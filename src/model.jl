#Note about output layer being tied to embedding: https://github.com/meta-llama/llama-models/issues/172

function apply_scaling(freqs::AbstractVector; scale_factor=8)
    #Hard-coded - I should move these to the main model struct and grab them from the config.
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192
    ###
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = similar(freqs)
    for (i, freq) in enumerate(freqs)
        wavelen = 2 * π / freq
        if wavelen < high_freq_wavelen
            new_freqs[i] = freq
        elseif wavelen > low_freq_wavelen
            new_freqs[i] = freq / scale_factor
        else
            @assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / 
                    (high_freq_factor - low_freq_factor)
            new_freqs[i] = (1 - smooth) * freq / scale_factor + smooth * freq
        end
    end
    return new_freqs
end

function precompute_freqs_cis(dim::Int, end_pos::Int; 
                            theta::T=10000f0, use_scaled=true, scale_factor=8) where T
    freqs = 1f0 ./ (theta .^ (T.(0:2:dim-1)[1:dim÷2] ./ dim))
    if use_scaled
        freqs = apply_scaling(freqs; scale_factor=scale_factor)
    end
    freqs_complex = cis.(T.(0:end_pos-1) * freqs')
    cos = permutedims(real(freqs_complex), (2, 1))  # (head_dim/2, seq_len)
    sin = permutedims(imag(freqs_complex), (2, 1))
    cos = reshape(cos, (dim÷2, end_pos, 1, 1))
    sin = reshape(sin, (dim÷2, end_pos, 1, 1))
    return cos, sin
end


#Note about Huggingface weights and rotary embeddings: https://discuss.huggingface.co/t/is-llama-rotary-embedding-implementation-correct/44509
#Use this one if you're using the Hugging Face weights.
function apply_rotary_emb(x, freqs_cis)
    head_dim, seq_len, n_heads, batch = size(x)
    x1 = @view x[1:head_dim÷2, :, :, :]
    x2 = @view x[head_dim÷2+1:end, :, :, :]
    cos, sin = freqs_cis
    out = vcat(  
        x1 .* cos .- x2 .* sin,
        x2 .* cos .+ x1 .* sin
    )
    return out
end

function update_kv_cache(cache::KVCache, start_pos::Int, xk::AbstractArray, xv::AbstractArray)
    seqlen = size(xk, 2)
    cache.cache_k[:, (start_pos+1):(start_pos+seqlen), :, :] .= xk
    cache.cache_v[:, (start_pos+1):(start_pos+seqlen), :, :] .= xv
    return cache.cache_k[:, 1:(start_pos+seqlen), :, :],
           cache.cache_v[:, 1:(start_pos+seqlen), :, :]
end

function repeat_kv(x::AbstractArray, n_rep::Int)
    if n_rep == 1
        return x
    end
    return repeat(x, 1, n_rep, 1, 1)
end

function forward_inference(model::Transformer{T}, tokens::AbstractArray{Int}, start_pos::Int) where T
    seqlen = size(tokens, 1) # tokens expected as (seq_len, batch)
    h = model.tok_embeddings(tokens) # Embedding: (dim, seq_len, batch)

    # Get relevant freqs_cis slice
    cos, sin = model.freqs_cis #@show size(cos) #(head_dim/2, max_RoPE, 1, 1)
    freqs_cis = (cos[:,start_pos+1:start_pos+seqlen,:,:], sin[:,start_pos+1:start_pos+seqlen,:,:])
    

    mask = create_mask(h)
    for layer in model.layers
        h = layer(h, start_pos, freqs_cis, mask)
    end
    h = model.norm(h)
    output = model.output(h)
    return output
end

function create_mask(h::AbstractArray)
    Flux.Zygote.ignore() do
        embeddim, seqlen, batch = size(h)
        mask = similar(h, seqlen, seqlen)
        T = eltype(h)
        mask .= T(-Inf)
        #mask = triu(mask, 1)
        mask = tril(mask, -1) #This is swapped because we're using the slightly more efficient dim setup
        return mask
    end
end

function forward_loss(model::Transformer{T}, inputs::AbstractArray, 
                     targets::AbstractArray; ignore_index::Int=-100,
                     mask = :auto) where T
    seqlen = size(inputs, 1) #(seq_len, batch)
    h = model.tok_embeddings(inputs) # (dim, seq_len, batch)
    cos, sin = model.freqs_cis #@show size(cos) #(head_dim/2, max_RoPE, 1, 1)
    freqs_cis = (cos[:,1:seqlen,:,:], sin[:,1:seqlen,:,:])
    # Forward through layers (start_pos = 0 disables KV caching)
    if mask == :auto
        mask = create_mask(h)
    end
    for layer in model.layers
        h = layer(h, 0, freqs_cis, mask)
    end
    h = model.norm(h)
    logits = model.output(h)
    # Need to reshape to (vocab_size, seq_len * batch)
    logits_2d = reshape(logits, size(logits,1), :)
    targets_1d = reshape(targets, :)
    # Mask out ignored indices - will handle this later.
    # Note: this is not the autoregressive mask, but the mask for the loss function
    #=
    mask = targets_1d .!= ignore_index
    if any(mask)
        loss = Flux.logitcrossentropy(
            logits_2d[:, mask],
            targets_1d[mask]
        )
    else
        loss = zero(Float32)
    end
    =#
    vocab_size = size(model.tok_embeddings.weight, 2)
    gt = Flux.onehotbatch(targets_1d, 1:vocab_size)
    loss = Flux.logitcrossentropy(logits_2d, gt)
    return loss
end


#https://discuss.huggingface.co/t/is-llama-rotary-embedding-implementation-correct/44509
#=
#Use this one if you're using the original Meta weights.
#You'll need to change the type of the freqs_cis field in Transformer to match.
function precompute_freqs_cis(dim::Int, end_pos::Int; 
                             theta::Float32=10000f0, 
                             use_scaled::Bool=true, scale_factor::Int=8)
    # Create frequencies for the first half of dimensions
    freqs = 1f0 ./ (theta .^ (Float32.(0:2:dim-1)[1:dim÷2] ./ dim))
    # Create position indices - note, using 0 indexing here because python consistency. Not sure if it makes a difference.
    t = Float32.(0:end_pos-1)
    if use_scaled
        freqs = apply_scaling(freqs; scale_factor=scale_factor)
    end
    # Compute outer product
    freqs = t * freqs'
    # Convert to complex exponentials
    # Note: Julia's cis(x) = exp(ix) = cos(x) + i*sin(x)
    freqs_complex = cis.(freqs)
    # Stack real and imaginary parts
    # Note: Julia's reshape is similar to PyTorch's stack
    freqs_cis_real = reshape(
        reinterpret(Float32, reshape(freqs_complex, :)), 
        (2, size(freqs)...)
    )
    # Permute to match PyTorch's dimension ordering
    return permutedims(freqs_cis_real, (2,3,1))
end

function apply_rotary_emb(x, freqs_cis)
    # x is (head_dim, seq_len, n_heads, batch) in Julia
    # freqs_cis is (seq_len, head_dim/2, 2)

    #@show size(freqs_cis)
    
    # Reshape x to separate real/imaginary pairs
    head_dim, seq_len, n_heads, batch = size(x)
    x_reshaped = reshape(x, (2, head_dim÷2, seq_len, n_heads, batch))
    
    # Reshape freqs_cis to broadcast correctly
    # Note: reshape to (2, head_dim/2, seq_len, 1, 1) for broadcasting
    freqs_cis = permutedims(freqs_cis, (3, 2, 1))  # now (2, head_dim/2, seq_len)
    freqs_cis = reshape(freqs_cis, (2, size(freqs_cis, 2), size(freqs_cis, 3), 1, 1))
    
    # Apply rotation using complex multiplication formula:
    # (a + bi)(c + di) = (ac-bd) + (ad+bc)i
    x_real = x_reshaped[1:1, :, :, :, :]
    x_imag = x_reshaped[2:2, :, :, :, :]
    f_real = freqs_cis[1:1, :, :, :, :]
    f_imag = freqs_cis[2:2, :, :, :, :]

    #@show size(f_real)
    #@show size(f_imag)

    #This is for checking the freqs_cis.
    #Note: the cos, sin values are repeated in python
    #g(f_real, f_imag) #passes
    
    out_real = x_real .* f_real .- x_imag .* f_imag
    out_imag = x_imag .* f_real .+ x_real .* f_imag
    
    # Combine and reshape back
    out = vcat(out_real, out_imag)
    return reshape(out, (head_dim, seq_len, n_heads, batch))
end
=#
