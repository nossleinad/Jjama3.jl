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
    # x is (head_dim, seq_len, n_heads, batch)
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


struct FeedForward
    w1::Dense
    w2::Dense
    w3::Dense
end

function FeedForward(dim::Int, ff_hidden_dim::Int)
    FeedForward(
        Dense(dim => ff_hidden_dim, bias=false),
        Dense(ff_hidden_dim => dim, bias=false),
        Dense(dim => ff_hidden_dim, bias=false)
    )
end

function (ff::FeedForward)(x)
    return ff.w2(Flux.swish(ff.w1(x)) .* ff.w3(x))
end

Flux.@layer :expand FeedForward

struct RMSNorm{T}
    weight::AbstractVector{T}
    eps::T
end

function RMSNorm(dim::Int; eps::T=1f-5) where T
    RMSNorm{T}(ones(T, dim), eps)
end

function (norm::RMSNorm)(x)
    rms = sqrt.(sum(abs2.(x), dims=1) ./ size(x,1) .+ norm.eps)
    return x .* (norm.weight ./ rms)
end

Flux.@layer RMSNorm

struct KVCache{T}
    cache_k::AbstractArray{T, 4}  # (head_dim, seq_len, n_kv_heads, batch)
    cache_v::AbstractArray{T, 4} 
end

function KVCache(T, batch_size::Int, seq_length::Int, n_kv_heads::Int, head_dim::Int; device = identity)
    cache_k = zeros(T, head_dim, seq_length, n_kv_heads, batch_size) |> device
    cache_v = zeros(T, head_dim, seq_length, n_kv_heads, batch_size) |> device
    KVCache(cache_k, cache_v)
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

mutable struct Attention
    wq::Dense
    wk::Dense
    wv::Dense
    wo::Dense
    n_heads::Int
    n_kv_heads::Int
    head_dim::Int
    n_rep::Int
    cache::Union{Nothing, KVCache}
end

function Attention(dim::Int, n_heads::Int, n_kv_heads=n_heads)
    head_dim = dim ÷ n_heads
    n_rep = n_heads ÷ n_kv_heads
    Attention(
        Dense(dim => n_heads * head_dim, bias=false),
        Dense(dim => n_kv_heads * head_dim, bias=false),
        Dense(dim => n_kv_heads * head_dim, bias=false),
        Dense(n_heads * head_dim => dim, bias=false),
        n_heads,
        n_kv_heads,
        head_dim,
        n_rep,
        nothing
    )
end

function (attn::Attention)(x::AbstractArray{T}, start_pos::Int, freqs_cis, mask=nothing) where T
    dim, seqlen, batch = size(x)

    xq = attn.wq(x)
    xk = attn.wk(x)
    xv = attn.wv(x)

    xq = reshape(xq, (attn.head_dim, attn.n_heads, seqlen, batch))
    xk = reshape(xk, (attn.head_dim, attn.n_kv_heads, seqlen, batch))
    xv = reshape(xv, (attn.head_dim, attn.n_kv_heads, seqlen, batch))

    #Lazy permute dims. Need to test CUDA.
    xq = PermutedDimsArray(xq, (1,3,2,4))
    xk = PermutedDimsArray(xk, (1,3,2,4))
    xv = PermutedDimsArray(xv, (1,3,2,4))

    xq_rope = apply_rotary_emb(xq, freqs_cis)
    xk_rope = apply_rotary_emb(xk, freqs_cis)

    if !isnothing(attn.cache)
        xk_rope, xv = update_kv_cache(attn.cache, start_pos, xk_rope, xv)
    end

    xk_rope = repeat_kv(xk_rope, attn.n_rep)
    xv      = repeat_kv(xv, attn.n_rep)
    
    xq_for_attn = reshape(xq_rope, attn.head_dim, :,  attn.n_heads * batch)
    xk_for_attn = reshape(xk_rope, attn.head_dim, :, attn.n_heads * batch)
    xv_for_attn = reshape(xv, attn.head_dim, :, attn.n_heads * batch)
    
    #=
    scores = batched_mul(
        permutedims(xq_for_attn, (2,1,3)),  # (seqlen, head_dim, batch*heads)
        #batched_transpose(xq_for_attn),  # (seqlen, head_dim, batch*heads)
        xk_for_attn                          # (head_dim, seqlen, batch*heads)
    ) ./ sqrt(T(attn.head_dim))
    if !isnothing(mask)
        scores = scores .+ mask
    end
    sm_scores = softmax(scores; dims=2) 
    output = batched_mul(sm_scores, permutedims(xv_for_attn, (2,1,3)))
    e_output = reshape(output, (seqlen, attn.head_dim, attn.n_heads, batch))
    p_output = permutedims(e_output, (2,3,1,4))  # (n_heads, head_dim, seqlen, batch)
    =#
    
    scores = batched_mul(
        batched_transpose(xk_for_attn),  
        xq_for_attn                         
    ) ./ sqrt(T(attn.head_dim))
    if !isnothing(mask)
        scores = scores .+ mask
    end
    sm_scores = softmax(scores; dims=1)
    output = batched_mul(xv_for_attn, sm_scores)
    e_output = reshape(output, (attn.head_dim, seqlen, attn.n_heads, batch))
    p_output = permutedims(e_output, (1,3,2,4)) 
    
    r_output = reshape(p_output, (attn.head_dim * attn.n_heads, seqlen, batch))
    proj = attn.wo(r_output)
    return proj
end

Flux.@layer :expand Attention

struct TransformerBlock
    attention::Attention
    feed_forward::FeedForward
    attention_norm::RMSNorm
    ffn_norm::RMSNorm
end

function TransformerBlock(dim::Int, n_heads::Int, n_kv_heads::Int=n_heads, ff_hidden_dim = 4 * dim;
                         norm_eps=1f-5)
    TransformerBlock(
        Attention(dim, n_heads, n_kv_heads),
        FeedForward(dim, ff_hidden_dim),
        RMSNorm(dim, eps=norm_eps),
        RMSNorm(dim, eps=norm_eps)
    )
end

function (block::TransformerBlock)(x, start_pos, freqs_cis, mask=nothing)
    h = x + block.attention(block.attention_norm(x), start_pos, freqs_cis, mask)
    out = h + block.feed_forward(block.ffn_norm(h))
    return out
end

Flux.@layer TransformerBlock

struct Transformer{T}
    tok_embeddings::Flux.Embedding
    layers::AbstractVector{TransformerBlock}
    norm::RMSNorm{T}
    output::Dense
    freqs_cis::Tuple{AbstractArray{T, 4}, AbstractArray{T, 4}}
end

function Transformer(vocab_size::Int, dim::Int, n_layers::Int, n_heads::Int, 
                    n_kv_heads::Int, max_seq_len::Int, ff_hidden_dim::Int;
                    norm_eps::T=1f-5,
                    rope_theta::T=500000f0,
                    use_scaled_rope=false,
                    scale_factor=8) where T
    
    tok_embeddings = Flux.Embedding(vocab_size => dim)
    layers = [TransformerBlock(dim, n_heads, n_kv_heads, ff_hidden_dim; norm_eps=norm_eps) for _ in 1:n_layers]
    norm = RMSNorm(dim, eps=norm_eps)
    output = Dense(dim => vocab_size, bias=false)
    freqs_cis = precompute_freqs_cis(
        dim ÷ n_heads,
        max_seq_len * 2;
        theta=rope_theta,
        use_scaled=use_scaled_rope,
        scale_factor=scale_factor
    )
    Transformer(tok_embeddings, layers, norm, output, freqs_cis)
end

Flux.@layer :expand Transformer trainable=(layers, norm)


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
    embeddim, seqlen, batch = size(h)
    mask = similar(h, seqlen, seqlen)
    T = eltype(h)
    mask .= T(-Inf)
    #mask = triu(mask, 1)
    mask = tril(mask, -1)
    return mask
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
