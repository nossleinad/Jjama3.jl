
function argmax_sampler(logits::AbstractVector; device = identity)
    return argmax(device(logits))
end

argmax_sampler(; device = identity) = logits -> argmax_sampler(logits; device = device)

function top_pk_sampler(logits::AbstractVector; p = 0.5f0, k = 5, device = identity)
    probs = device(Jjama3.softmax(logits))
    perm = partialsortperm(probs, 1:k, rev=true)
    sorted_probs = probs[perm]
    cumsum_probs = cumsum(sorted_probs)
    if cumsum_probs[1] > p
        return perm[1]
    else
        cutoff = findlast(cumsum_probs .< p)
        return sample(perm[1:cutoff], Weights(sorted_probs[1:cutoff]))
    end
end

top_pk_sampler(;p = 0.5f0, k = 5, device = identity) = logits -> top_pk_sampler(logits; p, k, device)

# This generate function seems to do one unnecessary forward pass when switching from the forward pass over the initial sequence
# to the sampling of each token. But when I try and fix it, the model gets slightly dumber.
# Vibes feel like a shift-by-1 in the RoPE, or something similar. Need to investigate when I find time.
"""
Takes an initial sequence of tokens, and generates new tokens one at a time until the end token is sampled. Uses a KV cache. No batch dim for now.
Runs on CPU by default. If the model is on the GPU (assuming Flux.jl, eg. `model = gpu(model)`), then pass `device = gpu` to `generate` to run on the GPU.

    tkn = llama3_tokenizer()
    generate(model, initial_tokens; max_new_tokens=100, sampler=top_pk_sampler(p=0.5f0, k=5), encoder_for_printing=tkn, end_token=128010)
"""
function generate(model::Transformer{T}, 
                 initial_tokens::AbstractArray{IntT};
                 max_new_tokens=100,
                 sampler::Function=argmax_sampler,
                 encoder_for_printing = nothing,
                 end_token = 128010,
                 device = identity) where {T, IntT}
    
    # Initialize sequence with a new copy of the tokens
    current_len = length(initial_tokens)
    tokens = Vector{IntT}(undef, current_len + max_new_tokens)
    tokens[1:current_len] = initial_tokens
    # Set up KV caches for all attention layers    
    for layer in model.layers
        layer.attention.cache = KVCache(
            T,  # eltype
            1,  # batch_size
            current_len + max_new_tokens,  # max possible sequence length
            layer.attention.n_kv_heads,
            layer.attention.head_dim,
            device = device
        )
    end
    # Process the initial sequence
    if current_len > 0
        input_tokens = device(reshape(initial_tokens, :, 1))  # (seq_len, batch=1)
        logits = forward_inference(model, input_tokens, 0)
        start_pos = current_len
    else
        start_pos = 0
    end
    # Generate new tokens one at a time
    for _ in 1:max_new_tokens
        # If sequence is empty or we want to process just the last token
        if start_pos == 0
            input_tokens = device(reshape([128001], :, 1))  # Use start of text token if empty
        else
            input_tokens = device(reshape([tokens[current_len]], :, 1))  # Just the last token
        end
        # Get logits for next token
        logits = forward_inference(model, input_tokens, start_pos)
        # Sample next token (logits are size vocab × 1 × 1)
        next_token = sampler(logits[:, end, 1])
        current_len += 1
        tokens[current_len] = next_token
        if !isnothing(encoder_for_printing)
            print(encoder_for_printing.decode([next_token]))
        end
        if next_token == end_token
            break
        end
        start_pos += 1
    end
    # Clear KV caches
    for layer in model.layers
        layer.attention.cache = nothing
    end
    return tokens[1:current_len]
end

