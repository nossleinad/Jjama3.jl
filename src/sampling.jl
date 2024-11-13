
function default_sampler(logits::AbstractVector)
    return argmax(logits)
end

function generate(model::Transformer{T}, 
                 initial_tokens::AbstractArray{IntT};
                 max_new_tokens=100,
                 sampler::Function=default_sampler,
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
        input_tokens = reshape(initial_tokens, :, 1)  # (seq_len, batch=1)
        logits = forward_inference(model, input_tokens, 0)
        start_pos = current_len
    else
        start_pos = 0
    end
    # Generate new tokens one at a time
    for _ in 1:max_new_tokens
        # If sequence is empty or we want to process just the last token
        if start_pos == 0
            input_tokens = reshape([128001], :, 1)  # Use start of text token if empty
        else
            input_tokens = reshape([tokens[current_len]], :, 1)  # Just the last token
        end
        # Get logits for next token
        logits = forward_inference(model, input_tokens, start_pos)
        # Sample next token (logits are size vocab × 1 × 1)
        next_token = sampler(vec(logits[:, end, 1]))
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

