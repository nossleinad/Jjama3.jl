function nexttoken!(tokens, model, sampler, logits, tokenizer_for_printing)
    tokens[model.pos+1] = sampler(logits[:, end, 1])
    !isnothing(tokenizer_for_printing) && print(decode(tokenizer_for_printing, [tokens[model.pos+1]], skip_special_tokens = false))
end

"""
    generate(model, initial_tokens; max_new_tokens=100, sampler=top_pk_sampler(p=0.5f0, k=5), tokenizer_for_printing=tkn, end_token=128010)

Takes an initial sequence of tokens, and generates new tokens one at a time until the end token is sampled. Uses a KV cache. No batch dim for now.
Runs on CPU by default. If the model is on the GPU (assuming Flux.jl, eg. `model = gpu(model)`), then pass `device = gpu` to `generate` to run on the GPU.

```julia
tkn = llama3_tokenizer()
generate(model, initial_tokens; max_new_tokens=100, sampler=top_pk_sampler(p=0.5f0, k=5), tokenizer_for_printing=tkn, end_token=128010)
```
"""
function generate(
    model::Transformer{T}, 
    initial_tokens::AbstractArray{<:Integer};
    max_new_tokens=100,
    sampler::Function=argmax_sampler,
    tokenizer_for_printing = nothing,
    end_token = 128010,
    clear_cache = true,
    pos_offset = 0,
    device = identity,
    sdpa_func = sdpa
) where T
    tokens = vcat(initial_tokens, similar(initial_tokens, max_new_tokens))
    if clear_cache
        clear_cache!(model)
        config_cache!(model, length(initial_tokens) + max_new_tokens)
    else
        extend_cache!(model, length(initial_tokens) + max_new_tokens)
    end
    input_tokens = device(reshape(initial_tokens, :, 1))  # (seq_len, batch=1)
    logits = model(input_tokens, sdpa_func = sdpa_func)
    if max_new_tokens > 0
        nexttoken!(tokens, model, sampler, logits, tokenizer_for_printing)
        tokens[model.pos+1] == end_token && return tokens[1:model.pos+1]
    else
        return tokens
    end
    for _ in 1:max_new_tokens-1
        input_tokens = device(reshape([tokens[model.pos+1]], :, 1))  # Just the last token
        logits = model(input_tokens, sdpa_func = sdpa_func)
        nexttoken!(tokens, model, sampler, logits, tokenizer_for_printing)
        tokens[model.pos+1] == end_token && break
    end
    return tokens[1:model.pos+1]
end

