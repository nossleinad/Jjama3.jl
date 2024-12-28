#Note about output layer being tied to embedding: https://github.com/meta-llama/llama-models/issues/172

function create_mask(h::AbstractArray{T}; precached_size = 0) where T<:AbstractFloat
    Flux.ChainRulesCore.ignore_derivatives() do
        dim, seqlen, batch = size(h)
        mask = similar(h, seqlen, seqlen)
        mask .= T(-Inf)
        mask = tril(mask, -1) #This is swapped because we're using the slightly more efficient dim setup
        if precached_size > 0
            pad = similar(h, precached_size, seqlen)
            pad .= T(0.0)
            mask = vcat(pad, mask)
        end
        return mask
    end
end


function masked_agg(ce, mask)
    if mask !== nothing
        ce = ce .* mask
    end
    return sum(ce)/sum(mask)
end

#Hoping this will wind up in Zygote.jl
"""
    eager_update!(state, model, update!)

Updates params during the backward pass, saving memory.

f(model, xs...) = model(xs...)
h = f(Zygote.eager_update!(state.layers[i], model.layers[i], Optimisers.update!), h, other_args...)
"""
function eager_update!(state, model, update!)
    function update_hook(dmodel)
        update!(state, model, dmodel)
        return nothing
    end
    return Flux.Zygote.hook(update_hook, model)
end


wrap(model, xs...) = model(xs...)
function (model::Transformer)(tokens::AbstractArray{Int}, opt_state; clear_cache = false, checkpoint_func = wrap, sdpa_func = sdpa)
    if clear_cache
        Flux.ChainRulesCore.ignore_derivatives() do
            Jjama3.clear_cache!(model)
        end
    end
    h = model.tok_embeddings(tokens) # Embedding: (dim, seq_len, batch)
    rope = model.rope[model.pos+1:model.pos+size(tokens, 1)]
    if size(h, 2) == 1 #If there is only one new token, then a 1-by-1 mask = 0 works, via broadcasting (if the attention functions allow it)
        mask = Jjama3.create_mask(h)
    else
        mask = Jjama3.create_mask(h; precached_size = model.pos)
    end
    for i in 1:length(model.layers)
        if !isnothing(opt_state)
            #If checkpoint_func is also just wrap, then this does nothing, but if its Zygote.checkpointed, then this is a checkpointed update
            h = checkpoint_func(wrap, eager_update!(opt_state.layers[i], model.layers[i], Optimisers.update!), h, model.pos, rope, mask, sdpa_func)   
        else
            h = checkpoint_func(wrap, model.layers[i], h, model.pos, rope, mask, sdpa_func)
        end
    end
    h = model.norm(h)
    output = model.output(h)
    model.pos += size(tokens, 1)
    return output
end

(model::Transformer)(tokens::AbstractArray{Int}; clear_cache = false, checkpoint_func = wrap, sdpa_func = sdpa) = model(tokens, nothing; clear_cache, checkpoint_func, sdpa_func)

function loss(logits, targets::AbstractArray; loss_mask = nothing)
    vocab_size = size(logits,1)
    gt = Flux.onehotbatch(targets, 1:vocab_size)
    if loss_mask !== nothing
        loss = Flux.logitcrossentropy(logits, gt, agg = x -> masked_agg(x, loss_mask))
    else
        loss = Flux.logitcrossentropy(logits, gt)
    end
    return loss
end

# compat
forward_inference(model, args...) = model(args...)
forward_loss(model::Transformer, inputs::AbstractArray, targets::AbstractArray; clear_cache = true, loss_mask = nothing) = loss(model(inputs, clear_cache = clear_cache), targets, loss_mask = loss_mask)
