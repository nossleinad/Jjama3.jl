struct KVCache{T}
    cache_k::AbstractArray{T, 4}  # (head_dim, seq_len, n_kv_heads, batch)
    cache_v::AbstractArray{T, 4} 
end

function KVCache(T, batch_size::Int, seq_length::Int, n_kv_heads::Int, head_dim::Int; device = identity)
    cache_k = zeros(T, head_dim, seq_length, n_kv_heads, batch_size) |> device
    cache_v = zeros(T, head_dim, seq_length, n_kv_heads, batch_size) |> device
    KVCache(cache_k, cache_v)
end

struct FeedForward
    w1::Union{Dense, LoRADense}
    w2::Union{Dense, LoRADense}
    w3::Union{Dense, LoRADense}
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

mutable struct Attention
    wq::Union{Dense, LoRADense}
    wk::Union{Dense, LoRADense}
    wv::Union{Dense, LoRADense}
    wo::Union{Dense, LoRADense}
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

    #Lazy permute dims. Need to test CUDA. Note: test fails.
    #xq = PermutedDimsArray(xq, (1,3,2,4))
    #xk = PermutedDimsArray(xk, (1,3,2,4))
    #xv = PermutedDimsArray(xv, (1,3,2,4))

    xq = permutedims(xq, (1,3,2,4))
    xk = permutedims(xk, (1,3,2,4))
    xv = permutedims(xv, (1,3,2,4))

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
    
    scores = batched_mul(batched_transpose(xk_for_attn), xq_for_attn) ./ sqrt(T(attn.head_dim))
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

Flux.@layer :expand Attention trainable=(wq, wv)

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

Flux.@layer TransformerBlock trainable=(attention, )

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

Flux.@layer :expand Transformer trainable=(layers, )
