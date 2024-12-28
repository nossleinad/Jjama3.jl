#Trying out some tricks for attention.

#Figure out where to thunk...

#Will use Zygote - for testing grad correctness:
function sdpa_norrule(xq::AbstractArray{T}, xk::AbstractArray{T}, xv::AbstractArray{T}, mask::AbstractArray{T}, head_dim::Int) where T
    A = softmax(batched_mul(batched_transpose(xk), xq) / sqrt(T(head_dim)) .+ mask; dims=1)
    return batched_mul(xv, A)
end

function ChainRulesCore.rrule(::typeof(sdpa),
                              xq::AbstractArray{T}, #(D, LQ, HB)
                              xk::AbstractArray{T}, #(D, LKV, HB)
                              xv::AbstractArray{T}, #(D, LKV, HB)
                              mask::AbstractArray{T}, #(LKV, LQ)
                              head_dim::Int
                              ) where {T}
    α = sqrt(T(head_dim))
    A = softmax(((batched_mul(batched_transpose(xk), xq) ./ α) .+ mask); dims=1) #(LKV, LQ, HB) "head-batch"
    y = batched_mul(xv, A) #(D, LQ, HB)
    function sdpa_pullback(ȳ)
        xv̄ = batched_mul(ȳ, batched_transpose(A)) #(D, LKV, HB)
        Ā  = batched_mul(batched_transpose(xv), ȳ) #(LKV, LQ, HB)
        dM = (A .* (Ā .- (sum(A .* Ā, dims=1)))) ./ α #(LKV, LQ, HB)
        xq̄ = batched_mul(xk, dM) #(D, LQ, HB)
        xk̄ = batched_mul(xq, batched_transpose(dM)) #(D, LKV, HB)
        return NoTangent(), xq̄, xk̄, xv̄, NoTangent(), NoTangent()
    end
    return y, sdpa_pullback
end


function keychunked_sdpa(xq::AbstractArray{T,3},
                      xk::AbstractArray{T,3},
                      xv::AbstractArray{T,3},
                      mask::AbstractArray{T},
                      head_dim::Int;
                      k_chunk_size::Int=128
                     ) where {T<:Real}

    k_len  = size(xk,2)
    q_len  = size(xq,2)
    nbatch = size(xq,3)

    scale = one(T) / sqrt(T(head_dim))
    
    partial_max  = fill!(similar(xq, 1, q_len, nbatch), -Inf)
    partial_expw = fill!(similar(xq, 1, q_len, nbatch), T(0))
    partial_vals = fill!(similar(xq, head_dim, q_len, nbatch), T(0))

    # Preallocate local buffers for each chunk
    attn      = fill!(similar(xq, k_chunk_size, q_len, nbatch), T(0))
    local_max = fill!(similar(xq, 1, q_len, nbatch), T(0))
    new_max   = similar(local_max)
    w_old     = similar(local_max)
    w_new     = similar(local_max)
    chunk_sum = similar(local_max)
    valpart   = fill!(similar(xq, head_dim, q_len, nbatch), T(0))

    kstart = 1
    while kstart <= k_len
        k_batch = min(k_chunk_size, k_len - kstart + 1)
        xk_chunk   = @view xk[:, kstart : kstart + k_batch - 1, :]
        xv_chunk   = @view xv[:, kstart : kstart + k_batch - 1, :]
        if length(mask) > 1
            mask_chunk = @view mask[kstart : kstart + k_batch - 1, :, :]
        else
            mask_chunk = mask #Handles the case where the mask is 1-by-1 for sampling a single token.
        end
        attn_view = @view attn[1:k_batch, 1:q_len, 1:nbatch]
        xkT_chunk = batched_transpose(xk_chunk)

        batched_mul!(attn_view, xkT_chunk, xq, scale, 0)  # attn_view = scale*(xkT_chunk*xq)
        attn_view .= attn_view .+ mask_chunk  # add mask

        local_max .= maximum(attn_view, dims=1)
        @. new_max = max(partial_max, local_max)
        @. w_old = exp(partial_max - new_max)
        @. w_new = exp(local_max   - new_max)
        @. attn_view = exp(attn_view - local_max)

        partial_vals .= partial_vals .* w_old # Rescale old accumulators by w_old
        partial_expw .= partial_expw .* w_old

        chunk_sum .= sum(attn_view, dims=1) .* w_new
        partial_expw .+= chunk_sum

        batched_mul!(valpart, xv_chunk, attn_view)
        valpart .= valpart .* w_new
        partial_vals .+= valpart
        partial_max .= new_max
        kstart += k_batch
    end

    y = partial_vals ./ partial_expw
    return y
end


#Todo: use this to ignore parts of the -Inf mask triangle, since we're processing over chunks of queries.
function querychunked_sdpa(
    xq::AbstractArray{T,3},
    xk::AbstractArray{T,3},
    xv::AbstractArray{T,3},
    mask::AbstractArray{T},
    head_dim::Int;
    q_chunk_size::Int=128
) where {T<:Real}
    q_len   = size(xq, 2)
    kv_len  = size(xv, 2)
    nbatch  = size(xq, 3)
    q_chunk_size = min(q_chunk_size, q_len)
    α = sqrt(T(head_dim))
    y = similar(xq)
    qk_chunk = similar(xq, kv_len, q_chunk_size, nbatch)
    Achunk = similar(xq, kv_len, q_chunk_size, nbatch)
    qstart = 1
    while qstart <= q_len
        q_batch = min(q_chunk_size, q_len - qstart + 1)
        qinds = qstart:qstart+q_batch-1
        qk_chunkview = view(qk_chunk,:,1:q_batch,:)
        batched_mul!(qk_chunkview,batched_transpose(xk), view(xq, :, qinds, :), 1/α)
        Achunk[:,1:q_batch,:] .= softmax((qk_chunkview .+ view(mask,:,qinds)); dims=1) #(LKV, LQ, HB) "head-batch"
        batched_mul!(view(y,:,qinds,:),xv, view(Achunk,:,1:q_batch,:)) #(D, LQ, HB)
        qstart += q_batch
    end
    return y
end

function ChainRulesCore.rrule(::typeof(querychunked_sdpa),
                              xq::AbstractArray{T}, #(D, LQ, HB)
                              xk::AbstractArray{T}, #(D, LKV, HB)
                              xv::AbstractArray{T}, #(D, LKV, HB)
                              mask::AbstractArray{T}, #(LKV, LQ)
                              head_dim::Int;
                              q_chunk_size = 128
                              ) where {T}
    y = querychunked_sdpa(xq, xk, xv, mask, head_dim, q_chunk_size=q_chunk_size)
    function sdpa_pullback(ȳ)
        k_len   = size(xk, 2)
        q_len   = size(xq, 2)
        kv_len  = size(xv, 2)
        nbatch  = size(xq, 3)
        q_chunk_size = min(q_chunk_size, q_len)
        α = sqrt(T(head_dim))
        
        xq̄, xk̄, xv̄ = similar(xq), fill!(similar(xk), 0), fill!(similar(xv), 0)
        Achunk = similar(xq, kv_len, q_chunk_size, nbatch)
        Āchunk = similar(xq, kv_len, q_chunk_size, nbatch)
        dMchunk = similar(xq, kv_len, q_chunk_size, nbatch)
        qk_chunk = similar(xq, kv_len, q_chunk_size, nbatch)
        qstart = 1
        while qstart <= q_len
            q_batch = min(q_chunk_size, q_len - qstart + 1)
            qinds = qstart:qstart+q_batch-1
            ȳview = view(ȳ,:,qinds,:)
            qk_chunkview = view(qk_chunk,:,1:q_batch,:)
            batched_mul!(qk_chunkview,batched_transpose(xk), view(xq, :, qinds, :), 1/α)
            Achunk[:,1:q_batch,:] .= softmax((qk_chunkview .+ view(mask,:,qinds)); dims=1)
            batched_mul!(xv̄, ȳview, batched_transpose(view(Achunk,:,1:q_batch,:)), one(T), one(T))
            Āchunkview = view(Āchunk,:,1:q_batch,:)
            batched_mul!(Āchunkview, batched_transpose(xv), ȳview)
            Achunkview = view(Achunk,:,1:q_batch,:)
            dMchunk[:,1:q_batch,:] .= (Achunkview .* (Āchunkview .- (sum(Achunkview .* Āchunkview, dims=1)))) ./ α #(LKV, LQ, HB)
            dMchunkview = view(dMchunk,:,1:q_batch,:)
            batched_mul!(xk̄, view(xq,:,qinds,:), batched_transpose(dMchunkview), one(T), one(T)) #(LKV, D, HB)
            batched_mul!(view(xq̄,:,qinds,:),xk, dMchunkview) #(D, LQ, HB)
            qstart += q_batch
        end
        return NoTangent(), xq̄, xk̄, xv̄, NoTangent(), NoTangent()
    end
    return y, sdpa_pullback
end

#=
#Testing forward passes
begin
    L1 = 872 #Query
    L2 = 267 #Key/Value
    D = 32
    HB = 80
    xq, xk, xv, mask = randn(Float32, D, L1, HB), randn(Float32, D, L2, HB), randn(Float32, D, L2, HB), zeros(Float32, L2, L1);
    f(xq, xk, xv, mask, hd) = (Jjama3.sdpa(xq, xk, xv, mask, hd));
    fqc(xq, xk, xv, mask, hd) = (Jjama3.querychunked_sdpa(xq, xk, xv, mask, hd, q_chunk_size = 64));
    fkc(xq, xk, xv, mask, hd) = (Jjama3.keychunked_sdpa(xq, xk, xv, mask, hd, k_chunk_size = 64));
    
    res = f(xq, xk, xv, mask, D);
    qcres = fqc(xq, xk, xv, mask, D);
    kcres = fkc(xq, xk, xv, mask, D);

    @assert isapprox(res, qcres)
    @assert isapprox(res, kcres)

    @show size(res)
    @show size(kcres)


    #@btime f($xq, $xk, $xv, $mask, $D)
    #@btime fqc($xq, $xk, $xv, $mask, $D)
    #@btime fkc($xq, $xk, $xv, $mask, $D)
end;
=#



#=
#Testing grads
begin
L1 = 1000
L2 = 1200
D = 32
HB = 80
xq, xk, xv, mask = randn(Float32, D, L1, HB), randn(Float32, D, L2, HB), randn(Float32, D, L2, HB), zeros(Float32, L2, L1);
fnr(xq, xk, xv, mask, hd) = sum(Zygote.checkpointed(Jjama3.sdpa_norrule,xq, xk, xv, mask, hd));
f(xq, xk, xv, mask, hd) = sum(Zygote.checkpointed(Jjama3.sdpa,xq, xk, xv, mask, hd));
flm(xq, xk, xv, mask, hd) = sum(Jjama3.querychunked_sdpa(xq, xk, xv, mask, hd, q_chunk_size = 64));
@time res = withgradient(f, xq, xk, xv, mask, D);
@time nrres = withgradient(fnr, xq, xk, xv, mask, D);
@time lmres = withgradient(flm, xq, xk, xv, mask, D);

@assert isapprox(res[1], nrres[1])
@assert isapprox(res[2][1], nrres[2][1])
@assert isapprox(res[2][2], nrres[2][2])
@assert isapprox(res[2][3], nrres[2][3])
@assert isapprox(res[1], lmres[1])
@assert isapprox(res[2][1], lmres[2][1])
@assert isapprox(res[2][2], lmres[2][2])
@assert isapprox(res[2][3], lmres[2][3])

GC.gc()
println("normal+rrule chechpointed:")
@time res = withgradient(f, xq, xk, xv, mask, D);
@time res = withgradient(f, xq, xk, xv, mask, D);

GC.gc()
println("normal+Zygote chechpointed:")
@time nrres = withgradient(fnr, xq, xk, xv, mask, D);
@time nrres = withgradient(fnr, xq, xk, xv, mask, D);

GC.gc()
println("chunked:")
@time lmres = withgradient(flm, xq, xk, xv, mask, D);
@time lmres = withgradient(flm, xq, xk, xv, mask, D);


println("btimed:")
GC.gc()
@btime res = withgradient(f, xq, xk, xv, mask, D);
GC.gc()
@btime nrres = withgradient(fnr, xq, xk, xv, mask, D);
GC.gc()
@btime lmres = withgradient(flm, xq, xk, xv, mask, D);

true
end
=#

