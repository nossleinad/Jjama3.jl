module MetalExt

#Note: Metal speeds things up a little for forward_inference and forward_loss calls, but is VERY slow for sampling.
#It seems that each single Metal call has some constant overhead that kills it.

using Metal, Jjama3.NNlib

function NNlib.batched_mul(a::MtlArray, b::MtlArray)
    a_shape = size(a)
    b_shape = size(b)
    a_reshaped = reshape(a, a_shape[1], a_shape[2], :)
    b_reshaped = reshape(b, b_shape[1], b_shape[2], :)
    res = Metal.zeros(a_shape[1], b_shape[2], size(a_reshaped)[3])
    Metal.MPS.matmul!(res, a_reshaped,b_reshaped)
    return reshape(res, a_shape[1], b_shape[2], a_shape[3:end]...)
end

end
