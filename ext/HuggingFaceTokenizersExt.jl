module HuggingFaceTokenizersExt

using Jjama3
using HuggingFaceTokenizers

encode(tkn::Tokenizer, str; kwargs...) = HuggingFaceTokenizers.encode(tkn, str; kwargs...).ids .+ 1
decode(tkn::Tokenizer, ids; kwargs...) = HuggingFaceTokenizers.decode(tkn, ids .- 1; kwargs...)

end
