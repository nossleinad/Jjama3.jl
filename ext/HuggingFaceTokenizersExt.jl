module HuggingFaceTokenizersExt

using Jjama3
using HuggingFaceTokenizers

Jjama3.encode(tkn::HuggingFaceTokenizers.Tokenizer, str; kwargs...) = HuggingFaceTokenizers.encode(tkn, str; kwargs...).ids .+ 1
Jjama3.decode(tkn::HuggingFaceTokenizers.Tokenizer, ids; kwargs...) = HuggingFaceTokenizers.decode(tkn, ids .- 1; kwargs...)

end
