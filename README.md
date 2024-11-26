# Jjama3 - Hackable Llama3.1 and Llama3.2 (text) in Julia

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/Jjama3.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/Jjama3.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/Jjama3.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/Jjama3.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/Jjama3.jl)

## Installation


We've split this into a few (unregistered) packages, so you'll need to add them all:
```julia
] add https://github.com/MurrellGroup/HuggingFaceTokenizers.jl
] add https://github.com/MurrellGroup/LowRankLayers.jl
] add https://github.com/MurrellGroup/LogitSamplers.jl
] add https://github.com/MurrellGroup/Jjama3.jl
```

## Quickstart

Download a Llama3 model `config.json`, `tokenizer.json`, and model safetensor weights from Hugging Face. Eg. [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/tree/main). Note: Hugging Face Llama3 models use a different RoPE convention to the original Meta implementation, and their weights have been permuted. This package works with the Huggingface convention, so if you load from the original Meta-Llama weights from a different source you'll need to do something horrible.

```julia
config = JSON3.read(read("SmolLM2-360M-Instruct/config.json", String))
model = load_llama3_from_safetensors("SmolLM2-360M-Instruct/model.safetensors", config)
tkn = tokenizer_from_file(Tokenizer, "SmolLM2-360M-Instruct/tokenizer.json")

prompt = smollm2_assistant_prompt(tkn,"Tell me the two worst things about Python.");
ts = generate(model, prompt, max_new_tokens=500, tokenizer_for_printing=tkn, end_token = encode(tkn, "<|im_end|>")[end]);
```

## Capability

- Seems to generate reasonable text from Llama3.1 and Llama3.2 models, loaded from Huggingface safetensors.
- Sampling accelerated with KV caching, with argmax and top-p sampling supported.
- Gradients seem to work on CPU, using Flux and Zygote. Untested on GPU.
- Sampling (and forward passes) work with CUDA, where everything is much faster. Gradients untested.
- Metal acceleration for forward_inference and forward_loss. Gradients untested. Sampling works, but is slower with Metal than with CPU.


## Samplers

The transformer emits "logits" which control the probability of the next token. A sampler takes these logits and converts them into a probability distribution over the vocabulary, and then samples from this distribution. There are [a few samplers available](https://github.com/MurrellGroup/LogitSamplers.jl), including argmax, top-p, top-k, min-p, and top-nσ. These can substantially affect the output of the model.

```julia
prompt = smollm2_assistant_prompt(tkn,"Tell me the two worst things about Python.");
ts = generate(model, prompt, max_new_tokens=500, tokenizer_for_printing=tkn, end_token = encode(tkn, "<|im_end|>")[end], sampler = top_nσ_sampler());
```

## Structured Sampling

You can pass in a custom sampler that places additional constraints on the sampling process. As an example, `structured_choice` is a sampler that always selects from a set of predefined options:

```julia
question = "In a Bayesian model, what do we call the probability distribution of parameters given the data?"
choices = ["Prior", "Likelihood", "Marginal Likelihood", "Evidence", "Posterior", "Margin Call"]
vocab = [decode(tkn, [i], skip_special_tokens = false) for i in 1:size(model.output.weight,1)]
eos = encode(tkn, "<|im_end|>")[end]
prompt = smollm2_instruct_prompt(tkn, "You are an expert in Statistics and Probability Theory who answers questions in as few words as possible.",question)
ts = generate(model, prompt, max_new_tokens=100, tokenizer_for_printing=tkn, end_token = eos, sampler = structured_choice(choices, vocab, eos));
```

This strategy can be extended to force the model outputs to follow specific formats.

## Finetuning

Often we want to adjust model parameters to better fit our specific use case, by further training the model on a new dataset. This can be done on all the model weights, but we also provide low-rank (via LoRA) finetuning.

```julia
using Jjama3, JSON3, Flux
config = JSON3.read(read("SmolLM2-360M-Instruct/config.json", String))
tkn = tokenizer_from_file(Tokenizer, "SmolLM2-360M-Instruct/tokenizer.json")

#Add LoRA to Q and V matrices when loading the model
model = load_llama3_from_safetensors("SmolLM2-360M-Instruct/model.safetensors", config, add_lora_to = [:Q, :V], lora_dim = 64)

#Set up a single, very silly, training example to finetune on
prompt = smollm2_assistant_prompt(tkn, "What language is the best for deep learning?");
ts = generate(model, prompt, max_new_tokens=50, tokenizer_for_printing=tkn, end_token = encode(tkn, "<|im_end|>")[end]);
trainsample = decode(tkn,prompt, skip_special_tokens = false) * "Ugh, bruh, what a stupid question.<|im_end|>";
train_toks = encode(tkn, trainsample);

#Set up the optimizer
opt_state = Flux.setup(AdamW(0.001f0), model);

#Train for 5 steps
for i in 1:5
    grads = Flux.gradient(model) do m
        forward_loss(m, train_toks[1:end-1,:], train_toks[2:end,:])
    end
    Flux.update!(opt_state, model, grads[1])
    println(i)
    generate(model, prompt, max_new_tokens=50, tokenizer_for_printing=tkn, end_token = encode(tkn, "<|im_end|>")[end])
    println()
end

#Ask the model an unrelated question:
prompt = smollm2_assistant_prompt(tkn, "Can you explain how tides work?");
generate(model, prompt, max_new_tokens=500, tokenizer_for_printing=tkn, end_token = encode(tkn, "<|im_end|>")[end], sampler = top_nσ_sampler());
```

