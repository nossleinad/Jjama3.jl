# Jjama3 - Hackable Llama3.1 and Llama3.2 (text) in Julia

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/Jjama3.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/Jjama3.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/Jjama3.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/Jjama3.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/Jjama3.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/Jjama3.jl)

# Installation

```julia
] add https://github.com/MurrellGroup/Jjama3.jl
```

# Quickstart

Download a Llama3 model `config.json` and safetensor weights from Huggingface. Eg. [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct). You might need access permissions for this. Note: Huggingface use a different RoPE convention to the original Meta implementation, and their weights have been permuted. This package works with the Huggingface convention, so if you load from the original weights you'll need to permute them.

```julia
config = JSON3.read(read("Llama3_2_1B_instruct/config.json", String));
model = load_llama3_from_safetensors("Llama3_2_1B_instruct/model.safetensors", config);
tkn = llama3_tokenizer();
prompt = assistant_prompt("Why would anyone implement the llama3 LLM in Julia?", tkn);
ts = generate(model, prompt, max_new_tokens=500, encoder_for_printing=tkn);
```

# Capability

- Seems to generate reasonable text from Llama3.1 and Llama3.2 models, loaded from Huggingface safetensors.
- Sampling accelerated with KV caching, with argmax and top-p sampling supported.
- Gradients seem to work on CPU, using Flux and Zygote. Untested on GPU.
- Sampling (and forward passes) work with CUDA, where everything is much faster. Gradients untested.
- Metal acceleration for forward_inference and forward_loss. Gradients untested. Sampling works, but is much slower with Metal than with CPU.
