# Jjama3

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/Jjama3.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/Jjama3.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/Jjama3.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/Jjama3.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/Jjama3.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/Jjama3.jl)

# Quickstart

```julia
config = JSON3.read(read("Llama3_2_1B_instruct/config.json", String));
model = load_llama3_from_safetensors("Llama3_2_1B_instruct/model.safetensors", config);
tkn = llama3_tokenizer();
prompt = assistant_prompt("Why would anyone implement the llama3 LLM in Julia?", tkn);
ts = generate(model, prompt, max_new_tokens=500, encoder_for_printing=tkn);
```