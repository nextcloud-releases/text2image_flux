# Nextcloud Local Image Generation: Flux

An ExApp that generates images from text using a quantized [FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) model (GGUF Q4_0) via [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp).

Weights downloaded on init:

* `flux-2-klein-4b-Q4_0.gguf` ([leejet/FLUX.2-klein-4B-GGUF](https://huggingface.co/leejet/FLUX.2-klein-4B-GGUF))
* `flux2-vae.safetensors` ([Comfy-Org/flux2-dev](https://huggingface.co/Comfy-Org/flux2-dev))
* `Qwen3-4B-Q4_K_M.gguf` ([unsloth/Qwen3-4B-GGUF](https://huggingface.co/unsloth/Qwen3-4B-GGUF))

The model runs completely on your machine via AppAPI. No private data leaves your servers.

Requires roughly 7GB VRAM on an NVIDIA GPU. CPU inference is also supported but slow. Use text2image_stablediffusion2 if you want a faster model.

## Ethical AI Rating

### Rating: 🟡

Positive:

* The software for training and inferencing of this model is open source
* The trained model is freely available under Apache 2.0, and thus can be ran on-premises

Negative:

* The training data isn't freely available, making it not possible to check or correct for bias or optimise the performance and CO2 usage.
