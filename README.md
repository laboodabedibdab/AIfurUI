# Simple SDXL Web UI

A straightforward Gradio-based web interface for running **Stable Diffusion XL (SDXL)** locally using PyTorch and Diffusers. Built as a custom script to handle text-to-image and image-to-image generation without heavy UI frameworks.

## Features
* **txt2img & img2img:** Generate images from prompts or iterate on existing ones with adjustable denoising strength.
* **VRAM Management:** Includes basic cleanup routines (`torch.cuda.empty_cache`, `gc.collect`) and supports `xformers`, `CPU offload`, `VAE slicing`, and `attention slicing` to help fit models into limited VRAM.
* **Local Assets:** Automatically scans local directories for model checkpoints (`.safetensors`, `.pt`, `.bin`) and custom VAEs.
