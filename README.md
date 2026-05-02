# ComfyUI Video Experiments

This repository contains ComfyUI workflows for image generation and image-to-video animation.

## Workflows

### 1. Image Generation (`comfyui/workflows/ImageGeneration.json`)

A straightforward workflow for generating high-quality portrait images using Stable Diffusion XL.

**Requirements:**
- Model: `Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors` (place in `models/checkpoints/`)

**Configuration:**
- Resolution: 1024x1024
- Sampler: Euler
- Scheduler: Normal
- Steps: 20
- CFG: 8

**Default Prompts:**
- Positive: "model portrait, classy, big smile, medium brown hairs, green eyes, 30y, head and shoulders portrait, hyperdetailed photography, cover"
- Negative: "bad eyes, cgi, airbrushed, plastic, deformed, watermark"

**Usage:**
Load this workflow in ComfyUI and adjust the prompts as needed for your desired portrait style.

---

### 2. Image to Video (`comfyui/workflows/ImageToVideo.json`)

Advanced workflow for converting static images into animated videos using Wan 2.2 Image-to-Video models with the Wan 2.1 VAE. This workflow uses a two-pass approach with high noise and low noise models for better quality.

**Requirements:**
- **Wan 2.2 I2V Models (GGUF format, Mac MPS compatible!):**
  - `Wan2.2-I2V-A14B-HighNoise-Q5_K_M.gguf` (place in `models/diffusion_models/`)
  - `Wan2.2-I2V-A14B-LowNoise-Q5_K_M.gguf` (place in `models/diffusion_models/`)

- **Text Encoder:**
  - `umt5_xxl_fp8_e4m3fn_scaled.safetensors` (T5 encoder for Wan)

- **VAE (Wan 2.1):**
  - `wan_2.1_vae.safetensors` (place in `models/vae/`)
  - Note: Wan 2.2 models are compatible with the Wan 2.1 VAE

**Configuration:**
- Input image resolution: 288x512
- Video length: 81 frames
- FPS: 16
- Sampler: Euler
- Scheduler: Simple
- Two-pass sampling:
  - Pass 1 (High Noise): Steps 0-10, CFG 3.5
  - Pass 2 (Low Noise): Steps 10-10000, CFG 3.5

**Default Prompts:**
- Positive: "Have her dance following:\n\n1. Basic Stance"
- Negative: "blurry, low quality, distorted, deformed, ugly, bad anatomy, watermark, text, static, frozen, duplicate limbs"

**Output Formats:**
- Animated WEBP (quality: 80)
- WEBM video (VP9 codec, CRF: 13.33)

**Usage:**
1. Load an input image using the LoadImage node
2. Adjust the positive prompt to describe the desired motion/animation
3. The workflow will run two sampling passes for optimal quality
4. Outputs will be saved as both WEBP and WEBM formats

**Notes:**
- This workflow is optimized for Mac MPS (Metal Performance Shaders)
- Uses quantized GGUF models (Q5_K_M) for efficient memory usage
- The two-pass approach (high noise followed by low noise) produces smoother animations

## Example Images

The `images/` folder contains example input images:
- `model.jpeg` - Portrait for image-to-video conversion
- `background.jpeg` - Background reference
- `movements.jpeg` - Movement reference

## Setup

1. Install ComfyUI and required custom nodes:
   - ComfyUI-GGUF support
   - Wan 2.2 I2V nodes

2. Download the required model files and place them in the appropriate directories

3. Load the workflow JSON files in ComfyUI

4. Adjust prompts and parameters as needed

## Tips

- For ImageToVideo workflow, keep input images at 288x512 for best results
- Experiment with different frame counts (default 81) for shorter/longer animations
- Adjust CFG values if animations are too literal (lower) or not following prompt enough (higher)
- The negative prompt is crucial for avoiding common video artifacts like flickering or distortion
