# Wan 2.2 Image-to-Video ComfyUI Workflow

A complete guide to the Wan 2.2 I2V (Image-to-Video) workflow running on Mac Apple Silicon using GGUF quantized models.

---

## Overview

This workflow takes a **still image** and generates a short **video clip** of that subject moving, guided by a text prompt. It uses Wan 2.2's Mixture-of-Experts (MoE) architecture with two specialized diffusion models — one for early (high noise) denoising and one for late (low noise) denoising.

**Workflow file:** `wan22_i2v_gguf.json`

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| RAM (Apple Silicon) | 32GB | 48GB+ |
| Storage | 40GB free | 60GB free |
| OS | macOS 13+ | macOS 14+ |
| ComfyUI | v0.8.36+ | latest |

> ⚠️ fp8 models are NOT supported on Apple Silicon MPS. Use GGUF quantized models instead.

---

## Model Dependencies

Download all files before running the workflow.

### Text Encoder
| File | Size | Destination | Link |
|---|---|---|---|
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | ~6.7GB | `models/text_encoders/` | [Download](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/blob/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors) |

### Diffusion Models (GGUF - Mac compatible)
| File | Size | Destination | Link |
|---|---|---|---|
| `Wan2.2-I2V-A14B-HighNoise-Q5_K_M.gguf` | ~10.8GB | `models/diffusion_models/` | [Download](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/main/HighNoise/Wan2.2-I2V-A14B-HighNoise-Q5_K_M.gguf) |
| `Wan2.2-I2V-A14B-LowNoise-Q5_K_M.gguf` | ~10.8GB | `models/diffusion_models/` | [Download](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/main/LowNoise/Wan2.2-I2V-A14B-LowNoise-Q5_K_M.gguf) |

### VAE
| File | Size | Destination | Link |
|---|---|---|---|
| `wan_2.1_vae.safetensors` | ~800MB | `models/vae/` | [Download](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors) |

> ⚠️ Use `wan_2.1_vae.safetensors` for the 14B models. The `wan2.2_vae.safetensors` is only for the 5B model.

### Required Custom Node
- **ComfyUI-GGUF** by city96 (v1.1.10+)
- Install via: ComfyUI Manager → Install Custom Nodes → search "ComfyUI-GGUF"

---

## Workflow Nodes Explained

### 1. CLIPLoader — T5 Text Encoder
**Node type:** `CLIPLoader`  
**Settings:** `type: wan`

Converts your text prompt into numerical embeddings the diffusion model can understand. Uses the T5-XXL architecture. The `wan` type must be selected (available in ComfyUI v0.8.36+).

---

### 2. CLIPTextEncode × 2 — Prompts
**Node type:** `CLIPTextEncode`

- **Positive prompt** — describes what you want to see and happen
- **Negative prompt** — describes what to avoid (blur, distortion, static frames)

Both connect to the same CLIPLoader output.

**Prompt tips:**
- Be descriptive about motion: *"fluid dancing motion, smooth movement"*
- Describe the subject clearly: *"woman in athletic wear"*
- Add cinematic quality words: *"cinematic, high quality, detailed"*

---

### 3. UnetLoaderGGUF × 2 — Diffusion Models
**Node type:** `UnetLoaderGGUF` (from ComfyUI-GGUF custom node)

Wan 2.2 uses a **Mixture of Experts (MoE)** architecture with two separate 14B parameter models:

- **High Noise model** — handles early denoising steps (steps 0-10). Focuses on overall structure, composition, and rough motion layout.
- **Low Noise model** — handles late denoising steps (steps 10-20). Focuses on fine details, texture, facial features, and smooth motion.

> The two-model approach allows the total parameter count to be ~27B while only running ~14B active parameters per step, keeping memory usage manageable.

---

### 4. ModelSamplingSD3 × 2 — Noise Schedule
**Node type:** `ModelSamplingSD3`  
**Setting:** `shift: 8.0`

Adjusts the sigma (noise level) schedule for each model. The value `8.0` defines the SNR (Signal-to-Noise Ratio) boundary point where the workflow hands off from the high noise model to the low noise model.

---

### 5. LoadImage — Input Image
**Node type:** `LoadImage`

Your starting image. The video will begin from this frame and animate forward. For best results:
- Use portrait orientation for portrait videos
- Higher resolution input = better quality output
- Centered, clear subject works best

---

### 6. VAELoader — Variational Autoencoder
**Node type:** `VAELoader`  
**File:** `wan_2.1_vae.safetensors`

The VAE compresses images into a compact "latent space" for efficient processing:
- **Encode** — converts input image to latent representation
- **Decode** — converts finished latent back to video frames at the end

Working in latent space is ~8x more efficient than working directly with pixels.

---

### 7. WanImageToVideo — Video Setup
**Node type:** `WanImageToVideo`

The conductor node that prepares the latent space for video generation. Key settings:

| Setting | Value | Notes |
|---|---|---|
| `width` | 288 | Portrait test (swap with height for landscape) |
| `height` | 512 | Portrait orientation |
| `length` | 81 | Number of frames (~5 sec at 16fps) |
| `batch_size` | 1 | Number of videos per run |

**Frame count guide:**
| Frames | Duration | Est. time (48GB Mac) |
|---|---|---|
| 33 | ~2 sec | ~25 min |
| 81 | ~5 sec | ~60 min |
| 121 | ~7.5 sec | ~90 min |

---

### 8. KSamplerAdvanced × 2 — Video Generation
**Node type:** `KSamplerAdvanced`

Where the actual generation happens. The diffusion process progressively removes noise over multiple steps guided by the text prompt.

**First pass (High Noise):**
| Setting | Value | Meaning |
|---|---|---|
| `add_noise` | enable | Start from pure noise |
| `steps` | 20 | Total steps in schedule |
| `cfg` | 3.5 | Prompt adherence strength |
| `sampler` | euler | Denoising algorithm |
| `scheduler` | simple | Step size schedule |
| `start_at_step` | 0 | Begin at step 0 |
| `end_at_step` | 10 | Stop at step 10 |
| `return_with_leftover_noise` | enable | Pass to second sampler |

**Second pass (Low Noise):**
| Setting | Value | Meaning |
|---|---|---|
| `add_noise` | disable | Continue from first pass |
| `steps` | 20 | Same total schedule |
| `cfg` | 3.5 | Same prompt adherence |
| `start_at_step` | 10 | Continue from step 10 |
| `end_at_step` | 10000 | Run to completion |
| `force_full_denoise` | disable | Normal completion |

**CFG guidance:**
- `2.0-3.0` — more creative, less prompt adherence
- `3.5` — balanced (recommended)
- `5.0+` — strict prompt adherence, can look over-saturated

---

### 9. VAEDecode — Render Frames
**Node type:** `VAEDecode`

Converts the finished latent tensor back into actual video frames (pixels). This step runs after both KSampler passes complete. Takes ~2 minutes on Mac.

---

### 10. SaveAnimatedWEBP + SaveWEBM — Output
Two output nodes saving the same frames in different formats:

| Node | Format | Best for |
|---|---|---|
| `SaveAnimatedWEBP` | Animated WEBP | Quick preview, sharing |
| `SaveWEBM` | VP9 video | Playback, editing, sharing |

**Output location:**
```
/Users/alexisroos/Documents/ComfyUI/output/
```

---

## Data Flow Diagram

```
[Text Prompt]
      │
      ▼
[CLIPLoader] ──────────────────────────────────────────┐
      │                                                 │
      ▼                                                 ▼
[Positive Prompt]                              [Negative Prompt]
      │                                                 │
      └─────────────────────┬───────────────────────────┘
                            │
[LoadImage] ──► [WanImageToVideo] ◄── [VAELoader]
                            │
                            ▼
                     [Latent Space]
                            │
                            ▼
[HighNoise GGUF] ► [ModelSamplingSD3] ► [KSampler Pass 1] (steps 0-10)
                                                │
                                                ▼
[LowNoise GGUF] ► [ModelSamplingSD3] ► [KSampler Pass 2] (steps 10-20)
                                                │
                                                ▼
                                         [VAEDecode]
                                                │
                                    ┌───────────┴───────────┐
                                    ▼                       ▼
                             [SaveWEBP]              [SaveWEBM]
```

---

## Performance Tips for Mac

### Before each run:
1. Log off and back on to clear wired/compressed memory
2. Close all unnecessary apps and browser tabs
3. Open only ComfyUI + one browser tab
4. Target <35GB memory used before starting

### Memory usage breakdown:
| Component | Memory |
|---|---|
| High Noise GGUF model | ~10.4GB |
| Low Noise GGUF model | ~10.4GB |
| T5 text encoder | ~6.4GB |
| VAE | ~0.2GB |
| Latents (81 frames, 512x288) | ~2GB |
| macOS wired/overhead | ~14GB |
| **Total** | **~43GB** |

### Why fp8 doesn't work on Mac:
Apple's MPS (Metal Performance Shaders) backend does not support the `Float8_e4m3fn` dtype. GGUF quantization works because it dequantizes to float16 at runtime which MPS supports.

### Speed expectations:
| Condition | Seconds/step |
|---|---|
| Clean memory, no swap | ~71s |
| Some swap (~2GB) | ~150-220s |
| Heavy swap (4GB+) | ~400-600s |

---

## Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `KeyError: 't5xxl'` | Wrong CLIP node (DualCLIPLoader) | Use CLIPLoader with type `wan` |
| `Float8_e4m3fn not supported on MPS` | fp8 models on Mac | Use GGUF models instead |
| `expected 36 channels but got 64` | Wrong VAE | Use `wan_2.1_vae.safetensors` not `wan2.2_vae` |
| Out of memory crash | Models too large | Use GGUF Q5_K_M, close all other apps |
| Very slow (400s+/step) | Swap memory in use | Log off/on to clear memory before run |

---

## Model Format Comparison

| Format | Size | Mac MPS | Quality | Notes |
|---|---|---|---|---|
| bf16 | ~27GB each | ✅ | Best | Too large for 48GB (crashes) |
| fp8 | ~14GB each | ❌ | Very good | Not supported on MPS |
| GGUF Q5_K_M | ~10.8GB each | ✅ | Good | **Recommended for Mac** |
| GGUF Q4_K_M | ~9.6GB each | ✅ | Decent | If memory is tight |
| GGUF Q3_K_M | ~7.2GB each | ✅ | Lower | Last resort |

---

## Version Info

| Component | Version |
|---|---|
| ComfyUI | v0.8.36 |
| ComfyUI-GGUF | v1.1.10 |
| Wan model | 2.2 I2V A14B |
| GGUF quantization | Q5_K_M |
| macOS | Apple Silicon |
