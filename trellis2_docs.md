# TRELLIS.2 Image-to-3D Pipeline

A complete guide to running Microsoft's TRELLIS.2 image-to-3D generation model on Google Colab with A100 GPU.

---

## Overview

TRELLIS.2 is a research model released by Microsoft in early 2025 that converts **2D images into 3D models (GLB/mesh format)**. It uses a Mixture-of-Experts (MoE) architecture with sparse and dense attention mechanisms.

**Notebook file:** `trellis.2/TRELLIS2_A100_v5.ipynb`

---

## What It Does

Input: A 2D image (PNG, JPEG, etc.)
Output: 3D mesh file (.glb) with geometry and optional colors

The model:
1. Removes background using BiRefNet (background removal)
2. Extracts image features using DINOv3
3. Generates sparse voxel structure via flow model
4. Refines details with diffusion model
5. Converts to mesh with texture attributes

---

## Hardware Requirements

| Component | Requirement |
|---|---|
| GPU | A100 (40GB VRAM) |
| RAM | 12GB+ system RAM |
| Storage | 40GB free (60GB recommended) |
| Platform | Google Colab Pro/Pro+ |

> ⚠️ This model requires an A100 GPU. T4 and V100 GPUs do not have sufficient VRAM.

---

## Prerequisites

### 1. Google Colab Setup
- Runtime → Change runtime type → **A100 GPU**
- Colab Pro or Pro+ subscription (for A100 access)

### 2. HuggingFace Account
- Create account at huggingface.co
- Generate access token with "Read access to public gated repos"
- Request access to gated models:
  - `facebook/dinov3-vitl16-pretrain-lvd1689m`
  - `ZhengPeng7/BiRefNet`

### 3. Google Drive
- Mount Drive to persist models between sessions (~12GB)
- Recommended folder structure:
  ```
  /MyDrive/TRELLIS.2-4B/          (model weights, ~12GB)
  /MyDrive/TRELLIS2_extensions/   (compiled CUDA extensions)
  ```

---

## Session Workflow

### First Time (~35 minutes)
1. Cell 1: Verify GPU
2. Cell 2: Login to HuggingFace
3. Cell 3: Mount Google Drive
4. Cell 4: Install dependencies & compile CUDA extensions
5. **Restart runtime** (Runtime → Restart session)
6. Cell 2: Login again
7. Cell 3: Mount Drive again
8. Cell 5: Download model weights (~12GB)
9. Cell 6: Patch BiRefNet cache
10. Cell 7: Load pipeline (~2-5 minutes)
11. Cell 8: Run inference on image
12. Cell 9: Export & download GLB

### Subsequent Sessions (~8 minutes)
1. Cell 1: Verify GPU
2. Cell 2: Login to HuggingFace
3. Cell 3: Mount Google Drive
4. Cell 4: Restore dependencies from Drive
5. **Restart runtime**
6. Cell 2 → Cell 3 → Cell 6 → Cell 7 → Cell 8 → Cell 9

---

## Why So Many Patches?

TRELLIS.2 is a research release that requires extensive modifications to run in Colab:

### 1. Missing Dependencies
- `sparse_structure_decoder` checkpoint from original TRELLIS (not included in TRELLIS.2)
- Must download separately and patch pipeline config

### 2. Gated Models
- DINOv3 and RMBG-2.0 require HuggingFace approval
- BiRefNet used instead of RMBG-2.0 (publicly accessible)

### 3. Dtype Mismatches
- Model uses `bfloat16` internally
- Colab PyTorch defaults to `float32`/`float16`
- Multiple patches required to cast tensors at module boundaries:
  - Linear layers
  - Conv3d layers
  - LayerNorm operations
  - Patch embeddings

### 4. Custom CUDA Extensions
Must be compiled from source (~20 minutes first time):
- **FlexGEMM** - Flexible sparse convolution operations
- **CuMesh** - CUDA mesh operations
- **o-voxel** - Voxel operations
- **nvdiffrast** - Differentiable rendering

### 5. `manual_cast` Bug
Utility function skips dtype casting when PyTorch autocast is active, causing NaN values. Fixed by forcing cast regardless of autocast state.

### 6. `sparse_structure_decoder` Incompatibility
Original TRELLIS decoder produces all-negative values with TRELLIS.2's flow model output. Fixed by using percentile threshold (top 50%) instead of `> 0`.

### 7. Attention Backend
- Sparse attention requires xformers
- Dense attention works with sdpa (scaled dot-product attention)

---

## Key Components

### Models Required

| File | Size | Purpose | Location |
|---|---|---|---|
| `TRELLIS.2-4B/` | ~12GB | Main model weights | `models/` |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | ~6.7GB | T5 text encoder | Included in TRELLIS.2-4B |
| `ss_dec_conv3d_16l8_fp16.safetensors` | ~200MB | Sparse structure decoder | Downloaded from original TRELLIS |
| DINOv3 ViT-L/16 | Auto-downloaded | Image feature extraction | HuggingFace cache |
| BiRefNet | Auto-downloaded | Background removal | HuggingFace cache |

### Pipeline Types

The model supports different quality/speed tradeoffs:

| Type | Resolution | Quality | Time | VRAM |
|---|---|---|---|---|
| `512` | 512³ voxels | Good | ~5 min | ~35GB |
| `1024` | 1024³ voxels | Better | ~12 min | ~38GB |
| `1024_cascade` | Two-stage | Better | ~15 min | ~38GB |
| `1536_cascade` | Two-stage | Best | ~25 min | ~40GB |

---

## Cell-by-Cell Guide

### Cell 1: Verify GPU
Checks that you have an A100 with CUDA support.

**Expected output:**
```
CUDA: True
GPU: NVIDIA A100-SXM4-40GB
VRAM: 40.5 GB
```

---

### Cell 2: HuggingFace Login
Authenticates with HuggingFace to access gated models. Paste your HF access token when prompted.

---

### Cell 3: Mount Google Drive
Mounts Drive at `/drive/MyDrive/` for persistent storage of models and extensions.

---

### Cell 4: Install Dependencies & Patch Source
The longest cell (~35 minutes first time, ~5 minutes if cached on Drive).

**What it does:**
1. Clones TRELLIS.2 repository
2. Installs Python packages (xformers, kornia, timm, transformers, etc.)
3. Compiles CUDA extensions:
   - nvdiffrast (fast, from git)
   - FlexGEMM (slow, ~10 mins)
   - CuMesh (slow, ~8 mins)
   - o-voxel (medium, ~2 mins)
4. Applies source file patches:
   - **Patch 1:** `manual_cast` - always cast regardless of autocast
   - **Patch 2:** `timestep_embedding` - preserve input dtype
   - **Patch 3:** BiRefNet wrapper - cast input to model dtype

**⚠️ After this cell completes:**
- **Runtime → Restart session**
- Patches are written to disk but won't take effect until restart
- Run Cell 2 → 3 → 5 after restart

---

### Cell 5: Download Model Weights
Downloads TRELLIS.2-4B (~12GB) and sparse_structure_decoder from HuggingFace.

**Skipped if already on Drive:**
- `/drive/MyDrive/TRELLIS.2-4B/`
- `/content/models/TRELLIS-extra/ckpts/ss_dec_conv3d_16l8_fp16.safetensors`

---

### Cell 6: Patch BiRefNet Cache
BiRefNet model is downloaded by HuggingFace transformers to cache. This cell patches the cached source file to add dtype casting at patch_embed layers.

**Must run every session** (cache is cleared on session end).

---

### Cell 7: Load Pipeline
Loads the full TRELLIS.2 pipeline into VRAM (~2-5 minutes).

**What it does:**
1. Clears cached Python modules
2. Creates stubs for o_voxel (to avoid import errors)
3. Applies runtime patches:
   - Linear layer dtype safety
   - LayerNorm dtype safety
   - Conv3d dtype safety
4. Patches transformers loading report (ignores mismatched size warnings)
5. Updates `pipeline.json` to point to sparse_structure_decoder
6. Loads all models to CUDA with bfloat16
7. Patches `sample_sparse_structure` to use percentile threshold
8. Sets attention backends (sparse=xformers, dense=sdpa)

**Memory usage after load:**
- Models: ~30GB
- Workspace: ~8GB
- Total: ~38GB / 40GB used

---

### Cell 8: Run Inference
Upload an image and generate 3D model.

**Settings:**
- `pipeline_type`: `'512'` (recommended), `'1024'`, `'1024_cascade'`, `'1536_cascade'`

**Generation time:** 5-25 minutes depending on pipeline type

**Output:** Mesh object with:
- `vertices`: (N, 3) coordinates
- `faces`: (M, 3) triangle indices
- `attrs`: (M, C) per-face attributes (color, roughness, metallic)
- `layout`: Dictionary mapping attribute names to channel indices

---

### Cell 9: Export & Download GLB
Converts mesh to GLB format and downloads.

**Two exports:**
1. **Geometry-only GLB** - Always works, no colors (~5-15MB)
2. **Colored GLB** - Attempts per-vertex color mapping (~5-15MB)

**Color mapping:**
- Face attributes (per-triangle colors) averaged to vertices
- Includes base_color, roughness, metallic if available
- May fail if attribute layout is incompatible

---

## Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `RuntimeError: CUDA out of memory` | Not enough VRAM | Use smaller pipeline type (`'512'`) |
| `KeyError: 'sparse_structure_decoder'` | Missing checkpoint | Run Cell 5 to download |
| `dtype mismatch` errors | Patches not applied | Ensure Cell 4 completed, then restart runtime |
| `ImportError: flex_gemm` | Extension not compiled | Re-run Cell 4, check for compile errors |
| BiRefNet crashes | Cache not patched | Run Cell 6 every session |
| All-empty mesh output | Decoder threshold bug | Cell 7 patches this (percentile threshold) |
| `Float8_e4m3fn not supported` | Wrong dtype | Should not occur (model uses bf16), check patches |

---

## Performance Tips

### Optimize Memory
1. Use `pipeline_type='512'` for testing
2. Clear CUDA cache before running: `torch.cuda.empty_cache()`
3. Use `pipeline.low_vram = True` if running out of memory (slower)

### Speed Up Subsequent Runs
1. Keep models on Google Drive (`/drive/MyDrive/TRELLIS.2-4B/`)
2. Keep compiled extensions on Drive (`/drive/MyDrive/TRELLIS2_extensions/`)
3. Both will be auto-restored in Cell 4 (~5 minutes vs ~35 minutes)

### Batch Processing
To process multiple images without reloading:
```python
for img_path in image_paths:
    image = Image.open(img_path).convert('RGB')
    outputs = pipeline.run(image, pipeline_type='512')
    # Export mesh...
```

---

## Architecture Notes

### Flow Model
TRELLIS.2 uses a **Rectified Flow** model for sparse structure generation. This is different from standard diffusion:
- Learns direct interpolation between noise and data
- Deterministic sampling (no stochastic noise)
- Faster convergence (fewer steps needed)

### Mixture of Experts (MoE)
Two separate 14B parameter models:
- **Sparse structure model** - Predicts which voxels are occupied
- **Refinement model** - Adds fine details and textures

Only one expert active at a time, keeping memory manageable.

### Sparse Attention
Uses xformers for efficient sparse attention on voxel grids. Computes attention only for occupied voxels, not full 3D grid.

---

## Output Format

GLB files contain:
- **Geometry:** Vertices and triangle faces
- **Materials (if colored):**
  - Base color (RGB)
  - Roughness (0-1)
  - Metallic (0-1)

Can be imported into:
- Blender (File → Import → glTF)
- Unity (drag and drop)
- Three.js / WebGL viewers
- Sketchfab (upload for web preview)

---

## Limitations

1. **Single object only** - Background is removed, focus on main subject
2. **View-dependent** - Quality depends on input image angle
3. **Texture limitations** - Generated colors may be approximate
4. **Geometric artifacts** - May produce holes or noise in complex regions
5. **No animation** - Static mesh only

---

## Version Info

| Component | Version |
|---|---|
| TRELLIS.2 | microsoft/TRELLIS.2-4B |
| Notebook | v5 |
| Platform | Google Colab (A100) |
| PyTorch | Auto-installed |
| CUDA | 12.2+ |
| Python | 3.10+ |

---

## Links

- [TRELLIS.2 GitHub](https://github.com/microsoft/TRELLIS.2)
- [TRELLIS.2 Model (HuggingFace)](https://huggingface.co/microsoft/TRELLIS.2-4B)
- [Original TRELLIS](https://github.com/microsoft/TRELLIS)
- [Paper (TRELLIS.2 if published)](https://arxiv.org/search/?query=TRELLIS)

---

## Credits

Model developed by Microsoft Research. Notebook includes extensive community patches for Colab compatibility.
