# video-experiments

Video & 3D generation workflows: ComfyUI (image generation, image-to-video) and TRELLIS.2 (image-to-3D).

## How it works

This is a documentation repository containing:

| Component | Type | Platform | Purpose |
|-----------|------|----------|---------|
| ComfyUI Image Generation | Workflow JSON | Local ComfyUI | SDXL portrait generation |
| ComfyUI Image-to-Video | Workflow JSON | Local ComfyUI (Mac MPS) | Wan 2.2 I2V animation |
| TRELLIS.2 Image-to-3D | Jupyter Notebook | Google Colab (A100) | 2D image → 3D mesh |

See README.md for quick start. See detailed guides for deep dives:
- `wan22_i2v_workflow_docs.md` (if exists, otherwise check README)
- `trellis2_docs.md`

## Workflows

### ComfyUI Workflows
- **Location:** `comfyui/workflows/*.json`
- **Usage:** Load in ComfyUI → adjust prompts → queue
- **Models:** Download separately (see README requirements)

### TRELLIS.2 Notebook
- **Location:** `trellis.2/TRELLIS2_A100_v5.ipynb`
- **Usage:** Upload to Colab → follow cell-by-cell guide
- **Models:** Auto-downloaded to Drive (~12GB)

## File layout

```
video-experiments/
├── comfyui/
│   └── workflows/
│       ├── ImageGeneration.json
│       └── ImageToVideo.json
├── trellis.2/
│   └── TRELLIS2_A100_v5.ipynb
├── images/              ← example input images
├── videos/              ← example output videos
├── README.md            ← project overview
├── trellis2_docs.md     ← TRELLIS.2 detailed guide
└── CLAUDE.MD            ← this file
```

## Git rules

- Commit workflow JSON files, notebooks, and documentation
- Commit example images/videos that are reasonable size (<20MB)
- Do NOT commit large model files (*.safetensors, *.gguf, *.ckpt) — use .gitignore
- Do NOT commit generated outputs from local runs

## Tools & dependencies

### ComfyUI Workflows
- **ComfyUI:** Local installation
- **Custom nodes:** ComfyUI-GGUF for Wan 2.2
- **Models:** See README requirements (Juggernaut-XL, Wan 2.2 GGUF models)
- **Platform:** Mac (MPS) or NVIDIA GPU

### TRELLIS.2 Notebook
- **Platform:** Google Colab Pro/Pro+ (A100 required)
- **HuggingFace:** Account with gated model access
- **Storage:** Google Drive (~12GB for models)
- **Dependencies:** Auto-installed in notebook (FlexGEMM, CuMesh, nvdiffrast, etc.)
