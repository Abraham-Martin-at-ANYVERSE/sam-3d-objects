# SAM 3D Objects — Agent Instructions

SAM 3D Objects is a foundation model for 3D object/scene reconstruction from single images, producing Gaussian Splats (`.ply`) and triangle meshes. See [README.md](README.md) for project overview.

## Environment Setup

**Prerequisites**: Linux 64-bit, NVIDIA GPU ≥32 GB VRAM, CUDA 12.1. Full setup in [doc/setup.md](doc/setup.md).

```bash
mamba env create -f environments/default.yml
mamba activate sam3d-objects

export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
pip install -e '.[dev]'
pip install -e '.[p3d]'

export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'
./patching/hydra   # required patch for hydra bug
```

Model checkpoints must be downloaded separately from HuggingFace (`facebook/sam-3d-objects`); see [doc/setup.md](doc/setup.md).

## Running Demos

```bash
python demo.py                # single-object reconstruction
# Notebooks in notebook/: demo_single_object.ipynb, demo_multi_object.ipynb
```

## Running Tests

```bash
pytest                        # test files follow *_test.py pattern
```

## Formatting & Linting

```bash
black .                       # formatter (line length 80)
flake8 .                      # linter
autoflake --remove-all-unused-imports -i -r .
```

## Architecture

| Module | Role |
|--------|------|
| `sam3d_objects/pipeline/` | Inference orchestration. `InferencePipelinePointMap` is the main pipeline class. |
| `sam3d_objects/model/backbone/tdfy_dit/` | Core DIT-based 3D generation model (shape + layout generators). |
| `sam3d_objects/model/backbone/generator/` | Base generator, classifier-free guidance, flow matching, shortcut modules. |
| `sam3d_objects/data/dataset/tdfy/` | Dataset preprocessing: image/mask transforms, pose targets. |
| `sam3d_objects/config/` | Hydra/OmegaConf config helpers. |
| `sam3d_objects/utils/visualization/` | Scene and mesh visualization (PyTorch3D, Plotly). |
| `notebook/inference.py` | **Public API**: `Inference(config_file)` → `__call__(image, mask, seed)`. |

**Inference data flow**: image+mask → preprocessing (pad, resize, rembg) → SS generator (shape) → SLAT generator (layout) → Gaussian/mesh decoder → `.ply` output.

## Key Conventions

- **Indentation**: 2 spaces (not 4, not tabs). See [CONTRIBUTING.md](CONTRIBUTING.md).
- **Line length**: 80 characters max.
- **Config**: Hydra 1.3 + OmegaConf. Config files live in `checkpoints/{tag}/pipeline.yaml`. Instantiation via `hydra.utils.instantiate`.
- **`LIDRA_SKIP_INIT`**: Set this env var to skip lazy model initialization (useful for import-only tasks).
- **Sparse tensors**: `spconv-cu121` is used throughout the model backbone — handle carefully when modifying sparse operations.
- **flash_attn**: Only activated on A100/H100/H200 GPUs; code must gracefully fall back when unavailable.
- **Test files**: Named `*_test.py`; `conftest.py` files are excluded from the built package.

## Dependency Groups (pyproject.toml)

| Extra | Contents |
|-------|----------|
| `p3d` | pytorch3d, flash_attn |
| `inference` | kaolin, gsplat, gradio, seaborn |
| `dev` | pytest, black, flake8, autoflake, findpydeps |

## Common Pitfalls

- **GPU memory**: ≥32 GB VRAM required. Reduce batch sizes or use `compile=False` on smaller GPUs.
- **Two-step torch install**: pytorch3d must be installed in a separate `pip install -e '.[p3d]'` pass due to dependency resolution order.
- **Hydra patch**: `./patching/hydra` must be applied after install or certain config instantiation will fail.
- **NVIDIA index**: Always set `PIP_EXTRA_INDEX_URL` before installing CUDA-specific packages.
- **Platform**: `environments/default.yml` targets `linux-64`; macOS/Windows not officially supported.
