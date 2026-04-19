# DCGAN on CIFAR-10 — Lab 1: Media generation via DL models

A Deep Convolutional GAN (Radford et al., 2015) that learns to generate new
32×32 RGB images that look like CIFAR-10 photos.

The full implementation lives in a single, heavily commented notebook:
[`dcgan_cifar10.ipynb`](./dcgan_cifar10.ipynb).

## What's inside

- A short theory recap of the GAN minimax game and the DCGAN architectural
  rules of thumb.
- One config cell at the top — including a `TARGET_CLASS` switch so you can
  train on a single CIFAR-10 class (fast, cleaner samples) or on the full
  10-class training set.
- A PyTorch implementation of the Generator and Discriminator, the
  paper-standard weight init, and a stable training loop with mild label
  smoothing and a fixed-noise sample grid saved every epoch.
- Loss curves, a final 8×8 grid of generated images, and saved model
  checkpoints (`checkpoints/generator.pt`, `checkpoints/discriminator.pt`).

## Requirements

- Python 3.11 or 3.12 (the project is pinned to 3.12 in `.python-version`).
- [`uv`](https://github.com/astral-sh/uv) for dependency management. All deps
  are pinned in [`uv.lock`](./uv.lock); the source declarations are in
  [`pyproject.toml`](./pyproject.toml).

The default install is **CPU-only PyTorch**. For CUDA, swap the `torch` /
`torchvision` lines in `pyproject.toml` for the matching CUDA wheels
(see <https://pytorch.org/get-started/locally/>) and re-run `uv lock`.

## Setup

```powershell
# Install pinned dependencies into a local .venv
uv sync

# Launch the notebook
uv run jupyter notebook dcgan_cifar10.ipynb
```

Or, in VS Code / Cursor: open `dcgan_cifar10.ipynb`, pick the
`.venv` interpreter created by `uv sync`, and `Run All`.

## Configuration

Open the **Hyperparameters** cell (section 3 of the notebook) and edit:

| Variable        | Default     | Notes                                                            |
| --------------- | ----------- | ---------------------------------------------------------------- |
| `TARGET_CLASS`  | `"horse"`   | Any of the 10 CIFAR-10 class names, or `None` to use all classes |
| `EPOCHS`        | `30`        | 30 epochs / single class is the demo default                     |
| `BATCH_SIZE`    | `128`       | Lower if you run out of memory                                   |
| `NZ`            | `100`       | Latent vector length                                             |
| `NGF` / `NDF`   | `64` / `64` | Generator / discriminator feature widths                         |
| `LR` / `BETA1`  | `2e-4`/`0.5`| DCGAN paper defaults                                             |

## Expected runtime

| Hardware             | `TARGET_CLASS="horse"` (5k images, 30 epochs) | `TARGET_CLASS=None` (50k images, 30 epochs) |
| -------------------- | --------------------------------------------- | ------------------------------------------- |
| Modern laptop CPU    | ~10–20 min                                    | ~1.5–3 hours                                |
| Single consumer GPU  | < 2 min                                       | ~10–20 min                                  |

For the all-classes run on a GPU, bump `EPOCHS` to 80–150 to get sharper
samples.

## Outputs

- `samples/epoch_XXX.png` — 8×8 grid of fakes from a fixed noise vector,
  written every epoch so you can flip through them as a stop-motion of training.
- `samples/final.png` — final 8×8 grid from a fresh noise vector.
- `checkpoints/generator.pt`, `checkpoints/discriminator.pt` — trained weights.
- `data/` — CIFAR-10 download cache (auto-populated on first run).

All four directories are git-ignored.

## Project layout

```
.
├── dcgan_cifar10.ipynb   # main deliverable: the notebook
├── main.py               # tiny CLI pointer to the notebook
├── pyproject.toml        # dependency declarations
├── uv.lock               # pinned lock file
├── .python-version       # 3.12 (used by uv / pyenv)
└── README.md             # you are here
```
