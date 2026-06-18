# Training the PyTorch DQN Model

## Context

ConstrictorsCoil has two DQN solvers:
- `dqn` — TensorFlow 1.x solver (existing, integrated with GUI)
- `dqn_torch` — PyTorch Double DQN solver (new, needs a trained model to run)

The PyTorch solver needs a `model.pt` file in the project root before it can be used for play.
The standalone training script is `train.py`.

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (RTX 5070 Ti confirmed to work)
- ~8-10 hours of GPU time for full 100k episode run

## Step 1 — Clone and enter the repo

```bash
git clone https://github.com/makhskham/ConstrictorsCoil.git
cd ConstrictorsCoil
```

## Step 2 — Create venv and install dependencies

```bash
uv venv .venv
.venv\Scripts\activate
```

Install PyTorch with CUDA (check your CUDA version first with `nvidia-smi`):

```bash
# CUDA 12.8 (RTX 5070 Ti)
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# Or let PyPI pick the right wheel automatically
uv pip install torch
```

Install the rest:

```bash
uv pip install numpy matplotlib pytest
```

## Step 3 — Verify GPU is detected

```python
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Must print `True` and the GPU name. If it prints `False`, PyTorch cannot see the GPU — fix CUDA/driver setup before training.

## Step 4 — Run training

```bash
python train.py
```

Optional flags:

```bash
python train.py --episodes 100000 --out model.pt
```

Progress prints every 200 episodes:

```
Episode:  50000 | Avg reward/score:  44.97 / 17.56 | Elapsed:   67 mins | Device: cuda
```

## Step 5 — Save the model

When training finishes, output is saved as `model.pt.tmp.<timestamp>`.
Rename it:

```bash
mv model.pt.tmp.<timestamp> model.pt
```

Then commit:

```bash
git add model.pt
git commit -m "Add trained DQN model"
git push origin master
```

Commit must be authored with `mkhamzaliyeva@gmail.com`:

```bash
git config user.email "mkhamzaliyeva@gmail.com"
git config user.name "Makhsuma Khamzaliyeva"
```

## Step 6 — Play with the trained model

```bash
python run.py -s dqn_torch
```

## Notes

- Training targets: avg score ~17-18 at 50k episodes, ~22-23 at 100k episodes
- The model uses an 8x8 grid (64 cells), Double DQN, epsilon decay over 50k episodes
- `model.pt` must be in the project root (same folder as `run.py`)
- Do not mention or reference any old GitHub accounts in commits or code
- All commits must use email `mkhamzaliyeva@gmail.com`
