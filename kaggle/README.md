# Kaggle GPU kernels

Standalone GPU training/benchmark scripts for the heavier studies of the SP-PINN
paper. Each kernel is **self-contained** — it re-declares the laser field and the
relativistic Hamiltonian inline rather than importing `relsim`, so it runs on a
fresh Kaggle worker with no project dependencies. Each subdirectory is a complete,
push-ready kernel (`<script>.py` + `kernel-metadata.json`).

| Kernel | Script | What it computes |
|---|---|---|
| `bench/` | `bench.py` | GPU throughput of Boris / RK4 / Tao pushers vs ensemble size `N` (batched, float64/float32). |
| `scaling/` | `scaling.py` | Surrogate accuracy **scaling law** `eps_theta(N_collocation, width)` and the leave-one-out **ablation** grid. |
| `multiseed/` | `multiseed.py` | Multi-seed (3 seeds) error bars on the scaling law. |
| `arch/` | `arch.py` | **Architecture sweep** (tanh+Fourier vs SIREN vs modified-MLP), then the winner at higher budget over 3 seeds; saves `best_model.pt`. |
| `param/` | `param.py` | **Parametric** surrogate: one network for the pulse-duration family `tau in [20,40]`; evaluates `eps_theta(tau)`. |

## How to run

These scripts are designed to be smoke-tested locally and then pushed to Kaggle:

```bash
# 1) local smoke test (CPU, seconds) — catches bugs before a real GPU run
SMOKE=1 python bench/bench.py

# 2) push to Kaggle (requires the Kaggle CLI configured; see below)
kaggle kernels push -p bench
kaggle kernels status  <user>/sp-pinn-bench
kaggle kernels output  <user>/sp-pinn-bench -p ./out
```

Outputs (`results.json`, `*.png`, and for `arch`/`param` a `*.pt` checkpoint) are
written to `/kaggle/working` on Kaggle and to the current directory locally.

## Infrastructure notes

- `kernel-metadata.json` sets `enable_gpu: true` and **`enable_internet: true`**.
  Internet is required because Kaggle sometimes assigns a P100 (sm_60) node whose
  preinstalled PyTorch lacks a Pascal kernel image; each script reinstalls the
  cu121 wheel (`pip install --force-reinstall --no-deps torch --index-url
  https://download.pytorch.org/whl/cu121`) before `import torch`.
- The `id` in each `kernel-metadata.json` is `<kaggle-username>/sp-pinn-<name>`.
  Replace the username with your own Kaggle account to push under your namespace.

## Credentials — IMPORTANT

These kernels are pushed with the [Kaggle CLI](https://github.com/Kaggle/kaggle-api),
which reads your **personal** API token from `~/.kaggle/kaggle.json`
(`{"username": "...", "key": "..."}`). **No API token is included in this
repository, and none should ever be committed** — the `.gitignore` explicitly
excludes `kaggle*.txt`, `*.token`, `*.key`, and `.kaggle/`. Configure your own token
locally (Kaggle account → Settings → Create New Token) before pushing.
