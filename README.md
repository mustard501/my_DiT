# DiT Reproduction

English | [中文](README_zh.md)

A simplified reproduction of Peebles & Xie (ICCV 2023) **Diffusion Transformers**: frozen SD KL-f8, class-conditional **DiT-S/2** on 32×32×4 latents. Backbone, loss, CFG, and 250-step respacing follow [facebookresearch/DiT](https://github.com/facebookresearch/DiT).

**Paper:**

- [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)

## Features

- Frozen official KL-f8 (CompVis `AutoencoderKL`; `sd-vae-ft-ema/mse` compatible)
- Forward diffusion; ε-prediction + LEARNED_RANGE variance; `L = MSE(ε) + VB(Σ)`
- Class conditioning: `c = t_emb + y_emb`, injected into each Transformer block via adaLN-Zero
- CFG: class-token dropout 0.1 at train time; `forward_with_cfg` when `cfg_scale≠1`, applied to the first 3 latent channels only
- Sampling: DDPM on a 250-step SpacedDiffusion (default / FID), or DDIM on the same subsequence (training previews)
- EMA 0.9999; training grids use fixed noise and `SAMPLE_CLASS_IDS`
- TensorBoard; FID vs the training ImageFolder (Inception-v3, 2048-d)

## Layout

```
my_DiT/
├── train.py              # train / recon / sample / FID
├── dit.py                # DiT-S/p (PatchEmbed, adaLN-Zero, CFG)
├── vae.py                # CompVis KL-f8 encoder/decoder and weight conversion
├── diffusion.py          # ADM schedule, loss, respace, DDPM/DDIM
├── dataset.py            # ImageFolder + ADM center-crop
├── requirements.txt
├── id_list.txt           # ImageNet-100 id ↔ class name
├── docs/
│   ├── DiT-note.html
│   └── figs/
├── data/imagenet100/
└── runs/
```

## Setup

Python 3.11+, NVIDIA driver with CUDA 12.0+.

```bash
pip install -r requirements.txt
```

VAE weights must be provided locally (not downloaded during training):

```bash
mkdir -p models/sd-vae-ft-ema
# https://huggingface.co/stabilityai/sd-vae-ft-ema/resolve/main/diffusion_pytorch_model.safetensors
# https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors
# CompVis: https://ommer-lab.com/files/latent-diffusion/kl-f8.zip
```

`--vae_ckpt` may be a `.safetensors` / `.ckpt` file or a directory containing one of the above.

## Usage

Entry point: `train.py`. Training is the default; `--recon` / `--sample` / `--eval_fid` select the other modes.

### VAE check

```bash
python train.py --recon \
  --vae_ckpt models/sd-vae-ft-ema/diffusion_pytorch_model.safetensors \
  --data_dir data/imagenet100/train \
  --out runs/vae_check
```

Writes `vae_recon.png` (top: input, bottom: `decode(mode(z))`). Falls back to random images if there is no ImageFolder.

### Training

`--data_dir` layout: `root/<class>/*.{jpg,png,webp}`. Class ids are the sorted folder names and must match `--num_classes`. For ImageNet-100 use `data/imagenet100/train`.

```bash
python train.py \
  --vae_ckpt models/sd-vae-ft-ema/diffusion_pytorch_model.safetensors \
  --data_dir data/imagenet100/train \
  --num_classes 100 \
  --out runs/dit_s2_in100
```

Effective batch `B = batch_size × (global_batch_size // batch_size)`. The paper uses `B = 256`, `lr = 1e-4`; scale linearly as `lr = 1e-4 × B / 256` (e.g. `1.25e-5` when `B = 32`).

Outputs: `ckpt.pt` (model / ema / opt / step), `samples_{step}.png`, `tb/`.

Preview classes are the first `--n_samples` entries of `SAMPLE_CLASS_IDS` in `train.py`; see `id_list.txt`.

### Resume

```bash
python train.py \
  --vae_ckpt models/sd-vae-ft-ema/diffusion_pytorch_model.safetensors \
  --data_dir data/imagenet100/train --num_classes 100 \
  --resume runs/dit_s2_in100/ckpt.pt --out runs/dit_s2_in100
```

### Sampling

Loads EMA. Default: 250-step DDPM with respacing.

```bash
python train.py --sample \
  --ckpt runs/dit_s2_in100/ckpt.pt \
  --vae_ckpt models/sd-vae-ft-ema/diffusion_pytorch_model.safetensors \
  --cfg_scale 1.5 --n_samples 8 \
  --out runs/dit_s2_in100
```

- Classes: `SAMPLE_CLASS_IDS`; `--class_label 16` fixes the whole batch to that class.
- `--sampler ddpm|ddim`, `--num_sampling_steps 250`.
- `--progress_every 10` writes `progression.png`: rows = samples, columns = `x0_hat` (left noisy → right clean).

### TensorBoard

```bash
tensorboard --logdir runs/dit_s2_in100/tb
```

`train/loss`, `train/ms_per_step`, `samples/grid`, `eval/fid`.

### FID

Computed against the training images in `--data_dir`. Paper setting: 50K images, 250-step DDPM, no CFG. Training default: `--fid_every 0`.

```bash
python train.py --eval_fid \
  --ckpt runs/dit_s2_in100/ckpt.pt \
  --vae_ckpt models/sd-vae-ft-ema/diffusion_pytorch_model.safetensors \
  --data_dir data/imagenet100/train \
  --n_fid 10000 --out runs/dit_s2_in100
```

`--fid_final` runs once after training; `--fid_every N` every N steps.

## Implementation

### Stage 1: KL-f8 encoder / decoder

Same as official DiT: Stable Diffusion KL-VAE (f=8), not VQ-GAN. The module tree follows CompVis `AutoencoderKL` and loads:

- HuggingFace `sd-vae-ft-*` (`convert_hf_vae_state_dict` maps `down_blocks` keys and Linear → 1×1 Conv)
- CompVis `kl-f8.zip` Lightning `.ckpt`

`KL_F8_DDCONFIG`: `ch=128`, `ch_mult=(1,2,4,4)`, 2 ResnetBlocks per level, 256→128→64→32. `attn_resolutions=[]` (no attention on the down/up path); bottleneck keeps `AttnBlock`. Encoder outputs 2×4 channels (mean / log-variance); `sample()` is then scaled by 0.18215. Decoder is symmetric. VAE is frozen during DiT training.

256×256 RGB → 32×32×4 latent. `--recon` uses `mode()`; diffusion training uses `sample()`.

### Transformer: DiT-S/2

`dit.py` matches official `models.py` without timm. S/2: 12 layers, `d=384`, 6 heads, `p=2`, mlp_ratio=4, ~33M parameters.

1. `PatchEmbed`: `Conv2d(4→384, k=2, s=2)` → 256 tokens.
2. Frozen 2D sin-cos (MAE; `grid_w` first).
3. `c = t_emb + y_emb`. Timestep: 256-d frequencies (cos then sin) + SiLU MLP. Labels: `Embedding(C+1, 384)`; the extra class is the CFG uncond token.
4. `DiTBlock`: affine-free LayerNorm + MHSA + GELU(tanh) MLP. Conditioning via adaLN-Zero: `SiLU→Linear(d→6d)` produces γ, β, α for attn and MLP. Modulation and FinalLayer linears are zero-init (identity at start).
5. `FinalLayer`: adaLN (2d) + Linear, unpatchify → 8×32×32 (first 4 channels ε, last 4 variance).

Attention is a handwritten `qkv` Linear + softmax with the same parameter layout as timm defaults.

### Conditioning and CFG

| Stage | Behavior |
|------|----------|
| Train `DiT.forward` | `y_embedder(y, self.training)`; when `train=True`, replace y with `num_classes` at p=0.1 |
| Training preview `quick_sample` | `diffusion_sample(..., cfg_scale=args.cfg_scale)`, default 1.0 (off) |
| `--sample` / FID | On when `--cfg_scale≠1` |

When on, `diffusion_sample` doubles the batch and labels as `[cond | uncond]` and calls `forward_with_cfg`. Guidance is applied to the first 3 output channels only (Appendix A):

```
ε̂_{1:3} = ε_uncond + s (ε_cond − ε_uncond)
```

CFG is not used in the training loss.

### Diffusion: DDPM respace vs DDIM

Training: `T=1000`, ADM linear `β = linspace(1e-4, 2e-2)`. 8-channel output; LEARNED_RANGE interpolates `[-1, 1]` to `[log β̃_t, log β_t]`. ε is detached in the VB term. `clip_denoised=False`.

All sampling first calls `make_spaced_diffusion` (ADM `SpacedDiffusion`): 250 `ᾱ` values are taken from the 1000-step chain, `β'_i = 1 − ᾱ_{t_i} / ᾱ_{t_{i-1}}`, then the reverse process runs on this coarser chain. Network timesteps are mapped back to `{0, …, 999}` via `tmap`. `--ddim_spacing` is unused.

| Call site | Sampler | Update |
|-----------|---------|--------|
| `--sample` / `--eval_fid` default | DDPM 250 | `μ_θ + σ_θ z`, using predicted Σ |
| Training `samples_{step}.png` | DDIM 250, η=0 | Deterministic; no Σ noise |

DDPM respace is a thinned Markov chain; DDIM is the non-Markov reverse process on the same subsequence. Paper FID uses the former.

## Parameters

| Flag | Default | Notes |
|------|---------|-------|
| `--patch_size` | 2 | S/2 → 256 tokens |
| `--num_classes` | 1000 | set 100 for ImageNet-100 |
| `--max_steps` | 400000 | optimizer steps |
| `--batch_size` | 32 | micro-batch |
| `--global_batch_size` | 32 | effective batch, see above |
| `--lr` | 1.25e-5 | linear in effective batch |
| `--T` / `--beta_start` / `--beta_end` | 1000 / 1e-4 / 2e-2 | ADM linear schedule |
| `--sampler` | `ddpm` | `--sample` and FID; training previews use DDIM |
| `--num_sampling_steps` | 250 | respace chain length |
| `--cfg_scale` | 1.0 | 1 = off |
| `--sample_every` | 5000 | fixed-noise preview interval |
| `--n_fid` | 50000 | generated images for FID |
| `--fid_every` | 0 | `0` disables |

## Dataset

| Item | This repo | Paper |
|------|-----------|-------|
| Dataset | [clane9/imagenet-100](https://huggingface.co/datasets/clane9/imagenet-100) | ImageNet-1K |
| Task | 100-class conditional generation | 1000-class conditional generation |
| Train images | 126689 | 1.28M |
| Resolution | short side 160, ADM crop to 256 | native ADM center-crop 256 |
| Latent | 32×32×4, ×0.18215 | same |
| Preprocess | `center_crop_arr` + hflip, [-1, 1] | same |
| FID reference | `--data_dir` training images | see paper |
| Class names | sorted folder names, `id_list.txt` | ImageNet synset order |

## Experiments

RTX 4090, DiT-S/2, ImageNet-100, batch 32, lr `1.25e-5`, 200k steps. Sampling: class 16 (ambulance), 250-step DDPM, `--cfg_scale 1.5`. FID not reported.

<p align="center">
  <img src="docs/figs/ambulance_samples_final.png" width="96%"/>
</p>
<p align="center"><em>Top: class 16 (ambulance) samples, 250-step DDPM, CFG 1.5.</em></p>

<p align="center">
  <img src="docs/figs/ambulance_progression.png" width="96%"/>
</p>
<p align="center"><em>Bottom: denoising progression. Rows = samples, columns = x0_hat, left noisy → right clean.</em></p>

## Differences from the paper

Backbone, loss, 3-channel CFG, 250-step DDPM respace, KL-f8, and adaLN-Zero match the official implementation.

| Item | This repo | Paper / official DiT |
|------|-----------|----------------------|
| Backbone | DiT-S/2, handwritten Attention/Mlp | same structure; official uses timm |
| Stage 1 | sd-vae-ft-* / kl-f8, frozen | `stabilityai/sd-vae-ft-ema` |
| Data | ImageNet-100, 160px upsampled to 256 | ImageNet-1K native crop |
| Classes | 100 | 1000 |
| Global batch | 32 | 256 (DDP) |
| Learning rate | 1.25e-5 | 1e-4 |
| Steps | 200k | S/2 scaling table 400K; XL/2 final 7M |
| Training previews | DDIM 250 | official train.py does not dump samples |
| `--sample` default | DDPM 250 + respace | `create_diffusion("250").p_sample_loop` |
| CFG | off by default; optional 1.5, first 3 channels | FID without CFG; SOTA figures s=1.5 |
| Parallel / precision | single GPU, optional grad accum | DDP; optional fp16 |
| FID | pytorch-fid | ADM TF Inception |

Notes: `docs/DiT-note.html`.

## Citation

```bibtex
@inproceedings{peebles2023dit,
  title={Scalable Diffusion Models with Transformers},
  author={Peebles, William and Xie, Saining},
  booktitle={ICCV},
  year={2023}
}
```
