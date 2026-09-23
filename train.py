import argparse
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import get_loader, is_imagefolder
from diffusion import GaussianDiffusion, diffusion_sample
from dit import build_dit
from vae import decode_to_image, load_frozen_vae

# Official DiT train.py: TF32 on Ampere (4090).
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 训练预览 / --sample 用的类别编号（ImageFolder 按文件夹名排序后的 0..C-1）。
# 取前 n_samples 个；不够则循环
SAMPLE_CLASS_IDS = [0, 1, 2, 3, 4, 5, 6, 7]


def sample_labels(n, num_classes, device):
    ids = SAMPLE_CLASS_IDS or list(range(num_classes))
    y = [ids[i % len(ids)] for i in range(n)]
    bad = [i for i in y if i < 0 or i >= num_classes]
    if bad:
        raise ValueError(f"SAMPLE_CLASS_IDS out of range [0, {num_classes}): {bad}")
    return torch.tensor(y, device=device, dtype=torch.long)


class EMA:
    """Exponential moving average of model weights (paper: decay 0.9999)."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)


@torch.no_grad()
def run_with_ema(model, ema, fn):
    """Swap EMA weights in-place (train weights parked on CPU) so sampling does not clone DiT on GPU."""
    device = next(model.parameters()).device
    backup = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema.shadow)
    model.eval()
    try:
        return fn(model)
    finally:
        model.load_state_dict({k: v.to(device) for k, v in backup.items()})
        model.train()


@torch.no_grad()
def generate_fid_samples(model, vae, diffusion, device, args, n, batch_size):
    """Sample in latent space (paper default: 250-step DDPM), VAE-decode, save PNGs."""
    out_dir = os.path.join(args.out, "fid", "fake")
    os.makedirs(out_dir, exist_ok=True)
    for old in os.listdir(out_dir):
        if old.endswith(".png"):
            os.remove(os.path.join(out_dir, old))
    idx = 0
    pbar = tqdm(total=n, desc=f"generate FID samples ({args.sampler} {args.num_sampling_steps})")
    while idx < n:
        bs = min(batch_size, n - idx)
        y = torch.randint(0, args.num_classes, (bs,), device=device)
        z, _ = diffusion_sample(
            diffusion, model, (bs, args.in_ch, args.latent_size, args.latent_size),
            device, y, args.cfg_scale, args.num_classes,
            sampler=args.sampler, steps=args.num_sampling_steps,
            spacing=args.ddim_spacing, eta=args.eta)
        imgs = decode_to_image(vae, z)
        for j in range(bs):
            save_image(imgs[j], os.path.join(out_dir, f"{idx + j:05d}.png"))
        idx += bs
        pbar.update(bs)
    pbar.close()
    return out_dir


def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Fréchet distance between two Gaussians (scipy>=1.14 compatible)."""
    import numpy as np
    from scipy import linalg

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2

    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if not np.isfinite(covmean).all():
        print(f"FID: singular product; adding {eps} to diagonal of cov estimates")
        eye = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + eye) @ (sigma2 + eye))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"Imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def _list_images(img_dir):
    """Collect image paths under img_dir (class subdirs allowed)."""
    files = []
    for dirpath, _, names in os.walk(img_dir):
        for f in names:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                files.append(os.path.join(dirpath, f))
    files.sort()
    if not files:
        raise FileNotFoundError(f"no images found in {img_dir}")
    return files


def compute_fid(fake_dir, ref_dir, device, batch_size=50):
    """Extract Inception features with pytorch-fid, Fréchet dist with our helper."""
    from pytorch_fid import fid_score
    from pytorch_fid.inception import InceptionV3

    fake_files = _list_images(fake_dir)
    ref_files = _list_images(ref_dir)
    print(f"FID: {len(fake_files)} fake vs {len(ref_files)} ref images")

    dims = 2048
    block = max(1, min(batch_size, 256))
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[dims]]).to(device)
    model.eval()

    mu_g, sigma_g = fid_score.calculate_activation_statistics(
        fake_files, model, block, dims, device, num_workers=0)
    mu_r, sigma_r = fid_score.calculate_activation_statistics(
        ref_files, model, block, dims, device, num_workers=0)
    return _frechet_distance(mu_g, sigma_g, mu_r, sigma_r)


def run_fid_eval(model, ema, vae, diffusion, device, args, step=None, writer=None):
    """Generate fakes with EMA, FID vs the training image folder."""
    if not is_imagefolder(args.data_dir):
        raise FileNotFoundError(
            f"no ImageFolder at {args.data_dir} (expected {args.data_dir}/<class>/*.jpg)")
    ref_dir = args.data_dir
    t0 = time.time()

    def _gen(m):
        generate_fid_samples(m, vae, diffusion, device, args, args.n_fid, args.fid_batch)

    if ema is not None:
        run_with_ema(model, ema, _gen)
    else:
        model.eval()
        _gen(model)
    fake_dir = os.path.join(args.out, "fid", "fake")
    fid = compute_fid(fake_dir, ref_dir, device, batch_size=args.fid_batch)
    dt = time.time() - t0
    tag = f"step {step}" if step is not None else "final"
    print(f"FID ({tag}, n={args.n_fid}, {args.sampler} {args.num_sampling_steps}): {fid:.2f}  ({dt / 60:.1f} min)")
    if writer is not None and step is not None:
        writer.add_scalar("eval/fid", fid, step)
    return fid


def _ckpt_args(ckpt, args):
    """Merge checkpoint hyperparams with CLI; accept our dict or official Namespace."""
    raw = ckpt.get("args", {})
    ca = dict(vars(raw)) if hasattr(raw, "__dict__") else dict(raw)
    ca.setdefault("in_ch", 4)
    ca.setdefault("patch_size", args.patch_size)
    ca.setdefault("num_classes", args.num_classes)
    ca.setdefault("class_dropout_prob", args.class_dropout_prob)
    ca.setdefault("T", args.T)
    ca.setdefault("beta_start", args.beta_start)
    ca.setdefault("beta_end", args.beta_end)
    ca.setdefault("vae_scale", args.vae_scale)
    image_size = ca.get("image_size", args.image_size)
    ca["image_size"] = image_size
    ca["latent_size"] = image_size // 8
    return ca


def _load_dit_state(ckpt):
    """Our ckpt.pt stores ema as a state dict; official DiT .pt uses the same key."""
    if "ema" in ckpt:
        state = ckpt["ema"]
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    return {k.replace("module.", "", 1): v for k, v in state.items()}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    args.latent_size = args.image_size // 8
    assert args.vae_ckpt, "training requires --vae_ckpt (official SD KL-f8 / sd-vae-ft-*)"
    assert args.image_size % 8 == 0
    assert args.latent_size % args.patch_size == 0

    vae = load_frozen_vae(args.vae_ckpt, device, scale_factor=args.vae_scale)
    model = build_dit(args).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DiT-S/{args.patch_size} params: {n_params:,} ({n_params / 1e6:.2f}M), "
          f"tokens {(args.latent_size // args.patch_size) ** 2}, device: {device}")

    diffusion = GaussianDiffusion(
        T=args.T, beta_start=args.beta_start, beta_end=args.beta_end).to(device)
    ema = EMA(model, decay=args.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    step, losses = 0, []
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        ema.shadow = ckpt["ema"]
        step, losses = ckpt["step"], ckpt.get("losses", [])
        print(f"resumed from checkpoint: step {step}")

    loader, ds = get_loader(args.data_dir, args.batch_size, size=args.image_size,
                            num_workers=args.num_workers)
    if len(ds.classes) > args.num_classes:
        raise ValueError(
            f"dataset has {len(ds.classes)} classes but --num_classes={args.num_classes}")
    accum = max(1, args.global_batch_size // args.batch_size)
    print(f"micro-batch {args.batch_size} × accum {accum} → global {accum * args.batch_size} "
          f"(paper 256)")
    tb_dir = os.path.join(args.out, "tb")
    writer = SummaryWriter(log_dir=tb_dir)
    writer.add_text("config", "\n".join(f"{k}: {v}" for k, v in sorted(vars(args).items())), 0)
    print(f"tensorboard: tensorboard --logdir {tb_dir}")

    def save_ckpt():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "ema": ema.shadow, "step": step, "losses": losses,
                    "args": vars(args)}, os.path.join(args.out, "ckpt.pt"))

    sample_noise = torch.randn(
        args.n_samples, args.in_ch, args.latent_size, args.latent_size, device=device)
    sample_y = sample_labels(args.n_samples, args.num_classes, device)
    print(f"preview classes: {sample_y.tolist()}")

    def quick_sample(tag):
        """Fixed latent noise + labels so samples_{step}.png is comparable across training."""

        def _sample(m):
            z, _ = diffusion_sample(
                diffusion, m, sample_noise.shape, device, sample_y,
                args.cfg_scale, args.num_classes, sampler="ddim",
                steps=args.num_sampling_steps, spacing=args.ddim_spacing,
                eta=args.eta, x_T=sample_noise)
            return decode_to_image(vae, z)

        imgs = run_with_ema(model, ema, _sample)
        save_image(imgs, os.path.join(args.out, f"samples_{tag}.png"),
                   nrow=4, value_range=(0, 1))
        writer.add_images("samples/grid", imgs, global_step=tag)

    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    micro = 0
    while step < args.max_steps:
        for x, y in loader:
            if step >= args.max_steps:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.no_grad():
                z = vae.encode(x).sample() * vae.scale_factor
            loss = diffusion.loss(model, z, y) / accum
            loss.backward()
            micro += 1
            if micro % accum != 0:
                continue
            opt.step()
            opt.zero_grad(set_to_none=True)
            ema.update(model)
            step += 1
            loss_val = loss.item() * accum

            if step % args.log_every == 0:
                losses.append(loss_val)
                dt = (time.time() - t0) / args.log_every * 1000
                t0 = time.time()
                print(f"step {step:>7d} | loss {loss_val:.4f} | {dt:.0f} ms/step")
                writer.add_scalar("train/loss", loss_val, step)
                writer.add_scalar("train/ms_per_step", dt, step)

            if step % args.sample_every == 0:
                quick_sample(step)
                print(f"    saved samples_{step}.png")

            if args.fid_every > 0 and step % args.fid_every == 0:
                run_fid_eval(model, ema, vae, diffusion, device, args, step=step, writer=writer)

            if step % args.ckpt_every == 0:
                save_ckpt()

    if step % args.sample_every != 0:
        quick_sample(step)
        print(f"    saved samples_{step}.png")
    save_ckpt()
    if args.fid_final:
        run_fid_eval(model, ema, vae, diffusion, device, args, step=step, writer=writer)
    writer.close()
    print(f"training done, {step} steps, checkpoint: {os.path.join(args.out, 'ckpt.pt')}")


def sample(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ca = _ckpt_args(ckpt, args)
    vae_path = args.vae_ckpt or ca.get("vae_ckpt")
    assert vae_path, "sampling requires --vae_ckpt or a checkpoint saved with vae_ckpt"

    vae = load_frozen_vae(vae_path, device, scale_factor=ca.get("vae_scale", args.vae_scale))
    model = build_dit(ca).to(device)
    model.load_state_dict(_load_dit_state(ckpt))
    model.eval()

    diffusion = GaussianDiffusion(
        T=ca["T"], beta_start=ca["beta_start"], beta_end=ca["beta_end"]).to(device)
    z_size = ca["latent_size"]
    num_classes = ca["num_classes"]
    if args.class_label is not None:
        y = torch.full((args.n_samples,), args.class_label, device=device, dtype=torch.long)
    else:
        y = sample_labels(args.n_samples, num_classes, device)
    print(f"sample classes: {y.tolist()}")
    x_T = None
    if args.same_noise:
        x_T = torch.randn(1, ca["in_ch"], z_size, z_size, device=device)
        x_T = x_T.repeat(args.n_samples, 1, 1, 1)
        print("sample noise: shared across windows")
    z, snaps = diffusion_sample(
        diffusion, model, (args.n_samples, ca["in_ch"], z_size, z_size), device, y,
        args.cfg_scale, num_classes, sampler=args.sampler, steps=args.num_sampling_steps,
        spacing=args.ddim_spacing, eta=args.eta, progress_every=args.progress_every,
        x_T=x_T)
    imgs = decode_to_image(vae, z)
    out = os.path.join(args.out, "samples_final.png")
    save_image(imgs, out, nrow=4, value_range=(0, 1))
    print(f"samples: {out}")

    if snaps:
        # 每行一张图，列是时间：左噪声 → 右干净（nrow = 时间帧数）
        decoded = torch.stack([decode_to_image(vae, s) for s in snaps], dim=1)
        b, t = decoded.shape[:2]
        out_p = os.path.join(args.out, "progression.png")
        save_image(decoded.reshape(b * t, *decoded.shape[2:]), out_p,
                   nrow=t, value_range=(0, 1))
        print(f"progression: {out_p} (rows=samples, cols=x0_hat, left=noisy → right=clean)")


def eval_fid(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ca = _ckpt_args(ckpt, args)
    vae_path = args.vae_ckpt or ca.get("vae_ckpt")
    assert vae_path, "FID requires --vae_ckpt or a checkpoint saved with vae_ckpt"

    vae = load_frozen_vae(vae_path, device, scale_factor=ca.get("vae_scale", args.vae_scale))
    model = build_dit(ca).to(device)
    model.load_state_dict(_load_dit_state(ckpt))
    model.eval()
    diffusion = GaussianDiffusion(
        T=ca["T"], beta_start=ca["beta_start"], beta_end=ca["beta_end"]).to(device)

    fid_args = argparse.Namespace(**{**ca, **{k: getattr(args, k) for k in (
        "data_dir", "out", "n_fid", "fid_batch", "sampler", "num_sampling_steps",
        "ddim_spacing", "eta", "cfg_scale")}})
    fid_args.latent_size = ca["latent_size"]
    fid_args.in_ch = ca["in_ch"]
    fid_args.num_classes = ca["num_classes"]
    run_fid_eval(model, None, vae, diffusion, device, fid_args)


@torch.no_grad()
def recon(args):
    """Encode/decode a batch with the frozen KL-f8 to verify official weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = load_frozen_vae(args.vae_ckpt, device, scale_factor=args.vae_scale)
    n = max(args.n_recon, 1)
    if is_imagefolder(args.data_dir):
        loader, _ = get_loader(args.data_dir, n, size=args.image_size, num_workers=0)
        x, _ = next(iter(loader))
        x = x[:n].to(device)
    else:
        print(f"no ImageFolder at {args.data_dir}; recon uses random images in [-1, 1]")
        x = torch.rand(n, 3, args.image_size, args.image_size, device=device) * 2 - 1
    posterior = vae.encode(x)
    z = posterior.mode()
    rec = vae.decode(z)
    z_scaled = z * vae.scale_factor
    print(f"recon: x {tuple(x.shape)} -> z {tuple(z.shape)} "
          f"(mean {z.mean().item():.4f}, std {z.std().item():.4f}; "
          f"scaled std {z_scaled.std().item():.4f})")
    expect = (n, 4, args.image_size // 8, args.image_size // 8)
    assert z.shape == expect, f"expected latent {expect}, got {tuple(z.shape)}"
    grid = torch.cat([((x + 1) / 2).clamp(0, 1), ((rec + 1) / 2).clamp(0, 1)], dim=0)
    out = os.path.join(args.out, "vae_recon.png")
    save_image(grid, out, nrow=x.shape[0], value_range=(0, 1))
    print(f"recon grid (top=input, bottom=decode(mode(z))): {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DiT-S/p class-conditional latent diffusion")

    # model (DiT-S/p on 32×32×4 latent)
    p.add_argument("--patch_size", type=int, default=2,
                   help="latent patch size p (DiT-S/2 is p=2 → 256 tokens)")
    p.add_argument("--in_ch", type=int, default=4)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--num_classes", type=int, default=1000)
    p.add_argument("--class_dropout_prob", type=float, default=0.1)

    # VAE (frozen official KL-f8)
    p.add_argument("--vae_ckpt", type=str, default=None,
                   help="local KL-f8 / sd-vae-ft-* weights (.ckpt / .safetensors / folder)")
    p.add_argument("--vae_scale", type=float, default=0.18215,
                   help="latent scale (z = sample() * scale); DiT / SD use 0.18215")

    # training (paper: 400K steps, global batch 256, lr 1e-4, wd 0)
    p.add_argument("--max_steps", type=int, default=400_000)
    p.add_argument("--batch_size", type=int, default=32,
                   help="micro-batch per forward; paper global batch is 256")
    p.add_argument("--global_batch_size", type=int, default=32,
                   help="optimizer step after global_batch_size/batch_size micro-batches")
    p.add_argument("--lr", type=float, default=1.25e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--resume", type=str, default=None)

    # diffusion (ADM linear: T=1000, β linspace(1e-4, 2e-2))
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=2e-2)

    # sampling (paper FID: 250-step DDPM, cfg=1; quality shots cfg=1.5)
    p.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    p.add_argument("--num_sampling_steps", type=int, default=250)
    p.add_argument("--ddim_spacing", type=str, default="uniform", choices=["uniform", "quad"])
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--cfg_scale", type=float, default=1.0,
                   help="classifier-free guidance; 1 = off")
    p.add_argument("--class_label", type=int, default=None,
                   help="fixed ImageNet class for --sample (default: 0..n cycling)")
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--same_noise", action="store_true",
                   help="--sample: all windows share one x_T (compare classes)")
    p.add_argument("--progress_every", type=int, default=10,
                   help="step interval for progression.png (only with --sample)")

    # FID
    p.add_argument("--fid_every", type=int, default=0,
                   help="FID every N training steps (0=disable)")
    p.add_argument("--fid_final", action="store_true",
                   help="run FID once after training finishes")
    p.add_argument("--n_fid", type=int, default=50_000,
                   help="generated images for FID (paper FID-50K)")
    p.add_argument("--fid_batch", type=int, default=8,
                   help="batch size for FID generation / Inception")

    # data / logging
    p.add_argument("--data_dir", type=str, default="data",
                   help="ImageFolder root (class subdirs), e.g. ImageNet train")
    p.add_argument("--out", type=str, default="runs/dit_s2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--sample_every", type=int, default=5000)
    p.add_argument("--ckpt_every", type=int, default=50_000)

    # run mode (default: train)
    p.add_argument("--sample", action="store_true", help="sample only (requires --ckpt)")
    p.add_argument("--recon", action="store_true",
                   help="VAE reconstruct a batch (no diffusion)")
    p.add_argument("--n_recon", type=int, default=8)
    p.add_argument("--eval_fid", action="store_true", help="FID only (requires --ckpt)")
    p.add_argument("--ckpt", type=str, default=None)
    args = p.parse_args()
    assert args.image_size % 8 == 0, "image_size must be divisible by 8 (KL-f8)"
    args.latent_size = args.image_size // 8

    os.makedirs(args.out, exist_ok=True)
    if args.recon:
        assert args.vae_ckpt, "--recon requires --vae_ckpt"
        recon(args)
    elif args.eval_fid:
        assert args.ckpt, "--eval_fid requires --ckpt"
        eval_fid(args)
    elif args.sample:
        assert args.ckpt, "--sample requires --ckpt"
        sample(args)
    else:
        train(args)
