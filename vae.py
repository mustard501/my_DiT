"""CompVis AutoencoderKL / SD KL-f8. Names match official weights (Lightning or HF)."""
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F


def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k) * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_).reshape(b, c, h, w)
        return x + self.proj_out(h_)


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "none"], f"attn_type {attn_type} unknown"
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    return nn.Identity()


class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False,
                 attn_type="vanilla", **ignore_kwargs):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                         temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels if double_z else z_channels,
                                  kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        temb = None
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        h = self.norm_out(h)
        h = nonlinearity(h)
        return self.conv_out(h)


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False,
                 use_linear_attn=False, attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                         temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        self.last_z_shape = z.shape
        temb = None
        h = self.conv_in(z)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class DiagonalGaussianDistribution:
    """CompVis ldm.modules.distributions.distributions.DiagonalGaussianDistribution."""

    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def mode(self):
        return self.mean


class _Dummy:
    """Stand-in for Lightning / OmegaConf objects inside official ckpts."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class _IgnoreMissingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(("pytorch_lightning", "lightning", "omegaconf")):
            return _Dummy
        return super().find_class(module, name)


class _ignore_missing_pickle:
    Unpickler = _IgnoreMissingUnpickler
    load = staticmethod(lambda f, **k: _IgnoreMissingUnpickler(f, **k).load())
    dump = pickle.dump
    dumps = pickle.dumps
    loads = pickle.loads


def load_pl_ckpt(path, map_location="cpu"):
    """Load a CompVis Lightning .ckpt without installing pytorch_lightning."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except ModuleNotFoundError:
        return torch.load(
            path, map_location=map_location, weights_only=False,
            pickle_module=_ignore_missing_pickle)


_HF_VAE_FILES = (
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.bin",
    "model.ckpt",
)


def _maybe_strip_prefix(sd, prefix):
    n = sum(k.startswith(prefix) for k in sd)
    if n == 0:
        return sd
    if n == len(sd) or n > len(sd) // 2:
        return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}
    return sd


def _is_hf_vae(sd):
    return any("down_blocks." in k or "mid_block." in k or "up_blocks." in k for k in sd)


def convert_hf_vae_state_dict(sd):
    """HuggingFace diffusers AutoencoderKL keys -> CompVis AutoencoderKL keys.

    Port of diffusers scripts/convert_diffusers_to_original_stable_diffusion.py
    (convert_vae_state_dict). Needed because DiT's sd-vae-ft-* is the HF layout,
    while Encoder/Decoder below keep CompVis names so Lightning .ckpt also loads.
    """
    vae_conversion_map = [
        ("nin_shortcut", "conv_shortcut"),
        ("norm_out", "conv_norm_out"),
        ("mid.attn_1.", "mid_block.attentions.0."),
    ]
    for i in range(4):
        for j in range(2):
            vae_conversion_map.append(
                (f"encoder.down.{i}.block.{j}.", f"encoder.down_blocks.{i}.resnets.{j}."))
        if i < 3:
            vae_conversion_map.append(
                (f"down.{i}.downsample.", f"down_blocks.{i}.downsamplers.0."))
            vae_conversion_map.append(
                (f"up.{3 - i}.upsample.", f"up_blocks.{i}.upsamplers.0."))
        for j in range(3):
            vae_conversion_map.append(
                (f"decoder.up.{3 - i}.block.{j}.", f"decoder.up_blocks.{i}.resnets.{j}."))
    for i in range(2):
        vae_conversion_map.append(
            (f"mid.block_{i + 1}.", f"mid_block.resnets.{i}."))

    vae_conversion_map_attn = [
        ("norm.", "group_norm."),
        ("q.", "query."),
        ("k.", "key."),
        ("v.", "value."),
        ("proj_out.", "proj_attn."),
    ]
    vae_extra_conversion_map = [
        ("to_q", "q"),
        ("to_k", "k"),
        ("to_v", "v"),
        ("to_out.0", "proj_out"),
    ]

    mapping = {k: k for k in sd}
    for k, v in mapping.items():
        for sd_part, hf_part in vae_conversion_map:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    for k, v in mapping.items():
        if "attentions" in k:
            for sd_part, hf_part in vae_conversion_map_attn:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    out = {v: sd[k] for k, v in mapping.items()}

    def reshape_weight_for_sd(w):
        return w if w.ndim == 1 else w.reshape(*w.shape, 1, 1)

    keys_to_rename = {}
    for k, w in list(out.items()):
        for name in ("q", "k", "v", "proj_out"):
            if f"mid.attn_1.{name}.weight" in k:
                out[k] = reshape_weight_for_sd(w)
        for hf_name, sd_name in vae_extra_conversion_map:
            if f"mid.attn_1.{hf_name}.weight" in k or f"mid.attn_1.{hf_name}.bias" in k:
                keys_to_rename[k] = k.replace(hf_name, sd_name)
    for old, new in keys_to_rename.items():
        out[new] = reshape_weight_for_sd(out[old])
        del out[old]
    return out


_VAE_WEIGHT_HINT = """\
Download official weights locally, then pass --vae_ckpt <file-or-folder>.
  DiT / SD ft-EMA (default):
    https://huggingface.co/stabilityai/sd-vae-ft-ema/resolve/main/diffusion_pytorch_model.safetensors
  SD ft-MSE (paper scaling-table decoder):
    https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors
  Original LDM KL-f8 (CompVis Lightning .ckpt zip):
    https://ommer-lab.com/files/latent-diffusion/kl-f8.zip
"""


def resolve_vae_path(path):
    """Local .ckpt / .safetensors / .bin, or a folder containing one of those files."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        for name in _HF_VAE_FILES:
            p = os.path.join(path, name)
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(f"no VAE weights in directory {path}\n{_VAE_WEIGHT_HINT}")
    raise FileNotFoundError(f"VAE weights not found: {path}\n{_VAE_WEIGHT_HINT}")


def load_vae_state_dict(path):
    path = resolve_vae_path(path)
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError("this checkpoint is .safetensors: pip install safetensors") from e
        sd = load_file(path)
    else:
        raw = load_pl_ckpt(path)
        sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    sd = _maybe_strip_prefix(sd, "first_stage_model.")
    sd = _maybe_strip_prefix(sd, "vae.")
    if _is_hf_vae(sd):
        sd = convert_hf_vae_state_dict(sd)
        print(f"converted HuggingFace VAE keys -> CompVis ({path})")
    return sd


class AutoencoderKL(nn.Module):
    """CompVis ldm.models.autoencoder.AutoencoderKL (no Lightning / loss)."""

    def __init__(self, ddconfig, embed_dim, ckpt_path=None, ignore_keys=(),
                 scale_factor=0.18215):
        super().__init__()
        assert ddconfig["double_z"]
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quant_conv = nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.scale_factor = scale_factor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=()):
        sd = load_vae_state_dict(path)
        keys = list(sd.keys())
        ignore_keys = tuple(ignore_keys) + ("loss.",)
        for k in keys:
            if any(k.startswith(ik) for ik in ignore_keys):
                del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"KL-f8 restored from {path}: {len(missing)} missing, {len(unexpected)} unexpected")
        if missing:
            print(f"  Missing Keys: {missing}")
            raise RuntimeError(f"VAE weights incomplete ({len(missing)} missing keys)")
        if unexpected:
            print(f"  Unexpected Keys: {unexpected}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return DiagonalGaussianDistribution(moments)

    def decode(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(z), posterior


KL_F8_DDCONFIG = dict(
    double_z=True,
    z_channels=4,
    resolution=256,
    in_channels=3,
    out_ch=3,
    ch=128,
    ch_mult=(1, 2, 4, 4),
    num_res_blocks=2,
    attn_resolutions=[],
    dropout=0.0,
)


def make_kl_f8(ckpt_path=None, scale_factor=0.18215):
    """Official SD / LDM KL-f8: 256^2 RGB -> 32x32x4 latent, then * scale_factor."""
    return AutoencoderKL(
        ddconfig=KL_F8_DDCONFIG, embed_dim=4, ckpt_path=ckpt_path,
        scale_factor=scale_factor)



def load_frozen_vae(ckpt_path, device, scale_factor=0.18215):
    """Load official SD KL-f8 / sd-vae-ft-* and freeze; diffusion never trains these weights."""
    vae = make_kl_f8(ckpt_path, scale_factor=scale_factor).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    n = sum(p.numel() for p in vae.parameters())
    print(f"KL-f8 frozen: {n / 1e6:.2f}M from {ckpt_path} (scale {scale_factor})")
    return vae


@torch.no_grad()
def decode_to_image(vae, z):
    """Scaled latent -> RGB in [0, 1]. z is the diffusion-space latent (already * scale_factor)."""
    return ((vae.decode(z / vae.scale_factor) + 1) / 2).clamp(0, 1)
