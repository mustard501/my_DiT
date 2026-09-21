"""ADM / DiT Gaussian diffusion: linear β, ε-prediction, LEARNED_RANGE Σ."""
import math

import numpy as np
import torch


def extract(a, t, x_shape):
    """Gather from precomputed tensor a[T] at timesteps t[B], reshape for broadcasting."""
    out = a.to(device=t.device).gather(0, t.long())
    return out.to(dtype=torch.float32).view(-1, *([1] * (len(x_shape) - 1)))


def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, tensor.ndim)))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """KL(N(mean1, exp(logvar1)) || N(mean2, exp(logvar2))); from ADM / DiT."""
    return 0.5 * (
        -1.0 + logvar2 - logvar1
        + torch.exp(logvar1 - logvar2)
        + (mean1 - mean2) ** 2 * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """ADM decoder NLL for values assumed to come from uint8 rescaled to [-1, 1]."""
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    cdf_plus = approx_standard_normal_cdf(inv_stdv * (centered_x + 1.0 / 255.0))
    cdf_min = approx_standard_normal_cdf(inv_stdv * (centered_x - 1.0 / 255.0))
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    log_delta = torch.log((cdf_plus - cdf_min).clamp(min=1e-12))
    return torch.where(x < -0.999, log_cdf_plus, torch.where(x > 0.999, log_one_minus_cdf_min, log_delta))


class GaussianDiffusion:
    """ADM / DiT: linear β, ε-prediction + LEARNED_RANGE Σ, L = MSE(ε) + VB.

    Paper: T=1000, β linspace(1e-4, 2e-2). Official create_diffusion uses LossType.MSE
    (VB term is not rescaled by 1/1000).
    """

    def __init__(self, T=1000, beta_start=1e-4, beta_end=2e-2, betas=None):
        if betas is None:
            # ADM get_named_beta_schedule("linear"): scaled so T=1000 → linspace(1e-4, 2e-2).
            scale = 1000 / T
            betas = np.linspace(scale * beta_start, scale * beta_end, T, dtype=np.float64)
        else:
            betas = np.array(betas, dtype=np.float64)
        T = int(len(betas))
        self.T = T
        betas = torch.from_numpy(betas)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        acp_prev = torch.cat([torch.ones(1), acp[:-1]])

        self.betas = betas
        self.log_betas = betas.log()
        self.acp = acp
        self.acp_prev = acp_prev
        self.sqrt_acp = acp.sqrt()
        self.sqrt_1m_acp = (1.0 - acp).sqrt()
        self.sqrt_recip_acp = (1.0 / acp).sqrt()
        self.sqrt_recipm1_acp = (1.0 / acp - 1.0).sqrt()

        posterior_variance = betas * (1.0 - acp_prev) / (1.0 - acp)
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([posterior_variance[1:2], posterior_variance[1:]]))
        self.posterior_mean_coef1 = betas * acp_prev.sqrt() / (1.0 - acp)
        self.posterior_mean_coef2 = (1.0 - acp_prev) * alphas.sqrt() / (1.0 - acp)

    def to(self, device):
        for k, v in vars(self).items():
            if torch.is_tensor(v):
                setattr(self, k, v.to(device))
        return self

    def forward_sample(self, x0, t, noise):
        """Closed-form forward: x_t = sqrt(ab_t)*x_0 + sqrt(1-ab_t)*eps."""
        return (extract(self.sqrt_acp, t, x0.shape) * x0
                + extract(self.sqrt_1m_acp, t, x0.shape) * noise)

    def _predict_xstart_from_eps(self, xt, t, eps):
        return (extract(self.sqrt_recip_acp, t, xt.shape) * xt
                - extract(self.sqrt_recipm1_acp, t, xt.shape) * eps)

    def _predict_eps_from_xstart(self, xt, t, x0):
        return ((extract(self.sqrt_recip_acp, t, xt.shape) * xt - x0)
                / extract(self.sqrt_recipm1_acp, t, xt.shape))

    def q_posterior_mean_variance(self, x0, xt, t):
        mean = (extract(self.posterior_mean_coef1, t, xt.shape) * x0
                + extract(self.posterior_mean_coef2, t, xt.shape) * xt)
        var = extract(self.posterior_variance, t, xt.shape)
        log_var = extract(self.posterior_log_variance_clipped, t, xt.shape)
        return mean, var, log_var

    def p_mean_variance(self, model, x, t, clip_denoised=False, model_kwargs=None):
        """p(x_{t-1}|x_t): split 2C output into ε + LEARNED_RANGE variance."""
        model_kwargs = {} if model_kwargs is None else model_kwargs
        B, C = x.shape[:2]
        out = model(x, t, **model_kwargs)
        assert out.shape == (B, C * 2, *x.shape[2:]), f"expected 2C={C * 2} channels, got {out.shape}"
        eps, var_values = out.split(C, dim=1)

        min_log = extract(self.posterior_log_variance_clipped, t, x.shape)
        max_log = extract(self.log_betas, t, x.shape)
        frac = (var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        model_variance = model_log_variance.exp()

        x0_pred = self._predict_xstart_from_eps(x, t, eps)
        if clip_denoised:
            x0_pred = x0_pred.clamp(-1, 1)
        mean, _, _ = self.q_posterior_mean_variance(x0_pred, x, t)
        return {
            "mean": mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": x0_pred,
            "eps": eps,
            "var_values": var_values,
        }

    def _vb_terms_bpd(self, model, x0, xt, t, model_kwargs=None):
        true_mean, _, true_log_var = self.q_posterior_mean_variance(x0, xt, t)
        out = self.p_mean_variance(model, xt, t, clip_denoised=False, model_kwargs=model_kwargs)
        kl = mean_flat(normal_kl(true_mean, true_log_var, out["mean"], out["log_variance"])) / math.log(2.0)
        decoder_nll = mean_flat(-discretized_gaussian_log_likelihood(
            x0, means=out["mean"], log_scales=0.5 * out["log_variance"])) / math.log(2.0)
        output = torch.where(t == 0, decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def loss(self, model, x0, y):
        """L = MSE(ε) + VB(Σ); VB mean is detached so it does not train the ε head."""
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.forward_sample(x0, t, noise)
        model_kwargs = dict(y=y)
        out = model(xt, t, **model_kwargs)
        B, C = xt.shape[:2]
        eps, var_values = out.split(C, dim=1)
        frozen = torch.cat([eps.detach(), var_values], dim=1)
        vb = self._vb_terms_bpd(lambda *a, r=frozen, **k: r, x0, xt, t, model_kwargs)["output"]
        mse = mean_flat((noise - eps) ** 2)
        return (mse + vb).mean()

    def predict_x0(self, model, xt, t, model_kwargs=None):
        tv = torch.full((xt.shape[0],), t, device=xt.device, dtype=torch.long)
        return self.p_mean_variance(model, xt, tv, clip_denoised=False,
                                    model_kwargs=model_kwargs)["pred_xstart"]

    @torch.no_grad()
    def p_sample(self, model, x, t, clip_denoised=False, model_kwargs=None):
        out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        noise = torch.randn_like(x)
        nonzero = (t != 0).float().view(-1, *([1] * (x.ndim - 1)))
        sample = out["mean"] + nonzero * (0.5 * out["log_variance"]).exp() * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    @torch.no_grad()
    def p_sample_loop(self, model, shape, device, model_kwargs=None, x_T=None,
                      clip_denoised=False, progress_every=None):
        """Full 1000-step DDPM reverse (paper FID uses 250-step respacing; here native T)."""
        x = torch.randn(shape, device=device) if x_T is None else x_T
        snaps = []
        for i in range(self.T - 1, -1, -1):
            tv = torch.full((x.shape[0],), i, device=device, dtype=torch.long)
            out = self.p_sample(model, x, tv, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
            x = out["sample"]
            if progress_every and (i % progress_every == 0 or i == 0):
                snaps.append(out["pred_xstart"])
        return x, snaps

    @torch.no_grad()
    def ddim_p_sample(self, model, x, t, eta=0.0, model_kwargs=None):
        """ADM ddim_sample: x_t → x_{t-1}. acp_prev[0] = 1 so t=0 returns x0_hat."""
        tv = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        out = self.p_mean_variance(model, x, tv, clip_denoised=False, model_kwargs=model_kwargs)
        x0_pred, eps = out["pred_xstart"], out["eps"]
        acp_t = extract(self.acp, tv, x.shape)
        acp_prev = extract(self.acp_prev, tv, x.shape)
        sigma = eta * ((1.0 - acp_prev) / (1.0 - acp_t)).sqrt() * (1.0 - acp_t / acp_prev).sqrt()
        x_prev = acp_prev.sqrt() * x0_pred + (1.0 - acp_prev - sigma ** 2).clamp(min=0.0).sqrt() * eps
        nonzero = (tv != 0).float().view(-1, *([1] * (x.ndim - 1)))
        if eta > 0:
            x_prev = x_prev + nonzero * sigma * torch.randn_like(x)
        return x_prev, x0_pred

    @torch.no_grad()
    def ddim_p_sample_loop(self, model, shape, device, num_steps=None, spacing="uniform",
                           eta=0.0, progress_every=None, x_T=None, model_kwargs=None):
        """Full DDIM reverse on *this* chain (paper: respace to 250, then loop T'=250)."""
        del spacing, num_steps
        x = torch.randn(shape, device=device) if x_T is None else x_T
        snaps = []
        for i in range(self.T - 1, -1, -1):
            x, x0_pred = self.ddim_p_sample(model, x, i, eta=eta, model_kwargs=model_kwargs)
            if progress_every and ((self.T - 1 - i) % progress_every == 0 or i == 0):
                snaps.append(x0_pred)
        return x, snaps


def space_timesteps(num_timesteps, num_wanted):
    """ADM/DiT one-section striding (create_diffusion('250') → 250 of 1000)."""
    if num_wanted >= num_timesteps:
        return list(range(num_timesteps))
    if num_wanted <= 1:
        return [0]
    frac = (num_timesteps - 1) / (num_wanted - 1)
    return sorted({int(round(i * frac)) for i in range(num_wanted)})


def make_spaced_diffusion(base, num_steps):
    """Official SpacedDiffusion: new β from kept ᾱ, model t remapped."""
    if num_steps >= base.T:
        return base, None
    use = space_timesteps(base.T, num_steps)
    acp = base.acp.detach().cpu()
    last = torch.tensor(1.0)
    new_betas = []
    for i in use:
        new_betas.append(float(1.0 - acp[i] / last))
        last = acp[i]
    return GaussianDiffusion(betas=new_betas), use



def wrap_sampler_model(model, tmap, use_cfg):
    """Map spaced t → original t; optionally run forward_with_cfg."""

    def call(x, t, **kwargs):
        if tmap is not None:
            t = torch.tensor(tmap, device=t.device, dtype=torch.long)[t.long()]
        if use_cfg:
            return model.forward_with_cfg(x, t, kwargs["y"], kwargs["cfg_scale"])
        return model(x, t, kwargs["y"])

    return call


def diffusion_sample(diffusion, model, shape, device, y, cfg_scale, num_classes,
                     sampler="ddpm", steps=250, spacing="uniform", eta=0.0,
                     x_T=None, progress_every=None):
    """DDPM (paper, uses learned Σ + optional respacing) or DDIM. CFG doubles the batch."""
    del spacing
    use_cfg = cfg_scale != 1.0
    n = shape[0]
    if x_T is None:
        x_T = torch.randn(shape, device=device)
    if use_cfg:
        x_T = torch.cat([x_T, x_T], 0)
        y = torch.cat([y, torch.full((n,), num_classes, device=device, dtype=torch.long)], 0)
        mk = dict(y=y, cfg_scale=cfg_scale)
        shape = x_T.shape
    else:
        mk = dict(y=y)

    if sampler == "ddpm":
        spaced, tmap = make_spaced_diffusion(diffusion, steps)
        spaced = spaced.to(device)
        m = wrap_sampler_model(model, tmap, use_cfg)
        z, snaps = spaced.p_sample_loop(
            m, shape, device, model_kwargs=mk, x_T=x_T,
            clip_denoised=False, progress_every=progress_every)
    elif sampler == "ddim":
        spaced, tmap = make_spaced_diffusion(diffusion, steps)
        spaced = spaced.to(device)
        m = wrap_sampler_model(model, tmap, use_cfg)
        z, snaps = spaced.ddim_p_sample_loop(
            m, shape, device, eta=eta, progress_every=progress_every,
            x_T=x_T, model_kwargs=mk)
    else:
        raise ValueError(f"unknown sampler {sampler!r}")
    if use_cfg:
        z, _ = z.chunk(2, dim=0)
        snaps = [s.chunk(2, dim=0)[0] for s in snaps]
    return z, snaps
