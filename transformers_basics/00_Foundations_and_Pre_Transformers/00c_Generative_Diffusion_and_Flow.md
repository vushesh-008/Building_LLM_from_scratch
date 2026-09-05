# Generative Flows and Diffusion

## Diffusion Models — DDPM & DDIM

Diffusion extends the denoising AE idea across **many small corruption steps** rather than one big one. The model learns to reverse a gradual noising process.

**Forward process — adding noise step by step:**

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\; \sqrt{1-\beta_t}\, x_{t-1},\; \beta_t I\right)$$

Over $T$ steps (typically $T=1000$), a clean image $x_0$ is gradually turned into pure Gaussian noise $x_T \sim \mathcal{N}(0, I)$.

```
  x₀ (clean)  →  x₁ (tiny noise)  →  x₂  →  ...  →  xT (pure noise ≈ N(0,I))

  Each step adds a small amount of Gaussian noise.
  After T steps, the image is completely destroyed.
```

A useful shortcut: you can sample $x_t$ **directly** from $x_0$ without stepping through $t$ intermediate states:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \varepsilon \qquad \varepsilon \sim \mathcal{N}(0, I)$$

where $\bar{\alpha}_t = \prod_{s=1}^{t}(1-\beta_s)$ is the cumulative noise schedule.

**Reverse process — denoising:**

A neural network (U-Net or Diffusion Transformer, DiT) is trained to predict the noise $\varepsilon$ that was added at each step:

$$L = \mathbb{E}_{x_0, \varepsilon, t}\!\left[\|\varepsilon - \varepsilon_\theta(x_t, t)\|^2\right]$$

At inference, start from $x_T \sim \mathcal{N}(0, I)$ and run the denoising network $T$ times to recover $x_0$.

```
  xT (noise)  →  xT₋₁  →  ...  →  x₁  →  x₀ (generated image)

  Each step: predict the noise, subtract a fraction of it.
  1000 steps at training → DDIM can do 50 steps at inference (deterministic).
```

**Classifier-Free Guidance (CFG):**

Without guidance, diffusion generates random images. CFG steers generation toward a condition (text prompt, class label) by interpolating between conditional and unconditional predictions:

$$\hat{\varepsilon} = \varepsilon_\theta(x_t, \varnothing) + w \cdot \left[\varepsilon_\theta(x_t, c) - \varepsilon_\theta(x_t, \varnothing)\right]$$

- $c$ = condition (e.g. text embedding from CLIP)
- $\varnothing$ = null condition (unconditional)
- $w$ = guidance scale (typically 7–12): higher = more faithful to prompt but less diverse

```
  w = 1:   output = unconditional prediction (ignores prompt)
  w = 7:   strong prompt following, slight quality gain
  w = 20:  over-saturated, artefacts
```

**DDIM (Denoising Diffusion Implicit Models):**

DDIM reformulates the reverse process as a **deterministic ODE** instead of a stochastic SDE. Same trained model, same math — but sampling becomes deterministic and you can skip steps:

- DDPM: 1000 stochastic steps required
- DDIM: 20–50 deterministic steps, nearly same quality

---

## Flow Matching

Flow Matching is the modern successor to diffusion (Flux, Stable Diffusion 3, and robot policy models like π₀ use it). The core idea is simpler: learn a **vector field** that transports noise to data along **straight paths** instead of the curved, step-by-step diffusion trajectory.

```
  Diffusion path (curved):          Flow Matching path (straight):
  xT  ↘                             xT  ──────────────────────────►  x₀
        ↘  ↘  ↘  ↘                        (one clean interpolation)
                    ↘  x₀
```

**Training:**

Pick a random $t \in [0,1]$. Interpolate between noise $x_1 \sim \mathcal{N}(0,I)$ and data $x_0$:

$$x_t = (1-t)\, x_0 + t\, x_1$$

The target vector field at this point is simply the direction from noise to data:

$$u_t(x_t) = x_0 - x_1$$

Train a network $v_\theta(x_t, t)$ to match this vector field:

$$L = \mathbb{E}_{t, x_0, x_1}\!\left[\|v_\theta(x_t, t) - (x_0 - x_1)\|^2\right]$$

**Inference:** start from noise $x_1$ and integrate the learned vector field with an ODE solver to reach $x_0$. Straight paths → fewer integration steps → faster sampling than diffusion.

**Why it matters for robotics:** VLA policies (like π₀) use flow matching to output **continuous action trajectories** directly — no action binning, no quantisation error. The policy learns a vector field over action space rather than predicting discrete tokens.

---

## Reconstruction / Generative Family Summary

| Model | Corruption | Loss | Latent | Generative? | Used for |
|---|---|---|---|---|---|
| Vanilla AE | None | MSE | Unstructured | No | Compression, anomaly detection |
| Denoising AE | Noise / masking | MSE on clean | Unstructured | No | Representation learning |
| VAE | None | ELBO (MSE + KL) | Gaussian | **Yes** | Generation, interpolation |
| MAE | 75% patch masking | MSE on masked | Unstructured | No | ViT pre-training |
| DDPM | Incremental Gaussian | MSE on noise $\varepsilon$ | Implicit | **Yes** | Image/video generation |
| Flow Matching | Interpolation $(1-t)x_0+tx_1$ | MSE on vector field | Implicit | **Yes** | Fast generation, robot policies |
