# XPBI — Position-Based Dynamics with Smoothing Kernels Handles Continuum Inelasticity

A full implementation of **"XPBI: Position-Based Dynamics with Smoothing Kernels
Handles Continuum Inelasticity"** (Chang Yu, Xuan Li, Lei Lan, Yin Yang, Chenfanfu
Jiang — SIGGRAPH 2024, [arXiv:2405.11694](https://arxiv.org/abs/2405.11694)),
including the constitutive models from the papers it builds on, as an interactive
real-time WebGPU demo that runs on the Mac GPU (Metal 3 via Chrome's WebGPU).

## Run it

```bash
cd ~/xpbi/web && python3 serve.py 8777
```

then open **http://localhost:8777** in Chrome (or any WebGPU-enabled browser).

## Controls

- **Drag** — orbit camera, **wheel** — zoom
- **Shift-drag** or **Poke mode (G)** — push a kinematic sphere through the material
- **Space** — pause; **Reset** — restart scene
- Sliders: substeps, solver iterations, time scale, and the live material
  parameters of each scene (friction angle φ, yield stress σY, Herschel-Bulkley
  exponent h, NACC hardening ξ, …)

## Scenes

| Scene | Model | Reference |
| --- | --- | --- |
| Sand: column collapse / two blocks | Drucker-Prager | Klár et al. 2016 |
| Snow: snowball smash | Non-Associated Cam-Clay | Wolper et al. 2019 |
| Snow: classic | Singular-value clamp + hardening | Stomakhin et al. 2013 |
| Plasticine: drop & poke | Von Mises | Li et al. 2022 |
| Goo: Herschel-Bulkley | HB viscoplasticity (b̄ᵉ form) | Yue et al. 2015 |
| Dam breach: water + sand | PBF + Drucker-Prager coupling | Macklin & Müller 2013 |
| Elastic: StVK block | StVK w/ Hencky strain, no plasticity | paper §3.1 |

## What is implemented (paper → code)

Everything in the paper's method and algorithm sections, on the GPU:

- **Velocity-based XPBD** (Eq. 15–19) with per-particle inelasticity constraints
  `C_p = √(2Ψ(F))`, `α = 1/V⁰` (Eq. 6), Jacobi iterations (the solver the paper
  itself uses for its real-time demos, §5.2), plus XPBD constraint damping
  (Macklin et al. 2016 Eq. 26 — named as necessary by the paper's §6).
- **StVK energy with Hencky strain** (Eq. 3) and its exact constraint gradients
  (Eq. 12–13) via 3×3 SVD in WGSL.
- **Updated-Lagrangian deformation gradient**: `F ← (I + Δt ∇v) Fⁿ` (Eq. 9) with
  the SPH velocity gradient (Eq. 11), Wendland C2 kernels, and Bonet–Lok kernel
  gradient correction with SVD pseudo-inverse (Eq. 10).
- **Implicit plasticity in the loop** (Eq. 14, Alg. 1): the return mapping runs on
  the trial F inside every solver iteration, and again at the final F update (Eq. 22).
- **Return mappings** transcribed from the referenced papers (see
  `docs/IMPLEMENTATION.md` for the exact formulas and sources): Drucker-Prager,
  NACC (with both tip projections, ellipse projection, and the fracture-friendly
  hardening), Von Mises, Herschel-Bulkley (analytic Bingham case + bisection for
  general h), and the classic snow clamp with exponential hardening.
- **Stability machinery**: XSPH damping c = 0.01 (Eq. 20), position-correction
  distance constraint at 0.75·r (Eq. 21), CFL velocity clamp.
- **Neighbor search**: uniform grid, counting sort with GPU prefix scan
  (Hoetzlein 2014, the paper's citation), support = 2r, particles sampled at
  spacing r (Table 2 note). Particle data is physically reordered into cell
  order every substep so neighbor gathers are cache-coherent — this alone is
  worth ~3× on Apple GPUs.
- **Performance** (M-series Mac, Chrome): 40–60 fps at 4k–28k particles with
  160–300 substeps/s. The hot neighbor loops are tuned for Apple GPUs: sorted
  particle data (cache-coherent gathers), row-merged cell ranges (9 range
  lookups per stencil instead of 27), V^n packed into vel.w (two vec4 loads
  per neighbor instead of three), and a fused constraint-gather + contact pass.
- **PBF water** (Macklin & Müller 2013): density constraint with CFM relaxation,
  s_corr anti-clustering, poly6/spiky kernels, two-way coupling with XPBI solids
  through the shared neighborhood.

Deviations from the paper (all deliberate, for real-time on a laptop GPU):
Jacobi with under-relaxation instead of grid-colored Gauss-Seidel, larger
timesteps with per-scene time scaling, softened stiffnesses, and a clamped NACC
hardening state. Details and rationale in `docs/IMPLEMENTATION.md`.

## Layout

```
web/            the WebGPU demo (no build step, no dependencies)
  index.html    UI shell
  serve.py      static server (no-store caching, required for ES-module dev)
  js/main.js    boot, camera, input, render pipeline, frame loop
  js/sim.js     buffers, compute pipelines, substep encoding
  js/scenes.js  scene sampling + material parameter sets
  js/shaders/   WGSL: constitutive models & return mappings (common.js),
                simulation kernels (simkernels.js), rendering (render.js)
prototype/      2D NumPy validation of the full algorithm (sand column collapse)
docs/           implementation notes, paper-equation → code mapping
```
