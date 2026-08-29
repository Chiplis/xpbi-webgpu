# Implementation notes: paper → code mapping

Primary paper: **XPBI** (Yu, Li, Lan, Yang, Jiang — SIGGRAPH 2024, arXiv:2405.11694).
All references below were pulled from the actual PDFs during implementation
(UCR/NSF-PAR/Waterloo/author-page mirrors), not from memory.

## Core algorithm (paper Alg. 1) → `web/js/sim.js: encodeSubstep` + `simkernels.js`

Per substep (the paper's "time step" — neighbor search per step, per §4.2):

| Paper step | Kernel(s) |
| --- | --- |
| Neighbor search using xⁿ (§4.2, Hoetzlein 2014) | `clearBuffer` + `countParticles` → `scanBlocks`/`scanApply` (GPU prefix scan) → `scatter` (counting sort via `atomicSub`) |
| Kernel gradient correction L_p (Eq. 10) | `computeL` — Wendland C2 gradients, Σ V_b ∇W ⊗ (x_b − x_p), SVD pseudo-inverse |
| v ← vⁿ + Δt M⁻¹ f_ext ; λ = 0 | folded into `scatter` |
| XPBD iterations: per-particle constraint loop | `solveA` (Δλ) + `solveB` (Δv gather + contacts + boundaries) |
| ∇v estimation (Eq. 11) | `velocityGradient()` — Σ V_b (v_b − v_p)(L_p ∇W_b)ᵀ |
| F trial + return mapping Z in the loop (Eq. 14) | in `solveA`: `Ftr = (I + Δt ∇v) Fⁿ; sigma = Z(svd(Ftr))` |
| Δλ_p (Eq. 17) with α = 1/V⁰, α̃ = α/Δt² | `solveA`, incl. XPBD damping term (see below) |
| Δv = M⁻¹∇C ᵀΔλ/Δt (Eq. 18) | `solveB` gathers `Sp = Δλ·(∂C/∂F Fⁿᵀ)L` from neighbor constraints + own self-gradient |
| Other constraints C_i (collisions) | position-correction distance constraint (Eq. 21, ε = 0.25r ⇒ active < 0.75r) + walls + kinematic sphere, merged into `solveB` |
| PBF water (if present) | `pbfA`/`pbfB` per iteration |
| XSPH smoothing (Eq. 20, c = 0.01) | `xsph` (water uses 10×, standard PBF viscosity range) |
| Final F update w/ return mapping (Eq. 22) | `updateF` — same velocity as position update, per the paper's consistency note |
| x ← x + Δt v | `integrate` (+ CFL clamp, see Deviations) |

Constraint definition (Eq. 6): `C = √(2Ψ)`, single-constraint variant the paper
recommends. Gradients (Eq. 12–13): `∇_{x_b}C_p = V_bⁿ (∂C/∂F Fⁿᵀ)(L_p ∇W_b)`,
with `∂C/∂F = P(Z(F_tr))/C` evaluated at the plastically projected state (the
projection is treated as constant during differentiation, per §3.3).

## Constitutive models → `web/js/shaders/common.js`

**StVK w/ Hencky strain** (paper Eq. 3; Klár et al. 2016 Eq. 25–26):
`Ψ = μ tr(log Σ)² + λ/2 (tr log Σ)²`, `∂Ψ/∂σᵢ = (2μ log σᵢ + λ tr(log Σ))/σᵢ`.

**Drucker-Prager** (Klár et al. 2016 §7.1, Eq. 27–31, verified against their
supplementary tech doc): Hencky strain ε = log Σ, three cases —
elastic (δγ ≤ 0), tip projection Σ ← 1 when tr(ε) > 0 (cohesionless free
separation; volume-preserving on the cone otherwise), cone projection
`ε ← ε − δγ ε̂/‖ε̂‖` with `δγ = ‖ε̂‖ + α (dλ+2μ)/(2μ)·tr(ε)` and
`α = √(2/3)·2 sin φ/(3 − sin φ)`. Note: Klár et al. has **no cohesion**; the
small `c0` knob (used by XPBI's own Table 2) is implemented as a trace shift.

**Von Mises** (Li, Li, Jiang 2022 Eq. 20–22): `δγ = ‖ε̂‖ − σ_Y/(2μ)`; subtract
along ε̂/‖ε̂‖ (volume preserved).

**NACC** (Wolper et al. 2019 + supplemental Alg. 2, their exact b^E formulation,
not a Hencky restatement): `p = −κ/2(J²−1)`, `ŝ = μJ^{−2/3}dev(Σ²)`,
`q = √(3/2)‖ŝ‖`, yield `y = (1+2β)q² + M²(p+βp₀)(p−p₀)`,
`p₀ = κ(10⁻⁵ + sinh(ξ·max(−α,0)))`, κ = (2/3)μ + λ. Case 1/2 tip projections set
`Σᵢ = J_new^{1/3}` with `α += log(J_tr/J_new)`; case 3 rescales the deviator of
Σ² onto the ellipse at fixed tr(Σ²) and applies their "fracture-friendly"
hardening via the ellipse–line intersection (quadratic in their supplement).

**Herschel-Bulkley** (Yue et al. 2015 Eq. 12–18 — their b̄ᵉ formulation):
trial `s_pre = μ‖dev(b̄ᵉ)‖` with `b̄ᵉ = J^{−2/3}Σ²`; yield `Φ = s − √(2/3)σ_Y`;
`μ̃ = tr(b̄ᵉ)μ/3`. h = 1 / η → 0 analytic (their Eq. 17); general h solved by
**bisection** on `η^{1/h}(s − s_pre) + 2μ̃Δt(s − √(2/3)σ_Y)^{1/h} = 0`
(their choice of solver). Deviator rescaled, trace kept, renormalized to det 1.

**Snow clamp** (Stomakhin et al. 2013 Eq. 11–12 + Eq. 2): σ clamped to
`[1−θc, 1+θs]`; plastic determinant tracked in the per-particle state and
μ, λ scaled by `exp(ξ(1−J_P))` (exponent capped for f32 safety).

**PBF** (Macklin & Müller 2013): density constraint `C = ρ/ρ₀ − 1` (unilateral),
`λ = −C/(Σ‖∇C‖² + ε)`, `Δp = Σ(λᵢ+λⱼ+s_corr)∇W_spiky·m/ρ₀`,
`s_corr = −k(W/W(Δq))⁴` with |Δq| = 0.2·support, poly6/spiky kernels; XSPH.
Coupling: solid particles contribute mass to water density and receive half
the position correction (mass-weighted two-way coupling in the spirit of
Macklin et al. 2014's unified solver).

## Deviations from the paper, and why

1. **Jacobi instead of grid-colored Gauss-Seidel.** The paper's own real-time
   demos use Jacobi (§5.2); colored GS mainly pays off for their offline
   high-stiffness scenes. Under-relaxation ω = 0.5 on Δλ was necessary for
   stability at real-time timesteps (without it, stiff scenes diverge —
   consistent with the paper's Fig. 14 Jacobi observations).
2. **XPBD constraint damping** (γ ≈ 0.4, Macklin et al. 2016 Eq. 26). The paper's
   §6 names "XSPH or XPBD constraint damping" as necessary; XSPH alone was not
   enough at our timesteps under sustained excitation (e.g. the poke collider).
3. **Timestep.** The paper uses Δt ∈ [5·10⁻⁵, 2·10⁻⁴] s offline. Real-time in a
   browser affords ~10⁻³ s, so scenes run with softened E, per-scene time
   scaling (slow motion), and a CFL safety clamp (≤ 0.45·support per substep).
4. **NACC hardening state clamped** to [−0.25, 0.4] and the sinh argument capped:
   at real-time timesteps the sinh(ξ|α|) feedback loop otherwise diverges
   between plasticity applications (benign at the paper's Δt).
5. **Kernel-gradient correction conditioning.** The pseudo-inverse tolerance is
   absolute (drop singular values < 0.2 of the ideal ≈ identity magnitude):
   deficient neighborhood directions (free surfaces, collider craters) are
   dropped rather than amplified.
6. **Counting-sort grid + full particle-data reordering.** Contiguous cell
   ranges instead of a linked-list grid (3–4×: pointer chasing through atomics
   bypasses the cache hierarchy on Apple GPUs), and the particle arrays
   (pos/vel/F/aux) are physically permuted into cell order every substep so
   the memory-bound neighbor gathers are cache-coherent and indirection-free
   (a further ~3×; `stagePVA`/`applyPVA`/`stageF`/`applyF`).
7. **Not implemented** from the demo suite: mesh-based cloth coupling (Fig. 3)
   and LBVH point-triangle collision (§5's timing breakdown) — the demo uses
   analytic boundaries (box, kinematic sphere) instead; the constraint framework
   they would plug into (the C_i loop of Alg. 1) is present.

## Validation

- `prototype/xpbi2d.py`: full 2D NumPy implementation of the same algorithm
  (velocity-based XPBD + corrected SPH gradients + implicit DP + XSPH + position
  correction). Sand column collapse produces a stable pile at the friction angle
  that stays static from t ≈ 1 s to t = 3 s (frames in `prototype/frames/`).
- 3D behaviors verified against the paper's qualitative results: sand piling at
  varying φ (their inset comparison), two-block collision (Fig. 13), snowball
  impact craters (Fig. 12), HB blob spreading vs h (Fig. 4), dam breach with
  water + sand (Fig. 7), and permanent dents from kneading plasticine (Fig. 2's
  material class).
