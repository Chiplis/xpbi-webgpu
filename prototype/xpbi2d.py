"""
XPBI 2D prototype — validates the core algorithm of
"XPBI: Position-Based Dynamics with Smoothing Kernels Handles Continuum
Inelasticity" (Yu, Li, Lan, Yang, Jiang — SIGGRAPH 2024, arXiv:2405.11694)
before the WebGPU port.

Implements, per the paper:
  - Velocity-based XPBD (Eq. 15-19), Jacobi iterations (paper §5.2 uses
    Jacobi for real-time scenes)
  - StVK w/ Hencky strain, single constraint C = sqrt(2 Psi), alpha = 1/V0
    (Eq. 3, 6)
  - Updated Lagrangian F: F^{n+1} = (I + dt grad_v) F^n (Eq. 9)
  - SPH velocity gradient w/ Wendland kernel + Bonet-Lok kernel gradient
    correction, SVD pseudo-inverse (Eq. 10, 11)
  - Constraint gradients (Eq. 12, 13)
  - Implicit plasticity in the loop (Eq. 14, Alg. 1): Drucker-Prager
    [Klar et al. 2016]
  - XSPH damping c = 0.01 (Eq. 20)
  - Position-correction distance constraint, eps = 0.25 r (Eq. 21)
  - Final F update with return mapping (Eq. 22)

Scene: sand column collapse -> should form a pile at roughly the friction angle.
"""

import numpy as np
from scipy.spatial import cKDTree
import os

# ---------------- parameters ----------------
dx = 0.008                     # particle spacing (= kernel radius r, per paper Table 2 note)
h = dx                         # SPH kernel radius r (support = 2h)
support = 2.0 * h
rho0 = 1000.0
E = 1e4
nu = 0.3
mu = E / (2 * (1 + nu))
lam = E * nu / ((1 + nu) * (1 - 2 * nu))
phi_f = np.deg2rad(35.0)       # friction angle
alpha_dp = np.sqrt(2.0 / 3.0) * 2.0 * np.sin(phi_f) / (3.0 - np.sin(phi_f))
dt = 2e-4
iters = 8
gravity = np.array([0.0, -9.8])
xsph_c = 0.01
d = 2                          # dimension

V0 = dx * dx
mass = rho0 * V0

DOMAIN = np.array([1.0, 0.6])

# ---------------- init: sand column ----------------
nx, ny = int(0.15 / dx), int(0.30 / dx)
xs, ys = np.meshgrid(np.arange(nx) * dx + 0.425, np.arange(ny) * dx + dx)
x = np.stack([xs.ravel(), ys.ravel()], axis=1)
rng = np.random.default_rng(0)
x += rng.uniform(-0.05 * dx, 0.05 * dx, x.shape)   # tiny jitter
N = len(x)
v = np.zeros_like(x)
F = np.tile(np.eye(2), (N, 1, 1))
print(f"N = {N} particles")

# ---------------- Wendland C2 kernel (2D), support 2h ----------------
W_SIG = 7.0 / (4.0 * np.pi * h * h)

def kernel_W(r):
    q = r / h
    out = np.zeros_like(q)
    m = q < 2.0
    t = 1.0 - 0.5 * q[m]
    out[m] = W_SIG * t**4 * (2.0 * q[m] + 1.0)
    return out

def kernel_dWdq(q):
    out = np.zeros_like(q)
    m = q < 2.0
    t = 1.0 - 0.5 * q[m]
    out[m] = W_SIG * (-5.0 * q[m]) * t**3
    return out

def kernel_gradW(dxv, r):
    # grad_{x_p} W(|x_p - x_b|) for dxv = x_p - x_b
    q = r / h
    coef = np.zeros_like(r)
    m = (q < 2.0) & (r > 1e-12)
    coef[m] = kernel_dWdq(q)[m] / (h * r[m])
    return dxv * coef[:, None]

# ---------------- SVD helpers ----------------
def batch_svd(Fs):
    U, S, Vt = np.linalg.svd(Fs)
    return U, S, Vt

def drucker_prager(S):
    """Return-mapping Z on singular values (Klar et al. 2016), batch."""
    eps = np.log(np.maximum(S, 1e-10))
    tr = eps.sum(axis=1)
    dev = eps - tr[:, None] / d
    dev_norm = np.linalg.norm(dev, axis=1)
    out = eps.copy()
    # Case II: expansion -> project to tip (cohesionless)
    case2 = tr > 0.0
    out[case2] = 0.0
    # Case III: shear yield
    dg = dev_norm + alpha_dp * (d * lam + 2.0 * mu) / (2.0 * mu) * tr
    case3 = (~case2) & (dg > 0.0) & (dev_norm > 1e-12)
    scale = 1.0 - dg[case3] / dev_norm[case3]
    out[case3] = dev[case3] * scale[:, None] + tr[case3, None] / d
    return np.exp(out)

def stvk_psi_and_dpsi(S):
    """Hencky StVK energy density and dPsi/dSigma per singular value."""
    logS = np.log(np.maximum(S, 1e-10))
    tr = logS.sum(axis=1)
    psi = mu * (logS**2).sum(axis=1) + 0.5 * lam * tr**2
    dpsi = (2.0 * mu * logS + lam * tr[:, None]) / np.maximum(S, 1e-10)
    return psi, dpsi

# ---------------- main loop ----------------
os.makedirs("frames", exist_ok=True)
frame = 0
steps_per_frame = int((1.0 / 60.0) / dt)

for step in range(int(3.0 / dt) + 1):
    # neighbor search on x^n
    tree = cKDTree(x)
    pairs_list = tree.query_ball_tree(tree, support)
    # flatten to (p, b) arrays excluding self
    counts = np.array([len(pl) - 1 for pl in pairs_list])
    P = np.repeat(np.arange(N), counts)
    B = np.array([b for i, pl in enumerate(pairs_list) for b in pl if b != i], dtype=np.int64)
    dxv = x[P] - x[B]
    rr = np.linalg.norm(dxv, axis=1)
    gW = kernel_gradW(dxv, rr)          # grad_{x_p} W_b(x_p)
    Wv = kernel_W(rr)

    # current volumes V^n = V0 det(F^n)
    Vn = V0 * np.linalg.det(F)

    # Bonet-Lok correction L_p = (sum_b V_b gradW ⊗ (x_b - x_p))^{-1} (SVD pseudo-inverse)
    A = np.zeros((N, 2, 2))
    np.add.at(A, P, Vn[B, None, None] * gW[:, :, None] * (-dxv)[:, None, :])
    Ua, Sa, Vta = np.linalg.svd(A)
    Sinv = np.where(Sa > 1e-6 * Sa.max(axis=1, keepdims=True).clip(min=1e-12), 1.0 / Sa, 0.0)
    L = np.einsum('nij,nj,nkj->nik', np.transpose(Vta, (0, 2, 1)), Sinv, Ua)

    # corrected kernel gradients per pair: g̃_pb = L_p gradW_b(x_p)
    gt = np.einsum('nij,nj->ni', L[P], gW)

    # external forces
    v = v + dt * gravity

    lam_p = np.zeros(N)

    for it in range(iters):
        # --- inelastic per-particle constraints (Jacobi) ---
        # velocity gradient (Eq. 11)
        gradv = np.zeros((N, 2, 2))
        dv = v[B] - v[P]
        np.add.at(gradv, P, Vn[B, None, None] * dv[:, :, None] * gt[:, None, :])
        # trial F and plastic projection (Eq. 14)
        Ftr = np.einsum('nij,njk->nik', np.eye(2)[None] + dt * gradv, F)
        U, S, Vt = batch_svd(Ftr)
        Sp = drucker_prager(S)
        Fp = np.einsum('nij,nj,njk->nik', U, Sp, Vt)
        psi, dpsi = stvk_psi_and_dpsi(Sp)
        C = np.sqrt(np.maximum(2.0 * psi, 0.0))
        # dC/dF = P(F)/C ; P = U diag(dpsi) Vt
        Cc = np.maximum(C, 1e-8)
        Pk = np.einsum('nij,nj,njk->nik', U, dpsi, Vt) / Cc[:, None, None]
        # G_p = (dC/dF) F^{nT}  (Eq. 12 core matrix)
        G = np.einsum('nij,nkj->nik', Pk, F)
        # pair gradients: grad_{x_b} C_p = V_b^n G_p g̃_pb   (b != p)
        gradC_b = Vn[B, None] * np.einsum('nij,nj->ni', G[P], gt)
        # self gradient: -sum_b
        gradC_p = np.zeros((N, 2))
        np.add.at(gradC_p, P, -gradC_b)
        # denominator sum_b 1/m |gradC|^2 (+ self)
        denom = np.zeros(N)
        np.add.at(denom, P, (gradC_b**2).sum(axis=1) / mass)
        denom += (gradC_p**2).sum(axis=1) / mass
        alpha_t = (1.0 / V0) / (dt * dt)
        dlam = (-C - alpha_t * lam_p) / (denom + alpha_t)
        active = C > 1e-9
        dlam = np.where(active, dlam, 0.0)
        lam_p += dlam
        # velocity updates (Eq. 18), Jacobi gather
        dv_acc = np.zeros((N, 2))
        np.add.at(dv_acc, B, gradC_b * dlam[P, None] / (mass * dt))
        dv_acc += gradC_p * dlam[:, None] / (mass * dt)
        v = v + dv_acc

        # --- position correction distance constraints (Eq. 21), on candidate pos ---
        xc = x + dt * v
        dxc = xc[P] - xc[B]
        dc = np.linalg.norm(dxc, axis=1)
        Cd = dc - (h - 0.25 * h)
        act = Cd < 0.0
        n_pb = np.where(dc[:, None] > 1e-12, dxc / np.maximum(dc, 1e-12)[:, None], 0.0)
        # hard constraint, symmetric mass: each pair contributes half to each side
        corr = np.where(act[:, None], -0.5 * Cd[:, None] * n_pb, 0.0)
        dvp = np.zeros((N, 2))
        np.add.at(dvp, P, corr)
        np.add.at(dvp, B, -corr)
        v = v + dvp / dt * 0.5   # relaxed Jacobi

        # --- boundary collisions (velocity level on candidate position) ---
        xc = x + dt * v
        for dim_i in range(2):
            lo = xc[:, dim_i] < dx
            hi = xc[:, dim_i] > DOMAIN[dim_i] - dx
            # friction: kill tangential proportionally when normal contact
            for m_, target in ((lo, dx), (hi, DOMAIN[dim_i] - dx)):
                if m_.any():
                    vn_needed = (target - x[m_, dim_i]) / dt
                    v_m = v[m_]
                    # Coulomb-ish: reduce tangential by mu_f * |dvn|
                    dvn = vn_needed - v_m[:, dim_i]
                    tdim = 1 - dim_i
                    vt = v_m[:, tdim]
                    fric = np.maximum(0.0, 1.0 - 0.4 * np.abs(dvn) / (np.abs(vt) + 1e-9))
                    v_m[:, tdim] = vt * fric
                    v_m[:, dim_i] = vn_needed
                    v[m_] = v_m

    # --- XSPH (Eq. 20) ---
    dvx = np.zeros((N, 2))
    np.add.at(dvx, P, Vn[B, None] * (v[B] - v[P]) * Wv[:, None])
    v = v + xsph_c * dvx

    # --- final F update w/ return mapping (Eq. 22) ---
    gradv = np.zeros((N, 2, 2))
    dv = v[B] - v[P]
    np.add.at(gradv, P, Vn[B, None, None] * dv[:, :, None] * gt[:, None, :])
    Fnew = np.einsum('nij,njk->nik', np.eye(2)[None] + dt * gradv, F)
    U, S, Vt = batch_svd(Fnew)
    Sp = drucker_prager(S)
    F = np.einsum('nij,nj,njk->nik', U, Sp, Vt)

    # --- position update ---
    x = x + dt * v
    x = np.clip(x, dx * 0.5, DOMAIN - dx * 0.5)

    if step % steps_per_frame == 0:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.6), dpi=90)
        ax.scatter(x[:, 0], x[:, 1], s=2.0, c='peru')
        ax.set_xlim(0, DOMAIN[0]); ax.set_ylim(0, DOMAIN[1])
        ax.set_aspect('equal'); ax.set_title(f"t = {step*dt:.2f}s")
        fig.savefig(f"frames/f{frame:04d}.png", bbox_inches='tight')
        plt.close(fig)
        print(f"frame {frame}  t={step*dt:.3f}  maxv={np.abs(v).max():.3f}  "
              f"meanJ={np.linalg.det(F).mean():.3f}")
        frame += 1

print("done")
