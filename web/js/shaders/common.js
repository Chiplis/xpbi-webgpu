// Shared WGSL: parameter structs, SPH kernels, 3x3 SVD, constitutive models,
// plastic return mappings. Implements the math of XPBI (arXiv:2405.11694) and
// its referenced constitutive papers:
//   Drucker-Prager  [Klar et al. 2016]
//   NACC            [Wolper et al. 2019]
//   Von Mises       [Li et al. 2022]
//   Herschel-Bulkley[Yue et al. 2015]
//   Snow (clamp)    [Stomakhin et al. 2013]
//   StVK w/ Hencky strain (paper Eq. 3)

export const WGSL_COMMON = /* wgsl */`

const MAT_ELASTIC : u32 = 0u;
const MAT_DP      : u32 = 1u;   // Drucker-Prager sand
const MAT_NACC    : u32 = 2u;   // Cam-Clay snow / fracture
const MAT_VM      : u32 = 3u;   // Von Mises plasticine/metal
const MAT_HB      : u32 = 4u;   // Herschel-Bulkley viscoplastic
const MAT_SNOW    : u32 = 5u;   // Stomakhin 2013 singular value clamp
const MAT_WATER   : u32 = 6u;   // PBF fluid (no F constraint)

struct Material {
  mu       : f32,
  lam      : f32,
  density  : f32,
  kind     : u32,
  // model params: DP:(alpha, cohesion) NACC:(alpha0(unused at runtime), beta, xi, M)
  // VM:(yield) HB:(yield, hexp, eta) SNOW:(thetaC, thetaS, xi)
  p0 : f32, p1 : f32, p2 : f32, p3 : f32,
  color : vec4f,
  boundFriction : f32,
  mass  : f32,       // rho * V0
  pad0 : f32, pad1 : f32,
};

struct Params {
  gridOrigin : vec3f,
  cellSize   : f32,
  gridDims   : vec3i,
  numCells   : u32,
  domainMin  : vec3f,
  dt         : f32,
  domainMax  : vec3f,
  h          : f32,       // kernel radius r (== particle spacing)
  gravity    : vec3f,
  support    : f32,       // 2h
  colliderPos : vec3f,
  colliderR   : f32,
  colliderVel : vec3f,
  colliderActive : f32,
  N          : u32,
  V0         : f32,       // rest volume per particle
  xsphC      : f32,
  omega      : f32,       // Jacobi under-relaxation for inelastic constraints
  pbfEps     : f32,       // PBF CFM relaxation
  sCorrK     : f32,
  time       : f32,
  contactOmega : f32,
  kDamp      : f32,       // XPBD constraint damping gamma [Macklin et al. 2016]
  pad0 : f32, pad1 : f32, pad2 : f32,
};

// ---------- Wendland C2 kernel, 3D, support radius = 2h ----------
fn kernelW(r : f32) -> f32 {
  let q = r / P.h;
  if (q >= 2.0) { return 0.0; }
  let t = 1.0 - 0.5 * q;
  let sig = 21.0 / (16.0 * 3.14159265 * P.h * P.h * P.h);
  return sig * t*t*t*t * (2.0*q + 1.0);
}
// grad_{xp} W(|xp-xb|), dxv = xp - xb
fn kernelGradW(dxv : vec3f) -> vec3f {
  let r = length(dxv);
  let q = r / P.h;
  if (q >= 2.0 || r < 1e-9) { return vec3f(0.0); }
  let t = 1.0 - 0.5 * q;
  let sig = 21.0 / (16.0 * 3.14159265 * P.h * P.h * P.h);
  let dwdq = sig * (-5.0 * q) * t*t*t;
  return dxv * (dwdq / (P.h * r));
}

// poly6 / spiky for PBF (support = 2h to match neighborhood)
fn kernelPoly6(r : f32) -> f32 {
  let H = P.support;
  if (r >= H) { return 0.0; }
  let x = H*H - r*r;
  return 315.0 / (64.0 * 3.14159265 * pow(H, 9.0)) * x*x*x;
}
fn kernelSpikyGrad(dxv : vec3f) -> vec3f {
  let H = P.support;
  let r = length(dxv);
  if (r >= H || r < 1e-9) { return vec3f(0.0); }
  let t = H - r;
  return dxv * (-45.0 / (3.14159265 * pow(H, 6.0)) * t * t / r);
}

// ---------- grid helpers ----------
fn cellCoord(p : vec3f) -> vec3i {
  let c = vec3i(floor((p - P.gridOrigin) / P.cellSize));
  return clamp(c, vec3i(0), P.gridDims - vec3i(1));
}
fn cellIndex(c : vec3i) -> u32 {
  return u32(c.x) + u32(c.y) * u32(P.gridDims.x) + u32(c.z) * u32(P.gridDims.x * P.gridDims.y);
}

// ---------- 3x3 SVD via Jacobi eigen-decomposition of F^T F ----------
struct SVD { U : mat3x3f, sig : vec3f, V : mat3x3f };

fn jacobiRot(Sm : ptr<function, mat3x3f>, Vm : ptr<function, mat3x3f>, p : i32, q : i32) {
  let Spq = (*Sm)[q][p]; // column-major: [col][row]; symmetric so either
  let Spp = (*Sm)[p][p];
  let Sqq = (*Sm)[q][q];
  if (abs(Spq) < 1e-12) { return; }
  let tau = (Sqq - Spp) / (2.0 * Spq);
  var t : f32;
  if (tau >= 0.0) { t = 1.0 / (tau + sqrt(1.0 + tau*tau)); }
  else            { t = 1.0 / (tau - sqrt(1.0 + tau*tau)); }
  let c = 1.0 / sqrt(1.0 + t*t);
  let s = t * c;
  // S = J^T S J where J is rotation in (p,q) plane
  var Sn = *Sm;
  for (var k = 0; k < 3; k++) {
    let Skp = (*Sm)[p][k];
    let Skq = (*Sm)[q][k];
    Sn[p][k] = c * Skp - s * Skq;
    Sn[q][k] = s * Skp + c * Skq;
  }
  var S2 = Sn;
  for (var k = 0; k < 3; k++) {
    let Spk = Sn[k][p];
    let Sqk = Sn[k][q];
    S2[k][p] = c * Spk - s * Sqk;
    S2[k][q] = s * Spk + c * Sqk;
  }
  *Sm = S2;
  var Vn = *Vm;
  for (var k = 0; k < 3; k++) {
    let Vkp = (*Vm)[p][k];
    let Vkq = (*Vm)[q][k];
    Vn[p][k] = c * Vkp - s * Vkq;
    Vn[q][k] = s * Vkp + c * Vkq;
  }
  *Vm = Vn;
}

fn svd3(F : mat3x3f) -> SVD {
  var S = transpose(F) * F;
  var V = mat3x3f(1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0);
  for (var sweep = 0; sweep < 6; sweep++) {
    jacobiRot(&S, &V, 0, 1);
    jacobiRot(&S, &V, 0, 2);
    jacobiRot(&S, &V, 1, 2);
  }
  var sig = vec3f(sqrt(max(S[0][0], 0.0)), sqrt(max(S[1][1], 0.0)), sqrt(max(S[2][2], 0.0)));
  // sort descending (columns of V follow)
  var v0 = V[0]; var v1 = V[1]; var v2 = V[2];
  if (sig.x < sig.y) { let ts = sig.x; sig.x = sig.y; sig.y = ts; let tv = v0; v0 = v1; v1 = tv; }
  if (sig.x < sig.z) { let ts = sig.x; sig.x = sig.z; sig.z = ts; let tv = v0; v0 = v2; v2 = tv; }
  if (sig.y < sig.z) { let ts = sig.y; sig.y = sig.z; sig.z = ts; let tv = v1; v1 = v2; v2 = tv; }
  // ensure det(V) = +1
  if (dot(cross(v0, v1), v2) < 0.0) { v2 = -v2; }
  V = mat3x3f(v0, v1, v2);
  // U columns
  var u0 : vec3f; var u1 : vec3f; var u2 : vec3f;
  let eps = 1e-8;
  if (sig.x > eps) { u0 = normalize(F * v0); } else { u0 = vec3f(1.0, 0.0, 0.0); }
  if (sig.y > eps) {
    u1 = F * v1 - dot(F * v1, u0) * u0;
    let l = length(u1);
    if (l > eps) { u1 = u1 / l; } else { u1 = normalize(cross(vec3f(0.0,1.0,0.0), u0) + vec3f(1e-4)); }
  } else {
    var a = cross(u0, vec3f(0.0, 1.0, 0.0));
    if (length(a) < 1e-4) { a = cross(u0, vec3f(1.0, 0.0, 0.0)); }
    u1 = normalize(a);
  }
  u2 = cross(u0, u1);
  // sign convention: det(U)=+1, negative sigma_min if F inverted
  if (determinant(F) < 0.0) { sig.z = -sig.z; }
  var U = mat3x3f(u0, u1, u2);
  return SVD(U, sig, V);
}

fn diagMul(U : mat3x3f, d : vec3f, V : mat3x3f) -> mat3x3f {
  return mat3x3f(U[0] * d.x, U[1] * d.y, U[2] * d.z) * transpose(V);
}

// ---------- StVK with Hencky strain (paper Eq. 3) ----------
fn safeSig(sig : vec3f) -> vec3f {
  return clamp(sig, vec3f(1e-3), vec3f(1e3));
}
fn stvkPsi(sig : vec3f, mu : f32, lam : f32) -> f32 {
  let e = log(safeSig(sig));
  let tr = e.x + e.y + e.z;
  return mu * dot(e, e) + 0.5 * lam * tr * tr;
}
// dPsi/dsigma_i
fn stvkDPsi(sig : vec3f, mu : f32, lam : f32) -> vec3f {
  let s = safeSig(sig);
  let e = log(s);
  let tr = e.x + e.y + e.z;
  return (2.0 * mu * e + vec3f(lam * tr)) / s;
}

// ---------- Return mappings Z(sigma) ----------

// Drucker-Prager [Klar et al. 2016], Hencky space, cohesionless + optional cohesion shift
fn rmDruckerPrager(sig : vec3f, m : Material) -> vec3f {
  let alpha = m.p0;
  let coh   = m.p1; // shift of trace (simple cohesion)
  let e = log(safeSig(sig)) - vec3f(coh);
  let tr = e.x + e.y + e.z;
  let dev = e - vec3f(tr / 3.0);
  let dn = length(dev);
  if (dn < 1e-9 || tr > 0.0) {
    // Case II: expanding -> project to tip
    return exp(vec3f(coh));
  }
  let dg = dn + alpha * (3.0 * m.lam + 2.0 * m.mu) / (2.0 * m.mu) * tr;
  if (dg <= 0.0) { return sig; } // Case I: elastic
  let en = e - (dg / dn) * dev;  // Case III: project onto cone
  return exp(en + vec3f(coh));
}

// Von Mises [Li et al. 2022], Hencky deviatoric projection
fn rmVonMises(sig : vec3f, m : Material) -> vec3f {
  let sigY = m.p0;
  let e = log(safeSig(sig));
  let tr = e.x + e.y + e.z;
  let dev = e - vec3f(tr / 3.0);
  let dn = length(dev);
  let dg = dn - sigY / (2.0 * m.mu);
  if (dg <= 0.0 || dn < 1e-9) { return sig; }
  let en = e - (dg / dn) * dev;
  return exp(en);
}

// Herschel-Bulkley viscoplasticity [Yue et al. 2015], their exact b^E formulation.
// Works on the unit-determinant left Cauchy-Green tensor b̄ᵉ (principal space:
// b̄_i = sigma_i^2 * J^{-2/3}). Trial deviatoric Kirchhoff stress s = mu dev(b̄ᵉ).
// Yield Phi(s)=||s||-sqrt(2/3) sigmaY (their Eq. 3); flow rate gamma =
// (Phi/eta)^{1/h} (Eq. 7). Backward Euler reduces to a scalar equation (Eq. 15/16):
//   eta^{1/h} (s - sPre) + 2 mũ dt (s - sqrt(2/3) sigmaY)^{1/h} = 0,
//   mũ = (1/3) tr(b̄ᵉ_pre) mu.
// h == 1 or eta -> 0: analytic (Eq. 17); general h: bisection (their choice).
fn rmHerschelBulkley(sig : vec3f, m : Material) -> vec3f {
  let sigY = m.p0;
  let hexp = m.p1;
  let eta  = m.p2;
  let s3 = safeSig(sig);
  let J = s3.x * s3.y * s3.z;
  let bbar = s3 * s3 * pow(J, -2.0 / 3.0);
  let trb = bbar.x + bbar.y + bbar.z;
  let devb = bbar - vec3f(trb / 3.0);
  let sPre = m.mu * length(devb);                 // ||s_pre||_F in principal space
  let sY = sqrt(2.0 / 3.0) * sigY;
  if (sPre <= sY || sPre < 1e-12) { return sig; }
  let muT = (trb / 3.0) * m.mu;                   // mũ
  var sNew : f32;
  if (abs(hexp - 1.0) < 1e-3 || eta < 1e-9) {
    sNew = sPre - (sPre - sY) / (1.0 + eta / (2.0 * muT * P.dt));
  } else {
    // bisection on [sY, sPre] for g(s) = eta^{1/h}(s - sPre) + 2 mũ dt (s - sY)^{1/h}
    let ih = 1.0 / hexp;
    let ei = pow(eta, ih);
    var lo = sY;
    var hi = sPre;
    for (var it = 0; it < 18; it++) {
      let mid = 0.5 * (lo + hi);
      let g = ei * (mid - sPre) + 2.0 * muT * P.dt * pow(max(mid - sY, 0.0), ih);
      if (g > 0.0) { hi = mid; } else { lo = mid; }
    }
    sNew = 0.5 * (lo + hi);
  }
  // shrink deviator, keep trace (their Eq. 18), then renormalize to det = 1
  var bNew = devb * (sNew / sPre) + vec3f(trb / 3.0);
  bNew = max(bNew, vec3f(1e-6));
  let detB = bNew.x * bNew.y * bNew.z;
  bNew = bNew * pow(detB, -1.0 / 3.0);
  // back to singular values, restoring the (unchanged) volumetric part J
  return sqrt(bNew) * pow(J, 1.0 / 3.0);
}

// Snow clamp [Stomakhin et al. 2013]: sigma in [1-thetaC, 1+thetaS]
fn rmSnowClamp(sig : vec3f, m : Material) -> vec3f {
  return clamp(sig, vec3f(1.0 - m.p0), vec3f(1.0 + m.p1));
}

// NACC [Wolper et al. 2019] — their exact formulation on b^E = F F^T
// (principal space: b̂ = Sigma^2). state = alpha (hardening variable).
//   p = -(kappa/2)(J^2 - 1),  ŝ = mu J^{-2/3} dev(Sigma^2),  q = sqrt(3/2)||ŝ||
//   yield y = (1+2beta) q^2 + M^2 (p + beta p0)(p - p0),
//   p0 = kappa (1e-5 + sinh(xi max(-alpha, 0))),  kappa = (2/3)mu + lam.
struct NaccResult { sig : vec3f, alpha : f32 };
fn rmNACC(sig : vec3f, m : Material, alphaIn : f32) -> NaccResult {
  let beta = m.p1;
  let xi   = m.p2;
  let Mf   = m.p3;
  let mu = m.mu;
  let kappa = (2.0 / 3.0) * mu + m.lam;
  var alpha = alphaIn;
  let s3 = safeSig(sig);
  let Jtr = s3.x * s3.y * s3.z;
  let b2 = s3 * s3;
  let trb = b2.x + b2.y + b2.z;
  let devb = b2 - vec3f(trb / 3.0);
  let sHat = mu * pow(Jtr, -2.0 / 3.0) * devb;      // deviatoric Kirchhoff stress
  let sNorm = length(sHat);
  let q = sqrt(1.5) * sNorm;
  let p = -0.5 * kappa * (Jtr * Jtr - 1.0);
  // clamp hardening state: sinh(xi*|alpha|) diverges violently at real-time
  // timesteps otherwise (paper runs dt <= 4e-5 s where this is benign)
  alpha = clamp(alpha, -0.25, 0.4);
  let p0 = kappa * (0.00001 + sinh(min(xi * max(-alpha, 0.0), 4.0)));
  let b = 1.0 + 2.0 * beta;
  var outSig = sig;
  if (p > p0) {
    // Case 1: too compressed -> project to (p0, 0) tip
    let Jn = sqrt(max(-2.0 * p0 / kappa + 1.0, 1e-6));
    outSig = vec3f(pow(Jn, 1.0 / 3.0));
    alpha = alpha + log(Jtr / Jn);
  } else if (p < -beta * p0) {
    // Case 2: too stretched -> project to (-beta p0, 0) tip
    let Jn = sqrt(max(2.0 * beta * p0 / kappa + 1.0, 1e-6));
    outSig = vec3f(pow(Jn, 1.0 / 3.0));
    alpha = alpha + log(Jtr / Jn);
  } else {
    let y = b * q * q + Mf * Mf * (p + beta * p0) * (p - p0);
    if (y > 1e-4 && sNorm > 1e-9) {
      // "fracture-friendly" case-3 hardening: intersect line through the
      // trial state and ellipse center with the yield ellipse (supp. Alg. 2)
      if (p0 > 1e-4 && p < p0 - 1e-4 && p > -beta * p0 + 1e-4) {
        let pc = (1.0 - beta) * p0 * 0.5;
        let lenDir = length(vec2f(pc - p, -q));
        if (lenDir > 1e-9) {
          let Dp = (pc - p) / lenDir;
          let Dq = -q / lenDir;
          let Aq = Mf * Mf * Dp * Dp + b * Dq * Dq;
          let Bq = Mf * Mf * Dp * (2.0 * pc - p0 + beta * p0);
          let Cq = Mf * Mf * (pc + beta * p0) * (pc - p0);
          let disc = Bq * Bq - 4.0 * Aq * Cq;
          if (disc > 0.0 && abs(Aq) > 1e-12) {
            let l1 = (-Bq + sqrt(disc)) / (2.0 * Aq);
            let l2 = (-Bq - sqrt(disc)) / (2.0 * Aq);
            let px1 = pc + l1 * Dp;
            let px2 = pc + l2 * Dp;
            var px = px1;
            if ((p - pc) * (px2 - pc) > 0.0) { px = px2; }
            let Jx2 = -2.0 * px / kappa + 1.0;
            if (Jx2 > 1e-8) {
              let Jx = sqrt(Jx2);
              if (Jx > 1e-4) { alpha = alpha + log(Jtr / Jx); }
            }
          }
        }
      }
      // Case 3 strain projection: q back onto ellipse at fixed p, keep tr(b)
      let qN = Mf * sqrt(max(-(p + beta * p0) * (p - p0), 0.0) / b);
      let sNewNorm = qN / sqrt(1.5);
      var bNew = (pow(Jtr, 2.0 / 3.0) / mu) * (sNewNorm / sNorm) * sHat + vec3f(trb / 3.0);
      bNew = max(bNew, vec3f(1e-6));
      outSig = sqrt(bNew);
    }
  }
  return NaccResult(outSig, alpha);
}

// hardening for classic snow: state = Jp, mu/lam scaled by exp(xi(1-Jp))
fn hardenedMat(m : Material, state : f32) -> Material {
  var mm = m;
  if (m.kind == MAT_SNOW) {
    let s = exp(min(m.p2 * (1.0 - state), 5.0));
    mm.mu = m.mu * s;
    mm.lam = m.lam * s;
  }
  return mm;
}

// Full return mapping dispatch. Returns projected sigma and new state.
struct RMResult { sig : vec3f, state : f32 };
fn returnMap(sig : vec3f, m : Material, state : f32) -> RMResult {
  switch (m.kind) {
    case 1u: { return RMResult(rmDruckerPrager(sig, m), state); }
    case 2u: { let r = rmNACC(sig, m, state); return RMResult(r.sig, r.alpha); }
    case 3u: { return RMResult(rmVonMises(sig, m), state); }
    case 4u: { return RMResult(rmHerschelBulkley(sig, m), state); }
    case 5u: {
      let mm = hardenedMat(m, state);
      let sNew = rmSnowClamp(sig, mm);
      // Jp update: plastic part absorbs det change
      let jOld = sig.x * sig.y * sig.z;
      let jNew = sNew.x * sNew.y * sNew.z;
      let st = clamp(state * jOld / max(jNew, 1e-6), 0.2, 2.0);
      return RMResult(sNew, st);
    }
    default: { return RMResult(sig, state); }
  }
}
`;
