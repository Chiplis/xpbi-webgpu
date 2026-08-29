// XPBI compute kernels (WGSL). One module string; each pass is an entry point.
// Pipeline per substep (paper Alg. 1):
//   clearBuffer(cellTab) -> countParticles -> scanBlocks -> scanApply ->
//   scatter (sort + Vn + predictor) -> computeL ->
//   iterate { solveA -> solveB (incl. contacts + boundaries) -> [pbfA -> pbfB] } ->
//   xsph -> updateF -> integrate
//
// Buffer packing (10 storage buffer limit on this adapter):
//   pos.w           = material index as float value
//   aux.x/y/z/w     = Vn / lambda / plastic state / pbf lambda
//   Sp[3i..3i+2]    = mat3 columns (xyz), selfDv vector in the three w slots
//   Sp also aliases the scan scratch (disjoint pass usage)
import { WGSL_COMMON } from './common.js';

export function buildSimWGSL() {
  return WGSL_COMMON + /* wgsl */`

@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var<uniform> MATS : array<Material, 8>;

@group(0) @binding(2) var<storage, read_write> pos     : array<vec4f>;
@group(0) @binding(3) var<storage, read_write> velIn   : array<vec4f>;
@group(0) @binding(4) var<storage, read_write> velOut  : array<vec4f>;
@group(0) @binding(5) var<storage, read_write> Fbuf    : array<mat3x3f>;
@group(0) @binding(6) var<storage, read_write> Lbuf    : array<mat3x3f>;
@group(0) @binding(7) var<storage, read_write> aux     : array<vec4f>;
@group(0) @binding(8) var<storage, read_write> SpBuf   : array<vec4f>;
// scanAux aliases the Sp buffer (disjoint pass usage): block sums for the scan
@group(0) @binding(8) var<storage, read_write> scanAux : array<u32>;
// counting-sort grid [Hoetzlein 2014]: after the scan + atomicSub scatter,
// cellTab[c] = start index of cell c in sortedIdx; end = cellTab[c+1].
// cellTabR is a non-atomic alias of the same binding for read-only passes
// (atomic loads bypass caches on Apple GPUs and dominate neighbor loops).
@group(0) @binding(9) var<storage, read_write> cellTab : array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> cellTabR : array<u32>;
@group(0) @binding(10) var<storage, read_write> sortedIdx : array<u32>;

const WG : u32 = 128u;
const NIL : u32 = 0xffffffffu;

// pos.w = material index + per-particle color seed in the fraction (< 0.5).
// Stored as a float VALUE (not bits: u32 1..7 as f32 bits are denormals,
// which Metal flushes to zero). Truncation recovers the index.
fn matOf(i : u32) -> u32 { return u32(pos[i].w); }
// takes a material INDEX, checks the referenced material's kind
fn isSolid(mi : u32) -> bool { return MATS[mi].kind != MAT_WATER; }
fn loadSp(i : u32) -> mat3x3f {
  return mat3x3f(SpBuf[3u*i].xyz, SpBuf[3u*i+1u].xyz, SpBuf[3u*i+2u].xyz);
}
fn loadSelfDv(i : u32) -> vec3f {
  return vec3f(SpBuf[3u*i].w, SpBuf[3u*i+1u].w, SpBuf[3u*i+2u].w);
}
fn storeSpSelf(i : u32, m : mat3x3f, s : vec3f) {
  SpBuf[3u*i]    = vec4f(m[0], s.x);
  SpBuf[3u*i+1u] = vec4f(m[1], s.y);
  SpBuf[3u*i+2u] = vec4f(m[2], s.z);
}

// [begin, end) range of a neighbor cell in sortedIdx; (0,0) if out of grid
fn cellRange(nc : vec3i) -> vec2u {
  if (any(nc < vec3i(0)) || any(nc >= P.gridDims)) { return vec2u(0u, 0u); }
  let nci = cellIndex(nc);
  let beg = cellTabR[nci];
  var end = P.N;
  if (nci + 1u < P.numCells) { end = cellTabR[nci + 1u]; }
  return vec2u(beg, min(end, beg + 128u));
}

// ---------------- counting-sort grid ----------------
// (cell table is zeroed with clearBuffer on the command encoder — no dispatch)
@compute @workgroup_size(WG)
fn countParticles(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= P.N) { return; }
  let c = cellIndex(cellCoord(pos[i].xyz));
  atomicAdd(&cellTab[c], 1u);
}

// inclusive scan of cellTab, per 256-cell block; block totals -> scanAux
var<workgroup> scanSh : array<u32, 256>;
@compute @workgroup_size(256)
fn scanBlocks(@builtin(global_invocation_id) gid : vec3u,
              @builtin(local_invocation_id) lid : vec3u,
              @builtin(workgroup_id) wid : vec3u) {
  let i = gid.x;
  let l = lid.x;
  var v = 0u;
  if (i < P.numCells) { v = atomicLoad(&cellTab[i]); }
  scanSh[l] = v;
  workgroupBarrier();
  for (var off = 1u; off < 256u; off = off << 1u) {
    var add = 0u;
    if (l >= off) { add = scanSh[l - off]; }
    workgroupBarrier();
    scanSh[l] = scanSh[l] + add;
    workgroupBarrier();
  }
  if (i < P.numCells) { atomicStore(&cellTab[i], scanSh[l]); }
  if (l == 255u) { scanAux[wid.x] = scanSh[255u]; }
}

// add block offsets -> global inclusive prefix sums. Each workgroup computes
// its own exclusive block offset by summing raw block totals (block count is
// small, so the serial loop is cheap and saves a whole dispatch + barrier).
@compute @workgroup_size(256)
fn scanApply(@builtin(global_invocation_id) gid : vec3u,
             @builtin(local_invocation_id) lid : vec3u,
             @builtin(workgroup_id) wid : vec3u) {
  let i = gid.x;
  if (lid.x == 0u) {
    var run = 0u;
    for (var b = 0u; b < wid.x; b++) { run = run + scanAux[b]; }
    scanSh[0] = run;
  }
  workgroupBarrier();
  if (i >= P.numCells) { return; }
  let offv = scanSh[0];
  if (offv > 0u) {
    let v = atomicLoad(&cellTab[i]);
    atomicStore(&cellTab[i], v + offv);
  }
}

// scatter: atomicSub turns inclusive sums into cell start offsets.
// Also computes V^n = V0 det(F) (was its own pass) and applies the predictor.
@compute @workgroup_size(WG)
fn scatter(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= P.N) { return; }
  let c = cellIndex(cellCoord(pos[i].xyz));
  let dst = atomicSub(&cellTab[c], 1u) - 1u;
  sortedIdx[dst] = i;
  aux[i].x = P.V0 * clamp(determinant(Fbuf[i]), 0.1, 10.0);
  let v = velIn[i].xyz + P.dt * P.gravity;
  velIn[i] = vec4f(v, 0.0);
  aux[i].y = 0.0; // lambda
}

// ---------------- data reordering into cell order ----------------
// Neighbor gathers dominate cost and are memory-bound; permuting the particle
// arrays into sorted (cell) order every substep makes them cache-coherent and
// removes the sortedIdx indirection from all neighbor loops.
// SpBuf (3 vec4 / particle) doubles as staging: two rounds.
@compute @workgroup_size(WG)
fn stagePVA(@builtin(global_invocation_id) gid : vec3u) {
  let j = gid.x;
  if (j >= P.N) { return; }
  let i = sortedIdx[j];
  SpBuf[3u*j]    = pos[i];
  SpBuf[3u*j+1u] = velIn[i];
  SpBuf[3u*j+2u] = aux[i];
}
@compute @workgroup_size(WG)
fn applyPVA(@builtin(global_invocation_id) gid : vec3u) {
  let j = gid.x;
  if (j >= P.N) { return; }
  pos[j]   = SpBuf[3u*j];
  velIn[j] = SpBuf[3u*j+1u];
  velOut[j] = SpBuf[3u*j+1u];
  aux[j]   = SpBuf[3u*j+2u];
}
@compute @workgroup_size(WG)
fn stageF(@builtin(global_invocation_id) gid : vec3u) {
  let j = gid.x;
  if (j >= P.N) { return; }
  let F = Fbuf[sortedIdx[j]];
  SpBuf[3u*j]    = vec4f(F[0], 0.0);
  SpBuf[3u*j+1u] = vec4f(F[1], 0.0);
  SpBuf[3u*j+2u] = vec4f(F[2], 0.0);
}
@compute @workgroup_size(WG)
fn applyF(@builtin(global_invocation_id) gid : vec3u) {
  let j = gid.x;
  if (j >= P.N) { return; }
  Fbuf[j] = mat3x3f(SpBuf[3u*j].xyz, SpBuf[3u*j+1u].xyz, SpBuf[3u*j+2u].xyz);
}

// ---------------- kernel gradient correction L (Eq. 10) ----------------
@compute @workgroup_size(WG)
fn computeL(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= P.N) { return; }
  let xi = pos[i].xyz;
  var A = mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0));
  let cc = cellCoord(xi);
  for (var dz = -1; dz <= 1; dz++) {
  for (var dy = -1; dy <= 1; dy++) {
  for (var dx = -1; dx <= 1; dx++) {
    let rge = cellRange(cc + vec3i(dx, dy, dz));
    for (var j = rge.x; j < rge.y; j++) {
      let b = j;
      if (b != i) {
        let dxv = xi - pos[b].xyz;
        if (dot(dxv, dxv) < P.support * P.support) {
          let g = kernelGradW(dxv);
          let r = pos[b].xyz - xi;
          A = A + mat3x3f(g * r.x, g * r.y, g * r.z) * aux[b].x;
        }
      }
    }
  }}}
  // SVD pseudo-inverse for stability (paper Sec 3.2). A healthy neighborhood
  // gives A ~ identity, so directions with tiny singular values are DEFICIENT
  // (free surface, carved hole) — drop them instead of amplifying them.
  let s = svd3(A);
  let sig = s.sig;
  let tol = 0.2;
  var inv = vec3f(0.0);
  if (abs(sig.x) > tol) { inv.x = 1.0 / sig.x; }
  if (abs(sig.y) > tol) { inv.y = 1.0 / sig.y; }
  if (abs(sig.z) > tol) { inv.z = 1.0 / sig.z; }
  Lbuf[i] = diagMul(s.V, inv, s.U); // V diag(inv) U^T
}

// ---------------- velocity gradient (Eq. 11) ----------------
fn velocityGradient(i : u32, xi : vec3f) -> mat3x3f {
  var G = mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0));
  let Li = Lbuf[i];
  let vi = velIn[i].xyz;
  let cc = cellCoord(xi);
  for (var dz = -1; dz <= 1; dz++) {
  for (var dy = -1; dy <= 1; dy++) {
  for (var dx = -1; dx <= 1; dx++) {
    let rge = cellRange(cc + vec3i(dx, dy, dz));
    for (var j = rge.x; j < rge.y; j++) {
      let b = j;
      if (b != i && isSolid(matOf(b))) {
        let dxv = xi - pos[b].xyz;
        if (dot(dxv, dxv) < P.support * P.support) {
          let gt = Li * kernelGradW(dxv);
          let dv = (velIn[b].xyz - vi) * aux[b].x;
          G = G + mat3x3f(dv * gt.x, dv * gt.y, dv * gt.z);
        }
      }
    }
  }}}
  return G;
}

// ---------------- inelastic constraints: pass A (Eq. 14, 17, Alg. 1) ----------------
@compute @workgroup_size(WG)
fn solveA(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= P.N) { return; }
  let mi = matOf(i);
  if (!isSolid(mi)) {
    storeSpSelf(i, mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0)), vec3f(0.0));
    return;
  }
  var m = MATS[mi];
  m = hardenedMat(m, aux[i].z);
  let xi = pos[i].xyz;
  let Fn = Fbuf[i];
  // trial F with current velocities (fixed-point plasticity in the loop)
  let gradV = velocityGradient(i, xi);
  let I3 = mat3x3f(1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0);
  let Ftr = (I3 + P.dt * gradV) * Fn;
  let sv = svd3(Ftr);
  let rm = returnMap(sv.sig, m, aux[i].z);
  let sig = rm.sig;
  // constraint C = sqrt(2 Psi) with alpha = 1/V0 (Eq. 6)
  let psi = stvkPsi(sig, m.mu, m.lam);
  let C = sqrt(max(2.0 * psi, 0.0));
  if (C < 1e-7) {
    storeSpSelf(i, mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0)), vec3f(0.0));
    return;
  }
  // dC/dF = P(F)/C at the projected state
  let dpsi = stvkDPsi(sig, m.mu, m.lam);
  let Pk = diagMul(sv.U, dpsi / C, sv.V);
  // G_i = (dC/dF) F^{nT}, per-neighbor grad = V_b * G_i (L_i gradW_b) (Eq. 12)
  let GL = (Pk * transpose(Fn)) * Lbuf[i];
  var denom = 0.0;
  var sgrad = vec3f(0.0);
  var cdot = 0.0;                       // grad C . v  (for XPBD damping)
  let cc = cellCoord(xi);
  for (var dz = -1; dz <= 1; dz++) {
  for (var dy = -1; dy <= 1; dy++) {
  for (var dx = -1; dx <= 1; dx++) {
    let rge = cellRange(cc + vec3i(dx, dy, dz));
    for (var j = rge.x; j < rge.y; j++) {
      let b = j;
      if (b != i && isSolid(matOf(b))) {
        let dxv = xi - pos[b].xyz;
        if (dot(dxv, dxv) < P.support * P.support) {
          let gb = aux[b].x * (GL * kernelGradW(dxv));
          denom = denom + dot(gb, gb) / MATS[matOf(b)].mass;
          sgrad = sgrad - gb;
          cdot = cdot + dot(gb, velIn[b].xyz);
        }
      }
    }
  }}}
  denom = denom + dot(sgrad, sgrad) / m.mass;
  cdot = cdot + dot(sgrad, velIn[i].xyz);
  let alphaT = (1.0 / P.V0) / (P.dt * P.dt);
  // damped XPBD update [Macklin et al. 2016, Eq. 26]: gamma = P.kDamp
  let g = P.kDamp;
  let dlam = P.omega * (-C - alphaT * aux[i].y - g * P.dt * cdot)
           / ((1.0 + g) * denom + alphaT);
  aux[i].y = aux[i].y + dlam;
  storeSpSelf(i, GL * dlam, sgrad * dlam);
}

// ---------------- inelastic constraints: pass B (gather Δv, Eq. 18) ----------------
// Also applies the position-correction contact constraint (Eq. 21) and
// boundary conditions in the same dispatch (Jacobi: neighbor candidates use
// pre-update velocities, which is one half-pass stale — acceptable).
@compute @workgroup_size(WG)
fn solveB(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= P.N) { return; }
  let mi = matOf(i);
  var dv = vec3f(0.0);
  let xi = pos[i].xyz;
  let m = MATS[mi];
  let solid = isSolid(mi);
  let cc = cellCoord(xi);
  if (solid) {
    dv = loadSelfDv(i); // own constraint self-gradient term
    let Vi = aux[i].x;
    for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let rge = cellRange(cc + vec3i(dx, dy, dz));
      for (var j = rge.x; j < rge.y; j++) {
        let p = j;
        if (p != i && isSolid(matOf(p))) {
          let dxv = pos[p].xyz - xi;  // x_p - x_i
          if (dot(dxv, dxv) < P.support * P.support) {
            // grad_{x_i} C_p = V_i * G_p L_p gradW_i(x_p); Sp already holds dlam*G*L
            dv = dv + Vi * (loadSp(p) * kernelGradW(dxv));
          }
        }
      }
    }}}
  }
  // cap the per-iteration constraint impulse: far above any healthy update,
  // but arrests Jacobi feedback blowups seeded by sharp velocity kicks
  var dvv = dv / (m.mass * P.dt);
  let dvl = length(dvv);
  let dvCap = 0.2 * P.support / P.dt;
  if (dvl > dvCap) { dvv = dvv * (dvCap / dvl); }
  var v = velIn[i].xyz + dvv;
  if (solid) {
    // pairwise distance constraint on candidate positions (paper Eq. 21)
    var corr = vec3f(0.0);
    let xci = xi + P.dt * v;
    let minD = 0.75 * P.h;
    for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let rge = cellRange(cc + vec3i(dx, dy, dz));
      for (var j = rge.x; j < rge.y; j++) {
        let b = j;
        if (b != i && isSolid(matOf(b))) {
          let xcb = pos[b].xyz + P.dt * velIn[b].xyz;
          let dxv = xci - xcb;
          let dist = length(dxv);
          if (dist < minD && dist > 1e-9) {
            let mb = MATS[matOf(b)].mass;
            let w = mb / (m.mass + mb);
            corr = corr + dxv / dist * ((minD - dist) * w);
          }
        }
      }
    }}}
    v = v + corr * (P.contactOmega / P.dt);
  }
  v = applyBoundary(xi + P.dt * v, v, xi, m.boundFriction);
  velOut[i] = vec4f(v, 0.0);
}

// ---------------- contacts: position correction (Eq. 21) + boundaries ----------------
fn applyBoundary(xcIn : vec3f, vIn : vec3f, xn : vec3f, fric : f32) -> vec3f {
  var vv = vIn;
  var xq = xcIn;
  for (var k = 0; k < 3; k++) {
    let lo = P.domainMin[k];
    let hi = P.domainMax[k];
    if (xq[k] < lo || xq[k] > hi) {
      let bound = select(hi, lo, xq[k] < lo);
      let vnNew = (bound - xn[k]) / P.dt;
      let dvn = abs(vnNew - vv[k]);
      var vt = vv; vt[k] = 0.0;
      let vtl = length(vt);
      if (vtl > 1e-9) {
        let drop = min(fric * dvn, vtl);
        vv = vv - vt / vtl * drop;
      }
      vv[k] = vnNew;
      xq = xn + P.dt * vv;
    }
  }
  // kinematic sphere collider: velocity-level projection (remove approaching
  // relative velocity) + a gentle bounded depenetration drift
  if (P.colliderActive > 0.5) {
    let d = xq - P.colliderPos;
    let dist = length(d);
    if (dist < P.colliderR && dist > 1e-9) {
      let n = d / dist;
      // pin the outward relative speed to a bounded depenetration target —
      // idempotent across solver iterations (never accumulates)
      let vOut = dot(vv - P.colliderVel, n);
      let vGoal = min((P.colliderR - dist) / P.dt, 0.6);
      if (vOut < vGoal) { vv = vv + n * (vGoal - vOut); }
    }
  }
  return vv;
}

// ---------------- PBF density constraint for water [Macklin & Muller 2013] ----------------
@compute @workgroup_size(WG)
fn pbfA(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= P.N) { return; }
  if (isSolid(matOf(i))) { aux[i].w = 0.0; return; }
  let m = MATS[matOf(i)];
  let rho0 = m.density;
  let xn = pos[i].xyz;
  let xci = xn + P.dt * velIn[i].xyz;
  var rho = m.mass * kernelPoly6(0.0);
  var gradSum = vec3f(0.0);
  var grad2 = 0.0;
  let cc = cellCoord(xn);
  for (var dz = -1; dz <= 1; dz++) {
  for (var dy = -1; dy <= 1; dy++) {
  for (var dx = -1; dx <= 1; dx++) {
    let rge = cellRange(cc + vec3i(dx, dy, dz));
    for (var j = rge.x; j < rge.y; j++) {
      let b = j;
      if (b != i) {
        let xcb = pos[b].xyz + P.dt * velIn[b].xyz;
        let dxv = xci - xcb;
        if (dot(dxv, dxv) < P.support * P.support) {
          let mb = MATS[matOf(b)].mass;
          rho = rho + mb * kernelPoly6(length(dxv));
          let g = kernelSpikyGrad(dxv) * (mb / rho0);
          gradSum = gradSum + g;
          grad2 = grad2 + dot(g, g);
        }
      }
    }
  }}}
  let C = max(rho / rho0 - 1.0, 0.0); // unilateral: only fix compression
  aux[i].w = -C / (grad2 + dot(gradSum, gradSum) + P.pbfEps);
}

@compute @workgroup_size(WG)
fn pbfB(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= P.N) { return; }
  let mi = matOf(i);
  let xn = pos[i].xyz;
  let xci = xn + P.dt * velIn[i].xyz;
  var dp = vec3f(0.0);
  let cc = cellCoord(xn);
  let li = aux[i].w;
  let iAmWater = !isSolid(mi);
  for (var dz = -1; dz <= 1; dz++) {
  for (var dy = -1; dy <= 1; dy++) {
  for (var dx = -1; dx <= 1; dx++) {
    let rge = cellRange(cc + vec3i(dx, dy, dz));
    for (var j = rge.x; j < rge.y; j++) {
      let b = j;
      if (b != i) {
        let bWater = !isSolid(matOf(b));
        if (iAmWater || bWater) {
          let xcb = pos[b].xyz + P.dt * velIn[b].xyz;
          let dxv = xci - xcb;
          if (dot(dxv, dxv) < P.support * P.support) {
            let mw = MATS[select(matOf(i), matOf(b), bWater)]; // a water material
            let lb = aux[b].w;
            let w = kernelPoly6(length(dxv));
            let w0 = kernelPoly6(0.2 * P.support);
            let sCorr = -P.sCorrK * pow(w / max(w0, 1e-12), 4.0);
            dp = dp + (li + lb + sCorr) * kernelSpikyGrad(dxv) * (mw.mass / mw.density);
          }
        }
      }
    }
  }}}
  // Δp is a position-space correction; convert to velocity. Solids only feel
  // half (one-way-ish coupling keeps the levee from being blasted apart).
  let scale = select(0.5, 1.0, iAmWater);
  velOut[i] = vec4f(velIn[i].xyz + dp * (scale / P.dt), 0.0);
}

// ---------------- XSPH (Eq. 20) ----------------
@compute @workgroup_size(WG)
fn xsph(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= P.N) { return; }
  let xi = pos[i].xyz;
  let vi = velIn[i].xyz;
  let mi = matOf(i);
  let iSolid = isSolid(mi);
  var acc = vec3f(0.0);
  let cc = cellCoord(xi);
  for (var dz = -1; dz <= 1; dz++) {
  for (var dy = -1; dy <= 1; dy++) {
  for (var dx = -1; dx <= 1; dx++) {
    let rge = cellRange(cc + vec3i(dx, dy, dz));
    for (var j = rge.x; j < rge.y; j++) {
      let b = j;
      if (b != i && (isSolid(matOf(b)) == iSolid)) {
        let dxv = xi - pos[b].xyz;
        if (dot(dxv, dxv) < P.support * P.support) {
          acc = acc + aux[b].x * (velIn[b].xyz - vi) * kernelW(length(dxv));
        }
      }
    }
  }}}
  // water gets stronger XSPH (PBF viscosity), solids the paper's c = 0.01
  var c = P.xsphC;
  if (!iSolid) { c = P.xsphC * 10.0; }
  velOut[i] = vec4f(vi + c * acc, 0.0);
}

// ---------------- final F update (Eq. 22) + plastic state ----------------
@compute @workgroup_size(WG)
fn updateF(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= P.N) { return; }
  let mi = matOf(i);
  if (!isSolid(mi)) { return; }
  var m = MATS[mi];
  m = hardenedMat(m, aux[i].z);
  let gradV = velocityGradient(i, pos[i].xyz);
  let I3 = mat3x3f(1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0);
  let Ftr = (I3 + P.dt * gradV) * Fbuf[i];
  let sv = svd3(Ftr);
  let rm = returnMap(sv.sig, m, aux[i].z);
  Fbuf[i] = diagMul(sv.U, safeSig(rm.sig), sv.V);
  aux[i].z = rm.state;
}

// ---------------- integrate ----------------
@compute @workgroup_size(WG)
fn integrate(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= P.N) { return; }
  // CFL-style safety clamp: never move more than ~half the kernel support
  // in one substep (keeps neighborhoods valid; invisible in normal motion)
  var v = velIn[i].xyz;
  let vmax = 0.45 * P.support / P.dt;
  let vl = length(v);
  if (vl > vmax) { v = v * (vmax / vl); }
  velIn[i] = vec4f(v, 0.0);
  var x = pos[i].xyz + P.dt * v;
  x = clamp(x, P.domainMin, P.domainMax);
  pos[i] = vec4f(x, pos[i].w);
}
`;
}
