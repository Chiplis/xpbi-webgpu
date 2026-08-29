// Scene definitions: particle sampling, materials, solver settings.
// Material parameter sets follow XPBI Table 2 conventions:
//   DP:(alpha from phi_f, cohesion) NACC:(alpha0->state, beta, xi, M)
//   VM:(sigmaY) HB:(sigmaY, h, eta) SNOW:(thetaC, thetaS, xi)
import { MAT } from './sim.js';

function sampleBox(out, mid, states, min, max, h, matIdx, jitter = 0.2, state0 = 0) {
  const n = [0, 1, 2].map(i => Math.max(1, Math.round((max[i] - min[i]) / h)));
  for (let k = 0; k < n[2]; k++)
    for (let j = 0; j < n[1]; j++)
      for (let i = 0; i < n[0]; i++) {
        out.push(
          min[0] + (i + 0.5) * h + (Math.random() - 0.5) * jitter * h,
          min[1] + (j + 0.5) * h + (Math.random() - 0.5) * jitter * h,
          min[2] + (k + 0.5) * h + (Math.random() - 0.5) * jitter * h,
        );
        mid.push(matIdx);
        states.push(state0);
      }
}

function sampleSphere(out, mid, states, c, r, h, matIdx, jitter = 0.2, state0 = 0) {
  const n = Math.ceil(2 * r / h);
  for (let k = 0; k <= n; k++)
    for (let j = 0; j <= n; j++)
      for (let i = 0; i <= n; i++) {
        const p = [c[0] - r + i * h, c[1] - r + j * h, c[2] - r + k * h];
        const d = Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]);
        if (d <= r) {
          out.push(
            p[0] + (Math.random() - 0.5) * jitter * h,
            p[1] + (Math.random() - 0.5) * jitter * h,
            p[2] + (Math.random() - 0.5) * jitter * h);
          mid.push(matIdx);
          states.push(state0);
        }
      }
}

const SAND = (phiDeg = 35, cohesion = 0) => ({
  kind: MAT.DP, E: 2000, nu: 0.3, density: 1.0,
  p0: Math.sqrt(2 / 3) * 2 * Math.sin(phiDeg * Math.PI / 180) / (3 - Math.sin(phiDeg * Math.PI / 180)),
  p1: cohesion,
  color: [0.85, 0.63, 0.28], boundFriction: 0.5,
});
const SNOW_NACC = () => ({
  // XPBI Table 2 "Snow Dive": NACC (4, 1e4, 0.3, -0.0005, 0.05, 30, 1.85).
  // E and xi softened for real-time timesteps (paper runs dt <= 4e-5 s offline).
  kind: MAT.NACC, E: 2000, nu: 0.3, density: 4.0,
  p0: -0.008 /* alpha0 -> state */, p1: 0.05, p2: 5, p3: 1.85,
  color: [0.93, 0.95, 0.98], boundFriction: 0.35,
});
const PLASTICINE = (sigY = 30) => ({
  // XPBI Table 2 "Noodles": VM (1, 2e4, 0.3, 76.9), softened for real-time
  // dough-like kneading (lower E, near-incompressible)
  kind: MAT.VM, E: 1200, nu: 0.42, density: 2.0,
  p0: sigY,
  color: [0.36, 0.76, 0.72], boundFriction: 0.4,
});
const GOO = (hExp = 1.0) => ({
  // XPBI Table 2 "Wrist": HB (100, 2250, 0.125, 10, 1, 10); yield stress and
  // viscosity raised for a blob-drop scene (theirs was a small wrist-scale toy)
  kind: MAT.HB, E: 2250, nu: 0.125, density: 100.0,
  p0: 45, p1: hExp, p2: 25,
  color: [0.91, 0.45, 0.62], boundFriction: 0.2,
});
const SNOW_CLASSIC = () => ({
  // Stomakhin 2013 Table 2 (E softened for real-time Jacobi)
  kind: MAT.SNOW, E: 9000, nu: 0.2, density: 4.0,
  p0: 3.5e-2, p1: 8e-3, p2: 10,
  color: [0.96, 0.97, 1.0], boundFriction: 0.35,
});
const WATER = () => ({
  kind: MAT.WATER, E: 100, nu: 0.3, density: 1.0,
  color: [0.25, 0.55, 0.95], boundFriction: 0.0,
});
const ELASTIC = () => ({
  kind: MAT.ELASTIC, E: 900, nu: 0.3, density: 2.0,
  color: [0.62, 0.52, 0.86], boundFriction: 0.3,
});

function base(h) {
  return {
    h,
    domainMin: [0, 0, 0], domainMax: [1, 1, 1],
    gravity: [0, -9.8, 0],
    substeps: 8, iterations: 3, timeScale: 0.42,
    omega: 0.5, contactOmega: 0.4, xsphC: 0.01,
    pbfEps: 100.0, sCorrK: 0.0001,
    camera: { theta: 0.5, phi: 0.32, dist: 2.1, target: [0.5, 0.28, 0.5] },
  };
}

export const SCENES = {
  'Sand: column collapse': () => {
    const s = base(0.016);
    const positions = [], matIds = [], states = [];
    sampleBox(positions, matIds, states, [0.36, 0.015, 0.36], [0.64, 0.62, 0.64], s.h, 0);
    return { ...s, positions, matIds, states, materials: [SAND(35)], hasWater: false,
      info: 'Drucker-Prager sand [Klár et al. 2016]. Watch it pile at the friction angle.' };
  },
  'Sand: two blocks collide': () => {
    const s = base(0.017);
    const positions = [], matIds = [], states = [];
    sampleBox(positions, matIds, states, [0.08, 0.015, 0.35], [0.38, 0.42, 0.65], s.h, 0);
    const n1 = matIds.length;
    sampleBox(positions, matIds, states, [0.62, 0.015, 0.35], [0.92, 0.42, 0.65], s.h, 0);
    const vel = new Float32Array(matIds.length * 3);
    for (let i = 0; i < matIds.length; i++) vel[i * 3] = i < n1 ? 1.6 : -1.6;
    return { ...s, positions, matIds, states, velocities: vel, materials: [SAND(35)], hasWater: false,
      info: 'Two sand blocks collide (paper Fig. 13) — continuum friction, no MPM grid gaps.' };
  },
  'Snow: snowball smash': () => {
    const s = base(0.018);
    s.substeps = 9; s.iterations = 3; s.timeScale = 0.45;
    const positions = [], matIds = [], states = [];
    const m = SNOW_NACC();
    // snow ground pack
    sampleBox(positions, matIds, states, [0.1, 0.015, 0.1], [0.9, 0.14, 0.9], s.h, 0, 0.2, m.p0);
    const n1 = matIds.length;
    sampleSphere(positions, matIds, states, [0.2, 0.55, 0.5], 0.11, s.h, 0, 0.2, m.p0);
    const vel = new Float32Array(matIds.length * 3);
    for (let i = n1; i < matIds.length; i++) { vel[i * 3] = 2.6; vel[i * 3 + 1] = -3.2; }
    return { ...s, positions, matIds, states, velocities: vel, materials: [m], hasWater: false,
      info: 'NACC snow [Wolper et al. 2019] — snowball impacting a snow pack (cf. paper Fig. 12).' };
  },
  'Plasticine: drop & poke': () => {
    const s = base(0.013);
    s.substeps = 8; s.iterations = 6; s.timeScale = 0.4;
    const positions = [], matIds = [], states = [];
    sampleSphere(positions, matIds, states, [0.5, 0.4, 0.5], 0.16, s.h, 0);
    return { ...s, positions, matIds, states, materials: [PLASTICINE(80)], hasWater: false,
      info: 'Von Mises plasticine [Li et al. 2022]. Drag with G held to knead it.' };
  },
  'Goo: Herschel-Bulkley': () => {
    const s = base(0.013);
    s.substeps = 9; s.iterations = 3; s.timeScale = 0.5;
    const positions = [], matIds = [], states = [];
    sampleSphere(positions, matIds, states, [0.5, 0.42, 0.5], 0.17, s.h, 0);
    return { ...s, positions, matIds, states, materials: [GOO(1.0)], hasWater: false,
      info: 'Herschel-Bulkley viscoplastic [Yue et al. 2015]. h<1 shear-thins, h>1 thickens (paper Fig. 4).' };
  },
  'Snow: classic (Stomakhin)': () => {
    const s = base(0.013);
    s.substeps = 10; s.timeScale = 0.38; s.iterations = 4;
    const positions = [], matIds = [], states = [];
    sampleSphere(positions, matIds, states, [0.35, 0.45, 0.42], 0.1, s.h, 0, 0.2, 1.0);
    sampleSphere(positions, matIds, states, [0.65, 0.5, 0.58], 0.1, s.h, 0, 0.2, 1.0);
    const vel = new Float32Array(matIds.length * 3);
    for (let i = 0; i < matIds.length; i++) {
      const left = positions[i * 3] < 0.5;
      vel[i * 3] = left ? 1.5 : -1.5;
    }
    return { ...s, positions, matIds, states, velocities: vel, materials: [SNOW_CLASSIC()], hasWater: false,
      info: 'Two snowballs, Stomakhin et al. 2013 clamp plasticity + hardening.' };
  },
  'Dam breach: water + sand': () => {
    const s = base(0.019);
    s.substeps = 6; s.iterations = 3; s.timeScale = 0.45;
    const positions = [], matIds = [], states = [];
    // sand levee in the middle
    sampleBox(positions, matIds, states, [0.42, 0.015, 0.05], [0.6, 0.34, 0.95], s.h, 0);
    // water reservoir on the left
    sampleBox(positions, matIds, states, [0.03, 0.015, 0.05], [0.36, 0.5, 0.95], s.h, 1);
    return { ...s, positions, matIds, states, materials: [SAND(30, 0.004), WATER()], hasWater: true,
      info: 'PBF water [Macklin & Müller 2013] coupled with Drucker-Prager sand (paper Fig. 7).' };
  },
  'Elastic: StVK block': () => {
    const s = base(0.014);
    s.substeps = 10; s.iterations = 4; s.timeScale = 0.5;
    const positions = [], matIds = [], states = [];
    sampleBox(positions, matIds, states, [0.38, 0.3, 0.38], [0.62, 0.54, 0.62], s.h, 0);
    return { ...s, positions, matIds, states, materials: [ELASTIC()], hasWater: false,
      info: 'Pure StVK-Hencky elasticity as XPBD constraints (paper §3.1) — no plasticity.' };
  },
};
