// XPBI GPU simulation: buffer management, pipelines, per-substep encoding.
import { buildSimWGSL } from './shaders/simkernels.js';

const MAT_STRIDE = 16; // floats per material
export const MAT = {
  ELASTIC: 0, DP: 1, NACC: 2, VM: 3, HB: 4, SNOW: 5, WATER: 6,
};

export class Simulation {
  constructor(device, scene) {
    this.device = device;
    this.scene = scene;
    const N = scene.positions.length / 3;
    this.N = N;
    this.h = scene.h;
    this.support = 2 * scene.h;
    this.V0 = scene.h ** 3;
    this.frame = 0;

    // grid sized to domain with 1-cell pad
    const cs = this.support;
    const pad = 2;
    this.gridOrigin = scene.domainMin.map(v => v - pad * cs);
    this.gridDims = [0, 1, 2].map(i =>
      Math.ceil((scene.domainMax[i] - scene.domainMin[i]) / cs) + 2 * pad);
    this.numCells = this.gridDims[0] * this.gridDims[1] * this.gridDims[2];

    this.createBuffers();
    this.createPipelines();
  }

  createBuffers() {
    const d = this.device;
    const N = this.N;
    const f4 = 16, m3 = 48;
    const mk = (size, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) =>
      d.createBuffer({ size: Math.max(size, 16), usage });

    this.posBuf = mk(N * f4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.VERTEX);
    this.velA = mk(N * f4);
    this.velB = mk(N * f4);
    this.Fbuf = mk(N * m3);
    this.Lbuf = mk(N * m3);
    this.auxBuf = mk(N * f4);      // x=Vn y=lambda z=state w=pbfLambda
    this.SpBuf = mk(Math.max(N * m3, Math.ceil(this.numCells / 256) * 4 + 16)); // 3 vec4/particle; also scan scratch
    this.cellTab = mk(this.numCells * 4);
    this.sortedIdx = mk(N * 4);

    this.paramsBuf = d.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.matsBuf = d.createBuffer({ size: 8 * MAT_STRIDE * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // upload initial state
    const s = this.scene;
    const pos = new Float32Array(N * 4);
    const F = new Float32Array(N * 12);
    const auxInit = new Float32Array(N * 4);
    for (let i = 0; i < N; i++) {
      pos[i * 4 + 0] = s.positions[i * 3 + 0];
      pos[i * 4 + 1] = s.positions[i * 3 + 1];
      pos[i * 4 + 2] = s.positions[i * 3 + 2];
      // material index + per-particle color seed in the fraction (stable
      // across the per-substep reordering, unlike the array index)
      pos[i * 4 + 3] = s.matIds[i] + Math.random() * 0.49;
      // identity F: mat3x3f is 3 columns of vec4-aligned vec3
      F[i * 12 + 0] = 1; F[i * 12 + 5] = 1; F[i * 12 + 10] = 1;
      auxInit[i * 4 + 0] = this.V0;
      auxInit[i * 4 + 2] = s.states ? s.states[i] : 0;
    }
    d.queue.writeBuffer(this.posBuf, 0, pos);
    const vel0 = new Float32Array(N * 4);
    if (s.velocities) {
      for (let i = 0; i < N; i++) {
        vel0[i * 4 + 0] = s.velocities[i * 3 + 0];
        vel0[i * 4 + 1] = s.velocities[i * 3 + 1];
        vel0[i * 4 + 2] = s.velocities[i * 3 + 2];
      }
    }
    d.queue.writeBuffer(this.velA, 0, vel0);
    d.queue.writeBuffer(this.Fbuf, 0, F);
    d.queue.writeBuffer(this.auxBuf, 0, auxInit);
    this.writeMaterials(s.materials);
  }

  writeMaterials(materials) {
    const arr = new Float32Array(8 * MAT_STRIDE);
    const u32 = new Uint32Array(arr.buffer);
    materials.forEach((m, i) => {
      const o = i * MAT_STRIDE;
      const mu = m.E / (2 * (1 + m.nu));
      const lam = m.E * m.nu / ((1 + m.nu) * (1 - 2 * m.nu));
      arr[o + 0] = mu; arr[o + 1] = lam; arr[o + 2] = m.density;
      u32[o + 3] = m.kind;
      arr[o + 4] = m.p0 ?? 0; arr[o + 5] = m.p1 ?? 0; arr[o + 6] = m.p2 ?? 0; arr[o + 7] = m.p3 ?? 0;
      arr[o + 8] = m.color[0]; arr[o + 9] = m.color[1]; arr[o + 10] = m.color[2]; arr[o + 11] = 1;
      arr[o + 12] = m.boundFriction ?? 0.4;
      arr[o + 13] = m.density * this.V0;   // mass
    });
    this.device.queue.writeBuffer(this.matsBuf, 0, arr);
  }

  createPipelines() {
    const d = this.device;
    if (!Simulation._cache || Simulation._cache.device !== d) {
      Simulation._cache = { device: d, built: null };
    }

    const entries = (velIn, velOut) => [
      { binding: 0, resource: { buffer: this.paramsBuf } },
      { binding: 1, resource: { buffer: this.matsBuf } },
      { binding: 2, resource: { buffer: this.posBuf } },
      { binding: 3, resource: { buffer: velIn } },
      { binding: 4, resource: { buffer: velOut } },
      { binding: 5, resource: { buffer: this.Fbuf } },
      { binding: 6, resource: { buffer: this.Lbuf } },
      { binding: 7, resource: { buffer: this.auxBuf } },
      { binding: 8, resource: { buffer: this.SpBuf } },
      { binding: 9, resource: { buffer: this.cellTab } },
      { binding: 10, resource: { buffer: this.sortedIdx } },
    ];

    // explicit shared layout so every pass can use the same two bind groups
    if (!Simulation._cache.built) {
      const module = d.createShaderModule({ code: buildSimWGSL() });
      const lay = [];
      lay.push({ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } });
      lay.push({ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } });
      for (let b = 2; b <= 10; b++) {
        lay.push({ binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } });
      }
      const bgl = d.createBindGroupLayout({ entries: lay });
      const pl = d.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const names = ['countParticles', 'scanBlocks', 'scanApply', 'scatter',
        'stagePVA', 'applyPVA', 'stageF', 'applyF',
        'computeL',
        'solveA', 'solveB', 'pbfA', 'pbfB', 'xsph', 'updateF', 'integrate'];
      const pipes = {};
      for (const n of names) {
        d.pushErrorScope('validation');
        pipes[n] = d.createComputePipeline({
          layout: pl,
          compute: { module, entryPoint: n },
        });
        d.popErrorScope().then(e => {
          if (e) console.error(`pipeline ${n} failed:`, e.message);
        });
      }
      Simulation._cache.built = { bgl, pipes };
    }
    const { bgl, pipes } = Simulation._cache.built;
    this.pipes = pipes;
    // two bind groups: (velA in, velB out) and (velB in, velA out)
    this.bindAB0 = d.createBindGroup({ layout: bgl, entries: entries(this.velA, this.velB) });
    this.bindBA0 = d.createBindGroup({ layout: bgl, entries: entries(this.velB, this.velA) });
    this.bindAB = new Proxy({}, { get: () => this.bindAB0 });
    this.bindBA = new Proxy({}, { get: () => this.bindBA0 });
  }

  // params: JS object with scene + UI settings
  writeParams(p) {
    const buf = new ArrayBuffer(256);
    const f = new Float32Array(buf);
    const i32 = new Int32Array(buf);
    const u32 = new Uint32Array(buf);
    // struct Params layout (WGSL std140-ish):
    // gridOrigin vec3f + cellSize f32          -> 0..3
    // gridDims vec3i + numCells u32            -> 4..7
    // domainMin vec3f + dt                     -> 8..11
    // domainMax vec3f + h                      -> 12..15
    // gravity vec3f + support                  -> 16..19
    // colliderPos vec3f + colliderR            -> 20..23
    // colliderVel vec3f + colliderActive       -> 24..27
    // N u32, V0, xsphC, omega                  -> 28..31
    // pbfEps, sCorrK, time, contactOmega       -> 32..35
    f.set(this.gridOrigin, 0); f[3] = this.support;
    i32[4] = this.gridDims[0]; i32[5] = this.gridDims[1]; i32[6] = this.gridDims[2];
    u32[7] = this.numCells;
    f.set(p.domainMin, 8); f[11] = p.dt;
    f.set(p.domainMax, 12); f[15] = this.h;
    f.set(p.gravity, 16); f[19] = this.support;
    f.set(p.colliderPos, 20); f[23] = p.colliderR;
    f.set(p.colliderVel, 24); f[27] = p.colliderActive ? 1 : 0;
    u32[28] = this.N; f[29] = this.V0; f[30] = p.xsphC; f[31] = p.omega;
    f[32] = p.pbfEps; f[33] = p.sCorrK; f[34] = p.time; f[35] = p.contactOmega;
    f[36] = p.kDamp ?? 0.4;
    this.device.queue.writeBuffer(this.paramsBuf, 0, buf);
  }

  // optional per-pass GPU profiling via timestamp queries
  initProfiler() {
    const d = this.device;
    if (!d.features.has('timestamp-query')) return false;
    this.tsCapacity = 64;
    this.tsQuery = d.createQuerySet({ type: 'timestamp', count: this.tsCapacity });
    this.tsResolve = d.createBuffer({ size: this.tsCapacity * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    this.tsRead = d.createBuffer({ size: this.tsCapacity * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    return true;
  }

  async profileSubstep(iterations, hasWater) {
    if (!this.tsQuery && !this.initProfiler()) return 'timestamp-query unavailable';
    const enc = this.device.createCommandEncoder();
    this.profileLabels = [];
    this.encodeSubstep(enc, iterations, hasWater, { query: this.tsQuery, labels: this.profileLabels });
    enc.resolveQuerySet(this.tsQuery, 0, this.profileLabels.length + 1, this.tsResolve, 0);
    enc.copyBufferToBuffer(this.tsResolve, 0, this.tsRead, 0, this.tsCapacity * 8);
    this.device.queue.submit([enc.finish()]);
    await this.tsRead.mapAsync(GPUMapMode.READ);
    const t = new BigInt64Array(this.tsRead.getMappedRange().slice(0));
    this.tsRead.unmap();
    const out = [];
    for (let i = 0; i < this.profileLabels.length; i++) {
      out.push(`${this.profileLabels[i]}: ${(Number(t[i + 1] - t[i]) / 1e6).toFixed(3)}ms`);
    }
    return out.join('\n');
  }

  // encode one full substep. velIn must be velA at entry and exits in velA.
  // With profile = {query, labels}: one compute pass per dispatch, with
  // timestamp writes, so each pass can be timed individually.
  encodeSubstep(enc, iterations, hasWater, profile = null) {
    const NG = Math.ceil(this.N / 128);
    const NB = Math.ceil(this.numCells / 256);
    enc.clearBuffer(this.cellTab);
    let pass = null;
    if (!profile) pass = enc.beginComputePass();
    let flip = false; // false: velA is "in"
    const run = (name, groups, swap = false) => {
      let p = pass;
      if (profile) {
        const qi = profile.labels.length;
        p = enc.beginComputePass({
          timestampWrites: {
            querySet: profile.query,
            beginningOfPassWriteIndex: qi === 0 ? 0 : undefined,
            endOfPassWriteIndex: qi + 1,
          },
        });
        profile.labels.push(name);
      }
      p.setPipeline(this.pipes[name]);
      p.setBindGroup(0, flip ? this.bindBA[name] : this.bindAB[name]);
      p.dispatchWorkgroups(groups);
      if (profile) p.end();
      if (swap) flip = !flip;
    };
    run('countParticles', NG);
    run('scanBlocks', NB);
    run('scanApply', NB);
    run('scatter', NG);
    // permute particle data into cell order: coherent neighbor gathers
    run('stagePVA', NG);
    run('applyPVA', NG);
    run('stageF', NG);
    run('applyF', NG);
    run('computeL', NG);
    for (let k = 0; k < iterations; k++) {
      run('solveA', NG);
      run('solveB', NG, true);   // includes contacts + boundary
      if (hasWater) {
        run('pbfA', NG);
        run('pbfB', NG, true);
      }
    }
    run('xsph', NG, true);
    run('updateF', NG);
    run('integrate', NG);
    if (pass) pass.end();
    // ensure final velocity ends in velA for next substep
    if (flip) {
      enc.copyBufferToBuffer(this.velB, 0, this.velA, 0, this.N * 16);
    }
  }
}
