// XPBI demo app: WebGPU setup, renderer, camera, UI, main loop.
import { Simulation } from './sim.js';
import { SCENES } from './scenes.js';
import { WGSL_RENDER } from './shaders/render.js';

const canvas = document.getElementById('gpu');
const ui = {
  scene: document.getElementById('scene'),
  info: document.getElementById('info'),
  stats: document.getElementById('stats'),
  pause: document.getElementById('pause'),
  reset: document.getElementById('reset'),
  sliders: document.getElementById('sliders'),
  err: document.getElementById('err'),
};

let device, context, format;
let sim = null, sceneCfg = null, sceneName = null;
let paused = false;
let renderer = null;

// ---------------- camera ----------------
const cam = { theta: 0.5, phi: 0.32, dist: 2.1, target: [0.5, 0.28, 0.5] };
function camEye() {
  const ct = Math.cos(cam.theta), st = Math.sin(cam.theta);
  const cp = Math.cos(cam.phi), sp = Math.sin(cam.phi);
  return [
    cam.target[0] + cam.dist * cp * st,
    cam.target[1] + cam.dist * sp,
    cam.target[2] + cam.dist * cp * ct,
  ];
}
function mat4LookAt(eye, tgt, up) {
  const z = norm3(sub3(eye, tgt));
  const x = norm3(cross3(up, z));
  const y = cross3(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot3(x, eye), -dot3(y, eye), -dot3(z, eye), 1,
  ]);
}
function mat4Persp(fovY, aspect, near, far) {
  const f = 1 / Math.tan(fovY / 2);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, far / (near - far), -1,
    0, 0, (near * far) / (near - far), 0,
  ]);
}
function mat4Mul(a, b) { // a*b, column-major
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      o[c * 4 + r] = s;
    }
  return o;
}
const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm3 = a => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

// ---------------- interaction state ----------------
const input = {
  orbiting: false, lastX: 0, lastY: 0,
  pokeMode: false, pokeActive: false,
  colliderPos: [0.5, 0.4, 0.5], colliderPrev: [0.5, 0.4, 0.5], colliderVel: [0, 0, 0],
  colliderR: 0.09,
};

canvas.addEventListener('pointerdown', e => {
  canvas.setPointerCapture(e.pointerId);
  if (input.pokeMode || e.shiftKey) {
    input.pokeActive = true;
    updateCollider(e);
    input.colliderPrev = [...input.colliderPos];
  } else {
    input.orbiting = true;
  }
  input.lastX = e.clientX; input.lastY = e.clientY;
});
canvas.addEventListener('pointermove', e => {
  if (input.orbiting) {
    cam.theta -= (e.clientX - input.lastX) * 0.006;
    cam.phi = Math.min(1.45, Math.max(-0.2, cam.phi + (e.clientY - input.lastY) * 0.006));
  } else if (input.pokeActive) {
    updateCollider(e);
  }
  input.lastX = e.clientX; input.lastY = e.clientY;
});
canvas.addEventListener('pointerup', () => { input.orbiting = false; input.pokeActive = false; });
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  cam.dist = Math.min(6, Math.max(0.6, cam.dist * Math.exp(e.deltaY * 0.001)));
}, { passive: false });
window.addEventListener('keydown', e => {
  if (e.key === 'g' || e.key === 'G') { togglePoke(); }
  if (e.key === ' ') { e.preventDefault(); paused = !paused; ui.pause.textContent = paused ? 'Resume' : 'Pause'; }
});
document.getElementById('poke').addEventListener('click', togglePoke);
function togglePoke() {
  input.pokeMode = !input.pokeMode;
  document.getElementById('poke').classList.toggle('active', input.pokeMode);
}

// unproject mouse to a camera-facing plane through the domain center
function updateCollider(e) {
  const rect = canvas.getBoundingClientRect();
  const nx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  const ny = 1 - ((e.clientY - rect.top) / rect.height) * 2;
  const eye = camEye();
  const fwd = norm3(sub3(cam.target, eye));
  const right = norm3(cross3(fwd, [0, 1, 0]));
  const up = cross3(right, fwd);
  const fovY = Math.PI / 4;
  const aspect = canvas.width / canvas.height;
  const th = Math.tan(fovY / 2);
  const dir = norm3([
    fwd[0] + nx * th * aspect * right[0] + ny * th * up[0],
    fwd[1] + nx * th * aspect * right[1] + ny * th * up[1],
    fwd[2] + nx * th * aspect * right[2] + ny * th * up[2],
  ]);
  // plane through target, normal = fwd
  const denom = dot3(dir, fwd);
  const t = denom > 1e-6 ? dot3(sub3(cam.target, eye), fwd) / denom : cam.dist;
  input.colliderPos = [eye[0] + dir[0] * t, eye[1] + dir[1] * t, eye[2] + dir[2] * t];
}

// ---------------- renderer ----------------
class Renderer {
  constructor(device, format) {
    this.device = device;
    const module = device.createShaderModule({ code: WGSL_RENDER });
    this.camBuf = device.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    // no MSAA: sphere impostors write frag_depth (late-Z), so multisampling
    // multiplies the most expensive path for little visual gain
    this.sampleCount = 1;

    const bglEntries = [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ];
    this.bglParticles = device.createBindGroupLayout({ entries: bglEntries });
    this.bglSimple = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
    });

    const depth = { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' };
    this.pipeParticles = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.bglParticles] }),
      vertex: { module, entryPoint: 'vsParticle' },
      fragment: { module, entryPoint: 'fsParticle', targets: [{ format }] },
      primitive: { topology: 'triangle-strip' },
      depthStencil: depth,
      multisample: { count: this.sampleCount },
    });
    this.pipeGround = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.bglSimple] }),
      vertex: { module, entryPoint: 'vsGround' },
      fragment: { module, entryPoint: 'fsGround', targets: [{ format }] },
      primitive: { topology: 'triangle-strip' },
      depthStencil: depth,
      multisample: { count: this.sampleCount },
    });
    this.pipeLines = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.bglSimple] }),
      vertex: {
        module, entryPoint: 'vsLines',
        buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }],
      },
      fragment: { module, entryPoint: 'fsLines', targets: [{ format }] },
      primitive: { topology: 'line-list' },
      depthStencil: depth,
      multisample: { count: this.sampleCount },
    });
    this.groundBind = device.createBindGroup({
      layout: this.bglSimple, entries: [{ binding: 0, resource: { buffer: this.camBuf } }],
    });
    this.lineBuf = null;
  }

  setScene(sim, cfg) {
    const d = this.device;
    this.particleBind = d.createBindGroup({
      layout: this.bglParticles,
      entries: [
        { binding: 0, resource: { buffer: this.camBuf } },
        { binding: 1, resource: { buffer: sim.posBuf } },
        { binding: 2, resource: { buffer: sim.matsBuf } },
        { binding: 3, resource: { buffer: sim.velA } },
      ],
    });
    // domain wireframe (12 edges)
    const [a, b] = [cfg.domainMin, cfg.domainMax];
    const C = [
      [a[0], a[1], a[2]], [b[0], a[1], a[2]], [b[0], a[1], b[2]], [a[0], a[1], b[2]],
      [a[0], b[1], a[2]], [b[0], b[1], a[2]], [b[0], b[1], b[2]], [a[0], b[1], b[2]],
    ];
    const E = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
    const verts = new Float32Array(E.length * 6);
    E.forEach((e, i) => { verts.set(C[e[0]], i * 6); verts.set(C[e[1]], i * 6 + 3); });
    this.lineBuf = d.createBuffer({ size: verts.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.lineBuf, 0, verts);
    this.lineCount = E.length * 2;
  }

  ensureTargets(w, h) {
    if (this.tw === w && this.th === h) return;
    this.tw = w; this.th = h;
    this.msaaTex?.destroy(); this.depthTex?.destroy();
    if (this.sampleCount > 1) {
      this.msaaTex = this.device.createTexture({
        size: [w, h], sampleCount: this.sampleCount, format,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    } else {
      this.msaaTex = null;
    }
    this.depthTex = this.device.createTexture({
      size: [w, h], sampleCount: this.sampleCount, format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  writeCamera(sim, cfg) {
    const eye = camEye();
    const view = mat4LookAt(eye, cam.target, [0, 1, 0]);
    const proj = mat4Persp(Math.PI / 4, canvas.width / canvas.height, 0.02, 50);
    const vp = mat4Mul(proj, view);
    const buf = new Float32Array(64);
    buf.set(view, 0); buf.set(proj, 16); buf.set(vp, 32);
    buf.set([...eye, 1], 48);
    buf.set(norm3([0.5, 0.9, 0.35]), 52); buf[55] = 0;
    buf[56] = sim.h * 0.62;              // particle render radius
    buf[57] = cfg.domainMin[1];
    this.device.queue.writeBuffer(this.camBuf, 0, buf);
  }

  render(enc, sim) {
    const view = context.getCurrentTexture().createView();
    this.ensureTargets(canvas.width, canvas.height);
    const colorAtt = this.msaaTex
      ? { view: this.msaaTex.createView(), resolveTarget: view, storeOp: 'discard' }
      : { view, storeOp: 'store' };
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        ...colorAtt,
        clearValue: { r: 0.10, g: 0.11, b: 0.13, a: 1 }, loadOp: 'clear',
      }],
      depthStencilAttachment: {
        view: this.depthTex.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard',
      },
    });
    pass.setPipeline(this.pipeGround);
    pass.setBindGroup(0, this.groundBind);
    pass.draw(4);
    pass.setPipeline(this.pipeLines);
    pass.setBindGroup(0, this.groundBind);
    pass.setVertexBuffer(0, this.lineBuf);
    pass.draw(this.lineCount);
    pass.setPipeline(this.pipeParticles);
    pass.setBindGroup(0, this.particleBind);
    pass.draw(4, sim.N);
    pass.end();
  }
}

// ---------------- scene management + UI ----------------
const OVERRIDES = {}; // per-scene UI overrides

function loadScene(name) {
  sceneName = name;
  const cfg = SCENES[name]();
  sceneCfg = cfg;
  Object.assign(cam, cfg.camera);
  sim = new Simulation(device, cfg);
  renderer.setScene(sim, cfg);
  ui.info.textContent = cfg.info;
  buildSliders(cfg);
  simTime = 0;
  // debug access
  window.__XPBI = {
    get sim() { return sim; }, get cfg() { return sceneCfg; }, device,
    async readPos(n = 4) {
      const N = sim.N;
      const stage = device.createBuffer({ size: N * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const e = device.createCommandEncoder();
      e.copyBufferToBuffer(sim.posBuf, 0, stage, 0, N * 16);
      device.queue.submit([e.finish()]);
      await stage.mapAsync(GPUMapMode.READ);
      const a = new Float32Array(stage.getMappedRange().slice(0));
      stage.unmap(); stage.destroy();
      return a;
    },
    async readVel() {
      const N = sim.N;
      const stage = device.createBuffer({ size: N * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const e = device.createCommandEncoder();
      e.copyBufferToBuffer(sim.velA, 0, stage, 0, N * 16);
      device.queue.submit([e.finish()]);
      await stage.mapAsync(GPUMapMode.READ);
      const a = new Float32Array(stage.getMappedRange().slice(0));
      stage.unmap(); stage.destroy();
      return a;
    },
  };
}

function slider(label, min, max, step, value, onchange, fmt = v => v) {
  const row = document.createElement('div');
  row.className = 'srow';
  const lab = document.createElement('label');
  const val = document.createElement('span');
  val.textContent = fmt(value);
  lab.textContent = label;
  const inp = document.createElement('input');
  inp.type = 'range'; inp.min = min; inp.max = max; inp.step = step; inp.value = value;
  inp.addEventListener('input', () => { val.textContent = fmt(parseFloat(inp.value)); onchange(parseFloat(inp.value)); });
  row.append(lab, inp, val);
  ui.sliders.append(row);
  return inp;
}

function buildSliders(cfg) {
  ui.sliders.innerHTML = '';
  slider('substeps / frame', 4, 60, 1, cfg.substeps, v => { cfg.substeps = v | 0; });
  slider('XPBD iterations', 1, 10, 1, cfg.iterations, v => { cfg.iterations = v | 0; });
  slider('time scale', 0.1, 1.5, 0.05, cfg.timeScale, v => { cfg.timeScale = v; });
  const m0 = cfg.materials[0];
  slider('Young’s modulus E', 200, 20000, 100, m0.E, v => { m0.E = v; sim.writeMaterials(cfg.materials); });
  if (m0.kind === 1) { // DP
    slider('friction angle φ (°)', 15, 45, 1, 35, v => {
      m0.p0 = Math.sqrt(2 / 3) * 2 * Math.sin(v * Math.PI / 180) / (3 - Math.sin(v * Math.PI / 180));
      sim.writeMaterials(cfg.materials);
    });
  }
  if (m0.kind === 3) { // VM
    slider('yield stress σY', 5, 400, 5, m0.p0, v => { m0.p0 = v; sim.writeMaterials(cfg.materials); });
  }
  if (m0.kind === 4) { // HB
    slider('HB exponent h', 0.3, 3.0, 0.1, m0.p1, v => { m0.p1 = v; sim.writeMaterials(cfg.materials); });
    slider('yield stress σY', 1, 100, 1, m0.p0, v => { m0.p0 = v; sim.writeMaterials(cfg.materials); });
    slider('viscosity η', 0.5, 100, 0.5, m0.p2, v => { m0.p2 = v; sim.writeMaterials(cfg.materials); });
  }
  if (m0.kind === 2) { // NACC
    slider('hardening ξ', 1, 60, 1, m0.p2, v => { m0.p2 = v; sim.writeMaterials(cfg.materials); });
    slider('friction M', 0.5, 3.5, 0.05, m0.p3, v => { m0.p3 = v; sim.writeMaterials(cfg.materials); });
  }
  slider('poke radius', 0.03, 0.2, 0.005, input.colliderR, v => { input.colliderR = v; });
}

// ---------------- main loop ----------------
let simTime = 0;
let lastT = performance.now();
let fpsFrames = 0, fpsT0 = performance.now(), fpsShown = 0;
let inFlight = 0;   // GPU backpressure: never queue more than 2 frames

function frame() {
  if (inFlight >= (window.__CAP ?? 3)) { requestAnimationFrame(frame); return; }
  const __f0 = performance.now();
  const now = performance.now();
  let dtWall = (now - lastT) / 1000;
  lastT = now;
  dtWall = Math.min(dtWall, 1 / 30);

  fpsFrames++;
  if (now - fpsT0 > 500) {
    fpsShown = fpsFrames * 1000 / (now - fpsT0);
    fpsFrames = 0; fpsT0 = now;
  }

  const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
  const w = Math.floor(canvas.clientWidth * dpr), h = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }

  const cfg = sceneCfg;

  if (!paused && sim) {
    const substeps = cfg.substeps;
    const dt = (1 / 60) * cfg.timeScale / substeps;
    // collider velocity estimate (world units / s)
    const cv = [0, 1, 2].map(i => (input.colliderPos[i] - input.colliderPrev[i]) / Math.max(dtWall, 1e-3));
    input.colliderPrev = [...input.colliderPos];
    const forced = window.__FORCE_COLLIDER;
    sim.writeParams({
      dt,
      domainMin: cfg.domainMin, domainMax: cfg.domainMax,
      gravity: cfg.gravity,
      colliderPos: forced || input.colliderPos, colliderR: input.colliderR,
      colliderVel: forced ? [0, 0, 0] : cv.map(v => Math.max(-1.5, Math.min(1.5, v || 0))),
      colliderActive: forced ? true : input.pokeActive,
      xsphC: cfg.xsphC, omega: cfg.omega, contactOmega: cfg.contactOmega,
      pbfEps: cfg.pbfEps, sCorrK: cfg.sCorrK, time: simTime,
      kDamp: cfg.kDamp ?? 0.4,
    });
    // chunked submits: small enough command buffers that Metal can preempt
    // for the compositor (a monolithic frame buffer starves it), large
    // enough to amortize submit overhead
    const CHUNK = 6;
    for (let s = 0; s < substeps; s += CHUNK) {
      const enc = device.createCommandEncoder();
      for (let k = s; k < Math.min(s + CHUNK, substeps); k++) {
        sim.encodeSubstep(enc, cfg.iterations, cfg.hasWater);
      }
      device.queue.submit([enc.finish()]);
    }
    simTime += (1 / 60) * cfg.timeScale;
  }

  renderer.writeCamera(sim, cfg);
  const enc = device.createCommandEncoder();
  renderer.render(enc, sim);
  device.queue.submit([enc.finish()]);
  inFlight++;
  const __f1 = performance.now();
  device.queue.onSubmittedWorkDone().then(() => {
    inFlight--;
    window.__TIMING = { encodeMs: __f1 - __f0, gpuMs: performance.now() - __f1 };
  });

  ui.stats.textContent =
    `${sim.N.toLocaleString()} particles · ${fpsShown.toFixed(0)} fps · ` +
    `${(cfg.substeps * 60 * cfg.timeScale).toFixed(0)} substeps/s`;

  requestAnimationFrame(frame);
}

// ---------------- boot ----------------
async function boot() {
  if (!navigator.gpu) { ui.err.textContent = 'WebGPU not available in this browser.'; return; }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) { ui.err.textContent = 'No WebGPU adapter.'; return; }
  device = await adapter.requestDevice({
    requiredFeatures: adapter.features.has('timestamp-query') ? ['timestamp-query'] : [],
    requiredLimits: {
      maxStorageBuffersPerShaderStage: Math.min(16, adapter.limits.maxStorageBuffersPerShaderStage),
      maxStorageBufferBindingSize: Math.min(256 * 1024 * 1024, adapter.limits.maxStorageBufferBindingSize),
    },
  });
  device.addEventListener('uncapturederror', e => {
    console.error('WebGPU error:', e.error.message);
    ui.err.textContent = 'GPU error: ' + e.error.message.slice(0, 300);
  });
  context = canvas.getContext('webgpu');
  format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  renderer = new Renderer(device, format);

  for (const name of Object.keys(SCENES)) {
    const opt = document.createElement('option');
    opt.value = name; opt.textContent = name;
    ui.scene.append(opt);
  }
  ui.scene.addEventListener('change', () => loadScene(ui.scene.value));
  ui.pause.addEventListener('click', () => { paused = !paused; ui.pause.textContent = paused ? 'Resume' : 'Pause'; });
  ui.reset.addEventListener('click', () => loadScene(sceneName));

  loadScene(Object.keys(SCENES)[0]);
  requestAnimationFrame(frame);
}
boot().catch(e => { console.error(e); ui.err.textContent = String(e).slice(0, 400); });
