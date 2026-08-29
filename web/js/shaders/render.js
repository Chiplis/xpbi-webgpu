// Rendering WGSL: particle sphere impostors, ground plane, domain box lines.

export const WGSL_RENDER = /* wgsl */`
struct Camera {
  view : mat4x4f,
  proj : mat4x4f,
  viewProj : mat4x4f,
  eye : vec4f,
  lightDirWorld : vec4f,   // xyz normalized
  particleRadius : f32,
  domainMinY : f32,
  pad0 : f32,
  pad1 : f32,
};

struct MaterialR {
  mu : f32, lam : f32, density : f32, kind : u32,
  p0 : f32, p1 : f32, p2 : f32, p3 : f32,
  color : vec4f,
  boundFriction : f32, mass : f32, padA : f32, padB : f32,
};

@group(0) @binding(0) var<uniform> cam : Camera;
@group(0) @binding(1) var<storage, read> pos : array<vec4f>;
@group(0) @binding(2) var<uniform> MATS : array<MaterialR, 8>;
@group(0) @binding(3) var<storage, read> vel : array<vec4f>;

struct VSOut {
  @builtin(position) clip : vec4f,
  @location(0) uv : vec2f,
  @location(1) viewCenter : vec3f,
  @location(2) color : vec3f,
  @location(3) speed : f32,
  @location(4) kind : f32,
};

fn hash11(n : f32) -> f32 {
  return fract(sin(n * 127.1) * 43758.5453);
}

@vertex
fn vsParticle(@builtin(vertex_index) vi : u32, @builtin(instance_index) ii : u32) -> VSOut {
  var corners = array<vec2f, 4>(vec2f(-1.0,-1.0), vec2f(1.0,-1.0), vec2f(-1.0,1.0), vec2f(1.0,1.0));
  let c = corners[vi];
  let wp = pos[ii].xyz;
  let vc = (cam.view * vec4f(wp, 1.0)).xyz;
  let r = cam.particleRadius;
  let vpos = vc + vec3f(c * r, 0.0);
  var o : VSOut;
  o.clip = cam.proj * vec4f(vpos, 1.0);
  o.uv = c;
  o.viewCenter = vc;
  let m = MATS[u32(pos[ii].w)];
  var col = m.color.rgb;
  // subtle per-particle albedo variation, seeded from pos.w's fraction so it
  // survives the sim's per-substep particle reordering
  let hv = fract(pos[ii].w) * 2.04;
  col = col * (0.88 + 0.24 * hv);
  o.color = col;
  o.speed = length(vel[ii].xyz);
  o.kind = f32(m.kind);
  return o;
}

struct FSOut {
  @location(0) color : vec4f,
  @builtin(frag_depth) depth : f32,
};

@fragment
fn fsParticle(in : VSOut) -> FSOut {
  let r2 = dot(in.uv, in.uv);
  if (r2 > 1.0) { discard; }
  let nz = sqrt(1.0 - r2);
  let nView = vec3f(in.uv, nz);
  // depth-correct sphere
  let vpos = in.viewCenter + nView * cam.particleRadius;
  let cp = cam.proj * vec4f(vpos, 1.0);
  var o : FSOut;
  o.depth = cp.z / cp.w;
  // lighting in view space
  let L = normalize((cam.view * vec4f(cam.lightDirWorld.xyz, 0.0)).xyz);
  let diff = max(dot(nView, L), 0.0);
  let up = normalize((cam.view * vec4f(0.0, 1.0, 0.0, 0.0)).xyz);
  let hemi = 0.5 + 0.5 * dot(nView, up);
  var col = in.color * (0.35 + 0.55 * diff + 0.35 * hemi * 0.5);
  // water: fresnel-ish rim + speed whitening
  if (in.kind > 5.5) {
    let fres = pow(1.0 - nz, 2.0);
    col = mix(col, vec3f(0.75, 0.9, 1.0), fres * 0.6);
    col = mix(col, vec3f(0.85, 0.95, 1.0), clamp(in.speed * 0.25, 0.0, 0.45));
  }
  let V = normalize(-vpos);
  let H = normalize(L + V);
  let spec = pow(max(dot(nView, H), 0.0), 48.0) * select(0.15, 0.6, in.kind > 5.5);
  col = col + vec3f(spec);
  o.color = vec4f(col, 1.0);
  return o;
}

// ---------------- ground + backdrop ----------------
struct GroundVSOut {
  @builtin(position) clip : vec4f,
  @location(0) world : vec3f,
};

@vertex
fn vsGround(@builtin(vertex_index) vi : u32) -> GroundVSOut {
  var corners = array<vec2f, 4>(vec2f(-1.0,-1.0), vec2f(1.0,-1.0), vec2f(-1.0,1.0), vec2f(1.0,1.0));
  let c = corners[vi] * 12.0 + vec2f(0.5, 0.5);
  var o : GroundVSOut;
  let wp = vec3f(c.x, cam.domainMinY, c.y);
  o.world = wp;
  o.clip = cam.viewProj * vec4f(wp, 1.0);
  return o;
}

@fragment
fn fsGround(in : GroundVSOut) -> @location(0) vec4f {
  let p = in.world.xz;
  // grid lines every 0.25
  let g = abs(fract(p * 4.0 - 0.5) - 0.5) / max(fwidth(p * 4.0), vec2f(1e-4));
  let line = 1.0 - min(min(g.x, g.y), 1.0);
  var base = vec3f(0.16, 0.17, 0.20);
  let lineCol = vec3f(0.24, 0.26, 0.30);
  var col = mix(base, lineCol, line * 0.8);
  // radial fade to background
  let dc = length(p - vec2f(0.5, 0.5));
  let fade = smoothstep(1.2, 5.0, dc);
  let bg = vec3f(0.10, 0.11, 0.13);
  col = mix(col, bg, fade);
  return vec4f(col, 1.0);
}

// ---------------- domain wireframe ----------------
struct LineVSOut { @builtin(position) clip : vec4f };
@vertex
fn vsLines(@location(0) p : vec3f) -> LineVSOut {
  var o : LineVSOut;
  o.clip = cam.viewProj * vec4f(p, 1.0);
  return o;
}
@fragment
fn fsLines() -> @location(0) vec4f {
  return vec4f(0.45, 0.48, 0.55, 1.0);
}
`;
