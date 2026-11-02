// Live ink-in-water background: a small stable-fluids simulation on the GPU (WebGL 2).
// It follows the pointer, drifts by itself when idle, pauses in hidden tabs, and stays off for reduced motion.
(() => {
  const canvas = document.getElementById("fluid");
  const calm = matchMedia("(prefers-reduced-motion: reduce)");
  if (!canvas || calm.matches) return;
  const gl = canvas.getContext("webgl2", { alpha: true, depth: false, stencil: false, antialias: false, premultipliedAlpha: true });
  if (!gl || !gl.getExtension("EXT_color_buffer_float")) return;

  const small = matchMedia("(max-width: 700px)").matches;
  const SIM = small ? 64 : 128;
  const DYE = small ? 384 : 768;
  const PRESSURE_STEPS = 18;
  const VELOCITY_FADE = 0.985;
  const DYE_FADE = 0.992;
  const CURL = 22;
  const RADIUS = 0.0035;
  const FORCE = 5200;
  const IDLE_SPLAT = 1700;
  const COLOURS = [[0.45, 0.32, 1.0], [0.13, 0.72, 0.95], [0.85, 0.28, 0.85], [0.3, 0.45, 1.0]];

  const VERTEX = `#version 300 es
  in vec2 point;
  uniform vec2 texel;
  out vec2 uv, uvL, uvR, uvT, uvB;
  void main() {
    uv = point * 0.5 + 0.5;
    uvL = uv - vec2(texel.x, 0.0); uvR = uv + vec2(texel.x, 0.0);
    uvT = uv + vec2(0.0, texel.y); uvB = uv - vec2(0.0, texel.y);
    gl_Position = vec4(point, 0.0, 1.0);
  }`;
  const HEAD = `#version 300 es
  precision highp float;
  in vec2 uv, uvL, uvR, uvT, uvB;
  out vec4 colour;
  `;
  const SHADERS = {
    splat: `uniform sampler2D target; uniform float aspect; uniform vec3 tint; uniform vec2 spot; uniform float radius;
      void main() { vec2 p = uv - spot; p.x *= aspect; colour = vec4(texture(target, uv).xyz + exp(-dot(p, p) / radius) * tint, 1.0); }`,
    advect: `uniform sampler2D velocity; uniform sampler2D source; uniform vec2 texel; uniform float dt; uniform float fade;
      void main() { vec2 from = uv - dt * texture(velocity, uv).xy * texel; colour = fade * texture(source, from); }`,
    curl: `uniform sampler2D velocity;
      void main() { float c = texture(velocity, uvR).y - texture(velocity, uvL).y - texture(velocity, uvT).x + texture(velocity, uvB).x; colour = vec4(0.5 * c, 0.0, 0.0, 1.0); }`,
    vorticity: `uniform sampler2D velocity; uniform sampler2D curl; uniform float strength; uniform float dt;
      void main() {
        float l = texture(curl, uvL).x, r = texture(curl, uvR).x, t = texture(curl, uvT).x, b = texture(curl, uvB).x, c = texture(curl, uv).x;
        vec2 force = 0.5 * vec2(abs(t) - abs(b), abs(r) - abs(l));
        force /= length(force) + 0.0001;
        force *= strength * c; force.y *= -1.0;
        colour = vec4(texture(velocity, uv).xy + force * dt, 0.0, 1.0);
      }`,
    divergence: `uniform sampler2D velocity;
      void main() {
        float l = texture(velocity, uvL).x, r = texture(velocity, uvR).x, t = texture(velocity, uvT).y, b = texture(velocity, uvB).y;
        vec2 c = texture(velocity, uv).xy;
        if (uvL.x < 0.0) l = -c.x; if (uvR.x > 1.0) r = -c.x; if (uvT.y > 1.0) t = -c.y; if (uvB.y < 0.0) b = -c.y;
        colour = vec4(0.5 * (r - l + t - b), 0.0, 0.0, 1.0);
      }`,
    pressure: `uniform sampler2D pressure; uniform sampler2D divergence;
      void main() {
        float l = texture(pressure, uvL).x, r = texture(pressure, uvR).x, t = texture(pressure, uvT).x, b = texture(pressure, uvB).x;
        colour = vec4((l + r + b + t - texture(divergence, uv).x) * 0.25, 0.0, 0.0, 1.0);
      }`,
    subtract: `uniform sampler2D pressure; uniform sampler2D velocity;
      void main() {
        float l = texture(pressure, uvL).x, r = texture(pressure, uvR).x, t = texture(pressure, uvT).x, b = texture(pressure, uvB).x;
        colour = vec4(texture(velocity, uv).xy - vec2(r - l, t - b), 0.0, 1.0);
      }`,
    clear: `uniform sampler2D target; uniform float value;
      void main() { colour = value * texture(target, uv); }`,
    show: `uniform sampler2D dye;
      void main() { vec3 c = texture(dye, uv).rgb; c = c / (1.0 + c); float a = max(c.r, max(c.g, c.b)); colour = vec4(c, a); }`,
  };

  function compile(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    return shader;
  }

  function program(body) {
    const made = gl.createProgram();
    gl.attachShader(made, compile(gl.VERTEX_SHADER, VERTEX));
    gl.attachShader(made, compile(gl.FRAGMENT_SHADER, HEAD + body));
    gl.bindAttribLocation(made, 0, "point");
    gl.linkProgram(made);
    if (!gl.getProgramParameter(made, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(made));
    const uniforms = {};
    for (let i = 0; i < gl.getProgramParameter(made, gl.ACTIVE_UNIFORMS); i += 1) {
      const name = gl.getActiveUniform(made, i).name;
      uniforms[name] = gl.getUniformLocation(made, name);
    }
    return { made, uniforms };
  }

  function target(width, height) {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, width, height, 0, gl.RGBA, gl.HALF_FLOAT, null);
    const buffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, buffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error("float target unsupported");
    return { texture, buffer, width, height };
  }

  function pair(width, height) {
    let read = target(width, height);
    let write = target(width, height);
    return { get read() { return read; }, get write() { return write; }, swap() { [read, write] = [write, read]; }, width, height };
  }

  let programs, velocity, dye, pressure, divergence, curl;
  try {
    programs = Object.fromEntries(Object.entries(SHADERS).map(([name, body]) => [name, program(body)]));
    const aspect = innerWidth / innerHeight;
    const size = (base) => (aspect >= 1 ? [Math.round(base * aspect), base] : [base, Math.round(base / aspect)]);
    velocity = pair(...size(SIM));
    dye = pair(...size(DYE));
    pressure = pair(...size(SIM));
    divergence = target(...size(SIM));
    curl = target(...size(SIM));
  } catch (error) {
    console.warn("Fluid background off:", error.message);
    return;
  }

  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(0);

  function run(name, output, settings = {}, textures = {}) {
    const { made, uniforms } = programs[name];
    gl.useProgram(made);
    const size = output || { width: canvas.width, height: canvas.height };
    gl.uniform2f(uniforms.texel, 1 / size.width, 1 / size.height);
    let unit = 0;
    for (const [key, texture] of Object.entries(textures)) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, texture.texture);
      gl.uniform1i(uniforms[key], unit);
      unit += 1;
    }
    for (const [key, setting] of Object.entries(settings)) {
      if (uniforms[key] === undefined) continue;
      if (Array.isArray(setting)) gl[`uniform${setting.length}f`](uniforms[key], ...setting);
      else gl.uniform1f(uniforms[key], setting);
    }
    gl.viewport(0, 0, size.width, size.height);
    gl.bindFramebuffer(gl.FRAMEBUFFER, output ? output.buffer : null);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  function splat(x, y, dx, dy, tint) {
    const aspect = canvas.width / canvas.height;
    run("splat", velocity.write, { aspect, tint: [dx, dy, 0], spot: [x, y], radius: RADIUS * Math.max(aspect, 1) }, { target: velocity.read });
    velocity.swap();
    run("splat", dye.write, { aspect, tint, spot: [x, y], radius: RADIUS * Math.max(aspect, 1) }, { target: dye.read });
    dye.swap();
  }

  function colour(strength = 0.3) {
    return COLOURS[Math.floor(Math.random() * COLOURS.length)].map((v) => v * strength);
  }

  function randomSplat() {
    const angle = Math.random() * Math.PI * 2;
    const speed = 500 + Math.random() * 700;
    splat(0.15 + Math.random() * 0.7, 0.25 + Math.random() * 0.6, Math.cos(angle) * speed, Math.sin(angle) * speed, colour(0.42));
  }

  function step(dt) {
    run("curl", curl, {}, { velocity: velocity.read });
    run("vorticity", velocity.write, { strength: CURL, dt }, { velocity: velocity.read, curl });
    velocity.swap();
    run("divergence", divergence, {}, { velocity: velocity.read });
    run("clear", pressure.write, { value: 0.8 }, { target: pressure.read });
    pressure.swap();
    for (let i = 0; i < PRESSURE_STEPS; i += 1) {
      run("pressure", pressure.write, {}, { pressure: pressure.read, divergence });
      pressure.swap();
    }
    run("subtract", velocity.write, {}, { pressure: pressure.read, velocity: velocity.read });
    velocity.swap();
    run("advect", velocity.write, { dt, fade: VELOCITY_FADE, texel: [1 / velocity.width, 1 / velocity.height] }, { velocity: velocity.read, source: velocity.read });
    velocity.swap();
    run("advect", dye.write, { dt, fade: DYE_FADE, texel: [1 / velocity.width, 1 / velocity.height] }, { velocity: velocity.read, source: dye.read });
    dye.swap();
  }

  function resize() {
    const scale = Math.min(devicePixelRatio || 1, 1.5);
    const width = Math.round(canvas.clientWidth * scale);
    const height = Math.round(canvas.clientHeight * scale);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
  }

  const pointer = { x: 0, y: 0, moved: false, last: 0 };
  addEventListener("pointermove", (event) => {
    const x = event.clientX / innerWidth;
    const y = 1 - event.clientY / innerHeight;
    if (pointer.last) {
      pointer.dx = (x - pointer.x) * FORCE;
      pointer.dy = (y - pointer.y) * FORCE;
      pointer.moved = Math.abs(pointer.dx) + Math.abs(pointer.dy) > 0;
    }
    pointer.x = x;
    pointer.y = y;
    pointer.last = performance.now();
  }, { passive: true });
  addEventListener("pointerdown", (event) => {
    for (let i = 0; i < 3; i += 1) {
      const angle = Math.random() * Math.PI * 2;
      splat(event.clientX / innerWidth, 1 - event.clientY / innerHeight, Math.cos(angle) * 900, Math.sin(angle) * 900, colour(0.35));
    }
  }, { passive: true });

  let then = performance.now();
  let idleSince = then;
  let frame = 0;
  function loop(now) {
    const dt = Math.min((now - then) / 1000, 1 / 30);
    then = now;
    resize();
    if (pointer.moved) {
      splat(pointer.x, pointer.y, pointer.dx, pointer.dy, colour());
      pointer.moved = false;
      idleSince = now;
    } else if (now - idleSince > IDLE_SPLAT) {
      randomSplat();
      idleSince = now;
    }
    step(dt);
    run("show", null, {}, { dye: dye.read });
    canvas.dataset.frames = String((frame += 1));
    request = requestAnimationFrame(loop);
  }

  resize();
  for (let i = 0; i < 6; i += 1) randomSplat();
  let request = requestAnimationFrame(loop);
  canvas.classList.add("live");
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) cancelAnimationFrame(request);
    else {
      then = performance.now();
      request = requestAnimationFrame(loop);
    }
  });
})();
