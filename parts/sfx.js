/* 彩還し sfx.js — all-synthesized WebAudio sound effects (no files, no network). Defines window.SFX = {unlock, play, setMuted, isMuted, toggleMuted}.
 * Integration: (1) include this file before the game script; (2) call SFX.unlock() on the first pointerdown/keydown (safe to call repeatedly);
 * (3) call SFX.play('fan'|'hit'|'kill'|'restore'|'hurt'|'heal'|'upgrade'|'boss'|'clear'|'retreat'|'death'|'click'[, {vol, pitch}]) at the matching game events;
 * (4) add a mute button that calls SFX.toggleMuted() (returns the new state; SFX.isMuted() for its label; persisted in localStorage "irogaeshi.muted");
 * (5) spam is fine: per-sound rate limits + a ~12-voice cap drop excess silently, and with no AudioContext every call is a silent no-op that never throws. */
(function (W) {
  'use strict';

  // ---------------------------------------------------------------- constants
  const KEY = 'irogaeshi.muted';
  const MASTER = 0.25;             // low master gain; a compressor sits behind it as a safety net
  const SOFT_CAP = 12;             // ordinary sounds never push the live voice count above this (tiered below)
  const HARD_CAP = 28;             // priority sounds (hurt, boss, clear, ...) may go this high
  const LEAD = 0.012;              // scheduling lead-time (s)
  const CHAIN_MS = 1500;           // kills closer together than this keep the pitch chain going
  const SEMI = [0, 2, 4, 7, 9];    // 宮 商 角 徴 羽  =  C D E G A
  const C4 = 261.6256;

  // pentatonic degree index -> Hz.  0 = C4, 4 = A4, 5 = C5 ... -1 = A3
  function pent(i) {
    const o = Math.floor(i / 5);
    return C4 * Math.pow(2, (o * 12 + SEMI[i - o * 5]) / 12);
  }
  function rr(a, b) { return a + Math.random() * (b - a); }
  function later(fn, ms) {
    try {
      if (typeof setTimeout === 'function') return setTimeout(fn, ms);
      if (W && typeof W.setTimeout === 'function') return W.setTimeout(fn, ms);
    } catch (e) { /* ignore */ }
    return 0;
  }

  // -------------------------------------------------------------------- state
  let ctx = null, master = null, dryBus = null, wetBus = null, noiseBuf = null;
  let failed = false;              // AudioContext unavailable / construction failed -> permanent silent no-op
  let primed = false;
  let active = 0;                  // live voices
  let DRY = false;                 // pre-warm mode: only renders pluck buffers, creates no nodes
  let gv = 1, gp = 0;              // per-call volume multiplier / pitch shift (cents)
  let lastResume = 0;
  let chain = 0, lastKill = -1e9, restoreN = 0, fanDir = 1;
  let cache = {}, cacheN = 0;
  let cur = '';                    // name of the sound currently being started
  const live = {};                 // name -> live voices of that sound (per-sound quota)
  const lastT = {};                // name -> audio-clock ms of the last accepted play

  let muted = false;
  try {
    const s = W.localStorage.getItem(KEY);
    muted = (s === '1' || s === 'true');
  } catch (e) { /* storage unavailable: default not muted */ }

  // ------------------------------------------------- Karplus-Strong (to buffer)
  // Rendered once per (pitch, decay, brightness) and cached; playing a pluck is then just a BufferSource.
  // WebAudio's DelayNode cannot loop faster than one render quantum (~344 Hz max), so the
  // string loop (noise burst -> delay line -> 2-tap lowpass -> fractional all-pass) is run in JS.
  function renderPluck(sr, freq, t60, bright) {
    const D = sr / freq - 0.5;                     // loop delay, minus the 0.5 sample of the averaging lowpass
    const N = Math.max(2, Math.floor(D - 0.5));
    const frac = D - N;                            // 0.5 .. 1.5 samples handled by the all-pass
    const a = (1 - frac) / (1 + frac);
    const len = Math.max(64, Math.floor(sr * t60 * 1.2));
    const out = new Float32Array(len);
    const ring = new Float32Array(N);
    let seed = ((freq * 977) | 0) || 1;            // deterministic xorshift noise
    const c = 0.12 + 0.88 * bright;                // excitation smoothing: low = dull/muffled, high = bright
    let y = 0, mean = 0, i;
    for (i = 0; i < N; i++) {
      seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5;
      y += c * ((((seed >>> 0) / 4294967296) * 2 - 1) - y);
      ring[i] = y; mean += y;
    }
    mean /= N;
    for (i = 0; i < N; i++) ring[i] -= mean;       // no DC
    const pick = Math.max(1, Math.round(N * 0.17)); // pluck-position comb (guzheng-ish)
    for (i = N - 1; i >= pick; i--) ring[i] -= 0.6 * ring[i - pick];
    const g = Math.pow(10, -3 / (freq * t60));     // per-period loss -> -60 dB after t60 seconds
    let p = 0, prev = 0, ax = 0, ay = 0, peak = 1e-9;
    for (let n = 0; n < len; n++) {
      const x = ring[p];
      const avg = 0.5 * (x + prev); prev = x;      // lowpass loss filter
      const z = a * avg + ax - a * ay; ax = avg; ay = z;   // fractional delay all-pass
      ring[p] = z * g;
      out[n] = x;
      const ab = x < 0 ? -x : x; if (ab > peak) peak = ab;
      if (++p >= N) p = 0;
    }
    const k = 0.9 / peak;
    const fadeIn = Math.min(len, Math.floor(sr * 0.0015));
    const fadeOut = Math.min(len, Math.floor(sr * 0.08));
    for (i = 0; i < len; i++) {
      let s = out[i] * k;
      if (i < fadeIn) s *= i / fadeIn;
      if (i >= len - fadeOut) s *= (len - 1 - i) / fadeOut;
      out[i] = s;
    }
    return out;
  }

  function getPluck(freq, t60, bright) {
    const key = ((freq * 100) | 0) + '|' + t60 + '|' + bright;
    let b = cache[key];
    if (b) return b;
    if (cacheN >= 160) { cache = {}; cacheN = 0; }
    const data = renderPluck(ctx.sampleRate, freq, t60, bright);
    b = ctx.createBuffer(1, data.length, ctx.sampleRate);
    b.getChannelData(0).set(data);
    cache[key] = b; cacheN++;
    return b;
  }

  // ------------------------------------------------------ voices & node helpers
  // A "voice" = one logical sound element (possibly several sources sharing a lifetime).
  // It counts once toward the voice cap and is disconnected when its sources end (plus a timer fallback).
  function voice(bus) { return { bus: bus, srcs: [], nodes: [], end: 0, name: cur }; }
  function add(v, src, nodes, end) {
    v.srcs.push(src);
    for (let i = 0; i < nodes.length; i++) v.nodes.push(nodes[i]);
    if (end > v.end) v.end = end;
  }
  function go(v) {
    if (DRY || !v.srcs.length) return;
    active++; live[v.name] = (live[v.name] || 0) + 1;
    let left = v.srcs.length, done = false;
    const fin = function () {
      if (done) return;
      done = true; active--; live[v.name]--;
      for (let i = 0; i < v.nodes.length; i++) { try { v.nodes[i].disconnect(); } catch (e) { /* already gone */ } }
    };
    for (let i = 0; i < v.srcs.length; i++) v.srcs[i].onended = function () { if (--left <= 0) fin(); };
    later(fin, Math.max(60, (v.end - ctx.currentTime) * 1000 + 400));
  }

  // 0 -> peak over `att`, then exponential decay reaching -60 dB after `t60`
  function env(p, t, peak, att, t60) {
    p.setValueAtTime(0.0001, t);
    p.linearRampToValueAtTime(peak, t + att);
    p.setTargetAtTime(0, t + att, t60 / 6.9);
  }

  // o: {t, type, f0, f1?, glide?, att, t60, vol, lp?}
  function addOsc(v, o) {
    if (DRY) return;
    const vol = o.vol * gv;
    if (!(vol > 0.0005)) return;
    const dur = o.att + o.t60 * 1.4 + 0.03;
    const s = ctx.createOscillator();
    s.type = o.type || 'sine';
    s.frequency.setValueAtTime(o.f0, o.t);
    if (o.f1) s.frequency.exponentialRampToValueAtTime(o.f1, o.t + (o.glide || dur));
    if (s.detune) s.detune.value = gp;
    const g = ctx.createGain();
    env(g.gain, o.t, vol, o.att, o.t60);
    const nodes = [s, g];
    if (o.lp) {
      const f = ctx.createBiquadFilter();
      f.type = 'lowpass'; f.frequency.value = o.lp; f.Q.value = 0.7;
      s.connect(f); f.connect(g); nodes.push(f);
    } else s.connect(g);
    g.connect(v.bus);
    s.start(o.t); s.stop(o.t + dur);
    add(v, s, nodes, o.t + dur);
  }

  // o: {t, type:'bandpass'|'lowpass'|'highpass', f0, f1?, glide?, q?, att, t60, vol}
  function addNoise(v, o) {
    if (DRY) return;
    const vol = o.vol * gv;
    if (!(vol > 0.0005)) return;
    const dur = o.att + o.t60 * 1.4 + 0.03;
    const s = ctx.createBufferSource();
    s.buffer = noiseBuf; s.loop = true;
    const f = ctx.createBiquadFilter();
    f.type = o.type || 'bandpass';
    f.Q.value = o.q || 0.7;
    f.frequency.setValueAtTime(o.f0, o.t);
    if (o.f1) f.frequency.exponentialRampToValueAtTime(o.f1, o.t + (o.glide || dur));
    const g = ctx.createGain();
    env(g.gain, o.t, vol, o.att, o.t60);
    s.connect(f); f.connect(g); g.connect(v.bus);
    s.start(o.t, Math.random() * Math.max(0.01, noiseBuf.duration - 0.05));
    s.stop(o.t + dur);
    add(v, s, [s, f, g], o.t + dur);
  }

  // o: {t, freq, t60, bright, vol, cents?, drop?, lp?}
  function addPluck(v, o) {
    const buf = getPluck(o.freq, o.t60, o.bright);
    if (DRY) return;
    const vol = o.vol * gv;
    if (!(vol > 0.0005)) return;
    const s = ctx.createBufferSource();
    s.buffer = buf;
    const cents = gp + (o.cents || 0);
    const ratio = Math.pow(2, cents / 1200);
    const drop = o.drop === undefined ? 24 : o.drop;     // quick pitch drop at the attack = guzheng "bend"
    if (s.detune && s.detune.setValueAtTime) {
      s.detune.setValueAtTime(cents + drop, o.t);
      s.detune.setTargetAtTime(cents, o.t, 0.022);
    } else s.playbackRate.value = ratio;
    const g = ctx.createGain();
    g.gain.value = vol;
    const nodes = [s, g];
    if (o.lp) {
      const f = ctx.createBiquadFilter();
      f.type = 'lowpass'; f.frequency.value = o.lp; f.Q.value = 0.6;
      s.connect(f); f.connect(g); nodes.push(f);
    } else s.connect(g);
    g.connect(v.bus);
    const end = o.t + buf.duration / ratio + 0.03;
    s.start(o.t); s.stop(end);
    add(v, s, nodes, end);
  }
  function pluck(o) { const v = voice(o.bus || wetBus); addPluck(v, o); go(v); }

  // ------------------------------------------------------------------ sounds
  // Each takes (t = start time on the audio clock, o = caller options).

  // fan: very short airy whoosh, alternating sweep direction, quiet
  function sFan(t) {
    const v = voice(dryBus);
    const up = fanDir > 0; fanDir = -fanDir;
    const lo = rr(650, 900), hi = rr(2300, 3100);
    addNoise(v, { t: t, type: 'bandpass', f0: up ? lo : hi, f1: up ? hi : lo, glide: 0.15, q: 0.8, att: 0.03, t60: 0.085, vol: 0.8 });
    go(v);
  }

  // hit: dull muffled ink "thup"
  function sHit(t) {
    const v = voice(dryBus);
    const f = rr(165, 215);
    addOsc(v, { t: t, type: 'triangle', f0: f, f1: f * 0.42, glide: 0.07, att: 0.002, t60: 0.075, vol: 0.8, lp: 900 });
    addNoise(v, { t: t, type: 'lowpass', f0: rr(650, 850), f1: 260, glide: 0.05, q: 0.7, att: 0.001, t60: 0.04, vol: 0.9 });
    go(v);
  }

  // kill: wet bubble pop + one pluck whose pitch climbs the scale along a chain of kills
  const KILL_BASE = 5, KILL_SPAN = 9, KILL_TAIL = [0, -1, 0, -2];   // C5 .. G6, then wobble near the top
  function sKill(t, o) {
    let step;
    if (o && typeof o.step === 'number') step = Math.max(0, o.step | 0);    // explicit step (also used by pre-warm)
    else {
      if (t * 1000 - lastKill > CHAIN_MS) chain = 0;
      lastKill = t * 1000;
      step = chain++;
    }
    const v = voice(dryBus);
    const f = rr(250, 340);
    addOsc(v, { t: t, type: 'sine', f0: f, f1: f * 3, glide: 0.032, att: 0.002, t60: 0.05, vol: 0.55 });
    addNoise(v, { t: t, type: 'highpass', f0: 2400, q: 0.7, att: 0.001, t60: 0.015, vol: 0.35 });
    go(v);
    const idx = KILL_BASE + (step < KILL_SPAN ? step : KILL_SPAN - 1 + KILL_TAIL[(step - KILL_SPAN) & 3]);
    pluck({ t: t + 0.008, freq: pent(idx), t60: 0.5, bright: 0.75, vol: 0.55 + 0.03 * Math.min(step, 8), bus: wetBus, drop: 22 });
  }

  // restore: soft shimmering chime, two plucks an octave apart, the upper one slightly detuned
  const RESTORE_ORDER = [0, 2, 1, 3, 4, 2, 0, 3];
  function sRestore(t, o) {
    const deg = (o && typeof o.deg === 'number') ? (o.deg | 0) : RESTORE_ORDER[restoreN++ % RESTORE_ORDER.length];
    const i = 5 + (((deg % 5) + 5) % 5);
    pluck({ t: t, freq: pent(i), t60: 0.8, bright: 0.55, vol: 0.5, bus: wetBus, drop: 6 });
    pluck({ t: t + 0.05, freq: pent(i + 5), t60: 0.8, bright: 0.6, vol: 0.36, bus: wetBus, cents: 9, drop: 4 });
  }

  // hurt: low thud + short descending tone
  function sHurt(t) {
    const v = voice(dryBus);
    addOsc(v, { t: t, type: 'sine', f0: 150, f1: 42, glide: 0.2, att: 0.003, t60: 0.2, vol: 1.0 });
    addOsc(v, { t: t, type: 'triangle', f0: 560, f1: 300, glide: 0.24, att: 0.004, t60: 0.17, vol: 0.3, lp: 1800 });
    addNoise(v, { t: t, type: 'lowpass', f0: 520, f1: 200, glide: 0.08, q: 0.7, att: 0.001, t60: 0.05, vol: 0.8 });
    go(v);
  }

  // heal: gentle rising two-note pluck (C5 -> G5)
  function sHeal(t) {
    pluck({ t: t, freq: pent(5), t60: 1.1, bright: 0.4, vol: 0.55, bus: wetBus, drop: 8 });
    pluck({ t: t + 0.12, freq: pent(8), t60: 1.1, bright: 0.4, vol: 0.6, bus: wetBus, drop: 8 });
  }

  // upgrade: bright 3-note rising arpeggio (E5 A5 C6) + a tiny sparkle
  function sUpgrade(t) {
    pluck({ t: t, freq: pent(7), t60: 1.1, bright: 0.95, vol: 0.6, bus: wetBus });
    pluck({ t: t + 0.075, freq: pent(9), t60: 1.1, bright: 0.95, vol: 0.65, bus: wetBus });
    pluck({ t: t + 0.15, freq: pent(10), t60: 1.1, bright: 0.95, vol: 0.7, bus: wetBus });
    const v = voice(wetBus);
    addOsc(v, { t: t + 0.15, type: 'sine', f0: 2093, att: 0.002, t60: 0.35, vol: 0.12 });
    go(v);
  }

  // boss: ominous gong (inharmonic partials, ~2.5 s decay) + low rumble
  const GONG = [[1, 1, 2.6], [1.0135, 0.8, 2.4], [1.47, 0.7, 2.3], [2.09, 0.55, 1.9], [2.56, 0.4, 1.5], [3.31, 0.3, 1.1], [4.12, 0.18, 0.8]];
  function sBoss(t) {
    const base = 82.41;                                  // E2
    const v = voice(wetBus);
    for (let i = 0; i < GONG.length; i++) {
      const f = base * GONG[i][0];
      addOsc(v, { t: t, type: 'sine', f0: f * 1.012, f1: f, glide: 0.5, att: 0.006, t60: GONG[i][2], vol: 0.28 * GONG[i][1] });
    }
    addNoise(v, { t: t, type: 'bandpass', f0: 800, q: 1.1, att: 0.002, t60: 0.07, vol: 0.5 });   // mallet strike
    go(v);
    const r = voice(dryBus);
    addNoise(r, { t: t, type: 'lowpass', f0: 240, f1: 80, glide: 2.2, q: 0.8, att: 0.3, t60: 1.4, vol: 3.2 });
    go(r);
  }

  // clear: 8-note ascending pentatonic run (~1.1 s) resolving to a sustained C chord
  function sClear(t) {
    for (let k = 0; k < 8; k++) {
      pluck({ t: t + k * 0.155, freq: pent(3 + k), t60: 1.0, bright: 0.7 + k * 0.03, vol: 0.42 + 0.035 * k, bus: wetBus, drop: 24 });
    }
    const tc = t + 1.2;                                  // C4 + G4 + E5, lightly rolled
    pluck({ t: tc, freq: pent(0), t60: 2.4, bright: 0.6, vol: 0.6, bus: wetBus });
    pluck({ t: tc + 0.03, freq: pent(3), t60: 2.4, bright: 0.6, vol: 0.5, bus: wetBus });
    pluck({ t: tc + 0.06, freq: pent(7), t60: 2.4, bright: 0.6, vol: 0.45, bus: wetBus });
    const v = voice(wetBus);                             // soft sustained pad under the chord
    [pent(0), pent(3), pent(5)].forEach(function (f) {
      addOsc(v, { t: tc, type: 'triangle', f0: f, att: 0.12, t60: 1.6, vol: 0.16, lp: 1300 });
    });
    go(v);
  }

  // retreat: neutral two-note descending pluck (D5 -> A4)
  function sRetreat(t) {
    pluck({ t: t, freq: pent(6), t60: 0.7, bright: 0.5, vol: 0.55, bus: wetBus });
    pluck({ t: t + 0.13, freq: pent(4), t60: 0.7, bright: 0.5, vol: 0.55, bus: wetBus });
  }

  // death: slow muffled descending phrase in the 羽 (A-minor-ish) mode: E5 C5 A4 E4 A3
  function sDeath(t) {
    const N = [7, 5, 4, 2, -1], T = [0, 0.3, 0.62, 1.0, 1.55], D = [0.8, 0.8, 0.9, 1.0, 2.0], V = [0.6, 0.6, 0.62, 0.65, 0.7];
    for (let i = 0; i < N.length; i++) {
      pluck({ t: t + T[i], freq: pent(N[i]), t60: D[i], bright: 0.18, vol: V[i], bus: wetBus, lp: 1100, drop: 14 });
    }
  }

  // click: tiny UI tick
  function sClick(t) {
    const v = voice(dryBus);
    addOsc(v, { t: t, type: 'sine', f0: 1900, f1: 1250, glide: 0.02, att: 0.001, t60: 0.018, vol: 0.4 });
    addNoise(v, { t: t, type: 'highpass', f0: 3200, q: 0.7, att: 0.0005, t60: 0.01, vol: 0.3 });
    go(v);
  }

  // name -> {gap: min ms between plays, need: voices it spawns, max: per-sound live-voice quota,
  //          cap: ceiling the global live-voice count may reach when this sound starts, fn}
  // Every play must pass three limits, else it is dropped silently: the per-sound rate limit (gap), the sound's own
  // quota (so a swarm of one sound can never hog the pool), and a tiered global ceiling (restore 12 > kill 10 > hit 8 >
  // fan 7, so the cheap, frequent ticks back off first and always leave headroom for kill/restore). The priority
  // sounds (hurt, heal, upgrade, boss, clear, retreat, death, click) have no quota and may use up to HARD_CAP.
  const DEFS = {
    fan:     { gap: 90,   need: 1,  max: 2,  cap: 7,        fn: sFan },
    hit:     { gap: 40,   need: 1,  max: 4,  cap: 8,        fn: sHit },
    kill:    { gap: 45,   need: 2,  max: 8,  cap: 10,       fn: sKill },
    restore: { gap: 120,  need: 2,  max: 4,  cap: SOFT_CAP, fn: sRestore },
    hurt:    { gap: 120,  need: 1,  max: 99, cap: HARD_CAP, fn: sHurt },
    heal:    { gap: 150,  need: 2,  max: 99, cap: HARD_CAP, fn: sHeal },
    upgrade: { gap: 200,  need: 4,  max: 99, cap: HARD_CAP, fn: sUpgrade },
    boss:    { gap: 1200, need: 2,  max: 99, cap: HARD_CAP, fn: sBoss },
    clear:   { gap: 1500, need: 12, max: 99, cap: HARD_CAP, fn: sClear },
    retreat: { gap: 300,  need: 2,  max: 99, cap: HARD_CAP, fn: sRetreat },
    death:   { gap: 1500, need: 5,  max: 99, cap: HARD_CAP, fn: sDeath },
    click:   { gap: 30,   need: 1,  max: 99, cap: HARD_CAP, fn: sClick }
  };

  // ------------------------------------------------------------ audio graph
  function build() {
    master = ctx.createGain();
    master.gain.value = muted ? 0 : MASTER;
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -14; comp.knee.value = 16; comp.ratio.value = 8;
    comp.attack.value = 0.004; comp.release.value = 0.2;
    master.connect(comp); comp.connect(ctx.destination);

    dryBus = ctx.createGain(); dryBus.connect(master);
    wetBus = ctx.createGain(); wetBus.connect(master);   // wetBus also feeds the little feedback-delay "room"
    const send = ctx.createGain(); send.gain.value = 0.3; wetBus.connect(send);
    const ret = ctx.createGain(); ret.gain.value = 0.6; ret.connect(master);
    [[0.087, 0.36, 3000], [0.131, 0.32, 2200]].forEach(function (p) {
      const d = ctx.createDelay(0.5); d.delayTime.value = p[0];
      const lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = p[2];
      const fb = ctx.createGain(); fb.gain.value = p[1];
      send.connect(d); d.connect(lp); lp.connect(fb); fb.connect(d); lp.connect(ret);
    });

    const n = Math.floor(ctx.sampleRate * 2);            // 2 s of white noise shared by all noise voices
    noiseBuf = ctx.createBuffer(1, n, ctx.sampleRate);
    const ch = noiseBuf.getChannelData(0);
    for (let i = 0; i < n; i++) ch[i] = Math.random() * 2 - 1;
  }

  // Render the pluck buffers the sounds will need, one job per timer tick, so no gameplay frame pays for it.
  function warm() {
    const jobs = [];
    const J = function (fn, o) { jobs.push(function () { fn(0, o); }); };
    for (let s = 0; s < KILL_SPAN; s++) J(sKill, { step: s });
    for (let d = 0; d < 5; d++) J(sRestore, { deg: d });
    [sHeal, sUpgrade, sRetreat, sDeath, sClear].forEach(function (fn) { J(fn); });
    let i = 0;
    (function next() {
      if (!ctx || i >= jobs.length) return;
      try { DRY = true; jobs[i++](); } catch (e) { /* ignore */ } finally { DRY = false; }
      later(next, 25);
    })();
  }

  // ---------------------------------------------------------------- public API
  let disarm = function () {};

  function unlock() {
    if (failed) return false;
    try {
      if (!ctx) {
        const C = W.AudioContext || W.webkitAudioContext;
        if (!C) { failed = true; return false; }
        try {
          ctx = new C();
          build();
        } catch (e) {
          try { if (ctx && ctx.close) ctx.close(); } catch (e2) { /* ignore */ }
          ctx = null; failed = true; return false;
        }
      }
      if (ctx.state !== 'running' && ctx.resume) {
        const p = ctx.resume();
        if (p && p.then) p.then(function () { if (ctx && ctx.state === 'running') disarm(); }, function () {});
      } else if (ctx.state === 'running') disarm();
      if (!primed) {
        primed = true;
        try {                                            // silent 1-sample blip: fully unlocks iOS Safari
          const b = ctx.createBuffer(1, 1, 22050), s = ctx.createBufferSource();
          s.buffer = b; s.connect(ctx.destination); s.start(0);
        } catch (e) { /* ignore */ }
        later(warm, 200);
      }
      return true;
    } catch (e) {
      return false;
    }
  }

  function play(name, opts) {
    try {
      if (muted || failed || !ctx) return false;
      if (!Object.prototype.hasOwnProperty.call(DEFS, name)) return false;
      if (ctx.state !== 'running') {                     // suspended / interrupted: try to wake, drop this one
        const nowMs = Date.now();
        if (ctx.resume && nowMs - lastResume > 1000) {
          lastResume = nowMs;
          const p = ctx.resume(); if (p && p.catch) p.catch(function () {});
        }
        return false;
      }
      const def = DEFS[name];
      const now = ctx.currentTime * 1000;
      const last = lastT[name];
      if (last !== undefined && now - last < def.gap) return false;      // per-sound rate limit
      if ((live[name] || 0) + def.need > def.max) return false;          // per-sound voice quota
      if (active + def.need > def.cap) return false;                     // global voice cap (tiered)
      const o = (opts && typeof opts === 'object') ? opts : {};
      gv = (typeof o.vol === 'number' && isFinite(o.vol)) ? Math.min(2, Math.max(0, o.vol)) : 1;
      if (gv <= 0.0005) return false;                                    // vol 0 = nothing to play
      lastT[name] = now;
      cur = name;
      gp = (typeof o.pitch === 'number' && isFinite(o.pitch)) ? Math.min(2400, Math.max(-2400, o.pitch)) : 0;
      def.fn(ctx.currentTime + LEAD, o);
      return true;
    } catch (e) {
      return false;
    } finally {
      gv = 1; gp = 0; cur = '';
    }
  }

  function setMuted(m) {
    muted = !!m;
    try { W.localStorage.setItem(KEY, muted ? '1' : '0'); } catch (e) { /* ignore */ }
    try {
      if (master) {
        const t = ctx.currentTime;
        master.gain.cancelScheduledValues(t);
        master.gain.setTargetAtTime(muted ? 0 : MASTER, t, 0.02);
      }
    } catch (e) { /* ignore */ }
    return muted;
  }
  function isMuted() { return !!muted; }
  function toggleMuted() { return setMuted(!muted); }

  // Convenience: also arm one-shot gesture listeners so audio unlocks even if the game forgets to call unlock().
  try {
    const d = W.document;
    if (d && typeof d.addEventListener === 'function') {
      const evs = ['pointerdown', 'keydown', 'touchstart', 'mousedown'];
      const onGesture = function () { unlock(); };
      disarm = function () {
        disarm = function () {};
        for (let i = 0; i < evs.length; i++) { try { d.removeEventListener(evs[i], onGesture, true); } catch (e) { /* ignore */ } }
      };
      for (let i = 0; i < evs.length; i++) d.addEventListener(evs[i], onGesture, true);
    }
  } catch (e) { /* ignore */ }

  W.SFX = {
    unlock: unlock,
    play: play,
    setMuted: setMuted,
    isMuted: isMuted,
    toggleMuted: toggleMuted,
    names: Object.keys(DEFS)
  };
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : {}));
