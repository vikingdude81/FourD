// Faithful JS port of FourD UniversalEngine (s3 manifold, gradual fatigue)
// Verified against the torch reference (see demo README).

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function makeGauss(rng) {
  let spare = null;
  return function () {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u, v, r;
    do { u = rng() * 2 - 1; v = rng() * 2 - 1; r = u * u + v * v; } while (r === 0 || r >= 1);
    const m = Math.sqrt(-2 * Math.log(r) / r);
    spare = v * m; return u * m;
  };
}

class Engine {
  constructor(prefs, macro, prm, rngSeed) {
    this.prefs = prefs;            // S x 4 (assumed unit)
    this.macro = macro;            // 24 x 4
    this.prm = prm;
    const rng = mulberry32(rngSeed), gauss = makeGauss(rng);
    this.rng = rng; this.gauss = gauss;
    this.N = prm.N; this.S = prefs.length;
    this.F = new Float64Array(this.N * this.S);
    this.u = new Float64Array(this.N * 4);
    for (let i = 0; i < this.N * 4; i++) this.u[i] = gauss();
    for (let i = 0; i < this.N; i++) {
      let n = 0;
      for (let d = 0; d < 4; d++) n += this.u[i * 4 + d] ** 2;
      n = Math.sqrt(n);
      for (let d = 0; d < 4; d++) this.u[i * 4 + d] /= n;
    }
    this.basins = [];              // steps arrays of Int8/Uint8 basin per agent (store transposed: steps[i][t]? keep per-agent arrays)
    this.clar = [];                // per-agent clarity arrays
  }

  step() {
    const { prefs, macro, prm, N, S } = this, u = this.u, F = this.F;
    const invA = prm.infl_a, invB = prm.infl_b, inv = new Float64Array(N * S);
    const act = new Float64Array(N * S);
    const rest = new Float64Array(N * S);
    const rowSum = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const u0 = u[i*4], u1 = u[i*4+1], u2 = u[i*4+2], u3 = u[i*4+3];
      let sum = 0;
      for (let j = 0; j < S; j++) {
        const p = prefs[j], p0 = p[0], p1 = p[1], p2 = p[2], p3 = p[3];
        const c = u0*p0 + u1*p1 + u2*p2 + u3*p3;
        let e = (invA + invB * c) * Math.exp(-F[i*S+j]) + prm.noise * this.gauss();
        if (e < prm.floor) e = prm.floor;
        inv[i*S+j] = e; rest[i*S+j] = Math.exp(-F[i*S+j]); sum += e;
      }
      rowSum[i] = sum + 1e-8;
      let fs = 0;
      for (let j = 0; j < S; j++) fs += rest[i*S+j];
      rest[i*S+0] += 0;            // (no-op, keeps structure)
      this._meanRest = fs / S;     // overwritten per-agent in loops below; kept for clarity
    }
    // activities
    for (let i = 0; i < N * S; i++) act[i] = inv[i] / rowSum[(i / S) | 0];
    // fatigue update (gradual)
    const invS = 1 / S, fr = prm.fatigue_rate, rec = prm.recovery;
    for (let k = 0; k < N * S; k++) {
      const a = act[k];
      let f = F[k] + fr * a;
      const ex = a - invS;
      f += 0.03 * (ex > 0.02 ? ex - 0.02 : 0);
      f -= (1 - a) * rec;
      F[k] = f < 0 ? 0 : (f > 3 ? 3 : f);
    }
    // forces & drive
    const nu = new Float64Array(N * 4);
    const clarity = new Float64Array(N);
    const basin = new Uint8Array(N);
    for (let i = 0; i < N; i++) {
      const u0 = u[i*4], u1 = u[i*4+1], u2 = u[i*4+2], u3 = u[i*4+3];
      let ax=0, ay=0, az=0, aw=0;      // activity_force
      let nx=0, ny=0, nz=0, nw=0;      // novelty raw sum rest*f
      let mx=0, my=0, mz=0, mw=0;      // mean_j f
      let meanRest = 0;
      for (let j = 0; j < S; j++) {
        const p = prefs[j];
        const c = u0*p[0] + u1*p[1] + u2*p[2] + u3*p[3];
        const fx = p[0] - c*u0, fy = p[1] - c*u1, fz = p[2] - c*u2, fw = p[3] - c*u3;
        const a = act[i*S+j], r = rest[i*S+j];
        ax += a*fx; ay += a*fy; az += a*fz; aw += a*fw;
        nx += r*fx; ny += r*fy; nz += r*fz; nw += r*fw;
        mx += fx; my += fy; mz += fz; mw += fw;
        meanRest += r;
      }
      meanRest /= S; mx /= S; my /= S; mz /= S; mw /= S;
      nx -= meanRest*mx; ny -= meanRest*my; nz -= meanRest*mz; nw -= meanRest*mw;
      let dx = 0.4*ax + 0.6*nx, dy = 0.4*ay + 0.6*ny, dz = 0.4*az + 0.6*nz, dw = 0.4*aw + 0.6*nw;
      // tangential noise
      const rn0 = prm.noise*this.gauss(), rn1 = prm.noise*this.gauss(),
            rn2 = prm.noise*this.gauss(), rn3 = prm.noise*this.gauss();
      const nr = rn0*u0 + rn1*u1 + rn2*u2 + rn3*u3;
      dx += rn0 - nr*u0; dy += rn1 - nr*u1; dz += rn2 - nr*u2; dw += rn3 - nr*u3;
      // clarity = |activity_force|
      clarity[i] = Math.sqrt(ax*ax + ay*ay + az*az + aw*aw);
      // update position
      let v0 = u0 + prm.steering*dx, v1 = u1 + prm.steering*dy,
          v2 = u2 + prm.steering*dz, v3 = u3 + prm.steering*dw;
      const vn = Math.sqrt(v0*v0 + v1*v1 + v2*v2 + v3*v3) || 1;
      v0/=vn; v1/=vn; v2/=vn; v3/=vn;
      nu[i*4]=v0; nu[i*4+1]=v1; nu[i*4+2]=v2; nu[i*4+3]=v3;
      // basin = argmax macro sim (softmax argmax == argmax)
      let best = -1e30, bi = 0;
      for (let m = 0; m < macro.length; m++) {
        const c2 = v0*macro[m][0] + v1*macro[m][1] + v2*macro[m][2] + v3*macro[m][3];
        if (c2 > best) { best = c2; bi = m; }
      }
      basin[i] = bi;
    }
    this.u = nu;
    for (let i = 0; i < N; i++) {
      if (!this.basins[i]) { this.basins[i] = []; this.clar[i] = []; }
      this.basins[i].push(basin[i]); this.clar[i].push(clarity[i]);
    }
  }

  run(steps) { for (let t = 0; t < steps; t++) this.step(); }

  meanFatiguePerSub() {
    const out = new Float64Array(this.S);
    for (let i = 0; i < this.N; i++)
      for (let j = 0; j < this.S; j++) out[j] += this.F[i*this.S+j];
    for (let j = 0; j < this.S; j++) out[j] /= this.N;
    return out;
  }
}

function transitionGap(basins, clar, N) {
  let cte = 0, cnt = 0, cdw = 0, cdn = 0;
  for (let i = 0; i < N; i++)
    for (let t = 1; t < basins[i].length; t++) {
      const c = clar[i][t];
      if (basins[i][t] !== basins[i][t-1]) { cte += c; cnt++; }
      else { cdw += c; cdn++; }
    }
  if (cnt < 10 || cdn < 10) return { gap: 0, tr: 0 };
  return { gap: cte/cnt - cdw/cdn, tr: cnt / (cnt + cdn) };
}

function boundaryZ(eng, nShuffle, rngSeed) {
  const N = eng.N, T = eng.basins[0].length;
  const { gap, tr } = transitionGap(eng.basins, eng.clar, N);
  const rng = mulberry32(rngSeed);
  const shuffled = [];
  for (let s = 0; s < nShuffle; s++) {
    const b = [];
    for (let i = 0; i < N; i++) {
      const row = eng.basins[i].slice();
      for (let k = row.length - 1; k > 0; k--) {
        const j = (rng() * (k + 1)) | 0;
        const tmp = row[k]; row[k] = row[j]; row[j] = tmp;
      }
      b.push(row);
    }
    shuffled.push(transitionGap(b, eng.clar, N).gap);
  }
  const mean = shuffled.reduce((a, b) => a + b, 0) / shuffled.length;
  const sd = Math.sqrt(shuffled.reduce((a, b) => a + (b-mean)**2, 0) / shuffled.length);
  return { null_z: (gap - mean) / (sd + 1e-15), transition_rate: tr };
}

// structure certificates: antipodal closure + 2nd-moment isotropy
function certificates(P) {
  const n = P.length;
  let closed = 0;
  for (let i = 0; i < n; i++) {
    let found = false;
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const d = P[i][0]*P[j][0] + P[i][1]*P[j][1] + P[i][2]*P[j][2] + P[i][3]*P[j][3];
      if (d < -0.999) { found = true; break; }
    }
    if (found) closed++;
  }
  const M2 = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
  for (const p of P) for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) M2[a][b] += p[a]*p[b];
  const tr = (M2[0][0]+M2[1][1]+M2[2][2]+M2[3][3]) / 4;
  let diff = 0, norm = 0;
  for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) {
    const t = M2[a][b] - (a === b ? tr : 0);
    diff += t*t; norm += M2[a][b]*M2[a][b];
  }
  return { antiFrac: closed / n, dev2: Math.sqrt(diff / norm) };
}

if (typeof module !== 'undefined') module.exports = { Engine, boundaryZ, certificates, mulberry32 };
