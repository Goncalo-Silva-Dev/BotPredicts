// Resultados e percentagens ao vivo. O resultado vem da ESPN (API pública) a cada 30 segundos,
// só para os jogos a decorrer; as percentagens são recalculadas aqui, no browser, com os números
// de antes do jogo (atributo data-m de cada jogo) e o resultado e o tempo atuais.
// A mesma matemática está em Python em aovivo_modelo.py (testada com jogos passados).
const LV = __LV__;

const Live = (() => {
  // Função de distribuição da normal (erro < 1e-7)
  function ncdf(x) {
    const t = 1 / (1 + 0.2316419 * Math.abs(x));
    const d = 0.3989422804014327 * Math.exp(-x * x / 2);
    const p = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    return x >= 0 ? 1 - p : p;
  }

  // ---------- Futebol ----------
  const N = 12;
  function soccerSteps(half, minute, stop) {
    const S = LV.soccer, prof = S.profile, s1 = S.stop1, s2 = S.stop2;
    const sum = a => a.reduce((x, y) => x + y, 0);
    const c1 = prof[45] / sum(s1), c2 = prof[91] / sum(s2);
    const stoppage = (surv, c, done) => {
      const base = done === 0 ? 1 : surv[Math.min(done, surv.length) - 1];
      const out = [];
      for (let u = done; u < surv.length; u++) out.push(c * surv[u] / base);
      return out;
    };
    let w = [];
    if (half === 1) {
      if (stop === null) {
        for (let m = minute + 1; m <= 45; m++) w.push(prof[m - 1]);
        w = w.concat(stoppage(s1, c1, 0));
      } else w = w.concat(stoppage(s1, c1, stop));
      for (let m = 46; m <= 90; m++) w.push(prof[m]);
      w = w.concat(stoppage(s2, c2, 0));
    } else if (stop === null) {
      for (let m = Math.max(minute + 1, 46); m <= 90; m++) w.push(prof[m]);
      w = w.concat(stoppage(s2, c2, 0));
    } else w = w.concat(stoppage(s2, c2, stop));
    return w;
  }
  function soccerMult(gh, ga, rh, ra) {
    const S = LV.soccer, clip = (x, a, b) => Math.max(a, Math.min(b, x));
    const redH = S.red[clip(ra - rh, -1, 1) + 1], redA = S.red[clip(rh - ra, -1, 1) + 1];
    const mh = new Float64Array(N * N), ma = new Float64Array(N * N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const d = clip(gh + i - ga - j, -2, 2);
      mh[i * N + j] = S.state[d + 2] * redH;
      ma[i * N + j] = S.state[-d + 2] * redA;
    }
    return [mh, ma];
  }
  function soccerAdded(lh, la, gh, ga, rh, ra, steps) {
    const [mh, ma] = soccerMult(gh, ga, rh, ra);
    let p = new Float64Array(N * N);
    p[0] = 1;
    for (const w of steps) {
      const nx = new Float64Array(N * N);
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        const v = p[i * N + j];
        if (v === 0) continue;
        const ph = 1 - Math.exp(-lh * w * mh[i * N + j]), pa = 1 - Math.exp(-la * w * ma[i * N + j]);
        nx[i * N + j] += v * (1 - ph) * (1 - pa);
        if (i + 1 < N) nx[(i + 1) * N + j] += v * ph * (1 - pa);
        if (j + 1 < N) nx[i * N + j + 1] += v * (1 - ph) * pa;
        if (i + 1 < N && j + 1 < N) nx[(i + 1) * N + j + 1] += v * ph * pa;
      }
      p = nx;
    }
    return p;
  }
  function soccer(par, st) {
    const [lh, la] = par.l, {gh, ga, rh, ra} = st;
    const p = soccerAdded(lh, la, gh, ga, rh, ra, soccerSteps(st.half, st.minute, st.stop));
    const out = {casa: 0, empate: 0, fora: 0, ambas: 0, mais: {}, top: []};
    const lines = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5];
    lines.forEach(l => out.mais[l] = 0);
    const scores = [];
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const v = p[i * N + j], h = gh + i, a = ga + j;
      if (h > a) out.casa += v; else if (h === a) out.empate += v; else out.fora += v;
      if (h > 0 && a > 0) out.ambas += v;
      lines.forEach(l => { if (h + a > l) out.mais[l] += v; });
      scores.push([`${h}-${a}`, v]);
    }
    out.top = scores.sort((x, y) => y[1] - x[1]).slice(0, 3);
    // próximo golo: sem mais golos = p[0]; o resto divide-se pela taxa de cada equipa agora
    const [mh, ma] = soccerMult(gh, ga, rh, ra), rH = lh * mh[0], rA = la * ma[0];
    out.next = {casa: (1 - p[0]) * rH / (rH + rA), nenhum: p[0], fora: (1 - p[0]) * rA / (rH + rA)};
    return out;
  }

  // ---------- Basquetebol ----------
  function basketCoefs(tau) {
    tau = Math.min(Math.max(tau, 0), 1);
    const keys = Object.keys(LV.basket).map(Number).sort((a, b) => a - b);
    const hi = keys.find(k => k >= tau), lo = keys[Math.max(keys.indexOf(hi) - 1, 0)];
    const f = hi === lo ? 0 : (tau - lo) / (hi - lo), a0 = LV.basket[String(lo)] || LV.basket[lo.toFixed(1)],
      a1 = LV.basket[String(hi)] || LV.basket[hi.toFixed(1)];
    const lin = i => a0[i] + f * (a1[i] - a0[i]);
    const vr = i => Math.sqrt(a0[i] ** 2 + f * (a1[i] ** 2 - a0[i] ** 2));
    return [lin(0), lin(1), vr(2), lin(3), lin(4), vr(5)];
  }
  function basket(par, st) {
    const [a, b, s, c, d, u] = basketCoefs(st.tau);
    const mu = st.lead + a * par.m + b * st.lead, sd = Math.max(s * par.sm, 0.05);
    const pOt = ncdf(par.m * Math.sqrt(5 / 48) / par.sm);
    const over = 1 - ncdf((0.5 - mu) / sd), tie = ncdf((0.5 - mu) / sd) - ncdf((-0.5 - mu) / sd);
    const tot = st.points + c * par.t + d * (st.points - (1 - st.tau) * par.t), sdt = Math.max(u * par.st, 0.05);
    return {casa: over + tie * pOt, margem: mu, sdMargem: sd, total: tot, sdTotal: sdt,
            totalOver: l => 1 - ncdf((l - tot) / sdt),
            // "casa -L" ganha se a diferença final passar L
            hcp: l => 1 - ncdf((l - mu) / sd)};
  }

  // ---------- Hóquei ----------
  function pois(l, n) {
    const out = [Math.exp(-l)];
    for (let k = 1; k < n; k++) out.push(out[k - 1] * l / k);
    return out;
  }
  function hockeyShare(side, period, left) {
    const s = LV.hockey[side], sh = [s[0], s[1], 1 - s[0] - s[1]];
    if (period > 3) return 0;
    let r = sh[period - 1] * left / 20;
    for (let k = period; k < 3; k++) r += sh[k];
    return r;
  }
  function hockey(par, st) {
    const H = LV.hockey, n = 14, {gh, ga, period, left} = st;
    const ph = pois(par.l[0] * hockeyShare("home", period, left), n);
    const pa = pois(par.l[1] * hockeyShare("away", period, left), n);
    const rest = period <= 3 ? (3 - period) * 20 + left : 0, k = Math.min(1, rest / 3);
    const [p1, q1, p2, q2] = H.kernel.map(x => x * k);
    const M = n + 1, out = new Float64Array(M * M);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const v = ph[i] * pa[j], d = gh + i - ga - j;
      if (Math.abs(d) !== 1 && Math.abs(d) !== 2) { out[i * M + j] += v; continue; }
      const [lead, trail] = Math.abs(d) === 1 ? [p1, q1] : [p2, q2];
      if (d > 0) { out[(i + 1) * M + j] += v * lead; out[i * M + j + 1] += v * trail; }
      else { out[i * M + j + 1] += v * lead; out[(i + 1) * M + j] += v * trail; }
      out[i * M + j] += v * (1 - lead - trail);
    }
    const r = {reg_casa: 0, reg_empate: 0, reg_fora: 0, over: {}};
    const lines = par.tl || [];
    lines.forEach(l => r.over[l] = 0);
    for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) {
      const v = out[i * M + j], h = gh + i, a = ga + j;
      if (h > a) r.reg_casa += v; else if (h === a) r.reg_empate += v; else r.reg_fora += v;
      lines.forEach(l => { if (h + a + (h === a ? 1 : 0) > l) r.over[l] += v; });  // empate: +1 no fim
    }
    r.casa = period > 3 ? H.ot : r.reg_casa + r.reg_empate * H.ot;
    return r;
  }

  // ---------- Ténis ----------
  function gameProb(p) {
    const q = 1 - p, deuce = p * p / (1 - 2 * p * q);
    return p ** 4 * (1 + 4 * q + 10 * q * q) + 20 * p ** 3 * q ** 3 * deuce;
  }
  function tiebreakProb(x, y) {
    const dp = Array.from({length: 8}, () => new Float64Array(8));
    dp[0][0] = 1;
    let win = 0;
    for (let n = 0; n < 12; n++) {
      for (let i = Math.max(0, n - 6); i <= Math.min(n, 6); i++) {
        const j = n - i;
        if (j > 6 || dp[i][j] === 0) continue;
        const p = Math.floor((n + 1) / 2) % 2 === 0 ? x : y;
        for (const [di, pr] of [[1, p], [0, 1 - p]]) {
          const a = i + di, b = j + 1 - di;
          if (a === 7 && b <= 5) win += dp[i][j] * pr;
          else if (b === 7 && a <= 5) { /* perde */ }
          else dp[a][b] += dp[i][j] * pr;
        }
      }
    }
    const both = x * y, none = (1 - x) * (1 - y);
    return win + dp[6][6] * both / (both + none);
  }
  const setDone = (i, j) => (i >= 6 && i - j >= 2) || (j >= 6 && j - i >= 2) || i === 7 || j === 7;
  function setTables(pa, pb) {
    const ga = gameProb(pa), gb = 1 - gameProb(pb), memo = new Map();
    function win(i, j, aServes) {
      if ((i >= 6 && i - j >= 2) || i === 7) return 1;
      if ((j >= 6 && j - i >= 2) || j === 7) return 0;
      const key = i * 100 + j * 2 + (aServes ? 1 : 0);
      if (memo.has(key)) return memo.get(key);
      let r;
      if (i === 6 && j === 6) {
        const t = aServes ? tiebreakProb(pa, 1 - pb) : tiebreakProb(1 - pb, pa);
        r = aServes ? t : 1 - t;
      } else {
        const p = aServes ? ga : gb;
        r = p * win(i + 1, j, !aServes) + (1 - p) * win(i, j + 1, !aServes);
      }
      memo.set(key, r);
      return r;
    }
    function reach(aFirst) {
      const r = Array.from({length: 8}, () => new Float64Array(8));
      r[0][0] = 1;
      for (let n = 0; n <= 12; n++) for (let i = 0; i <= n; i++) {
        const j = n - i;
        if (i > 7 || j > 7 || r[i][j] === 0 || setDone(i, j)) continue;
        if (i === 6 && j === 6) {
          const t = win(6, 6, aFirst);
          r[7][6] += r[6][6] * t; r[6][7] += r[6][6] * (1 - t);
          continue;
        }
        const aServes = n % 2 === 0 ? aFirst : !aFirst, p = aServes ? ga : gb;
        r[i + 1][j] += r[i][j] * p; r[i][j + 1] += r[i][j] * (1 - p);
      }
      return r;
    }
    return [win, reach(true), reach(false)];
  }
  function tennis(par, st) {
    const need = Math.floor(par.n / 2) + 1, done = st.done, [i, j] = st.cur;
    const sa = done.filter(([x, y]) => x > y).length, sb = done.length - sa;
    let nm = 0, ns = 0, den = 0;
    LV.tennis.z.forEach((z, k) => {
      const d = par.d + par.s * z, rnd = x => Math.round(x * 1000) / 1000;
      const pa = rnd(Math.min(Math.max(par.b + d / 2, 0.2), 0.97)), pb = rnd(Math.min(Math.max(par.b - d / 2, 0.2), 0.97));
      const [win, rA, rB] = setTables(pa, pb);
      let lik = 1;
      for (const [x, y] of done) lik *= Math.max(0.5 * ((rA[x] || [])[y] + (rB[x] || [])[y]) || 0, 1e-9);
      let la = (rA[i] || [])[j] || 0, lb = (rB[i] || [])[j] || 0;
      if (la + lb <= 0) { la = 1; lb = 1; }
      lik *= 0.5 * (la + lb);
      const aNow = (i + j) % 2 === 0;
      const pc = (la * win(i, j, aNow) + lb * win(i, j, !aNow)) / (la + lb);
      const ps = 0.5 * (win(0, 0, true) + win(0, 0, false)), memo = {};
      const match = (x, y) => {
        if (x === need) return 1;
        if (y === need) return 0;
        const key = x * 10 + y;
        if (!(key in memo)) memo[key] = ps * match(x + 1, y) + (1 - ps) * match(x, y + 1);
        return memo[key];
      };
      const pm = pc * match(sa + 1, sb) + (1 - pc) * match(sa, sb + 1);
      const w = LV.tennis.w[k];
      nm += w * lik * pm; ns += w * lik * pc; den += w * lik;
    });
    return {casa: nm / den, set: ns / den};
  }

  return {ncdf, soccerSteps, soccer, basket, hockey, tennis, setDone};
})();

if (typeof module !== "undefined") {
  module.exports = Live;  // testes (Node)
} else (() => {
  const cards = [...document.querySelectorAll("[data-espn]")].map(el => {
    const [slug, day, id, start] = el.dataset.espn.split("|");
    let par = null;
    try { par = el.dataset.m ? JSON.parse(el.dataset.m) : null; } catch (e) { par = null; }
    return {el, slug, day, id, start: Date.parse(start), done: false, par};
  });
  if (!cards.length) return;
  const ord = p => `${p}.º`;
  const pct = v => `${Math.round(Math.min(Math.max(v, 0), 1) * 100)}%`;
  const num = (v, k = 1) => v.toFixed(k).replace(".", ",");
  const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
  function status(ev, slug) {
    const s = ev.status, t = s.type, p = s.period || 0, c = s.displayClock || "", n = t.name || "";
    const sport = slug.split("/")[0], d = t.shortDetail || "";
    if (n === "STATUS_POSTPONED") return "Adiado";
    if (/CANCEL/.test(n)) return "Cancelado";
    if (/SUSPENDED|ABANDONED/.test(n)) return "Suspenso";
    if (sport === "tennis") {
      if (/RETIRED/.test(n)) return "Desistência";
      if (/WALKOVER/.test(n)) return "Falta de comparência";
      if (t.state === "post") return "Terminado";
      return t.state === "in" ? `${ord(p || 1)} set` : "";
    }
    if (t.state === "post") {
      if (sport === "soccer") return /PEN/.test(n) ? "Terminado após penáltis" : /AET/.test(n) ? "Terminado após prolongamento" : "Terminado";
      if (sport === "hockey" && /SO/.test(d)) return "Terminado após penáltis";
      return p > (sport === "hockey" ? 3 : 4) ? "Terminado após prolongamento" : "Terminado";
    }
    if (t.state !== "in") return /DELAY/.test(n) ? "Atrasado" : "";
    if (/HALFTIME/.test(n)) return "Intervalo";
    if (sport === "soccer") return /SHOOTOUT/.test(n) ? "Penáltis" : p > 2 ? `Prolongamento ${c}` : c;
    const unit = sport === "hockey" ? "período" : "quarto", last = sport === "hockey" ? 3 : 4;
    if (sport === "hockey" && p > last + 1) return "Penáltis";
    const name = p > last ? "Prolongamento" : `${ord(p)} ${unit}`;
    return /END_PERIOD/.test(n) ? `Fim do ${name.toLowerCase()}` : `${name} · ${c}`;
  }
  const clockMinutes = c => {  // "5:32" ou "45.2" (segundos) -> minutos que faltam no período
    if (!c) return 0;
    if (c.includes(":")) { const [m, s] = c.split(":").map(Number); return m + s / 60; }
    return Number(c) / 60 || 0;
  };
  // ---------- estado do jogo a partir da ESPN ----------
  function soccerState(comp, st) {
    const n = st.type.name || "", p = st.period || 0;
    if (st.type.state !== "in" || p > 2 || /SHOOTOUT/.test(n)) return null;
    const teams = comp.competitors, home = teams.find(x => x.homeAway === "home"), away = teams.find(x => x.homeAway === "away");
    const reds = side => (comp.details || []).filter(x => x.redCard && x.team && x.team.id === side.team.id).length;
    const base = {gh: Number(home.score) || 0, ga: Number(away.score) || 0, rh: reds(home), ra: reds(away)};
    if (/HALFTIME/.test(n)) return {...base, half: 2, minute: 45, stop: null};
    const m = /(\d+)'(?:\+(\d+)')?/.exec(st.displayClock || "");
    if (!m) return null;
    const min = Number(m[1]), extra = m[2] ? Number(m[2]) : null;
    if (p === 1) return extra !== null || min > 45 ? {...base, half: 1, minute: 45, stop: Math.max((extra || min - 45) - 1, 0)}
                                                 : {...base, half: 1, minute: Math.max(min - 1, 0), stop: null};
    return extra !== null || min > 90 ? {...base, half: 2, minute: 90, stop: Math.max((extra || min - 90) - 1, 0)}
                                      : {...base, half: 2, minute: Math.max(min - 1, 45), stop: null};
  }
  function scoreOf(comp) {
    const t = comp.competitors, s = side => Number(t.find(x => x.homeAway === side).score) || 0;
    return [s("home"), s("away")];
  }
  function basketState(comp, st, par) {
    if (st.type.state !== "in") return null;
    const p = st.period || 1, q = par.q, left = clockMinutes(st.displayClock);
    const tau = p <= 4 ? ((4 - p) * q + left) / (4 * q) : left / (4 * q);
    const [h, a] = scoreOf(comp);
    return {lead: h - a, points: h + a, tau};
  }
  function hockeyState(comp, st) {
    if (st.type.state !== "in") return null;
    const [h, a] = scoreOf(comp);
    return {gh: h, ga: a, period: st.period || 1, left: clockMinutes(st.displayClock)};
  }
  function tennisState(comp, par) {
    const players = comp.competitors || [];
    const A = players.find(x => String(x.id) === par.id) || players.find(x => x.order === 1);
    const B = players.find(x => x !== A);
    if (!A || !B) return null;
    const sa = (A.linescores || []).map(x => Number(x.value) || 0), sb = (B.linescores || []).map(x => Number(x.value) || 0);
    const sets = sa.map((x, k) => [x, sb[k] || 0]);
    let done = sets, cur = [0, 0];
    if (sets.length && !Live.setDone(...sets[sets.length - 1])) { done = sets.slice(0, -1); cur = sets[sets.length - 1]; }
    if (done.some(([x, y]) => x > 7 || y > 7) || cur[0] > 7 || cur[1] > 7) return null;
    return {done, cur, sets};
  }
  // ---------- desenho ----------
  const bar = parts => `<div class="bar">${parts.map(([c, v, t]) =>
    `<span class="${c}" style="flex:${Math.max(v, 0.0001).toFixed(4)}" title="${esc(t)}">${pct(v)}</span>`).join("")}</div>`;
  const facts = list => `<div class="facts">${list.map(f => `<span>${f}</span>`).join("")}</div>`;
  function render(card, ev) {
    const box = card.el.querySelector(".lv");
    if (!box || !card.par) return;
    const comp = ev.competitions[0], st = ev.status, par = card.par;
    let html = "";
    try {
      if (par.k === "f") {
        const s = soccerState(comp, st);
        if (s) {
          const r = Live.soccer(par, s), tot = s.gh + s.ga;
          const lines = [tot + 0.5, tot + 1.5, tot + 2.5];
          const f = lines.map(l => `Mais de ${num(l)} golos <b>${pct(r.mais[l] || 0)}</b>`);
          if (!(s.gh > 0 && s.ga > 0)) f.push(`Ambas marcam <b>${pct(r.ambas)}</b>`);
          f.push(`Próximo golo <b>casa ${pct(r.next.casa)} · nenhum ${pct(r.next.nenhum)} · fora ${pct(r.next.fora)}</b>`);
          f.push(`Resultado final <b>${r.top.map(([sc, v]) => `${sc} (${pct(v)})`).join(", ")}</b>`);
          if (s.rh || s.ra) f.push(`Expulsões <b>${s.rh} – ${s.ra}</b>`);
          html = bar([["h", r.casa, "Vitória da casa"], ["d", r.empate, "Empate"], ["a", r.fora, "Vitória de fora"]]) + facts(f);
        }
      } else if (par.k === "b") {
        const s = basketState(comp, st, par);
        if (s) {
          const r = Live.basket(par, s), fav = r.margem >= 0 ? "Casa" : "Fora";
          const f = [`Diferença no fim <b>${fav} por ${num(Math.abs(r.margem))}</b>`,
                     `Total no fim <b>${Math.round(r.total)}</b> (80%: ${Math.round(r.total - 1.2816 * r.sdTotal)} a ${Math.round(r.total + 1.2816 * r.sdTotal)})`];
          // linhas à volta do que se prevê agora (as de antes do jogo deixam de interessar)
          const around = (x, steps) => steps.map(k => Math.floor(x) + 0.5 + k);
          around(r.total, [-5, 0, 5]).forEach(l => f.push(`Mais de ${num(l)} pontos <b>${pct(r.totalOver(l))}</b>`));
          around(r.margem, [-3, 0, 3]).forEach(l => f.push(`Casa ${l > 0 ? "−" : "+"}${num(Math.abs(l))} <b>${pct(r.hcp(l))}</b>`));
          html = bar([["h", r.casa, "Vitória da casa"], ["a", 1 - r.casa, "Vitória de fora"]]) + facts(f);
        }
      } else if (par.k === "h") {
        const s = hockeyState(comp, st);
        if (s) {
          const r = Live.hockey(par, s);
          const f = [];
          if (s.period <= 3) {
            f.push(`60 minutos <b>${pct(r.reg_casa)} / ${pct(r.reg_empate)} / ${pct(r.reg_fora)}</b>`);
            f.push(`Vai a prolongamento <b>${pct(r.reg_empate)}</b>`);
            par.tl.filter(l => r.over[l] > 0.03 && r.over[l] < 0.97).slice(0, 3)
              .forEach(l => f.push(`Mais de ${num(l)} golos <b>${pct(r.over[l])}</b>`));
          } else f.push("Prolongamento ou penáltis: a casa ganha 52% destes jogos (NHL)");
          html = bar([["h", r.casa, "Vitória da casa"], ["a", 1 - r.casa, "Vitória de fora"]]) + facts(f);
        }
      } else if (par.k === "t") {
        const s = st.type.state === "in" ? tennisState(comp, par) : null;
        if (s) {
          const r = Live.tennis(par, s);
          html = bar([["h", r.casa, par.nh || "1.º jogador"], ["a", 1 - r.casa, par.na || "2.º jogador"]]) +
                 facts([`Set atual (${s.cur[0]}-${s.cur[1]}) <b>${esc(par.nh || "")} ${pct(r.set)} · ${esc(par.na || "")} ${pct(1 - r.set)}</b>`]);
        }
      }
    } catch (e) { html = ""; }
    const pre = card.el.querySelector(".lvp");
    if (html) {
      box.innerHTML = `<div class="lvh">Percentagens ao vivo, com o resultado e o tempo de agora</div>${html}`;
      box.hidden = false;
    } else box.hidden = true;
    if (pre) pre.hidden = box.hidden;
  }
  function show(card, ev) {
    const t = ev.status.type, comp = ev.competitions[0];
    const st = card.el.querySelector(".st"), sc = card.el.querySelector(".sc");
    card.done = t.state === "post";
    if (sc && (t.state === "in" || (t.state === "post" && t.completed))) {
      if (card.slug.startsWith("tennis")) {
        const s = card.par ? tennisState(comp, card.par) : null;
        if (s && s.sets.length) sc.textContent = s.sets.map(([x, y]) => `${x}-${y}`).join(" ");
      } else {
        const [h, a] = scoreOf(comp);
        sc.textContent = `${h} – ${a}`;
      }
      sc.classList.add("on");
    }
    if (st) {
      st.textContent = status(ev, card.slug);
      st.className = "st" + (t.state === "in" ? " in" : "");
    }
    render(card, ev);
  }
  function events(json, slug) {
    if (!slug.startsWith("tennis")) return json.events || [];
    // ténis: cada torneio tem vários jogos (competitions) dentro de groupings
    const out = [];
    (json.events || []).forEach(e => (e.groupings || []).forEach(g => (g.competitions || []).forEach(c =>
      out.push({id: c.id, status: c.status, competitions: [c]}))));
    return out;
  }
  async function tick() {
    const now = Date.now();
    const due = cards.filter(c => !c.done && now >= c.start - 15 * 60e3 && now <= c.start + 24 * 3600e3);
    const groups = {};
    due.forEach(c => (groups[`${c.slug}|${c.day}`] ||= []).push(c));
    if (!document.hidden) await Promise.all(Object.entries(groups).map(async ([key, list]) => {
      const [slug, day] = key.split("|");
      try {
        const r = await fetch(`https://site.api.espn.com/apis/site/v2/sports/${slug}/scoreboard?dates=${day}`, {cache: "no-store"});
        const byId = Object.fromEntries(events(await r.json(), slug).map(e => [String(e.id), e]));
        list.forEach(c => byId[c.id] && show(c, byId[c.id]));
      } catch (e) { /* sem ligação: tenta outra vez daqui a 30 segundos */ }
    }));
    setTimeout(tick, 30e3);
  }
  tick();
})();
