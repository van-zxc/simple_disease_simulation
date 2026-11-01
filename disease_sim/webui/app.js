'use strict';
/*
 * 疫情仪表盘前端：只负责「画」，不含任何模型逻辑。
 *
 * 数据来源（disease_sim/server.py 推送）：
 *   1) 文本 JSON {"type":"meta", ...}      连接时一次
 *   2) 二进制 Float32 (x,y,size_norm)*count 场所层（紧随 meta，也只发一次）
 *   3) 文本 JSON {"type":"frame", ...}      每帧
 *   4) 二进制 Float32 xy(2n) + Uint8 status(n)  紧随该帧 JSON
 *
 * 五张曲线图各自维护 500 点窗口并对纵轴做 1-2-5 档位自动缩放（带滞回，避免抖动）。
 */

const WINDOW_POINTS = 500;

// 8 个状态的中文注释（英文全称由服务端 meta.status_labels 给出）
const STATUS_ZH = ['易感', '免疫', '潜伏期', '有症状', '死亡', '康复', '无症状', '住院'];

// 显示用调色板：服务端 COLOR_MAP 的 Dead 是纯黑（深色画布上不可见），这里换成灰
function displayColors(metaColors) {
  const c = (metaColors || []).slice();
  if (c.length > 4) c[4] = '#6e7681';
  return c;
}

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const fmtInt = (v) => Math.round(v).toLocaleString('en-US');

function fmtAxis(v) {
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1e6) return (v / 1e6).toFixed(1) + 'M';
  if (a >= 1e4) return (v / 1e3).toFixed(0) + 'k';
  if (a >= 1e3) return (v / 1e3).toFixed(1) + 'k';
  if (a >= 100) return String(Math.round(v));
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  if (a >= 0.01) return v.toFixed(3);
  return v.toPrecision(2);
}

const S = {
  meta: null,
  map: null,
  venueCanvas: null,
  venues: null,
  pendingHeader: null,
  expectVenue: false,
  last: null,
  frame: -1,
  charts: {},
  kpi: [],
  dirty: false,
  displayColors: ['#B0B0B0', '#00BFFF', '#FFD700', '#FF4500', '#6e7681', '#32CD32', '#FF8C00', '#8A2BE2'],
};

const mapCanvas = document.getElementById('map');
const mapCtx = mapCanvas.getContext('2d');
const buckets = Array.from({ length: 8 }, () => []);

/* ------------------------------------------------------------------ 曲线 */

function niceCeil(v) {
  if (!(v > 0)) return 1;
  const e = Math.floor(Math.log10(v));
  const f = v / Math.pow(10, e);
  const nf = f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10;
  return nf * Math.pow(10, e);
}

/** 自动缩放：变大立刻跟上（曲线始终可见），变小要跌破 62% 才回落（滞回，刻度不抖）。 */
function autoscale(cur, maxVal) {
  if (!(maxVal > 0)) return cur > 0 ? Math.min(cur, 1) : 1;
  const want = niceCeil(maxVal * 1.08);
  if (want > cur) return want;
  if (want < cur * 0.62) return want;
  return cur;
}

class LineChart {
  constructor(id, cfg) {
    this.cv = document.getElementById(id);
    this.ctx = this.cv.getContext('2d');
    this.cfg = cfg;
    this.hist = {};
    for (const s of cfg.series) this.hist[s.key] = [];
    this.axL = 1;
    this.axR = 1;
    this.w = 0;
    this.h = 0;
    this.dpr = 1;
  }

  push(values) {
    for (const s of this.cfg.series) {
      const arr = this.hist[s.key];
      arr.push(Number(values[s.key]) || 0);
      if (arr.length > WINDOW_POINTS + 2) arr.shift();
    }
  }

  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.cv.getBoundingClientRect();
    const w = Math.max(120, Math.round(rect.width));
    const h = Math.max(80, Math.round(rect.height));
    if (w !== this.w || h !== this.h) {
      this.w = w;
      this.h = h;
      this.cv.width = Math.round(w * dpr);
      this.cv.height = Math.round(h * dpr);
    }
    this.dpr = dpr;
  }

  draw() {
    this._resize();
    const ctx = this.ctx;
    const cfg = this.cfg;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);

    const hasR = cfg.series.some((s) => s.axis === 'right');
    const m = { l: 44, r: hasR ? 46 : 12, t: 33, b: 17 };
    const plotW = Math.max(20, this.w - m.l - m.r);
    const plotH = Math.max(20, this.h - m.t - m.b);
    const x0 = m.l;
    const y0 = m.t;

    // 纵轴范围（取窗口内当前序列最大值）
    let maxL = 0;
    let maxR = 0;
    for (const s of cfg.series) {
      const arr = this.hist[s.key];
      let mx = 0;
      for (let i = 0; i < arr.length; i++) if (arr[i] > mx) mx = arr[i];
      if (s.axis === 'right') maxR = Math.max(maxR, mx);
      else maxL = Math.max(maxL, mx);
    }
    this.axL = autoscale(this.axL, maxL);
    if (hasR) this.axR = autoscale(this.axR, maxR);
    const fmtL = cfg.fmtL || fmtAxis;
    const fmtR = cfg.fmtR || fmtAxis;

    // 网格 + 左右刻度
    const ticks = 4;
    ctx.font = '10px "Segoe UI", system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    ctx.lineWidth = 1;
    for (let i = 0; i <= ticks; i++) {
      const y = Math.round(y0 + (plotH * i) / ticks) + 0.5;
      ctx.strokeStyle = i === ticks ? '#2b3a4b' : '#1a2432';
      ctx.beginPath();
      ctx.moveTo(x0, y);
      ctx.lineTo(x0 + plotW, y);
      ctx.stroke();
      ctx.textAlign = 'right';
      ctx.fillStyle = cfg.leftColor || '#8595a8';
      ctx.fillText(fmtL(this.axL * (1 - i / ticks)), x0 - 6, y);
      if (hasR) {
        ctx.textAlign = 'left';
        ctx.fillStyle = cfg.rightColor || '#8595a8';
        ctx.fillText(fmtR(this.axR * (1 - i / ticks)), x0 + plotW + 6, y);
      }
    }

    // 折线（最新帧贴右边缘，数据不足 500 点时从右侧向左生长）
    const stepX = plotW / (WINDOW_POINTS - 1);
    for (const s of cfg.series) {
      const arr = this.hist[s.key];
      const n = arr.length;
      if (n < 2) continue;
      const axMax = s.axis === 'right' ? this.axR : this.axL;
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.7;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const x = x0 + plotW - (n - 1 - i) * stepX;
        const y = y0 + plotH - clamp(arr[i] / axMax, 0, 1) * plotH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // 横轴提示
    ctx.fillStyle = '#6f8093';
    ctx.textAlign = 'left';
    ctx.fillText('-' + WINDOW_POINTS + ' 帧', x0, this.h - 6);
    ctx.textAlign = 'right';
    ctx.fillText('现在', x0 + plotW, this.h - 6);

    // 图例（含实时数值）
    const fmtVal = cfg.fmtVal || fmtAxis;
    let lx = x0;
    const ly = 13;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    for (const s of cfg.series) {
      const arr = this.hist[s.key];
      const val = arr.length ? arr[arr.length - 1] : 0;
      const text = s.label + ' ' + fmtVal(val);
      const tw = ctx.measureText(text).width;
      if (lx + tw + 16 > this.w - 4) break;
      ctx.fillStyle = s.color;
      ctx.fillRect(lx, ly - 1.5, 11, 3);
      ctx.fillStyle = '#c3d1df';
      ctx.fillText(text, lx + 15, ly);
      lx += tw + 30;
    }
  }
}

/* ------------------------------------------------------------------ 地图 */

function worldX(x, m) {
  return m.ox + x * m.scale;
}
function worldY(y, m) {
  return m.oy + (m.H - y) * m.scale;
}

function drawShape(c, shape, x, y, size) {
  const h = size / 2;
  if (shape === 'square') {
    c.fillRect(x - h, y - h, size, size);
    c.strokeRect(x - h, y - h, size, size);
    return;
  }
  c.beginPath();
  if (shape === 'triangle') {
    c.moveTo(x, y - h * 1.15);
    c.lineTo(x + h, y + h * 0.85);
    c.lineTo(x - h, y + h * 0.85);
    c.closePath();
  } else if (shape === 'circle') {
    c.arc(x, y, h * 1.05, 0, Math.PI * 2);
  } else {
    // cross（公园）
    const t = Math.max(0.8, size * 0.24);
    c.moveTo(x - t, y - h); c.lineTo(x + t, y - h); c.lineTo(x + t, y - t);
    c.lineTo(x + h, y - t); c.lineTo(x + h, y + t); c.lineTo(x + t, y + t);
    c.lineTo(x + t, y + h); c.lineTo(x - t, y + h); c.lineTo(x - t, y + t);
    c.lineTo(x - h, y + t); c.lineTo(x - h, y - t); c.lineTo(x - t, y - t);
    c.closePath();
  }
  c.fill();
  c.stroke();
}

function setupMapSize() {
  if (!S.meta) return;
  const rect = mapCanvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(200, Math.round(rect.width));
  const h = Math.max(200, Math.round(rect.height));
  mapCanvas.width = Math.round(w * dpr);
  mapCanvas.height = Math.round(h * dpr);
  const W = S.meta.world[0];
  const H = S.meta.world[1];
  const pad = 14;
  const scale = Math.min((w - pad * 2) / W, (h - pad * 2) / H);
  S.map = {
    w, h, dpr, W, H, scale,
    ox: (w - W * scale) / 2,
    oy: (h - H * scale) / 2,
  };
  buildVenueLayer();
}

/** 场所是静态层：画到离屏 canvas 一次，之后每帧只 blit，省掉上万次图元绘制。 */
function buildVenueLayer() {
  const m = S.map;
  if (!m || !S.venues) return;
  const off = document.createElement('canvas');
  off.width = Math.round(m.w * m.dpr);
  off.height = Math.round(m.h * m.dpr);
  const c = off.getContext('2d');
  c.setTransform(m.dpr, 0, 0, m.dpr, 0, 0);
  const coords = S.venues.coords;
  const [minPx, maxPx] = S.venues.sizePx;
  c.globalAlpha = 0.72;
  c.strokeStyle = 'rgba(255,255,255,.26)';
  c.lineWidth = 0.6;
  for (const layer of S.venues.layers) {
    c.fillStyle = layer.color;
    for (let i = 0; i < layer.count; i++) {
      const k = (layer.start + i) * 3;
      const x = worldX(coords[k], m);
      const y = worldY(coords[k + 1], m);
      const size = clamp(2.0 + 6.4 * coords[k + 2], minPx, maxPx);
      drawShape(c, layer.shape, x, y, size);
    }
  }
  S.venueCanvas = off;
}

function drawAgents() {
  const m = S.map;
  const last = S.last;
  if (!m || !last) return;
  const xy = last.xy;
  const st = last.st;
  const n = last.n;
  for (let s = 0; s < 8; s++) buckets[s].length = 0;
  for (let i = 0; i < n; i++) {
    const s = st[i];
    if (s < 8) buckets[s].push(xy[i * 2], xy[i * 2 + 1]);
  }
  const px = clamp(m.scale / 20, 1.3, 3.4);
  const half = px / 2;
  for (let s = 0; s < 8; s++) {
    const b = buckets[s];
    if (!b.length) continue;
    mapCtx.fillStyle = S.displayColors[s] || '#ffffff';
    mapCtx.beginPath();
    for (let i = 0; i < b.length; i += 2) {
      mapCtx.rect(worldX(b[i], m) - half, worldY(b[i + 1], m) - half, px, px);
    }
    mapCtx.fill();
  }
}

function drawMap() {
  const m = S.map;
  if (!m) return;
  mapCtx.setTransform(m.dpr, 0, 0, m.dpr, 0, 0);
  mapCtx.clearRect(0, 0, m.w, m.h);
  mapCtx.fillStyle = '#0d141c';
  mapCtx.fillRect(0, 0, m.w, m.h);
  // 城区边界（世界范围）
  mapCtx.strokeStyle = '#1e2b3a';
  mapCtx.lineWidth = 1;
  mapCtx.strokeRect(m.ox, m.oy, m.W * m.scale, m.H * m.scale);
  if (S.venueCanvas) {
    mapCtx.globalAlpha = 1;
    mapCtx.drawImage(S.venueCanvas, 0, 0, m.w, m.h);
  }
  drawAgents();
}

/* ------------------------------------------------------------------ 读数块 */

function kpiSpecs(meta) {
  const beds = meta.beds || 0;
  const specs = [
    { id: 'active', en: 'Active infections', zh: '活跃感染', sub: 'E + A + I + H',
      get: (h) => ({ v: fmtInt(h.total_infected) }) },
    { id: 'incubating', en: 'Incubating', zh: '潜伏期', sub: 'status = E',
      get: (h) => ({ v: fmtInt(h.counts[2]) }) },
    { id: 'asymptomatic', en: 'Asymptomatic', zh: '无症状', sub: 'status = A',
      get: (h) => ({ v: fmtInt(h.counts[6]) }) },
    { id: 'symptomatic', en: 'Symptomatic', zh: '有症状', sub: 'status = I',
      get: (h) => ({ v: fmtInt(h.counts[3]) }) },
    { id: 'hospitalized', en: 'Hospitalized', zh: '住院', sub: 'status = H',
      get: (h) => ({ v: fmtInt(h.counts[7]), u: '/' + beds }) },
    { id: 'cumulative', en: 'Cumulative infected', zh: '累计感染', sub: 'N − S − 初始免疫',
      get: (h) => ({ v: fmtInt(meta.population - h.counts[0] - h.counts[1]) }) },
    { id: 'susceptible', en: 'Susceptible', zh: '易感', sub: 'status = S',
      get: (h) => ({ v: fmtInt(h.counts[0]) }) },
    { id: 'recovered', en: 'Recovered', zh: '康复', sub: 'status = R',
      get: (h) => ({ v: fmtInt(h.counts[5]) }) },
    { id: 'dead', en: 'Dead', zh: '死亡', sub: 'status = D',
      get: (h) => ({ v: fmtInt(h.counts[4]) }) },
    { id: 'immune', en: 'Immune (initial)', zh: '初始免疫', sub: 'status = 免疫',
      get: (h) => ({ v: fmtInt(h.counts[1]) }) },
    { id: 'beds', en: 'Bed occupancy', zh: '床位占用', sub: '住院 / 床位',
      get: (h) => {
        const pct = beds > 0 ? (100 * h.counts[7]) / beds : 0;
        const cls = pct >= 100 ? 'bad' : pct >= 70 ? 'warn' : '';
        return { v: pct.toFixed(0), u: '%', cls };
      } },
    { id: 'medical', en: 'Medical status', zh: '医疗状态', sub: '床位是否击穿',
      get: (h) => ({ v: h.collapsed ? 'COLLAPSE' : 'OK', cls: h.collapsed ? 'bad' : 'ok' }) },
    { id: 'unmet', en: 'Unmet medical need', zh: '未满足医疗', sub: '等床 / 未收治',
      get: (h) => ({ v: fmtInt(h.untreated), cls: h.untreated > 0 ? 'bad' : '' }) },
    { id: 'immunity', en: 'Average immunity', zh: '平均免疫', sub: '存活者平均',
      get: (h) => ({ v: Number(h.immune_avg).toFixed(3) }) },
    { id: 'reinfection', en: 'Reinfection rate', zh: '再感染率', sub: '最近 50 帧',
      get: (h) => ({ v: Number(h.reinfection_pct).toFixed(1), u: '%' }) },
    { id: 'beta', en: 'Transmission β', zh: '传播率', sub: '感染者平均值',
      get: (h) => ({ v: fmtBeta(h.virus.beta) }) },
    { id: 'mu', en: 'Death rate μ', zh: '死亡率', sub: '每帧',
      get: (h) => ({ v: fmtMu(h.virus.mu) }) },
    { id: 'gen', en: 'Mean generation', zh: '平均代际', sub: '病毒传了几代',
      get: (h) => ({ v: Number(h.virus.generation).toFixed(2) }) },
    { id: 'lockdown', en: 'Lockdown', zh: '封控', sub: '接触乘子 0.15',
      get: (h) => ({ v: h.lockdown ? 'ON' : 'OFF', cls: h.lockdown ? 'warn' : '' }) },
    { id: 'home', en: 'At home', zh: '在家', sub: 'on site',
      get: (h) => ({ v: fmtInt(h.on_site.home) }) },
    { id: 'work', en: 'At work', zh: '在岗', sub: 'on site',
      get: (h) => ({ v: fmtInt(h.on_site.work) }) },
    { id: 'school', en: 'At school', zh: '在校', sub: 'on site',
      get: (h) => ({ v: fmtInt(h.on_site.school) }) },
    { id: 'leisure', en: 'At leisure', zh: '休闲', sub: 'on site',
      get: (h) => ({ v: fmtInt(h.on_site.leisure) }) },
    { id: 'travel', en: 'In transit', zh: '途中', sub: 'on site',
      get: (h) => ({ v: fmtInt(h.on_site.travel) }) },
  ];
  if (meta.interventions) {
    specs.push(
      { id: 'quarantined', en: 'Quarantined', zh: '密接隔离', sub: 'v3 干预',
        get: (h) => ({ v: fmtInt(h.interventions ? h.interventions.quarantined : 0) }) },
      { id: 'isolated', en: 'Isolated', zh: '自然隔离', sub: 'v3 干预',
        get: (h) => ({ v: fmtInt(h.interventions ? h.interventions.isolated : 0) }) },
      { id: 'testPending', en: 'Test pending', zh: '待检测', sub: 'v3 干预',
        get: (h) => ({ v: fmtInt(h.interventions ? h.interventions.test_pending : 0) }) },
      { id: 'tracePending', en: 'Trace pending', zh: '待追踪', sub: 'v3 干预',
        get: (h) => ({ v: fmtInt(h.interventions ? h.interventions.trace_pending : 0) }) },
      { id: 'contactMult', en: 'Contact multiplier', zh: '接触乘子', sub: '行为反馈',
        get: (h) => ({ v: (h.interventions ? h.interventions.contact_mult : 1).toFixed(2) }) },
    );
  }
  return specs;
}

function fmtBeta(v) {
  const x = Number(v) || 0;
  return x >= 0.1 ? x.toFixed(3) : x.toFixed(5);
}
function fmtMu(v) {
  const x = Number(v) || 0;
  if (x === 0) return '0';
  if (x >= 1e-4) return x.toFixed(5);
  return x.toExponential(1);
}

function buildKpi(meta) {
  const grid = document.getElementById('kpiGrid');
  grid.textContent = '';
  const cards = [];
  for (const spec of kpiSpecs(meta)) {
    const el = document.createElement('div');
    el.className = 'kpi';

    const k = document.createElement('div');
    k.className = 'k';
    const b = document.createElement('b');
    b.textContent = spec.en;
    k.appendChild(b);
    k.appendChild(document.createTextNode(' ' + spec.zh));

    const v = document.createElement('div');
    v.className = 'v';
    v.textContent = '—';

    const sub = document.createElement('div');
    sub.className = 'sub';
    sub.textContent = spec.sub || '';

    el.appendChild(k);
    el.appendChild(v);
    el.appendChild(sub);
    grid.appendChild(el);
    cards.push({ el, v, spec });
  }
  return cards;
}

function updateKpi(h) {
  for (const card of S.kpi) {
    const r = card.spec.get(h);
    card.v.textContent = r.v;
    if (r.u) {
      const u = document.createElement('span');
      u.className = 'u';
      u.textContent = r.u;
      card.v.appendChild(u);
    }
    card.el.className = 'kpi' + (r.cls ? ' ' + r.cls : '');
  }
}

/* ------------------------------------------------------------------ 覆盖文字 */

const timeLine = document.getElementById('timeLine');
const mapStats = document.getElementById('mapStats');

function updateOverlay(h) {
  const total = S.meta ? S.meta.frames : 0;
  const lock = h.lockdown ? '<span class="lock-on"> · 封控中 LOCKDOWN</span>' : '';
  timeLine.innerHTML = 'Day ' + h.day + ' · ' + h.clock + ' · t=' + (S.frame + 1) + '/' + total + lock;

  const cumulative = S.meta.population - h.counts[0] - h.counts[1];
  const lines = [];
  lines.push('活跃感染 Infected <b>' + fmtInt(h.total_infected) + '</b>'
    + '（潜伏 E ' + fmtInt(h.counts[2]) + ' · 无症状 A ' + fmtInt(h.counts[6])
    + ' · 有症状 I ' + fmtInt(h.counts[3]) + ' · 住院 H ' + fmtInt(h.counts[7]) + '）');
  lines.push('累计 Cumulative ' + fmtInt(cumulative)
    + ' · 易感 S ' + fmtInt(h.counts[0])
    + ' · 康复 R ' + fmtInt(h.counts[5])
    + ' · 死亡 D ' + fmtInt(h.counts[4]));
  const med = h.collapsed ? '<span class="bad">COLLAPSE 医疗击穿</span>' : 'OK';
  lines.push('医疗 Medical: ' + med
    + ' · 住院 ' + fmtInt(h.counts[7]) + '/' + fmtInt(h.beds)
    + ' · 未满足 ' + fmtInt(h.untreated)
    + ' | 免疫 Immunity ' + Number(h.immune_avg).toFixed(3)
    + ' · 再感染 ' + Number(h.reinfection_pct).toFixed(1) + '%');
  if (h.interventions) {
    const iv = h.interventions;
    lines.push('干预 v3: 密接隔离 ' + fmtInt(iv.quarantined)
      + ' · 自然隔离 ' + fmtInt(iv.isolated)
      + ' · 待检测 ' + fmtInt(iv.test_pending)
      + ' · 待追踪 ' + fmtInt(iv.trace_pending)
      + ' · 接触乘子 ' + Number(iv.contact_mult).toFixed(2));
  }
  mapStats.innerHTML = lines.join('<br>');
}

/* ------------------------------------------------------------------ 图例 */

function buildLegend(meta) {
  const box = document.getElementById('legend');
  box.textContent = '';
  const venueRow = document.createElement('div');
  venueRow.className = 'row';
  const layers = (meta.venue_layers || []);
  for (const layer of layers) {
    const it = document.createElement('span');
    it.className = 'it';
    const sw = document.createElement('span');
    sw.className = 'sw' + (layer.shape === 'cross' ? ' cross' : '');
    sw.style.background = layer.color;
    it.appendChild(sw);
    it.appendChild(document.createTextNode(layer.label));
    venueRow.appendChild(it);
  }
  const title1 = document.createElement('div');
  title1.className = 'title';
  title1.textContent = '场所 Venues';
  box.appendChild(title1);
  box.appendChild(venueRow);

  const title2 = document.createElement('div');
  title2.className = 'title';
  title2.textContent = '人群 Agents';
  box.appendChild(title2);
  const agentRow = document.createElement('div');
  agentRow.className = 'row';
  const labels = meta.status_labels || [];
  for (let s = 0; s < 8; s++) {
    const it = document.createElement('span');
    it.className = 'it';
    const sw = document.createElement('span');
    sw.className = 'sw';
    sw.style.background = S.displayColors[s];
    it.appendChild(sw);
    it.appendChild(document.createTextNode((labels[s] || s) + ' ' + STATUS_ZH[s]));
    agentRow.appendChild(it);
  }
  box.appendChild(agentRow);
}

/* ------------------------------------------------------------------ WebSocket */

const connEl = document.getElementById('conn');
const profileEl = document.getElementById('profile');
const lockBtn = document.getElementById('lockBtn');

function setConn(text, cls) {
  connEl.textContent = text;
  connEl.className = cls || '';
}

function onMeta(meta) {
  S.meta = meta;
  S.displayColors = displayColors(meta.colors);
  profileEl.textContent = '疾病档案 ' + meta.profile + ' · 人口 ' + fmtInt(meta.population)
    + ' · 世界 ' + meta.world[0].toFixed(1) + '×' + meta.world[1].toFixed(1)
    + ' · 床位 ' + fmtInt(meta.beds) + ' · 种子 ' + meta.seed
    + (meta.sample ? ' · 渲染抽样 ' + fmtInt(meta.render_points) + ' 点' : '');
  S.expectVenue = true;
  S.charts.active = new LineChart('chartActive', {
    leftColor: '#FFD700',
    series: [
      { key: 'E', label: '潜伏 E', color: '#FFD700', axis: 'left' },
      { key: 'A', label: '无症状 A', color: '#FF8C00', axis: 'left' },
      { key: 'I', label: '有症状 I', color: '#FF4500', axis: 'left' },
      { key: 'H', label: '住院 H', color: '#B39DFF', axis: 'left' },
    ],
    fmtVal: fmtInt,
  });
  S.charts.pools = new LineChart('chartPools', {
    leftColor: '#B0B0B0',
    series: [
      { key: 'S', label: '易感 S', color: '#B0B0B0', axis: 'left' },
      { key: 'R', label: '康复 R', color: '#32CD32', axis: 'left' },
      { key: 'Imm', label: '免疫', color: '#00BFFF', axis: 'left' },
      { key: 'D', label: '死亡 D', color: '#ff7b72', axis: 'left' },
    ],
    fmtVal: fmtInt,
  });
  S.charts.virus = new LineChart('chartVirus', {
    leftColor: '#ff7b72',
    rightColor: '#00BFFF',
    series: [
      { key: 'beta', label: 'β', color: '#ff7b72', axis: 'left' },
      { key: 'imm', label: '平均免疫', color: '#00BFFF', axis: 'right' },
    ],
    fmtVal: (v) => (Math.abs(v) >= 0.1 ? v.toFixed(3) : v.toFixed(4)),
  });
  S.charts.death = new LineChart('chartDeath', {
    leftColor: '#e3b341',
    rightColor: '#B39DFF',
    series: [
      { key: 'mu', label: 'μ ‰', color: '#e3b341', axis: 'left' },
      { key: 'gen', label: '代际', color: '#B39DFF', axis: 'right' },
    ],
    fmtVal: (v) => (Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3)),
  });
  S.charts.reinf = new LineChart('chartReinf', {
    leftColor: '#ff9c96',
    series: [{ key: 'reinf', label: '再感染 %', color: '#ff9c96', axis: 'left' }],
    fmtVal: (v) => v.toFixed(1) + '%',
  });
  S.kpi = buildKpi(meta);
  buildLegend(meta);
  setupMapSize();
  setConn('已连接', 'ok');
  schedule();
}

function decodeVenues(buffer) {
  const coords = new Float32Array(buffer);
  const layers = (S.meta && S.meta.venue_layers) || [];
  S.venues = {
    coords,
    layers,
    sizePx: (S.meta && S.meta.venue_size_px) || [1.6, 26],
  };
}

function onMessage(ev) {
  if (typeof ev.data === 'string') {
    let msg;
    try {
      msg = JSON.parse(ev.data);
    } catch (err) {
      console.error('bad json', err);
      return;
    }
    if (msg.type === 'meta') onMeta(msg);
    else if (msg.type === 'frame') S.pendingHeader = msg;
    return;
  }
  if (S.expectVenue) {
    S.expectVenue = false;
    decodeVenues(ev.data);
    buildVenueLayer();
    schedule();
    return;
  }
  const hdr = S.pendingHeader;
  S.pendingHeader = null;
  if (!hdr) return;
  const n = hdr.n;
  const xy = new Float32Array(ev.data, 0, n * 2);
  const st = new Uint8Array(ev.data, n * 8, n);
  S.frame = hdr.t;
  S.last = { xy, st, n };

  const c = hdr.counts;
  S.charts.active.push({ E: c[2], A: c[6], I: c[3], H: c[7] });
  S.charts.pools.push({ S: c[0], R: c[5], Imm: c[1], D: c[4] });
  S.charts.virus.push({ beta: hdr.virus.beta, imm: hdr.virus.immunity });
  S.charts.death.push({ mu: hdr.virus.mu * 1000, gen: hdr.virus.generation });
  S.charts.reinf.push({ reinf: hdr.reinfection_pct });

  S.lockdown = !!hdr.lockdown;
  lockBtn.textContent = S.lockdown ? 'RESUME 解封' : 'LOCKDOWN 封控';
  lockBtn.className = S.lockdown ? 'on' : '';
  updateKpi(hdr);
  updateOverlay(hdr);
  schedule();
}

function schedule() {
  if (S.dirty) return;
  S.dirty = true;
  requestAnimationFrame(() => {
    S.dirty = false;
    if (S.meta) {
      drawMap();
      for (const key of Object.keys(S.charts)) S.charts[key].draw();
    }
  });
}

let ws = null;
let retryTimer = 0;

function connect() {
  const proto = location.protocol === 'https:' ? 'wss://' : 'ws://';
  setConn('连接中…', '');
  try {
    ws = new WebSocket(proto + location.host + '/ws');
  } catch (err) {
    scheduleReconnect();
    return;
  }
  ws.binaryType = 'arraybuffer';
  ws.onmessage = onMessage;
  ws.onclose = () => {
    setConn('已断开', 'bad');
    scheduleReconnect();
  };
  ws.onerror = () => {
    setConn('连接错误', 'bad');
  };
}

function scheduleReconnect() {
  if (retryTimer) return;
  retryTimer = window.setTimeout(() => {
    retryTimer = 0;
    connect();
  }, 1500);
}

lockBtn.addEventListener('click', async () => {
  try {
    const res = await fetch('/api/toggle', { method: 'POST' });
    const data = await res.json();
    S.lockdown = !!data.lockdown;
    lockBtn.textContent = S.lockdown ? 'RESUME 解封' : 'LOCKDOWN 封控';
    lockBtn.className = S.lockdown ? 'on' : '';
  } catch (err) {
    console.error('toggle failed', err);
  }
});

window.addEventListener('resize', () => {
  setupMapSize();
  schedule();
});
if (typeof ResizeObserver !== 'undefined') {
  const ro = new ResizeObserver(() => {
    setupMapSize();
    schedule();
  });
  ro.observe(document.getElementById('mapPanel'));
}

connect();
