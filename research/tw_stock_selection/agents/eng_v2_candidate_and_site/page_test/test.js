// 本機頁面測試（Chromium／Playwright）。用法：node test.js <暫存資料夾> <改之前的 index.html>
// 每個情境：build.py 產生 h.html → 開頁 → 收集錯誤、越界修改、橫向捲動、兩個新區塊的文字 → 截圖存 shots/。
const { chromium } = require('playwright');
const { execFileSync } = require('child_process');
const fs = require('fs'), path = require('path');
const [,, TMP, BEFORE] = process.argv;
const SITE = '/home/user/-/xq_scripts/site';
const D3 = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/harness/d3.min.js';
const SHOTS = path.join(__dirname, 'shots');
const build = (page, out, sc) => execFileSync('python3', ['-I', '-B', path.join(__dirname, 'build.py'), page, path.join(SITE, 'data'), out, D3, JSON.stringify(sc || {})]);
const OLD_IDS = ['top', 'account', 'today', 'calc', 'holdings', 'exit', 'actions', 'trades', 'live', 'risk', 'top20', 'perf', 'yearly', 'howto'];
async function probe(b, file, width, scheme, shot) {
  const p = await b.newPage({ viewport: { width, height: 900 }, colorScheme: scheme });
  await p.goto('file://' + file); await p.waitForTimeout(500);
  const info = await p.evaluate((ids) => {
    const vis = (el) => !el.closest('[hidden]');
    const sec = (id) => { const e = document.getElementById(id); return e ? e.innerText : null; };
    const over = [...document.querySelectorAll('#rules *, #thermo *')].filter((e) => vis(e) && !e.closest('.tw') && e.getBoundingClientRect().right > document.documentElement.clientWidth + 0.5).map((e) => e.tagName + '.' + e.className).slice(0, 5);
    // 標記檢查：新區塊裡每個標了 data-source＋data-field＋（data-row 或 data-where）的元素，畫面上的字要等於那一格（數字比數值、文字比字串）
    const markBad = []; let markN = 0;
    document.querySelectorAll('#rules [data-source][data-field], #thermo [data-source][data-field]').forEach((e) => {
      if (!vis(e) || e.tagName === 'TH') return; const rowAttr = e.getAttribute('data-row'), where = e.getAttribute('data-where'); if (rowAttr === null && where === null) return;
      const d = dash.data(e.getAttribute('data-source')); if (d.status !== 'ok') return; let row = null;
      if (rowAttr !== null) row = d.data[+rowAttr]; else { const [k, v] = where.split('='); row = d.data.find((r) => String(r[decodeURIComponent(k)]) === decodeURIComponent(v)); }
      markN++; const cell = row ? row[e.getAttribute('data-field')] : undefined; const shown = e.textContent.trim();
      const ok = cell === null || cell === undefined ? shown === '—' : (typeof cell === 'number' ? parseFloat(shown) === cell : shown === String(cell));
      if (!ok) markBad.push(e.getAttribute('data-field') + ': 畫面「' + shown + '」資料「' + cell + '」');
    });
    return { markN, markBad, errors: window.__errors, faults: window.__faults, scrollW: document.documentElement.scrollWidth, clientW: document.documentElement.clientWidth,
      rules: sec('rules'), thermo: sec('thermo'), nRules: document.querySelectorAll('#rules ol.rules > li').length, nThermoRows: document.querySelectorAll('#thermo tbody tr').length,
      overflowing: over, old: Object.fromEntries(ids.map((i) => [i, sec(i)])),
      pills: [...document.querySelectorAll('#thermo .facts .pill')].map((e) => e.textContent + ':' + e.className) };
  }, OLD_IDS);
  if (shot) for (const id of ['rules', 'thermo']) { const el = await p.$('#' + id); if (el) await el.screenshot({ path: path.join(SHOTS, `${shot}_${id}.png`) }); }
  await p.close();
  return info;
}
(async () => {
  const b = await chromium.launch();
  const H = path.join(TMP, 'h.html'), H0 = path.join(TMP, 'h0.html');
  const out = { cases: {} };
  const sum = (i) => ({ marksChecked: i.markN, marksMismatch: i.markBad, errors: i.errors, faults: i.faults.length, faultSample: i.faults.slice(0, 3), hScroll: i.scrollW > i.clientW, scrollW: i.scrollW, clientW: i.clientW, nRules: i.nRules, nThermoRows: i.nThermoRows, overflowing: i.overflowing, pills: i.pills });
  // 改之前的頁面（同一份資料）：舊區塊的文字當基準
  build(BEFORE, H0, {}); const base = {};
  for (const w of [1280, 400]) base[w] = (await probe(b, H0, w, 'light', null));
  out.before = { errors: base[1280].errors, faults: base[1280].faults.length };
  // 1. 正常資料：兩種主題 × 桌機／手機
  build(path.join(SITE, 'index.html'), H, {});
  for (const [w, sch] of [[1280, 'light'], [1280, 'dark'], [400, 'light'], [400, 'dark'], [360, 'light']]) {
    const i = await probe(b, H, w, sch, `normal_${w}_${sch}`);
    const s = sum(i); const bw = base[w === 360 ? 400 : w];
    if (w !== 360) s.oldSectionsTextSameAsBefore = OLD_IDS.every((id) => i.old[id] === bw.old[id]);
    if (w === 1280 && sch === 'light') { s.rulesText = i.rules; s.thermoText = i.thermo; }
    out.cases[`normal_${w}_${sch}`] = s;
  }
  // 2～5. 缺資料、載入失敗、壞資料、空資料、冷＋完整月
  const variants = { missing_both: { drop: ['thermo', 'rules'] }, error_both: { error: ['thermo', 'rules'] }, bad_rows: { patch: { thermo: 'bad', rules: 'bad' } }, empty_rows: { patch: { thermo: 'empty', rules: 'empty' } }, cold_full_month: { patch: { thermo: 'cold_full' } } };
  for (const [name, sc] of Object.entries(variants)) {
    build(path.join(SITE, 'index.html'), H, sc);
    for (const w of [1280, 400]) {
      const i = await probe(b, H, w, 'light', w === 400 ? `${name}_400_light` : null);
      const s = sum(i); s.oldSectionsTextSameAsBefore = OLD_IDS.every((id) => i.old[id] === base[w].old[id]); s.rulesText = i.rules; s.thermoText = i.thermo;
      out.cases[`${name}_${w}`] = s;
    }
  }
  // 整頁手機截圖一張（確認新區塊在頁面裡的位置）
  build(path.join(SITE, 'index.html'), H, {});
  const p = await b.newPage({ viewport: { width: 400, height: 900 }, colorScheme: 'light' }); await p.goto('file://' + H); await p.waitForTimeout(500);
  await p.screenshot({ path: path.join(SHOTS, 'fullpage_400_light.png'), fullPage: true }); await p.close();
  await b.close();
  out.pass = Object.entries(out.cases).every(([k, c]) => c.errors.length === 0 && c.faults === 0 && !c.hScroll && c.overflowing.length === 0 && c.oldSectionsTextSameAsBefore !== false && (k.startsWith('bad_rows') || c.marksMismatch.length === 0));
  fs.writeFileSync(path.join(__dirname, 'page_test_result.json'), JSON.stringify(out, null, 1));
  for (const [k, c] of Object.entries(out.cases)) console.log(k, JSON.stringify({ err: c.errors, faults: c.faults, fs: c.faultSample, hScroll: c.hScroll, w: [c.scrollW, c.clientW], marks: [c.marksChecked, c.marksMismatch.length], nR: c.nRules, nT: c.nThermoRows, over: c.overflowing, oldSame: c.oldSectionsTextSameAsBefore, pills: c.pills }));
  console.log('before', JSON.stringify(out.before), 'PASS', out.pass);
})().catch((e) => { console.error(e); process.exit(1); });
