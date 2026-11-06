#!/usr/bin/env python3
"""步驟 3：探索（PREREG_DESIGN 之後；只用歷史線、G8P-1 殘差與 G8P-1 程式；不讀前瞻報酬）。
 (a) 歷史決策線（V7X GROW 100 萬；交叉核對 V7 FIX 100 萬）：逐 J 窗壓力較差的比例、挑選期平移 Δ_stress、Δ_w（依 PREREG_DESIGN.delta_w_rule）
 (b) 滾動 12／24 個月 sd（2019 年起，描述）
 (c) 重現兩位審查：G8P1 原樣 σ 掃描（D1、D3 縮放；P＝200,000）、成本平移 0.3／0.577／0.78 的 S1 機率（D1 σ_d；P＝400,000）
 (d) n＝12 樣本 sd 的抽樣分布（D1／D2／D3）與 studentized 尾部
 (e) 前瞻 meta：replay 的作廢頻率（只看時戳與覆蓋率欄位）、live runs.jsonl 的 rev_gate 狀態
用法：python3 -I -B s3_explore.py"""
import sys, json, math, glob
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import g8lib2 as L

L.check_reg('PREREG_DESIGN', files=False)
D = L.load_res()['PREREG_DESIGN']
BASE = 2026110631
out = dict(utc=L.now_utc())

# ---------------------------------------------------------------- (a)(b)
Mn, Ms, F = L.hist_windows()
Fn, Fs = L.hist_windows_v7fix()
def per(s, a, b):
    return s[(s.index >= pd.Period(a, 'M')) & (s.index <= pd.Period(b, 'M'))]
def line_stats(Mn, Ms, name):
    both = Mn.notna() & Ms.reindex_like(Mn).notna()
    worse_s = (Ms.reindex_like(Mn) < Mn) & both
    eq = (Ms.reindex_like(Mn) == Mn) & both
    n_line = L.jmed_line(Mn, worse=False); s_line = L.jmed_line(Ms, worse=False); w_line = L.jmed_line(Mn, Ms, worse=True)
    r = {}
    for nm, (a, b) in dict(sel=('2006-04', '2019-04'), post=('2019-05', '2026-10'), all=('2006-04', '2026-10')).items():
        msk = (Mn.index >= pd.Period(a, 'M')) & (Mn.index <= pd.Period(b, 'M'))
        z = pd.concat([n_line, s_line, w_line], axis=1, keys=['n', 's', 'w']).dropna()
        z = per(z, a, b)
        r[nm] = dict(n_windows=int(both[msk].values.sum()), frac_stress_worse=float(worse_s[msk].values.sum() / both[msk].values.sum()),
                     frac_equal=float(eq[msk].values.sum() / both[msk].values.sum()),
                     n_months=len(z), first=str(z.index[0]), last=str(z.index[-1]),
                     mean_normal=float(z.n.mean()), mean_worse=float(z.w.mean()),
                     shift_stress=float((z.s - z.n).mean()), shift_worse=float((z.w - z.n).mean()),
                     sd_normal=float(z.n.std(ddof=1)), sd_worse=float(z.w.std(ddof=1)))
    return r, n_line, w_line
ra, nx, wx = line_stats(Mn, Ms, 'V7X')
rb, nf, wf = line_stats(Fn, Fs, 'V7FIX')
out['hist'] = {'V7X_GROW100': ra, 'V7_FIX100': rb}
dw = math.floor(min(abs(ra['sel']['shift_worse']), abs(rb['sel']['shift_worse'])) * 1000) / 1000
out['delta_w'] = dw
out['delta_w_note'] = 'PREREG_DESIGN.delta_w_rule：兩條歷史線挑選期 shift_worse 絕對值較小者，3 位小數無條件捨去'
def roll(s, w):
    p = per(s, '2019-05', '2026-10')
    r = p.rolling(w).std(ddof=1).dropna()
    return dict(min=float(r.min()), q25=float(r.quantile(.25)), median=float(r.median()), q75=float(r.quantile(.75)), max=float(r.max()), n=len(r))
out['rolling_sd_post2019'] = {nm: {f'roll{w}': roll(s, w) for w in (12, 24)} for nm, s in
                              dict(V7X_worse=wx, V7X_normal=nx, V7FIX_normal=nf, V7FIX_worse=wf).items()}
out['rolling_sd_sel'] = {nm: {f'roll{w}': (lambda r: dict(q75=float(r.quantile(.75)), max=float(r.max())))(per(s, '2006-04', '2019-04').rolling(w).std(ddof=1).dropna())
                              for w in (12, 24)} for nm, s in dict(V7X_worse=wx, V7FIX_normal=nf).items()}
pd.DataFrame(dict(V7X_normal=nx, V7X_worse=wx, V7FIX_normal=nf, V7FIX_worse=wf)).to_csv(L.SCR / 'hist_lines_g8p2.csv')

# ---------------------------------------------------------------- (c) 重現審查
G8P1 = dict(stat='FIX_REFIT')
rep = dict(sigma_scan_D1={}, sigma_scan_D3={}, cost_shift_D1={})
for i, s in enumerate([6.141373712658076, 6.51, 7.0, 7.5, 8.0, 8.4, 9.0, 10.0, 10.4]):
    for dg in ('D1', 'D3'):
        X = L.gen(dg, 0.0, s, 200_000, 24, L.cell_rng(BASE, dg, i, 99, 0, 1))
        o = L.outcomes(L.zstats(X, G8P1), L.B_G8P1, L.CD_G8P1)
        rep[f'sigma_scan_{dg}'][f'{s:.2f}'] = dict(up=float(o['up'].mean()), up_seq_no_gate=None)
for k, m in enumerate((0.0, 0.3, 0.577, 0.78)):
    X = L.gen('D1', m, L.SIGMA_D, 400_000, 24, L.cell_rng(BASE, 'D1', 0, k, 0, 2))
    o = L.outcomes(L.zstats(X, G8P1), L.B_G8P1, L.CD_G8P1)
    rep['cost_shift_D1'][f'{m}'] = dict(s1=float(o['up'].mean()), literal_g8=float(o['lit'].mean()), fail=float(o['fail'].mean()))
rep['reviewer_stat'] = '4.8%（σ 6.51）、6.0%（7.0）、7.1%（7.5）、7.6%（8.0）、7.8%（9.0）；D3 縮放約 2%'
rep['reviewer_exec'] = 'S1：0.0393（μ 0）、6.4%（0.3）、9.6%（0.577）、12.5%（0.78）；字面 G8 0.50→0.63'
out['review_repro'] = rep

# ---------------------------------------------------------------- (d) n＝12 樣本 sd 與 studentized 尾部
cs = L.C_INFL * L.SIGMA_D
sdd = {}
for dg in ('D1', 'D2', 'D3'):
    X = L.gen(dg, 0.0, 1.0, 400_000, 24, L.cell_rng(BASE, dg, 0, 99, 0, 3))
    S = X.cumsum(1); S2 = (X * X).cumsum(1)
    r = {}
    for n in (12, 24):
        sh = L._sd(S[:, n - 1], S2[:, n - 1], n)
        tt = S[:, n - 1] / (sh * math.sqrt(n))
        from scipy import stats as st
        r[n] = dict(q05=float(np.quantile(sh, .05)), q25=float(np.quantile(sh, .25)), median=float(np.median(sh)), q75=float(np.quantile(sh, .75)),
                    p_t_gt_t95=float(np.mean(tt > st.t.isf(0.05, n - 1))), p_t_gt_t99=float(np.mean(tt > st.t.isf(0.01, n - 1))),
                    p_z_gt_bsqrt=float(np.mean(S[:, n - 1] / math.sqrt(n) > L.B_G8P1 * math.sqrt(24 / n))),
                    p_studentized_gt_bsqrt=float(np.mean(tt > L.B_G8P1 * math.sqrt(24 / n))),
                    p_sd_below_floor={lab: float(np.mean(sh * sg < cs)) for lab, sg in zip(L.SIG_LABEL, L.SIG_GRID)})
    sdd[dg] = r
out['sample_sd_unit'] = sdd
out['sample_sd_note'] = 'X 為單位 sd；s_n/σ 的分位數；p_sd_below_floor＝真實 σ 下 s_n < c·σ_d（V1 用 floor）的機率；p_studentized_gt_bsqrt＝S_n/(s_n√n) 超過 G8P-1 正態界限的機率（studentized 尾部）'

# ---------------------------------------------------------------- (e) 前瞻 meta（只讀時戳與覆蓋率）
sys.path.insert(0, str(L.G8P1_DIR))
import g8_evaluate as E1
sys.path.pop(0)
P1 = L.g8p1_protocol()
rp = L.AG / 'o_forward_infra_live/replay/seq/paper_lists'
vr = {}
for f in sorted(rp.glob('*.meta.json')):
    m = json.loads(f.read_text(encoding='utf-8'))
    vr[f.name] = E1.void_reason(m, P1)
cnt = {}
for k, v in vr.items():
    key = (k.split('_')[0], (v or 'VALID')[:2])
    cnt[f'{key[0]}|{key[1]}'] = cnt.get(f'{key[0]}|{key[1]}', 0) + 1
live_runs = []
for l in open(L.AG / 'live_tracking/main_forward/runs.jsonl', encoding='utf-8'):
    r = json.loads(l); g = r.get('rev_gate') or {}
    live_runs.append(dict(asof=r.get('asof'), mode=g.get('mode'), status=g.get('status'), coverage=g.get('coverage'), official_loaded=g.get('official_loaded'), j=g.get('j')))
out['void_meta'] = dict(replay_n_meta=len(vr), replay_void_by_line=cnt, replay_has_rev_gate=0,
                        replay_note='replay 的 28 份 meta 都沒有 rev_gate 欄（replay 早於 BF-5 閘門），依 G8P-1 V2 全部作廢；所以 replay 無法量化 J0_UNVERIFIED 的作廢頻率',
                        live_runs=live_runs,
                        live_note='live runs.jsonl 的 rev_gate：3 次非 J0 日（j 14～16）mode＝full、PASS、official_loaded＝false（官方月營收抓取 0 列）。'
                                  'make_forward.rev_gate 在 J0（j＝0 且非官方日）若 official_loaded＝false → status＝J0_UNVERIFIED 且不寫 coverage → G8P-1 V2 作廢 J0 窗。'
                                  '在官方抓取修好以前，每月 J0 窗會被機械作廢，決策線實際是 J1～J12 的中位數（12 窗 ≥ 7，不影響錨定月有效性）。')
(HERE / 'explore.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=L._js), encoding='utf-8')
L.save_key('EXPLORE', out, need=('PREREG_DESIGN',))
print(json.dumps({k: out[k] for k in ('hist', 'delta_w', 'rolling_sd_post2019', 'review_repro')}, ensure_ascii=False, indent=1, default=L._js))
print(json.dumps(out['sample_sd_unit'], ensure_ascii=False, default=L._js)[:3000])
print(json.dumps(out['void_meta']['replay_void_by_line'], ensure_ascii=False))
