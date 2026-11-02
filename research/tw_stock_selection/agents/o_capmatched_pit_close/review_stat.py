#!/usr/bin/env python3
"""統計審查（只讀 out/*.npz 與 results.json，不改研究結果）：NW t、安慰劑校準、j 間相關、區塊自助、分段、去一年、均值／波動拆解、4763 價格錯置。
python3 -I -B review_stat.py → 寫 results.json['review_stat']"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from s4_stats import load, tstat, JK
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
low = R['main']['P1']['lower']
def nw_t(s, L_=3):
    s = np.asarray(s, float); n = len(s); u = s - s.mean(); v = u @ u / n
    for l in range(1, L_ + 1): v += 2 * (1 - l / (L_ + 1)) * (u[l:] @ u[:-l]) / n
    return float(s.mean() / math.sqrt(v / n))
rng = np.random.default_rng(99)
def boot_p(x, B=5000, bl=6):
    n = len(x); m = []
    for _ in range(B):
        idx = np.concatenate([np.arange(s, s + bl) % n for s in rng.integers(0, n, n // bl + 1)])[:n]; m.append(x[idx].mean())
    m = np.array(m); return float(np.mean(m <= 0)), [float(np.percentile(m, 2.5) * 100), float(np.percentile(m, 97.5) * 100)]
OUT = {}
for cfg in ['a', 'b', 'o7']:
    D = load(f'run_{cfg}_MAIN')
    xs = {k: D[k]['ra'] - D[k]['Rr'].mean(axis=0) for k in JK}
    o = dict(t_jmed=float(np.median([tstat(xs[k]) for k in JK])), nw3_t_jmed=float(np.median([nw_t(xs[k]) for k in JK])))
    bp = [boot_p(xs[k]) for k in JK]; o['block_boot_p_le0_jmed'] = float(np.median([b[0] for b in bp]))
    # 安慰劑校準：拿 40 個隨機種子當「實際」，對其餘 999 個的平均算 t
    cal = []
    for k in JK[::3]:
        Rr = D[k]['Rr']; tot = Rr.sum(axis=0); n = Rr.shape[0]
        for s in range(40):
            cal.append(tstat(Rr[s] - (tot - Rr[s]) / (n - 1)))
    cal = np.array(cal)
    o['placebo_t_sd'] = float(cal.std()); o['placebo_t_mean'] = float(cal.mean()); o['placebo_share_t_ge_2'] = float(np.mean(cal >= 2.0)); o['placebo_t_p95'] = float(np.percentile(cal, 95))
    # j 間相關
    M = np.array([xs[k][:87] for k in JK]); C = np.corrcoef(M); o['corr_between_j_median'] = float(np.median(C[np.triu_indices(13, 1)]))
    # 分段
    for nm, (a, b) in {'2019-05_2022-12': ('2019-05-01', '2022-12-31'), '2023_2026': ('2023-01-01', '2026-12-31'), '2024_2026': ('2024-01-01', '2026-12-31')}.items():
        ts = []; ms = []
        for k in JK:
            sd = pd.DatetimeIndex(D[k]['sd']); m = (sd >= a) & (sd <= b); ts.append(tstat(xs[k][m])); ms.append(xs[k][m].mean() * 100)
        o[f't_jmed_{nm}'] = float(np.median(ts)); o[f'mean_pp_jmed_{nm}'] = float(np.median(ms))
    # 去一年
    loy = {}
    for y in range(2019, 2027):
        loy[y] = float(np.median([tstat(xs[k][pd.DatetimeIndex(D[k]['sd']).year != y]) for k in JK]))
    o['leave_one_year_out_t_jmed'] = loy
    # 均值／波動
    o['mean_pp_jmed'] = float(np.median([xs[k].mean() * 100 for k in JK])); o['sd_pp_jmed'] = float(np.median([xs[k].std(ddof=1) * 100 for k in JK]))
    o['mean_pp_off'] = float(D['OFF']['ra'].mean() * 0 + (D['OFF']['ra'] - D['OFF']['Rr'].mean(axis=0)).mean() * 100)
    o['sd_pp_off'] = float((D['OFF']['ra'] - D['OFF']['Rr'].mean(axis=0)).std(ddof=1) * 100)
    OUT[cfg] = o
    print(cfg, json.dumps(o, ensure_ascii=False, default=float))
# (i)～(iv) 的 t 變化拆成均值與波動
S = R['main']['by_set']
dec = {}
for c in ['o7', 'i', 'ii', 'a', 'b']:
    D = load(f'run_{c}_MAIN'); x = D['OFF']['ra'] - D['OFF']['Rr'].mean(axis=0)
    xj = [D[k]['ra'] - D[k]['Rr'].mean(axis=0) for k in JK]
    dec[c] = dict(off_mean_pp=float(x.mean() * 100), off_sd_pp=float(x.std(ddof=1) * 100), j_mean_pp_med=float(np.median([v.mean() * 100 for v in xj])),
                  j_sd_pp_med=float(np.median([v.std(ddof=1) * 100 for v in xj])))
OUT['mean_sd_decomposition'] = dec
# 4763 價格錯置（o7 面板 src 3：2025-05-21～06-27 有些日子是分割後價格）
sys.path.insert(0, str(HERE)); import pcore_lite as PL
C = pd.read_parquet(PL.O7 / 'raw' / 'panel' / 'close_raw.parquet', columns=['4763'])['4763']
D = load(f'run_{low}_MAIN')
glitch = []
for k in ['OFF'] + JK:
    for d in D[k]['sd']:
        if '2025-05-15' <= d <= '2025-06-30':
            p = C.loc[:d].dropna(); glitch.append((k, d, float(p.iloc[-1])))
OUT['p4763_price_on_signal_days_2025_05_06'] = glitch
R['review_stat'] = OUT
(HERE / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print(json.dumps(dec, ensure_ascii=False, indent=1))
