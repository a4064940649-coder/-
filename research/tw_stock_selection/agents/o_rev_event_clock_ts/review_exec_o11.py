#!/usr/bin/env python3
"""o11 反方審查（成本與可成交）：只讀研究員檔案與上游資料，檢查 EC 早買的漲停、處置／注意、零股比例、周轉低估。
python3 -I -B review_exec_o11.py"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent
D = H.parent / 'o_forward_infra_live' / 'raw' / 'screener' / 'data'
tr = pd.read_csv(H / 'data' / 'ec_sim_trades.csv', dtype={'sell': str, 'buy': str})
hist = pd.read_csv(D / 'history.csv.gz', dtype={'code': str}, parse_dates=['date'])
print('history range', hist.date.min().date(), hist.date.max().date())
O = hist.pivot(index='date', columns='code', values='open'); C = hist.pivot(index='date', columns='code', values='close')
V = hist.pivot(index='date', columns='code', values='volume')
cal = O.index[(C.notna().sum(axis=1) >= 300)]
def tick(p): return float(np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0))
def lu(pc):
    x = pc * 1.1; t = tick(x); return math.floor(x / t + 1e-9) * t
dsp = pd.read_csv(D / 'extras' / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
att = pd.read_csv(D / 'extras' / 'attention_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
attset = set(zip(att.date, att.code))
rows = []
for r in tr.itertuples():
    d = pd.Timestamp(r.day)
    k = cal.searchsorted(d)
    if k >= len(cal) or cal[k] != d or k + 1 >= len(cal): continue
    e = cal[k + 1]
    for side, c in (('buy', r.buy), ('sell', r.sell)):
        if not isinstance(c, str) or c not in O: continue
        pc = C.at[d, c]; o = O.at[e, c]
        if not (np.isfinite(pc) and np.isfinite(o)): continue
        gap = o / pc - 1
        indisp = bool(((dsp.code == c) & (dsp.start <= e) & (dsp.end >= e)).any())
        disp_next5 = bool(((dsp.code == c) & (dsp.start > e) & (dsp.start <= e + pd.Timedelta(days=10))).any())
        rows.append(dict(anchor=r.anchor, kind=r.kind, side=side, code=c, exec=e, gap=gap, open_lu=o >= lu(pc) - 1e-6,
                         gap_ge5=gap >= 0.05, att_d=(d, c) in attset, disp_e=indisp, disp_soon=disp_next5,
                         odd_only_100w=o * 1000 > 100000, odd_only_300w=o * 1000 > 300000))
X = pd.DataFrame(rows)
out = {}
for side, g in X.groupby('side'):
    out[side] = dict(n=len(g), open_limit_up=int(g.open_lu.sum()), gap_ge5pct=int(g.gap_ge5.sum()), mean_gap=float(g.gap.mean()),
                     median_gap=float(g.gap.median()), attention_on_signal_day=int(g.att_d.sum()), in_disposal_exec=int(g.disp_e.sum()),
                     disposal_within_10d=int(g.disp_soon.sum()), odd_only_share_100w=float(g.odd_only_100w.mean()),
                     odd_only_share_300w=float(g.odd_only_300w.mean()))
# 基準：同期間所有股票隨機日的開盤跳空（池內近似：前 300 名成交值）
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(X[X.side == 'buy'].sort_values('gap', ascending=False).head(10)[['anchor', 'code', 'exec', 'gap', 'att_d', 'disp_soon']].to_string())
# 周轉低估：extra 對 n_ann_valid 的迴歸外推
ex = json.loads((H / 'explore.json').read_text())
m = pd.DataFrame(ex['ec_sim_months']['V2'])
b = np.polyfit(m.n_ann_valid, m.extra, 1); bw = np.polyfit(m.n_ann_valid, m.n_window_buys, 1)
print('extra ~ n_ann_valid slope %.4f int %.3f; mean n_ann_valid %.1f' % (b[0], b[1], m.n_ann_valid.mean()))
for n in (250, 290):
    e_ = np.polyval(b, n); print(f'外推 n_ann_valid={n}: extra≈{e_:.2f}, window_buys≈{np.polyval(bw, n):.2f}')
# 成本外推（一般成本 mean_rt）
P = json.loads((H / 'results.json').read_text())['PREREG']['expected_cost']['breakeven_drift']
for n in (250, 290):
    e_ = np.polyval(b, n)
    print(n, {k: round(e_ * v['mean_rt'] * 100 / 10, 3) for k, v in P.items()})
# 成本需扣掉漲停未成交：比較的主策略月超額
print('main full-period monthly excess pp', (16.0 - 14.37) / 12, (16.8 - 14.37) / 12)
hol = json.loads((D / 'extras' / 'tw_holidays.json').read_text())['years'].keys()
print('holiday years in upstream table:', list(hol))
