#!/usr/bin/env python3
"""獨立複製（κ）：只依 frozen_spec.txt 第一～三節，從 raw/tw5m_head.parquet、raw/tw5m_daily.parquet、data/official_daily.parquet、
data/feat_all_2023.parquet 重算股日宇宙、bar 報酬，自行估 M3V（δ 格點 0.200～1.000 步長 0.005，閉式最小平方）與 M1/M2/M3 的 VAL 十分位比值。
不 import 本題其他 .py（calib_lib、iterate 等）。注意：複製者與研究員為同一代理，見 FINDINGS 揭露。
輸出 replicate/rep_kappa.json。用法：python3 -I -B replicate/rep_kappa.py
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parent; T = R.parent
BAD = ['2024-09-20', '2025-01-21', '2025-03-21', '2025-04-07', '2025-04-10', '2025-11-24']
H = pd.read_parquet(T / 'raw' / 'tw5m_head.parquet'); H['date'] = pd.to_datetime(H['date'])
Dy = pd.read_parquet(T / 'raw' / 'tw5m_daily.parquet'); Dy['date'] = pd.to_datetime(Dy['date'])
OF = pd.read_parquet(T / 'data' / 'official_daily.parquet'); OF['date'] = pd.to_datetime(OF['date'])
FA = pd.read_parquet(T / 'data' / 'feat_all_2023.parquet').sort_values(['code', 'date'])
for c in ('V60', 'sig60', 'vrank'): FA[c + '1'] = FA.groupby('code')[c].shift(1)
HM = ['09:00'] + [f'09:{m:02d}' for m in range(5, 60, 5)]
H = H[H.hm.isin(HM)].sort_values(['code', 'date', 'hm'])
cnt = H.groupby(['code', 'date']).hm.transform('size')
H = H[cnt == 12]
c = H.pivot(index=['code', 'date'], columns='hm', values='close')[HM]
v = H.pivot(index=['code', 'date'], columns='hm', values='volume')[HM]
o9 = H[H.hm == '09:00'].set_index(['code', 'date'])['open']
lr = np.log(c).diff(axis=1).iloc[:, 1:].abs()          # 11 根 |r|
q = v.iloc[:, 1:]
S = pd.DataFrame({'o9': o9}).join(Dy.set_index(['code', 'date'])[['nbar']]).reset_index()
S = S.merge(OF[['code', 'date', 'open', 'vol']], on=['code', 'date']).merge(FA[['code', 'date', 'V601', 'sig601', 'vrank1', 'prevc', 'fin']], on=['code', 'date'])
S = S[(S.vrank1 <= 300) & (S.fin == False) & (S.nbar >= 50) & (S.vol > 0) & (S.sig601 > 0) & np.isfinite(S.V601) & (np.abs(S.o9 / S.open - 1) <= 0.005)
      & ~S.date.dt.strftime('%Y-%m-%d').isin(BAD)]
S['Vsh'] = S.V601 / S.prevc
S['dec'] = np.ceil(S.vrank1 / 30)
S['per'] = np.where((S.date >= '2024-02-15') & (S.date <= '2025-06-30'), 'CAL', np.where((S.date >= '2025-07-01') & (S.date <= '2026-04-01'), 'VAL', 'X'))
key = pd.MultiIndex.from_frame(S[['code', 'date']])
Y = lr.reindex(key).to_numpy(); Qv = q.reindex(key).to_numpy()
sig = S.sig601.to_numpy()[:, None]; Vs = S.Vsh.to_numpy()[:, None]
ok = np.isfinite(Y) & (Qv > 0)
out = {'n_stockdays': {p: int((S.per == p).sum()) for p in ('CAL', 'VAL')}}
cal = (S.per == 'CAL').to_numpy(); val = (S.per == 'VAL').to_numpy()
def lsq(mask, delta):
    m = ok & mask[:, None]
    x = (sig * (Qv / Vs) ** delta)[m]; y = Y[m]
    n = len(x); sx, sy, sxx, sxy = x.sum(), y.sum(), (x * x).sum(), (x * y).sum()
    b = (sxy - sx * sy / n) / (sxx - sx * sx / n); a = (sy - b * sx) / n
    return a, b, float(((a + b * x - y) ** 2).sum())
best = min(((d,) + lsq(cal, d) for d in np.round(np.arange(0.2, 1.0001, 0.005), 3)), key=lambda r: r[3])
d, a, b, _ = best
out['M3V'] = dict(delta=float(d), a=float(a), b=float(b), impact_ref_per_sigma=float(b * 1e-3 ** d))
out['M3V_on_0.01_grid'] = (lambda r: dict(delta=float(r[0]), a=float(r[1]), b=float(r[2]), impact_ref_per_sigma=float(r[2] * 1e-3 ** r[0])))(
    min(((dd,) + lsq(cal, dd) for dd in np.round(np.arange(0.2, 1.0001, 0.01), 2)), key=lambda r: r[3]))
# VAL 十分位比值（M3V 與 M2）
def dec_ratio(pred):
    m = ok & val[:, None]
    yh = np.where(m, pred, np.nan); yy = np.where(m, Y, np.nan)
    sdh = np.nanmean(yh, axis=1)[val]; sdy = np.nanmean(yy, axis=1)[val]
    dd = S.dec.to_numpy()[val]
    r = {int(k): float(np.nansum(sdh[dd == k]) / np.nansum(sdy[dd == k])) for k in range(1, 11)}
    return dict(dec_ratio=r, in_band=bool(all(0.67 <= x <= 1.5 for x in r.values())), MdAE=float(np.nanmedian(np.abs(sdh - sdy))))
out['VAL_M3V'] = dec_ratio(a + b * sig * (Qv / Vs) ** d)
x2 = sig * np.sqrt(Qv / Vs); m = ok & cal[:, None]
Y2 = float(Y[m].sum() / x2[m].sum())
out['M2_Y'] = Y2; out['VAL_M2'] = dec_ratio(Y2 * x2)
fz = json.load(open(T / 'data' / 'model_calib.json'))
ref = fz['impact']['point']['b'] * 1e-3 ** fz['impact']['point']['delta']
out['frozen_impact_ref_per_sigma'] = ref
out['diff_pct_vs_frozen'] = float((out['M3V']['impact_ref_per_sigma'] / ref - 1) * 100)
out['diff_pct_vs_frozen_0.01grid'] = float((out['M3V_on_0.01_grid']['impact_ref_per_sigma'] / ref - 1) * 100)
out['pass_15pct'] = abs(out['diff_pct_vs_frozen']) <= 15
json.dump(out, open(R / 'rep_kappa.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
