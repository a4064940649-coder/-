#!/usr/bin/env python3
"""補不到的缺口的「季營收代理」B_Q（不是 R_FIX：季損益表營收與月營收合計的一致率只有約 85%，沒過 99% 閘門，只當估計值）。
月 k 的代理年增率＝「k 月（含）以前最後一個完整季」的季營收年增率（不用 k 之後的月份，避免前視）。
  季營收：taiwanan tse_earning（2013Q1 起，合併）；2012 各季（2013 各季的基期）用 tw_revenue 月營收三個月合計（個體，與面板 2013 年合併對個體的斷點同法）。
y3 代理（訊號月 m）＝ k＝m−1、m−2、m−3 的代理年增率平均。只輸出 yq（月 × 代碼）。
驗證（存活股、2014-04～2018-12 訊號、池內）：代理 y3 與真實 y3 的池內百分位差。不看報酬。
"""
import sys, json, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import v4lib as L
K, V = L.K, L.V

R = pd.read_parquet(HERE / 'raw_quarterly.parquet'); R = R[~R.suspect]
Q = {(c, q): v for c, q, v in zip(R.code, R.q, R.rev_k)}
d = pd.read_csv(K.LONG, dtype={'code': str, 'market': str}); d = d[d.market == 'sii']
d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
d = d[(d.p >= pd.Period('2012-01', 'M')) & (d.p <= pd.Period('2012-12', 'M'))]
d['q'] = d.p.dt.qyear.astype(str) + 'Q' + d.p.dt.quarter.astype(str)
g = d.groupby(['code', 'q']).revenue_k_twd.agg(['sum', 'size'])
for (c, q), r in g.iterrows():
    if r['size'] == 3 and (c, q) not in Q: Q[(c, q)] = float(r['sum'])
codes = sorted({c for c, _ in Q})
months = pd.period_range('2013-01', '2018-12', freq='M')
def lastq(p):           # k 月（含）以前最後一個完整季
    qe = pd.Period(p, 'Q')
    if p.month % 3 != 0: qe = qe - 1
    return qe
yq = pd.DataFrame(np.nan, index=months, columns=codes)
for p in months:
    q = lastq(p); qb = q - 4
    for c in codes:
        a, b = Q.get((c, str(q))), Q.get((c, str(qb)))
        if a and b and b > 0: yq.at[p, c] = (a / b - 1) * 100
f = HERE / 'yoy_qproxy.parquet'
yw = yq.copy(); yw.index = yw.index.astype(str); yw.to_parquet(f)

# 驗證：存活股（R_DEL 有真實 y3）在池內的百分位差
YRD = L.load_rdel(); Y5 = L.y5_orig()
L.setup_px()
S = L.V4(YRD, Y5, L.disp_mask_fn('trade'))
diffs = []; cors = []
for dd in S.sig_days:
    if not (pd.Timestamp('2014-04-01') <= dd <= pd.Timestamp('2018-12-31')): continue
    cs = S.cross_section(dd); el = cs['el']
    m = pd.Period(dd, 'M'); ms = [m - 1, m - 2, m - 3]
    Z = yq.reindex(index=ms, columns=K.COLS).to_numpy()
    yp = Z.mean(axis=0); yp[np.isnan(Z).any(axis=0)] = np.nan
    y3 = S.y3(dd)
    ok = el & np.isfinite(yp)
    if ok.sum() < 30: continue
    rt = pd.Series(np.where(el, y3, np.nan)).rank(pct=True).to_numpy()
    # 代理值放進真實分布裡排名（＝B_Q 的作法）
    vals = np.sort(y3[el])
    rp = np.searchsorted(vals, yp[ok], side='right') / len(vals)
    diffs.extend(list(np.abs(rp - rt[ok])))
    cors.append(pd.Series(rp).corr(pd.Series(rt[ok]), method='spearman'))
diffs = np.array(diffs)
val = {'n_stockmonths': int(len(diffs)), 'mean_abs_pct_diff': float(diffs.mean()), 'median_abs_pct_diff': float(np.median(diffs)),
       'share_within_0.1': float((diffs <= 0.1).mean()), 'share_within_0.2': float((diffs <= 0.2).mean()),
       'mean_spearman_by_month': float(np.nanmean(cors)), 'n_months': len(cors)}
meta = {'file': f.name, 'sha256': hashlib.sha256(f.read_bytes()).hexdigest(), 'n_codes': len(codes), 'validation_survivors_2014_04_2018_12': val,
        'note': '季營收代理只當補不到缺口的估計（B_Q），不進 R_FIX'}
(HERE / 'yoy_qproxy_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(meta, ensure_ascii=False, indent=1))
