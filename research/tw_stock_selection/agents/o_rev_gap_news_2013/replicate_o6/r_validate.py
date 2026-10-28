#!/usr/bin/env python3
"""獨立複製 o6：代理品質驗證（自寫）。
驗證格：訊號日 2013～2021、前 300 池 ok_other 且非 AB y3 有真值（非缺口格）。
把三個月（含退月）的官方年增率全部當成缺，只用新聞（水準÷官方 p−12 水準，或新聞年增率）與季報重建 N3／N2／N1／Q／NQ。
指標：重建值放回當日橫斷面後的百分位對真值百分位的 Spearman；百分位誤差中位數／p90；前 10 名一致率。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parent
AG = HERE.parent.parent
sys.path.insert(0, str(AG / 'o_v4_revenue_patch_official'))
import v4lib as L
import importlib.util
spec = importlib.util.spec_from_file_location('rp', HERE / 'r_proxy.py'); rp = importlib.util.module_from_spec(spec); spec.loader.exec_module(rp)

# 驗證時官方年增率全部當成缺
rp.off_yoy = lambda d, c, p: None

XR = L.setup_px()
YFX, Y5F = L.load_rfix()
S = L.V4(YFX, Y5F, L.disp_mask_fn('trade'))
sds = [d for d in S.sig_days if pd.Timestamp('2013-01-01') <= d <= pd.Timestamp('2021-12-31')]
VERS = ['N3', 'N2', 'N1', 'Q', 'NQ']
rows = []
for d in sds:
    cs = S.cross_section(d)
    el = cs['ok_other'] & np.isfinite(cs['y3'])
    idx = np.where(el)[0]
    y = cs['y3'][idx]; s52 = cs['s52'][idx]
    n = len(idx)
    pr52 = pd.Series(s52).rank(pct=True).to_numpy()
    prr = pd.Series(y).rank(pct=True).to_numpy()
    sc = 0.5 * pr52 + 0.5 * prr
    top_true = set(np.argsort(-sc, kind='stable')[:10])
    for k, j in enumerate(idx):
        c = L.COLS[j]
        px = rp.proxies(pd.Timestamp(d), c)
        rec = dict(sig=d.strftime('%Y-%m-%d'), code=c, y_true=y[k], pct_true=prr[k], top_true=k in top_true)
        for v in VERS:
            val = px[v]
            if np.isfinite(val):
                yy = y.copy(); yy[k] = val
                pr = pd.Series(yy).rank(pct=True).to_numpy()
                s2 = 0.5 * pr52 + 0.5 * pr
                rec[f'pct_{v}'] = pr[k]
                rec[f'top_{v}'] = k in set(np.argsort(-s2, kind='stable')[:10])
            else:
                rec[f'pct_{v}'] = np.nan; rec[f'top_{v}'] = None
        rows.append(rec)
V = pd.DataFrame(rows)
V.to_parquet(HERE / 'r_validation_cells.parquet', index=False)
out = dict(n_cells=len(V), n_days=len(sds))
for v in VERS:
    f = np.isfinite(V[f'pct_{v}'])
    a, b = V.loc[f, f'pct_{v}'], V.loc[f, 'pct_true']
    e = np.abs(a - b)
    tt = V.loc[f, 'top_true'].astype(bool); tp = V.loc[f, f'top_{v}'].astype(bool)
    out[v] = dict(cov=float(f.mean()), n=int(f.sum()), spearman=float(spearmanr(a, b).correlation) if f.sum() > 2 else None,
                  err_med=float(e.median()), err_p90=float(e.quantile(0.9)),
                  top10_agree_all=float((tt == tp).mean()),
                  top10_keep_given_true=float(tp[tt].mean()) if tt.any() else None,
                  top10_jaccard=float((tt & tp).sum() / max(1, (tt | tp).sum())),
                  n_top_true=int(tt.sum()))
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open(HERE / 'r_validation.json', 'w'), ensure_ascii=False, indent=1)
