#!/usr/bin/env python3
"""反方審查（成本與可成交、資料實際時點）o6：只讀研究員檔案，不改。
A. 訊號日相對月營收法定期限（10 日）的位置；新聞提及距訊號日天數。
B. B10 入選且有 N1 代理的 29 格：逐格列出所用提及（來源、日期、原文片段）、代理百分位，供人工核對解析錯置。
C. B10 入選 133 股月的可成交性：進場日開盤漲停（買不到）、V60 容量、持有窗內下市／鎖跌停。
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import rglib as G
import proxy as PX
K, V, L = G.K, G.V, G.L
out = {}
b = G.base(); S = G.mk()
M, LV, L5, Q = PX.load_inputs(K, HERE / 'data' / 'mentions_P3.parquet')
M['ts'] = pd.to_datetime(M.ts)
P = PX.Proxy(M, LV, L5, Q, S.YL, S.Y5, V.SWITCH)
X = pd.read_csv(HERE / 'gap_cells_2013_2021.csv', dtype={'code': str})
T = pd.read_csv(HERE / 'data' / 'proxy_table_gap_cells.csv', dtype={'code': str})
assert (T.sig.values == X.sig.values).all()

# A. 訊號日的日期分布
sd = sorted({pd.Timestamp(s) for s in X.sig})
out['A_sig_day_of_month'] = pd.Series([d.day for d in sd]).value_counts().sort_index().to_dict()

# B. 29 格逐格
rows = []
for s, c, t, ib, v in zip(T.sig, T.code, T.type, T.in_B10, T.proxy_N1):
    if not (ib and np.isfinite(v)): continue
    d = pd.Timestamp(s); cs = S.cross_section(d); y3 = S.y3(d)
    tv = y3[cs['ok_other'] & np.isfinite(y3)]
    pct = float((tv < v).mean() + 0.5 * (tv == v).mean())
    m = pd.Period(d, 'M'); lim = d + PX.CUT
    used = []
    for p in (m - 1, m - 2, m - 3):
        g = M[(M.code == c) & (M.ym == str(p)) & (M.ts <= lim)].sort_values('ts')
        if len(g):
            r = g.iloc[0]
            used.append(dict(ym=str(p), src=r.src, ts=str(r.ts.date()), days_before_sig=int((d - r.ts).days),
                             rev_k=None if not np.isfinite(r.rev_k) else float(r.rev_k), yoy=None if not np.isfinite(r.yoy) else float(r.yoy),
                             pre_monthend=bool(r.ts < (p + 1).start_time), snip=str(r.snip)[:90]))
    rows.append(dict(sig=s, code=c, type=t, proxy_y3=float(v), pct=pct, n_news_months=len(used), used=used))
out['B_n'] = len(rows)
out['B_by_code'] = pd.Series([r['code'] for r in rows]).value_counts().to_dict()
out['B_n_single_month'] = sum(r['n_news_months'] == 1 for r in rows)
out['B_any_pre_monthend'] = sum(any(u['pre_monthend'] for u in r['used']) for r in rows)
out['B_cells'] = rows
# 以股票為單位的中位數（每檔先取中位數）
bc = pd.DataFrame([(r['code'], r['pct']) for r in rows], columns=['code', 'pct'])
out['B_median_of_stock_medians'] = float(bc.groupby('code').pct.median().median())
out['B_n_stocks'] = int(bc.code.nunique())

# C. B10 入選 133 股月可成交性
On, Cn, Hn = K.On, K.Cn, K.H.to_numpy()
Vv = K.V.to_numpy()
IDX = K.IDX
sd_all = sorted(pd.Timestamp(x) for x in S.bounds()[2])
GAP = b['GAP']
crow = []
for s, c, t in zip(X.sig[X.in_B10], X.code[X.in_B10], X.type[X.in_B10]):
    d = pd.Timestamp(s); ti = IDX.get_loc(d); e = ti + 1; j = K.CI[c]
    nxt = [x for x in sd_all if x > d]
    te = IDX.get_loc(nxt[0]) + 1 if nxt else min(e + 21, len(IDX) - 1)
    gu = bool(K.gap_up_open(e)[j])
    v60 = float(S.V60[ti, j])
    # 持有窗內：最後交易日（下市）與鎖跌停日（開＝高＝收且當日跌幅 ≥ 6.5%/9.5%）
    seg = slice(e, te + 1)
    tr = K.TRADED[seg, j]
    last_trade_in_win = bool((~tr).any() and tr.any())
    lim = np.where(IDX[seg] < pd.Timestamp('2015-06-01'), 0.065, 0.095)
    pc = K.PREVC[seg, j]
    r = Cn[seg, j] / pc - 1
    locked_dn = int(np.nansum((r <= -lim) & np.isclose(On[seg, j], Cn[seg, j]) & np.isclose(Hn[seg, j], Cn[seg, j])))
    dl = GAP.get(c)
    crow.append(dict(sig=s, code=c, type=t, gapup_open_limit=gu, v60=v60, untraded_in_window=last_trade_in_win,
                     locked_limit_down_days=locked_dn, delist_date=dl, delist_in_window=bool(dl and pd.Timestamp(dl) <= IDX[min(te, len(IDX) - 1)])))
C = pd.DataFrame(crow)
out['C_n'] = len(C)
out['C_gapup_open_limit'] = int(C.gapup_open_limit.sum())
out['C_v60_quantiles_NTD'] = C.v60.quantile([0, .1, .5]).to_dict()
out['C_v60_lt_2e7'] = int((C.v60 < 2e7).sum())
out['C_1M_per_name_share_of_v60_max'] = float((1e6 / C.v60).max())
out['C_untraded_in_window'] = int(C.untraded_in_window.sum())
out['C_delist_in_window'] = C[C.delist_in_window][['sig', 'code', 'type', 'delist_date']].to_dict('records')
out['C_locked_limit_down_cells'] = C[C.locked_limit_down_days > 0][['sig', 'code', 'type', 'locked_limit_down_days']].to_dict('records')
print(json.dumps({k: v for k, v in out.items() if k != 'B_cells'}, ensure_ascii=False, indent=1, default=str))
for r in rows:
    print(r['sig'], r['code'], r['type'], round(r['proxy_y3'], 1), round(r['pct'], 2))
    for u in r['used']:
        print('   ', u['ym'], u['src'], u['ts'], u['days_before_sig'], u['rev_k'], u['yoy'], u['pre_monthend'], '|', u['snip'])

# D. 華亞科 3474 兩格：提及片段把南科（2408）的 38.05 億元配給華亞科；改用華亞科自己的 59.76 億元重算年增率與百分位
d_out = {}
base = P.off_level('3474', pd.Period('2012-09', 'M'))
for s in ('2013-11-11', '2013-12-11'):
    d = pd.Timestamp(s); cs = S.cross_section(d); y3 = S.y3(d)
    tv = y3[cs['ok_other'] & np.isfinite(y3)]
    yy = (5976000.0 / base - 1) * 100
    d_out[s] = dict(base_2012_09_k=base, yoy_parsed=(3805000.0 / base - 1) * 100, yoy_corrected=yy,
                    pct_corrected=float((tv < yy).mean() + 0.5 * (tv == yy).mean()))
print('D', json.dumps(d_out, ensure_ascii=False, default=float))
