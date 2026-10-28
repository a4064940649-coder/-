#!/usr/bin/env python3
"""探索 (1)：下市股缺營收月數表（修補前），依當時是否在前 300 流動性池加權。不跑任何策略報酬。
輸出 gap_delisted.csv（逐檔）、gap_pool_stockmonths.csv（逐池內股票月）。
缺營收月：面板第一筆交易月 ～ 最後交易月之間，tw_revenue（sii）與 taiwanan 都沒有營收水準的月份。
池內缺口：換股訊號日 d ≤ 終止上市日，營收以外都合格（前 300、52 週有值、成交日非處置、隔日有開盤、非金融、非 KY／DR），但 R_DEL 的 y3 缺值。
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import v4lib as L
K, V = L.K, L.V

L.check_v3_inputs()
L.setup_px()
YRD = L.load_rdel(); Y5 = L.y5_orig()
d = pd.read_csv(K.LONG, dtype={'code': str, 'market': str})
d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
LV = d[d.market == 'sii'].pivot_table(index='p', columns='code', values='revenue_k_twd', aggfunc='first')
T = pd.read_csv(L.V3_INPUTS['src_taiwanan'][0], dtype={'code': str}); T['p'] = pd.PeriodIndex(T['ym'], freq='M')
LT = T.pivot_table(index='p', columns='code', values='rev', aggfunc='last')
r5 = pd.read_csv(K.X / 'rev_5y.csv.gz', dtype={'code': str}); r5['p'] = pd.PeriodIndex(r5['ym'], freq='M')
L5 = r5.pivot_table(index='p', columns='code', values='revenue', aggfunc='last')
allp = pd.period_range('2004-01', '2026-09', freq='M')
def has_level(c, p):
    for X in (LV, LT, L5):
        if c in X.columns and p in X.index and np.isfinite(X.at[p, c]): return True
    return False

DL = pd.read_csv(L.V3_INPUTS['delist_official'][0], dtype=str)
DL = DL[(DL.four == 'True') & (DL.in_panel == 'True')]
DL['date'] = pd.to_datetime(DL['date'])
DL = DL[DL.date >= '2006-01-01'].drop_duplicates('code', keep='last')
names = dict(zip(DL.code, DL.name))

# 池內缺口
dm = L.disp_mask_fn('trade')
S = L.V4(YRD, Y5, dm)
dlmap = dict(zip(DL.code, DL.date))
rows = []
for dd in S.sig_days:
    cs = S.cross_section(dd)
    ok = cs['ok_other']; y3 = S.y3(dd)
    miss = ok & ~np.isfinite(y3)
    order = np.argsort(-np.where(np.isfinite(S.V60[cs['t']]), S.V60[cs['t']], -np.inf), kind='stable')
    rank = np.empty(K.N, int); rank[order] = np.arange(K.N)
    for j in np.where(miss)[0]:
        c = K.COLS[j]
        rows.append(dict(sig=str(dd.date()), code=c, delisted=c in dlmap and dd <= dlmap[c], liq_rank=int(rank[j]), pool=int(ok.sum()),
                         s52=float(cs['s52'][j]), s52_pct=float(pd.Series(np.where(ok, cs['s52'], np.nan)).rank(pct=True).iloc[j])))
G = pd.DataFrame(rows)
G.to_csv(HERE / 'gap_pool_stockmonths.csv', index=False)

# 逐檔下市股
out = []
for c, dt in zip(DL.code, DL.date):
    j = K.CI.get(c)
    tr = np.where(K.TRADED[:, j])[0] if j is not None else []
    if len(tr) == 0: continue
    p0 = max(pd.Period(K.IDX[tr[0]], 'M'), pd.Period('2005-01', 'M')); p1 = pd.Period(min(K.IDX[tr[-1]], dt), 'M')
    ps = pd.period_range(p0, p1, freq='M')
    # 該月營收在下月 10 日前公告；最後交易月的營收通常在下市後才公告 → 只算到最後交易月的前一個月
    ps = ps[ps < p1]
    miss = [p for p in ps if not has_level(c, p)]
    ymiss = [p for p in ps if not (c in YRD.columns and p in YRD.index and np.isfinite(YRD.at[p, c]))]
    g = G[(G.code == c) & G.delisted]
    lastlv = max([p for p in ps if has_level(c, p)], default=None)
    out.append(dict(code=c, name=names.get(c, ''), delist=str(dt.date()), first_m=str(p0), last_m=str(p1), months=len(ps),
                    miss_level=len(miss), miss_yoy=len(ymiss), last_level=str(lastlv) if lastlv else '',
                    miss_first=str(miss[0]) if miss else '', miss_last=str(miss[-1]) if miss else '',
                    pool_gap_sigs=int(len(g)), pool_gap_first=g.sig.min() if len(g) else '', pool_gap_last=g.sig.max() if len(g) else '',
                    pool_gap_s52pct_mean=float(g.s52_pct.mean()) if len(g) else np.nan))
O = pd.DataFrame(out).sort_values(['pool_gap_sigs', 'miss_level'], ascending=False)
O.to_csv(HERE / 'gap_delisted.csv', index=False)
summ = {'n_delisted_2006plus_in_panel': int(len(O)), 'n_with_any_missing_level': int((O.miss_level > 0).sum()),
        'n_all_missing': int((O.miss_level == O.months).sum()), 'n_missing_ge3': int((O.miss_level >= 3).sum()),
        'n_with_pool_gap': int((O.pool_gap_sigs > 0).sum()), 'pool_gap_stockmonths_delisted': int(G.delisted.sum()),
        'pool_gap_stockmonths_all': int(len(G)),
        'pool_gap_by_year_delisted': G[G.delisted].groupby(G.sig.str[:4]).size().to_dict(),
        'pool_gap_by_year_survivor_or_after': G[~G.delisted].groupby(G.sig.str[:4]).size().to_dict()}
print(json.dumps(summ, ensure_ascii=False, indent=1))
print(O.head(60).to_string())
(HERE / 'explore_gap_summary.json').write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding='utf-8')
