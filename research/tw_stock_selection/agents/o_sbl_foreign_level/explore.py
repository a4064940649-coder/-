#!/usr/bin/env python3
"""探索（只看挑選期：訊號日 2021-10-12～2023-12）。H3b 的描述統計；H3a 在挑選期沒有任何資料（借券 2026-07 起、融券 2024-01 起）。

輸出 explore.json。
"""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import sbl_lib as L

K, V, IDX, COLS, N = L.K, L.V, L.IDX, L.COLS, L.N
S = L.strategy(True)
SD = [d for d in S.sig_days if L.SEL[0] <= d <= L.SEL[1]]
ALLSD = S.sig_days + [S.end_day]
assert max(SD) <= pd.Timestamp('2023-12-31')

def next_exec(d):
    i = ALLSD.index(d)
    nd = ALLSD[i + 1]
    return IDX.get_loc(nd) + 1 if nd != S.end_day else IDX.get_loc(nd)

rows = []
for d in SD:
    cs = S.cross_section(d); el = cs['el']; t, e = cs['t'], cs['e']
    e2 = next_exec(d)
    f = L.fo_features(t)
    idx = np.where(el)[0]
    r = L.open_to_open(e, e2, idx, to_open=True)
    g = L.terciles(f['mcap'], el)
    # 近似 0050 成分：上市全體近似市值前 50
    mc_all = np.where(np.isfinite(f['mcap']) & K.TRADED[t], f['mcap'], -np.inf)
    top50 = np.zeros(N, bool); top50[np.argsort(-mc_all)[:50]] = True
    pr = lambda x: pd.Series(np.where(el, x, np.nan)).rank(pct=True).to_numpy()
    s52p, y3p = pr(cs['s52']), pr(cs['y3'])
    for k, j in enumerate(idx):
        rows.append(dict(d=d, code=COLS[j], j=j, r=r[k], lvl=f['lvl'][j], d6=f['d6'][j], d12=f['d12'][j], mcap=f['mcap'][j],
                         g=g[j], top50=top50[j], s52p=s52p[j], y3p=y3p[j], sc=cs['sc'][j]))
D = pd.DataFrame(rows)
D['logmc'] = np.log(D.mcap)
out = {'period': f'{SD[0].date()}～{SD[-1].date()}（{len(SD)} 個訊號日）', 'pool_size_mean': round(float(D.groupby('d').size().mean()), 1),
       'note': '報酬＝成交日開盤 → 下一成交日開盤（無成本）；外資持股用訊號日前一交易日；2025-10 前為回推值'}

def spread(D, col, q=5, within='g', min_n=10):
    """每月：在 within 組內依 col 分 q 組，最高組 − 最低組；組間平均。回傳月序列與各組均值。"""
    res = []; per_g = {0: [], 1: [], 2: []}; qm = []
    for d, x in D.groupby('d'):
        x = x[np.isfinite(x[col]) & np.isfinite(x.r)]
        sp = []; qq = np.zeros(q); cnt = 0
        for gg, y in x.groupby(within):
            if gg < 0 or len(y) < min_n: continue
            b = pd.qcut(y[col].rank(method='first'), q, labels=False)
            m = y.r.groupby(b).mean()
            s = m.iloc[-1] - m.iloc[0]
            sp.append(s); per_g[gg].append(s); qq += m.to_numpy(); cnt += 1
        if sp: res.append(np.mean(sp)); qm.append(qq / cnt)
    return np.array(res), per_g, np.mean(qm, axis=0) if qm else None

E = {}
for col in ['lvl', 'd6', 'd12']:
    s, pg, qm = spread(D, col)
    E[f'H3b_{col}_within_size_Q5minusQ1'] = {'all': L.summ(s), 'small': L.summ(pg[0]), 'mid': L.summ(pg[1]), 'large': L.summ(pg[2]),
                                             'quintile_means%': [round(float(v) * 100, 3) for v in qm]}
    # 不分市值組
    D['_all'] = 0
    s2, _, qm2 = spread(D, col, within='_all')
    E[f'H3b_{col}_pool_Q5minusQ1_no_size_control'] = {'all': L.summ(s2), 'quintile_means%': [round(float(v) * 100, 3) for v in qm2]}
# 市值本身
s, _, qm = spread(D.assign(_all=0), 'logmc', within='_all')
E['size_Q5minusQ1'] = {'all': L.summ(s), 'quintile_means%': [round(float(v) * 100, 3) for v in qm]}
# 主策略分數本身（參考）
s, _, qm = spread(D.assign(_all=0), 'sc', within='_all')
E['main_score_Q5minusQ1'] = {'all': L.summ(s), 'quintile_means%': [round(float(v) * 100, 3) for v in qm]}
# 在 0050 近似成分股外／內
for lab, msk in [('non_top50', ~D.top50), ('top50', D.top50)]:
    s, _, _ = spread(D[msk].assign(_all=0), 'lvl', q=3 if lab == 'top50' else 5, within='_all', min_n=6)
    E[f'H3b_lvl_{lab}_top_minus_bottom'] = L.summ(s)
out['spreads'] = E

# 相關矩陣（每月 Spearman 的平均）
cols = ['lvl', 'd6', 's52p', 'y3p', 'logmc', 'top50']
cm = []
for d, x in D.groupby('d'):
    x = x[cols].astype(float).dropna()
    if len(x) > 30: cm.append(x.rank().corr().to_numpy())
out['spearman_avg'] = {'cols': cols, 'matrix': np.round(np.nanmean(cm, axis=0), 3).tolist()}
# 外資持股的時間持續性
pers = []
for i in range(len(SD) - 1):
    a = D[D.d == SD[i]].set_index('code').lvl; b = D[D.d == SD[i + 1]].set_index('code').lvl
    k = a.index.intersection(b.index)
    pers.append(a[k].rank().corr(b[k].rank()))
out['lvl_month_to_month_rank_corr'] = round(float(np.mean(pers)), 4)
out['lvl_distribution_pool_pct'] = {q: round(float(D.lvl.quantile(q)) * 100, 2) for q in [0.1, 0.25, 0.5, 0.75, 0.9]}
out['d6_distribution_pool_pp'] = {q: round(float(D.d6.quantile(q)) * 100, 2) for q in [0.1, 0.25, 0.5, 0.75, 0.9]}

# ---------------- P9：主策略右尾 ----------------
pl = [p for p in S.plan(10) if L.SEL[0] <= p[3] <= L.SEL[1]]
t_end = IDX.get_loc(S.end_day)
con, per = V.stock_contrib(S.plan(10), t_end, a=L.SEL[0], b=L.SEL[1])
top20 = sorted(con, key=con.get, reverse=True)[:20]
ent = []
for (d, pk, r) in per:
    t = IDX.get_loc(d); f = L.fo_features(t)
    cs = S.cross_section(d)
    for j, x in zip(pk, r):
        ent.append(dict(d=d, code=COLS[j], r=x, d6=f['d6'][j], lvl=f['lvl'][j],
                        pool_lvl_pct=float(pd.Series(np.where(cs['el'], f['lvl'], np.nan)).rank(pct=True).iloc[j])))
EN = pd.DataFrame(ent)
v = EN[np.isfinite(EN.d6)]
out['P9'] = {'n_entries_with_d6': int(len(v)), 'share_d6_up_all_entries': round(float((v.d6 > 0).mean()), 3),
             'share_d6_up_top20_contrib_entries': round(float((v[v.code.isin(top20)].d6 > 0).mean()), 3),
             'n_top20_entries': int(v.code.isin(top20).sum()),
             'top_decile_period_return_entries_share_d6_up': round(float((v[v.r >= v.r.quantile(0.9)].d6 > 0).mean()), 3),
             'mean_d6_pp_all': round(float(v.d6.mean()) * 100, 2), 'mean_d6_pp_top20': round(float(v[v.code.isin(top20)].d6.mean()) * 100, 2),
             'main_picks_pool_lvl_pct_mean': round(float(EN.pool_lvl_pct.mean()), 3),
             'top20_codes': top20[:20]}
# 主策略入選股的外資持股水準 vs 池子：入選股中外資高（池內百分位 ≥ 0.5）與低的期報酬
hi = EN[EN.pool_lvl_pct >= 0.5]; lo = EN[EN.pool_lvl_pct < 0.5]
mm = []
for d, x in EN.groupby('d'):
    a = x[x.pool_lvl_pct >= 0.5].r; b = x[x.pool_lvl_pct < 0.5].r
    if len(a) and len(b): mm.append(a.mean() - b.mean())
out['main_picks_hi_minus_lo_fo_lvl'] = L.summ(mm)
out['H3a'] = '挑選期（2021-10～2023-12）沒有任何借券賣出餘額或融券餘額存檔可用（借券 2026-07-30 起、融券 2024-01 起），探索無法進行；見 h3a_pilot.py（前瞻追蹤登記後的描述性試算）'
(L.HERE / 'explore.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
