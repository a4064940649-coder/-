#!/usr/bin/env python3
"""探索（只用挑選期：訊號日 2006-01～2018-12，扣成本前為主）。輸出 explore.json。"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import hb_core as H
res = json.loads((H.HERE / 'results.json').read_text(encoding='utf-8'))
assert 'PLAN_sha256' in res and 'PREREG' not in res
K, V, R = H.K, H.V, H.R
st = H.setup(); S = H.make_S(); P = H.Prep(S)
days = [d for d in P.days if d <= H.SEL_B]
end_day = [d for d in P.days if d > H.SEL_B][0]
t0, t1, sd = H.bounds(days, end_day)
XR = st['XR']
out = {'period': f'{days[0].date()}..{days[-1].date()}（評價到 {end_day.date()} 收盤）', 'n_windows': len(days)}

# ---------- (1) 周轉拆解 ----------
pl = H.planner(P, 10, days=days)
for cname, cost in (('gross', (0.0, 0.0)), ('normal', H.NORMAL)):
    nav, lg = H.sim([(p[0], p[1]) for p in pl], *cost, t0, t1, xrules=XR)
    det = pd.DataFrame(lg['det'][1:])
    o = {k: float(det[k].mean()) for k in ('sell', 'buy', 'sell_exit', 'sell_trim', 'buy_new', 'buy_top', 'n_new', 'n_cont')}
    o['oneway_avg_(buy+sell)/2'] = float(((det.buy + det.sell) / 2).mean())
    o['reweight_part_(trim+top)/2'] = float(((det.sell_trim + det.buy_top) / 2).mean())
    o['slot_part_(exit+new)/2'] = float(((det.sell_exit + det.buy_new) / 2).mean())
    o['turnover_engine'] = lg['turnover']
    out[f'turnover_decomp_{cname}'] = o
    # 假想：續抱股完全不調回（band=∞）
    nav2, lg2 = H.sim([(p[0], p[1]) for p in pl], *cost, t0, t1, xrules=XR, band=1e9)
    out[f'turnover_noreweight_{cname}'] = lg2['turnover']
print('(1)', out['turnover_decomp_normal'], out['turnover_noreweight_normal'], flush=True)

# ---------- 每窗報酬（成交日開盤 → 下一成交日開盤；扣成本前；停牌者以最後價值持有） ----------
G = K.GROW
ex = [P.cs[d]['e'] for d in days] + [IDX_e for IDX_e in [P.cs[end_day]['e']]]
def win_ret(i):
    e, e2 = ex[i], ex[i + 1]
    r = np.where(K.TRADED[e] & np.isfinite(K.On[e]), K.Cn[e] / np.where(np.isfinite(K.On[e]), K.On[e], 1.0), np.nan)
    r = r * np.prod(G[e + 1:e2 + 1], axis=0)
    trd = K.TRADED[e2] & np.isfinite(K.On[e2])
    r = np.where(trd, r * K.On[e2] / np.where(trd, K.Cn[e2], 1.0), r)
    return r - 1.0
WR = [win_ret(i) for i in range(len(days))]

# ---------- (2) 轉移矩陣、(3) 分桶 ----------
def rb(r):
    if r is None: return 'inelig'
    r1 = r + 1
    return '01-10' if r1 <= 10 else '11-15' if r1 <= 15 else '16-20' if r1 <= 20 else '21-30' if r1 <= 30 else '31-50' if r1 <= 50 else '51+'
def hb(x):
    if not np.isfinite(x): return 'na'
    return '>=0.95' if x >= 0.95 else '0.90-0.95' if x >= 0.90 else '0.85-0.90' if x >= 0.85 else '<0.85'
trans = {}
rows = []
age = {}      # 連續持有月數
for i, d in enumerate(days):
    c = P.cs[d]; picks = set(pl[i][1].tolist())
    prev = set(pl[i - 1][1].tolist()) if i > 0 else set()
    sy = H.single_yoy(S, d)
    el = c['el']
    q = np.nanquantile(np.where(el, sy, np.nan), 2 / 3) if np.isfinite(np.where(el, sy, np.nan)).sum() > 30 else np.nan
    newage = {}
    for j in picks: newage[j] = age.get(j, 0) + 1
    for j in prev:
        r = c['rank'].get(j)
        trans[rb(r)] = trans.get(rb(r), 0) + 1
    for j in range(K.N):
        pass
    # 新進（本月原版前 10 且上月不持有）
    for j in picks - prev:
        rows.append(dict(i=i, d=d, j=j, grp='new', rank=c['rank'][j] + 1, s52=c['s52'][j], top3=bool(sy[j] >= q) if np.isfinite(sy[j]) else None,
                         r=WR[i][j], age=1))
    for j in picks & prev:
        rows.append(dict(i=i, d=d, j=j, grp='cont', rank=c['rank'][j] + 1, s52=c['s52'][j], top3=bool(sy[j] >= q) if np.isfinite(sy[j]) else None,
                         r=WR[i][j], age=newage[j]))
    for j in prev - picks:
        r = c['rank'].get(j)
        tr = K.TRADED[c['e'], j] and np.isfinite(K.On[c['e'], j])
        rows.append(dict(i=i, d=d, j=j, grp='old_out', rank=(r + 1) if r is not None else None, s52=c['s52'][j],
                         top3=bool(sy[j] >= q) if np.isfinite(sy[j]) else None, r=WR[i][j] if tr else np.nan, age=age.get(j, 0)))
    age = newage
df = pd.DataFrame(rows)
tot = sum(trans.values())
out['transition_prev_holdings_rank_now'] = {k: dict(n=v, share=v / tot) for k, v in sorted(trans.items())}
df['rb'] = [rb(None if not np.isfinite(x) else int(x) - 1) if x is not None else 'inelig' for x in df['rank'].astype(float)]
df['rb'] = np.where(df['rank'].isna(), 'inelig', df['rb'])
df['hb'] = [hb(x) for x in df['s52']]
df['rb4'] = df['rb'].replace({'31-50': '31+', '51+': '31+'})
newm = df[df.grp == 'new'].groupby('i')['r'].mean()

def cmp(mask, label):
    g = df[mask & df.r.notna()].groupby('i')['r'].mean()
    dd = (g - newm.reindex(g.index)).dropna()
    return dict(label=label, n_obs=int((mask & df.r.notna()).sum()), n_months=int(len(dd)), mean_pp=float(dd.mean() * 100) if len(dd) else None,
                t=R.tstat(dd.to_numpy()) if len(dd) >= 3 else None, grp_mean_pp=float(g.mean() * 100) if len(g) else None)
oo = df.grp == 'old_out'
tabs = {}
tabs['new_mean_pp'] = float(newm.mean() * 100)
tabs['cont_vs_new'] = cmp(df.grp == 'cont', '續抱（仍在前 10）')
for r_ in ('11-15', '16-20', '21-30', '31+', 'inelig'):
    tabs[f'rank_{r_}'] = cmp(oo & (df.rb4 == r_), f'舊股 排名 {r_}')
for h_ in ('>=0.95', '0.90-0.95', '0.85-0.90', '<0.85'):
    tabs[f'h_{h_}'] = cmp(oo & (df.rb4 != 'inelig') & (df.hb == h_), f'舊股（合格） 52週比 {h_}')
for r_ in ('11-15', '16-20', '21-30', '31+'):
    for h_ in ('>=0.95', '0.90-0.95', '0.85-0.90', '<0.85'):
        tabs[f'cell_{r_}_{h_}'] = cmp(oo & (df.rb4 == r_) & (df.hb == h_), f'{r_}×{h_}')
        for tp in (True, False):
            tabs[f'cell_{r_}_{h_}_top3={tp}'] = cmp(oo & (df.rb4 == r_) & (df.hb == h_) & (df.top3 == tp), f'{r_}×{h_}×營收前1/3={tp}')
in1130 = oo & df.rb4.isin(['11-15', '16-20', '21-30'])
tabs['P2a_rank11-30_h>=0.90'] = cmp(in1130 & (df.s52 >= 0.90), '排名 11～30 且 52 週比 ≥0.90')
tabs['P2b_rank11-30_h<0.85'] = cmp(in1130 & (df.s52 < 0.85), '排名 11～30 且 52 週比 <0.85')
tabs['P2b_all_old_h<0.85'] = cmp(oo & (df.rb4 != 'inelig') & (df.s52 < 0.85), '全部合格舊股且 52 週比 <0.85')
tabs['rank11-20_h>=0.90'] = cmp(oo & df.rb4.isin(['11-15', '16-20']) & (df.s52 >= 0.90), '排名 11～20 且 ≥0.90')
tabs['rank11-30_h>=0.95'] = cmp(in1130 & (df.s52 >= 0.95), '排名 11～30 且 ≥0.95')
tabs['rank11-30_h0.85-0.90'] = cmp(in1130 & (df.s52 >= 0.85) & (df.s52 < 0.90), '排名 11～30 且 0.85～0.90')
tabs['rank11-30_h>=0.90_top3'] = cmp(in1130 & (df.s52 >= 0.90) & (df.top3 == True), '排名 11～30、≥0.90、營收前 1/3')
tabs['rank11-30_h>=0.90_nottop3'] = cmp(in1130 & (df.s52 >= 0.90) & (df.top3 == False), '排名 11～30、≥0.90、營收非前 1/3')
tabs['rank21-30_h>=0.90'] = cmp(oo & (df.rb4 == '21-30') & (df.s52 >= 0.90), '排名 21～30 且 ≥0.90')
tabs['rank31+_h>=0.95'] = cmp(oo & (df.rb4 == '31+') & (df.s52 >= 0.95), '排名 31+ 且 ≥0.95')
out['bucket_vs_new'] = tabs
# 端點敏感度（方法教訓 5.3）：P2a／P2b／≥0.95 格
def sens(mask):
    g = df[mask & df.r.notna()].groupby('i')['r'].mean()
    dd = (g - newm.reindex(g.index)).dropna()
    o = {}
    for k in (1, 3, 5):
        o[f'drop_worst_{k}'] = R.tstat(dd.sort_values().iloc[k:].to_numpy())
        o[f'drop_best_{k}'] = R.tstat(dd.sort_values().iloc[:-k].to_numpy())
    w = dd.clip(dd.quantile(0.05), dd.quantile(0.95))
    o['winsor5_t'] = R.tstat(w.to_numpy()); o['winsor5_mean_pp'] = float(w.mean() * 100)
    o['median_pp'] = float(dd.median() * 100); o['share_months_neg'] = float((dd < 0).mean())
    # 年份去除
    yrs = pd.Series([days[i].year for i in dd.index], index=dd.index)
    o['drop_year_min_t'] = min((R.tstat(dd[yrs != y].to_numpy()), int(y)) for y in yrs.unique())
    o['drop_year_max_t'] = max((R.tstat(dd[yrs != y].to_numpy()), int(y)) for y in yrs.unique())
    # 個股層（不分群，只作對照）
    return o
out['P2_sensitivity'] = {'P2a_rank11-30_h>=0.90': sens(in1130 & (df.s52 >= 0.90)), 'P2b_all_old_h<0.85': sens(oo & (df.rb4 != 'inelig') & (df.s52 < 0.85)),
                         'rank11-30_h>=0.95': sens(in1130 & (df.s52 >= 0.95)), 'rank11-30_h0.90-0.95': sens(in1130 & (df.s52 >= 0.90) & (df.s52 < 0.95)),
                         'cont_top10': sens(df.grp == 'cont')}
tabs['rank11-30_h0.90-0.95'] = cmp(in1130 & (df.s52 >= 0.90) & (df.s52 < 0.95), '排名 11～30 且 0.90～0.95')
# 子期間
sub = {}
for lab, a, b in (('2006-2010', '2006-01-01', '2010-12-31'), ('2011-2014', '2011-01-01', '2014-12-31'), ('2015-2018', '2015-01-01', '2018-12-31')):
    m = (df.d >= a) & (df.d <= b)
    nm = df[m & (df.grp == 'new')].groupby('i')['r'].mean()
    def cm(mask):
        g = df[m & mask & df.r.notna()].groupby('i')['r'].mean(); dd = (g - nm.reindex(g.index)).dropna()
        return dict(n_months=int(len(dd)), mean_pp=float(dd.mean() * 100) if len(dd) else None, t=R.tstat(dd.to_numpy()) if len(dd) >= 3 else None)
    sub[lab] = {'rank11-30_h>=0.90': cm(in1130 & (df.s52 >= 0.90)), 'rank11-30_h<0.85': cm(in1130 & (df.s52 < 0.85)), 'cont': cm(df.grp == 'cont')}
out['bucket_subperiods'] = sub
# 依連續持有月數（原版持股）
agt = {}
inp = df.grp.isin(['new', 'cont'])
for a_ in (1, 2, 3, 4):
    msk = inp & ((df.age == a_) if a_ < 4 else (df.age >= 4))
    g = df[msk & df.r.notna()].groupby('i')['r'].mean()
    dd = (g - newm.reindex(g.index)).dropna()
    agt[f'age_{a_}{"+" if a_ == 4 else ""}'] = dict(n_obs=int(msk.sum()), mean_pp=float(g.mean() * 100), vs_new_pp=float(dd.mean() * 100) if a_ > 1 else 0.0,
                                                   t_vs_new=R.tstat(dd.to_numpy()) if a_ > 1 else None)
out['by_holding_age'] = agt
# 舊股整體：所有掉出者 vs 新進
out['old_out_all_vs_new'] = cmp(oo, '全部掉出舊股')
# 舊股 52 週比分布
x = df[oo & (df.rb4 != 'inelig')]
out['old_out_eligible_s52_quantiles'] = {str(q): float(x.s52.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
out['old_out_count_by_rb4_hb'] = {f'{a}|{b}': int(v) for (a, b), v in df[oo].groupby(['rb4', 'hb']).size().items()}

# ---------- (4) 容量 ----------
V60 = H._V60()
slots_new = []; slots_all = []
for i, d in enumerate(days):
    t = P.cs[d]['e'] - 1
    prev = set(pl[i - 1][1].tolist()) if i > 0 else set()
    for j in pl[i][1]:
        v = V60[t, j]
        over = bool(1e6 > 0.01 * v)
        slots_all.append(over)
        if j not in prev: slots_new.append(over)
out['capacity_10m'] = dict(share_new_slots_over_1pct_adv=float(np.mean(slots_new)), share_all_slots_over_1pct_adv=float(np.mean(slots_all)),
                           n_new=len(slots_new), n_all=len(slots_all), note='部位 100 萬（1,000 萬 ÷ 10），對訊號日 60 日均成交值（gapfix 窗）')
(H.HERE / 'explore.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
df.to_csv(H.HERE / 'explore_rows.csv', index=False)
for k, v in tabs.items():
    if not k.startswith('cell_'): print(k, v)
print(json.dumps({k: out[k] for k in ('P2_sensitivity', 'transition_prev_holdings_rank_now', 'by_holding_age', 'capacity_10m', 'bucket_subperiods')}, ensure_ascii=False, default=float))
