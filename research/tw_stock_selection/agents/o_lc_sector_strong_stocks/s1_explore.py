#!/usr/bin/env python3
"""探索（只用挑選期 SEL：訊號窗 2014-01～2016-12，終點截在 2016-12-30）。D1：2006～2013 沒有可核對的 PIT 股數（o8 閘門），不進探索。
輸出 explore.json。python3 -I -B s1_explore.py"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import lc_lib as X, lc_run as Rn

PER = 'SEL'
MODE = sys.argv[1] if len(sys.argv) > 1 else 'COMP'; X.set_mode(MODE)
out = {'period': X.PERIODS[PER], 'mode': MODE, 'mode_params': X.mode_params(), 'note': '只用挑選期；2017 年以後未讀取報酬'}

# ---------- 宇宙與產業描述 ----------
desc = []
for d in X.days_for(0, PER)[:-1]:
    cs = X.cs_cached(d); ST = cs['ST']
    desc.append(dict(n_U=len(cs['U']), n_sec_ge2=len(ST), w2330=float(cs['mc'][0] / cs['mc'].sum()) if X.COLS[cs['U'][0]] == '2330' else np.nan,
                     n_fin=int(cs['fin'].sum()), score_na=int(np.isnan(cs['score']).sum()), y3_na_nonfin=int((np.isnan(cs['y3']) & ~cs['fin']).sum()),
                     sec_sizes=sorted([len(v['members']) for v in cs['secs'].values()], reverse=True), top5=list(ST.index[:5])))
D = pd.DataFrame(desc)
out['universe'] = dict(n_U_mean=float(D.n_U.mean()), n_sec_ge2_mean=float(D.n_sec_ge2.mean()), n_sec_ge2_min=int(D.n_sec_ge2.min()),
                       w2330_mean=float(D.w2330.mean()), n_fin_mean=float(D.n_fin.mean()), score_na_mean=float(D.score_na.mean()),
                       y3_na_nonfin_mean=float(D.y3_na_nonfin.mean()),
                       median_sector_sizes=[float(np.median([s[i] if i < len(s) else 0 for s in D.sec_sizes])) for i in range(12)],
                       share_months_two_member_sector_in_top5=None)
from collections import Counter
cnt = Counter(s for t in D.top5 for s in t)
out['top5_sector_freq_j0'] = {k: v for k, v in cnt.most_common()}

# ---------- Q1 差中差與 IC（每 j；月份為單位）----------
def q1(j, n_split='half'):
    days = X.days_for(j, PER); SR = X.stock_win_ret(days)
    rows = []
    for k, d in enumerate(days[:-1]):
        cs = X.cs_cached(d); ST = cs['ST']; r = SR[k][cs['U']]
        if len(ST) < 10: continue
        KK = X.mode_params()['k']; top = list(ST.index[:KK]); bot = list(ST.index[-KK:])
        def spread(secs):
            diffs = []; ics = []
            for s in secs:
                m = [i for i in cs['secs'][s]['members'] if np.isfinite(cs['score'][i]) and np.isfinite(r[i])]
                if len(m) < 2: continue
                o = X.stock_order(cs, m); h = len(o) // 2
                hi, lo = o[:max(h, 1)], o[-max(h, 1):]
                diffs.append(np.mean(r[hi]) - np.mean(r[lo]))
                if len(m) >= 3: ics.append(pd.Series(cs['score'][m]).corr(pd.Series(r[m]), method='spearman'))
            return (np.mean(diffs) if diffs else np.nan), (np.nanmean(ics) if ics else np.nan)
        dt, it = spread(top); db, ib = spread(bot)
        mid = list(ST.index[KK:-KK]); dm, im = spread(mid)
        rows.append(dict(d=d, top=dt, bot=db, mid=dm, did=dt - db, ic_top=it, ic_bot=ib, ic_mid=im))
    return pd.DataFrame(rows)
Q1 = {}
for j in Rn.JS:
    q = q1(j)
    Q1[str(j)] = dict(n=len(q), top_pp=float(q.top.mean() * 100), bot_pp=float(q.bot.mean() * 100), mid_pp=float(q.mid.mean() * 100),
                      did_pp=float(q.did.mean() * 100), t_did=X.tstat(q.did), ic_top=float(q.ic_top.mean()), ic_mid=float(q.ic_mid.mean()), ic_bot=float(q.ic_bot.mean()))
out['Q1_did'] = dict(per_j=Q1, did_pp_jmed=float(np.median([v['did_pp'] for v in Q1.values()])), t_did_jmed=float(np.median([v['t_did'] for v in Q1.values()])),
                     ic_top_jmed=float(np.median([v['ic_top'] for v in Q1.values()])), ic_bot_jmed=float(np.median([v['ic_bot'] for v in Q1.values()])))

# ---------- 格點（紙上引擎，一般成本與零成本）----------
G = {}
for sp in Rn.GRID + Rn.REFS:
    nm = Rn.name(sp); xs = {}; xsB = {}; xg = {}; to = []; conc = []; c2330 = []; secto = []
    for j in Rn.JS:
        o = Rn.run(sp, j, PER, 'normal'); oz = Rn.run(sp, j, PER, 'zero'); oB = Rn.run(dict(kind='B', carrier='PURE'), j, PER, 'normal')
        xs[j] = o['x']; xsB[j] = o['r'] - oB['r']; xg[j] = oz['x']; to.append(o['to'].mean())
        for (idx, w) in o['plans']:
            ws = np.sort(w)[::-1]; conc.append(ws[:5].sum())
            c2330.append(float(w[idx == X.CI['2330']].sum()))
        sl = [tuple(i.get('sectors', [])) for i in o['info']]
        if sl and sl[0]: secto.append(np.mean([len(set(a) - set(b)) / max(len(a), 1) for a, b in zip(sl[1:], sl[:-1])]))
    G[nm] = dict(spec=sp, vs0050=Rn.summ_j(xs), vsB=Rn.summ_j(xsB), gross_vs0050=Rn.summ_j(xg), turnover_1side_mean=float(np.mean(to)),
                 top5_weight_mean=float(np.mean(conc)), w2330_mean=float(np.mean(c2330)), sector_turnover=float(np.mean(secto)) if secto else None)
    for k in ('vs0050', 'vsB', 'gross_vs0050'): G[nm][k].pop('per_j')
    print(nm, round(G[nm]['vs0050']['IR_jmed'], 3), round(G[nm]['vs0050']['t_jmed'], 2), round(G[nm]['vsB']['t_jmed'], 2), round(G[nm]['turnover_1side_mean'], 3), flush=True)
out['grid_SEL'] = G

# ---------- 2330 被選中比例、與主策略名單重疊 ----------
pj = pd.read_csv(X.AG / 'o_v7_listing_exec_engine' / 'data' / 'plans_j7.csv', dtype={'picks': str})
pj = pj[(pj.strat == 'AB_V7_B10') & (pj.seed == 0)]
ov = {}
for sp in [Rn.DEFAULT, dict(kind='CTRL', n=2, wt='cap', cap=0.15, carrier='SAT')]:
    vals = []; sel2330 = []
    for j in Rn.JS:
        o = Rn.run(sp, j, PER, 'normal'); days = o['days']
        g = pj[pj.j == f'J{j}'].copy(); g['signal'] = pd.to_datetime(g['signal'])
        mp = dict(zip(g.signal, g.picks))
        s = make = Rn.make(sp)
        for d in days[:-1]:
            cs = X.cs_cached(d); idx, w, info = s.sleeve(cs)
            codes = set(X.COLS[cs['U'][i]] for i in idx)
            sel2330.append('2330' in codes)
            if d in mp: vals.append(len(codes & set(mp[d].split(','))) / max(len(codes), 1))
    ov[Rn.name(sp)] = dict(overlap_with_main_mean=float(np.mean(vals)) if vals else None, n=len(vals), share_months_2330_selected=float(np.mean(sel2330)))
out['overlap_main_and_2330'] = ov

# ---------- 小產業（成員 2 家）內的選股是否等於隨機 ----------
two = []
for j in Rn.JS:
    days = X.days_for(j, PER); SR = X.stock_win_ret(days)
    for k, d in enumerate(days[:-1]):
        cs = X.cs_cached(d); r = SR[k][cs['U']]
        for s in list(cs['ST'].index[:X.mode_params()['k']]):
            m = [i for i in cs['secs'][s]['members'] if np.isfinite(cs['score'][i]) and np.isfinite(r[i])]
            if len(m) == 2:
                o = X.stock_order(cs, m); two.append(r[o[0]] - r[o[1]])
            elif len(m) >= 3:
                pass
out['two_member_sectors'] = dict(n_obs=len(two), mean_pp=float(np.mean(two) * 100) if two else None, t_naive=X.tstat(two) if two else None,
                                 note='跨 j 重疊觀測，t 只作描述')
out['drawdown_2008_2011'] = '挑選期 2014～2016 沒有 2008、2011 空頭；2006～2013 非 PIT 股數段只在凍結後作描述'
(Path(__file__).resolve().parent / f'explore_{MODE}.json').write_text(X.dumps(out), encoding='utf-8')
print('Q1', out['Q1_did']['did_pp_jmed'], out['Q1_did']['t_did_jmed'])
print(json.dumps(out['universe'], ensure_ascii=False, default=str))
print(json.dumps(ov, ensure_ascii=False))
print(out['two_member_sectors'])
