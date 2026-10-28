#!/usr/bin/env python3
"""探索複製（挑選期，扣成本前，原版名單）：周轉拆解、舊持股本月排名分布、新進 vs 舊股下一窗報酬、容量。"""
import sys, json, pickle
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import r4_core as C
K = C.K
P = pickle.load(open(HERE / 'plans.pkl', 'rb'))
PL, sd, t0, t1 = P['PL'], P['sd'], P['t0'], P['t1']
S = C.make_S()
SA, SB = pd.Timestamp('2006-04-01'), pd.Timestamp('2018-12-31')
G, Cn, On, TR = K.GROW, K.Cn, K.On, K.TRADED
out = {}

# ---- 周轉拆解（gross 引擎重跑，記錄每檔）----
def decompose(plan_rows, drift_new_only=False):
    reb = {p['e']: p for p in plan_rows}
    val = np.zeros(K.N); cash = 1.0; rec = []
    for t in range(t0, t1 + 1):
        if t in reb:
            p = reb[t]; pk = p['picks']
            tr = TR[t] & np.isfinite(On[t])
            h = val * np.where(tr, G[t] * On[t] / np.where(tr, Cn[t], 1.0), 1.0)
            frz = (h > 0) & ~tr
            for j in np.where(frz)[0]:
                r = C.XR.get(j)
                if r is not None and np.searchsorted(C.IDX, r[1]) <= t < (np.searchsorted(C.IDX, r[3]) if len(r) > 3 else 10 ** 9):
                    cash += h[j] * r[0]; h[j] = 0; frz[j] = False
            htr = np.where(frz, 0, h); Vt = htr.sum() + cash; Vall = Vt + h[frz].sum()
            inpk = np.zeros(K.N, bool); inpk[pk] = True
            held = htr > 0
            if drift_new_only:
                # 不調回：續抱股保持原市值，出場所得＋現金平均分給新進股
                cont = inpk & held
                newm = inpk & ~held
                tgt = np.where(cont, htr, 0.0)
                if newm.any(): tgt[newm] = (Vt - tgt.sum()) / newm.sum()
                else: tgt = np.where(cont, htr * Vt / htr[cont].sum(), 0.0)
            else:
                tgt = np.where(inpk, Vt / len(pk), 0.0)
            sell_exit = htr[held & ~inpk].sum() / Vall
            sell_trim = np.clip(htr - tgt, 0, None)[held & inpk].sum() / Vall
            buy_new = tgt[inpk & ~held].sum() / Vall
            buy_top = np.clip(tgt - htr, 0, None)[held & inpk].sum() / Vall
            rec.append((p['d'], sell_exit, sell_trim, buy_new, buy_top))
            nv = np.where(frz, h, tgt)
            val = np.where(frz, nv, nv * np.where(tr, Cn[t] / np.where(tr, On[t], 1.0), 1.0)); cash = 0
        elif t > t0:
            val = val * G[t]
    df = pd.DataFrame(rec, columns=['d', 'sell_exit', 'sell_trim', 'buy_new', 'buy_top']).set_index('d').iloc[1:]
    return (df[(df.index >= SA) & (df.index <= SB)].mean() * 100).to_dict()
out['decomp_base'] = decompose(PL['BASE'])
out['decomp_base_nodrift'] = decompose(PL['BASE'], drift_new_only=True)
print('decomp', out['decomp_base'], 'nodrift', out['decomp_base_nodrift'], flush=True)

# ---- 下一窗每檔報酬（成交日開盤 → 下一成交日開盤）----
rows = PL['BASE']
def fwd(i, js):
    t = rows[i]['e']; t2 = rows[i + 1]['e'] if i + 1 < len(rows) else None
    if t2 is None: return np.full(len(js), np.nan)
    js = np.asarray(js, int)
    r = Cn[t, js] / On[t, js] * np.prod(G[t + 1:t2 + 1, js], axis=0)
    trd = TR[t2, js] & np.isfinite(On[t2, js])
    r = np.where(trd, r * On[t2, js] / np.where(trd, Cn[t2, js], 1), r)
    return r - 1
age = {}
recs = []
for i, p in enumerate(rows):
    d = p['d']
    newage = {}
    for j in p['picks']: newage[int(j)] = age.get(int(j), 0) + 1
    if SA <= d <= SB and i > 0:
        # 舊持股：上月持股；只能用當期可成交者算報酬
        newj = [j for j in p['picks'] if int(j) not in set(p['held_prev'])]
        rn = fwd(i, newj) if newj else np.array([])
        for j, r in zip(newj, rn): recs.append(dict(d=d, j=int(j), kind='new', rank=p['rank'].get(int(j)), s52=p['s52'][j], age=0, r=r))
        oldj = list(p['held_prev'])
        okj = [j for j in oldj if TR[p['e'], j] and np.isfinite(On[p['e'], j])]
        ro = dict(zip(okj, fwd(i, okj))) if okj else {}
        for j in oldj:
            recs.append(dict(d=d, j=int(j), kind='old', rank=p['rank'].get(int(j)), el=bool(p['el'][j]), s52=p['s52'][j], age=age.get(int(j), 0), r=ro.get(j, np.nan)))
    age = newage
R = pd.DataFrame(recs)
old = R[R.kind == 'old']
def bucket(x):
    if not np.isfinite(x['rank']) if x['rank'] is not None else True: return 'inel'
    r = x['rank']
    return '1-10' if r <= 10 else '11-15' if r <= 15 else '16-20' if r <= 20 else '21-30' if r <= 30 else '31-50' if r <= 50 else '51+'
old = old.assign(rank=old['rank'].astype(float))
old['b'] = [('inel' if not np.isfinite(r) else '1-10' if r <= 10 else '11-15' if r <= 15 else '16-20' if r <= 20 else '21-30' if r <= 30 else '31-50' if r <= 50 else '51+') for r in old['rank']]
out['old_rank_dist'] = (old['b'].value_counts(normalize=True) * 100).round(1).to_dict()
print('rank dist', out['old_rank_dist'], flush=True)
new = R[R.kind == 'new']
nm = new.groupby('d')['r'].mean()
out['new_mean_pct'] = float(new['r'].mean() * 100)
out['new_mean_monthly_pct'] = float(nm.mean() * 100)
def grp(mask, lab):
    g = old[mask & old['r'].notna()]
    gm = g.groupby('d')['r'].mean()
    dd = (gm - nm.reindex(gm.index)).dropna()
    out.setdefault('groups', {})[lab] = dict(pp=float(dd.mean() * 100), t=C.tstat(dd), months=int(len(dd)), n=int(len(g)))
rk = old['rank']; s5 = old['s52']
grp(rk <= 10, 'top10')
grp((rk > 10) & (rk <= 30) & (s5 >= 0.90), 'P2a_11-30_s>=0.90')
grp((rk > 10) & (rk <= 30) & (s5 >= 0.95), '11-30_s>=0.95')
grp((rk > 10) & (rk <= 30) & (s5 >= 0.90) & (s5 < 0.95), '11-30_0.90-0.95')
grp((s5 < 0.85), 's<0.85_all')
grp((s5 < 0.85) & ~(rk <= 10), 's<0.85_nottop10')
grp((rk > 30), 'rank31+')
grp((rk > 30) | rk.isna(), 'rank31+_or_inel')
for a_, lab in ((1, 'age2'), (2, 'age3')): grp(old['age'] == a_, lab + '(age==%d)' % a_)
grp(old['age'] >= 3, 'age4+')
for a_ in (2, 3): grp(old['age'] == a_, 'age==%d' % a_)
grp(old['age'] >= 4, 'age>=4')
print(json.dumps(out['groups'], indent=0), flush=True)

# ---- 容量：新買名額部位（1,000 萬 / 10）> 60 日均成交值 1% ----
def cap(v):
    o = {}
    for per in ('SEL', 'POST', 'ALL'):
        a, b = map(pd.Timestamp, C.PER[per])
        flags = []; nnew = []
        for i, p in enumerate(PL[v]):
            if not (a <= p['d'] <= b) or i == 0: continue
            v60 = S.V60[p['e'] - 1]
            nn = [j for j in p['picks'] if int(j) not in set(p['held_prev'])]
            nnew.append(len(nn))
            flags += [1e6 > 0.01 * v60[j] for j in nn]
        o[per] = dict(frac=float(np.mean(flags) * 100), new_per_month=float(np.mean(nnew)))
    return o
out['capacity'] = {v: cap(v) for v in ('BASE', 'A20')}
print('capacity', out['capacity'])
(HERE / 'r_explore.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
