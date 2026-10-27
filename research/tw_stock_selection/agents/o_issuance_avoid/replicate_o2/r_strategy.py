#!/usr/bin/env python3
"""複製員：主策略（main_v2 V2：gapfix、排除金融、正式次排序）原版 vs V1（近 90 日現增／CB 事件跳過、已持有續抱）。"""
import sys, json, math
from pathlib import Path
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parent.parent / 'main_v2'))
import numpy as np, pandas as pd
import v2_core as V
sys.path.pop(0)
K = V.K
IDX, COLS, N = K.IDX, K.COLS, K.N
WIN = int(sys.argv[1]) if len(sys.argv) > 1 else 90
TYPES = sys.argv[2].split(',') if len(sys.argv) > 2 else ['SEO', 'CB']

import os
EVF = os.environ.get('EVF', str(H / 'my_events.csv'))
E = pd.read_csv(EVF, dtype={'code': str}, parse_dates=['recv', 'eff', 'list_date', 'anchor'])
E = E[E.etype.isin(TYPES) & E.code.isin(K.CI)]
E['known'] = E['anchor'] if os.environ.get('EVF') else E['recv'].fillna(E['list_date'])
EV = {c: np.sort(g['known'].values.astype('datetime64[ns]')) for c, g in E.groupby('code')}

def event_mask(d, win=WIN):
    m = np.zeros(N, bool)
    lo = np.datetime64(pd.Timestamp(d) - pd.Timedelta(days=win)); hi = np.datetime64(pd.Timestamp(d))
    for c, arr in EV.items():
        i = np.searchsorted(arr, lo)
        if i < len(arr) and arr[i] <= hi: m[K.CI[c]] = True
    return m

YL, Y5, Y5f = V.load_revenue()
S = V.V2('gapfix', YL, Y5, exfin=True)
t0, t1, sd = S.bounds()

def plan_avoid(n, avoid=True):
    out = []; held = set(); skipped = []
    for d in S.sig_days:
        cs = S.cross_section(d); el = cs['el']
        if el.sum() < 50: continue
        e = cs['e']; gu = K.gap_up_open(e)
        em = event_mask(d) if avoid else np.zeros(N, bool)
        cand = S.order(cs, 'official')[:int(el.sum())]
        pick = []
        for j in cand:
            if gu[j] and j not in held: continue
            if em[j] and j not in held: skipped.append((d, K.COLS[j])); continue
            pick.append(j)
            if len(pick) == n: break
        out.append((e, np.array(pick, int), np.where(el)[0], d))
        held = set(pick)
    return out, skipped

COST = {'normal': K.rates(0.001), 'stress': K.rates(0.004), 'stress_ia': K.rates('stress')}
SEG = {'sel': ('2006-04-01', '2019-04-30', '2019-03-31'), 'oos': ('2019-05-01', '2026-10-12', '2026-10-12'),
       'oosA': ('2019-05-01', '2023-12-31', '2023-12-31'), 'oosB': ('2024-01-01', '2026-10-12', '2026-10-12'),
       'ALL': ('2006-04-01', '2026-10-12', '2026-10-12')}

def winret(nav):
    D = pd.DatetimeIndex([d for d in sd if d in nav.index])
    x = nav.reindex(D)
    return (x.shift(-1) / x - 1).iloc[:-1]

def tst(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))

res = {'win': WIN, 'types': TYPES}
pl0 = S.plan(10, 'official')
ew, _ = K.simulate([(p[0], p[2]) for p in pl0], 0, 0, t0, t1)
res['bench'] = {k: {'E50': K.stats(ew, a, b, sd, K.E50)['0050_CAGR'] * 100, 'EW': K.stats(ew, a, b, sd)['CAGR'] * 100} for k, (a, b, _) in SEG.items()}
print('bench', res['bench'], flush=True)
for n in (10, 30):
    po, _ = plan_avoid(n, False)
    p1, skipped = plan_avoid(n, True)
    assert all((a[1] == b[1]).all() for a, b in zip(po, S.plan(n, 'official')))
    r = res.setdefault(f'n{n}', {})
    for c, (bb, ss) in COST.items():
        nav0, _ = K.simulate([(p[0], p[1]) for p in po], bb, ss, t0, t1)
        nav1, _ = K.simulate([(p[0], p[1]) for p in p1], bb, ss, t0, t1)
        w0, w1 = winret(nav0), winret(nav1)
        dif = (w1 - w0)
        rc = r.setdefault(c, {})
        for k, (a, b, bs) in SEG.items():
            s0 = V.seg_stats(nav0, ew, a, b, sd); s1 = V.seg_stats(nav1, ew, a, b, sd)
            dd = dif[(dif.index >= pd.Timestamp(a)) & (dif.index <= pd.Timestamp(bs))].dropna()
            rc[k] = {'CAGR0': s0['CAGR'], 'CAGR1': s1['CAGR'], 'dCAGR': s1['CAGR'] - s0['CAGR'], 'maxDD0': s0['maxDD'], 'maxDD1': s1['maxDD'],
                     't0050_0': s0['t_0050'], 't0050_1': s1['t_0050'], 'pair_mean_pp': dd.mean() * 100, 'pair_sd_pp': dd.std(ddof=1) * 100,
                     'pair_t': tst(dd), 'n_m': len(dd)}
            if k == 'ALL':
                yr = dd.groupby(dd.index.year).sum()
                best = int(yr.idxmax())
                rc[k]['best_year'] = best; rc[k]['pair_mean_ex_best_pp'] = dd[dd.index.year != best].mean() * 100
                # 依 V1 原版年化最好年份
                y0 = (1 + w1).groupby(w1.index.year).prod() - 1
                rc[k]['best_year_v1'] = int(y0.idxmax()); rc[k]['pair_mean_ex_bestv1_pp'] = dd[dd.index.year != int(y0.idxmax())].mean() * 100
        print(n, c, {k: {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in rc.items()}, flush=True)
        if c == 'normal':
            r['_dif'] = dif
    # 受影響：V1 建名單時被跳過的股票月
    sk = pd.DataFrame(skipped, columns=['d', 'code'])
    r['skipped_total'] = len(sk)
    r['skipped_sel'] = int((sk.d <= pd.Timestamp('2019-03-31')).sum()); r['skipped_oos'] = int((sk.d >= pd.Timestamp('2019-05-01')).sum())
    # 名單差異股月（原版有、V1 沒有）
    diffm = 0
    for a, b in zip(po, p1):
        diffm += len(set(a[1]) - set(b[1]))
    r['orig_not_in_v1'] = diffm
    # 原版入選股中近 90 日有事件者（挑選期）與下月相對報酬
    con, per = V.stock_contrib(po, t1)
    xs = []; tot = 0; nev = 0
    for (d, pk, rr) in per:
        if not (pd.Timestamp('2006-04-01') <= d <= pd.Timestamp('2019-03-31')): continue
        em = event_mask(d)
        tot += len(pk)
        for i, j in enumerate(pk):
            if em[j]:
                nev += 1
                oth = np.delete(rr, i)
                xs.append(rr[i] - oth.mean())
    r['sel_event_picks'] = {'n_event': nev, 'n_total': tot, 'share': nev / tot, 'rel_next_pct': float(np.mean(xs) * 100), 't': tst(xs)}
    # 2603 長榮 2021 被跳過的月份
    s26 = sk[(sk.code == '2603') & (sk.d.dt.year == 2021)]
    r['evergreen_2021'] = {'months': [str(x.date()) for x in s26.d], 'pair_sum_pp': float(r['_dif'].reindex(pd.DatetimeIndex(s26.d)).sum() * 100)}
    sk.to_csv(H / f'{os.environ.get("TAG", "my")}_skipped_n{n}_w{WIN}_{"_".join(TYPES)}.csv', index=False)
    del r['_dif']
    print(n, {k: v for k, v in r.items() if k not in COST}, flush=True)
(H / f'{os.environ.get("TAG", "my")}_strategy_w{WIN}_{"_".join(TYPES)}.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
