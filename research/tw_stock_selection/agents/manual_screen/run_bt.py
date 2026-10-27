#!/usr/bin/env python3
"""手動版 M1～M4（＋敏感度 S1～S3）對主策略 n10 回測。輸出 results_bt.json、navs.parquet、picks.json。"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import ms_core as K

T0 = time.time(); HERE = K.HERE
RULES = ['M1', 'M2', 'M3', 'M4', 'S1', 'S2', 'S3']
PER = {'long': {'A_2006-04_2013-12': ('2006-04-01', '2013-12-31'), 'B_2014_2020': ('2014-01-01', '2020-12-31'),
                'AB_2006-04_2020-12': ('2006-04-01', '2020-12-31')},
       'r5y': {'C_2021-03_2026-10': ('2021-03-01', '2026-10-31'), 'D_2024_2026-10': ('2024-01-01', '2026-10-31')}}
res = {}; navs = {}; picks = {}

def mstats(nav, a, b, sd, main, mainx, ew):
    o = K.stats(nav, a, b, sd, K.E50, main)
    o['t_vs_main'] = o.pop('t_EW', None)
    o['t_vs_main_exfin'] = K.stats(nav, a, b, sd, None, mainx).get('t_EW')
    o['t_EW'] = K.stats(nav, a, b, sd, None, ew).get('t_EW')
    D = pd.DatetimeIndex([d for d in sd if d in nav.index])
    def mr(x):
        xd = x.reindex(D); r = (xd.shift(-1) / xd - 1).iloc[:-1]; return r[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))]
    o['diff_vs_main_m'] = float((mr(nav) - mr(main)).mean())
    return o

def run(tag, Y, LV, BS, start):
    S = K.Strategy(Y, start)
    SX = K.Strategy(Y, start, fin_exclude=set(K.COLS[K.FIN]))
    t0 = K.IDX.get_loc(S.sig_days[0]); t1 = K.IDX.get_loc(S.end_day); sd = S.sig_days + [S.end_day]
    bb, ss = K.rates(0.001); bs_, ss_ = K.rates('stress')
    pm = S.plan(10); pmx = SX.plan(10)
    ew, _ = K.simulate([(p[0], p[2]) for p in pm], 0.0, 0.0, t0, t1)
    books = {'MAIN': (pm, None), 'MAIN_exfin': (pmx, None)}
    info = {'MAIN': {'blocked': S.info['blocked_n10']}, 'MAIN_exfin': {'blocked': SX.info['blocked_n10']}}
    for r in RULES:
        M = K.Manual(r, Y, LV, BS, S.sig_days)
        books[r] = (M.plan(), 10); info[r] = M.info
        print(tag, r, 'planned', f'{time.time()-T0:.0f}s', flush=True)
    nv = {}; nvs = {}
    for k, (pl, ns) in books.items():
        nv[k], lg = K.simulate([(p[0], p[1]) for p in pl], bb, ss, t0, t1, nslots=ns)
        nvs[k], _ = K.simulate([(p[0], p[1]) for p in pl], bs_, ss_, t0, t1, nslots=ns)
        info[k]['turnover'] = round(lg['turnover'], 3)
        navs[f'{tag}|{k}'] = nv[k]; navs[f'{tag}|{k}|stress'] = nvs[k]
        picks[f'{tag}|{k}'] = {str(p[3].date()): [K.COLS[j] for j in p[1]] for p in pl}
    navs[f'{tag}|EW'] = ew
    mainpk = {p[3]: set(p[1]) for p in pm}; mainxpk = {p[3]: set(p[1]) for p in pmx}
    out = {}
    for pname, (a, b) in PER[tag].items():
        po = {}
        for k in books:
            o = mstats(nv[k], a, b, sd, nv['MAIN'], nv['MAIN_exfin'], ew)
            o['CAGR_stress'] = K.stats(nvs[k], a, b, sd)['CAGR']
            pl = [p for p in books[k][0] if pd.Timestamp(a) <= p[3] <= pd.Timestamp(b)]
            o['overlap_main'] = float(np.mean([len(set(p[1]) & mainpk[p[3]]) for p in pl]))
            o['overlap_main_exfin'] = float(np.mean([len(set(p[1]) & mainxpk[p[3]]) for p in pl]))
            o['avg_n_held'] = float(np.mean([len(p[1]) for p in pl]))
            o['months_lt10'] = int(sum(len(p[1]) < 10 for p in pl)); o['n_sig'] = len(pl)
            po[k] = o
            print(tag, pname, k, {x: (round(v, 3) if isinstance(v, float) else v) for x, v in o.items() if x != 'yearly'}, flush=True)
        out[pname] = po
    out['info'] = {k: {x: v for x, v in i.items() if x != 'n_cand'} for k, i in info.items()}
    out['n_cand_median'] = {k: float(np.median(list(i['n_cand'].values()))) for k, i in info.items() if 'n_cand' in i}
    res[tag] = out

if __name__ == '__main__':
    YL = K.yoy_from_long(); LVL, BSL = K.rev_levels_long()
    run('long', YL, LVL, BSL, '2005-01-01')
    Y5 = K.yoy_from_rev5y(); LV5, BS5 = K.rev_levels_5y()
    run('r5y', Y5, LV5, BS5, '2021-01-01')
    (HERE / 'results_bt.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    pd.DataFrame(navs).to_parquet(HERE / 'navs.parquet')
    (HERE / 'picks.json').write_text(json.dumps(picks, ensure_ascii=False), encoding='utf-8')
    print('done', time.time() - T0)
