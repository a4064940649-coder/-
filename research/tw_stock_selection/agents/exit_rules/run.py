#!/usr/bin/env python3
"""exit_rules 第 1 階段：主策略 n10／n30 與 D 三條事件型規則 × 12 出場規則 × cash/next × 兩種成本。
只算挑選期（SEL）與驗證期（VAL），選中規則與判定寫入 results.json 的 RUN；淨值存 navs.parquet。保留期在 holdout.py。"""
import json, hashlib, sys, time
from pathlib import Path
import multiprocessing as mp
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.dont_write_bytecode = True
import xlib as X
L = X.L

res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert hashlib.sha256(json.dumps(res['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == res['PREREG_sha256'], 'PREREG 被改過'
assert 'ENGINE_CHECK' in res, '先跑 engine_check.py'

COSTS = {'slip0.001': L.cost_rates(0.001), 'stress': L.cost_rates('stress')}
SEGS_STD = {'SEL': ('2006-04-01', '2013-12-31'), 'VAL': ('2014-01-01', '2020-12-31')}
SEGS_SURGE = {'SEL': ('2008-03-01', '2013-01-31'), 'VAL': ('2016-03-01', '2020-12-31')}
VARIANTS = ['NONE'] + [f'{r}|{m}' for r in X.RULES for m in ('cash', 'next')]

STRATS = {}
def build_all():
    for n in (10, 30):
        it, _ = X.main_items(n); STRATS[f'main_n{n}'] = (it, 1)
    it, K = X.weekly_reversal_items(); STRATS['D_weekly_reversal'] = (it, K)
    it, K = X.high_volume_items(); STRATS['D_high_volume'] = (it, K)
    r = X.surge_items(); STRATS['D_revenue_sue'] = (r[0], r[1])

def job(args):
    sname, var, ck = args
    items, K = STRATS[sname]; b, s = COSTS[ck]
    T0 = items[0]['t']; T1 = len(X.IDX) - 1
    if var == 'NONE': rule, mode = None, 'cash'
    else:
        rn, mode = var.split('|'); rule = X.RULES[rn]
    nav, info = X.simulate(items, K, b, s, T0, T1, rule=rule, mode=mode)
    return sname, var, ck, nav, info

EWCOL = {'D_weekly_reversal': ('w3_weekly_reversal/navs.parquet', 'EW_pool'),
         'D_high_volume': ('w3_high_volume_premium/navs.parquet', 'main_GKM_weekly_20d_EW'),
         'D_revenue_sue': ('w3_revenue_sue/nav.parquet', 'S_main_EW')}

def main():
    t_start = time.time()
    build_all(); print('built', round(time.time() - t_start, 1), flush=True)
    jobs = [(sn, v, ck) for sn in STRATS for v in VARIANTS for ck in COSTS]
    navs, infos = {}, {}
    with mp.get_context('fork').Pool(4) as pool:
        for sn, v, ck, nav, info in pool.imap_unordered(job, jobs, chunksize=2):
            navs[f'{sn}|{v}|{ck}'] = nav
            if ck == 'slip0.001': infos[(sn, v)] = info
            if len(navs) % 20 == 0: print(len(navs), '/', len(jobs), round(time.time() - t_start, 1), flush=True)
    ND = pd.DataFrame(navs)
    ND.to_parquet(HERE / 'navs.parquet')
    import pickle
    (HERE / 'infos.pkl').write_bytes(pickle.dumps({f'{k[0]}|{k[1]}': {kk: vv for kk, vv in v.items()} for k, v in infos.items()}))
    out = {'strategies': {}, 'bear_windows': X.BEARS}
    for sn in STRATS:
        SEGS = SEGS_SURGE if sn == 'D_revenue_sue' else SEGS_STD
        ew = None
        if sn in EWCOL:
            p, c = EWCOL[sn]; ew = pd.read_parquet(X.AG / p)[c]
        base = {ck: ND[f'{sn}|NONE|{ck}'].dropna() for ck in COSTS}
        S = {'segments_def': SEGS, 'variants': {}}
        for v in VARIANTS:
            o = {'segments': {}, 'paired_vs_NONE': {}, 'paired_vs_NONE_stress': {}, 'counts': {}}
            for sk, (a, bnd) in SEGS.items():
                for ck in COSTS:
                    o['segments'][f'{sk}|{ck}'] = X.seg_stats(ND[f'{sn}|{v}|{ck}'].dropna(), a, bnd, ew=ew)
                if v != 'NONE':
                    o['paired_vs_NONE'][sk] = X.paired(ND[f'{sn}|{v}|slip0.001'].dropna(), base['slip0.001'], a, bnd)
                    o['paired_vs_NONE_stress'][sk] = X.paired(ND[f'{sn}|{v}|stress'].dropna(), base['stress'], a, bnd)
                o['counts'][sk] = X.counts(infos[(sn, v)], a, bnd)
            o['counts']['FULL'] = X.counts(infos[(sn, v)], '2005-01-01', '2026-10-31')
            o['turnover_per_rebalance'] = infos[(sn, v)]['turnover_per_rebalance']
            nv = ND[f'{sn}|{v}|slip0.001'].dropna()
            o['yearly_pct_to2020'] = {y: x for y, x in X.yearly(nv).items() if y <= 2020}
            o['bear_pct_to2020'] = {k: x for k, x in X.bear_rets(nv).items() if k != '2022'}
            if v != 'NONE':
                p = o['paired_vs_NONE']; c1 = p['SEL']['t_paired_m'] >= 1.5; c2 = p['VAL']['diff_pct_m'] > 0
                o['verdict_if_selected'] = '可用' if (c1 and c2) else ('待觀察' if (c1 or c2) else '不可用')
            S['variants'][v] = o
        # t 為 NaN（與無出場版逐月相同、差異變異為 0）視為 −inf，不可能被選中
        cand = {v: (lambda x: x if (x is not None and x == x) else float('-inf'))(S['variants'][v]['paired_vs_NONE']['SEL']['t_paired_m']) for v in VARIANTS if v != 'NONE'}
        sel = max(cand, key=lambda k: cand[k])
        so = S['variants'][sel]; p = so['paired_vs_NONE']
        S['SELECTED'] = sel
        S['SELECTED_SEL_t'] = p['SEL']['t_paired_m']; S['SELECTED_VAL_diff'] = p['VAL']['diff_pct_m']
        usable = p['SEL']['t_paired_m'] >= 1.5 and p['VAL']['diff_pct_m'] > 0
        S['VERDICT'] = '可用（待保留期）' if usable else '不可用（不設此出場）'
        if sn.startswith('D_'):
            sg = so['segments']
            ok = all(sg[f'{k}|slip0.001']['t_vs_EW_m'] >= 2 and sg[f'{k}|slip0.001']['CAGR'] >= sg[f'{k}|slip0.001']['0050_CAGR'] for k in ('SEL', 'VAL'))
            S['RESCUE_detail'] = {k: {'t_vs_EW_m': sg[f'{k}|slip0.001']['t_vs_EW_m'], 'CAGR': sg[f'{k}|slip0.001']['CAGR'], '0050_CAGR': sg[f'{k}|slip0.001']['0050_CAGR']} for k in ('SEL', 'VAL')}
            S['VERDICT'] = ('救回' if (usable and ok) else ('出場有幫助但沒救回' if usable else '出場沒幫助'))
        S['rank_SEL_t'] = sorted(cand.items(), key=lambda kv: -kv[1])
        out['strategies'][sn] = S
        print(sn, 'SELECTED', sel, S['SELECTED_SEL_t'], S['SELECTED_VAL_diff'], S['VERDICT'], flush=True)
    e = X.E50.reindex(ND.index) if hasattr(X, 'E50') else L.E50.reindex(ND.index)
    out['0050'] = {sk: X.R.seg_stats(e.dropna(), a, b) for sk, (a, b) in SEGS_STD.items()}
    out['0050_yearly_to2020'] = {y: x for y, x in X.yearly(e.loc['2006-01-01':]).items() if y <= 2020}
    out['0050_bear'] = X.bear_rets(e.loc['2006-01-01':])
    out['TAIEX_trigger_days'] = {'n_days_total_2006_2026': int(X.MKT_TRIG[(X.IDX >= pd.Timestamp('2006-01-01'))].sum()),
                                 'dates': [str(d.date()) for d in X.IDX[X.MKT_TRIG & (X.IDX >= pd.Timestamp('2006-01-01'))]]}
    res['RUN'] = out
    res['RUN_written_at'] = pd.Timestamp.now().isoformat(timespec='seconds')
    (HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print('done', round(time.time() - t_start, 1))

if __name__ == '__main__':
    main()
