#!/usr/bin/env python3
"""步驟 3（最後才跑）：保留期 2021-03～2026-10。選中 3 組合＋C1+C2 的保留期表現、成本敏感度、2022 空頭、129 組合保留期分佈、事前判定。結果寫入 results.json 的 HOLDOUT。"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.dont_write_bytecode = True
import clib as K
L = K.L
OUT = K.HERE
SEL = ('2006-04-01', '2013-12-31'); VAL = ('2014-01-01', '2020-12-31'); HOLD = ('2021-03-01', '2026-10-31'); MAINK = 'C1+C2'
KEEP = ('CAGR', 'vol', 'maxDD', 't_vs_0050_m', 't_vs_EW_m', 'excess_vs_EW_pct_m', 'excess_vs_0050_pct_m', 'yearly_pct', '0050_CAGR', 'EW_CAGR', '0050_maxDD', 'n_months')

def main():
    res = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
    top3 = res['SEARCH']['SELECTED_TOP3']
    M = pd.read_parquet(OUT / 'search_monthly.parquet'); NAV = pd.read_parquet(OUT / 'search_nav.parquet').astype(float)
    names = [K.cname(c) for c in K.COMBOS]
    S = L.Setup(L.yoy_long(), start='2005-01-01')
    ew = NAV['EW']
    W = {'holdout_2021-03_2026-10': [HOLD], 'bear_2022': [('2022-01-01', '2022-12-31')], 'all_2006-04_2026-10': [(SEL[0], HOLD[1])], 'oos_2006-04_2020-12': [(SEL[0], VAL[1])]}
    out = {'selected': top3}
    det = {}
    for nm in top3 + [MAINK]:
        for n in (30, 10):
            for sw in ('raw', 'sw'):
                k = f'{nm}|n{n}|{sw}'
                det[k] = {w: {kk: v for kk, v in (L.daily_stats(NAV[k].dropna(), sg, S.DATES, L.E50, ew) or {}).items() if kk in KEEP} for w, sg in W.items()}
                for w in ('bear_2022', 'all_2006-04_2026-10', 'oos_2006-04_2020-12'): det[k][w].pop('yearly_pct', None)
                pd_ = K.win(M[k] - M[f'{MAINK}|n{n}|{sw}'], *HOLD).dropna()
                det[k]['holdout_diff_vs_main_pct_m'] = round(float(pd_.mean() * 100), 3); det[k]['holdout_t_diff_vs_main'] = K.tstat(pd_) if nm != MAINK else None
                pv = K.win(M[k] - M[f'{MAINK}|n{n}|{sw}'], *VAL).dropna()
                det[k]['val_diff_vs_main_pct_m'] = round(float(pv.mean() * 100), 3); det[k]['val_t_diff_vs_main'] = K.tstat(pv) if nm != MAINK else None
    out['detail_slip0.1'] = det
    # 成本敏感度（重新模擬）
    comps = K.components(S, K.rev_level()); el = S.t300 & S.s52.notna() & ~S.inD & S.yoy3.notna(); prk = K.pct_ranks(comps, el)
    t0 = L.pos(S.dates[0]); t1 = L.pos(S.dates[-1])
    cs = {}
    for nm in top3 + [MAINK]:
        combo = tuple(nm.split('+'))
        for n in (30, 10):
            ev, _ = K.select(S, K.score(prk, combo), el, n)
            for slip in (0.003, 'stress'):
                b, s = L.cost_rates(slip); nav = K.sim(ev, b, s, t0, t1)
                cs[f'{nm}|n{n}|slip{slip}'] = {w: {kk: v for kk, v in L.daily_stats(nav, sg, S.DATES, L.E50, ew).items() if kk in ('CAGR', 'maxDD', 't_vs_0050_m', 't_vs_EW_m')}
                                               for w, sg in {'select': [SEL], 'valid': [VAL], 'holdout': [HOLD]}.items()}
    out['cost_sensitivity'] = cs
    # 129 組合保留期分佈
    ht = pd.Series({x: K.tstat(K.win(M[f'{x}|n30|raw'] - M['EW'], *HOLD).dropna()) for x in names})
    hd = pd.Series({x: float(K.win(M[f'{x}|n30|raw'] - M[f'{MAINK}|n30|raw'], *HOLD).dropna().mean() * 100) for x in names if x != MAINK})
    out['holdout_distribution_n30'] = {'t_EW_median': round(float(ht.median()), 2), 'main_t_EW': ht[MAINK], 'main_rank': int((ht > ht[MAINK]).sum() + 1),
                                       'share_better_than_main_t': round(float((ht.drop(MAINK) > ht[MAINK]).mean()), 3), 'diff_vs_main_median_pct_m': round(float(hd.median()), 3),
                                       'share_diff_positive': round(float((hd > 0).mean()), 3), 'top5_by_holdout_t': ht.sort_values(ascending=False).head(5).round(2).to_dict(),
                                       'val_top5_in_holdout': {x: ht[x] for x in pd.Series({y: res['SEARCH']['table_all129'][y]['val|n30|raw']['t_EW'] for y in names}).sort_values(ascending=False).head(5).index}}
    # 事前判定
    verdict = {}
    for nm in top3:
        v = res['SEARCH']['table_all129'][nm]['val|n30|raw']; h = det[f'{nm}|n30|raw']
        if v['t_EW'] >= 2 and v['diff_vs_main_pct_m'] > 0 and (v['t_diff_vs_main'] or 0) >= 1 and h['holdout_diff_vs_main_pct_m'] > 0: verdict[nm] = '可用'
        elif (v['t_EW'] >= 2 and (v['t_diff_vs_main'] or 0) < 1) or h['holdout_diff_vs_main_pct_m'] > 0 and v['t_EW'] >= 2: verdict[nm] = '待觀察'
        elif v['t_EW'] < 2 or (v['diff_vs_main_pct_m'] <= 0 and h['holdout_diff_vs_main_pct_m'] <= 0): verdict[nm] = '不可用'
        else: verdict[nm] = '待觀察'
    # 開關判定（以原主策略 n30 與選中組合各自檢查）
    sw = {}
    for nm in top3 + [MAINK]:
        r = {}
        for tag, w in (('valid', 'valid_2014-01_2020-12'), ('holdout', 'holdout_2021-03_2026-10')):
            a = res['SEARCH']['detail_slip0.1'][f'{nm}|n30|raw'][w] if tag == 'valid' else det[f'{nm}|n30|raw'][w]
            b = res['SEARCH']['detail_slip0.1'][f'{nm}|n30|sw'][w] if tag == 'valid' else det[f'{nm}|n30|sw'][w]
            r[tag] = {'maxDD_improve_pp': round((b['maxDD'] - a['maxDD']) * 100, 1), 'CAGR_change_pp': round((b['CAGR'] - a['CAGR']) * 100, 1)}
        r['verdict'] = '可用' if all(r[t]['maxDD_improve_pp'] >= 10 and r[t]['CAGR_change_pp'] > -5 for t in ('valid', 'holdout')) else '不可用（依事前門檻）'
        sw[nm] = r
    out['verdict_combo'] = verdict; out['verdict_switch'] = sw
    res['HOLDOUT'] = out
    (OUT / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    for k, v in det.items():
        print(k, {w: (x.get('CAGR'), x.get('maxDD'), x.get('t_vs_0050_m'), x.get('t_vs_EW_m')) for w, x in v.items() if isinstance(x, dict)}, v['holdout_diff_vs_main_pct_m'], v['holdout_t_diff_vs_main'], v['val_diff_vs_main_pct_m'], v['val_t_diff_vs_main'])
        print('   yearly', v['holdout_2021-03_2026-10']['yearly_pct'])
    print(json.dumps({k: out[k] for k in ('cost_sensitivity', 'holdout_distribution_n30', 'verdict_combo', 'verdict_switch')}, ensure_ascii=False))

if __name__ == '__main__':
    main()
