#!/usr/bin/env python3
"""步驟 2：挑選期（2006-04～2013-12）依事前規則選前 3 組合；驗證期（2014-01～2020-12）表現、129 組合分佈、max-t／SPA bootstrap、Deflated Sharpe。
不碰保留期（2021-03 起）。結果寫入 results.json 的 SEARCH。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.dont_write_bytecode = True
import clib as K
L = K.L
OUT = K.HERE
SEL = ('2006-04-01', '2013-12-31'); VAL = ('2014-01-01', '2020-12-31'); IFRS13 = ('2013-02-01', '2013-12-31')
MAINK = 'C1+C2'

def tt(x): return K.tstat(x)

def main():
    res = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
    M = pd.read_parquet(OUT / 'search_monthly.parquet')
    NAV = pd.read_parquet(OUT / 'search_nav.parquet').astype(float)
    names = [K.cname(c) for c in K.COMBOS]
    def ex(key, w, drop=None, vs='EW'):
        return K.win(M[key] - M[vs], *w, drop=drop).dropna()
    tab = {}
    for nm in names:
        r = {}
        for tag, w, drop in (('sel', SEL, None), ('sel_ex2013', SEL, IFRS13), ('val', VAL, None)):
            for n in (30, 10):
                for sw in ('raw', 'sw'):
                    k = f'{nm}|n{n}|{sw}'
                    e = ex(k, w, drop); e5 = ex(k, w, drop, '0050'); pdif = K.win(M[k] - M[f'{MAINK}|n{n}|{sw}'], *w, drop=drop).dropna()
                    r[f'{tag}|n{n}|{sw}'] = {'t_EW': tt(e), 'ex_EW_pct_m': round(float(e.mean() * 100), 3), 't_0050': tt(e5),
                                             'diff_vs_main_pct_m': round(float(pdif.mean() * 100), 3), 't_diff_vs_main': tt(pdif) if nm != MAINK else None}
        tab[nm] = r
    rank = sorted(names, key=lambda x: -tab[x]['sel|n30|raw']['t_EW'])
    top3 = rank[:3]
    rank_ex13 = sorted(names, key=lambda x: -tab[x]['sel_ex2013|n30|raw']['t_EW'])
    out = {'selection_rule': '挑選期 n30 滑價 0.1% 不加開關 對同池等權月窗 t 最高前 3',
           'SELECTED_TOP3': top3, 'main_rank_in_selection': rank.index(MAINK) + 1,
           'selection_top10': [(x, tab[x]['sel|n30|raw']['t_EW'], tab[x]['val|n30|raw']['t_EW']) for x in rank[:10]],
           'robust_ex2013_top10': [(x, tab[x]['sel_ex2013|n30|raw']['t_EW']) for x in rank_ex13[:10]],
           'robust_ex2013_top3_same_as_primary': rank_ex13[:3] == top3}
    # 驗證期 129 分佈
    vt = pd.Series({x: tab[x]['val|n30|raw']['t_EW'] for x in names}); st = pd.Series({x: tab[x]['sel|n30|raw']['t_EW'] for x in names})
    vd = pd.Series({x: tab[x]['val|n30|raw']['diff_vs_main_pct_m'] for x in names if x != MAINK})
    vt10 = pd.Series({x: tab[x]['val|n10|raw']['t_EW'] for x in names})
    out['distribution'] = {
        'sel_n30_t_EW': {'median': round(float(st.median()), 2), 'q25': round(float(st.quantile(.25)), 2), 'q75': round(float(st.quantile(.75)), 2), 'max': round(float(st.max()), 2), 'share_ge2': round(float((st >= 2).mean()), 3), 'main': tab[MAINK]['sel|n30|raw']['t_EW']},
        'val_n30_t_EW': {'median': round(float(vt.median()), 2), 'q25': round(float(vt.quantile(.25)), 2), 'q75': round(float(vt.quantile(.75)), 2), 'max': round(float(vt.max()), 2), 'argmax': vt.idxmax(), 'share_ge2': round(float((vt >= 2).mean()), 3), 'main': tab[MAINK]['val|n30|raw']['t_EW'],
                         'share_better_than_main_t': round(float((vt.drop(MAINK) > vt[MAINK]).mean()), 3), 'main_rank': int((vt > vt[MAINK]).sum() + 1)},
        'val_n30_diff_vs_main': {'median_pct_m': round(float(vd.median()), 3), 'share_positive': round(float((vd > 0).mean()), 3),
                                 'share_t_ge2': round(float((pd.Series({x: tab[x]['val|n30|raw']['t_diff_vs_main'] for x in vd.index}) >= 2).mean()), 3)},
        'val_n10_t_EW': {'median': round(float(vt10.median()), 2), 'main': tab[MAINK]['val|n10|raw']['t_EW'], 'share_better_than_main_t': round(float((vt10.drop(MAINK) > vt10[MAINK]).mean()), 3)},
        'rank_corr_sel_vs_val_spearman': round(float(st.rank().corr(vt.rank())), 3),
        'val_t_of_sel_top10_mean': round(float(vt[rank[:10]].mean()), 2), 'val_t_of_sel_bottom10_mean': round(float(vt[rank[-10:]].mean()), 2),
    }
    # 包含某元件的組合 vs 不含（驗證期 t 平均）
    out['component_marginal'] = {}
    for c in K.COMP:
        inn = [x for x in names if c in x.split('+')]; outn = [x for x in names if c not in x.split('+')]
        out['component_marginal'][c] = {'sel_mean_t_with': round(float(st[inn].mean()), 2), 'sel_mean_t_without': round(float(st[outn].mean()), 2),
                                        'val_mean_t_with': round(float(vt[inn].mean()), 2), 'val_mean_t_without': round(float(vt[outn].mean()), 2)}
    # bootstrap
    boot = {}
    for tag, w in (('sel', SEL), ('val', VAL)):
        X = pd.DataFrame({x: ex(f'{x}|n30|raw', w) for x in names}).dropna()
        r = K.maxt_bootstrap(X.to_numpy(), B=5000, mean_block=6, seed=1)
        boot[f'{tag}_vs_EW_maxt'] = {'t_obs_max': r['t_obs_max'], 'argmax': names[r['argmax']], 'p_maxt_adjusted': r['p_maxt'], 'boot_q95': r['boot_max_t_q95'], 'T': r['T'], 'K': r['K'],
                                     'p_adj_selected': {x: r['p_each'][names.index(x)] for x in top3 + [MAINK]}}
        D = pd.DataFrame({x: K.win(M[f'{x}|n30|raw'] - M[f'{MAINK}|n30|raw'], *w) for x in names if x != MAINK}).dropna()
        dn = list(D.columns)
        for spa in (False, True):
            r2 = K.maxt_bootstrap(D.to_numpy(), B=5000, mean_block=6, seed=2, spa=spa)
            boot[f'{tag}_vs_MAIN_{"SPAc" if spa else "RC"}'] = {'t_obs_max': r2['t_obs_max'], 'argmax': dn[r2['argmax']], 'p': r2['p_maxt'], 'boot_q95': r2['boot_max_t_q95'], 'K': r2['K'],
                                                                 'p_adj_selected': {x: r2['p_each'][dn.index(x)] for x in top3 if x in dn}}
        # Deflated Sharpe：最佳組合
        srs = (X.mean() / X.std(ddof=1)).to_numpy(); best = X.columns[int(np.argmax(srs))]
        rho = X.corr().to_numpy()[np.triu_indices(X.shape[1], 1)].mean(); Neff = rho + (1 - rho) * X.shape[1]
        boot[f'{tag}_DSR'] = {'best_by_SR': best, 'N129': K.deflated_sharpe(X[best], srs, 129), 'avg_corr': round(float(rho), 3),
                              'Neff': K.deflated_sharpe(X[best], srs, max(Neff, 2))}
        if tag == 'val':
            boot['val_DSR_selected_top1'] = K.deflated_sharpe(X[top3[0]], srs, 3)  # 驗證期只做 3 次試驗
    out['bootstrap'] = boot
    # 選中組合＋主策略：詳細（滑價 0.1%，日頻）
    S = L.Setup(L.yoy_long(), start='2005-01-01')
    W = {'select_2006-04_2013-12': [SEL], 'valid_2014-01_2020-12': [VAL], 'select_ex2013': [(SEL[0], '2013-01-31')],
         'bear_2008': [('2008-01-01', '2008-12-31')], 'bear_2011': [('2011-01-01', '2011-12-31')], 'bear_2015': [('2015-01-01', '2015-12-31')], 'bear_2018': [('2018-01-01', '2018-12-31')]}
    ew = NAV['EW']
    det = {}
    for nm in top3 + [MAINK]:
        for n in (30, 10):
            for sw in ('raw', 'sw'):
                k = f'{nm}|n{n}|{sw}'
                nav = NAV[k].dropna()
                det[k] = {w: {kk: v for kk, v in (L.daily_stats(nav, sg, S.DATES, L.E50, ew) or {}).items() if kk in ('CAGR', 'vol', 'maxDD', 't_vs_0050_m', 't_vs_EW_m', 'excess_vs_EW_pct_m', 'excess_vs_0050_pct_m', 'yearly_pct', '0050_CAGR', 'EW_CAGR', '0050_maxDD', 'n_months')} for w, sg in W.items()}
                for w in det[k]:
                    if not w.startswith(('select_2006', 'valid')): det[k][w].pop('yearly_pct', None)
    out['detail_slip0.1'] = det
    # 開關：129 組合驗證期日頻最大回撤改善分佈
    dd = {}
    for nm in names:
        a = L.daily_stats(NAV[f'{nm}|n30|raw'].dropna(), [VAL], S.DATES); b = L.daily_stats(NAV[f'{nm}|n30|sw'].dropna(), [VAL], S.DATES)
        dd[nm] = (b['maxDD'] - a['maxDD'], b['CAGR'] - a['CAGR'])
    d1 = pd.Series({k: v[0] for k, v in dd.items()}); d2 = pd.Series({k: v[1] for k, v in dd.items()})
    out['switch_effect_val_n30_all129'] = {'maxDD_improve_median_pp': round(float(d1.median() * 100), 1), 'share_maxDD_improved': round(float((d1 > 0).mean()), 3),
                                          'CAGR_change_median_pp': round(float(d2.median() * 100), 1), 'share_CAGR_better': round(float((d2 > 0).mean()), 3)}
    out['table_all129'] = tab
    res['SEARCH'] = out
    (OUT / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    for k in ('SELECTED_TOP3', 'main_rank_in_selection', 'selection_top10', 'robust_ex2013_top10', 'distribution', 'component_marginal', 'switch_effect_val_n30_all129'):
        print(k, json.dumps(out[k], ensure_ascii=False))
    print(json.dumps(boot, ensure_ascii=False))
    for k, v in det.items():
        print(k, {w: (x.get('CAGR'), x.get('maxDD'), x.get('t_vs_0050_m'), x.get('t_vs_EW_m')) for w, x in v.items()})

if __name__ == '__main__':
    main()
