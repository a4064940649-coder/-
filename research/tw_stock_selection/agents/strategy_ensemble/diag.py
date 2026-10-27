#!/usr/bin/env python3
"""事後診斷（看過 run.py 結果後才加，不參與挑選與判定）→ results.json 的 POSTHOC。
D1：S_recent 的 N10/N30 改用 exec_rebuild/main_nav.parquet（rev_5y 營收，現行建議實際用的版本）；2021-03-11 前以長資料版報酬接續。
D2：處置部位閒置改持現金（DISP_PX_cash、DISP_REAL_cash_ref）時的 E1～E6。
D3：窮舉組合在樣本外勝過 E6（Calmar 較高且對 0050 t 不低於 E6）的比例。
D4：去掉 2026 年後（2026-01～10 主策略異常強），E1～E6 與 N10 的保留期／驗證期。
執行：python3 -I -B diag.py（在 run.py 之後）"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H))
import elib as E   # noqa: E402
CAL = E.CAL; T = E.T; PR = E.PR; PER = PR['periods']
ME_IDX = np.array(sorted(E.MONTH_END))
FA = {'S_long': '2006-05-11', 'S_recent': '2021-02-26'}

def bounds(a, b):
    ia = int(CAL.searchsorted(pd.Timestamp(a))) - 1; ib = int(CAL.searchsorted(pd.Timestamp(b), 'right')) - 1
    return max(ia, 0), ib
def full(s):
    a = np.full(T, np.nan); a[CAL.get_indexer(s.index)] = s.to_numpy(); return a
def fstats(x, a, b):
    ia, ib = bounds(a, b); seg = x[ia:ib + 1]
    yrs = (CAL[ib] - CAL[ia]).days / 365.25
    cagr = (seg[-1] / seg[0]) ** (1 / yrs) - 1; dd = float((seg / np.maximum.accumulate(seg) - 1).min())
    me = ME_IDX[(ME_IDX > ia) & (ME_IDX < ib)]; mv = x[np.r_[ia, me, ib]]
    return {'CAGR': float(cagr), 'maxDD': dd, 'Calmar': float(cagr / abs(dd)), 'm': mv[1:] / mv[:-1] - 1}
def tst(d): d = np.asarray(d, float); return float(d.mean() / d.std(ddof=1) * math.sqrt(len(d)))
def rep(x, a, b, refs):
    s = fstats(x, a, b); o = {'CAGR': round(s['CAGR'], 4), 'maxDD': round(s['maxDD'], 4), 'Calmar': round(s['Calmar'], 3)}
    for k, r in refs.items(): o[f't_vs_{k}'] = round(tst(s['m'] - fstats(r, a, b)['m']), 2)
    return o

POST = {}
orig_P = E.P.copy()
def run_set(setname, wins, disp_override=None):
    out = {}
    refs = {'0050': E.P[:, E.CI['E0050']], 'N10': E.P[:, E.CI['N10']]}
    navs = {}
    for m in ['E1', 'E2', 'E3', 'E4', 'E5', 'E6']:
        if m == 'E6':
            d = disp_override or ('DISP_PX' if setname == 'S_long' else 'DISP_REAL')
            names, f = ['N10', d], E.w_fixed({'N10': 0.8, d: 0.2})
        else:
            names, f = PR['sets'][setname], E.METHODS[m]
        x = full(E.simulate(names, f, FA[setname])); navs[m] = x
        out[m] = {w: rep(x, a, b, refs) for w, (a, b) in wins.items()}
    out['N10'] = {w: rep(refs['N10'], a, b, {'0050': refs['0050']}) for w, (a, b) in wins.items()}
    out['N30'] = {w: rep(E.P[:, E.CI['N30']], a, b, {'0050': refs['0050'], 'N10': refs['N10']}) for w, (a, b) in wins.items()}
    return out

# D1
mn = pd.read_parquet(H.parent / 'exec_rebuild' / 'main_nav.parquet')
for k, col in (('N10', 'M2_n10_slip0.001'), ('N30', 'M2_n30_slip0.001')):
    s = mn[col]; s = s.reindex(s.index.union(CAL)).ffill().reindex(CAL)
    j = E.CI[k]; base = orig_P[:, j].copy(); i0 = int(np.argmax(np.isfinite(s.to_numpy())))
    new = base.copy(); new[i0:] = base[i0] * s.to_numpy()[i0:] / s.to_numpy()[i0]
    E.P[:, j] = new
E.RD[:] = pd.DataFrame(E.P).pct_change().to_numpy()
POST['D1_S_recent_with_rev5y_main'] = run_set('S_recent', PER['S_recent'])
E.P[:] = orig_P; E.RD[:] = pd.DataFrame(E.P).pct_change().to_numpy()

# D2：閒置現金版處置
cash = E.NAVDF[['DISP_PX_cash', 'DISP_REAL_cash_ref']].copy()
r5 = E.NAVDF['E0050'].pct_change()
rc = cash['DISP_REAL_cash_ref'].pct_change(); fv = cash['DISP_REAL_cash_ref'].first_valid_index()
rc[CAL <= fv] = 0.0                                # 2021-10 前閒置現金
cash['DISP_REAL_cash_ref'] = (1 + rc.fillna(0)).cumprod()
E.P[:, E.CI['DISP_PX']] = cash['DISP_PX_cash'].to_numpy(); E.P[:, E.CI['DISP_REAL']] = cash['DISP_REAL_cash_ref'].to_numpy()
E.RD[:] = pd.DataFrame(E.P).pct_change().to_numpy()
E.VOL60[:] = pd.DataFrame(E.RD).rolling(60, min_periods=60).std().shift(1).to_numpy()
E.MOM126[:] = (pd.DataFrame(E.P) / pd.DataFrame(E.P).shift(126)).shift(1).to_numpy() - 1
POST['D2_disposal_idle_cash'] = {s: run_set(s, PER[s]) for s in PR['sets']}
E.P[:] = orig_P; E.RD[:] = pd.DataFrame(E.P).pct_change().to_numpy()
E.VOL60[:] = pd.DataFrame(E.RD).rolling(60, min_periods=60).std().shift(1).to_numpy()
E.MOM126[:] = (pd.DataFrame(E.P) / pd.DataFrame(E.P).shift(126)).shift(1).to_numpy() - 1

# D3
R = json.loads((H / 'results.json').read_text(encoding='utf-8'))
ens = R['RESULTS']['ensembles']
X = pd.read_csv(H / 'exhaustive_all.csv.gz')
d3 = {}
for s in PR['sets']:
    x = X[X.set == s]
    for w in PER[s]:
        if w == 'selection': continue
        e6 = ens[f'{s}|E6']['windows'][w]
        bc = x[f'{w}_Calmar'] > e6['Calmar']; bt = x[f'{w}_t0050'] >= e6['t_vs_0050']
        d3[f'{s}|{w}'] = {'E6_Calmar': e6['Calmar'], 'E6_t_vs_0050': e6['t_vs_0050'], 'share_Calmar_gt_E6': round(float(bc.mean()), 3),
                          'share_t_ge_E6': round(float(bt.mean()), 3), 'share_both': round(float((bc & bt).mean()), 3),
                          'top5_by_Calmar': x.sort_values(f'{w}_Calmar', ascending=False).head(5)[['subset', 'w', f'{w}_CAGR', f'{w}_maxDD', f'{w}_Calmar', f'{w}_t0050']].round(3).to_dict('records')}
POST['D3_exhaustive_vs_E6'] = d3
# D4：去 2026
POST['D4_ex2026'] = {'S_long': run_set('S_long', {'holdout_ex2026': ['2021-03-01', '2025-12-31']}),
                     'S_recent': run_set('S_recent', {'validation_ex2026': ['2024-01-01', '2025-12-31']})}
R['POSTHOC'] = POST
(H / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
def show(d, pre=''):
    for k, v in d.items():
        print(pre + k, v)
for k, v in POST.items():
    print('=====', k)
    if k == 'D2_disposal_idle_cash':
        for s, vv in v.items(): show(vv, s + '|')
    elif k == 'D4_ex2026':
        for s, vv in v.items(): show(vv, s + '|')
    else: show(v)
