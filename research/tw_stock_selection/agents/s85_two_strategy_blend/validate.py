#!/usr/bin/env python3
"""S85 驗證段（2022-01～2026-10）＋描述表。只跑一次（valid_results.json 存在就拒跑）。先核對 FROZEN 雜湊。
輸出 valid_results.json、corr_env_table.csv、blend_table.csv、monthly_excess.csv。"""
import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s85lib as L
assert not (L.H / 'valid_results.json').exists(), '驗證只跑一次'
fzb = (L.H / 'FROZEN.json').read_bytes()
assert hashlib.sha256(fzb).hexdigest() == (L.H / 'FROZEN.sha256').read_text().split()[0]
FZ = json.loads(fzb)
sd, bench, C, th = L.load()
FULL = ('2016-02', '2026-10')
WINS = {'train': L.TRAIN, 'valid': L.VALID, 'full': FULL}
s_nav = sd['nav'].loc[L.START:]; s_blk = sd['nav_blk5pct'].loc[L.START:]
res = {'frozen': FZ, 'current': {}, 'cells': {}, 'candidates_alone': {}, 'corr': {}, 'env': {}, 'sens': {}}
for w, win in WINS.items():
    c = L.stats(s_nav, bench, th, win); c['ex_top5pct_cagr'] = L.stats(s_blk, bench, th, win)['cagr']; res['current'][w] = c
res['bench'] = {w: dict(cagr=L.stats(bench.loc[L.START:], bench, th, win)['cagr'], maxdd=L.stats(bench.loc[L.START:], bench, th, win)['maxdd']) for w, win in WINS.items()}
# ---- 合併格
for grp, names in (('comparable', L.COMPARABLE), ('reference', L.REFERENCE)):
    cells = L.all_cells(sd, bench, C, names); cb = L.all_cells(sd, bench, C, names, blk=True); c0 = L.all_cells(sd, bench, C, names, kappa=0.0)
    for k in cells:
        R = {'group': grp}
        for w, win in WINS.items():
            s = L.stats(cells[k], bench, th, win); s['ex_top5pct_cagr'] = L.stats(cb[k], bench, th, win)['cagr']
            s['cagr_kappa0'] = L.stats(c0[k], bench, th, win)['cagr']
            s.update(L.passes(s, res['current'][w])); R[w] = s
        res['cells'][k] = R
# 主策略壓力成本敏感度
Cs = dict(C); Cs['MAIN_N10'] = C['MAIN_N10_stress']
for k, nav in L.all_cells(sd, bench, Cs, ['MAIN_N10']).items():
    res['sens'][k + '|stress'] = {w: L.stats(nav, bench, th, win) for w, win in WINS.items()}
# ---- 候選單獨、相關、分環境
rb = L.mret(bench); exS = (L.mret(sd['nav']) - rb).loc[FULL[0]:FULL[1]]
state = th.reindex(exS.index)
rows = []; mex = {'SURGE': exS, 'state': state}
allc = L.COMPARABLE + L.REFERENCE + ['MAIN_N10_stress'] + L.DESCRIBE
for nm in allc:
    nav = C[nm].loc[L.START:]
    ex = (L.mret(nav) - rb).loc[FULL[0]:FULL[1]].dropna()
    if nm in L.SHORT: ex = ex.loc[:'2026-09']
    mex[nm] = ex
    res['candidates_alone'][nm] = {w: L.stats(nav if nm not in L.SHORT else nav.loc[:'2026-09-30'], bench, th, (win[0], min(win[1], '2026-09') if nm in L.SHORT else win[1])) for w, win in WINS.items()}
    j = pd.concat([exS, ex, state], axis=1, keys=['s', 'c', 'st']).dropna()
    def cr(m): return float(j.loc[m, 's'].corr(j.loc[m, 'c'])) if m.sum() > 3 else float('nan')
    tr = (j.index >= pd.Period(L.TRAIN[0])) & (j.index <= pd.Period(L.TRAIN[1])); va = j.index >= pd.Period(L.VALID[0])
    lose = j['s'] < 0
    row = dict(cand=nm, n=len(j), corr_full=cr(np.ones(len(j), bool)), corr_train=cr(tr), corr_valid=cr(va),
               corr_cold=cr((j['st'] == '冷').to_numpy()), corr_warm=cr((j['st'] == '溫').to_numpy()), corr_hot=cr((j['st'] == '熱').to_numpy()))
    for stn, lab in (('冷', 'cold'), ('溫', 'warm'), ('熱', 'hot')):
        m = j['st'] == stn
        row[f'ex_{lab}_ann'] = float(j.loc[m, 'c'].mean() * 12); row[f'n_{lab}'] = int(m.sum()); row[f'pos_{lab}'] = float((j.loc[m, 'c'] > 0).mean())
        mv = m & va; row[f'ex_{lab}_ann_valid'] = float(j.loc[mv, 'c'].mean() * 12); row[f'n_{lab}_valid'] = int(mv.sum())
    row.update(n_surge_lose=int(lose.sum()), ex_when_surge_loses_ann=float(j.loc[lose, 'c'].mean() * 12), pos_when_surge_loses=float((j.loc[lose, 'c'] > 0).mean()),
               ex_when_surge_wins_ann=float(j.loc[~lose, 'c'].mean() * 12),
               n_surge_lose_valid=int((lose & va).sum()), ex_when_surge_loses_ann_valid=float(j.loc[lose & va, 'c'].mean() * 12), pos_when_surge_loses_valid=float((j.loc[lose & va, 'c'] > 0).mean()))
    rows.append(row)
# SURGE 自己的分環境
j = pd.concat([exS, state], axis=1, keys=['s', 'st'])
va = j.index >= pd.Period(L.VALID[0])
res['env']['SURGE'] = {lab: dict(n=int((j['st'] == stn).sum()), ex_ann=float(j.loc[j['st'] == stn, 's'].mean() * 12), pos=float((j.loc[j['st'] == stn, 's'] > 0).mean()),
                                 n_valid=int(((j['st'] == stn) & va).sum()), ex_ann_valid=float(j.loc[(j['st'] == stn) & va, 's'].mean() * 12)) for stn, lab in (('冷', 'cold'), ('溫', 'warm'), ('熱', 'hot'))}
res['env']['SURGE']['n_lose_months'] = int((exS < 0).sum()); res['env']['SURGE']['n_months'] = int(len(exS)); res['env']['SURGE']['ex_when_lose_ann'] = float(exS[exS < 0].mean() * 12)
ct = pd.DataFrame(rows).set_index('cand'); ct.to_csv(L.H / 'corr_env_table.csv', float_format='%.4f')
res['corr'] = ct.to_dict('index')
pd.DataFrame(mex).to_csv(L.H / 'monthly_excess.csv', float_format='%.6f')
# ---- 判定
def verdict(key, label_pass):
    if key is None: return '沒有凍結格 → 維持現行'
    return label_pass if res['cells'][key]['valid']['passed'] else '凍結格驗證不過 → 維持現行'
res['verdict'] = dict(comparable=verdict(FZ['frozen_comparable'], '值得獨立複製'), reference=verdict(FZ['frozen_reference'], '值得先用同口徑重做'),
                      record_only=FZ['record_only_comparable'], record_only_valid_passed=res['cells'][FZ['record_only_comparable']]['valid']['passed'],
                      n_valid_pass_comparable=[k for k, v in res['cells'].items() if v['group'] == 'comparable' and v['valid']['passed']],
                      n_valid_pass_reference=[k for k, v in res['cells'].items() if v['group'] == 'reference' and v['valid']['passed']])
(L.H / 'valid_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
bt = []
for k, R in res['cells'].items():
    for w in WINS:
        s = R[w]; cur = res['current'][w]
        bt.append(dict(cell=k, group=R['group'], window=w, cagr=s['cagr'], d_cagr=s['cagr'] - cur['cagr'], maxdd=s['maxdd'], d_maxdd=s['maxdd'] - cur['maxdd'], excess=s['excess'], t=s['t'],
                       IR=s['IR'], d_IR=s['IR'] - cur['IR'], cold_excess=s['cold_excess'], d_cold=s['cold_excess'] - cur['cold_excess'], ex_top5pct_cagr=s['ex_top5pct_cagr'],
                       d_ex_top5=s['ex_top5pct_cagr'] - cur['ex_top5pct_cagr'], cagr_kappa0=s['cagr_kappa0'], passed=s['passed']))
bt = pd.DataFrame(bt); bt.to_csv(L.H / 'blend_table.csv', index=False, float_format='%.5f')
pd.set_option('display.width', 280); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 200)
P = lambda d: {k: (round(v * 100, 2) if k in ('cagr', 'maxdd', 'excess', 'cold_excess', 'ex_top5pct_cagr') else (round(v, 3) if isinstance(v, float) else v)) for k, v in d.items()}
for w in WINS: print('現行', w, P(res['current'][w]), '0050', {k: round(v * 100, 2) for k, v in res['bench'][w].items()})
v = bt[bt.window == 'valid'].set_index('cell').drop(columns=['group', 'window'])
pc = ['cagr', 'd_cagr', 'maxdd', 'd_maxdd', 'excess', 'cold_excess', 'd_cold', 'ex_top5pct_cagr', 'd_ex_top5', 'cagr_kappa0']
v[pc] = (v[pc] * 100).round(2); print(v.round(3).to_string())
f = bt[bt.window == 'full'].set_index('cell').drop(columns=['group', 'window']); f[pc] = (f[pc] * 100).round(2); print(f.round(3).to_string())
print(ct.round(3).T.to_string())
print(json.dumps(res['env'], ensure_ascii=False)); print(json.dumps(res['verdict'], ensure_ascii=False))
al = pd.DataFrame({k: v['valid'] for k, v in res['candidates_alone'].items()}).T; print(al.round(4).to_string())
al = pd.DataFrame({k: v['full'] for k, v in res['candidates_alone'].items()}).T; print(al.round(4).to_string())
print(json.dumps(res['sens'], ensure_ascii=False)[:1500])
