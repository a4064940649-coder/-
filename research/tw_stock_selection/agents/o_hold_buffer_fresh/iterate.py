#!/usr/bin/env python3
"""迭代建模（只用挑選期：訊號日 2006-01～2018-12，評價到 2019-01 第一個訊號日收盤）。
用法：python3 -I -B iterate.py <iter_id> <版本名,...>   結果累加進 iterations.json（已存在的版本不重跑）。
最後：python3 -I -B iterate.py select   依 PLAN 的選版規則選版。"""
import sys, json, time, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import hb_core as H
RES = json.loads((H.HERE / 'results.json').read_text(encoding='utf-8'))
assert 'PLAN_sha256' in RES
assert 'PREREG' not in RES, 'PREREG 已存在：迭代只能在凍結前做'
R = H.R
FAM = RES['PLAN']['model_family']
SPECS = {}
for f in 'ABCDE':
    for k, v in FAM[f].items(): SPECS[k] = dict(v, family=f)
# 跨族組合（依挑選期探索與迭代 1～4 決定；理由寫在 X_RATIONALE）
X_SPECS = {
    'X1': dict(K=30, h=0.95, band=0.40, family='X'),
    'X2': dict(delta=0.03, band=0.40, family='X'),
    'X3': dict(K=30, h=0.95, overlap=True, family='X'),
    'X4': dict(K=30, h=0.95, delta=0.03, family='X'),
}
X_RATIONALE = {
    'X1': '探索：排名 11～30 且 52 週比 ≥0.95 的舊股下一窗與新進股無差（+0.04pp，t 0.05），0.90～0.95 顯著較差 → 新鮮度門檻取 0.95；B30h95 是 B 族唯一壓力配對差 > 0 者，再疊漂移帶 D40（D 族壓力配對差較高者）',
    'X2': 'C03 是全部 16 版中壓力配對 t 最高者（1.12），疊 D40',
    'X3': 'B30h95 周轉 50% 未進 35～45% 帶；疊兩期重疊（E2 周轉 36%）以進入周轉帶，同時保留新鮮度條件',
    'X4': 'B30h95＋C03：新鮮舊股強制續抱，其餘舊股要被高出 0.03 分的新股才換',
}
SPECS.update(X_SPECS)
SPECS['BASE'] = dict(family='base')
ITF = H.HERE / 'iterations.json'
IT = json.loads(ITF.read_text(encoding='utf-8')) if ITF.exists() else {'versions': {}, 'iterations': [], 'n_versions_total': 0}

st = H.setup(); S = H.make_S(); P = H.Prep(S)
DAYS = [d for d in P.days if d <= H.SEL_B]
END = [d for d in P.days if d > H.SEL_B][0]
XR = st['XR']
COSTS = {'gross': (0.0, 0.0), 'normal': H.NORMAL, 'stress': H.STRESS, 'flat': H.FLAT, 'cap10m': 'cap'}
SUBS = {'SEL': (H.SEL_A, H.SEL_B), 'S1_2006-2013': (H.SEL_A, pd.Timestamp('2013-12-31')), 'S2a_2014-2018': (pd.Timestamp('2014-01-01'), H.SEL_B),
        'Y2009': (pd.Timestamp('2009-01-01'), pd.Timestamp('2009-12-31')), 'Y2008': (pd.Timestamp('2008-01-01'), pd.Timestamp('2008-12-31'))}
_t0, _t1, SD = H.bounds(DAYS, END)
E50 = H.E50

_BASE = {}
def base_nav(cost):
    if cost not in _BASE: _BASE[cost] = H.run(P, {}, COSTS[cost], DAYS, END, XR)[0]
    return _BASE[cost]

def excess_mean(nav):
    return R.tvs(nav, E50.reindex(nav.index), SD, H.SEL_A, H.SEL_B)

def breakeven(spec, target='mean', lo=0.0, hi=0.03):
    """單邊滑價 s：mean→對 0050 月均超額＝0；t2→對 0050 t＝2.0。"""
    def f(s):
        nav, _ = H.run(P, spec, H.rates_slip(s), DAYS, END, XR)
        x = excess_mean(nav)
        return x['mean_pp'] if target == 'mean' else x['t'] - 2.0
    flo, fhi = f(lo), f(hi)
    if flo <= 0: return 0.0
    if fhi > 0: return hi
    for _ in range(16):
        m = 0.5 * (lo + hi)
        if f(m) > 0: lo = m
        else: hi = m
    return 0.5 * (lo + hi)

def evaluate(name):
    spec = {k: v for k, v in SPECS[name].items() if k != 'family'}
    o = dict(spec=SPECS[name])
    for c in COSTS:
        nav, lg = H.run(P, spec, COSTS[c], DAYS, END, XR, log_detail=True)
        cg, mdd = R.cagr_mdd(nav, H.SEL_A, END)
        x = dict(CAGR=cg, maxDD=mdd, vs0050=excess_mean(nav), turnover=lg['turnover'])
        if c in ('gross', 'normal', 'stress'):
            det = pd.DataFrame(lg['det'])
            if len(det):
                x['n_fixed_mean'] = float(det['n_fixed'].iloc[1:].mean())
                x['n_cont_mean'] = float(det['n_cont'].iloc[1:].mean())
        if name != 'BASE':
            bn = base_nav(c)
            x['paired'] = {k: R.paired(nav, bn, SD, a, b) for k, (a, b) in SUBS.items()}
            dd = H.window_diff(nav, bn, SD, H.SEL_A, H.SEL_B)
            x['paired_drop_best'] = {f'drop_best_{k}': H.drop_best(dd, k) for k in (1, 3, 5)}
            x['paired_mean_sd'] = dict(mean_pp=float(dd.mean() * 100), sd_pp=float(dd.std(ddof=1) * 100))
        o[c] = x
    o['breakeven_slip_mean0'] = breakeven(spec, 'mean')
    o['breakeven_slip_t2'] = breakeven(spec, 't2')
    return o

def main():
    if sys.argv[1] == 'select':
        return select()
    it_id = sys.argv[1]; names = sys.argv[2].split(',')
    T0 = time.time()
    if 'BASE' not in IT['versions']:
        IT['versions']['BASE'] = evaluate('BASE')
    done = []
    for nm in names:
        if nm in IT['versions']: continue
        IT['versions'][nm] = evaluate(nm); IT['versions'][nm]['iteration'] = it_id
        done.append(nm)
        v = IT['versions'][nm]
        print(f'[{time.time()-T0:5.0f}s] {nm}', 'tov', round(v['normal']['turnover'], 3),
              {c: (round(v[c]['paired']['SEL']['mean_pp'], 3), round(v[c]['paired']['SEL']['t'], 2)) for c in ('gross', 'normal', 'stress', 'cap10m')},
              'be', round(v['breakeven_slip_mean0'] * 100, 3), round(IT['versions']['BASE']['breakeven_slip_mean0'] * 100, 3), flush=True)
    IT['iterations'].append(dict(id=it_id, versions=done, time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                                 note=sys.argv[3] if len(sys.argv) > 3 else ''))
    IT['X_rationale'] = X_RATIONALE
    IT['n_versions_total'] = len([k for k in IT['versions'] if k != 'BASE'])
    assert IT['n_versions_total'] <= 20
    ITF.write_text(json.dumps(IT, ensure_ascii=False, indent=1, default=float), encoding='utf-8')

NEIGH = {'A15': ['A20'], 'A20': ['A15', 'A25'], 'A25': ['A20', 'A30'], 'A30': ['A25'],
         'B20h85': ['B20h90', 'B30h85'], 'B20h90': ['B20h85', 'B20h95', 'B30h90'], 'B20h95': ['B20h90', 'B30h95'],
         'B30h85': ['B30h90', 'B20h85'], 'B30h90': ['B30h85', 'B30h95', 'B20h90'], 'B30h95': ['B30h90', 'B20h95'],
         'C03': ['C05'], 'C05': ['C03', 'C10'], 'C10': ['C05'], 'D20': ['D40'], 'D40': ['D20'], 'E2': [],
         'X1': ['B30h95', 'D40'], 'X2': ['C03', 'D40'], 'X3': ['B30h95', 'E2'], 'X4': ['B30h95', 'C03']}
CONSERV = {'A30': 'A25', 'A25': 'A20', 'A20': 'A15', 'A15': None, 'B30h85': 'B30h90', 'B30h90': 'B30h95', 'B30h95': 'B20h95', 'B20h85': 'B20h90',
           'B20h90': 'B20h95', 'B20h95': None, 'C10': 'C05', 'C05': 'C03', 'C03': None, 'D40': 'D20', 'D20': None, 'E2': None,
           'X1': 'B30h95', 'X2': 'C03', 'X3': 'B30h95', 'X4': 'B30h95'}

def select():
    vs = {k: v for k, v in IT['versions'].items() if k != 'BASE'}
    rows = []
    for k, v in vs.items():
        p = v['stress']['paired']['SEL']
        rows.append(dict(version=k, t_stress=p['t'], mean_stress=p['mean_pp'], turnover=v['normal']['turnover'],
                         in_band=bool(0.35 <= v['normal']['turnover'] <= 0.45)))
    cand = sorted([r for r in rows if r['in_band']], key=lambda r: -r['t_stress'])
    chain = []
    sel = cand[0]['version'] if cand else None
    while sel is not None:
        nb = [n for n in NEIGH.get(sel, []) if n in vs]
        bad = [n for n in nb if vs[n]['stress']['paired']['SEL']['mean_pp'] < 0]
        chain.append(dict(version=sel, neighbors=nb, negative_neighbors=bad))
        if not bad: break
        sel = CONSERV.get(sel)
        while sel is not None and sel not in vs: sel = CONSERV.get(sel)
    IT['selection'] = dict(rule=RES['PLAN']['selection_rule'], table=sorted(rows, key=lambda r: -r['t_stress']), candidates_in_band=[c['version'] for c in cand],
                           chain=chain, selected=sel, n_versions=len(vs))
    ITF.write_text(json.dumps(IT, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    print(json.dumps(IT['selection'], ensure_ascii=False, indent=1, default=float))

if __name__ == '__main__':
    main()
