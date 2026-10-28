# -*- coding: utf-8 -*-
"""樣本外（VAL、HOLD、POST）與全期報告。斷言 PREREG 已存在；規格只讀 PREREG。"""
import sys, json, math, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import rlib as L
RES_P = HERE / 'results.json'
res = json.loads(RES_P.read_text(encoding='utf-8'))
assert 'PREREG' in res and 'PREREG_sha256' in res, 'PREREG 不存在：不得計算樣本外'
PRE = res['PREREG']
assert hashlib.sha256(json.dumps(PRE, ensure_ascii=False, sort_keys=True, default=float).encode()).hexdigest() == res['PREREG_sha256']
for f in ('main_navs.parquet', 'main_plan.json', 'build_main.py', 'iterations.json', 'explore.json'):
    assert hashlib.sha256((HERE / f).read_bytes()).hexdigest() == PRE['files_sha256'][f], f
# rlib.py 在凍結後加了 A_override 選項（預設不變）：重算 SEL 預算格點逐格比對
for k, v in res['SEL_grid_budget'].items():
    w, ru = k.split('|'); r = L.sim(int(w) / 100, ru, *L.PER['SEL'])
    assert abs(L.cagr(r['nav']) * 100 - v['CAGR']) < 1e-9, k
O = {}
PERS = ['SEL', 'S1', 'S2', 'VAL', 'HOLD', 'POST', 'ALL']
W = [0.2, 0.3, 0.4]
def r2(x): return None if x is None else (round(float(x), 4) if isinstance(x, (float, np.floating)) else x)
# 1. 全格點 × 期間（新投資人起點）
grid = {}
for cost in ('normal', 'stress'):
    for P in PERS:
        a, b = L.PER[P]
        for w in W:
            m = L.sim(w, 'M', a, b, cost=cost)['nav']
            for ru in L.RULES:
                r = L.sim(w, ru, a, b, cost=cost); s = L.stats(r['nav'])
                s.update(n_trades=r['n_trades'], paired_vs_M=L.paired(r['nav'], m))
                grid[f'{cost}|{P}|{int(w*100)}|{ru}'] = s
O['grid'] = grid
# 1b. 連續路徑（2006 起不重置）的分段：主選版與每月
cont = {}
for cost in ('normal', 'stress'):
    for w in W:
        full = {ru: L.sim(w, ru, *L.PER['ALL'], cost=cost)['nav'] for ru in ('M', PRE['selected'][str(int(w*100))])}
        for P in ('VAL', 'HOLD', 'POST'):
            a, b = L.PER[P]
            i0 = max(int(L.IX.searchsorted(a)) - 1, 0); i1 = int(L.IX.searchsorted(b, 'right')) - 1
            seg = {k: v.iloc[i0:i1 + 1] / v.iloc[i0] for k, v in full.items()}
            ru = PRE['selected'][str(int(w*100))]
            cont[f'{cost}|{P}|{int(w*100)}'] = dict(rule=ru, paired_vs_M=L.paired(seg[ru], seg['M']), CAGR_rule=L.cagr(seg[ru]) * 100, CAGR_M=L.cagr(seg['M']) * 100)
O['continuous_path'] = cont
# 2. 判定（w=30 主選版）
def judge(cost):
    ru = PRE['primary']['rule']; w = PRE['primary']['w0050']
    a, b = L.PER['SEL']
    n1 = L.sim(w, ru, a, b, cost=cost)['nav']; n0 = L.sim(w, 'M', a, b, cost=cost)['nav']
    pS = L.paired(n1, n0)
    yd = L.yearly_diff(n1, n0); best = int(yd.idxmax())
    drop = L.ann_diff_drop_year(n1, n0, best)
    a2, b2 = L.PER['POST']
    pP = L.paired(L.sim(w, ru, a2, b2, cost=cost)['nav'], L.sim(w, 'M', a2, b2, cost=cost)['nav'])
    c = dict(SEL_diff_ge0=bool(pS['ann_diff_pp'] >= 0), POST_t_gt_m1=bool(pP['t'] > -1), SEL_drop_best_year_ge0=bool(drop >= 0))
    return dict(rule=ru, SEL=pS, POST=pP, SEL_yearly_diff_pp={int(k): float(v) for k, v in yd.items()}, best_year=best, SEL_ann_diff_drop_best_year=drop, checks=c, all_pass=bool(all(c.values())))
J = {c: judge(c) for c in ('normal', 'stress')}
J['final_pass'] = bool(J['normal']['all_pass'] and J['stress']['all_pass'])
O['judgment'] = J
# 3. 端點敏感度（主選版 vs 每月，POST 與 ALL）
def endpoint(P, cost='normal'):
    a, b = L.PER[P]; w = PRE['primary']['w0050']; ru = PRE['primary']['rule']
    d = (L.win(L.sim(w, ru, a, b, cost=cost)['nav']) - L.win(L.sim(w, 'M', a, b, cost=cost)['nav'])).dropna()
    srt = d.sort_values(ascending=False)
    out = dict(t=L.tstat(d), t_drop_last=L.tstat(d.iloc[:-1]))
    for k in (1, 3, 5): out[f't_drop_best{k}'] = L.tstat(d.drop(srt.index[:k]))
    return out
O['endpoint'] = {P: endpoint(P) for P in ('SEL', 'POST', 'ALL')}
# 4. 追蹤誤差預算表：SEL 選出的格子在各期間
bt = {}
for T, cell in PRE['budget_cells'].items():
    if cell is None: bt[T] = None; continue
    w, ru = cell.split('|'); w = int(w) / 100
    row = dict(cell=cell)
    for P in PERS:
        a, b = L.PER[P]
        r = L.sim(w, ru if w > 0 else 'M', a, b); s = L.stats(r['nav'])
        row[P] = dict(CAGR=s['CAGR'], TE=s['TE'], MDD=s['MDD'], longest_lag_m=s['longest_lag_m'], roll36_lag_frac=s['roll36_lag_frac'],
                      ex_pp=s['ex_pp'], t_vs0050=s['t_vs0050'], n_trades=r['n_trades'])
    bt[T] = row
O['budget_table'] = bt
# 5. 前緣（每月與 NONE、w 0～50%）與放棄風險代理
fr = {}
for ru in ('M', 'NONE', 'B10'):
    for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        for P in PERS:
            a, b = L.PER[P]
            r = L.sim(w, ru if w > 0 else 'M', a, b); s = L.stats(r['nav'])
            fr[f'{ru}|{int(w*100)}|{P}'] = dict(CAGR=s['CAGR'], TE=s['TE'], MDD=s['MDD'], longest_lag_m=s['longest_lag_m'],
                                               roll12_lag_frac=s['roll12_lag_frac'], roll36_lag_frac=s['roll36_lag_frac'], t_vs0050=s['t_vs0050'], ex_pp=s['ex_pp'])
O['frontier'] = fr
slope = {}
for P in PERS:
    x0 = fr[f'M|0|{P}']; xs = [fr[f'M|{w}|{P}'] for w in (10, 20, 30, 40, 50)]
    slope[P] = dict(cost_pp_per_TEpp={w: (x0['CAGR'] - x['CAGR']) / (x0['TE'] - x['TE']) if x0['TE'] != x['TE'] else None for w, x in zip((10, 20, 30, 40, 50), xs)},
                    TE_ratio_vs_1mw={w: x['TE'] / x0['TE'] / (1 - w / 100) for w, x in zip((10, 20, 30, 40, 50), xs)})
O['frontier_slope'] = slope
# 6. 三者並列（＋0050、同池等權）
def trio(cost):
    out = {}
    for P in PERS:
        a, b = L.PER[P]
        rows = {}
        A = L.sim(0.0, 'M', a, b, cost=cost)['nav']; rows['MAIN100'] = L.stats(A, ew=True)
        rows['MAIN70_0050_30_M'] = L.stats(L.sim(0.3, 'M', a, b, cost=cost)['nav'])
        rows['MAIN70_2330_30_M'] = L.stats(L.sim(0.3, 'M', a, b, cost=cost, B='A2330')['nav'])
        rows['MAIN70_0050_30_SELECTED'] = L.stats(L.sim(0.3, PRE['primary']['rule'], a, b, cost=cost)['nav'])
        rows['E50'] = L.stats(L.bench('E50', a, b)); rows['POOL_EW_gross'] = L.stats(L.bench('POOL_EW|gross', a, b))
        rows['A2330'] = L.stats(L.bench('A2330', a, b))
        out[P] = rows
    return out
O['trio'] = {c: trio(c) for c in ('normal', 'stress')}
# 7. 成本損益兩平（主策略單邊滑價）
SL = pd.read_parquet(HERE / 'main_navs_slip.parquet')
be = {}
for col in SL.columns:
    row = {}
    for P in ('ALL', 'SEL', 'POST', 'HOLD'):
        a, b = L.PER[P]
        for nm, w, ru in (('MAIN100', 0.0, 'M'), ('M70_30_M', 0.3, 'M'), ('M70_30_SEL', 0.3, PRE['primary']['rule'])):
            s = L.stats(L.sim(w, ru, a, b, A_override=SL[col])['nav'])
            row[f'{nm}|{P}'] = dict(CAGR=s['CAGR'], ex_pp=s['ex_pp'], t=s['t_vs0050'])
    be[col] = row
O['cost_grid'] = be
def interp(key, field, target):
    xs = [float(c[4:]) for c in SL.columns if c.startswith('slip')]
    ys = [be[c][key][field] for c in SL.columns if c.startswith('slip')]
    for i in range(len(xs) - 1):
        if (ys[i] - target) * (ys[i + 1] - target) <= 0 and ys[i] != ys[i + 1]:
            return xs[i] + (target - ys[i]) * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i])
    return None
O['breakeven_slip_one_side'] = {f'{k}|{fld}={tg}': interp(k, fld, tg) for k in ('MAIN100|ALL', 'M70_30_M|ALL', 'M70_30_SEL|ALL', 'MAIN100|POST', 'M70_30_M|POST', 'MAIN100|HOLD', 'M70_30_M|HOLD')
                                for fld, tg in (('t', 2.0), ('ex_pp', 0.0))}
# 再平衡本身的成本拖累：w=30 每月，移轉成本歸零 vs 一般
a, b = L.PER['ALL']
r1 = L.sim(0.3, 'M', a, b); 
L.C['zero'] = {k: 0.0 for k in L.C['normal'] if k != 'note'}
r0 = L.sim(0.3, 'M', a, b, cost='zero') if False else None
O['rebalance_turnover_ALL_w30_M'] = dict(n_trades=r1['n_trades'], sum_transfer_frac=r1['turnover'], approx_cost_pp_per_year=r1['turnover'] * (L.C['normal']['stock_sell'] + L.C['normal']['etf_buy']) / ((L.IX[-1] - L.IX[0]).days / 365.25) * 100)
# 8. 實盤打折（0050＋0.5×超額）
O['haircut'] = {P: {k: O['trio']['normal'][P]['E50']['CAGR'] + 0.5 * (O['trio']['normal'][P][k]['CAGR'] - O['trio']['normal'][P]['E50']['CAGR'])
                    for k in ('MAIN100', 'MAIN70_0050_30_M', 'MAIN70_2330_30_M', 'MAIN70_0050_30_SELECTED')} for P in ('ALL', 'POST', 'HOLD')}
# 9. 逐年（ALL 連續路徑）：主策略、70/30 每月、選版、0050
a, b = L.PER['ALL']
navs = {'MAIN100': L.sim(0.0, 'M', a, b)['nav'], 'M70_30_M': L.sim(0.3, 'M', a, b)['nav'], 'M70_30_SEL': L.sim(0.3, PRE['primary']['rule'], a, b)['nav'],
        'M70_2330_30_M': L.sim(0.3, 'M', a, b, B='A2330')['nav'], 'E50': L.bench('E50', a, b)}
yr = {}
for k, v in navs.items():
    y = v.groupby(v.index.year).last(); y = pd.concat([pd.Series([v.iloc[0]], index=[2005]), y]); yr[k] = {int(i): float(x) for i, x in ((y / y.shift(1) - 1) * 100).iloc[1:].items()}
O['yearly_ALL'] = yr
wpath = L.sim(0.3, PRE['primary']['rule'], a, b)['wb']
O['selected_wB_path_ALL'] = {str(d.date()): float(x) for d, x in wpath.iloc[::12].items()}
# 10. 資金可行性（100 萬／300 萬，主策略腿 70%）
MP = L.MP
feas = {}
for cap in (1_000_000, 3_000_000):
    rows = []
    for p in MP['plan']:
        alloc = cap * 0.7 / 10
        for pr in p['open_raw']:
            if not (pr and pr > 0): continue
            lot = pr * 1000
            whole = math.floor(alloc / lot) * lot
            rows.append(dict(y=int(p['exec'][:4]), lot_ok=alloc >= lot, whole_frac=whole / alloc))
    df = pd.DataFrame(rows)
    feas[cap] = dict(slots=len(df), frac_slots_below_one_lot=float((~df.lot_ok).mean()), mean_whole_lot_frac=float(df.whole_frac.mean()),
                     by_period={k: dict(frac_below_one_lot=float((~g.lot_ok).mean()), mean_whole_lot_frac=float(g.whole_frac.mean()))
                                for k, g in (('2006_2018', df[df.y <= 2018]), ('2019_2023', df[(df.y >= 2019) & (df.y <= 2023)]), ('2024_2026', df[df.y >= 2024]))})
    # 2330 30%
    c2330 = L.NAVS.index.to_series().map(lambda d: None)
O['capital_feasibility_main_leg'] = feas
C = pd.read_parquet('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/close_raw.parquet', columns=['2330'])['2330']
px = {int(y): float(C[C.index.year == y].dropna().iloc[-1]) for y in (2006, 2010, 2015, 2018, 2020, 2023, 2024, 2025, 2026)}
O['p2330_year_end'] = px
O['p2330_lot_vs_30pct'] = {cap: {y: dict(lot_ntd=p * 1000, alloc=cap * 0.3, lots=math.floor(cap * 0.3 / (p * 1000))) for y, p in px.items()} for cap in (1_000_000, 3_000_000)}
# 每月移轉金額分布（w=30 每月，ALL）
r = L.sim(0.3, 'M', *L.PER['ALL'])
wb = r['wb']; dev = (wb - 0.3).abs()
O['monthly_transfer_frac'] = dict(median=float(dev.median()), p90=float(dev.quantile(0.9)), max=float(dev.max()))
O['min_fee_drag'] = {cap: dict(orders_per_year=24, fee_floor_ntd=20, drag_pp=24 * 20 / cap * 100) for cap in (1_000_000, 3_000_000)}
res['OOS'] = O
RES_P.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
# 摘要
def p(*a): print(*a, flush=True)
p('JUDGE', json.dumps({c: dict(SEL=round(J[c]['SEL']['ann_diff_pp'], 3), SELt=round(J[c]['SEL']['t'], 2), POST=round(J[c]['POST']['ann_diff_pp'], 3), POSTt=round(J[c]['POST']['t'], 2),
                       drop=round(J[c]['SEL_ann_diff_drop_best_year'], 3), best=J[c]['best_year'], checks=J[c]['checks']) for c in ('normal', 'stress')}, ensure_ascii=False, default=str), J['final_pass'])
for c in ('normal', 'stress'):
    for P in PERS:
        t = O['trio'][c][P]
        p('TRIO', c, P, {k: (round(v['CAGR'], 2), round(v['t_vs0050'], 2) if v['t_vs0050'] == v['t_vs0050'] else None, round(v['MDD'], 1), round(v['TE'], 1), v['longest_lag_m']) for k, v in t.items()})
for P in ('VAL', 'HOLD', 'POST', 'ALL'):
    for w in (20, 30, 40):
        p('GRID', P, w, {ru: (round(grid[f'normal|{P}|{w}|{ru}']['CAGR'], 2), round(grid[f'normal|{P}|{w}|{ru}']['paired_vs_M']['ann_diff_pp'], 2), round(grid[f'normal|{P}|{w}|{ru}']['paired_vs_M']['t'], 2) if ru != 'M' else 0, grid[f'normal|{P}|{w}|{ru}']['n_trades']) for ru in L.RULES})
p('CONT', {k: (round(v['paired_vs_M']['ann_diff_pp'], 2), round(v['paired_vs_M']['t'], 2)) for k, v in cont.items()})
p('ENDPOINT', O['endpoint'])
p('BUDGET', json.dumps({T: (v['cell'], {P: (round(v[P]['CAGR'], 1), round(v[P]['TE'], 1), round(v[P]['MDD'], 1), v[P]['longest_lag_m']) for P in PERS}) for T, v in bt.items()}, ensure_ascii=False))
p('SLOPE', json.dumps(slope, default=float))
p('BE', O['breakeven_slip_one_side'])
p('HAIR', O['haircut'])
p('FEAS', json.dumps(feas), O['p2330_lot_vs_30pct'][1000000][2026], O['monthly_transfer_frac'], O['rebalance_turnover_ALL_w30_M'])
for ru in ('M', 'NONE'):
    for P in ('S1', 'S2', 'VAL', 'HOLD', 'ALL'):
        p('ABANDON', ru, P, {w: (round(fr[f'{ru}|{w}|{P}']['TE'], 1), fr[f'{ru}|{w}|{P}']['longest_lag_m'], round(fr[f'{ru}|{w}|{P}']['roll36_lag_frac'], 2) if fr[f'{ru}|{w}|{P}']['roll36_lag_frac'] is not None else None, round(fr[f'{ru}|{w}|{P}']['roll12_lag_frac'], 2), round(fr[f'{ru}|{w}|{P}']['CAGR'], 1)) for w in (0, 10, 20, 30, 40, 50)})
