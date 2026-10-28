#!/usr/bin/env python3
"""穩健性與事後比較（PREREG 之後；不改凍結規格，結果寫進 results.json 的 ROBUST，事後比較一律標示「事後」）。
 (a) 事後：衝擊分母改用全日 V60（o4 審查的口徑 0.1%＋0.5σ√(Q/ADV)，參與上限 20%×ADV）——用來和 o4 的 1,000 萬 t 2.21 銜接，不是判定欄。
 (b) 分年：各級距（主格 A）對零衝擊 AB_V4 的年報酬差。
 (c) 分產業、去掉最大成本股：seed 0～9、300 萬與 1,000 萬（FIX、A）的衝擊成本依產業與個股集中度。
 (d) 主策略 main_v2（EXFIN n10 normal）與零衝擊 AB_V4 的對照。
用法：python3 -I -B robust.py
"""
import sys, json, math, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import ecore as E

res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert 'PREREG' in res and 'RESULTS' in res
MODEL = json.loads((E.DATA / 'model_spec.json').read_text(encoding='utf-8'))
_D = {}
def _init(): _D['D'] = E.Data()
def job(a):
    seed, tier, ver, kind = a
    D = _D['D']
    m = dict(MODEL)
    if kind == 'adv':
        m['oa'] = dict(MODEL['oa']); m['oa']['vars'] = ['const']; m['oa']['coef'] = [math.log(1.0 / MODEL['oa']['kappa'])]
    nav, info = E.simulate(D, seed, tier=E.TIERS[tier], version=ver, fee='A', model=E.ExecModel(m), collect=(kind == 'rows'))
    out = {'a': a, 'nav': nav.to_numpy()}
    if kind == 'rows':
        out['rows'] = [(r[0], r[1], r[3], r[6], r[7]) for r in info['rows'] if r[5] == 'lot']
    return out

if __name__ == '__main__':
    T0 = time.time()
    D = E.Data(); IDXN = D.IDX[D.t0:D.t1 + 1]; sd = D.sig_days
    E50 = D.E50.reindex(IDXN); AB = D.AB.reindex(IDXN); AB = AB / AB.iloc[0]
    jobs = [(s, t, v, 'adv') for s in range(40) for t in E.TIERS for v in ('GROW', 'FIX')]
    jobs += [(s, t, 'FIX', 'rows') for s in range(10) for t in ('300萬', '1000萬')]
    with ProcessPoolExecutor(max_workers=4, initializer=_init) as ex:
        OUT = list(ex.map(job, jobs, chunksize=4))
    print('sims', time.time() - T0, flush=True)
    rob = {}
    adv = {}
    for t in E.TIERS:
        for v in ('GROW', 'FIX'):
            R = pd.DataFrame({o['a'][0]: pd.Series(o['nav'], index=IDXN).pct_change() for o in OUT if o['a'][1:] == (t, v, 'adv')})
            nav = (1 + R.mean(axis=1).fillna(0)).cumprod()
            adv[f'{t}|{v}'] = {'CAGR': E.cagr_mdd(nav, E.A_, E.B_)[0], 't': E.tstat(E.excess(nav, E50, sd)),
                               'HOLD_CAGR': E.cagr_mdd(nav, *E.PER['HOLD'])[0], 'SEL_t': E.tstat(E.excess(nav, E50, sd, *E.PER['SEL']))}
    rob['posthoc_ADV_denominator_impact'] = adv
    NA = pd.read_parquet(HERE / 'navs_abavg.parquet')
    yr = {}
    rab = AB.pct_change()
    for t in E.TIERS:
        for v in ('GROW', 'FIX'):
            r = NA[f'main|{t}|{v}|A'].pct_change()
            yr[f'{t}|{v}'] = {int(y): round(float((1 + g).prod() - (1 + rab[g.index].fillna(0)).prod()) * 100, 2) for y, g in r.iloc[1:].groupby(r.index[1:].year)}
    rob['yearly_diff_vs_zero_impact_pp'] = yr
    sl = pd.read_csv(HERE / 'raw' / 'screener' / 'stock_list.csv', dtype=str).set_index('code')['industry']
    conc = {}
    for t in ('300萬', '1000萬'):
        rows = [r for o in OUT if o['a'][1:] == (t, 'FIX', 'rows') for r in o['rows']]
        df = pd.DataFrame(rows, columns=['t', 'j', 'side', 'val', 'imp'])
        df['cost'] = df.val * (df.imp - 0.001) / (1 + df.side * df.imp)
        df['code'] = [D.CODES[int(j)] for j in df.j]; df['ind'] = df.code.map(sl).fillna('（已下市／無產業別）')
        tot = df.cost.sum()
        bi = (df.groupby('ind').cost.sum() / tot).sort_values(ascending=False)
        bs = (df.groupby('code').cost.sum() / tot).sort_values(ascending=False)
        vi = (df.groupby('ind').val.sum() / df.val.sum())
        conc[t] = {'impact_share_by_industry_top8': {k: [round(float(v), 3), round(float(vi[k]), 3)] for k, v in bi.head(8).items()},
                   'top1_stock': [bs.index[0], round(float(bs.iloc[0]), 3)], 'top10_stocks_share': round(float(bs.head(10).sum()), 3), 'n_stocks': int(len(bs))}
    rob['impact_concentration_FIX_seed0_9'] = conc
    mv2 = pd.read_parquet(E.HERE.parent / 'main_v2' / 'navs_gapfix.parquet')['EXFIN|n10|normal'].reindex(IDXN)
    mv2 = mv2 / mv2.iloc[0]
    rob['main_v2_EXFIN_n10'] = {'CAGR': E.cagr_mdd(mv2, E.A_, E.B_)[0], 't': E.tstat(E.excess(mv2, E50, sd)), 'HOLD_CAGR': E.cagr_mdd(mv2, *E.PER['HOLD'])[0]}
    rob['note'] = '(a) 為事後比較，不是判定欄；(b)～(d) 為描述。'
    res['ROBUST'] = rob
    (HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)), encoding='utf-8')
    print(json.dumps(rob, ensure_ascii=False, default=float)[:5000])
