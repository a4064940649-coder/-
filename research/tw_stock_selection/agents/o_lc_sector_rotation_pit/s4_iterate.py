#!/usr/bin/env python3
"""迭代（只用挑選期 SEL 2014-01～2016-12）：40 格 × 13 j，一般成本與零成本，對 B 的 j 中位數 IR。
版本（最多 3 次迭代；原因寫在 iterations.json）：
  V0 事前設定（產業合併門檻 2、超配上限 3×B、緩衝帶 k+2）
  V1 合併門檻 2→3（探索：前 5 名平均 1.9 個是 B<2% 的小產業）
  V2 緩衝帶 k+2→k+3（探索：預設格一般成本吃掉約一半的毛超額）
  V3 超配上限 3×B→2×B（同 V1 的小產業疑慮，改上限公式而不是合併）
選定規則（事前寫死）：全部版本×40 格依「一般成本 j 中位數 IR（對 B）」排序，依序檢查
  (a) 3 個相鄰格（k 互換、A 互換、頻率互換）一般成本 j 中位數超額與它同號且 ≥ 它的 50%；
  (b) 零成本安慰劑（1,000 組隨機產業排名，持續度相同）百分位 ≥ p90。
  第一個兩者都過的就是選定版；都不過 → 選定＝事前預設（COMP、k5、A30、每月、V0），並註明「沒有格點通過選定規則」。"""
import sys, json, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lclib as X, lcrun as R
M0, M1, ME = '2014-01', '2016-12', '2017-01'
VERS = {'V0': dict(R.P0), 'V1': dict(R.P0, sect_min=3), 'V2': dict(R.P0, buf=3), 'V3': dict(R.P0, cap_mult=2.0)}
WHY = {'V0': '事前設定', 'V1': '探索：COMP 前 5 名平均 1.9 個產業 B 權重 < 2%，擔心小產業主導 → 允許的「產業合併門檻 2→3」',
       'V2': '探索：預設格月周轉 8.9%、一般成本 0.065pp／月，約吃掉一半毛超額（0.104pp）→ 允許的「緩衝帶 k+2→k+3」',
       'V3': '同 V1 的小產業疑慮，改用允許的「上限公式」：單一產業超配 ≤ min(A/k, 2×B)'}
R.init()
PERS = json.loads((X.HERE / 'explore.json').read_text(encoding='utf-8'))['persistence_jmed']

def gjob(a):
    j, vn = a; P = VERS[vn]
    B = R.build(j, M0, M1, ME, P=P)
    res = R.run_all(B)
    e50 = X.bench_win(B['days'])
    out = {}
    for nm in list(B['plans']) + ['B']:
        d = {}
        for cost in ('normal', 'zero'):
            s, inf = res[(nm, cost)]; sb = res[('B', cost)][0]
            d[cost] = dict(vsB=X.stats(s - sb), vs0050=X.stats(s - e50), turnover=float(np.mean(inf['turnover'])), cost_pp=float(np.mean(inf['cost']) * 100),
                           cagr=X.cagr(s))
        out[nm] = d
    return vn, j, out

def agg(rows):
    """rows：{j: out}。回傳每格 j 中位數。"""
    names = rows[0].keys(); A = {}
    for nm in names:
        r = {}
        for cost in ('normal', 'zero'):
            for k in ('mean_pp', 't', 'IR', 'TE'):
                v = [rows[j][nm][cost]['vsB'][k] for j in rows]
                r[f'{cost}_vsB_{k}'] = float(np.median(v))
                if k in ('mean_pp', 't'): r[f'{cost}_vsB_{k}_p10'] = float(np.percentile(v, 10))
            for k in ('mean_pp', 't'):
                r[f'{cost}_vs0050_{k}'] = float(np.median([rows[j][nm][cost]['vs0050'][k] for j in rows]))
            r[f'{cost}_turnover'] = float(np.median([rows[j][nm][cost]['turnover'] for j in rows]))
            r[f'{cost}_cost_pp'] = float(np.median([rows[j][nm][cost]['cost_pp'] for j in rows]))
        A[nm] = r
    return A

def neighbors(nm):
    s, k, A, f = nm.split('|')
    k2 = 'k5' if k == 'k3' else 'k3'; A2 = 'A15' if A == 'A30' else 'A30'; f2 = 'B2' if f == 'M' else 'M'
    return ['|'.join([s, k2, A, f]), '|'.join([s, k, A2, f]), '|'.join([s, k, A, f2])]

def cfg_tuple(nm):
    s, k, A, f = nm.split('|'); return (s, int(k[1:]), int(A[1:]) / 100, f)

if __name__ == '__main__':
    t0 = time.time()
    with Pool(4) as p: res = p.map(gjob, [(j, v) for v in VERS for j in range(13)])
    byv = {v: {} for v in VERS}
    for vn, j, out in res: byv[vn][j] = out
    AG = {v: agg(byv[v]) for v in VERS}
    print('grid done', round(time.time() - t0), flush=True)
    cand = sorted([(AG[v][nm]['normal_vsB_IR'], v, nm) for v in VERS for nm in AG[v] if nm != 'B'], reverse=True)
    log = []; chosen = None
    pool = Pool(4)
    for ir, v, nm in cand:
        r = AG[v][nm]
        if not (r['normal_vsB_mean_pp'] > 0): log.append(dict(version=v, cfg=nm, IR=ir, result='超額 ≤ 0，停止往下找')); break
        nb = neighbors(nm); best = r['normal_vsB_mean_pp']
        nbv = {n: AG[v][n]['normal_vsB_mean_pp'] for n in nb}
        ok_a = all(x > 0 and x >= 0.5 * best for x in nbv.values())
        ent = dict(version=v, cfg=nm, IR=ir, mean_pp=best, neighbors=nbv, a_pass=ok_a)
        if ok_a:
            sig = nm.split('|')[0]; phi = R.phi_from_rank_ac(PERS[sig])
            draws = pool.map(R.placebo_draw, [(1000 + i, cfg_tuple(nm), phi, M0, M1, ME, VERS[v]) for i in range(1000)], chunksize=25)
            g = np.array([d[0] for d in draws]); n = np.array([d[1] for d in draws])
            act_g = r['zero_vsB_mean_pp']; act_n = r['normal_vsB_mean_pp']
            ent.update(phi=phi, placebo_gross_pct=float((g < act_g).mean() * 100), placebo_net_pct=float((n < act_n).mean() * 100),
                       placebo_gross_p90=float(np.percentile(g, 90)), placebo_gross_p95=float(np.percentile(g, 95)), actual_gross=act_g)
            ent['b_pass'] = ent['placebo_gross_pct'] >= 90
            print(ent, flush=True)
            if ent['b_pass']: chosen = (v, nm); log.append(ent); break
        log.append(ent)
        if len(log) >= 12: log.append(dict(note='檢查 12 個候選後停止')); break
    pool.close()
    fallback = chosen is None
    if fallback: chosen = ('V0', R.DEFAULT)
    it = dict(period='SEL 2014-01～2016-12（36 個月；D1 未過，挑選期太短，檢定力很低）', versions={v: dict(params=VERS[v], why=WHY[v]) for v in VERS},
              n_versions=len(VERS), n_cells_tried=len(VERS) * 40, selection_log=log, chosen=dict(version=chosen[0], cfg=chosen[1], fallback_to_default=fallback),
              default=dict(version='V0', cfg=R.DEFAULT, stats=AG['V0'][R.DEFAULT]),
              grid_SEL={v: {nm: {k: round(x, 4) for k, x in AG[v][nm].items()} for nm in AG[v]} for v in VERS},
              top10_by_IR=[dict(version=v, cfg=nm, IR=round(ir, 3)) for ir, v, nm in cand[:10]])
    (X.HERE / 'iterations.json').write_text(X.dumps(it), encoding='utf-8')
    print('chosen', chosen, 'fallback', fallback)
    print('default', {k: round(x, 3) for k, x in AG['V0'][R.DEFAULT].items()})
    print('top10', it['top10_by_IR'])
    print('B', {k: round(x, 3) for k, x in AG['V0']['B'].items() if 'vs0050' in k})
    print('time', round(time.time() - t0))
