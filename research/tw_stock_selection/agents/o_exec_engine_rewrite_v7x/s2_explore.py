#!/usr/bin/env python3
"""步驟二：探索（只用挑選期 2006-04～2019-04-30，原設定引擎 v0 解讀，B10、j 0～12、seed 0～4）。
看四項偏樂觀設定在機制上是否看得到：
 (1) FIX 每次換股的提領／注資占權益比例（未計成本的交易量）；
 (2) 處置期賣出筆數與「賣出金額÷當日成交值」；
 (3) 2020 前盤後零股單量÷(s_dec×V60)（若 >ratio_pre 就會被代理上限擋下）；
 (4) 開盤量前視：成交日「當日成交值÷V60_{t−1}」分布（>1 表示原設定用到比事前預期更大的開盤量）；全部股日的同一比例作對照。
另報 Q/OA 參與率（教訓 7：>10% 時 TICK 不是下界）。"""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_data as D, x_exec as X

HERE = Path(__file__).resolve().parent
T1 = int(np.searchsorted(D.DATES, np.datetime64('2019-04-30'), side='right')) - 1
CFGS = [(3e5, 'FIX'), (1e6, 'FIX'), (1e7, 'FIX'), (1e7, 'GROW'), (3e5, 'GROW')]


def job(key_pl):
    key, pl = key_pl
    res = {}
    for tier, mode in CFGS:
        nav, acc, trk, part = X.run_track(pl, dict(tier=tier, mode=mode, model='K5T', cost='normal'), t1=T1)
        res[f'{int(tier/1e4)}萬|{mode}'] = dict(acc=acc, trk={k: list(map(float, v)) for k, v in trk.items()}, part=list(map(float, part)))
    return key, res


def q(v, ps=(10, 50, 90)):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if not len(v): return None
    return {f'p{p}': float(np.percentile(v, p)) for p in ps} | {'n': int(len(v)), 'mean': float(v.mean())}


if __name__ == '__main__':
    P = D.plans('AB_V7_B10', js=list(range(13)), seeds=list(range(5)))
    agg = {}
    with Pool(4) as pool:
        for key, res in pool.imap_unordered(job, sorted(P.items(), key=lambda kv: kv[0])):
            for c, r in res.items():
                a = agg.setdefault(c, dict(acc=[], disp_ratio=[], odd_pre_ratio=[], flow_ratio=[], oa_ratio=[], part=[]))
                a['acc'].append(r['acc'])
                for k in ('disp_ratio', 'odd_pre_ratio', 'flow_ratio', 'oa_ratio'): a[k] += r['trk'][k]
                a['part'] += r['part']
    out = {'period': '2006-04 ～ 2019-04-30（挑選期）', 'plans': 'B10 j0～12 × seeds 0～4（65 份名單）', 'engine': '原設定（interp v0＋併購延遲 v1）、K5T、一般成本'}
    for c, a in agg.items():
        acc = {k: float(np.mean([x[k] for x in a['acc']])) for k in a['acc'][0]}
        fr = np.abs(np.array(a['flow_ratio']))
        out[c] = dict(n_disp_sells_per_plan=acc['n_disp_sells'], n_cap_hit_per_plan=acc['n_cap_hit'], deferred_days_per_plan=acc['deferred_days'],
                      disp_sell_value_over_day_value=q(a['disp_ratio']),
                      disp_share_over_20pct=float(np.mean(np.array(a['disp_ratio']) > 0.2)) if a['disp_ratio'] else None,
                      odd_pre2020_order_over_sdecV60=q(a['odd_pre_ratio']),
                      odd_pre2020_share_over_0p5=float(np.mean(np.array(a['odd_pre_ratio']) > 0.5)) if a['odd_pre_ratio'] else None,
                      fix_abs_flow_over_equity=q(fr) if len(fr) else None,
                      fix_abs_flow_annual_sum_mean=float(fr.sum() / len(a['acc']) / 13.0) if len(fr) else None,
                      exec_day_value_over_V60=q(a['oa_ratio']),
                      exec_day_value_gt_V60=float(np.mean(np.array(a['oa_ratio']) > 1)) if a['oa_ratio'] else None,
                      part_Q_over_OA=q(a['part'], (50, 90)),
                      part_gt10pct=float(np.mean(np.array(a['part']) > 0.10)) if a['part'] else None)
    # 對照：挑選期前 300 池所有股日的 value_t ÷ V60_{t−1}
    t0 = int(np.searchsorted(D.DATES, np.datetime64('2006-04-01')))
    VR = D.VRp[t0:T1 + 1]; m = np.isfinite(VR) & (VR <= 300)
    rr = (D.V[t0:T1 + 1] / D.V60p[t0:T1 + 1])[m]
    out['pool_all_stockdays_value_over_V60'] = q(rr)
    out['note'] = ('exec_day_value_over_V60 是整股成交日（換股日與順延日）的 當日成交值÷V60_{t−1}；中位數 >1 表示原設定用當日量算 OA，比事前預測寬鬆。'
                   'odd_pre2020_order_over_sdecV60 >1 表示該筆盤後零股超過『2026 盤中零股占比×V60』。')
    (HERE / 'explore.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1)[:6000])
