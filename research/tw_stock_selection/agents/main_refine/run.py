#!/usr/bin/env python3
"""main_refine 第 1 階段：全部改良版本（滑價 0.1%、壓力）× 挑選期／驗證期的統計與判定。保留期在 holdout.py 才計算。
輸出：results.json 的 RUN（追加）、navs.parquet。"""
import json, hashlib, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.dont_write_bytecode = True
import rlib as R
L = R.L

res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
s = json.dumps(res['PREREG'], ensure_ascii=False, sort_keys=True)
assert hashlib.sha256(s.encode()).hexdigest() == res['PREREG_sha256'], 'PREREG 被改過'

SEGS = {'SEL': ('2006-04-01', '2013-12-31'), 'SEL_exIFRS': ('2006-04-01', '2013-01-31'), 'VAL': ('2014-01-01', '2020-12-31')}
COSTS = {'slip0.001': L.cost_rates(0.001), 'stress': L.cost_rates('stress')}

def main():
    t_start = time.time()
    Y = L.yoy_long(); S = L.Setup(Y, start='2005-01-01'); F = R.Frames(S.dates, Y)
    LEV = R.rev_level_long(); LEG = F.legs(LEV)
    # 引擎重現檢查
    b, sl = COSTS['slip0.001']
    ev, _ = S.selection(10); nl, _ = L.sim_portfolio(ev, b, sl, L.pos(S.dates[0]), L.pos(S.dates[-1]))
    pl_base = F.plan(LEG['yoy3']); T0 = pl_base[0][0]; T1 = len(R.IDX) - 1
    nb, _ = R.simulate(pl_base, b, sl, T0, T1)
    md = float((nb.reindex(nl.index) / nl - 1).abs().max()); assert md < 1e-12, md
    # R5 訊號
    dm = R.monthend_dates(Y); F5 = R.Frames(dm, Y)
    plans = {'BASE': (pl_base, {}),
             'R1a_yoy1': (F.plan(LEG['yoy1']), {}), 'R1c_yoy6': (F.plan(LEG['yoy6']), {}),
             'R1d_accel': (F.plan(LEG['accel']), {}), 'R1e_hi12': (F.plan(LEG['hi12']), {}),
             'R2_w03': (F.plan(LEG['yoy3'], w52=0.3), {}), 'R2_w07': (F.plan(LEG['yoy3'], w52=0.7), {}),
             'R3_sl15_cash': (pl_base, dict(stop=0.15, stop_mode='cash')), 'R3_sl20_cash': (pl_base, dict(stop=0.20, stop_mode='cash')),
             'R3_sl15_next': (pl_base, dict(stop=0.15, stop_mode='next')), 'R3_sl20_next': (pl_base, dict(stop=0.20, stop_mode='next')),
             'R4_bimonthly': (pl_base, dict(every=2)), 'R4_buf20': (pl_base, dict(buffer_k=20)),
             'R5_firstday': (F5.plan(F5.yoy3), {})}
    navs, info = {}, {}
    for name, (pl, kw) in plans.items():
        for ck, (bb, ss) in COSTS.items():
            nav, inf = R.simulate(pl, bb, ss, T0, T1, **kw)
            navs[f'{name}|{ck}'] = nav
            if ck == 'slip0.001':
                inf['first_exec'] = str(R.IDX[[p[0] for p in pl if p[0] >= T0][0]].date()); info[name] = inf
        print(name, info[name], round(time.time() - t_start, 1), flush=True)
    ND = pd.DataFrame(navs)
    ND.to_parquet(HERE / 'navs.parquet')
    out = {'engine_check_max_abs_diff_vs_lhlib': md, 'base_first_exec': str(R.IDX[T0].date()), 'last_day': str(R.IDX[T1].date()),
           'R5_first_signal': str(dm[0].date()), 'R5_n_signals': len(dm), 'info': info, 'variants': {}}
    def eff_start(name, a):
        fe = pd.Timestamp(info[name]['first_exec']); fb = pd.Timestamp(info['BASE']['first_exec'])
        if fe > fb: return max(pd.Timestamp(a), (fe + pd.offsets.MonthBegin(1)))
        return pd.Timestamp(a)
    for name in plans:
        v = {'info': info[name], 'segments': {}, 'paired_vs_BASE': {}, 'paired_vs_BASE_stress': {}}
        for sk, (a, bnd) in SEGS.items():
            a2 = eff_start(name, a)
            for ck in COSTS:
                st = R.seg_stats(ND[f'{name}|{ck}'], a2, bnd)
                if st is not None: st['window'] = f'{a2.date()}~{bnd}'
                v['segments'][f'{sk}|{ck}'] = st
            if name != 'BASE':
                v['paired_vs_BASE'][sk] = R.paired(ND[f'{name}|slip0.001'], ND['BASE|slip0.001'], a2, bnd)
                v['paired_vs_BASE_stress'][sk] = R.paired(ND[f'{name}|stress'], ND['BASE|stress'], a2, bnd)
                if a2 != pd.Timestamp(a): v['paired_vs_BASE'][sk]['base_same_window'] = R.seg_stats(ND['BASE|slip0.001'], a2, bnd)
        yr = R.yearly(ND[f'{name}|slip0.001']); v['yearly_pct_SEL_VAL'] = {y: x for y, x in yr.items() if y <= 2020}
        if name != 'BASE':
            p = v['paired_vs_BASE']; c1 = p['SEL']['t_paired_m'] >= 1.5; c2 = p['VAL']['diff_pct_m'] > 0
            v['verdict'] = '可用' if (c1 and c2) else ('待觀察' if (c1 or c2) else '不可用')
            v['verdict_detail'] = {'SEL_t>=1.5': bool(c1), 'VAL_diff>0': bool(c2)}
        out['variants'][name] = v
        print(name, {k: (x['CAGR'], x['maxDD'], x['t_vs_0050_m']) for k, x in v['segments'].items() if x and 'slip' in k},
              {k: (x['diff_pct_m'], x['t_paired_m']) for k, x in v['paired_vs_BASE'].items()}, v.get('verdict'), flush=True)
    e = R.E50.reindex(ND.index)
    out['0050'] = {sk: R.seg_stats(e, a, bnd) for sk, (a, bnd) in SEGS.items()}
    out['0050_yearly_pct'] = {y: x for y, x in R.yearly(e.loc[ND.index[0]:]).items() if y <= 2020}
    res['RUN'] = out
    (HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print('done', round(time.time() - t_start, 1))

if __name__ == '__main__':
    main()
