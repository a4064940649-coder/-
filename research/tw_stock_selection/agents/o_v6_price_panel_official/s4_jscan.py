#!/usr/bin/env python3
"""換股日分布 j＝0～12（PREREG 後一次跑完）：沿用 o_stagger_tranche/st_core 的 j 定義（錨定月 m 的法定期限 D_m＝m 月 10 日起第一個
加權指數交易日，2019-02／2023-04 特例；J{j}＝D_m 之後第 j 個交易日；不在 GOOD 日或次一 GOOD 日不存在則該月跳過）與 st_core.unavail_for 的 AB
（seed 0～39）。欄位：V6、V6_B10（不做 AB）、AB_V6、AB_V6_B10（40 seeds 平均）；成本一般與壓力；另把 s4_run 的可執行拖累（o5 移植公式）
同法加到每個 j 淨值上。錨定月＝V6 正式訊號日的月份（含 2024-01）。輸出 RESULTS_JSCAN、navs_jscan_v6.parquet。"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from multiprocessing import Pool
import v6common as C
C.check_prereg()
import v6env, v6stats as ST
L = v6env.load('v6'); K, V = L.K, L.V
XR = L.setup_px()
YFX, Y5F = L.load_rfix(); DMT = L.disp_mask_fn('trade'); GAP = L.gap_dict()
sys.path.insert(0, str(C.AG / 'o_stagger_tranche'))
import st_core as STC          # noqa: E402（只讀 import；在 P 補價之後 import）
sys.path.pop(0)
OFF = L.V4(YFX, Y5F, DMT)
ANCH = [pd.Period(d, 'M') for d in OFF.sig_days]

class JV6(L.V4):
    def set_days(self, days, end_day):
        self.anchor = {d: m for m, d in days}; self.sig_days = [d for m, d in days]; self.end_day = end_day; self._cs = {}
    def y3(self, d, back=0):
        m = self.anchor[d] - back
        Y = self.Y5 if d >= V.SWITCH else self.YL
        y = K.yoy3_for_month(Y, m)
        if self.exfin: y = np.where(V.FINMASK, np.nan, y)
        return y

def days_for(j):
    out, skip = [], []
    for m in ANCH:
        d = STC.day_for(m, j)
        if d is None: skip.append((str(m), 'no_day')); continue
        t = K.IDX.get_loc(d)
        if not K.GOOD[t + 1]: skip.append((str(m), 'next_not_good')); continue
        out.append((m, d))
    de = STC.day_for(pd.Period('2026-10', 'M'), j)
    if de is None or de > K.IDX[-1]: de = K.IDX[-1]
    out = [(m, d) for m, d in out if d < de]
    return out, de, skip

SPEC = {'V6': dict(), 'V6_B10': dict(gap=GAP, pfix=1.0), 'AB_V6': dict(), 'AB_V6_B10': dict(gap=GAP, pfix=1.0)}
def job(a):
    spec, j, seed = a
    days, de, skip = days_for(j)
    kw = SPEC[spec]
    if spec.startswith('AB'):
        S = JV6(YFX, Y5F, DMT, unavail=(lambda d, _s=seed: STC.unavail_for(S.anchor[d], d, _s)), **kw)
    else:
        S = JV6(YFX, Y5F, DMT, **kw)
    S.set_days(days, de)
    pl = S.plan(10, 'official')
    t0, t1 = K.IDX.get_loc(S.sig_days[0]), K.IDX.get_loc(de)
    r = {c: L.sim_plan(pl, t0, t1, c, XR)[0].pct_change() for c in ('normal', 'stress')}
    return spec, j, seed, r, [p[3] for p in pl], de, skip

if __name__ == '__main__':
    T0 = time.time()
    jobs = [(s, j, 0) for s in ('V6', 'V6_B10') for j in range(13)] + [(s, j, sd) for s in ('AB_V6', 'AB_V6_B10') for j in range(13) for sd in range(40)]
    with Pool(4) as P: out = P.map(job, jobs, chunksize=2)
    by = {}
    for spec, j, s, r, sdays, de, skip in out: by.setdefault((spec, j), []).append((s, r, sdays, de, skip))
    CAP = C.AG / 'o_exec_capacity_tiers'
    EX = pd.read_parquet(CAP / 'navs_abavg.parquet'); BEN = pd.read_parquet(CAP / 'data' / 'bench.parquet')
    rzi = BEN['AB_V4|normal'].pct_change()
    TIERS = [f'main|{t}|{m}|A' for t in ('30萬', '100萬', '300萬', '1000萬') for m in ('FIX', 'GROW')]
    E50 = K.E50
    res = {}; navs = {}
    for spec in SPEC:
        for c in ('normal', 'stress'):
            rows = {}
            for j in range(13):
                lst = sorted(by[(spec, j)], key=lambda x: x[0])
                sdays, de, skip = lst[0][2], lst[0][3], lst[0][4]
                assert all(x[2] == sdays for x in lst)
                nav = ST.mean_nav([x[1][c] for x in lst]); sdx = list(sdays) + [de]
                x = ST.xs(nav, E50, sdx)
                o = {'t': ST.tstat(x), 'n_windows': int(len(x)), 'CAGR': ST.cagr_mdd(nav, ST.A0, min(de, pd.Timestamp(ST.B0)))[0], 'n_skip': len(skip), 'end_day': str(de.date())}
                if len(lst) > 1:
                    st = [ST.tstat(ST.xs((1 + y[1][c].fillna(0)).cumprod(), E50, sdx)) for y in lst]
                    o.update(seed_t_median=float(np.median(st)), seed_t_min=float(min(st)), seed_t_max=float(max(st)))
                xh = ST.xs(nav, E50, sdx, ST.HOLD_CAL_END, ST.B0, attr='end'); o['t_HOLD_cal'] = ST.tstat(xh)
                if c == 'normal' and spec.startswith('AB'):
                    for col in TIERS:
                        drag = (EX[col].pct_change() - rzi).reindex(nav.index).fillna(0.0)
                        ne = (1 + nav.pct_change().fillna(0) + drag).cumprod()
                        o[f'exec|{col}'] = {'t': ST.tstat(ST.xs(ne, E50, sdx)), 'CAGR': ST.cagr_mdd(ne, ST.A0, min(de, pd.Timestamp(ST.B0)))[0]}
                rows[j] = o; navs[f'{spec}|{c}|J{j}'] = nav
            ts = np.array([rows[j]['t'] for j in range(13)]); cg = np.array([rows[j]['CAGR'] for j in range(13)])
            summ = {'t_median': float(np.median(ts)), 't_p10': float(np.percentile(ts, 10)), 't_min': float(ts.min()), 't_max': float(ts.max()),
                    'CAGR_median': float(np.median(cg)), 'CAGR_p10': float(np.percentile(cg, 10)), 'n_ge_2.30': int((ts >= 2.30).sum()), 'n_ge_2.0': int((ts >= 2.0).sum()),
                    'by_j': rows}
            if c == 'normal' and spec.startswith('AB'):
                summ['exec_j'] = {col: {'t_median': float(np.median([rows[j][f'exec|{col}']['t'] for j in range(13)])),
                                        't_p10': float(np.percentile([rows[j][f'exec|{col}']['t'] for j in range(13)], 10)),
                                        'CAGR_median': float(np.median([rows[j][f'exec|{col}']['CAGR'] for j in range(13)])),
                                        'CAGR_p10': float(np.percentile([rows[j][f'exec|{col}']['CAGR'] for j in range(13)], 10))} for col in TIERS}
            res[f'{spec}|{c}'] = summ
    res['runtime_s'] = time.time() - T0
    C.save_key('RESULTS_JSCAN', res)
    pd.DataFrame(navs).to_parquet(C.HERE / 'navs_jscan_v6.parquet')
    print({k: (round(v['t_median'], 3), round(v['t_p10'], 3)) for k, v in res.items() if isinstance(v, dict)}, flush=True)
