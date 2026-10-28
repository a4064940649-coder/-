#!/usr/bin/env python3
"""換股日分布（樣本外，PREREG 後一次跑完）：o_stagger_tranche/st_core.py 的 j 定義（錨定月 m 的法定期限 D_m＝m 月 10 日起第一個加權指數交易日，
2019-02／2023-04 特例；J{j}＝D_m 之後第 j 個交易日；不在面板或有值檔數 < 300 則該月跳過）、st_core.unavail_for 的 AB（seed 0～39）、
V5 資料（首次值替換）與 V4 規則。j＝0～12 共 13 版，每版 AB 40 seed 平均淨值，報對 0050 t（全部窗）。
成交日＝訊號日次一面板交易日；若次一面板日不是有值檔數 ≥ 300 的日子，該月跳過（記錄次數）。"""
import sys, json, hashlib, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from multiprocessing import Pool
import v5lib as L
K, V, L4 = L.K, L.V, L.L4
res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert 'PREREG' in res and hashlib.sha256(json.dumps(res['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == res['PREREG_sha256'], 'PREREG 不存在或被改過'
assert L.sha(L.FREEZE5) == res['PREREG']['data_freeze_v5_sha256']
L.check_freeze5()
g = L.setup()
sys.path.insert(0, str(L.AG / 'o_stagger_tranche'))
import st_core as ST          # noqa: E402（只讀 import；在 P 補價之後 import，GOODSET 用補價後的 GOOD）
sys.path.pop(0)
OFF = L.make(fv=L.FVAdj())
ANCH = [pd.Period(d, 'M') for d in OFF.sig_days]

class JV5(L.V5):
    def set_days(self, days, end_day):
        self.anchor = {d: m for m, d in days}; self.sig_days = [d for m, d in days]; self.end_day = end_day; self._cs = {}
    def y3(self, d, back=0):
        m = self.anchor[d] - back
        Y, panel = (self.Y5, '5') if d >= V.SWITCH else (self.YL, 'L')
        y = K.yoy3_for_month(Y, m)
        if self.exfin: y = np.where(V.FINMASK, np.nan, y)
        if self.fv is None: return y
        return self.fv.adjust_y3(Y, m, d, panel, y, V.FINMASK if self.exfin else np.zeros(L.N, bool))

def days_for(j):
    out, skip = [], []
    for m in ANCH:
        d = ST.day_for(m, j)
        if d is None: skip.append((str(m), 'no_day')); continue
        t = K.IDX.get_loc(d)
        if not K.GOOD[t + 1]: skip.append((str(m), 'next_not_good')); continue
        out.append((m, d))
    de = ST.day_for(pd.Period('2026-10', 'M'), j)
    if de is None or de > K.IDX[-1]: de = K.IDX[-1]
    out = [(m, d) for m, d in out if d < de]
    return out, de, skip

def job(a):
    j, seed = a
    days, de, skip = days_for(j)
    S = JV5(g['YL'], g['Y5'], g['DM'], fv=L.FVAdj(), unavail=(lambda d, _s=seed: ST.unavail_for(S.anchor[d], d, _s)))
    S.set_days(days, de)
    pl = S.plan(10, 'official')
    t0, t1 = K.IDX.get_loc(S.sig_days[0]), K.IDX.get_loc(de)
    nav = L.sim(pl, 'normal', t0, t1)
    return j, seed, nav.pct_change(), [p[3] for p in pl], de
if __name__ == '__main__':
    T0 = time.time()
    with Pool(4) as P: out = P.map(job, [(j, s) for j in range(13) for s in range(40)], chunksize=2)
    by = {}
    for j, s, r, sd, de in out: by.setdefault(j, []).append((s, r, sd, de))
    rows = {}; navs = {}
    for j in range(13):
        lst = sorted(by[j], key=lambda x: x[0])
        nav = L.mean_nav([x[1] for x in lst]); sd = lst[0][2]; de = lst[0][3]
        assert all(x[2] == sd for x in lst)
        x = L4.excess(nav, K.E50, sd + [de], L.A0, L.B0)
        seed_t = [L.tstat(L4.excess((1 + y[1].fillna(0)).cumprod(), K.E50, sd + [de], L.A0, L.B0)) for y in lst]
        days, de2, skip = days_for(j)
        rows[j] = {'t': L.tstat(x), 'n_windows': int(len(x)), 'mean_pp': float(x.mean() * 100), 'CAGR': L.R.cagr_mdd(nav, L.A0, min(de, pd.Timestamp(L.B0)))[0],
                   'seed_t_median': float(np.median(seed_t)), 'seed_t_min': float(min(seed_t)), 'seed_t_max': float(max(seed_t)), 'n_skip': len(skip), 'skip': skip, 'end_day': str(de.date())}
        navs[f'J{j}'] = nav
    ts = np.array([rows[j]['t'] for j in range(13)])
    summ = {'t_median': float(np.median(ts)), 't_p10': float(np.percentile(ts, 10)), 't_min': float(ts.min()), 't_max': float(ts.max()),
            'n_ge_2.30': int((ts >= 2.30).sum()), 'n_ge_2.0': int((ts >= 2.0).sum()), 'by_j': rows, 'runtime_s': time.time() - T0}
    res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
    res['RESULTS_JSCAN'] = summ
    (HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)), encoding='utf-8')
    pd.DataFrame(navs).to_parquet(HERE / 'navs_jscan_v5.parquet')
    print(json.dumps({k: v for k, v in summ.items() if k != 'by_j'}, indent=1), {j: round(rows[j]['t'], 3) for j in rows})
