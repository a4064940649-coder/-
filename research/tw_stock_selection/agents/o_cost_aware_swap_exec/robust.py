#!/usr/bin/env python3
"""凍結後穩健性補充（只描述，不選擇；先斷言三份 PREREG 與 results.json['OOS'] 已存在）。寫入 results.json['ROBUST']。
描述代表 NR 與主策略（BASE）的差，拆到個股與產業：REAL_opt、seed 0、13 j、100 萬一般（單行程）。
作法：不改引擎，只在 CASim 子類別把 _eqm 換成 property，每天收盤記錄持股市值快照（S×CF）與淨值；
  個股日貢獻＝前一日權重 × 個股含息日報酬（ADJ[t+1]×CF[t+1]／CF[t]−1）；差＝NR−BASE 的貢獻差，逐股、逐產業、逐年加總（算術和，pp）。
  這是權重歸因（不含成交時點與成本差），另報「貢獻差合計」對「淨值對數報酬差」的對帳殘差。
產業：stock_list.csv 的現行產業別（非 PIT，只作描述分組；查不到者＝「已下市或無資料」）。
python3 -I -B robust.py"""
import sys, json, math, pickle
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import cas_lib as CL; import run_lib as RL; sys.path.pop(0)
RES = HERE / 'results.json'
R = json.loads(RES.read_text(encoding='utf-8'))
for k in ('PREREG', 'PREREG_POWER', 'PREREG_FREEZE'): assert CL.canon(R[k]) == R[k + '_sha256']
assert 'OOS' in R
CH = R['PREREG_FREEZE']['chosen']; CHO = R['PREREG_FREEZE']['chosen_opt']
E = CL.engine(); D = E['D']
P = pickle.load(open(RL.PCACHE, 'rb'))

class Snap(E['CASim']):
    @property
    def _eqm(self): return self.__dict__.get('_eqm_v')
    @_eqm.setter
    def _eqm(self, v):
        self.__dict__['_eqm_v'] = v
        if v is None or not hasattr(self, 'S'): return
        t = self.__dict__.get('_tc', self.t0)
        H = np.flatnonzero(self.S > 0)
        self.__dict__.setdefault('_snaps', []).append((t, H.copy(), (self.S[H] * D.CF[t, H]).copy(), float(v)))
        self.__dict__['_tc'] = t + 1

def contrib(v, j, s, opt):
    pl, cands = P[(v, j, s)]
    cands = {e: ([int(x) for x in c], gu) for e, (c, gu) in cands.items()}
    sim = Snap(pl, dict(CL.cfg_for(E, '100萬', 'normal')), settle=E['ST'], cands=cands, opt=opt, rng_seed=1000 * s + j)
    nav = sim.run()
    T = len(D.IDX); C = np.zeros((2027 - 2006, D.N))
    for t, H, val, eq in sim._snaps:
        if t + 1 >= T or len(H) == 0: continue
        with np.errstate(invalid='ignore', divide='ignore'):
            r = D.ADJ[t + 1, H] * D.CF[t + 1, H] / D.CF[t, H] - 1.0
        r = np.where(np.isfinite(r), r, 0.0)
        C[D.IDX[t + 1].year - 2006, H] += val / eq * r
    return nav, C

SL = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype=str).set_index('code')
ind = np.array([SL.industry.get(str(c), '已下市或無資料') for c in D.COLS], dtype=object)
ind = np.array([x if isinstance(x, str) else '已下市或無資料' for x in ind], dtype=object)
dC = []; resid = []; navd = []
for j in range(13):
    n0, c0 = contrib('REAL_opt', j, 0, {})
    n1, c1 = contrib('REAL_opt', j, 0, CHO)
    dC.append(c1 - c0)
    yrs = (n0.index[-1] - n0.index[0]).days / 365.25
    ld = (math.log(n1.iloc[-1]) - math.log(n0.iloc[-1])) * 100 / yrs
    navd.append(CL.cagr(n1, n0.index[0], n0.index[-1]) - CL.cagr(n0, n0.index[0], n0.index[-1]))
    resid.append(float((c1 - c0).sum() * 100 / yrs - ld))
    print('j', j, round(navd[-1], 3), round(float((c1 - c0).sum() * 100 / yrs), 3), flush=True)
dC = np.array(dC)                      # 13 × 年 × 股
YRS = 20.5
tot_j = dC.sum(axis=(1, 2)) * 100 / YRS                                  # 每 j 的年化算術貢獻差（pp／年）
by_stock = dC.sum(axis=1).mean(axis=0) * 100 / YRS                       # 13 j 平均
order = np.argsort(-np.abs(by_stock))
top = [dict(code=str(D.COLS[i]), industry=str(ind[i]), pp_yr=float(by_stock[i])) for i in order[:10]]
top_pos = int(np.argmax(by_stock))
by_ind = pd.Series(by_stock).groupby(ind).sum().sort_values()
by_year = dC.sum(axis=2).mean(axis=0) * 100                              # 逐年（pp，13 j 平均）
# 逐 j 拿掉該 j 最大正貢獻股
drop1 = [float((dC[k].sum(axis=0).sum() - dC[k].sum(axis=0).max()) * 100 / YRS) for k in range(13)]
# 2019 起
y19 = dC[:, 2019 - 2006:, :]
tot19 = y19.sum(axis=(1, 2)) * 100 / 7.8
bs19 = y19.sum(axis=1).mean(axis=0) * 100 / 7.8
out = dict(written_utc=pd.Timestamp.now('UTC').isoformat(), design=__doc__, chosen=CH, variant='REAL_opt', seed=0, tier='100萬', cost='normal',
           nav_cagr_diff_by_j=navd, nav_cagr_diff_med=float(np.median(navd)),
           weight_contrib_diff_pp_yr=dict(by_j=tot_j.tolist(), med=float(np.median(tot_j)), mean=float(tot_j.mean())),
           recon_resid_pp_yr=dict(by_j=resid, mean=float(np.mean(resid)), note='貢獻差合計 − 淨值對數報酬差（年化）；殘差＝成本差＋成交時點＋現金部位'),
           n_stocks_nonzero=int((np.abs(by_stock) > 1e-9).sum()),
           top10_abs_stocks=top,
           top_positive_stock=dict(code=str(D.COLS[top_pos]), industry=str(ind[top_pos]), pp_yr=float(by_stock[top_pos])),
           drop_top_positive_stock=dict(mean_all=float(by_stock.sum()), mean_drop1=float(by_stock.sum() - by_stock[top_pos]),
                                        by_j_own_top_dropped_med=float(np.median(drop1)), by_j_own_top_dropped_min=float(np.min(drop1))),
           share_pos_stocks=float((by_stock > 1e-9).sum() / max(1, (np.abs(by_stock) > 1e-9).sum())),
           by_industry_pp_yr={str(k): float(v) for k, v in by_ind.items() if abs(v) > 1e-6},
           by_year_pp={str(2006 + i): float(x) for i, x in enumerate(by_year)},
           y19on=dict(med=float(np.median(tot19)), mean=float(tot19.mean()), top5=[dict(code=str(D.COLS[i]), industry=str(ind[i]), pp_yr=float(bs19[i])) for i in np.argsort(-np.abs(bs19))[:5]],
                      drop_top_positive=float(bs19.sum() - bs19.max())))
r = json.loads(RES.read_text(encoding='utf-8')); r['ROBUST'] = out
RES.write_text(CL.dumps(r), encoding='utf-8')
print(CL.dumps({k: v for k, v in out.items() if k != 'design'})[:5000])
