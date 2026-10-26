#!/usr/bin/env python3
"""52 週高點動能：樣本外檢定（2019-05～2026-10，上市股）＋ 樣本內重跑（2005～2019-04，同一套上市股面板）＋ 存活者偏誤估計。

規則與第一輪完全相同（事前固定，不調參）：
  訊號 = 還原收盤 / 過去 252 個交易日還原收盤最高值；每月最後交易日收盤換股，持有一個月。
  可投資：60 日平均成交值 ≥ NT$1,000 萬、收盤價 ≥ 10 元、過去 252 日每天有成交。
  A. 全體五分位（等權）：前 1/5 vs 後 1/5、vs 全體等權
  B. 實務版：流動性前 300 名中買訊號最高 30 檔，等權；成本 0.5% / 1% 來回
  C. 流動性前 500 名中買 50 檔
  D. 實務版 + 剔除 60 日波動較高的一半
  E. 實務版但延後一天執行（訊號用 t−1 收盤，t 收盤成交）
存活者偏誤：2010-01～2019-04，「全部上市股（含已下市）」vs「只用 k66 現存股」重跑 A、B。
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from btlib import month_ends, period_returns, bench_returns, run_sort, summarize

P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
OUT = Path(__file__).resolve().parent / 'output'; OUT.mkdir(exist_ok=True)
adj = pd.read_parquet(P / 'adj.parquet'); C = pd.read_parquet(P / 'close_raw.parquet'); V = pd.read_parquet(P / 'value.parquet'); R1 = pd.read_parquet(P / 'ret.parquet')
k66_codes = {p.stem for p in (P.parent / 'k66').glob('*.csv')}
me = month_ends(adj.index)
R = period_returns(adj, me)
S52 = (adj / adj.rolling(252).max())
V60 = V.rolling(60).mean()
VOL60 = R1.rolling(60).std()
e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj']
B = bench_returns(e50, me)

def at(M, lag=0):
    return (M.shift(lag) if lag else M).reindex(me)

def elig_base(lag=0):
    return (at(V60, lag) >= 1e7) & (at(C, lag) >= 10) & at(S52, lag).notna()

def top_liq(elig, n, lag=0):
    v = at(V60, lag).where(elig)
    rk = v.rank(axis=1, ascending=False)
    return elig & (rk <= n)

res, raw = {}, {}
sig0, sig1 = at(S52), at(S52, 1)
E0 = elig_base(); E1 = elig_base(1)
variants = {
    'A_quintiles_all': (sig0, E0, None),
    'B_top30_of_top300': (sig0, top_liq(E0, 300), 30),
    'C_top50_of_top500': (sig0, top_liq(E0, 500), 50),
    'E_top30_of_top300_lag1': (sig1, top_liq(E1, 300, 1), 30),
}
lowvol = at(VOL60).le(at(VOL60).where(top_liq(E0, 300)).median(axis=1), axis=0)
variants['D_top30_of_top300_lowvolhalf'] = (sig0, top_liq(E0, 300) & lowvol, 30)
periods = {'IS_2005_2019-04': ('2005-01-01', '2019-04-30'), 'OOS_2019-05_2026-10': ('2019-05-01', '2026-10-31'), 'FULL_2005_2026': ('2005-01-01', '2026-10-31')}
for name, (sig, el, top_n) in variants.items():
    rr = run_sort(sig, R, el, nq=5, top_n=top_n)
    raw[name] = rr
    for pname, (a, b) in periods.items():
        res[f'{name}|{pname}'] = summarize(rr, me, B, label=name, start=a, end=b)
# 存活者偏誤：2010-01～2019-04
surv = pd.DataFrame(False, index=E0.index, columns=E0.columns); surv[[c for c in E0.columns if c in k66_codes]] = True
for name, el, top_n in [('A_quintiles_all', E0, None), ('B_top30_of_top300', top_liq(E0, 300), 30)]:
    full = run_sort(sig0, R, el, top_n=top_n)
    elS = (E0 & surv) if top_n is None else top_liq(E0 & surv, 300)
    so = run_sort(sig0, R, elS, top_n=top_n)
    res[f'SURV_{name}|full_2010_2019-04'] = summarize(full, me, B, label=name + ' all listed incl. delisted', start='2010-01-01', end='2019-04-30')
    res[f'SURV_{name}|survivors_only_2010_2019-04'] = summarize(so, me, B, label=name + ' survivors only (k66)', start='2010-01-01', end='2019-04-30')
(OUT / 'bt_52w_oos.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
def line(k, v):
    g = v.get('best_gross', {}); n = v.get('best_net', {}); n1 = v.get('best_net_1pct', {}); u = v.get('universe', {}); b = v.get('bench', {})
    return (f"{k:58s} n={v['n_names']:5.1f} to={v['turnover']:.2f} | gross {g.get('CAGR')} net.5 {n.get('CAGR')} net1 {n1.get('CAGR')} MDD {n.get('maxDD')} | "
            f"EW {u.get('CAGR')} 0050 {b.get('CAGR')} | Q5-Q1 {v['Q5_minus_Q1_pct_per_period']}%/m t={v['t_Q5_Q1']} | net-EW t={v['t_net_vs_universe']} | net-0050 {v.get('net_minus_bench_pct_per_period')}%/m t={v.get('t_net_vs_bench')} | yrs {v.get('years_beat_bench')}")
for k, v in res.items(): print(line(k, v))
k = 'B_top30_of_top300|OOS_2019-05_2026-10'; print('OOS yearly net', res[k]['yearly_net_pct']); print('OOS yearly 0050', res[k]['yearly_bench_pct']); print('OOS yearly EW', res[k]['yearly_universe_pct'])
