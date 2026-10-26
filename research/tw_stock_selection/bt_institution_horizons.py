#!/usr/bin/env python3
"""法人的優勢在哪個時間尺度？（上市股，2021-10～2026-10，60 日均成交值 ≥ NT$1,000 萬）
每天依「當日買超股數 ÷ 當日成交量」把股票分五組（外資／投信／自營商分別做），比較前 1/5 與後 1/5 在各區間的平均報酬差：
  D0   當天（前一日收盤→當日收盤）：法人自己交易的那天，散戶不知道
  ON   當日收盤→隔日開盤（隔夜；資料盤後公布，最早的可成交點是隔日開盤）
  D1   隔日開盤→隔日收盤
  D2_5 第 2～5 日；D6_20 第 6～20 日
t 值以每日價差的時間序列計算（Newey-West 未調整，僅供比較）。輸出 output/bt_institution_horizons.json
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
S = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad'); P = S / 'panel'
X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras'); OUT = Path(__file__).resolve().parent / 'output'
adj = pd.read_parquet(P / 'adj.parquet'); C = pd.read_parquet(P / 'close_raw.parquet'); O = pd.read_parquet(P / 'open.parquet'); V = pd.read_parquet(P / 'value.parquet'); R1 = pd.read_parquet(P / 'ret.parquet')
inst = pd.read_csv(X / 'inst_5y.csv.gz', dtype={'code': str}, parse_dates=['date']); inst = inst[inst['code'].isin(adj.columns)]
idx = adj.index[(adj.index >= inst['date'].min()) & (adj.index <= inst['date'].max())]
liq = (V.rolling(60).mean() >= 1e7).reindex(idx)
vol = (V / C).reindex(idx)
A = adj.reindex(adj.index)
# 區間報酬（用還原指數；隔夜用開盤／前收，除權息日剔除）
cl = adj; op = adj * (O / C)                      # 還原開盤
r_D0 = R1.reindex(idx)
r_ON = (op.shift(-1) / cl - 1).reindex(idx)
r_D1 = (cl.shift(-1) / op.shift(-1) - 1).reindex(idx)
r_D2_5 = (cl.shift(-5) / cl.shift(-1) - 1).reindex(idx)
r_D6_20 = (cl.shift(-20) / cl.shift(-5) - 1).reindex(idx)
horiz = {'D0_same_day': r_D0, 'ON_overnight': r_ON, 'D1_next_day_intraday': r_D1, 'D2_5': r_D2_5, 'D6_20': r_D6_20}
res = {}
for who in ['foreign', 'trust', 'dealer']:
    M = inst.pivot_table(index='date', columns='code', values=who, aggfunc='sum').reindex(idx).reindex(columns=adj.columns) * 1000
    sig = (M / vol).where(liq & vol.gt(0))
    rk = sig.rank(axis=1, pct=True)
    q5, q1 = rk >= 0.8, rk <= 0.2
    out = {}
    for hn, r in horiz.items():
        r = r.where(r.abs() < 0.5)
        sp = (r.where(q5).mean(axis=1) - r.where(q1).mean(axis=1)).dropna()
        out[hn] = {'Q5_minus_Q1_bp': round(float(sp.mean() * 1e4), 1), 't': round(float(sp.mean() / (sp.std() / np.sqrt(len(sp)))), 1)}
    # 跟單策略：每天隔日開盤買前 1/5、持有 1 日／5 日的扣成本淨值（來回 0.5%；當沖 0.23% 不適用於隔日）
    res[who] = out
(OUT / 'bt_institution_horizons.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
for k, v in res.items(): print(k, v)
