#!/usr/bin/env python3
"""覆蓋率：(a) 下市股價格修補 (b) B10 情境實際出清事件的對價覆蓋 (c) 終止上市名單計數。"""
import sys, json
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_infra_v3_rebuild/replicate_o3')
import r3_lib as L, r3_bounds_lib as B
sys.path.pop(0)
import numpy as np, pandas as pd
K, V = L.K, L.V
out = {}
off = L.OFF
w = off[(off['date'] >= '2019-05-17') & (off['date'] <= '2023-12-31')]
out['official_2019-05-17_2023_rows'] = len(w); out['four_digit'] = int((w['code'].str.len() == 4).sum())
out['four_in_panel'] = int(((w['code'].str.len() == 4) & w['code'].isin(K.COLS)).sum())
codes4 = sorted(w[w['code'].str.len() == 4]['code'])
dp = pd.read_parquet(L.O3 / 'delist_prices.parquet')
out['patched_codes'] = dp['code'].nunique(); out['patched_rows'] = len(dp); out['codes_match'] = sorted(dp['code'].unique()) == codes4
# 應有股票日：面板最後一筆之後到實際最後交易日（merger_consideration.csv 的 last_trade；缺則用 delist_prices 最後一筆）
C0 = pd.read_parquet('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/close_raw.parquet')
mc = L.MC.set_index('code')
need = 0; per = {}
for c in codes4:
    pl = C0[c].last_valid_index()
    lt = mc.at[c, 'last_trade'] if c in mc.index else dp[dp.code == c]['date'].max()
    nd = int(((C0.index > pl) & (C0.index <= lt)).sum()); got = int((dp.code == c).sum())
    need += nd; per[c] = (str(pl.date()), str(pd.Timestamp(lt).date()), nd, got)
out['stockdays_need'] = need; out['stockdays_got'] = len(dp); out['rate'] = len(dp) / need
out['full_codes'] = sum(1 for v in per.values() if v[3] >= v[2]); out['per'] = per
print({k: v for k, v in out.items() if k != 'per'})
# B10 情境實際出清事件
L.apply_patch()
YL, Y5 = L.load_rev(1); xr = L.XRules('main'); b, s = K.rates()
ev = {}
for D in ('hi',):
    for U in ('EXFIN', 'ALL'):
        sb = B.SB(YL, Y5, U == 'EXFIN', D, 1.0); t0, t1, sd = sb.bounds()
        for n in (10, 30):
            nav, lg = L.simulate(sb.plan(n), b, s, t0, t1, xr)
            for c in lg['conv']: ev.setdefault(c[1], []).append((D, U, n, c[0]))
cs = sorted(ev)
def usable(c):
    if c not in mc.index: return False
    t = mc.at[c, 'type']
    return t in ('swap', 'failure') or (t == 'cash' and np.isfinite(mc.at[c, 'cash']))
out['B10_exit_codes'] = cs
out['B10_exit_n'] = len(cs)
out['B10_exit_type_known'] = sum(c in mc.index and mc.at[c, 'type'] != 'unknown' for c in cs)
out['B10_exit_usable'] = sum(usable(c) for c in cs)
out['B10_exit_news'] = sum(c in mc.index and mc.at[c, 'verified'] == 'news' for c in cs)
print({k: out[k] for k in ('B10_exit_n', 'B10_exit_type_known', 'B10_exit_usable', 'B10_exit_news')}, cs)
(L.HERE / 'my_cov.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
