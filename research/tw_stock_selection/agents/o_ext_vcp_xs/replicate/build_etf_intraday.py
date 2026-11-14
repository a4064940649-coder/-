# -*- coding: utf-8 -*-
"""0050 當日開盤→收盤報酬（審查 A 的口徑修正用）。來源：scratchpad/data_ext/tsec_0050.csv（～2019-05-16）＋ jb7k 0050.csv（2015 起，需先 curl 到 scratchpad/rep_vcp/jb/0050.csv）。
輸出 scratchpad/rep_vcp/etf_intraday.csv。python3 -I -B build_etf_intraday.py"""
import pandas as pd
S = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/'
def roc(s):
    y, m, d = s.split('/'); return pd.Timestamp(int(y) + 1911, int(m), int(d))
t = pd.read_csv(S + 'data_ext/tsec_0050.csv', header=None, names=['d', 'vol', 'val', 'o', 'h', 'l', 'c', 'chg', 'n'], dtype=str)
t['d'] = [roc(x) for x in t.d]
for c in ('o', 'c'): t[c] = pd.to_numeric(t[c].str.replace(',', ''), errors='coerce')
a = t.set_index('d')[['o', 'c']].dropna()
j = pd.read_csv(S + 'rep_vcp/jb/0050.csv', parse_dates=['date']).set_index('date')[['open', 'close']].dropna(); j.columns = ['o', 'c']
x = pd.concat([a[a.index < j.index[0]], j])
(x.c / x.o - 1).to_frame('intr').to_csv(S + 'rep_vcp/etf_intraday.csv')
