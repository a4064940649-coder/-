#!/usr/bin/env python3
"""S3d（凍結後，描述）：「強觸發」（純價格條件 r6>門檻且相對大盤>20pp，連續 3 個交易日）卻不在 universe 的事件數，逐年；
2021-09～2025 有 disposal_5y 獨立名單交叉驗證（100%／98.4%），用來校準「面板強觸發但無處置」在完整年份有多常見。
python3 -I -B s7b_strong_unmatched.py → s7b_strong_unmatched.json"""
import json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
adj = pd.read_parquet(P / 'adj.parquet'); cr = pd.read_parquet(P / 'close_raw.parquet')
keep = cr.notna().sum(axis=1) >= 300; adj = adj[keep]; cols = [c for c in adj.columns if len(c) == 4 and not c.startswith('0')]; adj = adj[cols]; IDX = adj.index
TX = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].reindex(IDX).ffill()
r6 = adj / adj.shift(6) - 1; m6 = TX / TX.shift(6) - 1
th = np.where(IDX >= pd.Timestamp('2015-06-01'), 0.32, 0.25)[:, None]
up = (r6.to_numpy() > th) & (r6.sub(m6, axis=0).to_numpy() > 0.2)
F = pd.DataFrame(up, index=IDX, columns=cols).astype(int)
trig = F.rolling(3).sum() == 3
prev = trig.astype(int).shift(1).rolling(15, min_periods=1).sum().fillna(0) > 0
new = trig & ~prev
U = pd.read_csv(HERE / 'data/universe_announce.csv', dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
D5 = pd.read_csv(HERE / 'raw/screener_extras/disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
pos = {}
for c, a, e in zip(U.code, U.announce, U.end): pos.setdefault(c, []).append((IDX.searchsorted(a), IDX.searchsorted(e, 'right') - 1))
pos5 = {}
for c, a, e in zip(D5.code, D5.date, D5.end): pos5.setdefault(c, []).append((IDX.searchsorted(a), IDX.searchsorted(e, 'right') - 1))
rows = []
ii, jj = np.where(new.to_numpy())
for i, j in zip(ii, jj):
    c = cols[j]
    hit = any(a - 5 <= i <= b + 3 for a, b in pos.get(c, []))
    hit5 = any(a - 5 <= i <= b + 3 for a, b in pos5.get(c, []))
    rows.append(dict(code=c, trig=str(IDX[i].date()), year=IDX[i].year, hit=hit, hit_d5=hit5, r6=round(float(r6.iat[i, j]), 3)))
E = pd.DataFrame(rows)
by = {int(y): dict(n_strong=int(len(g)), n_unmatched=int((~g.hit).sum()), share_unmatched=round(float((~g.hit).mean()), 3),
                   n_unmatched_also_not_in_disposal5y=int((~g.hit & ~g.hit_d5).sum()) if y >= 2022 else None) for y, g in E.groupby('year')}
out = dict(by_year=by, unmatched_2019=E[(E.year == 2019) & ~E.hit].to_dict('records'), unmatched_2020_12=E[(E.trig >= '2020-12-01') & (E.trig <= '2020-12-31') & ~E.hit].to_dict('records'),
           note='強觸發＝純價格條件連續 3 日（不含量能條件）；比對窗＝[公告−5, 結束+3] 交易日。2022～2025 的未比對事件同時不在 disposal_5y → 代表「面板強觸發但官方未處置」在完整名單年份的基準率。')
(HERE / 's7b_strong_unmatched.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
for y, v in by.items(): print(y, v)
print(out['unmatched_2019']); print(out['unmatched_2020_12'])
