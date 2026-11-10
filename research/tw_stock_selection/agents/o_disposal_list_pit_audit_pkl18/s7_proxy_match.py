#!/usr/bin/env python3
"""S3c（凍結後）：面板近似觸發事件（凍結模型的 K_reg 事件定義）與 universe 的逐事件比對率（逐年），並列出 2019 及殘差 < −2σ 年份的未比對事件（S3 候選）。
python3 -I -B s7_proxy_match.py → s7_proxy_match.json、data/proxy_unmatched_candidates.csv"""
import json, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8')); assert 'PREREG' in R and 'RESULTS_MODEL' in R
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
adj = pd.read_parquet(P / 'adj.parquet'); cr = pd.read_parquet(P / 'close_raw.parquet'); val = pd.read_parquet(P / 'value.parquet')
keep = cr.notna().sum(axis=1) >= 300; adj, cr, val = adj[keep], cr[keep], val[keep]
cols = [c for c in adj.columns if len(c) == 4 and not c.startswith('0')]; adj, cr, val = adj[cols], cr[cols], val[cols]
IDX = adj.index
TX = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].reindex(IDX).ffill()
r6 = adj / adj.shift(6) - 1; m6 = TX / TX.shift(6) - 1; vol = val / cr; vr = vol / vol.shift(1).rolling(60, min_periods=40).mean()
th = np.where(IDX >= pd.Timestamp('2015-06-01'), 0.32, 0.25)[:, None]
d = r6.sub(m6, axis=0).to_numpy(); R6 = r6.to_numpy()
PF = ((R6 > th) & (d > 0.2)) | ((R6 < -th) & (d < -0.2)); VF = (vr.to_numpy() >= 5) & (np.abs(R6) > 0.10)
F = pd.DataFrame(PF | VF, index=IDX, columns=cols).astype(int)
pre = (IDX < pd.Timestamp('2014-04-01'))[:, None]
trig = ((F.rolling(5).sum() == 5).to_numpy() & pre) | ((F.rolling(3).sum() == 3).to_numpy() & ~pre) | (F.rolling(10).sum() >= 6).to_numpy()
trig = pd.DataFrame(trig, index=IDX, columns=cols)
prev = trig.astype(int).shift(1).rolling(10, min_periods=1).sum().fillna(0) > 0
new = trig & ~prev
U = pd.read_csv(HERE / 'data/universe_announce.csv', dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
U = U[U.code.isin(cols)]
pos = {c: [] for c in cols}
for r in U.itertuples(): pos[r.code].append((IDX.searchsorted(r.announce), IDX.searchsorted(r.end, 'right') - 1))
ev = []
ii, jj = np.where(new.to_numpy())
for i, j in zip(ii, jj):
    c = cols[j]
    hit = any(a - 3 <= i <= b + 3 for a, b in pos[c])   # 觸發日落在某筆處置 [公告−3, 結束+3] 交易日內
    ev.append(dict(code=c, trig=IDX[i], year=IDX[i].year, hit=hit, r6=float(R6[i, j]), vr=float(vr.iat[i, j]) if np.isfinite(vr.iat[i, j]) else np.nan,
                   n_flag10=int(F.iloc[max(0, i - 9):i + 1, j].sum())))
E = pd.DataFrame(ev)
# 反向：universe 列有無近似事件（召回）
rec = []
evp = {c: g.trig.map(IDX.get_loc).to_numpy() for c, g in E.groupby('code')}
for r in U.itertuples():
    a = IDX.searchsorted(r.announce); ok = r.code in evp and np.any((evp[r.code] >= a - 10) & (evp[r.code] <= a + 1))
    rec.append(dict(year=r.start.year, hit=bool(ok)))
Rc = pd.DataFrame(rec)
byy = {int(y): dict(n_proxy=int(len(g)), precision=round(float(g.hit.mean()), 3)) for y, g in E.groupby('year')}
for y, g in Rc.groupby('year'): byy.setdefault(int(y), {}).update(n_universe=int(len(g)), recall=round(float(g.hit.mean()), 3))
cand_years = sorted(set([2019] + [y for y in R['RESULTS_MODEL']['years_below_minus2sigma']]))
C = E[(~E.hit) & E.year.isin(cand_years)].sort_values(['year', 'n_flag10', 'r6'], ascending=[True, False, False])
C.to_csv(HERE / 'data/proxy_unmatched_candidates.csv', index=False)
# 2019 與相鄰年 precision 比較（逐年同一定義）
out = dict(by_year=byy, candidate_years=cand_years, n_candidates={int(y): int((C.year == y).sum()) for y in cand_years},
           note='precision＝近似事件落在某筆 universe 處置 [公告−3, 結束+3] 的比例；recall＝universe 列在公告前 10～後 1 個交易日內有近似事件的比例。2019 若明顯低於相鄰年，支持 H_inc。')
(HERE / 's7_proxy_match.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
for y, v in sorted(byy.items()): print(y, v)
print(out['n_candidates'])
print(C[C.year == 2019].head(40).to_string())
