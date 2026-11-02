"""資料閘門：EPS 對 hawan0228／sinotrade、季營收對月營收加總、股池覆蓋率。"""
from pathlib import Path
import json, glob
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
D = HERE / 'data'
fin = pd.read_parquet(D / 'fin.parquet')
key = fin.set_index(['stock_id', 'period']).eps
res = {}
# hawan0228（單季 EPS，日期＝季底）
rows = []
for f in glob.glob('/home/user/hawan0228/fintech_ai/data/external/raw/financials/*_financial.csv'):
    d = pd.read_csv(f, dtype={'stock_id': str}, encoding='utf-8-sig')
    d = d[d.type == 'EPS']
    for _, r in d.iterrows():
        t = pd.Timestamp(r.date)
        rows.append((r.stock_id, f'{t.year}Q{(t.month - 1) // 3 + 1}', float(r.value)))
h = pd.DataFrame(rows, columns=['stock_id', 'period', 'v'])
h['mine'] = [key.get((a, b), np.nan) for a, b in zip(h.stock_id, h.period)]
h = h.dropna()
res['hawan'] = dict(n=len(h), agree=float(np.mean(np.abs(h.v - h.mine) <= 0.011)))
# sinotrade（單季）
rows = []
for f in glob.glob('/home/user/sinotrade/shioaji-data/v1/tw/financials/json/*.json'):
    d = json.load(open(f))
    s = d.get('statements') or {}
    iq = s.get('income_quarter') or {}
    per = s.get('periods') or []
    e = iq.get('基本每股盈餘（元）')
    if not e:
        continue
    for p, v in zip(per, e):
        if v is not None:
            rows.append((d['code'], p, float(v)))
s = pd.DataFrame(rows, columns=['stock_id', 'period', 'v'])
s['mine'] = [key.get((a, b), np.nan) for a, b in zip(s.stock_id, s.period)]
s = s.dropna()
res['sinotrade'] = dict(n=len(s), agree=float(np.mean(np.abs(s.v - s.mine) <= 0.011)))
# 季營收 vs 三個月月營收加總（千元）
rv = pd.read_parquet(D / 'rev_m.parquet')
rv['y'] = rv.period.str[:4].astype(int); rv['m'] = rv.period.str[5:7].astype(int); rv['q'] = (rv.m - 1) // 3 + 1
rv = rv.drop_duplicates(['stock_id', 'period'])
g = rv.groupby(['stock_id', 'y', 'q']).agg(s=('rev', 'sum'), n=('rev', 'count')).reset_index()
g = g[g.n == 3]
g['period'] = g.y.astype(str) + 'Q' + g.q.astype(str)
m = g.merge(fin[fin.industry == 'ci'][['stock_id', 'period', 'rev']], on=['stock_id', 'period']).dropna()
m = m[m.rev.abs() > 0]
res['rev_sum'] = dict(n=len(m), within2=float(np.mean(np.abs(m.s * 1000 / m.rev - 1) <= 0.02)))
# 股池覆蓋率：各季 A 日股池（ci 業、4 碼）中有事件者比例 → 以「季中股池股票有事件」近似
E = pd.read_parquet(D / 'events_all.parquet')
av = pd.read_parquet(D / 'avail.parquet')
res['A_src_share'] = av.src.value_counts(normalize=True).round(3).to_dict()
print(json.dumps(res, indent=1, ensure_ascii=False))
json.dump(res, open(HERE / 'r7_gates.json', 'w'), indent=1, ensure_ascii=False)
