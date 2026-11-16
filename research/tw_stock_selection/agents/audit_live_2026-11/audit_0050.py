#!/usr/bin/env python3
"""稽核第 3 項補充：只用官方日 K 自建 0050 含息指數（日報酬＝收盤 ÷（收盤 − 漲跌價差）− 1，停牌日報酬 0），
檢查 2025-06 分割（1 拆 4）與停牌日，算逐年報酬，和引擎 0050（finlab 還原價）與網站 yearly.json 比。輸出 audit_0050.json。python3 -I -B audit_0050.py"""
import subprocess, json, io
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
RAW = '/home/user/-/research/tw_stock_selection/raw/jb7k6295r7-bot_tw-stock-data'
TSD = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/tsd'


def rows(repo, dates_only=None):
    ls = subprocess.run(['git', '-C', repo, 'ls-tree', 'HEAD', 'data/universe/daily/'], capture_output=True, text=True, check=True).stdout.split('\n')
    items = [(l.split('\t')[1].rsplit('/', 1)[1][:10], l.split()[2]) for l in ls if l]
    if dates_only is not None:
        items = [x for x in items if x[0] in dates_only]
    p = subprocess.Popen(['git', '-C', repo, 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out = {}
    for d, sha in items:
        p.stdin.write((sha + '\n').encode()); p.stdin.flush()
        hdr = p.stdout.readline().split(); n = int(hdr[2]); b = p.stdout.read(n); p.stdout.read(1)
        for line in b.split(b'\n'):
            if b',0050,' in line[:40]:
                f = line.decode().split(',')
                if f[2] == '0050':
                    out[d] = (float(f[5]) if f[5] else np.nan, float(f[8]) if f[8] else np.nan, float(f[11]) if f[11] else np.nan)
        out.setdefault(d, None)
    p.stdin.close(); p.wait()
    return out


a = rows(RAW)
b = rows(TSD, dates_only={d for d in ['2026-11-02', '2026-11-03', '2026-11-04', '2026-11-05', '2026-11-06', '2026-11-09', '2026-11-10', '2026-11-11', '2026-11-12', '2026-11-13']})
a.update(b)
days = sorted(a)
miss = [d for d in days if a[d] is None]
C = pd.Series({d: a[d][1] if a[d] else np.nan for d in days}); CH = pd.Series({d: a[d][2] if a[d] else np.nan for d in days})
C.index = pd.to_datetime(C.index); CH.index = C.index
ref = C - CH
r = (C / ref - 1)
big = r[np.abs(r) > 0.15]
r = r.fillna(0.0)
idx = (1 + r).cumprod()
s = idx.loc['2015-07-01':'2026-10-22']
ye = s.groupby(s.index.year).last(); yr = {}
for y in ye.index:
    prev = s.iloc[0] if y == s.index[0].year else ye.loc[y - 1]
    yr[int(y)] = round(float(ye.loc[y] / prev - 1) * 100, 1)
# 2015-07-01 的報酬不應算入（起點＝當日收盤）
site = {int(x['年']): x['0050'] for x in json.load(open('/home/user/-/xq_scripts/site/data/yearly.json', encoding='utf-8'))['rows']}
bt = {int(k): v for k, v in json.load(open(HERE / 'audit_backtest.json', encoding='utf-8'))['yearly_0050'].items()}
dd = (s / s.cummax() - 1)
out = dict(n_days=len(days), days_without_0050=miss, abs_ret_gt_15pct=[(str(k.date()), round(float(v), 4)) for k, v in big.items()],
           split=dict(close_0610=float(C.loc['2025-06-10']), close_0618=float(C.loc['2025-06-18']), ref_0618=float(ref.loc['2025-06-18']),
                      ratio=round(float(C.loc['2025-06-10'] / ref.loc['2025-06-18']), 4)),
           yearly_official=yr, yearly_engine=bt, yearly_site=site,
           max_abs_official_vs_engine=max(abs(yr[y] - bt[y]) for y in yr if y in bt), max_abs_official_vs_site=max(abs(yr[y] - site[y]) for y in yr if y in site),
           maxdd_official=float(dd.min()), maxdd_trough=str(dd.idxmin().date()),
           live=dict(open_1112=a['2026-11-12'][0], close_1112=a['2026-11-12'][1], close_1113=a['2026-11-13'][1]))
json.dump(out, open(HERE / 'audit_0050.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False))
