#!/usr/bin/env python3
"""S94：KY 股的營收資料特性（從資料看得到的部分）：公告時點（revenue_seen 2026-09、2026-10 兩個月）、備註欄（幣別／合併）、SURGE 分布。python3 -I -B s94_ky_checks.py"""
import json, glob, os
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; OUT = HERE / 'out'
RAW = HERE.parents[1] / 'raw' / 'tsd_extra' / 'revenue_raw' / 'first'
inv = json.load(open(OUT / 'inventory.json')); KY = set(inv['ky_codes']); R = {}
# 1) 公告時點
for mon, dl in (('2026-09', '2026-10-12'), ('2026-10', '2026-11-10')):     # 2026-10-10 國慶日（週六）→ 期限順延到 10-12（週一）
    seen = {}
    fs = sorted(glob.glob(str(RAW / 'meta/revenue_seen' / mon / '*.csv.gz')))
    for f in fs:
        ts = pd.Timestamp(os.path.basename(f)[:17].replace('T', ' ')[:13] + ':' + os.path.basename(f)[13:15] + ':' + os.path.basename(f)[15:17], tz='UTC').tz_convert('Asia/Taipei')
        d = pd.read_csv(f, dtype=str)
        for c, mk in zip(d.stock_id, d.market):
            seen.setdefault(c, (ts, mk))
    s = pd.DataFrame([(c, t, mk) for c, (t, mk) in seen.items()], columns=['code', 'first_seen_tw', 'market'])
    s = s[s.market == 'sii']; s['ky'] = s.code.isin(KY)
    cut = pd.Timestamp(dl + ' 23:59:59', tz='Asia/Taipei')
    R['seen_' + mon] = dict(files=[os.path.basename(f) for f in fs], first_file_time_tw=str(s.first_seen_tw.min()), deadline=dl,
                            ky=dict(n=int(s.ky.sum()), by_deadline=int((s[s.ky].first_seen_tw <= cut).sum()), in_first_file=int((s[s.ky].first_seen_tw == s.first_seen_tw.min()).sum()),
                                    late=s[s.ky & (s.first_seen_tw > cut)].assign(t=lambda q: q.first_seen_tw.astype(str))[['code', 't']].to_dict('records')),
                            non_ky=dict(n=int((~s.ky).sum()), by_deadline=int((s[~s.ky].first_seen_tw <= cut).sum()), in_first_file=int((s[~s.ky].first_seen_tw == s.first_seen_tw.min()).sum()),
                                        late=s[~s.ky & (s.first_seen_tw > cut)].assign(t=lambda q: q.first_seen_tw.astype(str))[['code', 't']].to_dict('records')))
# 2) 出表日期與備註（最近 4 個月的含出表日期原表）
fr = []
for f in sorted(glob.glob(str(RAW / 'mops/revenue/2026-*.csv.gz'))):
    if '_esb' in f: continue
    fr.append(pd.read_csv(f, dtype=str))
d = pd.concat(fr, ignore_index=True); d = d[d.market == 'twse'].reset_index(drop=True); d['ky'] = d.stock_id.isin(KY); d['note'] = d['備註'].fillna('-')
R['report_dates'] = {str(k): v for k, v in d.groupby(['period', 'ky'])['出表日期'].agg(lambda s: dict(s.value_counts())).to_dict().items()}
has = d.note.str.strip().ne('-') & d.note.str.strip().ne('')
kw = lambda pat: {('KY' if k else '非KY'): [int(g[has.loc[g.index]].note.str.contains(pat).sum()), int(len(g))] for k, g in d.groupby('ky')}
R['notes'] = dict(with_note=kw('.'), fx=kw('匯率|幣|美元|人民幣|USD|RMB'), consolidated=kw('合併'), examples_ky=d[d.ky & has].note.drop_duplicates().head(12).tolist())
# 3) SURGE 分布（可選股池內）：KY 對非 KY
el = pd.read_csv(OUT / 'B/elig.csv.gz', dtype={'code': str}); el['ky'] = el.code.isin(KY)
R['surge_dist'] = {('KY' if k else '非KY'): dict(n=len(g), mean=float(g.surge.mean()), std=float(g.surge.std()), share_ge2=float((g.surge >= 2).mean()), share_ge2_5=float((g.surge >= 2.5).mean()),
                                                share_ma60=float(g.ma60.mean()), median_v60_m=float(g.v60_m.median())) for k, g in el.groupby('ky')}
res = json.load(open(OUT / 'B/res.json')); T = res['targets']
# 每月 L_MA 前 10／前 20 名中 KY 的占比
top = el[el.ma60 == 1].sort_values(['month', 'surge'], ascending=[True, False]); top['rk'] = top.groupby('month').cumcount() + 1
R['ky_share'] = dict(pool=float(el[el.ma60 == 1].ky.mean()), top10=float(top[top.rk <= 10].ky.mean()), top20=float(top[top.rk <= 20].ky.mean()), targets=float(np.mean([c in KY for v in T.values() for c in v])))
json.dump(R, open(OUT / 'ky_checks.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(R, ensure_ascii=False, default=str)[:6000])
# 4) 各快照時點累計已看到的比例（上市；KY 對非 KY）
cum = {}
for mon in ('2026-09', '2026-10'):
    seen = {}
    for f in sorted(glob.glob(str(RAW / 'meta/revenue_seen' / mon / '*.csv.gz'))):
        b = os.path.basename(f); ts = pd.Timestamp(f'{b[:10]} {b[11:13]}:{b[13:15]}:{b[15:17]}', tz='UTC').tz_convert('Asia/Taipei')
        d = pd.read_csv(f, dtype=str); d = d[(d.market == 'sii') & d.stock_id.str.fullmatch(r'[1-9]\d{3}')]
        for c in d.stock_id: seen.setdefault(c, ts)
    s = pd.Series(seen); ky = s.index.isin(list(KY))
    cum[mon] = {str(t)[:16]: dict(KY=f'{int((s[ky] <= t).sum())}/{int(ky.sum())}', nonKY=f'{int((s[~ky] <= t).sum())}/{int((~ky).sum())}') for t in sorted(s.unique())}
R['seen_cum'] = cum
for k in ('seen_2026-09', 'seen_2026-10'):
    for g in ('ky', 'non_ky'): R[k][g]['n_late'] = len(R[k][g].pop('late'))
json.dump(R, open(OUT / 'ky_checks.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: v for k, v in R.items()}, ensure_ascii=False, default=str)[:5000])
