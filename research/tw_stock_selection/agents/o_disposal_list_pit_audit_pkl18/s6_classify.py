#!/usr/bin/env python3
"""S3b：本地提及 → 狀態分類（依 SPEC_draft 狀態表，程式化部分），逐年比對率；未比對事件輸出供逐筆閱讀。
python3 -I -B s6_classify.py → data/local_mentions_classified.csv、data/unmatched_events.csv、s6_classify.json"""
import json, re, sys
SUF = sys.argv[1] if len(sys.argv) > 1 else ""
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
cr = pd.read_parquet(P / 'close_raw.parquet'); TD = cr.index[cr.notna().sum(axis=1) >= 300]
# 面板外日期（2024 缺口等）用工作日補
TD = TD.union(pd.bdate_range('2024-01-02', '2024-02-14'))
U = pd.read_csv(HERE / 'data/universe_announce.csv', dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
B = pd.read_csv(HERE / 'raw/lineage/disposal_hist_rejudge.csv', dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
LM = pd.read_csv(AG / 'o_v7_listing_exec_engine/listing_master.csv', dtype={'code': str})
seg = {}
for c, s in zip(LM.code, LM.segments.fillna('')):
    for part in str(s).split(';'):
        if '~' in part:
            a, b = part.split('~'); seg.setdefault(c, []).append((pd.Timestamp(a), pd.Timestamp(b)))
sl = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype={'code': str})
mkt = dict(zip(sl.code, sl.market))
def tp(x): return TD.searchsorted(x)
iv = {}
for r in U.itertuples(): iv.setdefault(r.code, []).append((tp(r.announce) - 10, tp(r.end) + 10, r.start))
ivB = {}
for r in B.itertuples(): ivB.setdefault(r.code, []).append((tp(r.announce) - 10, tp(r.end) + 10))
M = pd.read_csv(HERE / f'data/local_mentions{SUF}.csv', dtype={'code': str}, parse_dates=['date'])
def twse_at(c, d):
    if c in seg: return any(a <= d <= b for a, b in seg[c])
    return mkt.get(c) == 'TWSE'
st = []
for r in M.itertuples():
    t = tp(r.date)
    if any(a <= t <= b for a, b, _ in iv.get(r.code, [])): st.append('ALREADY_LISTED')
    elif not twse_at(r.code, r.date): st.append('NOT_COMMON_OR_TPEX')
    elif any(a <= t <= b for a, b in ivB.get(r.code, [])): st.append('ALREADY_LISTED_B')   # 在 rejudge（非普通股列不會有，保險）
    else: st.append('UNMATCHED')
M['status_auto'] = st
M.to_csv(HERE / f'data/local_mentions_classified{SUF}.csv', index=False)
y = M.date.dt.year
tab = M.groupby([y, 'src', 'status_auto']).size().unstack(fill_value=0)
rate = {}
for (yy, s), g in M.groupby([y, 'src']):
    tw = g[g.status_auto.isin(['ALREADY_LISTED', 'UNMATCHED'])]
    rate[f'{yy}|{s}'] = dict(n_twse=int(len(tw)), matched=int((tw.status_auto == 'ALREADY_LISTED').sum()),
                             rate=round(float((tw.status_auto == 'ALREADY_LISTED').mean()), 3) if len(tw) else None,
                             n_unique_code_matched=int(tw[tw.status_auto == 'ALREADY_LISTED'].code.nunique()), n_unique_code_unmatched=int(tw[tw.status_auto == 'UNMATCHED'].code.nunique()))
# universe 事件被至少一個來源提及的比例（新聞覆蓋度；同年同來源）
cov = {}
for s in M.src.unique():
    ms = M[(M.src == s) & (M.status_auto == 'ALREADY_LISTED')]
    yrs = sorted(ms.date.dt.year.unique())
    for yy in yrs:
        Uy = U[U.start.dt.year == yy]
        hit = 0
        for r in Uy.itertuples():
            a, b = tp(r.announce) - 10, tp(r.end) + 10
            mm = ms[ms.code == r.code]
            if len(mm) and any(a <= tp(d) <= b for d in mm.date): hit += 1
        cov[f'{yy}|{s}'] = dict(n_universe=int(len(Uy)), mentioned=hit, share=round(hit / max(len(Uy), 1), 3))
# 未比對事件：同代碼 10 個交易日內合併
un = M[M.status_auto == 'UNMATCHED'].sort_values(['code', 'date'])
ev = []; 
for c, g in un.groupby('code'):
    cur = None
    for r in g.itertuples():
        t = tp(r.date)
        if cur is None or t - cur['t_last'] > 10:
            if cur: ev.append(cur)
            cur = dict(code=c, first=r.date, last=r.date, t_last=t, srcs=set(), n=0, snips=[])
        cur['last'] = r.date; cur['t_last'] = t; cur['srcs'].add(r.src); cur['n'] += 1
        if len(cur['snips']) < 4: cur['snips'].append(f'[{r.src} {r.date.date()} {r.how} {r.kw}] {r.snip}')
    ev.append(cur)
E = pd.DataFrame([dict(code=e['code'], first=e['first'].date(), last=e['last'].date(), year=e['first'].year, srcs='+'.join(sorted(e['srcs'])), n_src=len(e['srcs']), n_mentions=e['n'], snips=' || '.join(e['snips'])) for e in ev])
E.to_csv(HERE / f'data/unmatched_events{SUF}.csv', index=False)
out = dict(status_by_year_src={f'{a}|{b}': {k: int(v) for k, v in r.items()} for (a, b), r in tab.iterrows()}, match_rate=rate, news_coverage_of_universe=cov,
           unmatched_events_by_year=E.groupby('year').size().astype(int).to_dict() if len(E) else {}, unmatched_events_multi_src=int((E.n_src >= 2).sum()) if len(E) else 0)
(HERE / f's6_classify{SUF}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
for k, v in rate.items(): print(k, v, cov.get(k))
print(out['unmatched_events_by_year'], out['unmatched_events_multi_src'])
