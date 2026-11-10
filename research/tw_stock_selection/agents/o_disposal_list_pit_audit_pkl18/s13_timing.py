#!/usr/bin/env python3
"""S1b：公告發布時點的公開證據（盤後）。ELD（Yahoo 股市新聞，2016～2018，post_time 含時:分）中含處置字樣、可對應到 universe 列的文章：
發布時間相對於 announce 日與 start 日開盤（09:00）的位置；並抽出「明起／今起／明日起」字樣。
python3 -I -B s13_timing.py → s13_timing.json"""
import re, json
from pathlib import Path
import pandas as pd, numpy as np
HERE = Path(__file__).resolve().parent; AG = HERE.parent
EL = AG / 'o_rev_gap_news_2013/raw/roccqqck_stock_eland/5pa_article.csv'
e = pd.read_csv(EL); e['pt'] = pd.to_datetime(e['post_time'], errors='coerce')
U = pd.read_csv(HERE / 'data/universe_announce.csv', dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
U = U[(U.start >= '2015-12-01') & (U.start <= '2019-01-31')]
rj = pd.read_csv(HERE / 'raw/lineage/disposal_hist_rejudge.csv', dtype={'code': str})
nm = {re.sub(r'\s+', '', n): c for c, n in zip(rj.code, rj.name.fillna('')) if len(c) == 4}
KW = re.compile(r'列處置|列為處置|處置股|分盤交易|預收款券|人工管制')
rows = []
for r in e.itertuples():
    t = f"{r.title if isinstance(r.title, str) else ''} {r.content if isinstance(r.content, str) else ''}"
    if not KW.search(t) or pd.isna(r.pt): continue
    for m in KW.finditer(t):
        w = t[max(0, m.start() - 40): m.end() + 30]
        codes = set(re.findall(r'\((\d{4})\)', w)) | {c for n, c in nm.items() if len(n) >= 2 and n in w}
        for c in codes:
            x = U[(U.code == c) & (U.announce >= r.pt.normalize() - pd.Timedelta(days=4)) & (U.announce <= r.pt.normalize() + pd.Timedelta(days=1))]
            for u in x.itertuples():
                so = u.start + pd.Timedelta(hours=9)
                rows.append(dict(code=c, title=str(r.title)[:60], post_time=str(r.pt), announce=str(u.announce.date()), start=str(u.start.date()),
                                 post_minus_announce_h=round((r.pt - u.announce).total_seconds() / 3600, 2), before_start_open=bool(r.pt < so),
                                 phrase=(re.findall(r'(明起|今起|明日起|今日起|今\s*\(\d+\)\s*日起)', w) or [''])[0], snip=w.replace('<BR>', ' ')[:110]))
R = pd.DataFrame(rows).drop_duplicates(['code', 'post_time', 'announce'])
out = dict(n=int(len(R)), n_codes=int(R.code.nunique()) if len(R) else 0,
           post_on_announce_day=int((R.post_minus_announce_h.between(0, 24)).sum()) if len(R) else 0,
           post_on_announce_day_after_1330=int((R.post_minus_announce_h.between(13.5, 24)).sum()) if len(R) else 0,
           post_on_announce_day_before_1330=int((R.post_minus_announce_h.between(0, 13.5, inclusive='left')).sum()) if len(R) else 0,
           earliest_hour_on_announce_day=float(R[R.post_minus_announce_h.between(0, 24)].post_minus_announce_h.min()) if len(R) else None,
           rows=R.sort_values('post_time').to_dict('records'))
(HERE / 's13_timing.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print({k: v for k, v in out.items() if k != 'rows'})
for r in out['rows']: print(r['code'], r['post_time'], 'ann', r['announce'], 'start', r['start'], r['post_minus_announce_h'], r['phrase'], '|', r['title'])
