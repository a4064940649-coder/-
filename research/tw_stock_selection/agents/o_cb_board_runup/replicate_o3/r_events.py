#!/usr/bin/env python3
"""複製員：董事會決議發行公司債事件（自己的分類規則）＋ FSC 覆蓋率。"""
import json
from pathlib import Path
import pandas as pd, numpy as np
HERE = Path(__file__).resolve().parent
d = pd.read_parquet(HERE / 'my_titles.parquet')
t = d['title'].fillna('').str.replace(r'\s+', '', regex=True)
d['t'] = t
bond = t.str.contains('公司債')
board = t.str.contains('董事會') | t.str.contains('擬發行|擬辦理|擬募集')
act = t.str.contains('發行|募集|辦理|私募')
BAD = ('更正|補充|買回|註銷|停止轉換|轉換價格|代子公司|取得|到期|賣回|收足|募集完成|訂價|定價|撤回|撤銷|廢止|子公司|利率|延長|展延|取消|變更|修正'
       '|除息|除權|配息|基準日|股東會|股東常會|認購|增資基準|贖回|代收|存儲|專戶|資金|計畫|計劃|用途')
cand = d[bond & board & act & ~t.str.contains(BAD)].copy()
def kind(s):
    if '私募' in s: return 'PCB' if ('轉換' in s) else 'PSB'
    if '交換' in s: return 'EB'
    if '海外' in s or '外國' in s: return 'ECB'
    if '轉換' in s: return 'CB'
    return 'SB'
cand['kind'] = cand['t'].map(kind)
cand = cand.sort_values(['code', 'kind', 'date'])
GAP = int(__import__('os').environ.get('GAP', 30))
ev = []
for (c, k), g in cand.groupby(['code', 'kind']):
    last = None
    for _, r in g.iterrows():
        if last is not None and (r['date'] - last).days <= GAP:
            last = r['date']; continue
        ev.append({'code': c, 'kind': k, 'board': r['date'], 'title': r['title']}); last = r['date']
ev = pd.DataFrame(ev).sort_values(['board', 'code']).reset_index(drop=True)
out = {'n_by_kind': ev['kind'].value_counts().to_dict()}

# ---------- FSC 上市國內公募 CB（2011-01～2020-08）----------
f = pd.read_csv(HERE.parent.parent / 'o_issuance_avoid/fsc_cases_all.csv', dtype=str)
f['item2'] = f['item'].fillna('').str.replace(r'\s+', '', regex=True)
f['recv'] = pd.to_datetime(f['recv'], errors='coerce'); f['eff'] = pd.to_datetime(f['eff'], errors='coerce')
cb = f[f.item2.str.contains('轉換公司債') & ~f.item2.str.contains('海外|私募|補辦|外國') & (f.ctype == '上市')].copy()
x = cb[(cb.recv >= '2011-01-01') & (cb.recv < '2020-09-01')].copy()
SE = set(json.loads(json.dumps(__import__('ast').literal_eval((HERE.parent / 'raw/nccu_mops/databases/SE.txt').read_text(encoding='utf-8')))))
codes_have = set(d['code'])
evc = ev[ev.kind == 'CB']
def match(r, lo=0, hi=180):
    e = evc[(evc.code == r.code) & (evc.board <= r.recv) & ((r.recv - evc.board).dt.days <= hi)]
    return e['board'].max() if len(e) else pd.NaT
x['board'] = [match(r) for r in x.itertuples()]
x['hit'] = x['board'].notna()
x['insurv'] = x['code'].isin(codes_have)
wd = x['result'] != '生效'
out['fsc_n'] = len(x); out['fsc_hit'] = int(x.hit.sum()); out['cov'] = round(x.hit.mean() * 100, 1)
out['cov_by_year'] = {int(k): round(v * 100) for k, v in x.groupby(x.recv.dt.year)['hit'].mean().items()}
out['wd_n'] = int(wd.sum()); out['wd_hit'] = int(x.hit[wd].sum())
out['miss_not_in_list'] = int((~x.hit & ~x.insurv).sum()); out['miss_in_list'] = int((~x.hit & x.insurv).sum())
out['board_to_recv_median_days'] = float((x.recv - x.board).dt.days[x.hit].median()); out['n_b2r'] = int(x.hit.sum())
print(json.dumps(out, ensure_ascii=False, default=str))
ev.to_csv(HERE / 'my_events.csv', index=False)
x.to_csv(HERE / 'my_fsc_match.csv', index=False)
