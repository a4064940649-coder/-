#!/usr/bin/env python3
"""integrate_o19 名單層前置（不算報酬）：把審查題 2019（及 2006、2008 殘差 < −2σ 年）面板代理「未比對」事件做成擬處置列，
供 I3 敏感度（2019 處置清單疑似不完整的影響界限）使用。擬處置期＝觸發日的下一交易日起 10 個交易日（2019 年制的一般處置長度；
與審查題 K_reg「近似起日＝觸發日下一交易日」一致）。只讀審查題檔案；寫 integrate_o19/data/。
  I3a：s7b_strong_unmatched.json 的 unmatched_2019（強觸發、未比對，9 起）
  I3b：data/proxy_unmatched_candidates.csv 中 year＝2019 全部（K_reg 代理未比對，極端上界）
  I3c：同上 year ∈ {2006, 2008, 2019}
輸出：data/DU_I3{a,b,c}.csv＝DISP_OFFICIAL_v2（嚴格）＋擬處置列（code,start,end,src）。
python3 -I -B mk_pseudo.py"""
import sys, json, hashlib
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
AU = AG / 'o_disposal_list_pit_audit_pkl18'
sys.path.insert(0, str(AG / 'o_rev_avail_ds_pit_count')); import pclib as P; sys.path.pop(0)
IDX = P.IDX
V2 = pd.read_csv(AU / 'data/DISP_OFFICIAL_v2.csv', dtype={'code': str}, parse_dates=['start', 'end'])[['code', 'start', 'end']]
V2['src'] = 'DISP_OFFICIAL_v2'
s7b = json.loads((AU / 's7b_strong_unmatched.json').read_text(encoding='utf-8'))
PC = pd.read_csv(AU / 'data/proxy_unmatched_candidates.csv', dtype={'code': str}, parse_dates=['trig'])
def pseudo(ev, tag):
    rows = []
    for c, tr in ev:
        i = IDX.searchsorted(pd.Timestamp(tr), 'right')
        if i + 9 >= len(IDX): continue
        rows.append(dict(code=c, start=IDX[i], end=IDX[i + 9], src=tag))
    return pd.DataFrame(rows)
SETS = {'I3a': [(r['code'], r['trig']) for r in s7b['unmatched_2019']],
        'I3b': list(zip(PC[PC.year == 2019].code, PC[PC.year == 2019].trig)),
        'I3c': list(zip(PC[PC.year.isin([2006, 2008, 2019])].code, PC[PC.year.isin([2006, 2008, 2019])].trig))}
out = {}
for k, ev in SETS.items():
    ps = pseudo(ev, 'pseudo_' + k)
    D = pd.concat([V2, ps], ignore_index=True)
    f = HERE / f'data/DU_{k}.csv'; D.to_csv(f, index=False, date_format='%Y-%m-%d')
    out[k] = dict(n_events=len(ev), n_pseudo_rows=len(ps), n_in_CI=int(ps.code.isin(P.CI).sum()), by_year=ps.start.dt.year.value_counts().sort_index().to_dict(),
                  file=f.name, sha256=hashlib.sha256(f.read_bytes()).hexdigest())
out['src_sha256'] = {str(p.relative_to(AG)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (AU / 'data/DISP_OFFICIAL_v2.csv', AU / 's7b_strong_unmatched.json', AU / 'data/proxy_unmatched_candidates.csv', AU / 'data/DISP_OFFICIAL_v2_posthoc.csv')}
(HERE / 'pseudo.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, default=str))
