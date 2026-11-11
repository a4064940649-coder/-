#!/usr/bin/env python3
"""integrate_o20 前視計數第二步：以實際交易日曆（run_paper 日淨值索引＝o18 面板 IDX）重判 lookahead_check 的 7 列（公告日落在週末），
並查這些列是否進入名單差（紙上 M1→M2、V7X base→M2）。python3 -I -B lookahead_check2.py"""
import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert hashlib.sha256(json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256']
IDX = pd.DatetimeIndex(pd.read_csv(HERE / 'data/paper_daily_navs.csv.gz', usecols=[0], index_col=0, parse_dates=True).index)
rows = R['LOOKAHEAD_CHECK']['a_rows']
LD = R['RESULTS_PAPER']['list_diffs']
VD = pd.read_csv(HERE / 'data/v7x_plan_diff_M2.csv.gz', dtype=str) if (HERE / 'data/v7x_plan_diff_M2.csv.gz').exists() else None
out = []
for code, ordr, ann, st, en in rows:
    ann, st, en = map(pd.Timestamp, (ann, st, en))
    ann_trading = bool(ann in IDX)
    d_last = IDX[IDX < st][-1]  # 起日前一個交易日：此日 e＝起日
    la = bool(ann > d_last)
    pap = {k: [x for x in v if x['signal'] == str(d_last.date()) and (code in x['into'] or code in x['out'])] for k, v in LD.items()}
    vx = int(((VD.signal == str(d_last.date())) & (VD['into'].fillna('').str.contains(code) | VD['out'].fillna('').str.contains(code))).sum()) if VD is not None else None
    vx_any_day = int(((VD['into'].fillna('').str.contains(code)) & (VD.signal >= str((st - pd.Timedelta(days=10)).date())) & (VD.signal <= str(en.date()))).sum()) if VD is not None else None
    out.append(dict(code=code, ordr=ordr, announce=str(ann.date()), start=str(st.date()), announce_is_trading_day=ann_trading, d_before_start=str(d_last.date()),
                    lookahead=la, paper_diff_hits={k: v for k, v in pap.items() if v}, v7x_diff_rows_on_d=vx, v7x_diff_rows_into_near=vx_any_day))
res = dict(rows=out, n_lookahead=sum(o['lookahead'] for o in out), n_affect_paper=sum(bool(o['paper_diff_hits']) for o in out),
           n_affect_v7x=(sum((o['v7x_diff_rows_on_d'] or 0) > 0 for o in out) if VD is not None else None))
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8')); R['LOOKAHEAD_CHECK2'] = res
(HERE / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(res, ensure_ascii=False, indent=0))
