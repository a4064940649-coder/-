#!/usr/bin/env python3
"""o18 題 2 獨立複製（步驟 6，描述）：C0 union 對 orig 的移出／移入股下一期報酬（訊號日收盤→下一訊號日收盤，含息 MYADJ）。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r3_paper as R3      # 會重跑 r3（名單＋模擬），只為取名單；輸出覆寫 r3_paper.json（內容相同）
sys.path.pop(0)
K = R3.K; A = K.MYADJ
SD = R3.SD; END = R3.END
pts = list(SD[SD < END]) + [END]
nxt = dict(zip(pts[:-1], pts[1:]))
def r(c, a, b):
    s = A[c].dropna(); x = s[s.index <= a]; y = s[s.index <= b]
    return float(y.iloc[-1] / x.iloc[-1] - 1) if len(x) and len(y) else np.nan
out = {}
for var in ('union', 'replace'):
    o, u = R3.PK[('C0', 'orig')], R3.PK[('C0', var)]
    rows = []
    for d in o:
        if d not in nxt: continue
        for c in o[d] - u[d]: rows.append(('out', str(d.date()), c, r(c, d, nxt[d])))
        for c in u[d] - o[d]: rows.append(('in', str(d.date()), c, r(c, d, nxt[d])))
    df = pd.DataFrame(rows, columns=['side', 'd', 'code', 'ret'])
    ro, ri = df[df.side == 'out'], df[df.side == 'in']
    contrib = (ri.ret.sum() - ro.ret.sum()) * 0.1 * 100
    y19 = df[df.d >= '2019-01-01']
    c19 = (y19[y19.side == 'in'].ret.sum() - y19[y19.side == 'out'].ret.sum()) * 0.1 * 100
    out[var] = dict(n_out=len(ro), n_in=len(ri), mean_out=float(ro.ret.mean() * 100), mean_in=float(ri.ret.mean() * 100),
                    contrib_pp=float(contrib), contrib_2019on_pp=float(c19), rows=rows)
(HERE / 'r6_removed.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
for k, v in out.items(): print(k, {a: b for a, b in v.items() if a != 'rows'})
