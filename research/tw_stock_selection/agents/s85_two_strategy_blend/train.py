#!/usr/bin/env python3
"""S85 訓練段（2016-02～2021-12）：只算訓練段數字，依 PREREG 選凍結格，寫 FROZEN.json＋sha256、train_results.json。已存在就拒跑。"""
import sys, json, hashlib
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s85lib as L
assert not (L.H / 'FROZEN.json').exists(), '已凍結，拒絕重跑'
sd, bench, C, th = L.load()
cut = '2021-12-31'
sdT = sd.loc[:cut]; bT = bench.loc[:cut]; CT = {k: v.loc[:cut] for k, v in C.items()}      # 只餵訓練段資料
cur = L.stats(sdT['nav'].loc[L.START:], bT, th, L.TRAIN)
cur['ex_top5pct_cagr'] = L.stats(sdT['nav_blk5pct'].loc[L.START:], bT, th, L.TRAIN)['cagr']
res = {'current': cur, 'cells': {}, 'candidates_alone': {}}
for nm in L.COMPARABLE + L.REFERENCE:
    res['candidates_alone'][nm] = L.stats(CT[nm].loc[L.START:], bT, th, L.TRAIN)
for grp, names in (('comparable', L.COMPARABLE), ('reference', L.REFERENCE)):
    cells = L.all_cells(sdT, bT, CT, names); cb = L.all_cells(sdT, bT, CT, names, blk=True)
    for k, nav in cells.items():
        s = L.stats(nav, bT, th, L.TRAIN); s['ex_top5pct_cagr'] = L.stats(cb[k], bT, th, L.TRAIN)['cagr']
        s['group'] = grp; s.update(L.passes(s, cur)); res['cells'][k] = s
def pick(grp, need_pass):
    xs = [(v['IR'], v['cagr'], k) for k, v in res['cells'].items() if v['group'] == grp and (v['passed'] or not need_pass)]
    return max(xs)[2] if xs else None
fz = dict(frozen_comparable=pick('comparable', True), record_only_comparable=pick('comparable', False), frozen_reference=pick('reference', True),
          train_qualified=[k for k, v in res['cells'].items() if v['passed']], prereg_sha256=(L.H / 'PREREG.sha256').read_text().split()[0])
(L.H / 'train_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
b = json.dumps(fz, ensure_ascii=False, indent=1).encode(); (L.H / 'FROZEN.json').write_bytes(b)
(L.H / 'FROZEN.sha256').write_text(hashlib.sha256(b).hexdigest() + '  FROZEN.json\n')
pd.set_option('display.width', 250)
df = pd.DataFrame(res['cells']).T
for c in ('cagr', 'maxdd', 'excess', 'cold_excess', 'ex_top5pct_cagr'):
    df[c] = (df[c].astype(float) * 100).round(2)
print('現行', {k: round(v * 100, 2) if k in ('cagr', 'maxdd', 'excess', 'cold_excess', 'ex_top5pct_cagr') else (round(v, 3) if isinstance(v, float) else v) for k, v in cur.items()})
print(df[['cagr', 'maxdd', 'excess', 't', 'IR', 'cold_excess', 'ex_top5pct_cagr', 'ret_ok', 'dd_ok', 'cold_ok', 'ir_ok', 'passed']].to_string())
print(pd.DataFrame(res['candidates_alone']).T.round(4).to_string())
print(fz)
