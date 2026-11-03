#!/usr/bin/env python3
"""題 1 凍結選定版的產業排名 → data/t1_sector_order.json。
先核對題 1 PREREG 存在、lclib.py 與資料檔雜湊相符，再用題 1 程式（唯讀 import）在本題每個訊號日（13 個 j＋官方日，2006-04 起）
算可輪動產業依選定訊號由高到低的排序（同分比 B 權重，與題 1 sector_targets 相同）。緩衝帶（k+2）由本題 Strat 逐路徑套用（同題 1 規則）。
python3 -I -B s0b_t1_adapter.py"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import lc_lib as X
T1 = X.T1DIR
pre = json.loads((T1 / 'data' / 'PREREG.json').read_text(encoding='utf-8'))
P = pre['PREREG']
for f, h in P['code_sha256'].items():
    if f == 'lclib.py': assert X.sha(T1 / f) == h, '題 1 lclib.py 與其 PREREG 不符'
for f, h in P['data_sha256'].items(): assert X.sha(T1 / f) == h, f
cfg = P['chosen']['cfg']; sig, kk = cfg.split('|')[0], int(cfg.split('|')[1][1:])
params = P['chosen']['params']
m = X.t1_module(); m.SECT_MIN = params['sect_min']
caps = m.Caps(); sidx = m.SectorIndex(caps); S = m.Signals(caps, sidx)
days = set()
for j in range(13): days |= set(X.days_for(j, 'FULL')[:-1])
days |= set(X.sig_days('off', '2006-04-01', '2026-10-31')[:-1])
order = {}
for d in sorted(days):
    cs = S.at(d); ST = cs['ST']
    if len(ST) == 0: order[str(d.date())] = []; continue
    Bs = ST['B'] / cs['mc'].sum(); v = ST[sig].dropna()
    order[str(d.date())] = sorted(v.index, key=lambda s: (-v[s], -Bs[s]))
out = dict(source='o_lc_sector_rotation_pit 凍結選定版', topic1_prereg_sha256=pre['sha256'], cfg=cfg, signal=sig, k=kk, params=params,
           lclib_sha256=X.sha(T1 / 'lclib.py'), n_days=len(order), order=order)
(HERE / 'data' / 't1_sector_order.json').write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
print('done', cfg, params, len(order))
