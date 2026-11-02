#!/usr/bin/env python3
"""第 3 步：PREREG（sha256＋時間戳）寫入 results.json。雜湊涵蓋 frozen_spec.txt、全部程式、explore.json、iterations.json、gap_map.json、build_meta.json。"""
import json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
files = ['frozen_spec.txt', 'explore.json', 'iterations.json', 'data/gap_map.json', 'data/build_meta.json', 'data/gate_engine.json'] + sorted(p.name for p in HERE.glob('*.py'))
h = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in files}
spec = (HERE / 'frozen_spec.txt').read_text()
pre = dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), spec_sha256=h['frozen_spec.txt'], files_sha256=h,
           main='v4_ALL400_n10_cap50', main_spec=dict(pool='ALL', K=400, n=10, cap=0.5), report=dict(R1=dict(pool='ALL', K=500, n=10), R2=dict(pool='TP', K=300, n=10)),
           base=dict(pool='TW', K=300, n=10), disposal='d1_ALL_all', satellite_ratios=[0.3, 0.5],
           neighbors=dict(n5=dict(pool='ALL', K=400, n=5, cap=0.5), n15=dict(pool='ALL', K=400, n=15, cap=0.5), K300=dict(pool='ALL', K=300, n=10, cap=0.5),
                          K500=dict(pool='ALL', K=500, n=10, cap=0.5), floor0=dict(pool='ALL', K=400, n=10, cap=0.5, floor=1.0), floor2k=dict(pool='ALL', K=400, n=10, cap=0.5, floor=2e7)),
           note='打開驗證期之前寫入；s4_oos.py 先斷言 PREREG 存在且 frozen_spec.txt 雜湊相符')
pre['prereg_sha256'] = hashlib.sha256(json.dumps(pre, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
R = HERE / 'results.json'
doc = json.loads(R.read_text()) if R.exists() else {}
assert 'PREREG' not in doc, 'PREREG 已存在'
doc['PREREG'] = pre
R.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
print(pre['time_utc'], pre['prereg_sha256'])
