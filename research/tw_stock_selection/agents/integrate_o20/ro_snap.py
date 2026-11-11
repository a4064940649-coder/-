#!/usr/bin/env python3
"""integrate_o20 唯讀快照（自 integrate_o19/ro_snap.py 複製）：live_tracking/main_forward、o_forward_infra_live、o_g8p2a_protocol_addendum 全部檔案 sha256。
用法：python3 -I -B ro_snap.py start|end"""
import sys, json, hashlib, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; AG = HERE.parent
DIRS = [AG/'live_tracking'/'main_forward', AG/'o_forward_infra_live', AG/'o_g8p2a_protocol_addendum']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
rows = []
for d in DIRS:
    for p in sorted(d.rglob('*')):
        if p.is_file():
            st = p.stat(); rows.append([str(p.relative_to(AG)), sha(p), st.st_size, st.st_mtime])
tag = sys.argv[1]
out = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), n=len(rows),
           sha=hashlib.sha256(json.dumps([r[:3] for r in rows]).encode()).hexdigest(), rows=rows)
(HERE/f'ro_snapshot_{tag}.json').write_text(json.dumps(out, ensure_ascii=False, indent=0))
print(out['utc'], out['n'], out['sha'])
