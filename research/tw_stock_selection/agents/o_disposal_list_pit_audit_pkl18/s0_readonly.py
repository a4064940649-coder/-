#!/usr/bin/env python3
"""唯讀快照：live_tracking/main_forward、o_forward_infra_live、o_g8p2a_protocol_addendum 全部檔案 sha256＋mtime。
python3 -I -B s0_readonly.py <tag>  → raw/readonly_<tag>.json"""
import sys, json, hashlib, os, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; AG = HERE.parent
DIRS = ['live_tracking/main_forward', 'o_forward_infra_live', 'o_g8p2a_protocol_addendum']
tag = sys.argv[1]
out = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'files': {}}
for d in DIRS:
    for p in sorted((AG / d).rglob('*')):
        if p.is_file():
            out['files'][str(p.relative_to(AG))] = [hashlib.sha256(p.read_bytes()).hexdigest(), os.path.getmtime(p)]
(HERE / 'raw' / f'readonly_{tag}.json').write_text(json.dumps(out, indent=0))
print(tag, out['utc'], len(out['files']))
