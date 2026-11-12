#!/usr/bin/env python3
"""o_cost_model_sensitivity_recon 唯讀快照：live_tracking/main_forward、o_forward_infra_live、o_g8p2a_protocol_addendum 全部檔案，
以及本題只讀輸入（o_turnover_fixedcost_diagnosis/data/*、引擎 .py、V7X x_cache.npz、integrate_o19/o20 results.json）的 sha256。只讀，不寫入那些資料夾。
用法：python3 -I -B ro_snap.py start|end"""
import sys, json, hashlib, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; AG = HERE.parent
DIRS = [AG / 'live_tracking' / 'main_forward', AG / 'o_forward_infra_live', AG / 'o_g8p2a_protocol_addendum', AG / 'o_turnover_fixedcost_diagnosis' / 'data']
FILES = [AG / 'o_engine_delist_consideration' / f for f in ('cz_v7x.py', 'cz_engine.py', 'events_v2.csv', 'results.json')] + \
        [AG / 'o_exec_engine_rewrite_v7x' / f for f in ('x_exec.py', 'x_data.py', 'x_stats.py', 'results.json', 'data/x_cache.npz')] + \
        [AG / 'integrate_o19' / 'results.json', AG / 'integrate_o20' / 'results.json', AG / 'o_turnover_fixedcost_diagnosis' / 'results.json']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
rows = []
for d in DIRS:
    if not d.exists(): rows.append([str(d.relative_to(AG)), 'MISSING_DIR', 0, 0]); continue
    for p in sorted(d.rglob('*')):
        if p.is_file():
            st = p.stat(); rows.append([str(p.relative_to(AG)), sha(p), st.st_size, st.st_mtime])
for p in FILES:
    st = p.stat(); rows.append([str(p.relative_to(AG)), sha(p), st.st_size, st.st_mtime])
tag = sys.argv[1]
out = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), n=len(rows),
           sha=hashlib.sha256(json.dumps([r[:3] for r in rows]).encode()).hexdigest(), rows=rows)
(HERE / f'ro_snapshot_{tag}.json').write_text(json.dumps(out, ensure_ascii=False, indent=0))
print(out['utc'], out['n'], out['sha'])
