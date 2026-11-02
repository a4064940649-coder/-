#!/usr/bin/env python3
"""截斷重播（A1／A4 驗收用）：依每日排程（每天台北 06:49，asof＝前一個交易日）逐日執行 make_forward.py --replay。
用法：python3 -I -B replay.py --out replay/seq --from 2026-09-10 --to 2026-11-02 [--recheck 5]"""
import argparse, json, subprocess, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fwd_data as FD  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=str(HERE / 'replay/seq'))
ap.add_argument('--from', dest='d0', default='2026-09-10')
ap.add_argument('--to', dest='d1', default='2026-11-02')
ap.add_argument('--start', default='2026-09')
ap.add_argument('--disp-start', default='2026-09-01')
ap.add_argument('--recheck', type=int, default=5)
a = ap.parse_args()
src = FD.Source()
days = src.hcal[(src.hcal >= a.d0) & (src.hcal <= a.d1)]
log = Path(a.out).parent / (Path(a.out).name + '_driver.jsonl')
for d in days:
    clock = (pd.Timestamp(d.date()) + pd.Timedelta(days=1, hours=6, minutes=49)).strftime('%Y-%m-%dT%H:%M+08:00')
    cmd = [sys.executable, '-I', '-B', str(HERE / 'make_forward.py'), '--replay', '--sim-clock', clock, '--asof', str(d.date()),
           '--start', a.start, '--disp-start', a.disp_start, '--out', a.out, '--recheck', str(a.recheck)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    last = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ''
    rec = dict(asof=str(d.date()), clock=clock, rc=p.returncode, err=p.stderr[-2000:] if p.returncode else '')
    try:
        j = json.loads(last); rec.update(frozen_now=j.get('frozen_now'), n_list_diffs=j.get('n_list_diffs'), status=j.get('status'))
    except Exception:
        rec['stdout_tail'] = p.stdout[-500:]
    with open(log, 'a', encoding='utf-8') as fh: fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    if p.returncode: sys.exit(p.returncode)
