# 凍結、錨定、登錄簿、唯讀檢查的獨立核對（不執行研究員判定器）
import json, hashlib, os, subprocess, datetime as dt
H = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(H, '..')
AG = os.path.join(D, '..')
REPO = os.path.abspath(os.path.join(AG, '..', '..', '..'))
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
o = {}
for f in ('G8_PROTOCOL_v2.json', 'FREEZE_MANIFEST_v2.json', 'g8_evaluate_v2.py', 'g8lib2.py', 'calib_registry.jsonl', 'smoke/SMOKE_MANIFEST.json'):
    o[f] = sha(os.path.join(D, f))
man = json.load(open(os.path.join(D, 'FREEZE_MANIFEST_v2.json')))
pro = json.load(open(os.path.join(D, 'G8_PROTOCOL_v2.json')))
o['manifest_protocol_match'] = man['protocol_sha256'] == o['G8_PROTOCOL_v2.json']
o['manifest_evaluator_match'] = man['evaluator_sha256'] == o['g8_evaluate_v2.py']
o['manifest_g8lib2_match'] = man['g8lib2_sha256'] == o['g8lib2.py']
o['frozen_utc'] = pro['frozen_utc']
o['frozen_before_deadline'] = dt.datetime.fromisoformat(pro['frozen_utc']) < dt.datetime.fromisoformat(pro['deadline_utc'])
# 登錄簿 GENESIS
lines = [json.loads(l) for l in open(os.path.join(D, 'calib_registry.jsonl')) if l.strip()]
g = dict(lines[0]); e = g.pop('entry_sha')
o['genesis_entry_sha_ok'] = hashlib.sha256(json.dumps(g, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest() == e == man['registry_genesis_entry_sha']
o['genesis_prev_zero'] = lines[0]['prev_sha'] == '0' * 64
# 登記
R = json.load(open(os.path.join(D, 'results.json')))
for k in ('PREREG_DESIGN', 'PREREG'):
    h = hashlib.sha256(json.dumps(R[k], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    o[k] = dict(sha_ok=h == R[k + '_sha256'] == man['results_registrations'][k], utc=R[k + '_utc'])
# 煙霧測試
sm = json.load(open(os.path.join(D, 'smoke/SMOKE_MANIFEST.json')))
o['smoke_files_ok'] = all(sha(os.path.join(D, p)) == h for p, h in sm['files'].items())
o['smoke_n_py'] = sum(p.endswith('.py') for p in sm['files']); o['smoke_n_out'] = sum(p.endswith('.out') for p in sm['files'])
o['smoke_manifest_in_protocol'] = pro['smoke_manifest_sha256'] == o['smoke/SMOKE_MANIFEST.json']
# 外部錨定
bl = open(os.path.join(AG, 'BACKLOG.md'), encoding='utf-8').read()
o['backlog_has_manifest_sha'] = man and (o['FREEZE_MANIFEST_v2.json'] in bl)
gl = subprocess.run(['git', '-C', REPO, 'log', '--format=%H %cI', '--', os.path.relpath(os.path.join(D, 'FREEZE_MANIFEST_v2.json'), REPO)],
                    capture_output=True, text=True).stdout.strip()
o['git_commits_of_manifest'] = gl.splitlines()
problems = []
if not o['backlog_has_manifest_sha']: problems.append('ANCHOR_BACKLOG_MISSING')
if not gl: problems.append('ANCHOR_GIT_MISSING')
o['expected_official_label'] = 'INVALID' if problems else 'OK'
o['expected_problems'] = problems
# 唯讀檢查
files = []
for root in ('live_tracking/main_forward', 'o_forward_infra_live'):
    for dp, dn, fn in os.walk(os.path.join(AG, root)):
        for f in fn:
            files.append(os.path.relpath(os.path.join(dp, f), AG))
files.sort()
listing = ''.join(f"{sha(os.path.join(AG, f))}  {f}\n" for f in files)
o['ro_n_files'] = len(files); o['ro_listing_sha'] = hashlib.sha256(listing.encode()).hexdigest()
base = json.load(open(os.path.join(AG, 'o_g8_protocol_fin_erratum/results.json')))['READONLY_CHECK']['files']
now = {f: sha(os.path.join(AG, f)) for f in files}
changed = sorted(f for f in now if base.get(f) != now[f])
removed = sorted(f for f in base if f not in now)
o['ro_baseline_n'] = len(base)
o['ro_n_changed_or_new'] = len(changed); o['ro_n_new'] = sum(f not in base for f in changed); o['ro_n_removed'] = len(removed)
mt = [dt.datetime.fromtimestamp(os.path.getmtime(os.path.join(AG, f)), dt.timezone.utc) for f in changed]
o['ro_changed_mtime_range'] = [min(mt).isoformat(), max(mt).isoformat()] if mt else None
lo = dt.datetime(2026, 11, 5, 22, 51, 5, tzinfo=dt.timezone.utc); hi = dt.datetime(2026, 11, 5, 22, 51, 46, tzinfo=dt.timezone.utc)
o['ro_changed_outside_window'] = [f for f, m in zip(changed, mt) if not (lo <= m <= hi)]
json.dump(o, open(os.path.join(H, 'r5_integrity.json'), 'w'), ensure_ascii=False, indent=1)
print(json.dumps(o, ensure_ascii=False, indent=1))
