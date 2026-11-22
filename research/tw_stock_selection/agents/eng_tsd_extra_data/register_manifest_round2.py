import sys, json
sys.path.insert(0, sys.argv[1])
import update_daily as u
out = sys.argv[1]; stage = sys.argv[2]
old = json.load(open(out + '/manifest.json', encoding='utf-8'))['datasets']
import os
res = {}
for spec in u.DATASETS:
    n = spec['name']
    if n not in old and os.path.isdir(os.path.join(out, n)):
        res[n] = dict(status='已更新', last_change_commit=sys.argv[3], last_change_commit_time=sys.argv[4])
u.write_manifest(out, res); print(sorted(res))
