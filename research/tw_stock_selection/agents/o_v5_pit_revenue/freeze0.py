#!/usr/bin/env python3
"""步驟 0：資料凍結底稿。複製 data_freeze_v4.json 全部 sha256，並重新計算（斷言一致）；加上 scratchpad 面板檔的 sha。
輸出 data_freeze_v5_draft.json（之後 freeze_v5.py 會在 PREREG 前寫正式 data_freeze_v5.json）。"""
import json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
V4 = HERE.parent / 'o_v4_revenue_patch_official'
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
fz = json.loads((V4 / 'data_freeze_v4.json').read_text(encoding='utf-8'))
out = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
       'data_freeze_v4_sha256': sha(V4 / 'data_freeze_v4.json'), 'files': {}, 'mismatch': []}
for grp in ('v3_base', 'panel', 'other'):
    for k, v in fz[grp].items():
        h = sha(v['path']); out['files'][f'{grp}/{k}'] = {'path': v['path'], 'sha256': h, 'v4_sha256': v['sha256']}
        if h != v['sha256']: out['mismatch'].append(k)
for k in ('R_FIX', 'Y5_FIX'):
    f = V4 / fz[k]['file']; h = sha(f); out['files'][f'v4/{k}'] = {'path': str(f), 'sha256': h, 'v4_sha256': fz[k]['sha256']}
    if h != fz[k]['sha256']: out['mismatch'].append(k)
for k, v in fz['this_topic_files'].items():
    f = V4 / v['file']; h = sha(f); out['files'][f'v4/{k}'] = {'path': str(f), 'sha256': h, 'v4_sha256': v['sha256']}
    if h != v['sha256']: out['mismatch'].append(k)
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
for f in sorted(P.glob('*')):
    out['files'][f'scratch_panel/{f.name}'] = {'path': str(f), 'sha256': sha(f)}
assert not out['mismatch'], out['mismatch']
(HERE / 'data_freeze_v5_draft.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(len(out['files']), 'files; mismatch', out['mismatch'])
