#!/usr/bin/env python3
"""正式資料凍結（PREREG 前）：data_freeze_v5.json＝底稿（V4 凍結檔＋scratchpad 面板）＋本題處理檔＋原始 clone 的 HEAD＋全部腳本與引擎庫的 sha256。
凍結後任何列出的檔案改變，run／jscan 會在 check_freeze5() 斷言失敗。"""
import json, hashlib, datetime, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
dr = json.loads((HERE / 'data_freeze_v5_draft.json').read_text(encoding='utf-8'))
files = {k: {'path': v['path'], 'sha256': v['sha256']} for k, v in dr['files'].items()}
for f in sorted((HERE / 'data').glob('*')):
    files[f'data/{f.name}'] = {'path': str(f), 'sha256': sha(f)}
for f in ('iterations.json', 'explore.json', 'explore_fv.json', 'sources_log.json', 'smoke.json', 'data_freeze_v5_draft.json'):
    files[f'topic/{f}'] = {'path': str(HERE / f), 'sha256': sha(HERE / f)}
scripts = sorted(HERE.glob('*.py')) + sorted((HERE / 'rep').glob('*.py'))
libs = [AG / 'o_v4_revenue_patch_official' / 'v4lib.py', AG / 'o_infra_v3_rebuild' / 'o3_core.py', AG / 'main_v2' / 'v2_core.py',
        AG / 'independent_audit' / 'ia_core.py', AG / 'o_disposal_rejudge' / 'drj_core.py', AG / 'o_stagger_tranche' / 'st_core.py',
        AG / 'exec_rebuild' / 'exlib.py', AG / 'o_v4_revenue_patch_official' / 'replicate_o4' / 'rcore.py']
for f in scripts + libs:
    files[f'script/{f.relative_to(AG)}'] = {'path': str(f), 'sha256': sha(f)}
other = {'raw_news_meta_v4': json.loads((AG / 'o_v4_revenue_patch_official' / 'raw_news_meta.json').read_text(encoding='utf-8')),
         'data_freeze_v4_sha256': dr['data_freeze_v4_sha256']}
heads = {}
for d in sorted((HERE / 'raw').iterdir()):
    try:
        heads[d.name] = {'head': subprocess.check_output(['git', '-C', str(d), 'rev-parse', 'HEAD'], text=True).strip(),
                         'origin': subprocess.check_output(['git', '-C', str(d), 'remote', 'get-url', 'origin'], text=True).strip()}
    except Exception as e: heads[d.name] = str(e)
out = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), 'files': files, 'raw_clones': heads, 'other': other,
       'note': '日線面板 8 檔在 scratchpad（研究中心共用、restore_env.sh 可重建），只登記 sha；本題自行產生的處理資料全部在 data/。'}
f = HERE / 'data_freeze_v5.json'
assert not f.exists(), 'data_freeze_v5.json 已存在；凍結後改檔須重新登記（另存新版）'
f.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(len(files), 'files frozen', sha(f))
