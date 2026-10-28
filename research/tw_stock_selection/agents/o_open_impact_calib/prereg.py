#!/usr/bin/env python3
"""步驟 4：凍結（PREREG）。把 frozen_spec.txt、選定模型、判定規則與全部腳本／資料的 sha256 寫進 results.json。
雜湊涵蓋：本題全部 .py（含 replicate/、ecore_o5.py、xlib_o5.py、review_exec_o5_ref.py）、explore.json、iterations.json、frozen_spec.txt、
data/ 全部檔案（容量回測輸出 cap_* 除外，它們在 PREREG 之後才產生）、raw/ 全部檔案。
用法：python3 -I -B prereg.py
"""
import json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
RES = HERE / 'results.json'
res = json.loads(RES.read_text(encoding='utf-8')) if RES.exists() else {}
assert 'PREREG' not in res, 'PREREG 已存在'
files = sorted([p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'results.json'
                and not (p.parent == HERE / 'data' and p.name.startswith('cap_'))
                and not (p.parent == HERE / 'replicate' and p.suffix == '.json')])
fh = {str(p.relative_to(HERE)): sha(p) for p in files}
IT = json.load(open(HERE / 'iterations.json')); MC = json.load(open(HERE / 'data' / 'model_calib.json'))
P = dict(
    timestamp_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    frozen_spec=(HERE / 'frozen_spec.txt').read_text(encoding='utf-8'),
    selected=IT['selected'], model_calib_sha256=sha(HERE / 'data' / 'model_calib.json'),
    n_versions=dict(impact_models=7, open_volume_methods=3, crowding_tests=3, rounds=3),
    seen_before_prereg=['check_engine.py：impact=O5 的 o5 主格重現與 o5 成本審查 ALL 摩擦（已發表數字），不涉及本題校準模型',
                        'run_cap.py／analyze.py 以 CAP_DRYRUN=1 冒煙（2 seeds、30 萬與 1,000 萬），只確認流程不報錯，輸出不印不存於研究資料夾',
                        'replicate/rep_kappa.py：κ 獨立複製（校準層，不涉及策略報酬）'],
    file_sha256=fh)
canon = json.dumps(P, ensure_ascii=False, sort_keys=True).encode()
res['PREREG'] = P; res['PREREG_sha256'] = hashlib.sha256(canon).hexdigest()
RES.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print(res['PREREG_sha256'], P['timestamp_utc'], len(fh))
