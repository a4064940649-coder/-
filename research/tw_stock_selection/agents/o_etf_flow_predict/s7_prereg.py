"""凍結（PREREG）：本題在資料閘門結案，沒有可進樣本外的衛星。凍結的是 (1) P1 規則重建器（臺灣50 v4、0056 y4），
(2) 前瞻追蹤與重啟條件。寫入 results.json['PREREG']（spec 全文 sha256、程式 sha256、時間戳）。之後的前瞻程式（s6_forward.py）斷言 PREREG 存在。"""
import sys, json, hashlib, datetime as dt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eflib as E
SPEC = open(E.HERE / 'frozen_spec.txt', encoding='utf-8').read()
code = {f: hashlib.sha256((E.HERE / f).read_bytes()).hexdigest() for f in ('eflib.py', 's1_gate.py', 's2_tw50_p1.py', 's3_0056_p1.py', 's4_tw50_control.py', 's5_dose_overlap_mde.py', 's6_forward.py')}
pre = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), spec_sha256=hashlib.sha256(SPEC.encode()).hexdigest(), code_sha256=code,
           gate_sha256=json.load(open(E.HERE / 'gate.json'))['sha256'], spec=SPEC)
rp = E.HERE / 'results.json'
R = json.load(open(rp)) if rp.exists() else {}
assert 'PREREG' not in R, 'PREREG 已存在，不可覆寫'
R['PREREG'] = pre
json.dump(R, open(rp, 'w'), ensure_ascii=False, indent=1)
print(pre['created_utc'], pre['spec_sha256'])
