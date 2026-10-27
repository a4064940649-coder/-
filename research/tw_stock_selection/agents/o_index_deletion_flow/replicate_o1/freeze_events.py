"""步驟 2：確認事件表與凍結版一致（sha256），記錄複製員自己的凍結時間；此時尚未算任何報酬。"""
import sys, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rlib as L
h = L.sha(L.EVF)
rec = {'events_v2_sha256': h, 'matches_frozen': h == L.FROZEN_SHA,
       'replica_freeze_utc': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
       'returns_computed_before_freeze': False}
L.dump(rec, 'freeze.json'); print(rec)
assert rec['matches_frozen']
