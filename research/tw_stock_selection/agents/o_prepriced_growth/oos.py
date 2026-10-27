#!/usr/bin/env python3
"""樣本外閘門：python3 -I -B oos.py

斷言 PREREG 存在且 sha256 與內容相符；閘門未開（挑選期 t < 1.5）時，只把拒絕紀錄寫進 results.json 的 OOS，
不載入任何 2014 年以後的報酬、不計算驗證期與保留期。閘門打開時才會進入樣本外計算（本輪未發生，故未實作計算段，
避免「先寫好再看要不要跑」的誘惑；若未來重做且閘門打開，需另行事前登記樣本外程式）。
"""
import json, hashlib, datetime, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert res.get('PREREG'), '先執行 prereg.py'
body = json.dumps(res['PREREG'], ensure_ascii=False, sort_keys=True)
assert hashlib.sha256(body.encode('utf-8')).hexdigest() == res['PREREG_sha256'], 'PREREG 內容被改過'
assert hashlib.sha256((HERE / 'iterations.json').read_bytes()).hexdigest() == res['iterations_json_sha256'], 'iterations.json 在凍結後被改過'
g = res['PREREG']['gate']
now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
if not g['open']:
    res['OOS'] = {'status': 'NOT_OPENED', 'reason': f"挑選期最佳版本 {res['PREREG']['selected_version']} 對同池等權 t {g['best_t']:.2f} < 1.5（P5 失敗條件）",
                  'verdict': '不可用', 'time_utc': now}
    (HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('閘門未開：不計算驗證期與保留期。', res['OOS'])
    sys.exit(0)
sys.exit('閘門打開：需另行事前登記樣本外程式（本輪未發生）')
