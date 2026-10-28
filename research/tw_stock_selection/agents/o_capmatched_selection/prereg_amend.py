#!/usr/bin/env python3
"""凍結後的引擎修正登記（PREREG v2；o4 教訓 4：凍結後改引擎要重新登記）。
原因：第一次樣本外執行（results_oos_v1.json，已看過）發現 cmlib.simulate_w／cmexec 把「之後再也沒有成交」的持股（已下市成分：2311、2325、3474、2823、2888 等）
永久凍結在最後價值，不轉現金，使 2006 年起的連續淨值逐年累積死部位（複製 0050 在挑選期月均 −0.058pp，而 2019 年起新跑只有 −0.002pp）。
修正：換股日若凍結持股的最後成交日 < 當日，以最後收盤價值扣賣出成本轉現金再配置（o2 教訓 7）。選股規格、參數、版本、門檻全部不變；
挑選期重跑（iterations.json）選中版本不變（各指標最大差 0.02）。v1 樣本外結果保留在 results_oos_v1.json／navs_oos_v1.parquet 並在報告中揭露。"""
import json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
PR = R['PREREG']
it = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
assert it['round1']['chosen'] == PR['frozen_version']
skip = {'results.json'}
files = sorted([p for p in HERE.rglob('*') if p.is_file() and p.name not in skip and '__pycache__' not in p.parts
                and not p.name.startswith(('results_smoke', 'navs_smoke', 'results_oos', 'navs_oos', 'results_live_check', 'oos.log'))])
new = {str(p.relative_to(HERE)): sha(p) for p in files}
changed = sorted([f for f in new if PR['sha256_files'].get(f) != new[f]])
R['PREREG_AMEND'] = {'version': 2, 'reason': __doc__, 'changed_files': changed, 'sha256_files': new,
                     'frozen_version_unchanged': PR['frozen_version'],
                     'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                     'note': '修正是在看過 v1 樣本外結果之後做的；v1 的凍結版 VAL：+0.285pp／月、t 1.88（results_oos_v1.json）。兩版都報告。'}
R['PREREG_AMEND']['sha256'] = hashlib.sha256(json.dumps(R['PREREG_AMEND'], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
(HERE / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
print('AMEND', R['PREREG_AMEND']['sha256'], R['PREREG_AMEND']['timestamp_utc'], 'changed:', changed)
