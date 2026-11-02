#!/usr/bin/env python3
"""第 4 步：凍結。把 frozen_spec.txt、全部程式、explore.json、iterations.json、事件表與面板的 sha256 寫進 results.json 的 PREREG。
只能執行一次（已存在就拒絕覆寫）。"""
import json, sys, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from pcore import HERE, DATA, sha

rp = HERE / 'results.json'
R = json.load(open(rp)) if rp.exists() else {}
assert 'PREREG' not in R, 'PREREG 已存在，不可覆寫'
files = sorted(glob.glob(str(HERE / '*.py'))) + [str(HERE / 'frozen_spec.txt'), str(HERE / 'explore.json'), str(HERE / 'iterations.json')]
dfiles = ['events.parquet', 'avail_dates.parquet', 'fin_long.parquet', 'rev_long.parquet', 'quality.json', 'predef.json', 'panel_info.json']
H = {Path(f).name: sha(f) for f in files}
H.update({'data/' + f: sha(DATA / f) for f in dfiles})
predictions = {
    'MAIN_100w_GROW_normal_oos_ex_ann_jmed': '事前預測 +1%～+5%／年（挑選期 +7.2% 打五折以下），月 t 0.3～1.5；G2 不會過（全期 t < 3）',
    'R1_lowatt': '低關注版樣本外仍不贏 MAIN（挑選期 −1.5%／年）；P1 組間差 t < 2（挑選期 0.06）→ 預期「機制的低關注部分不支持」',
    'j_decay': '樣本外 j9 的異常報酬約為 j0 的 1/3～1/2（挑選期 0.75% vs 2.2%）',
    'P2_FM_SUEP_t': '樣本外 ≥ 2（挑選期 4.6）',
    'G7': '1,000 萬參與率中位數 > 20% 上限，預期超額明顯縮小甚至 < 0',
    'C50_vs_main': '配對 t < 1.5（PEAD 單腿弱於主策略）；C50 對 0050 t 可能因波動下降而接近主策略',
}
R['PREREG'] = {'utc': pd.Timestamp.utcnow().isoformat(), 'frozen_spec': (HERE / 'frozen_spec.txt').read_text(), 'hashes': H,
               'predictions': predictions, 'choice': 'v3', 'report_versions': ['v1', 'v8'],
               'n_selection_variants': 8, 'n_neighbor_runs': 3}
R['PREREG']['sha256'] = __import__('hashlib').sha256(json.dumps({k: R['PREREG'][k] for k in ['utc', 'frozen_spec', 'hashes', 'predictions']},
                                                                  sort_keys=True, ensure_ascii=False).encode()).hexdigest()
json.dump(R, open(rp, 'w'), ensure_ascii=False, indent=1)
print('PREREG', R['PREREG']['sha256'], R['PREREG']['utc'])
