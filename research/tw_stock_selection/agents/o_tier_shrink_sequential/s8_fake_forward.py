#!/usr/bin/env python3
"""步驟 6b：假前瞻序列（供 forward_monitor.py 測試與獨立複製逐位比對）。
產生 data/fake_forward.csv（line,start,end,ret_line,ret_0050；30 個月；固定種子；真實 α 依線設定，與歷史無關），
連跑兩次 forward_monitor.py（--forward），斷言輸出逐位相同，並記錄輸出 sha256 到 results.json 的 FAKE_FORWARD_TEST。
用法：python3 -I -B s8_fake_forward.py
"""
import sys, json, subprocess, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import tlib as T

LINES = {'L1': (-0.5, 5.3), 'L2_MAIN80_DISP20': (-0.4, 4.9), 'L3_MAIN70_E50_30': (-0.35, 3.7), 'L4_MAIN70_2330_30': (0.2, 3.5),
         'L5_CAPM_F': (0.1, 1.8), 'L6_CAPM_C50': (0.05, 0.9), 'L7_CAPM_C70': (0.03, 0.55), 'L8_REGIME': (0.3, 3.7),
         'L9_REGIME_CORE50': (0.15, 1.85), 'L10_REGIME_D': (0.2, 2.4)}

def main():
    T.check_block('PREREG_FORWARD', files=False)
    rng = np.random.default_rng(20261111)
    starts = pd.date_range('2026-11-11', periods=30, freq='MS') + pd.Timedelta(days=10)
    rows = []
    r50 = rng.normal(0.01, 0.05, 30)
    for ln, (a, s) in LINES.items():
        x = rng.normal(a, s, 30) / 100
        for i, d in enumerate(starts):
            rows.append(dict(line=ln, start=str(d.date()), end=str((starts[i + 1] if i + 1 < 30 else d + pd.Timedelta(days=31)).date()),
                             ret_line=round(float(r50[i] + x[i]), 10), ret_0050=round(float(r50[i]), 10)))
    F = pd.DataFrame(rows); fp = T.DATA / 'fake_forward.csv'; F.to_csv(fp, index=False)
    outs = []
    for k in range(2):
        o = T.DATA / f'fake_monitor_out_{k}.json'
        subprocess.run([sys.executable, '-I', '-B', str(HERE / 'forward_monitor.py'), '--forward', str(fp), '--out', str(o), '--tier', '100萬', '--mode', 'FIX'],
                       check=True, capture_output=True)
        outs.append(o.read_bytes())
    assert outs[0] == outs[1], '兩次輸出不同'
    h = hashlib.sha256(outs[0]).hexdigest()
    res = {'fake_forward_sha256': T.sha(fp), 'monitor_out_sha256': h, 'identical_two_runs': True, 'utc': T.now_utc(),
           'summary': {ln: {k: v for k, v in r.items() if k in ('n_months', 'post_P_alpha_gt_0', 'E_up', 'E_down', 'wealth_ratio_vs_0050', 'trigger_now', 'allocation')}
                       for ln, r in json.loads(outs[0])['lines'].items()}}
    T.save_key('FAKE_FORWARD_TEST', res)
    print(json.dumps(res, ensure_ascii=False, indent=1)[:4000])

if __name__ == '__main__':
    main()
