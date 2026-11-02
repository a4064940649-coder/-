#!/usr/bin/env python3
"""步驟五：全期（含驗證期 2019-05～、保留期 2024～）一次跑完，凍結後不改規格。
斷言：results.json 有 PREREG、所有雜湊與 PREREG 相同、RESULTS_OOS 尚不存在（或指定 --resume 只補缺的 run）。
Runs（B10、j 0～12、seeds 0～19、32 格；R0 另跑官方日 40 seeds）：
  R0 原設定（＝選定版本解讀）→ 與 V7 原表對帳
  R1 ＋F1 → R2 ＋F1＋F2 → R3 ＋F1＋F2＋F3(0.5) → R4 ＋F1＋F2＋F3＋F4（＝V7X 正式）
  單項：S_F2、S_F3、S_F4（各自只開一項，對 R0）；敏感度：R4 的 f3＝0.25、1.0。"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import x_data as D, x_run as XR
import s4_prereg as PR

R = json.loads((HERE / 'results.json').read_text())
assert 'PREREG' in R, 'PREREG 不存在'
H = PR.hashes()
bad = [k for k, v in R['PREREG']['hashes'].items() if k not in ('s5_oos.py', 's6_summary.py', 's7_apply.py', 'iterations.json') and H.get(k) != v]
assert not bad, f'雜湊不符：{bad}'
base = dict(R['PREREG']['orig_cfg'])
RUNS = [('R0', {}), ('R1', dict(f1=True)), ('R2', dict(f1=True, f2=True)), ('R3', dict(f1=True, f2=True, f3=0.5)),
        ('R4', dict(f1=True, f2=True, f3=0.5, f4=True)), ('S_F2', dict(f2=True)), ('S_F3', dict(f3=0.5)), ('S_F4', dict(f4=True)),
        ('R4_z025', dict(f1=True, f2=True, f3=0.25, f4=True)), ('R4_z100', dict(f1=True, f2=True, f3=1.0, f4=True))]

if __name__ == '__main__':
    only = sys.argv[1:] or [r for r, _ in RUNS]
    for name, ex in RUNS:
        if name not in only: continue
        if (HERE / 'out' / f'navs_oos_{name}.parquet').exists(): print('skip', name); continue
        cfg = dict(base); cfg.update(ex)
        XR.run_grid('oos_' + name, js=range(13), seeds=range(20), extra=cfg)
        if name in ('R0', 'R4'):
            XR.run_grid('oos_' + name + '_OFF', seeds=range(40), extra=cfg, official=True)
