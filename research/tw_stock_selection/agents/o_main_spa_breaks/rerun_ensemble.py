#!/usr/bin/env python3
"""缺淨值成員重跑：strategy_ensemble 的 5,076 組窮舉（子集 × 加權）只存了統計，沒存淨值。
只讀 import strategy_ensemble/elib.py（它會斷言自己的 PREREG），用同一個 simulate 重建每一組的日頻淨值，
在主策略訊號窗格點上取窗報酬，輸出 data/ensemble_ret.parquet。不寫入原資料夾。
核對：重建的 S_long|E1 全組合（9 條等權）與原檔 ensemble_nav.parquet 的 S_long|E1 比對最大差。"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fam_common as C
sys.path.insert(0, str(C.AG / 'strategy_ensemble'))
import elib as E            # noqa: E402（只讀 import）
sys.path.pop(0)
FIRST_ALLOC = {'S_long': '2006-05-11', 'S_recent': '2021-02-26'}   # 與 strategy_ensemble/run.py 相同

def main():
    t0 = time.time()
    ex = pd.read_csv(C.AG / 'strategy_ensemble' / 'exhaustive_all.csv.gz')
    out = {}; chk = {}
    for i, r in ex.iterrows():
        names = r['subset'].split('+'); m = r['w']
        nav = E.simulate(names, E.METHODS[m], FIRST_ALLOC[r['set']])
        key = f"strategy_ensemble/exhaustive::{r['set']}|{r['subset']}|{m}"
        out[key] = C.win_ret(nav)
        if r['set'] == 'S_long' and len(names) == 9 and m == 'E1': chk['S_long|E1'] = nav
        if i % 500 == 0: print(i, round(time.time() - t0, 1), flush=True)
    R = pd.DataFrame(out)
    orig = pd.read_parquet(C.AG / 'strategy_ensemble' / 'ensemble_nav.parquet')['S_long|E1']
    a = chk['S_long|E1'].dropna(); b = orig.reindex(a.index)
    d = float(np.nanmax(np.abs(a.to_numpy() / a.iloc[0] - b.to_numpy() / b.iloc[0])))
    print('check S_long|E1 max abs diff', d)
    assert d < 1e-9, '重建與原檔不符'
    R.to_parquet(C.DATA / 'ensemble_ret.parquet')
    print('done', R.shape, round(time.time() - t0, 1))

if __name__ == '__main__':
    main()
