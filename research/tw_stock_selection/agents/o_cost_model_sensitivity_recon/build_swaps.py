#!/usr/bin/env python3
"""資料整理（不算統計量）：中心格成交日誌 → exit 層換股表（o20 B 配對；開盤價 edge、實際成交價 edge、含成本成交價 edge、rtc、ĉ、順延資訊），全部期間、6 格 × 3 變體。
重現檢查：100 萬一般 REAL_opt 的 edge_B／rtc_B 與 o_turnover data/swaps_exitlevel_100n_REAL_opt_seed0.csv.gz 逐列一致（|差| ≤ 1e−6）。
輸出：scratchpad/csr/swaps_center.parquet、data/swaps_center_seed0.csv.gz（關鍵中間檔）。
python3 -I -B build_swaps.py"""
import sys, json, pickle, time, hashlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import csr_lib as L; import diag as DG; sys.path.pop(0)
SCR = L.SCR; t0 = time.time()
M = pickle.load(open(SCR / 'center_meta.pkl', 'rb')); NC = pd.read_parquet(SCR / 'navs_center.parquet'); NC.index = DG.IDX
P0 = L.load_plans(); P0 = P0[P0.seed == 0]
SIG = {(v, j): dict(zip(g.exec_idx.astype(int), g.signal)) for (v, j), g in P0.groupby(['variant', 'j'])}
rows = []
for c in L.CELLS:
    F = pd.read_parquet(SCR / f'fills_center_{L.CELLKEY[c].replace("|", "_")}.parquet')
    F['variant'] = F.variant.astype(str)
    c0 = L.cash0(DG.E, c[1])
    for v in L.VARS:
        for j in range(13):
            key = f'{v}|{j}|{L.CELLKEY[c]}'
            f = DG.prep_fills(F, M['rebs'][key], v, j)
            rows += [dict(r, cell=L.CELLKEY[c]) for r in DG.swaps(f, M['rebs'][key], v, j, SIG[(v, j)], c0, NC[key], c[1])]
    print(c, len(rows), round(time.time() - t0), flush=True)
X = pd.DataFrame(rows)
X['date_e'] = DG.IDX[X.e.values]
# 重現 o20
O = pd.read_csv(L.TFC / 'data/swaps_exitlevel_100n_REAL_opt_seed0.csv.gz', dtype={'code': str})
m = X[(X.cell == 'normal|100萬|std') & (X.variant == 'REAL_opt')].merge(O[['j', 'k', 'code', 'edge_B', 'rtc_B', 'r', 'deferred']], on=['j', 'k', 'code'], suffixes=('', '_o20'), how='outer', indicator=True)
chk = dict(n_mine=int((m._merge != 'right_only').sum()), n_o20=int((m._merge != 'left_only').sum()), n_both=int((m._merge == 'both').sum()))
b = m[m._merge == 'both']
for k in ('edge_B', 'rtc_B', 'r', 'deferred'):
    d = (b[k] - b[k + '_o20']).abs(); d = d[np.isfinite(d)]
    chk['maxabs_' + k] = float(d.max())
chk['nan_mismatch_edge'] = int((np.isfinite(b.edge_B) != np.isfinite(b.edge_B_o20)).sum())
print(chk)
assert chk['n_mine'] == chk['n_o20'] == chk['n_both'] and chk['maxabs_edge_B'] <= 1e-6 and chk['maxabs_rtc_B'] <= 1e-6 and chk['nan_mismatch_edge'] == 0, chk
X.to_parquet(SCR / 'swaps_center.parquet')
f = HERE / 'data/swaps_center_seed0.csv.gz'
X.drop(columns=['date_e']).to_csv(f, index=False, compression='gzip', float_format='%.8g')
mf = HERE / 'build_swaps_meta.json'
mf.write_text(json.dumps(dict(utc=pd.Timestamp.now('UTC').isoformat(), runtime_s=round(time.time() - t0, 1), n=len(X), o20_check=chk,
                              file=str(f.relative_to(HERE)), sha256=hashlib.sha256(f.read_bytes()).hexdigest()), ensure_ascii=False, indent=1))
print('done', len(X), round(time.time() - t0))
