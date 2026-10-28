#!/usr/bin/env python3
"""族群矩陣定稿（PREREG 雜湊的對象）：
  data/family_excess.parquet   T×N：各成員窗報酬 − 0050 窗報酬（一般成本後）；NaN＝成員當時不存在
  data/family_excess_ew.parquet T×N：各成員窗報酬 − 主策略同池等權（AB_V4|EW，零成本）
  data/family_members.csv      成員清單：id、資料夾、F_main、F_long、各窗覆蓋
  data/synth_excess.parquet    合成成員（缺淨值者；只用於敏感度）
覆蓋規則（事前寫死）：成員在某窗內缺值的窗數 ≤ max(2, 2.5%×窗長) 才算「有覆蓋」，缺的窗以超額 0 填補
  （＝當時持有 0050）；F_long＝在 W_long 有覆蓋的成員。去重：與已收成員窗報酬完全相同者剔除。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fam_common as C
MAIN_ID = 'o_v4_revenue_patch_official/navs_v4.parquet::AB_V4|normal'

def coverage(x, idx, a, b):
    m = C.wmask(idx, a, b)
    n = int(m.sum()); miss = int((~np.isfinite(x[m])).sum())
    return miss <= max(2, int(0.025 * n)), miss, n

def main():
    F = pd.read_parquet(C.DATA / 'family_ret.parquet')
    E = pd.read_parquet(C.DATA / 'ensemble_ret.parquet')
    B = pd.read_parquet(C.DATA / 'bench.parquet')
    M = pd.read_csv(C.DATA / 'members_all.csv')
    meta = M[M.include == True].set_index('member_id')
    # 合併並去重（窗報酬完全相同）
    R = pd.concat([F, E], axis=1)
    X = R.to_numpy(); keep = []; seen = {}; dropped = {}
    for j, mid in enumerate(R.columns):
        x = X[:, j]; fin = np.isfinite(x)
        key = (int(fin.sum()), round(float(np.nansum(x)), 8))
        dup = None
        for k in seen.get(key, []):
            y = X[:, k]
            if np.array_equal(fin, np.isfinite(y)) and np.nanmax(np.abs(x - y)) < 1e-10: dup = R.columns[k]; break
        if dup is None: seen.setdefault(key, []).append(j); keep.append(mid)
        else: dropped[mid] = dup
    R = R[keep]
    idx = R.index
    D = R.sub(B['r0050'], axis=0)
    Dew = R.sub(B['rEW_ABV4'], axis=0)
    rows = []
    for mid in R.columns:
        x = D[mid].to_numpy()
        r = dict(member_id=mid, folder=mid.split('/')[0], F_main=bool(meta.loc[mid, 'F_main']) if mid in meta.index else False,
                 source='ensemble_rerun' if mid.startswith('strategy_ensemble/exhaustive') else 'nav_file')
        for w, (a, b) in C.WINDOWS.items():
            ok, miss, n = coverage(x, idx, a, b); r[f'cov_{w}'] = ok; r[f'miss_{w}'] = miss
        rows.append(r)
    MM = pd.DataFrame(rows)
    MM['F_long'] = MM['cov_W_long']
    assert MAIN_ID in set(MM.member_id)
    assert MM.set_index('member_id').loc[MAIN_ID, ['cov_W_long', 'cov_W_19', 'cov_W_21']].all()
    MM.loc[MM.member_id == MAIN_ID, 'F_main'] = True
    D.to_parquet(C.DATA / 'family_excess.parquet'); Dew.to_parquet(C.DATA / 'family_excess_ew.parquet')
    MM.to_csv(C.DATA / 'family_members.csv', index=False)
    pd.Series(dropped, name='duplicate_of').to_csv(C.DATA / 'family_dedup_dropped.csv')
    # 合成成員
    S = pd.read_csv(C.DATA / 'missing_members.csv')
    syn = {}
    for _, s in S.iterrows():
        base = D[s.similar_member].to_numpy().copy()
        out = np.full(len(base), np.nan)
        if s.span == 'long':
            segs = [C.wmask(idx, '2006-04-01', '2019-04-30'), C.wmask(idx, '2019-05-01', '2026-10-12')]
        else:
            segs = [C.wmask(idx, '2021-03-01', '2023-12-31'), C.wmask(idx, '2024-01-01', '2026-10-12')]
        for m, t in zip(segs, (s.t1, s.t2)):
            y = base[m]; f = np.isfinite(y)
            mu = t * y[f].std(ddof=1) / np.sqrt(f.sum())
            out[m] = y - y[f].mean() + mu
        syn[f'SYN::{s.id}'] = out
    SY = pd.DataFrame(syn, index=idx)
    SY.to_parquet(C.DATA / 'synth_excess.parquet')
    print('members', len(MM), 'dedup dropped', len(dropped), 'F_main', int(MM.F_main.sum()), 'F_long', int(MM.F_long.sum()),
          'cov W_19', int(MM.cov_W_19.sum()), 'cov W_21', int(MM.cov_W_21.sum()), 'synthetic', SY.shape[1])
    for w in C.WINDOWS:
        print(w, 'F_all', int(MM[f'cov_{w}'].sum()), 'F_main', int((MM[f'cov_{w}'] & MM.F_main).sum()), 'F_long', int((MM[f'cov_{w}'] & MM.F_long).sum()))

if __name__ == '__main__':
    main()
