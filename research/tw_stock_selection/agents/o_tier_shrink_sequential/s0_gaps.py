#!/usr/bin/env python3
"""步驟 0b：所有輸入序列的跳日與覆蓋（o5 教訓 1）。只讀 raw/。輸出 data/gaps.json（之後併入 explore.json）。
用法：python3 -I -B s0_gaps.py
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import tlib as T

def gaps(s, name, thr=5):
    s = s.dropna(); d = s.index
    out = []
    for a, b in zip(d[:-1], d[1:]):
        nb = int(np.busday_count(a.date(), b.date())) - 1
        if nb > thr:
            out.append({'series': name, 'from': str(a.date()), 'to': str(b.date()), 'missing_weekdays': nb})
    return out

def cover(s, name):
    s = s.dropna()
    return {'series': name, 'first': str(s.index[0].date()), 'last': str(s.index[-1].date()), 'n_days': int(len(s))}

def main():
    out = {'gaps_gt5_weekdays': [], 'coverage': [], 'notes': []}
    series = {}
    e50 = T.e50_nav(); series['0050'] = e50[e50.index >= '2006-04-01']
    V4 = pd.read_parquet(T.RAW / 'navs' / 'navs_v4.parquet')
    for c in ['AB_V4|normal', 'AB_V4_B10|normal', 'AB_V4|stress', 'AB_V4_B10|stress', 'AB_V4|EW']: series['V4:' + c] = V4[c]
    V6 = pd.read_parquet(T.RAW / 'navs' / 'navs_v6.parquet')
    for c in ['AB_V6|normal', 'AB_V6_B10|normal', 'AB_V6_B10|stress']: series['V6:' + c] = V6[c]
    JS = pd.read_parquet(T.RAW / 'navs' / 'navs_jscan_v6.parquet')
    for c in ['AB_V6_B10|normal|J0', 'AB_V6_B10|normal|J12', 'AB_V6|normal|J6']: series['JS:' + c] = JS[c]
    EX = pd.read_parquet(T.RAW / 'navs' / 'navs_abavg.parquet')
    for c in ['main|30萬|FIX|A', 'main|1000萬|GROW|A', 'stress|100萬|FIX|B']: series['EXEC:' + c] = EX[c]
    IO = pd.read_parquet(T.RAW / 'navs' / 'integrate_o6_navs.parquet')
    for c in ['M_E50_30|B10|normal', 'M_2330_30|B10|normal']: series['IO6:' + c] = IO[c]
    CM = pd.read_parquet(T.RAW / 'newstrat' / 'capmatched_navs_oos.parquet')
    for c in ['F1_K5_N20_sqrt|normal', 'C50', 'C70', 'REP']: series['CM:' + c] = CM[c]
    for k, s in series.items():
        out['gaps_gt5_weekdays'] += gaps(s, k); out['coverage'].append(cover(s, k))
    # 窗格點層級：每個序列在 245 窗上的缺值
    wcov = {}
    for k, s in series.items():
        r = T.win_ret(s)
        wcov[k] = {'n_win': int(r.notna().sum()), 'first_na': [str(i.date()) for i in r.index[r.isna()]][:5]}
    out['window_coverage'] = wcov
    W = pd.read_parquet(T.RAW / 'newstrat' / 'regime_W_full.parquet')
    out['regime_W_full'] = {'first': str(W.index[0].date()), 'last': str(W.index[-1].date()), 'n': int(len(W)),
                            'grid_overlap': int(len(set(W.index) & set(T.WIN))), 'grid_n': int(len(T.WIN))}
    TS = pd.read_parquet(T.RAW / 'spa' / 'tsmc_inputs.parquet')
    out['tsmc_inputs'] = {'n': int(len(TS)), 'idx_equals_grid': bool((pd.DatetimeIndex(TS.index) == T.WIN).all()),
                          'r2330_na': int(TS['r2330'].isna().sum())}
    X, M = T.load_family()
    D = pd.read_parquet(T.RAW / 'spa' / 'family_excess.parquet')[M.member_id.tolist()]
    out['family'] = {'n_members': int(X.shape[1]), 'n_windows': int(X.shape[0]), 'n_missing_cells_filled0': int(D.isna().sum().sum()),
                     'missing_in_first_window': int(D.iloc[0].isna().sum()), 'idx_equals_grid': bool((pd.DatetimeIndex(D.index) == T.WIN).all())}
    # 窗長
    wl = pd.Series([int(np.busday_count(a.date(), b.date())) for a, b in zip(T.GRID[:-1], T.GRID[1:])], index=T.WIN)
    out['long_windows_gt27_weekdays'] = [{'window_start': str(i.date()), 'weekdays': int(v)} for i, v in wl.items() if v > 27]
    out['notes'] = ['V4 系列 2023-12-29→2024-02-15 有 33 個工作天無資料（已知；V6 補上），V4 格點窗 2023-12-11 長 48 個工作天。',
                    'V6 格點比 V4 多 2024-01-11 一個訊號日；本題一律用 V4 格點（與族群相同），V6 序列以日頻淨值在 V4 格點取樣。',
                    '0050 的 >5 工作天缺口全部是農曆年休市。']
    (T.DATA / 'gaps.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    g = pd.DataFrame(out['gaps_gt5_weekdays'])
    print(g.groupby('series').size().to_string() if len(g) else 'no gaps')
    print(g[g.missing_weekdays > 8].to_string() if len(g) else '')
    print(json.dumps({k: out[k] for k in ('regime_W_full', 'tsmc_inputs', 'family', 'long_windows_gt27_weekdays')}, ensure_ascii=False))
    print({k: v['n_win'] for k, v in wcov.items()})

if __name__ == '__main__':
    main()
