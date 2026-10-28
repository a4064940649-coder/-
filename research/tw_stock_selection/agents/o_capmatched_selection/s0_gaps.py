#!/usr/bin/env python3
"""步驟 0b：資料跳日清點（o5 教訓 1）與「已看過帳本」。不看任何策略報酬。寫 explore.json 的 STEP0。"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import cmlib as L

def runs_missing(have, cal, minlen=6):
    """cal 中不在 have 的連續區段（長度 ≥ minlen，即 > 5 個交易日）。"""
    hv = set(pd.DatetimeIndex(have)); out = []; cur = []
    for d in cal:
        if d in hv:
            if len(cur) >= minlen: out.append((str(cur[0].date()), str(cur[-1].date()), len(cur)))
            cur = []
        else: cur.append(d)
    if len(cur) >= minlen: out.append((str(cur[0].date()), str(cur[-1].date()), len(cur)))
    return out

cal = L.TCAL[(L.TCAL >= pd.Timestamp('2004-01-02')) & (L.TCAL <= pd.Timestamp('2026-10-22'))]
res = {}
res['panel_dates_missing_runs'] = runs_missing(L.IDX, cal)
lowcov = pd.Series(L.NOBS, index=L.IDX)
res['panel_low_coverage_days'] = {str(d.date()): int(v) for d, v in lowcov[lowcov < 0.9 * lowcov.rolling(21, center=True, min_periods=5).median()].items()}
res['E50_raw_missing_runs'] = runs_missing(L.E50_RAW.index, cal[cal >= L.E50_RAW.index.min()])
for f, cols in [('navs_v4.parquet', ['AB_V4|normal', 'AB_V4|EW', 'AB_V4_B10|normal']), ('navs_v6.parquet', ['AB_V6|normal']),
                ('integrate_o6_navs.parquet', ['M_E50_30|AB|normal', 'M_2330_30|AB|normal']), ('main_v2_navs_gapfix.parquet', ['EXFIN|n10|normal'])]:
    d = pd.read_parquet(L.RAW / 'navs' / f)
    for c in cols:
        s = d[c].dropna()
        res[f'{f}:{c}_missing_runs'] = runs_missing(s.index, cal[(cal >= s.index.min()) & (cal <= s.index.max())])
        # 淨值連續不變（停在同一值）> 5 日
        same = (s.diff() == 0).astype(int)
        grp = (same != same.shift()).cumsum(); ln = same.groupby(grp).sum()
        res[f'{f}:{c}_flat_runs_gt5'] = int((ln > 5).sum())
# 個股層級：股數錨（vf）與 2330
res['stock_2330_missing_runs'] = runs_missing(L.IDX[L.TRADED[:, L.J2330]], cal)
res['src3_period'] = ['2024-02-15', '2025-09-23', '來源 3（Yahoo 5 分 K 近似收盤 13:20）；敏感度：排除 src 3 期間的換股窗']

SEEN = {
    'o_capacity_tsmc_gap (o1)': '看過 1～50 名組 2019 年後（2019-04～2026-10）的報酬與「分數前 1/5 減組內等權」（驗證期 1～50 名組 t 2.0）',
    'o_largecap_leg (o3)': '看過成交值前 50 大池 2019～2026 的選股力（同一分數，t 0.62）與 70/30 合成',
    'integrate_o1／o2／o6': '看過 TSMC30（2330 30%＋主策略）與 0050 混合（M_E50_20/30/50）的 2019～2026 全段',
    'o_sbl_foreign_level (o2)': '看過 2021-10 以後的大型組（外資持股，近似 0050 內／外）',
    'main_v2／o4／o5／o6': '主策略 AB_V4、B10、V6 全期（含 2019～2026）；o6 看過 2019 年起對 0050 t 1.33',
    'etf_rebalance (r2)': '臺灣50 納入／刪除事件報酬（2015～2026）',
    '結論': '2019 年後的大型股報酬全部不是盲測。本題挑選期 2019-01～2022-12、驗證期 2023-01～2026-10 都已被前面各題部分看過；歷史判定上限＝待觀察；唯一乾淨樣本是 2026-11 起的前瞻登記。',
}
out = json.loads((L.HERE / 'explore.json').read_text(encoding='utf-8')) if (L.HERE / 'explore.json').exists() else {}
out['STEP0'] = {'gaps': res, 'seen_ledger': SEEN,
                'lessons_read': 'original_o1～o6 DIRECTOR_REPORT 5.3 共 22 點（o1 1～6、o2 1～9、o3 1～8、o4 1～10、o5 1～10、o6 11～22）'}
(L.HERE / 'explore.json').write_text(L.dumps(out), encoding='utf-8')
print(L.dumps(res)[:3000])
