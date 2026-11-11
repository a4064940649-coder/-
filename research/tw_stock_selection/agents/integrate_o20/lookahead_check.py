#!/usr/bin/env python3
"""integrate_o20 D1 前視計數（PREREG D1_lookahead_check）：基準遮罩在每個交易日 d（2006～2026）排除的股票中，有多少列的官方公告日 > d。
檢查三個來源：(a) ≤ 2021-09-23 disposal_official_universe 涵蓋 d 或 e（紙上 M0／M1 與 V7X dm_all 共用）；(b) ia_core disposal_5y 涵蓋 d（紙上 M0／M1）；
(c) V7X disp_mask_fn(trade) 本身已要求公告 ≤ d（構造上 0，仍列出）。公告日以 universe_announce（ordr 對齊）與 disposal_5y.date 為準。
python3 -I -B lookahead_check.py"""
import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert hashlib.sha256(json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256']
U = pd.read_csv(AG / 'o_disposal_list_pit_audit_pkl18/data/universe_announce.csv', dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
DU = pd.read_csv(AG / 'o_rev_avail_ds_pit_count/raw/disposal_official_universe.csv', dtype={'code': str}, parse_dates=['start', 'end'])
D5 = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
J = DU.merge(U[['code', 'ordr', 'start', 'announce']], on=['code', 'ordr', 'start'], how='left')
assert J.announce.notna().all() and len(J) == len(DU)
# 交易日曆：用處置表與 0050 以外的公開日曆不可靠 → 用 universe 本身的保守判定：公告日 > 前一個「日曆日中最近的週一～五」即可能前視
IDX = pd.bdate_range('2003-01-01', '2026-12-31')
pos = IDX.searchsorted(J.start.values)
prev_bd = IDX[np.maximum(pos - 1, 0)]
la_e = J[(J.announce.values > prev_bd.values) & (J.start <= pd.Timestamp('2021-09-24'))]
res = dict(
    a_DU_cover_e_announce_after_prev_weekday=int(len(la_e)), a_rows=la_e[['code', 'ordr', 'announce', 'start', 'end']].astype(str).values.tolist()[:20],
    a_DU_cover_d_announce_ge_start=int((J.announce >= J.start).sum()),
    b_D5_date_ge_start=int((D5.date >= D5.start).sum()),
    c_disp_mask_trade='構造上要求 announce ≤ d，前視列 0',
    note='涵蓋 d 的列公告日必 < 起日 ≤ d；涵蓋 e 而不涵蓋 d 的列起日＝e，公告日 < e。以週一～五近似交易日，公告日落在 e 前一個平日之後才可能晚於 d。')
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8')); assert 'LOOKAHEAD_CHECK' not in R; R['LOOKAHEAD_CHECK'] = res
(HERE / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(res, ensure_ascii=False))
