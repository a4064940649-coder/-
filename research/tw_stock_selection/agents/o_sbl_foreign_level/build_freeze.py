#!/usr/bin/env python3
"""建立訊號面板並在看任何報酬之前凍結（sha256＋時間戳）。

輸出：
  fo_panel.parquet   外資持股比率（小數），index＝面板日、columns＝上市代碼；2025-10-07 前為回推值
  sbl_panel.parquet  借券賣出當日餘額（股），2026-07-30 起
  short_panel.parquet 融券今日餘額（張），2024-01 起
  shares_anchor.csv  2025-10-07 發行股數（市值近似用）
  data_freeze.json   上列與 raw/ 全部檔案的 sha256
"""
import sys, json, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import sbl_lib as L

H = L.HERE
P, flag = L.fo_panel()
P.astype('float32').to_parquet(H / 'fo_panel.parquet')

s = pd.read_parquet(L.RAW / 'bb_sbl_twt93u.parquet')
s['Date'] = pd.to_datetime(s.Date)
SB = s.pivot_table(index='Date', columns='Ticker', values='SBL_Balance', aggfunc='last').reindex(columns=L.COLS)
# 2026-07-30 的前日餘額 → 補 2026-07-29
first = s[s.Date == s.Date.min()].set_index('Ticker').SBL_Prev_Balance.reindex(L.COLS)
d0 = L.IDX[L.IDX < s.Date.min()][-1]
SB = pd.concat([pd.DataFrame([first.to_numpy()], index=[d0], columns=L.COLS), SB]).sort_index()
SB.to_parquet(H / 'sbl_panel.parquet')

m = pd.read_parquet(L.RAW / 'bb_margin.parquet')
m['Date'] = pd.to_datetime(m.Date)
SH = m.pivot_table(index='Date', columns='Ticker', values='Short_Balance', aggfunc='last').reindex(columns=L.COLS)
SH.to_parquet(H / 'short_panel.parquet')

R, S, Fs = L.load_qfiis()
S.loc[L.ANCHOR].reindex(L.COLS).rename('shares').to_csv(H / 'shares_anchor.csv')

fz = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
      'note': '在任何報酬計算之前建立；之後的程式一律檢查這些 sha256',
      'fo_panel_rows_with_values': int(P.notna().any(axis=1).sum()), 'fo_first_day': str(P.dropna(how='all').index.min().date()),
      'files': {}}
for f in sorted(list(L.RAW.glob('*')) + [H / 'fo_panel.parquet', H / 'sbl_panel.parquet', H / 'short_panel.parquet', H / 'shares_anchor.csv']):
    fz['files'][str(f.relative_to(H))] = L.sha(f)
(H / 'data_freeze.json').write_text(json.dumps(fz, ensure_ascii=False, indent=1), encoding='utf-8')
neg = (P < -0.005).sum().sum()
print(fz['created_utc'], fz['fo_first_day'], 'negative backcast cells(<-0.5pp):', int(neg), 'of', int(P.notna().sum().sum()))
