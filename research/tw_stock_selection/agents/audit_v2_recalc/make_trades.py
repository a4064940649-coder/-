#!/usr/bin/env python3
"""數據稽核部 2026-11-21：由 S75 V_BRIDGE 交易明細＋淨值重產「交易紀錄_2008_2026」csv／xlsx（格式照 xq_scripts/ 原檔；原產生程式已不在倉庫）。
用法：python3 -I -B make_trades.py <trades.csv.gz> <nav.csv.gz> <輸出資料夾> [--check-old]
  --check-old：用 S75 原檔重產後，逐格和 xq_scripts/ 原 csv、xlsx 四張表比對（確認我的寫法＝原檔的產生方式），不符就結束碼非 0。
名稱欄：沿用原檔的「代號→名稱」對照（原檔說明：取自公開財報檔的公司名稱，查不到留空白）；原檔沒有的新代號留空白並列出。
xlsx：以原檔為樣板（欄寬、格式、凍結窗格、篩選不變），只換儲存格的值；「說明」表只換筆數那一列與「狀態」那一列。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import openpyxl

tr_p, nav_p, out_dir = sys.argv[1:4]
CHECK = '--check-old' in sys.argv
OUT = Path(out_dir); OUT.mkdir(parents=True, exist_ok=True)
XQ = Path('/home/user/-/xq_scripts')
OLD_CSV, OLD_XLSX = XQ / '交易紀錄_2008_2026.csv', XQ / '交易紀錄_2008_2026.xlsx'
COLS = ['訊號日', '買進日', '賣出日', '代號', '名稱', '股數', '買進均價', '賣出均價', '買進金額（含成本）', '賣出金額（扣成本）', '損益金額', '報酬率%', '持有交易日', '買進時營收驚奇分數', '狀態']

old = pd.read_csv(OLD_CSV, dtype={'代號': str}, encoding='utf-8-sig', keep_default_na=False)
NAME = dict(zip(old['代號'], old['名稱']))
t = pd.read_csv(tr_p, dtype={'code': str}); t = t[t['version'] == 'V_BRIDGE'].sort_values(['ent_date', 'code']).reset_index(drop=True)
nav = pd.read_csv(nav_p, index_col=0, parse_dates=True)
END = str(nav.index[-1].date())
r0 = lambda x: int(np.floor(x + 0.5)) if x >= 0 else -int(np.floor(-x + 0.5))
d = pd.DataFrame({
    '訊號日': t['sig_date'], '買進日': t['ent_date'], '賣出日': t['exit_date'], '代號': t['code'], '名稱': [NAME.get(c, '') for c in t['code']],
    '股數': t['shares'].astype(int), '買進均價': (t['cost'].round(0) / t['shares']).round(2), '賣出均價': (t['proceeds'].round(0) / t['shares']).round(2),   # 原檔＝四捨五入到元的金額 ÷ 股數（逐格驗過）
    '買進金額（含成本）': t['cost'].round(0).astype(int), '賣出金額（扣成本）': t['proceeds'].round(0).astype(int), '損益金額': t['pnl'].round(0).astype(int),
    '報酬率%': (t['ret'] * 100).round(2), '持有交易日': t['hold_days'].astype(int), '買進時營收驚奇分數': t['score'].round(2),
    '狀態': np.where(t['why'] == 'open_end', f'期末仍持有（以 {END} 收盤計）', '已賣出')})[COLS]
missing_names = sorted(set(t['code']) - set(NAME))
d.to_csv(OUT / '交易紀錄_2008_2026.csv', index=False, encoding='utf-8-sig')

# ---- 逐年交易統計（依買進日的年份；含期末仍持有）
y = d.assign(年=d['買進日'].str[:4])
ys = []
for k, g in y.groupby('年'):
    ys.append([k, len(g), int((g['報酬率%'] > 0).sum()), round(float((g['報酬率%'] > 0).mean() * 100), 1), round(float(g['報酬率%'].mean()), 2), float(g['報酬率%'].max()),
               float(g['報酬率%'].min()), int(g['損益金額'].sum()), float(np.round(g['持有交易日'].mean(), 1))])   # 原檔用 numpy 的四捨五入（33.45→33.4，逐格驗過）
# ---- 逐年帳戶報酬（年底 ÷ 前一年底 − 1；第一年以 100 萬起算）
ya = []
last = nav.groupby(nav.index.year).last(); pb, pe = 1e6, 1e6
for yr in last.index:
    a, b = last.loc[yr, 'V_BRIDGE'] / pb - 1, last.loc[yr, '0050'] / pe - 1
    lab = str(yr) if yr != last.index[-1] else f'{yr}（到 {nav.index[-1].month}/{nav.index[-1].day}）'
    ya.append([lab, round(a * 100, 2), round(b * 100, 2), round((a - b) * 100, 2)])
    pb, pe = last.loc[yr, 'V_BRIDGE'], last.loc[yr, '0050']

wb = openpyxl.load_workbook(OLD_XLSX)
ws = wb['交易明細']
assert ws.max_row - 1 >= 1
tmpl = [ws.cell(2, c)._style for c in range(1, 16)]
if ws.max_row > len(d) + 1:
    ws.delete_rows(len(d) + 2, ws.max_row - len(d) - 1)
for i, row in enumerate(d.itertuples(index=False), start=2):
    for c, v in enumerate(row, start=1):
        cell = ws.cell(i, c); cell.value = v.item() if hasattr(v, 'item') else v; cell._style = tmpl[c - 1]
ws.auto_filter.ref = f'A1:O{len(d) + 1}'
for name, rows in (('逐年交易統計', ys), ('逐年帳戶報酬', ya)):
    w = wb[name]
    assert w.max_row - 1 == len(rows), (name, w.max_row, len(rows))
    for i, row in enumerate(rows, start=2):
        for c, v in enumerate(row, start=1):
            w.cell(i, c).value = v
n_open = int((t['why'] == 'open_end').sum())
w = wb['說明']
line3 = f'共 {len(d)} 筆買進：{len(d) - n_open} 筆已賣出、{n_open} 筆到 {END} 仍持有（以當天收盤計）。'
assert w.cell(3, 1).value.startswith('共 ') and w.cell(10, 1).value.startswith('狀態：')
w.cell(3, 1).value = line3
if not CHECK:
    w.cell(1, 1).value = w.cell(1, 1).value + '——2026-11-21 新口徑（0050 改用官方含息序列、減資恢復買賣日用參考價）'
    w.cell(10, 1).value = '狀態：2026-11-21 起改用新口徑重算（0050 自 2015 年起改用官方含息序列、減資恢復買賣日改用參考價）。數據稽核部已用同一套引擎獨立重跑，逐筆、逐日與工程部結果相符；引擎本身與 2015 年以前的原始資料沒有另外重驗。'
wb.save(OUT / '交易紀錄_2008_2026.xlsx')
info = dict(rows=len(d), closed=len(d) - n_open, open_end=n_open, end=END, missing_names=missing_names, yearly_trades=ys, yearly_account=ya)

if CHECK:
    new = pd.read_csv(OUT / '交易紀錄_2008_2026.csv', dtype={'代號': str}, encoding='utf-8-sig', keep_default_na=False)
    ok_csv = new.shape == old.shape and bool((new.astype(str).to_numpy() == old.astype(str).to_numpy()).all())
    bytes_same = (OUT / '交易紀錄_2008_2026.csv').read_bytes() == OLD_CSV.read_bytes()
    a = openpyxl.load_workbook(OLD_XLSX); b = openpyxl.load_workbook(OUT / '交易紀錄_2008_2026.xlsx')
    diffs = []
    for s in a.sheetnames:
        ra, rb = list(a[s].iter_rows(values_only=True)), list(b[s].iter_rows(values_only=True))
        if len(ra) != len(rb):
            diffs.append((s, 'rows', len(ra), len(rb))); continue
        for i, (x, z) in enumerate(zip(ra, rb)):
            for j, (p, q) in enumerate(zip(x, z)):
                if p != q and not (isinstance(p, (int, float)) and isinstance(q, (int, float)) and abs(p - q) < 1e-9):
                    diffs.append((s, i + 1, j + 1, p, q))
    info['check_old'] = dict(csv_cells_equal=ok_csv, csv_bytes_equal=bytes_same, xlsx_cell_diffs=len(diffs), first=diffs[:10])
    print(json.dumps(info['check_old'], ensure_ascii=False, default=str))
    json.dump(info, open(OUT / 'make_trades_info.json', 'w'), ensure_ascii=False, indent=1, default=str)
    sys.exit(0 if (ok_csv and not diffs) else 1)
json.dump(info, open(OUT / 'make_trades_info.json', 'w'), ensure_ascii=False, indent=1, default=str)
print('ok', len(d), n_open, missing_names)
