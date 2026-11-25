#!/usr/bin/env python3
"""ENG-V3-SYNC 2026-11-26：用 S75 v3（漲跌幅照當時制度）的 V_BRIDGE 交易與淨值重產 xq_scripts/交易紀錄_2008_2026.csv／.xlsx。
寫法照 audit_v2_recalc/make_trades.py（該檔不改；它已逐格驗過與原檔的產生方式相同），差別：
  1. 交易讀 v3_2026-11-25_limit7/trades_v3.csv.gz、淨值讀 nav_2008_2026_v3.csv.gz；
  2. 樣板（欄寬、格式、凍結窗格、名稱對照）固定用 audit_v2_recalc/public/ 下的 v2 存檔，不讀 xq_scripts/ 的現行檔，所以可以重複執行；
     v2 檔沒出現過的代號（v3 只有 1233）名稱取自 o_concentration_regime/raw/stock_list.csv，查不到留空白；
  3. 「說明」分頁改寫標題、筆數、成交價、狀態四列，寫明 v3 口徑與更新日期；
  4. 寫檔前核對筆數、逐年帳戶報酬與 external_numbers_v3.json／yearly_v3.csv 相符，不符就報錯、不寫檔。
用法：python3 -I -B make_trades_v3.py [輸出資料夾]（預設 xq_scripts/）→ 另寫本資料夾 make_trades_v3_info.json"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import openpyxl
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
V3 = HERE.parent / 's75_continuous_2008_2026' / 'v3_2026-11-25_limit7'
TPL = HERE.parent / 'audit_v2_recalc' / 'public'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / 'xq_scripts'
OUT.mkdir(parents=True, exist_ok=True)
UPDATED = '2026-11-26'
COLS = ['訊號日', '買進日', '賣出日', '代號', '名稱', '股數', '買進均價', '賣出均價', '買進金額（含成本）', '賣出金額（扣成本）', '損益金額', '報酬率%', '持有交易日', '買進時營收驚奇分數', '狀態']

old = pd.read_csv(TPL / '交易紀錄_2008_2026.csv', dtype={'代號': str}, encoding='utf-8-sig', keep_default_na=False)
NAME = dict(zip(old['代號'], old['名稱']))
SL = pd.read_csv(HERE.parent / 'o_concentration_regime' / 'raw' / 'stock_list.csv', dtype=str)   # v2 檔沒出現過的代號才用這份補名稱（同為簡稱；v2 已有的 2031、2106 等寫法相同）
NAME2 = dict(zip(SL['code'], SL['name']))
t = pd.read_csv(V3 / 'trades_v3.csv.gz', dtype={'code': str}); t = t[t['version'] == 'V_BRIDGE'].sort_values(['ent_date', 'code']).reset_index(drop=True)
nav = pd.read_csv(V3 / 'nav_2008_2026_v3.csv.gz', index_col=0, parse_dates=True)
END = str(nav.index[-1].date())
d = pd.DataFrame({
    '訊號日': t['sig_date'], '買進日': t['ent_date'], '賣出日': t['exit_date'], '代號': t['code'], '名稱': [NAME[c] if c in NAME else NAME2.get(c, '') for c in t['code']],
    '股數': t['shares'].astype(int), '買進均價': (t['cost'].round(0) / t['shares']).round(2), '賣出均價': (t['proceeds'].round(0) / t['shares']).round(2),
    '買進金額（含成本）': t['cost'].round(0).astype(int), '賣出金額（扣成本）': t['proceeds'].round(0).astype(int), '損益金額': t['pnl'].round(0).astype(int),
    '報酬率%': (t['ret'] * 100).round(2), '持有交易日': t['hold_days'].astype(int), '買進時營收驚奇分數': t['score'].round(2),
    '狀態': np.where(t['why'] == 'open_end', f'期末仍持有（以 {END} 收盤計）', '已賣出')})[COLS]
codes_not_in_v2 = sorted(set(t['code']) - set(NAME))          # v2 檔沒出現過的代號（名稱留空白）
n_open = int((t['why'] == 'open_end').sum())

# ---- 逐年交易統計（依買進日的年份；含期末仍持有）
ys = []
for k, g in d.assign(年=d['買進日'].str[:4]).groupby('年'):
    ys.append([k, len(g), int((g['報酬率%'] > 0).sum()), round(float((g['報酬率%'] > 0).mean() * 100), 1), round(float(g['報酬率%'].mean()), 2), float(g['報酬率%'].max()),
               float(g['報酬率%'].min()), int(g['損益金額'].sum()), float(np.round(g['持有交易日'].mean(), 1))])
# ---- 逐年帳戶報酬（年底 ÷ 前一年底 − 1；第一年以 100 萬起算）
ya = []
last = nav.groupby(nav.index.year).last(); pb, pe = 1e6, 1e6
for yr in last.index:
    a, b = last.loc[yr, 'V_BRIDGE'] / pb - 1, last.loc[yr, '0050'] / pe - 1
    lab = str(yr) if yr != last.index[-1] else f'{yr}（到 {nav.index[-1].month}/{nav.index[-1].day}）'
    ya.append([lab, round(a * 100, 2), round(b * 100, 2), round((a - b) * 100, 2)])
    pb, pe = last.loc[yr, 'V_BRIDGE'], last.loc[yr, '0050']

# ---- 核對（不符就不寫檔）
X = json.loads((V3 / 'external_numbers_v3.json').read_text(encoding='utf-8'))
xb = {r['項目']: r['v3新值'] for r in X['V_BRIDGE']}
assert len(d) == xb['交易筆數（全部，含期末未平倉）'] == 627 and len(d) - n_open == xb['交易筆數（已出場）'] == 619, (len(d), n_open)
yv = pd.read_csv(V3 / 'yearly_v3.csv')
assert [int(r[0][:4]) for r in ya] == list(yv['year'])
dev = max(max(abs(r[1] - v * 100), abs(r[2] - e * 100)) for r, v, e in zip(ya, yv['V_BRIDGE'], yv['e0050']))
assert dev < 0.005 + 1e-9, dev                              # 只差四捨五入到小數 2 位
assert [r[1] for r in ya] == [r['新'] for r in X['逐年_V_BRIDGE']] and [r[2] for r in ya] == [r['e0050'] for r in X['逐年_V_BRIDGE']]
assert round(float((d['報酬率%'] > 0).mean() * 100), 4) == xb['賺錢比例（全部交易，文件「47.8%」的口徑）']
assert abs(float(t['pnl'].sum()) - xb['選股交易損益合計']) < 1

d.to_csv(OUT / '交易紀錄_2008_2026.csv', index=False, encoding='utf-8-sig')
wb = openpyxl.load_workbook(TPL / '交易紀錄_2008_2026.xlsx')
ws = wb['交易明細']
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
w = wb['說明']
BASE = '月營收驚奇策略：2008-01～2026-10 回測交易紀錄（研究中心 S75，連續帳戶，100 萬起算）'
assert w.max_row == 11 and w.cell(1, 1).value.startswith(BASE) and w.cell(3, 1).value.startswith('共 ') and w.cell(6, 1).value.startswith('成交價：') and w.cell(10, 1).value.startswith('狀態：')
w.cell(1, 1).value = BASE + f'——v3 口徑（{UPDATED} 更新：漲跌停改照當時制度；0050 官方含息序列、減資恢復買賣日用參考價不變）'
w.cell(3, 1).value = f'共 {len(d)} 筆買進：{len(d) - n_open} 筆已賣出、{n_open} 筆到 {END} 仍持有（以當天收盤計）。'
w.cell(6, 1).value = ('成交價：訊號日隔天開盤買進／賣出。開盤就鎖漲停的當天不買、也不遞補；開盤鎖跌停的延到賣得掉那天。'
                      '漲跌停照當時制度：2015-06-01 以前 7%、2008-10-13～10-24 跌幅 3.5%、2015-06-01 起 10%。2014 年以前除權息日的報酬以 0 計（偏保守）。')
w.cell(10, 1).value = (f'狀態：{UPDATED} 更新為 v3 口徑（2026-11-25 起採用）。和上一版（v2，2026-11-21）的差別只有一項：2015-06-01 以前的漲跌停由 10% 改回當時的 7%'
                       '（2008-10-13～10-24 跌幅 3.5%），所以早年開盤就鎖漲停的幾筆不再算買得到，筆數 632 → 627，之後帳戶的資金與股數跟著小幅變動。'
                       '7% 的部分已由數據稽核部獨立重算（R97），這條路徑逐日、逐筆相符；引擎本身與 2015 年以前的原始資料沒有另外重驗。'
                       '出處：agents/s97_limit_lock_execution、agents/r97_replicate_limit7、agents/s75_continuous_2008_2026/v3_2026-11-25_limit7。')
wb.save(OUT / '交易紀錄_2008_2026.xlsx')
dropped = sorted(set(zip(old['買進日'], old['代號'])) - set(zip(d['買進日'], d['代號'])))
added = sorted(set(zip(d['買進日'], d['代號'])) - set(zip(old['買進日'], old['代號'])))
info = dict(source='v3_2026-11-25_limit7/trades_v3.csv.gz（V_BRIDGE）、nav_2008_2026_v3.csv.gz', updated=UPDATED, rows=len(d), closed=len(d) - n_open, open_end=n_open, end=END,
            codes_not_in_v2={c: NAME2.get(c, '') for c in codes_not_in_v2}, v2_trades_not_in_v3=dropped, v3_trades_not_in_v2=added, yearly_vs_yearly_v3_max_abs_dev_pp=dev, yearly_trades=ys, yearly_account=ya)
json.dump(info, open(HERE / 'make_trades_v3_info.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print('ok', len(d), len(d) - n_open, n_open, 'codes_not_in_v2', codes_not_in_v2, 'v2 有 v3 沒有', dropped, 'v3 有 v2 沒有', added)
