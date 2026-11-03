#!/usr/bin/env python3
"""第 0 步（看任何報酬之前）：原始資料複製到 raw/（附 sha256 manifest）＋PIT 產業表 raw/industry_pit.csv。
產業表規則（事前寫死）：
  觀測：證交所產業別公開快照 2015-05-22、2016-04-01、2017-07-06（weihautin 封存 StockList）、2023-01-16（證交所 ISIN 檔，V7 raw 封存）、
        2026-10（tw-stock-screener stock_list.csv）。只取上市股。
  逐段生效：兩次觀測不同 → 新類別自較晚觀測日起生效（當時可得，偏保守）；例外：2023-01-16 → 現行 的差異視為 2023-07-03 產業別調整生效。
  第一次觀測以前：沿用最早觀測（2015-05 以前無公開快照，標「回溯」）。2007-07-02 以前的電子 8 類合併在 lclib.ind_at 處理。
  沒有任何快照的下市股：依公司名稱與主業人工標註（標「人工推定」，未能線上核對證交所舊頁）。
  觀光事業→觀光餐旅（2023-07 更名）一律寫成觀光餐旅。"""
import json, shutil, hashlib
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
RAW = HERE / 'raw'; RAW.mkdir(exist_ok=True)
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()

W = AG / 'o_0050_tilt_pit' / 'raw' / 'weihautin' / 'update_csv'
SRC = {
    'snap/StockList_20150522.csv': W / 'StockList_20150522.csv',
    'snap/StockList_20160401.csv': W / 'StockList.csv-old',
    'snap/StockList_20170706.csv': W / 'StockList_20170706.csv',
    'snap/stock_list_202610.csv': AG / 'o_0050_tilt_pit' / 'raw' / 'screener' / 'stock_list.csv',
    'delist_official.csv': AG / 'o_0050_tilt_pit' / 'raw' / 'agents' / 'delist_official.csv',
    'bench/etf_0050.csv': Path('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv'),
    'bench/taiex_twii.csv': AG / 'o_0050_tilt_pit' / 'raw' / 'bench' / 'taiex_twii.csv',
    'listing_master.csv': AG / 'o_v7_listing_exec_engine' / 'listing_master.csv',
}
ISIN = AG / 'o_v7_listing_exec_engine' / 'raw' / 'isin' / 'snap' / 'isin_02_64993c2e1699ae56b9752b47e87f90721be5fd50.txt'
man = {'note': '下載／外部資料一律視為不可信，只讀。', 'files': {}}
for k, p in SRC.items():
    q = RAW / k; q.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, q)
    man['files'][k] = dict(src=str(p), sha256=sha(q), bytes=q.stat().st_size)
# ISIN 2023-01-16：只抽上市股 code,name,industry（原檔 8MB，記 sha）
d = json.loads(ISIN.read_text(encoding='utf-8')); rows = []
for g in d['list']:
    for x in g['list']:
        if x.get('market') == '上市' and x.get('industry'):
            rows.append((x['code'], x['name'], x['industry']))
q = RAW / 'snap' / 'isin_20230116_listed.csv'
pd.DataFrame(rows, columns=['code', 'name', 'industry']).to_csv(q, index=False)
man['files']['snap/isin_20230116_listed.csv'] = dict(src=str(ISIN), src_sha256=sha(ISIN), sha256=sha(q), note='ISIN 2023/01/16 快照（o_lc_sector_strong_stocks 同口徑認定日期）')
# twsec 年度 EPS／純益率（D1b 診斷用）
TW = AG / 'o_0050_tilt_pit' / 'raw' / 'twsec' / 'yearly_earning_raw_data'
(RAW / 'twsec_yearly').mkdir(exist_ok=True)
n = 0
for f in TW.glob('*.csv'):
    shutil.copyfile(f, RAW / 'twsec_yearly' / f.name); n += 1
man['files']['twsec_yearly/'] = dict(src=str(TW), n_files=n, note='taiwanan/TW_Stock_Exchange_Crawler 年度財務比率（EPS、純益率）')
REV = Path('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv')
man['files']['(in place) tw_revenue.csv'] = dict(src=str(REV), sha256=sha(REV), note='只讀原地引用（D1b 年營收）')

# ---------------- PIT 產業表 ----------------
OBS = []
def snapcsv(fn, date, tag):
    t = pd.read_csv(RAW / 'snap' / fn, dtype=str); t.columns = [c.strip('﻿"') for c in t.columns]
    mk = t['市場'].astype(str).str.strip()
    t = t[mk.isin(['上市', '市'])]
    for c, i in zip(t['代號'].str.strip(), t['產業別'].astype(str).str.strip()):
        OBS.append((c, pd.Timestamp(date), i, tag))
snapcsv('StockList_20150522.csv', '2015-05-22', '快照 2015-05-22（weihautin StockList）')
snapcsv('StockList_20160401.csv', '2016-04-01', '快照 2016-04-01（weihautin StockList.csv-old，以 3008/2227 收盤比對定日）')
snapcsv('StockList_20170706.csv', '2017-07-06', '快照 2017-07-06（weihautin StockList）')
t = pd.read_csv(q, dtype=str)
for c, i in zip(t.code, t.industry): OBS.append((c, pd.Timestamp('2023-01-16'), i, '證交所 ISIN 2023-01-16'))
t = pd.read_csv(RAW / 'snap' / 'stock_list_202610.csv', dtype=str)
t = t[t.market == 'TWSE']
for c, i in zip(t.code, t.industry):
    if isinstance(i, str) and i: OBS.append((c, pd.Timestamp('2026-10-01'), i, 'tw-stock-screener stock_list 2026-10'))
MANUAL = {   # 沒有任何快照的下市股（人工推定，依公司主業；未能線上核對）
    '1606': '電器電纜', '2315': '電腦及週邊設備業', '2336': '電腦及週邊設備業', '2350': '其他電子業', '2366': '通信網路業',
    '2384': '光電業', '2391': '通信網路業', '2394': '光電業', '2403': '電子通路業', '2418': '電子零組件業', '2446': '電子零組件業',
    '2469': '電子零組件業', '2473': '資訊服務業', '2526': '建材營造業', '2807': '金融保險業', '2822': '金融保險業', '2854': '金融保險業',
    '3007': '電子零組件業', '3009': '光電業', '3012': '光電業', '3061': '光電業', '3063': '半導體業', '3214': '光電業',
    '3271': '通信網路業', '3367': '通信網路業', '3534': '半導體業', '3638': '半導體業', '3697': '半導體業', '5280': '半導體業',
    '5854': '金融保險業', '6004': '金融保險業', '8008': '電腦及週邊設備業', '8078': '通信網路業',
    # 2015 以前下市、快照沒有但在 o8 名單常見者（同 o_lc_sector_strong_stocks 的 MANUAL_DELIST 口徑）
    '7822': '半導體業',   # 倍利科（2026-03 上市，screener 產業欄空白）
    '2311': '半導體業', '2325': '半導體業', '2448': '光電業', '3474': '半導體業', '2823': '金融保險業', '2833': '金融保險業', '2847': '金融保險業',
}
PREFIX = {'11': '水泥工業', '12': '食品工業', '13': '塑膠工業', '14': '紡織纖維', '15': '電機機械', '16': '電器電纜', '17': '化學工業',
          '18': '玻璃陶瓷', '19': '造紙工業', '20': '鋼鐵工業', '21': '橡膠工業', '22': '汽車工業', '25': '建材營造業', '26': '航運業',
          '27': '觀光餐旅', '28': '金融保險業', '29': '貿易百貨業', '99': '其他業'}
def norm(i): return '觀光餐旅' if i in ('觀光事業', '觀光餐旅業') else ('金融保險業' if i in ('金控業', '銀行業', '保險業', '證券業') else i)
O = pd.DataFrame(OBS, columns=['code', 'date', 'industry', 'source'])
O['industry'] = O['industry'].map(norm)
O = O.sort_values(['code', 'date'])
codes = sorted(set(pd.read_parquet(AG / 'o_v7_listing_exec_engine' / 'data' / 'close_raw.parquet').columns))
REG23 = pd.Timestamp('2023-07-03')
rows = []
for c in codes:
    o = O[O.code == c]
    if c == '7822': o = o.iloc[0:0]
    if len(o) == 0:
        if c in MANUAL: rows.append((c, '2004-01-01', '', MANUAL[c], '人工推定（無公開快照；依公司主業）'))
        elif c[:2] in PREFIX and not c.startswith('91'): rows.append((c, '2004-01-01', '', PREFIX[c[:2]], '代碼前兩碼推定（無快照）'))
        else: rows.append((c, '2004-01-01', '', '未分類', '無資料'))
        continue
    o = o.drop_duplicates(['date'], keep='first').reset_index(drop=True)
    segs = []
    cur = o.industry.iloc[0]; src = o.source.iloc[0]
    segs.append(['2004-01-01', None, cur, '回溯：沿用最早觀測（' + src + '）'])
    for k in range(1, len(o)):
        if o.industry.iloc[k] != cur:
            dt = o.date.iloc[k]
            if o.source.iloc[k].startswith('tw-stock-screener') and o.source.iloc[k - 1].startswith('證交所 ISIN'): dt = REG23
            segs[-1][1] = str(dt.date())
            cur = o.industry.iloc[k]
            segs.append([str(dt.date()), None, cur, o.source.iloc[k] + ('（2023-07-03 產業別調整生效）' if dt == REG23 else '（較晚觀測日起生效）')])
    for s in segs: rows.append((c, s[0], s[1] or '', s[2], s[3]))
T = pd.DataFrame(rows, columns=['code', 'start', 'end', 'industry', 'source'])
T.to_csv(RAW / 'industry_pit.csv', index=False)
man['files']['industry_pit.csv'] = dict(sha256=sha(RAW / 'industry_pit.csv'), rows=len(T), n_changed=int((T.groupby('code').size() > 1).sum()),
                                        n_manual=int(T.source.str.startswith('人工').sum()), n_prefix=int(T.source.str.startswith('代碼').sum()))
(RAW / 'manifest.json').write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(man['files']['industry_pit.csv'], ensure_ascii=False))
print(T[T.code.isin(T.code[T.groupby('code').code.transform('size') > 1])].head(40).to_string())
