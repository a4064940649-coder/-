#!/usr/bin/env python3
"""獨立複製（上市身分）：只依 frozen_spec.txt 第一節第 2 點，從 raw/ 與 V6 面板 src 欄自己建逐日上市矩陣（不讀研究員的 s1_listing.py／srcload7.py／data/src_*.parquet）。
寫法刻意不同：全部證據併成長表 → 每檔依日期排序、同日取最高優先 → merge_asof 前向／後向帶入最近證據 → 規則判定。
輸出 replicate/data/listed_rep.parquet。"""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; TOP = H.parent; RAW = TOP / 'raw'; OUT = H / 'data'; OUT.mkdir(exist_ok=True)
V6 = TOP.parent / 'o_v6_price_panel_official' / 'data'
c6 = pd.read_parquet(V6 / 'close_raw.parquet'); s6 = pd.read_parquet(V6 / 'src.parquet')
COLS = list(c6.columns)
tx = pd.to_datetime(pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv')['date'])
CAL = c6.index.union(pd.DatetimeIndex(sorted(set(tx[(tx >= '2004-01-01') & (tx <= '2026-10-27')]))))
ok4 = lambda s: s.astype(str).str.fullmatch(r'[1-9]\d{3}')
ev = []
def add(d, c, m, pr, src): ev.append(pd.DataFrame({'date': pd.to_datetime(d), 'code': pd.Series(c).astype(str).str.strip().to_numpy(), 'm': m, 'pr': pr, 'src': src}))
st = c6.notna() & s6.isin([1, 2, 4]); x = st.stack(); x = x[x]; add(x.index.get_level_values(0), x.index.get_level_values(1), 'TW', 9, 'panel')
st = c6.notna() & (s6 == 3); x = st.stack(); x = x[x]; add(x.index.get_level_values(0), x.index.get_level_values(1), 'TW', 4, 'y5m')
for f in glob.glob(str(RAW / 'leochen1_stock_v' / 'data' / 'history' / '*.csv')):
    d = pd.read_csv(f, encoding='utf-8-sig', usecols=[0, 5]); d.columns = ['date', 'v']; d = d[pd.to_numeric(d.v, errors='coerce') > 0]
    d = d[pd.to_datetime(d.date) >= '2019-01-01']; add(d.date, [Path(f).stem] * len(d), 'TW', 7, 'leo')
for f in glob.glob(str(RAW / 'bacardi' / 'prices_*.csv')):
    d = pd.read_csv(f, dtype={'stock_id': str}, encoding='utf-8-sig', usecols=['date', 'stock_id', 'market', 'volume', 'close'])
    d = d[(pd.to_numeric(d.volume, errors='coerce') > 0) & (pd.to_numeric(d.close, errors='coerce') > 0)]
    for mk, m in (('TWSE', 'TW'), ('TPEx', 'OT')):
        q = d[d.market == mk]; add(q.date, q.stock_id, m, 8, 'bac')
for f in glob.glob(str(RAW / 'tasi' / '淨值' / '*.csv')):
    nm = Path(f).name; m = 'TW' if '市淨值' in nm else 'OT' if '櫃淨值' in nm else None
    if m is None: continue
    try: d = pd.read_csv(f, dtype=str, encoding='utf-8-sig')
    except Exception: continue
    cc = [c for c in d.columns if '代號' in c]
    if cc: add([nm[:8]] * len(d), d[cc[0]], m, 5, 'pbr')
ad = pd.read_csv(RAW / 'tasi_file_add_dates.csv', header=None, names=['f', 'added'])
ad = ad[ad.f.str.startswith('營收/')]; adm = ad.assign(ym=ad.f.str[3:9]).groupby('ym').added.min()
for f in glob.glob(str(RAW / 'tasi' / '營收' / '*.csv')):
    ym = Path(f).stem
    if ym not in adm.index: continue
    a = pd.Timestamp(adm[ym]); end = (pd.Period(ym, 'M') + 1).to_timestamp()
    if not (-1 <= (a - end).days <= 45): continue
    m = 'TW'; cs, ms = [], []
    for ln in Path(f).read_text(encoding='utf-8-sig', errors='replace').splitlines()[1:]:
        p = ln.split(',')[0]
        if p.startswith('全部國內上市公司合計'): m = 'OT'; continue
        if p.startswith('全部國內上櫃公司合計'): break
        cs.append(p); ms.append(m)
    ev.append(pd.DataFrame({'date': a, 'code': cs, 'm': ms, 'pr': 6, 'src': 'rev'}))
isin = {}
for f in glob.glob(str(RAW / 'isin' / 'snap' / 'isin_0*.txt')):
    d = json.load(open(f, encoding='utf-8')); v = pd.Timestamp(d['version'].replace('/', '-'))
    for g in d['list']:
        for it in g['list']:
            c = it['code'].strip()
            if Path(f).name.startswith('isin_02') and it['market'].startswith('上市') and g['name'] in ('股票', '普通股'):
                ev.append(pd.DataFrame({'date': [v], 'code': [c], 'm': ['TW'], 'pr': [7], 'src': ['isin']})); isin[c] = pd.Timestamp(it['listedDate'].replace('/', '-'))
            if Path(f).name.startswith('isin_04') and it['market'] == '上櫃' and g['name'] == '股票':
                ev.append(pd.DataFrame({'date': [v], 'code': [c], 'm': ['OT'], 'pr': [7], 'src': ['isin']}))
sl = pd.read_csv(RAW / 'screener' / 'stock_list.csv', dtype=str); slm = dict(zip(sl.code, sl.market))
h = pd.read_csv(RAW / 'screener' / 'history.csv.gz', dtype={'code': str}, usecols=['date', 'code', 'volume', 'close'])
h = h[(h.volume > 0) & (h.close > 0)]; mm = h.code.map(slm).map({'TWSE': 'TW', 'TPEX': 'OT'})
add(h.date[mm.notna()], h.code[mm.notna()], mm[mm.notna()].to_numpy(), 1, 'hist')
dl = pd.read_csv(TOP.parent / 'o_infra_v3_rebuild' / 'delist_official.csv', dtype=str); add(dl.date, dl.code, 'DL', 10, 'dl')
EV = pd.concat(ev, ignore_index=True); EV = EV[ok4(EV.code)]
EV = EV.sort_values(['code', 'date', 'pr']).drop_duplicates(['code', 'date'], keep='last')
man = json.loads((TOP / 'listing_manual.json').read_text(encoding='utf-8'))
codes = sorted(set(COLS) | set(EV[EV.m == 'TW'].code))
LM = {}
cal = pd.DataFrame({'date': CAL})
DAILY = {'panel', 'y5m', 'leo', 'bac', 'pbr', 'isin'}
for c, e in EV.groupby('code'):
    if c not in codes or not (e.m == 'TW').any(): continue
    e = e[['date', 'm', 'src']].sort_values('date')
    b = pd.merge_asof(cal, e.rename(columns={'m': 'prev'})[['date', 'prev']], on='date', direction='backward')
    f = pd.merge_asof(cal, e.rename(columns={'m': 'nxt'})[['date', 'nxt']], on='date', direction='forward')
    prev, nxt = b.prev.to_numpy(), f.nxt.to_numpy()
    if c in isin and not e[(e.date < isin[c]) & e.m.isin(['OT', 'DL'])].shape[0]:
        start = isin[c]
    else:
        start = e[(e.m == 'TW') & e.src.isin(DAILY)].date.min()
        if pd.isna(start): start = e[e.m == 'TW'].date.min()
    lst = (prev == 'TW') | ((pd.isna(prev) | (prev == None)) & (nxt == 'TW') & (CAL >= start))
    lst &= ~((prev == 'TW') & False)
    if c in isin and not e[(e.date < isin[c]) & e.m.isin(['OT', 'DL'])].shape[0]:
        lst &= CAL >= isin[c]
    for a, bb in man.get(c, {}).get('exclude', []): lst &= ~((CAL >= pd.Timestamp(a)) & (CAL <= pd.Timestamp(bb)))
    # 規格：之後無證據 → 上市到最後一筆證據或成交（不延伸到面板終點之後無資料的期間）；DL 由 prev=='DL' 處理
    last = e[e.m == 'TW'].date.max()
    tr = c6[c].dropna().index if c in c6.columns else pd.DatetimeIndex([])
    lasttr = max([d for d in [last, tr.max() if len(tr) else None] if d is not None and not pd.isna(d)])
    lst &= ~((CAL > lasttr) & (pd.isna(nxt) | (nxt == None)))
    LM[c] = lst
OUTM = pd.DataFrame(LM, index=CAL).reindex(columns=sorted(LM)).fillna(False)
OUTM.to_parquet(OUT / 'listed_rep.parquet')
print('codes', OUTM.shape, 'listed cells', int(OUTM.to_numpy().sum()))
