#!/usr/bin/env python3
"""第 0 步 (a)-1：下市股價格修補表 delist_prices.parquet（不看任何策略報酬）。
python3 -I -B build_delist.py

應有名單：證交所「終止上市公司」頁面（twse.com.tw/zh/listed/suspend-listing.html）的鏡像
  raw/songdoyouho/delisted_stocks.csv（GitHub songdoyouho/MomentumStrategy，2001-01～2024-09，256 列）。
  只取 4 碼代碼（6 碼 DR／外國股不在面板宇宙內）。
價格來源（raw/ 視為不可信，只用 git 讀檔，不執行其中任何程式）：
  A. taiwanan/TW_Stock_Exchange_Crawler tse_trading_raw_data（證交所 STOCK_DAY 格式，2015-01～2021-04-29）
     ── o_rev_survivorship 已 clone 在 ../o_rev_survivorship/raw/，每檔取最後存在的版本。
  B. henry753951/finance data/stocks（FinMind TaiwanStockPrice 原始價，含 spread；2005-09～2022-12-30），每檔取 git 歷史中最後存在的版本。
修補範圍：面板 2019-05-17～2023-12 中，終止上市名單內 4 碼股票在「面板最後一筆」之後、來源最後一筆之前的交易日。
  來源優先序：A（官方格式）＞ B。日期必須落在面板日曆上。
日報酬：A＝收盤 ÷ 參考價 −1（參考價＝收盤 ±|漲跌價差| 中最接近前收者，同 o_rev_survivorship）；
        B＝收盤 ÷ (收盤 − spread) − 1（FinMind 除權息日 spread＝0 → 報酬 0，與面板 flag 1 慣例相同）。成交量 0 的列視為無成交。
交叉核對（門檻：收盤一致率 ≥ 99%）：A vs 面板、B vs 面板（2005～2019-05-16 重疊段）、A vs B（2019-05-17～2021-04-29）。
覆蓋率：應有股票日＝面板最後一筆之後到「實際最後交易日」的面板交易日數；實際最後交易日＝來源最後一筆
  （若來源最後一筆停在來源截止日 A 2021-04-29／B 2022-12-30，改用「終止上市日 − 該年代中位落差」推估）。
"""
import subprocess, csv, re, json, hashlib, datetime, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw'
TA = HERE.parent / 'o_rev_survivorship' / 'raw' / 'taiwanan_TW_Stock_Exchange_Crawler'
HE = RAW / 'henry753951_finance'
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
C = pd.read_parquet(P / 'close_raw.parquet'); RET = pd.read_parquet(P / 'ret.parquet')
IDX = C.index
lt = C.apply(lambda s: s.last_valid_index())
CUT = pd.Timestamp('2019-05-16')
A_END, B_END = pd.Timestamp('2021-04-29'), pd.Timestamp('2022-12-30')

def git(repo, *a): return subprocess.run(['git', '-C', str(repo), *a], capture_output=True, check=True).stdout
def last_version(repo, path):
    h = git(repo, 'log', '--all', '-1', '--format=%H', '--', path).decode().strip()
    if not h: return None
    try: return git(repo, 'show', f'{h}:{path}')
    except subprocess.CalledProcessError: return git(repo, 'show', f'{h}^:{path}')
def num(x):
    x = re.sub(r'[^0-9.\-]', '', str(x).replace(',', ''))
    try: return float(x)
    except ValueError: return np.nan

# ---------- 應有名單 ----------
d = pd.read_csv(RAW / 'songdoyouho' / 'delisted_stocks.csv', dtype=str); d.columns = ['roc', 'name', 'code']
d['date'] = [pd.Timestamp(int(a) + 1911, int(b), int(c)) for a, b, c in d['roc'].str.split('/')]
d['four'] = d['code'].str.fullmatch(r'\d{4}')
d['in_panel'] = d['code'].isin(C.columns)
d['panel_last'] = d['code'].map(lt)
d = d.sort_values('date').reset_index(drop=True)
d.to_csv(HERE / 'delist_official.csv', index=False)
tgt = d[(d['date'] > CUT) & (d['date'] <= pd.Timestamp('2023-12-31')) & d['four']].copy()

# ---------- 來源 A / B ----------
def load_A(c):
    raw = last_version(TA, f'tse_trading_raw_data/{c}.csv')
    if raw is None: return None, None
    rows = []
    for r in csv.reader(raw.decode('utf-8-sig', errors='replace').splitlines()[2:]):
        if len(r) < 9 or not re.fullmatch(r'\d{2,3}/\d{2}/\d{2}', r[0].strip()): continue
        y, m, dd = r[0].strip().split('/')
        rows.append((pd.Timestamp(int(y) + 1911, int(m), int(dd)), num(r[2]), num(r[3]), num(r[5]), num(r[6]), num(r[8]), num(r[1])))
    D = pd.DataFrame(rows, columns=['date', 'open', 'high', 'close', 'chg', 'value', 'vol']).drop_duplicates('date', keep='last').sort_values('date')
    D = D[np.isfinite(D['close']) & (D['close'] > 0) & (D['vol'].fillna(0) > 0)]
    pc = D['close'].shift(1); a = D['chg'].abs()
    c1, c2 = D['close'] - a, D['close'] + a
    ref = np.where(np.abs(c1 - pc) <= np.abs(c2 - pc), c1, c2); ref = np.where(pc.isna(), c1, ref)
    D['ret'] = np.where(ref > 0, D['close'] / ref - 1, np.nan)
    return D.set_index('date'), hashlib.sha256(raw).hexdigest()

def load_B(c):
    raw = last_version(HE, f'data/stocks/{c}.csv')
    if raw is None: return None, None
    D = pd.read_csv(pd.io.common.BytesIO(raw), usecols=['date', 'Trading_Volume', 'Trading_money', 'open', 'max', 'close', 'spread'])
    D['date'] = pd.to_datetime(D['date'])
    D = D.drop_duplicates('date', keep='last').sort_values('date')
    D = D[(D['Trading_Volume'] > 0) & (D['close'] > 0)]
    ref = D['close'] - D['spread']
    D = pd.DataFrame({'date': D['date'], 'open': D['open'], 'high': D['max'], 'close': D['close'], 'value': D['Trading_money'],
                      'ret': np.where(ref > 0, D['close'] / ref - 1, np.nan)})
    return D.set_index('date'), hashlib.sha256(raw).hexdigest()

def agree(x, y):
    j = x.index.intersection(y.index)
    xx, yy = x.reindex(j), y.reindex(j); m = xx.notna() & yy.notna()
    return int(m.sum()), int((np.abs(xx[m] - yy[m]) < 1e-6).sum())

# 交叉核對：用「所有終止上市 4 碼且在面板內」的股票（不只目標）比對面板重疊段，擴大樣本
chk = {'A_vs_panel': [0, 0], 'B_vs_panel': [0, 0], 'A_vs_B_2019_2021': [0, 0], 'A_vs_panel_ret0.1pp': [0, 0], 'B_vs_panel_ret0.1pp': [0, 0]}
rows = []; meta = {}
for _, r in tgt.iterrows():
    c = r['code']
    A, ha = load_A(c); B, hb = load_B(c)
    meta[c] = {'name': r['name'], 'delist_date': str(r['date'].date()), 'panel_last': str(r['panel_last'].date()) if pd.notna(r['panel_last']) else None,
               'A_last': str(A.index.max().date()) if A is not None and len(A) else None, 'B_last': str(B.index.max().date()) if B is not None and len(B) else None,
               'A_sha': ha, 'B_sha': hb}
    pl = r['panel_last'] if pd.notna(r['panel_last']) else pd.Timestamp('1900-01-01')
    pan_c = C[c].dropna() if c in C.columns else pd.Series(dtype=float)
    pan_r = RET[c].dropna() if c in RET.columns else pd.Series(dtype=float)
    for nm, S in (('A', A), ('B', B)):
        if S is None: continue
        n, k = agree(S['close'][S.index <= pl], pan_c); chk[f'{nm}_vs_panel'][0] += n; chk[f'{nm}_vs_panel'][1] += k
        j = S.index[S.index <= pl].intersection(pan_r.index); rr = S['ret'].reindex(j); pr = pan_r.reindex(j); m = rr.notna() & pr.notna()
        chk[f'{nm}_vs_panel_ret0.1pp'][0] += int(m.sum()); chk[f'{nm}_vs_panel_ret0.1pp'][1] += int((np.abs(rr[m] - pr[m]) <= 0.001).sum())
    if A is not None and B is not None:
        n, k = agree(A['close'][A.index > CUT], B['close'][B.index > CUT]); chk['A_vs_B_2019_2021'][0] += n; chk['A_vs_B_2019_2021'][1] += k
    # 修補列
    take = []
    if A is not None:
        a = A[(A.index > pl)]; a = a.assign(src='A'); take.append(a)
    if B is not None:
        b = B[(B.index > pl)]
        if A is not None: b = b[~b.index.isin(A.index)]
        take.append(b.assign(src='B'))
    if take:
        T = pd.concat(take).sort_index()
        T = T[T.index.isin(IDX)]
        T['code'] = c
        rows.append(T.reset_index())
patch = pd.concat(rows, ignore_index=True)[['date', 'code', 'open', 'high', 'close', 'value', 'ret', 'src']]
# 第一筆修補列的報酬：以面板最後收盤為前收（跨停牌）；A 的參考價規則已處理，B 只在 spread 缺時補
patch = patch.sort_values(['code', 'date']).reset_index(drop=True)

# ---------- 實際最後交易日與覆蓋率 ----------
gap_hist = d[d['in_panel'] & d['four'] & (d['date'] < CUT) & (d['date'] > pd.Timestamp('2010-01-01'))]
med_gap = int(np.median([np.searchsorted(IDX, x) - np.searchsorted(IDX, y) for x, y in zip(gap_hist['date'], gap_hist['panel_last'])]))
cov = []
for _, r in tgt.iterrows():
    c = r['code']; m = meta[c]
    srcl = patch.loc[patch['code'] == c, 'date'].max() if (patch['code'] == c).any() else None
    pl = r['panel_last'] if pd.notna(r['panel_last']) else None
    capped = srcl is not None and (srcl in (A_END, B_END) or abs((srcl - A_END).days) <= 1 or abs((srcl - B_END).days) <= 1)
    if srcl is None or capped:
        i = np.searchsorted(IDX, r['date']) - med_gap
        last_true = IDX[max(i, 0)] if r['date'] <= IDX[-1] else None
        est = True
    else:
        last_true = srcl; est = False
    if pl is None: need = 0
    else: need = int(((IDX > pl) & (IDX <= last_true)).sum()) if last_true is not None else 0
    got = int(((patch['code'] == c) & (patch['date'] <= last_true)).sum()) if last_true is not None else 0
    cov.append({'code': c, 'name': r['name'], 'delist_date': str(r['date'].date()), 'panel_last': m['panel_last'], 'src_last': str(srcl.date()) if srcl is not None else None,
                'true_last_trade': str(last_true.date()) if last_true is not None else None, 'true_last_estimated': est, 'need_days': need, 'got_days': got})
cov = pd.DataFrame(cov)
cov.to_csv(HERE / 'delist_coverage.csv', index=False)
f = HERE / 'delist_prices.parquet'; patch.to_parquet(f, index=False)
out = {'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
       'official_list': {'file': 'raw/songdoyouho/delisted_stocks.csv', 'sha256': hashlib.sha256((RAW / 'songdoyouho' / 'delisted_stocks.csv').read_bytes()).hexdigest(),
                         'rows': int(len(d)), 'range': [str(d['date'].min().date()), str(d['date'].max().date())]},
       'n_target_4digit_2019-05-17_2023': int(len(tgt)), 'n_target_in_panel': int(tgt['in_panel'].sum()),
       'n_6digit_excluded_2019_2023': int(((d['date'] > CUT) & (d['date'] <= pd.Timestamp('2023-12-31')) & ~d['four']).sum()),
       'median_gap_tradingdays_lasttrade_to_delist': med_gap,
       'cross_check': {k: {'n': v[0], 'agree': v[1], 'rate': (v[1] / v[0]) if v[0] else None} for k, v in chk.items()},
       'patch_rows': int(len(patch)), 'patch_rows_by_src': patch['src'].value_counts().to_dict(),
       'coverage_stockdays': {'need': int(cov['need_days'].sum()), 'got': int(cov['got_days'].sum()),
                              'rate': float(cov['got_days'].sum() / cov['need_days'].sum())},
       'stocks_full': int((cov['got_days'] >= cov['need_days']).sum()), 'stocks_partial': cov.loc[cov['got_days'] < cov['need_days'], 'code'].tolist(),
       'sha256': hashlib.sha256(f.read_bytes()).hexdigest(), 'meta': meta}
(HERE / 'delist_prices_meta.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps({k: v for k, v in out.items() if k != 'meta'}, ensure_ascii=False, indent=1))
print(cov.to_string())
