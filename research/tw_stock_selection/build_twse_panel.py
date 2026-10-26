#!/usr/bin/env python3
"""建立上市（TWSE）普通股日線面板 2004-01 ～ 2026-10，拼接四個公開來源並處理除權息／公司行動／資料錯誤。

來源與優先序（依日期）：
  1 Asoul/tsec（證交所原始行情，含已下市股）：≤ 2019-05-16，且僅用於 k66 沒有的股票／日期（2010 前、已下市股）
  2 k66inthesky/TW-stock（證交所 STOCK_DAY，抓取時仍上市者）：上市日或 2010-01-04 ～ 2023-12-29
    Asoul 與 k66 重疊期逐日比對，> 20% 個股收盤不符的日期判定為 Asoul 錯置（如 2017-08-03 複製了 07-31），全市場刪除
  3 voidful/tw_stocker（Yahoo 5 分 K 彙總成日 K，抓取時仍上市者）：2024-02-15 ～ 2025-09-23
    收盤為 13:20 最後一根，非官方收盤（與官方收盤差異中位數 0%）；成交量不含收盤集合競價，依個股重疊期比例放大
  4 stevenhsiehsheng-ctrl/tw-stock-screener history.csv.gz（全市場官方日 K）：2025-09-24 ～ 2026-10-22

日報酬規則（官方來源 1、2 有「漲跌價差」欄，其基準是證交所公告的參考價）：
  A 參考價（收盤 − 漲跌價差）＝ 前收 → R = 收盤 / 前收 − 1
  B 除權息日：行情檔標 X，或 2017-06～2019-05 的 Asoul 以「0.00」取代 X（漲跌價差 0 但收盤≠前收）
    → 有 exdiv_5y 參考價（2021-10 起）則 R = 收盤 / 前收 × F − 1，否則 R = 0（近似，除權息日的價格變動略去）
  C 參考價 ≠ 前收且漲跌價差非 0（減資、除權息有公告漲跌）→ R = 收盤 / 參考價 − 1
  D 其餘全市場異常日（> 50% 官方來源個股的漲跌價差與前收矛盾）→ 整日刪除
非官方來源 3、4：R = 收盤 / 前收 × F − 1，F = 兩筆觀測間 exdiv_5y 事件的（前收／參考價）連乘；
  開盤偏離前收超過漲跌幅 + 0.5%（近似收盤日 + 1.5%）且無除權息 → 視為公司行動，R = 收盤 / 開盤 − 1。
資料缺口（該期間沒有任何來源涵蓋，例如 2024-01-02 ～ 2024-02-14）：保留跨缺口報酬（含 F），只剔除 −70% / +150% 以外。
其餘超過「漲跌幅 ^ 間隔天數 + 1%（近似收盤日 + 3%）」者視為資料錯誤，R 設為缺值（還原指數視為 0）。
flag：0 正常、1 除權息日設 0、2 除權息參考價調整、3 參考價重設（公司行動）、4 資料缺口、5 資料錯誤剔除
輸出：scratchpad/panel/*.parquet 寬表（close_raw, open, high, value, adj, ret, flag, src）＋ meta.json
"""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd

S = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad')
SCR = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data')
OUT = S / 'panel'
D_ASOUL_END, D_K66_START, D_K66_END = pd.Timestamp('2019-05-16'), pd.Timestamp('2019-05-17'), pd.Timestamp('2023-12-29')
D_YAH_START, D_YAH_END, D_HIST_START = pd.Timestamp('2024-02-15'), pd.Timestamp('2025-09-23'), pd.Timestamp('2025-09-24')
LIMIT_CHANGE = pd.Timestamp('2015-06-01')   # 漲跌幅 7% → 10%
COLS = ['d', 'vol', 'val', 'o', 'h', 'l', 'c', 'chg', 'n']

def is4(c): return isinstance(c, str) and len(c) == 4 and c.isdigit() and c[0] != '0'

def read_exchange(path, has_header):
    if has_header:
        df = pd.read_csv(path, dtype=str); df.columns = COLS[:len(df.columns)]
    else:
        df = pd.read_csv(path, header=None, names=COLS, dtype=str)
    d = df['d'].astype(str).str.strip()
    if d.str.contains('/').all():
        p = d.str.split('/', expand=True).astype(int); p[0] += 1911
        df['date'] = pd.to_datetime(dict(year=p[0], month=p[1], day=p[2]), errors='coerce')
    else:
        df['date'] = pd.to_datetime(d, errors='coerce')
    for c in ['o', 'h', 'c', 'val']:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce')
    ch = df['chg'].astype(str).str.strip()
    df['x'] = ch.str.startswith('X')
    df['chgv'] = pd.to_numeric(ch.str.replace('+', '', regex=False).str.replace(',', ''), errors='coerce').where(~df['x'])
    df['plus'] = ch.str.startswith('+')      # Asoul：櫃買（TPEx）行情的漲跌價差帶「+」號，證交所（TWSE）不帶
    df = df.dropna(subset=['date', 'c']); df = df[df['c'] > 0]
    df = df.drop_duplicates('date').set_index('date').sort_index()
    return df[['o', 'h', 'c', 'val', 'x', 'chgv', 'plus']]

def inconsistent_rows(df):
    """官方來源中，漲跌價差隱含參考價與前收矛盾（且漲跌價差非 0）的列。"""
    c = df['c'].to_numpy(float); chg = df['chgv'].to_numpy(float)
    prev = np.r_[np.nan, c[:-1]]
    off = df['src'].isin([1, 2]).to_numpy()
    ok = off & np.isfinite(chg) & np.isfinite(prev) & (chg != 0)
    inc = ok & (np.abs((c - chg) - prev) > 0.011 + 1e-4 * prev)
    return ok, inc

def compute_returns(df, cover_iv, ex_events, CAL):
    """df：單一股票，index=日期（升冪），欄位 o,h,c,val,x,chgv,src。回傳 (R, flag) ndarray。"""
    pos = CAL.get_indexer(df.index)
    covered = np.zeros(len(CAL), bool)
    for a, b in cover_iv:
        covered[CAL.searchsorted(a):CAL.searchsorted(b, 'right')] = True
    unc = np.concatenate([[0], np.cumsum(~covered)])
    c = df['c'].to_numpy(float); o = df['o'].to_numpy(float); x = df['x'].to_numpy(bool)
    chg = df['chgv'].to_numpy(float); src = df['src'].to_numpy(float)
    N = len(df); prev = np.r_[np.nan, c[:-1]]; prev_src = np.r_[np.nan, src[:-1]]
    ppos = np.r_[-1, pos[:-1]]
    nskip = np.where(ppos >= 0, pos - ppos - 1, 0)
    gap_unc = np.where(ppos >= 0, unc[pos] - unc[np.maximum(ppos + 1, 0)], 0)
    F = np.ones(N)
    if ex_events is not None:
        for ed, f in zip(ex_events['date'], ex_events['f']):
            k = df.index.searchsorted(ed)
            if 0 < k < N: F[k] *= f
    dates = df.index
    lim = np.where(dates < LIMIT_CHANGE, 0.07, 0.10)
    obs_no = np.arange(N)
    is_gap = gap_unc > 0
    official = np.isin(src, [1, 2])
    approx = (src == 3) | ((src == 4) & (prev_src == 3))
    with np.errstate(divide='ignore', invalid='ignore'):
        r_prev = c / prev * F - 1
        ref = c - chg
        consistent = official & np.isfinite(chg) & (np.abs(ref - prev) <= 0.011 + 1e-4 * prev)
        lostref = official & np.isfinite(chg) & (chg == 0) & ~consistent
        xday = official & x
        caseC = official & np.isfinite(chg) & (chg != 0) & ~consistent & (ref > 0)
        r = r_prev.copy(); flag = np.zeros(N, np.int8)
        r[consistent] = (c / prev - 1)[consistent]
        mB = (xday | lostref) & ~is_gap
        r[mB & (F > 1)] = r_prev[mB & (F > 1)]; flag[mB & (F > 1)] = 2
        r[mB & (F == 1)] = 0.0; flag[mB & (F == 1)] = 1
        mC = caseC & ~is_gap & ~mB
        r[mC] = (c / ref - 1)[mC]; flag[mC] = 3
        other = ~consistent & ~mB & ~mC & ~is_gap
        flag[other & (F > 1)] = 2
        open_dev = np.abs(o / prev - 1)
        ca = other & (F == 1) & (obs_no >= 5) & np.isfinite(open_dev) & (open_dev > lim + np.where(approx, 0.015, 0.005)) & (o > 0)
        r[ca] = (c / o - 1)[ca]; flag[ca] = 3
        flag[is_gap] = 4
        n = nskip + 1; m_err = np.where(approx, 0.03, 0.01)
        up_lim = (1 + lim) ** n - 1 + m_err; dn_lim = 1 - (1 - lim) ** n + m_err
        ipo = obs_no < 5
        bad = (~is_gap & ~ipo & ((r > up_lim) | (r < -dn_lim))) | (is_gap & ((r < -0.7) | (r > 1.5))) | (~is_gap & ipo & ((r > 1.5) | (r < -0.7)))
    r[bad] = np.nan; flag[bad] = 5
    r[0] = np.nan; flag[0] = 0
    return r, flag

def load_sources():
    twse_list = pd.read_csv(S / 'piptest/twstock_wheel/twstock/codes/twse_equities.csv', dtype=str)
    tpex_list = pd.read_csv(S / 'piptest/twstock_wheel/twstock/codes/tpex_equities.csv', dtype=str)
    scr_list = pd.read_csv(SCR / 'stock_list.csv', dtype=str)
    twse_codes = {c for c in twse_list.loc[twse_list['type'] == '股票', 'code'] if is4(c)} | {c for c in scr_list.loc[scr_list['market'] == 'TWSE', 'code'] if is4(c)}
    tpex_codes = {c for c in tpex_list.loc[tpex_list['type'] == '股票', 'code'] if is4(c)} | {c for c in scr_list.loc[scr_list['market'] == 'TPEX', 'code'] if is4(c)}
    asoul = {Path(p).stem: p for p in glob.glob(str(S / 'tsec_all/*.csv'))}
    k66 = {Path(p).stem: p for p in glob.glob(str(S / 'k66/*.csv'))}
    yah = {Path(p).stem: p for p in glob.glob(str(S / 'tw_stocker_daily/*.csv'))}
    hist = pd.read_csv(SCR / 'history.csv.gz', dtype={'code': str}, parse_dates=['date'])
    hist = hist[hist['code'].map(is4) & (hist['volume'] > 0) & (hist['close'] > 0)]
    hist_g = {c: g.set_index('date').sort_index() for c, g in hist.groupby('code')}
    ex = pd.read_csv(SCR / 'extras/exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
    ex = ex[ex['code'].map(is4) & (ex['ref_price'] > 0) & (ex['prev_close'] > 0)].copy()
    ex['f'] = ex['prev_close'] / ex['ref_price']
    ex = ex[(ex['f'] >= 1.0) & (ex['f'] < 3.0)]
    ex_g = {c: g.sort_values('date') for c, g in ex.groupby('code')}
    return twse_codes, tpex_codes, asoul, k66, yah, hist_g, ex_g

def main():
    OUT.mkdir(exist_ok=True)
    twse_codes, tpex_codes, asoul, k66, yah, hist_g, ex_g = load_sources()
    cal = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date'])['date']
    delisted_any = {c for c in asoul if is4(c) and c not in twse_codes and c not in tpex_codes}
    # 已下市股的市場別：Asoul 檔中上漲日帶「+」號比例 > 0.3 → 櫃買（實測：現存櫃買 718 檔全為 1.0、現存上市 90% 為 0）
    def plus_share(code):
        a = read_exchange(asoul[code], False); pos = a['chgv'] > 0
        return float(a['plus'].sum() / max(int(pos.sum()), 1))
    delisted_tpex = {c for c in delisted_any if plus_share(c) > 0.3}
    delisted_unknown = delisted_any - delisted_tpex      # 視為上市（TWSE）下市股
    print('delisted in Asoul:', len(delisted_any), '| TPEx-format', len(delisted_tpex), '| TWSE-format', len(delisted_unknown), flush=True)
    universe = sorted({c for c in asoul if is4(c) and c in twse_codes} | delisted_unknown | {c for c in k66 if is4(c)}
                      | {c for c in yah if is4(c) and c in twse_codes} | {c for c in hist_g if c in twse_codes})
    print('universe codes', len(universe), '| delisted (market unknown, from Asoul)', len(delisted_unknown), flush=True)
    # Asoul 與 k66 重疊期（2010-01～2019-05）比對：找出 Asoul 錯置的日期（多數個股收盤與 k66 不符）
    mism, tot = [], []
    for code in sorted(set(asoul) & set(k66)):
        if not is4(code): continue
        a = read_exchange(asoul[code], False)['c']; k = read_exchange(k66[code], True)['c']
        j = pd.concat([a, k], axis=1, join='inner', keys=['a', 'k']).dropna()
        tot.append(j.index); mism.append(j.index[(j['a'] / j['k'] - 1).abs() > 0.001])
    tot = pd.Series(np.concatenate(tot)).value_counts(); mism = pd.Series(np.concatenate(mism)).value_counts().reindex(tot.index).fillna(0)
    share = (mism / tot)[tot >= 100]
    asoul_bad = sorted(share[share > 0.2].index)
    print('Asoul dates disagreeing with k66 (>20% of stocks):', [str(d.date()) for d in asoul_bad], flush=True)
    raw, cover, vol_scale = {}, {}, {}
    for code in universe:
        parts, iv = [], []
        k = read_exchange(k66[code], True).loc[:D_K66_END] if code in k66 else None
        k_first = k.index[0] if k is not None and len(k) else None
        if code in asoul:
            a = read_exchange(asoul[code], False).loc[:D_ASOUL_END]
            if a['plus'].any(): a = a.loc[a.index > a.index[a['plus'].to_numpy()].max()]   # 去掉轉上市前的櫃買時期
            a = a[~a.index.isin(asoul_bad)].drop(columns='plus')
            if k_first is not None: a = a.loc[:k_first - pd.Timedelta(days=1)]
            if len(a): a['src'] = 1; parts.append(a); iv.append((a.index[0], a.index[-1] if k_first is not None else D_ASOUL_END))
        if k is not None and len(k):
            k = k.drop(columns='plus').copy(); k['src'] = 2; parts.append(k); iv.append((k_first, D_K66_END))
        if code in yah:
            y0 = pd.read_csv(yah[code], parse_dates=['date']).set_index('date').sort_index()
            y0 = y0[(y0['volume'] > 0) & (y0['close'] > 0)]
            scale = np.nan
            if code in hist_g:   # 成交量放大倍數：官方 / Yahoo，重疊期中位數
                j = hist_g[code][['volume']].join(y0[['volume']], rsuffix='_y', how='inner'); j = j[j['volume_y'] > 0]
                if len(j) >= 20: scale = float(np.clip((j['volume'] / j['volume_y']).median(), 1.0, 3.0))
            vol_scale[code] = scale
            y = y0.loc[D_YAH_START:D_YAH_END]
            if len(y):
                sc = scale if np.isfinite(scale) else 1 / 0.69
                parts.append(pd.DataFrame({'o': y['open'], 'h': y['high'], 'c': y['close'], 'val': y['volume'] * y['close'] * sc,
                                           'x': False, 'chgv': np.nan, 'src': 3}))
                iv.append((D_YAH_START, D_YAH_END))
        if code in hist_g:
            h = hist_g[code].loc[D_HIST_START:]
            if len(h):
                parts.append(pd.DataFrame({'o': h['open'], 'h': h['high'], 'c': h['close'], 'val': h['volume'] * h['close'], 'x': False, 'chgv': np.nan, 'src': 4}))
                iv.append((D_HIST_START, pd.Timestamp('2030-01-01')))
        if not parts: continue
        df = pd.concat(parts).sort_index(); df = df[~df.index.duplicated(keep='last')]
        df = df.loc[df.index >= pd.Timestamp('2004-01-01')]
        if len(df) < 60: continue
        raw[code], cover[code] = df, iv
    # D：其餘全市場異常日（單輪；> 50% 官方來源個股的漲跌價差與前收矛盾）
    oks, incs = [], []
    for df in raw.values():
        ok, inc = inconsistent_rows(df); oks.append(df.index[ok]); incs.append(df.index[inc])
    den = pd.Series(np.concatenate(oks)).value_counts(); num = pd.Series(np.concatenate(incs)).value_counts().reindex(den.index).fillna(0)
    frac = (num / den)[den >= 100]
    bad_dates = sorted(set(frac[frac > 0.5].index))
    print('other market-wide inconsistent dates (src 1/2):', [(str(d.date()), round(float(frac[d]), 2)) for d in bad_dates], flush=True)
    for code in raw:
        df = raw[code]; raw[code] = df[~(df.index.isin(bad_dates) & df['src'].isin([1, 2]))]
    bad_dates = sorted(set(bad_dates) | set(asoul_bad))
    CAL = pd.DatetimeIndex(sorted((set(cal) | set().union(*[set(d.index) for d in raw.values()])) - (set(bad_dates) - set().union(*[set(d.index) for d in raw.values()]))))
    CAL = CAL[CAL >= pd.Timestamp('2004-01-01')]
    print('codes with data', len(raw), '| calendar', CAL[0].date(), CAL[-1].date(), len(CAL), flush=True)
    cols = {k: {} for k in ['c', 'o', 'h', 'val', 'src', 'ret', 'flag']}
    for code, df in raw.items():
        df = df[CAL.get_indexer(df.index) >= 0]
        r, flag = compute_returns(df, cover[code], ex_g.get(code), CAL)
        for k, v in [('c', df['c']), ('o', df['o']), ('h', df['h']), ('val', df['val']), ('src', df['src'])]:
            cols[k][code] = v.astype(float)
        cols['ret'][code] = pd.Series(r, index=df.index); cols['flag'][code] = pd.Series(flag.astype(float), index=df.index)
    M = {k: pd.DataFrame(v).reindex(CAL) for k, v in cols.items()}
    keep = M['c'].notna().any(axis=1)          # 去掉所有個股皆無資料的日子（2024-01-02～02-14 缺口）
    M = {k: v.loc[keep] for k, v in M.items()}
    C, R, flag = M['c'], M['ret'], M['flag']
    adj = (1 + R.fillna(0)).cumprod().where(C.notna())
    vs = pd.Series(vol_scale).dropna()
    meta = {'n_delisted_tpex_excluded': len(delisted_tpex), 'n_codes': int(C.shape[1]), 'dates': [str(C.index.min().date()), str(C.index.max().date())], 'n_dates': int(C.shape[0]),
            'n_delisted_unknown_market': len(delisted_unknown), 'market_wide_corrupted_dates_removed': [str(d.date()) for d in bad_dates],
            'tw_stocker_volume_scale_median': round(float(vs.median()), 3) if len(vs) else None, 'tw_stocker_volume_scale_n': int(len(vs)),
            'flag_counts_by_year': {int(y): {int(f): int(n) for f, n in g.stack().value_counts().sort_index().items() if f > 0} for y, g in flag.groupby(flag.index.year)},
            'stocks_with_price_median_by_year': {int(y): int(v) for y, v in C.notna().sum(axis=1).groupby(C.index.year).median().items()},
            'src_share_by_year': {int(y): {int(s): round(float(v), 3) for s, v in g.stack().value_counts(normalize=True).sort_index().items()} for y, g in M['src'].groupby(M['src'].index.year)}}
    for name, mat in [('close_raw', C), ('open', M['o']), ('high', M['h']), ('value', M['val']), ('adj', adj), ('ret', R), ('flag', flag), ('src', M['src'])]:
        mat.to_parquet(OUT / f'{name}.parquet')
    (OUT / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    (OUT / 'delisted_unknown.json').write_text(json.dumps(sorted(delisted_unknown)), encoding='utf-8')
    print(json.dumps({k: v for k, v in meta.items() if k != 'src_share_by_year'}, ensure_ascii=False))

if __name__ == '__main__':
    main()
