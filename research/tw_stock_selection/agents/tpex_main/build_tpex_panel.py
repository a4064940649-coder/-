#!/usr/bin/env python3
"""建立上櫃（TPEx）普通股日線面板 2007-04～2026-10，格式比照上市面板（adj, close_raw, open, high, value, ret, flag, src）。

來源（見 prereg.py）：A Asoul/tsec、B taiwanan otc_trading_raw_data、C henry753951/finance（FinMind）、
D hschi1106/stock_GA＋hawan0228/fintech_ai（FinMind，只補 2023-01～09）、E shohokuno10/Tasi kbar、F tw-stock-screener history.csv.gz。
報酬規則：匯入 build_twse_panel.compute_returns（不改原檔）。src 代碼：1＝A、2＝B／C／D（有漲跌價差）、4＝E／F（無漲跌價差，用 exdiv_5y）。
輸出：scratchpad/tpex_panel/*.parquet（reindex 到上市面板日期）＋ meta.json；另存 rows.pkl（各股原始列，供驗證）。
"""
import json, glob, sys, pickle
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path('/home/user/-/research/tw_stock_selection')
sys.path.insert(0, str(ROOT))
import build_twse_panel as BT  # noqa: E402  只用其函式

S = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad')
LS = S / 'tpex_ls'
OUT = S / 'tpex_panel'
SCR = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data')
TW = S / 'panel'
A_END_FULL = pd.Timestamp('2016-12-31')    # Asoul 上櫃 2017 起缺日 → 之後視為未涵蓋
A_USE_END = pd.Timestamp('2014-12-31')     # 有 B 的股票只用 A 到 2014 年底
B_END = pd.Timestamp('2021-04-29'); C_END = pd.Timestamp('2022-12-30')
D_START, D_END = pd.Timestamp('2023-01-01'), pd.Timestamp('2023-09-30')
D2_START, D2_END = pd.Timestamp('2025-04-01'), pd.Timestamp('2025-09-23')   # Tasi 上櫃缺漏期，用 D 補（hschi 到 2025-05-28）
E_GOOD_DAYS = []
E_START, E_END = pd.Timestamp('2023-10-01'), pd.Timestamp('2025-09-23')
F_START = pd.Timestamp('2025-09-24')
is4 = BT.is4


def roc_date(s):
    s = s.astype(str).str.strip()
    p = s.str.split('/', expand=True)
    ok = p.notna().all(axis=1) & p[0].str.isdigit()
    y = pd.to_numeric(p[0].where(ok), errors='coerce') + 1911
    return pd.to_datetime(dict(year=y, month=pd.to_numeric(p[1].where(ok), errors='coerce'), day=pd.to_numeric(p[2].where(ok), errors='coerce')), errors='coerce')


def num(x):
    return pd.to_numeric(x.astype(str).str.replace(',', '').str.strip(), errors='coerce')


def finish(df):
    df = df.dropna(subset=['date', 'c']); df = df[df['c'] > 0]
    df = df.drop_duplicates('date').set_index('date').sort_index()
    return df[['o', 'h', 'c', 'val', 'x', 'chgv']]


def read_A(path):
    d = BT.read_exchange(path, False)
    return d.drop(columns='plus')


def read_B(path):
    df = pd.read_csv(path, skiprows=2, header=None, names=['d', 'vol', 'o', 'h', 'l', 'c', 'chg', 'n', 'val'], dtype=str)
    df['date'] = roc_date(df['d'])
    for k in ['o', 'h', 'c', 'val']: df[k] = num(df[k])
    ch = df['chg'].astype(str).str.strip()
    df['x'] = ch.str.startswith('X')
    df['chgv'] = pd.to_numeric(ch.str.replace('+', '', regex=False).str.replace(',', ''), errors='coerce').where(~df['x'])
    return finish(df)


def read_fm(path):
    try:
        df = pd.read_csv(path, dtype={'stock_id': str}, encoding='utf-8-sig')
    except Exception:
        return pd.DataFrame(columns=['o', 'h', 'c', 'val', 'x', 'chgv'], index=pd.DatetimeIndex([]))
    if not {'close', 'open', 'max', 'Trading_money', 'spread', 'Trading_Volume', 'date'} <= set(df.columns) or len(df) == 0:
        return pd.DataFrame(columns=['o', 'h', 'c', 'val', 'x', 'chgv'], index=pd.DatetimeIndex([]))
    df = df.rename(columns={'open': 'o', 'max': 'h', 'close': 'c', 'Trading_money': 'val', 'spread': 'chgv'})
    df['date'] = pd.to_datetime(df['date'], errors='coerce'); df['x'] = False
    df = df[df['Trading_Volume'] > 0]
    return finish(df)


def load_tasi(cal):
    fs = sorted(glob.glob(str(LS / 'tasi/kbar/kbar*.csv')))
    parts = []
    for f in fs:
        d = pd.Timestamp(Path(f).stem[4:])
        if d not in cal or d < E_START or d > E_END: continue
        try:
            x = pd.read_csv(f, dtype={'stoc': str})
        except Exception:
            continue
        if 'stoc' not in x.columns or len(x) == 0: continue     # 空檔（週末、假日、未抓到）
        x = x[x['stoc'].map(is4)]
        x['date'] = pd.to_datetime(x['date'])
        x = x[x['date'] == d]
        parts.append(x)
    t = pd.concat(parts)
    t = t[(t['vol'] > 0) & (t['clo'] > 0)]
    t = t.rename(columns={'stoc': 'code', 'ope': 'o', 'max': 'h', 'clo': 'c'})
    t['val'] = t['vol'] * 1000 * t['c']; t['x'] = False; t['chgv'] = np.nan
    # 同一檔連續兩日 OHLC 與張數完全相同（可能是複製前一日的檔），視為缺值
    t = t.sort_values(['code', 'date'])
    same = (t[['o', 'h', 'c', 'vol']].values == t.groupby('code')[['o', 'h', 'c', 'vol']].shift().values).all(axis=1)
    t = t[~same]
    # Tasi 2025-04 起上櫃股大量缺漏（多日只有 0～440 檔）：只保留「現存上櫃清單中有值檔數 ≥ 700」的日子
    tp = set(pd.read_csv(SCR / 'stock_list.csv', dtype=str).query("market=='TPEX'")['code'])
    n_tp = t[t['code'].isin(tp)].groupby('date').size()
    good = set(n_tp[n_tp >= 700].index)
    t = t[t['date'].isin(good)]
    global E_GOOD_DAYS
    E_GOOD_DAYS = sorted(good)
    return {c: g.set_index('date')[['o', 'h', 'c', 'val', 'x', 'chgv']].sort_index() for c, g in t.groupby('code')}


def main():
    OUT.mkdir(exist_ok=True)
    twse_list = pd.read_csv(S / 'piptest/twstock_wheel/twstock/codes/twse_equities.csv', dtype=str)
    tpex_list = pd.read_csv(S / 'piptest/twstock_wheel/twstock/codes/tpex_equities.csv', dtype=str)
    scr_list = pd.read_csv(SCR / 'stock_list.csv', dtype=str)
    twse_now = {c for c in twse_list.loc[twse_list['type'] == '股票', 'code'] if is4(c)} | {c for c in scr_list.loc[scr_list['market'] == 'TWSE', 'code'] if is4(c)}
    tpex_now = {c for c in tpex_list.loc[tpex_list['type'] == '股票', 'code'] if is4(c)} | {c for c in scr_list.loc[scr_list['market'] == 'TPEX', 'code'] if is4(c)}
    pe = pd.read_csv(SCR / 'extras/pe_5y.csv.gz', dtype={'code': str}, usecols=['date', 'code', 'market'])
    pe_last = pe.sort_values('date').drop_duplicates('code', keep='last').set_index('code')['market']
    pe_tpex = set(pe_last[pe_last == 'TPEX'].index); pe_twse = set(pe_last[pe_last == 'TWSE'].index)
    tw_adj = pd.read_parquet(TW / 'adj.parquet'); TWIDX = tw_adj.index
    tw_first = tw_adj.apply(lambda s: s.first_valid_index())          # 上市面板中各代碼第一個有值日（轉上市日）
    cal = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date'])['date']
    CAL0 = pd.DatetimeIndex(sorted(set(cal[cal >= '2004-01-01']) | set(TWIDX)))

    asoul = {Path(p).stem: p for p in glob.glob(str(S / 'tsec_all/*.csv')) if is4(Path(p).stem)}
    # Asoul 中的上櫃股：上漲日帶「+」比例 > 0.3（build_twse_panel 的判定），只取第一個帶「+」到最後一個帶「+」之間
    A_tp = {}
    for c, p in asoul.items():
        a = BT.read_exchange(p, False)
        pos = a['chgv'] > 0
        if pos.sum() == 0: continue
        if a['plus'].sum() / pos.sum() > 0.3 or (a['plus'].any() and c in tw_first.index):
            if not a['plus'].any(): continue
            lastp = a.index[a['plus'].to_numpy()].max()
            a = a.loc[:lastp]
            if c in tw_first.index and pd.notna(tw_first[c]): a = a.loc[a.index < tw_first[c]]
            if len(a): A_tp[c] = a.drop(columns='plus')
    B_tp = {Path(p).stem: p for p in glob.glob(str(LS / 'twan/otc_trading_raw_data/*.csv')) if is4(Path(p).stem)}
    henry = {Path(p).stem: p for p in glob.glob(str(LS / 'henry/data/stocks/*.csv')) if is4(Path(p).stem)}
    hschi = {Path(p).stem: p for p in glob.glob('/home/user/hschi1106/stock_ga/data/stock_dayK/*.csv') if is4(Path(p).stem)}
    hawan = {Path(p).stem.split('_')[0]: p for p in glob.glob('/home/user/hawan0228/fintech_ai/data/external/raw/prices/*_price.csv')}
    E = load_tasi(set(CAL0))
    hist = pd.read_csv(SCR / 'history.csv.gz', dtype={'code': str}, parse_dates=['date'])
    hist = hist[hist['code'].map(is4) & (hist['volume'] > 0) & (hist['close'] > 0)]
    F = {c: pd.DataFrame({'o': g['open'], 'h': g['high'], 'c': g['close'], 'val': g['volume'] * g['close'], 'x': False, 'chgv': np.nan}).set_index(g['date']).sort_index()
         for c, g in hist.groupby('code')}

    # 上櫃代碼集合：Asoul 上櫃、taiwanan 上櫃、現存上櫃清單、pe_5y 最後市場別為 TPEX；
    # 另外 Tasi／history 中不在任何上市清單、也不在上市面板的代碼視為上櫃（2023-10 後下櫃者）
    tp_codes = set(A_tp) | set(B_tp) | tpex_now | pe_tpex
    extra = {c for c in set(E) | set(F) if c not in twse_now and c not in pe_twse and c not in tw_first.index}
    print('extra not in tpex_now/pe_tpex/A/B:', sorted(extra - tpex_now - pe_tpex - set(A_tp) - set(B_tp)), flush=True)
    tp_codes |= extra
    print('codes: Asoul', len(A_tp), 'taiwanan', len(B_tp), 'tpex_now', len(tpex_now), 'pe_tpex', len(pe_tpex), 'extra(Tasi/hist)', len(extra), 'total', len(tp_codes), flush=True)

    ex = pd.read_csv(SCR / 'extras/exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
    ex = ex[ex['code'].map(is4) & (ex['ref_price'] > 0) & (ex['prev_close'] > 0)].copy()
    ex['f'] = ex['prev_close'] / ex['ref_price']; ex = ex[(ex['f'] >= 1.0) & (ex['f'] < 3.0)]
    ex_g = {c: g.sort_values('date') for c, g in ex.groupby('code')}

    raw, cover, srcmap = {}, {}, {}
    for code in sorted(tp_codes):
        cut = tw_first[code] if code in tw_first.index and pd.notna(tw_first[code]) else pd.Timestamp('2100-01-01')
        parts, iv = [], []
        hasB = code in B_tp
        # 依優先序：B > A > C > D > E > F；每個來源只在自己的期間
        if hasB:
            b = read_B(B_tp[code]).loc[:B_END]
            if len(b): b['src'] = 2; b['s'] = 'B'; parts.append(b); iv.append((b.index[0], B_END))
        if code in A_tp:
            a = A_tp[code]
            a = a.loc[:A_USE_END] if hasB else a
            a = a[~a.index.isin(pd.to_datetime(['2017-08-03', '2018-07-12']))]   # 上市面板判定的 Asoul 錯置日
            if len(a):
                a = a.copy(); a['src'] = 1; a['s'] = 'A'; parts.append(a); iv.append((a.index[0], min(a.index[-1], A_END_FULL)))
        if code in henry:
            c_ = read_fm(henry[code])
            c_ = c_.loc[(c_.index > (B_END if hasB else pd.Timestamp('2014-12-31'))) & (c_.index <= C_END)]
            if len(c_): c_['src'] = 2; c_['s'] = 'C'; parts.append(c_); iv.append((c_.index[0], C_END))
        dd = []
        for srcd in (hschi, hawan):
            if code in srcd:
                x_ = read_fm(srcd[code])
                d_ = pd.concat([x_.loc[D_START:D_END], x_.loc[D2_START:D2_END]])
                if len(d_): dd.append(d_)
        if dd:
            d_ = pd.concat(dd); d_ = d_[~d_.index.duplicated()].sort_index(); d_['src'] = 2; d_['s'] = 'D'; parts.append(d_)
            if (d_.index <= D_END).any(): iv.append((d_.index[d_.index >= D_START][0], D_END))
            if (d_.index >= D2_START).any(): iv.append((d_.index[d_.index >= D2_START][0], d_.index[-1]))
        if code in E:
            e = E[code].copy(); e['src'] = 4; e['s'] = 'E'; parts.append(e)
            gd = pd.DatetimeIndex([d for d in E_GOOD_DAYS if d >= e.index[0]])
            if len(gd):   # 涵蓋區間＝Tasi 有效日的連續段（中間缺日 > 3 個交易日即斷開）
                pos = CAL0.get_indexer(gd); brk = np.where(np.diff(pos) > 3)[0]
                st = np.r_[0, brk + 1]; en = np.r_[brk, len(gd) - 1]
                for a_, b_ in zip(st, en): iv.append((gd[a_], gd[b_]))
        if code in F:
            f = F[code].loc[F_START:].copy()
            if len(f): f['src'] = 4; f['s'] = 'F'; parts.append(f); iv.append((F_START, pd.Timestamp('2030-01-01')))
        if not parts: continue
        df = pd.concat(parts)
        df = df[~df.index.duplicated(keep='first')].sort_index()
        df = df.loc[(df.index >= pd.Timestamp('2004-01-01')) & (df.index < cut)]
        df = df[df.index.isin(CAL0)]
        if len(df) < 60: continue
        raw[code] = df; cover[code] = iv
    print('codes with data', len(raw), flush=True)
    cols = {k: {} for k in ['c', 'o', 'h', 'val', 'src', 'ret', 'flag']}
    for code, df in raw.items():
        r, flag = BT.compute_returns(df, cover[code], ex_g.get(code), CAL0)
        for k in ['c', 'o', 'h', 'val', 'src']:
            cols[k][code] = df[k].astype(float)
        cols['ret'][code] = pd.Series(r, index=df.index); cols['flag'][code] = pd.Series(flag.astype(float), index=df.index)
    M = {k: pd.DataFrame(v).reindex(CAL0) for k, v in cols.items()}
    adj_full = (1 + M['ret'].fillna(0)).cumprod().where(M['c'].notna())
    # 輸出到上市面板日期（adj 保留全日曆累積，跨過被刪掉的日子不會遺失報酬）
    out = {'close_raw': M['c'], 'open': M['o'], 'high': M['h'], 'value': M['val'], 'adj': adj_full, 'ret': M['ret'], 'flag': M['flag'], 'src': M['src']}
    for k, v in out.items():
        v.reindex(TWIDX).to_parquet(OUT / f'{k}.parquet')
    # ret 在上市日曆上：用 adj 重算（被刪掉日子的報酬併入下一個保留日）
    a2 = adj_full.reindex(TWIDX)
    (a2 / a2.ffill().shift() - 1).where(a2.notna()).to_parquet(OUT / 'ret.parquet')
    srcs = pd.DataFrame({c: df['s'] for c, df in raw.items()}).reindex(TWIDX)
    srcs.to_parquet(OUT / 'srcname.parquet')
    C = M['c'].reindex(TWIDX); flag = M['flag'].reindex(TWIDX)
    delisted = sorted(c for c in raw if c not in tpex_now and c not in twse_now)
    meta = {'n_codes': len(raw), 'n_transfer_to_twse': int(sum(1 for c in raw if c in tw_first.index)), 'n_not_currently_listed': len(delisted),
            'dates': [str(C.dropna(how='all').index.min().date()), str(C.dropna(how='all').index.max().date())],
            'stocks_with_price_median_by_year': {int(y): int(v) for y, v in C.notna().sum(axis=1).groupby(C.index.year).median().items()},
            'not_currently_listed_with_price_median_by_year': {int(y): int(v) for y, v in C[delisted].notna().sum(axis=1).groupby(C.index.year).median().items()},
            'flag_counts_by_year': {int(y): {int(f): int(n) for f, n in g.stack().value_counts().sort_index().items() if f > 0} for y, g in flag.groupby(flag.index.year)},
            'src_share_by_year': {int(y): {str(s): round(float(v), 3) for s, v in g.stack().value_counts(normalize=True).sort_index().items()} for y, g in srcs.groupby(srcs.index.year)}}
    (OUT / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    with open(OUT / 'rows.pkl', 'wb') as fh: pickle.dump({'raw': raw, 'cover': cover}, fh)
    print(json.dumps({k: v for k, v in meta.items() if k not in ('src_share_by_year', 'flag_counts_by_year')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
