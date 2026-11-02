#!/usr/bin/env python3
"""第 0 步：上市＋上櫃合併面板、補洞、PIT 市場別、gap_map.json、存活者偏誤表。
只讀：V7 凍結上市面板（o_v7_listing_exec_engine/data）、tpex_main 上櫃面板（scratchpad/tpex_panel）、
V7 資料夾內已封存的 GitHub 來源（src_hschi＝hschi1106/stock_GA、src_tasi＝shohokuno10/Tasi、src_bacardi_tpex＝bacardi1117-ai）、
ISIN 封存（src_isin）、exdiv_5y、tw_revenue.csv、stock_list.csv。輸出：scratchpad/o_tpex_ext/panel/*.parquet、data/gap_map.json、data/build_meta.json。
用法：python3 -I -B s0_build.py"""
import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
V7D = AG / 'o_v7_listing_exec_engine' / 'data'
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad')
TPP = SCR / 'tpex_panel'
OUT = SCR / 'o_tpex_ext' / 'panel'
EXT = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data')
DATA = HERE / 'data'
OUT.mkdir(parents=True, exist_ok=True); DATA.mkdir(exist_ok=True)
F = ['close_raw', 'open', 'high', 'value', 'ret', 'flag']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    TW = {k: pd.read_parquet(V7D / f'{k}.parquet') for k in F}
    IDX = TW['close_raw'].index
    TP = {k: pd.read_parquet(TPP / f'{k}.parquet').reindex(IDX) for k in F}
    tw_cols = set(TW['close_raw'].columns); tp_cols = list(TP['close_raw'].columns)
    meta = {'sha': {'v7_close': sha(V7D / 'close_raw.parquet'), 'tp_close': sha(TPP / 'close_raw.parquet')}}
    # 上市首日（V7 面板有值的第一天）：轉上市股的上櫃期間只取這天之前
    tw_first = {c: TW['close_raw'][c].first_valid_index() for c in tp_cols if c in tw_cols}
    ex = pd.read_csv(EXT / 'extras' / 'exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
    exref = {(c, d): r for c, d, r in zip(ex.code, ex.date, ex.ref_price)}
    fills = {}
    # ---------- 補洞：hschi（2021-05～2023-09）、Tasi（2023-10～2025-03 缺值格，主要是 2024-01-02～02-14）、bacardi（2025-04～09-23）----------
    plan = [('hschi', 'src_hschi.parquet', '2021-05-01', '2023-09-30'),
            ('tasi', 'src_tasi.parquet', '2023-10-01', '2025-03-31'),
            ('bacardi', 'src_bacardi_tpex.parquet', '2025-04-01', '2025-09-23')]
    tpset = set(tp_cols)
    ORIG = TP['close_raw'].notna().to_numpy().copy()
    for name, f, a, b in plan:
        s = pd.read_parquet(V7D / f)
        s = s[s.code.isin(tpset)]
        # 重疊期一致率（全期間、該來源與現有面板都有值的格）
        Cp = TP['close_raw']
        ii = IDX.get_indexer(pd.DatetimeIndex(s.date)); jj = np.array([tp_cols.index(c) for c in s.code])
        okk = ii >= 0
        cur = Cp.to_numpy()[ii[okk], jj[okk]]; new = s['close'].to_numpy()[okk]
        both = np.isfinite(cur) & np.isfinite(new)
        # 代碼撞號（例：hschi 的 6208 是 ETF 006208 去掉前導 0）：該代碼重疊格不一致率 > 20% → 整檔不採用
        badc = pd.Series(np.abs(cur[both] - new[both]) > 0.0049).groupby(s['code'].to_numpy()[okk][both]).mean()
        collide = sorted(badc[badc > 0.2].index)
        agree_all = float(np.mean(np.abs(cur[both] - new[both]) <= 0.0049)) if both.any() else None
        keepc = ~np.isin(s['code'].to_numpy()[okk], collide)
        both = both & keepc
        s = s[~s.code.isin(collide)]
        # 轉上市股在上市期間的列不比（上櫃面板只有上櫃期）
        agree = float(np.mean(np.abs(cur[both] - new[both]) <= 1e-6 * np.maximum(1, np.abs(cur[both])) + 0.0049)) if both.any() else None
        s = s[(s.date >= a) & (s.date <= b)].copy()
        s['ii'] = IDX.get_indexer(pd.DatetimeIndex(s.date)); s['jj'] = [tp_cols.index(c) for c in s.code]
        s = s[s.ii >= 0]
        twf = np.array([tw_first.get(c) or pd.Timestamp('2100-01-01') for c in s.code], dtype='datetime64[ns]')
        s = s[(s.date.values.astype('datetime64[ns]') < twf) & np.isfinite(s.close) & (s.close > 0)]
        Cn = TP['close_raw'].to_numpy(copy=True); On = TP['open'].to_numpy(copy=True); Hn = TP['high'].to_numpy(copy=True)
        Vn = TP['value'].to_numpy(copy=True); Rn = TP['ret'].to_numpy(copy=True); Fn = TP['flag'].to_numpy(copy=True)
        ii = s.ii.to_numpy(); jj = s.jj.to_numpy()
        nw = ~np.isfinite(Cn[ii, jj]); s = s[nw]; ii, jj = ii[nw], jj[nw]
        cl = s.close.to_numpy(float); o = s.open.to_numpy(float); h = s.high.to_numpy(float)
        val = s.value.to_numpy(float); vol = s.vol.to_numpy(float)
        v = np.where(np.isfinite(val) & (val > 0), val, vol * cl)   # src_tasi 的 vol 已是股數（V7 封存口徑）
        Cn[ii, jj] = cl; On[ii, jj] = np.where(np.isfinite(o) & (o > 0), o, np.nan); Hn[ii, jj] = h; Vn[ii, jj] = v
        n_fill = len(s)
        # 報酬：除權息參考價（exdiv_5y）優先；其次官方漲跌（chg≠0，Tasi 無）；否則前收
        PCn = pd.DataFrame(Cn).ffill().shift(1).to_numpy()
        pc = PCn[ii, jj]
        er = np.array([exref.get((c, d), np.nan) for c, d in zip(s.code, s.date)], float)
        ch = s.chg.to_numpy(float)
        use_ch = np.isfinite(ch) & (ch != 0) & (name != 'tasi')
        ref = np.where(np.isfinite(er), er, np.where(use_ch, cl - ch, pc))
        fl = np.where(np.isfinite(er), 2, np.where(use_ch, 0, np.where(np.isfinite(pc), 1, 3)))
        r = cl / ref - 1
        bad = ~np.isfinite(r) | (np.abs(r) > 0.105) & (s.date.values >= np.datetime64('2015-06-01')) | (np.abs(r) > 0.075) & (s.date.values < np.datetime64('2015-06-01'))
        fl = np.where(bad, 3, fl); r = np.where(np.isfinite(r) & ~bad, r, 0.0)   # 超過漲跌幅（減資、復牌參考價不符）→ 報酬記 0、flag 3（同上櫃面板慣例）
        keep = ~np.isfinite(Rn[ii, jj])
        Rn[ii[keep], jj[keep]] = r[keep]; Fn[ii[keep], jj[keep]] = fl[keep]
        for k, A in (('close_raw', Cn), ('open', On), ('high', Hn), ('value', Vn), ('ret', Rn), ('flag', Fn)):
            TP[k] = pd.DataFrame(A, index=IDX, columns=tp_cols)
        fills[name] = {'n_cells_filled': int(n_fill), 'overlap_close_agree': agree, 'overlap_close_agree_incl_collision': agree_all, 'collision_codes_dropped': collide, 'overlap_n': int(both.sum()), 'window': [a, b]}
        print(name, fills[name], flush=True)
    meta['fills'] = fills
    # 補洞後修正：原面板在缺口之後第一個交易日的報酬是「跨缺口」累積報酬（相對缺口前最後收盤），
    # 缺口補上後會重複計算 → 這些格改以補值後的前一筆收盤（或除權息參考價）重算。（2026-11-02 PREREG 後發現的資料錯誤）
    Cn = TP['close_raw'].to_numpy(copy=True); Rn = TP['ret'].to_numpy(copy=True); Fn = TP['flag'].to_numpy(copy=True)
    NEWV = np.isfinite(Cn); T_, N_ = Cn.shape
    ar = np.arange(T_)[:, None] * np.ones((1, N_))
    lastnew = pd.DataFrame(np.where(NEWV, ar, np.nan)).ffill().shift(1).to_numpy()
    lastold = pd.DataFrame(np.where(ORIG, ar, np.nan)).ffill().shift(1).to_numpy()
    fix = ORIG & NEWV & np.isfinite(lastnew) & (np.nan_to_num(lastnew, nan=-1) != np.nan_to_num(lastold, nan=-1))
    ii, jj = np.where(fix)
    pc = Cn[lastnew[ii, jj].astype(int), jj]
    er = np.array([exref.get((tp_cols[j], IDX[i]), np.nan) for i, j in zip(ii, jj)], float)
    ref = np.where(np.isfinite(er), er, pc)
    r = Cn[ii, jj] / ref - 1
    lim = np.where(IDX[ii] >= pd.Timestamp('2015-06-01'), 0.105, 0.075)
    badf = ~np.isfinite(r) | (np.abs(r) > lim)
    Rn[ii, jj] = np.where(badf, 0.0, r); Fn[ii, jj] = np.where(badf, 3, np.where(np.isfinite(er), 2, 1))
    TP['ret'] = pd.DataFrame(Rn, index=IDX, columns=tp_cols); TP['flag'] = pd.DataFrame(Fn, index=IDX, columns=tp_cols)
    meta['postfill_ret_fix'] = dict(n_cells=int(len(ii)), by_year={int(k): int(v) for k, v in pd.Series(IDX[ii].year).value_counts().sort_index().items()},
                                    note='跨缺口累積報酬重算（PREREG 後發現；只修資料，不改規格）')
    print('postfill fix', meta['postfill_ret_fix'], flush=True)
    # ---------- 合併：上市（V7）＋上櫃；轉上市股的上櫃期間接在上市之前 ----------
    only_tp = [c for c in tp_cols if c not in tw_cols]
    cols = list(TW['close_raw'].columns) + only_tp
    M = {}
    for k in F:
        A = TW[k].reindex(columns=cols).copy()
        A[only_tp] = TP[k][only_tp]
        M[k] = A
    MKT = pd.DataFrame(False, index=IDX, columns=cols)
    MKT[only_tp] = TP['close_raw'][only_tp].notna() | True
    trans = {}
    for c, fd in tw_first.items():
        if fd is None: continue
        pre = TP['close_raw'][c].loc[:fd - pd.Timedelta(days=1)].dropna()
        if not len(pre): continue
        mask = IDX < fd
        for k in F:
            M[k].loc[mask, c] = TP[k].loc[mask, c]
        MKT.loc[mask, c] = True
        # 上市首日報酬：若缺值，以上櫃末日收盤為參考價
        if not np.isfinite(M['ret'].at[fd, c]):
            M['ret'].at[fd, c] = M['close_raw'].at[fd, c] / pre.iloc[-1] - 1; M['flag'].at[fd, c] = 1
        trans[c] = {'tpex_last': str(pre.index[-1].date()), 'twse_first': str(fd.date())}
    meta['n_transfer'] = len(trans); meta['transfers'] = trans
    # MKT 只在有資料的上櫃期間為 True；之後無資料也維持上櫃身分（下櫃前）
    MKT = MKT.astype(bool)
    for k in F:
        M[k].astype(float).to_parquet(OUT / f'{k}.parquet')
    MKT.to_parquet(OUT / 'mkt_tpex.parquet')
    meta['cols'] = len(cols); meta['n_tw'] = len(tw_cols); meta['n_tp_only'] = len(only_tp)
    # ---------- gap_map ----------
    Ct = M['close_raw'].where(MKT)          # 只看上櫃期間
    has = Ct.notna()
    first = has.idxmax().where(has.any()); last = has[::-1].idxmax().where(has.any())
    months = pd.period_range('2007-05', '2026-10', freq='M')
    alive_end = {}
    gm = {}
    for m in months:
        a, b = m.start_time, m.end_time
        sub = has.loc[a:b]
        if not len(sub): gm[str(m)] = {'n_days': 0}; continue
        nprice = int(sub.any().sum())
        med_daily = float(sub.sum(axis=1).median())
        alive = int(((first <= b) & (last >= a)).sum())
        # 上櫃期間的轉上市股：轉換後不算
        gm[str(m)] = {'n_days': int(len(sub)), 'n_with_price': nprice, 'median_daily': med_daily, 'n_alive_bracket': alive,
                      'coverage': round(med_daily / alive, 4) if alive else None}
    # ISIN 官方錨點（2023-01-16 上櫃股票 809 檔）
    isin = pd.read_parquet(V7D / 'src_isin.parquet')
    iso = isin[(isin.version == '2023-01-16') & (isin.market == '上櫃') & (isin.group == '股票')]
    for mm in ('2022-12', '2023-01'):
        if mm in gm: gm[mm]['isin_official_2023_01_16'] = int(len(iso)); gm[mm]['coverage_vs_isin'] = round(gm[mm]['median_daily'] / len(iso), 4)
    # 營收覆蓋：上櫃有價格且 60 日均值 ≥ 1,000 萬的股票中，有「上月營收」的比例
    rv = pd.read_csv('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv', dtype={'code': str})
    rv['p'] = pd.PeriodIndex(pd.to_datetime(rv['month']), freq='M')
    rvs = rv.groupby('p').code.apply(set).to_dict()
    V60 = M['value'].where(MKT).rolling(60, min_periods=40).mean()
    for m in months:
        if str(m) not in gm or gm[str(m)].get('n_days', 0) == 0: continue
        a = m.start_time
        t = IDX.searchsorted(a)
        if t >= len(IDX): continue
        el = V60.iloc[t]; el = set(el[el >= 1e7].index)
        rs = rvs.get(m - 1, set())
        gm[str(m)]['rev_cov_liquid'] = round(len(el & rs) / len(el), 4) if el else None
        gm[str(m)]['n_liquid'] = len(el)
    for k, v in gm.items():
        cv = v.get('coverage')
        v['hole'] = bool(cv is not None and cv < 0.85) or v.get('n_days', 0) == 0
    holes = [k for k, v in gm.items() if v['hole']]
    meta['holes'] = holes
    # ---------- 下櫃股（存活者偏誤）----------
    sl = pd.read_csv(EXT / 'stock_list.csv', dtype=str)
    listed_now = set(sl.code)
    endp = IDX[-1] - pd.Timedelta(days=30)
    dl = [c for c in only_tp if pd.notna(last[c]) and last[c] < endp and c not in listed_now]
    dly = pd.Series({c: last[c].year for c in dl}).value_counts().sort_index()
    alive_y = {y: int(((first <= pd.Timestamp(f'{y}-12-31')) & (last >= pd.Timestamp(f'{y}-01-01'))).sum()) for y in range(2008, 2027)}
    rate = {int(y): round(float(dly.get(y, 0)) / alive_y[y], 4) for y in range(2008, 2027)}
    base_rate = float(np.mean([rate[y] for y in range(2008, 2017)]))
    exp_missing = {int(y): round(base_rate * alive_y[y] - float(dly.get(y, 0)), 1) for y in range(2019, 2027)}
    surv = {'delisted_found_by_last_year': {int(k): int(v) for k, v in dly.items()}, 'alive_by_year': alive_y, 'delist_rate': rate,
            'base_rate_2008_2016': base_rate, 'expected_missing_2019_2026': exp_missing,
            'expected_missing_total': round(sum(max(0, v) for v in exp_missing.values()), 1),
            'note': '櫃買官方終止上櫃名單無法取得（tpex.org.tw 被代理擋、GitHub 找不到鏡像）；以 2008～2016 有完整下櫃資料期間的年下櫃率外推 2019 年後應有下櫃數，差額＝缺漏估計。'}
    meta['survivorship'] = surv
    meta['delisted_tp_codes'] = dl
    (DATA / 'gap_map.json').write_text(json.dumps(gm, ensure_ascii=False, indent=0))
    (DATA / 'build_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=0, default=str))
    print('holes', holes)
    print('surv', json.dumps(surv, ensure_ascii=False)[:800])


if __name__ == '__main__':
    main()
