#!/usr/bin/env python3
"""閘門母體（不看報酬、不用 V4 股池）：每個換股訊號日 d，以 listing_master 的上市普通股為母體，
用「外部官方成交值」（hschi Trading_money、bacardi turnover、history 收×量、leochen／Tasi 的 OHLC4×量；都沒有時才用 V6 面板 src 1/2/4 官方值）
算 V60（最近 60 個加權指數交易日、只計上市期間、≥ 40 筆）與最後收盤，取 V60 ≥ 1,000 萬、收盤 ≥ 10 的前 300 名＝P(d)。
母體股日＝P(d) × [d, 下一訊號日) 之間「上市且任一來源有成交」的交易日。
保守版另加：上市但在 d 前 60 日完全沒有任何價格資料的股票，以「資料恢復後第一個可算的 V60」代入排名，進前 300 者，其 [d, 下一訊號日) 的所有上市交易日都算進母體（多半沒有官方 OHLC）。
覆蓋＝母體股日中，面板有官方 OHLC（src ∈ {1,2,4,5,6,7} 且開高收齊全；替換窗 2024-01-02～2025-09-23 內另要求 low）的比例。閘門用保守版。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import v7common as C, srcload7 as S

def external_values(cols_all, cal):
    rows = []
    h = S.get('hschi'); rows.append(h[['date', 'code', 'close']].assign(v=h['value'], pr=1))
    b = S.get('bacardi'); rows.append(b[['date', 'code', 'close']].assign(v=b['value'], pr=2))
    for nm, pr in (('history', 3), ('leochen', 4), ('tasi', 5)):
        d = S.get(nm); rows.append(d[['date', 'code', 'close']].assign(v=(d.open + d.high + d.low + d.close) / 4 * d.vol, pr=pr))
    X = pd.concat(rows, ignore_index=True)
    X = X[X.code.isin(set(cols_all))].sort_values(['date', 'code', 'pr']).drop_duplicates(['date', 'code'])
    V = X.pivot(index='date', columns='code', values='v').reindex(index=cal, columns=cols_all)
    P = X.pivot(index='date', columns='code', values='close').reindex(index=cal, columns=cols_all)
    return V, P

def population(sig_days, end_day, conservative=True, a='2020-01-01'):
    LM = pd.read_parquet(C.DATA / 'listed_twse.parquet')
    cal = C.taiex_cal(); cal = cal[(cal >= '2019-06-01') & (cal <= '2026-10-27')]
    cols_all = list(LM.columns)
    V, P = external_values(cols_all, cal)
    cl6 = pd.read_parquet(C.V6DATA / 'close_raw.parquet'); v6 = pd.read_parquet(C.V6DATA / 'value.parquet'); s6 = pd.read_parquet(C.V6DATA / 'src.parquet')
    off6 = s6.isin([1, 2, 4]) & cl6.notna()
    V = V.fillna(v6.where(off6).reindex(index=cal, columns=cols_all)); P = P.fillna(cl6.where(off6).reindex(index=cal, columns=cols_all))
    Lc = LM.reindex(index=cal, columns=cols_all).fillna(False).astype(bool)
    V = V.where(Lc); P = P.where(Lc)
    V60 = V.rolling(60, min_periods=40).mean(); PX = P.ffill()
    anyp = V.notna()
    rows = []; extra = []
    sd = [d for d in sig_days if d >= pd.Timestamp(a)]
    for i, d in enumerate(sd):
        nxt = sd[i + 1] if i + 1 < len(sd) else end_day
        k = cal.searchsorted(d, side='right') - 1
        v = V60.iloc[k]; px = PX.iloc[k]; lst = Lc.iloc[k]
        ok = (v >= 1e7) & (px >= 10) & lst
        vv = v.where(ok)
        if conservative:
            # 上市但近 60 日沒有任何價格：以資料恢復後第一個可算的 V60 代入
            no = lst & (anyp.iloc[max(0, k - 59):k + 1].sum() == 0)
            for c in no[no].index:
                fut = V60[c].iloc[k + 1:k + 250].dropna()
                if len(fut) and fut.iloc[0] >= 1e7:
                    vv[c] = fut.iloc[0]; extra.append((str(d.date()), c, float(fut.iloc[0])))
        top = vv.dropna().sort_values(ascending=False).index[:300]
        prox = {e[1] for e in extra if e[0] == str(d.date())}
        days = cal[(cal >= d) & (cal < nxt)]
        for t in days:
            for c in top:
                if not Lc.at[t, c]: continue
                ta = bool(anyp.at[t, c])
                if ta or c in prox: rows.append((t, c, ta, c in prox, str(d.date())))
    POP = pd.DataFrame(rows, columns=['date', 'code', 'traded_any', 'proxy', 'sig'])
    ex = pd.DataFrame(extra, columns=['sig', 'code', 'proxy_V60'])
    return POP, ex

def coverage(POP, panel_dir, conservative=True):
    P = panel_dir if isinstance(panel_dir, dict) else {k: pd.read_parquet(Path(panel_dir) / f'{k}.parquet') for k in ('close_raw', 'open', 'high', 'low', 'src')}
    idx = P['close_raw'].index; cols = P['close_raw'].columns
    pop = POP[POP.code.isin(cols)].copy()
    out_cols = POP[~POP.code.isin(cols)]
    ii = idx.get_indexer(pop.date); jj = cols.get_indexer(pop.code)
    ok_i = ii >= 0
    def g(k):
        a = np.full(len(pop), np.nan); a[ok_i] = P[k].to_numpy()[ii[ok_i], jj[ok_i]]; return a
    c, o, h, l, s = g('close_raw'), g('open'), g('high'), g('low'), g('src')
    pop['panel_traded'] = np.isfinite(c)
    inw = ((pop.date >= C.W_A) & (pop.date <= C.W_B)).to_numpy()
    # 官方 OHLC：開高收齊全且來源為官方；low 只在替換窗內要求（窗外是 V4 官方行情，V4 面板沒有 low 欄）
    pop['official'] = np.isfinite(c) & np.isfinite(o) & np.isfinite(h) & (np.isfinite(l) | ~inw) & np.isin(s, [1, 2, 4, 5, 6, 7])
    pop = pd.concat([pop, out_cols.assign(panel_traded=False, official=False)])
    # 分母：任一來源有成交的股日；保守版另含代入股（近 60 日無任何資料）的全部上市交易日
    den = pop if conservative else pop[~pop.proxy]
    return pop, {'n_stock_days': int(len(den)), 'official_share': float(den.official.mean()) if len(den) else None,
                 'n_not_official': int((~den.official).sum()),
                 'n_no_data_anywhere': int((~den.traded_any).sum()),
                 'n_traded_but_panel_missing': int((den.traded_any & ~den.panel_traded).sum()),
                 'n_panel_nonofficial': int((den.panel_traded & ~den.official).sum())}
