#!/usr/bin/env python3
"""複製員 o6：依凍結規格第一節自己重建 V6 面板（不讀研究員程式）。
輸入：V4 面板（scratchpad/panel）、replicate_o6/data/src_*.parquet（r1_sources.py）、exdiv_5y、corp_actions、TAIEX 日曆。
輸出：replicate_o6/data/{adj,close_raw,open,high,low,value,ret,flag,src}.parquet。
low：窗內＝官方 low；窗外先沿用研究員以外的來源較費時，這裡窗外 low 只做 hschi/bacardi/leochen（收盤相同才採用），
     henry（2005～2022）另在 r2b_low.py 補（只影響 low 比對，不影響策略）。"""
from pathlib import Path
import json, sys
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
D_OUT = HERE / 'data'
P4 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
XT = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
TX = Path('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv')
WA, WB = pd.Timestamp('2024-01-02'), pd.Timestamp('2025-09-23')
LIM = 0.105

def main(args=()):
    vacuous_v0 = 'vol0_final' in args     # 對照：成交量 0 的列直接當「沒有成交」、不往下一來源找
    # 'spec'（預設）：事件日若是休市日（颱風停市），照規格「該股沒有成交」移到 5 日內第一個成交日
    # 'compat'：休市日（不在面板日曆）的事件不移動、直接不用（＝研究員面板的實際行為，比對後發現）
    mode = 'compat' if 'compat' in args else 'spec'
    V4 = {k: pd.read_parquet(P4 / f'{k}.parquet') for k in ('adj', 'close_raw', 'open', 'high', 'value', 'ret', 'flag', 'src')}
    COLS = V4['close_raw'].columns; I4 = V4['close_raw'].index
    tx = pd.to_datetime(pd.read_csv(TX)['date'])
    txw = pd.DatetimeIndex(tx[(tx >= WA) & (tx <= WB)])
    IDX = I4.union(txw).sort_values()
    newd = IDX.difference(I4)
    print('new dates', len(newd), newd.min(), newd.max())
    T, N = len(IDX), len(COLS)
    win = (IDX >= WA) & (IDX <= WB)
    WD = IDX[win]
    V = {k: v.reindex(IDX) for k, v in V4.items()}

    # ---------- 來源 ----------
    def piv(df, col):
        x = df.pivot(index='date', columns='code', values=col)
        return x.reindex(index=WD, columns=COLS)
    srcs = {}
    for name in ('hschi', 'bacardi', 'leochen', 'tasi'):
        d = pd.read_parquet(D_OUT / f'src_{name}.parquet')
        d = d[d['code'].isin(set(COLS)) & d['date'].isin(set(WD))]
        if name == 'bacardi':
            raw_twse = d[d['market'] == 'TWSE']
            d = raw_twse
        ok = (d['vol'] > 0) & (d[['o', 'h', 'l', 'c']] > 0).all(axis=1)
        srcs[name] = dict(all=d, ok=d[ok])
    # 上市身分：leochen 有列、bacardi TWSE 有列、V4 src 3
    exist = lambda name: piv(srcs[name]['all'].assign(one=1.0), 'one').notna().to_numpy()
    v4src3 = (V['src'].loc[WD].to_numpy() == 3)
    listed = exist('leochen') | exist('bacardi') | v4src3
    jb = (WD >= pd.Timestamp('2024-01-02')) & (WD <= pd.Timestamp('2024-02-14'))
    a1 = V['close_raw'].loc['2023-12-21':'2023-12-29'].notna().any(axis=0).to_numpy()
    a2 = V['close_raw'].loc['2024-02-15':'2024-02-22'].notna().any(axis=0).to_numpy()
    listed[jb] |= (a1 & a2)[None, :]

    # ---------- OHLC 選取 ----------
    order = [('hschi', 5), ('bacardi', 7), ('leochen', 7), ('tasi', 6)]
    o = np.full((len(WD), N), np.nan); h = o.copy(); l = o.copy(); c = o.copy(); vol = o.copy(); srcw = o.copy()
    sname = np.full((len(WD), N), '', dtype=object)
    blocked = np.zeros((len(WD), N), bool)     # vol0_final 對照用
    for name, code in order:
        okd = srcs[name]['ok']
        P = {k: piv(okd, k).to_numpy() for k in ('o', 'h', 'l', 'c', 'vol')}
        has = np.isfinite(P['c']) & listed & np.isnan(c) & ~blocked
        for arr, k in ((o, 'o'), (h, 'h'), (l, 'l'), (c, 'c')): arr[has] = P[k][has]
        vol[has] = P['vol'][has] * (1000.0 if name == 'tasi' else 1.0)
        srcw[has] = code; sname[has] = name
        if vacuous_v0:
            al = piv(srcs[name]['all'], 'c').to_numpy()
            blocked |= np.isfinite(al) & ~np.isfinite(P['c']) & np.isnan(c)
    # 最後備援：V4 src 3
    fb = np.isnan(c) & v4src3 & ~blocked
    for arr, k in ((o, 'open'), (h, 'high'), (c, 'close_raw')):
        arr[fb] = V[k].loc[WD].to_numpy()[fb]
    srcw[fb] = 3; sname[fb] = 'src3'
    print('src counts window', pd.Series(srcw.ravel()).value_counts().to_dict())

    # ---------- 成交值 ----------
    val = np.full((len(WD), N), np.nan); vsrc = np.full((len(WD), N), '', dtype=object)
    for name, mcol in (('hschi', 'money'), ('bacardi', 'money')):
        al = srcs[name]['all']
        pc = piv(al, 'c').to_numpy(); pm = piv(al, 'money').to_numpy()
        use = np.isnan(val) & np.isfinite(c) & np.isfinite(pc) & (np.abs(pc - c) < 1e-9) & np.isfinite(pm) & (sname != 'src3')
        val[use] = pm[use]; vsrc[use] = name
    est = np.isnan(val) & np.isfinite(c) & np.isfinite(vol) & (sname != 'src3')
    val[est] = ((o + h + l + c) / 4 * vol)[est]; vsrc[est] = 'est'
    fbv = np.isnan(val) & np.isfinite(c)
    val[fbv] = V['value'].loc[WD].to_numpy()[fbv]; vsrc[fbv] = 'src3'
    print('value src', pd.Series(vsrc[np.isfinite(c)]).value_counts().to_dict())

    # ---------- 合成全期 OHLC ----------
    C = V['close_raw'].to_numpy().copy(); O = V['open'].to_numpy().copy(); H = V['high'].to_numpy().copy()
    VAL = V['value'].to_numpy().copy(); SRC = V['src'].to_numpy().copy()
    wi = np.where(win)[0]
    C[wi] = c; O[wi] = o; H[wi] = h; VAL[wi] = val; SRC[wi] = srcw
    L = np.full((T, N), np.nan); L[wi] = l
    RET = V['ret'].to_numpy().copy(); FLAG = V['flag'].to_numpy().copy()
    RET[wi] = np.nan; FLAG[wi] = np.nan

    # ---------- 事件 ----------
    ex = pd.read_csv(XT / 'exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
    ex = ex[(ex['market'] == 'TWSE')][['date', 'code', 'prev_close', 'ref_price']].assign(kind='exdiv')
    ca = pd.read_csv(XT / 'corp_actions.csv', dtype={'code': str}, parse_dates=['date'])
    ca = ca[(ca['market'] == 'TWSE')][['date', 'code', 'prev_close', 'ref_price']].assign(kind='corp')
    ev = pd.concat([ex, ca], ignore_index=True)
    ev = ev[ev['code'].isin(set(COLS)) & (ev['date'] >= WA - pd.Timedelta(days=10)) & (ev['date'] <= WB + pd.Timedelta(days=10))]
    CI = {k: j for j, k in enumerate(COLS)}
    TR = np.isfinite(C)
    # 前一筆收盤（跨停牌）與相隔交易日
    prevC = np.full((T, N), np.nan); gap = np.zeros((T, N), int)
    last = np.full(N, np.nan); lasti = np.full(N, -1)
    nobs = np.zeros((T, N), int); cnt = np.zeros(N, int)
    for t in range(T):
        prevC[t] = last; gap[t] = np.where(lasti >= 0, t - lasti, 0)
        tr = TR[t]
        cnt[tr] += 1; nobs[t] = cnt
        last = np.where(tr, C[t], last); lasti = np.where(tr, t, lasti)
    st = dict(exdiv_in=0, corp_in=0, moved_notrade=0, moved_lastcum=0, rejected=0, dropped=0, used=0)
    evref = {}
    for r in ev.itertuples(index=False):
        j = CI[r.code]
        t = IDX.searchsorted(r.date)
        if t >= T: continue
        if not (WA <= IDX[t] <= WB) and not (IDX[t] == r.date): pass
        # 事件日沒有成交 → 5 個交易日內第一個成交日
        if IDX[t] != r.date and mode == 'compat':
            st['dropped'] += 1; continue
        if IDX[t] != r.date or not TR[t, j]:
            tt = [u for u in range(t, min(t + 5, T)) if TR[u, j]]
            if not tt: st['dropped'] += 1; continue
            if tt[0] != t or IDX[t] != r.date: st['moved_notrade'] += 1
            t = tt[0]
        if r.kind == 'exdiv' and np.isfinite(r.prev_close):
            pc = prevC[t, j]
            if abs(r.prev_close - pc) > 1e-6 and abs(r.prev_close - C[t, j]) < 1e-6:
                tt = [u for u in range(t + 1, min(t + 6, T)) if TR[u, j]]
                if not tt: st['dropped'] += 1; continue
                t = tt[0]; st['moved_lastcum'] += 1
        if not win[t]: continue
        st[r.kind + '_in'] += 1
        if not np.isfinite(prevC[t, j]): st['dropped'] += 1; continue   # 面板第一筆：沒有前收，事件不用
        rv = C[t, j] / r.ref_price - 1; rp = C[t, j] / prevC[t, j] - 1
        if abs(rv) > LIM and np.isfinite(rp) and abs(rp) <= LIM:
            st['rejected'] += 1; continue
        if (t, j) not in evref: evref[(t, j)] = r.ref_price; st['used'] += 1
    print('events', st)

    # ---------- 窗內 ret / flag ----------
    nf = dict(f2=0, f4=0, f4nan=0, f5=0)
    for t in wi:
        tr = TR[t]
        for j in np.where(tr)[0]:
            if (t, j) in evref:
                RET[t, j] = C[t, j] / evref[(t, j)] - 1; FLAG[t, j] = 2; nf['f2'] += 1; continue
            pc = prevC[t, j]
            if not np.isfinite(pc):
                RET[t, j] = np.nan; FLAG[t, j] = 0; continue
            r = C[t, j] / pc - 1
            if gap[t, j] > 20:
                FLAG[t, j] = 4; nf['f4'] += 1
                if abs(r) > LIM: r = np.nan; nf['f4nan'] += 1
                RET[t, j] = r; continue
            if abs(r) > LIM and nobs[t, j] >= 6:
                FLAG[t, j] = 5; RET[t, j] = np.nan; nf['f5'] += 1; continue
            RET[t, j] = r; FLAG[t, j] = 0
    print('flags', nf)
    # ---------- 窗後第一筆 ----------
    after = np.where(IDX > WB)[0]
    nre = 0; kept2 = 0
    for j in range(N):
        tt = after[TR[after, j]]
        if len(tt) == 0: continue
        t = tt[0]
        if FLAG[t, j] == 2: kept2 += 1; continue
        pc = prevC[t, j]
        if np.isfinite(pc):
            RET[t, j] = C[t, j] / pc - 1; nre += 1
            FLAG[t, j] = 0      # 重算的列 flag 歸 0（與研究員面板一致；規格未寫明）
    print('post-window first recomputed', nre, 'kept flag2', kept2)

    # ---------- adj ----------
    A4 = V['adj'].to_numpy()
    ADJ = A4.copy()
    t0 = wi[0]
    ADJ[t0:] = np.nan
    for j in range(N):
        prev = A4[:t0, j]; pv = prev[np.isfinite(prev)]
        base = pv[-1] if len(pv) else 1.0
        g = np.where(TR[t0:, j], 1 + np.nan_to_num(RET[t0:, j], nan=0.0), 1.0)
        a = base * np.cumprod(g)
        ADJ[t0:, j] = np.where(TR[t0:, j], a, np.nan)

    out = dict(adj=ADJ, close_raw=C, open=O, high=H, low=L, value=VAL, ret=RET, flag=FLAG, src=SRC)
    suf = ('_v0final' if vacuous_v0 else '') + ('_compat' if mode == 'compat' else '')
    for k, v in out.items():
        pd.DataFrame(v, index=IDX, columns=COLS).to_parquet(D_OUT / f'{k}{suf}.parquet')
    json.dump(dict(events=st, flags=nf, post_recomputed=nre, kept_flag2=kept2,
                   src_counts={str(k): int(v) for k, v in pd.Series(srcw.ravel()).value_counts().items()},
                   value_src={str(k): int(v) for k, v in pd.Series(vsrc[np.isfinite(c)]).value_counts().items()}),
              open(D_OUT / f'build_meta{suf}.json', 'w'), ensure_ascii=False, indent=1)

if __name__ == '__main__':
    main(sys.argv[1:])
