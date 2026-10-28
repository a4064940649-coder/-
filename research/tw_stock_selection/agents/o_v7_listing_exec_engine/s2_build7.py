#!/usr/bin/env python3
"""步驟 2：建 V7 面板（由 o_v6_price_panel_official/s2_build.py 複製後修改；V6 的 D8 為底）。
V7 修補：(a) 上市身分改用 listing_master 的逐日上市矩陣（data/listed_twse.parquet；全市場上市身分），取代「leochen／bacardi／V4 src 3 任一有值」；
         (b) 颱風停市日（不在面板日期中）的除權息／公司行動事件，順延到下一個面板日期後再套用 D8 的事件日對齊；
         (c) 拿掉 2024-01-02～02-14「V4 在 2023-12-21～29 與 2024-02-15～22 都有值」的上市判定（前視）。
以下為 V6 原說明：
步驟 2：建 V6 面板（只迭代資料；判準只看官方對官方一致率、除權息參考價相符率，不看任何策略報酬）。
用法：python3 -I -B s2_build.py iterate   → 跑全部資料版本、寫 iterations_data（iterations.json 的 data 段）
      python3 -I -B s2_build.py final <版本> → 用選定版本寫 data/{adj,close_raw,open,high,low,value,ret,flag,src}.parquet

V6 口徑（與現行面板相同）：
  * 日期＝V4 面板日期 ∪ TAIEX 交易日（2024-01-02～2025-09-23）。替換窗 W＝2024-01-02～2025-09-23；窗外各欄沿用 V4（只有窗後每檔第一筆的 ret 依新的前收重算）。
  * close_raw／open／high／low：原始價；value＝成交值（元）；ret＝收盤 ÷ 參考價 − 1；adj＝第一筆 1.0、之後乘 (1+ret)（ret 缺值當 0）。
  * src：2／4＝原官方；5＝FinMind（hschi）；6＝Tasi kbar；7＝其他官方封存（bacardi＝證交所全市場日 K、leochen＝STOCK_DAY 格式）；3＝Yahoo 5 分 K 近似（只當最後備援）。
  * flag：0 一般；1 除權息／事件日但無參考價（ret 設 0，同 V4）；2 除權息或減資等事件、用官方參考價；5 疑似資料錯誤（無事件但超過漲跌幅 0.5% 以上，ret 設缺值，同 V4）。
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import v7common as C, srcload7 as SL, gatepop as GP

T0 = time.time()
def log(*a): print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)

VERSIONS = {
    # V7 版本：只迭代資料（判準：上市母體覆蓋率、官方對官方一致率、除權息前收相符率、颱風日事件參考價相符），不看策略報酬
    'L1': dict(order=['hschi', 'bacardi', 'leochen', 'tasi'], vest='ohlc4', ev='lists', twse_only=True, align=True, listing='master', typhoon=True),
    'L2': dict(order=['hschi', 'bacardi', 'leochen', 'tasi'], vest='ohlc4', ev='lists', twse_only=True, align=True, listing='master_exTIB', typhoon=True),
    'L3': dict(order=['hschi', 'bacardi', 'leochen', 'tasi'], vest='ohlc4', ev='lists', twse_only=True, align=True, listing='v6_nolook', typhoon=True),
    'L4': dict(order=['hschi', 'bacardi', 'leochen', 'tasi'], vest='ohlc4', ev='lists', twse_only=True, align=True, listing='master', typhoon=False),
    'L5': dict(order=['hschi', 'bacardi', 'leochen', 'tasi'], vest='ohlc4', ev='lists', twse_only=True, align=True, listing='master_strong', typhoon=True),
    'V6REPRO': dict(order=['hschi', 'bacardi', 'leochen', 'tasi'], vest='ohlc4', ev='lists', twse_only=True, align=True, listing='v6', typhoon=False),
}
SRCCODE = {'hschi': C.SRC_FINMIND, 'tasi': C.SRC_TASI, 'bacardi': C.SRC_OTHER, 'leochen': C.SRC_OTHER}

def tick(p):
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)

def load_all():
    P4 = C.load_panel(C.PANEL4)
    S = {k: SL.get(k) for k in ('hschi', 'tasi', 'bacardi', 'leochen', 'history')}
    cols = P4['close_raw'].columns
    for k in S: S[k] = S[k][S[k].code.isin(set(cols))]
    X = C.XTR
    ex = pd.read_csv(X / 'exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
    ex = ex[ex.market == 'TWSE']
    ca = pd.read_csv(X / 'corp_actions.csv', dtype={'code': str}, parse_dates=['date'])
    ca = ca[ca.market == 'TWSE']
    return P4, S, ex, ca

def wide(d, f, idx, cols):
    return d.pivot(index='date', columns='code', values=f).reindex(index=idx, columns=cols)

def build(cfg, P4, S, ex, ca, calib=None):
    IDX4, COLS = P4['close_raw'].index, P4['close_raw'].columns
    cal = C.taiex_cal()
    wd = cal[(cal >= C.W_A) & (cal <= C.W_B)]
    IDX = IDX4.union(wd)
    win = (IDX >= C.W_A) & (IDX <= C.W_B)
    WIDX = IDX[win]
    F = {k: P4[k].reindex(IDX) for k in C.FIELDS}
    F['low'] = pd.DataFrame(np.nan, index=IDX, columns=COLS)
    # ---- 窗內：依優先序取 OHLC ----
    ohlc = {f: pd.DataFrame(np.nan, index=WIDX, columns=COLS) for f in ('open', 'high', 'low', 'close', 'vol')}
    src = pd.DataFrame(np.nan, index=WIDX, columns=COLS); srcname = {}
    for k in cfg['order']:
        d = S[k][(S[k].date >= C.W_A) & (S[k].date <= C.W_B)]
        W = {f: wide(d, f, WIDX, COLS) for f in ('open', 'high', 'low', 'close', 'vol')}
        take = ohlc['close'].isna() & W['close'].notna()
        for f in W: ohlc[f] = ohlc[f].where(~take, W[f])
        src = src.where(~take, SRCCODE[k]); srcname[k] = take
    # 上市身分：該股日在 leochen（STOCK_DAY＝上市）、bacardi（market=TWSE）或 V4 src 3（Yahoo .TW）任一有值才算上市；
    # 2024-01-02～02-14 另外允許「2023-12 最後 5 個交易日與 2024-02-15 起 5 個交易日在 V4 都有值」的股票
    if cfg.get('twse_only'):
        lmode = cfg.get('listing', 'v6')
        if lmode in ('v6', 'v6_nolook'):
            lst = pd.DataFrame(False, index=WIDX, columns=COLS)
            for k in ('leochen', 'bacardi'):
                d = S[k][(S[k].date >= C.W_A) & (S[k].date <= C.W_B)]
                lst |= wide(d, 'close', WIDX, COLS).notna()
            lst |= P4['close_raw'].reindex(WIDX).notna()
            pre = P4['close_raw'].loc['2023-12-21':'2023-12-29'].notna().any()
            gapd = (WIDX >= C.GAP_A) & (WIDX <= C.GAP_B)
            if lmode == 'v6':
                post = P4['close_raw'].loc['2024-02-15':'2024-02-22'].notna().any()
                lst.loc[gapd, (pre & post).to_numpy()] = True
            else:   # 拿掉前視：只用缺口前已知的上市（2023-12 最後 5 個交易日有值），缺口後是否還有值不看
                lst.loc[gapd, pre.to_numpy()] = True
        else:
            LMX = pd.read_parquet(C.DATA / 'listed_twse.parquet')
            lst = LMX.reindex(index=WIDX, columns=COLS).fillna(False).astype(bool)
            if lmode == 'master_exTIB':
                tib = [c for c in json.loads((C.HERE / 'explore.json').read_text(encoding='utf-8'))['LISTING']['TIB_codes'] if c in lst.columns]
                lst.loc[:, tib] = False
            if lmode == 'master_strong':
                LMS = pd.read_parquet(C.DATA / 'listed_twse_strong.parquet')
                lst = LMS.reindex(index=WIDX, columns=COLS).fillna(False).astype(bool)
        for f in ohlc: ohlc[f] = ohlc[f].where(lst)
        drop = (src.notna() & ~lst)
        src = src.where(lst)
        for k in srcname: srcname[k] = srcname[k] & lst
        tw_drop = int(drop.to_numpy().sum())
    else:
        tw_drop = 0
        lst = pd.DataFrame(True, index=WIDX, columns=COLS)
    # 官方成交值：hschi＞bacardi；其餘用估計
    val = pd.DataFrame(np.nan, index=WIDX, columns=COLS)
    vsrc = pd.DataFrame('', index=WIDX, columns=COLS)
    for k in ('hschi', 'bacardi'):
        d = S[k][(S[k].date >= C.W_A) & (S[k].date <= C.W_B)]
        W = wide(d, 'value', WIDX, COLS); Wc = wide(d, 'close', WIDX, COLS)
        ok = val.isna() & W.notna() & (np.abs(Wc - ohlc['close']) < 1e-6)       # 只用與所選收盤一致的官方成交值
        val = val.where(~ok, W); vsrc = vsrc.mask(ok, k)
    est = ohlc['close'] * ohlc['vol'] if cfg['vest'] == 'close' else (ohlc['open'] + ohlc['high'] + ohlc['low'] + ohlc['close']) / 4 * ohlc['vol']
    if cfg['vest'] == 'calib' and calib is not None: est = est * calib
    need = val.isna() & ohlc['close'].notna()
    val = val.where(~need, est); vsrc = vsrc.mask(need & est.notna(), 'est')
    # 最後備援：V4 src 3
    v4w = {f: P4[f].reindex(WIDX) for f in ('close_raw', 'open', 'high', 'value', 'src')}
    fb = ohlc['close'].isna() & v4w['close_raw'].notna() & lst
    ohlc['close'] = ohlc['close'].where(~fb, v4w['close_raw']); ohlc['open'] = ohlc['open'].where(~fb, v4w['open'])
    ohlc['high'] = ohlc['high'].where(~fb, v4w['high']); val = val.where(~fb, v4w['value'])
    src = src.where(~fb, C.SRC_Y5M); vsrc = vsrc.mask(fb, 'src3')
    # 寫入窗內
    for f, g in (('close_raw', 'close'), ('open', 'open'), ('high', 'high'), ('low', 'low')):
        F[f].loc[WIDX] = ohlc[g].to_numpy()
    F['value'].loc[WIDX] = val.to_numpy(); F['src'].loc[WIDX] = src.to_numpy()
    # ---- 報酬與 flag：窗內全部重算；窗後每檔第一筆依新前收重算 ----
    Cn = F['close_raw'].to_numpy(); Rn = F['ret'].to_numpy().copy(); Fn = F['flag'].to_numpy().copy()
    T = len(IDX); N = len(COLS); ci = {c: j for j, c in enumerate(COLS)}
    t_w0, t_w1 = int(np.argmax(win)), int(len(win) - 1 - np.argmax(win[::-1]))
    evref = {}
    stats_ty = {'typhoon_events_shifted': 0, 'typhoon_events_list': []}
    def tloc(d):
        if d in IDX: return IDX.get_loc(d)
        if not cfg.get('typhoon'): return None
        k = IDX.searchsorted(d)              # 事件日不是面板日期（颱風停市）→ 順延到下一個面板日期
        if k >= len(IDX) or (IDX[k] - d).days > 10: return None
        return k
    for d, c, r in zip(ex.date, ex.code, ex.ref_price):
        if c in ci:
            k = tloc(d)
            if k is not None:
                evref[(k, ci[c])] = ('exdiv', float(r))
                if d not in IDX: stats_ty['typhoon_events_shifted'] += 1; stats_ty['typhoon_events_list'].append((str(d.date()), c, str(IDX[k].date())))
    for d, c, r in zip(ca.date, ca.code, ca.ref_price):
        if c in ci:
            k = tloc(d)
            if k is not None:
                evref[(k, ci[c])] = ('corp', float(r))
                if d not in IDX: stats_ty['typhoon_events_shifted'] += 1; stats_ty['typhoon_events_list'].append((str(d.date()), c, str(IDX[k].date())))
    # 官方漲跌價差（FinMind／bacardi）：chg==0 且收盤≠前收 → 事件日
    chg = pd.DataFrame(np.nan, index=WIDX, columns=COLS)
    for k in ('hschi', 'bacardi'):
        d = S[k][(S[k].date >= C.W_A) & (S[k].date <= C.W_B)]
        W = wide(d, 'chg', WIDX, COLS); Wc = wide(d, 'close', WIDX, COLS)
        ok = chg.isna() & W.notna() & (np.abs(Wc - ohlc['close']) < 1e-6)
        chg = chg.where(~ok, W)
    CH = np.full((T, N), np.nan); CH[t_w0:t_w1 + 1] = chg.to_numpy()
    stats = {'exdiv_events': 0, 'corp_events': 0, 'chg_unlisted_event_flag1': 0, 'chg_ref_differs_used': 0, 'flag5': 0,
             'exdiv_prevclose_match': 0, 'exdiv_prevclose_n': 0, 'post_window_first_recomputed': 0}
    exprev = {}
    for d, c, pc in zip(ex.date, ex.code, ex.prev_close):
        if c in ci:
            k = tloc(d)
            if k is not None: exprev[(k, ci[c])] = float(pc)
    traded = np.isfinite(Cn)
    stats.update(event_moved_to_next_trade=0, event_rejected_implausible=0, long_gap_first_obs=0)
    if cfg.get('align'):
        # 事件日對齊：事件日該股沒有成交 → 移到之後第一個成交日（5 個交易日內）；
        # 除權息表的前收≠面板前收、但等於事件日當天收盤 → 表列日期是最後含權日，移到下一個成交日
        ev2, ep2 = {}, {}
        for (t, j), v in evref.items():
            tt = t
            if not traded[t, j]:
                nx = np.where(traded[t:t + 6, j])[0]
                tt = t + int(nx[0]) if len(nx) else None
            elif (t, j) in exprev:
                pv = np.where(traded[:t, j])[0]
                if len(pv) and abs(exprev[(t, j)] - Cn[pv[-1], j]) > 1e-6 and abs(exprev[(t, j)] - Cn[t, j]) < 1e-6:
                    nx = np.where(traded[t + 1:t + 6, j])[0]
                    tt = t + 1 + int(nx[0]) if len(nx) else None
            if tt is None: continue
            if tt != t: stats['event_moved_to_next_trade'] += 1
            ev2[(tt, j)] = v
            if (t, j) in exprev: ep2[(tt, j)] = exprev[(t, j)]
        evref, exprev = ev2, ep2
    for j in range(N):
        tr = np.where(traded[:, j])[0]
        if len(tr) == 0: continue
        sel = tr[(tr >= t_w0) & (tr <= t_w1)]
        post = tr[tr > t_w1]
        todo = list(sel) + ([post[0]] if len(post) else [])
        for t in todo:
            k = np.searchsorted(tr, t)
            if k == 0:
                Rn[t, j] = np.nan; Fn[t, j] = 0; continue
            prev = Cn[tr[k - 1], j]
            if t > t_w1:   # 窗後第一筆：V4 是事件日（flag 2）就保留原 ret
                if Fn[t, j] == 2: continue
                Rn[t, j] = Cn[t, j] / prev - 1; Fn[t, j] = 0; stats['post_window_first_recomputed'] += 1; continue
            ev = evref.get((t, j))
            if (t, j) in exprev:
                stats['exdiv_prevclose_n'] += 1
                stats['exdiv_prevclose_match'] += int(abs(exprev[(t, j)] - prev) < 1e-6)
            lim0 = 0.10 + 0.005
            if ev is not None and cfg.get('align') and k > 5 and abs(Cn[t, j] / ev[1] - 1) > lim0 and abs(Cn[t, j] / prev - 1) <= lim0:
                stats['event_rejected_implausible'] += 1; ev = None      # 事件參考價使報酬超過漲跌幅、但對前收在漲跌幅內 → 表列事件不採用
            if cfg.get('align') and ev is None and (t - tr[k - 1]) > 20 and k > 0:
                r = Cn[t, j] / prev - 1                                  # 與前一筆相隔 > 20 個交易日（轉上市、長期停牌）
                Rn[t, j] = r if abs(r) <= lim0 else np.nan; Fn[t, j] = 4; stats['long_gap_first_obs'] += 1
                continue
            if ev is not None:
                Rn[t, j] = Cn[t, j] / ev[1] - 1; Fn[t, j] = 2
                stats['exdiv_events' if ev[0] == 'exdiv' else 'corp_events'] += 1
                continue
            r = Cn[t, j] / prev - 1; f = 0
            if 'chg' in cfg['ev'] and np.isfinite(CH[t, j]):
                ref_o = Cn[t, j] - CH[t, j]
                if CH[t, j] == 0 and abs(Cn[t, j] - prev) > 1e-6:
                    r, f = 0.0, 1; stats['chg_unlisted_event_flag1'] += 1
                elif abs(ref_o - prev) > 1e-6 and ref_o > 0:
                    r, f = Cn[t, j] / ref_o - 1, 2; stats['chg_ref_differs_used'] += 1
            ipo_days = k   # 上市後前 5 個交易日無漲跌幅
            lim = 0.10
            if f == 0 and ipo_days > 5 and abs(r) > lim + 0.005:
                r, f = np.nan, 5; stats['flag5'] += 1
            Rn[t, j] = r; Fn[t, j] = f
    F['ret'] = pd.DataFrame(Rn, index=IDX, columns=COLS); F['flag'] = pd.DataFrame(Fn, index=IDX, columns=COLS)
    # adj：窗起點之前沿用 V4；之後依 ret 連乘（只在有收盤的日子）
    A = F['adj'].to_numpy().copy()
    G = np.where(traded, 1.0 + np.nan_to_num(Rn, nan=0.0), 1.0)
    for j in range(N):
        tr = np.where(traded[:, j])[0]
        tr2 = tr[tr >= t_w0]
        if len(tr2) == 0: continue
        pre = tr[tr < t_w0]
        a = A[pre[-1], j] if len(pre) else None
        for t in tr2:
            a = 1.0 if a is None else a * G[t, j]
            A[t, j] = a
        A[np.setdiff1d(np.arange(t_w0, T), tr2), j] = np.nan
    F['adj'] = pd.DataFrame(A, index=IDX, columns=COLS)
    # low：窗外沿用 V6 面板的 low（V6 用 henry／hschi／bacardi／history 補；本題不重做窗外）
    L6 = pd.read_parquet(C.V6DATA / 'low.parquet').reindex(index=IDX, columns=COLS)
    F['low'] = F['low'].where(pd.DataFrame(np.repeat(win[:, None], N, 1), index=IDX, columns=COLS), L6)
    stats['non_twse_cells_dropped'] = tw_drop
    stats.update(stats_ty)
    meta = {'stats': stats, 'n_window_days': int(win.sum()), 'n_new_dates': int(len(IDX) - len(IDX4)),
            'src_counts_window': {str(int(k)): int(v) for k, v in pd.Series(src.to_numpy().ravel()).value_counts().items()},
            'value_src_counts_window': {k: int(v) for k, v in pd.Series(vsrc.to_numpy().ravel()).value_counts().items() if k},
            'srcname_cells': {k: int(v.to_numpy().sum()) for k, v in srcname.items()}}
    return F, meta, (ohlc, val, vsrc, src)

def typhoon_check(F, meta):
    """颱風停市日事件：順延後的事件日，面板前收（前一個成交日收盤）＝除權息表前收、ret＝收盤÷表列參考價−1、flag＝2 的比例。"""
    ex = pd.read_csv(C.XTR / 'exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date']); ex = ex[ex.market == 'TWSE']
    ex = ex[ex.date.isin(C.TYPHOON) & ex.code.isin(F['close_raw'].columns)]
    IDX = F['close_raw'].index; rows = []
    for d, c, pc, rf in zip(ex.date, ex.code, ex.prev_close, ex.ref_price):
        k = IDX.searchsorted(d); cl = F['close_raw'][c]
        tr = cl.iloc[k:k + 6].dropna()
        if not len(tr): rows.append((str(d.date()), c, None, False, False, False)); continue
        t = tr.index[0]; prevs = cl.loc[:IDX[k - 1]].dropna()
        prev = prevs.iloc[-1] if len(prevs) else np.nan
        r = F['ret'].at[t, c]; fl = F['flag'].at[t, c]
        rows.append((str(d.date()), c, str(t.date()), bool(abs(prev - pc) < 1e-6), bool(np.isfinite(r) and abs(r - (tr.iloc[0] / rf - 1)) < 1e-9), bool(fl == 2)))
    D = pd.DataFrame(rows, columns=['date', 'code', 'applied_on', 'prevclose_match', 'ret_uses_ref', 'flag2'])
    return {'n_events': int(len(D)), 'prevclose_match_rate': float(D.prevclose_match.mean()) if len(D) else None,
            'ref_applied_rate': float((D.ret_uses_ref & D.flag2).mean()) if len(D) else None, 'not_applied': D[~(D.ret_uses_ref & D.flag2)].values.tolist()}

def metrics(cfg, F, meta, parts, S):
    """資料品質判準（不看報酬）：所選格與「另一個官方來源」的一致率；除權息前收相符率；value 估計誤差。"""
    ohlc, val, vsrc, src = parts
    WIDX = ohlc['close'].index
    out = {}
    agree = {}
    for k in ('hschi', 'tasi', 'bacardi', 'leochen'):
        d = S[k][(S[k].date >= C.W_A) & (S[k].date <= C.W_B)]
        Wc = wide(d, 'close', WIDX, ohlc['close'].columns); Wo = wide(d, 'open', WIDX, ohlc['close'].columns)
        agree[k] = (Wc, Wo)
    # 對每個所選格，找「不是它自己來源」的其他官方，比對收盤與開盤
    tot = 0; ok_c = 0; ok_o = 0; by_src = {}
    for k in ('hschi', 'tasi', 'bacardi', 'leochen'):
        sel = parts[3].eq(SRCCODE[k]) & ohlc['close'].notna()
        # 該格實際來源＝第一個在優先序中有值者
        for k2 in ('hschi', 'tasi', 'bacardi', 'leochen'):
            if SRCCODE[k2] == SRCCODE[k] and k2 != k: pass
        others = [o for o in ('hschi', 'tasi', 'bacardi', 'leochen') if o != k]
        for o in others:
            Wc, Wo = agree[o]
            m = sel & Wc.notna() & agree[k][0].notna() & (np.abs(agree[k][0] - ohlc['close']) < 1e-6)
            n = int(m.to_numpy().sum())
            if n == 0: continue
            c = int((m & (np.abs(Wc - ohlc['close']) < 1e-6)).to_numpy().sum())
            oo = int((m & (np.abs(Wo - ohlc['open']) < 1e-6)).to_numpy().sum())
            by_src[f'{k}_vs_{o}'] = {'n': n, 'close_agree': c / n, 'open_agree': oo / n}
            tot += n; ok_c += c; ok_o += oo
    out['chosen_vs_other_official'] = by_src
    out['chosen_close_agree_all'] = ok_c / tot if tot else None
    out['chosen_open_agree_all'] = ok_o / tot if tot else None
    st = meta['stats']
    out['exdiv_prevclose_match_rate'] = st['exdiv_prevclose_match'] / max(st['exdiv_prevclose_n'], 1)
    # value 估計：在 hschi 有官方值的格上，以 leochen 成交量估計，看誤差
    h = S['hschi'][(S['hschi'].date >= '2020-01-01') & (S['hschi'].date <= '2023-12-31')]
    l = S['leochen'][(S['leochen'].date >= '2020-01-01') & (S['leochen'].date <= '2023-12-31')]
    m = h.merge(l[['date', 'code', 'vol']], on=['date', 'code'], suffixes=('', '_l'))
    e = {'close': m.close * m.vol_l, 'ohlc4': (m.open + m.high + m.low + m.close) / 4 * m.vol_l}
    ve = {}
    for kk, v in e.items():
        r = v / m.value
        ve[kk] = {'median_ratio': float(r.median()), 'abs_err_median_pct': float((r - 1).abs().median() * 100), 'within_2pct': float(((r - 1).abs() <= 0.02).mean())}
    out['value_estimator_on_2020_2023'] = ve
    out['n_est_value_cells_window'] = meta['value_src_counts_window'].get('est', 0)
    out['n_fallback_src3_cells_window'] = meta['src_counts_window'].get('3', 0)
    return out

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'iterate'
    P4, S, ex, ca = load_all()
    log('loaded')
    if mode == 'iterate':
        sig = [pd.Timestamp(x) for x in pd.read_csv(C.V6DIR / 'plan_V6.csv')['signal']]
        POP, EXTRA = GP.population(sig, pd.Timestamp('2026-10-12'), conservative=True)
        POP.to_parquet(C.DATA / 'gate_population.parquet'); EXTRA.to_csv(C.DATA / 'gate_population_proxy.csv', index=False)
        V6P = {k: pd.read_parquet(C.V6DATA / f'{k}.parquet') for k in ('close_raw', 'ret', 'adj', 'value', 'flag')}
        log('population', len(POP), 'proxy rows', len(EXTRA))
        res = {}
        calib = None
        for v, cfg in VERSIONS.items():
            if cfg['vest'] == 'calib' and calib is None:
                ve = res['D4']['metrics']['value_estimator_on_2020_2023']['ohlc4']
                calib = 1.0 / ve['median_ratio']
            F, meta, parts = build(cfg, P4, S, ex, ca, calib)
            mt = metrics(cfg, F, meta, parts, S)
            _, cv = GP.coverage(POP, F, conservative=True); _, cvs = GP.coverage(POP, F, conservative=False)
            mt['pop_coverage_conservative'] = cv; mt['pop_coverage_strict'] = cvs
            g24 = POP[(POP.date >= C.GAP_A) & (POP.date <= C.GAP_B)]
            mt['pop_coverage_2024_01_02_to_02_14'] = GP.coverage(g24, F, conservative=True)[1]
            mt['typhoon_check'] = typhoon_check(F, meta)
            mt['diff_vs_V6_cells_window'] = {k: int(((F[k].loc[C.W_A:C.W_B] != V6P[k].reindex(F[k].index).loc[C.W_A:C.W_B]) & ~(F[k].loc[C.W_A:C.W_B].isna() & V6P[k].reindex(F[k].index).loc[C.W_A:C.W_B].isna())).to_numpy().sum()) for k in ('close_raw', 'ret', 'value')}
            mt['diff_vs_V6_cells_outside_window'] = {k: int(((F[k].drop(F[k].loc[C.W_A:C.W_B].index) != V6P[k].reindex(F[k].index).drop(F[k].loc[C.W_A:C.W_B].index)) & ~(F[k].drop(F[k].loc[C.W_A:C.W_B].index).isna() & V6P[k].reindex(F[k].index).drop(F[k].loc[C.W_A:C.W_B].index).isna())).to_numpy().sum()) for k in ('close_raw', 'ret', 'adj', 'value', 'flag')}
            res[v] = {'cfg': cfg, 'calib': calib if cfg['vest'] == 'calib' else None, 'meta': meta, 'metrics': mt}
            log(v, {k: mt[k] for k in ('chosen_close_agree_all', 'chosen_open_agree_all', 'exdiv_prevclose_match_rate', 'n_est_value_cells_window', 'n_fallback_src3_cells_window')}, meta['stats'])
        f = C.HERE / 'iterations.json'
        it = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
        it['data_versions'] = res
        it['data_rule'] = '判準只看：(1) 所選格對另一官方來源的收盤／開盤一致率；(2) 除權息日 exdiv_5y 前收與面板前收相符率；(3) 上市母體覆蓋（s3_gate7）；(4) 颱風日事件。不看任何策略報酬。'
        C.jdump(it, f)
    elif mode == 'final':
        v = sys.argv[2]
        it = json.loads((C.HERE / 'iterations.json').read_text(encoding='utf-8'))
        cfg = VERSIONS[v]; calib = it['data_versions'][v].get('calib')
        F, meta, parts = build(cfg, P4, S, ex, ca, calib)
        C.DATA.mkdir(exist_ok=True)
        for k in C.FIELDS + ['low']:
            F[k].to_parquet(C.DATA / f'{k}.parquet')
        C.jdump({'version': v, 'cfg': cfg, 'calib': calib, 'meta': meta}, C.DATA / 'panel_v7_meta.json')
        log('written', v, meta['stats'])

if __name__ == '__main__':
    main()
