#!/usr/bin/env python3
"""步驟 2：建 V6 面板（只迭代資料；判準只看官方對官方一致率、除權息參考價相符率，不看任何策略報酬）。
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
import v6common as C, srcload as SL

T0 = time.time()
def log(*a): print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)

VERSIONS = {
    # 版本：OHLC 來源優先序、value 估計法、事件參考價規則
    'D1': dict(order=['hschi', 'tasi', 'bacardi', 'leochen'], vest='close', ev='lists'),          # 事前寫死的優先序（官方 history＞FinMind＞Tasi＞其他）
    'D2': dict(order=['hschi', 'tasi', 'bacardi', 'leochen'], vest='ohlc4', ev='lists'),
    'D3': dict(order=['hschi', 'bacardi', 'leochen', 'tasi'], vest='ohlc4', ev='lists'),           # Tasi 降到其他官方之後
    'D4': dict(order=['hschi', 'bacardi', 'leochen', 'tasi'], vest='ohlc4', ev='lists+chg'),       # 另用官方漲跌價差找未列名事件
    'D5': dict(order=['hschi', 'bacardi', 'leochen', 'tasi'], vest='calib', ev='lists+chg'),       # value 估計再乘挑選樣本校準係數
    # D1～D5 發現：hschi／Tasi 也含上櫃股（同代碼在上櫃期間的價格被誤收進上市面板，例如 1563、6446），FinMind 漲跌價差在無成交日後記 0（D4 的事件判定誤觸）
    'D6': dict(order=['hschi', 'bacardi', 'leochen', 'tasi'], vest='ohlc4', ev='lists', twse_only=True),   # D3＋上市身分過濾
    'D7': dict(order=['hschi', 'tasi', 'bacardi', 'leochen'], vest='ohlc4', ev='lists', twse_only=True),   # 事前優先序＋上市身分過濾
    # D6 發現：exdiv_5y／corp_actions 少數日期差一天（1519、1538）、少數事件參考價不合理（2429 權）、跨市場轉入股與前一筆相隔數年（6446）
    'D8': dict(order=['hschi', 'bacardi', 'leochen', 'tasi'], vest='ohlc4', ev='lists', twse_only=True, align=True),
}
SRCCODE = {'hschi': C.SRC_FINMIND, 'tasi': C.SRC_TASI, 'bacardi': C.SRC_OTHER, 'leochen': C.SRC_OTHER}

def tick(p):
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)

def load_all():
    P4 = C.load_panel()
    S = {k: SL.get(k) for k in ('hschi', 'tasi', 'bacardi', 'leochen', 'henry', 'history')}
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
        lst = pd.DataFrame(False, index=WIDX, columns=COLS)
        for k in ('leochen', 'bacardi'):
            d = S[k][(S[k].date >= C.W_A) & (S[k].date <= C.W_B)]
            lst |= wide(d, 'close', WIDX, COLS).notna()
        lst |= P4['close_raw'].reindex(WIDX).notna()
        pre = P4['close_raw'].loc['2023-12-21':'2023-12-29'].notna().any()
        post = P4['close_raw'].loc['2024-02-15':'2024-02-22'].notna().any()
        gapd = (WIDX >= C.GAP_A) & (WIDX <= C.GAP_B)
        lst.loc[gapd, (pre & post).to_numpy()] = True
        for f in ohlc: ohlc[f] = ohlc[f].where(lst)
        drop = (src.notna() & ~lst)
        src = src.where(lst)
        for k in srcname: srcname[k] = srcname[k] & lst
        tw_drop = int(drop.to_numpy().sum())
    else:
        tw_drop = 0
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
    fb = ohlc['close'].isna() & v4w['close_raw'].notna()
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
    for d, c, r in zip(ex.date, ex.code, ex.ref_price):
        if c in ci and d in IDX: evref[(IDX.get_loc(d), ci[c])] = ('exdiv', float(r))
    for d, c, r in zip(ca.date, ca.code, ca.ref_price):
        if c in ci and d in IDX: evref[(IDX.get_loc(d), ci[c])] = ('corp', float(r))
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
        if c in ci and d in IDX: exprev[(IDX.get_loc(d), ci[c])] = float(pc)
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
    # low：窗外用官方來源補（收盤必須與 V4 相同）
    for k in ('henry', 'hschi', 'bacardi', 'history'):
        d = S[k]
        W = wide(d, 'low', IDX, COLS); Wc = wide(d, 'close', IDX, COLS)
        ok = F['low'].isna() & W.notna() & (np.abs(Wc - F['close_raw']) < 1e-6) & ~pd.DataFrame(np.repeat(win[:, None], N, 1), index=IDX, columns=COLS)
        F['low'] = F['low'].where(~ok, W)
    stats['non_twse_cells_dropped'] = tw_drop
    meta = {'stats': stats, 'n_window_days': int(win.sum()), 'n_new_dates': int(len(IDX) - len(IDX4)),
            'src_counts_window': {str(int(k)): int(v) for k, v in pd.Series(src.to_numpy().ravel()).value_counts().items()},
            'value_src_counts_window': {k: int(v) for k, v in pd.Series(vsrc.to_numpy().ravel()).value_counts().items() if k},
            'srcname_cells': {k: int(v.to_numpy().sum()) for k, v in srcname.items()}}
    return F, meta, (ohlc, val, vsrc, src)

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
        res = {}
        calib = None
        for v, cfg in VERSIONS.items():
            if cfg['vest'] == 'calib' and calib is None:
                ve = res['D4']['metrics']['value_estimator_on_2020_2023']['ohlc4']
                calib = 1.0 / ve['median_ratio']
            F, meta, parts = build(cfg, P4, S, ex, ca, calib)
            mt = metrics(cfg, F, meta, parts, S)
            res[v] = {'cfg': cfg, 'calib': calib if cfg['vest'] == 'calib' else None, 'meta': meta, 'metrics': mt}
            log(v, {k: mt[k] for k in ('chosen_close_agree_all', 'chosen_open_agree_all', 'exdiv_prevclose_match_rate', 'n_est_value_cells_window', 'n_fallback_src3_cells_window')}, meta['stats'])
        f = C.HERE / 'iterations.json'
        it = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
        it['data_versions'] = res
        it['data_rule'] = '判準只看：(1) 所選格對另一官方來源的收盤／開盤一致率（≥ 99.5%）；(2) 除權息日 exdiv_5y 前收與面板前收相符率（≥ 99%）；(3) value 估計誤差。不看任何策略報酬。'
        C.jdump(it, f)
    elif mode == 'final':
        v = sys.argv[2]
        it = json.loads((C.HERE / 'iterations.json').read_text(encoding='utf-8'))
        cfg = VERSIONS[v]; calib = it['data_versions'][v].get('calib')
        F, meta, parts = build(cfg, P4, S, ex, ca, calib)
        C.DATA.mkdir(exist_ok=True)
        for k in C.FIELDS + ['low']:
            F[k].to_parquet(C.DATA / f'{k}.parquet')
        C.jdump({'version': v, 'cfg': cfg, 'calib': calib, 'meta': meta}, C.DATA / 'panel_v6_meta.json')
        log('written', v, meta['stats'])

if __name__ == '__main__':
    main()
