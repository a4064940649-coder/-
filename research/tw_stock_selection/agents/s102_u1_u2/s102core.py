#!/usr/bin/env python3
"""S102 純函式（正式引擎與 2008 起連續引擎共用）：格的定義、U1 訊號（make_signal_s10 的 E0 部分逐行複製＋遞補）、近 5 日漲幅、執行包裝、描述統計。
s10lib／cslib 在呼叫時才從 sys.modules 取（兩種引擎載入路徑不同，但都是同一個模組）。"""
import math
import numpy as np, pandas as pd

# 事前固定（見 PREREG.json）；U1、U2 各自選格、各自判定
U1_CELLS = {
    'U1|r20>0': dict(topic='U1', fill=dict(mom=0.0)),        # 遞補條件：近 20 日漲幅 > 0
    'U1|r20>5%': dict(topic='U1', fill=dict(mom=0.05)),      # 近 20 日漲幅 > 5%
    'U1|MA': dict(topic='U1', fill=dict(mom=None)),          # 只要仍在季線上前 20 名（不看漲幅）
}
U2_CELLS = {
    'U2|any': dict(topic='U2', rebuy=dict(r5=False)),        # 站回季線（且仍在前 20 名）即買回
    'U2|r5>0': dict(topic='U2', rebuy=dict(r5=True)),        # 再加近 5 日漲幅 > 0
}
CELLS = {**U1_CELLS, **U2_CELLS}
FILL_OFFSET = 1e6                                            # 遞補股的 rank＝SURGE − 1e6 → 引擎買進順序排在原名單之後（原名單內、遞補股內各自依 SURGE）


def ret5(P):
    """近 5 日漲幅＝還原收盤 ÷ 5 個交易日前的還原收盤 − 1（同 RET20 的寫法）。"""
    G = P.G
    out = np.full(G.shape, np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        out[5:] = G[5:] / G[:-5] - 1
    return out


def make_signal_u(P, univ, RT, shift=0, fill=None, depth=10, mom_lo=0.10, nsel=10):
    """s10lib.make_signal_s10 的 E0 部分逐行複製（進場名單、rank、X、info、RQ、sigrows 同原函式）；另附 score[k]／fin[k]（U2 用）。
    fill＝dict(mom)：訊號列 k 的原名單 tgt 不足 nsel 檔時，從「上一個訊號列的季線上 SURGE 前 10 名（pool＝o_ma[:10]，不論當時動能是否過門檻）」中，挑本次仍在季線上 SURGE 前 20 名（o_ma[:20]）、
    不在 tgt、且（mom 有值時）近 20 日漲幅 > mom 的股票，依本次 SURGE 由高到低遞補到 nsel 檔（名單檔數＝原名單＋遞補；已持有者由引擎略過）。"""
    import s10lib
    cslib = s10lib.cslib
    S, av = RT['S'], RT['av']
    sig = np.zeros((P.T, P.N), bool); rank = np.full((P.T, P.N), np.nan); X = np.zeros((P.T, P.N), bool)
    info = []; RQ = {}; SC = {}; FN = {}; prev = None; FILL = {}
    for m in S.index:
        k0 = av[m]
        if k0 >= P.T or k0 < 61:
            continue
        s = S.loc[m].to_numpy()
        if int((univ[k0] & np.isfinite(s)).sum()) < 50:
            continue
        k = k0 + shift
        if k >= P.T:
            continue
        s = cslib.ins15_mask_row(P, m, k, s)
        fin = univ[k] & np.isfinite(s)
        ok_ma = fin & P.F_MA60[k]
        o_ma = np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]
        r20 = P.RET20[k]
        pool = o_ma[:depth]
        okr = r20[pool] > mom_lo
        tgt = pool[okr][:nsel]
        add = []
        if fill is not None and prev is not None and len(tgt) < nsel:
            ts = set(int(j) for j in tgt)
            for j in o_ma[:20]:
                j = int(j)
                if j in prev and j not in ts and (fill['mom'] is None or r20[j] > fill['mom']):
                    add.append(j)
            add = add[:nsel - len(tgt)]
        sig[k, tgt] = True; rank[k, tgt] = s[tgt]
        if add:
            sig[k, add] = True; rank[k, add] = s[add] - FILL_OFFSET; FILL[k] = list(add)
        RQ[k] = set(int(j) for j in tgt) | set(add)
        X[k, :] = True; X[k, o_ma[:20]] = False
        SC[k] = s; FN[k] = fin
        info.append(dict(month=str(m), k=int(k), sig_date=str(P.IDX[k].date()), n_ma=int(ok_ma.sum()), n_tgt=int(len(tgt) + len(add)), n_orig=int(len(tgt)), n_fill=int(len(add)),
                         targets=[str(P.COLS[j]) for j in tgt], fills=[str(P.COLS[j]) for j in add]))
        prev = set(int(j) for j in pool)                   # 上一個訊號列的「季線上 SURGE 前 10 名」（不論動能是否過 10%）
    sigrows = np.zeros(P.T, bool); sigrows[[x['k'] for x in info]] = True
    return dict(sig=sig, rank=rank, X=X, XV={'E0': X}, SD=None, max_hold=None, info=info, RQ=RQ, sigrows=sigrows, score=SC, fin=FN, FILL=FILL)


def sim_core(P, S, U, cell, kw, base_sim):
    """CUR／U1 → base_sim（原引擎 simulate_s10）；U2 → s102sim.simulate_u(rebuy=…)。"""
    if cell == 'CUR' or CELLS[cell]['topic'] == 'U1':
        return base_sim(P, S, U, **kw)
    import s102sim
    rb = dict(CELLS[cell]['rebuy']); rb['RET5'] = P.RET5
    return s102sim.simulate_u(P, S, U, rebuy=rb, **kw)


def scaled_current(nav_cur, bench, lam, a, b):
    """R2 對照（同 S100 s100util.scaled_current）：窗內每日報酬＝lam × 現行日報酬 ＋ (1 − lam) × 0050 日報酬（每日再平衡、不另計成本）。回傳窗內淨值（起點 1）。"""
    rc = nav_cur.loc[a:b].pct_change().iloc[1:]; rb = bench.reindex(nav_cur.index).loc[a:b].pct_change().iloc[1:]
    r = lam * rc + (1.0 - lam) * rb
    return pd.concat([pd.Series([1.0], index=[nav_cur.loc[a:b].index[0]]), (1.0 + r).cumprod()])


def roll_sr_neg(nav, win=252):
    """滾動一年（252 個交易日）SR < 0 的比例（S101 第 7 項口徑：日報酬滾動平均 < 0 即 SR < 0；只算滿窗）。"""
    r = nav.dropna().pct_change().dropna()
    m = r.rolling(win).mean().dropna()
    return float((m < 0).mean()) if len(m) else float('nan')


def top5_share(lg, top_frac=0.05):
    """逐筆集中度（S99／S101 口徑）：報酬率前 5% 交易（ceil）的損益 ÷ 全部交易損益（含期末未出場）。"""
    r = lg['ret'].to_numpy(float); pnl = lg['pnl'].to_numpy(float); n = len(r); m = int(math.ceil(n * top_frac)); o = np.argsort(-r, kind='stable')[:m]
    return dict(n_all=int(n), n_top=m, share_of_trade_pnl=float(pnl[o].sum() / pnl.sum()), rest_mean_ret=float(np.delete(r, o).mean()), mean_ret=float(r.mean()),
                median_ret=float(np.median(r)), win=float((r > 0).mean()), pnl_sum=float(pnl.sum()))
