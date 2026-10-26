#!/usr/bin/env python3
"""第 7 步：技術訊號當中心主策略（bt_final_rule.py：52 週高點 × 近 3 月營收年增，月換 30 檔）的進出場時機濾網。

主策略照原程式：每月第一個 ≥ 11 日交易日「收盤」換股（沿用中心慣例）。本檔用日頻引擎重算（先驗證無濾網時與原程式一致）。
濾網一律只用「換股日前一交易日收盤」或「當日收盤 → 隔日開盤執行」的資訊（不偷看）。事前固定的濾網：
 進場濾網（換股時只買符合者；R＝依原排名往下遞補到 30 檔，D＝不遞補、空位放現金）：
  E1 站上季線且季線上彎（收 > MA60 且 MA60 > 5 日前）  E2 RSI(14) ≤ 70（不買超買）  E3 MACD DIF > DEA
  E4 收 > MA20  E5 KD 的 K ≤ 80（不買超買）
 出場／時機濾網（第 s 日收盤成立 → 第 s+1 日開盤賣，現金放到下次換股日）：
  X1 個股收盤跌破 MA20  X2 個股 MACD 死亡交叉  X5 個股收盤跌破 MA60
  X3 大盤（加權指數）收盤 < MA60 → 全部出場、站回 MA60 隔日開盤買回當月名單
  X4 大盤 MACD DIF < DEA → 全部出場、DIF > DEA 隔日開盤買回
 成本：每次買、賣各 0.25%（來回 0.5%）。期間：全期 2021-03～2026-10；挑選 2021–2023；驗證 2024–2026（另列剔除 src 3 日）。
 事前判定「有用」：挑選期 CAGR 高於原策略且 MDD 不更差，或 MDD 改善 ≥ 5 個百分點且 CAGR 少不到 2 個百分點；有用者再看驗證期。
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import tlib as L
import bt_final_rule as M

H = 0.0025


def picks_with(mask=None, drop=False, n=30):
    """mask：dates × codes 的 bool（True＝可買）；None＝原策略。"""
    el = M.t300 & M.yoy3.notna() & M.s52.notna() & ~M.inD
    sc = 0.5 * M.s52.where(el).rank(axis=1, pct=True) + 0.5 * M.yoy3.where(el).rank(axis=1, pct=True)
    out = {}
    for t in M.R.index:
        ok = el.loc[t] & M.R.loc[t].notna()
        if ok.sum() < 50: continue
        s = sc.loc[t][ok].sort_values(ascending=False)
        if mask is None:
            out[t] = list(s.index[:n])
        elif drop:
            out[t] = [c for c in s.index[:n] if mask.loc[t, c]]
        else:
            out[t] = [c for c in s.index if mask.loc[t, c]][:n]
    return out


def engine(D, picks, stop=None, mkt_bad=None, n=30):
    """日頻模擬。picks：{換股日: [代碼]}；stop：T×N bool（收盤成立→隔日開盤賣）；mkt_bad：T bool Series（True＝應空手）。"""
    idx = D['idx']; cols = list(D['AC'].columns); ci = {c: k for k, c in enumerate(cols)}
    AC = D['adj'].ffill().to_numpy(); AO = D['AO'].to_numpy()
    rd = sorted(picks); i0 = idx.get_loc(rd[0]); iend = idx.get_loc(M.dates[-1])
    rset = {idx.get_loc(t): t for t in rd}
    S = stop.to_numpy() if stop is not None else None
    MB = mkt_bad.reindex(idx).fillna(False).to_numpy() if mkt_bad is not None else None
    pos, cash = {}, 1.0
    target = []; stopped = set(); pend = set(); p_exit = p_entry = False; out_mkt = False
    vals = np.full(len(idx), np.nan); vals[i0] = 1.0
    nst = 0; nsw = 0
    for i in range(i0, iend + 1):
        bought = set()
        if i > i0:
            if p_exit:
                for j, v in pos.items():
                    cash += v * (AO[i, j] / AC[i - 1, j] if np.isfinite(AO[i, j]) else 1.0) * (1 - H)
                pos = {}; out_mkt = True; p_exit = False; pend = set(); nsw += 1
            elif p_entry:
                names = [ci[c] for c in target if ci[c] not in stopped and np.isfinite(AO[i, ci[c]])]
                if names:
                    per = cash * (1 - H) / n
                    for j in names:
                        pos[j] = per; bought.add(j); cash -= per / (1 - H)
                out_mkt = False; p_entry = False; nsw += 1
            for j in list(pend):
                if j in pos and np.isfinite(AO[i, j]):
                    cash += pos[j] * AO[i, j] / AC[i - 1, j] * (1 - H); del pos[j]; stopped.add(j); pend.discard(j); nst += 1
            for j in pos:
                if j in bought:
                    pos[j] *= AC[i, j] / AO[i, j]
                else:
                    pos[j] *= AC[i, j] / AC[i - 1, j]
        if i in rset and i < iend:
            t = rset[i]; target = picks[t]; stopped = set(); pend = set()
            tot = cash + sum(pos.values())
            if out_mkt:
                cash = tot - sum(v * H for v in pos.values()); pos = {}
            else:
                new = [ci[c] for c in target]
                sold = sum(v for j, v in pos.items() if j not in new)
                tot -= sold * H
                keep = {j for j in pos if j in new}
                per = tot / n
                buy_amt = per * len([j for j in new if j not in keep])
                tot2 = tot - buy_amt * H
                per = tot2 / n
                pos = {j: per for j in new}
                cash = tot2 - per * len(new)
        vals[i] = cash + sum(pos.values())
        # 收盤後的訊號
        if S is not None:
            for j in pos:
                if S[i, j]: pend.add(j)
        if MB is not None:
            if not out_mkt and MB[i] and pos: p_exit = True
            if out_mkt and not MB[i] and target: p_entry = True
    v = pd.Series(vals, index=idx).iloc[i0:iend + 1]
    r = v.pct_change().dropna()
    r.attrs['stops'] = nst; r.attrs['switches'] = nsw
    return r


def stats(D, r, base=None, a=None, b=None, off=False):
    m = pd.Series(True, index=r.index)
    if a: m &= r.index >= pd.Timestamp(a)
    if b: m &= r.index <= pd.Timestamp(b)
    if off: m &= D['off_day'].reindex(r.index).fillna(False)
    x = r[m]; b50 = D['r0050'].reindex(x.index).fillna(0)
    o = {**L.perf(x), 'sharpe0': None}
    if o.get('vol'): o['sharpe0'] = round(o['CAGR'] / o['vol'], 2)
    mx, mb = L.monthly(x), L.monthly(b50)
    o['t_vs_0050'] = L.tstat(mx - mb); o['0050_CAGR'] = L.perf(b50).get('CAGR')
    if base is not None:
        mbase = L.monthly(base[m].reindex(x.index))
        o['diff_vs_base_pct_m'] = round(float((mx - mbase).mean() * 100), 3); o['t_vs_base'] = L.tstat(mx - mbase)
    return o


def main():
    D = L.load()
    C = D['AC']
    m20, m60 = L.sma(C, 20), L.sma(C, 60)
    dif, dea = L.macd(D); K, _ = L.kd(D); rs = L.rsi(D)
    prev = lambda X: X.shift(1).reindex(M.dates)        # 換股日前一交易日收盤
    masks = {'E1_above_MA60_rising': prev((C > m60) & (m60 > m60.shift(5))), 'E2_RSI<=70': prev(rs <= 70),
             'E3_MACD_DIF>DEA': prev(dif > dea), 'E4_close>MA20': prev(C > m20), 'E5_K<=80': prev(K <= 80)}
    stops = {'X1_stop_close<MA20': (C < m20), 'X2_stop_MACD_dead': L.cross_dn(dif, dea), 'X5_stop_close<MA60': (C < m60)}
    tx = pd.read_csv(L.BENCH / 'taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index()
    tx = tx.reindex(D['idx'].union(tx.index)).ffill().reindex(D['idx'])
    tdif = tx.ewm(span=12, adjust=False).mean() - tx.ewm(span=26, adjust=False).mean(); tdea = tdif.ewm(span=9, adjust=False).mean()
    mkts = {'X3_TAIEX<MA60_cash': tx < tx.rolling(60).mean(), 'X4_TAIEX_MACD<0_cash': tdif < tdea}
    base_p = picks_with()
    # 驗證：與原程式選股一致
    _, _, _, orig = M.run(M.yoy3)
    assert all(orig[t] == base_p[t] for t in orig), 'picks mismatch'
    variants = {'BASE': engine(D, base_p)}
    for k, mk in masks.items():
        mk = mk.fillna(False).astype(bool)
        variants[k + '_R'] = engine(D, picks_with(mk))
        variants[k + '_D'] = engine(D, picks_with(mk, drop=True))
    for k, st in stops.items():
        variants[k] = engine(D, base_p, stop=st.fillna(False))
    for k, mb in mkts.items():
        variants[k] = engine(D, base_p, mkt_bad=mb)
    base = variants['BASE']
    W = {'FULL_2021-03_2026-10': (None, None, False), 'SEL_2021_2023': ('2021-01-01', '2023-12-31', False),
         'VAL_2024_2026': ('2024-01-01', '2026-12-31', False), 'VAL_2024_2026_off': ('2024-01-01', '2026-12-31', True)}
    res = {'check_original_bt_final_rule_CAGR': M.ann(M.run(M.yoy3)[0], M.dates)}
    rows = {w: [] for w in W}
    for k, r in variants.items():
        res[k] = {w: stats(D, r, None if k == 'BASE' else base, *a) for w, a in W.items()}
        res[k]['stops'] = r.attrs.get('stops'); res[k]['switches'] = r.attrs.get('switches')
        res[k]['yearly_pct'] = L.yearly(r)
        for w in W:
            s = res[k][w]
            rows[w].append({'variant': k, 'CAGR': s.get('CAGR'), 'MDD': s.get('maxDD'), 'vol': s.get('vol'), 'sh': s.get('sharpe0'),
                            't50': s.get('t_vs_0050'), 'd_base%m': s.get('diff_vs_base_pct_m'), 't_base': s.get('t_vs_base')})
    sb = res['BASE']['SEL_2021_2023']
    useful = []
    for k in variants:
        if k == 'BASE': continue
        s = res[k]['SEL_2021_2023']
        ok = (s['CAGR'] > sb['CAGR'] and s['maxDD'] >= sb['maxDD']) or (s['maxDD'] - sb['maxDD'] >= 0.05 and sb['CAGR'] - s['CAGR'] < 0.02)
        res[k]['useful_in_selection'] = bool(ok)
        if ok: useful.append(k)
    res['useful_in_selection'] = useful
    # 對照：把原策略「部分放現金」調到與濾網版本相同波動（同期間），看濾網是否只是等同減碼
    vm = {}
    for k, r in variants.items():
        if k == 'BASE': continue
        vm[k] = {}
        for w, (a, b, off) in W.items():
            mk = pd.Series(True, index=r.index)
            if a: mk &= r.index >= pd.Timestamp(a)
            if b: mk &= r.index <= pd.Timestamp(b)
            if off: mk &= D['off_day'].reindex(r.index).fillna(False)
            x, bb = r[mk], base.reindex(r.index)[mk]
            wgt = float(x.std() / bb.std())
            pb = L.perf(wgt * bb)
            vm[k][w] = {'cash_blend_weight': round(wgt, 3), 'blend_CAGR': pb.get('CAGR'), 'blend_maxDD': pb.get('maxDD'),
                        'filter_CAGR': res[k][w]['CAGR'], 'filter_maxDD': res[k][w]['maxDD']}
    res['vol_matched_cash_blend_of_BASE'] = vm
    res['tables'] = rows
    L.save('results_main_filter.json', res)
    print('orig', res['check_original_bt_final_rule_CAGR'])
    for w in W:
        print('==', w); print(pd.DataFrame(rows[w]).to_string(index=False))
    print('useful in selection:', useful)
    print(pd.DataFrame({k: {f'{w}_{m}': v[w][m] for w in v for m in ('cash_blend_weight', 'blend_CAGR', 'blend_maxDD', 'filter_CAGR', 'filter_maxDD')} for k, v in vm.items()}).T.to_string())
    print({k: (res[k]['stops'], res[k]['switches']) for k in variants})


if __name__ == '__main__':
    main()
