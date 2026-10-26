#!/usr/bin/env python3
"""經典技術分析 21 條規則（事前固定，預設參數，不調參），個股訊號 → 隔日開盤進場 → 依規則出場；
交易層級統計＋上限 30 檔等權組合。輸出 results_rules.json、rules_events.pkl（scratchpad，供合成用）。

規則（強度＝同日訊號多於空位時的排序依據，越大越優先）：
 T01 KD(9,3,3)：K 上穿 D 且 K<20 買；K 下穿 D 賣。強度 20−K。
 T02 MACD(12,26,9)：DIF 上穿 DEA 買；下穿賣。強度 (DIF−DEA)/收盤。
 T03 RSI(14)：RSI<30 買；RSI>70 賣。強度 30−RSI。
 T04 布林(20,2) 突破上軌：收盤上穿上軌買；收盤跌破中軌賣。強度 %B。
 T05 布林(20,2) 觸下軌：收盤 ≤ 下軌買；收盤 ≥ 中軌賣。強度 −%B。
 T06 通道收窄突破：過去 10 日內帶寬曾為 120 日最低，今日收盤上穿上軌買；跌破中軌賣。強度 %B。
 T07a-e 帶量突破：收盤 > 前 60 日最高收盤、且成交值 > 前 20 日均值 2 倍。
        a 持有 5 日／b 20 日／c 60 日／d 跌破 20 日均線賣（不設期限）／e 60 日或跌破 20 日均線先到者。強度＝成交值倍數。
 T08 均線 5/20 黃金交叉買、死亡交叉賣。強度 收盤/MA20−1。
 T09 均線 20/60 黃金交叉買、死亡交叉賣。強度 收盤/MA60−1。
 T10 均線糾結突破：過去 5 日內曾有 max(MA5,10,20,60)/min−1 < 2%，今日收盤上穿四線最高者買；跌破 MA20 賣。強度 收盤/四線最高−1。
 T11 站上季線且季線上彎：收盤上穿 MA60 且 MA60 高於 5 日前買；收盤跌破 MA60 賣。強度 收盤/MA60−1。
 T12a/b 連三紅：連 3 日收 > 開 且收 > 前收（第 3 日訊號），持有 5／20 日。強度 3 日報酬。
 T13a/b 長紅突破：實體（收/開−1）> 6%、收 ≥ 最高×0.99、收 > 前 20 日最高收盤，持有 5／20 日。強度 實體大小。
 T14a/b 跳空上漲未回補 3 日：第 g 日 min(開,收) > 前日最高（低點以實體下緣近似），之後 3 日 min(開,收) 都 > g−1 日最高，
        第 g+3 日收盤確認、隔日開盤買，持有 5／20 日。強度 缺口大小。
"""
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tlib as L


def build_rules(D):
    C, O, H = D['AC'], D['AO'], D['AH']
    V = D['value']
    rules = {}
    K, Dd = L.kd(D)
    rules['T01_KD_golden_K<20'] = (L.cross_up(K, Dd) & (K < 20), 20 - K, L.cross_dn(K, Dd), None)
    dif, dea = L.macd(D)
    rules['T02_MACD_cross'] = (L.cross_up(dif, dea), (dif - dea) / C, L.cross_dn(dif, dea), None)
    r = L.rsi(D)
    rules['T03_RSI<30_buy_>70_sell'] = (r < 30, 30 - r, r > 70, None)
    m20, up, lo = L.boll(D)
    pb = (C - lo) / (up - lo).replace(0, np.nan)
    rules['T04_BB_break_upper'] = (L.cross_up(C, up), pb, C < m20, None)
    rules['T05_BB_touch_lower'] = (C <= lo, -pb, C >= m20, None)
    bw = (up - lo) / m20
    sq = (bw <= bw.rolling(120, min_periods=120).min() + 1e-12)
    sq10 = sq.astype(float).shift(1).rolling(10, min_periods=1).max() > 0
    rules['T06_BB_squeeze_breakout'] = (sq10 & L.cross_up(C, up), pb, C < m20, None)
    hi60 = C.shift(1).rolling(60, min_periods=60).max()
    vr = V / V.shift(1).rolling(20, min_periods=20).mean()
    vb = (C > hi60) & (vr > 2)
    rules['T07a_volbreak_hold5'] = (vb, vr, None, 5)
    rules['T07b_volbreak_hold20'] = (vb, vr, None, 20)
    rules['T07c_volbreak_hold60'] = (vb, vr, None, 60)
    rules['T07d_volbreak_exitMA20'] = (vb, vr, C < m20, None)
    rules['T07e_volbreak_60orMA20'] = (vb, vr, C < m20, 60)
    m5, m10, m60 = L.sma(C, 5), L.sma(C, 10), L.sma(C, 60)
    rules['T08_MA5x20'] = (L.cross_up(m5, m20), C / m20 - 1, L.cross_dn(m5, m20), None)
    rules['T09_MA20x60'] = (L.cross_up(m20, m60), C / m60 - 1, L.cross_dn(m20, m60), None)
    arr = [m5.to_numpy(), m10.to_numpy(), m20.to_numpy(), m60.to_numpy()]
    mx = pd.DataFrame(np.maximum.reduce(arr), index=C.index, columns=C.columns)
    mn = pd.DataFrame(np.minimum.reduce(arr), index=C.index, columns=C.columns)
    conv = (mx / mn - 1 < 0.02)
    conv5 = conv.astype(float).shift(1).rolling(5, min_periods=1).max() > 0
    rules['T10_MA_converge_breakout'] = (conv5 & L.cross_up(C, mx), C / mx - 1, C < m20, None)
    rules['T11_above_MA60_rising'] = (L.cross_up(C, m60) & (m60 > m60.shift(5)), C / m60 - 1, C < m60, None)
    red = (C > O) & (C > C.shift(1))
    r3 = red & red.shift(1, fill_value=False) & red.shift(2, fill_value=False)
    ret3 = C / C.shift(3) - 1
    rules['T12a_3red_hold5'] = (r3, ret3, None, 5)
    rules['T12b_3red_hold20'] = (r3, ret3, None, 20)
    body = C / O - 1
    lr = (body > 0.06) & (C >= H * 0.99) & (C > C.shift(1).rolling(20, min_periods=20).max())
    rules['T13a_longred_break_hold5'] = (lr, body, None, 5)
    rules['T13b_longred_break_hold20'] = (lr, body, None, 20)
    bl = np.minimum(O, C)                     # 低點近似：實體下緣
    Hp = H.shift(1)
    gap = bl > Hp
    ref = Hp                                   # 缺口下緣＝g−1 日最高
    ok3 = gap.shift(3, fill_value=False)
    for k in (0, 1, 2):
        ok3 &= bl.shift(k) > ref.shift(3)
    gsize = (bl / Hp - 1).shift(3)
    rules['T14a_gap_unfilled3_hold5'] = (ok3, gsize, None, 5)
    rules['T14b_gap_unfilled3_hold20'] = (ok3, gsize, None, 20)
    return rules


def main():
    t0 = time.time()
    D = L.load()
    print('loaded', round(time.time() - t0, 1))
    rules = build_rules(D)
    print('indicators', round(time.time() - t0, 1))
    res, rows, evs, pfs = {}, {w: [] for w in L.WIN}, {}, {}
    for name, (sig, stg, ex, hold) in rules.items():
        ev = L.make_events(D, sig, stg, ex, hold, name)
        o, pf, evd = L.evaluate(D, ev, name)
        res[name] = o; evs[name] = ev; pfs[name] = pf
        for w in L.WIN: rows[w].append(L.row(o, w))
        s = o['S']
        print(f"{name:30s} ev={len(ev):7d} S: n={s['trades_all_signals'].get('n')} net%={s['trades_all_signals'].get('mean_net_pct')} "
              f"CAGR={s['portfolio']['net'].get('CAGR')} tEW={s['portfolio']['t_vs_EW']} pass={o['pass_S']}  ({time.time()-t0:.0f}s)", flush=True)
    for w in L.WIN:
        print('\n==', w); print(pd.DataFrame(rows[w]).to_string(index=False))
    L.save('results_rules.json', {'rules': res, 'tables': rows,
                                   'candidates_pass_S': [k for k, v in res.items() if v['pass_S']]})
    with open(L.PKL, 'wb') as f:
        pickle.dump({'events': evs, 'daily': {k: v for k, v in pfs.items()}}, f)
    print('done', round(time.time() - t0, 1))


if __name__ == '__main__':
    main()
