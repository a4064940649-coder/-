#!/usr/bin/env python3
"""台股交易成本、損益兩平與期望值試算（研究報告附件）

用途：把「交易成本如何吃掉短線策略的期望值」用公式與數字講清楚。
所有參數都可在下方 PARAMS 區修改後重跑：python3 cost_expectancy_calc.py

注意：費率以 2026-10 已查證之公開資料為準（見報告），券商折扣因人而異。
"""
from dataclasses import dataclass

# ---------- PARAMS（可修改） ----------
COMM_RATE = 0.001425          # 券商手續費上限 0.1425%（單邊）
TAX_STOCK = 0.003             # 證交稅 0.3%（賣出時課）
TAX_DAYTRADE = 0.0015         # 現股當沖證交稅減半 0.15%（至 2027-12-31）
TAX_ETF = 0.001               # ETF 證交稅 0.1%
FUT_TAX = 0.00002             # 期貨交易稅：契約金額的 十萬分之二（單邊）
DISCOUNTS = {"無折扣": 1.0, "6折": 0.6, "2.8折": 0.28}
TAIEX_LEVEL = 46000           # 試算用指數水位：加權指數 2026-10-22 收盤 46,041（取整）
FUT_SPECS = {                 # 名稱: (每點價值 NT$, 單邊手續費 NT$ 假設)
    "大台 TX": (200, 45),     # 單邊手續費為券商常見報價區間中值（未查證官方；含期交所費 20）
    "小台 MTX": (50, 22),     # 含期交所費 12.5
    "微台 TMF": (10, 12),     # 含期交所費 8
}
SLIPPAGE_TICKS = 1            # 假設每趟各吃 1 檔滑價（進+出共 2 檔）

@dataclass
class RoundTrip:
    name: str
    cost_pct: float           # 來回總成本（佔成交金額 %）

def stock_roundtrip(discount: float, tax: float) -> float:
    return COMM_RATE * discount * 2 + tax

def tick_size(price: float) -> float:
    """台股升降單位（上市股票）"""
    if price < 10: return 0.01
    if price < 50: return 0.05
    if price < 100: return 0.1
    if price < 500: return 0.5
    if price < 1000: return 1.0
    return 5.0

def futures_roundtrip_pct(point_value: float, comm_per_side: float, index: float) -> float:
    notional = point_value * index
    tax = notional * FUT_TAX * 2
    comm = comm_per_side * 2
    return (tax + comm) / notional

def expectancy_R(p: float, rr: float, cost_in_R: float) -> float:
    """每筆期望值（以 R = 單筆風險 為單位）
    E = p×RR − (1−p)×1 − cost_in_R
    """
    return p * rr - (1 - p) * 1 - cost_in_R

def breakeven_winrate(rr: float, cost_in_R: float) -> float:
    """令 E = 0 解 p：p = (1 + cost) / (1 + RR)"""
    return (1 + cost_in_R) / (1 + rr)

def kelly(p: float, rr: float) -> float:
    return p - (1 - p) / rr


def prob_losing_streak(n_trades: int, streak: int, loss_prob: float) -> float:
    """在 n_trades 筆交易中，至少出現一次「連續 >= streak 筆虧損」的機率（馬可夫鏈精算）"""
    states = [0.0] * streak
    states[0] = 1.0
    absorbed = 0.0
    for _ in range(n_trades):
        new = [0.0] * streak
        for k, pr in enumerate(states):
            if pr == 0.0:
                continue
            new[0] += pr * (1 - loss_prob)          # 獲利 → 連虧歸零
            if k + 1 >= streak:
                absorbed += pr * loss_prob          # 達到門檻
            else:
                new[k + 1] += pr * loss_prob
        states = new
    return absorbed

def main():
    print("=" * 72)
    print("① 現股來回交易成本（佔成交金額 %）")
    print("=" * 72)
    rows = []
    for dname, d in DISCOUNTS.items():
        swing = stock_roundtrip(d, TAX_STOCK)
        dt = stock_roundtrip(d, TAX_DAYTRADE)
        etf = stock_roundtrip(d, TAX_ETF)
        rows.append((dname, swing, dt, etf))
        print(f"{dname:>6} | 隔日沖/波段(稅0.3%) {swing*100:.3f}% | 現股當沖(稅0.15%) {dt*100:.3f}% | ETF(稅0.1%) {etf*100:.3f}%")
    print("\n公式：來回成本 = 手續費×折扣×2 + 證交稅（賣出）")
    print(f"例：6折當沖 = 0.1425%×0.6×2 + 0.15% = {0.1425*0.6*2:.4f}% + 0.15% = {stock_roundtrip(0.6, TAX_DAYTRADE)*100:.4f}%")

    print("\n" + "=" * 72)
    print("② 成本換算成「檔數」（含每趟 1 檔滑價）")
    print("=" * 72)
    for price in (30, 80, 150, 600, 1200):
        t = tick_size(price)
        tick_pct = t / price
        cost = stock_roundtrip(0.6, TAX_DAYTRADE)
        total = cost + 2 * tick_pct * SLIPPAGE_TICKS
        print(f"股價 {price:>5} | 1檔={t} ({tick_pct*100:.3f}%) | 當沖成本(6折)={cost*100:.3f}% ≈ {cost/tick_pct:.2f}檔 | 含滑價總計 {total*100:.3f}% ≈ {total/tick_pct:.2f}檔")

    print("\n" + "=" * 72)
    print(f"③ 台指期來回成本（指數 {TAIEX_LEVEL} 點試算）")
    print("=" * 72)
    for name, (pv, comm) in FUT_SPECS.items():
        pct = futures_roundtrip_pct(pv, comm, TAIEX_LEVEL)
        pts = pct * TAIEX_LEVEL
        notional = pv * TAIEX_LEVEL
        print(f"{name:>7} | 契約值 NT${notional:,.0f} | 來回稅 NT${notional*FUT_TAX*2:,.1f} + 手續費 NT${comm*2} | 合計 {pct*100:.4f}% ≈ {pts:.2f} 點")
    print("公式：來回成本% = (契約值×0.002%×2 + 手續費×2) / 契約值")

    print("\n" + "=" * 72)
    print("④ 期望值：成本如何吃掉優勢（R = 單筆停損距離）")
    print("=" * 72)
    scenarios = [
        ("現股當沖 6折, 停損 1%", stock_roundtrip(0.6, TAX_DAYTRADE), 0.01),
        ("現股當沖 2.8折, 停損 1%", stock_roundtrip(0.28, TAX_DAYTRADE), 0.01),
        ("現股波段 6折, 停損 5%", stock_roundtrip(0.6, TAX_STOCK), 0.05),
        ("小台當沖, 停損 0.5% (≈230點)", futures_roundtrip_pct(50, 22, TAIEX_LEVEL), 0.005),
        ("小台當沖, 停損 0.2% (≈92點)", futures_roundtrip_pct(50, 22, TAIEX_LEVEL), 0.002),
        ("微台當沖, 停損 0.2% (≈92點)", futures_roundtrip_pct(10, 12, TAIEX_LEVEL), 0.002),
    ]
    p, rr = 0.40, 2.0
    print(f"假設：勝率 p={p:.0%}、盈虧比 RR={rr:.1f} → 毛期望值 = {p}×{rr} − {1-p:.1f}×1 = {p*rr-(1-p):+.2f} R/筆")
    for name, cost, stop in scenarios:
        cR = cost / stop
        e = expectancy_R(p, rr, cR)
        be = breakeven_winrate(rr, cR)
        print(f"{name:<30} | 成本={cost*100:.3f}% = {cR:.2f}R | 淨期望值 {e:+.2f}R/筆 | 損益兩平勝率 {be:.1%}")
    print("\n公式：E = p×RR − (1−p) − 成本/R ；損益兩平勝率 p* = (1+成本/R)/(1+RR)")

    print("\n" + "=" * 72)
    print("⑤ 凱利公式與破產風險提醒")
    print("=" * 72)
    for p_, rr_ in [(0.40, 2.0), (0.50, 1.5), (0.35, 3.0), (0.55, 1.0)]:
        k = kelly(p_, rr_)
        print(f"p={p_:.0%}, RR={rr_:.1f} → 全凱利 f*={k:+.1%}；實務建議 ≤ 1/4 凱利 = {max(k,0)/4:.1%} 每筆風險")
    print("公式：f* = p − (1−p)/RR")

    print("\n" + "=" * 72)
    print("⑥ 連續虧損機率（勝率 40% → 單筆虧損機率 60%；馬可夫鏈精算）")
    print("=" * 72)
    for n in (100, 250, 500):
        row = " | ".join(f"連虧≥{k}: {prob_losing_streak(n, k, 0.6):.1%}" for k in (8, 10, 12))
        print(f"{n:>3} 筆交易 | {row}")
    print("解讀：一年 250 筆交易幾乎必然碰到連虧 8 筆，近半機率碰到連虧 10 筆；部位大小必須以此為前提。")

if __name__ == "__main__":
    main()
