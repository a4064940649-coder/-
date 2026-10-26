#!/usr/bin/env python3
"""用「現股」實際執行最終規則：整股（一張 1,000 股）或零股 × 資金規模，逐期模擬真實帳戶。

規則與 bt_final_rule.py 相同（52 週高點 × 近 3 月營收年增率，前 300 名流動性、排除處置股、前 30 檔）。
帳戶模擬（2021-03-11 起，每月換股日）：
  * 只用現金買賣（現股）：不融資、不融券、不當沖。同一交易日賣出與買進的款項 T+2 淨額交割，賣出所得可當天再買。
  * 賣出：不在新名單的持股全部賣出。續抱：不動。買進：新名單中尚未持有者，每檔預算 = min(總資產/30, 可用現金/新檔數)。
  * 整股：股數取 1,000 的倍數（買不起一張就跳過、現金閒置）；「只選買得起」版本先剔除一張超過預算的股票再取前 30 名。
  * 零股：股數取整數。
成本（假設，見報告）：手續費 0.1425% × 2.8 折、單筆最低 20 元（零股另測 1 元）；賣出證交稅 0.3%；
  滑價每邊整股 0.1%、零股 0.2%（另測 0.4%）。現金不計利息。持有期間報酬用含息還原報酬。
自我檢查：零成本、可買分數股、每期全部調回等權時，必須重現 bt_final_rule 的毛報酬。
輸出：output/bt_cash_execution.json
"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bt_final_rule as fr
from btlib import ann, yearly

OUT = Path(__file__).resolve().parent / 'output'
dates, R, B, PX = fr.dates, fr.R, fr.B, fr.px
EL = fr.t300 & fr.yoy3.notna() & fr.s52.notna() & ~fr.inD
SC = 0.5 * fr.s52.where(EL).rank(axis=1, pct=True) + 0.5 * fr.yoy3.where(EL).rank(axis=1, pct=True)
COMM, TAX = 0.001425 * 0.28, 0.003

def simulate(K, lot, min_fee, slip, affordable_only=False, n=30, reequalize=False, comm=COMM, tax=TAX):
    cash, pos = float(K), {}
    vals, held, idle, costs = [], [], [], 0.0
    def fee(v): return max(min_fee, v * comm) if v > 0 else 0.0
    for i, t in enumerate(dates):
        if i > 0:
            tp = dates[i - 1]
            for c in pos:
                r = R.at[tp, c]
                pos[c] *= (1 + r) if np.isfinite(r) else 1.0
        V = cash + sum(pos.values()); vals.append(V)
        if i == len(dates) - 1: break
        ok = EL.loc[t] & R.loc[t].notna()
        s = SC.loc[t][ok].sort_values(ascending=False)
        if affordable_only:
            s = s[(PX.loc[t, s.index] * lot) <= V / n]
        target = list(s.index[:n])
        exits = [c for c in pos if c not in target] if not reequalize else list(pos)
        for c in exits:
            v = pos.pop(c); k = fee(v) + v * (tax + slip); cash += v - k; costs += k
        new = [c for c in target if c not in pos]
        if new:
            budget = min(V / n, cash / len(new))
            for c in new:
                p = PX.at[t, c]
                if not (np.isfinite(p) and p > 0): continue
                sh = budget / (p * (1 + slip + comm)) if lot is None else math.floor(budget / (p * (1 + slip + comm)) / lot) * lot
                if sh <= 0: continue
                v = sh * p; k = fee(v) + v * slip
                if v + k > cash * (1 + 1e-9) + 1e-6: continue      # 相對容差，避免浮點誤差誤判現金不足
                cash -= v + k; costs += k; pos[c] = v
        held.append(len(pos)); idle.append(cash / V if V > 0 else 0.0)
    v = pd.Series(vals, index=pd.DatetimeIndex(dates))
    r = (v.shift(-1) / v - 1).iloc[:-1]
    a = ann(r, dates)
    return {'CAGR': a['CAGR'], 'maxDD': a['maxDD'], 'final_value': round(float(v.iloc[-1])), 'avg_names_held': round(float(np.mean(held)), 1),
            'min_names_held': int(np.min(held)), 'avg_idle_cash_pct': round(float(np.mean(idle)) * 100, 1),
            'cost_pct_of_avg_assets_per_year': round(float(costs / v.mean() / a['years'] * 100), 2), 'yearly_pct': yearly(r, dates)}

res = {'window': [str(dates[0].date()), str(dates[-1].date())], 'benchmark_0050_CAGR': ann(B, dates)['CAGR']}
# 自我檢查
chk = simulate(1e12, None, 0.0, 0.0, reequalize=True, comm=0.0, tax=0.0)
g, _, _, _ = fr.run(fr.yoy3, rt=0.0)
res['self_check'] = {'sim_frictionless_CAGR': chk['CAGR'], 'bt_final_rule_gross_CAGR': ann(g, dates)['CAGR']}
print('self-check', res['self_check'], flush=True)
res['ideal_fractional_with_costs'] = simulate(1e12, None, 0.0, 0.001)
modes = {'board_lot_skip_unaffordable': dict(lot=1000, min_fee=20, slip=0.001),
         'board_lot_affordable_only': dict(lot=1000, min_fee=20, slip=0.001, affordable_only=True),
         'odd_lot_min_fee_1': dict(lot=1, min_fee=1, slip=0.002),
         'odd_lot_min_fee_20': dict(lot=1, min_fee=20, slip=0.002),
         'odd_lot_min_fee_20_n20': dict(lot=1, min_fee=20, slip=0.002, n=20),
         'odd_lot_min_fee_20_slip_0.4pct': dict(lot=1, min_fee=20, slip=0.004)}
for m, kw in modes.items():
    res[m] = {}
    for K in [300_000, 1_000_000, 3_000_000, 10_000_000, 30_000_000]:
        x = simulate(K, **kw); res[m][str(K)] = x
        print(f"{m:30s} K={K:>11,} CAGR {x['CAGR']:.4f} MDD {x['maxDD']:.4f} names {x['avg_names_held']:5.1f} (min {x['min_names_held']:2d}) idle {x['avg_idle_cash_pct']:5.1f}% cost {x['cost_pct_of_avg_assets_per_year']:.2f}%/yr final {x['final_value']:,}", flush=True)
(OUT / 'bt_cash_execution.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print('ideal fractional with costs', {k: res['ideal_fractional_with_costs'][k] for k in ['CAGR', 'maxDD', 'cost_pct_of_avg_assets_per_year']}, '| 0050', res['benchmark_0050_CAGR'])
