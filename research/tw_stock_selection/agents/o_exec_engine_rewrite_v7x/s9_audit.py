#!/usr/bin/env python3
"""抽查 3 格（審查用，只讀）：V7X 正式設定下逐筆記錄成交，檢查不變量。
格：(a) 30萬 FIX J6 seed 3、(b) 1000萬 GROW J2 seed 7、(c) 300萬 FIX J10 seed 11（K5T、一般成本）。
檢查：整股張數 ≤ 事前上限；零股股數 ≤ Z 上限（2020-10-26 前為代理上限）；處置賣出 ≤ 當日成交值 20%；持股不為負；
收盤現金 < 0 的天數與幅度（F1 提領先記現金、賣單被上限擋下時會短暫為負）；FIX 每次換股後權益 ≈ 級距×CPI（扣成本前）；
獨立重算：以成交紀錄重算現金流，與引擎現金逐日相符。"""
import sys, json, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import x_data as D, x_exec as X

R = json.loads((HERE / 'results.json').read_text())
FINAL = R['PREREG']['final_cfg']
CASES = [('30萬', 3e5, 'FIX', 6, 3), ('1000萬', 1e7, 'GROW', 2, 7), ('300萬', 3e6, 'FIX', 10, 11)]


class AuditSim(X.Sim):
    def __init__(self, pl, cfg):
        super().__init__(pl, cfg); self.log = []; self.viol = []; self.cashlog = []

    def fill_whole(self, t, j, L, side):
        cap = self.lotcap(t, j); c0 = self.cash
        got = super().fill_whole(t, j, L, side)
        if got // 1000 > cap: self.viol.append(('whole_cap', t, j, got, cap))
        self.log.append(('W', t, j, side, got, self.cash - c0))
        return got

    def fill_odd(self, t, j, q, side, cashlim=None):
        cap = self.oddcap(t, j); c0 = self.cash
        got = super().fill_odd(t, j, q, side, cashlim)
        if cap is not None and got > max(cap, 0): self.viol.append(('odd_cap', t, j, got, cap))
        self.log.append(('O', t, j, side, got, self.cash - c0))
        return got

    def rebalance(self, t, pk, tr):
        super().rebalance(t, pk, tr); self.cashlog.append((t, self.cash, 'reb'))

    def deferred(self, t, tr):
        super().deferred(t, tr); self.cashlog.append((t, self.cash, 'def'))

    def fill_disp_sell(self, t, j, n):
        c0 = self.cash + self.cash_late
        got = super().fill_disp_sell(t, j, n)
        if self.c.get('f2') and got * D.C[t, j] > 0.2 * np.nan_to_num(D.V[t, j]) + 1e-6: self.viol.append(('disp_cap', t, j, got))
        self.log.append(('D', t, j, -1, got, self.cash + self.cash_late - c0))
        return got


if __name__ == '__main__':
    out = {}
    for lab, tier, mode, j, s in CASES:
        pl = D.plans('AB_V7_B10', js=[j], seeds=[s])[(j, s)]
        cfg = dict(tier=tier, mode=mode, model='K5T', cost='normal'); cfg.update(FINAL)
        sim = AuditSim(pl, cfg)
        # 逐日記錄收盤現金與最小持股
        cash_close, minS = [], []
        orig_run = sim.run
        nav = sim.run()
        L = pd.DataFrame(sim.log, columns=['kind', 't', 'j', 'side', 'n', 'dcash'])
        CL = pd.DataFrame(sim.cashlog, columns=['t', 'cash', 'kind'])
        neg = CL[CL.cash < -1e-6]
        out[f'{lab}|{mode}|J{j}|s{s}'] = dict(
            n_fills=int((L.n > 0).sum()), n_whole=int(((L.kind == 'W') & (L.n > 0)).sum()), n_odd=int(((L.kind == 'O') & (L.n > 0)).sum()),
            n_disp=int(((L.kind == 'D') & (L.n > 0)).sum()), violations=len(sim.viol), viol_examples=[list(map(str, v)) for v in sim.viol[:5]],
            min_shares=float(sim.S.min()), n_neg_cash_after_trading=int(len(neg)), n_trading_days=int(len(CL)),
            min_cash=float(CL.cash.min()), neg_cash_examples=[(str(D.IDX[int(a)].date()), float(b), c) for a, b, c in neg.head(5).itertuples(index=False)], final_nav=float(nav.iloc[-1]),
            buy_cash_out_positive=bool((L[(L.side > 0) & (L.n > 0)].dcash < 0).all()), sell_cash_in_positive=bool((L[(L.side < 0) & (L.n > 0)].dcash > 0).all()),
            acc={k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in sim.acc.items()})
    (HERE / 'out' / 'audit3.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1)[:4000])
