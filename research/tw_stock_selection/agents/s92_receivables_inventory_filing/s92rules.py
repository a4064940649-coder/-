#!/usr/bin/env python3
"""S92 格子（事前固定，見 PREREG.json）。門檻與方向只看訓練期（訊號 2016-02～2021-12）現行交易的分組決定。
新格 10 格（時點＝實際公告日 fd；只判一般業；缺值不擋）：
  INV20／INV30＝存貨年增率 − 單季營收年增率 > 20／30 個百分點 → 不買
  AR30        ＝應收（帳款＋關係人＋票據）年增率 − 單季營收年增率 > 30 個百分點 → 不買
  BOTH10      ＝應收、存貨兩者都比營收快 10 個百分點以上 → 不買
  EITHER30    ＝應收或存貨任一比營收快 30 個百分點以上 → 不買
  DIO20       ＝存貨周轉天數比去年同季多 20 天以上 → 不買
  處理：Z＝不遞補（錢照引擎放 0050）；F＝由第 11～20 名中動能過關且沒被同一規則擋者依 SURGE 遞補。
S88 時點檢驗 2 格（不參與選格）：R09FD|Z、R09FD|F＝S88 凍結規則（應計比率在股池最高 20% 或近四季營業現金流為負）改用實際公告日選季。
參考（不算格、不參與選格）：R09S88|Z＝S88 原時點估法在新口徑（五開關）下重跑，用來和 R09FD|Z 對照。"""
import numpy as np
import s92lib as L

NEW_RULES = {
    'INV20': lambda X: X['inv_gap'] > 0.20,
    'INV30': lambda X: X['inv_gap'] > 0.30,
    'AR30': lambda X: X['ar_gap'] > 0.30,
    'BOTH10': lambda X: (X['ar_gap'] > 0.10) & (X['inv_gap'] > 0.10),
    'EITHER30': lambda X: (X['ar_gap'] > 0.30) | (X['inv_gap'] > 0.30),
    'DIO20': lambda X: X['dio_chg'] > 20.0,
}
NEW_CELLS = ('INV20|Z', 'INV20|F', 'INV30|Z', 'INV30|F', 'AR30|Z', 'AR30|F', 'BOTH10|Z', 'EITHER30|Z', 'DIO20|Z', 'DIO20|F')
S88_CELLS = ('R09FD|Z', 'R09FD|F')
REF_CELLS = ('R09S88|Z',)
ALL_CELLS = NEW_CELLS + S88_CELLS + REF_CELLS
assert len(NEW_CELLS) + len(S88_CELLS) == 12


class Rules:
    """badfn(cell) → 函式 k ↦ 長度 N 布林。每個訊號列的特徵有快取（換位移時 clear）。"""

    def __init__(self, P, U, FN, dry=False):
        self.P, self.U, self.FN, self.dry = P, U, FN, dry
        self._c = {}

    def clear(self):
        self._c = {}; self.FN.clear()

    def _x(self, k):
        if ('x', k) not in self._c:
            self._c[('x', k)] = L.featx(self.P, self.FN, k, 'fd')
        return self._c[('x', k)]

    def _r09(self, k, pit):
        if (pit, k) not in self._c:
            F = L.feat88(self.P, self.U, self.FN, k, pit)
            self._c[(pit, k)] = F['ACC_HI'] | F['OCF_NEG']
        return self._c[(pit, k)]

    def badfn(self, cell):
        rule = cell.split('|')[0]
        if self.dry:
            return lambda k: np.zeros(self.P.N, bool)
        if rule in NEW_RULES:
            fn = NEW_RULES[rule]

            def f(k):
                with np.errstate(invalid='ignore'):
                    return np.asarray(fn(self._x(k)), bool)
            return f
        if rule == 'R09FD':
            return lambda k: self._r09(k, 'fd')
        if rule == 'R09S88':
            return lambda k: self._r09(k, 's88')
        raise ValueError(cell)

    def apply(self, RT, S, cell, rng=None):
        return L.apply_bad(self.P, self.U, RT, S, self.badfn(cell), cell.split('|')[1], rng=rng)
