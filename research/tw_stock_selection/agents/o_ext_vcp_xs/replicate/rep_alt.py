# -*- coding: utf-8 -*-
"""以獨立官方日 K（jb7k，2015 起）重建替代面板：補回 2024-01-02～02-07 缺口、把 Yahoo 段換成官方開高低收量、
把 2019-05 斷點股往後接。用來量化資料缺口對 2019 起數字的影響（診斷用，凍結後才跑）。"""
import numpy as np, pandas as pd
import rep_lib as L


class AltData(L.Data):
    def __init__(self, base, fill_gap=True, replace_from='2015-01-05', extend=True):
        of = pd.read_parquet(L.S + 'rep_vcp/official_long.parquet')
        of['date'] = pd.to_datetime(of['date'])
        of = of[(of.market == 'twse') & (of.volume > 0) & of.close.notna()]
        W = {c: of.pivot(index='date', columns='code', values=c) for c in ('open', 'high', 'low', 'close', 'volume', 'amount')}
        od = W['close'].index
        dates = base.dates.union(od[(od > '2023-12-29') & (od < '2024-02-15')]) if fill_gap else base.dates
        self.dates = dates
        self.codes = base.codes
        pidx = base.dates
        P = {k: pd.DataFrame(getattr(base, k), index=pidx, columns=base.codes).reindex(dates) for k in ('C', 'O', 'H', 'L', 'V', 'VAL', 'F', 'FLAG', 'SRC')}
        A = {c: W[c].reindex(index=dates, columns=base.codes) for c in W}
        has_off = A['close'].notna().any(axis=0).values           # 該檔在官方有資料
        d2 = np.broadcast_to(dates.values[:, None], P['C'].shape)
        rep = (d2 >= np.datetime64(replace_from)) & has_off[None, :]
        pc = ~np.isnan(P['C'].values)
        rows = np.arange(len(dates))[:, None]
        firstbar = np.where(pc.any(0), pc.argmax(0), len(dates))
        lastbar = np.where(pc.any(0), len(dates) - 1 - pc[::-1].argmax(0), -1)
        rep = rep & (rows >= firstbar[None, :])                    # 面板第一根之前不補
        if not extend:
            rep = rep & (rows <= lastbar[None, :])                 # 面板最後一根之後不接
        self.n_added_after_last = int((rep & (rows > lastbar[None, :]) & A['close'].notna().values).sum())
        self.n_added_gap = int((rep & (d2 > np.datetime64('2023-12-29')) & (d2 < np.datetime64('2024-02-15')) & A['close'].notna().values).sum())
        self.n_added_inside = int((rep & ~pc & A['close'].notna().values & (rows <= lastbar[None, :])).sum()) - (self.n_added_gap - int((rep & (rows > lastbar[None, :]) & (d2 > np.datetime64('2023-12-29')) & (d2 < np.datetime64('2024-02-15')) & A['close'].notna().values).sum()))
        self.n_removed = int((rep & pc & A['close'].isna().values).sum())
        m = {'C': 'close', 'O': 'open', 'H': 'high', 'L': 'low', 'V': 'volume', 'VAL': 'amount'}
        for k, c in m.items():
            setattr(self, k, np.where(rep, A[c].values.astype(float), P[k].values))
        F = P['F'].ffill().values                               # 還原因子：缺口日與接續段沿用前值
        self.F = np.where(np.isnan(self.C), np.nan, F)
        self.A = self.C * self.F
        self.T, self.N = self.C.shape
        a = pd.DataFrame(self.A)
        af = a.ffill()
        self.R = (a / af.shift(1) - 1).values
        self.FLAG = np.where(np.isnan(P['FLAG'].values), 0.0, P['FLAG'].values)
        self.FLAG[np.isnan(self.C)] = np.nan
        self.SRC = P['SRC'].values
        self.replaced = rep & ~np.isnan(self.C)
        ix = pd.read_csv(L.IDX, parse_dates=['date']).set_index('date')['close']
        self.IDXC = ix.reindex(ix.index.union(dates)).ffill().reindex(dates).values
        ef = pd.read_csv(L.ETF, parse_dates=['date']).set_index('date')['adj']
        self.ETF = ef.reindex(ef.index.union(dates)).ffill().reindex(dates).values
        self.AO = self.O * self.F; self.AH = self.H * self.F; self.AL = self.L * self.F
        self.lim = np.where(dates < pd.Timestamp('2015-06-01'), 0.07, 0.10)
        self._pack()
        self._aux()
