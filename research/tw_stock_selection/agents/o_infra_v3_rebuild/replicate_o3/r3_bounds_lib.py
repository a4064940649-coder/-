"""B05／B10：終止上市名單內 4 碼國內股，訊號日 ≤ 終止日，營收以外條件都合格但缺營收 → 放回股池，營收百分位設 0.5／1.0。"""
import sys
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_infra_v3_rebuild/replicate_o3')
import r3_lib as L
sys.path.pop(0)
import numpy as np, pandas as pd
K, V = L.K, L.V

OFFd = L.OFF[(L.OFF['four'] == True) & (L.OFF['in_panel'] == True)].copy()
def domestic(nm):
    return not ('KY' in nm or 'DR' in nm)
OFFd = OFFd[OFFd['name'].map(domestic) & ~OFFd['code'].str.startswith('91')]
TERM = {}
for c, g in OFFd.groupby('code'):
    TERM[c] = sorted(g['date'])

def cand_mask(d):
    m = np.zeros(K.N, bool)
    for c, ds in TERM.items():
        if any(d <= x for x in ds) and c in K.CI: m[K.CI[c]] = True
    return m

class SB(L.S3):
    def __init__(self, YL, Y5, exfin, dmode, pct, order='post', rankm='a', addy='inf'):
        super().__init__(YL, Y5, exfin, dmode)
        self.addy = addy; self.pct = pct; self.order_mode = order; self.rankm = rankm; self.added = {}

    def cross_section(self, d):
        if d in self._cs: return self._cs[d]
        t = K.IDX.get_loc(d); e = t + 1
        v60 = self.V60[t]; px = K.Cn[t]
        base = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10)
        vv = np.where(base, v60, -np.inf)
        order = np.argsort(-vv, kind='stable')
        top = np.zeros(K.N, bool); top[order[:min(300, base.sum())]] = True
        top &= base
        y3 = self.y3(d)
        s52 = self.S52[t]
        ok = top & np.isfinite(s52) & ~self.dmask(d) & np.isfinite(K.On[e]) & K.TRADED[e]
        el = ok & np.isfinite(y3)
        add = ok & ~np.isfinite(y3) & cand_mask(d)
        if self.exfin: add &= ~V.FINMASK
        for c in getattr(self, 'force_add', {}).get(d, []): add[K.CI[c]] = ok[K.CI[c]] and not np.isfinite(y3[K.CI[c]])
        for c in getattr(self, 'force_drop', {}).get(d, []): add[K.CI[c]] = False
        def extra(el):
            if self.dmode == 'hi' and L.DHI_EXTRA.get(d, 0) > 0 and d < L.DSWITCH:
                for _ in range(L.DHI_EXTRA[d]):
                    s = np.where(el, s52, -np.inf); j = np.where(s == s.max())[0][0]; el[j] = False
            return el
        if self.order_mode == 'pre':
            el = extra(el); full = el | add
        else:
            full = extra(el | add); add = add & full; el = full & ~add
        if add.any(): self.added[d] = [K.COLS[j] for j in np.where(add)[0]]
        r52 = pd.Series(np.where(full, s52, np.nan)).rank(pct=True).to_numpy()
        if self.rankm == 'a':
            ry = pd.Series(np.where(el, y3, np.nan)).rank(pct=True).to_numpy()
            ry = np.where(add, self.pct, ry)
        else:   # b：一起排名，放回者營收值＝+inf（B10）或池內中位數（B05）
            yv = np.where(add, np.inf if self.pct >= 1 else np.nanmedian(np.where(el, y3, np.nan)), y3)
            ry = pd.Series(np.where(full, yv, np.nan)).rank(pct=True).to_numpy()
        sc = 0.5 * r52 + 0.5 * ry
        sc[~full] = np.nan
        y3o = np.where(add, {'nan': np.nan, 'inf': np.inf, 'pctinf': (np.inf if self.pct >= 1 else np.nan)}[self.addy], y3)
        cs = dict(t=t, e=e, el=full, sc=sc, s52=s52, y3=y3o, nold=0, add=add)
        self._cs[d] = cs
        return cs
