#!/usr/bin/env python3
"""主策略（main_v2 正式：gapfix、排除金融、n10、正式次排序）加上盤中／隔夜訊號的三種結合方式。
(a) third：分數＝(52 週百分位＋營收百分位＋訊號百分位)/3
(b) mult ：分數＝0.5×(52 週百分位×訊號百分位)＋0.5×營收百分位
(c) drop ：原分數前 20 名中，剔除訊號百分位最低（最「隔夜型」）的 5 檔，其餘照原分數順序遞補
訊號缺值（例如盤中占比在總報酬 ≤ 0 時）給中位百分位 0.5。
src3_mode：'exclude'＝回看窗內 src 3 日數 > 20% 的訊號月改用原主策略分數（主版本）；'keep'＝照用（敏感度）。"""
import numpy as np, pandas as pd
import olib as L
K, V = L.K, L.V
IDX, COLS, N = L.IDX, L.COLS, L.N

def pr(x, el):
    return pd.Series(np.where(el, x, np.nan)).rank(pct=True).to_numpy()

class Combo(V.V2):
    def __init__(self, spec, combo, sp, YL, Y5, src3_mode='exclude', **kw):
        super().__init__('gapfix', YL, Y5, exfin=True, **kw)
        self.spec, self.combo, self.sp, self.src3_mode = spec, combo, sp, src3_mode
        self._cc = {}; self.f3 = {}; self.excluded = []

    def cross_section(self, d):
        if d in self._cc: return self._cc[d]
        base = super().cross_section(d)
        cs = dict(base)
        t = base['t']; el = base['el']
        Lw, S, form = self.spec
        CI, CO, ok, f3 = self.sp.window(t, Lw, S)
        self.f3[d] = f3
        X = L.form_signal(CI, CO, form)
        px = pr(X, el); px = np.where(el & ~np.isfinite(px), 0.5, px)
        cs['px'] = px; cs['X'] = X
        excl = (self.src3_mode == 'exclude') and np.isfinite(f3) and f3 > 0.20
        cs['excluded'] = bool(excl)
        if excl:
            self.excluded.append(d)
        elif self.combo in ('third', 'mult'):
            r52 = pr(base['s52'], el); rrv = pr(base['y3'], el)
            sc = (r52 + rrv + px) / 3 if self.combo == 'third' else 0.5 * r52 * px + 0.5 * rrv
            sc[~el] = np.nan
            cs['sc'] = sc
        self._cc[d] = cs
        return cs

    def plan(self, n, tie='official'):
        out = []; held = set(); nblk = 0
        for d in self.sig_days:
            cs = self.cross_section(d)
            el = cs['el']
            if el.sum() < 50: continue
            e = cs['e']; gu = K.gap_up_open(e)
            cand = list(self.order(cs, tie)[:int(el.sum())])
            if self.combo == 'drop' and not cs['excluded']:
                top20 = cand[:20]
                # 百分位最低的 5 檔（同值時排名較後者先剔除）
                o = sorted(range(len(top20)), key=lambda i: (cs['px'][top20[i]], -i))
                drop = {top20[i] for i in o[:5]}
                cand = [j for j in cand if j not in drop]
            pick = []
            for j in cand:
                if gu[j] and j not in held: nblk += 1; continue
                pick.append(j)
                if len(pick) == n: break
            out.append((e, np.array(pick, int), np.where(el)[0], d))
            held = set(pick)
        self.nblk = nblk
        return out
