# -*- coding: utf-8 -*-
"""替代題迭代（只用挑選期：訊號日 2006-04～2019-03，月窗在 2019-04-30 以前結束）。主策略 n10、排除金融、一般成本。
選版規則（事前寫死、跑之前）：受影響入選股月份（原版入選、版本未入選）≥ 30 的版本中，取挑選期配對差 t 最大者。
版本（6 版，題目上限 6）：
  A1 跳過並遞補：上市 1～3 年的新上市股（初次上市＋轉上市）
  A2 跳過並遞補：上市 1～2 年的新上市股
  A3 跳過並遞補：上市 1～3 年、只限初次上市
  A4 跳過並遞補：上市 1～3 年初次上市股，且持有期內含上市後 12/18/24/30/36 月週年（候選集保解除日）
  A5 跳過並遞補：上市 2～4 年初次上市股（舊制「屆滿 2 年後每半年領回 1/5」的解除期；探索看到 3～4 年負值後追加，標記為資料導向）
  A6 跳過並遞補：上市 1～3 年初次上市股，且上市年為熱市（IPO 家數 ≥ 2004～2018 中位數 18）
「權重減半」需要改引擎的等權假設，本輪不做（記為未測）。"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import alib as A
V, K = A.V, A.K
IDX, N = A.IDX, A.N

def mk(lo, hi, ipo_only=False, anniv=False, hot=False):
    def f(d, e, e2):
        a = A.age_years(d)
        m = (a >= lo) & (a < hi) & (A.IPOMASK if ipo_only else A.NEWMASK)
        if hot: m &= A.HOT
        if anniv:
            m = m & np.array([A.half_year_anniv_in(j, e, e2) if m[j] else False for j in range(N)])
        return m
    return f

VERS = {'A1': mk(1, 3), 'A2': mk(1, 2), 'A3': mk(1, 3, ipo_only=True), 'A4': mk(1, 3, ipo_only=True, anniv=True),
        'A5': mk(2, 4, ipo_only=True), 'A6': mk(1, 3, ipo_only=True, hot=True)}
SEL = ('2006-04-01', '2019-04-30')

def monthly(nav, sd):
    D = pd.DatetimeIndex([d for d in sd if d in nav.index]); nd = nav.reindex(D)
    return (nd.shift(-1) / nd - 1).iloc[:-1]

def run(versions, a, b, label, S=None, cost='normal'):
    S = S or A.strategy(exfin=True)
    t0, t1, sd = S.bounds()
    bb, ss = K.rates(0.001) if cost == 'normal' else K.rates('stress')
    base = S.plan(10, 'official')
    nav0, _ = K.simulate([(p[0], p[1]) for p in base], bb, ss, t0, t1)
    ew, _ = K.simulate([(p[0], p[2]) for p in base], 0, 0, t0, t1)
    m0 = monthly(nav0, sd)
    sdw = [d for d in sd if pd.Timestamp(a) <= d <= pd.Timestamp(b)]
    out = {'base': dict(seg=V.seg_stats(nav0, ew, a, b, sd))}
    navs = {'base': nav0, 'EW': ew}
    for k, fn in versions.items():
        pl, sk = A.plan_avoid(S, 10, fn)
        nav, _ = K.simulate([(p[0], p[1]) for p in pl], bb, ss, t0, t1)
        m = monthly(nav, sd)
        dd = (m - m0).dropna(); dd = dd[(dd.index >= pd.Timestamp(a)) & (dd.index <= pd.Timestamp(b)) & (dd.index < pd.Timestamp(b) - pd.Timedelta(days=20))]
        bp = {p[3]: set(p[1]) for p in base}; vp = {p[3]: set(p[1]) for p in pl}
        aff = sum(len(bp[d] - vp.get(d, set())) for d in bp if pd.Timestamp(a) <= d <= pd.Timestamp(b) - pd.Timedelta(days=20))
        out[k] = dict(seg=V.seg_stats(nav, ew, a, b, sd), paired=A.msum(dd), affected=int(aff), n_skipped=len([s for s in sk if pd.Timestamp(a) <= s[0] <= pd.Timestamp(b)]))
        navs[k] = nav
    return out, navs, S

if __name__ == '__main__':
    out, navs, S = run(VERS, *SEL, 'sel')
    elig = {k: v for k, v in out.items() if k != 'base' and v['affected'] >= 30}
    pick = max(elig, key=lambda k: elig[k]['paired']['t']) if elig else None
    res = dict(rule='受影響入選股月份 ≥ 30 的版本中，取挑選期配對差 t 最大者', window=SEL, n_versions=len(VERS),
               versions={k: dict(CAGR=v['seg']['CAGR'], maxDD=v['seg']['maxDD'], t_0050=v['seg']['t_0050'],
                                 paired=v.get('paired'), affected=v.get('affected'), n_skipped=v.get('n_skipped')) for k, v in out.items()},
               chosen=pick, doc=__doc__)
    A.dump(res, 'iterations.json')
    print(json.dumps(res['versions'], ensure_ascii=False, indent=1, default=A.conv)); print('chosen', pick)
