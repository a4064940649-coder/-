#!/usr/bin/env python3
"""反方審查（成本與可成交）o_disposal_rejudge。只讀研究員檔案，不寫入任何檔案（結果印到 stdout）。
1) 預收款券的現金限制：滿倉散戶在換股日 e 開盤賣舊股的價款 T+2 才入帳，處置股（需預收款項）無法在 e 開盤買進。
   對「e 當天處置中、且非續抱」的新買入，改為 e+2 開盤買（期間停 0050），重算 E1／E5／E0。
   E1 也受影響：公告日 ≤ d、start ∈ (d, e] 的股票 E1 照買（live SOP 也照買，備註「需預收款券」）。
2) 換手：E1 把續抱中的處置股強制賣出（賣出也需預收券、分盤撮合）→ 計數。
3) 容量：受影響名額在 e 日的成交值，與 30 萬／1,000 萬帳戶一檔部位的占比。
4) 2021-06 那個月 E0 多買的處置股，e 日開盤相對前收、是否開盤漲停（開盤競價在分盤撮合下是否可得）。
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import drj_core as R
K = R.K
IDX, COLS = R.IDX, R.COLS
NORMAL = K.rates(0.001)
D = R.load_list()
DS = R.Disp(D)
VAL = K.V.to_numpy() if hasattr(K, 'V') else None
out = {}

def in_disp_e(j, e, d):
    """e 日處置中、且公告日 ≤ d（e 開盤前已知）。"""
    return DS.in_disp_day(j, e, known_by=IDX.get_loc(d)) is not None

def build(ver):
    S = R.make_strategy(ver, DS)
    t0, t1, sd = S.bounds()
    pl = S.plan(10, 'official')
    return S, t0, t1, sd, pl

RUN = {}
for ver in ('E1', 'E5', 'E0'):
    RUN[ver] = build(ver)

PER = {'SEL': (R.SEL_A, R.SEL_B), 'POST': (R.VAL_A, R.HOLD_B), 'ALL': (R.SEL_A, R.HOLD_B)}
def per_of(d):
    return 'SEL' if d <= R.SEL_B else 'POST'

# ---------- 1/2/3：計數 ----------
cnt = {}
rows = []
for ver, (S, t0, t1, sd, pl) in RUN.items():
    held = set(); c = {'SEL': dict(new_buy_in_disp=0, new_buy_announced_on_d=0, forced_sell_in_disp_at_d=0, sell_while_in_disp_at_e=0, slots=0),
                       'POST': dict(new_buy_in_disp=0, new_buy_announced_on_d=0, forced_sell_in_disp_at_d=0, sell_while_in_disp_at_e=0, slots=0)}
    for (e, pk, _, d) in pl:
        p = per_of(d); c[p]['slots'] += len(pk)
        info = DS.at(d)
        pks = set(int(j) for j in pk)
        for j in pks - held:
            if in_disp_e(j, e, d):
                c[p]['new_buy_in_disp'] += 1
                g = DS.by_code[COLS[j]]
                x = g[(g.i0 <= e) & (g.i1 >= e) & (g.announce <= d)]
                if (x.announce == d).any(): c[p]['new_buy_announced_on_d'] += 1
                v = VAL[e, j] if VAL is not None else np.nan
                rows.append(dict(ver=ver, sig=str(d.date()), code=COLS[j], disp_at_d=j in info, start=str(x.start.min().date()),
                                 e_value_mn=float(v) / 1e6 if np.isfinite(v) else None))
        for j in held - pks:
            if j in info: c[p]['forced_sell_in_disp_at_d'] += 1
            if in_disp_e(j, e, d): c[p]['sell_while_in_disp_at_e'] += 1
        held = pks
    cnt[ver] = c
out['counts_n10'] = cnt
B = pd.DataFrame(rows)
out['new_buys_in_disp_detail'] = {v: g.drop(columns='ver').to_dict('records') for v, g in B.groupby('ver')} if len(B) else {}
if len(B) and B.e_value_mn.notna().any():
    ev = B.dropna(subset=['e_value_mn'])
    out['capacity'] = dict(median_e_value_mn=float(ev.e_value_mn.median()), min_e_value_mn=float(ev.e_value_mn.min()),
                           share_1000wan_slot_median=float((1.0 / ev.e_value_mn).median()), share_1000wan_slot_max=float((1.0 / ev.e_value_mn).max()))

# ---------- 1：預收款券延後 T+2 ----------
def defer_t2(pl):
    dm = {}; held = set()
    for (e, pk, _, d) in pl:
        m = {}
        for j in pk:
            j = int(j)
            if j not in held and in_disp_e(j, e, d): m[j] = e + 2
        if m: dm[e] = m
        held = set(int(j) for j in pk)
    return dm

res = {}
NAV = {}
for ver, (S, t0, t1, sd, pl) in RUN.items():
    b, s = NORMAL
    nav0, _ = R.simulate2(pl, b, s, t0, t1)
    navd, lg = R.simulate2(pl, b, s, t0, t1, defer=defer_t2(pl), DS=None)
    navs, _ = R.simulate2(pl, b, s, t0, t1, extra=0.005, DS=DS)   # 研究員的壓力成本（不延後）
    NAV[ver] = dict(base=nav0, t2=navd, stress=navs, sd=sd)
    e50 = R.E50.reindex(nav0.index)
    o = {}
    for k, (a, bb) in PER.items():
        last = min(pd.Timestamp(bb), nav0.index[-1])
        o[k] = {lab: dict(CAGR=round(R.cagr_mdd(n, a, last)[0], 3), t50=round(R.tvs(n, e50, sd, a, bb)['t'], 3))
                for lab, n in (('base', nav0), ('t2', navd), ('stress', navs))}
    o['defer_entered'] = lg['defer_entered']; o['defer_never'] = lg['defer_never']
    res[ver] = o
out['prepay_T2'] = res
pair = {}
for v in ('E5', 'E0'):
    pair[v] = {}
    for lab in ('base', 't2', 'stress'):
        pair[v][lab] = {k: {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in R.paired(NAV[v][lab], NAV['E1'][lab], NAV['E1']['sd'], a, b).items()}
                        for k, (a, b) in PER.items()}
out['paired_vs_E1'] = pair

# ---------- 4：2021-06 ----------
chk = []
for code in ('2465', '2615', '2636', '3530', '1476', '2542', '3014', '3305'):
    if code not in K.CI: continue
    j = K.CI[code]
    for (e, pk, _, d) in RUN['E0'][4]:
        if str(d.date()).startswith('2021-06'):
            ref = K.PREVC[e, j]
            gu = bool(K.gap_up_open(e)[j])
            chk.append(dict(code=code, sig=str(d.date()), e=str(IDX[e].date()), open=float(K.On[e, j]), prev_close=float(ref),
                            open_gap_pct=round((K.On[e, j] / ref - 1) * 100, 2), gap_up_open=gu, in_disp_e=in_disp_e(j, e, d),
                            e_value_mn=float(VAL[e, j]) / 1e6 if VAL is not None else None))
out['jun2021'] = chk
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
