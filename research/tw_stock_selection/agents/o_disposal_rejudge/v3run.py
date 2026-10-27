#!/usr/bin/env python3
"""在第 1 題（o_infra_v3_rebuild）的 V3 資料上重跑：P＝下市股價格修補、X＝永久停牌照對價出清（只讀 import o3_core，資料 sha256 由 o3_core 斷言）。
版本 E1／E0／選中版本（一般成本；E4 需要延後進場引擎，V3 上不跑）。營收：ORIG（v2 正式）與 R_DEL（修補版）各一。
另：本題官方名單 vs 第 1 題近似名單的一致率。PREREG 必須已存在。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import drj_core as R
res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert 'PREREG_sha256' in res, 'PREREG 不存在'
SELV = res['PREREG']['selected_version']
sys.path.insert(0, str(HERE.parent / 'o_infra_v3_rebuild'))
import o3_core as O3            # noqa: E402（只讀 import）
sys.path.pop(0)
K, V = R.K, R.V
assert O3.K is K
npatch = O3.apply_patch(require_freeze=True)
E = O3.load_events(require_freeze=True)
XR = O3.exit_rules(E, 'main')
D = R.load_list(); DS = R.Disp(D)
NORMAL = K.rates(0.001)
PER = {'SEL': (R.SEL_A, R.SEL_B), 'POST': (R.VAL_A, R.HOLD_B), 'VAL': (R.VAL_A, R.VAL_B), 'HOLD': (R.HOLD_A, R.HOLD_B), 'ALL': (R.SEL_A, R.HOLD_B)}
YL0, Y50 = R.revenue()
YRD = O3.load_rdel()
out = {'n_patched_cells': npatch, 'n_exit_rules': len(XR), 'runs': {}, 'paired': {}}
for revname, YL in (('ORIG', YL0), ('R_DEL', YRD)):
    navs = {}
    for ver in ('E1', 'E0', SELV):
        S = V.V2('gapfix', YL, Y50, exfin=True)
        base = R.make_strategy(ver, DS)._mask
        orig = S.cross_section
        def cs(d, _o=orig, _m=base):
            old = K.disposal_mask; K.disposal_mask = _m
            try: return _o(d)
            finally: K.disposal_mask = old
        S.cross_section = cs
        t0, t1, sd = S.bounds()
        pl = S.plan(10, 'official')
        b, s = NORMAL
        nav, lg = O3.simulate([(p[0], p[1]) for p in pl], b, s, t0, t1, xrules=XR)
        ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)
        navs[ver] = nav
        e50 = R.E50.reindex(nav.index)
        st = {}
        for k, (a, bb) in PER.items():
            last = min(bb, nav.index[-1])
            cg, mdd = R.cagr_mdd(nav, a, last)
            st[k] = dict(CAGR=cg, maxDD=mdd, vs0050=R.tvs(nav, e50, sd, a, bb), vsEW=R.tvs(nav, ew, sd, a, bb), E50_CAGR=R.cagr_mdd(e50, a, last)[0])
        out['runs'][f'{revname}|{ver}'] = dict(stats=st, n_exits=len(lg['exits']), n_frozen=len(lg['frozen']))
        if ver != 'E1':
            out['paired'][f'{revname}|{ver}'] = {k: R.paired(nav, navs['E1'], sd, a, bb) for k, (a, bb) in PER.items()}
        print(revname, ver, {k: (round(v['CAGR'], 2), round(v['vs0050']['t'], 2), round(v['maxDD'], 1)) for k, v in st.items()},
              {k: (round(v['mean_pp'], 3), round(v['t'], 2)) for k, v in out['paired'].get(f'{revname}|{ver}', {}).items()}, flush=True)
# 與第 1 題近似名單的一致率（2006-04～2021-08，面板內 regular 事件，±5 日曆日）
H = O3.load_disposal_hist(require_freeze=True)
A = D[(D.src == 'A')]
def match(a, b):
    bs = {}
    for c, d in zip(b.code, b.start): bs.setdefault(c, []).append(d)
    return np.array([any(abs((d - x).days) <= 5 for x in bs.get(c, [])) for c, d in zip(a.code, a.start)])
agr = {}
for lab, a, b in (('2006-04_2021-08', '2006-04-01', '2021-08-31'), ('2005_2021-09-23', '2005-01-01', '2021-09-23')):
    a_ = A[(A.start >= a) & (A.start <= b)]; h_ = H[(H.start >= a) & (H.start <= b)]
    agr[lab] = dict(n_official=len(a_), n_infra_proxy=len(h_), official_found_in_infra=float(match(a_, h_).mean()), infra_found_in_official=float(match(h_, a_).mean()))
out['agree_with_infra_v3_list'] = agr
print(agr)
(HERE / 'v3_results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
