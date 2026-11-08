#!/usr/bin/env python3
"""獨立複製之二：紙上（ia_core 資料）consideration 結算，自寫引擎（部位用 dict，不 import 研究員的 cz_engine）。
資料只讀 import independent_audit/ia_core.py（研究中心共用資料層）；名單只讀 import o_finmask_delisted_fin_erratum 的 fclass／flib（與 integrate_o16 同）。
事件表＝rp_events.csv（自建）；結算規則照 SPEC 文字（results.json PREREG.settlement_rules）自行實作。
輸出 rp_ia.json：C0／U_opt／U_opt_F2 × 一般 × freeze／cons_main／cons_lo／cons_hi／cons_c50 的判定欄與逐日淨值摘要。
python3 -I -B rp_ia.py"""
import sys, json, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent.parent
sys.path.insert(0, str(AG / 'independent_audit')); import ia_core as K; sys.path.pop(0)
FM = AG / 'o_finmask_delisted_fin_erratum'
SPEC = json.loads((FM / 'ERRATUM_SPEC_FM.json').read_text(encoding='utf-8'))
sys.path.insert(0, str(FM)); import fclass as FCL, flib as FL; sys.path.pop(0)
D_BOUND = 0.057
T = pd.read_csv(HERE / 'rp_events.csv', dtype={'code': str, 'successor': str})
TAG = sys.argv[1] if len(sys.argv) > 1 else 'asis'
if TAG == 'aligned':
    for c, t in json.loads((HERE / 'adjudication.json').read_text(encoding='utf-8'))['align_to_researcher'].items(): T.loc[T.code == c, 'type'] = t
IDX = K.IDX; DT = IDX.values


def tpos(d): return int(np.searchsorted(DT, np.datetime64(pd.Timestamp(d))))


def make_rules(bound, c50=False):
    """{j: rule}：rule＝dict(trig, parts=[(kind, per_share, release_idx, succ, ratio)]) ；kind∈cash／stock／sellfix／zero。"""
    dl = {'main': 0.0, 'lo': -D_BOUND, 'hi': D_BOUND}[bound]
    out = {}
    for r in T.itertuples():
        j = K.CI.get(r.code)
        if j is None: continue
        ltU = pd.Timestamp(r.last_trade); term = pd.Timestamp(r.terminate_date) if isinstance(r.terminate_date, str) and r.terminate_date else ltU + pd.Timedelta(days=14)
        tr = np.flatnonzero(K.TRADED[:, j]); tr = tr[IDX[tr] <= max(term, ltU)]
        lt = int(tr[-1]); lc = float(K.Cn[lt, j]); ti = tpos(term); pay = min(ti + 20, len(IDX) - 1)
        parts = []
        if IDX[lt] < ltU - pd.Timedelta(days=7):
            parts.append(('sellfix', lc, lt + 1, None, None))
        else:
            sj = K.CI.get(r.successor) if isinstance(r.successor, str) and r.successor else None
            s = None
            if sj is not None:
                w = np.flatnonzero(K.TRADED[:, sj] & np.isfinite(K.On[:, sj])); w = w[w >= ti]
                s = int(w[0]) if len(w) else None
            t = r.type
            if t == 'cash': parts.append(('cash', r.cash, pay, None, None))
            elif t in ('swap', 'conversion', 'mixed') and s is not None:
                parts.append(('stock', lc, s, sj, r.ratio))
                if t == 'mixed': parts.append(('cash', r.cash, pay, None, None))
                if t == 'mixed' and np.isfinite(r.pref_ratio): parts.append(('sellfix', r.pref_ratio * r.pref_value, s, None, None))
            elif t == 'swap_ur' and s is not None and np.isfinite(K.Cn[lt, sj]):
                parts.append(('stock', lc, s, sj, lc / K.Cn[lt, sj] * (1 + dl)))
            elif t in ('swap_ur', 'acq_ur'):
                parts.append(('cash', lc * (1 + dl), pay, None, None))
            elif t == 'unknown':
                if bound == 'lo': parts.append(('zero', lc, lt + 1, None, None))
                else: parts.append(('cash', lc, pay, None, None))
        out[j] = dict(trig=lt + 1, lc=lc, parts=parts, c50=c50, code=r.code)
    return out


def run(plan, buy, sell, t0, t1, rules=None):
    reb = {e: list(p) for e, p in plan}
    pos = {}; cash = 1.0; cl = []
    trig = {}
    for j, ru in (rules or {}).items(): trig.setdefault(ru['trig'], []).append(j)
    e50 = K.E50.to_numpy()
    navs = []
    for t in range(t0, t1 + 1):
        if rules:
            for j in trig.get(t, []):
                if pos.get(j, 0) > 0 and t > t0:
                    ru = rules[j]; n = pos.pop(j) / ru['lc']
                    for kind, ps, rel, sj, ratio in ru['parts']:
                        cl.append(dict(kind=kind, v=n * ps, rel=max(rel, t), sj=sj, q=(n * ratio if kind == 'stock' else 0), c50=ru['c50']))
            if t > t0:
                g50 = e50[t] / e50[t - 1] if np.isfinite(e50[t]) and np.isfinite(e50[t - 1]) else 1.0
                for c in cl:
                    if c['kind'] == 'cash' and c['c50'] and t <= c['rel']: c['v'] *= g50
        isreb = t in reb
        openadd = {}
        rest = []
        for c in cl:
            if t < c['rel']: rest.append(c); continue
            if c['kind'] == 'cash': cash += c['v']
            elif c['kind'] == 'stock':
                if isreb: openadd[c['sj']] = openadd.get(c['sj'], 0.0) + c['q'] * K.On[t, c['sj']]
                else: pos[c['sj']] = pos.get(c['sj'], 0.0) + c['q'] * K.Cn[t, c['sj']] / K.GROW[t, c['sj']]
            elif isreb: cash += 0.0 if c['kind'] == 'zero' else c['v'] * (1 - sell)
            else: rest.append(c)
        cl = rest
        if isreb:
            ok = lambda j: K.TRADED[t, j] and np.isfinite(K.On[t, j])
            h = {}
            for j, v in pos.items():
                h[j] = v * (K.GROW[t, j] * K.On[t, j] / K.Cn[t, j] if ok(j) else 1.0)
            for j, v in openadd.items(): h[j] = h.get(j, 0.0) + v
            frozen = {j: v for j, v in h.items() if v > 0 and not ok(j)}
            live = {j: v for j, v in h.items() if j not in frozen}
            picks = reb[t]; k = len(picks); W = sum(live.values()) + cash
            def need(x):
                c = 0.0
                for j in set(live) | set(picks):
                    tg = x if j in picks else 0.0; cur = live.get(j, 0.0)
                    c += buy * max(tg - cur, 0.0) + sell * max(cur - tg, 0.0)
                return x * k + c
            lo, hi = 0.0, W / k
            for _ in range(60):
                mid = (lo + hi) / 2
                if need(mid) > W: hi = mid
                else: lo = mid
            pos = dict(frozen)
            for j in picks:
                pos[j] = pos.get(j, 0.0) + lo * (K.Cn[t, j] / K.On[t, j])
            cash = 0.0
        elif t > t0:
            pos = {j: v * K.GROW[t, j] for j, v in pos.items()}
        navs.append(sum(pos.values()) + cash + sum(c['v'] for c in cl))
    return pd.Series(navs, index=IDX[t0:t1 + 1])


def main():
    cls = SPEC['classifiers']; CL = FCL.Classifier(cls['keywords'], cls['f2_manual_exclusion'], cls['name_exclusion'])
    C0M = FL.C0MASK.copy(); F2V = np.array([CL.f2(c) for c in FL.COLS]); LO, HI = FL.E.LO, FL.E.HI
    MASK = {'C0': lambda d: C0M, 'F2': lambda d: F2V if LO <= pd.Timestamp(d) <= HI else C0M}
    VARS = {'C0': ('C0', 'base'), 'U_opt': ('C0', 'opt'), 'U_opt_F2': ('F2', 'opt')}
    buy, sell = K.rates(0.001)
    RU = {'cons_main': make_rules('main'), 'cons_lo': make_rules('lo'), 'cons_hi': make_rules('hi'), 'cons_c50': make_rules('main', True)}
    sys.path.insert(0, str(AG / 'o_forward_sequential_v2')); import fslib as F; sys.path.pop(0)
    E50 = F.e50(); END = pd.Timestamp('2026-10-12')
    out = {}; navd = {}
    for k, (mk, mode) in VARS.items():
        S = FL.VM(MASK[mk], mode, 'SURV'); pl = S.plan(10, 'official'); t0, t1, sd = S.bounds()
        plan = [(int(p[0]), np.array(p[1], int)) for p in pl]
        SD = pd.DatetimeIndex(sd); pts = list(SD[SD < END]) + [END]
        for eg in ['freeze'] + list(RU):
            nav = run(plan, buy, sell, t0, t1, RU.get(eg))
            navd[f'{k}|{eg}'] = nav
            r = np.array([F._at(nav, b) / F._at(nav, a) - 1 for a, b in zip(pts[:-1], pts[1:])])
            r5 = np.array([F._at(E50, b) / F._at(E50, a) - 1 for a, b in zip(pts[:-1], pts[1:])])
            yr = np.array([a.year for a in pts[:-1]]); per = pd.PeriodIndex([pd.Period(a, 'M') for a in pts[:-1]])
            def seg(a, b):
                m = np.ones(len(r), bool)
                if a: m &= per >= pd.Period(a, 'M')
                if b: m &= per <= pd.Period(b, 'M')
                x, y = r[m], r5[m]; ex = x - y
                an = lambda z: (np.prod(1 + z) ** (12 / len(z)) - 1) * 100
                return dict(ann=an(x), gex=an(x) - an(y), t=float(ex.mean() / ex.std(ddof=1) * math.sqrt(len(ex))))
            o = dict(ALL=seg(None, None), SEG1=seg('2006-01', '2012-12'), SEG2=seg('2013-01', '2018-12'), SEG3=seg('2019-01', None))
            ex = pd.Series(r - r5).groupby(yr).sum(); b3 = ex.sort_values(ascending=False).index[:3]; m = ~np.isin(yr, b3); z = (r - r5)[m]
            o['t_drop_best3'] = float(z.mean() / z.std(ddof=1) * math.sqrt(len(z)))
            out[f'{k}|{eg}'] = o
        print(k, {eg: round(out[f'{k}|{eg}']['ALL']['ann'], 4) for eg in ['freeze'] + list(RU)}, flush=True)
    pd.DataFrame(navd).to_parquet(Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/o17_delist/rp_navs_ia_' + TAG + '.parquet'))
    (HERE / f'rp_ia_{TAG}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
