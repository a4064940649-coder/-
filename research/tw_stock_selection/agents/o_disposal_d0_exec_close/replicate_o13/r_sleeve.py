# Satellite sleeve simulation (10 slots, idle 0050, strict prepay, cash pool k slots)
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from r_exec import *
C50B, C50S = 0.000855, 0.000855 + 0.001

def sim(ev, ent, ex, tier=1e6, mode='FIX', prepay='strict', k=2, slots=10, start='2006-04-01', **kw):
    days = [i for i in np.where(GOOD)[0] if IDX[i] >= pd.Timestamp(start)]
    pos_of = {}
    for n, i in enumerate(days): pos_of[i] = n
    # events by entry day
    evs = ev[ev.adopt & ev.ann_ok & (ev.s >= pd.Timestamp(start))]
    by_day = {}
    for r in evs.itertuples():
        ei = next_good(r.si) if ent == 'O2' else r.si
        if ei >= NT: continue
        by_day.setdefault(ei, []).append(r)
    eq0 = tier * cpi_ratio(IDX[days[0]].year)
    slot = eq0 / slots
    cash = k * slot; e50v = (eq0 - cash) * (1 - C50B); recv = []  # (settle_idx, amt)
    positions = []  # dict(code, tranches(list [sh, lvl, P]), sells(list), scale)
    nav = []; stats = dict(entered=0, cash_skip=0, full_skip=0, held_skip=0, nofill=0, partial=0)
    cash_hist = []; disp_hist = []
    eq_prev = eq0; unit = 1.0; month_prev = IDX[days[0]].month
    reserved_next = 0.0
    for n, t in enumerate(days):
        d = IDX[t]
        e50v *= E50[t] / E50[days[n - 1]] if n > 0 else 1.0
        # settle receivables
        keep = []
        for (si_, a) in recv:
            if si_ <= n: cash += a
            else: keep.append((si_, a))
        recv = keep
        # FIX monthly scaling (before trading on first day of month)
        sc = 1.0
        if n > 0 and d.month != month_prev:
            if mode == 'FIX':
                target = tier * cpi_ratio(d.year)
                # equity at prior close
                sc = target / eq_prev
                cash *= sc; e50v *= sc; recv = [(a, b * sc) for a, b in recv]
                for p in positions: p['scale'] *= sc
        month_prev = d.month
        slot_now = (tier * cpi_ratio(d.year) / slots) if mode == 'FIX' else (eq_prev * sc / slots)
        # entries
        for r in sorted(by_day.get(t, []), key=lambda r: (r.announce, r.fo)):
            if r.bear or not r.r4: continue
            if any(p['code'] == r.code for p in positions): stats['held_skip'] += 1; continue
            if len(positions) >= slots: stats['full_skip'] += 1; continue
            if prepay == 'strict':
                if cash >= slot_now: B = slot_now
                elif cash >= 0.5 * slot_now: B = cash; stats['partial'] += 1
                else: stats['cash_skip'] += 1; continue
            else:
                B = slot_now
            o = run_trade(r, ent, ex, None, budget=B, ret_detail=True, **kw)
            if o is None: stats['nofill'] += 1; continue
            if prepay != 'strict' and o['cost'] > cash:
                need = o['cost'] - cash; e50v -= need / (1 - C50S); cash += need
            cash -= o['cost']; stats['entered'] += 1
            positions.append(dict(code=r.code, ci=r.ci, tr=[[sh, lvl, P] for (sh, lvl, P, _) in o['tranches']],
                                  sells=sorted(o['sells']), scale=1.0, lotflag=[x[3] for x in o['tranches']]))
        # exits today
        newpos = []
        for p in positions:
            rem = []
            for (kk, amt, lsold, osold) in p['sells']:
                if kk == t:
                    recv.append((n + 2, amt * p['scale']))
                    for q, tr in enumerate(p['tr']):
                        if p['lotflag'][q] == 1: tr[0] -= lsold
                        else: tr[0] -= osold
                elif kk > t: rem.append((kk, amt, lsold, osold))
            p['sells'] = rem
            if sum(tr[0] for tr in p['tr']) > 0 and (p['sells'] or True):
                if not p['sells'] and all(tr[0] <= 0 for tr in p['tr']): continue
                newpos.append(p)
        positions = [p for p in newpos if sum(tr[0] for tr in p['tr']) > 0]
        # position value
        pv = 0.0
        for p in positions:
            for sh, lvl, P in p['tr']:
                pv += p['scale'] * sh * P * np.exp(LA[t, p['ci']] - lvl)
        # cash management at close: reserve for events with entry next day
        nt_ = days[n + 1] if n + 1 < len(days) else None
        res = 0.0
        if nt_ is not None:
            nfree = slots - len(positions)
            for r in by_day.get(nt_, []):
                if r.bear or not r.r4 or any(p['code'] == r.code for p in positions): continue
                if nfree <= 0: break
                res += slot_now; nfree -= 1
        rsum = sum(a for _, a in recv)
        target = k * slot_now + res
        if prepay == 'strict':
            if cash + rsum < target and e50v > 0:
                x = min(target - cash - rsum, e50v * (1 - C50S))
                e50v -= x / (1 - C50S); recv.append((n + 2, x))
            elif cash > target + slot_now:
                y = cash - target; cash -= y; e50v += y * (1 - C50B)
        else:
            if cash > k * slot_now + slot_now:
                y = cash - k * slot_now; cash -= y; e50v += y * (1 - C50B)
        eq = e50v + cash + sum(a for _, a in recv) + pv
        unit *= eq / (eq_prev * sc)
        eq_prev = eq
        nav.append(unit); cash_hist.append((cash + sum(a for _, a in recv)) / eq); disp_hist.append(pv / eq)
    nav = pd.Series(nav, index=IDX[days])
    stats['avg_cash'] = float(np.mean(cash_hist)); stats['avg_disp'] = float(np.mean(disp_hist))
    stats['skip_ratio'] = stats['cash_skip'] / max(1, stats['entered'] + stats['cash_skip'])
    return nav, stats

E50S = pd.Series(E50, index=IDX)
def month_excess(nav):
    m = nav.resample('ME').last(); b = E50S[nav.index].resample('ME').last()
    b0 = E50S[E50S.index < nav.index[0]].iloc[-1]
    rm = m.pct_change(); rm.iloc[0] = m.iloc[0] / 1.0 - 1
    rb = b.pct_change(); rb.iloc[0] = b.iloc[0] / b0 - 1
    return rm - rb, rm, rb
def tstat(x): x = x.dropna(); return x.mean() / x.std(ddof=1) * np.sqrt(len(x))
def per(x, a, b): return x[(x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(b) + pd.offsets.MonthEnd(0))]

if __name__ == '__main__':
    import pickle
    from r_data import load_events
    ev = load_events()
    tag = sys.argv[1] if len(sys.argv) > 1 else 'main'
    cfg = dict(main={}, k1=dict(k=1), k3=dict(k=3), stress=dict(stress=True), g1=dict(mode='GROW'), f3=dict(tier=3e6), f10=dict(tier=1e7), f03=dict(tier=3e5), none=dict(prepay='none'))[tag]
    out = {}
    for a in ['O1', 'C1', 'O2']:
        for b in ['CL1', 'CL', 'OA']:
            nav, st = sim(ev, a, b, **cfg)
            ex, rm, rb = month_excess(nav)
            out[(a, b)] = (nav, st, ex)
            print(a, b, 'VAL t %.2f FULL t %.2f SEL t %.2f' % (tstat(per(ex, '2024-01-01', '2026-10-22')), tstat(ex), tstat(per(ex, '2021-10-01', '2023-12-31'))), {k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()}, flush=True)
    pickle.dump(out, open(f'sleeve_{tag}.pkl', 'wb'))
    for nm, (a, b) in dict(VAL=('2024-01-01', '2026-10-22'), FULL=('2006-04-01', '2026-10-22'), SEL=('2021-10-01', '2023-12-31'), Y19=('2019-01-01', '2026-10-22')).items():
        ts = [tstat(per(v[2], a, b)) for v in out.values()]; mu = [per(v[2], a, b).mean() * 100 for v in out.values()]
        print(nm, 'median t %.2f p10 t %.2f median mean %.3fpp p10 mean %.3fpp' % (np.median(ts), np.percentile(ts, 10), np.median(mu), np.percentile(mu, 10)))
    print('skip ratio median', np.median([v[1]['skip_ratio'] for v in out.values()]), 'avg cash', np.median([v[1]['avg_cash'] for v in out.values()]), 'avg disp', np.median([v[1]['avg_disp'] for v in out.values()]), 'full skip', np.median([v[1]['full_skip'] for v in out.values()]))
