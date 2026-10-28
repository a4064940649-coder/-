"""Independent replication engine (o5). Written from the frozen spec only."""
import numpy as np, pandas as pd, json, os, math
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data'); RAW = os.path.join(HERE, '..', 'raw')
T0, T1 = pd.Timestamp('2006-04-11'), pd.Timestamp('2026-10-12')
ODD_SWITCH = pd.Timestamp('2020-10-26'); OA_HALF_END = pd.Timestamp('2020-03-23')

_G = {}
def load():
    if _G: return _G
    O = pd.read_parquet(os.path.join(D, 'px_open.parquet'))
    codes = list(O.columns); idx = O.index
    rd = lambda f: pd.read_parquet(os.path.join(D, f)).reindex(index=idx, columns=codes)
    C = rd('px_close.parquet'); G = rd('px_grow.parquet'); TR = rd('px_traded.parquet'); PC = rd('prevclose.parquet')
    CF = C.ffill()
    tr = TR.values == 1
    adj = (G.values * PC.values / C.values)
    adj = np.where(np.isfinite(adj), adj, 1.0); adj = np.where(tr, adj, 1.0)
    Ov = O.values; PCv = PC.values
    ok = tr & np.isfinite(Ov)
    gap = np.where(ok, Ov / PCv - 1, np.nan); gap = np.where(np.isfinite(gap), gap, 0.0)
    f = pd.read_parquet(os.path.join(D, 'feat_daily.parquet'))
    def lag(col):
        w = f.pivot(index='date', columns='code', values=col).reindex(columns=codes)
        w = w.reindex(w.index.union(idx)).sort_index()
        # value as of the previous trading date in the panel index (t-1)
        w = w.reindex(idx)
        return w.shift(1).values
    spec = json.load(open(os.path.join(D, 'model_spec.json')))
    cpi = json.load(open(os.path.join(RAW, 'cpi_tw.json')))['index_2006_100']
    xr = json.load(open(os.path.join(D, 'xrules.json')))
    xrules = {}
    for c, (fac, rel, kind, vu) in xr.items():
        xrules[codes.index(c)] = (fac, pd.Timestamp(rel), pd.Timestamp(vu) if vu else pd.Timestamp('2100-01-01'))
    i0 = idx.get_loc(T0); i1 = idx.get_loc(T1)
    _G.update(dict(idx=idx, codes=codes, cidx={c: i for i, c in enumerate(codes)}, O=Ov, C=C.values, CF=CF.values, PC=PCv,
                   ADJ=adj, OK=ok, TR=tr, GAP=gap, V60=lag('V60'), SIG=lag('sig60'), RK=lag('vrank'),
                   spec=spec, cpi={int(k): v for k, v in cpi.items()}, xrules=xrules, i0=i0, i1=i1,
                   years=np.array([d.year for d in idx])))
    return _G

def plans(seed):
    fn = 'plan_v4.csv' if seed == -1 else 'plans_ab.csv'
    p = pd.read_csv(os.path.join(D, fn), dtype={'picks': str})
    p = p[p.seed == seed]
    g = load()
    return {int(r.exec_idx): [g['cidx'][c] for c in r.picks.split(',')] for r in p.itertuples()}

FEES = {  # rate, min_lot, min_odd
    'A': (0.001425 * 0.28, 20.0, 20.0), 'B': (0.001425 * 0.6, 20.0, 20.0), 'C': (0.001425 * 0.28, 20.0, 1.0),
    'D1': (0.001425 * 0.6, 0.0, 0.0)}

DEF = dict(fee='A', odd_time=True, odd_spread=True, impact=True, cap=True, kappa=None, oa_half=False,
           odd_stress=False, keep=False)

def simulate(seed, tier, mode, **kw):
    P = dict(DEF); P.update(kw)
    g = load(); spec = g['spec']
    O, C, CF, PC, ADJ, OK, GAP, V60, SIG, RK = (g[k] for k in ('O', 'C', 'CF', 'PC', 'ADJ', 'OK', 'GAP', 'V60', 'SIG', 'RK'))
    idx = g['idx']; i0, i1 = g['i0'], g['i1']; N = O.shape[1]
    cpi = g['cpi']; xrules = g['xrules']
    rate, min_lot, min_odd = FEES[P['fee']]
    coef = np.array(spec['oa']['coef']); kappa = P['kappa'] if P['kappa'] is not None else spec['oa']['kappa']
    d_edges = np.array(spec['drift']['edges']); d_vals = np.array(spec['drift']['vals'])
    pe = np.array(spec['odd']['price_edges']); re_ = np.array(spec['odd']['rank_edges'])
    obuy = np.array(spec['odd']['buy']); osell = np.array(spec['odd']['sell'])
    if P['odd_stress']:
        obuy = obuy * 2; osell = np.where(osell <= 0, 0.0, osell)
    plan = plans(seed)
    S = np.zeros(N); cash = tier * cpi[2006] / cpi[2026]
    E_init = cash; E_prev = cash; unit = 1.0
    nT = i1 - i0 + 1
    nav = np.empty(nT); nav[0] = 1.0
    comp = np.zeros((nT, 6))  # tax, fee, impact_excess, base_slip, odd_spread, (unused)
    tvf = np.zeros(nT); cashf = np.zeros(nT)
    st = dict(new=0, new_oddonly=0, new_pre=0, new_oddonly_pre=0, lot_orders=0, ex10=0, ex20=0,
              lot_pre19=0, ex10_pre19=0, lot_post19=0, ex10_post19=0, unfilled=0.0, imp_amt=[])
    pend = {}
    ctx = {}

    def fee(val, odd):
        return max(min_odd if odd else min_lot, rate * val)

    def oa_of(j, t, pick, drop):
        v = V60[t, j]
        if not np.isfinite(v): return np.nan
        rk = RK[t, j]; s = SIG[t, j]
        if not (np.isfinite(rk) and np.isfinite(s)): return np.nan
        x = np.array([1.0, math.log(max(rk, 1.0)), math.log(PC[t, j]), math.log(max(s, 1e-4)), abs(GAP[t, j]), pick, drop])
        oa = v * kappa * math.exp(coef @ x)
        if P['oa_half'] and idx[t] < OA_HALF_END: oa *= 0.5
        return oa

    def odd_base(j, t):
        if not P['odd_time']: return O[t, j]
        if idx[t] < ODD_SWITCH: return C[t, j]
        b = d_vals[np.searchsorted(d_edges, GAP[t, j], 'right')]
        return O[t, j] * (1 + b)

    def odd_c(j, t, side):
        if not P['odd_spread']: return 0.0
        pb = np.searchsorted(pe, PC[t, j], 'right'); rk = RK[t, j]
        rb = np.searchsorted(re_, rk if np.isfinite(rk) else 1e9, 'right')
        return obuy[pb, rb] if side > 0 else osell[pb, rb]

    def lot_cost(j, t, side, L, oa, comps=None):
        if L <= 0: return 0.0
        n = 1000.0 * L * O[t, j]
        imp = 0.001
        if P['impact']:
            s = SIG[t, j]
            if np.isfinite(oa) and oa > 0 and np.isfinite(s):
                imp += 0.5 * s * math.sqrt(1000.0 * L * PC[t, j] / oa)
        val = n * (1 + side * imp); f = fee(val, False); tax = 0.003 * val if side < 0 else 0.0
        if comps is not None:
            comps[0] += tax; comps[1] += f; comps[2] += (imp - 0.001) * n; comps[3] += 0.001 * n; comps[5] += n
        return side * val + f + tax

    def odd_cost(j, t, side, q, comps=None):
        if q <= 0: return 0.0
        b = odd_base(j, t); c = odd_c(j, t, side)
        val = q * b * (1 + side * c); f = fee(val, True); tax = 0.003 * val if side < 0 else 0.0
        if comps is not None:
            comps[0] += tax; comps[1] += f; comps[4] += q * b * c; comps[5] += q * b
        return side * val + f + tax

    def cap_lots(j, t, oa):
        if not P['cap'] or not (np.isfinite(oa) and oa > 0): return 10 ** 12
        return max(1, int(math.floor(0.20 * oa / (1000.0 * PC[t, j]))))

    for k in range(1, nT):
        t = i0 + k
        S *= ADJ[t]
        comps = np.zeros(6)
        E_pre = None
        if t in plan:
            # a) clear pending
            for j, q in pend.items():
                pc = PC[t, j] if np.isfinite(PC[t, j]) else CF[t - 1, j]
                st['unfilled'] += abs(q) * pc
            pend = {}
            held = S > 0
            frozen = held & ~OK[t]
            for j in np.nonzero(frozen)[0]:
                if j in xrules:
                    fac, rel, vu = xrules[j]
                    if rel <= idx[t] < vu:
                        cash += S[j] * CF[t - 1, j] * fac; S[j] = 0.0
            frozen = (S > 0) & ~OK[t]
            trad = (S > 0) & OK[t]
            pcf = np.where(np.isfinite(PC[t]), PC[t], CF[t - 1])
            E_pre = np.nansum(S[trad] * O[t, trad]) + np.nansum(S[frozen] * pcf[frozen]) + cash
            if mode == 'FIX':
                tgt_eq = tier * cpi[idx[t].year] / cpi[2026]
                sc = tgt_eq / E_pre
                unit *= E_pre / E_prev
                S *= sc; cash *= sc; E_scaled = tgt_eq
            else:
                E_scaled = E_pre
            # d) fractional shares
            tj = np.nonzero(trad)[0]
            fl = np.floor(S[tj] + 1e-9); cash += np.sum((S[tj] - fl) * O[t, tj]); S[tj] = fl
            # e) rebalance
            picks = plan[t]; k_ = len(picks); pset = set(picks)
            cur_set = [j for j in tj if S[j] > 0]
            hi = (np.sum(S[cur_set] * O[t, cur_set]) if cur_set else 0.0) + cash
            hi /= k_
            # order universe
            J = list(picks) + [j for j in cur_set if j not in pset]
            m = len(J); Ja = np.array(J)
            cur = np.round(S[Ja])
            isp = np.array([j in pset for j in J])
            oa = np.array([oa_of(j, t, 1.0 if isp[i] else 0.0, 0.0 if isp[i] else 1.0) for i, j in enumerate(J)])
            p_ = PC[t, Ja]; o_ = O[t, Ja]; s_ = SIG[t, Ja]
            ob = np.array([odd_base(j, t) for j in J])
            cb = np.array([odd_c(j, t, 1) for j in J]); cs = np.array([odd_c(j, t, -1) for j in J])
            impok = P['impact'] & np.isfinite(oa) & (oa > 0) & np.isfinite(s_)
            oa_s = np.where(impok, oa, 1.0); s_s = np.where(impok, s_, 0.0)
            keepmask = isp & (cur > 0) if P['keep'] else np.zeros(m, bool)

            def targets(T):
                L = np.floor(T / (1000 * p_) + 1e-12)
                q = np.floor((T - 1000 * L * p_) / p_ + 1e-9)
                tg = 1000 * L + np.maximum(q, 0)
                tg = np.where(isp, tg, 0.0)
                return np.where(keepmask, cur, tg)

            def need(T):
                dl = targets(T) - cur
                side = np.sign(dl); a = np.abs(dl)
                L = a // 1000; q = a % 1000
                n = 1000 * L * o_
                imp = 0.001 + 0.5 * s_s * np.sqrt(1000 * L * p_ / oa_s)
                v1 = n * (1 + side * imp)
                f1 = np.where(L > 0, np.maximum(min_lot, rate * v1), 0.0)
                c = np.where(side > 0, cb, cs)
                v2 = q * ob * (1 + side * c)
                f2 = np.where(q > 0, np.maximum(min_odd, rate * v2), 0.0)
                tax = np.where(side < 0, 0.003 * (v1 + v2), 0.0)
                return np.sum(side * (v1 + v2) + f1 + f2 + tax)

            if need(hi) <= cash:
                T = hi
            else:
                lo, h = 0.0, hi
                for _ in range(40):
                    mid = 0.5 * (lo + h)
                    if need(mid) <= cash: lo = mid
                    else: h = mid
                T = lo
            tg = targets(T); dl = tg - cur
            # exec stats
            for i, j in enumerate(J):
                if isp[i] and cur[i] == 0:
                    pre = idx[t] < ODD_SWITCH
                    st['new'] += 1; st['new_pre'] += pre
                    if tg[i] < 1000:
                        st['new_oddonly'] += 1; st['new_oddonly_pre'] += pre
                Ld = abs(dl[i]) // 1000
                if Ld > 0 and dl[i] != 0:
                    st['lot_orders'] += 1
                    r = 1000 * Ld * p_[i] / oa[i] if np.isfinite(oa[i]) and oa[i] > 0 else 0.0
                    st['ex10'] += r > 0.1; st['ex20'] += r > 0.2
                    if idx[t].year <= 2018: st['lot_pre19'] += 1; st['ex10_pre19'] += r > 0.1
                    else: st['lot_post19'] += 1; st['ex10_post19'] += r > 0.1
            order = np.argsort(dl, kind='stable')
            for i in order:
                d = dl[i]
                if d == 0: continue
                j = J[i]; side = 1 if d > 0 else -1; a = abs(d)
                L = int(a // 1000); q = int(a % 1000)
                if side > 0:
                    full = odd_cost(j, t, 1, q) + lot_cost(j, t, 1, L, oa[i])
                    if full > cash + 1e-9:
                        pend[j] = d; continue
                cash -= odd_cost(j, t, side, q, comps)
                S[j] += side * q
                Lx = min(L, cap_lots(j, t, oa[i]))
                c0 = comps[2]
                cash -= lot_cost(j, t, side, Lx, oa[i], comps)
                st['imp_amt'].append(comps[2] - c0)
                S[j] += side * 1000 * Lx
                if Lx < L: pend[j] = side * 1000 * (L - Lx)
        elif pend:
            trad = (S > 0) & OK[t]; frozen = (S > 0) & ~OK[t]
            pcf = np.where(np.isfinite(PC[t]), PC[t], CF[t - 1])
            E_pre = np.nansum(S[trad] * O[t, trad]) + np.nansum(S[frozen] * pcf[frozen]) + cash
            items = [(j, q) for j, q in pend.items() if OK[t, j]]
            items.sort(key=lambda x: (0 if x[1] < 0 else 1, abs(x[1])))
            for j, q in items:
                oa = oa_of(j, t, 0.0, 0.0)
                if q < 0:
                    qty = min(-q, int(round(S[j])))
                    if qty <= 0: continue
                    L = qty // 1000; qq = qty % 1000
                    cash -= odd_cost(j, t, -1, qq, comps); S[j] -= qq
                    Lx = min(L, cap_lots(j, t, oa))
                    cash -= lot_cost(j, t, -1, Lx, oa, comps); S[j] -= 1000 * Lx
                    pend[j] = q + qq + 1000 * Lx
                else:
                    Lp = int(q // 1000); qp = int(q % 1000)
                    lm = int(math.floor(cash / (O[t, j] * 1000 * 1.05)))
                    nl = min(Lp, lm); done = 0
                    if qp > 0:
                        cst = odd_cost(j, t, 1, qp)
                        if not (cst > cash - nl * O[t, j] * 1050):
                            cash -= odd_cost(j, t, 1, qp, comps); S[j] += qp; done += qp
                    Lx = min(nl, cap_lots(j, t, oa))
                    if Lx > 0:
                        cash -= lot_cost(j, t, 1, Lx, oa, comps); S[j] += 1000 * Lx; done += 1000 * Lx
                    pend[j] = q - done
            pend = {j: q for j, q in pend.items() if q != 0}
        cfv = np.where(S != 0, CF[t], 0.0)
        E_close = cash + np.sum(S * np.nan_to_num(cfv))
        if mode == 'FIX':
            if t in plan:
                unit *= E_close / E_scaled
            else:
                unit *= E_close / E_prev
        else:
            unit = E_close / E_init
        nav[k] = unit
        if E_pre is not None:
            comp[k] = comps / (E_scaled if t in plan else E_pre)
        cashf[k] = cash / E_close if E_close > 0 else 0
        E_prev = E_close
    st['imp_amt'] = None
    return dict(nav=nav, comp=comp, cashf=cashf, st=st, E_end=E_close, unfilled=st['unfilled'])
