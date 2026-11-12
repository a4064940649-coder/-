#!/usr/bin/env python3
"""資料產生（不算任何正式統計量；各 run 只把淨值、成交日誌、成交彙總存進 scratchpad/csr）。
階段（python3 -I -B run_engine.py <stage>）：
  check  ：SensSim 預設 vs CSim 逐位（6 組）；乘數明寫 1.0 也逐位；日誌 vs o_turnover tradelog（100 萬／30 萬 REAL_opt seed 0，13 j）。
  center ：中心格（mo=mc=mt=1）6 格 × 3 變體 × 13 j seed 0，完整日誌＋換股紀錄＋逐日成本。
  grid   ：其餘 8 個 (mo, mc) 格 × 6 格 × 3 變體 × 13 j seed 0，淨值＋成交彙總。
  tick   ：mt=0.5（半個跳動點）中心格 × 6 格 × 39，淨值＋成交彙總。
  refund ：100 萬一般 × 39 × refund∈{cash, prop}，淨值＋逐日成本＋退款總額。
  coal   ：Shapley 6 因子 64 組合 × 39（100 萬一般），淨值（時點關的 32 組在 close 資料層的 worker 跑）。
  seeds  ：四角＋中心 × 6 格 × 39 × 20 seeds → 每 (格, 變體, j) 的 20 seeds 平均淨值（x_stats.avg_nav，seed 0..19 依序）。"""
import sys, json, time, pickle, itertools
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import csr_lib as L; sys.path.pop(0)
SCR = L.SCR; SCR.mkdir(parents=True, exist_ok=True)
_G = {}

def init(close=False, seeds='0'):
    E = L.engine(close=close); _G['E'] = E
    P = L.load_plans()
    if seeds == '0': P = P[P.seed == 0]
    _G['G'] = {k: g for k, g in P.groupby(['variant', 'j', 'seed'])}

def mk(v, j, s, cell, close=False, **kw):
    E = _G['E']; D = E['D']
    pl = L.to_plan(_G['G'][(v, j, s)], D, close=close)
    cfg = L.cfg_for(E, cell[1], cell[0])
    return E['Sim'](pl, dict(cfg), settle=E['ST'], cmode='consideration', wholelot=(cell[2] == 'wholelot'), **kw)

def full(nav):
    D = _G['E']['D']; a = np.full(len(D.IDX), np.nan); a[len(D.IDX) - len(nav):] = nav.values
    assert nav.index[-1] == D.IDX[-1]
    return a

SUMPER = ['ALL', 'SEL14', 'S0618', 'Y19on', 'PREODD', 'POSTODD', 'V1520', 'H2126']
def summarize(sim, nav, j, cell):
    """成交彙總：各期間 (s0, s1] 的下單日、順延延遲、成本金額與 Σ成本/E_pre、雙邊周轉、零股比例、最低手續費比例。"""
    E = _G['E']; D = E['D']; S = E['S']
    c0 = L.cash0(E, cell[1])
    e = S.end_day('AB_V7_B10', j); nv = nav.loc[:e].dropna()
    F = pd.DataFrame(sim.fills, columns=L.FILL_COLS)
    F = F[F.shares > 0]
    rt = np.array(sorted(sim.reb)); k = np.searchsorted(rt, F.t.values, side='right') - 1
    F['treb'] = rt[k]
    F['dl_td'] = F.t - F.treb
    F['dl_cd'] = (D.IDX[F.t.values] - D.IDX[F.treb.values]).days
    dc = pd.Series(sim.daycost)
    navv = nav.reindex(D.IDX[F.t.values]).values
    F['Epre'] = c0 * navv + dc.reindex(F.t.values).fillna(0).values
    F['cost'] = F.fee + F.tax + F.tick_amt + F.imp_amt + F.odd_amt + F.stress_amt
    F['w'] = 1.0 / F.Epre
    F['hi'] = (F.ref >= 500).astype(int)
    F['date'] = D.IDX[F.t.values]
    out = {}
    for p in SUMPER:
        a, b = L.PER[p]
        try: s0, s1 = L.seg(nv, a, b, xs=(p in L.XS_START))
        except IndexError: continue
        f = F[(F.date > s0) & (F.date <= s1)]
        yrs = (s1 - s0).days / 365.25
        td = int(((nv.index > s0) & (nv.index <= s1)).sum())
        days = f.groupby('t').ctx.agg(lambda x: 'R' if (x == 'R').any() else 'P')
        Pf = f[f.ctx == 'P']
        ch = Pf.groupby(['treb', 'j']).dl_td.max() if len(Pf) else pd.Series(dtype=float)
        chc = Pf.groupby(['treb', 'j']).dl_cd.max() if len(Pf) else pd.Series(dtype=float)
        q = lambda x, pp: float(np.percentile(x, pp)) if len(x) else None
        o = dict(yrs=yrs, n_tdays=td, n_fill_days=int(len(days)), n_R_days=int((days == 'R').sum()), n_P_days=int((days == 'P').sum()),
                 n_fills=int(len(f)), n_odd=int((f.kind == 'O').sum()), n_P=int(len(Pf)),
                 not_tot=float(f.notional.sum()), not_odd=float(f.notional[f.kind == 'O'].sum()), not_P=float(Pf.notional.sum()),
                 turn2=float((f.notional * f.w).sum()), fee_min_share=float((f.fee <= 20.0001).mean()) if len(f) else None,
                 n_chains=int(len(ch)), chain_td_p50=q(ch, 50), chain_td_p90=q(ch, 90), chain_td_max=(float(ch.max()) if len(ch) else None), chain_td_mean=(float(ch.mean()) if len(ch) else None),
                 chain_cd_p50=q(chc, 50), chain_cd_p90=q(chc, 90), chain_cd_max=(float(chc.max()) if len(chc) else None),
                 fill_td_p50=q(Pf.dl_td, 50), fill_td_p90=q(Pf.dl_td, 90))
        for comp in ('fee', 'tax', 'tick_amt', 'imp_amt', 'odd_amt', 'stress_amt', 'cost'):
            o['amt_' + comp] = float(f[comp].sum()); o['drag_' + comp] = float((f[comp] * f.w).sum())
            o['drag_' + comp + '_hi'] = float((f[comp] * f.w)[f.hi == 1].sum())
        o['not_hi'] = float(f.notional[f.hi == 1].sum()); o['not_odd_hi'] = float(f.notional[(f.hi == 1) & (f.kind == 'O')].sum())
        out[p] = o
    return out

def job_check(a):
    v, j, cell, mode = a
    E = _G['E']
    if mode == 'ident':
        n1 = mk(v, j, 0, cell).run(); n2 = mk(v, j, 0, cell, mo=1.0, mc=1.0, mt=1.0, sw=dict(L.ALLON), refund=None, log=True).run()
        pl = L.to_plan(_G['G'][(v, j, 0)], E['D']); cfg = L.cfg_for(E, cell[1], cell[0])
        Cls = E['CV'].CSim
        if cell[2] == 'wholelot':
            sys.path.insert(0, str(L.AG / 'integrate_o19')); XE = E['XE']
            class WholeLotPre(E['CV'].CSim):
                def oddcap(self, t, j_):
                    if t < XE.I_ODD: return 0
                    return super().oddcap(t, j_)
            Cls = WholeLotPre
        ref = Cls(pl, dict(cfg), settle=E['ST'], cmode='consideration').run()
        return a, float(np.nanmax(np.abs(ref.values - n1.values))), float(np.nanmax(np.abs(ref.values - n2.values)))
    sim = mk(v, j, 0, cell, log=True); sim.run()
    F = pd.DataFrame(sim.fills, columns=L.FILL_COLS); F = F[F.shares > 0]
    return a, F, None

def job_center(a):
    v, j, cell = a
    sim = mk(v, j, 0, cell, log=True); nav = sim.run()
    fd = pd.DataFrame(sim.fills, columns=L.FILL_COLS)
    for c in ('t', 'j', 'shares', 'req'): fd[c] = fd[c].astype(np.int32)
    fd['side'] = fd.side.astype(np.int8); fd['kind'] = fd.kind.astype('category'); fd['ctx'] = fd.ctx.astype('category')
    return a, full(nav), fd, sim.rebs, dict(sim.daycost), dict(sim.acc), summarize(sim, nav, j, cell), L.pstats(_G['E']['S'], nav, j)

def job_grid(a):
    v, j, cell, mo, mc, mt = a
    sim = mk(v, j, 0, cell, mo=mo, mc=mc, mt=mt, log=True); nav = sim.run()
    return a, full(nav), dict(sim.acc), summarize(sim, nav, j, cell), L.pstats(_G['E']['S'], nav, j)

def job_refund(a):
    v, j, mode = a
    cell = ('normal', '100萬', 'std')
    sim = mk(v, j, 0, cell, refund=mode); nav = sim.run()
    return a, full(nav), dict(sim.daycost), sim.refunded, L.pstats(_G['E']['S'], nav, j)

def job_coal(a):
    coal, v, j = a
    close = _G['E']['close']
    sw = dict(tax=int('X' in coal), fee=int('F' in coal), tick=int('K' in coal), imp=int('I' in coal), caps=int('C' in coal), timing='open' if 'T' in coal else 'close')
    assert (sw['timing'] == 'close') == close
    sim = mk(v, j, 0, ('normal', '100萬', 'std'), close=close, sw=sw); nav = sim.run()
    return a, full(nav), L.pstats(_G['E']['S'], nav, j, pers=['Y19on'])['Y19on']['cagr']

def job_seeds(a):
    v, j, cell, mo, mc = a
    S = _G['E']['S']
    navs = [mk(v, j, s, cell, mo=mo, mc=mc).run() for s in range(20)]
    av = S.avg_nav(navs)
    return a, full(av)

def pool(fn, jobs, init_args=(False, "0"), n=3, every=100, t0=None):
    t0 = t0 or time.time(); out = []
    with Pool(n, initializer=init, initargs=init_args) as P:
        for i, r in enumerate(P.imap_unordered(fn, jobs, chunksize=1)):
            out.append(r)
            if i % every == 0: print(fn.__name__, i, len(jobs), round(time.time() - t0), flush=True)
    return out

if __name__ == '__main__':
    st = sys.argv[1]; t0 = time.time(); meta = {}
    if st == 'check':
        jobs = [(v, j, c, 'ident') for (v, j, c) in [('REAL_opt', 0, L.CELLS[1]), ('REAL_opt', 6, L.CELLS[1]), ('REAL_p90', 3, L.CELLS[0]),
                                                      ('REAL_carry', 9, L.CELLS[3]), ('REAL_opt', 4, L.CELLS[4]), ('REAL_p90', 11, L.CELLS[5]), ('REAL_carry', 2, L.CELLS[2])]]
        jobs += [('REAL_opt', j, c, 'log') for c in (L.CELLS[1], L.CELLS[0]) for j in range(13)]
        res = pool(job_check, jobs, every=10)
        idn = [(list(map(str, a)), d1, d2) for a, d1, d2 in res if a[3] == 'ident']
        assert all(d1 == 0.0 and d2 == 0.0 for _, d1, d2 in idn), idn
        cmp = {}
        for tag, c in (('100n', L.CELLS[1]), ('30n', L.CELLS[0])):
            T = pd.read_csv(L.TFC / f'data/tradelog_{tag}_REAL_opt_seed0.csv.gz', dtype={'code': str})
            mine = pd.concat([F.assign(jj=a[1]) for a, F, _ in res if a[3] == 'log' and a[2] == c], ignore_index=True)
            E = L.engine(); D = E['D']
            mine['code'] = [D.COLS[x] for x in mine.j]; mine['date'] = D.IDX[mine.t.values].strftime('%Y-%m-%d')
            mine = mine.sort_values(['jj', 't', 'code', 'side', 'kind', 'ctx', 'shares']).reset_index(drop=True)
            T = T.sort_values(['j', 'date', 'code', 'side', 'kind', 'ctx', 'shares']).reset_index(drop=True)
            assert len(mine) == len(T), (tag, len(mine), len(T))
            assert (mine.date.values == T.date.values).all() and (mine.code.values == T.code.values).all() and (mine.shares.values == T.shares.values).all()
            cmp[tag] = dict(n=len(T), maxabs={k: float(np.max(np.abs(mine[k].values - T[k].values) / np.maximum(1.0, np.abs(T[k].values)))) for k in ('notional', 'fee', 'tax', 'px')})
        meta = dict(identity=idn, tradelog_cmp=cmp)
    elif st == 'center':
        jobs = [(v, j, c) for c in L.CELLS for v in L.VARS for j in range(13)]
        res = pool(job_center, jobs, every=40)
        NV = {}; FL = {c: [] for c in L.CELLS}; RB = {}; DC = {}; ACC = {}; SM = {}; PS = {}
        for a, nav, fills, rebs, dc, acc, sm, ps in res:
            v, j, c = a; key = f'{v}|{j}|{L.CELLKEY[c]}'
            NV[key] = nav; f = fills; f.insert(0, 'j_', np.int16(j)); f.insert(0, 'variant', v); FL[c].append(f)
            RB[key] = rebs; DC[key] = dc; ACC[key] = acc; SM[key] = sm; PS[key] = ps
        pd.DataFrame(NV).to_parquet(SCR / 'navs_center.parquet')
        for c in L.CELLS:
            X = pd.concat(FL[c], ignore_index=True); X['variant'] = X.variant.astype('category'); X['kind'] = X.kind.astype(str); X['ctx'] = X.ctx.astype(str)
            X.to_parquet(SCR / f'fills_center_{L.CELLKEY[c].replace("|", "_")}.parquet'); FL[c] = None
        pickle.dump(dict(rebs=RB, daycost=DC, acc=ACC, summ=SM, pstats=PS), open(SCR / 'center_meta.pkl', 'wb'))
    elif st in ('grid', 'tick'):
        if st == 'grid': pts = [(mo, mc, 1.0) for (mo, mc) in L.GRID if (mo, mc) != (1.0, 1.0)]
        else: pts = [(1.0, 1.0, 0.5)]
        jobs = [(v, j, c, mo, mc, mt) for (mo, mc, mt) in pts for c in L.CELLS for v in L.VARS for j in range(13)]
        res = pool(job_grid, jobs, every=200)
        NV = {}; ACC = {}; SM = {}; PS = {}
        for a, nav, acc, sm, ps in res:
            v, j, c, mo, mc, mt = a; key = f'{mo}|{mc}|{mt}|{v}|{j}|{L.CELLKEY[c]}'
            NV[key] = nav; ACC[key] = acc; SM[key] = sm; PS[key] = ps
        pd.DataFrame(NV).to_parquet(SCR / f'navs_{st}.parquet')
        pickle.dump(dict(acc=ACC, summ=SM, pstats=PS), open(SCR / f'{st}_meta.pkl', 'wb'))
    elif st == 'refund':
        jobs = [(v, j, m) for m in ('cash', 'prop') for v in L.VARS for j in range(13)]
        res = pool(job_refund, jobs, every=20)
        NV = {}; M = {}
        for a, nav, dc, rf, ps in res:
            key = f'{a[2]}|{a[0]}|{a[1]}'; NV[key] = nav; M[key] = dict(daycost=dc, refunded=rf, pstats=ps)
        pd.DataFrame(NV).to_parquet(SCR / 'navs_refund.parquet'); pickle.dump(M, open(SCR / 'refund_meta.pkl', 'wb'))
    elif st == 'coal':
        FAC = ['T', 'X', 'F', 'K', 'I', 'C']
        coals = [''.join(f for f, b in zip(FAC, bits) if b) for bits in itertools.product([0, 1], repeat=6)]
        NV = {}; Y = {}
        for close in (False, True):
            jobs = [(c, v, j) for c in coals if (('T' not in c) == close) for v in L.VARS for j in range(13)]
            for a, nav, y in pool(job_coal, jobs, init_args=(close, '0'), every=200, t0=t0):
                key = f'{a[0] or "none"}|{a[1]}|{a[2]}'; NV[key] = nav; Y[key] = y
        pd.DataFrame(NV).to_parquet(SCR / 'navs_coal.parquet')
        SH = pd.read_csv(L.TFC / 'data/shapley_coalitions.csv.gz', dtype={'coalition': str})
        SH = SH[SH.tier == '100萬']; FACS = 'TXFKIC'
        def key(s): return ''.join(f for f in FACS if f in s) or 'none'
        d = [abs(Y[f'{key(r.coalition) if r.coalition != "none" else "none"}|{r.variant}|{r.j}'] - r.CAGR_Y19on) for r in SH.itertuples()]
        meta = dict(n=len(NV), coal_vs_tfc_maxabs_pp=float(np.max(d)), n_cmp=len(d))
        assert meta['coal_vs_tfc_maxabs_pp'] <= 1e-6, meta
    elif st == 'seeds':
        jobs = [(v, j, c, mo, mc) for (mo, mc) in L.CORNERS for c in L.CELLS for v in L.VARS for j in range(13)]
        res = pool(job_seeds, jobs, init_args=(False, 'all'), every=50)
        NV = {f'{mo}|{mc}|{v}|{j}|{L.CELLKEY[c]}': nav for (v, j, c, mo, mc), nav in res}
        pd.DataFrame(NV).to_parquet(SCR / 'navs_seeds20.parquet')
    meta.update(stage=st, utc=pd.Timestamp.now('UTC').isoformat(), runtime_s=round(time.time() - t0, 1))
    mf = HERE / 'run_engine_meta.json'
    M = json.loads(mf.read_text(encoding='utf-8')) if mf.exists() else {}
    M[st] = meta; mf.write_text(json.dumps(M, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print('done', st, round(time.time() - t0), json.dumps(meta, ensure_ascii=False, default=str)[:2000])
