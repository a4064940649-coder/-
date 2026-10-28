#!/usr/bin/env python3
"""樣本外（PREREG 之後一次跑完，不改規格）：各資金級距的可執行回測、拆解、敏感度、統計與判定。
必須先有 results.json 的 PREREG（sha256 吻合），而且 PREREG 列出的每個檔案雜湊都要與現在相同。
輸出：results.json 的 RESULTS、navs_main.parquet（主格逐 seed 淨值）、navs_abavg.parquet（各變體 40 seed 平均日報酬淨值）
用法：python3 -I -B run_oos.py
"""
import sys, json, hashlib, time, math, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import ecore as E

DRY = os.environ.get('EXEC_DRYRUN') == '1'      # 程式碼冒煙測試：2 seeds、輸出到 scratchpad、不印結果（PREREG 前只測流程）
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/exec_dry')
RES_F = HERE / 'results.json' if not DRY else SCR / 'results.json'
res = json.loads(RES_F.read_text(encoding='utf-8')) if not DRY else {'PREREG': {'file_sha256': {}}, 'PREREG_sha256': None}
if not DRY:
  assert 'PREREG' in res and 'PREREG_sha256' in res, 'PREREG 不存在：先跑 prereg.py'
  canon = json.dumps(res['PREREG'], ensure_ascii=False, sort_keys=True).encode()
  assert hashlib.sha256(canon).hexdigest() == res['PREREG_sha256'], 'PREREG 被改過'
  assert 'RESULTS' not in res, '樣本外已跑過（一次跑完）'
  for f, h in res['PREREG']['file_sha256'].items():
      assert E.sha(Path(f)) == h, f'{f} 與 PREREG 雜湊不符'
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)

MODEL = json.loads((E.DATA / 'model_spec.json').read_text(encoding='utf-8'))
SEEDS = list(range(40)) if not DRY else [0, 1]; TIERS = list(E.TIERS); VERS = ('GROW', 'FIX'); FEES = ('A', 'B', 'C')
DECOMP = {   # 依序加入摩擦（fee A）
    'd1_integer': dict(integer=True, odd_time=False, odd_spread=False, fees=False, impact='flat', cap=False),
    'd2_oddtime': dict(integer=True, odd_time=True, odd_spread=False, fees=False, impact='flat', cap=False),
    'd3_oddspread': dict(integer=True, odd_time=True, odd_spread=True, fees=False, impact='flat', cap=False),
    'd4_fees': dict(integer=True, odd_time=True, odd_spread=True, fees=True, impact='flat', cap=False),
    'd5_impact': dict(integer=True, odd_time=True, odd_spread=True, fees=True, impact='sqrt', cap=False),
}
SENS = {
    'kappa_lo': dict(model=dict(oa_scale=MODEL['oa']['kappa_lo_p25'] / MODEL['oa']['kappa'])),
    'kappa_hi': dict(model=dict(oa_scale=MODEL['oa']['kappa_hi_upper'] / MODEL['oa']['kappa'])),
    'oa_pre2020_half': dict(model=dict(oa_pre_cut='2020-03-23', oa_pre_scale=0.5)),
    'odd_stress': dict(model=dict(odd_stress=True)),
    'keep': dict(keep=True),
    'stress': dict(fee='B', model=dict(oa_scale=MODEL['oa']['kappa_lo_p25'] / MODEL['oa']['kappa'], odd_stress=True)),
}

_D = {}
def _init():
    _D['D'] = E.Data()

def job(spec):
    D = _D['D']
    name, seed, tier, ver, fee = spec['name'], spec['seed'], spec['tier'], spec['ver'], spec['fee']
    m = dict(MODEL); m.update(spec.get('model', {}))
    M = E.ExecModel(m)
    collect = spec.get('collect', False)
    nav, info = E.simulate(D, seed, tier=E.TIERS[tier], version=ver, fee=fee, model=M, flags=spec.get('flags'), keep=spec.get('keep', False), collect=collect)
    acc = info['acc']
    tv = np.zeros(len(nav))
    for t, v in acc['_tv'].items(): tv[t - D.t0] = v
    out = dict(spec=spec, nav=nav.to_numpy(), tv=tv.astype(np.float32),
               acc={k: float(v) for k, v in acc.items() if not k.startswith('_')}, ndays=len(nav))
    if collect:
        R = info['rows']
        out['rows'] = np.array([(r[0], r[1], r[3], r[4], 1 if r[5] == 'lot' else 0, r[6], r[7], r[8] if r[8] == r[8] else np.nan, r[9], 1 if r[2] == 'reb' else 0) for r in R], dtype=float)
    return out

def specs():
    L = []
    for seed in SEEDS:
        for tier in TIERS:
            for ver in VERS:
                for fee in FEES:
                    L.append(dict(name='main', seed=seed, tier=tier, ver=ver, fee=fee, collect=(fee == 'A')))
                for k, fl in DECOMP.items():
                    L.append(dict(name=k, seed=seed, tier=tier, ver=ver, fee='A', flags=fl))
            for k, s in SENS.items():
                vs = VERS if k == 'stress' else ('GROW',)
                for ver in vs:
                    d = dict(name=k, seed=seed, tier=tier, ver=ver, fee=s.get('fee', 'A'))
                    if 'model' in s: d['model'] = s['model']
                    if 'keep' in s: d['keep'] = True
                    L.append(d)
    for tier in TIERS:
        for ver in VERS:
            L.append(dict(name='V4plan', seed=-1, tier=tier, ver=ver, fee='A'))
    return L

if __name__ == '__main__':
    D = E.Data()
    IDXN = D.IDX[D.t0:D.t1 + 1]
    SPECS = specs()
    log('jobs', len(SPECS))
    OUT = []
    with ProcessPoolExecutor(max_workers=4, initializer=_init) as ex:
        for i, o in enumerate(ex.map(job, SPECS, chunksize=4)):
            OUT.append(o)
            if i % 200 == 0: log('done', i)
    log('sims done')
    # ---------------- 零衝擊（引擎）逐 seed ----------------
    Z = pd.read_parquet(E.DATA / 'zero_impact_seed_navs.parquet')
    ZS = {s: (Z[f's{s:02d}'] / Z[f's{s:02d}'].iloc[0]).reindex(IDXN) for s in SEEDS}
    E50 = D.E50.reindex(IDXN); AB = D.AB.reindex(IDXN); AB = AB / AB.iloc[0]; ABEW = D.ABEW.reindex(IDXN); ABEW = ABEW / ABEW.iloc[0]
    ABS = D.ABS.reindex(IDXN); ABS = ABS / ABS.iloc[0]
    sd = D.sig_days
    def key(o): s = o['spec']; return (s['name'], s['tier'], s['ver'], s['fee'])
    groups = {}
    for o in OUT:
        groups.setdefault(key(o), {})[o['spec']['seed']] = o
    def abavg(g):
        R = pd.DataFrame({s: pd.Series(g[s]['nav'], index=IDXN).pct_change() for s in SEEDS if s in g})
        return (1 + R.mean(axis=1).fillna(0)).cumprod()
    def stats(nav, full=True):
        o = {}
        for p, (a, b) in E.PER.items():
            cg, mdd = E.cagr_mdd(nav, a, b); x = E.excess(nav, E50, sd, a, b); xe = E.excess(nav, ABEW, sd, a, b)
            o[p] = dict(CAGR=cg, maxDD=mdd, t_0050=E.tstat(x), mean_pp=float(x.mean() * 100), n=int(len(x)), t_EW=E.tstat(xe),
                        E50_CAGR=E.cagr_mdd(E50, a, b)[0], ZI_AB_CAGR=E.cagr_mdd(AB, a, b)[0])
        if full:
            x = E.excess(nav, E50, sd); xs = x.sort_values(ascending=False)
            o['endpoint'] = {f'drop_best{k}': E.tstat(x.drop(xs.index[:k])) for k in (1, 3, 5, 10)}
            o['endpoint']['drop_last1'] = E.tstat(x.iloc[:-1]); o['endpoint']['drop_first_year'] = E.tstat(x[x.index >= pd.Timestamp('2007-04-01')])
            o['endpoint']['start_2007'] = E.tstat(x[x.index >= pd.Timestamp('2007-01-01')])
            r = nav.pct_change(); r50 = E50.pct_change()
            o['yearly'] = {int(y): [round(float((1 + g).prod() - 1) * 100, 2), round(float((1 + r50[g.index].fillna(0)).prod() - 1) * 100, 2)] for y, g in r.iloc[1:].groupby(r.index[1:].year)}
        return o
    def paired(a, b, x0=E.A_, x1=E.B_):
        ra, rb = E.win_ret(a, sd), E.win_ret(b, sd)
        d = (ra - rb); d = d[(d.index >= pd.Timestamp(x0)) & (d.index <= pd.Timestamp(x1))].dropna()
        se = d.std(ddof=1) / math.sqrt(len(d))
        return dict(mean_pp_month=float(d.mean() * 100), t=E.tstat(d), n=int(len(d)), MDE80_pp_month=float(2.8 * se * 100), MDE80_pp_year=float(2.8 * se * 100 * 12))
    seedstats = lambda g: {s: (E.cagr_mdd(pd.Series(g[s]['nav'], index=IDXN), E.A_, E.B_)[0], E.tstat(E.excess(pd.Series(g[s]['nav'], index=IDXN), E50, sd))) for s in SEEDS if s in g}
    ZSTAT = {s: (E.cagr_mdd(ZS[s], E.A_, E.B_)[0], E.tstat(E.excess(ZS[s], E50, sd))) for s in SEEDS}
    R = {'zero_impact_AB_V4': stats(AB), 'zero_impact_AB_V4_stress': stats(AB, False), 'pool_EW': stats(ABEW, False), '0050': stats(E50 / E50.iloc[0], False),
         'zero_impact_seed': {'t_median': float(np.median([v[1] for v in ZSTAT.values()])), 't_min': float(min(v[1] for v in ZSTAT.values())),
                              't_max': float(max(v[1] for v in ZSTAT.values())), 'n_t_below_2': int(sum(v[1] < 2 for v in ZSTAT.values())),
                              'CAGR_median': float(np.median([v[0] for v in ZSTAT.values()]))}}
    NAVAB = {}
    tab = {}
    for (name, tier, ver, fee), g in groups.items():
        if name == 'V4plan':
            nv = pd.Series(g[-1]['nav'], index=IDXN); tab[f'{name}|{tier}|{ver}|{fee}'] = {'stats': stats(nv, False), 'acc': g[-1]['acc']}
            continue
        nav = abavg(g); NAVAB[f'{name}|{tier}|{ver}|{fee}'] = nav
        ss = seedstats(g)
        ts = np.array([v[1] for v in ss.values()]); cg = np.array([v[0] for v in ss.values()])
        e = {'stats': stats(nav, full=(name == 'main')),
             'seed_t': {'median': float(np.median(ts)), 'min': float(ts.min()), 'max': float(ts.max()), 'n_below_2.0': int((ts < 2.0).sum()), 'n_below_2.3': int((ts < 2.3).sum())},
             'seed_CAGR': {'median': float(np.median(cg)), 'min': float(cg.min()), 'max': float(cg.max())},
             'paired_vs_zero_impact_AB': paired(nav, AB),
             'paired_seed_vs_zero_seed': {'mean_dCAGR_pp': float(np.mean([ss[s][0] - ZSTAT[s][0] for s in ss])),
                                          'median_dt': float(np.median([ss[s][1] - ZSTAT[s][1] for s in ss]))},
             'acc_mean': {k: float(np.mean([g[s]['acc'][k] for s in g])) for k in g[SEEDS[0]]['acc']}}
        yrs = (IDXN[-1] - IDXN[0]).days / 365.25
        e['annual_cost_pct'] = {k[4:]: float(np.mean([g[s]['acc'][k] for s in g])) / yrs * 100 for k in g[SEEDS[0]]['acc'] if k.startswith('pct_')}
        e['mean_cash_frac'] = float(np.mean([g[s]['acc']['cash_frac_sum'] / (len(IDXN) - 1) for s in g]))
        e['end_equity_median'] = float(np.median([g[s]['acc']['end_equity'] for s in g]))
        tab[f'{name}|{tier}|{ver}|{fee}'] = e
    log('tables')
    # ---------------- 拆解（ΔCAGR、Δt，相對前一步；d0＝零衝擊引擎）----------------
    steps = ['zero', 'd1_integer', 'd2_oddtime', 'd3_oddspread', 'd4_fees', 'd5_impact', 'main']
    dec = {}
    for tier in TIERS:
        for ver in VERS:
            prev = AB; prevname = 'zero'; o = {}
            for st in steps[1:]:
                nv = NAVAB[f'{st}|{tier}|{ver}|A']
                o[st] = {'dCAGR_pp': E.cagr_mdd(nv, E.A_, E.B_)[0] - E.cagr_mdd(prev, E.A_, E.B_)[0],
                         'dt': E.tstat(E.excess(nv, E50, sd)) - E.tstat(E.excess(prev, E50, sd)),
                         'dCAGR_pre2020_10_pp': E.cagr_mdd(nv, *E.PER['PRE_ODD'])[0] - E.cagr_mdd(prev, *E.PER['PRE_ODD'])[0],
                         'dCAGR_post2020_10_pp': E.cagr_mdd(nv, *E.PER['POST_ODD'])[0] - E.cagr_mdd(prev, *E.PER['POST_ODD'])[0]}
                prev = nv
            # 取整追蹤誤差（逐 seed d1 對零衝擊）
            g = groups[('d1_integer', tier, ver, 'A')]
            te = [float((pd.Series(g[s]['nav'], index=IDXN).pct_change() - ZS[s].pct_change()).std() * math.sqrt(252) * 100) for s in SEEDS]
            o['rounding_TE_pp_median'] = float(np.median(te))
            gm = groups[('main', tier, ver, 'A')]
            te2 = [float((pd.Series(gm[s]['nav'], index=IDXN).pct_change() - ZS[s].pct_change()).std() * math.sqrt(252) * 100) for s in SEEDS]
            o['total_exec_TE_pp_median'] = float(np.median(te2))
            dec[f'{tier}|{ver}'] = o
    # ---------------- 損益兩平（在可執行成本之上再加單邊滑價 x）----------------
    be = {}
    grid = np.round(np.arange(0, 0.00401, 0.0001), 6)
    for tier in TIERS:
        for ver in VERS:
            g = groups[('main', tier, ver, 'A')]
            Rm = pd.DataFrame({s: pd.Series(g[s]['nav'], index=IDXN).pct_change() for s in SEEDS})
            TV = pd.DataFrame({s: pd.Series(g[s]['tv'].astype(float), index=IDXN) for s in SEEDS})
            pts = []
            for x in grid:
                nv = (1 + (Rm - x * TV).mean(axis=1).fillna(0)).cumprod()
                pts.append((float(x), E.tstat(E.excess(nv, E50, sd)), E.cagr_mdd(nv, E.A_, E.B_)[0]))
            def cross(th):
                if pts[0][1] < th: return 0.0
                for (a1, b1, _), (a2, b2, _) in zip(pts, pts[1:]):
                    if b1 >= th > b2: return a1 + (a2 - a1) * (b1 - th) / (b1 - b2)
                return float('inf')
            be[f'{tier}|{ver}'] = {'extra_slip_at_t2.3': cross(2.3), 'extra_slip_at_t2.0': cross(2.0), 'annual_turnover_two_sided': float(TV.mean(axis=1).sum() / ((IDXN[-1] - IDXN[0]).days / 365.25)),
                                   'grid_head': pts[:6]}
    # ---------------- 執行量（rows：主格 fee A）----------------
    PL = {s: D.plans[s] for s in SEEDS}
    newset = {}
    for s in SEEDS:
        prev = set()
        for e, pk, sg in PL[s]:
            newset[(s, e)] = set(pk) - prev; prev = set(pk)
    exq = {}
    for tier in TIERS:
        for ver in VERS:
            g = groups[('main', tier, ver, 'A')]
            lot = []; odd = []; pure = []
            for s in SEEDS:
                r = g[s]['rows']
                if r is None or len(r) == 0: continue
                df = pd.DataFrame(r, columns=['t', 'j', 'side', 'n', 'islot', 'val', 'cost', 'qoa', 'intended', 'reb'])
                df['seed'] = s
                lot.append(df[df.islot == 1]); odd.append(df[df.islot == 0])
                buys = df[(df.reb == 1) & (df.side > 0)]
                for e_, gg in buys.groupby('t'):
                    nw = newset.get((s, int(e_)), set())
                    for j, gj in gg.groupby('j'):
                        if int(j) in nw: pure.append((int(e_), int(j), int((gj.islot == 1).sum() == 0)))
            L_ = pd.concat(lot) if lot else pd.DataFrame(); O_ = pd.concat(odd) if odd else pd.DataFrame()
            o = {}
            if len(L_):
                Lr = L_[L_.reb == 1].copy()
                Lr['ratio_int'] = Lr.qoa * Lr.intended / Lr.n
                Lr['date'] = D.IDX[Lr.t.astype(int)]; Lr['vrank'] = D.VRANK[Lr.t.astype(int) - 1, Lr.j.astype(int)]
                Lr['absgap'] = np.abs(D.GAP[Lr.t.astype(int), Lr.j.astype(int)])
                o['lot_orders_reb'] = int(len(Lr))
                o['share_gt10pct_OA'] = float((Lr.ratio_int > 0.10).mean()); o['share_gt20pct_OA'] = float((Lr.ratio_int > 0.20).mean())
                o['share_gt10_2006_18'] = float((Lr[Lr.date.dt.year <= 2018].ratio_int > 0.10).mean())
                o['share_gt10_2019_26'] = float((Lr[Lr.date.dt.year >= 2019].ratio_int > 0.10).mean())
                o['share_gt10_rank201_300'] = float((Lr[Lr.vrank > 200].ratio_int > 0.10).mean())
                o['median_ratio'] = float(Lr.ratio_int.median())
                L_['imp_cost'] = L_.val * (L_.cost - 0.001) / (1 + L_.side * L_.cost)
                L_['vrank'] = D.VRANK[L_.t.astype(int) - 1, L_.j.astype(int)]; L_['absgap'] = np.abs(D.GAP[L_.t.astype(int), L_.j.astype(int)])
                tot = L_.imp_cost.sum()
                o['impact_cost_share_rank201_300'] = float(L_[L_.vrank > 200].imp_cost.sum() / tot) if tot > 0 else None
                o['lot_value_share_rank201_300'] = float(L_[L_.vrank > 200].val.sum() / L_.val.sum())
                o['impact_cost_share_absgap_gt3pct'] = float(L_[L_.absgap > 0.03].imp_cost.sum() / tot) if tot > 0 else None
                o['lot_value_share_absgap_gt3pct'] = float(L_[L_.absgap > 0.03].val.sum() / L_.val.sum())
                o['mean_impact_bp_by_rank'] = {str(k): round(float(v) * 1e4, 2) for k, v in L_.groupby(pd.cut(L_.vrank, [0, 100, 200, 300]), observed=True).cost.mean().items()}
                o['cap_hit_orders_share'] = float((Lr.ratio_int > 0.2).mean())
            if len(O_):
                O_['date'] = D.IDX[O_.t.astype(int)]; O_['p'] = D.PC[O_.t.astype(int), O_.j.astype(int)]
                O_['spread_cost'] = O_.val * O_.cost / (1 + O_.side * O_.cost)
                def oc(m):
                    x = O_[m]; return float(x.spread_cost.sum() / x.val.sum() * 1e4) if len(x) and x.val.sum() > 0 else None
                o['odd_cost_bp_price_gt300'] = oc(O_.p > 300); o['odd_cost_bp_price_le300'] = oc(O_.p <= 300)
                o['odd_cost_bp_pre2020_10'] = oc(O_.date < E.ODD_INTRADAY); o['odd_cost_bp_post2020_10'] = oc(O_.date >= E.ODD_INTRADAY)
                o['odd_buy_cost_bp'] = oc(O_.side > 0); o['odd_sell_cost_bp'] = oc(O_.side < 0)
                o['odd_value_share_of_trades'] = float(O_.val.sum() / (O_.val.sum() + (L_.val.sum() if len(L_) else 0)))
            if pure:
                P_ = pd.DataFrame(pure, columns=['t', 'j', 'pure'])
                P_['date'] = D.IDX[P_.t]
                o['pure_odd_new_entry_share'] = float(P_.pure.mean())
                o['pure_odd_pre2020_10'] = float(P_[P_.date < E.ODD_INTRADAY].pure.mean())
                o['pure_odd_post2020_10'] = float(P_[P_.date >= E.ODD_INTRADAY].pure.mean())
            exq[f'{tier}|{ver}'] = o
    log('exec stats')
    # ---------------- 可證偽：30 萬 vs 300 萬 ----------------
    fals = {}
    for ver in VERS:
        a, b = NAVAB[f'main|30萬|{ver}|A'], NAVAB[f'main|300萬|{ver}|A']
        fals[ver] = dict(dCAGR_pp=E.cagr_mdd(a, E.A_, E.B_)[0] - E.cagr_mdd(b, E.A_, E.B_)[0], paired=paired(a, b),
                         paired_pre2020_10=paired(a, b, *E.PER['PRE_ODD']), paired_post2020_10=paired(a, b, *E.PER['POST_ODD']))
    # ---------------- 判定 ----------------
    zi = E.cagr_mdd(AB, E.A_, E.B_)[0]; c50 = E.cagr_mdd(E50, E.A_, E.B_)[0]
    def verdict(st):
        t = st['ALL']['t_0050']; cg = st['ALL']['CAGR']
        if t < 2.0 or cg - c50 < 2.0: return '不建議'
        if t >= 2.3 and zi - cg <= 1.5: return '建議'
        return '可行但打折'
    VER = {}
    for tier in TIERS:
        VER[tier] = {}
        for ver in VERS:
            st = tab[f'main|{tier}|{ver}|A']['stats']
            VER[tier][ver] = {'verdict': verdict(st), 'CAGR': st['ALL']['CAGR'], 't': st['ALL']['t_0050'], 'gap_vs_zero_impact_pp': zi - st['ALL']['CAGR'],
                              'excess_vs_0050_pp': st['ALL']['CAGR'] - c50,
                              'verdict_feeB': verdict(tab[f'main|{tier}|{ver}|B']['stats']), 'verdict_stress': verdict(tab[f'stress|{tier}|{ver}|B']['stats'])}
    out = {'n_sims': len(OUT), 'table': tab, 'decomposition': dec, 'breakeven': be, 'exec_quantities': exq, 'falsification_30v300': fals,
           'verdict': VER, 'reference': R, 'runtime_s': time.time() - T0,
           'judgement_rule': '主欄：GROW、手續費 A、40 seed 平均日報酬淨值；建議＝t≥2.3 且年化比零衝擊 AB_V4 少 ≤1.5pp；可行但打折＝t≥2.0；不建議＝t<2.0 或對 0050 年化超額 <2pp。歷史回測判定上限待觀察。'}
    res['RESULTS'] = out
    RES_F.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)), encoding='utf-8')
    M = {}
    for o in OUT:
        s = o['spec']
        if s['name'] == 'main': M[f"{s['tier']}|{s['ver']}|{s['fee']}|s{s['seed']:02d}"] = o['nav']
    OD = HERE if not DRY else SCR
    pd.DataFrame(M, index=IDXN).to_parquet(OD / 'navs_main.parquet')
    pd.DataFrame(NAVAB).to_parquet(OD / 'navs_abavg.parquet')
    log('done', json.dumps(VER, ensure_ascii=False) if not DRY else '(dry run：不印結果)')
