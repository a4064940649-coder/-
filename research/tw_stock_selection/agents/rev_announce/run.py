#!/usr/bin/env python3
"""rev_announce 第 2 步：BASE／A_exFIN／B_pit_mc／B_pit_worst／AB（n10、n30）回測與配對比較、判定。
引擎＝main_refine/rlib.simulate（import，不改原檔）；資料、股池、分數全部來自 revenue_longhistory/lhlib.py。
輸出 results.json 的 RUN、navs.parquet。"""
import json, hashlib, sys, time, copy
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent / 'main_refine'))
import rlib as R  # noqa: E402
L = R.L

res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert hashlib.sha256(json.dumps(res['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == res['PREREG_sha256'], 'PREREG 被改過'
assert 'ANN' in res, '先跑 ann.py'
SEGS = {'P1': ('2006-04-01', '2020-12-31'), 'P2': ('2021-01-01', '2026-10-31'),
        'SEL': ('2006-04-01', '2013-12-31'), 'VAL': ('2014-01-01', '2020-12-31'), 'HOLD': ('2021-03-01', '2026-10-31')}
COSTS = {'slip0.001': L.cost_rates(0.001), 'stress': L.cost_rates('stress')}
NMC = int(sys.argv[1]) if len(sys.argv) > 1 else 200
SCR = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data')


def main():
    t0 = time.time()
    sl = pd.read_csv(SCR / 'stock_list.csv', dtype=str).drop_duplicates('code').set_index('code')
    COLS = list(R.COLS)
    FIN = set(sl.index[sl['industry'] == '金融保險業']) | {c for c in COLS if c[:2] == '28' or c in ('5876', '5880')}
    pd_ = json.loads((HERE / 'p_delay.json').read_text(encoding='utf-8'))
    pind, INS15 = pd_['p_by_industry'], set(pd_['INS15'])
    ind = [sl['industry'].get(c, '（stock_list 無）') if isinstance(sl['industry'].get(c), str) else '（stock_list 無）' for c in COLS]
    pcol = np.array([1.0 if c in FIN else pind.get(i, pd_['p_overall']) for c, i in zip(COLS, ind)])
    isfin = np.array([c in FIN for c in COLS]); is15 = np.array([c in INS15 for c in COLS])
    cal = pd.read_csv(HERE / 'deadline_calendar.csv', dtype=str).set_index('signal_day')
    fs = pd.read_csv(HERE / 'firstseen_2026-09.csv', dtype=str, index_col=0)

    Y = L.yoy_long(); S = L.Setup(Y, start='2005-01-01')
    F = R.Frames(S.dates, Y)
    Yx = Y.copy(); Yx[[c for c in Yx.columns if c in FIN]] = np.nan
    FA = R.Frames(S.dates, Yx)
    DATES = F.DATES; nd = len(DATES)
    # 舊營收（m−2 為止的 3 個月平均）
    mm2 = [p - 1 for p in F.mm]
    def stale(Fr):
        return pd.DataFrame(Fr.YOY3.reindex(mm2).to_numpy(), index=DATES, columns=Fr.YOY3.columns).reindex(columns=COLS)
    ST, STA = stale(F), stale(FA)
    # 可得性：每個訊號日的類別
    case = np.array(['D<S'] * nd, dtype=object); det = np.zeros((nd, len(COLS)), bool); rnd = np.zeros(nd, bool)
    for k, d in enumerate(DATES):
        key = str(d.date()); m = str(pd.Period(d, 'M') - 1)
        if key in cal.index: case[k] = cal.loc[key, 'case']
        if m == '2026-09':
            case[k] = 'ANN'
            ok = set(fs.index[fs['cat'] == 'A_before'])
            det[k] = ~np.array([c in ok for c in COLS])
        elif case[k] == 'D>S':
            det[k] = True
        elif case[k] == 'D=S':
            rnd[k] = True
        if '2026-01' <= m <= '2026-08':
            det[k] |= is15
    cases = pd.Series(case, index=DATES)

    def plan_with(Fr, STx, U):
        cur = Fr.yoy3
        y = cur.where(~U, STx)
        G = copy.copy(Fr); G.yoy3 = y
        return G.plan(y)

    def masks(seed):
        if seed == 'worst':
            U = det.copy(); U[rnd] = True; return pd.DataFrame(U, index=DATES, columns=COLS)
        if seed == 'none':
            return pd.DataFrame(np.zeros_like(det), index=DATES, columns=COLS)
        rng = np.random.default_rng(seed)
        U = det.copy(); u = rng.random((int(rnd.sum()), len(COLS)))
        U[rnd] |= u < pcol[None, :]
        return pd.DataFrame(U, index=DATES, columns=COLS)

    pl_base = F.plan(F.yoy3); T0 = pl_base[0][0]; T1 = len(R.IDX) - 1
    # 引擎檢查
    mr = pd.read_parquet(HERE.parent / 'main_refine' / 'navs.parquet')['BASE|slip0.001']
    b, s = COSTS['slip0.001']
    nb, _ = R.simulate(pl_base, b, s, T0, T1, n=10)
    md = float((nb.reindex(mr.index) / mr - 1).abs().max()); assert md < 1e-12, md
    # 「none」遮罩必須重現 BASE
    nb2, _ = R.simulate(plan_with(F, ST, masks('none')), b, s, T0, T1, n=10)
    assert float((nb2 / nb - 1).abs().max()) < 1e-12
    print('engine ok', round(time.time() - t0, 1), flush=True)

    navs = {}; info = {}
    det_plans = {'BASE': pl_base, 'A_exFIN': FA.plan(FA.yoy3), 'B_pit_worst': plan_with(F, ST, masks('worst')),
                 'AB_worst': plan_with(FA, STA, masks('worst'))}
    for name, pl in det_plans.items():
        for n in (10, 30):
            for ck, (bb, ss) in COSTS.items():
                nav, inf = R.simulate(pl, bb, ss, T0, T1, n=n)
                navs[f'{name}_n{n}|{ck}'] = nav
                if ck == 'slip0.001': info[f'{name}_n{n}'] = inf
    print('det done', round(time.time() - t0, 1), flush=True)
    # Monte Carlo
    mc = {f'{v}_n{n}|{ck}': [] for v in ('B_pit_mc', 'AB') for n in (10, 30) for ck in COSTS}
    flagged = {10: [], 30: []}
    base_picks = {}
    for seed in range(NMC):
        U = masks(seed)
        for v, Fr, STx in (('B_pit_mc', F, ST), ('AB', FA, STA)):
            pl = plan_with(Fr, STx, U)
            for n in (10, 30):
                for ck, (bb, ss) in COSTS.items():
                    nav, _ = R.simulate(pl, bb, ss, T0, T1, n=n)
                    mc[f'{v}_n{n}|{ck}'].append(nav.pct_change())
        if seed % 20 == 0: print('mc', seed, round(time.time() - t0, 1), flush=True)
    mc_cagr = {}
    for key, lst in mc.items():
        M = pd.concat(lst, axis=1)
        avg = M.mean(axis=1); avg.iloc[0] = 0.0
        navs[key] = (1 + avg).cumprod()
        # 每次的年化分佈
        dist = {}
        for sk in ('P1', 'P2'):
            a, bnd = SEGS[sk]
            cg = []
            for j in range(M.shape[1]):
                nj = (1 + M.iloc[:, j].fillna(0)).cumprod()
                st = R.seg_stats(nj, a, bnd); cg.append(st['CAGR'])
            dist[sk] = {'p05': round(float(np.percentile(cg, 5)), 4), 'p50': round(float(np.percentile(cg, 50)), 4), 'p95': round(float(np.percentile(cg, 95)), 4)}
        mc_cagr[key] = dist
    ND = pd.DataFrame(navs)
    ND.to_parquet(HERE / 'navs.parquet')
    print('mc done', round(time.time() - t0, 1), flush=True)

    # ---------- 統計 ----------
    out = {'engine_check_max_abs_diff_vs_main_refine_BASE': md, 'first_exec': str(R.IDX[T0].date()), 'last_day': str(R.IDX[T1].date()),
           'n_signal_dates': nd, 'signal_case_counts': cases.value_counts().to_dict(), 'NMC': NMC, 'info': info, 'variants': {}, 'mc_CAGR_dist': mc_cagr}
    names = ['BASE', 'A_exFIN', 'B_pit_mc', 'AB', 'B_pit_worst', 'AB_worst']
    for n in (10, 30):
        for v in names:
            key = f'{v}_n{n}'
            o = {'segments': {}, 'paired_vs_BASE': {}, 'paired_vs_BASE_stress': {}}
            for sk, (a, bnd) in SEGS.items():
                for ck in COSTS:
                    o['segments'][f'{sk}|{ck}'] = R.seg_stats(ND[f'{key}|{ck}'], a, bnd)
                if v != 'BASE':
                    p = R.paired(ND[f'{key}|slip0.001'], ND[f'BASE_n{n}|slip0.001'], a, bnd)
                    p['dCAGR_pp'] = round((o['segments'][f'{sk}|slip0.001']['CAGR'] - R.seg_stats(ND[f'BASE_n{n}|slip0.001'], a, bnd)['CAGR']) * 100, 2)
                    o['paired_vs_BASE'][sk] = p
                    o['paired_vs_BASE_stress'][sk] = R.paired(ND[f'{key}|stress'], ND[f'BASE_n{n}|stress'], a, bnd)
            if v != 'BASE':
                sig = {}
                for sk in ('P1', 'P2'):
                    p = o['paired_vs_BASE'][sk]
                    sig[sk] = bool(p['diff_pct_m'] < 0 and (p['t_paired_m'] <= -2.0 or p['dCAGR_pp'] <= -3.0))
                o['明顯下降'] = sig; o['verdict'] = '明顯下降' if any(sig.values()) else '沒有明顯下降'
            o['yearly_pct'] = R.yearly(ND[f'{key}|slip0.001'])
            out['variants'][key] = o
            print(key, {sk: (o['segments'][f'{sk}|slip0.001']['CAGR'], o['segments'][f'{sk}|slip0.001']['maxDD']) for sk in ('P1', 'P2')},
                  {sk: (x['diff_pct_m'], x['t_paired_m'], x['dCAGR_pp']) for sk, x in o['paired_vs_BASE'].items() if sk in ('P1', 'P2')}, o.get('verdict'), flush=True)
    e = R.E50.reindex(ND.index)
    out['0050'] = {sk: R.seg_stats(e, a, bnd) for sk, (a, bnd) in SEGS.items()}

    # ---------- 診斷：BASE 持股 ----------
    diag = {}
    for n in (10, 30):
        ks = {}
        rows = []
        for k, (ex, cand) in enumerate(pl_base):
            d = R.IDX[ex - 1]
            kk = DATES.get_loc(d)
            pick = []
            for j in cand:
                if L.GAP_UP_OPEN_ERA[ex, j]: continue  # 近似（不考慮續抱例外）
                pick.append(j)
                if len(pick) == n: break
            rows.append({'date': d, 'case': cases.iloc[kk], 'n_fin': int(isfin[pick].sum()), 'n_det_unavail': int(det[kk, pick].sum()),
                         'exp_unavail_DeqS': float(pcol[pick].sum()) if rnd[kk] else 0.0})
        df = pd.DataFrame(rows)
        for sk in ('P1', 'P2'):
            a, bnd = SEGS[sk]; g = df[(df.date >= a) & (df.date <= bnd)]
            ks[sk] = {'rebalances': int(len(g)), 'slots': int(len(g) * n), 'fin_slots': int(g.n_fin.sum()), 'fin_share': round(float(g.n_fin.sum() / (len(g) * n)), 4),
                      'rebal_by_case': g['case'].value_counts().to_dict(), 'det_unavail_slots': int(g.n_det_unavail.sum()),
                      'expected_unavail_slots_DeqS': round(float(g.exp_unavail_DeqS.sum()), 1),
                      'expected_unavail_share_all_slots': round(float((g.exp_unavail_DeqS.sum() + g.n_det_unavail.sum()) / (len(g) * n)), 4)}
        diag[f'n{n}'] = ks
    # 依換股期（訊號日→下個訊號日）比較：D=S 月 vs D<S 月的 B−BASE 差
    for n in (10, 30):
        bse = ND[f'BASE_n{n}|slip0.001']
        for v in ('B_pit_mc', 'B_pit_worst', 'A_exFIN', 'AB'):
            x = ND[f'{v}_n{n}|slip0.001']
            pr = lambda s: (s.reindex(DATES).shift(-1) / s.reindex(DATES) - 1)
            dd = (pr(x) - pr(bse)).iloc[:-1].dropna()
            g = {}
            for sk in ('P1', 'P2'):
                a, bnd = SEGS[sk]; z = dd[(dd.index >= a) & (dd.index <= bnd)]
                c = cases.reindex(z.index)
                g[sk] = {cs: {'n': int((c == cs).sum()), 'mean_pct': round(float(z[c == cs].mean() * 100), 3) if (c == cs).any() else None,
                              't': L.tstat(z[c == cs]) if (c == cs).sum() > 2 else None} for cs in ('D<S', 'D=S', 'D>S', 'ANN')}
            diag[f'by_case_{v}_n{n}'] = g
    out['DIAG'] = diag
    res['RUN'] = out
    (HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(json.dumps(diag, ensure_ascii=False, default=str)[:4000])
    print('done', round(time.time() - t0, 1))

if __name__ == '__main__':
    main()
