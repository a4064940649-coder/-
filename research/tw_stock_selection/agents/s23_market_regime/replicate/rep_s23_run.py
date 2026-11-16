#!/usr/bin/env python3
"""S23 獨立複製主程式：驗收 A0～A2 → 訓練期 8 設定 × 20 路徑（全期連續切訓練窗）→ 自行依 PREREG 選規則凍結 →
驗證期（重新起跑）8 設定 × 20 路徑、V1～V4（判定只用凍結規則；其餘 6 條為事後描述）→ 全期連續描述 → 結構下限 →
登記的隨機對照（種子 20261116＋i，20 次）＋ 加抽 200 次看方向。
輸出 replicate/results_rep.json。python3 -I -B rep_s23_run.py"""
import sys, json, time, resource, gc, hashlib
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rep_s23_lib as L
from rep_s23_lib import s20lib, s17lib, s10lib, s4lib, s2lib, stlib, COSTS

H = Path(__file__).resolve().parent
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = L.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22')
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
G = s20lib.growth_features(P)
RT = dict(S=G['S'], av=G['av'])
del G
s17lib.e50_fix_suspension(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
print('載入', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE = P.IDX[P.t0], P.IDX[-1]
TA, TB, T18 = lastb('2016-02-01'), lastb('2022-01-01'), lastb('2019-01-01')
D19, D24 = T18, lastb('2024-01-01')
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
CANDS = list(L.RULES)
CFG = ['CUR'] + CANDS
SEGV = ('2022', '2024-07~2025-04')
cg = stlib.cagr; mdd = stlib.maxdd
med = lambda v: float(np.median(v))


def sim(S, cap, fresh=False, block=None):
    if fresh:
        S = L.fresh(S)
    return s10lib.simulate_s10(P, S, U, cost=COSTS['strict'], capital=cap, block=block, **CELL)


def build(S0, df, c, trig_override=None):
    if c == 'CUR':
        return S0, 0
    tg = L.triggered(df, c) if trig_override is None else trig_override
    return L.apply_rule(P, S0, df, tg, L.RULES[c]['action'])


def cont_metrics(nav, lg, cap):
    sg = s17lib.seg_dd(nav)
    tr = cg(nav, TA, TB); dd = mdd(nav, TA, TB)
    return dict(train=tr, train_dd=dd, calmar=tr / abs(dd), y1618=cg(nav, TA, T18), y1921=cg(nav, T18, TB),
                since2019=cg(nav, D19, DE), full=cg(nav, D0, DE), valid=cg(nav, TB, DE), since2024=cg(nav, D24, DE), maxdd=mdd(nav, D0, DE),
                maxdd2019=mdd(nav, D19, DE), seg=sg, top5=s17lib.top5_share(lg, nav, cap))


def fresh_metrics(nav):
    sg = s17lib.seg_dd(nav)
    return dict(valid=cg(nav, TB, DE), valid_dd=mdd(nav, TB, DE), y2022_2023=cg(nav, TB, D24), since2024=cg(nav, D24, DE), seg={k: sg[k] for k in SEGV})


keyof = lambda cap, s: f'{int(cap/1e4)}萬|+{s}'
res = {'note': '獨立複製：未 import／執行／讀取 s23lib.py、train.py、validate.py、posthoc.py；引擎 s20lib→s17lib→s10lib；規則名單、環境變數、選擇、判定全部自寫',
       'acceptance': {}}

# ---------------------------------------------------------------- 第一輪：全期連續 8 設定 × 20 路徑；同一位移也跑重新起跑
CONT = {c: {} for c in CFG}
FR = {c: {} for c in CFG}
DFS = {}; S0_0 = None
for s in range(4):
    S0 = s10lib.make_signal_s10(P, U, RT, shift=s)
    df = L.regime_rows(P, U, RT, s)
    ref = {int(x['k']): x['targets'] for x in S0['info']}
    assert len(df) == len(ref) and all([str(P.COLS[j]) for j in f1] == ref[k] for k, f1 in zip(df['k'], df['f1']))
    DFS[s] = df
    if s == 0:
        S0_0 = S0
        navc, lgc, stc = sim(S0, 1e6)
        a0 = dict(since2019=cg(navc, D19, DE), full=cg(navc, D0, DE), maxdd=mdd(navc, D0, DE))
        a0['ok'] = bool(abs(a0['since2019'] - 0.4813) < 0.00025 and abs(a0['full'] - 0.3408) < 0.00025 and abs(a0['maxdd'] + 0.3340) < 0.00025)
        Sn, _ = L.apply_rule(P, S0, df, np.zeros(len(df), bool), 'exit')        # 門檻不觸發（全部 False）
        Sn2, _ = L.apply_rule(P, S0, df, L.triggered(df, 'BRD_STOP_Q20', thr=-1.0), 'stop')
        cols = ['code', 'ent_t', 'exit_t', 'why']
        a1 = {}
        for nm, SS in (('exit_no_trigger', Sn), ('stop_thr_minus1', Sn2)):
            nava, lga, _ = sim(SS, 1e6)
            a1[nm] = dict(trades_identical=bool(len(lga) == len(lgc) and (lga[cols].reset_index(drop=True) == lgc[cols].reset_index(drop=True)).all().all()),
                          nav_identical=bool(np.allclose(nava.to_numpy(), navc.to_numpy(), rtol=0, atol=1e-6)))
        # A2：MA_EXIT 觸發列的次日開盤，衛星持股全部賣出（停板／無開盤延後除外）
        Se, _ = build(S0, df, 'MA_EXIT')
        nave, lge, ste = sim(Se, 1e6)
        tk = [k for k, g in zip(df['k'], L.triggered(df, 'MA_EXIT')) if g]
        held = lge[[any(e <= k < x for k in tk) for e, x in zip(lge['ent_t'], lge['exit_t'])]]
        late = []
        for r in held.itertuples():
            k = max(k for k in tk if r.ent_t <= k < r.exit_t)
            if r.exit_t != k + 1:
                late.append(dict(code=r.code, k=str(P.IDX[k].date()), exit=str(r.exit_date.date()), why=r.why))
        res['acceptance'] = dict(A0=a0, A1=a1, A2=dict(positions_checked=int(len(held)), sold_later_than_next_open=len(late), late=late),
                                 F1_identical='4 個位移每個訊號列自寫 F1 與 make_signal_s10 info.targets 相同（assert）')
        print('驗收', json.dumps(res['acceptance'], ensure_ascii=False), flush=True)
        # 主跑（位移 0、100 萬）CUR 結構
        res['main_CUR_cont_stockfrac'] = dict(
            mean_since2019=float(stc['stockfrac'].loc[D19:].mean()),
            mean_since2019_after_20190101=float(stc['stockfrac'].loc['2019-01-01':].mean()),
            sat_share_since2019=float((stc['stockfrac'] / (1 - stc['B'] / navc)).loc['2019-01-01':].mean()),
            seg_mean={k: float(stc['stockfrac'].loc[a:b].mean()) for k, (a, b) in s17lib.SEGMENTS.items()},
            avg_npos=stc['avg_npos'])
        del navc, lgc, stc, nave, lge, ste, Sn, Sn2, Se
    for c in CFG:
        Sx, _ = build(S0, df, c)
        for cap in CAPS:
            nav, lg, _ = sim(Sx, cap)
            CONT[c][keyof(cap, s)] = cont_metrics(nav, lg, cap)
            del nav, lg
        del Sx
    gc.collect()
    print('連續 位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
keys = list(CONT['CUR'])


def pair(D, a, b, f, sub=None):
    g = (lambda d: d[f]) if sub is None else (lambda d: d[f][sub])
    d = np.array([g(D[a][k]) - g(D[b][k]) for k in keys]) * 100
    return dict(median_pp=med(d), npos=int((d > 0).sum()), min_pp=float(d.min()), max_pp=float(d.max()))


ntrig_tr0 = {c: int((L.triggered(DFS[0], c) & (DFS[0]['month'] <= L.TRAIN_LAST).to_numpy()).sum()) for c in CANDS}
train = {'CUR': dict(train_median=med([CONT['CUR'][k]['train'] for k in keys]), train_dd_median=med([CONT['CUR'][k]['train_dd'] for k in keys]))}
for c in CANDS:
    imp = []
    for k in keys:
        a, b = CONT[c][k], CONT['CUR'][k]
        better = (a['train_dd'] - b['train_dd'] > 0) or (a['calmar'] > b['calmar'])
        imp.append(bool(better and (a['train'] - b['train']) * 100 >= -1.0))
    p = pair(CONT, c, 'CUR', 'train')
    cal = np.array([CONT[c][k]['calmar'] - CONT['CUR'][k]['calmar'] for k in keys])
    train[c] = dict(train_median=med([CONT[c][k]['train'] for k in keys]), train_dd_median=med([CONT[c][k]['train_dd'] for k in keys]),
                    n_improved=int(sum(imp)), cagr_diff=p, dd_diff=pair(CONT, c, 'CUR', 'train_dd'),
                    calmar_diff_median=med(cal), calmar_npos=int((cal > 0).sum()),
                    y1618=pair(CONT, c, 'CUR', 'y1618'), y1921=pair(CONT, c, 'CUR', 'y1921'),
                    seg2018=pair(CONT, c, 'CUR', 'seg', '2018'), seg2020=pair(CONT, c, 'CUR', 'seg', '2020-03'),
                    months_triggered_shift0_train=ntrig_tr0[c])
order = sorted(CANDS, key=lambda c: (-train[c]['n_improved'], -train[c]['calmar_diff_median'], train[c]['months_triggered_shift0_train']))
FZ = order[0]
T0 = bool(train[FZ]['n_improved'] >= 14 and train[FZ]['cagr_diff']['median_pp'] >= -1.0)
bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
res['train'] = dict(window=[str(TA.date()), str(TB.date())], summary=train, order=order, frozen_rep=FZ, T0_pass=T0,
                    bench_0050=dict(cagr=cg(bench, TA, TB), maxdd=mdd(bench, TA, TB)))
print('訓練', json.dumps({c: (train[c].get('n_improved'), round(train[c].get('cagr_diff', {}).get('median_pp', 0), 3), round(train[c].get('dd_diff', {}).get('median_pp', 0), 3)) for c in CANDS}, ensure_ascii=False), '\n凍結', FZ, 'T0', T0, flush=True)

# ---------------------------------------------------------------- 重新起跑：資金比例、主跑、禁買聯集
S0 = S0_0; df0 = DFS[0]
navx = sim(S0, 1e6, fresh=True)[0]
ratio = float(navx.loc[TB] / 1e6)
FCAP = {cap: cap / ratio for cap in CAPS}
MAIN = {}; mainlg = {}; mainnav = {}
for c in ('CUR', FZ):
    Sx, _ = build(S0, df0, c)
    for mode in ('fresh', 'cont'):
        fr = mode == 'fresh'
        nav, lg, st = sim(Sx, FCAP[1e6] if fr else 1e6, fresh=fr)
        b0 = TB if fr else D0
        R = dict(valid=cg(nav, TB, DE), valid_dd=mdd(nav, TB, DE), maxdd=mdd(nav, b0, DE), seg=s17lib.seg_dd(nav),
                 top5_share=s17lib.top5_share(lg, nav.loc[b0:], float(nav.loc[b0])), n_trades=int(len(lg)))
        s_ = nav.loc[b0:]; dd = s_ / s_.cummax() - 1; tr_ = dd.idxmin()
        R['dd_peak_trough'] = [str(s_.loc[:tr_].idxmax().date()), str(tr_.date())]
        if not fr:
            R.update(since2019=cg(nav, D19, DE), full=cg(nav, D0, DE))
        else:
            R['first_entry'] = str(lg['ent_date'].min().date())
        MAIN[f'{c}|{mode}'] = R; mainlg[f'{c}|{mode}'] = lg; mainnav[f'{c}|{mode}'] = nav
res['acceptance']['A3_fresh_no_trade_before_2022'] = bool((mainlg['CUR|fresh']['ent_date'] >= pd.Timestamp('2022-01-01')).all())
res['fresh_capital_ratio'] = ratio
res['main'] = MAIN
res['bench_0050'] = dict(seg=s17lib.seg_dd(bench), since2019=cg(bench, D19, DE), valid=cg(bench, TB, DE))
blkU = {}
for nm in ('CUR|fresh', f'{FZ}|fresh'):
    b, _ = s2lib.block_from(mainlg[nm], 5)
    for j, v in b.items():
        blkU.setdefault(j, []).extend(v)
res['V3_block_union'] = {str(P.COLS[j]): [[str(P.IDX[a].date()), str(P.IDX[b].date())] for a, b in v] for j, v in blkU.items()}
print('主跑', json.dumps(MAIN, ensure_ascii=False, default=str)[:1500], flush=True)

# ---------------------------------------------------------------- 第二輪：重新起跑 × 20 路徑（8 設定；禁買聯集只跑 CUR 與凍結）
FRB = {'CUR': {}, FZ: {}}
for s in range(4):
    S0s = S0_0 if s == 0 else s10lib.make_signal_s10(P, U, RT, shift=s)
    df = DFS[s]
    for c in CFG:
        Sx, _ = build(S0s, df, c)
        for cap in CAPS:
            k = keyof(cap, s)
            FR[c][k] = fresh_metrics(sim(Sx, FCAP[cap], fresh=True)[0])
            if c in FRB:
                FRB[c][k] = dict(valid=cg(sim(Sx, FCAP[cap], fresh=True, block=blkU)[0], TB, DE))
        del Sx
    if s:
        del S0s
    gc.collect()
    print('重新起跑 位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

FRX = {f'{c}|fresh': v for c, v in FR.items()}
FRX.update({f'{c}|fresh_blk': v for c, v in FRB.items()})
pf = lambda c: dict(valid=pair(FRX, f'{c}|fresh', 'CUR|fresh', 'valid'), valid_dd=pair(FRX, f'{c}|fresh', 'CUR|fresh', 'valid_dd'),
                    y2022_2023=pair(FRX, f'{c}|fresh', 'CUR|fresh', 'y2022_2023'), since2024=pair(FRX, f'{c}|fresh', 'CUR|fresh', 'since2024'),
                    seg={g: pair(FRX, f'{c}|fresh', 'CUR|fresh', 'seg', g) for g in SEGV})
pv = pf(FZ)
pv['blk'] = pair(FRX, f'{FZ}|fresh_blk', 'CUR|fresh_blk', 'valid')
V_ = dict(V1=pv['valid_dd']['npos'] >= 15, V2=pv['valid']['median_pp'] >= -2.0, V3=pv['blk']['median_pp'] >= -2.0,
          V4=all(pv['seg'][g]['median_pp'] > 0 for g in SEGV))
V_ = {k: bool(v) for k, v in V_.items()}
res['valid'] = dict(pairs_FZ_vs_CUR=pv, medians={nm: {f: med([D[k][f] for k in keys]) for f in D[keys[0]] if f != 'seg'} for nm, D in FRX.items()})
res['verdict'] = dict(T0=T0, **V_, overall='通過' if (T0 and all(V_.values())) else '不通過')
res['posthoc_candidates_valid'] = {c: pf(c) for c in CANDS}
cp = {c: {f: pair(CONT, c, 'CUR', f) for f in ('since2019', 'full', 'valid', 'maxdd', 'top5')} for c in CANDS}
for c in CANDS:
    cp[c]['seg'] = {g: pair(CONT, c, 'CUR', 'seg', g) for g in s17lib.SEGMENTS}
res['cont_paths'] = dict(pairs=cp, medians={c: {f: med([CONT[c][k][f] for k in keys]) for f in ('since2019', 'full', 'valid', 'maxdd', 'top5')} for c in CFG},
                         seg_medians={c: {g: med([CONT[c][k]['seg'][g] for k in keys]) for g in s17lib.SEGMENTS} for c in CFG})
print('判定', res['verdict'], json.dumps(pv, ensure_ascii=False), flush=True)

# ---------------------------------------------------------------- 登記的隨機對照（位移 0、100 萬）＋加抽
base_f = MAIN['CUR|fresh']; base_c = MAIN['CUR|cont']
fz_f = dict(d=(MAIN[f'{FZ}|fresh']['valid'] - base_f['valid']) * 100, dd=(MAIN[f'{FZ}|fresh']['valid_dd'] - base_f['valid_dd']) * 100)
fz_c = dict(d=(MAIN[f'{FZ}|cont']['since2019'] - base_c['since2019']) * 100, dd=(MAIN[f'{FZ}|cont']['maxdd'] - base_c['maxdd']) * 100)
trig_fz = L.triggered(df0, FZ)
vmask = (df0['month'] >= L.VALID_FIRST).to_numpy()
n_v = int((trig_fz & vmask).sum()); n_c = int(trig_fz.sum())
act = L.RULES[FZ]['action']
del mainlg, mainnav
gc.collect()


def rand_draw(rng):
    iv = np.flatnonzero(vmask)
    tv = np.zeros(len(df0), bool); tv[rng.choice(iv, size=n_v, replace=False)] = True
    tc = np.zeros(len(df0), bool); tc[rng.choice(len(df0), size=n_c, replace=False)] = True
    Sv, _ = L.apply_rule(P, S0_0, df0, tv, act)
    navf = sim(Sv, FCAP[1e6], fresh=True)[0]
    Sc, _ = L.apply_rule(P, S0_0, df0, tc, act)
    navc = sim(Sc, 1e6)[0]
    return dict(fresh_d=(cg(navf, TB, DE) - base_f['valid']) * 100, fresh_dd=(mdd(navf, TB, DE) - base_f['valid_dd']) * 100,
                cont_d=(cg(navc, D19, DE) - base_c['since2019']) * 100, cont_dd=(mdd(navc, D0, DE) - base_c['maxdd']) * 100)


def summarize(R):
    rv = pd.DataFrame(R)
    q = lambda x: [float(x.quantile(0.1)), float(x.median()), float(x.quantile(0.9))]
    pct = lambda x, v: float((x <= v).mean())
    return dict(n=len(rv), fresh_d_q10_50_90=q(rv['fresh_d']), fresh_dd_q10_50_90=q(rv['fresh_dd']),
                cont_d_q10_50_90=q(rv['cont_d']), cont_dd_q10_50_90=q(rv['cont_dd']),
                FZ_pct=dict(fresh_d=pct(rv['fresh_d'], fz_f['d']), fresh_dd=pct(rv['fresh_dd'], fz_f['dd']),
                            cont_d=pct(rv['cont_d'], fz_c['d']), cont_dd=pct(rv['cont_dd'], fz_c['dd'])))


R20 = [rand_draw(np.random.default_rng(20261116 + i)) for i in range(20)]
print('隨機 20', json.dumps(summarize(R20), ensure_ascii=False), f'{time.time()-t_all:.0f}s', flush=True)
R200 = [rand_draw(np.random.default_rng(7_000_000 + i)) for i in range(200)]
res['random_control'] = dict(n_months=dict(valid=n_v, cont=n_c), FZ=dict(fresh=fz_f, cont=fz_c),
                             registered_seeds_20261116_plus_i=summarize(R20), extra_200_seeds_7000000_plus_i=summarize(R200), draws20=R20)
print('隨機 200', json.dumps(res['random_control']['extra_200_seeds_7000000_plus_i'], ensure_ascii=False), flush=True)
res['triggered_months_FZ'] = dict(valid=[str(m) for m, g, v in zip(df0['month'], trig_fz, vmask) if g and v], all=[str(m) for m, g in zip(df0['month'], trig_fz) if g])
res['paths_cont'] = CONT; res['paths_fresh'] = FR; res['paths_fresh_blk'] = FRB
res['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss(),
                   code_sha256={f: hashlib.sha256((H / f).read_bytes()).hexdigest() for f in ('rep_s23_lib.py', 'rep_s23_run.py', 'rep_s23_prep.py')})
(H / 'results_rep.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print('完成', res['meta'])
