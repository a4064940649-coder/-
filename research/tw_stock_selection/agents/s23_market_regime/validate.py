#!/usr/bin/env python3
"""S23 驗證期（凍結後只跑一次）：凍結規則 FZ 對現行 CUR。由 S21 validate.py 改寫（規則套用改成 s23lib.apply_regime；對照組改成隨機減碼）。
(1) 判定用：重新起跑（只保留營收月份 ≥ 2021-12 的訊號；起始資金縮放使 2021-12-30 淨值約＝60～200 萬）× 20 條路徑，驗證窗 2021-12-30 → 2026-10-22；V1～V4。
(2) 描述用：全期連續模擬（主跑＋20 條路徑）：逐年、最大回撤、禁買最賺 5 筆、成本加倍、四段空頭。
(3) 登記的隨機對照：同窗口同月數隨機觸發同動作，20 個種子（位移 0、100 萬），重新起跑與全期連續各一組。
輸出 results.json、trades.csv.gz。PREREG.json、FROZEN.json 的 sha256 必須相符；results.json 已存在就拒絕重跑。python3 -I -B validate.py"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s23lib
from s23lib import s17lib, s10lib, s4lib, s2lib, stlib, COSTS

H = Path(__file__).resolve().parent
pre = (H / 'PREREG.json').read_bytes(); sha = hashlib.sha256(pre).hexdigest()
assert sha == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與 PREREG.sha256 不符'
fz = (H / 'FROZEN.json').read_bytes(); fsha = hashlib.sha256(fz).hexdigest()
assert fsha == (H / 'FROZEN.sha256').read_text().split()[0], 'FROZEN.json 與 FROZEN.sha256 不符'
assert not (H / 'results.json').exists(), 'results.json 已存在：驗證期只跑一次，拒絕重跑'
FZ = json.loads(fz)
RULE = FZ['frozen_rule']
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s23lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
RT = s23lib.surge(P)
s17lib.e50_fix_suspension(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
print('載入完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE = P.IDX[P.t0], P.IDX[-1]
D19, D24, TB = lastb('2019-01-01'), lastb('2024-01-01'), lastb('2022-01-01')
WIN = {'since2019': (D19, DE), 'full': (D0, DE), 'since2024': (D24, DE), 'y2019_2023': (D19, D24), 'valid': (TB, DE), 'y2022_2023': (TB, D24)}
W = lambda nav, ks=WIN: {k: stlib.window(nav, bench, *WIN[k]) for k in ks}
cg = lambda nav, k: stlib.cagr(nav, *WIN[k])
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
SEGV = ('2022', '2024-07~2025-04')


def variant(S, F, rule, trig=None):
    if rule is None:
        return S, []
    if trig is None:
        trig = s23lib.triggered(F, rule, rule.get('thr')).to_numpy()
    return s23lib.apply_regime(S, F, trig, rule['action'])


def sim(S1, cap=1e6, cost='strict', fresh=False, block=None):
    if fresh:
        S1 = s23lib.fresh_from(S1, s23lib.VALID_FIRST_MONTH)
    S2, ekw = s10lib.variant_S(S1, 'E0')
    return s10lib.simulate_s10(P, S2, U, cost=COSTS[cost], capital=cap, block=block, **CELL, **ekw)


def slim(w):
    return {k: dict(cagr=v['cagr'], cagr_0050=v['cagr_0050'], diff=v['diff'], maxdd=v['maxdd'], maxdd_0050=v['maxdd_0050']) for k, v in w.items()}


# ---------------------------------------------------------------- 起始資金縮放
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
F0 = s23lib.regime_features(P, U, RT, S0)
nav1, lg1, _ = sim(S0, 1e6, fresh=True)
ratio = float(nav1.loc[TB] / 1e6)
acc = dict(A3_no_trade_before_2022=bool((lg1['ent_date'] >= pd.Timestamp('2022-01-01')).all()), ratio_nav_TB=ratio, first_entry=str(lg1['ent_date'].min().date()))
assert acc['A3_no_trade_before_2022'], acc
del nav1, lg1
FCAP = {cap: cap / ratio for cap in CAPS}
trig0 = s23lib.triggered(F0, RULE, RULE.get('thr')).to_numpy()
mper0 = pd.PeriodIndex(F0['month'], freq='M')
vmask0 = np.asarray(mper0 >= s23lib.VALID_FIRST_MONTH)
cmask0 = np.asarray(mper0 >= pd.Period('2016-01', 'M'))

# ---------------------------------------------------------------- 主跑（位移 0）
res = {'PREREG_sha256': sha, 'FROZEN_sha256': fsha, 'frozen_rule': RULE, 'T0_pass': FZ['T0_pass'], 'acceptance': acc,
       'meta': dict(start=str(D0.date()), valid_from=str(TB.date()), end=str(DE.date())),
       'triggered_months': dict(valid=[m for m, g, v in zip(F0['month'], trig0, vmask0) if g and v], all=[m for m, g in zip(F0['month'], trig0) if g]),
       'bench_0050': {k: dict(cagr=stlib.cagr(bench, *WIN[k]), maxdd=stlib.maxdd(bench, *WIN[k])) for k in WIN}, 'main': {}}
res['bench_0050']['seg'] = s17lib.seg_dd(bench)
trades = []; mainlg = {}
VAR0 = {'CUR': S0, 'FZ': variant(S0, F0, RULE)[0]}
for mode in ('fresh', 'cont'):
    fresh = mode == 'fresh'
    cap = FCAP[1e6] if fresh else 1e6
    a = '2022-01-01' if fresh else '1900-01-01'
    for c in ('CUR', 'FZ'):
        nav, lg, st = sim(VAR0[c], cap, fresh=fresh)
        nm = f'{c}|{mode}'
        mainlg[nm] = lg
        ks = ('valid', 'y2022_2023', 'since2024') if fresh else tuple(WIN)
        w = W(nav, ks)
        blk, top = s2lib.block_from(lg[lg['ent_date'] >= pd.Timestamp(a)], 5)
        nb = sim(VAR0[c], cap, fresh=fresh, block=blk)[0]
        nx = sim(VAR0[c], cap, cost='strict_x2', fresh=fresh)[0]
        kk = ('valid',) if fresh else ('since2019', 'full', 'valid')
        b0 = TB if fresh else D0
        res['main'][nm] = dict(windows=slim(w), maxdd=stlib.maxdd(nav, b0, DE), dd_dates=s10lib.dd_dates(nav.loc[b0:]), seg=s17lib.seg_dd(nav),
                               yearly=stlib.yearly(nav.loc[b0:], bench), trades=stlib.trade_stats(lg, None if not fresh else a),
                               block5=slim(W(nb, kk)), block5_top=[dict(code=y.code, ent=str(y.ent_date.date()), exit=str(y.exit_date.date()), ret=float(y.ret), pnl=float(y.pnl)) for y in top.itertuples()],
                               costx2=slim(W(nx, kk)), top5_share=s17lib.top5_share(lg[lg['ent_date'] >= pd.Timestamp(a)], nav.loc[b0:], float(nav.loc[b0])),
                               n_buy=int(st['n_buy']), avg_npos=float(st['avg_npos']), avg_stockfrac=float(np.nanmean(st['stockfrac'].loc[b0:].to_numpy())))
        x = lg.copy(); x['run'] = nm
        trades.append(x[['run', 'code', 'sig_date', 'ent_date', 'exit_date', 'hold_days', 'pnl', 'ret', 'why']])
        del nav, nb, nx, st
    gc.collect()
print('主跑完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

blkU = {}
for nm in ('CUR|fresh', 'FZ|fresh'):
    b, _ = s2lib.block_from(mainlg[nm], 5)
    for j, v in b.items():
        blkU.setdefault(j, []).extend(v)
res['V3_block_union'] = {str(P.COLS[j]): [[str(P.IDX[a].date()), str(P.IDX[b].date())] for a, b in v] for j, v in blkU.items()}
del mainlg

# ---------------------------------------------------------------- 20 條路徑
PATH = {f'{c}|{m}': {} for c in ('CUR', 'FZ') for m in ('fresh', 'cont', 'fresh_blk')}
ntrig = {}
for s in range(4):
    S = S0 if s == 0 else s10lib.make_signal_s10(P, U, RT, shift=s)
    F = F0 if s == 0 else s23lib.regime_features(P, U, RT, S)
    SF, lgv = variant(S, F, RULE)
    ntrig[f'+{s}'] = int(sum(1 for x in lgv if pd.Period(x['month'], 'M') >= s23lib.VALID_FIRST_MONTH))
    VV = {'CUR': S, 'FZ': SF}
    for cap in CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        for c in ('CUR', 'FZ'):
            nav = sim(VV[c], FCAP[cap], fresh=True)[0]
            sg = s17lib.seg_dd(nav)
            PATH[f'{c}|fresh'][key] = dict(valid=cg(nav, 'valid'), valid_dd=stlib.maxdd(nav, TB, DE), y2022_2023=cg(nav, 'y2022_2023'), since2024=cg(nav, 'since2024'),
                                           seg={k: sg[k] for k in SEGV})
            nav, lg, _ = sim(VV[c], cap)
            PATH[f'{c}|cont'][key] = dict(since2019=cg(nav, 'since2019'), full=cg(nav, 'full'), valid=cg(nav, 'valid'), y2019_2023=cg(nav, 'y2019_2023'),
                                          since2024=cg(nav, 'since2024'), maxdd=stlib.maxdd(nav, D0, DE), maxdd2019=stlib.maxdd(nav, D19, DE), seg=s17lib.seg_dd(nav),
                                          top5=s17lib.top5_share(lg, nav, cap))
            nav = sim(VV[c], FCAP[cap], fresh=True, block=blkU)[0]
            PATH[f'{c}|fresh_blk'][key] = dict(valid=cg(nav, 'valid'))
            del nav
    del VV, SF
    if s:
        del S, F
    gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
res['paths'] = PATH; res['n_triggered_valid_by_shift'] = ntrig
keys = list(PATH['CUR|fresh'])
med = lambda v: float(np.median(v))


def pair(a, b, f, sub=None):
    g = (lambda d: d[f]) if sub is None else (lambda d: d[f][sub])
    d = np.array([g(PATH[a][k]) - g(PATH[b][k]) for k in keys]) * 100
    return dict(npos=int((d > 0).sum()), median_pp=med(d), min_pp=float(d.min()), max_pp=float(d.max()))


summ = {}
for nm, D in PATH.items():
    summ[nm] = {f: med([D[k][f] for k in keys]) for f in D[keys[0]] if f != 'seg'}
    if 'seg' in D[keys[0]]:
        summ[nm]['seg'] = {sg: med([D[k]['seg'][sg] for k in keys]) for sg in D[keys[0]]['seg']}
res['paths_median'] = summ
pv = {'FZ_vs_CUR|fresh': dict(valid=pair('FZ|fresh', 'CUR|fresh', 'valid'), valid_dd=pair('FZ|fresh', 'CUR|fresh', 'valid_dd'),
                              y2022_2023=pair('FZ|fresh', 'CUR|fresh', 'y2022_2023'), since2024=pair('FZ|fresh', 'CUR|fresh', 'since2024'),
                              seg={sg: pair('FZ|fresh', 'CUR|fresh', 'seg', sg) for sg in SEGV}),
      'FZ_vs_CUR|cont': {f: pair('FZ|cont', 'CUR|cont', f) for f in ('since2019', 'full', 'valid', 'y2019_2023', 'since2024', 'maxdd', 'maxdd2019', 'top5')},
      'FZ_vs_CUR|fresh_blk': dict(valid=pair('FZ|fresh_blk', 'CUR|fresh_blk', 'valid'))}
pv['FZ_vs_CUR|cont']['seg'] = {sg: pair('FZ|cont', 'CUR|cont', 'seg', sg) for sg in s17lib.SEGMENTS}
res['pairs'] = pv

# ---------------------------------------------------------------- 判定
p = pv['FZ_vs_CUR|fresh']
V = dict(V1=p['valid_dd']['npos'] >= 15, V2=p['valid']['median_pp'] >= -2.0, V3=pv['FZ_vs_CUR|fresh_blk']['valid']['median_pp'] >= -2.0,
         V4=all(p['seg'][sg]['median_pp'] > 0 for sg in SEGV))
V = {k: bool(v) for k, v in V.items()}
passed = bool(FZ['T0_pass'] and all(V.values()))
res['verdict'] = dict(T0=FZ['T0_pass'], **V, n_pass_V=int(sum(V.values())), overall='通過' if passed else '不通過',
                      text=('凍結規則在驗證期降低回撤且報酬代價在容忍內' if passed else '不通過：停在描述，不換規則、不換門檻、不寫 XQ 修改建議'))
print('判定', json.dumps(res['verdict'], ensure_ascii=False), flush=True)

# ---------------------------------------------------------------- 登記的隨機對照（位移 0、100 萬）
navC_f = sim(S0, FCAP[1e6], fresh=True)[0]; navC_c = sim(S0, 1e6)[0]
base = dict(fresh=dict(valid=cg(navC_f, 'valid'), dd=stlib.maxdd(navC_f, TB, DE), seg={k: s17lib.seg_dd(navC_f)[k] for k in SEGV}),
            cont=dict(since2019=cg(navC_c, 'since2019'), dd=stlib.maxdd(navC_c, D0, DE), seg={k: s17lib.seg_dd(navC_c)[k] for k in SEGV}))
navF_f = sim(VAR0['FZ'], FCAP[1e6], fresh=True)[0]; navF_c = sim(VAR0['FZ'], 1e6)[0]
frz = dict(fresh=dict(valid=cg(navF_f, 'valid'), dd=stlib.maxdd(navF_f, TB, DE), seg={k: s17lib.seg_dd(navF_f)[k] for k in SEGV}),
           cont=dict(since2019=cg(navF_c, 'since2019'), dd=stlib.maxdd(navF_c, D0, DE), seg={k: s17lib.seg_dd(navF_c)[k] for k in SEGV}))
n_v = int((trig0 & vmask0).sum()); n_c = int((trig0 & cmask0).sum())
rnd = dict(fresh=[], cont=[])
for i in range(20):
    rng = np.random.default_rng(20261116 + i)
    tv = s23lib.random_trig(len(F0), vmask0, n_v, rng)
    nav = sim(variant(S0, F0, RULE, tv)[0], FCAP[1e6], fresh=True)[0]
    rnd['fresh'].append(dict(valid=cg(nav, 'valid'), dd=stlib.maxdd(nav, TB, DE), seg={k: s17lib.seg_dd(nav)[k] for k in SEGV}))
    tc = s23lib.random_trig(len(F0), cmask0, n_c, rng)
    nav = sim(variant(S0, F0, RULE, tc)[0], 1e6)[0]
    rnd['cont'].append(dict(since2019=cg(nav, 'since2019'), dd=stlib.maxdd(nav, D0, DE), seg={k: s17lib.seg_dd(nav)[k] for k in SEGV}))
    del nav
pct = lambda x, arr: float((np.array(arr) < x).mean() + 0.5 * (np.array(arr) == x).mean())
rc = dict(n_months=dict(valid=n_v, cont=n_c), base_CUR=base, frozen=frz, random=rnd)
rc['summary'] = dict(
    fresh=dict(frozen_valid_diff_pp=(frz['fresh']['valid'] - base['fresh']['valid']) * 100, frozen_dd_diff_pp=(frz['fresh']['dd'] - base['fresh']['dd']) * 100,
               random_valid_diff_pp_q10_50_90=[float(np.quantile([(r['valid'] - base['fresh']['valid']) * 100 for r in rnd['fresh']], q)) for q in (0.1, 0.5, 0.9)],
               random_dd_diff_pp_q10_50_90=[float(np.quantile([(r['dd'] - base['fresh']['dd']) * 100 for r in rnd['fresh']], q)) for q in (0.1, 0.5, 0.9)],
               frozen_pct_valid=pct(frz['fresh']['valid'], [r['valid'] for r in rnd['fresh']]), frozen_pct_dd=pct(frz['fresh']['dd'], [r['dd'] for r in rnd['fresh']])),
    cont=dict(frozen_since2019_diff_pp=(frz['cont']['since2019'] - base['cont']['since2019']) * 100, frozen_dd_diff_pp=(frz['cont']['dd'] - base['cont']['dd']) * 100,
              random_since2019_diff_pp_q10_50_90=[float(np.quantile([(r['since2019'] - base['cont']['since2019']) * 100 for r in rnd['cont']], q)) for q in (0.1, 0.5, 0.9)],
              random_dd_diff_pp_q10_50_90=[float(np.quantile([(r['dd'] - base['cont']['dd']) * 100 for r in rnd['cont']], q)) for q in (0.1, 0.5, 0.9)],
              frozen_pct_since2019=pct(frz['cont']['since2019'], [r['since2019'] for r in rnd['cont']]), frozen_pct_dd=pct(frz['cont']['dd'], [r['dd'] for r in rnd['cont']])))
res['random_control'] = rc
res['meta'].update(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss(), fresh_capital_scale=ratio)
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
tr = pd.concat(trades, ignore_index=True)
tr['pnl'] = tr['pnl'].round(0); tr['ret'] = tr['ret'].round(4)
for c in ('sig_date', 'ent_date', 'exit_date'):
    tr[c] = pd.to_datetime(tr[c]).dt.strftime('%Y-%m-%d')
tr.to_csv(H / 'trades.csv.gz', index=False)
print(json.dumps(pv, ensure_ascii=False)[:2500])
print(json.dumps(rc['summary'], ensure_ascii=False))
print(res['meta'])
