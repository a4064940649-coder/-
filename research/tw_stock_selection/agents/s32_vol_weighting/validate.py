#!/usr/bin/env python3
"""S32 驗證期（凍結後只跑一次）：凍結規則 FZ、現行 CUR、對照 HV3_CASH（CTL）。由 S21 validate.py 改寫（規則套用換成 s32lib.build_wmul＋simulate_w；判定改 PREREG 的 V1～V4）。
(1) 判定用：重新起跑（只保留營收月份 ≥ 2021-12 的訊號；起始資金縮放使 2021-12-30 淨值約＝60～200 萬）× 20 條路徑，驗證窗 2021-12-30 → 2026-10-22。
(2) 描述用：全期連續模擬（主跑＋20 條路徑）：逐年、最大回撤、交易、禁買最賺 5 筆、成本加倍、四段空頭、八項檢查、CAPM β、單檔最大權重、有效持股數。
輸出 results.json、trades.csv.gz。PREREG／FROZEN sha256 必須相符；results.json 已存在就拒絕重跑。python3 -I -B validate.py [--dry]"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s32lib
from s32lib import s17lib, s10lib, s4lib, s2lib, stlib, COSTS

H = Path(__file__).resolve().parent
pre = (H / 'PREREG.json').read_bytes(); sha = hashlib.sha256(pre).hexdigest()
assert sha == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與 PREREG.sha256 不符'
fz = (H / 'FROZEN.json').read_bytes(); fsha = hashlib.sha256(fz).hexdigest()
assert fsha == (H / 'FROZEN.sha256').read_text().split()[0], 'FROZEN.json 與 FROZEN.sha256 不符'
DRY = '--dry' in sys.argv                                      # 乾跑：規則換成等額（m≡1），只測程式路徑，輸出到 scratchpad
assert DRY or not (H / 'results.json').exists(), 'results.json 已存在：驗證期只跑一次，拒絕重跑'
FZ = json.loads(fz)
RULE = FZ['frozen_rule']; CTLR = FZ['control']
if DRY:
    RULE = None; CTLR = None
OUT = s32lib.S32 / 'dry' if DRY else H
OUT.mkdir(parents=True, exist_ok=True)
CFG = {'CUR': None, 'FZ': RULE, 'CTL': CTLR}
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s32lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
RT = s32lib.surge(P)
s17lib.e50_fix_suspension(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
print('載入完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE = P.IDX[P.t0], P.IDX[-1]
DM = P.IDX[(P.t0 + P.T - 1) // 2]
D19, D24, TB = lastb('2019-01-01'), lastb('2024-01-01'), lastb('2022-01-01')
iTB = int(np.searchsorted(P.IDX.values, TB.to_datetime64()))
VM = P.IDX[(iTB + P.T - 1) // 2]
WIN = {'since2019': (D19, DE), 'full': (D0, DE), 'since2024': (D24, DE), 'half1': (D0, DM), 'half2': (DM, DE), 'y2019_2023': (D19, D24),
       'valid': (TB, DE), 'valid_h1': (TB, VM), 'valid_h2': (VM, DE), 'y2022_2023': (TB, D24)}
W = lambda nav, ks=WIN: {k: stlib.window(nav, bench, *WIN[k]) for k in ks}
cg = lambda nav, k: stlib.cagr(nav, *WIN[k])
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
SEGV = ('2022', '2024-07~2025-04')
VC = {}


def variant(S, c, rule):
    key = (id(S), c)
    if key not in VC:
        VC[key] = s32lib.build_wmul(P, S, rule)
    return VC[key]


def sim(S, c, rule, cap=1e6, cost='strict', fresh=False, block=None):
    S1, _ = variant(S, c, rule)
    if fresh:
        S1 = s32lib.fresh_from(S1, s32lib.VALID_FIRST_MONTH)
    S2, ekw = s10lib.variant_S(S1, 'E0')
    return s32lib.simulate_w(P, S2, U, cost=COSTS[cost], capital=cap, block=block, **CELL, **ekw)


def slim(w):
    return {k: dict(cagr=v['cagr'], cagr_0050=v['cagr_0050'], diff=v['diff'], maxdd=v['maxdd']) for k, v in w.items()}


def divers(st, a, b):
    mw = st['maxw'].loc[a:b]; hh = st['hhi'].loc[a:b]; npd = st['nposd'].loc[a:b]
    return dict(maxw_max=float(mw.max()), maxw_mean=float(mw[npd > 0].mean()), effn_mean=float((1.0 / hh[npd > 0]).mean()), npos_mean=float(npd.mean()))


def tstats(lg, a=None):
    o = stlib.trade_stats(lg.copy(), a)
    o['n_open_end'] = int(((lg['why'] == 'open_end') & (lg['ent_date'] >= pd.Timestamp(a or '1900-01-01'))).sum())
    return o


# ---------------------------------------------------------------- 驗收 A1、起始資金縮放、A3
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
S2, ekw = s10lib.variant_S(S0, 'E0')
navc = s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=1e6, **CELL, **ekw)[0]
acc = dict(A1=bool(np.array_equal(navc.to_numpy(), sim(S0, 'CUR', None)[0].to_numpy())))
nav1, lg1, _ = sim(S0, 'CUR', None, 1e6, fresh=True)
ratio = float(nav1.loc[TB] / 1e6)
acc.update(A3_no_trade_before_2022=bool((lg1['ent_date'] >= pd.Timestamp('2022-01-01')).all()), ratio_nav_TB=ratio, first_entry=str(lg1['ent_date'].min().date()))
assert acc['A1'] and acc['A3_no_trade_before_2022'], acc
del navc, nav1, lg1
FCAP = {cap: cap / ratio for cap in CAPS}

# ---------------------------------------------------------------- 主跑（位移 0）
res = {'PREREG_sha256': sha, 'FROZEN_sha256': fsha, 'frozen_rule': FZ['frozen_rule'], 'T0_pass': FZ['T0_pass'], 'acceptance': acc,
       'meta': dict(start=str(D0.date()), mid=str(DM.date()), valid_from=str(TB.date()), valid_mid=str(VM.date()), end=str(DE.date())),
       'bench_0050': {k: dict(cagr=stlib.cagr(bench, *WIN[k]), maxdd=stlib.maxdd(bench, *WIN[k])) for k in WIN}, 'main': {}}
res['bench_0050']['seg'] = s17lib.seg_dd(bench)
trades = []; mainlg = {}
for mode in ('fresh', 'cont'):
    fresh = mode == 'fresh'
    cap = FCAP[1e6] if fresh else 1e6
    for c, rule in CFG.items():
        nav, lg, st = sim(S0, c, rule, cap, fresh=fresh)
        nm = f'{c}|{mode}'
        mainlg[nm] = lg
        a = '2022-01-01' if fresh else None
        ks = ('valid', 'valid_h1', 'valid_h2', 'y2022_2023', 'since2024') if fresh else tuple(WIN)
        w = W(nav, ks)
        lga = lg[lg['ent_date'] >= pd.Timestamp(a or '1900-01-01')]
        blk, top = s2lib.block_from(lga, 5)
        nb = sim(S0, c, rule, cap, fresh=fresh, block=blk)[0]
        nx = sim(S0, c, rule, cap, cost='strict_x2', fresh=fresh)[0]
        kk = ('valid',) if fresh else ('since2019', 'full', 'valid')
        b0 = TB if fresh else D0
        wl = variant(S0, c, rule)[1]
        mm = [v for x in wl if (not fresh or pd.Period(x['month'], 'M') >= s32lib.VALID_FIRST_MONTH) for v in x['m']]
        R = dict(windows=slim(w), maxdd=stlib.maxdd(nav, b0, DE), calmar=s32lib.calmar(nav, b0, DE), dd_dates=s10lib.dd_dates(nav.loc[b0:]), seg=s17lib.seg_dd(nav),
                 yearly=stlib.yearly(nav.loc[b0:], bench), trades=tstats(lg, a),
                 block5=slim(W(nb, kk)), block5_top=[dict(code=y.code, ent=str(y.ent_date.date()), exit=str(y.exit_date.date()), ret=float(y.ret), pnl=float(y.pnl)) for y in top.itertuples()],
                 costx2=slim(W(nx, kk)), top5_share=s17lib.top5_share(lga, nav.loc[b0:], float(nav.loc[b0])),
                 capm=s32lib.capm_beta(nav, bench, b0, DE), capm_valid=s32lib.capm_beta(nav, bench, TB, DE), divers=divers(st, b0, DE),
                 target_mult=dict(min=float(min(mm)) if mm else 1.0, max=float(max(mm)) if mm else 1.0, n=len(mm)),
                 n_buy=int(st['n_buy']), avg_stockfrac=float(np.nanmean(st['stockfrac'].loc[b0:].to_numpy())),
                 blk_min=int(st['blk_min']), shrunk=int(st['shrunk']), skip_cash=int(st['skip_cash']))
        if not fresh:
            chk = dict(net_since2019=w['since2019']['diff'] > 0, net_full=w['full']['diff'] > 0, costx2_since2019=R['costx2']['since2019']['diff'] > 0,
                       costx2_full=R['costx2']['full']['diff'] > 0, block5_since2019=R['block5']['since2019']['diff'] > 0, block5_full=R['block5']['full']['diff'] > 0,
                       half1=w['half1']['diff'] > 0, half2=w['half2']['diff'] > 0)
            R['checks8'] = {k: bool(v) for k, v in chk.items()}; R['checks8_pass'] = int(sum(chk.values()))
            # 最賺 5 筆的進場倍數
            x = lg[lg['why'] != 'open_end'].sort_values('pnl', ascending=False).head(5)
            WM = variant(S0, c, rule)[0]['wmul']
            R['top5_trades'] = [dict(code=y.code, ent=str(y.ent_date.date()), ret=float(y.ret), pnl=float(y.pnl), mult=float(WM.get((int(y.sig_t), int(y.j)), 1.0))) for y in x.itertuples()]
        res['main'][nm] = R
        x = lg.copy(); x['run'] = nm
        trades.append(x[['run', 'code', 'sig_date', 'ent_date', 'exit_date', 'hold_days', 'cost', 'pnl', 'ret', 'why']])
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
PATH = {f'{c}|{m}': {} for c in CFG for m in ('fresh', 'cont')}
PATH.update({'CUR|fresh_blk': {}, 'FZ|fresh_blk': {}})
for s in range(4):
    S = S0 if s == 0 else s10lib.make_signal_s10(P, U, RT, shift=s)
    for cap in CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        for c, rule in CFG.items():
            nav, lg, st = sim(S, c, rule, FCAP[cap], fresh=True)
            sg = s17lib.seg_dd(nav)
            PATH[f'{c}|fresh'][key] = dict(valid=cg(nav, 'valid'), valid_dd=stlib.maxdd(nav, TB, DE), calmar=s32lib.calmar(nav, TB, DE),
                                           y2022_2023=cg(nav, 'y2022_2023'), since2024=cg(nav, 'since2024'), seg={k: sg[k] for k in SEGV},
                                           top5=s17lib.top5_share(lg, nav, FCAP[cap]), beta=s32lib.capm_beta(nav, bench, TB, DE)['beta'], **divers(st, TB, DE))
            nav, lg, st = sim(S, c, rule, cap)
            PATH[f'{c}|cont'][key] = dict(since2019=cg(nav, 'since2019'), full=cg(nav, 'full'), valid=cg(nav, 'valid'), y2019_2023=cg(nav, 'y2019_2023'), since2024=cg(nav, 'since2024'),
                                          maxdd=stlib.maxdd(nav, D0, DE), calmar=s32lib.calmar(nav, D0, DE), seg=s17lib.seg_dd(nav), top5=s17lib.top5_share(lg, nav, cap),
                                          beta=s32lib.capm_beta(nav, bench, D0, DE)['beta'], **divers(st, D0, DE))
            if c in ('CUR', 'FZ'):
                nav = sim(S, c, rule, FCAP[cap], fresh=True, block=blkU)[0]
                PATH[f'{c}|fresh_blk'][key] = dict(valid=cg(nav, 'valid'))
            del nav, lg, st
    VC.clear()
    if s:
        del S
    gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
res['paths'] = PATH
keys = list(PATH['CUR|fresh'])
med = lambda v: float(np.median(v))


def pair(a, b, f, sub=None, scale=100.0):
    g = (lambda d: d[f]) if sub is None else (lambda d: d[f][sub])
    d = np.array([g(PATH[a][k]) - g(PATH[b][k]) for k in keys]) * scale
    return dict(npos=int((d > 0).sum()), median=med(d), min=float(d.min()), max=float(d.max()))


summ = {}
for nm, D in PATH.items():
    summ[nm] = {f: med([D[k][f] for k in keys]) for f in D[keys[0]] if f != 'seg'}
    if 'seg' in D[keys[0]]:
        summ[nm]['seg'] = {sg: med([D[k]['seg'][sg] for k in keys]) for sg in D[keys[0]]['seg']}
res['paths_median'] = summ
pv = {}
for c in ('FZ', 'CTL'):
    a, b = f'{c}|fresh', 'CUR|fresh'
    dc = np.array([PATH[a][k]['calmar'] - PATH[b][k]['calmar'] for k in keys])
    dd = np.array([PATH[a][k]['valid_dd'] - PATH[b][k]['valid_dd'] for k in keys]) * 100
    pv[f'{c}_vs_CUR|fresh'] = dict(valid=pair(a, b, 'valid'), valid_dd=pair(a, b, 'valid_dd'), calmar=pair(a, b, 'calmar', scale=1.0),
                                   n_imp=int(((dc >= 0.05) | (dd >= 0.5)).sum()), n_calmar_imp=int((dc >= 0.05).sum()), n_dd_imp=int((dd >= 0.5).sum()),
                                   y2022_2023=pair(a, b, 'y2022_2023'), since2024=pair(a, b, 'since2024'), top5=pair(a, b, 'top5'), beta=pair(a, b, 'beta', scale=1.0),
                                   maxw_max=pair(a, b, 'maxw_max'), effn_mean=pair(a, b, 'effn_mean', scale=1.0),
                                   seg={sg: pair(a, b, 'seg', sg) for sg in SEGV})
    a = f'{c}|cont'; b = 'CUR|cont'
    pv[f'{c}_vs_CUR|cont'] = {f: pair(a, b, f) for f in ('since2019', 'full', 'valid', 'y2019_2023', 'since2024', 'maxdd', 'top5', 'maxw_max')}
    pv[f'{c}_vs_CUR|cont'].update(calmar=pair(a, b, 'calmar', scale=1.0), beta=pair(a, b, 'beta', scale=1.0), effn_mean=pair(a, b, 'effn_mean', scale=1.0))
    pv[f'{c}_vs_CUR|cont']['seg'] = {sg: pair(a, b, 'seg', sg) for sg in s17lib.SEGMENTS}
pv['FZ_vs_CUR|fresh_blk'] = dict(valid=pair('FZ|fresh_blk', 'CUR|fresh_blk', 'valid'))
res['pairs'] = pv

p = pv['FZ_vs_CUR|fresh']
V = dict(V1=p['n_imp'] >= 15, V2=p['valid']['median'] >= -1.5, V3=pv['FZ_vs_CUR|fresh_blk']['valid']['median'] >= -1.5,
         V4=all(p['seg'][sg]['median'] > 0 for sg in SEGV))
V = {k: bool(v) for k, v in V.items()}
passed = bool(FZ['T0_pass'] and all(V.values()))
res['verdict'] = dict(T0=FZ['T0_pass'], **V, n_pass_V=int(sum(V.values())), overall='通過' if passed else '不通過',
                      text=('凍結規則在驗證期有用' if passed else '不通過：停在描述，不換規則、不換上下限、不寫 XQ 修改建議'))
res['meta'].update(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss(), fresh_capital_scale=ratio, dry=DRY)
(OUT / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
tr = pd.concat(trades, ignore_index=True)
tr['pnl'] = tr['pnl'].round(0); tr['cost'] = tr['cost'].round(0); tr['ret'] = tr['ret'].round(4)
for c in ('sig_date', 'ent_date', 'exit_date'):
    tr[c] = pd.to_datetime(tr[c]).dt.strftime('%Y-%m-%d')
tr.to_csv(OUT / 'trades.csv.gz', index=False)
print('判定', json.dumps(res['verdict'], ensure_ascii=False))
print(json.dumps(pv['FZ_vs_CUR|fresh'], ensure_ascii=False))
print(res['meta'])
