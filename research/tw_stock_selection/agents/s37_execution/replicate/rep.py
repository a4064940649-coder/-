#!/usr/bin/env python3
"""S37 獨立複製主程式（python3 -I -B rep.py）。只依 SPEC.json 文字與 FINDINGS.md 條件撰寫；未讀、未 import s37lib.py／run.py。
引擎：s20lib → s17lib → s10lib（不改），cslib.EXRIGHT_FIX = True；成交價掛鉤在 xengine.simulate_x（fill='open' 必須與 simulate_s10 逐位相同）。
輸出：results_rep.json（本資料夾）。暫存：scratchpad/s37rep/（結束刪除）。"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent.parent
SC = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s37rep/cache')
SC.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(AG / 's20_persistent_growth'))
sys.path.insert(0, str(HERE))
import s20lib                                                  # noqa: E402
from s20lib import s17lib, s10lib, s4lib, stlib, cslib, COSTS   # noqa: E402
import xengine                                                 # noqa: E402
s20lib.S20 = SC; s17lib.S17 = SC; s10lib.S10 = SC
for n in ('s9lib', 's8lib', 's7lib', 's6lib', 's5lib', 's4lib', 's3lib', 's2lib'):
    setattr(getattr(s10lib, n), 'S' + n[1:-3], SC)
s10lib.stlib.S1 = SC; cslib.TMP = SC
cslib.EXRIGHT_FIX = True

T0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
QUICK = '--quick' in sys.argv
P = s20lib.load()
assert P.exright_fix is True
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
F = s17lib.rev_family(P, defs=['STD'])
RT = dict(S=F['S']['STD'], av=F['av']); del F
s17lib.e50_fix_suspension(P)
for a in ('CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
AV, A4, NFB, NFB_T0 = xengine.make_px(P)
print('載入', round(time.time() - T0), 's', rss(), 'MB', flush=True)

lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19 = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01')
CELL = dict(sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'])
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
SH = (0, 1, 2, 3)
SIG = {s: s10lib.variant_S(s10lib.make_signal_s10(P, U, RT, shift=s), 'E0')[0] for s in SH}


def summ(nav, lg, st):
    return dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE),
                n_closed=int((lg['why'] != 'open_end').sum()), blk_limup=int(st['blk_limup']), sell_defer_limdn=int(st['sell_defer_limdn']),
                n_buy=int(st['n_buy']), n_defer_try=int(st['n_defer_try']), n_defer_buy=int(st['n_defer_buy']),
                buy_val=float(st['buy_val']), slip=float(st['slip']))


res = dict(meta=dict(start=str(D0.date()), since2019_from=str(D19.date()), end=str(DE.date()), avgv_fallback_cells_all=NFB, avgv_fallback_cells_from_t0=NFB_T0))

# ---------------------------------------------------------------- 驗收：標準路徑、A1（20 條路徑 simulate_x(open) vs simulate_s10）
a1 = 0.0; base_runs = {}
for s in SH:
    for cap in CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        nav0, lg0, st0 = s10lib.simulate_s10(P, SIG[s], U, capital=cap, cap_pct=0.05, **CELL)
        nav1, lg1, st1 = xengine.simulate_x(P, SIG[s], U, s10lib, fill='open', capital=cap, cap_pct=0.05, **CELL)
        a1 = max(a1, float(np.nanmax(np.abs(nav0.to_numpy() - nav1.to_numpy()))))
        assert len(lg0) == len(lg1) and int(st0['n_buy']) == int(st1['n_buy']) and int(st0['blk_limup']) == int(st1['blk_limup'])
        base_runs[key] = (nav1, lg1, st1)
        del nav0, lg0, st0
    if QUICK:
        break
std = summ(*base_runs['100萬|+0'])
res['acceptance'] = dict(standard=std, A1_max_abs_nav_diff_vs_simulate_s10=a1,
                         match=bool(round(std['since2019'] * 100, 2) == 48.99 and round(std['full'] * 100, 2) == 34.54 and round(std['maxdd'] * 100, 2) == -32.11 and std['n_closed'] == 381))
print('驗收', res['acceptance'], round(time.time() - T0), 's', rss(), 'MB', flush=True)
assert res['acceptance']['match'], '標準路徑不符，停止'
assert a1 < 1e-6, 'A1 失敗'

# ---------------------------------------------------------------- 第 1 部分：描述
O, H, L, C, REF, LU, LD, VOL, G = P.O, P.H, P.L, P.C, P.REF, P.LIMUP, P.LIMDN, P.VOL, P.G
with np.errstate(invalid='ignore', divide='ignore'):
    VM20 = pd.DataFrame(VOL).rolling(20, min_periods=20).mean().shift(1).to_numpy()


def qd(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0)
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return dict(n=int(len(x)), mean=float(x.mean()), p5=float(q[0]), p25=float(q[1]), p50=float(q[2]), p75=float(q[3]), p95=float(q[4]), gt0=float((x > 0).mean()))


def fwd20(t, j, base_open=True):
    """t 日開盤 → t+20 日收盤（含息還原）。"""
    t2 = t + 20
    if t2 >= P.T or not (np.isfinite(O[t, j]) and O[t, j] > 0) or not np.isfinite(G[t2, j]) or not np.isfinite(G[t, j]):
        return np.nan
    return float(G[t2, j] / G[t, j] * C[t, j] / O[t, j] - 1)


def describe(keys, side):
    t = np.array([k[0] for k in keys]); j = np.array([k[1] for k in keys])
    o = O[t, j]; ok = np.isfinite(o) & (o > 0)
    t, j, o = t[ok], j[ok], o[ok]
    out = dict(n=int(len(keys)), n_no_trade=int((~ok).sum()))
    gap = o / REF[t, j] - 1
    out['gap'] = qd(gap)
    for th, nm in ((0.03, 'gap_gt3'), (0.05, 'gap_gt5'), (0.07, 'gap_gt7')):
        out[nm] = float((gap > th).mean())
    out['gap_lt_m3'] = float((gap < -0.03).mean())
    out['avgv_vs_open'] = qd(AV[t, j] / o - 1); out['avg4_vs_open'] = qd(A4[t, j] / o - 1)
    out['high_vs_open'] = qd(H[t, j] / o - 1); out['low_vs_open'] = qd(L[t, j] / o - 1); out['close_vs_open'] = qd(C[t, j] / o - 1)
    vr = VOL[t, j] / VM20[t, j]
    if side == 'B':
        lu = o >= LU[t, j] - 1e-9; one = L[t, j] >= LU[t, j] - 1e-9
        out.update(open_limup=int(lu.sum()), open_limup_rate=float(lu.sum() / len(keys)), one_line_limup=int((lu & one).sum()), open_limup_then_opened=int((lu & ~one).sum()),
                   limup_vol_ratio_median=float(np.nanmedian(vr[lu])) if lu.any() else None,
                   limup_vol_ratio_lt05=float((vr[lu] < 0.5).mean()) if lu.any() else None, all_vol_ratio_median=float(np.nanmedian(vr)))
    else:
        ld = o <= LD[t, j] + 1e-9; one = H[t, j] <= LD[t, j] + 1e-9
        out.update(open_limdn=int(ld.sum()), open_limdn_rate=float(ld.sum() / len(keys)), one_line_limdn=int((ld & one).sum()), open_limdn_then_opened=int((ld & ~one).sum()))
    return out


ent = {}; ex = {}; ent_std = {}; ex_std = {}; defer_days = {}; n_exit_events = 0
for key, (nav, lg, st) in base_runs.items():
    for t, j, r in st['ent_att']:
        ent.setdefault((t, j), r)
        if key == '100萬|+0':
            ent_std.setdefault((t, j), r)
    first = {}
    for t, j, r, ft in st['ex_att']:
        ex.setdefault((t, j), r); n_exit_events += 1
        if key == '100萬|+0':
            ex_std.setdefault((t, j), r)
        if r in ('sold', 'partial') and t > ft:
            defer_days[t - ft] = defer_days.get(t - ft, 0) + 1
p1 = {}
p1['entry_all_paths'] = describe(list(ent), 'B'); p1['entry_standard'] = describe(list(ent_std), 'B')
p1['exit_all_paths'] = describe(list(ex), 'S'); p1['exit_standard'] = describe(list(ex_std), 'S')
p1['exit_defer'] = dict(n_exit_events=n_exit_events, n_deferred=int(sum(defer_days.values())), defer_days={str(k): v for k, v in sorted(defer_days.items())})
p1['entry_result_counts'] = pd.Series(list(ent.values())).value_counts().to_dict()
p1['exit_result_counts'] = pd.Series(list(ex.values())).value_counts().to_dict()
yr = {}
for (t, j) in ent:
    y = P.IDX[t].year; d = yr.setdefault(y, dict(n=0, open_limup=0, gaps=[], av=[]))
    d['n'] += 1
    if np.isfinite(O[t, j]) and O[t, j] > 0:
        d['open_limup'] += int(O[t, j] >= LU[t, j] - 1e-9); d['gaps'].append(O[t, j] / REF[t, j] - 1); d['av'].append(AV[t, j] / O[t, j] - 1)
p1['entry_yearly'] = {str(y): dict(n=d['n'], open_limup=d['open_limup'], gap_median=float(np.median(d['gaps'])), avgv_vs_open_median=float(np.median(d['av']))) for y, d in sorted(yr.items())}

# 限價模擬（買進嘗試，全路徑去重）
LO = {}
keys = [k for k in ent if np.isfinite(O[k]) and O[k] > 0]
nall = len(ent)
for x in (0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 'limup'):
    fills = []; miss_f = []; fill_f = []; q = 0; miss = 0
    for (t, j) in keys:
        lim = LU[t, j] if x == 'limup' else REF[t, j] * (1 + x)
        if lim >= LU[t, j] - 1e-9 and O[t, j] >= LU[t, j] - 1e-9:
            q += 1; continue                                   # 開盤鎖漲停：排隊，視為沒買到
        if O[t, j] <= lim + 1e-9:
            px = O[t, j]
        elif L[t, j] <= lim + 1e-9:
            px = lim
        else:
            miss += 1; miss_f.append(fwd20(t, j)); continue
        fills.append((px / O[t, j] - 1, px / REF[t, j] - 1)); fill_f.append(fwd20(t, j))
    fa = np.array(fills)
    LO[str(x)] = dict(fill_rate=len(fills) / nall, queue_at_limup=q, miss=miss, fill_vs_open_mean=float(fa[:, 0].mean()), fill_vs_ref_mean=float(fa[:, 1].mean()),
                      fwd20_from_open_filled_median=float(np.nanmedian(fill_f)), fwd20_from_open_missed_median=float(np.nanmedian(miss_f)) if miss_f else None)
p1['entry_limit_order_sim'] = LO
fu = []
for (t, j) in ent:
    if np.isfinite(O[t, j]) and O[t, j] >= LU[t, j] - 1e-9:
        n1 = t + 1
        fu.append(dict(date=str(P.IDX[t].date()), code=str(P.COLS[j]), one_line=bool(L[t, j] >= LU[t, j] - 1e-9),
                       next_open_limup=bool(n1 < P.T and O[n1, j] >= LU[n1, j] - 1e-9),
                       next_open_vs_open=float(O[n1, j] / O[t, j] - 1) if n1 < P.T else None,
                       fwd20_from_next_open=fwd20(n1, j) if n1 < P.T else None, fwd20_from_open_hypo=fwd20(t, j)))
p1['entry_limup_followup'] = fu
ed = sorted(set(int(x['k']) + 1 for s in SH for x in SIG[s]['info'] if int(x['k']) + 1 < P.T))
rl = []; gm = []
for t in ed:
    m = U[t - 1] & np.isfinite(O[t]) & (O[t] > 0)
    rl.append(float((O[t, m] >= LU[t, m] - 1e-9).mean())); gm.append(float(np.median(O[t, m] / REF[t, m] - 1)))
p1['universe_exec_days'] = dict(n_days=len(ed), open_limup_rate_mean=float(np.mean(rl)), gap_median_mean=float(np.mean(gm)))
res['part1'] = p1
print('第 1 部分完成', round(time.time() - T0), 's', rss(), 'MB', flush=True)

# ---------------------------------------------------------------- 第 2 部分：成交價敏感度
VAR = dict(BASE=dict(fill='open'), AVGV=dict(fill='avgv'), AVG4=dict(fill='avg4'), OPEN05=dict(fill='open05'), DEFER=dict(fill='open', defer=True))
paths = {v: {} for v in VAR}
for key, (nav, lg, st) in base_runs.items():
    paths['BASE'][key] = summ(nav, lg, st)
del base_runs; gc.collect()
for v, kw in VAR.items():
    if v == 'BASE':
        continue
    for s in SH:
        for cap in CAPS:
            key = f'{int(cap/1e4)}萬|+{s}'
            nav, lg, st = xengine.simulate_x(P, SIG[s], U, s10lib, AV=AV, A4=A4, capital=cap, cap_pct=0.05, **kw, **CELL)
            paths[v][key] = summ(nav, lg, st)
        if QUICK:
            break
    print(v, paths[v]['100萬|+0']['since2019'], round(time.time() - T0), 's', rss(), 'MB', flush=True)
summary = {}
for v in VAR:
    ks = list(paths[v])
    d = dict(standard=paths[v]['100萬|+0'], paths_median={f: float(np.median([paths[v][k][f] for k in ks])) for f in ('since2019', 'full', 'maxdd', 'n_closed', 'blk_limup', 'n_defer_buy')})
    if v != 'BASE':
        d['vs_BASE'] = {}
        for f in ('since2019', 'full', 'maxdd'):
            df = np.array([(paths[v][k][f] - paths['BASE'][k][f]) * 100 for k in ks])
            d['vs_BASE'][f] = dict(standard_pp=float((paths[v]['100萬|+0'][f] - paths['BASE']['100萬|+0'][f]) * 100), median_pp=float(np.median(df)),
                                   min_pp=float(df.min()), max_pp=float(df.max()), n_worse=int((df < 0).sum()))
    summary[v] = d
res['part2'] = dict(summary=summary, paths=paths)

# ---------------------------------------------------------------- 第 3 部分：容量
CAP3 = (1e6, 5e6, 1e7, 3e7); PCT = (0.0, 0.01, 0.02, 0.05)
raw = {}
for capital in CAP3:
    for pct in PCT:
        for s in SH:
            nav, lg, st = xengine.simulate_x(P, SIG[s], U, s10lib, fill='open', capital=capital, cap_pct=pct, **CELL)
            ce = max(st['cap_enter'], 1)
            raw[(capital, pct, s)] = dict(since2019=stlib.cagr(nav, D19, DE), maxdd=stlib.maxdd(nav, D0, DE), capped_rate=st['blk_cap'] / ce,
                                          dropped_rate=st['cap_dropped'] / ce, sell_partial_rate=st['sell_partial_cap'] / max(st['n_sell'], 1),
                                          n_buy=int(st['n_buy']), cap_enter=int(st['cap_enter']),
                                          part=qd(st['part']) if s == 0 else None)
            if QUICK:
                break
    print('容量', capital, round(time.time() - T0), 's', rss(), 'MB', flush=True)
p3 = {}
for capital in CAP3:
    for pct in PCT:
        ss = [s for s in SH if (capital, pct, s) in raw]
        r0 = raw[(capital, pct, 0)]; n0 = raw[(capital, 0.0, 0)]
        med = {f: float(np.median([raw[(capital, pct, s)][f] for s in ss])) for f in ('since2019', 'maxdd', 'capped_rate', 'dropped_rate', 'sell_partial_rate')}
        p3[f'{int(capital/1e4)}萬|{pct:.2f}'] = dict(shift0={k: v for k, v in r0.items() if k != 'part'}, shift0_vs_nocap_pp=(r0['since2019'] - n0['since2019']) * 100,
                                                   median4=med, median4_vs_nocap_pp=float(np.median([(raw[(capital, pct, s)]['since2019'] - raw[(capital, 0.0, s)]['since2019']) * 100 for s in ss])),
                                                   participation_shift0=r0['part'])
res['part3'] = dict(summary=p3)
res['meta']['runtime_s'] = round(time.time() - T0, 1); res['meta']['maxrss_mb'] = rss()
out = HERE / ('results_rep_quick.json' if QUICK else 'results_rep.json')
out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: int(o) if isinstance(o, np.integer) else float(o)))
print('完成', out, res['meta'], flush=True)
