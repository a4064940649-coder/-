#!/usr/bin/env python3
"""S27 獨立複製（第 2 步）：從 rep_engine.py 產生的日淨值（暫存區 navs_daily.npz）重算 part1～part3 與 site table 主要數字。
只依 SPEC.json 文字與 FINDINGS.md 條件自行實作；未讀、未 import s27 的 build.py／analyze.py。
種子：numpy default_rng(777_2027)（刻意與原研究不同；三情境共用同一組抽樣索引）。
用法：python3 -I -B rep_analyze.py → replicate/results_rep.json"""
import json, time, resource
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
TMP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s27rep')
t0 = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
SEED = 7772027
NSIM = 10000

z = np.load(TMP / 'navs_daily.npz')
IDX = pd.DatetimeIndex(z['IDX']); KEYS = [str(k) for k in z['keys']]
DN = pd.DataFrame(z['NAV'], index=IDX, columns=KEYS); DB = pd.Series(z['E50'], index=IDX)
MN = DN.groupby(IDX.to_period('M')).last(); MB = DB.groupby(IDX.to_period('M')).last()
STD = '100萬|+0'


def pct(a, q):
    return float(np.percentile(a, q))


def r4(x):
    return None if x is None else round(float(x), 4)


# ---------------------------------------------------------------- part 1
def lag_stats(W):
    """W：月底相對財富（第 0 個＝基準月，視為高點）。lag_t＝t − 最近一次創新高（≥ 前高）的月序。"""
    W = np.asarray(W, float); h = 0; hi = W[0]; lag = np.zeros(len(W), int); spells = []; cur = 0; cur_start = 0
    for t in range(1, len(W)):
        if W[t] >= hi:
            if cur > 0:
                spells.append((cur, cur_start))
            hi = W[t]; h = t; cur = 0
        else:
            lag[t] = t - h; cur = t - h; cur_start = h
    if cur > 0:
        spells.append((cur, cur_start))
    return lag, spells


def dd_episodes(x, idx):
    x = np.asarray(x, float); eps = []; pk = 0; i = 1; n = len(x)
    while i < n:
        if x[i] >= x[pk]:
            pk = i; i += 1; continue
        j = i
        while j < n and x[j] < x[pk]:
            j += 1
        seg = x[pk:j]; tr = pk + int(np.argmin(seg)); rec = j if j < n else None
        end = rec if rec is not None else n - 1
        eps.append(dict(peak=str(idx[pk].date()), trough=str(idx[tr].date()), recover=None if rec is None else str(idx[rec].date()),
                        depth=float(x[tr] / x[pk] - 1), trough_to_recover_m=(end - tr) / 21, underwater_m=(end - pk) / 21, unrecovered=rec is None))
        if rec is None:
            break
        pk = j; i = j + 1
    return eps


def mdd(x):
    x = np.asarray(x, float)
    return float((x / np.maximum.accumulate(x) - 1).min())


def path_hist(nav_m, b_m, nav_d, months):
    s = nav_m.to_numpy(); b = b_m.to_numpy()
    r12 = (s[12:] / s[:-12]) - (b[12:] / b[:-12])
    r36 = (s[36:] / s[:-36]) ** (1 / 3) - (b[36:] / b[:-36]) ** (1 / 3)
    out = {}
    for nm, v, off in (('R12', r12, 12), ('R36', r36, 36)):
        out[nm] = dict(min=float(v.min()), p5=pct(v, 5), p10=pct(v, 10), p25=pct(v, 25), p50=pct(v, 50), p75=pct(v, 75),
                       frac_neg=float((v < 0).mean()), n_windows=int(len(v)), min_end=str(months[off + int(np.argmin(v))]))
    neg = r12 < 0; best = cur = 0
    for f in neg:
        cur = cur + 1 if f else 0; best = max(best, cur)
    out['R12_longest_neg_run_m'] = best
    W = s / b; lag, spells = lag_stats(W)
    lmax = max(spells, key=lambda q: q[0]) if spells else (0, 0)
    out.update(lag_longest_m=int(lmax[0]), lag_longest_start=str(months[lmax[1]]), lag_current_m=int(lag[-1]),
               lag_spells_ge12=int(sum(1 for q in spells if q[0] >= 12)), rel_maxdd=mdd(W),
               maxdd_daily=mdd(nav_d.to_numpy()), maxdd_monthly=mdd(s))
    eps = dd_episodes(nav_d.to_numpy(), nav_d.index)
    e10 = [e for e in eps if e['depth'] <= -0.10]; e20 = [e for e in eps if e['depth'] <= -0.20]
    out.update(n_dd_ge10=len(e10), n_dd_ge20=len(e20))
    for tag, E in (('ge10', e10), ('ge20', e20)):
        rc = [e['trough_to_recover_m'] for e in E if not e['unrecovered']] or [np.nan]
        uw = [e['underwater_m'] for e in E] or [np.nan]
        out.update({f'recover_{tag}_median_m': float(np.median(rc)), f'recover_{tag}_max_m': float(np.max(rc)),
                    f'underwater_{tag}_median_m': float(np.median(uw)), f'underwater_{tag}_max_m': float(np.max(uw))})
    out['longest_underwater_m'] = float(max(e['underwater_m'] for e in eps))
    out['dd_episodes_ge10'] = [{k: (r4(v) if isinstance(v, float) else v) for k, v in e.items()} for e in e10]
    # 歷史警戒線檢查（A：36 個月窗起、B：12 個月窗起、C／D／E 全期）
    dd_m = s / np.maximum.accumulate(s) - 1; rel = W / np.maximum.accumulate(W) - 1
    out['hist_series'] = dict(A=(months[36:], r36), B=(months[12:], r12), C=(months[1:], dd_m[1:]), D=(months[1:], lag[1:]), E=(months[1:], rel[1:]))
    return out


WORSE_HIGH = ('frac_neg', 'longest', 'lag_', 'n_dd', 'recover', 'underwater')


def summarize(per):
    keys = ['R12.min', 'R12.p5', 'R12.p10', 'R12.p25', 'R12.p50', 'R36.min', 'R36.p5', 'R36.p10', 'R36.p25', 'R36.p50', 'rel_maxdd', 'maxdd_daily',
            'maxdd_monthly', 'R12.frac_neg', 'R36.frac_neg', 'R12_longest_neg_run_m', 'lag_longest_m', 'lag_spells_ge12', 'n_dd_ge10', 'n_dd_ge20',
            'recover_ge10_max_m', 'recover_ge20_max_m', 'underwater_ge10_max_m', 'underwater_ge20_max_m', 'longest_underwater_m',
            'recover_ge20_median_m', 'underwater_ge20_median_m']
    out = {}
    for k in keys:
        v = np.array([(p[k.split('.')[0]][k.split('.')[1]] if '.' in k else p[k]) for p in per.values()], float)
        hi_bad = any(w in k for w in WORSE_HIGH)
        out[k] = dict(median=r4(np.median(v)), worst=r4(v.max() if hi_bad else v.min()), best=r4(v.min() if hi_bad else v.max()))
    return out


def part1(base):
    m0 = pd.Period(base, 'M'); months = [m for m in MN.index if m >= m0]
    d0 = IDX[IDX.to_period('M') == m0][-1]
    per = {k: path_hist(MN.loc[months, k], MB.loc[months], DN[k].loc[d0:], months) for k in KEYS}
    dmm = np.array([per[k]['maxdd_daily'] - per[k]['maxdd_monthly'] for k in KEYS])
    b_eps = dd_episodes(DB.loc[d0:].to_numpy(), DB.loc[d0:].index)
    b20 = [e for e in b_eps if e['depth'] <= -0.20]
    bench = dict(maxdd_daily=r4(mdd(DB.loc[d0:])), maxdd_monthly=r4(mdd(MB.loc[months])), n_dd_ge10=sum(e['depth'] <= -0.1 for e in b_eps),
                 n_dd_ge20=len(b20), longest_underwater_m=r4(max(e['underwater_m'] for e in b_eps)),
                 recover_ge20_max_m=r4(max(e['trough_to_recover_m'] for e in b20)), underwater_ge20_max_m=r4(max(e['underwater_m'] for e in b20)))
    return per, months, dict(base_month=base, n_months=len(months) - 1, summary_20paths=summarize(per),
                             daily_minus_monthly_maxdd=dict(median=r4(np.median(dmm)), worst=r4(dmm.min())), bench_0050=bench)


per16, months16, P1_16 = part1('2016-01')
per19, months19, P1_19 = part1('2018-12')
HIST_WORST_M = min(per16[k]['maxdd_monthly'] for k in KEYS)
HIST_WORST_D = min(per16[k]['maxdd_daily'] for k in KEYS)
print('part1 done', f'{time.time()-t0:.0f}s', flush=True)


# ---------------------------------------------------------------- part 2／3 模擬
def returns(months):
    s = MN.loc[months].to_numpy(); b = MB.loc[months].to_numpy()
    return s[1:] / s[:-1] - 1, b[1:] / b[:-1] - 1          # (T,20), (T,)


def draw_idx(rng, T, H, n, blk):
    out = np.empty((n, H), np.int64)
    for i in range(n):
        pos = 0
        while pos < H:
            L = int(rng.integers(6, 13)) if blk is None else blk
            st = int(rng.integers(0, T)); L2 = min(L, H - pos)
            out[i, pos:pos + L2] = (st + np.arange(L2)) % T; pos += L2
    return out


def maxdd_paths(V):                     # V：(n, H+1) 含起點 1
    return (V / np.maximum.accumulate(V, axis=1) - 1).min(axis=1)


def lag_paths(W):                       # W：(n, H+1)，回傳每個時點 lag（t − 最近創新高 t）
    n, H1 = W.shape; hi = np.maximum.accumulate(W, axis=1)
    newhi = W >= hi
    t = np.arange(H1)[None, :]
    last = np.maximum.accumulate(np.where(newhi, t, 0), axis=1)
    return t - last


def qd(a):
    return {f'p{q}': r4(np.percentile(a, q)) for q in (5, 10, 25, 50, 75, 90, 95)}


GRID = dict(A=np.round(np.arange(-30, 6, 1) / 100, 4), B=np.round(np.arange(-80, 1, 2) / 100, 4),
            C=np.round(np.arange(0, 41, 1) / 100, 4), D=np.arange(6, 97), E=np.round(np.arange(-80, 1, 1) / 100, 4))


def simulate(months, blk, hist_worst_m, alerts=True):
    rs, rb = returns(months); T = len(rb)
    dbar = (rs - rb[:, None]).mean(axis=0)
    rng = np.random.default_rng(SEED + (0 if blk is None else blk) + (1000 if months[0].year >= 2018 else 0))
    H = 96 if alerts else 60
    pth = rng.integers(0, 20, NSIM); I = draw_idx(rng, T, H, NSIM, blk)
    B = rb[I]; Bv = np.concatenate([np.ones((NSIM, 1)), np.cumprod(1 + B, axis=1)], axis=1)
    res = dict(dbar_monthly_by_path=dict(median=r4(np.median(dbar)), min=r4(dbar.min()), max=r4(dbar.max())),
               dbar_annual_simple_median=r4(np.median(dbar) * 12), T=T, scenarios={})
    trig_store = {}
    for sc, f in (('EFF', 0.0), ('HALF', 0.5), ('ZERO', 1.0)):
        S = rs[I, pth[:, None]] - f * dbar[pth][:, None]
        Sv = np.concatenate([np.ones((NSIM, 1)), np.cumprod(1 + S, axis=1)], axis=1)
        W = Sv / Bv; LAG = lag_paths(W[:, :61])
        R = {}
        for h in (12, 36, 60):
            cs = Sv[:, h] - 1; cb = Bv[:, h] - 1
            ann = (1 + cs) ** (12 / h) - 1; annb = (1 + cb) ** (12 / h) - 1
            md = maxdd_paths(Sv[:, :h + 1]); lg = LAG[:, :h + 1].max(axis=1)
            R[f'{h}m'] = dict(strat_cum=qd(cs), strat_ann=qd(ann), b0050_ann=qd(annb), excess_ann=qd(ann - annb),
                             p_lose_to_0050=r4((cs < cb).mean()), p_strat_loss=r4((cs < 0).mean()), p_0050_loss=r4((cb < 0).mean()),
                             maxdd_p50=r4(np.percentile(md, 50)), maxdd_p90=r4(np.percentile(md, 10)), maxdd_p95=r4(np.percentile(md, 5)),
                             p_dd_worse_than_hist=r4((md < hist_worst_m).mean()), p_lag_ge12=r4((lg >= 12).mean()), p_lag_ge24=r4((lg >= 24).mean()),
                             lag_p50=float(np.percentile(lg, 50)), lag_p90=float(np.percentile(lg, 90)))
        r12w = Sv[:, 12:61] / Sv[:, :49] - Bv[:, 12:61] / Bv[:, :49]
        r36w = Sv[:, 36:61] / Sv[:, :25] < Bv[:, 36:61] / Bv[:, :25]
        md60 = maxdd_paths(Sv[:, :61]); lg60 = LAG.max(axis=1)
        R['events_5y'] = dict(any_12m_behind=r4((r12w < 0).any(axis=1).mean()), any_36m_behind=r4(r36w.any(axis=1).mean()),
                              lag_ge12=r4((lg60 >= 12).mean()), lag_ge24=r4((lg60 >= 24).mean()), dd_ge30=r4((md60 <= -0.30).mean()),
                              dd_ge40=r4((md60 <= -0.40).mean()), r12_le_m20=r4((r12w <= -0.20).any(axis=1).mean()),
                              year1_behind=R['12m']['p_lose_to_0050'])
        res['scenarios'][sc] = R
        if alerts:   # 檢查月 37～96（1 起算）＝ Sv 欄 37～96
            t = np.arange(37, 97)
            A = (Sv[:, t] / Sv[:, t - 36]) ** (1 / 3) - (Bv[:, t] / Bv[:, t - 36]) ** (1 / 3)
            Bm = Sv[:, t] / Sv[:, t - 12] - Bv[:, t] / Bv[:, t - 12]
            C = (Sv / np.maximum.accumulate(Sv, axis=1) - 1)[:, t]
            D = lag_paths(W)[:, t]
            E = (W / np.maximum.accumulate(W, axis=1) - 1)[:, t]
            trig_store[sc] = dict(A=A, B=Bm, C=C, D=D, E=E)
        del S, Sv, W
    if alerts:
        res['alerts'] = alert_eval(trig_store, hist_worst_m)
    return res


def fire(v, k, thr):
    if k in ('A', 'B', 'C', 'E'):
        return v < thr
    return v >= thr


HOR = {'1y': 12, '3y': 36, '5y': 60}


def alert_eval(TS, hw):
    eff = TS['EFF']; thr = {}
    for fa_name, fa in (('fa10', 0.10), ('fa5', 0.05)):
        th = {}
        for k in 'ABCDE':
            v = eff[k]
            if k in ('A', 'B', 'E'):
                ext = v.min(axis=1); ok = [g for g in GRID[k] if (ext < g).mean() <= fa]; th[k] = float(max(ok))
            elif k == 'C':
                ext = v.min(axis=1); ok = [round(hw - g, 4) for g in GRID['C'] if (ext < hw - g).mean() <= fa]; th[k] = float(max(ok))
            else:
                ext = v.max(axis=1); ok = [int(g) for g in GRID['D'] if (ext >= g).mean() <= fa]; th[k] = int(min(ok))
        thr[fa_name] = th
    out = {}
    for fa_name, th in thr.items():
        o = {}
        F = {sc: {k: fire(TS[sc][k], k, th[k]) for k in 'ABCDE'} for sc in TS}
        for k in 'ABCDE':
            o[k] = dict(threshold=th[k], prob={sc: {h: r4(F[sc][k][:, :n].any(axis=1).mean()) for h, n in HOR.items()} for sc in TS})
            o[k]['zero_over_eff_5y'] = r4(o[k]['prob']['ZERO']['5y'] / max(o[k]['prob']['EFF']['5y'], 1e-9))
        combos = dict(ANY=lambda f: f['A'] | f['B'] | f['C'] | f['D'] | f['E'], A_and_D=lambda f: f['A'] & f['D'],
                      A_and_E_posthoc=lambda f: f['A'] & f['E'], TWO_OR_MORE_posthoc=lambda f: (sum(f[k].astype(int) for k in 'ABCDE') >= 2))
        for nm, fn in combos.items():
            o[nm] = dict(prob={sc: {h: r4(fn(F[sc])[:, :n].any(axis=1).mean()) for h, n in HOR.items()} for sc in TS})
        out[fa_name] = o
    return out


P2 = {}
P2['main_2016_b6to12'] = simulate(months16, None, HIST_WORST_M); print('main done', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
P2['sens_2016_b6'] = simulate(months16, 6, HIST_WORST_M)
P2['sens_2016_b12'] = simulate(months16, 12, HIST_WORST_M)
hw19 = min(per19[k]['maxdd_monthly'] for k in KEYS)
P2['sens_2019_b6to12'] = simulate(months19, None, hw19)
print('sims done', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

# ---------------------------------------------------------------- part 3 歷史檢查
TH = P2['main_2016_b6to12']['alerts']['fa10']
th10 = {k: TH[k]['threshold'] for k in 'ABCDE'}
hist_pp = {}
for key in KEYS:
    hs = per16[key]['hist_series']; d = {}
    for k in 'ABCDE':
        ms, v = hs[k]; f = fire(np.asarray(v), k, th10[k])
        d[k] = dict(n_months=int(f.sum()), months=[str(ms[i]) for i in np.flatnonzero(f)],
                    worst=r4(np.max(v) if k == 'D' else np.min(v)), current_2026_10=r4(v[-1]) if k != 'D' else int(v[-1]))
    hist_pp[key] = d
hist_sum = {}
for k in 'ABCDE':
    cur = np.array([hist_pp[p][k]['current_2026_10'] for p in KEYS], float); wv = np.array([hist_pp[p][k]['worst'] for p in KEYS], float)
    hist_sum[k] = dict(paths_triggered=int(sum(hist_pp[p][k]['n_months'] > 0 for p in KEYS)), current_median=r4(np.median(cur)),
                       current_worst=r4(cur.max() if k == 'D' else cur.min()), worst_value_median=r4(np.median(wv)),
                       worst_value_worst=r4(wv.max() if k == 'D' else wv.min()))
# 2026-10 標準路徑現值＋C 的日內口徑（日淨值對前高）
nd = DN[STD].loc[IDX[IDX.to_period('M') == pd.Period('2016-01', 'M')][-1]:]
cur_daily_dd = float(nd.iloc[-1] / nd.cummax().iloc[-1] - 1)

for per in (per16, per19):
    for p in per.values():
        p.pop('hist_series', None)
P1_16['std_path'] = per16[STD]; P1_19['std_path'] = per19[STD]
for P1, per in ((P1_16, per16), (P1_19, per19)):
    P1['per_path'] = {k: {kk: vv for kk, vv in v.items() if kk != 'dd_episodes_ge10'} for k, v in per.items()}


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    return o


out = dict(seed=SEED, n_sims=NSIM, note='獨立複製；未讀 build.py／analyze.py；種子與原研究不同',
           engine_check=json.load(open(HERE / 'engine_check.json')),
           part1_history=dict(since2016=P1_16, since2019=P1_19), part2_simulation=P2,
           part3_alerts=dict(thresholds_fa10=th10, thresholds_fa5={k: P2['main_2016_b6to12']['alerts']['fa5'][k]['threshold'] for k in 'ABCDE'},
                             hist_worst_mdd_monthly=r4(HIST_WORST_M), hist_worst_mdd_daily=r4(HIST_WORST_D),
                             history_check_summary=hist_sum, history_check_std=hist_pp[STD], history_check_per_path=hist_pp,
                             std_current_daily_dd_2026_10=r4(cur_daily_dd),
                             uninformative_events_5y={sc: P2['main_2016_b6to12']['scenarios'][sc]['events_5y'] for sc in ('EFF', 'HALF', 'ZERO')}),
           peak_mb=rss(), secs=round(time.time() - t0))
json.dump(clean(out), open(HERE / 'results_rep.json', 'w'), ensure_ascii=False, indent=1)
print('done', out['secs'], 's', out['peak_mb'], 'MB')
