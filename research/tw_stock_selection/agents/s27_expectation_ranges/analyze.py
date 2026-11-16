#!/usr/bin/env python3
"""S27 第二步：歷史描述（part1）、成對區塊 bootstrap 前瞻模擬（part2）、失效警戒線（part3）、網站表（part4）。
輸入：monthly_paths.csv.gz、暫存區 daily_paths.pkl（build.py 產生）。輸出：results.json、site_table.json。
python3 -I -B analyze.py"""
import json, hashlib, time, resource
from pathlib import Path
import numpy as np, pandas as pd

H = Path(__file__).resolve().parent
assert hashlib.sha256((H / 'SPEC.json').read_bytes()).hexdigest() == (H / 'SPEC.sha256').read_text().split()[0], 'SPEC.json 與 SPEC.sha256 不符'
TMP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s27')
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
r4 = lambda x: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 4)

MON = pd.read_csv(H / 'monthly_paths.csv.gz', index_col='month')
DAILY = pd.read_pickle(TMP / 'daily_paths.pkl')
PATHS = [c for c in MON.columns if c != '0050']
STD = '100萬|+0'
assert len(PATHS) == 20
PERIODS = {'since2016': '2016-01', 'since2019': '2018-12'}      # 基準月（月報酬從下一個月起）
QS = [5, 10, 25, 50, 75, 90, 95]


# ------------------------------------------------------------------ 共用小工具
def run_under(W):
    """W：(..., T) 相對財富或淨值（第 0 欄為起點）。回傳每個時點『連續幾個月底低於前高』（起點為 0）。"""
    W = np.asarray(W, float)
    peak = np.maximum.accumulate(W, axis=-1)
    under = W < peak * (1 - 1e-12)
    out = np.zeros(W.shape, dtype=np.int32)
    for t in range(1, W.shape[-1]):
        out[..., t] = np.where(under[..., t], out[..., t - 1] + 1, 0)
    return out


def dd_series(N):
    N = np.asarray(N, float)
    return N / np.maximum.accumulate(N, axis=-1) - 1


def longest_true_run(b):
    best = cur = 0
    for v in b:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def episodes(nav):
    """日淨值回撤段：深度、谷底到收復（月＝交易日/21）、前高到收復（月）、是否未收復。"""
    v = nav.to_numpy(float); out = []
    pk = 0; i = 1; n = len(v)
    while i < n:
        if v[i] >= v[pk]:
            pk = i; i += 1; continue
        j = i; tr = i
        while j < n and v[j] < v[pk]:
            if v[j] < v[tr]:
                tr = j
            j += 1
        rec = j < n
        out.append(dict(peak=str(nav.index[pk].date()), trough=str(nav.index[tr].date()), recover=str(nav.index[j].date()) if rec else None,
                        depth=float(v[tr] / v[pk] - 1), trough_to_recover_m=(j - tr) / 21 if rec else (n - 1 - tr) / 21,
                        underwater_m=(j - pk) / 21 if rec else (n - 1 - pk) / 21, unrecovered=not rec))
        pk = j; i = j + 1
    return out


def summarize(vals, worst='min'):
    v = np.array(vals, float)
    return dict(median=r4(np.median(v)), worst=r4(v.min() if worst == 'min' else v.max()), best=r4(v.max() if worst == 'min' else v.min()))


# ------------------------------------------------------------------ part1 歷史描述
def hist_path(n, b, nd, bd):
    """n、b：月底淨值（含基準月）；nd、bd：日淨值（含基準日）。"""
    out = {}
    N = n.to_numpy(float); B = b.to_numpy(float)
    for w, key in ((12, 'R12'), (36, 'R36')):
        if w == 12:
            ex = (N[w:] / N[:-w]) - (B[w:] / B[:-w])
        else:
            ex = (N[w:] / N[:-w]) ** (12 / w) - (B[w:] / B[:-w]) ** (12 / w)
        out[key] = dict(min=r4(ex.min()), **{f'p{q}': r4(np.percentile(ex, q)) for q in (5, 10, 25, 50, 75)},
                        frac_neg=r4((ex < 0).mean()), n_windows=int(len(ex)), min_end=str(n.index[w + int(ex.argmin())]))
        if w == 12:
            out['R12_longest_neg_run_m'] = int(longest_true_run(ex < 0))
    W = N / B
    ru = run_under(W)
    rd = dd_series(W)
    # 相對落後段
    spells = []; t = 1
    while t < len(W):
        if ru[t] > 0:
            s0 = t
            while t < len(W) and ru[t] > 0:
                t += 1
            spells.append(dict(start=str(n.index[s0 - 1]), months=int(t - s0), ended=bool(t < len(W)), depth=r4(rd[s0 - 1:t].min())))
        t += 1
    out['lag_longest_m'] = int(ru.max())
    out['lag_longest_start'] = max(spells, key=lambda x: x['months'])['start'] if spells else None
    out['lag_current_m'] = int(ru[-1])
    out['lag_spells_ge12'] = int(sum(s['months'] >= 12 for s in spells))
    out['rel_maxdd'] = r4(rd.min())
    # 回撤（日）
    ep = episodes(nd)
    big10 = [e for e in ep if e['depth'] <= -0.10]; big20 = [e for e in ep if e['depth'] <= -0.20]
    out['maxdd_daily'] = r4(min(e['depth'] for e in ep))
    out['maxdd_monthly'] = r4(dd_series(N).min())
    out['n_dd_ge10'] = len(big10); out['n_dd_ge20'] = len(big20)
    for nm, L in (('ge10', big10), ('ge20', big20)):
        if L:
            out[f'recover_{nm}_median_m'] = r4(np.median([e['trough_to_recover_m'] for e in L]))
            out[f'recover_{nm}_max_m'] = r4(max(e['trough_to_recover_m'] for e in L))
            out[f'underwater_{nm}_median_m'] = r4(np.median([e['underwater_m'] for e in L]))
            out[f'underwater_{nm}_max_m'] = r4(max(e['underwater_m'] for e in L))
    out['longest_underwater_m'] = r4(max(e['underwater_m'] for e in ep))
    out['dd_episodes_ge10'] = [{k: (r4(v) if isinstance(v, float) else v) for k, v in e.items()} for e in big10]
    return out


part1 = {}
for per, base in PERIODS.items():
    m = MON.loc[base:]
    d0 = DAILY.index[DAILY.index.to_period('M') <= pd.Period(base, 'M')][-1]
    dd = DAILY.loc[d0:]
    res = {p: hist_path(m[p], m['0050'], dd[p], dd['0050']) for p in PATHS}
    bench = hist_path(m['0050'], m['0050'], dd['0050'], dd['0050'])
    keys_min = ['R12.min', 'R12.p5', 'R12.p10', 'R12.p25', 'R12.p50', 'R36.min', 'R36.p5', 'R36.p10', 'R36.p25', 'R36.p50',
                'rel_maxdd', 'maxdd_daily', 'maxdd_monthly']
    keys_max = ['R12.frac_neg', 'R36.frac_neg', 'R12_longest_neg_run_m', 'lag_longest_m', 'lag_spells_ge12', 'n_dd_ge10', 'n_dd_ge20',
                'recover_ge10_max_m', 'recover_ge20_max_m', 'underwater_ge10_max_m', 'underwater_ge20_max_m', 'longest_underwater_m',
                'recover_ge20_median_m', 'underwater_ge20_median_m']
    get = lambda r, k: (r[k.split('.')[0]][k.split('.')[1]] if '.' in k else r.get(k, np.nan))
    summ = {k: summarize([get(res[p], k) for p in PATHS], 'min') for k in keys_min}
    summ.update({k: summarize([get(res[p], k) if get(res[p], k) is not None else np.nan for p in PATHS], 'max') for k in keys_max})
    gap = [res[p]['maxdd_daily'] - res[p]['maxdd_monthly'] for p in PATHS]
    part1[per] = dict(base_month=base, n_months=int(len(m) - 1), summary_20paths=summ,
                      daily_minus_monthly_maxdd=dict(median=r4(np.median(gap)), worst=r4(min(gap))),
                      std_path=res[STD], bench_0050={k: bench[k] for k in ('maxdd_daily', 'maxdd_monthly', 'n_dd_ge10', 'n_dd_ge20', 'longest_underwater_m',
                                                                           'recover_ge20_max_m', 'underwater_ge20_max_m', 'dd_episodes_ge10') if k in bench},
                      per_path={p: {k: v for k, v in res[p].items() if k != 'dd_episodes_ge10'} for p in PATHS})
    print('part1', per, f'{time.time()-t_all:.0f}s', flush=True)
HIST_WORST_MDD_M = part1['since2016']['summary_20paths']['maxdd_monthly']['worst']
HIST_WORST_MDD_D = part1['since2016']['summary_20paths']['maxdd_daily']['worst']


# ------------------------------------------------------------------ part2 bootstrap
def boot_idx(T, L, nsim, rng, bmin=6, bmax=12):
    idx = np.empty((nsim, L), dtype=np.int32)
    for i in range(nsim):
        pos = 0
        while pos < L:
            bl = int(rng.integers(bmin, bmax + 1)); st = int(rng.integers(0, T))
            k = min(bl, L - pos)
            idx[i, pos:pos + k] = (st + np.arange(k)) % T
            pos += k
    return idx


def simulate(base, bmin=6, bmax=12, nsim=10000, L=96, seed=20261127):
    m = MON.loc[base:]
    RS = (m[PATHS].pct_change().iloc[1:]).to_numpy(float).T          # (20, T)
    RB = m['0050'].pct_change().iloc[1:].to_numpy(float)              # (T,)
    T = RB.shape[0]
    dbar = (RS - RB[None, :]).mean(axis=1)                            # (20,)
    rng = np.random.default_rng(seed)
    pi = rng.integers(0, 20, nsim)
    idx = boot_idx(T, L, nsim, rng, bmin, bmax)
    rs0 = RS[pi[:, None], idx]; rb = RB[idx]
    return dict(rs0=rs0, rb=rb, dbar=dbar[pi], dbar_all=dbar, T=T)


def horizon_stats(rs, rb):
    out = {}
    NS = np.concatenate([np.ones((rs.shape[0], 1)), np.cumprod(1 + rs, axis=1)], axis=1)
    NB = np.concatenate([np.ones((rb.shape[0], 1)), np.cumprod(1 + rb, axis=1)], axis=1)
    for h in (12, 36, 60):
        ns, nb = NS[:, :h + 1], NB[:, :h + 1]
        cs, cb = ns[:, -1] - 1, nb[:, -1] - 1
        ann = lambda c: (1 + c) ** (12 / h) - 1
        mdd = dd_series(ns).min(axis=1)
        lag = run_under(ns / nb).max(axis=1)
        out[f'{h}m'] = dict(
            strat_cum={f'p{q}': r4(np.percentile(cs, q)) for q in QS},
            strat_ann={f'p{q}': r4(np.percentile(ann(cs), q)) for q in QS},
            b0050_ann={f'p{q}': r4(np.percentile(ann(cb), q)) for q in QS},
            excess_ann={f'p{q}': r4(np.percentile(ann(cs) - ann(cb), q)) for q in QS},
            p_lose_to_0050=r4((cs < cb).mean()), p_strat_loss=r4((cs < 0).mean()), p_0050_loss=r4((cb < 0).mean()),
            maxdd_p50=r4(np.percentile(mdd, 50)), maxdd_p90=r4(np.percentile(mdd, 10)), maxdd_p95=r4(np.percentile(mdd, 5)),
            p_dd_worse_than_hist=r4((mdd < HIST_WORST_MDD_M).mean()),
            p_lag_ge12=r4((lag >= 12).mean()), p_lag_ge24=r4((lag >= 24).mean()), lag_p50=r4(np.median(lag)), lag_p90=r4(np.percentile(lag, 90)))
        if h == 60:
            r12 = ns[:, 12:] / ns[:, :-12] - nb[:, 12:] / nb[:, :-12]
            r36 = ns[:, 36:] / ns[:, :-36] - nb[:, 36:] / nb[:, :-36]
            out['events_5y'] = dict(any_12m_behind=r4((r12 < 0).any(axis=1).mean()), any_36m_behind=r4((r36 < 0).any(axis=1).mean()),
                                    lag_ge12=r4((lag >= 12).mean()), lag_ge24=r4((lag >= 24).mean()), dd_ge30=r4((mdd <= -0.30).mean()),
                                    dd_ge40=r4((mdd <= -0.40).mean()),
                                    r12_le_m20=r4((r12 <= -0.20).any(axis=1).mean()), year1_behind=r4((ns[:, 12] < nb[:, 12]).mean()))
    return out


SCEN = {'EFF': 0.0, 'HALF': 0.5, 'ZERO': 1.0}


def alert_stats(rs, rb):
    """回傳每次模擬在 1／3／5 年監看期（檢查點 NAV 索引 37..48／37..72／37..96）各條件的『最壞值』。"""
    NS = np.concatenate([np.ones((rs.shape[0], 1)), np.cumprod(1 + rs, axis=1)], axis=1)
    NB = np.concatenate([np.ones((rb.shape[0], 1)), np.cumprod(1 + rb, axis=1)], axis=1)
    W = NS / NB
    t = np.arange(NS.shape[1])
    A = np.full(NS.shape, np.nan); Bc = np.full(NS.shape, np.nan)
    A[:, 36:] = (NS[:, 36:] / NS[:, :-36]) ** (1 / 3) - (NB[:, 36:] / NB[:, :-36]) ** (1 / 3)
    Bc[:, 12:] = NS[:, 12:] / NS[:, :-12] - NB[:, 12:] / NB[:, :-12]
    C = dd_series(NS); D = run_under(W); E = dd_series(W)
    out = {}
    for yrs, last in ((1, 48), (3, 72), (5, 96)):
        sl = slice(37, last + 1)
        out[yrs] = dict(A=A[:, sl].min(axis=1), B=Bc[:, sl].min(axis=1), C=C[:, sl].min(axis=1), D=D[:, sl].max(axis=1), E=E[:, sl].min(axis=1))
    out['raw'] = dict(A=A, B=Bc, C=C, D=D, E=E)
    return out


GRID = {'A': np.round(np.arange(-0.30, 0.0501, 0.01), 4), 'B': np.round(np.arange(-0.80, 0.0001, 0.02), 4),
        'C': np.round(HIST_WORST_MDD_M - np.arange(0, 0.4001, 0.01), 4), 'D': np.arange(6, 97), 'E': np.round(np.arange(-0.80, 0.0001, 0.01), 4)}


def trig(stat, key, thr):
    return stat < thr if key != 'D' else stat >= thr


def pick_threshold(stat5, key, fa):
    g = GRID[key]
    ok = [x for x in g if trig(stat5, key, x).mean() <= fa]
    if not ok:
        return None
    return float(max(ok)) if key != 'D' else int(min(ok))


def run_block(base, bmin, bmax, full=True):
    sim = simulate(base, bmin, bmax)
    out = dict(dbar_monthly_by_path=dict(median=r4(np.median(sim['dbar_all'])), min=r4(sim['dbar_all'].min()), max=r4(sim['dbar_all'].max())),
               dbar_annual_simple_median=r4(np.median(sim['dbar_all']) * 12), T=sim['T'], scenarios={})
    al = {}
    for sc, k in SCEN.items():
        rs = sim['rs0'] - k * sim['dbar'][:, None]
        out['scenarios'][sc] = horizon_stats(rs[:, :60], sim['rb'][:, :60])
        al[sc] = alert_stats(rs, sim['rb'])
    # 警戒線
    alerts = {}
    for fa in (0.10, 0.05):
        th = {key: pick_threshold(al['EFF'][5][key], key, fa) for key in 'ABCDE'}
        tab = {}
        for key in 'ABCDE':
            if th[key] is None:
                tab[key] = None; continue
            tab[key] = dict(threshold=th[key], prob={sc: {f'{y}y': r4(trig(al[sc][y][key], key, th[key]).mean()) for y in (1, 3, 5)} for sc in SCEN})
            e5, z5 = tab[key]['prob']['EFF']['5y'], tab[key]['prob']['ZERO']['5y']
            tab[key]['zero_over_eff_5y'] = r4(z5 / e5) if e5 else None
        # 組合：任一條；A 且 D 同月
        anyp = {}; adp = {}
        for sc in SCEN:
            raw = al[sc]['raw']
            hit = np.zeros(raw['A'].shape, bool)
            for key in 'ABCDE':
                if th[key] is not None:
                    x = raw[key]
                    hit |= (np.nan_to_num(x, nan=np.inf) < th[key]) if key != 'D' else (x >= th[key])
            ad = (np.nan_to_num(raw['A'], nan=np.inf) < th['A']) & (raw['D'] >= th['D']) if th['A'] is not None and th['D'] is not None else np.zeros_like(hit)
            anyp[sc] = {f'{y}y': r4(hit[:, 37:last + 1].any(axis=1).mean()) for y, last in ((1, 48), (3, 72), (5, 96))}
            adp[sc] = {f'{y}y': r4(ad[:, 37:last + 1].any(axis=1).mean()) for y, last in ((1, 48), (3, 72), (5, 96))}
        tab['ANY'] = dict(prob=anyp); tab['A_and_D'] = dict(prob=adp)
        alerts[f'fa{int(fa*100)}'] = tab
    out['alerts'] = alerts
    # 固定參考門檻的觸發曲線（方便產品擁有者看）
    if full:
        curves = {}
        for key, vals in (('A', [-0.20, -0.15, -0.10, -0.05, 0.0]), ('B', [-0.50, -0.40, -0.30, -0.20, -0.10]),
                          ('C', [-0.30, -0.35, -0.40, -0.45, -0.50, -0.55]), ('D', [12, 18, 24, 30, 36, 48]), ('E', [-0.20, -0.30, -0.40, -0.50, -0.60])):
            curves[key] = {str(v): {sc: r4(trig(al[sc][5][key], key, v).mean()) for sc in SCEN} for v in vals}
        out['alert_curves_5y'] = curves
    return out, al


part2 = {}
main, AL_MAIN = run_block('2016-01', 6, 12)
part2['main_2016_b6to12'] = main
print('part2 main', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
for nm, base, a, b in (('sens_2016_b6', '2016-01', 6, 6), ('sens_2016_b12', '2016-01', 12, 12), ('sens_2019_b6to12', '2018-12', 6, 12)):
    r, _ = run_block(base, a, b, full=False)
    part2[nm] = r
    print('part2', nm, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
del AL_MAIN

# ------------------------------------------------------------------ part3 歷史上是否觸發過（主門檻，fa10）
TH = {k: (v['threshold'] if v else None) for k, v in main['alerts']['fa10'].items() if k in 'ABCDE'}
m = MON.loc['2016-01':]
hist_trig = {}
for p in PATHS:
    N = m[p].to_numpy(float); B = m['0050'].to_numpy(float); W = N / B
    A = np.full(len(N), np.nan); Bc = np.full(len(N), np.nan)
    A[36:] = (N[36:] / N[:-36]) ** (1 / 3) - (B[36:] / B[:-36]) ** (1 / 3)
    Bc[12:] = N[12:] / N[:-12] - B[12:] / B[:-12]
    stats = dict(A=A, B=Bc, C=dd_series(N), D=run_under(W), E=dd_series(W))
    hp = {}
    for key, x in stats.items():
        if TH[key] is None:
            continue
        hit = (np.nan_to_num(x, nan=np.inf) < TH[key]) if key != 'D' else (x >= TH[key])
        worst = np.nanmin(x) if key != 'D' else np.nanmax(x)
        hp[key] = dict(n_months=int(hit.sum()), months=[str(m.index[i]) for i in np.where(hit)[0]][:24], worst=r4(worst) if key != 'D' else int(worst))
    hist_trig[p] = hp
hist_sum = {key: dict(paths_triggered=int(sum(hist_trig[p].get(key, {}).get('n_months', 0) > 0 for p in PATHS)),
                      worst_value_median=r4(np.median([hist_trig[p][key]['worst'] for p in PATHS])) if TH[key] is not None else None,
                      worst_value_worst=(r4(min(hist_trig[p][key]['worst'] for p in PATHS)) if key != 'D' else int(max(hist_trig[p][key]['worst'] for p in PATHS))) if TH[key] is not None else None)
            for key in 'ABCDE'}
part3 = dict(thresholds_fa10=TH, thresholds_fa5={k: (v['threshold'] if v else None) for k, v in main['alerts']['fa5'].items() if k in 'ABCDE'},
             hist_worst_mdd_monthly=HIST_WORST_MDD_M, hist_worst_mdd_daily=HIST_WORST_MDD_D,
             history_check_summary=hist_sum, history_check_std=hist_trig[STD], history_check_per_path=hist_trig,
             uninformative_events_5y={sc: main['scenarios'][sc]['events_5y'] for sc in SCEN})

# ------------------------------------------------------------------ part4 網站表
S = main['scenarios']
pct = lambda x: int(round(100 * x))
def rng_(d, lo='p10', hi='p90'):
    return f'{pct(d[lo])}% ～ {pct(d[hi])}%'
dgap = part1['since2016']['daily_minus_monthly_maxdd']
h1 = part1['since2016']['summary_20paths']
rows = []
for h, nm in ((12, '未來 1 年'), (36, '未來 3 年'), (60, '未來 5 年')):
    rows.append(dict(item=f'{nm}報酬（年化）', normal_range=rng_(S['EFF'][f'{h}m']['strat_ann']),
                     reference_0050=rng_(S['EFF'][f'{h}m']['b0050_ann']),
                     note=f'模擬 P10～P90（策略有效情境；過去 10 年重抽，偏樂觀）。α 歸零時中位數 {pct(S["ZERO"][f"{h}m"]["strat_ann"]["p50"])}%。'))
rows.append(dict(item='某 12 個月輸 0050', normal_range=f'5 年內發生機率 {pct(S["EFF"]["events_5y"]["any_12m_behind"])}%；歷史 20 條中位數有 {pct(h1["R12.frac_neg"]["median"])}% 的 12 個月窗輸 0050',
                 alert=f'滾動 12 個月落後 0050 超過 {-pct(TH["B"])} 個百分點' if TH['B'] is not None else None,
                 note='單看「這一年輸 0050」不是失效證據（α 歸零時也只多一點）。'))
rows.append(dict(item='輸 0050 的機率', normal_range=f'1 年 {pct(S["EFF"]["12m"]["p_lose_to_0050"])}%／3 年 {pct(S["EFF"]["36m"]["p_lose_to_0050"])}%／5 年 {pct(S["EFF"]["60m"]["p_lose_to_0050"])}%',
                 note=f'α 歸零時 1 年 {pct(S["ZERO"]["12m"]["p_lose_to_0050"])}%／3 年 {pct(S["ZERO"]["36m"]["p_lose_to_0050"])}%／5 年 {pct(S["ZERO"]["60m"]["p_lose_to_0050"])}%。'))
rows.append(dict(item='滾動 3 年對 0050 年化超額', normal_range=f'歷史 20 條中位數最低 {pct(h1["R36.min"]["median"])} 個百分點（最差路徑 {pct(h1["R36.min"]["worst"])}）',
                 alert=f'低於 {pct(TH["A"])} 個百分點／年' if TH['A'] is not None else None, note='每月底檢查；需要 36 個月資料。'))
rows.append(dict(item='連續落後 0050（相對財富未創新高）', normal_range=f'歷史最長 {int(h1["lag_longest_m"]["median"])} 個月（20 條中位數），最差 {int(h1["lag_longest_m"]["worst"])} 個月；5 年內 ≥ 12 個月機率 {pct(S["EFF"]["60m"]["p_lag_ge12"])}%',
                 alert=f'連續 {TH["D"]} 個月底都沒追回前高' if TH['D'] is not None else None, note=f'α 歸零時 5 年內 ≥ 12 個月機率 {pct(S["ZERO"]["60m"]["p_lag_ge12"])}%，所以「落後一年多」本身不是失效證據。'))
rows.append(dict(item='最大回撤（5 年內）', normal_range=f'月底口徑中位數 {pct(S["EFF"]["60m"]["maxdd_p50"])}%、P90 {pct(S["EFF"]["60m"]["maxdd_p90"])}%；歷史日口徑最差 {pct(HIST_WORST_MDD_D)}%',
                 alert=f'月底淨值對前高回撤深於 {pct(TH["C"])}%' if TH['C'] is not None else None,
                 note=f'日內回撤通常比月底口徑再深約 {-pct(dgap["median"])} 個百分點（歷史最多 {-pct(dgap["worst"])}）。'))
rows.append(dict(item='相對 0050 的最深落後（相對財富對前高）', normal_range=f'歷史 20 條中位數 {pct(h1["rel_maxdd"]["median"])}%，最差 {pct(h1["rel_maxdd"]["worst"])}%',
                 alert=f'深於 {pct(TH["E"])}%' if TH['E'] is not None else None))
site = dict(title='現行策略（E0／W10_70）正常範圍與警戒線', as_of='2026-10-22', source='agents/s27_expectation_ranges/results.json',
            caveat='模擬從 2016-02～2026-10 的月報酬重抽（成對、區塊 6～12 個月、10,000 次），假設未來像過去 10 年；實際可能更差。警戒線觸發代表「值得檢討」，不是自動停用。',
            rows=rows)

results = dict(spec_sha256=(H / 'SPEC.sha256').read_text().split()[0], spec_deviations=[], build_meta=json.load(open(H / 'build_meta.json')),
               part1_history=part1, part2_simulation=part2, part3_alerts=part3, peak_mb=None, secs=None)
results['peak_mb'] = rss(); results['secs'] = round(time.time() - t_all)
json.dump(results, open(H / 'results.json', 'w'), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
json.dump(site, open(H / 'site_table.json', 'w'), ensure_ascii=False, indent=1)
print('done', results['secs'], 's', results['peak_mb'], 'MB')
