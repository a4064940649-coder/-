#!/usr/bin/env python3
"""S61 第二步：歷史燈號（樣本內／樣本外）、亮燈後超額、bootstrap 誤報率（同 S27 口徑）、與 S27 五條線重疊、現況、
compute_health.py 驗收。輸入：build_official.py 的輸出、S55 oos_*、S59 env_monthly.csv。輸出：results.json、lamp_monthly.csv。
python3 -I -B analyze.py"""
import json, hashlib, time, resource, subprocess, sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

H = Path(__file__).resolve().parent
AG = H.parent
SITE = Path('/home/user/-/xq_scripts/site')
SCR = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s61')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(H / 'SPEC.json') == (H / 'SPEC.sha256').read_text().split()[0]
assert sha(H / 'PREREG.json') == (H / 'PREREG.sha256').read_text().split()[0]
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
r4 = lambda x: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 4)
spec = importlib.util.spec_from_file_location('compute_health', H / 'compute_health.py')
CH = importlib.util.module_from_spec(spec); spec.loader.exec_module(CH)
TH27 = dict(A=-0.08, B=-0.26, C=-0.2991, D=50, E=-0.24)            # S27 results.json thresholds_fa10
res = dict(spec_sha256=sha(H / 'SPEC.json'), prereg_sha256=sha(H / 'PREREG.json'), spec_deviations=[],
           build_meta=json.loads((H / 'build_meta.json').read_text())['acceptance'])

# ------------------------------------------------------------------ 資料
PM = pd.read_csv(H / 'paths_monthly.csv.gz')
PATHS = list(dict.fromkeys(PM['path'])); STD = '100萬|+0'
assert len(PATHS) == 20


def path_monthly(p):
    d = PM[PM['path'] == p].set_index('month'); d.index = pd.PeriodIndex(d.index, freq='M')
    M = pd.DataFrame(index=d.index)
    M['ret_s'] = d['nav'].pct_change(); M['ret_b'] = d['b0050'].pct_change(); M['excess'] = M['ret_s'] - M['ret_b']
    M['w_act'] = d['w_act']; M['n_closed'] = d['n_closed']; M['n_win'] = d['n_win']; M['n_picks'] = d['n_picks']
    fv = M['n_picks'].first_valid_index()
    M.loc[M.index >= fv, 'n_picks'] = M.loc[M.index >= fv, 'n_picks'].fillna(0)      # 第一個名單月之後缺名單＝0 檔
    return M


IS = {p: path_monthly(p) for p in PATHS}
# 樣本外：S55 月報酬＋交易，S59 的 NTGT／W_ACT
om = pd.read_csv(AG / 's55_pre2015_oos' / 'oos_monthly.csv', index_col=0); om.index = pd.PeriodIndex(om.index, freq='M')
ot = pd.read_csv(AG / 's55_pre2015_oos' / 'oos_trades.csv.gz', dtype={'code': str})
ot = ot[ot['why'] != 'open_end'].copy(); ot['ret'] = ot['proceeds'] / ot['cost'] - 1
ot['xm'] = pd.to_datetime(ot['exit_date']).dt.to_period('M')
env = pd.read_csv(AG / 's59_regime_attribution' / 'env_monthly.csv', index_col=0); env.index = pd.PeriodIndex(env.index, freq='M')
OOS = pd.DataFrame(index=om.index)
OOS['ret_s'] = om['strategy']; OOS['ret_b'] = om['b0050']; OOS['excess'] = om['excess']
assert float((env['excess'].reindex(OOS.index) - OOS['excess']).abs().max()) < 1e-10
OOS['w_act'] = env['W_ACT'].reindex(OOS.index)
OOS['n_closed'] = ot.groupby('xm').size().reindex(OOS.index, fill_value=0)
OOS['n_win'] = ot[ot['ret'] > 0].groupby('xm').size().reindex(OOS.index, fill_value=0)
OOS['n_picks'] = env['NTGT'].reindex(OOS.index)
OOS.loc[OOS.index >= pd.Period('2008-02', 'M'), 'n_picks'] = OOS.loc[OOS.index >= pd.Period('2008-02', 'M'), 'n_picks'].fillna(0)   # STUCK：沒名單＝0 檔
OOS = OOS.loc[:'2013-12']

L_IS = {p: CH.lamp_table(IS[p]) for p in PATHS}
L_OOS = CH.lamp_table(OOS, first_signal='2008-02')
SEG = {'IS': (L_IS[STD], IS[STD], '2017-01', '2026-10'), 'OOS': (L_OOS, OOS, '2009-01', '2013-01'), 'STUCK': (L_OOS, OOS, '2013-02', '2013-12')}
END_FOR_FWD = {'IS': pd.Period('2026-10', 'M'), 'OOS': pd.Period('2013-01', 'M'), 'STUCK': pd.Period('2013-12', 'M')}


# ------------------------------------------------------------------ S27 五條線（歷史，月底）
def s27_lines(M, base):
    m = M.loc[pd.Period(base, 'M') + 1:]
    NS = np.concatenate([[1.0], np.cumprod(1 + m['ret_s'].to_numpy())]); NB = np.concatenate([[1.0], np.cumprod(1 + m['ret_b'].to_numpy())])
    W = NS / NB
    A = np.full(len(NS), np.nan); B = np.full(len(NS), np.nan)
    A[36:] = (NS[36:] / NS[:-36]) ** (1 / 3) - (NB[36:] / NB[:-36]) ** (1 / 3)
    B[12:] = NS[12:] / NS[:-12] - NB[12:] / NB[:-12]
    C = NS / np.maximum.accumulate(NS) - 1; E = W / np.maximum.accumulate(W) - 1
    D = np.zeros(len(W), int); pk = np.maximum.accumulate(W)
    for t in range(1, len(W)):
        D[t] = D[t - 1] + 1 if W[t] < pk[t] * (1 - 1e-12) else 0
    idx = pd.PeriodIndex([pd.Period(base, 'M')] + list(m.index), freq='M')
    out = pd.DataFrame(dict(A=A, B=B, C=C, D=D, E=E), index=idx).iloc[1:]
    for k in 'ABCE':
        out['S_' + k] = out[k] < TH27[k]
    out['S_D'] = out['D'] >= TH27['D']
    out['S_n'] = out[['S_A', 'S_B', 'S_C', 'S_D', 'S_E']].sum(axis=1)
    out['S_any'] = out['S_n'] >= 1; out['S_red'] = out['S_n'] >= 2
    return out


S27H = {'IS': s27_lines(IS[STD], '2016-01'), 'OOS': s27_lines(OOS, '2007-12')}
S27H['STUCK'] = S27H['OOS']
chk = S27H['IS'].loc['2026-10']
res['s27_recompute_check_2026_10'] = dict(A=r4(chk['A']), B=r4(chk['B']), C=r4(chk['C']), D=int(chk['D']), E=r4(chk['E']),
                                          note='與網站 RISK 常數（正式淨值重算：A 15.0、B 33.5、C −5.4、E −7.1）對照')


# ------------------------------------------------------------------ part1 歷史
def fwd_excess(M, m, h, end):
    i = M.index.get_loc(m)
    if m + h > end or i + h >= len(M):
        return np.nan
    w = M.iloc[i + 1:i + 1 + h]
    return float((np.prod(1 + w['ret_s']) - np.prod(1 + w['ret_b'])) * 100)


def fwd_block(M, months, end, allm):
    o = {}
    for h in (3, 6, 12):
        v = np.array([fwd_excess(M, m, h, end) for m in months]); v = v[np.isfinite(v)]
        b = np.array([fwd_excess(M, m, h, end) for m in allm]); b = b[np.isfinite(b)]
        o[f'{h}m'] = dict(n=int(len(v)), mean_pp=r4(v.mean()) if len(v) else None, median_pp=r4(np.median(v)) if len(v) else None,
                          frac_positive=r4((v > 0).mean()) if len(v) else None, baseline_all_months_mean_pp=r4(b.mean()) if len(b) else None,
                          baseline_n=int(len(b)))
    return o


def runs(mask):
    out = []; cur = None
    for m, v in mask.items():
        if v and cur is None:
            cur = [m, m]
        elif v:
            cur[1] = m
        elif cur is not None:
            out.append(cur); cur = None
    if cur is not None:
        out.append(cur)
    return [dict(start=str(a), end=str(b), months=int(b.ordinal - a.ordinal + 1)) for a, b in out]


part1 = {}; tables = []
for seg, (L, M, a, b) in SEG.items():
    l = L.loc[a:b]
    red = l['n_lit'] >= 2; yel = l['n_lit'] == 1; anyl = l['n_lit'] >= 1
    allm = list(l.index); end = END_FOR_FWD[seg]
    rr = runs(red)
    o = dict(period=f'{a}～{b}', months=int(len(l)), red=int(red.sum()), yellow=int(yel.sum()), green=int((l['n_lit'] == 0).sum()),
             L1=int(l['L1'].sum()), L2=int(l['L2'].sum()), L3=int(l['L3'].sum()),
             first_red=str(l.index[red][0]) if red.any() else None, first_any=str(l.index[anyl][0]) if anyl.any() else None,
             first_L1=str(l.index[l['L1']][0]) if l['L1'].any() else None, first_L2=str(l.index[l['L2']][0]) if l['L2'].any() else None,
             first_L3=str(l.index[l['L3']][0]) if l['L3'].any() else None,
             red_runs=rr, red_months=[str(x) for x in l.index[red]],
             combos={k: int(v) for k, v in l.loc[red, ['L1', 'L2', 'L3']].astype(int).astype(str).agg(''.join, axis=1).value_counts().items()})
    if seg != 'STUCK':
        o['after'] = dict(red=fwd_block(M, list(l.index[red]), end, allm),
                          red_first_of_run=fwd_block(M, [pd.Period(x['start'], 'M') for x in rr], end, allm),
                          yellow_or_red=fwd_block(M, list(l.index[anyl]), end, allm),
                          L1=fwd_block(M, list(l.index[l['L1']]), end, allm), L2=fwd_block(M, list(l.index[l['L2']]), end, allm),
                          L3=fwd_block(M, list(l.index[l['L3']]), end, allm))
    part1[seg] = o
    t = l.copy(); t['seg'] = seg
    t = t.join(M[['excess']]); t['excess_pp'] = t['excess'] * 100
    tables.append(t)
# 樣本內 20 條路徑：紅燈月數分布
p20 = {p: dict(red=int((L_IS[p].loc['2017-01':'2026-10', 'n_lit'] >= 2).sum()), yellow=int((L_IS[p].loc['2017-01':'2026-10', 'n_lit'] == 1).sum()),
               first_red=(str(L_IS[p].loc['2017-01':'2026-10'].index[L_IS[p].loc['2017-01':'2026-10', 'n_lit'] >= 2][0])
                          if (L_IS[p].loc['2017-01':'2026-10', 'n_lit'] >= 2).any() else None)) for p in PATHS}
for p in PATHS:
    lp = L_IS[p].loc['2017-01':'2026-10']; rm = lp['n_lit'] >= 2
    p20[p]['red_months'] = [str(x) for x in lp.index[rm]]
    p20[p]['red_combos'] = {k: int(v) for k, v in lp.loc[rm, ['L1', 'L2', 'L3']].astype(int).astype(str).agg(''.join, axis=1).value_counts().items()}
yrs = pd.Series([m[:4] for p in PATHS for m in p20[p]['red_months']]).value_counts().sort_index()
part1['IS_20paths_red_path_months_by_year'] = {k: int(v) for k, v in yrs.items()}
pool = {}
for h in (3, 6, 12):
    v = []; b = []
    for p in PATHS:
        lp = L_IS[p].loc['2017-01':'2026-10']
        v += [fwd_excess(IS[p], m, h, END_FOR_FWD['IS']) for m in lp.index[lp['n_lit'] >= 2]]
        b += [fwd_excess(IS[p], m, h, END_FOR_FWD['IS']) for m in lp.index]
    v = np.array(v); v = v[np.isfinite(v)]; b = np.array(b); b = b[np.isfinite(b)]
    pool[f'{h}m'] = dict(n_path_months=int(len(v)), mean_pp=r4(v.mean()), median_pp=r4(np.median(v)), frac_positive=r4((v > 0).mean()),
                         baseline_all_months_mean_pp=r4(b.mean()))
part1['IS_20paths_after_red_pooled'] = pool
part1['IS_20paths'] = dict(per_path=p20, paths_with_red=int(sum(v['red'] > 0 for v in p20.values())),
                           red_months_median=float(np.median([v['red'] for v in p20.values()])), red_months_max=int(max(v['red'] for v in p20.values())))
# 樣本外 2011～2013-01 子段
l = L_OOS.loc['2011-01':'2013-01']
part1['OOS_2011_2013_01'] = dict(months=int(len(l)), red=int((l['n_lit'] >= 2).sum()), yellow=int((l['n_lit'] == 1).sum()),
                                 L1=int(l['L1'].sum()), L2=int(l['L2'].sum()), L3=int(l['L3'].sum()))
res['part1_history'] = part1
LM = pd.concat(tables)
LM.index = LM.index.astype(str)
LM['picks_last3'] = LM['picks_last3'].astype(str)
LM[['seg', 'win12', 'n_closed12', 'unit12_pp', 'excess12_sum_pp', 'picks_last3', 'L1', 'L2', 'L3', 'n_lit', 'lamp', 'excess_pp']].round(4).to_csv(H / 'lamp_monthly.csv', index_label='month')
print('part1', f'{time.time()-t_all:.0f}s', flush=True)

# ------------------------------------------------------------------ part4 歷史重疊
part4 = {}
for seg in ('IS', 'OOS', 'STUCK'):
    _, _, a, b = SEG[seg]
    l = SEG[seg][0].loc[a:b]; s = S27H[seg].loc[a:b]
    hz = dict(L1=l['L1'], L2=l['L2'], L3=l['L3'], any=l['n_lit'] >= 1, red=l['n_lit'] >= 2)
    sz = dict(A=s['S_A'], B=s['S_B'], C=s['S_C'], D=s['S_D'], E=s['S_E'], any=s['S_any'], red=s['S_red'])
    o = dict(s27_lit_months={k: int(v.sum()) for k, v in sz.items()},
             s27_worst=dict(A=r4(s['A'].min()), B=r4(s['B'].min()), C=r4(s['C'].min()), D=int(s['D'].max()), E=r4(s['E'].min())), pairs={})
    for hk, hv in hz.items():
        for sk, sv in sz.items():
            both = int((hv & sv).sum()); either = int((hv | sv).sum())
            o['pairs'][f'{hk}×{sk}'] = dict(both=both, health=int(hv.sum()), s27=int(sv.sum()), jaccard=r4(both / either) if either else None,
                                            share_of_health_months=r4(both / hv.sum()) if hv.sum() else None)
    part4[seg] = o
res['part4_overlap_history'] = part4

# ------------------------------------------------------------------ part2 bootstrap
NSIM, LEN, SEED = 10000, 96, 20261161


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


def stack(Ms, a, b):
    cols = ['ret_s', 'ret_b', 'w_act', 'n_closed', 'n_win', 'n_picks']
    return np.stack([M.loc[a:b, cols].to_numpy(float) for M in Ms])            # (P, T, 6)


def roll(x, n):
    c = np.cumsum(x, axis=1); out = c.copy(); out[:, n:] = c[:, n:] - c[:, :-n]; out[:, :n - 1] = np.nan
    return out


def sim_lamps(X):
    """X：(nsim, L, 6)。回傳各項亮燈布林（nsim, L）、S27 五條線（以 NAV 索引 1..L 對齊月 1..L）。"""
    rs, rb, w, nc, nw, pk = [X[..., i] for i in range(6)]
    ex = rs - rb
    C12 = roll(nc, 12); W12 = roll(nw, 12)
    with np.errstate(invalid='ignore', divide='ignore'):
        win = W12 / C12
        L1 = (C12 > 0) & (win < CH.WIN_TH)
        sx = roll(ex, 12); sw = roll(w, 12)
        L2 = (sw > 0) & (sx / sw < 0)
    pm = np.full(pk.shape, np.nan); pm[:, 2:] = np.maximum(np.maximum(pk[:, 2:], pk[:, 1:-1]), pk[:, :-2])
    L3 = pm <= CH.PICK_TH
    L1 &= np.isfinite(C12); L2 &= np.isfinite(sx)
    NS = np.concatenate([np.ones((rs.shape[0], 1)), np.cumprod(1 + rs, axis=1)], axis=1)
    NB = np.concatenate([np.ones((rs.shape[0], 1)), np.cumprod(1 + rb, axis=1)], axis=1)
    W = NS / NB
    A = np.full(NS.shape, np.nan); B = np.full(NS.shape, np.nan)
    A[:, 36:] = (NS[:, 36:] / NS[:, :-36]) ** (1 / 3) - (NB[:, 36:] / NB[:, :-36]) ** (1 / 3)
    B[:, 12:] = NS[:, 12:] / NS[:, :-12] - NB[:, 12:] / NB[:, :-12]
    Cc = NS / np.maximum.accumulate(NS, axis=1) - 1; E = W / np.maximum.accumulate(W, axis=1) - 1
    pkW = np.maximum.accumulate(W, axis=1); und = W < pkW * (1 - 1e-12)
    D = np.zeros(W.shape, np.int32)
    for t in range(1, W.shape[1]):
        D[:, t] = np.where(und[:, t], D[:, t - 1] + 1, 0)
    S = dict(A=np.nan_to_num(A, nan=np.inf)[:, 1:] < TH27['A'], B=np.nan_to_num(B, nan=np.inf)[:, 1:] < TH27['B'], C=Cc[:, 1:] < TH27['C'],
             D=D[:, 1:] >= TH27['D'], E=E[:, 1:] < TH27['E'])
    return dict(L1=L1, L2=L2, L3=L3), S


def summarize(Lb, S):
    n = Lb['L1'].astype(int) + Lb['L2'].astype(int) + Lb['L3'].astype(int)
    red = n >= 2; anyl = n >= 1
    Sn = sum(S[k].astype(int) for k in 'ABCDE'); Sany = Sn >= 1; Sred = Sn >= 2
    out = {}
    for y, last in ((1, 48), (3, 72), (5, 96)):
        sl = slice(36, last)                                                  # 月 37..last（陣列索引 36..last−1）
        f = lambda z: r4(z[:, sl].any(axis=1).mean())
        out[f'{y}y'] = dict(red=f(red), yellow_or_red=f(anyl), L1=f(Lb['L1']), L2=f(Lb['L2']), L3=f(Lb['L3']),
                            s27_any=f(Sany), s27_red=f(Sred), red_and_s27any_same_month=f(red & Sany), red_and_s27red_same_month=f(red & Sred),
                            any_and_s27any_same_month=f(anyl & Sany))
    sl = slice(36, 96)
    rm = red[:, sl]; am = anyl[:, sl]
    out['month_level_5y'] = dict(red_month_share=r4(rm.mean()), yellow_or_red_month_share=r4(am.mean()),
                                 s27any_share_within_red_months=r4((rm & Sany[:, sl]).sum() / rm.sum()) if rm.sum() else None,
                                 s27any_share_within_lit_months=r4((am & Sany[:, sl]).sum() / am.sum()) if am.sum() else None,
                                 red_share_within_s27any_months=r4((rm & Sany[:, sl]).sum() / Sany[:, sl].sum()) if Sany[:, sl].sum() else None,
                                 mean_red_months_given_any_red=r4(rm.sum(axis=1)[rm.any(axis=1)].mean()) if rm.any() else None)
    return out


part2 = dict(nsim=NSIM, length=LEN, seed=SEED, monitoring='月 37～48／37～72／37～96', scenarios={})
rng = np.random.default_rng(SEED)
XI = stack([IS[p] for p in PATHS], '2016-02', '2026-10')                 # (20, 129, 6)
T = XI.shape[1]
pi = rng.integers(0, len(PATHS), NSIM)
idx = boot_idx(T, LEN, NSIM, rng)
X = XI[pi[:, None], idx]
dbar = (XI[..., 0] - XI[..., 1]).mean(axis=1)
Lb, S = sim_lamps(X)
part2['scenarios']['EFF'] = summarize(Lb, S)
X0 = X.copy(); X0[..., 0] -= dbar[pi][:, None]
Lb0, S0 = sim_lamps(X0)
part2['scenarios']['ZERO_EX'] = summarize(Lb0, S0)
part2['dbar_monthly_median'] = r4(np.median(dbar))
del X, X0, Lb, S, Lb0, S0
for nm, a, b in (('OOS', '2008-02', '2013-01'), ('BAD', '2011-01', '2013-01')):
    XO = stack([OOS], a, b)
    rng2 = np.random.default_rng(SEED)
    rng2.integers(0, len(PATHS), NSIM)                                     # 保持與 EFF 相同的亂數序列結構
    io = boot_idx(XO.shape[1], LEN, NSIM, rng2)
    Lb, S = sim_lamps(XO[0][io])
    part2['scenarios'][nm] = summarize(Lb, S)
    part2['scenarios'][nm]['T'] = int(XO.shape[1])
    del Lb, S
res['part2_false_alarm'] = part2
print('part2', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

# ------------------------------------------------------------------ part3 現況
cur = {}
for m in ('2026-10',):
    r = L_IS[STD].loc[m]
    cur[m] = dict(win12=r4(r['win12']), n_closed12=int(r['n_closed12']), unit12_pp=r4(r['unit12_pp']), excess12_sum_pp=r4(r['excess12_sum_pp']),
                  picks_last3=r['picks_last3'], L1=bool(r['L1']), L2=bool(r['L2']), L3=bool(r['L3']), lamp=r['lamp'],
                  note='2026-10 只到 2026-10-22（正式淨值最後一天）')
top = json.loads((SITE / 'data' / 'top20.json').read_text())
st = json.loads((SITE / 'data' / 'status.json').read_text())['rows'][0]
n11 = sum(1 for x in top['rows'] if x.get('入選') == '是')
pk3 = [int(v) for v in IS[STD]['n_picks'].loc['2026-09':'2026-10']] + [n11]
cur['2026-11_latest'] = dict(sig_date=st['本月訊號日'], n_picks_2026_11=n11, picks_last3=pk3, L3=bool(max(pk3) <= 2),
                             L1=cur['2026-10']['L1'], L2=cur['2026-10']['L2'],
                             lamp={0: '綠', 1: '黃'}.get(int(cur['2026-10']['L1']) + int(cur['2026-10']['L2']) + int(max(pk3) <= 2), '紅'),
                             note='2026-11 名單來自 xq_scripts/site/data/top20.json『入選』欄（唯讀）；①②沒有 2026-10-22 之後的正式淨值與交易，沿用 2026-10')
cur['2026-10_20paths'] = {p: L_IS[p].loc['2026-10', 'lamp'] for p in PATHS}
res['part3_current'] = cur

# ------------------------------------------------------------------ compute_health.py 驗收
SCR.mkdir(parents=True, exist_ok=True)
acc = {}
for nm, extra in (('exact_weights', ['--weights', str(H / 'official_weights.csv.gz')]), ('approx_weights', [])):
    out = SCR / f'ch_{nm}.json'
    subprocess.run([sys.executable, '-I', '-B', str(H / 'compute_health.py'), '--nav', str(SITE / 'navs_official.csv.gz'), '--trades', str(H / 'official_trades.csv.gz'),
                    '--picks', str(H / 'official_picks.csv'), '--history', '--out', str(out)] + extra, check=True)
    J = json.loads(out.read_text())
    hh = {x['月']: x for x in J['逐月']}
    ref = L_IS[STD].loc['2017-01':'2026-10']
    mism = [m for m in ref.index.astype(str) if hh[m]['燈號'] != ref.loc[m, 'lamp'] or sorted(hh[m]['亮']) != sorted(k for k in ('L1', 'L2', 'L3') if ref.loc[m, k])]
    du = [abs(hh[m]['單位部位超額'] - ref.loc[m, 'unit12_pp']) for m in ref.index.astype(str) if hh[m]['單位部位超額'] is not None]
    acc[nm] = dict(months=int(len(ref)), lamp_or_items_mismatch=mism, unit12_abs_diff_max_pp=r4(max(du)), unit12_abs_diff_median_pp=r4(np.median(du)),
                   current=dict(lamp=J['燈號'], items=[(x['項目'], x['數值'], x['亮']) for x in J['三項']]))
res['compute_health_acceptance'] = acc
res['additions_after_seeing_results'] = ['part1_history.IS_20paths 的 red_months／red_combos 與 IS_20paths_red_path_months_by_year、IS_20paths_after_red_pooled：看到標準路徑 0 個紅燈月、但 20 條路徑多數有紅燈後才加的描述，不改任何規則或門檻']
res['peak_mb'] = rss(); res['secs'] = round(time.time() - t_all)
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k: res[k] for k in ('part3_current', 'compute_health_acceptance')}, ensure_ascii=False, indent=1, default=str)[:4000])
print('done', res['secs'], 's', res['peak_mb'], 'MB')
