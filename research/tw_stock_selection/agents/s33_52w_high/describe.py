#!/usr/bin/env python3
"""S33 描述（不論通過與否都做；驗證期判定之後才跑，不影響判定）。輸出 results_describe.json、f1_pth.csv.gz。python3 -I -B describe.py
由 S29 describe.py 改寫（分數換成 PTH）。
(1) F1 逐筆（位移 0；E0 持有期，s20lib.outcomes，不扣成本）：對 0050 超額（ex50）與對同業超額（exP；同 S25／S29）。
    PTH 三分組：(i) TF1＝當月 F1 內 PTH 名次百分位三分（主表；當月 F1 ≥ 3 檔才分組）；(ii) pPTH＝同期股池百分位三分。
    各組 n、平均、中位數、勝率、月份群聚 t；高組 − 低組逐月差的平均與 t；逐月 IC（Spearman，當月 ≥ 4 檔）平均與 t。
    另：L_MA 第 11～20 名依 PTH ≥ 0.90 分兩組（FILL_PTH 的遞補來源）；L_MA 前 10 名中 RET20 ≤ 10% 但 PTH ≥ 0.90 者（GATE_PTH 新增來源）
        與 RET20 > 10% 但 PTH < 0.90 者（GATE_PTH 剔除來源）。
(2) 隨機對照：凍結規則的 PTH 換成當月 L_MA 前 20 名 PTH 值的隨機重排，20 次；驗證窗（重新起跑）與 2019 起（全期連續），位移 0、100 萬。"""
import sys, json, time, resource, gc, math
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s33lib
from s33lib import s20lib, s17lib, s10lib, s4lib, stlib, COSTS
sys.path.insert(0, str(s33lib.AG / 's25_factor_momentum'))
import s25lib                                              # 只借用 industry_matrix／last_listed／hold_ret（不改）
sys.path.insert(0, str(s33lib.AG / 's29_course_qmv'))
for mod, attr in ((s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s33lib.s9lib, 'S9'), (s33lib.s8lib, 'S8'), (s33lib.s7lib, 'S7'), (s33lib.s6lib, 'S6'),
                  (s33lib.s5lib, 'S5'), (s4lib, 'S4'), (s33lib.s3lib, 'S3'), (s33lib.s2lib, 'S2'), (stlib, 'S1'), (s33lib.cslib, 'TMP')):
    setattr(mod, attr, s33lib.S33)
assert s33lib.cslib.EXRIGHT_FIX

H = Path(__file__).resolve().parent
assert (H / 'results.json').exists(), '先跑 validate.py'
FZ = json.loads((H / 'FROZEN.json').read_text())
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s33lib.load()
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
RT = s33lib.surge(P)
s17lib.e50_fix_suspension(P)
PS = s33lib.PTHS(P, U)
import s29lib                                              # 只借用 mcap_panel（同業排除最大市值股；不改）
MC = s29lib.mcap_panel(P)
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
ks = [x['k'] for x in S0['info']]
IND, NAMES = s25lib.industry_matrix(P, ks)
LL = s25lib.last_listed(P)
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TB, D19, DE = lastb('2022-01-01'), lastb('2019-01-01'), P.IDX[-1]
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
out = {'note': '描述，未參與判定；驗證期判定（results.json）之後才跑', 'FROZEN_sha256': (H / 'FROZEN.sha256').read_text().split()[0]}


def tstat(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(v.mean() / (v.std(ddof=1) / math.sqrt(len(v)))) if len(v) >= 3 and v.std(ddof=1) > 0 else float('nan')


# ---------------------------------------------------------------- (1) 逐筆
df = s33lib.cand_table(P, U, RT, S0, PS)
o = s20lib.outcomes(P, S0['X'], ks, df, P.T - 1)
df = df.join(o)
peer = np.full(len(df), np.nan); npeer = np.zeros(len(df), int); indn = []
for i, y in enumerate(df.itertuples()):
    j, k = int(y.j), int(y.k); e = k + 1
    ind = IND[k]; g = ind[j]
    indn.append(NAMES[g] if g >= 0 else '')
    if g < 0 or not np.isfinite(y.ret):
        continue
    x = None if y.how == 'truncated' else int(y.exit_t)
    cand = np.flatnonzero(U[k] & (ind == g))
    ca = np.union1d(cand, [j])
    big = int(ca[np.argmax(np.nan_to_num(MC[k, ca].astype(float), nan=-1))])
    peers = [p for p in cand if p != j and p != big and np.isfinite(P.O[e, p]) and P.O[e, p] > 0]
    pr = np.array([s25lib.hold_ret(P, p, e, x, LL) for p in peers], float)
    pr = pr[np.isfinite(pr)]
    npeer[i] = len(pr)
    if len(pr) >= 5:
        peer[i] = pr.mean()
df['peer'] = peer; df['n_peer'] = npeer; df['industry'] = indn
df['ex50'] = df['excess']; df['exP'] = df['ret'] - df['peer']
df['mper'] = pd.PeriodIndex(df['month'], freq='M')
df = df[df['mper'] >= pd.Period('2016-01', 'M')].copy()
df['period'] = np.where(df['mper'] <= s33lib.TRAIN_LAST_MONTH, 'train', 'valid')
# 當月 F1 內 PTH 名次百分位（缺值不排）
f1m = df['grp'] == 'F1'
df['rF1'] = np.nan
for m, g in df[f1m].groupby('month'):
    g = g[np.isfinite(g['PTH'])]
    if len(g) >= 3:
        r = g['PTH'].rank(method='average')
        df.loc[g.index, 'rF1'] = (r - 0.5) / len(g)
print('(1) 逐筆完成', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)


def grp_stats(d, col):
    d = d[np.isfinite(d[col])]
    v = d[col].to_numpy()
    if len(v) == 0:
        return dict(n=0)
    mm = d.groupby('month')[col].mean()
    return dict(n=int(len(v)), mean_pct=float(v.mean() * 100), median_pct=float(np.median(v) * 100), win=float((v > 0).mean()),
                t_cluster_month=tstat(mm.to_numpy()), n_months=int(len(mm)),
                mean_drop_top5_pct=float(np.sort(v)[:-5].mean() * 100) if len(v) > 5 else None)


def tert(x):
    return np.where(~np.isfinite(x), 'NA', np.where(x < 1 / 3, 'L', np.where(x < 2 / 3, 'M', 'H')))


desc = {}
for per in ('train', 'valid', 'all'):
    sel = (df['grp'] == 'F1') & ((df['period'] == per) | (per == 'all'))
    d0 = df[sel]
    o1 = {'ALL': {c: grp_stats(d0, c) for c in ('ex50', 'exP')}}
    for sc in ('rF1', 'pPTH'):
        tg = tert(d0[sc].to_numpy())
        r = {'PTH_range_by_group': {g: [float(np.nanmin(d0['PTH'][tg == g])), float(np.nanmedian(d0['PTH'][tg == g])), float(np.nanmax(d0['PTH'][tg == g]))]
                                    for g in ('L', 'M', 'H') if (tg == g).any()}}
        for col in ('ex50', 'exP'):
            rr = {g: grp_stats(d0[tg == g], col) for g in ('L', 'M', 'H', 'NA')}
            dd = d0.assign(tg=tg)[np.isfinite(d0[col])]
            gm = dd.groupby(['month', 'tg'])[col].mean().unstack()
            if 'H' in gm and 'L' in gm:
                hl = (gm['H'] - gm['L']).dropna()
                rr['H_minus_L_month'] = dict(mean_pct=float(hl.mean() * 100), t=tstat(hl.to_numpy()), n_months=int(len(hl)))
            ics = []
            for m, g in dd.groupby('month'):
                g = g[np.isfinite(g['PTH'])]
                if len(g) >= 4 and g['PTH'].nunique() > 1 and g[col].nunique() > 1:
                    ics.append(g[['PTH', col]].corr(method='spearman').iloc[0, 1])
            rr['IC_month'] = dict(mean=float(np.mean(ics)) if ics else None, t=tstat(ics), n_months=len(ics))
            r[col] = rr
        o1[sc] = r
    pm = (df['period'] == per) | (per == 'all')
    d1 = df[pm & (df['grp'] == 'R11_20')]
    d1r = d1[d1['ret20'] > 0.10]
    o1['R11_20_by_PTH_ge_0.9'] = {col: dict(hi=grp_stats(d1[d1['PTH'] >= 0.9], col), lo=grp_stats(d1[~(d1['PTH'] >= 0.9)], col),
                                            ret20_gt_10_all=grp_stats(d1r, col)) for col in ('ex50', 'exP')}
    d2 = df[pm & (df['lma_rank'] <= 10)]
    o1['top10_gate_sources'] = {col: dict(added_ret20_le10_pth_ge90=grp_stats(d2[(d2['ret20'] <= 0.10) & (d2['PTH'] >= 0.9)], col),
                                          removed_ret20_gt10_pth_lt90=grp_stats(d2[(d2['ret20'] > 0.10) & (d2['PTH'] < 0.9)], col),
                                          kept_both=grp_stats(d2[(d2['ret20'] > 0.10) & (d2['PTH'] >= 0.9)], col)) for col in ('ex50', 'exP')}
    desc[per] = o1
out['describe_F1'] = desc
f1 = df[df['grp'] == 'F1']
out['coverage_F1'] = dict(n=int(len(f1)), PTH_ok=float(np.isfinite(f1['PTH']).mean()), rF1_ok=float(np.isfinite(f1['rF1']).mean()),
                          peer_ok=float(np.isfinite(f1['peer']).mean()), n_peer_median=float(f1['n_peer'].median()))
out['F1_top10_ex50'] = [dict(code=r.code, month=r.month, ex50_pct=round(r.ex50 * 100, 1), PTH=round(r.PTH, 3) if np.isfinite(r.PTH) else None,
                             rF1=round(r.rF1, 2) if np.isfinite(r.rF1) else None) for r in f1.sort_values('ex50', ascending=False).head(10).itertuples()]
cols = ['month', 'code', 'grp', 'lma_rank', 'surge', 'ret20', 'PTH', 'pPTH', 'rF1', 'industry', 'ret', 'r50', 'peer', 'n_peer', 'ex50', 'exP', 'how', 'hold_days']
df[cols].to_csv(H / 'f1_pth.csv.gz', index=False, float_format='%.5f')

# ---------------------------------------------------------------- (2) 隨機對照
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
del MC
gc.collect()


def sim(S1, cap=1e6, fresh=False):
    if fresh:
        S1 = s33lib.fresh_from(S1, s33lib.VALID_FIRST_MONTH)
    S2, ekw = s10lib.variant_S(S1, 'E0')
    return s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=cap, **CELL, **ekw)


ratio = json.loads((H / 'results.json').read_text())['meta']['fresh_capital_scale']
RULE = FZ['frozen_rule']
base = {}
for nm, rule in (('CUR', None), ('FZ', RULE), ('FILL_ALL', FZ['control'])):
    S1, _ = s33lib.build_variant(P, U, RT, S0, PS, rule)
    base[nm] = dict(valid=stlib.cagr(sim(S1, 1e6 / ratio, fresh=True)[0], TB, DE), since2019=stlib.cagr(sim(S1)[0], D19, DE))
rng = np.random.default_rng(33)
R = []
for it in range(20):
    S1, _ = s33lib.build_variant(P, U, RT, S0, PS, RULE, rng=rng)
    R.append(dict(valid=stlib.cagr(sim(S1, 1e6 / ratio, fresh=True)[0], TB, DE), since2019=stlib.cagr(sim(S1)[0], D19, DE)))
    del S1
    gc.collect()
rv = np.array([r['valid'] for r in R]); rs = np.array([r['since2019'] for r in R])
out['random_control'] = dict(rule=RULE['id'], how='當月 L_MA 前 20 名的 PTH 值隨機重排（分布與機制同凍結規則）', n=20, seed=33, base=base,
                             valid=dict(median=float(np.median(rv)), p10=float(np.quantile(rv, .1)), p90=float(np.quantile(rv, .9)), n_below_FZ=int((rv < base['FZ']['valid']).sum())),
                             since2019=dict(median=float(np.median(rs)), p10=float(np.quantile(rs, .1)), p90=float(np.quantile(rs, .9)), n_below_FZ=int((rs < base['FZ']['since2019']).sum())),
                             draws=R)
out['meta'] = dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
(H / 'results_describe.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k: out[k] for k in ('coverage_F1', 'random_control', 'meta')}, ensure_ascii=False, default=str)[:3000])
