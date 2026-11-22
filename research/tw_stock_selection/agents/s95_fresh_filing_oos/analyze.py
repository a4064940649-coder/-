#!/usr/bin/env python3
"""S95 主檢定（依 PREREG.json）。python3 -I -B analyze.py → results.json、trades_s95.csv.gz"""
import json, glob, hashlib
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; AG = H.parent; RAW = AG.parent / 'raw' / 'tsd_extra'
assert hashlib.sha256((H / 'PREREG.json').read_bytes()).hexdigest() == (H / 'PREREG.sha256').read_text().split()[0]
NB = 10000
e50 = pd.read_csv(AG / 's75_continuous_2008_2026/v2_2026-11-21/e50_spliced.csv', parse_dates=['date']).set_index('date')['adj']

def prep(t):
    t = t.copy()
    for c in ('sig_date', 'exit_date'):
        t[c] = pd.to_datetime(t[c])
    t['code'] = t['code'].astype(str)
    t['b50'] = e50.reindex(t.exit_date, method='ffill').to_numpy() / e50.reindex(t.sig_date, method='ffill').to_numpy() - 1   # 0050 缺日用前一日收盤
    assert t.b50.notna().all()
    t['ex'] = t.ret - t.b50
    t['m'] = t.sig_date.dt.month
    return t

v2 = pd.read_csv(AG / 's75_continuous_2008_2026/v2_2026-11-21/trades_v2.csv.gz')
oos = prep(v2[(v2.version == 'V_BRIDGE') & (v2.sig_date <= '2015-12-31')])
oosm = prep(v2[(v2.version == 'V_MASK') & (v2.sig_date <= '2012-12-31')])
ins = prep(pd.read_csv(AG / 's92_receivables_inventory_filing/trades_features.csv.gz'))
# 公告日
fd = pd.concat([pd.read_parquet(f, columns=['code', 'year', 'season', 'uploaded_at']) for f in sorted(glob.glob(str(RAW / 'filing_dates/*/*.parquet')))])
fd['code'] = fd.code.astype(str)
F = fd.groupby(['code', 'year', 'season']).uploaded_at.min().dt.normalize().reset_index()
Fd = {c: np.sort(g.uploaded_at.to_numpy('datetime64[D]')) for c, g in F.groupby('code')}
def since(r):
    a = Fd.get(r.code)
    if a is None: return np.nan
    i = np.searchsorted(a, np.datetime64(r.sig_date.date(), 'D'), 'left')
    return np.nan if i == 0 else float((np.datetime64(r.sig_date.date(), 'D') - a[i - 1]).astype(int))
for t in (oos, ins):
    t['since'] = [since(r) for r in t.itertuples()]
agree = dict(n=int(ins.since_filing.notna().sum()), same=int((ins.since == ins.since_filing).sum()),
             same_bucket=int(((ins.since <= 10) == (ins.since_filing <= 10))[ins.since_filing.notna()].sum()))

def stats(ex, lab):
    a, b = ex[lab == 1], ex[lab == 0]
    return float(a.mean() - b.mean()), float(np.median(a) - np.median(b))
def desc(t, lab):
    o = {}
    for nm, v in (('treat', 1), ('ctrl', 0)):
        x = t.ex[lab == v]; r = t.ret[lab == v]
        o[nm] = dict(n=int(len(x)), n_clusters=int(t.sig_date[lab == v].nunique()), mean_ex=float(x.mean()), median_ex=float(x.median()), mean_ret=float(r.mean()), win=float((r > 0).mean()), win_vs0050=float((x > 0).mean())) if len(x) else dict(n=0)
    return o
def h1(t, months_by_stratum, seed=95):
    """months_by_stratum: list of (mask, months)。分層聚類置換。"""
    t = t.reset_index(drop=True); lab = np.full(len(t), -1); strat = np.full(len(t), -1)
    for s, (mk, ms) in enumerate(months_by_stratum):
        mk = mk(t).to_numpy(); lab[mk] = t.m[mk].isin(ms).astype(int); strat[mk] = s
    assert (lab >= 0).all()
    ex = t.ex.to_numpy(); cl, ci = np.unique(t.sig_date.to_numpy(), return_inverse=True)
    cl_lab = np.array([lab[ci == i][0] for i in range(len(cl))]); cl_st = np.array([strat[ci == i][0] for i in range(len(cl))])
    d, md = stats(ex, lab)
    cm = np.array([ex[ci == i].mean() for i in range(len(cl))]); dcl = float(cm[cl_lab == 1].mean() - cm[cl_lab == 0].mean())
    rng = np.random.default_rng(seed); ge = gem = gec = 0
    for _ in range(NB):
        p = cl_lab.copy()
        for s in np.unique(cl_st):
            ix = np.flatnonzero(cl_st == s); p[ix] = rng.permutation(cl_lab[ix])
        pl = p[ci]; a, b = stats(ex, pl)
        ge += a >= d; gem += b >= md; gec += (cm[p == 1].mean() - cm[p == 0].mean()) >= dcl
    o = desc(t, pd.Series(lab)); o.update(diff_mean=d, p_mean=(ge + 1) / (NB + 1), diff_median=md, p_median=(gem + 1) / (NB + 1), diff_clustermean=dcl, p_clustermean=(gec + 1) / (NB + 1))
    # 描述：拿掉報酬最高 5 筆
    k = np.argsort(-t.ret.to_numpy())[5:]; o['drop_top5_diff_mean'] = stats(ex[k], lab[k])[0]
    return o
def h2(t, seed=95):
    t = t[t.since.notna()].reset_index(drop=True)
    lab = np.where(t.since <= 10, 1, np.where(t.since > 60, 0, -1)); ex = t.ex.to_numpy()
    o = desc(t, pd.Series(lab)); o['n_mid_11_60'] = int((lab == -1).sum()); o['mid_mean_ex'] = float(ex[lab == -1].mean()) if (lab == -1).any() else None
    if o['treat']['n'] == 0 or o['ctrl']['n'] == 0: return o
    d, md = stats(ex, lab); cl, ci = np.unique(t.sig_date.to_numpy(), return_inverse=True)
    idx = [np.flatnonzero(ci == i) for i in range(len(cl))]
    rng = np.random.default_rng(seed); D = []; M = []; bad = 0
    for _ in range(NB):
        s = np.concatenate([idx[i] for i in rng.integers(0, len(cl), len(cl))])
        if (lab[s] == 1).sum() == 0 or (lab[s] == 0).sum() == 0: bad += 1; continue
        a, b = stats(ex[s], lab[s]); D.append(a); M.append(b)
    D = np.array(D); M = np.array(M)
    o.update(diff_mean=d, p_mean=float((D <= 0).mean()), ci90_mean=[float(np.quantile(D, .05)), float(np.quantile(D, .95))], diff_median=md, p_median=float((M <= 0).mean()), n_boot_discarded=bad)
    keep = lab >= 0; k = np.flatnonzero(keep)[np.argsort(-t.ret.to_numpy()[keep])[5:]]; o['drop_top5_diff_mean'] = stats(ex[k], lab[k])[0]
    return o

Y = lambda a, b: (lambda t: (t.sig_date.dt.year >= a) & (t.sig_date.dt.year <= b))
NEW = [3, 5, 8, 11]; OLD = [5, 9, 11]
sub = lambda t, a, b: t[Y(a, b)(t)]
R = dict(prereg_sha256=(H / 'PREREG.sha256').read_text().split()[0], since_agree_with_s92=agree, H1={}, H2={})
R['H1']['OOS_2008_2012_主要(5,9,11)'] = h1(sub(oos, 2008, 2012), [(Y(2008, 2012), OLD)])
R['H1']['OOS_2013_2015(3,5,8,11)'] = h1(sub(oos, 2013, 2015), [(Y(2013, 2015), NEW)])
R['H1']['OOS_2008_2015_合併'] = h1(oos, [(Y(2008, 2012), OLD), (Y(2013, 2015), NEW)])
R['H1']['次要_2008_2012(4,5,9,11)'] = h1(sub(oos, 2008, 2012), [(Y(2008, 2012), [4, 5, 9, 11])])
R['H1']['次要_2008_2012(3,5,8,11)'] = h1(sub(oos, 2008, 2012), [(Y(2008, 2012), NEW)])
R['H1']['敏感度_V_MASK_2008_2012(5,9,11)'] = h1(oosm, [(Y(2008, 2012), OLD)])
R['H1']['IS_2016_2021'] = h1(sub(ins, 2016, 2021), [(Y(2016, 2021), NEW)])
R['H1']['IS_2022_2026'] = h1(sub(ins, 2022, 2026), [(Y(2022, 2026), NEW)])
R['H1']['IS_2016_2026'] = h1(ins, [(Y(2016, 2026), NEW)])
R['H2']['OOS_2013_2015'] = h2(sub(oos, 2013, 2015)); R['H2']['OOS_2013_2015']['n_no_filing_excluded'] = int(sub(oos, 2013, 2015).since.isna().sum())
R['H2']['IS_2016_2021'] = h2(sub(ins, 2016, 2021)); R['H2']['IS_2022_2026'] = h2(sub(ins, 2022, 2026)); R['H2']['IS_2016_2026'] = h2(ins)
# 描述：分月
def bym(t):
    g = t.groupby('m'); return {int(k): dict(n=int(len(x)), mean_ex=round(float(x.ex.mean()), 4), mean_ret=round(float(x.ret.mean()), 4)) for k, x in g}
R['by_signal_month'] = {k: bym(sub(t, a, b)) for k, (t, a, b) in {'2008_2012': (oos, 2008, 2012), '2013_2015': (oos, 2013, 2015), '2016_2021': (ins, 2016, 2021), '2022_2026': (ins, 2022, 2026)}.items()}
(H / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
pd.concat([oos.assign(src='s75v2_V_BRIDGE'), ins.assign(src='s92_official')])[['src', 'code', 'sig_date', 'exit_date', 'ret', 'b50', 'ex', 'm', 'since']].round(6).to_csv(H / 'trades_s95.csv.gz', index=False)
for h in ('H1', 'H2'):
    for k, v in R[h].items():
        print(h, k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items() if a not in ('treat', 'ctrl')}, '\n   T', {a: round(b, 4) for a, b in v['treat'].items()}, '\n   C', {a: round(b, 4) for a, b in v['ctrl'].items()})
print(agree); print(json.dumps(R['by_signal_month']))
