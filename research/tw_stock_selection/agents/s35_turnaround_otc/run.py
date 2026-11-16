#!/usr/bin/env python3
"""S35 主程式（PREREG.json 寫好、sha256 記下之後才跑）。python3 -I -B run.py → results.json、f1_otc.csv.gz
上櫃 F1 逐筆（位移 0；E0 持有期；s20lib.outcomes，不扣成本）：
- ex50＝持有期報酬 − 0050 同期；exP＝持有期報酬 − 同業等權同期（s35lib.peer_returns）。
- 主要檢定：exP ~ 合成指標 TA（三者 ≥ 2），OLS、訊號月群聚（CR1）、單尾 α＝5%，df＝月數−1。
- 次要：ex50 同法；兩段（2016～2020、2021～2026）方向；三個單一指標（Holm，探索）；同月內配對差；去掉最賺 5 筆；上櫃除權息悲觀界敏感度。
- 描述：谷底翻揚比例、最賺 10 筆裡幾筆是谷底翻揚、各組中位數與勝率。
- 若主要檢定通過：上市前瞻紙上追蹤的統計力估算（PREREG followup）。"""
import sys, json, time, resource, gc, math
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s35lib
from s35lib import s20lib, s17lib, s10lib, stlib, cslib, cluster_diff, within_month_diff, grp_stats

H = Path(__file__).resolve().parent
PR = json.loads((H / 'PREREG.json').read_text())
PR_SHA = (H / 'PREREG.sha256').read_text().split()[0]
import hashlib
assert hashlib.sha256((H / 'PREREG.json').read_bytes()).hexdigest() == PR_SHA, 'PREREG.json 與 sha256 不符'
ALPHA = PR['primary']['alpha_one_sided']
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s35lib.load()
UU = s35lib.universes(P)
U, UALL = UU['OTC'], UU['ALL']
del UU
RT = s17lib.rev_family(P, defs=['STD'])
RT = dict(S=RT['S']['STD'], av=RT['av'])
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
G = s20lib.growth_features(P)
RO = s35lib.ROE(P)
df = s35lib.f1_table(P, U, RT, S0, G, RO)
del G, RO
df = df[df['grp'] == 'F1'].reset_index(drop=True)
ks_all = [x['k'] for x in S0['info']]
IND, NAMES, SRC = s35lib.industry_matrix(P, sorted(set(df['k'])))
for a in ('H', 'L', 'VOL', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'F_HI', 'IDXC', 'AO', 'NEARHI', 'F_MA20', 'PTH'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
print('prep', len(df), f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)


def attach(df0):
    o = s20lib.outcomes(P, S0['X'], ks_all, df0, P.T - 1)
    d = df0.drop(columns=[c for c in ('ret', 'r50', 'exit_t', 'how', 'limup_open', 'excess', 'hold_days', 'peer', 'n_peer', 'ex50', 'exP') if c in df0]).join(o)
    peer, npeer = s35lib.peer_returns(P, d, UALL, IND, P.V60)
    d['peer'] = peer; d['n_peer'] = npeer
    d['ex50'] = d['excess']; d['exP'] = d['ret'] - d['peer']
    return d


SMOKE = '--smoke' in sys.argv                              # 冒煙測試：報酬全部換成亂數（不看真實報酬），輸出到 scratchpad
_rng = np.random.default_rng(0)


def smoke(d):
    if SMOKE:
        for col in ('ret', 'ex50', 'exP', 'peer', 'r50', 'excess'):
            if col in d:
                d[col] = np.where(np.isfinite(d[col]), _rng.normal(0, 0.2, len(d)), np.nan)
    return d


df = smoke(attach(df))
df = df[np.isfinite(df['ret'])].reset_index(drop=True)
print('outcomes', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
out = {'PREREG_sha256': PR_SHA, 'note': '上櫃 F1 逐筆，E0 持有期，不扣成本；超額單位為小數（_pct 為 %）'}


def test_block(d, ind):
    r = {}
    for col in ('exP', 'ex50'):
        dd = d[np.isfinite(d[col])]
        r[col] = dict(cluster=cluster_diff(dd[col], dd[ind].astype(float), dd['month']),
                      within_month=within_month_diff(dd[col].to_numpy(), dd[ind].to_numpy(), dd['month'].to_numpy()),
                      grp_TA=grp_stats(dd.loc[dd[ind], col]), grp_other=grp_stats(dd.loc[~dd[ind], col]))
    return r


# ---------------------------------------------------------------- 主要與次要
prim = test_block(df, 'TA')
c = prim['exP']['cluster']
passed = bool(np.isfinite(c['t']) and c['diff'] > 0 and c['p_one'] < ALPHA)
out['primary'] = dict(spec=PR['primary']['statistic'], result=c, passed=passed)
out['secondary_ex50'] = prim['ex50']['cluster']
out['within_month'] = {k: prim[k]['within_month'] for k in ('exP', 'ex50')}
halves = {}
for h, g in df.groupby('half'):
    tb = test_block(g, 'TA')
    halves[h] = {col: tb[col]['cluster'] for col in ('exP', 'ex50')}
out['halves'] = halves
out['halves_same_direction'] = {col: bool(all(halves[h][col]['diff'] > 0 for h in halves) or all(halves[h][col]['diff'] < 0 for h in halves)) for col in ('exP', 'ex50')}
out['halves_both_positive'] = {col: bool(all(halves[h][col]['diff'] > 0 for h in halves)) for col in ('exP', 'ex50')}
singles = {}
for ind in ('PTH_LOW', 'MU_LE0', 'ROE_LOW'):
    tb = test_block(df, ind)
    singles[ind] = {col: tb[col]['cluster'] for col in ('exP', 'ex50')}
    singles[ind]['halves_exP'] = {h: cluster_diff(g['exP'], g[ind].astype(float), g['month'])['diff'] for h, g in df[np.isfinite(df['exP'])].groupby('half')}
for col in ('exP', 'ex50'):
    ps = sorted(((singles[i][col]['p_one'], i) for i in singles), key=lambda x: x[0])
    m = len(ps); run = 0.0
    for r_, (p, i) in enumerate(ps):
        run = max(run, min(1.0, (m - r_) * p))
        singles[i][col]['p_holm'] = run
out['single_indicators'] = singles
# 去掉最賺 5 筆（依 exP）後
d5 = df[np.isfinite(df['exP'])].sort_values('exP', ascending=False).iloc[5:]
out['drop_top5_exP'] = cluster_diff(d5['exP'], d5['TA'].astype(float), d5['month'])
# 中位數差（描述）與 2 檔以上同月
dd = df[np.isfinite(df['exP'])]
out['median_diff_exP_pct'] = float((dd.loc[dd['TA'], 'exP'].median() - dd.loc[~dd['TA'], 'exP'].median()) * 100)
out['n_ind_dose'] = {int(k): grp_stats(g['exP']) for k, g in dd.groupby('n_ind')}
print('tests done', flush=True)

# ---------------------------------------------------------------- 上櫃除權息悲觀界（訊號、名單不變；只換持有期與同業報酬的價格序列）
C, has = P.C, P.has
CFt = pd.DataFrame(np.where(has, C, np.nan)).ffill().to_numpy()
prev = np.vstack([np.full((1, P.N), np.nan), CFt[:-1]])
z = (np.nan_to_num(P.CHG, nan=0.0) == 0) & has & np.isfinite(prev) & (np.abs(C - prev) > 1e-9) & (P.MK == 2)
R2 = np.where(z, C / prev - 1, P.R)
R2 = np.where(np.abs(R2) > 0.5, P.R, R2)
G0, REF0 = P.G, P.REF
P.G = np.cumprod(1 + R2, axis=0); P.REF = np.where(z, prev, REF0)
del R2, CFt
dfs = smoke(attach(df[['month', 'k', 'code', 'j', 'grp', 'TA', 'half']].copy()))
P.G, P.REF = G0, REF0
del G0, REF0, prev, z
gc.collect()
ds = dfs[np.isfinite(dfs['exP'])]
out['sensitivity_otc_exright_pessimistic'] = dict(how='上櫃漲跌價差＝0 但收盤對前一成交收盤有變的格子改用收盤÷前一成交收盤（股利當跌）；名單不變',
                                                 exP=cluster_diff(ds['exP'], ds['TA'].astype(float), ds['month']),
                                                 ex50=cluster_diff(dfs['ex50'], dfs['TA'].astype(float), dfs['month']))
print('sens', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

# ---------------------------------------------------------------- 描述
desc = dict(n_F1=int(len(df)), n_TA=int(df['TA'].sum()), share_TA=float(df['TA'].mean()),
            share_single={i: float(df[i].mean()) for i in ('PTH_LOW', 'MU_LE0', 'ROE_LOW')},
            share_TA_by_half={h: float(g['TA'].mean()) for h, g in df.groupby('half')},
            peer_coverage=float(np.isfinite(df['exP']).mean()), n_peer_median=float(df['n_peer'].median()),
            how=df['how'].value_counts().to_dict(), hold_days_median=float(df['hold_days'].median()))
for col in ('exP', 'ex50', 'ret'):
    top = df[np.isfinite(df[col])].sort_values(col, ascending=False).head(10)
    desc[f'top10_{col}'] = dict(n_TA=int(top['TA'].sum()), rows=[dict(code=r.code, month=r.month, val_pct=round(getattr(r, col) * 100, 1), TA=bool(r.TA),
                                                                       PTH_LOW=bool(r.PTH_LOW), MU_LE0=bool(r.MU_LE0), ROE_LOW=bool(r.ROE_LOW),
                                                                       PTH=None if not np.isfinite(r.PTH) else round(r.PTH, 3),
                                                                       mu=None if not np.isfinite(r.mu) else round(r.mu), ROE=None if not np.isfinite(r.ROE) else round(r.ROE, 3))
                                                                  for r in top.itertuples()])
    bot = df[np.isfinite(df[col])].sort_values(col).head(10)
    desc[f'bottom10_{col}'] = dict(n_TA=int(bot['TA'].sum()))
    desc[f'groups_{col}'] = dict(TA=grp_stats(df.loc[df['TA'], col]), other=grp_stats(df.loc[~df['TA'], col]), all=grp_stats(df[col]))
    desc[f'groups_{col}_by_half'] = {h: dict(TA=grp_stats(g.loc[g['TA'], col]), other=grp_stats(g.loc[~g['TA'], col])) for h, g in df.groupby('half')}
# 大贏家（exP 前 10%）中谷底翻揚比例
q90 = df['exP'].quantile(0.9)
bw = df[df['exP'] >= q90]
desc['big_winners_top10pct_exP'] = dict(threshold_pct=float(q90 * 100), n=int(len(bw)), share_TA=float(bw['TA'].mean()), base_share_TA=float(df.loc[np.isfinite(df['exP']), 'TA'].mean()))
out['describe'] = desc

# ---------------------------------------------------------------- 前瞻紙上追蹤統計力（只在通過時；PREREG followup）
if passed:
    fu = PR['followup']
    v = dd['exP'].to_numpy(); sd = float(v.std(ddof=1))
    iid_se = sd * math.sqrt(1 / c['n1'] + 1 / c['n0'])
    deff = (c['se'] / iid_se) ** 2
    from scipy import stats
    z_a, z_b = stats.norm.ppf(1 - ALPHA), stats.norm.ppf(fu['power'])
    pw = {}
    for nm, eff in (('observed_otc', c['diff']), ('half_observed', c['diff'] / 2)):
        rr = {}
        for npm in fu['listed_f1_per_month_scenarios']:
            n_per = npm; f = desc['share_TA']
            # se(M) ＝ sd·√(deff)·√(1/(f·n·M)＋1/((1−f)·n·M))
            M = (z_a + z_b) ** 2 * sd ** 2 * deff * (1 / f + 1 / (1 - f)) / (n_per * eff ** 2)
            rr[str(npm)] = dict(months=float(M), years=float(M / 12))
        pw[nm] = dict(effect_pct=float(eff * 100), by_f1_per_month=rr)
    out['followup_power'] = dict(sd_exP_pct=sd * 100, design_effect=float(deff), share_TA=desc['share_TA'], alpha=ALPHA, power=fu['power'], result=pw,
                                 note='以上櫃樣本的效果與變異估算；上市 F1 每月檔數取情境值；假設上市與上櫃變異相近')
cols = ['month', 'code', 'mk', 'lma_rank', 'surge', 'ret20', 'PTH', 'mu', 'mu_sigma', 'ROE', 'roe_lagq', 'PTH_LOW', 'MU_LE0', 'ROE_LOW', 'n_ind', 'TA', 'half',
        'ret', 'r50', 'peer', 'n_peer', 'ex50', 'exP', 'how', 'hold_days']
OUTD = s35lib.S35 if SMOKE else H
df[cols].to_csv(OUTD / 'f1_otc.csv.gz', index=False, float_format='%.5f')
out['meta'] = dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=rss(), panel_last=str(P.IDX[-1].date()))
(OUTD / ('smoke_results.json' if SMOKE else 'results.json')).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
if SMOKE:
    print('SMOKE OK', out['meta'])
    sys.exit(0)
print(json.dumps({k: out[k] for k in ('primary', 'secondary_ex50', 'halves', 'meta')}, ensure_ascii=False, default=str)[:4000])
