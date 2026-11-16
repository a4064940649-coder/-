#!/usr/bin/env python3
"""S35 第 0 步（寫 PREREG 之前；不算任何報酬）：上櫃資料檢查。python3 -I -B data_check.py → data_check.json
(1) 營收補洞檔（s16）sha256、併入列數；上櫃 SURGE 有值比例；與 s16lib.rev_tables_s16 的 SURGE（σ>0）逐格比對；
(2) 除權息：EXRIGHT_FIX 只作用在上市（TWT49U）；上櫃「漲跌價差＝0 但收盤對前一成交收盤有變」格數（逐年）與上市修正前後對照；
(3) 上櫃 F1：每月檔數、空月、三個指標的有值比例與成立比例、合成指標比例（只看訊號，不看報酬）；
(4) 同業：產業來源（PIT／快照／缺）、同業可用檔數（只看進場日有開盤，不算報酬）。"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s35lib
from s35lib import s20lib, s17lib, s10lib, stlib, cslib

H = Path(__file__).resolve().parent
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
out = {'note': '資料檢查；沒有計算任何報酬'}
P = s35lib.load()
UU = s35lib.universes(P)
U = UU['OTC']
print('load', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
out['panel'] = dict(T=int(P.T), N=int(P.N), first=str(P.IDX[0].date()), last=str(P.IDX[-1].date()), exright_fix=bool(P.exright_fix), exright_info=P.exright_info)
out['rev_patch'] = dict(patch_csv=str(s35lib.PATCH_CSV), patch_sha256=s35lib.sha256(s35lib.PATCH_CSV), rows_added=int(P.n_patch_rows),
                        rev_orig=str(s35lib.REV_ORIG), rev_patched_sha256=s35lib.sha256(s35lib.REV_PATCHED))

# (1) SURGE
RT = s17lib.rev_family(P, defs=['STD'])
RT = dict(S=RT['S']['STD'], av=RT['av'], n_sigma0=RT['n_sigma0']['STD'])
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
cov = {}
for x in S0['info']:
    m = pd.Period(x['month'], 'M')
    if m < s35lib.FIRST_MONTH:
        continue
    k = x['k']; s = RT['S'].loc[m].to_numpy()
    cov.setdefault(m.year, []).append((int(U[k].sum()), float(np.isfinite(s[U[k]]).mean()) if U[k].any() else np.nan))
out['otc_universe_by_year'] = {y: dict(mean_n=float(np.mean([a for a, _ in v])), surge_ok=float(np.nanmean([b for _, b in v]))) for y, v in sorted(cov.items())}
out['n_sigma0_cells'] = int(RT['n_sigma0'])
sys.path.insert(0, str(s35lib.AG / 's16_surge_otc'))
import s16lib                                              # 只用 rev_tables_s16 對照（不改）
s35lib._redirect()
R16 = s16lib.rev_tables_s16(P, patch=False)               # stlib.REV_CSV 已指向補洞檔 → 等同 patch=True
a = RT['S'].loc[s35lib.FIRST_MONTH:].to_numpy(); b = R16['S'].reindex(index=RT['S'].index, columns=RT['S'].columns).loc[s35lib.FIRST_MONTH:].to_numpy()
otc_cols = (P.MK == 2).any(axis=0)
both = np.isfinite(a) & np.isfinite(b)
out['surge_vs_s16'] = dict(otc_cells_both=int(both[:, otc_cols].sum()), max_abs_diff=float(np.nanmax(np.abs(a - b)[both])),
                           only_s35=int((np.isfinite(a) & ~np.isfinite(b))[:, otc_cols].sum()), only_s16=int((~np.isfinite(a) & np.isfinite(b))[:, otc_cols].sum()))
del R16, a, b
print('surge', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

# (2) 除權息
chg_raw = P.CHG.copy()                                     # 修正後（上市已改）的漲跌價差
C = P.C; has = P.has
CF = pd.DataFrame(np.where(has, C, np.nan)).ffill().to_numpy()
prev = np.vstack([np.full((1, P.N), np.nan), CF[:-1]])
z = (np.nan_to_num(chg_raw, nan=0.0) == 0) & has & np.isfinite(prev) & (np.abs(C - prev) > 1e-9)
yrs = P.IDX.year.to_numpy()
by = {}
for mk, nm in ((2, 'otc'), (1, 'twse_after_fix')):
    zz = z & (P.MK == mk)
    by[nm] = {int(y): int(zz[yrs == y].sum()) for y in np.unique(yrs)}
    by[nm]['total'] = int(zz.sum())
prev_untraded = np.vstack([np.zeros((1, P.N), bool), ~has[:-1]])
zo = z & (P.MK == 2)
dn = zo & ~prev_untraded
out['exright_otc'] = dict(
    zero_chg_price_changed=by,
    otc_split=dict(prev_day_untraded=int((zo & prev_untraded).sum()), prev_day_traded=int(dn.sum()),
                   prev_traded_close_below_prev=float((C[dn] < prev[dn]).mean()) if dn.any() else None,
                   prev_traded_mean_close_over_prev_minus1=float(np.mean(C[dn] / prev[dn] - 1)) if dn.any() else None),
    how='上櫃沒有 TWT49U 參考價，EXRIGHT_FIX 只修上市（MK＝1）。上櫃這些格子 CHG＝0 → 參考價＝收盤 → 當日含息報酬記 0：除權息日不把股利當跌（方向對），'
        '但當天對參考價的真實漲跌遺失；前一日無成交的格子也記 0。上市稽核：除權息日對參考價平均 +0.36%（兩邊相抵後很小）。主要檢定是兩組相減，兩組都受同樣影響；'
        'run.py 另做悲觀界敏感度（上櫃這些格子改用收盤÷前一成交收盤，等於把股利當跌）。')
del chg_raw, CF, prev, z
print('exright', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

# (3) F1 與指標
G = s20lib.growth_features(P)
RO = s35lib.ROE(P)
df = s35lib.f1_table(P, U, RT, S0, G, RO)
f1 = df[df['grp'] == 'F1']
nm = f1.groupby('month').size()
allm = [x['month'] for x in S0['info'] if pd.Period(x['month'], 'M') >= s35lib.FIRST_MONTH]
out['f1'] = dict(n_rows=int(len(f1)), n_signal_months=len(allm), n_empty_months=int(len(allm) - len(nm)), mean_per_month=float(len(f1) / len(allm)),
                 months_with_1=int((nm == 1).sum()), size_dist={int(a): int(b) for a, b in nm.value_counts().sort_index().items()},
                 coverage=dict(PTH=float(np.isfinite(f1['PTH']).mean()), mu=float(np.isfinite(f1['mu']).mean()), ROE=float(np.isfinite(f1['ROE']).mean()),
                               roe_lag_quarters_median=float(np.nanmedian(f1['roe_lagq']))),
                 share=dict(PTH_LOW=float(f1['PTH_LOW'].mean()), MU_LE0=float(f1['MU_LE0'].mean()), ROE_LOW=float(f1['ROE_LOW'].mean()), TA=float(f1['TA'].mean())),
                 n_TA=int(f1['TA'].sum()), n_TA_by_half={h: int(g['TA'].sum()) for h, g in f1.groupby('half')}, n_by_half={h: int(len(g)) for h, g in f1.groupby('half')},
                 corr_indicators=f1[['PTH_LOW', 'MU_LE0', 'ROE_LOW']].astype(float).corr().round(3).to_dict(),
                 months_with_both_groups=int(f1.groupby('month')['TA'].agg(lambda s: s.any() and (~s).any()).sum()))
print('f1', out['f1']['n_rows'], f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

# (4) 同業可用性（不算報酬）
ks = sorted(set(f1['k']))
IND, NAMES, SRC = s35lib.industry_matrix(P, ks)
src = np.array([SRC[int(r.k)][int(r.j)] for r in f1.itertuples()])
npeer = []
for r in f1.itertuples():
    k, j = int(r.k), int(r.j); g = IND[k][j]
    if g < 0:
        npeer.append(0); continue
    c = np.flatnonzero(UU['ALL'][k] & (IND[k] == g)); c = c[c != j]
    e = k + 1
    npeer.append(int(max(0, sum(1 for p in c if np.isfinite(P.O[e, p]) and P.O[e, p] > 0) - 1)))
npeer = np.array(npeer)
out['peer'] = dict(industry_source=dict(pit=int((src == 1).sum()), snapshot=int((src == 2).sum()), missing=int((src == 0).sum())),
                   n_peer_median=float(np.median(npeer)), share_ge5=float((npeer >= s35lib.MIN_PEERS).mean()),
                   industries_F1=f1.assign(ind=[NAMES[IND[int(r.k)][int(r.j)]] if IND[int(r.k)][int(r.j)] >= 0 else '' for r in f1.itertuples()])['ind'].value_counts().head(12).to_dict())
out['meta'] = dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
(H / 'data_check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps(out, ensure_ascii=False, default=str)[:6000])
