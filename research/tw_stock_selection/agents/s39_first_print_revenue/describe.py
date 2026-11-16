#!/usr/bin/env python3
"""S39 第 2 步（描述 1）：首次值 FP vs 目前營收檔 CUR（tw_revenue.csv）——更正比例、幅度、方向、產業／月份集中度；
窗內（TL≠FP）vs 窗後（CUR≠TL）拆解；舊月份（首次快照的去年當月營收 V12 vs CUR）。輸出 results.json 的 describe_1 鍵、data/corrections.csv.gz。
python3 -I -B describe.py"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s39lib
from s39lib import s4lib, stlib, s17lib

H = Path(__file__).resolve().parent
t0 = time.time()
RES = H / 'results.json'
res = json.loads(RES.read_text()) if RES.exists() else {}

fp = s39lib.load_fp()
r = pd.read_csv(stlib.REV_CSV, dtype={'code': str}, parse_dates=['month'])
r['m'] = pd.PeriodIndex(r['month'], freq='M')
r['pri'] = (r['market'] != 'sii').astype(int)
r = r.sort_values(['m', 'code', 'pri']).drop_duplicates(['m', 'code'], keep='first')
IND = s39lib.industry_latest()

# 策略股池：訊號列股池內、現行 SURGE 有值
P = s39lib.load()
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
F = s17lib.rev_family(P, defs=['STD'])
S, av = F['S']['STD'], F['av']
pool = set()
for m in s39lib.MONTHS:
    k = av[m]
    if k >= P.T:
        continue
    s = S.loc[m].to_numpy()
    for j in np.flatnonzero(U[k] & np.isfinite(s)):
        pool.add((str(m), P.COLS[j]))


def stats(d, a='FP', b='CUR'):
    d = d[(d[a] > 0) & (d[b] > 0)].copy()
    d['corr'] = d[a].round() != d[b].round()
    d['rel'] = d[b] / d[a] - 1
    c = d[d['corr']]
    ab = c['rel'].abs()
    o = dict(n=int(len(d)), n_corr=int(len(c)), share_corr=float(d['corr'].mean()) if len(d) else None,
             up=int((c['rel'] > 0).sum()), down=int((c['rel'] < 0).sum()),
             abs_rel_bins={'<0.1%': int((ab < 0.001).sum()), '0.1~1%': int(((ab >= 0.001) & (ab < 0.01)).sum()),
                           '1~10%': int(((ab >= 0.01) & (ab < 0.1)).sum()), '>=10%': int((ab >= 0.1).sum())},
             rel_quantiles={str(q): float(c['rel'].quantile(q)) for q in (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)} if len(c) else {},
             abs_rel_median=float(ab.median()) if len(c) else None, abs_rel_mean=float(ab.mean()) if len(c) else None)
    return o, d


out = {}
# 主比較：FP vs CUR（上市；同代碼優先上市後 market == sii）
x = fp.merge(r[['m', 'code', 'market', 'revenue_k_twd']], on=['m', 'code'], how='outer', indicator=True)
x = x[x['m'].isin(set(s39lib.MONTHS))]
x = x.rename(columns={'revenue_k_twd': 'CUR'})
x['ind'] = x['code'].map(IND).fillna('（無產業資料）')
cov = {}
for mk in ('sii', 'otc'):
    y = x[(x['market'] == mk) | (x['market'].isna())]
    cov[mk] = dict(only_cur=int(((y['_merge'] == 'right_only') & (y['market'] == mk)).sum()), both=int(((y['_merge'] == 'both') & (y['market'] == mk)).sum()))
cov['only_fp_any_market'] = int((x['_merge'] == 'left_only').sum())
cov['only_fp_in_P_COLS'] = int(((x['_merge'] == 'left_only') & x['code'].isin(set(P.COLS))).sum())
cov['note'] = 'Tasi 快照其實含上市＋上櫃（S36 註記只有上市不正確，但對上市股池無影響）；only_fp 多為合計列、新上市或研究中心營收檔缺的公司（如 2881／2882／2884 等金控部分月份）。'
out['coverage'] = cov

sii = x[(x['_merge'] == 'both') & (x['market'] == 'sii')].copy()
sii['in_pool'] = [(str(m), c) in pool for m, c in zip(sii['m'], sii['code'])]
o_all, dall = stats(sii)
o_pool, dpool = stats(sii[sii['in_pool']])
o_otc, _ = stats(x[(x['_merge'] == 'both') & (x['market'] == 'otc')])
out['fp_vs_cur'] = dict(sii_all=o_all, strategy_pool=o_pool, otc_reference=o_otc)
# 窗內／窗後
w = dall.copy()
w['in_win'] = w['FP'].round() != w['TL'].round()
w['post_win'] = w['TL'].round() != w['CUR'].round()
out['decompose'] = dict(n=int(len(w)), in_window_changed=int(w['in_win'].sum()), post_window_changed=int(w['post_win'].sum()),
                        corrected_only_in_window=int((w['corr'] & w['in_win'] & ~w['post_win']).sum()),
                        corrected_post_window=int((w['corr'] & w['post_win']).sum()),
                        tl_equals_cur_share=float((~w['post_win']).mean()))
# 依月份、產業
bym = dall.groupby(dall['m'].astype(str)).agg(n=('corr', 'size'), n_corr=('corr', 'sum'), share=('corr', 'mean'),
                                               up=('rel', lambda s: int((s > 0).sum())), down=('rel', lambda s: int((s < 0).sum())))
bym_pool = dpool.groupby(dpool['m'].astype(str))['corr'].agg(['size', 'sum'])
bym['pool_n'] = bym_pool['size']; bym['pool_corr'] = bym_pool['sum']
out['by_month'] = {k: {kk: (float(vv) if kk == 'share' else int(vv)) for kk, vv in v.items()} for k, v in bym.fillna(0).to_dict('index').items()}
# 營收月份距今（月）與更正率的關係
age = dall.groupby(dall['m'].astype(str))['corr'].mean()
out['by_month_share_first_half_vs_second'] = dict(first14=float(dall[dall['m'] <= pd.Period('2025-07', 'M')]['corr'].mean()),
                                                  last14=float(dall[dall['m'] > pd.Period('2025-07', 'M')]['corr'].mean()))
bi = dall.groupby('ind').agg(n=('corr', 'size'), n_corr=('corr', 'sum'), share=('corr', 'mean'))
bi = bi[bi['n'] >= 200].sort_values('share', ascending=False)
out['by_industry_top'] = {k: dict(n=int(v['n']), n_corr=int(v['n_corr']), share=float(v['share'])) for k, v in bi.head(12).iterrows()}
out['by_industry_bottom'] = {k: dict(n=int(v['n']), n_corr=int(v['n_corr']), share=float(v['share'])) for k, v in bi.tail(5).iterrows()}
# 產業集中度卡方近似（描述）：各產業更正率的離散度
p0 = dall['corr'].mean()
bi2 = dall.groupby('ind')['corr'].agg(['size', 'sum'])
exp = bi2['size'] * p0
chi = float((((bi2['sum'] - exp) ** 2) / exp.clip(lower=1e-9)).sum() + (((bi2['size'] - bi2['sum']) - (bi2['size'] - exp)) ** 2 / (bi2['size'] - exp).clip(lower=1e-9)).sum())
out['industry_dispersion'] = dict(chi2=chi, df=int(len(bi2) - 1), overall_share=float(p0))
# 公司集中度：被更正的股月有多少家公司、重複更正者
cc = dall[dall['corr']].groupby('code').size().sort_values(ascending=False)
out['company_concentration'] = dict(n_companies_corrected=int(len(cc)), n_companies=int(dall['code'].nunique()),
                                    top10=[dict(code=c, n=int(n), ind=IND.get(c, '')) for c, n in cc.head(10).items()],
                                    share_corr_from_repeaters=float(cc[cc >= 3].sum() / max(cc.sum(), 1)))
# 大幅更正明細（|rel| ≥ 10%）
big = dall[dall['corr'] & (dall['rel'].abs() >= 0.10)].sort_values('rel')
out['big_corrections'] = [dict(code=y.code, m=str(y.m), FP=float(y.FP), TL=float(y.TL), CUR=float(y.CUR), rel=float(y.rel), in_pool=bool(y.in_pool), ind=y.ind)
                          for y in big.itertuples()]
# 更正性質分類：用 m+12 首次快照的「去年當月營收」（同一來源 MOPS 一年後顯示的 m 營收）
vv = fp[['code', 'm', 'V12']].copy(); vv['m'] = vv['m'] - 12
cl = dall[dall['corr']].drop(columns=['V12', 'V1']).merge(vv, on=['code', 'm'], how='left')
def cls(y):
    if not np.isfinite(y.V12):
        return '無一年後資料（m ≥ 2025-10）'
    if round(y.V12) == round(y.CUR):
        return 'MOPS 事後更正（一年後 MOPS 顯示＝目前值）'
    if round(y.V12) == round(y.FP):
        return '一年後 MOPS 仍顯示首次值（研究中心檔來源／口徑不同，或一年後才改）'
    return '三者皆不同'
cl['cls'] = [cls(y) for y in cl.itertuples()]
out['correction_nature'] = {k: dict(n=int(len(g)), in_pool=int(g['in_pool'].sum()), abs_rel_median=float(g['rel'].abs().median()))
                            for k, g in cl.groupby('cls')}
# 舊月份：首次快照的「去年當月營收」V12 vs CUR(m−12)
v = fp[['code', 'm', 'V12']].copy(); v['m'] = v['m'] - 12
v = v.merge(r[r['market'] == 'sii'][['m', 'code', 'revenue_k_twd']], on=['m', 'code']).rename(columns={'revenue_k_twd': 'CUR'})
o_old, dold = stats(v, 'V12', 'CUR')
o_old['months'] = f"{v['m'].min()}～{v['m'].max()}"
o_old['note'] = 'V12＝m 首次公布當時 MOPS 顯示的 m−12 營收（已含 m−12 到那時為止的更正）；與 CUR 不同＝一年以後又被改，或兩個來源不同。'
out['older_months_v12_vs_cur'] = o_old
# V12 vs 該月自己的首次值（m−12 ≥ 2024-06 時）：一年內的更正
v2 = fp[['code', 'm', 'V12']].copy(); v2['m'] = v2['m'] - 12
v2 = v2.merge(fp[['code', 'm', 'FP']], on=['code', 'm'])
o_y, _ = stats(v2, 'FP', 'V12')
o_y['note'] = 'FP(m) vs 一年後公告 m+12 時顯示的 m 營收（同一來源 MOPS，排除兩檔案來源差異）'
out['fp_vs_one_year_later_same_source'] = o_y
res['describe_1'] = out
res['describe_1_runtime_s'] = round(time.time() - t0, 1)
RES.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
dall[dall['corr']][['code', 'm', 'FP', 'TL', 'CUR', 'rel', 'in_pool', 'ind', 'T_fp', 'n_chg_window']].assign(m=lambda z: z['m'].astype(str)).to_csv(H / 'data' / 'corrections.csv.gz', index=False)
pr = {k: v for k, v in out.items() if k not in ('by_month', 'big_corrections')}
print(json.dumps(pr, ensure_ascii=False, indent=1, default=str)[:9000])
print('big', len(out['big_corrections']), 'maxrss MB', round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
