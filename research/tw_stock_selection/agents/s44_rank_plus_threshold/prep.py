#!/usr/bin/env python3
"""S44 第 0 步（描述，不算任何策略報酬）：
(1) 每月 L_MA 第 10 名（與第 1、20 名）的 SURGE 分布（全期、依年、依日曆月份；1、2 月另列）；
(2) F1 名單（位移 0）中 SURGE < 2、< 2.5 的比例與月份；F1 在當月全股池的 SURGE 分位分布；
(3) 各候選規則（ABS 1.5／2.0／2.5、REL 第 95／97.5／99 百分位）在訓練期、驗證期會剔除的 F1 檔數與比例（只數檔數）；
(4) 驗收 A1：本檔 f1_rows 的 F1 名單與 make_signal_s10 的 info 相同；pct 與 L_MA 名次合理。
輸出 data_check.json、describe_months.csv.gz、describe_f1.csv.gz。python3 -I -B prep.py"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s44lib
from s44lib import s10lib

H = Path(__file__).resolve().parent
t = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s44lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
U, RT = s44lib.base(P)
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
F1, MO = s44lib.f1_rows(P, U, RT, S0)
print('載入', f'{time.time()-t:.0f}s', rss(), 'MB', flush=True)
chk = dict(panel=dict(T=P.T, N=P.N, last=str(P.IDX[-1].date()), exright_fix=P.exright_fix))
chk['A1_f1_matches_info'] = bool(sum(len(x['targets']) for x in S0['info']) == len(F1) and bool((F1['rank_ma'] <= 10).all()))
F1['period'] = np.where(F1['mper'] <= s44lib.TRAIN_LAST_MONTH, 'train', 'valid')
MO['period'] = np.where(MO['mper'] <= s44lib.TRAIN_LAST_MONTH, 'train', 'valid')
MO['year'] = MO['mper'].dt.year; MO['cm'] = MO['mper'].dt.month
F1['year'] = F1['mper'].dt.year; F1['cm'] = F1['mper'].dt.month
first = pd.Period('2016-01', 'M')
MOw = MO[MO['mper'] >= first]; F1w = F1[F1['mper'] >= first]


def dist(v):
    v = pd.Series(v).dropna()
    return dict(n=int(len(v)), min=float(v.min()), p10=float(v.quantile(.1)), p25=float(v.quantile(.25)), median=float(v.median()),
                p75=float(v.quantile(.75)), p90=float(v.quantile(.9)), max=float(v.max()), lt2=int((v < 2).sum()), lt25=int((v < 2.5).sum()))


chk['note'] = '營收月份 2016-01 起（訊號日 2016-02 起）＝訓練期起點；2015 年下半年訊號列也列在 describe_months，但不進摘要。'
chk['s10_dist'] = dict(all=dist(MOw['s10']), train=dist(MOw.loc[MOw['period'] == 'train', 's10']), valid=dist(MOw.loc[MOw['period'] == 'valid', 's10']))
chk['s10_by_year'] = {int(y): dict(median=float(g['s10'].median()), min=float(g['s10'].min()), max=float(g['s10'].max()), n_lt2=int((g['s10'] < 2).sum()),
                                   n_lt25=int((g['s10'] < 2.5).sum()), n=int(len(g))) for y, g in MOw.groupby('year')}
chk['s10_by_calmonth'] = {int(m): dict(median=float(g['s10'].median()), min=float(g['s10'].min()), n_lt2=int((g['s10'] < 2).sum()), n=int(len(g)),
                                       pool_ge2_median=float(g['pool_ge2'].median())) for m, g in MOw.groupby('cm')}
chk['s1_s20_dist'] = dict(s1=dist(MOw['s1']), s20=dist(MOw['s20']))
chk['months_s10_lt2'] = [dict(month=r.month, s10=round(r.s10, 3), s1=round(r.s1, 3), n_tgt=int(r.n_tgt), pool_ge2=round(r.pool_ge2, 4)) for r in MOw[MOw['s10'] < 2].itertuples()]
chk['pool_size'] = dist(MOw['n_pool']) | dict(note='「當月全股池」＝訊號列股池（只上市、排除創新板）且 SURGE 有值；L_MA 再加季線之上。')
chk['n_ma'] = dist(MOw['n_ma'])
chk['corr_s10_pool_ge2'] = float(np.corrcoef(MOw['s10'], MOw['pool_ge2'])[0, 1])

# ---------------------------------------------------------------- F1 中 SURGE < 2、< 2.5
f1d = {}
for per, d in (('all', F1w), ('train', F1w[F1w['period'] == 'train']), ('valid', F1w[F1w['period'] == 'valid'])):
    f1d[per] = dict(n=int(len(d)), n_months_with_f1=int(d['month'].nunique()), surge=dist(d['surge']),
                    frac_lt15=float((d['surge'] < 1.5).mean()), frac_lt2=float((d['surge'] < 2).mean()), frac_lt25=float((d['surge'] < 2.5).mean()),
                    months_with_lt2=int(d.loc[d['surge'] < 2, 'month'].nunique()), months_with_lt25=int(d.loc[d['surge'] < 2.5, 'month'].nunique()),
                    pct=dist(d['pct']), frac_pct_lt95=float((d['pct'] < .95).mean()), frac_pct_lt975=float((d['pct'] < .975).mean()), frac_pct_lt99=float((d['pct'] < .99).mean()))
chk['f1_surge'] = f1d
chk['f1_lt2_months'] = {m: [dict(code=r.code, surge=round(r.surge, 3), pct=round(r.pct, 4), rank_ma=int(r.rank_ma)) for r in g.itertuples()]
                        for m, g in F1w[F1w['surge'] < 2].groupby('month')}
chk['f1_lt25_months_list'] = sorted(F1w.loc[F1w['surge'] < 2.5, 'month'].unique().tolist())
chk['f1_lt2_by_calmonth'] = {int(m): dict(n=int(len(g)), n_lt2=int((g['surge'] < 2).sum()), n_lt25=int((g['surge'] < 2.5).sum())) for m, g in F1w.groupby('cm')}
chk['f1_lt2_by_year'] = {int(y): dict(n=int(len(g)), n_lt2=int((g['surge'] < 2).sum()), n_lt25=int((g['surge'] < 2.5).sum())) for y, g in F1w.groupby('year')}
chk['pool_percentile_levels'] = {c: dist(MOw[c]) for c in ('p90', 'p95', 'p975', 'p99')}

# ---------------------------------------------------------------- 候選規則剔除檔數（只數）
prev = {}
cands = [dict(id=f'ABS_{th}', kind='ABS', theta=th) for th in s44lib.THETAS] + [dict(id=f'REL_{c}', kind='REL', col=c) for c in ('p90', 'p95', 'p975', 'p99')]
for r in cands:
    kp = s44lib.keep(F1w, r)
    o = {}
    for per in ('train', 'valid'):
        m = (F1w['period'] == per).to_numpy()
        o[per] = dict(removed=int((~kp[m]).sum()), frac=float((~kp[m]).mean()), months=int(F1w.loc[m & ~kp.to_numpy(), 'month'].nunique()),
                      months_emptied=int(sum(1 for mm, g in F1w[m].groupby('month') if not kp[g.index].any())))
    prev[r['id']] = o
chk['rule_effect_preview'] = prev
chk['meta'] = dict(runtime_s=round(time.time() - t, 1), maxrss_mb=rss())
(H / 'data_check.json').write_text(json.dumps(chk, ensure_ascii=False, indent=1, default=str))
MO.drop(columns=['mper']).round(4).to_csv(H / 'describe_months.csv.gz', index=False)
F1.drop(columns=['mper']).assign(sig_date=lambda d: d['sig_date'].dt.strftime('%Y-%m-%d')).round(4).to_csv(H / 'describe_f1.csv.gz', index=False)
print(json.dumps({k: chk[k] for k in ('A1_f1_matches_info', 's10_dist', 'f1_surge', 'rule_effect_preview', 'pool_percentile_levels', 'corr_s10_pool_ge2', 'meta')}, ensure_ascii=False, indent=0)[:6000])
