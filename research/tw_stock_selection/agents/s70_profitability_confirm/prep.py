#!/usr/bin/env python3
"""S70 第 0 步＋資料檢查（寫 SPEC／PREREG 之前；只算現行，不算任何候選的報酬）：python3 -I -B prep.py → data_check.json
- P0 面板；正式開關全開；A0 現行 100 萬位移 0 逐位重現 48.76%／34.40%／−32.12%／381 筆；A2 現行 20 條路徑與 deadline_engine_fix ALL 逐條相同；
- A1 候選機制：財報全缺值時 C1／C2／C3 的交易與淨值與現行逐位相同（缺值＝保留、順序＝SURGE）；
- 財報時點檢查：名單上的涵蓋率、資料齡、用到的季別；
- 第 0 步描述（現行主跑 100 萬位移 0 實際買進的股票；買進當下＝訊號日當時可得的最近一季）：
    營業利益率、ROE、毛利／資產、是否虧損的分布（全樣本／訓練／驗證都只算「特徵」），並和同期股池比較；
    報酬比較（虧損 vs 不虧損等）在寫 PREREG 前只看訓練期（訊號 2016-02～2021-12）；
- 候選涉及的檔數（不含報酬）：各候選每期剔除檔數、改動月份數；C1 與「最近四季 EPS ≤ 0」的重疊；和 S29（FLT_V／負 E/P 描述）的重疊；
- C3（只改買進順序）的機制檢查：只數「買進的股票集合有沒有不同」，不看報酬。"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s70lib as L
from s70lib import s10lib, stlib, cslib

H = Path(__file__).resolve().parent
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = L.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
U, RT = L.base(P)
fix = L.s17lib.e50_fix_suspension(P)
L.slim(P); gc.collect()
FN = L.Fin(P)
print('載入完成', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19, TB = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01'), lastb('2022-01-01')
dl = json.loads((L.AG / 'deadline_engine_fix' / 'out' / 'results.json').read_text())
cols = ['j', 'ent_t', 'exit_t', 'why', 'shares', 'cost', 'proceeds']
same = lambda a, b: bool(np.array_equal(a[0].to_numpy(), b[0].to_numpy(), equal_nan=True) and a[1][cols].equals(b[1][cols]))
out = dict(flags=dict(EXRIGHT_FIX=cslib.EXRIGHT_FIX, DEADLINE_FIX=cslib.DEADLINE_FIX, INS15_MASK=cslib.INS15_MASK),
           P0=dict(T=P.T, N=P.N, first=str(D0.date()), last=str(DE.date()), train_end=str(TB.date())), e50_fix=fix, paths={})


class NoFin:                                               # 全缺值財報：A1 用
    def row(self, d):
        return {c: np.full(P.N, np.nan) for c in L.FIELDS}


SIG = {}
a1 = True; mech = {}
for s in L.SHIFTS:
    sg = s10lib.make_signal_s10(P, U, RT, shift=s)
    SIG[s] = sg
    S3, _ = L.apply_cand(P, U, FN, sg, 'C3')
    ndiff = 0; nos = 0
    for cap in L.CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        r0 = L.sim(P, sg, U, cap=cap)
        nav, lg, st = r0
        out['paths'][key] = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE))
        r3 = L.sim(P, S3, U, cap=cap)                      # C3 機制檢查：只比買進集合
        b0 = set(zip(lg['sig_t'], lg['j'])); b3 = set(zip(r3[1]['sig_t'], r3[1]['j']))
        ndiff += len(b0 ^ b3); nos += int(st['noslot'])
        mech[key] = dict(noslot=int(st['noslot']), blk_nocash=int(st['blk_nocash']), shrunk=int(st['shrunk']), n_buy=int(st['n_buy']),
                         c3_buy_set_diff=len(b0 ^ b3), c3_trades_identical=same(r0, r3))
        if s == 0 and cap == 1e6:
            main = dict(out['paths'][key], n_closed=int((lg['why'] != 'open_end').sum()))
            LG0 = lg
            for c in L.CANDS:
                Sx, _ = L.apply_cand(P, U, NoFin(), sg, c)
                a1 &= same(r0, L.sim(P, Sx, U, cap=cap))
        del r0, r3
    gc.collect()
    print('位移', s, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
out['A0'] = dict(main, tol_ok=bool(abs(main['since2019'] - L.ACC['since2019']) < 2.5e-4 and abs(main['full'] - L.ACC['full']) < 2.5e-4
                                   and abs(main['maxdd'] - L.ACC['maxdd']) < 2.5e-4 and main['n_closed'] == L.ACC['n_closed']),
                 bitwise_vs_deadline_ALL=bool(main['since2019'] == dl['standard']['ALL']['since2019'] and main['full'] == dl['standard']['ALL']['full']
                                              and main['maxdd'] == dl['standard']['ALL']['maxdd_full']))
d2 = [abs(out['paths'][k]['since2019'] - v['since2019']) for k, v in dl['paths']['ALL'].items()]
out['A2'] = dict(n=len(d2), max_abs_diff=float(max(d2)), ok=bool(len(d2) == 20 and max(d2) < 1e-9))
out['A1_nofin_identical'] = bool(a1)
out['C3_mech'] = dict(note='C3 只改買進順序；順序只有在名額（10 檔）或現金不夠時才有作用。只數買進集合差異，不看報酬。',
                      paths=mech, n_paths_identical=int(sum(v['c3_trades_identical'] for v in mech.values())),
                      total_buy_set_diff=int(sum(v['c3_buy_set_diff'] for v in mech.values())), total_noslot=int(sum(v['noslot'] for v in mech.values())))

# ---------------------------------------------------------------- 名單（F1）逐檔特徵
S0 = SIG[0]
rows = []; urows = []
for x in S0['info']:
    k = x['k']; r = L.features(P, U, FN, k)
    um = U[k]
    opi = r['opi_q'][um]; nt = r['ni_ttm'][um]
    urows.append(dict(month=x['month'], k=k, n_univ=int(um.sum()), cov_opi=float(np.isfinite(opi).mean()),
                      loss_op=float((opi[np.isfinite(opi)] < 0).mean()), loss_ttm=float((nt[np.isfinite(nt)] <= 0).mean()),
                      med_opm=float(np.nanmedian(r['opm_q'][um])), med_roe=float(np.nanmedian(r['roe'][um])), med_gpa=float(np.nanmedian(r['gpa'][um]))))
    for c in x['targets']:
        j = P.ci[c]; fl = L.flags(r, j)
        rows.append(dict(month=x['month'], k=k, sig_date=x['sig_date'], code=c, j=j, surge=float(S0['rank'][k, j]), **{f: float(r[f][j]) for f in L.FIELDS},
                         gpa_pct=float(r['gpa_pct'][j]), roe_pct=float(L.pct_in(r['roe'], um)[j]), opm_pct=float(L.pct_in(r['opm_q'], um)[j]),
                         ind=str(FN.ind.iloc[j]), **fl))
F1 = pd.DataFrame(rows); UV = pd.DataFrame(urows)
F1['train'] = (F1['sig_date'] >= '2016-02-01') & (F1['sig_date'] < '2022-01-01')
F1['valid'] = F1['sig_date'] >= '2022-01-01'
UV['sd'] = [str(P.IDX[k].date()) for k in UV['k']]
UV['train'] = (UV['sd'] >= '2016-02-01') & (UV['sd'] < '2022-01-01'); UV['valid'] = UV['sd'] >= '2022-01-01'
q = lambda v: {f'p{p}': float(np.nanpercentile(v, p)) for p in (10, 25, 50, 75, 90)} if np.isfinite(v).any() else {}


def feat_stats(x):
    n = len(x)
    return dict(n=n, cov_opi=float(np.isfinite(x['opi_q']).mean()), cov_roe=float(np.isfinite(x['roe']).mean()), cov_gpa=float(np.isfinite(x['gpa']).mean()),
                age_days_median=float(x['age_days'].median()), age_days_max=float(x['age_days'].max()),
                loss_op_q=float(x['OPLOSS'].mean()), n_loss_op_q=int(x['OPLOSS'].sum()),
                loss_ni_ttm=float(x['EPS_TTM'].mean()), n_loss_ni_ttm=int(x['EPS_TTM'].sum()),
                loss_eps_q=float((x['eps_q'] < 0).mean()), both_loss=int((x['OPLOSS'] & x['EPS_TTM']).sum()), either_loss=int((x['OPLOSS'] | x['EPS_TTM']).sum()),
                gpa_low30=float(x['GPA_LOW'].mean()), n_gpa_low30=int(x['GPA_LOW'].sum()),
                opm_q=q(x['opm_q'].to_numpy() * 100), roe_ttm=q(x['roe'].to_numpy() * 100), gpa=q(x['gpa'].to_numpy() * 100),
                opm_pct_in_univ=q(x['opm_pct'].to_numpy()), roe_pct_in_univ=q(x['roe_pct'].to_numpy()), gpa_pct_in_univ=q(x['gpa_pct'].to_numpy()),
                roe_below_univ_median=float((x['roe_pct'] < 0.5).sum() / np.isfinite(x['roe_pct']).sum()),
                gpa_below_univ_median=float((x['gpa_pct'] < 0.5).sum() / np.isfinite(x['gpa_pct']).sum()))


def univ_stats(u):
    return dict(n_months=len(u), n_univ_median=float(u['n_univ'].median()), cov_opi=float(u['cov_opi'].mean()), loss_op_q=float(u['loss_op'].mean()),
                loss_ni_ttm=float(u['loss_ttm'].mean()), median_opm_q=float(u['med_opm'].mean() * 100), median_roe_ttm=float(u['med_roe'].mean() * 100),
                median_gpa=float(u['med_gpa'].mean() * 100))


out['step0_list_F1（訊號名單，特徵）'] = dict(all=feat_stats(F1), train=feat_stats(F1[F1['train']]), valid=feat_stats(F1[F1['valid']]))
out['step0_universe（同期股池，各訊號月平均）'] = dict(all=univ_stats(UV), train=univ_stats(UV[UV['train']]), valid=univ_stats(UV[UV['valid']]))

# ---- 實際買進（現行主跑交易）
LG = LG0.copy(); LG['code'] = LG['code'].astype(str)
key = {(int(a), str(b)): i for i, (a, b) in enumerate(zip(F1['k'], F1['code']))}
idx = [key.get((int(a), str(b)), -1) for a, b in zip(LG['sig_t'], LG['code'])]
assert min(idx) >= 0, '有交易對不到名單'
T = pd.concat([LG.reset_index(drop=True), F1.iloc[idx].drop(columns=['code', 'j', 'sig_date', 'k']).reset_index(drop=True)], axis=1)
T['r50'] = P.E50[T['exit_t'].to_numpy()] / P.E50O[T['ent_t'].to_numpy()] - 1
T['ex'] = T['ret'] - T['r50']
out['step0_bought（現行實際買進的交易，特徵）'] = dict(all=feat_stats(T), train=feat_stats(T[T['train']]), valid=feat_stats(T[T['valid']]))
out['step0_bought_industry_missing_opi'] = T.loc[~np.isfinite(T['opi_q']), 'ind'].value_counts().to_dict()


def ret_stats(z, total_pnl):
    if not len(z):
        return dict(n=0)
    mm = z.groupby('sig_t')['ex'].mean()
    return dict(n=int(len(z)), n_months=int(z['sig_t'].nunique()), mean_ret=float(z['ret'].mean()), median_ret=float(z['ret'].median()), win=float((z['ret'] > 0).mean()),
                mean_ex=float(z['ex'].mean()), median_ex=float(z['ex'].median()), win_vs_0050=float((z['ex'] > 0).mean()),
                month_t=(float(mm.mean() / (mm.std(ddof=1) / np.sqrt(len(mm)))) if len(mm) > 2 else None),
                pnl_sum=float(z['pnl'].sum()), pnl_share=float(z['pnl'].sum() / total_pnl), best=float(z['ret'].max()),
                top3=[dict(code=str(a), sig=str(b.date()), ret=float(c)) for a, b, c in z.sort_values('pnl', ascending=False).head(3)[['code', 'sig_date', 'ret']].itertuples(index=False)])


TT = T[T['train']]; tp = float(TT['pnl'].sum())
grp = {'營業利益<0（最近一季）': TT['OPLOSS'], '營業利益≥0': ~TT['OPLOSS'] & np.isfinite(TT['opi_q']), '營業利益缺值': ~np.isfinite(TT['opi_q']),
       '近四季淨利≤0': TT['EPS_TTM'], '近四季淨利>0': ~TT['EPS_TTM'] & np.isfinite(TT['ni_ttm']),
       '毛利／資產最低30%': TT['GPA_LOW'], '毛利／資產中間40%': (TT['gpa_pct'] >= 0.3) & (TT['gpa_pct'] < 0.7), '毛利／資產最高30%': TT['gpa_pct'] >= 0.7,
       '毛利／資產缺值': ~np.isfinite(TT['gpa_pct'])}
out['step0_train_returns_by_group（只看訓練期；現行主跑交易，報酬已扣成本）'] = dict(ALL=ret_stats(TT, tp), **{g: ret_stats(TT[m], tp) for g, m in grp.items()})

# ---- 候選改動檔數（不含報酬）
eff = {}
for nm in ('C1', 'C2', 'EPS_TTM'):
    e = {}
    for s in L.SHIFTS:
        _, rec = L.apply_cand(P, U, FN, SIG[s], nm)
        for lab, lo, hi in (('train', '2016-01', '2021-11'), ('valid', '2021-12', '2099-12')):
            rr = [r for r in rec if lo <= r['month'] <= hi]
            e[f'{lab}|+{s}'] = dict(n_orig=sum(len(r['orig']) for r in rr), removed=sum(len(r['removed']) for r in rr), months_changed=sum(1 for r in rr if r['removed']), n_months=len(rr))
    eff[nm] = e
_, rec3 = L.apply_cand(P, U, FN, S0, 'C3')
eff['C3'] = dict(months_order_changed=sum(1 for r in rec3 if r['orig'] != r['order']), n_months=len(rec3))
out['rule_effect_preview'] = eff
tb = lambda x: dict(n=len(x), oploss=int(x['OPLOSS'].sum()), eps_ttm=int(x['EPS_TTM'].sum()), both=int((x['OPLOSS'] & x['EPS_TTM']).sum()),
                    gpa_low=int(x['GPA_LOW'].sum()), oploss_and_gpa_low=int((x['OPLOSS'] & x['GPA_LOW']).sum()))
out['overlap_between_rules_F1'] = dict(all=tb(F1), train=tb(F1[F1['train']]), valid=tb(F1[F1['valid']]))

# ---- 和 S29 的重疊（S29 的 F1 表：V＝E/P；FLT_V 剔除 pV < 0.2；事後描述 F1_negative_EP＝V < 0）
q29 = pd.read_csv(L.AG / 's29_course_qmv' / 'f1_qmv.csv.gz', dtype={'code': str})
q29 = q29[q29['grp'] == 'F1'][['month', 'code', 'Q', 'V', 'pQ', 'pV']]
M = F1.merge(q29, on=['month', 'code'], how='left')
ov = {}
for lab, x in (('all', M), ('train', M[M['train']]), ('valid', M[M['valid']])):
    o = dict(n=len(x), matched_s29=int(x['pV'].notna().sum() + (x['V'].isna() & x['Q'].notna()).sum()))
    for nm in ('OPLOSS', 'EPS_TTM', 'GPA_LOW'):
        z = x[x[nm]]
        o[nm] = dict(n=int(len(z)), in_s29_FLT_V_removed=int((z['pV'] < 0.2).sum()), s29_negative_EP=int((z['V'] < 0).sum()), s29_pQ_low_third=int((z['pQ'] < 1 / 3).sum()))
    ov[lab] = o
out['overlap_with_S29'] = ov
F1.drop(columns=['j']).to_csv(H / 'f1_features.csv.gz', index=False)
out['meta'] = dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
(H / 'data_check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
pr = {k: v for k, v in out.items() if k not in ('paths',)}
pr['C3_mech'] = {k: v for k, v in out['C3_mech'].items() if k != 'paths'}
print(json.dumps(pr, ensure_ascii=False, indent=1, default=str))
