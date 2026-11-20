#!/usr/bin/env python3
"""S77 驗證（只跑一次；valid_results.json 已存在就拒絕）：核對 FROZEN／PREREG／程式 sha256 後，完整面板跑 2016-01～2026-10。
python3 -I -B validate.py → valid_results.json、valid_family.csv、nav_s77.csv.gz、yearly.csv、trades.csv.gz、paths.csv"""
import sys, json, time, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
assert not (HERE / 'valid_results.json').exists(), '驗證只跑一次'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for f in ('PREREG', 'FROZEN'):
    assert sha(HERE / f'{f}.json') == (HERE / f'{f}.sha256').read_text().split()[0], f
for line in (HERE / 'code.sha256').read_text().strip().split('\n'):
    h, f = line.split(); assert sha(HERE / f) == h, f'{f} 與登記的 sha256 不符'
import numpy as np, pandas as pd
import s77lib as M
from s77lib import K, R

t0 = time.time()
FZ = json.load(open(HERE / 'FROZEN.json')); TR = json.load(open(HERE / 'train_results.json'))
assert FZ['train_results_sha256'] == sha(HERE / 'train_results.json')
NEW = FZ['frozen_rule']; CUR = M.CURRENT
RULES = {'new': NEW, 'current': CUR}
P, U = M.load()
ZS, av = M.signals(P)
D0, DE = P.IDX[P.t0], P.IDX[-1]
assert (str(D0.date()), str(DE.date())) == ('2008-01-02', '2026-10-22')
bench = M.bench_of(P)
DT = K.last_before(P.IDX, '2016-01-01')                      # 2015-12-31
qs = lambda x: {q: float(np.quantile(x, p)) for q, p in (('min', 0), ('p25', .25), ('median', .5), ('p75', .75), ('max', 1))}
out = dict(task='S77 驗證段 2016-01～2026-10（凍結後只跑一次）', sha=dict(PREREG=sha(HERE / 'PREREG.json'), FROZEN=sha(HERE / 'FROZEN.json')), frozen_rule_id=FZ['frozen_rule_id'], current_rule_id=M.cid(CUR))

# ---------------------------------------------------------------- 連續帳戶（S75 口徑）
tn = pd.read_csv(HERE / 'train_nav.csv.gz', index_col=0, parse_dates=True)
NAV = {}; LG = {}; cont = {}; prow = []
for name, c in RULES.items():
    sg, nav, lg, st = M.run_cfg(P, U, ZS, av, c)
    NAV[name] = nav; LG[name] = lg
    w = {k: K.window_block(nav, bench, a, b) for k, (a, b) in (('train', M.TRAIN), ('valid', M.VALID), ('full', ('2008-01', '2026-10')), ('2016～2018', ('2016-01', '2018-12')), ('2019～2026', ('2019-01', '2026-10')))}
    cont[name] = dict(windows=w, yearly=K.yearly_table(nav, bench), final_nav=float(nav.iloc[-1]), maxdd_full=K.dd_info(nav), maxdd_valid=K.dd_info(nav.loc[DT:]),
                      trades=dict(train=K.trade_block(P, lg, st, None, '2015-12-31'), valid=K.trade_block(P, lg, st, '2016-01-01', None)),
                      turnover=dict(train=M.turnover(lg, nav, None, '2015-12-31'), valid=M.turnover(lg, nav, '2016-01-01', None)),
                      bears=K.bear_block(nav, bench), limit7=K.limit7_diag(P, lg), why=lg['why'].value_counts().to_dict(),
                      mean_targets=dict(train=float(np.mean([i['n_tgt'] for i in sg['info'] if i['sig_date'] < '2016-01-01'])), valid=float(np.mean([i['n_tgt'] for i in sg['info'] if i['sig_date'] >= '2016-01-01']))),
                      train_nav_same_as_train_py=float((nav.reindex(tn.index) / tn[name] - 1).abs().max()))
    x = lg[(lg['why'] != 'open_end') & (lg['ent_date'] >= pd.Timestamp('2016-01-01'))].sort_values('pnl', ascending=False)
    cont[name]['top5_valid'] = [dict(code=str(a), ent=str(b.date()), exit=str(d.date()), ret=float(e), pnl=float(f)) for a, b, d, e, f in zip(x['code'].head(5), x['ent_date'].head(5), x['exit_date'].head(5), x['ret'].head(5), x['pnl'].head(5))]
    cont[name]['top5_valid_pnl_share'] = float(x['pnl'].head(5).sum() / (nav.iloc[-1] - nav.loc[DT]))
    pp = {}
    for s in K.SHIFTS:
        for cap in K.CAPS:
            _, n2, l2, _ = M.run_cfg(P, U, ZS, av, c, cap=cap, shift=s)
            pp[f'{int(cap / 1e4)}萬|+{s}'] = dict(cagr_train=K.cagr(n2, D0, DT), cagr_valid=K.cagr(n2, DT, DE), cagr_full=K.cagr(n2), maxdd_full=R.maxdd(n2), maxdd_valid=R.maxdd(n2.loc[DT:]))
            prow.append(dict(account='continuous', rule=name, capital=int(cap), shift=s, **pp[f'{int(cap / 1e4)}萬|+{s}']))
    cont[name]['paths20'] = {k: dict(median=float(np.median([p[k] for p in pp.values()])), min=float(min(p[k] for p in pp.values())), max=float(max(p[k] for p in pp.values())))
                             for k in ('cagr_train', 'cagr_valid', 'cagr_full', 'maxdd_full', 'maxdd_valid')}
    cont[name]['paths'] = pp
out['e0050'] = dict(cagr_train=K.cagr(bench, D0, DT), cagr_valid=K.cagr(bench, DT, DE), cagr_full=K.cagr(bench), maxdd_train=R.maxdd(bench.loc[:DT]), maxdd_valid=R.maxdd(bench.loc[DT:]), maxdd_full=R.maxdd(bench), final_nav=float(bench.iloc[-1]))
out['continuous'] = cont
d = [cont['new']['paths'][k]['cagr_valid'] - cont['current']['paths'][k]['cagr_valid'] for k in cont['new']['paths']]
out['continuous_paired_valid'] = dict(median_diff_pp=float(np.median(d) * 100), n_pos=int(sum(x > 0 for x in d)), n=len(d))

# ---------------------------------------------------------------- 家族（147 格；凍結後才算）
fam = []
for k, v in TR['configs'].items():
    c = v['cfg']
    _, nav, lg, _ = M.run_cfg(P, U, ZS, av, c)
    wv = K.window_block(nav, bench, *M.VALID); wt = K.window_block(nav, bench, *M.TRAIN)
    assert abs(wt['cagr_diff_pp'] - v['cagr_diff_pp']) < 1e-9, k
    fam.append(dict(id=k, stage=v['stage'], **c, train_score=v['score'], train_cagr_diff_pp=v['cagr_diff_pp'], train_t_nw6=v['t_nw6'], valid_cagr=wv['cagr'], valid_cagr_diff_pp=wv['cagr_diff_pp'],
                    valid_ex_ann_pp=wv['ex_ann_pp'], valid_t_nw6=wv['t_nw6'], valid_maxdd=wv['maxdd'], full_cagr=K.cagr(nav),
                    valid_n_closed=int(((lg['why'] != 'open_end') & (lg['ent_date'] >= pd.Timestamp('2016-01-01'))).sum())))
fam = pd.DataFrame(fam); fam.to_csv(HERE / 'valid_family.csv', index=False)
pct = lambda col, k: float((fam[col] <= fam.loc[fam['id'] == k, col].iloc[0]).mean() * 100)
rk = lambda col, k: int((fam[col] > fam.loc[fam['id'] == k, col].iloc[0]).sum() + 1)
nid, cidd = FZ['frozen_rule_id'], M.cid(CUR)
s1 = fam[fam['stage'] == 1]
out['family'] = dict(n=len(fam), valid_cagr=qs(fam['valid_cagr']), valid_cagr_diff_pp=qs(fam['valid_cagr_diff_pp']), train_cagr_diff_pp=qs(fam['train_cagr_diff_pp']), valid_t_nw6=qs(fam['valid_t_nw6']),
                     n_valid_beat_0050=int((fam['valid_cagr_diff_pp'] > 0).sum()), n_train_beat_0050=int((fam['train_cagr_diff_pp'] > 0).sum()), n_valid_t_ge_1_65=int((fam['valid_t_nw6'] >= 1.65).sum()),
                     new=dict(valid_cagr_rank=rk('valid_cagr', nid), valid_cagr_percentile=pct('valid_cagr', nid)), current=dict(valid_cagr_rank=rk('valid_cagr', cidd), valid_cagr_percentile=pct('valid_cagr', cidd)),
                     spearman_train_score_vs_valid_cagr=float(fam['train_score'].rank().corr(fam['valid_cagr'].rank())),
                     spearman_train_cagr_vs_valid_cagr=float(fam['train_cagr_diff_pp'].rank().corr(fam['valid_cagr'].rank())),
                     top10_train_score_valid_cagr_median=float(fam.sort_values('train_score', ascending=False).head(10)['valid_cagr'].median()),
                     top5_valid=fam.sort_values('valid_cagr', ascending=False).head(5)[['id', 'valid_cagr', 'valid_cagr_diff_pp', 'train_cagr_diff_pp', 'train_score']].to_dict('records'),
                     by_signal=dict(train=s1.groupby('sig')['train_cagr_diff_pp'].median().to_dict(), valid=s1.groupby('sig')['valid_cagr_diff_pp'].median().to_dict()),
                     by_filter=dict(train=s1.groupby('filt')['train_cagr_diff_pp'].median().to_dict(), valid=s1.groupby('filt')['valid_cagr_diff_pp'].median().to_dict()),
                     by_n=dict(train={int(a): b for a, b in s1.groupby('n')['train_cagr_diff_pp'].median().items()}, valid={int(a): b for a, b in s1.groupby('n')['valid_cagr_diff_pp'].median().items()}))

# ---------------------------------------------------------------- 2015-12-31 重新起跑（S25 的 V1～V4）
P.t0 = int(np.searchsorted(P.IDX.values, np.datetime64(DT)))
b2 = M.bench_of(P)
SEG = {'2022': ('2022-01-01', '2022-12-31'), '2024-07～2025-04': ('2024-07-01', '2025-04-30')}
segdd = lambda s: {k: float((s.loc[a:b] / s.loc[a:b].cummax() - 1).min()) for k, (a, b) in SEG.items()}
fresh = {}; top = {}
for name, c in RULES.items():
    sg, nav, lg, st = M.run_cfg(P, U, ZS, av, c)
    rs, rb = R.monthly_returns(nav, b2); blk = R.stats_block(rs, rb)
    x = lg[lg['why'] != 'open_end'].sort_values('pnl', ascending=False).head(5)
    top[name] = [(int(j), int(a), int(b)) for j, a, b in zip(x['j'], x['ent_t'], x['exit_t'])]
    fresh[name] = dict(main=dict(cagr=K.cagr(nav), cagr_0050=K.cagr(b2), final_nav=float(nav.iloc[-1]), maxdd=K.dd_info(nav), ex_ann_pp=blk['ex_ann_pp'], IR=blk['IR'], t_nw6=blk['t_nw6'], t_iid=blk['t_iid'], months=blk['months'],
                                 trades=K.trade_block(P, lg, st), turnover=M.turnover(lg, nav, None, None), seg_dd=segdd(nav), yearly=K.yearly_table(nav, b2),
                                 top5=[dict(code=str(a), ent=str(b.date()), exit=str(d.date()), ret=float(e), pnl=float(f)) for a, b, d, e, f in zip(x['code'], x['ent_date'], x['exit_date'], x['ret'], x['pnl'])],
                                 top5_pnl_share=float(x['pnl'].sum() / (nav.iloc[-1] - 1e6))))
    if name == 'new':
        NAV['new_fresh'] = nav
    else:
        NAV['current_fresh'] = nav
block = {}
for name in top:
    for j, a, b in top[name]:
        block.setdefault(j, []).append((a, b))
for name, c in RULES.items():
    pp = {}
    for s in K.SHIFTS:
        for cap in K.CAPS:
            _, n2, _, _ = M.run_cfg(P, U, ZS, av, c, cap=cap, shift=s)
            _, n3, _, _ = M.run_cfg(P, U, ZS, av, c, cap=cap, shift=s, block=block)
            pp[f'{int(cap / 1e4)}萬|+{s}'] = dict(cagr=K.cagr(n2), maxdd=R.maxdd(n2), cagr_ban=K.cagr(n3), **{f'dd_{k}': v for k, v in segdd(n2).items()})
            prow.append(dict(account='fresh_2015-12-31', rule=name, capital=int(cap), shift=s, **pp[f'{int(cap / 1e4)}萬|+{s}']))
    fresh[name]['paths'] = pp
    fresh[name]['paths20'] = {k: dict(median=float(np.median([p[k] for p in pp.values()])), min=float(min(p[k] for p in pp.values())), max=float(max(p[k] for p in pp.values()))) for k in ('cagr', 'maxdd', 'cagr_ban')}
keys = list(fresh['new']['paths'])
dif = lambda k: np.array([fresh['new']['paths'][q][k] - fresh['current']['paths'][q][k] for q in keys]) * 100
d1 = dif('cagr'); d3 = dif('cagr_ban'); d4 = {k: dif(f'dd_{k}') for k in SEG}
V = dict(V1=dict(n_pos=int((d1 > 0).sum()), n=20, passed=bool((d1 > 0).sum() >= 15)), V2=dict(median_diff_pp=float(np.median(d1)), min=float(d1.min()), max=float(d1.max()), passed=bool(np.median(d1) >= 2.0)),
         V3=dict(banned=sorted({str(P.COLS[j]) for j in block}), median_diff_pp=float(np.median(d3)), passed=bool(np.median(d3) > 0)),
         V4={k: dict(median_diff_pp=float(np.median(v)), passed=bool(np.median(v) >= -2.0)) for k, v in d4.items()})
V['overall'] = bool(V['V1']['passed'] and V['V2']['passed'] and V['V3']['passed'] and all(x['passed'] for x in V['V4'].values()))
out['fresh_2015_12_31'] = fresh; out['S25'] = V

# ---------------------------------------------------------------- 事前寫死的判定
nv, cv = cont['new']['windows']['valid'], cont['current']['windows']['valid']; ntr = cont['new']['windows']['train']
dq1 = (nv['cagr'] - cv['cagr']) * 100
pc = out['family']['new']['valid_cagr_percentile']
out['verdicts'] = dict(
    Q1=dict(new_minus_current_valid_cagr_pp=dq1, verdict='比現行好' if dq1 > 3 else ('比現行差' if dq1 < -3 else '差不多')),
    Q1b=dict(ex_ann_pp=nv['ex_ann_pp'], cagr_diff_pp=nv['cagr_diff_pp'], t_nw6=nv['t_nw6'], verdict='後段仍明顯贏 0050' if (nv['t_nw6'] >= 1.65 and nv['cagr_diff_pp'] > 0) else ('方向一致但不顯著' if nv['cagr_diff_pp'] > 0 else '沒贏')),
    Q2=dict(train_t_nw6=ntr['t_nw6'], n_trials=147, bonferroni_t=3.39, verdict='校正後顯著' if ntr['t_nw6'] >= 3.39 else ('只在不校正時顯著' if ntr['t_nw6'] >= 1.65 else '不顯著')),
    Q3=dict(train_cagr_diff_pp=ntr['cagr_diff_pp'], train_t_nw6=ntr['t_nw6'],
            verdict='前段有對得上的規則' if (ntr['cagr_diff_pp'] >= 8 and ntr['t_nw6'] >= 2) else ('那個年代這類訊號本來就賺得少' if (ntr['cagr_diff_pp'] < 5 or ntr['t_nw6'] < 1.65) else '介於兩者，無法判定'),
            family_median_train_pp=out['family']['train_cagr_diff_pp']['median'], family_median_valid_pp=out['family']['valid_cagr_diff_pp']['median']),
    family=dict(percentile=pc, verdict='規則本身挑得好' if pc >= 75 else ('挑到的在後段反而偏差' if pc < 25 else '和隨便一格差不多')))
out['runtime_s'] = round(time.time() - t0, 1)
(HERE / 'valid_results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
df = pd.DataFrame({'new': NAV['new'], 'current': NAV['current'], '0050': bench.reindex(NAV['new'].index), 'new_fresh2016': NAV['new_fresh'], 'current_fresh2016': NAV['current_fresh'], '0050_fresh2016': b2}).round(2)
df.index.name = 'date'; df.to_csv(HERE / 'nav_s77.csv.gz')
pd.DataFrame([dict(year=a['year'], new=a['strat'], current=b['strat'], e0050=a['e0050'], new_diff_pp=a['diff_pp'], current_diff_pp=b['diff_pp']) for a, b in zip(cont['new']['yearly'], cont['current']['yearly'])]).to_csv(HERE / 'yearly.csv', index=False)
pd.concat([LG[n].drop(columns=['j']).assign(rule=n) for n in ('new', 'current')]).to_csv(HERE / 'trades.csv.gz', index=False)
pd.DataFrame(prow).to_csv(HERE / 'paths.csv', index=False)
print(json.dumps(out['verdicts'], ensure_ascii=False, indent=1)); print(json.dumps(out['S25'], ensure_ascii=False, indent=1)); print('done', out['runtime_s'])
