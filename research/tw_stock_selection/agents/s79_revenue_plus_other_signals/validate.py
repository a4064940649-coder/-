#!/usr/bin/env python3
"""S79 驗證（2016-01～2026-10，只跑一次；valid_results.json 已存在就拒絕）。先核對 PREREG／FROZEN／程式 sha256。
全部試驗（第一＋第二階段）× 20 路徑：驗證窗與全期統計；現行、凍結規則、0050 的三段完整成績單。
python3 -I -B validate.py → valid_results.json、valid_trials.csv、nav_frozen_vs_current.csv.gz、yearly.csv、trades_frozen.csv.gz"""
import sys, json, hashlib, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(HERE / 'PREREG.json') == (HERE / 'PREREG.sha256').read_text().split()[0]
assert sha(HERE / 'FROZEN.json') == (HERE / 'FROZEN.sha256').read_text().split()[0]
for line in (HERE / 'code.sha256').read_text().strip().split('\n'):
    h, f = line.split(); assert sha(HERE / f) == h, f
assert not (HERE / 'valid_results.json').exists(), '驗證只跑一次'
import numpy as np, pandas as pd
import s79lib as Q
from s79lib import K, R

t0 = time.time()
FZ = json.loads((HERE / 'FROZEN.json').read_text())
TRN = json.loads((HERE / 'train_results.json').read_text())
P, U, F, RT = Q.load(); FE = Q.features(P, U, F)
B = Q.bench(P)
D0, DE = P.IDX[P.t0], P.IDX[-1]
WINS = {'train': Q.TRAIN, 'valid': Q.VALID, 'full': ('2008-01', '2026-10')}
ALLY = tuple(range(2008, 2027))


def wcagr(nav, a, e):
    pa, pb = pd.Period(a, 'M'), pd.Period(e, 'M'); idx = nav.index
    pre = idx[idx < pa.to_timestamp()]; d0 = pre[-1] if len(pre) else idx[0]; d1 = idx[idx <= pb.to_timestamp(how='end')][-1]
    return K.cagr(nav.loc[d0:d1])


def evaluate(spec, detail=False):
    res = Q.run_paths(P, U, RT, FE, spec)
    pc = {w: [] for w in WINS}; pdd = {w: [] for w in WINS}
    for (cap, s), (nav, lg, st, sg) in res.items():
        for w, (a, e) in WINS.items():
            pc[w].append(wcagr(nav, a, e))
        if cap == 1e6 and s == 0:
            std = (nav, lg, st, sg)
    nav, lg, st, sg = std
    out = dict(paths={w: dict(median=float(np.median(v)), min=float(np.min(v)), max=float(np.max(v))) for w, v in pc.items()})
    for w, (a, e) in WINS.items():
        blk = Q.window(nav, B, a, e)
        pa, pb = pd.Period(a, 'M').to_timestamp(), pd.Period(e, 'M').to_timestamp(how='end')
        ent = lg[(lg['ent_date'] >= pa) & (lg['ent_date'] <= pb)]
        cl = ent[ent['why'] != 'open_end']
        out[w] = dict(cagr=blk['cagr'], cagr_0050=blk['cagr_0050'], cagr_diff_pp=blk['cagr_diff_pp'], ex_ann_pp=blk['ex_ann_pp'], t_nw6=blk['t_nw6'], t_iid=blk['t_iid'], IR=blk['IR'],
                      maxdd=blk['maxdd'], maxdd_0050=blk['maxdd_0050'], months=blk['months'], trades_entered=int(len(ent)), trades_closed=int(len(cl)),
                      win_rate=float((cl['ret'] > 0).mean()) if len(cl) else None, mean_ret=float(cl['ret'].mean()) if len(cl) else None,
                      avg_idle=float(1 - st['sf'].loc[pa:pb].dropna().mean()),
                      mean_targets=float(np.mean([i['n_tgt'] for i in sg['info'] if a <= i['sig_date'][:7] <= e])))
    if detail:
        out['yearly'] = K.yearly_table(nav, B.reindex(nav.index))
        out['final_nav'] = float(nav.iloc[-1])
        x = lg[lg['why'] != 'open_end'].sort_values('pnl', ascending=False)
        out['top5'] = [dict(code=str(c), ent=str(e_.date()), exit=str(q.date()), ret=float(r), pnl=float(p_)) for c, e_, q, r, p_ in zip(x['code'].head(5), x['ent_date'].head(5), x['exit_date'].head(5), x['ret'].head(5), x['pnl'].head(5))]
        out['top5_pnl_share_of_gain'] = float(x['pnl'].head(5).sum() / (nav.iloc[-1] - 1e6))
        out['periods'] = {k: K.window_block(nav, B.reindex(nav.index), a, e) for k, (a, e) in K.PERIODS.items()}
        out['_nav'] = nav; out['_lg'] = lg
    return out


specs = FZ['all_trial_specs']
RES = {}
for i, (tid, spec) in enumerate(specs.items()):
    RES[tid] = evaluate(spec, detail=tid in ('F0.base.current', FZ['frozen_rule']))
    if i % 20 == 0:
        print(i, tid, f'{time.time() - t0:.0f}s', flush=True)
cur = RES['F0.base.current']; fz = RES[FZ['frozen_rule']]
navc, navf = cur.pop('_nav'), fz['_nav'] if FZ['is_new_rule'] else None
lgc = cur.pop('_lg')
if FZ['is_new_rule']:
    navf = fz.pop('_nav'); lgf = fz.pop('_lg')
else:
    navf, lgf = navc, lgc
# ---- 全體分佈
rows = []
for tid, r in RES.items():
    tr = TRN['results'][tid]
    rows.append(dict(id=tid, fam=tid.split('.')[0], train_J=tr['J'], train_J_adj=tr['J_adj'], train_cagr_ex=tr['cagr_ex'], train_eligible=tr['eligible'] if 'eligible' in tr else False,
                     train_cagr_med=r['paths']['train']['median'], valid_cagr_med=r['paths']['valid']['median'], full_cagr_med=r['paths']['full']['median'],
                     valid_minus_current_pp=(r['paths']['valid']['median'] - cur['paths']['valid']['median']) * 100,
                     full_minus_current_pp=(r['paths']['full']['median'] - cur['paths']['full']['median']) * 100,
                     valid_ex_ann_pp=r['valid']['ex_ann_pp'], valid_t=r['valid']['t_nw6'], valid_maxdd=r['valid']['maxdd'], valid_trades=r['valid']['trades_entered'],
                     train_ex_ann_pp=r['train']['ex_ann_pp'], train_t=r['train']['t_nw6'], full_ex_ann_pp=r['full']['ex_ann_pp'], full_t=r['full']['t_nw6']))
df = pd.DataFrame(rows)
df.to_csv(HERE / 'valid_trials.csv', index=False)
oth = df[df['id'] != 'F0.base.current']
v = oth['valid_minus_current_pp']
sp = lambda a, b: float(pd.Series(a).rank().corr(pd.Series(b).rank()))
dist = dict(n=int(len(oth)), valid_minus_current_pp=dict(median=float(v.median()), p10=float(v.quantile(0.1)), p25=float(v.quantile(0.25)), p75=float(v.quantile(0.75)), p90=float(v.quantile(0.9)), min=float(v.min()), max=float(v.max()),
                                                          share_better_than_current=float((v > 0).mean()), share_within_2pp_or_better=float((v >= -2).mean())),
            valid_cagr_med=dict(median=float(oth['valid_cagr_med'].median()), min=float(oth['valid_cagr_med'].min()), max=float(oth['valid_cagr_med'].max())),
            current_rank_in_valid=int((df['valid_cagr_med'] > cur['paths']['valid']['median']).sum() + 1), frozen_rank_in_valid=int((df['valid_cagr_med'] > fz['paths']['valid']['median']).sum() + 1), n_all=int(len(df)),
            share_trials_beating_0050_valid=float((df['valid_ex_ann_pp'] > 0).mean()), share_trials_valid_t_ge_1p65=float((df['valid_t'] >= 1.65).mean()),
            share_trials_beating_0050_train=float((df['train_ex_ann_pp'] > 0).mean()), share_trials_train_t_ge_2=float((df['train_t'] >= 2).mean()),
            spearman_trainJadj_vs_valid_minus_current=sp(oth['train_J_adj'], oth['valid_minus_current_pp']),
            spearman_trainCagrEx_vs_valid_minus_current=sp(oth['train_cagr_ex'], oth['valid_minus_current_pp']),
            by_family={f: dict(n=int(len(g)), train_cagr_ex_median=float(g['train_cagr_ex'].median()), train_cagr_ex_max=float(g['train_cagr_ex'].max()), train_J_median=float(g['train_J'].median()), train_J_max=float(g['train_J'].max()),
                               valid_minus_current_median=float(g['valid_minus_current_pp'].median()), valid_minus_current_max=float(g['valid_minus_current_pp'].max()),
                               full_minus_current_median=float(g['full_minus_current_pp'].median()), full_minus_current_max=float(g['full_minus_current_pp'].max()))
                       for f, g in oth.groupby('fam')},
            train_top10_by_J_adj=[dict(id=r_['id'], train_J_adj=r_['train_J_adj'], train_cagr_ex=r_['train_cagr_ex'], valid_minus_current_pp=r_['valid_minus_current_pp'], full_minus_current_pp=r_['full_minus_current_pp'])
                                  for _, r_ in oth.sort_values('train_J_adj', ascending=False).head(10).iterrows()],
            train_top10_by_cagr_ex=[dict(id=r_['id'], train_J_adj=r_['train_J_adj'], train_cagr_ex=r_['train_cagr_ex'], valid_minus_current_pp=r_['valid_minus_current_pp'], full_minus_current_pp=r_['full_minus_current_pp'])
                                    for _, r_ in oth.sort_values('train_cagr_ex', ascending=False).head(10).iterrows()])
N = len(df)
import math
mt = dict(n_trials=N, expected_max_t_under_null=math.sqrt(2 * math.log(N)), bonferroni_one_sided_5pct_t=float(__import__('statistics').NormalDist().inv_cdf(1 - 0.05 / N)),
          best_train_t=float(df['train_t'].max()), best_train_t_id=str(df.loc[df['train_t'].idxmax(), 'id']), frozen_train_t=fz['train']['t_nw6'], current_train_t=cur['train']['t_nw6'])
pc = dict(P1=bool(fz['paths']['valid']['median'] >= cur['paths']['valid']['median'] - 0.02), P2=bool(fz['valid']['maxdd'] >= cur['valid']['maxdd'] - 0.05),
          P3=bool(fz['valid']['ex_ann_pp'] > 0 and fz['valid']['t_nw6'] >= 1.65), T1=bool(fz['train']['ex_ann_pp'] >= 5 and fz['train']['t_nw6'] >= 2.0))
verdict = '值得獨立複製的候選規則' if (FZ['is_new_rule'] and all(pc.values())) else '沒有比現行好，維持現行'
bw = {w: dict(cagr=cur[w]['cagr_0050'], maxdd=cur[w]['maxdd_0050']) for w in WINS}
out = dict(task='S79', sha=dict(PREREG=sha(HERE / 'PREREG.json'), FROZEN=sha(HERE / 'FROZEN.json')), frozen_rule=FZ['frozen_rule'], is_new_rule=FZ['is_new_rule'], frozen_spec=FZ['spec'],
           current=cur, frozen=fz, e0050=bw, pass_criteria=pc, verdict=verdict, distribution=dist, multiple_testing=mt,
           all=RES if not FZ['is_new_rule'] else {k: v for k, v in RES.items()}, runtime_s=round(time.time() - t0, 1))
(HERE / 'valid_results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
pd.DataFrame({'current': navc, 'frozen': navf, '0050': B.reindex(navc.index)}).round(2).to_csv(HERE / 'nav_frozen_vs_current.csv.gz')
yc = cur['yearly']; yf = fz['yearly']
pd.DataFrame([dict(year=a['year'], current=a['strat'], frozen=b['strat'], e0050=a['e0050'], current_diff_pp=a['diff_pp'], frozen_diff_pp=b['diff_pp']) for a, b in zip(yc, yf)]).to_csv(HERE / 'yearly.csv', index=False)
lgf.drop(columns=['j']).to_csv(HERE / 'trades_frozen.csv.gz', index=False)
print(verdict, pc, json.dumps(dist['valid_minus_current_pp']), f'{time.time() - t0:.0f}s')
