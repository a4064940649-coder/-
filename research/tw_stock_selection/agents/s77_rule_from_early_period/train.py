#!/usr/bin/env python3
"""S77 訓練（只跑一次；train_results.json 已存在就拒絕）：面板截到 2015-12-31，照 PREREG 的兩階段挑選，寫 FROZEN.json＋FROZEN.sha256。
python3 -I -B train.py → train_results.json、train_family.csv、train_nav.csv.gz、FROZEN.json、FROZEN.sha256"""
import sys, json, time, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
assert not (HERE / 'train_results.json').exists() and not (HERE / 'FROZEN.json').exists(), '訓練只跑一次'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(HERE / 'PREREG.json') == (HERE / 'PREREG.sha256').read_text().split()[0]
for line in (HERE / 'code.sha256').read_text().strip().split('\n'):
    h, f = line.split(); assert sha(HERE / f) == h, f'{f} 與登記的 sha256 不符'
assert json.load(open(HERE / 'accept.json'))['pass']
import numpy as np, pandas as pd
import s77lib as M
from s77lib import K

t0 = time.time()
P, U = M.load(M.TRAIN_END)
assert str(P.IDX[-1].date()) == '2015-12-31' and str(P.IDX[P.t0].date()) == '2008-01-02'
ZS, av = M.signals(P, last_month='2015-12')
bench = M.bench_of(P)
RES = {}; NAVS = {}


def run(c):
    k = M.cid(c)
    if k not in RES:
        sg, nav, lg, st = M.run_cfg(P, U, ZS, av, c)
        RES[k] = dict(cfg=c, **M.train_metrics(P, nav, lg, st, bench))
        RES[k]['eligible'] = bool(RES[k]['n_closed'] >= 40 and RES[k]['maxdd'] >= RES[k]['maxdd_0050'] - 0.05)
        NAVS[k] = nav
    return RES[k]


def neigh_check(c, neigh):
    me = run(c); nb = [run(x)['cagr_diff_pp'] for x in neigh(c)]
    med = float(np.median(nb))
    return dict(own=me['cagr_diff_pp'], neighbor_median=med, n_neighbors=len(nb), passed=bool(med >= me['cagr_diff_pp'] - 3.0))


def ordered(cfgs):
    ks = [M.cid(c) for c in cfgs]
    return sorted(range(len(cfgs)), key=lambda i: (-RES[ks[i]]['score'], -RES[ks[i]]['cagr_diff_pp'], i))


# ---- 第一階段
S1 = M.stage1_cfgs()
for c in S1:
    run(c)
trail1 = []; pick1 = None; flag1 = '過鄰居規則'
for i in ordered(S1):
    c = S1[i]
    if not RES[M.cid(c)]['eligible']:
        continue
    chk = neigh_check(c, M.neighbors1)
    trail1.append(dict(id=M.cid(c), score=RES[M.cid(c)]['score'], **chk))
    if chk['passed']:
        pick1 = c; break
if pick1 is None:
    el = [S1[i] for i in ordered(S1) if RES[M.cid(S1[i])]['eligible']]
    pick1 = el[0]; flag1 = '未過鄰居規則'
# ---- 第二階段
S2 = M.stage2_cfgs(pick1)
for c in S2:
    run(c)
base = RES[M.cid(pick1)]
trail2 = []; pick2 = pick1
for i in ordered(S2):
    c = S2[i]; r = RES[M.cid(c)]
    if M.cid(c) == M.cid(pick1) or r['score'] < base['score'] + 1.0:
        break
    if not r['eligible']:
        trail2.append(dict(id=M.cid(c), score=r['score'], passed=False, reason='不合格')); continue
    chk = neigh_check(c, M.neighbors2)
    trail2.append(dict(id=M.cid(c), score=r['score'], **chk))
    if chk['passed']:
        pick2 = c; break
for k in list(RES):
    RES[k]['stage'] = 1 if k in {M.cid(c) for c in S1} else 2
assert len(RES) == 147, len(RES)
fk = M.cid(pick2)
# ---- 凍結規則在訓練段的補充（都只用訓練段）
sg, nav, lg, st = M.run_cfg(P, U, ZS, av, pick2)
paths = {}
for s in K.SHIFTS:
    for cap in K.CAPS:
        for name, c in (('new', pick2), ('current', M.CURRENT)):
            _, n2, _, _ = M.run_cfg(P, U, ZS, av, c, cap=cap, shift=s)
            paths.setdefault(name, {})[f'{int(cap / 1e4)}萬|+{s}'] = dict(cagr=K.cagr(n2), maxdd=K.R.maxdd(n2))
b0 = bench
df = pd.DataFrame([dict(id=k, **{a: b for a, b in v.items() if a not in ('cfg', 'yearly_excess_pp')}, **v['cfg']) for k, v in RES.items()])
df.to_csv(HERE / 'train_family.csv', index=False)
el = df[df['eligible']]
out = dict(task='S77 訓練段（2008-01-02～2015-12-31）', prereg_sha256=sha(HERE / 'PREREG.json'), panel=dict(T=P.T, N=P.N, last=str(P.IDX[-1].date())),
           e0050=dict(cagr=K.cagr(b0), maxdd=K.R.maxdd(b0), yearly=[y['e0050'] for y in K.yearly_table(b0, b0)]),
           n_trials=len(RES), n_eligible=int(df['eligible'].sum()),
           family=dict(cagr_diff_pp={q: float(df['cagr_diff_pp'].quantile(p)) for q, p in (('min', 0), ('p25', .25), ('median', .5), ('p75', .75), ('max', 1))},
                       score={q: float(df['score'].quantile(p)) for q, p in (('min', 0), ('p25', .25), ('median', .5), ('p75', .75), ('max', 1))},
                       t_nw6={q: float(df['t_nw6'].quantile(p)) for q, p in (('min', 0), ('p25', .25), ('median', .5), ('p75', .75), ('max', 1))},
                       n_t_ge_1_65=int((df['t_nw6'] >= 1.65).sum()), n_cagr_diff_pos=int((df['cagr_diff_pp'] > 0).sum()),
                       best_by_cagr=df.sort_values('cagr_diff_pp', ascending=False).head(5)[['id', 'cagr_diff_pp', 'score', 't_nw6', 'maxdd', 'n_closed']].to_dict('records'),
                       best_by_score=df.sort_values('score', ascending=False).head(10)[['id', 'cagr_diff_pp', 'score', 't_nw6', 'maxdd', 'n_closed', 'eligible']].to_dict('records'),
                       by_signal_median_cagr_diff=df[df['stage'] == 1].groupby('sig')['cagr_diff_pp'].median().to_dict(),
                       by_filter_median_cagr_diff=df[df['stage'] == 1].groupby('filt')['cagr_diff_pp'].median().to_dict(),
                       by_n_median_cagr_diff={int(a): b for a, b in df[df['stage'] == 1].groupby('n')['cagr_diff_pp'].median().items()}),
           stage1=dict(pick=M.cid(pick1), flag=flag1, trail=trail1), stage2=dict(pick=fk, trail=trail2, table=[dict(id=M.cid(c), **{a: RES[M.cid(c)][a] for a in ('score', 'cagr_diff_pp', 't_nw6', 'maxdd', 'n_closed', 'eligible')}) for c in S2]),
           frozen=dict(id=fk, **RES[fk]), current=dict(id=M.cid(M.CURRENT), **RES[M.cid(M.CURRENT)]),
           current_rank_by_score=int((df['score'] > RES[M.cid(M.CURRENT)]['score']).sum() + 1), current_rank_by_cagr=int((df['cagr_diff_pp'] > RES[M.cid(M.CURRENT)]['cagr_diff_pp']).sum() + 1),
           frozen_trades=K.trade_block(P, lg, st), frozen_limit7=K.limit7_diag(P, lg), frozen_yearly=K.yearly_table(nav, bench),
           paths20_train={n: dict(cagr_median=float(np.median([p['cagr'] for p in v.values()])), cagr_min=float(min(p['cagr'] for p in v.values())), cagr_max=float(max(p['cagr'] for p in v.values())),
                                  maxdd_median=float(np.median([p['maxdd'] for p in v.values()])), paths=v) for n, v in paths.items()},
           configs=RES, runtime_s=round(time.time() - t0, 1))
(HERE / 'train_results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
pd.DataFrame({'new': NAVS[fk], 'current': NAVS[M.cid(M.CURRENT)], '0050': bench}).to_csv(HERE / 'train_nav.csv.gz')
fz = dict(id='S77', frozen_rule=pick2, frozen_rule_id=fk, stage1_pick=M.cid(pick1), stage1_flag=flag1, prereg_sha256=sha(HERE / 'PREREG.json'), train_results_sha256=sha(HERE / 'train_results.json'),
          train=dict(score=RES[fk]['score'], cagr=RES[fk]['cagr'], cagr_0050=RES[fk]['cagr_0050'], cagr_diff_pp=RES[fk]['cagr_diff_pp'], t_nw6=RES[fk]['t_nw6'], maxdd=RES[fk]['maxdd'], n_closed=RES[fk]['n_closed']),
          note='凍結後才跑 validate.py；不可回頭改。')
(HERE / 'FROZEN.json').write_text(json.dumps(fz, ensure_ascii=False, indent=1))
(HERE / 'FROZEN.sha256').write_text(sha(HERE / 'FROZEN.json') + '  FROZEN.json\n')
print(json.dumps({k: v for k, v in out.items() if k not in ('configs',)}, ensure_ascii=False, indent=1, default=str)[:9000])
print('FROZEN', fk, sha(HERE / 'FROZEN.json'))
