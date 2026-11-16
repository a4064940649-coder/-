#!/usr/bin/env python3
"""S31 訓練期：4 條候選規則 × 20 條路徑，只算訓練窗（2016-01-29 → 2021-12-30）；依 PREREG 選一條凍結。另報訓練窗分散度（4 條全報）。
輸出 results_train.json、FROZEN.json、FROZEN.sha256、code.sha256。PREREG.json 的 sha256 必須相符；FROZEN.json 已存在就拒絕重跑。python3 -I -B train.py"""
import sys, json, hashlib, time, resource, gc, datetime
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s31lib
from s31lib import s17lib, s10lib, s4lib, stlib, COSTS

H = Path(__file__).resolve().parent
pre = (H / 'PREREG.json').read_bytes()
sha = hashlib.sha256(pre).hexdigest()
assert sha == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與 PREREG.sha256 不符'
assert not (H / 'FROZEN.json').exists(), 'FROZEN.json 已存在，拒絕重跑'
PRE = json.loads(pre)
RULES = PRE['rule_form']['candidates（4 條，全部計入 N；參與訓練期選擇）']
assert RULES == json.loads(json.dumps(s31lib.RULES, ensure_ascii=False)), '規則與 PREREG 不符'
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s31lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
RT = s31lib.surge(P)
fix = s17lib.e50_fix_suspension(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
IL = s31lib.IndLookup()
print('載入完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TA, TB = lastb('2016-02-01'), lastb('2022-01-01')
T18 = lastb('2019-01-01'); D19 = T18
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
CELL = dict(sat_w=0.7, sizing='W10', lock=False)


def sim(S1, cap=1e6, cost='strict'):
    S2, ekw = s10lib.variant_S(S1, 'E0')
    return s10lib.simulate_s10(P, S2, U, cost=COSTS[cost], capital=cap, **CELL, **ekw)


def tr_metrics(nav):
    return dict(train=stlib.cagr(nav, TA, TB), train_dd=stlib.maxdd(nav, TA, TB), y1618=stlib.cagr(nav, TA, T18), y1921=stlib.cagr(nav, T18, TB))


# ---------------------------------------------------------------- 驗收 A0、A1
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
IND0, _ = s31lib.industry_matrix(P, [x['k'] for x in S0['info']])
acc = {'P0': True, 'exright_fix': P.exright_fix, 'e50_fix': fix}
navc, lgc, _ = sim(S0)
a0 = dict(since2019=stlib.cagr(navc, D19, P.IDX[-1]), full=stlib.cagr(navc, P.IDX[P.t0], P.IDX[-1]), maxdd=stlib.maxdd(navc, P.IDX[P.t0], P.IDX[-1]),
          n_closed=int((lgc['why'] != 'open_end').sum()))
a0['ok'] = bool(abs(a0['since2019'] - 0.4899) < 0.00025 and abs(a0['full'] - 0.3454) < 0.00025 and abs(a0['maxdd'] + 0.3211) < 0.00025 and a0['n_closed'] == 381)
acc['A0'] = a0
SA, _ = s31lib.build_rule(P, U, RT, S0, IND0, dict(id='ID', kind='score', score='identity'))
nava, lga, _ = sim(SA)
cols = ['code', 'ent_t', 'exit_t', 'why']
acc['A1'] = dict(trades_identical=bool(len(lga) == len(lgc) and (lga[cols].reset_index(drop=True) == lgc[cols].reset_index(drop=True)).all().all()),
                 nav_identical=bool(np.allclose(nava.to_numpy(), navc.to_numpy(), rtol=0, atol=1e-6)))
assert a0['ok'] and acc['A1']['trades_identical'] and acc['A1']['nav_identical'], ('驗收失敗', acc)
del navc, lgc, nava, lga, SA
print('驗收', json.dumps({k: v for k, v in acc.items() if k != 'e50_fix'}, ensure_ascii=False), flush=True)

# ---------------------------------------------------------------- 20 條路徑 × 5 個設定（只算訓練窗）
CFG = {'CUR': None}
for r in RULES:
    CFG[r['id']] = r
M = {c: {} for c in CFG}
eff = {c: {} for c in CFG if c != 'CUR'}
DIV = {}
for s in range(4):
    S = S0 if s == 0 else s10lib.make_signal_s10(P, U, RT, shift=s)
    IND = IND0 if s == 0 else s31lib.industry_matrix(P, [x['k'] for x in S['info']])[0]
    VAR = {}
    for c, rule in CFG.items():
        S1, lg = s31lib.build_rule(P, U, RT, S, IND, rule)
        VAR[c] = S1
        if c != 'CUR':
            L = [x for x in lg if pd.Period(x['month'], 'M') <= s31lib.TRAIN_LAST_MONTH and pd.Timestamp(P.IDX[x['k']]) > TA]
            eff[c][f'+{s}'] = dict(removed=sum(len(set(x['orig']) - set(x['new'])) for x in L), added=sum(len(set(x['new']) - set(x['orig'])) for x in L),
                                   n_orig=sum(len(x['orig']) for x in L), list_div=s31lib.list_div(L))
    for cap in CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        for c in CFG:
            nav, lg, st = sim(VAR[c], cap)
            M[c][key] = tr_metrics(nav)
            if s == 0 and cap == 1e6:                       # 標準路徑：訓練窗分散度
                lt = lg[(lg['ent_date'] > TA) & (lg['ent_date'] <= TB)]
                DIV[c] = dict(holdings=s31lib.holdings_div(P, lg, TA, TB, IL), n_trades_train=int(len(lt)))
                # 訓練窗最賺 5 筆占比：訓練窗內進場且已平倉，損益 ÷ 訓練窗淨值增加
                x = lt[lt['why'] != 'open_end'].sort_values('pnl', ascending=False).head(5)
                DIV[c]['top5_share_train'] = float(x['pnl'].sum() / (nav.loc[TB] - nav.loc[TA]))
            del nav, lg, st
    del VAR
    if s:
        del S, IND
    gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

keys = list(M['CUR'])
res = {'PREREG_sha256': sha, 'acceptance': acc, 'window': dict(train=[str(TA.date()), str(TB.date())], y1618=[str(TA.date()), str(T18.date())], y1921=[str(T18.date()), str(TB.date())]),
       'rule_effect_train': eff, 'diversification_train': DIV, 'paths': M, 'summary': {}}
bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
res['bench_0050_train'] = stlib.cagr(bench, TA, TB)
for c in CFG:
    v = np.array([M[c][k]['train'] for k in keys])
    o = dict(train_median=float(np.median(v)), train_min=float(v.min()), train_max=float(v.max()),
             train_dd_median=float(np.median([M[c][k]['train_dd'] for k in keys])))
    if c != 'CUR':
        d = np.array([M[c][k]['train'] - M['CUR'][k]['train'] for k in keys]) * 100
        dd = np.array([M[c][k]['train_dd'] - M['CUR'][k]['train_dd'] for k in keys]) * 100
        d1 = np.array([M[c][k]['y1618'] - M['CUR'][k]['y1618'] for k in keys]) * 100
        d2 = np.array([M[c][k]['y1921'] - M['CUR'][k]['y1921'] for k in keys]) * 100
        e0 = eff[c]['+0']
        o.update(score_median_pp=float(np.median(d)), npos=int((d > 0).sum()), n_improve_025=int((d >= 0.25).sum()), diff_min_pp=float(d.min()), diff_max_pp=float(d.max()),
                 dd_diff_median_pp=float(np.median(dd)), y1618_median_pp=float(np.median(d1)), y1618_npos=int((d1 > 0).sum()),
                 y1921_median_pp=float(np.median(d2)), y1921_npos=int((d2 > 0).sum()), n_changed_shift0=int(e0['removed'] + e0['added']))
    res['summary'][c] = o
cand = [r['id'] for r in RULES]
best = sorted(cand, key=lambda c: (res['summary'][c]['score_median_pp'], res['summary'][c]['n_improve_025'], -res['summary'][c]['n_changed_shift0']), reverse=True)[0]
bs = res['summary'][best]
t0 = bool(bs['score_median_pp'] >= 1.0 and bs['n_improve_025'] >= 14)
rule = next(r for r in RULES if r['id'] == best)
res['selection'] = dict(order=sorted(cand, key=lambda c: res['summary'][c]['score_median_pp'], reverse=True), frozen=best, T0_pass=t0,
                        T0_text='訓練期找到可用的產業中性化規則' if t0 else '訓練期沒有找到（仍凍結分數最高的一條，驗證期只描述）')
res['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss())
(H / 'results_train.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
FZ = dict(written_at=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), PREREG_sha256=sha, frozen_rule=rule, T0_pass=t0, T0_text=res['selection']['T0_text'],
          train_score=bs, note='凍結後 validate.py 只跑一次；不得換規則或門檻。')
txt = json.dumps(FZ, ensure_ascii=False, indent=1, default=str)
(H / 'FROZEN.json').write_text(txt)
fsha = hashlib.sha256(txt.encode()).hexdigest()
(H / 'FROZEN.sha256').write_text(f'{fsha}  FROZEN.json\n')
(H / 'code.sha256').write_text(''.join(f'{hashlib.sha256((H / p).read_bytes()).hexdigest()}  {p}\n' for p in ('s31lib.py', 'prep.py', 'prereg_write.py', 'train.py')))
for c in CFG:
    print(c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res['summary'][c].items()}, flush=True)
    print('   分散', json.dumps(DIV[c], ensure_ascii=False))
print('凍結', best, 'T0', t0, fsha, res['meta'])
