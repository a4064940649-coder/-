#!/usr/bin/env python3
"""S20 訓練期：7 條候選規則＋安慰劑 × 20 條路徑，只算訓練窗（2016-01-29 → 2021-12-30）；依 PREREG 選一條凍結。
輸出 results_train.json、FROZEN.json、FROZEN.sha256。PREREG.json 的 sha256 必須相符；FROZEN.json 已存在就拒絕重跑。
python3 -I -B train.py"""
import sys, json, hashlib, time, resource, gc, datetime
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s20lib
from s20lib import s17lib, s10lib, s4lib, s2lib, stlib, COSTS

H = Path(__file__).resolve().parent
pre = (H / 'PREREG.json').read_bytes()
sha = hashlib.sha256(pre).hexdigest()
assert sha == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與 PREREG.sha256 不符'
assert not (H / 'FROZEN.json').exists(), 'FROZEN.json 已存在，拒絕重跑'
PRE = json.loads(pre)
RULES = PRE['rule_form']['candidates（7 條，全部計入 N；門檻＝訓練期位移 0 F1 名單的分位數，numpy linear，凍結成數字）']
PLACEBO = PRE['rule_form']['placebo（只描述，不參與選擇與判定）']
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s20lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
G = s20lib.growth_features(P)
RT = dict(S=G['S'], av=G['av'])
fix = s17lib.e50_fix_suspension(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
print('載入完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TA, TB = lastb('2016-02-01'), lastb('2022-01-01')           # 訓練窗 2016-01-29 → 2021-12-30
T18 = lastb('2019-01-01')
D19 = lastb('2019-01-01')
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
CELL = dict(sat_w=0.7, sizing='W10', lock=False)


def sim(S, rule, cap=1e6, cost='strict'):
    S1, _ = s20lib.apply_rule(P, S, G, rule)
    S2, ekw = s10lib.variant_S(S1, 'E0')
    return s10lib.simulate_s10(P, S2, U, cost=COSTS[cost], capital=cap, **CELL, **ekw)


def tr_metrics(nav):
    return dict(train=stlib.cagr(nav, TA, TB), train_dd=stlib.maxdd(nav, TA, TB), y1618=stlib.cagr(nav, TA, T18), y1921=stlib.cagr(nav, T18, TB))


# ---------------------------------------------------------------- 門檻核對、驗收
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
f1 = s20lib.f1_table(P, S0, G)
f1['mper'] = pd.PeriodIndex(f1['month'], freq='M')
tr = f1[f1['mper'] <= s20lib.TRAIN_LAST_MONTH]
qs = [float(np.quantile(tr['MS'], q)) for q in (1 / 3, 1 / 2, 2 / 3)]
for r, q in zip(RULES[:3], qs):
    assert abs(r['thr'] - q) < 1e-12, ('門檻與 PREREG 不符', r, q)
acc = {'P0': True, 'e50_fix': fix}
navc, lgc, stc = sim(S0, None)
a0 = dict(since2019=stlib.cagr(navc, D19, P.IDX[-1]), full=stlib.cagr(navc, P.IDX[P.t0], P.IDX[-1]), maxdd=stlib.maxdd(navc, P.IDX[P.t0], P.IDX[-1]))
a0['ok'] = bool(abs(a0['since2019'] - 0.4813) < 0.00025 and abs(a0['full'] - 0.3408) < 0.00025 and abs(a0['maxdd'] + 0.3340) < 0.00025)
acc['A0'] = a0
nava, lga, _ = sim(S0, dict(feat='MS', op='>=', thr=-np.inf))
cols = ['code', 'ent_t', 'exit_t', 'why']
acc['A2'] = dict(trades_identical=bool(len(lga) == len(lgc) and (lga[cols].reset_index(drop=True) == lgc[cols].reset_index(drop=True)).all().all()),
                 nav_identical=bool(np.allclose(nava.to_numpy(), navc.to_numpy(), rtol=0, atol=1e-6)))
assert a0['ok'] and acc['A2']['trades_identical'] and acc['A2']['nav_identical'], ('驗收失敗', acc)
del navc, lgc, stc, nava, lga
print('驗收', json.dumps({k: v for k, v in acc.items() if k != 'e50_fix'}, ensure_ascii=False), flush=True)

# ---------------------------------------------------------------- 訓練期 F1 留下比例（位移 0）
keep_frac = {}
for r in RULES + [PLACEBO]:
    k = [s20lib.rule_keep_value(G, r['feat'], r['op'], r['thr'], m, c) for m, c in zip(tr['mper'], tr['code'])]
    keep_frac[r['id']] = float(np.mean(k))

# ---------------------------------------------------------------- 20 條路徑 × 9 個設定（只算訓練窗）
CFG = {'CUR': None}
for r in RULES + [PLACEBO]:
    CFG[r['id']] = dict(feat=r['feat'], op=r['op'], thr=r['thr'])
M = {c: {} for c in CFG}
for s in range(4):
    S = S0 if s == 0 else s10lib.make_signal_s10(P, U, RT, shift=s)
    for cap in CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        for c, rule in CFG.items():
            nav, lg, st = sim(S, rule, cap)
            M[c][key] = tr_metrics(nav)
            del nav, lg, st
    if s:
        del S
    gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

keys = list(M['CUR'])
res = {'PREREG_sha256': sha, 'acceptance': acc, 'window': dict(train=[str(TA.date()), str(TB.date())], y1618=[str(TA.date()), str(T18.date())], y1921=[str(T18.date()), str(TB.date())]),
       'thresholds_check': qs, 'keep_frac_train_f1': keep_frac, 'bench_0050_train': stlib.cagr(bench, TA, TB), 'paths': M, 'summary': {}}
for c in CFG:
    v = np.array([M[c][k]['train'] for k in keys])
    o = dict(train_median=float(np.median(v)), train_min=float(v.min()), train_max=float(v.max()),
             train_dd_median=float(np.median([M[c][k]['train_dd'] for k in keys])))
    if c != 'CUR':
        d = np.array([M[c][k]['train'] - M['CUR'][k]['train'] for k in keys]) * 100
        dd = np.array([M[c][k]['train_dd'] - M['CUR'][k]['train_dd'] for k in keys]) * 100
        d1 = np.array([M[c][k]['y1618'] - M['CUR'][k]['y1618'] for k in keys]) * 100
        d2 = np.array([M[c][k]['y1921'] - M['CUR'][k]['y1921'] for k in keys]) * 100
        o.update(score_median_pp=float(np.median(d)), npos=int((d > 0).sum()), diff_min_pp=float(d.min()), diff_max_pp=float(d.max()),
                 dd_diff_median_pp=float(np.median(dd)), y1618_median_pp=float(np.median(d1)), y1618_npos=int((d1 > 0).sum()),
                 y1921_median_pp=float(np.median(d2)), y1921_npos=int((d2 > 0).sum()), keep_frac=keep_frac[c])
    res['summary'][c] = o
cand = [r['id'] for r in RULES]
best = sorted(cand, key=lambda c: (res['summary'][c]['score_median_pp'], res['summary'][c]['npos'], res['summary'][c]['keep_frac']), reverse=True)[0]
bs = res['summary'][best]
t0 = bool(bs['score_median_pp'] >= 1.0 and bs['npos'] >= 14)
rule = next(r for r in RULES if r['id'] == best)
res['selection'] = dict(order=sorted(cand, key=lambda c: res['summary'][c]['score_median_pp'], reverse=True), frozen=best, T0_pass=t0,
                        T0_text='訓練期找到可用的持續成長篩選' if t0 else '訓練期沒有找到（仍凍結分數最高的一條，驗證期只描述）')
res['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss())
(H / 'results_train.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
FZ = dict(written_at=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), PREREG_sha256=sha, frozen_rule=rule, T0_pass=t0, T0_text=res['selection']['T0_text'],
          train_score=bs, placebo=PLACEBO, note='凍結後 validate.py 只跑一次；不得換門檻或特徵。')
txt = json.dumps(FZ, ensure_ascii=False, indent=1, default=str)
(H / 'FROZEN.json').write_text(txt)
fsha = hashlib.sha256(txt.encode()).hexdigest()
(H / 'FROZEN.sha256').write_text(f'{fsha}  FROZEN.json\n')
for c in CFG:
    print(c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res['summary'][c].items()}, flush=True)
print('凍結', best, 'T0', t0, fsha, res['meta'])
