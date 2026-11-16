#!/usr/bin/env python3
"""S34 事後描述（驗證期判定已凍結；本檔不改判定）。
(1) 事前登記的「受影響月份逐筆超額比較」：4 條規則 × 訓練／驗證，校正前（現行）名單 vs 校正後名單，E0 持有期超額（對 0050，扣成本前），群聚 t。
(2) 事前登記的隨機對照：凍結規則 ADJ2 的機制套到每年隨機相鄰兩月（200 組）；訓練窗與驗證窗（重新起跑）20 條路徑配對差中位數的分布與百分位。
    另外（未登記）同樣做 JF_SKIP、WD、NOBUY 各 200 組，只描述。
(3) 未登記：其他 3 條規則的驗證窗（重新起跑）20 條路徑；ADJ2 驗證期逐年差；ADJ2 與現行名單差異最大的股票（只在一邊的交易損益）。
輸出 results_posthoc.json。python3 -I -B posthoc.py"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s34lib
from s34lib import s17lib, s10lib, s20lib, s4lib, stlib, COSTS

H = Path(__file__).resolve().parent
for nm in ('PREREG', 'FROZEN'):
    assert hashlib.sha256((H / f'{nm}.json').read_bytes()).hexdigest() == (H / f'{nm}.sha256').read_text().split()[0]
FZ = json.loads((H / 'FROZEN.json').read_text())
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s34lib.load()
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
TD = s34lib.trading_days(); Rv = s34lib.revenue(P)
RT = s34lib.surge_rule(P, Rv, 'none', TD)
s17lib.e50_fix_suspension(P)
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TA, TB, DE = lastb('2016-02-01'), lastb('2022-01-01'), P.IDX[-1]
D24 = lastb('2024-01-01')
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
RULES = {r['id']: r for r in s34lib.RULES}
per = lambda m: 'train' if pd.Period(m, 'M') <= s34lib.TRAIN_LAST_MONTH else ('valid' if pd.Period(m, 'M') >= s34lib.VALID_FIRST_MONTH else 'pre')
res = {'frozen': FZ['frozen_rule']['id']}


def sim(S1, cap=1e6, fresh=False, block=None):
    if fresh:
        S1 = s34lib.fresh_from(S1, s34lib.VALID_FIRST_MONTH)
    S2, ekw = s10lib.variant_S(S1, 'E0')
    return s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=cap, block=block, **CELL, **ekw)


S0s = {s: s10lib.make_signal_s10(P, U, RT, shift=s) for s in range(4)}
nav1, _, _ = sim(S0s[0], 1e6, fresh=True)
ratio = float(nav1.loc[TB] / 1e6)
FCAP = {c: c / ratio for c in CAPS}
CUR = {}
for s in range(4):
    for cap in CAPS:
        k = f'{int(cap/1e4)}萬|+{s}'
        CUR[k] = dict(train=stlib.cagr(sim(S0s[s], cap)[0], TA, TB), valid=stlib.cagr(sim(S0s[s], FCAP[cap], fresh=True)[0], TB, DE))


def paths(builder):
    """builder(s) → S1；回傳訓練、驗證（重新起跑）20 條路徑對 CUR 配對差（個百分點）。"""
    dt, dv = [], []
    for s in range(4):
        S1 = builder(s)
        S1 = {k: v for k, v in S1.items() if k not in ('XV', 'RQ')} | {'XV': {'E0': S1['X']}}
        for cap in CAPS:
            k = f'{int(cap/1e4)}萬|+{s}'
            dt.append((stlib.cagr(sim(S1, cap)[0], TA, TB) - CUR[k]['train']) * 100)
            dv.append((stlib.cagr(sim(S1, FCAP[cap], fresh=True)[0], TB, DE) - CUR[k]['valid']) * 100)
        del S1
    dt, dv = np.array(dt), np.array(dv)
    return dict(train_median=float(np.median(dt)), train_npos=int((dt > 0).sum()), valid_median=float(np.median(dv)), valid_npos=int((dv > 0).sum()),
                valid_min=float(dv.min()), valid_max=float(dv.max()))


# ---------------------------------------------------------------- (3) 其他規則驗證窗、ADJ2 逐年
# 記憶體：每個訊號約 100MB，只保留位移 0（逐筆比較用），其他位移用完即丟
SR = {rid: s34lib.build(P, U, Rv, TD, r, 0, S_cur=S0s[0], RT_cur=RT)[0] for rid, r in RULES.items()}
res['all_rules_paths（驗證窗未登記描述；訓練窗同 results_train）'] = {
    rid: paths(lambda s, rid=rid, r=r: SR[rid] if s == 0 else s34lib.build(P, U, Rv, TD, r, s, S_cur=S0s[s], RT_cur=RT)[0]) for rid, r in RULES.items()}
gc.collect()
print('(3) 規則路徑', f'{time.time()-t_all:.0f}s', flush=True)
bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
navc, lgc, _ = sim(S0s[0]); nava, lga, _ = sim(SR['ADJ2'])
yc, ya = stlib.yearly(navc, bench), stlib.yearly(nava, bench)
res['ADJ2_yearly_cont_100萬'] = {y: dict(CUR=yc[y]['strat'], ADJ2=ya[y]['strat'], diff_pp=(ya[y]['strat'] - yc[y]['strat']) * 100) for y in yc}
# 只在一邊的交易（全期連續主跑，依進場訊號日）
kc = set(zip(lgc['code'], lgc['sig_date'].astype(str))); ka = set(zip(lga['code'], lga['sig_date'].astype(str)))
oc = lgc[[(c, d) not in ka for c, d in zip(lgc['code'], lgc['sig_date'].astype(str))]]
oa = lga[[(c, d) not in kc for c, d in zip(lga['code'], lga['sig_date'].astype(str))]]
fmt = lambda d: [dict(code=r.code, sig=str(pd.Timestamp(r.sig_date).date()), exit=str(pd.Timestamp(r.exit_date).date()), ret=round(float(r.ret), 4), pnl=round(float(r.pnl))) for r in d.itertuples()]
res['ADJ2_only_one_side_trades_cont_100萬'] = dict(
    CUR_only=dict(n=len(oc), pnl_sum=float(oc['pnl'].sum()), mean_ret=float(oc['ret'].mean()), top=fmt(oc.sort_values('pnl', ascending=False).head(8)), bottom=fmt(oc.sort_values('pnl').head(5))),
    ADJ2_only=dict(n=len(oa), pnl_sum=float(oa['pnl'].sum()), mean_ret=float(oa['ret'].mean()), top=fmt(oa.sort_values('pnl', ascending=False).head(8)), bottom=fmt(oa.sort_values('pnl').head(5))),
    by_sig_month_cal={g: dict(CUR_only_pnl=float(oc[pd.to_datetime(oc['sig_date']).dt.month.isin(ms)]['pnl'].sum()), ADJ2_only_pnl=float(oa[pd.to_datetime(oa['sig_date']).dt.month.isin(ms)]['pnl'].sum()))
                      for g, ms in (('訊號 2～4 月（營收 1～3 月）', [2, 3, 4]), ('其他訊號月', [1, 5, 6, 7, 8, 9, 10, 11, 12]))})
del navc, nava

# ---------------------------------------------------------------- (1) 受影響月份逐筆超額
END = P.T - 1


def outc(S, rows):
    """S：訊號；rows：[(month, k, [j...])] → DataFrame（month, code, excess）。"""
    ks = [x['k'] for x in S['info'] if S['sig'][x['k']].any()] + [x['k'] for x in S['info']]
    sig_ks = sorted(set(x['k'] for x in S['info']))
    rr = [(m, k, j) for m, k, js in rows for j in js]
    if not rr:
        return pd.DataFrame(columns=['month', 'k', 'j', 'code', 'excess'])
    cand = pd.DataFrame(rr, columns=['month', 'k', 'j'])
    o = s20lib.outcomes(P, S['X'], sig_ks, cand, END)
    cand['excess'] = o['excess'].to_numpy(); cand['code'] = [str(P.COLS[j]) for j in cand['j']]
    return cand


def ctest(d):
    d = np.array([x for x in d if np.isfinite(x)])
    if len(d) < 2:
        return dict(n_months=int(len(d)), mean=float(d.mean()) if len(d) else None, t=None)
    return dict(n_months=int(len(d)), mean=float(d.mean()), t=float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if d.std(ddof=1) > 0 else None)


TL = {}
for rid, r in RULES.items():
    S1 = SR[rid]; S0 = S0s[0]
    aff = (1, 2, 3) if rid == 'ADJ2' else (1, 2)
    i1 = {x['month']: x['k'] for x in S1['info']}
    rows0, rows1 = [], []
    for x in S0['info']:
        m = x['month']
        if pd.Period(m, 'M').month not in aff or per(m) == 'pre':
            continue
        rows0.append((m, x['k'], list(np.flatnonzero(S0['sig'][x['k']]))))
        k1 = i1.get(m)
        rows1.append((m, k1, list(np.flatnonzero(S1['sig'][k1])) if k1 is not None else []))
    o0 = outc(S0, rows0); o1 = outc(S1, rows1)
    out = {}
    for p in ('train', 'valid', 'all'):
        ms = [m for m, _, _ in rows0 if (p == 'all' or per(m) == p)]
        diffs, detail = [], []
        sh = dict(common=[], cur_only=[], new_only=[])
        for m in ms:
            a = o0[o0['month'] == m]; b = o1[o1['month'] == m]
            ea = float(a['excess'].mean()) if len(a) else 0.0
            eb = float(b['excess'].mean()) if len(b) else 0.0
            if len(a) == 0 and len(b) == 0:
                continue
            diffs.append(eb - ea)
            ca, cb = set(a['code']), set(b['code'])
            sh['common'] += list(a[a['code'].isin(cb)]['excess']); sh['cur_only'] += list(a[~a['code'].isin(cb)]['excess']); sh['new_only'] += list(b[~b['code'].isin(ca)]['excess'])
            detail.append(dict(month=m, n_cur=len(a), n_new=len(b), cur_mean=ea, new_mean=eb, diff=eb - ea, cur_only=sorted(ca - cb), new_only=sorted(cb - ca)))
        out[p] = dict(cluster=ctest(diffs), groups={g: dict(n=len(v), mean=float(np.nanmean(v)) if v else None) for g, v in sh.items()},
                      detail=detail if p != 'all' else None)
    TL[rid] = out
res['affected_months_trade_level'] = TL
del SR
gc.collect()
print('(1) 逐筆', f'{time.time()-t_all:.0f}s', flush=True)

# ---------------------------------------------------------------- (2) 隨機對照
years = sorted(set(Rv.index.year))
NS = 200
RC = {}
for rid, r in RULES.items():
    vals = []
    for i in range(NS):
        pr = s34lib.random_pairs(20261116 + i, years)
        if r['kind'] == 'surge':
            skip = [pd.Period(f'{y}-{a:02d}', 'M') for y, a, b in pr] if r['mode'] == 'jf' else None
            RTr = s34lib.surge_rule(P, Rv, r['mode'], TD, months=pr, skip=skip)
            bld = lambda s, RTr=RTr: s10lib.make_signal_s10(P, U, RTr, shift=s)
        else:
            ms = set(pd.Period(f'{y}-{m:02d}', 'M') for y, a, b in pr for m in (a, b))
            bld = lambda s, ms=ms: s34lib.nobuy(S0s[s], ms)[0]
        vals.append(paths(bld))
    tm = np.array([v['train_median'] for v in vals]); vm = np.array([v['valid_median'] for v in vals])
    real = res['all_rules_paths（驗證窗未登記描述；訓練窗同 results_train）'][rid]
    RC[rid] = dict(registered=(rid == FZ['frozen_rule']['id']), n=NS,
                   train=dict(real=real['train_median'], q05=float(np.quantile(tm, .05)), q50=float(np.median(tm)), q95=float(np.quantile(tm, .95)), pct_rank=float((tm < real['train_median']).mean())),
                   valid=dict(real=real['valid_median'], q05=float(np.quantile(vm, .05)), q50=float(np.median(vm)), q95=float(np.quantile(vm, .95)), pct_rank=float((vm < real['valid_median']).mean())),
                   frac_random_pass_T0like=float(np.mean([(v['train_median'] >= 1.0) for v in vals])))
    print('(2) 隨機', rid, f'{time.time()-t_all:.0f}s', rss(), 'MB', json.dumps(RC[rid], ensure_ascii=False), flush=True)
    gc.collect()
res['random_control'] = RC
res['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss(), fresh_capital_scale=ratio)
(H / 'results_posthoc.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k: v for k, v in res.items() if k not in ('affected_months_trade_level',)}, ensure_ascii=False, default=str)[:6000])
for rid in TL:
    print(rid, {p: (TL[rid][p]['cluster'], TL[rid][p]['groups']) for p in TL[rid]})
