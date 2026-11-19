#!/usr/bin/env python3
"""S74（描述性，不改規則）：月中跌破 60 日均線的持股，撐到換股執行日開盤對 0050 的超額；依深度、距換股日天數、買進以來賺賠、大盤位置分組。
規格 SPEC.json、用詞規則 PREREG.json（寫於任何事件報酬計算之前，sha256 啟動時核對）。只跑現行引擎主跑一條（100 萬、位移 0），不跑任何規則變體。
用法：python3 -I -B run.py。載入與訊號逐行照 agents/s57_intramonth_exit/run.py 的現行格。暫存 scratchpad/s74/（結束刪除）。"""
import sys, json, time, resource, gc, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s74')
CACHE = TMP / 'cache'
CACHE.mkdir(parents=True, exist_ok=True)
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for f in ('SPEC', 'PREREG'):
    assert sha(HERE / f'{f}.json') == (HERE / f'{f}.sha256').read_text().split()[0], f'{f}.json 與登記的 sha256 不符'

sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402
for mod, attr in ((s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                  (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(mod, attr, CACHE)
assert cslib.DEADLINE_FIX is False and cslib.INS15_MASK is False
cslib.EXRIGHT_FIX = True

P = s20lib.load()
assert P.exright_fix
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), '面板不符'
with np.errstate(invalid='ignore'):
    MA = stlib.rmean(P.G, 60)
    BELOW = np.isfinite(P.G) & np.isfinite(MA) & (P.G <= MA)
    DEPTH = P.G / MA - 1
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
F = s17lib.rev_family(P, defs=['STD'])
s17lib.e50_fix_suspension(P)
S0 = F['S']['STD']; months = list(S0.index)
del F; gc.collect()
cslib.DEADLINE_FIX = True; cslib.INS15_MASK = True
AV = cslib.rev_avail_days(P, months, True)
RT = dict(S=S0, av=AV)
print('載入', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TB = P.IDX[-1]; D19 = lastb('2019-01-01'); FULL0 = P.IDX[P.t0]
sg = s10lib.make_signal_s10(P, U, RT, shift=0, depth=10, mom_lo=0.10, nsel=10)
S2, ekw = s10lib.variant_S(sg, 'E0')
nav, lg, st = s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=1e6, sat_w=0.7, sizing='W10', lock=False, **ekw)
ref = json.loads((AG / 'deadline_engine_fix' / 'out' / 'results.json').read_text())['standard']['ALL']
acc = dict(since2019=stlib.cagr(nav, D19, TB), ref_since2019=ref['since2019'], n_closed=int((lg['why'] != 'open_end').sum()), ref_n_closed=ref['n_closed'],
           full=stlib.cagr(nav, FULL0, TB), ref_full=ref['full'], maxdd_full=stlib.maxdd(nav, FULL0, TB), ref_maxdd_full=ref['maxdd_full'])
acc['A1_bitwise'] = bool(acc['since2019'] == acc['ref_since2019'] and acc['n_closed'] == acc['ref_n_closed'] and acc['full'] == acc['ref_full']
                         and acc['maxdd_full'] == acc['ref_maxdd_full'])
print('驗收', json.dumps(acc, ensure_ascii=False), flush=True)
assert acc['A1_bitwise'], '驗收 A1 失敗'

# ---------------- 事件 ----------------
T = P.T; G, O, REF = P.G, P.O, P.REF
E50 = np.asarray(P.E50, float); E50O = np.asarray(P.E50O, float)
with np.errstate(invalid='ignore'):
    GO = np.full_like(G, np.nan); GO[1:] = G[:-1] * (O[1:] / REF[1:])     # 還原開盤（含息）；無開盤價為 NaN
    GO[~(np.isfinite(O) & (O > 0))] = np.nan
    m50 = stlib.rmean(E50[:, None], 60)[:, 0]
    MKT_BELOW = np.isfinite(m50) & (E50 <= m50)
X = S2['X']; sr = sg['sigrows']; sk = np.flatnonzero(sr)
ev = []; cnt = dict(n_hold=int(len(lg)), never=0, first_on_signal=0, mid=0, excl_no_next=0, exec_open_shifted=0)
for h in lg.itertuples(index=False):
    j, a, b = int(h.j), int(h.ent_t), int(h.exit_t)
    last = b if h.why == 'open_end' else b - 1             # 最後一個持有收盤日
    bl = np.flatnonzero(BELOW[a:last + 1, j])
    if not len(bl):
        cnt['never'] += 1; continue
    t = a + int(bl[0])
    if sr[t]:
        cnt['first_on_signal'] += 1; continue
    nk = sk[sk > t]
    if not len(nk) or int(nk[0]) + 1 >= T:
        cnt['excl_no_next'] += 1; continue
    k = int(nk[0]); e0 = k + 1; e = e0
    while e < T and not np.isfinite(GO[e, j]):
        e += 1
    if e >= T:
        cnt['excl_no_next'] += 1; continue
    if e != e0:
        cnt['exec_open_shifted'] += 1
    cnt['mid'] += 1
    rs = GO[e, j] / G[t, j] - 1; rb = E50O[e] / E50[t] - 1
    d = dict(code=str(h.code), j=j, ent_date=str(P.IDX[a].date()), event_date=str(P.IDX[t].date()), sig_date=str(P.IDX[k].date()), exec_date=str(P.IDX[e].date()),
             k=k, depth=float(DEPTH[t, j]), D=int(e0 - t), days_held=int(t - a), gain=float(G[t, j] / GO[a, j] - 1) if np.isfinite(GO[a, j]) else None,
             mkt_below=bool(MKT_BELOW[t]), ret_stock=float(rs), ret_0050=float(rb), ex=float(rs - rb))
    if t + 1 < T and np.isfinite(GO[t + 1, j]):
        d['ex_nextopen'] = float(GO[e, j] / GO[t + 1, j] - 1 - (E50O[e] / E50O[t + 1] - 1))
    else:
        d['ex_nextopen'] = None
    # 換股日結果
    d['above_at_k'] = bool(not BELOW[k, j] and np.isfinite(G[k, j]) and np.isfinite(MA[k, j]))
    d['sold_at_k'] = bool(X[k, j])
    d['depth_at_k'] = float(DEPTH[k, j])
    d['n_below_days'] = int(BELOW[t:k + 1, j].sum()); d['n_days_to_k'] = int(k - t + 1)
    if not d['sold_at_k']:
        if h.why == 'open_end':
            x1 = T - 1; s1 = G[x1, j]; b1 = E50[x1]
        else:
            x1 = b; s1 = GO[x1, j] if np.isfinite(GO[x1, j]) else G[x1 - 1, j]; b1 = E50O[x1]
        d['kept_ret'] = float(s1 / GO[e, j] - 1); d['kept_ex'] = float(s1 / GO[e, j] - 1 - (b1 / E50O[e] - 1)); d['kept_days'] = int(x1 - e)
        d['final_exit_date'] = str(P.IDX[x1].date()); d['final_why'] = str(h.why)
    d['trade_ret'] = float(h.ret); d['why'] = str(h.why)
    ev.append(d)
E = pd.DataFrame(ev)
print('事件', cnt, flush=True)


def clus_t(x, g):
    x = np.asarray(x, float); n = len(x)
    if n < 2:
        return None
    m = x.mean(); s = pd.Series(x - m).groupby(np.asarray(g)).sum().to_numpy()
    G_ = len(s)
    if G_ < 2:
        return None
    se = np.sqrt((s ** 2).sum() * G_ / (G_ - 1)) / n
    return float(m / se) if se > 0 else None


def stats(df, col='ex'):
    d = df[df[col].notna()]; x = d[col].to_numpy(float); n = len(x)
    if n == 0:
        return dict(n=0)
    return dict(n=n, mean=float(x.mean()), median=float(np.median(x)), win=float((x > 0).mean()), p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)),
                n_months=int(d['k'].nunique()), t_cluster=clus_t(x, d['k'].to_numpy()), unreliable=bool(n < 20))


def diff_t(df, mask, col='ex'):
    """該組 − 同維度其餘：以換股月份分群的群聚 t（對 x = a + b·組別 的 b）。"""
    d = df[df[col].notna()]; mk = np.asarray(mask)[df[col].notna().to_numpy()]
    y = d[col].to_numpy(float); n1, n0 = int(mk.sum()), int((~mk).sum())
    if n1 < 2 or n0 < 2:
        return None
    Xm = np.column_stack([np.ones(len(y)), mk.astype(float)])
    beta = np.linalg.lstsq(Xm, y, rcond=None)[0]; u = y - Xm @ beta
    XtXi = np.linalg.inv(Xm.T @ Xm); meat = np.zeros((2, 2)); gs = d['k'].to_numpy(); ng = 0
    for g in np.unique(gs):
        s = (Xm[gs == g] * u[gs == g, None]).sum(0); meat += np.outer(s, s); ng += 1
    V = XtXi @ meat @ XtXi * ng / (ng - 1)
    return dict(diff_mean=float(beta[1]), t_cluster=float(beta[1] / np.sqrt(V[1, 1])) if V[1, 1] > 0 else None,
                diff_median=float(np.median(y[mk]) - np.median(y[~mk])))


def block(df, masks):
    out = {}
    for name, mk in masks.items():
        sub = df[mk]
        o = dict(ex=stats(sub), ex_nextopen=stats(sub, 'ex_nextopen'), stock=stats(sub, 'ret_stock'), vs_rest=diff_t(df, mk),
                 above_at_k=float(sub['above_at_k'].mean()) if len(sub) else None, sold_at_k=float(sub['sold_at_k'].mean()) if len(sub) else None)
        v = o['vs_rest']
        o['clearly_different'] = bool(o['ex'].get('n', 0) >= 20 and v is not None and v['t_cluster'] is not None and abs(v['t_cluster']) >= 2.5
                                      and np.sign(v['diff_mean']) == np.sign(v['diff_median']))
        out[name] = o
    return out


dp = E['depth'].to_numpy(); D = E['D'].to_numpy(); yr = pd.to_datetime(E['event_date']).dt.year.to_numpy()
gain = E['gain'].to_numpy(float)
GRP = {
 '深度': {'0～−2%': dp > -0.02, '−2～−5%': (dp <= -0.02) & (dp > -0.05), '−5～−10%': (dp <= -0.05) & (dp > -0.10), '< −10%': dp <= -0.10},
 '距換股執行日': {'≤5 日': D <= 5, '6～10 日': (D >= 6) & (D <= 10), '>10 日': D > 10},
 '買進以來': {'賺': gain >= 0, '賠': gain < 0},
 '大盤（0050）': {'也在季線下': E['mkt_below'].to_numpy(), '在季線上': ~E['mkt_below'].to_numpy()},
 '時期': {'2023 年底以前': yr <= 2023, '2024 年起': yr >= 2024},
}
ALL = dict(ex=stats(E), ex_nextopen=stats(E, 'ex_nextopen'), stock=stats(E, 'ret_stock'), bench=stats(E, 'ret_0050'))
a = ALL['ex']
ALL['wording'] = ('幫忙' if (a['mean'] > 0 and a['t_cluster'] >= 2 and a['median'] > 0) else
                  '幫倒忙' if (a['mean'] < 0 and a['t_cluster'] <= -2 and a['median'] < 0) else '看不出差別')
groups = {g: block(E, m) for g, m in GRP.items()}
dm = [groups['深度'][k]['ex'].get('mean') for k in GRP['深度']]
deep = diff_t(E, dp <= -0.05)
depth_q = dict(means=dm, monotone_decreasing=bool(all(x is not None for x in dm) and all(dm[i] > dm[i + 1] for i in range(3))), deep_vs_shallow=deep,
               deep=stats(E[dp <= -0.05]), shallow=stats(E[dp > -0.05]))
depth_q['verdict'] = '有跡象' if (depth_q['monotone_decreasing'] and deep and deep['t_cluster'] is not None and deep['t_cluster'] <= -2.5 and deep['diff_median'] < 0) else '資料不支持'

sold = E[E['sold_at_k']]; kept = E[~E['sold_at_k']]
outcome = dict(n=int(len(E)), above_at_k=float(E['above_at_k'].mean()), kept=dict(n=int(len(kept)), frac=float(len(kept) / len(E))),
               sold=dict(n=int(len(sold)), frac=float(len(sold) / len(E)), still_below=int((~sold['above_at_k']).sum()), above_but_sold=int(sold['above_at_k'].sum())),
               ex_sold=stats(sold), ex_kept=stats(kept), stock_sold=stats(sold, 'ret_stock'), stock_kept=stats(kept, 'ret_stock'),
               kept_after_ex=stats(kept, 'kept_ex'), kept_after_ret=stats(kept, 'kept_ret'), kept_after_days_median=float(kept['kept_days'].median()) if len(kept) else None,
               sold_still_below_ex=stats(sold[~sold['above_at_k']]), sold_above_ex=stats(sold[sold['above_at_k']]),
               frac_days_below_median=float((E['n_below_days'] / E['n_days_to_k']).median()))
s57 = json.loads((AG / 's57_intramonth_exit' / 'results.json').read_text())['cells']
rule_ref = {c: dict(path_median_since2019=s57[c]['path']['median'], maxdd_full_median=s57[c]['path']['maxdd_full_median'], calmar_median=s57[c]['path']['calmar_median'],
                    vs_current={k: v for k, v in s57[c].get('vs_current', {}).items() if k != 'seg_dd'}, post_sell_main=s57[c]['post_sell_main'])
            for c in ('CUR', 'D_MA', 'D_MA2')}
res = dict(spec_sha256=sha(HERE / 'SPEC.json'), prereg_sha256=sha(HERE / 'PREREG.json'), run_sha256=sha(Path(__file__)),
           flags=dict(EXRIGHT_FIX=cslib.EXRIGHT_FIX, DEADLINE_FIX=cslib.DEADLINE_FIX, INS15_MASK=cslib.INS15_MASK), acceptance=acc,
           counts=dict(cnt, frac_mid=cnt['mid'] / cnt['n_hold'], n_events=int(len(E)), first_event=str(E['event_date'].min()), last_event=str(E['event_date'].max()),
                       depth_median=float(np.median(dp)), D_median=float(np.median(D)), days_held_median=float(E['days_held'].median())),
           all=ALL, groups=groups, depth_question=depth_q, rebalance_outcome=outcome, rule_reference_S57=rule_ref,
           by_year={int(y): stats(E[yr == y]) for y in sorted(set(yr))},
           events=[{k_: (None if isinstance(v_, float) and v_ != v_ else v_) for k_, v_ in r_.items()} for r_ in E.drop(columns=['j', 'k']).to_dict('records')], runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
f = lambda s: 'n=0' if not s.get('n') else f"n={s['n']} 平均 {s['mean']*100:+.2f}% 中位 {s['median']*100:+.2f}% 勝率 {s['win']*100:.0f}% P10 {s['p10']*100:+.1f}% P90 {s['p90']*100:+.1f}% 月數 {s['n_months']} t {s['t_cluster'] if s['t_cluster'] is None else round(s['t_cluster'],2)}"
print('計數', json.dumps(res['counts'], ensure_ascii=False))
print('全部 超額', f(ALL['ex']), '|', ALL['wording']); print('  次日開盤起', f(ALL['ex_nextopen'])); print('  股票自身', f(ALL['stock'])); print('  0050', f(ALL['bench']))
for g, o in groups.items():
    for k, v in o.items():
        print(g, k, f(v['ex']), '| 次開', f(v['ex_nextopen']), '| 對其餘', None if v['vs_rest'] is None else {a_: round(b_, 4) for a_, b_ in v['vs_rest'].items() if b_ is not None},
              '站回', v['above_at_k'] and round(v['above_at_k'], 2), '賣', v['sold_at_k'] and round(v['sold_at_k'], 2), '明顯不同' if v['clearly_different'] else '')
print('深度問題', json.dumps({k: v for k, v in depth_q.items() if k not in ('deep', 'shallow')}, ensure_ascii=False)); print(' 深', f(depth_q['deep'])); print(' 淺', f(depth_q['shallow']))
print('換股日', json.dumps({k: v for k, v in outcome.items() if not isinstance(v, dict) or k in ('kept', 'sold')}, ensure_ascii=False))
for k in ('ex_sold', 'ex_kept', 'stock_sold', 'stock_kept', 'kept_after_ex', 'kept_after_ret', 'sold_still_below_ex', 'sold_above_ex'):
    print(' ', k, f(outcome[k]))
for y, s in res['by_year'].items():
    print(y, f(s))
print('done', res['runtime_s'], 's', res['maxrss_mb'], 'MB')
