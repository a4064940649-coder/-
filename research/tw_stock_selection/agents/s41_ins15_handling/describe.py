#!/usr/bin/env python3
"""S41 第 1 步（描述，事前登記之前跑；只看現行 CUR 與遮蔽格數，不算任何處理 A／B／C 的報酬）：
(0) 驗收：A0＝CUR 重現 48.99／34.54／−32.11／381；A1＝A、B、C 在「這群公司一律看得到」（cells 空）時交易與淨值與 CUR 逐位相同。
(1) 2016-02～2026-10 訊號：這群公司（S40 名單 18 檔＋2884）在股池、有 SURGE、L_MA 前 20、F1 的月數；CUR 標準路徑交易次數、損益與占比。
(2) 反事實（一直是 15 日期限）每個位移被遮的股月數、遮蔽月份占比；遮蔽格中「現行會在 F1／前 20／持有中」的數量。
(3) 2026 起實際受影響：依法定期限（每月）與 Tasi 首次可見時間（S40 資料）各列月份與檔數；CUR 在這些月份這群公司的 F1／前 20／持股。
輸出 results_describe.json。python3 -I -B describe.py"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s41lib as L
from s41lib import s10lib, stlib

H = Path(__file__).resolve().parent
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT = L.prep_panel()
S0, av = RT['S'], RT['av']
assert list(S0.columns) == list(P.COLS)
GJ = {c: int(np.flatnonzero(P.COLS == c)[0]) for c in L.GROUP if (P.COLS == c).any()}
res = dict(group=L.GROUP, group_in_panel=list(GJ), group_missing=[c for c in L.GROUP if c not in GJ])
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D19, DE = lastb('2019-01-01'), P.IDX[-1]

# ---------------------------------------------------------------- (0) 驗收
sgC = s10lib.make_signal_s10(P, U, RT, shift=0)
navC, lgC, _ = L.sim(P, U, sgC)
a0 = dict(since2019=stlib.cagr(navC, D19, DE), full=stlib.cagr(navC, P.IDX[P.t0], DE), maxdd=stlib.maxdd(navC, P.IDX[P.t0], DE),
          n_trades=int((lgC['why'] != 'open_end').sum()))
a0['ok'] = bool(abs(a0['since2019'] - L.ACC['since2019']) < 0.00025 and abs(a0['full'] - L.ACC['full']) < 0.00025
                and abs(a0['maxdd'] - L.ACC['maxdd']) < 0.00025 and a0['n_trades'] == L.ACC['n_trades'])
acc = dict(A0=a0)
cols = ['code', 'ent_t', 'exit_t', 'why', 'shares']
for tr in 'ABC':
    sg, _ = L.build(P, U, RT, tr, [], 0)
    nav, lg, _ = L.sim(P, U, sg)
    acc[f'A1_{tr}'] = dict(nav_bitwise=bool(np.array_equal(nav.to_numpy(), navC.to_numpy())),
                           trades_identical=bool(len(lg) == len(lgC) and (lg[cols].reset_index(drop=True) == lgC[cols].reset_index(drop=True)).all().all()),
                           X_identical=bool(np.array_equal(sg['X'], sgC['X'])), sig_identical=bool(np.array_equal(sg['sig'], sgC['sig'])))
res['acceptance'] = acc
assert a0['ok'] and all(all(v.values()) for k, v in acc.items() if k.startswith('A1')), acc
print('驗收', json.dumps(acc, ensure_ascii=False), flush=True)

# ---------------------------------------------------------------- (1) 出現頻率（CUR、位移 0）
info = {x['month']: x for x in sgC['info']}
months = [pd.Period(m, 'M') for m in info if pd.Period(m, 'M') >= pd.Period('2016-01', 'M')]
per = {c: dict(univ=0, surge=0, lma=0, top20=0, f1=0) for c in GJ}
mrow = []
for m in months:
    k = info[str(m)]['k']; s = S0.loc[m].to_numpy()
    fin = U[k] & np.isfinite(s); ok_ma = fin & P.F_MA60[k]
    o_ma = np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]
    top20 = set(o_ma[:20].tolist()); f1 = set(np.flatnonzero(sgC['sig'][k]).tolist())
    nt = nf = 0
    for c, j in GJ.items():
        per[c]['univ'] += int(U[k, j]); per[c]['surge'] += int(fin[j]); per[c]['lma'] += int(ok_ma[j])
        per[c]['top20'] += int(j in top20); per[c]['f1'] += int(j in f1)
        nt += int(j in top20); nf += int(j in f1)
    mrow.append(dict(month=str(m), sig_date=info[str(m)]['sig_date'], n_top20=nt, n_f1=nf,
                     codes_top20=[str(P.COLS[j]) for j in o_ma[:20] if str(P.COLS[j]) in GJ], codes_f1=[str(P.COLS[j]) for j in f1 if str(P.COLS[j]) in GJ]))
tot = dict(n_months=len(months), top20_slots=20 * len(months), f1_total=int(sum(len(np.flatnonzero(sgC['sig'][info[str(m)]['k']])) for m in months)))
gsum = {f: int(sum(per[c][f] for c in per)) for f in ('univ', 'surge', 'lma', 'top20', 'f1')}
gsum['months_any_top20'] = int(sum(1 for r in mrow if r['n_top20'] > 0)); gsum['months_any_f1'] = int(sum(1 for r in mrow if r['n_f1'] > 0))
lgC = lgC.copy()
g = lgC[lgC['code'].isin(list(GJ))]
gain = float(navC.iloc[-1] - navC.iloc[0])
tr = dict(n_trades_all=int(len(lgC)), n_trades_group=int(len(g)), n_closed_group=int((g['why'] != 'open_end').sum()),
          pnl_group=float(g['pnl'].sum()), pnl_all=float(lgC['pnl'].sum()), nav_gain=gain,
          pnl_group_share_of_trades_pnl=float(g['pnl'].sum() / lgC['pnl'].sum()), pnl_group_share_of_nav_gain=float(g['pnl'].sum() / gain),
          mean_ret_group=float(g['ret'].mean()) if len(g) else None, median_ret_group=float(g['ret'].median()) if len(g) else None,
          win_rate_group=float((g['ret'] > 0).mean()) if len(g) else None, mean_ret_all=float(lgC['ret'].mean()),
          trades=[dict(code=y.code, sig=str(y.sig_date.date()), ent=str(y.ent_date.date()), exit=str(y.exit_date.date()), why=y.why,
                       ret=round(float(y.ret), 4), pnl=round(float(y.pnl)), hold_days=int(y.hold_days)) for y in g.sort_values('ent_t').itertuples()])
for nm, (a, b) in dict(train=('2016-02-01', '2021-12-31'), valid=('2022-01-01', '2026-12-31')).items():
    x = g[(g['sig_date'] >= a) & (g['sig_date'] <= b)]; y = lgC[(lgC['sig_date'] >= a) & (lgC['sig_date'] <= b)]
    tr[nm] = dict(n_group=int(len(x)), n_all=int(len(y)), pnl_group=float(x['pnl'].sum()), pnl_all=float(y['pnl'].sum()))
res['describe_cur'] = dict(per_code=per, group_sum=gsum, totals=tot, trades=tr, months_with_group=[r for r in mrow if r['n_top20'] or r['n_f1']])
print('描述', json.dumps(dict(gsum=gsum, tot=tot, tr={k: v for k, v in tr.items() if k != 'trades'}), ensure_ascii=False), flush=True)


# ---------------------------------------------------------------- (2) 反事實遮蔽格數
def held_at(lg, k):
    return set(int(j) for j, e, x in zip(lg['j'], lg['ent_t'], lg['exit_t']) if e <= k < x)


cf = {}
for s in range(4):
    cells = L.hidden_cells(P, av, S0, s)
    hm = sorted(set(m for m, _ in cells if m >= pd.Period('2016-01', 'M')))
    allm = [m for m in S0.index if av.get(m, P.T) + s < P.T and m >= pd.Period('2016-01', 'M')]
    cf[f'+{s}'] = dict(n_cells=len(cells), n_months_hidden=len(hm), n_months_total=len(allm),
                       months_visible=[str(m) for m in allm if m not in set(hm)])
# 位移 0：遮蔽格中現行會是 F1／前 20／持有中的
cells0 = L.hidden_cells(P, av, S0, 0)
hit = dict(f1=0, top20=0, held=0)
hl = []
for m, j in cells0:
    x = info.get(str(m))
    if x is None or m < pd.Period('2016-01', 'M'):
        continue
    k = x['k']; s = S0.loc[m].to_numpy(); ok_ma = U[k] & np.isfinite(s) & P.F_MA60[k]
    o_ma = np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]
    a, b, c = bool(sgC['sig'][k, j]), bool(j in set(o_ma[:20].tolist())), bool(j in held_at(lgC, k))
    hit['f1'] += a; hit['top20'] += b; hit['held'] += c
    if a or b or c:
        hl.append(dict(month=str(m), code=str(P.COLS[j]), f1=a, top20=b, held=c))
cf['shift0_hits'] = hit; cf['shift0_hit_list'] = hl
res['counterfactual_cells'] = cf
print('反事實', json.dumps({k: (v if not isinstance(v, dict) else {a: b for a, b in v.items() if a != 'months_visible'}) for k, v in cf.items() if k != 'shift0_hit_list'}, ensure_ascii=False), flush=True)

# ---------------------------------------------------------------- (3) 2026 起實際受影響
real = []
for m in pd.period_range('2026-01', '2026-10', freq='M'):
    d10 = (m + 1).to_timestamp() + pd.Timedelta(days=9)
    if m in av and av[m] < P.T:
        sd = P.IDX[av[m]]; d15 = P.IDX[L.d15_index(P, m)] if L.d15_index(P, m) < P.T else None
    else:                                                  # 面板外（2026-10 營收、2026-11 訊號）：週末推算（不含國定假日）
        dd = pd.bdate_range(d10, periods=3)
        sd = dd[1]; d15 = pd.bdate_range((m + 1).to_timestamp() + pd.Timedelta(days=14), periods=1)[0]
    real.append(dict(rev_month=str(m), signal_day=str(sd.date()), deadline15=str(d15.date()) if d15 is not None else None,
                     hidden_by_deadline=bool(d15 is None or d15 > sd), n_codes=len(GJ)))
seen = pd.read_csv(L.AG / 'o_rev_event_clock_ts' / 'data' / 'firstseen_sources.csv.gz', dtype={'code': str}, usecols=['source', 'code', 'ym', 'T'])
seen = seen[(seen['source'] == 'tasi') & seen['code'].isin(list(GJ))]
seen['T'] = pd.to_datetime(seen['T'], utc=True).dt.tz_convert('Asia/Taipei')
ra = pd.read_csv(L.AG / 'rev_announce' / 'firstseen_2026-09.csv', dtype=str)
ra.columns = ['code'] + list(ra.columns[1:])
ra = ra.set_index('code')['first_seen']
for r in real:
    if r['rev_month'] > '2026-09':
        r['tasi'] = '無 Tasi 資料（面板外）'; continue
    send = pd.Timestamp(r['signal_day']).tz_localize('Asia/Taipei') + pd.Timedelta(hours=23, minutes=59, seconds=59)
    sm = seen[seen['ym'] == r['rev_month']].groupby('code')['T'].min()
    vis, late = [], []
    for c in GJ:
        t = sm.get(c)
        if r['rev_month'] == '2026-09' and c in ra.index and isinstance(ra[c], str):
            t2 = pd.Timestamp(ra[c]).tz_convert('Asia/Taipei') if pd.Timestamp(ra[c]).tzinfo else pd.Timestamp(ra[c]).tz_localize('UTC').tz_convert('Asia/Taipei')
            t = t2 if (t is None or pd.isna(t) or t2 < t) else t
        (vis if (t is not None and not pd.isna(t) and t <= send) else late).append(c)
    r['visible_by_signal_night'] = vis; r['not_visible'] = late; r['n_not_visible'] = len(late)
    m = pd.Period(r['rev_month'], 'M'); x = info.get(str(m))
    if x is not None:
        k = x['k']; s = S0.loc[m].to_numpy(); ok_ma = U[k] & np.isfinite(s) & P.F_MA60[k]
        o_ma = np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]
        hd = held_at(lgC, k)
        r['cur_top20'] = [str(P.COLS[j]) for j in o_ma[:20] if str(P.COLS[j]) in GJ]
        r['cur_f1'] = [str(P.COLS[j]) for j in np.flatnonzero(sgC['sig'][k]) if str(P.COLS[j]) in GJ]
        r['cur_held_at_signal'] = [str(P.COLS[j]) for j in hd if str(P.COLS[j]) in GJ]
res['real_2026'] = real
res['meta'] = dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
(H / 'results_describe.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print(json.dumps(real, ensure_ascii=False, default=str)[:4000])
print(res['meta'])
