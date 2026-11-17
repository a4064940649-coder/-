#!/usr/bin/env python3
"""S65 新起跑散戶的進場方式：規格 SPEC.json、判讀 PREREG.json（sha256 啟動時核對）。python3 -I -B run.py → results.json、starts.csv.gz"""
import sys, json, time, resource, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
t0c = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for f in ('SPEC', 'PREREG'):
    assert sha(HERE / f'{f}.json') == (HERE / f'{f}.sha256').read_text().split()[0], f'{f}.json 與登記的 sha256 不符'
sys.path.insert(0, str(HERE))
import s65lib as L                                         # noqa: E402
from s65lib import s10lib, stlib, COSTS                     # noqa: E402

P, U, RT = L.load()
T = P.T; IDX = pd.DatetimeIndex(P.IDX); E50 = np.asarray(P.E50, float)
CAP = L.CAP
SB = {s: L.base_signal(P, U, RT, s) for s in L.SHIFTS}
print('載入', f'{time.time()-t0c:.0f}s', rss(), 'MB', flush=True)

# ---------------------------------------------------------------- 驗收
acc = {}
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]; DE = P.IDX[-1]
a2 = True; a3 = True
for s in L.SHIFTS:
    nav, lg, st = s10lib.simulate_s10(P, SB[s], U, capital=CAP, sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'])
    if s == 0:
        acc['since2019'] = stlib.cagr(nav, D19, DE); acc['n_closed'] = int((lg['why'] != 'open_end').sum())
        acc['A1'] = bool(acc['since2019'] == L.ACC['since2019'] and acc['n_closed'] == 381)
    n2 = L.sim(P, SB[s], U)[0]
    a2 &= bool(np.array_equal(nav.to_numpy(), n2.to_numpy()))
    n3 = L.sim(P, SB[s], U, start_t=P.t0, inj={}, nobuy_t=None, end_t=None, odd_from='hist')[0]
    a3 &= bool(np.array_equal(nav.to_numpy(), n3.to_numpy()))
    n4 = L.sim(P, SB[s], U, end_t=P.t0 + 800)[0].to_numpy()
    a3 &= bool(np.array_equal(nav.to_numpy()[:801], n4[:801]) and np.isnan(n4[801:]).all())
acc['A2'] = a2; acc['A3'] = a3
print('驗收', acc, flush=True)
assert acc['A1'] and acc['A2'] and acc['A3'], '驗收失敗'

# ---------------------------------------------------------------- 工具
DD50 = E50 / pd.Series(E50).rolling(252, min_periods=1).max().to_numpy() - 1.0     # 0050 含息還原收盤對近 252 日（含當天）最高的回落
METHODS = ('A', 'B', 'C', 'D', 'E', 'E_D', 'L0')
NAME = {'A': 'A 一次到位（現行）', 'B': 'B 先全買 0050', 'C': 'C 分 3 個月', 'D': 'D 分 6 個月', 'E': 'E 等回落 10%（每月看）', 'E_D': 'E_D 等回落（每天看，敏感度）', 'L0': '一次買 0050（基準）'}


def hrow(k, months):
    d = IDX[k] + pd.DateOffset(months=months)
    if d > IDX[-1]:
        return None
    return int(np.searchsorted(IDX.values, np.datetime64(d), side='right') - 1)


def wealth(S, S_empty, info, i, m, odd, hend):
    """回傳 (W[k..hend] 陣列, extra dict)。"""
    k = info[i]['k']; ex = {}
    W = np.full(hend - k + 1, CAP)

    def put(nav, start, outside=None):
        v = nav.to_numpy()[: hend - start + 1]
        assert np.isfinite(v).all()
        W[start - k:] = v if outside is None else v + outside[: len(v)]

    if m == 'L0':
        put(L.sim(P, S_empty, U, start_t=k, end_t=hend, odd_from=odd)[0], k)
    elif m == 'A':
        nav, lg, st = L.sim(P, S, U, start_t=k, end_t=hend, odd_from=odd)
        put(nav, k)
        ex['n_buy_day1'] = int(sum(1 for f in st['fills'] if f[0] == k + 1 and f[2] == 'B')); ex['n_tgt'] = int(info[i]['n_tgt'])
        ex['lg'] = lg
    elif m == 'B':
        put(L.sim(P, S, U, start_t=k, end_t=hend, odd_from=odd, nobuy_t=k + 1)[0], k)
    elif m in ('C', 'D'):
        n = 3 if m == 'C' else 6
        part = CAP / n
        inj = {info[i + j]['k'] + 1: part for j in range(1, n)}
        nav = L.sim(P, S, U, capital=part, start_t=k, end_t=hend, odd_from=odd, inj=inj)[0]
        out = np.full(hend - k + 1, CAP - part)
        for t in inj:
            if t <= hend:
                out[t - k:] -= part
        ex['injected_total'] = float(part + sum(v for t, v in inj.items() if t <= hend))
        put(nav, k, out)
    elif m in ('E', 'E_D'):
        if m == 'E':
            rows = [info[i + j]['k'] for j in range(0, 13)]
        else:
            rows = list(range(k, info[i + 12]['k'] + 1))
        st_row = rows[-1]; trig = False
        for r in rows[:-1]:
            if DD50[r] <= -0.10:
                st_row = r; trig = True; break
        if not trig and DD50[rows[-1]] <= -0.10:
            trig = True
        ex['start_row'] = int(st_row); ex['wait_days'] = int(st_row - k); ex['triggered'] = bool(trig)
        ex['wait_months'] = round((IDX[st_row] - IDX[k]).days / 30.4375, 2)
        if st_row < hend:
            put(L.sim(P, S, U, start_t=st_row, end_t=hend, odd_from=odd)[0], st_row)
    return W, ex


def mdd(w):
    return float((w / np.maximum.accumulate(w) - 1).min())


def run_set(s, odd, methods=METHODS, keep=False):
    S = SB[s]; info = S['info']
    S_empty = dict(S); S_empty['sig'] = np.zeros_like(S['sig'])
    rows = []; keepd = {}
    for i, x in enumerate(info):
        k = x['k']
        if k + 1 >= T:
            continue
        d1 = IDX[k + 1]
        if not (pd.Timestamp('2016-02-01') <= d1 <= pd.Timestamp('2025-10-31')):
            continue
        h12 = hrow(k, 12); h36 = hrow(k, 36)
        assert h12 is not None
        hend = h36 if h36 is not None else h12
        yrs36 = (IDX[h36] - IDX[k]).days / 365.25 if h36 is not None else None
        for m in methods:
            W, ex = wealth(S, S_empty, info, i, m, odd, hend)
            r = dict(shift=s, odd=odd, method=m, start_month=str(d1)[:7], sig_date=str(IDX[k].date()), exec_date=str(d1.date()), rev_month=x['month'],
                     r12=float(W[h12 - k] / CAP - 1), mdd12=mdd(W[: h12 - k + 1]), mdd12_trough=str(IDX[k + int(np.argmin(W[: h12 - k + 1] / np.maximum.accumulate(W[: h12 - k + 1])))].date()),
                     r36=float(W[h36 - k] / CAP - 1) if h36 is not None else np.nan,
                     ann36=float((W[h36 - k] / CAP) ** (1 / yrs36) - 1) if h36 is not None else np.nan,
                     mdd36=mdd(W[: h36 - k + 1]) if h36 is not None else np.nan)
            for kk in ('n_buy_day1', 'n_tgt', 'wait_months', 'triggered', 'injected_total'):
                if kk in ex:
                    r[kk] = ex[kk]
            if m == 'A' and 'lg' in ex:
                lg = ex['lg']; c = lg[(lg['why'] != 'open_end') & (lg['exit_t'] <= h12)] if len(lg) else lg
                r['n_closed_12m'] = int(len(c)); r['win_12m'] = float((c['ret'] > 0).mean()) if len(c) else np.nan
            if m in ('C', 'D'):
                assert abs(ex['injected_total'] - CAP) < 1e-6, 'A4 對帳失敗'
            rows.append(r)
    return pd.DataFrame(rows)


def dist(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) == 0:
        return None
    return dict(n=int(len(v)), median=float(np.median(v)), p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)), worst=float(v.min()), best=float(v.max()), mean=float(v.mean()))


def summarize(df):
    out = {}
    key = ['shift', 'start_month']
    base = df[df.method == 'L0'].set_index(key)
    A = df[df.method == 'A'].set_index(key)
    for m in [x for x in METHODS if x in set(df.method)]:
        d = df[df.method == m].set_index(key)
        ex12 = d['r12'] - base['r12'].reindex(d.index); ex36 = d['ann36'] - base['ann36'].reindex(d.index); ex36c = d['r36'] - base['r36'].reindex(d.index)
        o = dict(r12=dist(d['r12']), ex12=dist(ex12), mdd12=dist(d['mdd12']), r36=dist(d['r36']), ann36=dist(d['ann36']), ex36_ann=dist(ex36), ex36_cum=dist(ex36c), mdd36=dist(d['mdd36']),
                 lose12=float((ex12 < 0).mean()), lose36=float((ex36.dropna() < 0).mean()), loss_first_year=float((d['r12'] < 0).mean()),
                 loss_36m=float((d['r36'].dropna() < 0).mean()), mdd12_worse_than_20=float((d['mdd12'] < -0.20).mean()))
        # 12 個月指標限縮到有 36 個月視窗的起始月（同樣本對照）
        has36 = d['ann36'].notna()
        o['r12_on36sample'] = dist(d.loc[has36, 'r12']); o['ex12_on36sample'] = dist(ex12[has36])
        if m not in ('A', 'L0'):
            p12 = d['r12'] - A['r12'].reindex(d.index); p36 = (d['ann36'] - A['ann36'].reindex(d.index)).dropna()
            pm = d['mdd12'] - A['mdd12'].reindex(d.index)
            o['vs_A'] = dict(d_r12=dist(p12), beat_A_12=float((p12 > 0).mean()), d_ann36=dist(p36), beat_A_36=float((p36 > 0).mean()), d_mdd12=dist(pm), shallower_mdd12=float((pm > 0).mean()))
        if m in ('E', 'E_D'):
            wm = d['wait_months']
            o['wait'] = dict(immediate=float((wm == 0).mean()), waited_then_triggered=float(((wm > 0) & d['triggered']).mean()), timeout_12m=float((~d['triggered'].astype(bool)).mean()), wait_months_median=float(wm.median()), wait_months_mean=float(wm.mean()))
        out[m] = o
    return out


def verdict(sm, sm_pool):
    out = {}
    for m in ('B', 'C', 'D', 'E', 'E_D'):
        def cls(S_):
            dm = (S_[m]['ex36_ann']['median'] - S_['A']['ex36_ann']['median']) * 100; dp = (S_[m]['ex36_ann']['p10'] - S_['A']['ex36_ann']['p10']) * 100
            if abs(dm) <= 0.2 and abs(dp) <= 0.2:
                c = '實質相同'
            elif dm >= 1.0 and dp >= -0.5:
                c = '較好'
            elif dp >= 1.0 and dm >= -1.0:
                c = '較穩（保守者可選）'
            elif dm <= -1.0 and dp < 1.0:
                c = '較差'
            else:
                c = '沒有明顯差別'
            return c, dm, dp
        c0, dm0, dp0 = cls(sm); c1, dm1, dp1 = cls(sm_pool)
        final = c0
        if c0 in ('較好', '較穩（保守者可選）') and c1 != c0:
            final = '不穩、沒有明顯差別'
        out[m] = dict(shift0=c0, d_median_pp=dm0, d_p10_pp=dp0, pooled=c1, pooled_d_median_pp=dm1, pooled_d_p10_pp=dp1, final=final)
    return out


# ---------------------------------------------------------------- 跑
frames = []
for s in L.SHIFTS:
    frames.append(run_set(s, 'all')); print('shift', s, 'all', f'{time.time()-t0c:.0f}s', rss(), 'MB', flush=True)
hist = run_set(0, 'hist', methods=('A', 'B', 'C', 'D', 'E', 'L0')); print('hist', f'{time.time()-t0c:.0f}s', flush=True)
df = pd.concat(frames, ignore_index=True)
main = df[df['shift'] == 0]
SM = summarize(main); SMP = summarize(df); SMH = summarize(hist)
V = verdict(SM, SMP)

# 描述：A 第一年最慘的 10 個起始月
piv = main.pivot(index='start_month', columns='method', values='r12')
pmdd = main.pivot(index='start_month', columns='method', values='mdd12')
Arow = main[main.method == 'A'].set_index('start_month')
worst = []
for sm_ in piv['A'].sort_values().index[:10]:
    a = Arow.loc[sm_]
    worst.append(dict(start_month=sm_, exec_date=a['exec_date'], r12={m: float(piv.loc[sm_, m]) for m in METHODS}, ex12_A=float(piv.loc[sm_, 'A'] - piv.loc[sm_, 'L0']),
                      mdd12={m: float(pmdd.loc[sm_, m]) for m in METHODS}, mdd12_trough_A=a['mdd12_trough'], n_tgt=int(a['n_tgt']), n_buy_day1=int(a['n_buy_day1']),
                      n_closed_12m=int(a['n_closed_12m']), win_12m=None if not np.isfinite(a['win_12m']) else float(a['win_12m']), dd50_at_start=float(DD50[int(np.searchsorted(IDX.values, np.datetime64(a['sig_date'])))])))
worst_ex = []
exA = (piv['A'] - piv['L0'])
for sm_ in exA.sort_values().index[:10]:
    worst_ex.append(dict(start_month=sm_, r12_A=float(piv.loc[sm_, 'A']), r12_0050=float(piv.loc[sm_, 'L0']), ex12_A=float(exA.loc[sm_]), mdd12_A=float(pmdd.loc[sm_, 'A'])))
# 各起始年
by_year = {}
for y, g in main.assign(y=main.start_month.str[:4]).groupby('y'):
    pv = g.pivot(index='start_month', columns='method', values='r12'); pa = g.pivot(index='start_month', columns='method', values='ann36')
    by_year[y] = dict(n=int(len(pv)), r12_median={m: float(pv[m].median()) for m in METHODS}, ex12_median={m: float((pv[m] - pv['L0']).median()) for m in METHODS if m != 'L0'},
                      lose12_A=float(((pv['A'] - pv['L0']) < 0).mean()), ann36_median={m: (None if pa[m].notna().sum() == 0 else float(pa[m].median())) for m in METHODS},
                      ex36_median={m: (None if pa[m].notna().sum() == 0 else float((pa[m] - pa['L0']).median())) for m in METHODS if m != 'L0'}, n36=int(pa['A'].notna().sum()))
# A 與 B 差在哪
dAB = (piv['B'] - piv['A'])
ab = dict(r12_B_minus_A=dist(dAB), first_month_note='B 第一個換股日不買個股；之後每月照名單，持股約在 A 的部位陸續出場後趨於一致',
          n_buy_day1_A=dist(Arow['n_buy_day1']), n_tgt=dist(Arow['n_tgt']), frac_filled=float(Arow['n_buy_day1'].sum() / max(Arow['n_tgt'].sum(), 1)),
          same_day_roundtrip_cost_est_pct_of_capital=float(0.7 * Arow['n_buy_day1'].mean() / 10 * (2 * stlib.FEE + stlib.TAX_ETF + 2 * 0.0003) * 100))
res = dict(acceptance=acc, src_sha=L.SRC_SHA, n_starts_shift0=int(main.start_month.nunique()), n_starts_36m=int(main[main.method == 'A']['ann36'].notna().sum()),
           first_start=main.start_month.min(), last_start=main.start_month.max(), last_start_36m=main[main.ann36.notna()].start_month.max(),
           names=NAME, main_shift0_oddall=SM, pooled_shift0to3_oddall=SMP, sens_shift0_oddhist=SMH, verdict=V,
           worst10_first_year_A=worst, worst10_first_year_excess_A=worst_ex, by_start_year=by_year, A_vs_B=ab,
           secs=round(time.time() - t0c), peak_mb=rss())
json.dump(res, open(HERE / 'results.json', 'w'), ensure_ascii=False, indent=1, default=str)
pd.concat([df, hist], ignore_index=True).round(6).to_csv(HERE / 'starts.csv.gz', index=False)
pc = lambda x: f'{x*100:7.2f}'
for tag, S_ in (('主（位移0、零股全期）', SM), ('位移0～3合併', SMP), ('零股歷史制度', SMH)):
    print('==', tag)
    for m, o in S_.items():
        print(f"{m:4} r12 中{pc(o['r12']['median'])} P10{pc(o['r12']['p10'])} P90{pc(o['r12']['p90'])} 差{pc(o['r12']['worst'])} | ex12 中{pc(o['ex12']['median'])} P10{pc(o['ex12']['p10'])} P90{pc(o['ex12']['p90'])} 差{pc(o['ex12']['worst'])} 輸{pc(o['lose12'])} 虧{pc(o['loss_first_year'])} | mdd12 中{pc(o['mdd12']['median'])} P10{pc(o['mdd12']['p10'])} 差{pc(o['mdd12']['worst'])}")
        print(f"     ann36 中{pc(o['ann36']['median'])} P10{pc(o['ann36']['p10'])} P90{pc(o['ann36']['p90'])} 差{pc(o['ann36']['worst'])} | ex36 中{pc(o['ex36_ann']['median'])} P10{pc(o['ex36_ann']['p10'])} P90{pc(o['ex36_ann']['p90'])} 差{pc(o['ex36_ann']['worst'])} 輸{pc(o['lose36'])} | mdd36 中{pc(o['mdd36']['median'])} P10{pc(o['mdd36']['p10'])} 差{pc(o['mdd36']['worst'])} n36={o['ann36']['n']}")
print(json.dumps(V, ensure_ascii=False, indent=1))
print('done', res['secs'], 's', res['peak_mb'], 'MB')
