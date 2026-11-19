#!/usr/bin/env python3
"""S73 起跑後 6 日／首月大跌的歷史分布與其後條件分布。規格 SPEC.json、判讀 PREREG.json（啟動時核對 sha256）。
python3 -I -B run.py → results.json、starts_a.csv.gz、dense_b.csv.gz"""
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
import s73lib as L3                                        # noqa: E402
L = L3.B
from s65lib import s10lib, stlib, COSTS                     # noqa: E402

P, U, RT = L.load()
T = P.T; IDX = pd.DatetimeIndex(P.IDX); CAP = L.CAP
SB = {s: L.base_signal(P, U, RT, s) for s in L.SHIFTS}
print('載入', f'{time.time()-t0c:.0f}s', rss(), 'MB', flush=True)

# ---------------------------------------------------------------- 驗收
acc = {}
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]; DE = P.IDX[-1]
a2 = True
for s in L.SHIFTS:
    nav, lg, st = s10lib.simulate_s10(P, SB[s], U, capital=CAP, sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'])
    if s == 0:
        acc['since2019'] = stlib.cagr(nav, D19, DE); acc['n_closed'] = int((lg['why'] != 'open_end').sum())
        acc['A1'] = bool(acc['since2019'] == L.ACC['since2019'] and acc['n_closed'] == 381)
        nav0 = nav
    a2 &= bool(np.array_equal(nav.to_numpy(), L.sim(P, SB[s], U)[0].to_numpy()))
acc['A2'] = a2
OFF = pd.read_csv(L3.NAVS, parse_dates=['date']).set_index('date')
nv = nav0.dropna(); nv.index = pd.DatetimeIndex(nv.index)
cm = OFF.index.intersection(nv.index)
rel = np.abs(OFF.loc[cm, 'E0|W10_70'].to_numpy() / nv.reindex(cm).to_numpy() - 1)
acc['A3_n_common'] = int(len(cm)); acc['A3_n_official'] = int(len(OFF)); acc['A3_max_rel_diff'] = float(rel.max())
acc['A3'] = bool(len(cm) == len(OFF) and rel.max() <= 1e-8)
print('驗收', acc, flush=True)
assert acc['A1'] and acc['A2'] and acc['A3'], '驗收失敗'


# ---------------------------------------------------------------- 共用指標
def hpos(dates, j, months):
    """dates：DatetimeIndex；回傳日期 ≤ dates[j]＋months 個日曆月的最後一列位置；超出資料 → None"""
    d = dates[j] + pd.DateOffset(months=months)
    if d > dates[-1]:
        return None
    return int(np.searchsorted(dates.values, np.datetime64(d), side='right') - 1)


def after(W, W0, dates, j, base, tag):
    """W、W0：從起點（位置 0）到資料尾的策略／0050 淨值；j：觸發點位置；base：起始資金（W[0]）。"""
    o = {}
    h12 = hpos(dates, j, 12)
    for m in (3, 6, 12):
        h = hpos(dates, j, m)
        if h is None:
            o[f'{tag}_s{m}'] = o[f'{tag}_b{m}'] = o[f'{tag}_x{m}'] = np.nan
        else:
            o[f'{tag}_s{m}'] = W[h] / W[j] - 1; o[f'{tag}_b{m}'] = W0[h] / W0[j] - 1; o[f'{tag}_x{m}'] = o[f'{tag}_s{m}'] - o[f'{tag}_b{m}']
    if h12 is None:
        o[f'{tag}_fall'] = o[f'{tag}_fall_base'] = o[f'{tag}_fall0'] = o[f'{tag}_r12_from_start'] = o[f'{tag}_stop12_from_start'] = np.nan
    else:
        seg = W[j:h12 + 1]
        o[f'{tag}_fall'] = seg.min() / W[j] - 1; o[f'{tag}_fall_base'] = seg.min() / base - 1; o[f'{tag}_fall0'] = W0[j:h12 + 1].min() / W0[j] - 1
        o[f'{tag}_r12_from_start'] = W[h12] / base - 1; o[f'{tag}_stop12_from_start'] = W[j] / base * W0[h12] / W0[j] - 1
    ok = np.flatnonzero(W[j:] >= base)
    if len(ok):
        o[f'{tag}_rec_td'] = int(ok[0]); o[f'{tag}_rec_mo'] = (dates[j + ok[0]] - dates[j]).days / 30.4375; o[f'{tag}_rec_date'] = str(dates[j + ok[0]].date())
    else:
        o[f'{tag}_rec_td'] = np.nan; o[f'{tag}_rec_mo'] = np.inf; o[f'{tag}_rec_date'] = None
    return o


def dist(v, pcts=(10, 90)):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(n=0)
    d = dict(n=int(len(v)), median=float(np.median(v)), mean=float(v.mean()), worst=float(v.min()), best=float(v.max()), pos=float((v > 0).mean()))
    for p in pcts:
        d[f'p{p}'] = float(np.percentile(v, p))
    return d


def rec_stats(mo):
    mo = np.asarray(mo, float); mo = mo[~np.isnan(mo)]
    fin = mo[np.isfinite(mo)]
    return dict(n=int(len(mo)), not_recovered=int(np.isinf(mo).sum()), already_above=float((mo == 0).mean()) if len(mo) else None,
                median_months=(None if len(mo) == 0 else (float(np.median(mo)) if np.isfinite(np.median(mo)) else 'inf')),
                p90_months=(None if len(mo) == 0 else (float(np.percentile(mo, 90)) if np.isfinite(np.percentile(mo, 90)) else 'inf')),
                longest_recovered_months=float(fin.max()) if len(fin) else None)


def r_dist(df, col_s, col_b):
    s = df[col_s].to_numpy(float); b = df[col_b].to_numpy(float); d = s - b
    o = dict(strategy=dist(s, (5, 1, 10)), b0050=dist(b, (5, 1, 10)), diff=dist(d, (5, 1, 10)),
             neg_frac_strategy=float((s < 0).mean()), neg_frac_0050=float((b < 0).mean()))
    return o


def where_current(df, col_s='r6', col_b='r6_b'):
    s = df[col_s].to_numpy(float); b = df[col_b].to_numpy(float); n = len(s)
    return dict(n=n, strat_le_cur=dict(n=int((s <= L3.CUR_S).sum()), frac=float((s <= L3.CUR_S).mean())),
                b0050_le_cur=dict(n=int((b <= L3.CUR_B).sum()), frac=float((b <= L3.CUR_B).mean())),
                strat_le_13=dict(n=int((s <= -0.13).sum()), frac=float((s <= -0.13).mean())),
                strat_le_10=dict(n=int((s <= -0.10).sum()), frac=float((s <= -0.10).mean())),
                b0050_le_13=dict(n=int((b <= -0.13).sum()), frac=float((b <= -0.13).mean())),
                b0050_le_10=dict(n=int((b <= -0.10).sum()), frac=float((b <= -0.10).mean())),
                strat_worst=float(s.min()), b0050_worst=float(b.min()))


def cond(df, mask, tag, cost=0.005):
    g = df[mask]; o = dict(n=int(len(g)))
    if len(g) == 0:
        return o
    o['trigger_ret'] = dist(g['r6' if tag == 'a6' else 'rM']); o['trigger_ret_0050'] = dist(g['r6_b' if tag == 'a6' else 'rM_b'])
    for m in (3, 6, 12):
        o[f'strat_{m}m'] = dist(g[f'{tag}_s{m}']); o[f'b0050_{m}m'] = dist(g[f'{tag}_b{m}']); o[f'excess_{m}m'] = dist(g[f'{tag}_x{m}'])
        # 停手對照：繼續 − 停手（以觸發點資金為分母）；c＝0 等於超額
        o[f'stop_c0_{m}m'] = dict(o[f'excess_{m}m'], continue_win=o[f'excess_{m}m'].get('pos'))
        dc = (g[f'{tag}_s{m}'] - ((1 - cost) * (1 + g[f'{tag}_b{m}']) - 1))
        o[f'stop_c50bp_{m}m'] = dict(dist(dc), continue_win=float((dc.dropna() > 0).mean()) if dc.notna().any() else None)
    o['further_fall_12m'] = dist(g[f'{tag}_fall']); o['further_fall_12m_0050'] = dist(g[f'{tag}_fall0']); o['deepest_vs_start_12m'] = dist(g[f'{tag}_fall_base'])
    o['recover'] = rec_stats(g[f'{tag}_rec_mo'])
    v = g[f'{tag}_r12_from_start'].dropna(); w = g[f'{tag}_stop12_from_start'].dropna()
    o['from_start_12m_after_trigger'] = dict(continue_=dist(v), stop=dist(w), continue_loss_frac=float((v < 0).mean()) if len(v) else None, stop_loss_frac=float((w < 0).mean()) if len(w) else None)
    if 'exec_date' in g:
        o['distinct_year_months'] = sorted(set(str(x)[:7] for x in g['exec_date'])); o['distinct_years'] = sorted(set(str(x)[:4] for x in g['exec_date']))
    return o


def verdict_stop(c):
    if c.get('n', 0) == 0 or c['stop_c0_12m'].get('n', 0) == 0:
        return dict(n=0, word='無資料')
    d = c['stop_c0_12m']['median'] * 100; w = c['stop_c0_12m']['continue_win']; n = c['stop_c0_12m']['n']
    if n < 10:
        word = '個數不足 10，只描述'
    elif d >= 1 and w >= 0.6:
        word = '歷史上停手是幫倒忙'
    elif d <= -1 and w <= 0.4:
        word = '歷史上停手有幫上忙'
    else:
        word = '歷史上停不停手沒有明顯差別'
    return dict(n=n, d_median_pp=d, continue_win=w, word=word)


def word_common(p):
    return ('這十年沒有發生過這麼差的開局' if p == 0 else '非常少見（百次不到一次）' if p < 0.01 else '少見' if p < 0.05 else '不常見但會遇到' if p < 0.10 else '常見')


# ---------------------------------------------------------------- 樣本 A：每個換股日新起跑
rows = []
for s in L.SHIFTS:
    S = SB[s]; info = S['info']
    S_empty = dict(S); S_empty['sig'] = np.zeros_like(S['sig'])
    for i, x in enumerate(info):
        k = x['k']
        if k + 1 >= T or i + 1 >= len(info):
            continue
        d1 = IDX[k + 1]
        if not (pd.Timestamp('2016-02-01') <= d1 <= pd.Timestamp('2025-10-31')):
            continue
        nav, lg, st = L.sim(P, S, U, start_t=k, odd_from='all')
        nb = L.sim(P, S_empty, U, start_t=k, odd_from='all')[0]
        assert pd.Timestamp(nav.index[0]) == IDX[k] and nav.iloc[0] == CAP and nb.iloc[0] == CAP, 'A4 失敗'
        W = nav.to_numpy(); W0 = nb.to_numpy(); assert len(W) == T - k and np.isfinite(W).all() and np.isfinite(W0).all()
        dates = IDX[k:]; km = info[i + 1]['k'] - k
        r = dict(shift=s, sig_date=str(IDX[k].date()), exec_date=str(d1.date()), rev_month=x['month'], n_tgt=int(x['n_tgt']),
                 n_buy_day1=int(sum(1 for f in st['fills'] if f[0] == k + 1 and f[2] == 'B')),
                 day6_date=str(IDX[k + 6].date()), r6=W[6] / CAP - 1, r6_b=W0[6] / CAP - 1, month_days=int(km), month_end=str(IDX[k + km].date()),
                 rM=W[km] / CAP - 1, rM_b=W0[km] / CAP - 1)
        h12 = hpos(dates, 0, 12); r['r12'] = W[h12] / CAP - 1 if h12 is not None else np.nan; r['r12_b'] = W0[h12] / CAP - 1 if h12 is not None else np.nan
        r.update(after(W, W0, dates, 6, CAP, 'a6')); r.update(after(W, W0, dates, km, CAP, 'aM'))
        rows.append(r)
    print('shift', s, f'{time.time()-t0c:.0f}s', rss(), 'MB', flush=True)
A = pd.DataFrame(rows); A0 = A[A['shift'] == 0].reset_index(drop=True)

# ---------------------------------------------------------------- 樣本 B：正式連續帳戶每個交易日
od = pd.DatetimeIndex(OFF.index); s_ = OFF['E0|W10_70'].to_numpy(float); b_ = OFF['0050'].to_numpy(float)
rows = []
for t in range(len(od) - 6):
    W = s_[t:]; W0 = b_[t:]; dates = od[t:]
    r = dict(date=str(od[t].date()), exec_date=str(od[t].date()), day6_date=str(od[t + 6].date()), r6=W[6] / W[0] - 1, r6_b=W0[6] / W0[0] - 1, rM=np.nan, rM_b=np.nan)
    r.update(after(W, W0, dates, 6, W[0], 'a6'))
    rows.append(r)
BF = pd.DataFrame(rows)
BM = BF[(BF.date >= '2016-02-01') & (BF.date <= '2025-10-31')].reset_index(drop=True)
print('樣本', len(A0), len(A), len(BM), len(BF), flush=True)


def episodes(df, thr=-0.10, gap=6):
    idx = np.flatnonzero(df['r6'].to_numpy() <= thr); out = []
    if len(idx) == 0:
        return out
    grp = [[idx[0]]]
    for a in idx[1:]:
        (grp[-1].append(a) if a - grp[-1][-1] <= gap else grp.append([a]))
    for g in grp:
        sub = df.iloc[g]; w = sub.loc[sub['r6'].idxmin()]
        out.append(dict(first=sub['date'].iloc[0], last=sub['date'].iloc[-1], n_days=int(len(g)), worst_date=w['date'], worst_day6=w['day6_date'], worst_r6=float(w['r6']), r6_0050_same_day=float(w['r6_b']),
                        after12_strat_median=None if sub['a6_s12'].notna().sum() == 0 else float(sub['a6_s12'].median()), after12_excess_median=None if sub['a6_x12'].notna().sum() == 0 else float(sub['a6_x12'].median())))
    return out


def block(df, name, with_month):
    o = dict(n=int(len(df)), r6=r_dist(df, 'r6', 'r6_b'), where_current=where_current(df))
    thr6 = float(np.percentile(df['r6'], 10)); o['p10_threshold_r6'] = thr6
    o['cond_all_a6'] = cond(df, np.ones(len(df), bool), 'a6')
    o['cond_worst10_r6'] = cond(df, (df['r6'] <= thr6).to_numpy(), 'a6')
    o['cond_le_minus10_r6'] = cond(df, (df['r6'] <= -0.10).to_numpy(), 'a6')
    o['verdict_stop_worst10_r6'] = verdict_stop(o['cond_worst10_r6']); o['verdict_stop_le_minus10_r6'] = verdict_stop(o['cond_le_minus10_r6'])
    if with_month:
        o['rM'] = r_dist(df, 'rM', 'rM_b'); o['month_days'] = dist(df['month_days'])
        o['where_current_month'] = where_current(df, 'rM', 'rM_b')
        thrM = float(np.percentile(df['rM'], 10)); o['p10_threshold_rM'] = thrM
        o['cond_all_aM'] = cond(df, np.ones(len(df), bool), 'aM'); o['cond_worst10_rM'] = cond(df, (df['rM'] <= thrM).to_numpy(), 'aM')
        o['verdict_stop_worst10_rM'] = verdict_stop(o['cond_worst10_rM'])
        o['n_buy_day1'] = dist(df['n_buy_day1'])
    return o


res = dict(acceptance=acc, src_sha=L.SRC_SHA, current=dict(strategy=L3.CUR_S, b0050=L3.CUR_B, note='網站模擬帳戶 2026-11-12 開盤～11-19 收盤；不在面板內'),
           A_shift0=block(A0, 'A0', True), A_pooled=block(A, 'A', True), B_main=block(BM, 'BM', False), B_full=block(BF, 'BF', False))
cols = ['shift', 'sig_date', 'exec_date', 'day6_date', 'n_tgt', 'n_buy_day1', 'r6', 'r6_b', 'rM', 'rM_b', 'r12', 'r12_b', 'a6_s3', 'a6_s6', 'a6_s12', 'a6_x12', 'a6_fall', 'a6_fall_base', 'a6_rec_mo', 'a6_rec_date']
res['A_list_le_minus10'] = json.loads(A[A.r6 <= -0.10].sort_values(['shift', 'sig_date'])[cols].to_json(orient='records'))
res['A_shift0_worst10_list'] = json.loads(A0[A0.r6 <= np.percentile(A0.r6, 10)].sort_values('r6')[cols].to_json(orient='records'))
res['A_shift0_worst10_month_list'] = json.loads(A0[A0.rM <= np.percentile(A0.rM, 10)].sort_values('rM')[cols + ['aM_s12', 'aM_x12', 'aM_fall', 'aM_rec_mo']].to_json(orient='records'))
res['B_main_episodes_le_minus10'] = episodes(BM); res['B_full_episodes_le_minus10'] = episodes(BF)
res['B_main_worst10_episodes'] = episodes(BM, thr=float(np.percentile(BM.r6, 10)))
res['B_full_days_le_minus10'] = json.loads(BF[BF.r6 <= -0.10][['date', 'day6_date', 'r6', 'r6_b', 'a6_s3', 'a6_s6', 'a6_s12', 'a6_x12', 'a6_fall', 'a6_rec_mo']].to_json(orient='records'))
# 白話用詞（依 PREREG）
pA0 = res['A_shift0']['where_current']['strat_le_cur']['frac']; pA = res['A_pooled']['where_current']['strat_le_cur']['frac']; pB = res['B_main']['where_current']['strat_le_cur']['frac']; pBF = res['B_full']['where_current']['strat_le_cur']['frac']
c_w = res['A_shift0']['cond_worst10_r6']; c_a = res['A_shift0']['cond_all_a6']
dq2 = (c_w['strat_12m']['median'] - c_a['strat_12m']['median']) * 100
vs = dict(A_shift0_worst10=res['A_shift0']['verdict_stop_worst10_r6'], A_pooled_worst10=res['A_pooled']['verdict_stop_worst10_r6'], B_main_worst10=res['B_main']['verdict_stop_worst10_r6'],
          A_shift0_le10=res['A_shift0']['verdict_stop_le_minus10_r6'], A_pooled_le10=res['A_pooled']['verdict_stop_le_minus10_r6'], B_main_le10=res['B_main']['verdict_stop_le_minus10_r6'], B_full_le10=res['B_full']['verdict_stop_le_minus10_r6'],
          A_shift0_worst10_month=res['A_shift0']['verdict_stop_worst10_rM'])
words = [v['word'] for v in vs.values()]
res['prereg_words'] = dict(q1=dict(A_shift0=dict(p=pA0, word=word_common(pA0)), A_pooled=dict(p=pA, word=word_common(pA)), B_main=dict(p=pB, word=word_common(pB)), B_full=dict(p=pBF, word=word_common(pBF))),
                           q2=dict(diff_median_12m_pp=dq2, word='之後通常比平常好' if dq2 > 5 else '之後通常比平常差' if dq2 < -5 else '之後和平常差不多'),
                           q3=dict(primary=vs['A_shift0_worst10'], all=vs, unstable=bool(('歷史上停手是幫倒忙' in words) and ('歷史上停手有幫上忙' in words))))
res['secs'] = round(time.time() - t0c); res['peak_mb'] = rss()
json.dump(res, open(HERE / 'results.json', 'w'), ensure_ascii=False, indent=1, default=str)
A.round(6).to_csv(HERE / 'starts_a.csv.gz', index=False); BF.round(6).to_csv(HERE / 'dense_b.csv.gz', index=False)
print(json.dumps(res['prereg_words'], ensure_ascii=False, indent=1))
print('done', res['secs'], 's', res['peak_mb'], 'MB')
