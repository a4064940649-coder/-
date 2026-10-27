#!/usr/bin/env python3
"""步驟 5：名次歸因（事件層級）、全池訊號效力、事件數、前 100 名對自身等權的配對差。"""
import sys, json, pickle
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_capacity_tsmc_gap/replicate_o1')
import rcore as R
from r2_tsmc import crosses
from rstat import PER, S

NAV = pd.read_parquet(R.HERE / 'navs_tsmc.parquet')
NP = pd.read_parquet(R.HERE / 'navs_pools.parquet')
PK = pickle.load(open(R.HERE / 'picks_main.pkl', 'rb'))
buy, sell = R.COST['normal']
RT = (1 - sell) / (1 + buy) - 1  # 來回成本（乘法）
Sl = [d for d in S if d < R.EVAL_END]
nxt = {Sl[k]: (Sl[k + 1] if k + 1 < len(Sl) else R.EVAL_END) for k in range(len(Sl))}


def stock_win(d, mode):
    """每檔股票在窗內的報酬：'oo'＝成交日開盤→下次成交日開盤；'cc'＝訊號日收盤→下一訊號日收盤；'oc'＝成交日開盤→下一訊號日收盤。"""
    i = R.POS[d]; e = nxt[d]; ie = R.IDX.get_loc(e)
    if mode == 'cc':
        return R.AC[ie] / R.AC[i] - 1
    a = R.AO[i + 1]
    if mode == 'oc':
        return R.AC[ie] / a - 1
    if ie + 1 < len(R.IDX) and e < R.EVAL_END:
        b = np.where(np.isfinite(R.AO[ie + 1]), R.AO[ie + 1], R.AC[ie + 1])
    else:
        b = R.AC[ie]
    return b / a - 1


def ew_win(nav, d):
    e = nxt[d]
    return float(nav.loc[e] / nav.loc[d] - 1)


def period_mask(idx, p):
    ma, mb = PER[p][2], PER[p][3]
    return (idx >= pd.Timestamp(ma)) & (idx <= pd.Timestamp(mb))


def summarize_month(series):
    s = pd.Series(series).dropna()
    o = {}
    for p in ('SEL', 'VAL', 'ALL', 'HOLD'):
        x = s[period_mask(s.index, p)]
        o[p] = {'pp': round(x.mean() * 100, 3), 't': round(R.tstat(x), 3), 'n': int(len(x))}
    return o


out = {}
ewnav = NAV['EW']
for n in (10, 30):
    picks = PK[(n, 0.0)]
    out[f'events_n{n}'] = {'ALL': sum(len(v) for v in picks.values()),
                           'SEL': sum(len(v) for d, v in picks.items() if d <= pd.Timestamp('2019-03-31')),
                           'VAL': sum(len(v) for d, v in picks.items() if d >= pd.Timestamp('2019-04-01'))}
    for mode in ('cc', 'oo', 'oc'):
        for costm in ('net', 'gross'):
            dif_pool = {}; dif_grp = {}; g1_50 = {}; g1_50_ew = {}
            for d in Sl:
                cr = crosses[d]; pk = picks.get(d)
                if pk is None: continue
                r = stock_win(d, mode)
                if costm == 'net': r = (1 + r) * (1 + RT) - 1
                ew = ew_win(ewnav, d) if mode == 'cc' else np.nanmean(stock_win(d, mode)[cr.el])
                rk = cr.vrank
                A = [j for j in pk if rk[j] > 100]; Bq = [j for j in pk if rk[j] <= 100]
                grpA = cr.el & (rk > 100); grpB = cr.el & (rk <= 100)
                rr = stock_win(d, mode)
                if A and Bq:
                    dif_pool[d] = np.mean(r[A]) - np.mean(r[Bq])
                    dif_grp[d] = (np.mean(r[A]) - np.nanmean(rr[grpA])) - (np.mean(r[Bq]) - np.nanmean(rr[grpB]))
                C1 = [j for j in pk if rk[j] <= 50]; g = cr.el & (rk <= 50)
                if C1:
                    g1_50[d] = np.mean(r[C1]) - ew
                    g1_50_ew[d] = np.nanmean(rr[g]) - ew
            out[f'attr_n{n}_{mode}_{costm}'] = {'vs_pool': summarize_month(dif_pool), 'vs_group': summarize_month(dif_grp),
                                                 'g1_50_picks_vs_pool': summarize_month(g1_50), 'g1_50_EW_vs_pool': summarize_month(g1_50_ew)}

# 全池訊號效力：各名次組內分數前 1/5 減組內等權（第 1 個月）
groups = [(1, 50), (51, 100), (101, 200), (201, 300)]
for mode in ('cc', 'oo', 'oc'):
    res = {}
    for lo, hi in groups:
        ser = {}
        for d in Sl:
            cr = crosses[d]; r = stock_win(d, mode)
            g = np.where(cr.el & (cr.vrank >= lo) & (cr.vrank <= hi))[0]
            if len(g) < 5: continue
            sc = cr.score[g]
            order = g[np.argsort(-sc, kind='stable')]
            k = int(np.floor(len(g) / 5))
            top = order[:k]
            ser[d] = np.nanmean(r[top]) - np.nanmean(r[g])
        res[f'{lo}-{hi}'] = summarize_month(ser)
    out[f'signal_q5_{mode}'] = res

# 前 100 名：對自身等權的超額 減 原版對自身等權的超額
pr = R.win_returns(NP['1-100'], S); pe = R.win_returns(NP['1-100|EW'], S)
p0 = R.win_returns(NAV['n10|w0|normal'], S); pe0 = R.win_returns(NAV['EW'], S)
out['top100_excessEW_minus_orig'] = summarize_month((pr - pe) - (p0 - pe0))
out['top100_vs_orig'] = summarize_month(pr - p0)
json.dump(out, open(R.HERE / 'res_attr.json', 'w'), ensure_ascii=False, indent=1)
for k, v in out.items(): print(k, json.dumps(v, ensure_ascii=False))
