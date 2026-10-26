#!/usr/bin/env python3
"""檢定 3：配對交易的只做多版。

事前固定規則：
  配對形成（每月最後交易日 t）：t 當日可投資（60 日均值 ≥ 1,000 萬、價 ≥ 10）、流動性前 300 名、有現行產業分類；
    同產業兩兩計算過去 250 個交易日日報酬相關（至少 240 天有值），≥ 0.7 者列入，供 (t, 下一月底] 期間進場使用。
  價差：x = log(還原價A) − log(還原價B)；z = (x − 過去 250 日均值) / 過去 250 日標準差（含當日）。
  進場：收盤 z ≤ −2 → 隔日開盤買 A（落後者）；z ≥ +2 → 隔日開盤買 B。同一檔已持有則不重複；同日多個訊號取 |z| 最大的配對。
        隔日開盤鎖漲停（開=高 且 漲幅 ≥ 漲停−0.5%）或無開盤價 → 放棄。
  出場：落後者方向的 z 回到 ≥ 0（收盤確認，隔日開盤賣）或持有滿 20 個交易日（第 20 個交易日開盤賣），先到者。
  成本：每筆來回 0.5%（壓力 1%）。持有期間停止交易以最後價計。
  組合：每日把所有持倉等權（空手時 0 報酬、現金），彙總成月報酬，對 0050、同池（前 300 名有產業）等權比較；
        另報告逐筆：平均淨報酬、勝率、同期間 0050 超額（以進場前一日收盤 → 出場前一日收盤近似）。
  敏感度（只報告、不挑）：相關 ≥ 0.8；進場 |z| ≥ 2.5。
  挑選期 2010-01～2019-04、驗證期 2019-05～2026-10（依進場日歸屬）。
"""
import json, warnings
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
warnings.filterwarnings('ignore')

adj, C, O, H, V, Rd = load('adj', 'close_raw', 'open', 'high', 'value', 'ret')
ind = industries(); cols = [c for c in adj.columns if c in ind.index]
adj, C, O, H, V, Rd = adj[cols], C[cols], O[cols], H[cols], V[cols], Rd[cols]
dates = adj.index; N = len(dates); pos = {d: i for i, d in enumerate(dates)}
me = month_ends(dates)
g = ind.reindex(cols)
el, v60 = eligibility(me, C, V, lag=0); U = top_liq(el, v60, 300)

# 開盤還原價、收盤還原價（ffill 用於停牌）
Aadj = adj.ffill(limit=60)
Oadj = (O * adj / C)
prevC = C.ffill().shift(1)
thr = pd.Series(np.where(dates < pd.Timestamp('2015-06-01'), 0.065, 0.095), index=dates)
lockO = ((O >= H - 1e-9) & (O / prevC - 1).ge(thr, axis=0))
buyable = Oadj.notna() & ~lockO
e50 = e50_series().reindex(dates, method='ffill')

# 每月配對清單
def form_pairs(minc):
    plist = {}
    for t in me:
        if t < pd.Timestamp('2009-12-01'): continue
        i = pos[t]; names = U.loc[t][U.loc[t]].index
        W = Rd.iloc[i - 249:i + 1][names]; W = W.loc[:, W.notna().sum() >= 240]
        prs = []
        gg = g.reindex(W.columns)
        for k, idx in gg.groupby(gg).groups.items():
            if len(idx) < 2: continue
            cm = W[idx].corr(min_periods=200).values; iu = np.triu_indices(len(idx), 1)
            sel = cm[iu] >= minc
            prs += [(idx[a], idx[b]) for a, b in zip(iu[0][sel], iu[1][sel])]
        plist[t] = prs
    return plist

LA = np.log(Aadj)
zcache = {}
def zser(a, b):
    if (a, b) not in zcache:
        x = LA[a] - LA[b]
        m = x.rolling(250, min_periods=240).mean(); s = x.rolling(250, min_periods=240).std()
        zcache[(a, b)] = ((x - m) / s).values
    return zcache[(a, b)]

AA = Aadj.values; OA = Oadj.ffill(limit=60).values; BUY = buyable.values; cidx = {c: j for j, c in enumerate(cols)}
E50 = e50.values

def simulate(plist, zin=2.0, cost=0.005, hold=20):
    # 每日可用的配對：屬於「最近一個 ≤ d−? 的月底」形成的清單（形成日 t 之後的交易日）
    active = [None] * N; mlist = sorted(plist)
    k = 0
    for i, d in enumerate(dates):
        while k + 1 < len(mlist) and mlist[k + 1] < d: k += 1
        if mlist and mlist[k] < d: active[i] = plist[mlist[k]]
    holding = {}   # code -> dict(entry_i, pair, sign)
    trades = []
    for i in range(N - 1):
        # 出場檢查（收盤 i 確認 → i+1 開盤賣）或時間到（i+1 為第 20 天開盤）
        for c in list(holding):
            h = holding[c]; z = zser(*h['pair'])[i] * h['sign']
            if (i + 1 - h['ei'] >= hold) or (np.isfinite(z) and z >= 0):
                xi = i + 1; j = cidx[c]
                r = OA[xi, j] / OA[h['ei'], j] - 1
                ld = cidx[h['pair'][1] if h['pair'][0] == c else h['pair'][0]]
                rl = OA[xi, ld] / OA[h['ei'], ld] - 1   # 領先者同期間報酬（診斷：價差是否收斂）
                trades.append({'code': c, 'pair': h['pair'], 'ei': h['ei'], 'xi': xi, 'ret': r - cost,
                               'r50': E50[xi - 1] / E50[h['ei'] - 1] - 1, 'zin': h['z'], 'cost': cost, 'gross': r, 'leader': rl})
                del holding[c]
        if active[i] is None: continue
        cand = {}
        for a, b in active[i]:
            z = zser(a, b)[i]
            if not np.isfinite(z): continue
            if z <= -zin: lag, sign = a, 1
            elif z >= zin: lag, sign = b, -1
            else: continue
            if lag in holding: continue
            if lag not in cand or abs(z) > abs(cand[lag][1]): cand[lag] = ((a, b), z, sign)
        for c, (pr, z, sign) in cand.items():
            j = cidx[c]
            if not BUY[i + 1, j]: continue
            holding[c] = {'ei': i + 1, 'pair': pr, 'sign': sign, 'z': z}
    return pd.DataFrame(trades)

def daily_portfolio(tr):
    """持倉等權的日報酬（成本在進場日扣）。"""
    s = np.zeros(N); n = np.zeros(N)
    for row in tr.itertuples():
        j = cidx[row.code]
        # 進場日：開→收；中間：收→收；出場日：前收→開；成本在進場日扣
        vals = [AA[row.ei, j] / OA[row.ei, j] - 1] + [AA[k, j] / AA[k - 1, j] - 1 for k in range(row.ei + 1, row.xi)] + [OA[row.xi, j] / AA[row.xi - 1, j] - 1]
        vals = np.nan_to_num(np.array(vals), nan=0.0); vals[0] -= row.cost
        idx = np.arange(row.ei, row.xi + 1)
        s[idx] += vals; n[idx] += 1
    r = np.where(n > 0, s / np.maximum(n, 1), 0.0)
    return pd.Series(r, index=dates), pd.Series(n, index=dates)

def to_monthly(dr):
    eq = (1 + dr).cumprod(); m = eq.reindex(me)
    return (m.shift(-1) / m - 1).iloc[:-1]

R_m = period_returns(adj, me); pool_m = pool_ew(R_m, U)
B = bench_returns(e50_series(), me)

res = {}
periods = {'IS_2010-01_2019-04': ('2010-01-01', '2019-04-30'), 'OOS_2019-05_2026-10': (OOS_START, OOS_END)}
P07 = form_pairs(0.7); P08 = form_pairs(0.8)
res['pairs_per_month'] = {str(y): round(float(v), 1) for y, v in pd.Series({t: len(p) for t, p in P07.items()}).groupby(lambda d: d.year).mean().items()}
configs = {'main_c0.7_z2_cost0.5': (P07, 2.0, 0.005), 'stress_cost1pct': (P07, 2.0, 0.01),
           'sens_c0.8_z2': (P08, 2.0, 0.005), 'sens_c0.7_z2.5': (P07, 2.5, 0.005)}
for name, (pl, zin, cost) in configs.items():
    tr = simulate(pl, zin, cost)
    tr['entry'] = dates[tr.ei.values]; tr['days'] = tr.xi - tr.ei
    dr, npos = daily_portfolio(tr)
    mr = to_monthly(dr)
    for pn, (a, b) in periods.items():
        tt = tr[(tr.entry >= pd.Timestamp(a)) & (tr.entry <= pd.Timestamp(b))]
        ex = tt.ret - tt.r50
        exm = ex.groupby(tt.entry.dt.to_period('M')).mean()
        o = {'n_trades': int(len(tt)), 'avg_net_ret_pct': round(float(tt.ret.mean() * 100), 3), 'median_net_pct': round(float(tt.ret.median() * 100), 3),
             'win_rate': round(float((tt.ret > 0).mean()), 3), 'avg_days': round(float(tt.days.mean()), 1),
             'exit_by_z0_share': round(float((tt.days < 20).mean()), 3),
             'avg_ex_0050_pct_per_trade': round(float(ex.mean() * 100), 3), 't_ex_0050_trade_months': tstat(exm),
             'avg_laggard_minus_leader_gross_pct': round(float((tt.gross - tt.leader).mean() * 100), 3),
             't_laggard_minus_leader_months': tstat((tt.gross - tt.leader).groupby(tt.entry.dt.to_period('M')).mean()),
             'by_industry_n': tt.code.map(g).value_counts().head(6).to_dict(),
             'avg_positions_when_any': round(float(cut(npos[npos > 0], a, b).mean()), 2),
             'days_invested_share': round(float((cut(npos, a, b) > 0).mean()), 3)}
        o['portfolio'] = eval_monthly(mr, me, B, pool_m, a, b)
        res[f'{name}|{pn}'] = o
        print(f"{name:22s} {pn:20s} trades={o['n_trades']} avg={o['avg_net_ret_pct']}% win={o['win_rate']} days={o['avg_days']} ex0050/trade={o['avg_ex_0050_pct_per_trade']}% t={o['t_ex_0050_trade_months']} invested={o['days_invested_share']} z0exit={o['exit_by_z0_share']} lag-lead={o['avg_laggard_minus_leader_gross_pct']}% t={o['t_laggard_minus_leader_months']} | port {fmt(o['portfolio'])}")
print(res['pairs_per_month'])
(OUTDIR / 'results_s3.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
