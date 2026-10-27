#!/usr/bin/env python3
"""探索：只用挑選期（訊號日 < 2019-04-11、報酬到 2019-04-11 換股日為止）。輸出 explore.json。"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import ocore as O
K, V = O.K, O.V
T0 = time.time()
assert json.loads((O.HERE / 'results.json').read_text(encoding='utf-8')).get('PLAN_sha256'), '先執行 plan.py'
SPLIT = O.SPLIT
tS = O.IDX.get_loc(SPLIT)
res = {'note': '只用挑選期：訊號日 < 2019-04-11；所有期報酬在 2019-04-12（2019-04 成交日）開盤以前結束。', 'periods': {'SEL_a': '2006-04-11', 'SEL_b': '2019-04-11'}}

S = O.V2R(1, 300)
t0, t1, sd = S.bounds()
plans = {n: S.plan(n) for n in (10, 30)}

# ---------- 1. 入選股名次歸因 ----------
res['attrib'] = {}
EV = {}
for n in (10, 30):
    df = O.event_table(S, plans[n])
    df = df[df['d'] < SPLIT].copy()
    # 期報酬終點必須在 SPLIT 之前或當天：最後一個 SEL 期（2019-03）結束於 2019-04-12 開盤，只多 1 天開盤，視同邊界
    EV[n] = df
    r = {}
    yr = df.assign(y=df['d'].dt.year).groupby(['y', 'bucket']).size().unstack(fill_value=0)
    yr.columns = [O.BNAME[c] for c in yr.columns]
    r['rank_dist_by_year'] = {int(y): {**row.to_dict(), 'mean_rank': round(float(df[df['d'].dt.year == y]['rank'].mean()), 1),
                                        'median_rank': float(df[df['d'].dt.year == y]['rank'].median())} for y, row in yr.iterrows()}
    r['rank_share_SEL'] = {O.BNAME[b]: round(float((df['bucket'] == b).mean()), 3) for b in range(4)}
    for lab, lo, hi in (('SEL', '2006-01-01', '2019-04-10'), ('2006_2013', '2006-01-01', '2013-12-31'), ('2014_2019Q1', '2014-01-01', '2019-04-10')):
        x = df[(df['d'] >= pd.Timestamp(lo)) & (df['d'] <= pd.Timestamp(hi))]
        r[lab] = {'ex_pool': O.group_summary(x, 'ex_pool'), 'ex_bucket': O.group_summary(x, 'ex_bucket'),
                  'net_raw': O.group_summary(x.assign(netc=x['net']), 'netc'),
                  'mid_minus_large_ex_pool': O.mml_stats(O.mid_minus_large(x, 'ex_pool')),
                  'mid_minus_large_ex_bucket': O.mml_stats(O.mid_minus_large(x, 'ex_bucket')),
                  'clustered_ex_pool': O.clustered_reg(x, 'ex_pool'), 'clustered_ex_bucket': O.clustered_reg(x, 'ex_bucket'),
                  'bucket_ew_minus_pool_ew_pp': {O.BNAME[b]: round(float((x[x['bucket'] == b]['ew_bucket'] - x[x['bucket'] == b]['ew_pool']).mean()) * 100, 3) for b in range(4)}}
    res['attrib'][f'n{n}'] = r
    print(f'[{time.time()-T0:5.0f}s] n{n}', r['SEL']['mid_minus_large_ex_pool'], r['SEL']['mid_minus_large_ex_bucket'], flush=True)

# ---------- 2. 全池訊號效力：各名次組內分數前 1/5 減組內等權，第 1～3 個月 ----------
ex = [p[0] for p in plans[10]]
sig = [p[3] for p in plans[10]]
def oo(t, t2, idx):
    """成交日 t 開盤 → t2 開盤（t2 沒成交則 NaN）。"""
    g = K.Cn[t, idx] / K.On[t, idx] * np.prod(K.GROW[t + 1:t2 + 1, idx], axis=0)
    ok = K.TRADED[t2, idx] & np.isfinite(K.On[t2, idx])
    return np.where(ok, g * K.On[t2, idx] / np.where(ok, K.Cn[t2, idx], 1.0), np.nan) - 1
eff = {b: {h: [] for h in (1, 2, 3)} for b in range(4)}
for i, d in enumerate(sig):
    if d >= SPLIT: break
    cs = S.cross_section(d)
    el = np.where(cs['el'])[0]
    bk = O.bucket_of(cs['rank'][el])
    for b in range(4):
        m = el[bk == b]
        if len(m) < 15: continue
        scb = cs['sc'][m]
        q = np.quantile(scb, 0.8)
        topq = m[scb >= q]
        for h in (1, 2, 3):
            if i + h >= len(ex) or ex[i + h] > tS + 1: continue
            ta, tb = ex[i + h - 1], ex[i + h]
            ra = oo(ta, tb, m); rt = oo(ta, tb, topq)
            eff[b][h].append((d, np.nanmean(rt) - np.nanmean(ra)))
res['signal_efficacy_by_bucket'] = {O.BNAME[b]: {f'month{h}': O.mml_stats(pd.Series([v for _, v in eff[b][h]])) for h in (1, 2, 3)} for b in range(4)}
res['signal_efficacy_note'] = '每個名次組內（合格池），依主策略分數取前 1/5，減組內等權；第 h 個月＝訊號後第 h 個持有期（成交日開盤→下一成交日開盤），毛報酬，月序列 t。'
print(f'[{time.time()-T0:5.0f}s] eff', {k: v['month1'] for k, v in res['signal_efficacy_by_bucket'].items()}, flush=True)

# ---------- 3. 2330 歸因（只用 SEL 的訊號窗） ----------
ew, _ = K.simulate([(p[0], p[2]) for p in plans[10]], 0, 0, t0, t1)
nav, _ = K.simulate([(p[0], p[1]) for p in plans[10]], *O.NORMAL, t0, t1)
W = pd.DataFrame({'S': O.window_returns(nav, sd), 'E50': O.window_returns(K.E50, sd), 'TSMC': O.window_returns(O.TSMC_IDX, sd), 'EW': O.window_returns(ew, sd)})
W = W[W.index < SPLIT]
W['y'] = W['S'] - W['E50']; W['x_tsmc'] = W['TSMC'] - W['E50']; W['x_ew'] = W['EW'] - W['E50']
reg = {}
for lab, lo, hi in (('SEL', '2006-01-01', '2019-04-10'), ('2006_2013', '2006-01-01', '2013-12-31'), ('2014_2019Q1', '2014-01-01', '2019-04-10')):
    x = W[(W.index >= pd.Timestamp(lo)) & (W.index <= pd.Timestamp(hi))]
    a = O.ols(x['y'], x[['x_tsmc']]); b = O.ols(x['y'], x[['x_ew']]); c = O.ols(x['y'], x[['x_tsmc', 'x_ew']])
    reg[lab] = {'on_tsmc': {k: v for k, v in a.items() if k != 'resid'}, 'on_ew': {k: v for k, v in b.items() if k != 'resid'},
                'on_both': {k: v for k, v in c.items() if k != 'resid'}, 'mean_y_pp': round(float(x['y'].mean()) * 100, 3),
                'mean_x_tsmc_pp': round(float(x['x_tsmc'].mean()) * 100, 3), 'corr_tsmc_E50': round(float(x[['TSMC', 'E50']].corr().iloc[0, 1]), 3)}
res['tsmc_regression_SEL'] = reg
print(f'[{time.time()-T0:5.0f}s] reg', {k: (v['on_tsmc']['coef'], v['on_tsmc']['R2']) for k, v in reg.items()}, flush=True)

# ---------- 4. 台積電在 0050 的權重（粗略近似，只到 SEL 終點） ----------
# 每年：以當年第一個訊號日流動性前 70 名當作成分股近似，0050 日報酬對這 70 檔日報酬做非負最小平方（無截距），取 2330 的係數。
res['tsmc_weight_in_0050_proxy_SEL'] = O.tsmc_weight_proxy(S, [p[3] for p in plans[10]], 2006, 2018, t_max=tS)
res['tsmc_weight_proxy_note'] = '粗略：成分股用流動性前 70 名近似、權重年內視為固定；2025-09～2026-09 的實際權重 57～60%（extras/etf0050_w.csv、ex_tsmc.csv）在 oos.py 用來校對。'
print(f'[{time.time()-T0:5.0f}s] wproxy', res['tsmc_weight_in_0050_proxy_SEL'], flush=True)

# ---------- 5. 容量（SEL） ----------
cap = {}
for n in (10, 30):
    df = EV[n]
    c = {}
    for C in (1e6, 1e7, 5e7):
        ratio = (C / n) / df['adv20']
        c[f'{int(C/1e4)}萬'] = {'median_pct': round(float(ratio.median()) * 100, 3), 'p90_pct': round(float(ratio.quantile(0.9)) * 100, 3),
                               'p99_pct': round(float(ratio.quantile(0.99)) * 100, 3), 'share_gt5pct': round(float((ratio > 0.05).mean()), 4),
                               'share_gt1pct': round(float((ratio > 0.01).mean()), 4),
                               'by_bucket_share_gt5pct': {O.BNAME[b]: round(float((ratio[df['bucket'] == b] > 0.05).mean()), 4) for b in range(4)}}
    c['adv20_median_by_bucket_億'] = {O.BNAME[b]: round(float(df[df['bucket'] == b]['adv20'].median()) / 1e8, 3) for b in range(4)}
    cap[f'n{n}'] = c
# 全池各名次組的 ADV（換股日，SEL 各年中位數），以及「1% 權重的基金規模，每天只占 ADV 10% 需要幾天建倉」的換算基礎
adv_year = {}
for d in sig:
    if d >= SPLIT: break
    cs = S.cross_section(d)
    rk = cs['rank']; v = cs['v60']
    for b, (lo, hi) in enumerate(O.BUCKETS):
        m = (rk >= lo) & (rk <= hi)
        adv_year.setdefault(d.year, {}).setdefault(O.BNAME[b], []).append(float(np.median(v[m])))
cap['pool_v60_median_by_bucket_億_by_year'] = {y: {b: round(float(np.median(x)) / 1e8, 2) for b, x in v.items()} for y, v in adv_year.items()}
res['capacity_SEL'] = cap
print(f'[{time.time()-T0:5.0f}s] cap', cap['n10']['5000萬'], flush=True)

# ---------- 6. 名次組的產業分布（SEL，n10） ----------
ind = V.SL['industry']
df = EV[10]
res['industry_by_bucket_SEL_n10'] = {O.BNAME[b]: df[df['bucket'] == b]['code'].map(lambda c: ind.get(c, '（無）')).value_counts().head(6).to_dict() for b in range(4)}
(O.HERE / 'explore.json').write_text(O.dumps(res), encoding='utf-8')
print('done', time.time() - T0)
