#!/usr/bin/env python3
"""探索（只用挑選期 2006-01～2013-12）：python3 -I -B explore.py → explore.json

事前寫死的停止規則（在看到任何數字之前寫進程式）：
  以「營收前 1/5」（AVG 分數）的股票，按 52 週比股池內百分位分 10 段，看下一個換股窗（訊號日收盤→下一訊號日收盤）
  對同池等權的平均超額。若 (a) 10 段平均超額與段次的 Spearman ≥ 0.8，且 (b) 第 10 段是 10 段中最高、
  且 (c) 中段（第 4～8 段）平均 ≤ 低段（第 1～3 段）平均，則判「報酬隨 52 週高點單調增加、沒有還沒反映的溢酬」→ 停止，判不可用。
  （只要 (c) 不成立，即中段 > 低段，就繼續；(a)(b) 只是單調的描述。）
所有前瞻窗都在 2013-12-31 收盤截斷，不碰 2014 年以後的價格。
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import olib as O
V, K = O.V, O.K
T0 = time.time()
STOP_RULE = __doc__.split('事前寫死的停止規則')[1].split('所有前瞻窗')[0].strip()

R = O.revenue()
S = V.V2('gapfix', R['YL'], R['Y5'], exfin=True)
L = O.Leg(S)
d0 = O.leg_start(L)
days = [d for d in S.sig_days if d0 <= d <= O.SEL_END]
ends = days[1:] + [O.SEL_END]                       # 最後一窗截在 2013-12-31
assert max(ends) <= O.SEL_END
ti = [K.IDX.get_loc(d) for d in days]; tend = K.IDX.get_indexer(pd.DatetimeIndex(ends), method='ffill')
assert K.IDX[tend[-1]] <= O.SEL_END
A = K.MYADJ.ffill().to_numpy()
RW = np.array([A[b] / A[a] - 1 for a, b in zip(ti, tend)])          # 窗×股
S52 = O.V.rolling_signals('gapfix')[0]
NHFLAG = np.nan_to_num(S52, nan=0) >= 0.999                           # 當天收盤＝252 日新高
cum_nh = np.cumsum(NHFLAG, axis=0)
sel_end_t = K.IDX.get_loc(K.IDX[K.IDX <= O.SEL_END][-1])

rows = []   # 每月股池資訊
CS = []
for k, d in enumerate(days):
    o = L.leg_cs(d); el = o['el']
    cs = o['cs']
    sc = cs['sc']; main_rank = pd.Series(np.where(el, sc, np.nan)).rank(ascending=False, method='first').to_numpy()
    v60 = S.V60[cs['t']]; liq_rank = pd.Series(np.where(el, v60, np.nan)).rank(ascending=False, method='first').to_numpy()
    CS.append(dict(el=el, p52=o['p52'], pacc=o['pacc'], pnh=o['pnh'], avg=o['avg'], py3=o['py3'], main_rank=main_rank, liq_rank=liq_rank, t=cs['t']))

def ew_ex(k, mask):
    el = CS[k]['el']; r = RW[k]
    base = np.nanmean(r[el])
    return np.nanmean(r[mask & el]) - base if (mask & el).sum() else np.nan

def pavg(x, el):
    """AVG 分數在股池內的百分位（avg 是兩個百分位的平均，再取一次百分位以便切 1/5）。"""
    return O.pct_in(el & np.isfinite(x), x)

out = {'stop_rule_prespecified': STOP_RULE, 'period': [str(days[0].date()), str(O.SEL_END.date())], 'n_months': len(days)}

# ---------- (2) 形狀：營收前 1/5 × 52 週百分位 10 段 ----------
def decile_table(revkey):
    M = np.full((len(days), 10), np.nan); C = np.zeros((len(days), 10))
    for k in range(len(days)):
        c = CS[k]; el = c['el']
        rv = c[revkey] if revkey != 'avg' else pavg(c['avg'], el)
        strong = el & (rv >= 0.8)
        dec = np.clip(np.floor(c['p52'] * 10 - 1e-9), 0, 9)
        for q in range(10):
            m = strong & (dec == q)
            C[k, q] = m.sum()
            if m.sum() >= 1: M[k, q] = ew_ex(k, m)
    mean = np.nanmean(M, axis=0) * 100
    t = [O.tstat(M[:, q]) for q in range(10)]
    sp = float(pd.Series(mean).corr(pd.Series(np.arange(10)), method='spearman'))
    low = float(np.nanmean(M[:, 0:3])) * 100; mid = float(np.nanmean(M[:, 3:8])) * 100; top = float(np.nanmean(M[:, 8:10])) * 100
    return {'mean_ex_pct': mean.round(3).tolist(), 't': [round(x, 2) for x in t], 'avg_count': C.mean(axis=0).round(1).tolist(),
            'spearman_decile': round(sp, 3), 'low_d1_3': round(low, 3), 'mid_d4_8': round(mid, 3), 'top_d9_10': round(top, 3),
            'd10_is_max': bool(np.nanargmax(mean) == 9),
            't_mid_minus_low': round(O.tstat(np.nanmean(M[:, 3:8], axis=1) - np.nanmean(M[:, 0:3], axis=1)), 2),
            't_mid_minus_top': round(O.tstat(np.nanmean(M[:, 3:8], axis=1) - np.nanmean(M[:, 8:10], axis=1)), 2),
            't_mid_vs_EW': round(O.tstat(np.nanmean(M[:, 3:8], axis=1)), 2)}

shape = {k: decile_table(k) for k in ('avg', 'pacc', 'pnh', 'py3')}
# 無條件（全股池）的 52 週十分位
M = np.full((len(days), 10), np.nan)
for k in range(len(days)):
    c = CS[k]; dec = np.clip(np.floor(c['p52'] * 10 - 1e-9), 0, 9)
    for q in range(10): M[k, q] = ew_ex(k, c['el'] & (dec == q))
shape['all_pool'] = {'mean_ex_pct': (np.nanmean(M, axis=0) * 100).round(3).tolist(), 't': [round(O.tstat(M[:, q]), 2) for q in range(10)]}
# 剔除 2013（IFRS 斷點）
keep = np.array([d.year != 2013 for d in days])
out['shape'] = shape
sa = shape['avg']
stop = (sa['spearman_decile'] >= 0.8) and sa['d10_is_max'] and (sa['mid_d4_8'] <= sa['low_d1_3'])
out['stop_decision'] = {'monotone_spearman>=0.8': sa['spearman_decile'] >= 0.8, 'd10_is_max': sa['d10_is_max'],
                        'mid<=low': sa['mid_d4_8'] <= sa['low_d1_3'], 'STOP': bool(stop)}

# ---------- (1) 描述：營收強（AVG 前 1/5）且 52 週百分位 [0.4,0.85] ----------
def group_masks(k):
    c = CS[k]; el = c['el']; pa = pavg(c['avg'], el)
    strong = el & (pa >= 0.8); weak = el & (pa < 0.4)
    mid = (c['p52'] >= 0.4) & (c['p52'] <= 0.85); low = c['p52'] < 0.3; high = c['p52'] > 0.85
    return dict(G=strong & mid, LOWSTRONG=strong & low, HIGHSTRONG=strong & high, MIDWEAK=weak & mid, POOL=el,
                REST=el & ~(strong & high) & ~(strong & mid))
desc = {'count_per_month': {}, 'industry_share_G': {}, 'industry_share_pool': {}, 'elec_share': {}, 'liq_rank_quartiles': {}}
gc = {g: [] for g in ('G', 'LOWSTRONG', 'HIGHSTRONG', 'MIDWEAK', 'POOL', 'REST')}
indG, indP, liqG, liqP, elG, elP = {}, {}, [], [], [], []
for k in range(len(days)):
    gm = group_masks(k)
    for g in gc: gc[g].append(int(gm[g].sum()))
    for j in np.where(gm['G'])[0]: indG[O.INDUSTRY[j]] = indG.get(O.INDUSTRY[j], 0) + 1
    for j in np.where(gm['POOL'])[0]: indP[O.INDUSTRY[j]] = indP.get(O.INDUSTRY[j], 0) + 1
    liqG += CS[k]['liq_rank'][gm['G']].tolist(); liqP += CS[k]['liq_rank'][gm['POOL']].tolist()
    elG.append(O.ELEC[gm['G']].mean() if gm['G'].sum() else np.nan); elP.append(O.ELEC[gm['POOL']].mean())
desc['count_per_month'] = {g: {'mean': round(float(np.mean(v)), 1), 'min': int(np.min(v)), 'max': int(np.max(v))} for g, v in gc.items()}
tg, tp = sum(indG.values()), sum(indP.values())
desc['industry_share_G'] = {k: round(v / tg, 3) for k, v in sorted(indG.items(), key=lambda x: -x[1])[:12]}
desc['industry_share_pool'] = {k: round(v / tp, 3) for k, v in sorted(indP.items(), key=lambda x: -x[1])[:12]}
desc['elec_share'] = {'G': round(float(np.nanmean(elG)), 3), 'POOL': round(float(np.nanmean(elP)), 3)}
desc['liq_rank_quartiles'] = {'G': np.percentile(liqG, [25, 50, 75]).round(0).tolist(), 'POOL': np.percentile(liqP, [25, 50, 75]).round(0).tolist(),
                              'G_share_rank_101_300': round(float(np.mean(np.array(liqG) > 100)), 3), 'POOL_share_rank_101_300': round(float(np.mean(np.array(liqP) > 100)), 3)}
out['describe'] = desc

# ---------- 轉移機率：之後 1/3/6 個月內收盤創 252 日新高 ----------
H = {'1m': 21, '3m': 63, '6m': 126}
def reach_high(t, h):
    te = t + h
    if te > sel_end_t: return None
    return (cum_nh[te] - cum_nh[t]) > 0
trans = {}
for hn, h in H.items():
    acc = {g: [] for g in gc}
    accdec = [[] for _ in range(10)]
    for k in range(len(days)):
        rh = reach_high(CS[k]['t'], h)
        if rh is None: continue
        gm = group_masks(k)
        for g in gc:
            if gm[g].sum(): acc[g].append(rh[gm[g]].mean())
        c = CS[k]; pa = pavg(c['avg'], c['el']); strong = c['el'] & (pa >= 0.8)
        dec = np.clip(np.floor(c['p52'] * 10 - 1e-9), 0, 9)
        for q in range(10):
            m = strong & (dec == q)
            if m.sum(): accdec[q].append(rh[m].mean())
    trans[hn] = {'by_group': {g: round(float(np.mean(v)), 3) for g, v in acc.items()},
                 'strong_rev_by_52w_decile': [round(float(np.mean(v)), 3) if v else None for v in accdec], 'n_months': len(acc['G'])}
out['transition_to_new_high'] = trans

# ---------- P2：之後 1～3 個月進入主策略前 30 名的機率 ----------
p2 = {g: [] for g in gc}
for k in range(len(days) - 3):
    gm = group_masks(k)
    future = np.zeros(O.N, bool)
    for h in (1, 2, 3):
        future |= (CS[k + h]['main_rank'] <= 30)
    for g in gc:
        m = gm[g] & ~(CS[k]['main_rank'] <= 30)          # 當下不在前 30 名的
        if m.sum(): p2[g].append(future[m].mean())
out['P2_enter_main_top30_within_3m'] = {g: round(float(np.mean(v)), 3) for g, v in p2.items()}
out['P2_t_G_minus_REST'] = round(O.tstat(np.array(p2['G']) - np.array(p2['REST'])), 2)
out['P2_t_G_minus_LOWSTRONG'] = round(O.tstat(np.array(p2['G']) - np.array(p2['LOWSTRONG'])), 2)

# ---------- 報酬集中：G 之後 3 個月內有沒有走到新高 ----------
conc = {'reached': [], 'not': [], 'share_reached': []}
for k in range(len(days) - 3):
    rh = reach_high(CS[k]['t'], 63)
    if rh is None: continue
    gm = group_masks(k); el = CS[k]['el']
    r3 = (1 + RW[k:k + 3]).prod(axis=0) - 1; base = np.nanmean(r3[el])
    a = gm['G'] & rh; b = gm['G'] & ~rh
    if a.sum(): conc['reached'].append(np.nanmean(r3[a]) - base)
    if b.sum(): conc['not'].append(np.nanmean(r3[b]) - base)
    if gm['G'].sum(): conc['share_reached'].append(rh[gm['G']].mean())
out['return_concentration_3m'] = {'ex_reached_high_pct': round(float(np.mean(conc['reached'])) * 100, 2),
                                  'ex_not_reached_pct': round(float(np.mean(conc['not'])) * 100, 2),
                                  'share_reached': round(float(np.mean(conc['share_reached'])), 3),
                                  'note': '用到未來資訊分組（事後），只用來確認報酬是不是來自「後來追上」的那批，不是可交易訊號'}

# ---------- 事件時間：G 形成後第 h 個月的超額（資訊擴散速度） ----------
evt = {}
for g in ('G', 'LOWSTRONG', 'HIGHSTRONG', 'MIDWEAK'):
    seq = []
    for h in range(0, 6):
        xs = []
        for k in range(len(days) - h):
            gm = group_masks(k)
            if gm[g].sum() == 0: continue
            el = CS[k + h]['el']; r = RW[k + h]
            xs.append(np.nanmean(r[gm[g]]) - np.nanmean(r[el]))
        seq.append((round(float(np.mean(xs)) * 100, 3), round(O.tstat(xs), 2)))
    evt[g] = {'month_h_mean_ex_pct_t': seq}
out['event_time'] = evt

# ---------- 子樣本：G 的下月超額 ----------
sub = {}
def sub_ex(cond_fn, label, kmask=None):
    xs = []
    for k in range(len(days)):
        if kmask is not None and not kmask[k]: continue
        gm = group_masks(k); m = gm['G'] & cond_fn(k)
        if m.sum() == 0: continue
        el = CS[k]['el'] & cond_fn(k)   # 同子樣本等權
        xs.append(np.nanmean(RW[k][m]) - np.nanmean(RW[k][el]))
    sub[label] = {'mean_ex_pct': round(float(np.mean(xs)) * 100, 3), 't': round(O.tstat(xs), 2), 'n': len(xs)}
allm = lambda k: np.ones(O.N, bool)
sub_ex(allm, 'G_all')
sub_ex(lambda k: CS[k]['liq_rank'] <= 100, 'liq_1_100')
sub_ex(lambda k: CS[k]['liq_rank'] > 100, 'liq_101_300')
sub_ex(lambda k: O.ELEC, 'elec')
sub_ex(lambda k: ~O.ELEC, 'non_elec')
sub_ex(allm, 'year_2013_only', kmask=np.array([d.year == 2013 for d in days]))
sub_ex(allm, 'ex_2013', kmask=np.array([d.year != 2013 for d in days]))
sub_ex(allm, '2006_2008', kmask=np.array([d.year <= 2008 for d in days]))
sub_ex(allm, '2009_2012', kmask=np.array([2009 <= d.year <= 2012 for d in days]))
out['subsample_G_next_window'] = sub
# 只看 G 的頂端（營收分數最高的 10 檔），對照主策略前 10 名的下月超額與兩者相關
top10, main10 = [], []
for k in range(len(days)):
    c, cand_ok = L.candidates(days[k], 'AVG', 'B2')
    pk = c[:10]; el = CS[k]['el']; base = np.nanmean(RW[k][el])
    top10.append(np.nanmean(RW[k][pk]) - base)
    mk = np.where(CS[k]['main_rank'] <= 10)[0]
    main10.append(np.nanmean(RW[k][mk]) - base)
out['quick_AVG_B2_top10_vs_main_top10'] = {'leg_mean_ex_pct': round(float(np.mean(top10)) * 100, 3), 'leg_t': round(O.tstat(top10), 2),
    'main_mean_ex_pct': round(float(np.mean(main10)) * 100, 3), 'main_t': round(O.tstat(main10), 2), 'corr': round(O.corr(np.array(top10), np.array(main10)), 3),
    'note': '收盤到收盤、未扣成本、無開盤漲停遞補，只是描述；不是版本迭代'}
out['runtime_s'] = round(time.time() - T0, 1)
(O.HERE / 'explore.json').write_text(json.dumps(O.r4(out), ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(O.r4(out), ensure_ascii=False, indent=1))
