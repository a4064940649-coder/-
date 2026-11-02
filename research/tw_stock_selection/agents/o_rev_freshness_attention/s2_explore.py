#!/usr/bin/env python3
"""步驟 2：探索（只用前期 2006-04～2018-12 的報酬；2019 年後只看不含報酬的描述統計）。
  E1 時間戳閘門（s1_gate 的結果摘要）
  E2 流程檢查：本題的排序與漲停遞補，在全 el 取前 10 時，要重現 o_v7 plans_j7.csv 的 AB_V7 seed 0 名單（前期、13 個 j）
  E3 關注度三分位的穩定性（官方訊號日，月對月轉移矩陣；全期，不含報酬）
  E4 V60 百分位與主策略分數的 Spearman（每月，全期，不含報酬）；各三分位占主策略前 10（o_v7 plans_ab7 AB_V7 seed 0）的比例
  E5 0050 成分、外資持股組的池內檔數（不含報酬）
  E6 前期 α_g(j)：A 三分位 × j 的平均、斜率 b_g、I_前（前期安慰劑）；營收腿與價格腿；M50、COMBO（前期）
輸出 explore.json、data/alpha_pre_v1.parquet、data/contrib_pre_v1.parquet。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import fa_common as C, fa_build as B, fa_stats as ST

E = C.engine(); K, L, STC = E['K'], E['L'], E['STC']
OUT = {}
G = C.load_res().get('GATE_TIMESTAMP', {})
OUT['E1_gate'] = {k: G.get(k) for k in ('population_stock_months', 'coverage_exact_or_upper_bound', 'pass', 'verdict')}

# ---------- E2 流程檢查 ----------
PJ = pd.read_csv(C.V7DIR / 'data' / 'plans_j7.csv', dtype={'picks': str})
PJ = PJ[(PJ.strat == 'AB_V7') & (PJ.seed == 0)]
S0, J, amap = B.strat(0)
_, V60x = E['V'].rolling_signals('gapfix')
agree = {}; nmis = 0; ntot = 0
for j in C.JS:
    ref = {pd.Timestamp(r.signal): r.picks.split(',') for r in PJ[PJ.j == j].itertuples()}
    held = set(); ok = 0; n = 0
    for m in sorted(J[j]):
        d, e, e2 = J[j][m]
        if m > C.PRE[1]: break
        cs = S0.cross_section(d)
        if cs['el'].sum() < 50: continue
        gu = K.gap_up_open(e)
        pk = B.pick(cs['sc'], cs['el'], 10, gu, held, cs['y3'])
        held = set(pk)
        if d in ref:
            n += 1; ok += int(sorted(K.COLS[pk]) == sorted(ref[d]))
    agree[j] = ok / n if n else None; nmis += n - ok; ntot += n
OUT['E2_plumbing_top10_vs_plans_j7'] = {'by_j': agree, 'overall': 1 - nmis / ntot, 'n_month_j': ntot}

# ---------- E3/E4/E5 描述（全期，不含報酬） ----------
OFF = E['OFF']
PA = pd.read_csv(C.V7DIR / 'data' / 'plans_ab7.csv', dtype={'picks': str})
PA = PA[(PA.strat == 'AB_V7') & (PA.seed == 0)]
ref_off = {pd.Timestamp(r.signal): r.picks.split(',') for r in PA.itertuples()}
prevg = None; trans = {'PRE': np.zeros((3, 3)), 'POST': np.zeros((3, 3))}
sp = {'PRE': [], 'POST': []}; share = {'PRE': np.zeros(3), 'POST': np.zeros(3)}
sizes = {'PRE': [], 'POST': []}
for d in OFF.sig_days:
    m = pd.Period(d, 'M')
    per = 'PRE' if m <= C.PRE[1] else ('POST' if m >= C.POST[0] else None)
    cs = dict(OFF.cross_section(d)); cs['v60x'] = V60x[cs['t']]
    gr = B.groups_for(cs, d, cs['e'], B.DEFAULT)
    gA = gr['A']
    if per:
        if prevg is not None:
            both = (gA >= 0) & (prevg >= 0)
            for a in range(3):
                for b in range(3): trans[per][a, b] += ((prevg == a) & (gA == b) & both).sum()
        el = cs['el']
        sp[per].append(pd.Series(cs['v60x'][el]).rank().corr(pd.Series(cs['sc'][el]).rank()))
        if d in ref_off:
            for c in ref_off[d]:
                g = gA[K.CI[c]]
                if g >= 0: share[per][g] += 1
        sizes[per].append(dict(el=int(el.sum()), m50=int((gr['M50'] == 1).sum()), fo=int((gr['FO'] >= 0).sum()), combo=int((gr['COMBO'] == 0).sum())))
    prevg = gA
OUT['E3_tercile_transition_row_normalized'] = {p: (trans[p] / trans[p].sum(axis=1, keepdims=True)).round(3).tolist() for p in trans}
OUT['E3_stay_prob'] = {p: float(np.trace(trans[p]) / trans[p].sum()) for p in trans}
OUT['E4_spearman_V60_score'] = {p: dict(mean=float(np.mean(v)), p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90))) for p, v in sp.items()}
OUT['E4_share_of_top10_by_tercile_low_mid_high'] = {p: (share[p] / share[p].sum()).round(3).tolist() for p in share}
OUT['E5_group_sizes_mean'] = {p: pd.DataFrame(v).mean().round(1).to_dict() for p, v in sizes.items()}

# ---------- E6 前期 α ----------
months = set(pd.period_range(C.PRE[0], C.PRE[1], freq='M'))
A, CR = B.build(months)
A.to_parquet(C.DATA / 'alpha_pre_v1.parquet')
CF = B.contrib_frame(CR); CF.to_parquet(C.DATA / 'contrib_pre_v1.parquet')
pre = {}
for sch, leg, hi, lo in (('A', 'main', 2, 0), ('A', 'rev', 2, 0), ('A', 'px', 2, 0), ('M50', 'main', 1, 0)):
    s, I, bh, bl = ST.interaction(A, sch, leg, hi, lo, per=C.PRE)
    pre[f'{sch}|{leg}'] = s
pre['A|main|by_j'] = {g: ST.alpha_by_j(A, 'A', 'main', g, per=C.PRE) for g in (0, 1, 2)}
pre['A|main|mid'] = ST.summ(ST.month_slopes(A, 'A', 'main', 1, per=C.PRE))
pre['COMBO|main'] = ST.summ(ST.month_slopes(A, 'COMBO', 'main', 0, per=C.PRE))
pre['A|main|alpha_level_pp'] = {g: float(A[(A.scheme == 'A') & (A.leg == 'main') & (A.g == g)].alpha.mean() * 100) for g in (0, 1, 2)}
pre['A|main|gross_level_pp'] = {g: float(A[(A.scheme == 'A') & (A.leg == 'main') & (A.g == g)].alpha_gross.mean() * 100) for g in (0, 1, 2)}
pre['turnover_newfrac'] = {g: float(A[(A.scheme == 'A') & (A.leg == 'main') & (A.g == g)].newf.mean()) for g in (0, 1, 2)}
c, nm = ST.stock_contrib(CF, C.PRE)
pre['stock_contrib_check'] = dict(sum_c_pp=float(c.sum() * 100), months=nm, max_share=float((c / c.sum()).max()))
OUT['E6_pre_alpha'] = pre
C.jdump(OUT, C.HERE / 'explore.json')
print(json.dumps(C.conv({k: v for k, v in OUT.items() if k != 'E6_pre_alpha'}), ensure_ascii=False)[:3000])
print(json.dumps(C.conv({k: pre[k] for k in ('A|main', 'A|rev', 'A|px', 'M50|main', 'COMBO|main', 'A|main|alpha_level_pp', 'stock_contrib_check')}), ensure_ascii=False, indent=0)[:3000])
