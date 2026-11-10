# 反方審查 o19（資料與統計）：只讀研究員檔案，不修改
import json, numpy as np, pandas as pd, os
H = os.path.dirname(os.path.abspath(__file__))
D7 = '/home/user/-/research/tw_stock_selection/agents/o_v7_listing_exec_engine/data/'
def rpq(p):
    with open(p, 'rb') as f: return pd.read_parquet(f)
A = json.load(open(os.path.join(H, 'analysis.json')))
out = {}
# 1) 紙上：凍結規格是日曆年化；ia 口徑是解封後換算
pp = A['paper_ia']
out['paper_calendar_vs_official'] = {'C0_cal': pp['C0|normal|freeze']['ann_calendar'], 'C0_off': 28.28,
    'C0_diff_cal': pp['C0|normal|freeze']['ann_calendar'] - 28.28,
    'opt_cal': pp['ALLPIT_REAL_opt|normal|cons_main']['ann_calendar'], 'opt_diff_cal': pp['ALLPIT_REAL_opt|normal|cons_main']['ann_calendar'] - 22.22}
# 2) 修正版 vs 原引擎逐年差：2026 單年集中
yd = A['vs_orig_engine']['fixed']['yearly_median_diff_pp']
out['fixed_vs_orig_2026'] = {k: round(v['2026'], 3) for k, v in yd.items()}
out['fixed_vs_orig_other_years_maxabs'] = {k: round(max(abs(x) for y, x in v.items() if y != '2026'), 3) for k, v in yd.items()}
out['fixed_per_j_maxabs_ann'] = {f"{r['variant']}|{r['cell']}": round(r['ann_diff_maxabs'], 3) for r in A['vs_orig_engine']['fixed']['summary']}
# 3) 面板：GOOD 日上成交值缺值（G1 ffill(limit=3) 會補）逐年頻率
C = rpq(D7 + 'close_raw.parquet'); V = rpq(D7 + 'value.parquet')[C.columns]
TR = C.notna(); good = TR.sum(1) >= 300
Vg = V[good]; Cg = C[good]
# 在 GOOD 日缺值、但前後 5 個 GOOD 日內有值（零星缺口）
na = Vg.isna(); prev = Vg.ffill(limit=3).notna()
bridged = (na & prev)
yr = Vg.index.year
out['good_days_per_year'] = good.groupby(good.index.year).sum().astype(int).to_dict()
out['bridged_cells_per_goodday_by_year'] = (bridged.sum(1).groupby(yr).mean()).round(2).to_dict()
out['last_panel_date'] = str(C.index[-1].date())
# 4) P4 邊際：時點類 2.39 vs 門檻 3；成本類 14.37 vs 12
st = A['P4_ladder']['steps']
out['P4'] = {k: round(v['d_ann_y19'], 3) for k, v in st.items()}
out['P4_full_period'] = {k: round(v['d_ann_full'], 3) for k, v in st.items()}
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))

# 5) FIX5（嚴格）vs MIMIC（原讀法）逐 j、逐窗：2019 起差異落在哪些窗
R = json.load(open(os.path.join(H, 'results.json')))
res5 = {}
for cell in ['E|REAL_carry|cons_main|N100', 'E|REAL_opt|cons_main|N100', 'E|REAL_p90|cons_main|N100', 'E|REAL_opt|cons_main|N1000']:
    F = R['RESULTS_FIX5']['cells'].get(cell); M = R['RESULTS_MIMIC']['cells'].get(cell)
    if not F or not M: continue
    dj = {j: round(100 * (F['per_j'][j]['exc_y19'] - M['per_j'][j]['exc_y19']), 3) for j in F['per_j']}
    # 逐窗差（月窗超額），最後 24 窗 vs 其餘
    tail = []; head = []
    for j in F['win']:
        a = np.array(F['win'][j]['ex']); b = np.array(M['win'][j]['ex']); n = min(len(a), len(b))
        d = (a[:n] - b[:n]) * 100
        tail.append(float(np.sum(np.abs(d[-10:])))); head.append(float(np.sum(np.abs(d[:-10]))))
    res5[cell] = dict(y19_diff_per_j_pp=dj, med=float(np.median(list(dj.values()))), min=min(dj.values()), max=max(dj.values()),
                      sumabs_last10win_med=float(np.median(tail)), sumabs_other_win_med=float(np.median(head)),
                      worst_med_fix=100*F['sum']['exc_y19_med'], worst_med_mimic=100*M['sum']['exc_y19_med'])
print(json.dumps(res5, ensure_ascii=False, indent=1))
# 6) t 值：月窗超額自相關（重疊／序列相關會讓 ddof1 t 偏高或偏低）
C5 = R['RESULTS_FIX5']['cells']['E|REAL_p90|cons_main|N100']
ac = []
for j in C5['win']:
    x = np.array(C5['win'][j]['ex']); x = x - x.mean(); ac.append(float(np.sum(x[1:] * x[:-1]) / np.sum(x * x)))
print('AR1 of window excess (p90 N100, per j) median', np.median(ac), 'range', min(ac), max(ac))
# 13 j 間相關：中位數 j 不是獨立樣本
X = np.array([np.array(C5['win'][j]['ex'])[:240] for j in C5['win']])
cc = np.corrcoef(X); print('mean pairwise corr across j of window excess', float((cc.sum() - len(cc)) / (len(cc) ** 2 - len(cc))))
