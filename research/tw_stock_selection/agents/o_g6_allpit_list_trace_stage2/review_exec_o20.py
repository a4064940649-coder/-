# -*- coding: utf-8 -*-
"""反方審查（成本與可成交）o20：只讀研究員輸出與共用面板，不改其檔案、不 import 研究員程式。python3 -I -B review_exec_o20.py
1) M1->M2 多出的 6～7 槽：成交日 e 開盤是否漲停（買不到）、處置期何時結束、e 日成交值（1,000 萬容量）、該槽在窗內的報酬
2) 3057 停牌恢復後賣出日：是否跌停鎖死（賣不掉）
3) 年化口徑：PREREG 凍結主口徑為日曆年化，標題 22.06／22.21 是 ia 口徑
4) 周轉：p2 的成本兩平點只用名單換股數／10，忽略等權再平衡；以面板重算含再平衡的單邊周轉與兩平點
"""
import json, numpy as np, pandas as pd
D = '/home/user/-/research/tw_stock_selection/agents/o_g6_allpit_list_trace_stage2/'
PN = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/'
C = pd.read_parquet(PN + 'close_raw.parquet'); O = pd.read_parquet(PN + 'open.parquet'); H = pd.read_parquet(PN + 'high.parquet')
R = pd.read_parquet(PN + 'ret.parquet'); F = pd.read_parquet(PN + 'flag.parquet'); VAL = pd.read_parquet(PN + 'value.parquet')
IDX = C.index
U = pd.read_csv('/home/user/-/research/tw_stock_selection/agents/o_disposal_list_pit_audit_pkl18/data/universe_announce.csv',
                dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
PREVC = C.ffill().shift(1)
def tick(p): return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)
def lim_px(ref, lim, up=True):
    raw = ref * (1 + lim if up else 1 - lim); t = tick(raw)
    return np.floor(raw / t + 1e-9) * t if up else np.ceil(raw / t - 1e-9) * t
def ref_of(t, c):
    r = R.iat[t, R.columns.get_loc(c)]; cl = C.iat[t, C.columns.get_loc(c)]
    if F.iat[t, F.columns.get_loc(c)] == 1 or not np.isfinite(r): return PREVC.iat[t, PREVC.columns.get_loc(c)]
    return cl / (1 + r)
def lim_of(d): return 0.07 if d < pd.Timestamp('2015-06-01') else 0.10
out = {}

print('== 1) M1->M2 多出的槽')
P2 = json.load(open(D + 'out/p2_paper_compare.json'))
rows = []
for v in ['ALLPIT_REAL_opt', 'ALLPIT_REAL_carry']:
    for d, gone, new in P2['list_diffs'][f'{v}|M1->M2']['days']:
        d = pd.Timestamp(d); t = IDX.get_loc(d); e = IDX[t + 1]
        for c in new:
            if c not in C.columns: rows.append(dict(v=v, d=str(d.date()), code=c, note='不在面板')); continue
            te = t + 1; ref = ref_of(te, c); op = O.at[e, c]; hi = H.at[e, c]; cl = C.at[e, c]
            lu = lim_px(ref, lim_of(e)); ld = lim_px(ref, lim_of(e), up=False)
            dsp = U[(U.code == c) & (U.start <= d) & (U.end >= d - pd.Timedelta(days=40))][['announce', 'start', 'end', 'measure']]
            # 下一個訊號日（下月第一個日≥11 交易日）近似：找 k_paper 名單下一列
            rows.append(dict(v=v, d=str(d.date()), e=str(e.date()), code=c, replaced=gone, prevclose_ref=round(float(ref), 2), open=float(op),
                             limit_up=float(lu), open_at_limit_up=bool(op >= lu - 1e-6), high=float(hi), close=float(cl),
                             open_gap_pct=round((op / ref - 1) * 100, 2), close_vs_open_pct=round((cl / op - 1) * 100, 2),
                             value_e_mn=round(float(VAL.at[e, c]) / 1e6, 1),
                             value_med20_mn=round(float(VAL[c].iloc[max(0, t - 19):t + 1].median()) / 1e6, 1),
                             disposition=[(str(a.date()), str(s.date()), str(en.date()), m) for a, s, en, m in dsp.itertuples(index=False)]))
for r in rows: print(r)
out['m1_m2_slots'] = rows

# 該槽在窗內的貢獻：M2-M1 逐窗差
Z = np.load(D + 'out/k_paper_windows.npz')
per = Z['M2|ALLPIT_REAL_opt|normal|per']
for c in ['normal', 'stress']:
    dd = Z[f'M2|ALLPIT_REAL_opt|{c}|r'] - Z[f'M1|ALLPIT_REAL_opt|{c}|r']
    nz = np.where(np.abs(dd) > 1e-12)[0]
    out[f'opt_{c}_window_diff'] = {str(per[i]): round(float(dd[i]) * 100, 3) for i in nz}
    print(c, '逐窗 M2-M1（pp，該窗報酬差）', out[f'opt_{c}_window_diff'])

print('\n== 2) 3057 恢復交易後（複製員讀法賣出日）漲跌停檢查')
c = '3057'; r3057 = []
for ds in ['2013-10-01', '2013-10-02', '2013-10-03', '2013-10-04', '2013-10-07', '2013-10-08', '2013-10-22']:
    e = pd.Timestamp(ds); te = IDX.get_loc(e); ref = ref_of(te, c)
    r3057.append(dict(d=ds, ref=round(float(ref), 2), open=float(O.at[e, c]), high=float(H.at[e, c]), close=float(C.at[e, c]),
                      lu=float(lim_px(ref, 0.07)), ld=float(lim_px(ref, 0.07, up=False)), value_mn=round(float(VAL.at[e, c]) / 1e6, 2)))
for r in r3057: print(r)
out['c3057'] = r3057

print('\n== 3) 年化口徑')
S = P2['stats']
for m in ['M0', 'M1', 'M2']:
    k = f'{m}|ALLPIT_REAL_opt|normal'; print(m, 'opt ia', round(S[k]['ann_ia'], 3), 'cal（凍結主口徑）', round(S[k]['ann_cal'], 3), 'exc_cal', round(S[k]['exc_cal'], 3))
out['annualization'] = {m: dict(ia=S[f'{m}|ALLPIT_REAL_opt|normal']['ann_ia'], cal=S[f'{m}|ALLPIT_REAL_opt|normal']['ann_cal']) for m in ['M0', 'M1', 'M2']}

print('\n== 4) 含等權再平衡的周轉（M2 opt）')
L = pd.read_csv(D + 'out/k_paper_lists.csv.gz', dtype=str)
Lv = L[(L['mask'] == 'M2') & (L.variant == 'ALLPIT_REAL_opt')].sort_values('signal')
sig = [pd.Timestamp(s) for s in Lv.signal]; picks = [s.split(',') for s in Lv.picks]
GROW = (1 + R.fillna(0.0)).where(C.notna(), 1.0)
tov_list, tov_full = [], []
for i in range(1, len(sig)):
    t0 = IDX.get_loc(sig[i - 1]) + 1; t1 = IDX.get_loc(sig[i]) + 1   # e_{i-1} 開盤 → e_i 開盤（以收盤報酬近似）
    prev = [p for p in picks[i - 1] if p in GROW.columns]
    g = GROW[prev].iloc[t0 + 1:t1 + 1].prod().to_numpy()   # 近似 e 日開→下個 e 日（忽略開收差）
    w = g / g.sum(); new = set(picks[i])
    sell = sum(wi for wi, p in zip(w, prev) if p not in new)
    keep = [(wi, p) for wi, p in zip(w, prev) if p in new]
    over = sum(max(wi - 0.1, 0) for wi, p in keep)
    tov_list.append(len(set(prev) - new) / 10); tov_full.append(sell + over)
tl, tf = float(np.mean(tov_list)), float(np.mean(tov_full))
print('名單換股/10', round(tl, 4), '含再平衡單邊', round(tf, 4), '比值', round(tf / tl, 3))
# 兩平點：用含再平衡周轉重算（同 p2 方法，改 tau）
r = Z['M2|ALLPIT_REAL_opt|normal|r']; r50 = Z['M2|ALLPIT_REAL_opt|normal|r50']; n = len(r)
tau = np.array([1.0] + tov_full)[:n]; g5 = np.prod(1 + r50) ** (12 / n)
lo, hi = 0.0, 0.05
for _ in range(60):
    mid = (lo + hi) / 2
    if np.prod(1 + r - 2 * mid * tau) ** (12 / n) - g5 > 0: lo = mid
    else: hi = mid
print('兩平額外單邊成本（含再平衡）', round(lo * 100, 3), '% （研究員 0.417%）')
# 每年已內含的成本拖累
fee = 0.001425 * 0.6; buy = fee + 0.001; sell_ = fee + 0.003 + 0.001
print('正常成本每年拖累約', round(tf * 12 * (buy + sell_) * 100, 2), '%；其中證交稅', round(tf * 12 * 0.003 * 100, 2), '%')
out['turnover'] = dict(list_only=tl, with_rebal=tf, breakeven_with_rebal_pct=lo * 100,
                       annual_cost_drag_pct=tf * 12 * (buy + sell_) * 100, annual_tax_pct=tf * 12 * 0.003 * 100)
json.dump(out, open('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/review_exec_o20.json', 'w'),
          ensure_ascii=False, indent=1, default=str)
