#!/usr/bin/env python3
"""反方審查（資料與統計）o7 市值匹配選股。只讀研究員檔案，不寫入任何研究員檔案；輸出到 stdout。
python3 -I -B review_stat_o7.py"""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import cmlib as L, cmstrat as T

FROZEN = 'F1_K5_N20_sqrt'
END = pd.Timestamp('2026-10-12')
G = L.grow_exfix()
DAYS = [d for d in L.official_days('2006-04-01', END) if d < END] + [END]
DAYS19 = [d for d in DAYS if d >= pd.Timestamp('2019-01-01')]
PER = {'SEL': ('2019-01-01', '2022-12-31'), 'VAL': ('2023-01-01', '2026-10-12'), 'POST2019': ('2019-01-01', '2026-10-12')}
NAVS = pd.read_parquet(L.HERE / 'navs_oos.parquet')
FN = NAVS[f'{FROZEN}|normal']; REP = NAVS['REP']
E = L.E50.reindex(FN.index).ffill()
def P(*a): print(*a, flush=True)

# ---------- A. 存活者：面板在 2019-05-16 截斷的股票 ----------
P('=== A. 面板截斷（2019-05-16 之後無資料、實際 2019-05～2024 才下市）===')
last = pd.Series(L.IDX[np.clip(L.LAST_TRADE, 0, None)], index=L.COLS)
trunc = last[last == pd.Timestamp('2019-05-16')].index.tolist()
P('截斷檔數', len(trunc))
d0 = L.IDX[L.IDX <= pd.Timestamp('2019-05-15')][-1]
cs0 = T.cross_section(L.official_days('2019-05-01', '2019-05-31')[0])
rk = {c: int(cs0['rank'][L.CI[c]]) for c in trunc}
P('2019-05 換股日市值名次（≤300）:', {c: r for c, r in sorted(rk.items(), key=lambda x: x[1]) if r <= 300})
P('其中 2019-05 合格衛星候選:', [c for c in trunc if cs0['elig'][L.CI[c]]])
dl = pd.read_csv(L.RAW / 'agents' / 'delist_official.csv', dtype=str)
dl = dl[dl.code.isin(trunc)][['code', 'name', 'date']]
P(dl.sort_values('date').to_string(index=False))

# ---------- B. 未來股數：2019 年的股數＝2025 錨點的比例 ----------
P('\n=== B. 未來股數（前視）===')
S = L.shares_est()
t19 = L.IDX.get_loc(L.IDX[L.IDX >= pd.Timestamp('2019-06-03')][0]); t21 = L.IDX.get_loc(L.IDX[L.IDX >= pd.Timestamp('2021-10-07')][0])
ta = L.IDX.get_loc(L.IDX[L.IDX >= pd.Timestamp('2025-10-07')][0])
same = np.isfinite(S[t19]) & (S[t19] == S[t21])
P('2019-06 與 2021-10 股數完全相同的比例（有股數者）', round(float(same.sum() / np.isfinite(S[t19]).sum()), 3))
# 衛星持股在挑選期的市值名次分布（邊界敏感度）
plan_sel, info_sel = T.build_plan(FROZEN, DAYS19)
rows = []
for d in DAYS19[:-1]:
    cs = T.cross_section(d); e0 = cs['e']
    if e0 not in plan_sel: continue
    js, ws = plan_sel[e0]
    i = DAYS.index(d); e1 = L.IDX.get_loc(DAYS[i + 1]) + 1 if DAYS[i + 1] < END else L.IDX.get_loc(END)
    f = L.window_factors_all(e0, min(e1, L.IDX.get_loc(END)), G)
    rr = float(np.nansum(cs['R'] * (np.nan_to_num(f, nan=1.0) - 1)))
    w = np.zeros(L.N); w[js] = ws; a_ = w - cs['R']
    for j in np.where(np.abs(a_) > 1e-9)[0]:
        rows.append((d, L.COLS[j], int(cs['rank'][j]), a_[j], a_[j] * ((f[j] if np.isfinite(f[j]) else 1.0) - 1 - rr), bool(L.TRADED[min(e1, len(L.IDX) - 1), j])))
C = pd.DataFrame(rows, columns=['d', 'code', 'rank', 'act', 'c', 'tr_end'])
for k, (a, b) in PER.items():
    z = C[(C.d >= a) & (C.d <= b)]; nw = z.d.nunique()
    bk = pd.cut(z['rank'], [0, 50, 100, 150, 200, 250, 300, 10 ** 10], labels=['1-50', '51-100', '101-150', '151-200', '201-250', '251-300', '>300'])
    g = z.groupby(bk, observed=False)
    P(k, '主動貢獻 pp/月 依市值名次（正主動=衛星，負=被替換成分）:', (g.c.sum() / nw * 100).round(3).to_dict())
    P(k, '  其中只看衛星（主動>0）:', (z[z.act > 0].groupby(bk[z.act > 0], observed=False).c.sum() / nw * 100).round(3).to_dict(), ' 合計', round(float(z[z.act > 0].c.sum() / nw * 100), 3),
      ' 被替換（主動<0）合計', round(float(z[z.act < 0].c.sum() / nw * 100), 3))

# ---------- C. β 與 α（對 0050、對複製）----------
P('\n=== C. 對 0050 的 β／α（訊號窗月報酬）===')
def reg(y, x):
    d = pd.concat([y, x], axis=1).dropna(); A = np.column_stack([np.ones(len(d)), d.iloc[:, 1]])
    b, *_ = np.linalg.lstsq(A, d.iloc[:, 0], rcond=None); r = d.iloc[:, 0] - A @ b
    s2 = r @ r / (len(d) - 2); se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
    return dict(alpha_pp=round(b[0] * 100, 3), t_alpha=round(b[0] / se[0], 2), beta=round(b[1], 3), t_beta1=round((b[1] - 1) / se[1], 2), n=len(d))
rF = L.win_ret(FN, DAYS); r5 = L.win_ret(E, DAYS); rR = L.win_ret(REP, DAYS)
for k, (a, b) in PER.items():
    m = (rF.index >= a) & (rF.index <= b)
    P(k, 'F~0050', reg(rF[m], r5[m]), ' REP~0050', reg(rR[m], r5[m]))
    up = m & (r5 > 0); dn = m & (r5 <= 0)
    P(k, '  0050 上漲月超額 pp', round(float((rF - r5)[up].mean() * 100), 3), 'n', int(up.sum()), '｜下跌月', round(float((rF - r5)[dn].mean() * 100), 3), 'n', int(dn.sum()))

# ---------- D. 逐年、去一年、NW t、自助法 ----------
P('\n=== D. 逐年、去一年、HAC ===')
x = (rF - r5).dropna(); x19 = x[(x.index >= '2019-01-01') & (x.index <= '2026-10-12')]
yr = x19.groupby(x19.index.year)
P('逐年月超額均值 pp:', (yr.mean() * 100).round(3).to_dict(), ' 加總 pp:', (yr.sum() * 100).round(2).to_dict())
for y in sorted(set(x19.index.year)):
    z = x19[x19.index.year != y]; P(f'  去掉 {y}: mean {z.mean() * 100:.3f}pp t {L.tstat(z):.2f}')
z = x19[~x19.index.year.isin([2020, 2021])]; P(f'  去掉 2020+2021: mean {z.mean() * 100:.3f} t {L.tstat(z):.2f} n {len(z)}')
def nw_t(s, L_=3):
    s = np.asarray(s, float); n = len(s); u = s - s.mean(); g0 = u @ u / n; v = g0
    for l in range(1, L_ + 1): v += 2 * (1 - l / (L_ + 1)) * (u[l:] @ u[:-l]) / n
    return s.mean() / math.sqrt(v / n)
for k, (a, b) in PER.items():
    z = x[(x.index >= a) & (x.index <= b)]
    P(k, f'OLS t {L.tstat(z):.2f}  NW(3) t {nw_t(z):.2f}  中位數 {z.median() * 100:.3f}pp  勝率 {(z > 0).mean():.2f}  偏態 {z.skew():.2f}')
rng = np.random.default_rng(7)
def block_boot_mean(z, B=5000, bl=6):
    z = np.asarray(z); n = len(z); out = []
    for _ in range(B):
        idx = np.concatenate([np.arange(s, s + bl) % n for s in rng.integers(0, n, n // bl + 1)])[:n]
        out.append(z[idx].mean())
    return np.percentile(out, [2.5, 5, 50]) * 100
for k, (a, b) in PER.items():
    z = x[(x.index >= a) & (x.index <= b)]; P(k, '區塊自助（6 月）均值 2.5/5/50 百分位 pp', np.round(block_boot_mean(z), 3))

# IR 比較：本題 vs 主策略（2019 年後）
M = pd.read_parquet(L.RAW / 'navs' / 'navs_v4.parquet')['AB_V4|normal']
rM = L.win_ret(M.reindex(FN.index).ffill(), DAYS)
xm = (rM - r5).dropna(); xm19 = xm[(xm.index >= '2019-01-01') & (xm.index <= '2026-10-12')]
j = x19.index.intersection(xm19.index); a1 = x19[j].to_numpy(); a2 = xm19[j].to_numpy()
def ir(v): return v.mean() * 12 / (v.std(ddof=1) * math.sqrt(12))
d_ir = []
for _ in range(5000):
    idx = np.concatenate([np.arange(s, s + 6) % len(j) for s in rng.integers(0, len(j), len(j) // 6 + 1)])[:len(j)]
    d_ir.append(ir(a1[idx]) - ir(a2[idx]))
P(f'IR 本題 {ir(a1):.2f} 主策略 {ir(a2):.2f} 差 {ir(a1) - ir(a2):.2f}；自助 P(差≤0)={np.mean(np.array(d_ir) <= 0):.3f}；90% 區間 {np.percentile(d_ir, [5, 95]).round(2)}')

# ---------- E. 安慰劑：同骨架、同合格池，分數換成隨機／反向／單一成分 ----------
P('\n=== E. 安慰劑（同結構，替換分數）===')
t0 = L.IDX.get_loc(DAYS19[0]); t1 = L.IDX.get_loc(END)
b_, s_ = L.rates('normal')
def run_with(score_fn):
    plan = {}; held = set()
    for d in DAYS19:
        cs = T.cross_section(d); c2 = dict(cs); c2['sc'] = score_fn(cs)
        js, ws, _, _ = T.target(c2, FROZEN, held); plan[cs['e']] = (js, ws); held = set(js.tolist())
    nav, _ = L.simulate_w(plan, b_, s_, t0, t1, G=G)
    return {k: L.rel_stats(nav, L.E50, DAYS19, a, b) for k, (a, b) in PER.items()}
def stat_line(r): return {k: (round(v['mean_pp'], 3), round(v['t'], 2)) for k, v in r.items()}
act = run_with(lambda cs: cs['sc'])
P('實際分數（2019 起重跑）', stat_line(act))
def pr(cs, x):
    return pd.Series(np.where(cs['elig'], x, np.nan)).rank(pct=True).to_numpy()
P('只用 52 週比', stat_line(run_with(lambda cs: pr(cs, cs['s52']))))
P('只用營收年增', stat_line(run_with(lambda cs: pr(cs, cs['y3']))))
P('反向分數（最低分 20 檔）', stat_line(run_with(lambda cs: np.where(cs['elig'], 1 - cs['sc'], np.nan))))
res = []
NSEED = 200
for sd in range(NSEED):
    rg = np.random.default_rng(1000 + sd)
    res.append(run_with(lambda cs: np.where(cs['elig'], rg.random(L.N), np.nan)))
for k in PER:
    m = np.array([r[k]['mean_pp'] for r in res]); tt = np.array([r[k]['t'] for r in res])
    P(k, f'隨機分數 {NSEED} 次：均值中位數 {np.median(m):.3f}pp（p5 {np.percentile(m, 5):.3f}, p95 {np.percentile(m, 95):.3f}），t 中位數 {np.median(tt):.2f}；'
         f'實際 {act[k]["mean_pp"]:.3f}pp 的排列 p＝{np.mean(m >= act[k]["mean_pp"]):.3f}；實際−隨機中位數＝{act[k]["mean_pp"] - np.median(m):.3f}pp')

# ---------- F. 版本挑選：驗證期 12 版的分布與挑選期排序相關 ----------
P('\n=== F. 版本挑選 ===')
R = json.loads((L.HERE / 'results_oos.json').read_text(encoding='utf-8'))
V = R['versions']
tab = pd.DataFrame({v: {'SEL_IR': V[v]['normal']['vs_0050']['SEL']['IR'], 'VAL_pp': V[v]['normal']['vs_0050']['VAL']['mean_pp'], 'VAL_t': V[v]['normal']['vs_0050']['VAL']['t'],
                        'VAL_IR': V[v]['normal']['vs_0050']['VAL']['IR']} for v in V}).T.sort_values('SEL_IR', ascending=False)
P(tab.round(3).to_string())
P('SEL IR 與 VAL IR 的 Spearman', round(tab.SEL_IR.corr(tab.VAL_IR, method='spearman'), 3))
P('j 掃描 POST2019 t:', {j: round(o['F|POST2019']['t'], 2) for j, o in R['jscan']['by_j'].items()})

# ---------- G. 實際分數 − 隨機分數平均 的時間序列 t（安慰劑的樣本路徑不確定性）----------
P('\n=== G. 實際 − 隨機平均（50 次）的月序列 t ===')
def nav_with(score_fn):
    plan = {}; held = set()
    for d in DAYS19:
        cs = T.cross_section(d); c2 = dict(cs); c2['sc'] = score_fn(cs)
        js, ws, _, _ = T.target(c2, FROZEN, held); plan[cs['e']] = (js, ws); held = set(js.tolist())
    return L.simulate_w(plan, b_, s_, t0, t1, G=G)[0]
nA = nav_with(lambda cs: cs['sc']); rA = L.win_ret(nA, DAYS19)
rr = []
for sd in range(50):
    rg = np.random.default_rng(5000 + sd); rr.append(L.win_ret(nav_with(lambda cs: np.where(cs['elig'], rg.random(L.N), np.nan)), DAYS19))
rRand = pd.concat(rr, axis=1).mean(axis=1)
r50 = L.win_ret(L.E50.reindex(nA.index).ffill(), DAYS19)
for k, (a, b) in PER.items():
    m = (rA.index >= a) & (rA.index <= b)
    d1 = (rA - rRand)[m]; d2 = (rRand - r50.reindex(rRand.index))[m]
    P(k, f'實際−隨機 {d1.mean() * 100:.3f}pp t {L.tstat(d1):.2f}｜隨機結構−0050 {d2.mean() * 100:.3f}pp t {L.tstat(d2):.2f}')
