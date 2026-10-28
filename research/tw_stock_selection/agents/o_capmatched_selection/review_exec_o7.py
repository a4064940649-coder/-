#!/usr/bin/env python3
"""反方審查（成本與可成交）o7 市值匹配選股。只讀研究員檔案，不寫入任何研究員檔案；輸出到 stdout。
python3 -I -B review_exec_o7.py"""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import cmlib as L, cmstrat as T, cmexec as X

FROZEN = 'F1_K5_N20_sqrt'
END = pd.Timestamp('2026-10-12')
G = L.grow_exfix()
DAYS = [d for d in L.official_days('2006-04-01', END) if d < END] + [END]
DAYS19 = [d for d in DAYS if d >= pd.Timestamp('2019-01-01')]
PER = {'SEL': ('2019-01-01', '2022-12-31'), 'VAL': ('2023-01-01', '2026-10-12'), 'POST2019': ('2019-01-01', '2026-10-12')}
ODD_INTRA = pd.Timestamp('2020-10-26')
def P(*a): print(*a, flush=True)
def st(nav, days=DAYS19):
    return {k: (round(r['mean_pp'], 3), round(r['t'], 2), round(r['IR'], 2)) for k in PER for r in [L.rel_stats(nav, L.E50, days, *PER[k])]}

t0 = L.IDX.get_loc(DAYS19[0]); t1 = L.IDX.get_loc(END)
plan, info = T.build_plan(FROZEN, DAYS19)
I = pd.DataFrame(info)

# ---------- A. 成本分解：零成本／一般／壓力／全額手續費＋滑價 ----------
P('=== A. 成本分解（2019 年起重跑同一計畫）===')
for nm, c in [('zero', 'zero'), ('normal', 'normal'), ('stress', 'stress'), ('fullfee+0.1%', ('fullfee', 0.001)),
              ('6折+0.3%滑價', ('slip', 0.003)), ('6折+0.5%滑價', ('slip', 0.005))]:
    nav, lg = L.simulate_w(plan, *L.rates(c), t0, t1, G=G)
    P(f'  {nm:14s}', st(nav), 'turnover', round(lg['turnover'], 4))
b, s = L.rates('normal'); P('  normal 單邊：買', round(b * 100, 4), '% 賣', round(s * 100, 4), '%；stress 0.5%/0.5% → 來回只多', round((1.0 - (b + s) * 100), 3), '%')

# ---------- B. 每期換手與交易筆數（以前一日收盤權重為起點）----------
P('=== B. 換手與交易筆數 ===')
navN, lgN = L.simulate_w(plan, *L.rates('normal'), t0, t1, G=G, record=True)
W = lgN['W']
rows = []
for d in DAYS19[:-1]:
    cs = T.cross_section(d); e = cs['e']
    if e not in plan: continue
    js, ws = plan[e]; tgt = np.zeros(L.N); tgt[js] = ws
    prev = W.get(e - 1, np.zeros(L.N))
    dw = tgt - prev
    rows.append(dict(d=d, e=e, oneway=0.5 * np.abs(dw).sum(), ntr=int((np.abs(dw) > 1e-6).sum()), nnew=int(((tgt > 0) & (prev <= 1e-9)).sum()),
                     ndrop=int(((tgt <= 1e-9) & (prev > 1e-9)).sum()), nsmall=int(((np.abs(dw) > 1e-6) & (np.abs(dw) < 0.001)).sum()), nhold=int((tgt > 0).sum()),
                     satw=float(tgt[(tgt > 0) & (cs['R'] == 0)].sum()), satmax=float(tgt[(tgt > 0) & (cs['R'] == 0)].max()) if ((tgt > 0) & (cs['R'] == 0)).any() else 0.0,
                     w2330=float(tgt[L.J2330])))
B = pd.DataFrame(rows)
for k, (a, bb) in PER.items():
    z = B[(B.d >= a) & (B.d <= bb)]
    P(f'  {k}: 單邊換手 {z.oneway.mean():.3f}（中位 {z.oneway.median():.3f}）；每月成交檔數 {z.ntr.mean():.1f}，其中 |Δw|<0.1% 的微調 {z.nsmall.mean():.1f}；'
      f'新進 {z.nnew.mean():.1f}／剔除 {z.ndrop.mean():.1f}；持股 {z.nhold.mean():.1f}；非成分衛星權重 {z.satw.mean():.3f}、單檔最大 {z.satmax.mean():.3f}；台積電 {z.w2330.mean():.3f}')

# ---------- C. 漲停買不到（開盤漲停跳過遞補）、跌停賣不掉 ----------
P('=== C. 漲跌停 ===')
for k, (a, bb) in PER.items():
    z = I[(pd.to_datetime(I.d) >= a) & (pd.to_datetime(I.d) <= bb)]
    P(f'  {k}: 開盤漲停被跳過的新買進檔數 合計 {int(z.n_blocked.sum())}，發生月數 {int((z.n_blocked > 0).sum())}/{len(z)}')
def limit_down_open(t):
    lim = 0.07 if L.IDX[t] < pd.Timestamp('2015-06-01') else 0.10
    with np.errstate(invalid='ignore', divide='ignore'):
        ref = np.where(np.nan_to_num(L.Fn[t], nan=0) == 1, L.PREVC[t], L.Cn[t] / (1 + L.Rn[t]))
        ref = np.where(np.isfinite(ref), ref, L.PREVC[t])
        raw = ref * (1 - lim); tk = L.tick(raw)
        ld = np.ceil(raw / tk - 1e-9) * tk
    return np.isfinite(L.On[t]) & np.isfinite(ld) & (L.On[t] <= ld + 1e-6)
nld = 0; wld = 0.0; nsell = 0; lu_buy_held = 0
for r in rows:
    e = r['e']; js, ws = plan[e]; tgt = np.zeros(L.N); tgt[js] = ws; prev = W.get(e - 1, np.zeros(L.N))
    sell = (prev - tgt) > 1e-4; nsell += int(sell.sum())
    ld = limit_down_open(e) & sell
    nld += int(ld.sum()); wld += float((prev - tgt)[ld].sum())
    lu = L.limit_up_open(e) & ((tgt - prev) > 1e-4)
    lu_buy_held += int(lu.sum())
P(f'  2019 年起：賣出（Δw<−0.01%）共 {nsell} 筆，其中開盤跌停 {nld} 筆、權重合計 {wld:.4f}（引擎假設照開盤賣出）')
P(f'  2019 年起：加碼（Δw>+0.01%）時開盤漲停 {lu_buy_held} 筆（已持有者照開盤買進，引擎不擋）')
# 遞補規則需要「看到開盤價」：改成盤前就送單、開盤漲停的新買進買不到（權重按比例分給其他持股），不遞補
_orig = L.limit_up_open
L.limit_up_open = lambda t: np.zeros(L.N, bool)
T._CS.clear()
plan_nb, info_nb = T.build_plan(FROZEN, DAYS19)
L.limit_up_open = _orig
plan_ns = {}
nskip = 0
held = set()
for d in DAYS19:
    cs = T.cross_section(d); e = cs['e']
    js, ws = plan_nb[e]
    blk = L.limit_up_open(e)[js] & ~np.isin(js, list(held))
    nskip += int(blk.sum())
    plan_ns[e] = (js[~blk], ws[~blk] / ws[~blk].sum())
    held = set(js[~blk].tolist())
nav_nb, _ = L.simulate_w(plan_nb, *L.rates('normal'), t0, t1, G=G)
nav_ns, _ = L.simulate_w(plan_ns, *L.rates('normal'), t0, t1, G=G)
P('  凍結版（開盤漲停跳過＋遞補）', st(navN))
P('  不擋漲停（照開盤價買進漲停股，樂觀）', st(nav_nb))
P(f'  盤前送單、不遞補（漲停新買進買不到 {nskip} 筆，權重給其他持股）', st(nav_ns))

# ---------- D. 容量：開盤集合競價參與率（o5 OA 模型）----------
P('=== D. 容量（買賣金額 ÷ 估計開盤集合競價成交值）===')
navg = navN / navN.iloc[0]
for tier in (1e7, 3e7, 1e8):
    parts = []
    for r in rows:
        e = r['e']; js, ws = plan[e]; tgt = np.zeros(L.N); tgt[js] = ws; prev = W.get(e - 1, np.zeros(L.N))
        A = tier * X.cpi_scale(L.IDX[e]) * float(navg.iloc[e - 1 - t0]) if e - 1 - t0 >= 0 else tier
        dw = tgt - prev; k = np.where(np.abs(dw) > 1e-6)[0]
        Q = np.abs(dw[k]) * A
        pc = L.PREVC[e, k]; rk = X.vrank(e - 1)[k]; sg = X.SIG[e - 1, k]
        gap = np.where(np.isfinite(L.On[e, k]) & (pc > 0), L.On[e, k] / pc - 1, 0.0)
        pick = tgt[k] > 0; drop = (prev[k] > 0) & ~pick
        x = np.vstack([np.ones(len(k)), np.log(np.clip(np.nan_to_num(rk, nan=1e4), 1, None)), np.log(pc), np.log(np.clip(np.nan_to_num(sg, nan=0.02), 1e-4, None)),
                       np.abs(gap), pick.astype(float), drop.astype(float)])
        oa = np.nan_to_num(X.V60[e - 1, k], nan=1e7) * X.KAPPA * np.exp(X.B @ x)
        v60 = X.V60[e - 1, k]
        for q, o, v, j in zip(Q, oa, v60, k):
            parts.append((L.IDX[e], L.COLS[j], q, q / o, q / v if np.isfinite(v) and v > 0 else np.nan))
    D = pd.DataFrame(parts, columns=['e', 'code', 'Q', 'p_oa', 'p_v60'])
    for k_, (a, bb) in PER.items():
        z = D[(D.e >= a) & (D.e <= bb)]
        P(f'  {tier/1e4:.0f}萬 {k_}: 筆數 {len(z)}；開盤參與率 p50 {z.p_oa.median():.3f} p95 {z.p_oa.quantile(.95):.3f} max {z.p_oa.max():.2f}；>20% 筆數 {(z.p_oa > 0.2).sum()}；'
          f'占 60 日均量 p95 {z.p_v60.quantile(.95):.4f} max {z.p_v60.max():.3f}')

# ---------- E. 零股：2020-10-26 盤中零股開放前，小資金的零股成交無法在開盤成交 ----------
P('=== E. 零股占成交值（GROW；股數連續）===')
for tier in (3e5, 1e6, 3e6):
    out = []
    for r in rows:
        e = r['e']; js, ws = plan[e]; tgt = np.zeros(L.N); tgt[js] = ws; prev = W.get(e - 1, np.zeros(L.N))
        A = tier * X.cpi_scale(L.IDX[e]) * float(navg.iloc[e - 1 - t0])
        dw = tgt - prev; k = np.where(np.abs(dw) > 1e-6)[0]
        Q = np.abs(dw[k]) * A; lot = 1000 * L.PREVC[e, k]
        odd = np.where(Q < lot, Q, Q % lot)
        hold = tgt * A; lotall = 1000 * L.PREVC[e]
        nsub = int(((tgt > 0) & (hold < lotall)).sum())
        out.append((L.IDX[e], odd.sum() / Q.sum(), nsub, int((tgt > 0).sum()), int(((Q < 1000)).sum()), len(k)))
    Z = pd.DataFrame(out, columns=['e', 'oddshare', 'nsub', 'nhold', 'ntiny', 'ntr'])
    pre = Z[Z.e < ODD_INTRA]; post = Z[Z.e >= ODD_INTRA]
    P(f'  {tier/1e4:.0f}萬：盤中零股開放前 {len(pre)} 個月零股占成交值 {pre.oddshare.mean():.2f}、未滿一張的持股 {pre.nsub.mean():.1f}/{pre.nhold.mean():.1f} 檔；'
      f'開放後 {post.oddshare.mean():.2f}、{post.nsub.mean():.1f}/{post.nhold.mean():.1f} 檔；每月成交 < 1,000 元的筆數 {Z.ntiny.mean():.1f}/{Z.ntr.mean():.1f}')

# ---------- F. 資料時點：營收、src3 開盤價 ----------
P('=== F. 資料時點 ===')
gaps = []
for d in DAYS19[:-1]:
    m = pd.Period(d, 'M'); dl = L.deadline(m); u = L.unavail(d)
    gaps.append((d, dl, int(np.busday_count(dl.date(), d.date())), bool(u.all())))
Gp = pd.DataFrame(gaps, columns=['d', 'deadline', 'bdays_after', 'all_unavail'])
P('  訊號日與營收期限日的營業日差分佈（0＝期限日當天→改用舊營收）:', Gp.bdays_after.value_counts().sort_index().to_dict())
P('  訊號日＝期限日次一營業日（期限日晚上才公告完的營收當天收盤就用）月數:', int((Gp.bdays_after == 1).sum()), '/', len(Gp))
srcv = pd.Series(L.SRCn[(L.IDX >= '2024-02-15') & (L.IDX <= '2025-09-23')].ravel()).value_counts().head(5)
P('  src3 期間來源代碼分佈:', srcv.to_dict())
m3 = (L.IDX >= '2024-02-15') & (L.IDX <= '2025-09-23')
for d in DAYS19:
    e = L.IDX.get_loc(d) + 1
    if not m3[min(e, len(L.IDX) - 1)]: continue
cnt = 0; eqc = 0; tot = 0
for r in rows:
    e = r['e']
    if not ((L.IDX[e] >= pd.Timestamp('2024-02-15')) & (L.IDX[e] <= pd.Timestamp('2025-09-23'))): continue
    js, ws = plan[e]
    o = L.On[e, js]; c = L.Cn[e, js]; pc = L.PREVC[e, js]
    tot += len(js); cnt += int(np.isfinite(o).sum()); eqc += int((np.isfinite(o) & ((o == c) | (o == pc))).sum())
P(f'  src3 期間成交日持股：開盤價有值 {cnt}/{tot}；開盤＝收盤或＝前收 {eqc}')
# 2024-01 面板缺口：換股日
P('  2023-12～2024-03 換股日:', [str(d.date()) for d in DAYS19 if pd.Timestamp('2023-12-01') <= d <= pd.Timestamp('2024-03-31')])

# ---------- G. 不利方向合計：j 中位數 × 可執行成本 ----------
R = json.loads((L.HERE / 'results_oos.json').read_text(encoding='utf-8'))
js_ = R['jscan']['summary']; ex = R['executable']; vr = R['versions'][FROZEN]['normal']['vs_0050']
P('=== G. 不利方向合計 ===')
for k in ('SEL', 'VAL', 'POST2019'):
    base = vr[k]['mean_pp']; jm = js_[f'F|{k}']['mean_pp_median']
    for tn in ('100萬', '300萬', '1000萬'):
        e5 = ex[f'F|{tn}|FIX|o5'][k]['mean_pp']
        P(f'  {k} {tn}: 官方日一般成本 {base:.3f} → 可執行(FIX,o5) {e5:.3f}；j 中位數 {jm:.3f}；兩者疊加約 {jm - (base - e5):.3f}pp／月')

# ---------- H. 對照：src3 以外期間，成交日持股開盤＝收盤或＝前收的比例 ----------
def eq_rate(a, bb):
    tot = eqc = 0
    for r in rows:
        e = r['e']
        if not (pd.Timestamp(a) <= L.IDX[e] <= pd.Timestamp(bb)): continue
        js, ws = plan[e]; o = L.On[e, js]; c = L.Cn[e, js]; pc = L.PREVC[e, js]
        tot += len(js); eqc += int((np.isfinite(o) & ((o == c) | (o == pc))).sum())
    return eqc, tot
P('=== H. 開盤＝收盤或前收比例 ===')
for a, bb in [('2019-01-01', '2024-01-31'), ('2024-02-15', '2025-09-23'), ('2025-09-24', '2026-10-12')]:
    P(' ', a, bb, eq_rate(a, bb))
