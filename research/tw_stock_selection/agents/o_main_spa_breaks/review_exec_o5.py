#!/usr/bin/env python3
"""反方審查（成本與可成交）：o_main_spa_breaks。只讀研究員的檔案與 V4 引擎（只讀 import），結果只印在畫面，不寫任何檔案。
python3 -I -B review_exec_o5.py
1) 判定欄與「選股力 p 0.037」在壓力成本下：族群不變，把焦點換成 AB_V4|stress（同種子、同 bootstrap）。
2) 2019 後、保留期（含 2023-12-11 跨面板缺口那一窗）的壓力成本表現。
3) 周轉與成本拖累：零成本／一般／壓力三版的年化差。
4) 主策略持股的可成交性：換股日 20 日均成交值、10 檔各 100 萬（1,000 萬資金）時的參與率；分期間。
5) 五分位價差（Q5−Q1，零成本、不做漲停遞補）：Q5 的開盤漲停比例；排除開盤漲停後重算 2024-01 前後均值；
   改成可做多的 Q5（一般成本）−同池等權。
"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import spa_lib as S            # 只讀 import
V4D = H.parent / 'o_v4_revenue_patch_official'
MAIN = 'o_v4_revenue_patch_official/navs_v4.parquet::AB_V4|normal'
GRID_F = V4D / 'plan_V4.csv'

def t_(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return x.mean() / x.std(ddof=1) * math.sqrt(len(x))

def nw_t(x, lag=None):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    lag = lag or int(4 * (n / 100) ** (2 / 9)); e = x - x.mean(); s = e @ e / n
    for i in range(1, lag + 1): s += 2 * (1 - i / (lag + 1)) * (e[i:] @ e[:-i]) / n
    return x.mean() / math.sqrt(s / n)

def chow_mean(x, idx, date):
    a = x[idx < pd.Timestamp(date)]; b = x[idx >= pd.Timestamp(date)]
    d = b.mean() - a.mean(); se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return a.mean() * 100, b.mean() * 100, d / se

def main():
    D = pd.read_parquet(H / 'data/family_excess.parquet'); Dew = pd.read_parquet(H / 'data/family_excess_ew.parquet')
    MM = pd.read_csv(H / 'data/family_members.csv'); Bn = pd.read_parquet(H / 'data/bench.parquet')
    BI = pd.read_parquet(H / 'data/break_inputs.parquet')
    nv = pd.read_parquet(V4D / 'navs_v4.parquet')
    sys.path.insert(0, str(H)); import fam_common as C
    rS = C.win_ret(nv['AB_V4|stress']); rN = C.win_ret(nv['AB_V4|normal']); rZ = BI['AB_V4|zero']
    r50 = Bn['r0050']; rEW = Bn['rEW_ABV4']
    idx = D.index
    # ---------- 1) SPA_c 焦點：壓力成本 ----------
    print('== 1) SPA_c focus（F_long／F_main，W_long，L6，B=10000，研究員同種子）')
    m = C.wmask(idx, '2006-04-01', '2026-10-12'); n = int(m.sum())
    Wc = S.sb_counts(n, 6, 10000, 20261028 + 0 + 6)
    for fam in ('F_long', 'F_main'):
        mem = MM[MM.cov_W_long & MM[fam]].member_id.tolist()
        f = mem.index(MAIN)
        for loss, DD, bench in (('x0050', D, r50), ('xEW', Dew, rEW)):
            X = DD.loc[m, mem].fillna(0.0).to_numpy()
            o = S.mcp_tests(X, Wc, 6, 'pr', focus=f)
            Xs = X.copy(); Xs[:, f] = (rS - bench)[m].to_numpy()
            os_ = S.mcp_tests(Xs, Wc, 6, 'pr', focus=f)
            print(f'  {fam} {loss}: normal p_SPA_c_focus {o["p_SPA_c_focus"]:.3f} rank {o["focus_rank"]} t {o["focus_t"]:.2f} | '
                  f'STRESS p {os_["p_SPA_c_focus"]:.3f} rank {os_["focus_rank"]} t {os_["focus_t"]:.2f} StepM {os_["p_StepM_focus"]:.3f}')
    # ---------- 2) 子期間壓力成本 ----------
    print('== 2) 對 0050 / 對同池等權，三種成本（窗 t iid / HAC；年化差）')
    def ann(r, a, b):
        mm = C.wmask(r.index, a, b); x = r[mm]; return (np.prod(1 + x)) ** (12 / len(x)) - 1
    for lab, (a, b) in {'ALL': ('2006-04-01', '2026-10-12'), 'POST19': ('2019-05-01', '2026-10-12'),
                        'HOLD_grid(2024-02-15+)': ('2024-01-01', '2026-10-12'), 'HOLD+gapwin(2023-12-11+)': ('2023-12-01', '2026-10-12')}.items():
        mm = C.wmask(idx, a, b)
        for cl, r in (('zero', rZ), ('normal', rN), ('stress', rS)):
            x = (r - r50)[mm]; xe = (r - rEW)[mm]
            print(f'  {lab:26s} {cl:6s} n={mm.sum():3d} x0050 mean {x.mean()*100:5.2f}pp t {t_(x):5.2f}/{nw_t(x):5.2f} | xEW t {t_(xe):5.2f} | '
                  f'CAGR main {ann(r, a, b)*100:5.1f}% 0050 {ann(r50, a, b)*100:5.1f}%')
    # ---------- 3) 周轉 ----------
    pl = pd.read_csv(GRID_F)
    P = [set(s.split(',')) for s in pl['picks']]
    chg = np.array([len(P[i] - P[i - 1]) / 10 for i in range(1, len(P))])
    sig = pd.to_datetime(pl['signal']).iloc[1:].to_numpy()
    print('== 3) 周轉（plan_V4 官方版，每月換掉比例）：全期 %.2f，2019 後 %.2f，2024 後 %.2f；年單邊周轉約 %.1f 倍' % (
        chg.mean(), chg[sig >= np.datetime64('2019-05-01')].mean(), chg[sig >= np.datetime64('2024-01-01')].mean(), chg.mean() * 12))
    for cl, r in (('normal', rN), ('stress', rS)):
        print(f'   成本拖累 {cl}: 窗均 {(rZ - r).mean()*100:.2f} pp／窗；年化差 {(ann(rZ,"2006-04-01","2026-10-12")-ann(r,"2006-04-01","2026-10-12"))*100:.1f} pp')
    # ---------- 4) 容量 ----------
    Pn = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
    V = pd.read_parquet(Pn / 'value.parquet'); V20 = V.rolling(20, min_periods=10).mean().shift(1)
    O = pd.read_parquet(Pn / 'open.parquet'); Cc = pd.read_parquet(Pn / 'close_raw.parquet')
    rows = []
    for _, rw in pl.iterrows():
        e = pd.Timestamp(rw['exec']); codes = rw['picks'].split(',')
        if e not in V20.index: continue
        v = V20.loc[e, [c for c in codes if c in V20.columns]]
        vd = V.loc[e, [c for c in codes if c in V.columns]]
        rows.append(dict(e=e, med20=v.median(), min20=v.min(), part_med=1e6 / v.median(), part_max=(1e6 / v).max(),
                         day_part_max=(1e6 / vd.replace(0, np.nan)).max()))
    Q = pd.DataFrame(rows).set_index('e')
    print('== 4) 10 檔各 100 萬（1,000 萬資金）：持股 20 日均成交值與參與率')
    for lab, (a, b) in {'2006-2018': ('2006', '2018-12-31'), '2019-2023': ('2019', '2023-12-31'), '2024-2026': ('2024', '2026-12-31')}.items():
        q = Q.loc[a:b]
        print(f'  {lab}: 持股 20 日均值中位數 {q.med20.median()/1e6:7.1f} 百萬；最小檔中位數 {q.min20.median()/1e6:6.1f} 百萬；'
              f'最小檔參與率中位 {q.part_max.median()*100:.1f}%；>5% 的月比例 {(q.part_max>0.05).mean()*100:.0f}%；'
              f'換股日當天最小檔參與率 >10% 的月比例 {(q.day_part_max>0.10).mean()*100:.0f}%')
    # ---------- 5) 五分位 Q5：開盤漲停 ----------
    print('== 5) 五分位 Q5 的開盤漲停與可成交版本')
    sys.path.insert(0, str(V4D)); import v4lib as L
    L.check_freeze(); XR = L.setup_px(); YFX, Y5F = L.load_rfix(); DMT = L.disp_mask_fn('trade')
    import ia_core as IA
    SW = L.V4(YFX, Y5F, DMT, unavail=L.V.make_unavail('worst')); t0, t1, _ = SW.bounds()
    q5, q5f, q1 = [], [], []; lu_rows = []
    for dd in SW.sig_days:
        cs = SW.cross_section(dd); el = cs['el']
        if el.sum() < 50: continue
        sc = cs['sc']; ii = np.where(el & np.isfinite(sc))[0]
        order = ii[np.argsort(sc[ii], kind='stable')]; parts = np.array_split(order, 5)
        gu = IA.gap_up_open(cs['e'])
        p5 = np.array(parts[4], int); p1 = np.array(parts[0], int)
        q5.append((cs['e'], p5)); q1.append((cs['e'], p1)); q5f.append((cs['e'], p5[~gu[p5]]))
        lu_rows.append(dict(d=pd.Timestamp(dd), q5_lu=gu[p5].mean(), q1_lu=gu[p1].mean(), n5=len(p5)))
    LU = pd.DataFrame(lu_rows).set_index('d')
    for lab, (a, b) in {'2006-2018': ('2006', '2018-12-31'), '2019-2023': ('2019', '2023-12-31'), '2024-2026': ('2024', '2026-12-31')}.items():
        q = LU.loc[a:b]; print(f'  {lab}: Q5 開盤漲停比例 {q.q5_lu.mean()*100:.2f}%（Q1 {q.q1_lu.mean()*100:.2f}%），Q5 平均檔數 {q.n5.mean():.0f}')
    out = {}
    for name, plan, cost in (('Q5z', q5, 'zero'), ('Q5f_z', q5f, 'zero'), ('Q1z', q1, 'zero'), ('Q5f_n', q5f, 'normal')):
        b_, s_ = L.cost_rates(cost); nav, lg = L.O3.simulate(plan, b_, s_, t0, t1, xrules=XR); out[name] = C.win_ret(nav)
        if name == 'Q5f_n': print(f'  Q5（排除開盤漲停）月周轉 {lg["turnover"]:.2f}')
    ii = out['Q5z'].index
    for lab, x in (('Q5−Q1 零成本（研究員）', out['Q5z'] - out['Q1z']), ('Q5排除漲停−Q1 零成本', out['Q5f_z'] - out['Q1z']),
                   ('Q5排除漲停 一般成本 − 同池等權（可做多版）', out['Q5f_n'] - rEW)):
        x = x.dropna(); a_, b2, z = chow_mean(x.to_numpy(), x.index, '2024-01-01')
        print(f'  {lab}: 全期 {x.mean()*100:.2f}pp t {t_(x):.2f}；2024-01 前 {a_:.2f} → 後 {b2:.2f}（Welch z {z:.2f}）；2019-05 後 {x[x.index>="2019-05-01"].mean()*100:.2f}')

if __name__ == '__main__':
    main()
