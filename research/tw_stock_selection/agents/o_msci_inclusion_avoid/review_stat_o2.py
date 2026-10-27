#!/usr/bin/env python3
"""反方審查（資料與統計角度）o_msci_inclusion_avoid。只讀研究員程式與資料，不寫入研究員檔案。
輸出：scratchpad/review_stat_o2_out.json"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import oa_lib as O
K = O.K; IDX, COLS, CI = O.IDX, O.COLS, O.CI
OUT = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/review_stat_o2_out.json'
out = {}

INC = O.inclusion_events()
T = O.event_metrics(INC)
T = T[T.ok == True].copy()
T['grp'] = T.eff.dt.strftime('%Y-%m-%d')
main = T[~T.src3].copy()

def cm(x, g):
    s = pd.Series(np.asarray(x, float), index=np.asarray(g)).dropna(); m = s.groupby(level=0).mean()
    t = m.mean() / (m.std(ddof=1) / math.sqrt(len(m))) if len(m) > 2 else np.nan
    return m, round(float(m.mean()) * 100, 2), round(float(t), 2)

# ---------- 1. P1 的群平均、刪群、分年 ----------
for p in ('select', 'oos'):
    g = main[(main.cls == 'STD') & (main.period == p)]
    m, mu, t = cm(g.car_p1_5, g.grp)
    loo = {k: cm(g[g.grp != k].car_p1_5, g[g.grp != k].grp)[2] for k in m.index}
    srt = m.sort_values()
    d1 = cm(g[g.grp != srt.index[0]].car_p1_5, g[g.grp != srt.index[0]].grp)
    d2 = cm(g[~g.grp.isin(srt.index[:2])].car_p1_5, g[~g.grp.isin(srt.index[:2])].grp)
    d3 = cm(g[~g.grp.isin(srt.index[:3])].car_p1_5, g[~g.grp.isin(srt.index[:3])].grp)
    # 刪除最負的單一股票
    worst = g.car_p1_5.idxmin()
    gw = g.drop(worst)
    out[f'P1_{p}'] = {'mean%': mu, 't': t, 'cluster_means%': (m * 100).round(2).to_dict(),
                      'n_clusters_negative': int((m < 0).sum()), 'n_clusters': int(len(m)),
                      'leave_one_cluster_out_t_range': [min(loo.values()), max(loo.values())],
                      'drop_worst1_cluster': d1[1:], 'drop_worst2_clusters': d2[1:], 'drop_worst3_clusters': d3[1:],
                      'drop_worst_stock': [g.loc[worst, 'name'], round(float(g.loc[worst, 'car_p1_5']) * 100, 2), cm(gw.car_p1_5, gw.grp)[1:]],
                      'stock_level_naive_t': round(float(g.car_p1_5.mean() / (g.car_p1_5.std(ddof=1) / math.sqrt(len(g)))), 2),
                      'stocks_negative_share': round(float((g.car_p1_5 < 0).mean()), 2)}

# ---------- 2. 基準敏感度：等權市場（研究員）vs 0050 vs 流動性加權市場 ----------
R = np.where(np.isfinite(K.Rn), np.clip(K.Rn, -0.5, 0.5), np.nan)
E50 = K.E50.to_numpy(); R50 = np.r_[np.nan, E50[1:] / E50[:-1] - 1]
W = np.nan_to_num(O.V60, nan=0.0); W = np.vstack([np.zeros((1, W.shape[1])), W[:-1]])   # 前一日 60 日均成交值權重
VW = np.nansum(np.nan_to_num(R) * W, axis=1) / np.where(W.sum(axis=1) > 0, W.sum(axis=1), np.nan)
RN0 = np.clip(np.nan_to_num(K.Rn, nan=0.0), -0.5, 0.5)
def car_b(j, a, b, bench): return float(np.sum(RN0[a:b + 1, j] - bench[a:b + 1]))
for nm, bench in (('vs_0050', R50), ('vs_liq_weighted', VW), ('raw', np.zeros(len(IDX)))):
    for p in ('select', 'oos'):
        g = main[(main.cls == 'STD') & (main.period == p)]
        x = [car_b(CI[r.code], r.iE + 1, r.iE + 5, bench) for r in g.itertuples()]
        out.setdefault('P1_benchmark', {})[f'{nm}|{p}'] = cm(x, g.grp)[1:]
    g = main[main.cls == 'TW50']
    x = [car_b(CI[r.code], r.iE + 1, r.iE + 5, bench) for r in g.itertuples()]
    out['P1_benchmark'][f'{nm}|TW50_all'] = cm(x, g.grp)[1:]

# ---------- 3. 視窗敏感度（E+2..+6 去掉 E+1 收盤競價反彈；E+1..+3） ----------
for p in ('select', 'oos'):
    g = main[(main.cls == 'STD') & (main.period == p)]
    for nm, (a, b) in {'E+1': (1, 1), 'E+1..+3': (1, 3), 'E+2..+6': (2, 6), 'E+6..+20': (6, 20)}.items():
        x = [O.ar(CI[r.code], r.iE + a, r.iE + b) for r in g.itertuples()]
        out.setdefault('P1_windows', {})[f'{nm}|{p}'] = cm(x, g.grp)[1:]

# ---------- 4. 成交量對齊：E 是否真的是被動資金成交日 ----------
for c in ('STD', 'SC', 'TW50'):
    for p in ('select', 'oos'):
        g = main[(main.cls == c) & (main.period == p)]
        if not len(g): continue
        rr = {}
        for k in (-1, 0, 1):
            v = []
            for r in g.itertuples():
                j = CI[r.code]; i = r.iE
                base = np.nanmedian(K.V.iloc[i - 25:i - 5, j].to_numpy())
                v.append(K.V.iat[i + k, j] / base if base > 0 else np.nan)
            rr[f'E{k:+d}'] = round(float(np.nanmedian(v)), 2)
        out.setdefault('volume_ratio_median', {})[f'{c}|{p}'] = rr

# ---------- 5. 對照組：以「公告前 10 日到 E」的漲幅配對（研究員只用 E−10..E） ----------
allinc = set(zip(INC.review, INC.code))
def control(reviews, top_n=300, q=0.8):
    rows = []
    for rv in reviews:
        sub = INC[(INC.review == rv)]
        E = sub.eff.iloc[0]; i = IDX.get_loc(E); iA = int(sub.iA.iloc[0])
        if iA < 0: continue
        v60 = O.V60[i - 1]; px = K.Cn[i - 1]
        ok = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10) & K.TRADED[i]
        order = np.argsort(-np.where(ok, v60, -np.inf))[:top_n]
        top = [j for j in order if ok[j] and (rv, COLS[j]) not in allinc]
        ru = np.array([O.ar(j, iA - 10, i) for j in top]); p1_ = np.array([O.ar(j, i + 1, i + 5) for j in top]); qq = np.quantile(ru, q)
        rows += [dict(grp=str(E.date()), car_p1_5=b, ru=a) for a, b in zip(ru, p1_) if a >= qq]
    return pd.DataFrame(rows)
for p in ('select', 'oos'):
    g = main[(main.cls == 'STD') & (main.period == p)]
    ru_std = (g.car_preA10 + g.car_A_E)
    rv = sorted(g.review.unique())
    for top_n, q in ((300, 0.8), (100, 0.8), (100, 0.9)):
        C = control(rv, top_n, q)
        out.setdefault('control_long_runup', {})[f'{p}|top{top_n}|q{q}'] = {'ctrl_runup_mean%': round(float(C.ru.mean()) * 100, 2), 'ctrl_p1_5': cm(C.car_p1_5, C.grp)[1:]}
    out['control_long_runup'][f'{p}|STD_runup_mean%(A-10..E)'] = round(float(ru_std.mean()) * 100, 2)
    # 同審核配對：STD 減 對照（top100 q0.8）
    C = control(rv, 100, 0.8)
    a = g.groupby('grp').car_p1_5.mean(); b = C.groupby('grp').car_p1_5.mean(); d = (a - b).dropna()
    out['control_long_runup'][f'{p}|STD_minus_ctrl_top100_paired'] = [round(float(d.mean()) * 100, 2), round(float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))), 2), int(len(d))]

# ---------- 6. STD 與安慰劑臺灣50 是否可區分 ----------
def welch(a, b):
    return round(float((a.mean() - b.mean()) / math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))), 2)
gS = main[main.cls == 'STD']; gT = main[main.cls == 'TW50']; gH = main[main.cls == 'HDETF']
mS = gS.groupby('grp').car_p1_5.mean(); mT = gT.groupby('grp').car_p1_5.mean(); mH = gH.groupby('grp').car_p1_5.mean()
mS19 = gS[gS.period == 'oos'].groupby('grp').car_p1_5.mean(); mT19 = gT[gT.period == 'oos'].groupby('grp').car_p1_5.mean()
out['STD_vs_placebo'] = {'STD_all_mean%': round(mS.mean() * 100, 2), 'TW50_all_mean%': round(mT.mean() * 100, 2),
                         'welch_t_STD_minus_TW50_all': welch(mS, mT), 'welch_t_STD_minus_TW50_oos': welch(mS19, mT19),
                         'welch_t_STD_minus_HDETF': welch(mS19, mH)}

# ---------- 7. 分年 ----------
g = main[main.cls == 'STD'].copy(); g['yr'] = g.eff.dt.year
out['P1_by_year'] = g.groupby('yr').car_p1_5.agg(lambda s: (round(float(s.mean()) * 100, 2), int(len(s)))).to_dict()

# ---------- 8. 主策略層：7 個事件的配對差（月）重抽樣 ----------
RES = json.load(open(O.HERE / 'results.json'))
it = json.load(open(O.HERE / 'iterations.json'))
st = RES['OOS']['strategy']
pa = dict(it['results']['B2|n10|normal']['paired']['affected_list']); pa.update(st['B2|n10|normal']['oos']['paired']['affected_list'])
x = np.array(list(pa.values())) / 100
rng = np.random.default_rng(0); bs = np.array([rng.choice(x, len(x)).mean() for _ in range(20000)])
out['strategy_affected_months_all'] = {'list%': pa, 'mean%': round(float(x.mean()) * 100, 3), 't': round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2),
                                       'boot_P(mean>0)': round(float((bs > 0).mean()), 3), 'boot_95%': [round(float(np.quantile(bs, .025)) * 100, 2), round(float(np.quantile(bs, .975)) * 100, 2)]}
out['strategy_summary'] = {k: {p: (v[p]['CAGR'], v[p]['base_CAGR'], v[p]['dCAGR_pp'], v[p]['t_0050'], v[p]['base_t_0050'], v[p]['t_vs_main_all_months'])
                               for p in ('select', 'oos', 'ALL')} for k, v in st.items() if k.startswith(('B2|n10', 'B1|n10', 'A1|n10', 'B2_no_src3|n10', 'B2_plus_supp|n10'))}
json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=str)
for k, v in out.items(): print(k, json.dumps(v, ensure_ascii=False, default=str)[:1500])
