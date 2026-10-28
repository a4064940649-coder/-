#!/usr/bin/env python3
"""事後診斷（POSTHOC，不參與判定，PREREG 之後撰寫）：
1. 修正 s4 的 describe_2019_2026.segments_main_state（s4 用錯索引，起點日期錯位；episodes 欄正確）。
2. 輪動對靜態混合的來源拆解：(w−w̄)(r50−rM)＝(w−w̄)[(r50−rEW) − (rM−rEW)]，看賺的是 d 的擇時還是主策略 α 的擇時。
3. 驗證期利得集中度（最大 3 個月）。
4. 前瞻腳本的 top5（官方日 K，history.csv.gz）與面板 top5 在重疊期的核對；以及 2026-10-12 決策日的目前狀態。
寫入 results.json 的 POSTHOC。"""
import sys, json, importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import cr_lib as L, cr_eval as E

RES = L.load_json(); assert 'OOS' in RES
PR = RES['PREREG']; spec = [s for s in PR['versions'] if s['id'] == PR['frozen']['main_version']][0]
W = pd.read_parquet(L.DATA / 'W_full.parquet')
NV4 = pd.read_parquet(L.RAW / 'navs_v4.parquet'); main = NV4['AB_V4|normal'].dropna(); END = main.index[-1]
Wv = W[W['end'] <= END]
e50 = L.e50_series()
OUT = {}
S, st, w = E.version_states(Wv, spec)
s19 = L.sel(st.dropna(), ('2019-01-01', '2026-10-31'))
OUT['segments_2019_2026_corrected'] = [dict(start=str(s19.index[a].date()), end_window_start=str(s19.index[b].date()), months=int(b - a + 1), state=l) for a, b, l in L.runs(list(s19))]

sd = list(Wv.index) + [END]
rM = L.win_ret(main, sd); r50 = L.win_ret(L.on_cal(e50, main.index), sd); rEW = Wv['rEW']
df = pd.DataFrame(dict(rM=rM, r50=r50, rEW=rEW.reindex(rM.index), st=st.reindex(rM.index), w=w.reindex(rM.index).astype(float))).dropna()
df['x'] = df.r50 - df.rM; df['dsim'] = df.r50 - df.rEW; df['alpha'] = df.rM - df.rEW
dec = {}
for pn, per in (('PRE', L.PRE), ('SEL', L.SEL), ('VAL', L.VAL)):
    g = L.sel(df, per); wb = g.w.mean()
    tot = ((g.w - wb) * g.x).sum(); from_d = ((g.w - wb) * g.dsim).sum(); from_a = -((g.w - wb) * g.alpha).sum()
    byst = g.groupby('st')[['x', 'dsim', 'alpha']].mean().mul(100).round(3).to_dict('index')
    dec[pn] = dict(n=int(len(g)), wbar=float(wb), timing_total_pp=float(tot * 100), from_d_pp=float(from_d * 100), from_alpha_pp=float(from_a * 100), mean_by_state_pp=byst,
                   t_x_C_minus_D=float((g[g.st == 'C'].x.mean() - g[g.st == 'D'].x.mean()) / np.sqrt(g[g.st == 'C'].x.var() / (g.st == 'C').sum() + g[g.st == 'D'].x.var() / (g.st == 'D').sum())))
OUT['timing_decomposition'] = dec
# 驗證期集中度
o = RES['OOS']['main_VAL_static_real_full']
g = L.sel(df, L.VAL); wb = g.w.mean(); c = ((g.w - wb) * g.x).sort_values(ascending=False) * 100
OUT['VAL_top_contributions_pp'] = {str(k.date()): round(float(v), 2) for k, v in c.head(5).items()}
OUT['VAL_bottom_contributions_pp'] = {str(k.date()): round(float(v), 2) for k, v in c.tail(5).items()}
OUT['VAL_share_top3_of_total'] = float(c.head(3).sum() / c.sum())

# 前瞻腳本核對
spec_l = importlib.util.spec_from_file_location('o7live', str(L.HERE.parent / 'live_tracking' / 'o7_regime' / 'o7_regime_live.py'))
mod = importlib.util.module_from_spec(spec_l); spec_l.loader.exec_module(mod)
Vh, Ch = mod.load_hist('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data')
ddl = mod.decision_dates(Vh.index)
P = L.Panel()
rows = []
for d in ddl:
    if d < pd.Timestamp('2025-12-01'): continue
    try: t_live = mod.top5_share(Vh, Ch, d)[0]
    except Exception: continue
    if d in P.IDX:
        idx, v = P.pool(d); t_pan = float(np.sort(v)[::-1][:5].sum() / v.sum())
    else: t_pan = np.nan
    rows.append(dict(date=str(d.date()), top5_live=round(t_live, 4), top5_panel=None if not np.isfinite(t_pan) else round(t_pan, 4)))
OUT['live_vs_panel_top5'] = rows
# 目前狀態（2026-10-12 決策日，面板口徑）：top5(2026-10-12) − top5(6 個決策日前)
dts = P.official_dates('2025-01-01'); dts = [d for d in dts if d <= pd.Timestamp('2026-10-22')]
t5 = {}
for d in dts:
    idx, v = P.pool(d); t5[d] = float(np.sort(v)[::-1][:5].sum() / v.sum())
k = sorted(t5); last = k[-1]
OUT['current_state_panel'] = dict(decision=str(last.date()), top5=t5[last], top5_6ago=t5[k[-7]], S2=t5[last] - t5[k[-7]], state='C' if t5[last] - t5[k[-7]] > 0 else 'D')
# 全期（2006-04～2026-10）一條連續模擬：主版本、同平均權重靜態、0050、主策略（只描述）
B = E.build(Wv, spec, main, e50, '2006-04-01', 'normal', per=('2006-01-01', '2026-10-31'))
o_all = E.evaluate(B, Wv, main, e50, ('2006-01-01', '2026-10-31'), END, label='ALL')
OUT['ALL_2006_2026'] = {k: v for k, v in E.strip(o_all).items() if k not in ('whipsaw',)}
OUT['ALL_2006_2026']['whipsaw_summary'] = {k: v for k, v in o_all['whipsaw'].items() if k != 'episodes'}
for nm, per in (('2019_2026', ('2019-01-01', '2026-10-31')),):
    o_ = E.evaluate(B, Wv, main, e50, per, END, label=nm)
    OUT[f'SUB_{nm}_from_ALL_sim'] = dict(CAGR_rot=o_['CAGR_rot'], MDD_rot=o_['MDD_rot'], CAGR_static=o_['CAGR_static'], CAGR_main=o_['CAGR_main'], CAGR_0050=o_['CAGR_0050'],
                                         vs_static=dict(mean_pp=o_['vs_static']['mean_pp'], t=o_['vs_static']['t']), vs_0050=o_['vs_0050'], vs_main=o_['vs_main'], switches=o_['switches'], wbar=o_['wbar'])
RES['POSTHOC'] = OUT
L.save_json(RES)
print(json.dumps({kk: vv for kk, vv in OUT.items() if kk != 'live_vs_panel_top5'}, ensure_ascii=False, indent=0, default=str)[:4000])
print(rows)
