#!/usr/bin/env python3
"""反方審查（成本與可成交）o_v5_pit_revenue：只讀研究員與 o_exec_capacity_tiers 的檔案，不寫任何研究員檔案。
python3 -I -B review_exec_o5.py  → 只印到 stdout。"""
import sys, json, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent
CAP = AG / 'o_exec_capacity_tiers'
A0, B0 = pd.Timestamp('2006-04-11'), pd.Timestamp('2026-10-12')

NV = pd.read_parquet(HERE / 'navs_v5.parquet')
BEN = pd.read_parquet(CAP / 'data' / 'bench.parquet')
EX = pd.read_parquet(CAP / 'navs_abavg.parquet')
E50 = BEN['E50']
sd = [pd.Timestamp(x) for x in pd.read_csv(HERE / 'plan_V5.csv')['signal']] + [B0]

def win(nav):
    nav = nav.dropna(); D = pd.DatetimeIndex([d for d in sd if d in nav.index]); x = nav.reindex(D)
    return (x.shift(-1) / x - 1).iloc[:-1]
def t(x): x = np.asarray(x, float); return float(x.mean() / x.std(ddof=1) * math.sqrt(len(x)))
def cagr_from_daily(r, a=A0):
    r = r[(r.index > a) & (r.index <= B0)].fillna(0); nav = (1 + r).prod()
    return float(nav ** (365.25 / (B0 - a).days) - 1) * 100
out = {}
W50 = win(E50.reindex(NV.index))
def xs(nav): return (win(nav) - W50).dropna()

# ---------- 0. 重現 ----------
out['repro_t_AB_V5'] = t(xs(NV['AB_V5|normal'])); out['repro_t_B10'] = t(xs(NV['AB_V5_B10|normal']))
ZI = BEN['AB_V4|normal'].reindex(NV.index)
out['AB_V5_vs_ZI_AB_V4_max_win_diff_pp'] = float((win(NV['AB_V5|normal']) - win(ZI)).abs().max() * 100)

# ---------- 1. 可執行拖累移植：o_exec_capacity_tiers 的 (可執行 AB_V4 − 零衝擊 AB_V4) 逐窗／逐日差，加到 AB_V5、AB_V5_B10 ----------
rzi = ZI.pct_change()
trans = {}
for base in ('AB_V5|normal', 'AB_V5_B10|normal'):
    xb = xs(NV[base]); rb = NV[base].pct_change()
    for col in [c for c in EX.columns if c.startswith(('main|', 'stress|', 'kappa_lo|', 'keep|'))]:
        ne = EX[col].reindex(NV.index)
        drag_w = (win(ne) - win(ZI)).reindex(xb.index)
        xe = xb + drag_w
        re = rb + (ne.pct_change() - rzi)
        trans[f'{base.split("|")[0]}<-{col}'] = {
            't': round(t(xe), 3), 't_from2008': round(t(xe[xe.index >= '2008-01-01']), 3),
            'drop_best1': round(t(xe.drop(xe.sort_values(ascending=False).index[:1])), 3),
            't_S1': round(t(xe[xe.index <= '2013-12-31']), 3),
            't_HOLD': round(t(xe[xe.index >= '2024-01-01']), 3),
            'CAGR': round(cagr_from_daily(re), 2), 'drag_pp_per_month': round(float(drag_w.mean() * 100), 3)}
out['exec_transfer'] = trans
e50c = cagr_from_daily(E50.reindex(NV.index).pct_change())
out['E50_CAGR'] = e50c
# SOP 實盤預期（0050＋超額×0.5）用可執行 B10 重算
sop = {}
for tier in ('30萬', '100萬', '300萬', '1000萬'):
    for m in ('FIX', 'GROW'):
        k = f'AB_V5_B10<-main|{tier}|{m}|A'
        c = trans[k]['CAGR']; sop[f'{tier}|{m}'] = {'B10_exec_CAGR': c, 'live_expect': round(e50c + (c - e50c) * 0.5, 2), 'B10_exec_t': trans[k]['t'], 'from2008_t': trans[k]['t_from2008']}
out['SOP_live_expect_by_tier'] = sop

# ---------- 2. 周轉（名單換手）----------
P = pd.read_csv(HERE / 'plan_AB_V5_seeds.csv', dtype={'picks': str})
P['signal'] = pd.to_datetime(P['signal'])
tov = []
for s, g in P.groupby('seed'):
    g = g.sort_values('signal'); prev = None
    for _, r in g.iterrows():
        cur = set(r['picks'].split(','))
        if prev is not None: tov.append(len(cur - prev) / max(len(cur), 1))
        prev = cur
out['name_turnover_per_rebalance_mean'] = float(np.mean(tov))
out['name_turnover_annual_oneway_x'] = float(np.mean(tov) * 12)
out['tax_drag_pp_per_year_approx'] = float(np.mean(tov) * 12 * 0.3)
out['normal_cost_roundtrip_pct'] = (0.001425 * 0.6 + 0.001) * 2 * 100 + 0.3
cb = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))['RESULTS']['breakeven_AB_V5']
out['researcher_breakeven'] = {k: cb[k] for k in ('slip_at_t2.3', 'slip_at_t2.0', 'fullfee_t')}

# ---------- 3. 換股日開盤跌停（鎖住賣不掉）與處置中賣出 ----------
PO = pd.read_parquet(CAP / 'data' / 'px_open.parquet'); PC = pd.read_parquet(CAP / 'data' / 'prevclose.parquet')
PT = pd.read_parquet(CAP / 'data' / 'px_traded.parquet')
IDXd = PO.index
gap = PO / PC - 1
sys.path.insert(0, str(AG / 'o_v4_revenue_patch_official'))
import v4lib as L4   # 只讀 import（處置名單）
sys.path.pop(0)
D = L4.R.load_list()
D = D[['code', 'announce', 'start', 'end']].copy()
for c in ('announce', 'start', 'end'): D[c] = pd.to_datetime(D[c])
def in_disp(code, sig, e):
    x = D[(D['code'] == code) & (D['announce'] <= sig) & (D['start'] <= e) & (D['end'] >= e)]
    return len(x) > 0
n_drop = n_lock_dn = n_near_dn = n_disp_sell = n_new = n_upnear = 0
lock_cost = []
for s, g in P.groupby('seed'):
    if s % 4: continue   # 10 個 seed 足夠估比例
    g = g.sort_values('signal'); prev = None
    for _, r in g.iterrows():
        sig = r['signal']; i = IDXd.get_loc(sig); e = IDXd[i + 1] if i + 1 < len(IDXd) else None
        cur = r['picks'].split(',')
        if e is None: break
        lim = 0.07 if e < pd.Timestamp('2015-06-01') else 0.10
        if prev is not None:
            for c in set(prev) - set(cur):
                if c not in PO.columns or not bool(PT.at[e, c]): continue
                n_drop += 1; gp = gap.at[e, c]
                if np.isfinite(gp) and gp <= -(lim - 0.006):
                    n_lock_dn += 1
                    j = IDXd.get_loc(e)
                    nxt = PO[c].iloc[j + 1:j + 6].dropna()
                    if len(nxt): lock_cost.append(float(nxt.iloc[0] / PO.at[e, c] - 1))
                if np.isfinite(gp) and gp <= -0.05: n_near_dn += 1
                if in_disp(c, sig, e): n_disp_sell += 1
            for c in set(cur) - set(prev):
                if c not in PO.columns: continue
                n_new += 1; gp = gap.at[e, c]
                if np.isfinite(gp) and gp >= 0.05: n_upnear += 1
        prev = cur
out['sells'] = {'n_drop_orders': n_drop, 'share_open_locked_limit_down': n_lock_dn / max(n_drop, 1),
                'share_open_gap_le_-5pct': n_near_dn / max(n_drop, 1), 'share_sold_while_in_disposal': n_disp_sell / max(n_drop, 1),
                'locked_dn_next_open_ret_mean_pct': float(np.mean(lock_cost) * 100) if lock_cost else None, 'n_locked': len(lock_cost)}
out['buys'] = {'n_new': n_new, 'share_new_open_gap_ge_+5pct': n_upnear / max(n_new, 1)}

# ---------- 4. 容量：入選股 V60 與部位比例 ----------
F = pd.read_parquet(CAP / 'data' / 'feat_daily.parquet', columns=['date', 'code', 'V60', 'vrank'])
F['date'] = pd.to_datetime(F['date']); F = F.set_index(['date', 'code'])
rows = []
for _, r in P[P['seed'] == 0].iterrows():
    for c in r['picks'].split(','):
        k = (r['signal'], c)
        if k in F.index: rows.append((r['signal'], c, float(F.at[k, 'V60']), float(F.at[k, 'vrank'])))
Q = pd.DataFrame(rows, columns=['sig', 'code', 'V60', 'vrank'])
cap = {'median_V60_mn': float(Q['V60'].median() / 1e6), 'p10_V60_mn': float(Q['V60'].quantile(0.1) / 1e6),
       'share_vrank_gt200': float((Q['vrank'] > 200).mean())}
for acct in (3e5, 1e6, 3e6, 1e7):
    pos = acct / 10
    cap[f'{int(acct/1e4)}萬_pos_over_V60_median_pct'] = float((pos / Q['V60']).median() * 100)
    cap[f'{int(acct/1e4)}萬_share_pos_gt_5pct_V60'] = float((pos / Q['V60'] > 0.05).mean())
out['capacity_seed0'] = cap
CR = json.loads((CAP / 'results.json').read_text(encoding='utf-8'))['RESULTS']
out['cap_study_verdict'] = {k: {m: (round(v[m]['CAGR'], 2), round(v[m]['t'], 3), v[m]['verdict']) for m in ('GROW', 'FIX')} for k, v in CR['verdict'].items()}
out['cap_study_turnover_two_sided'] = CR['breakeven']['30萬|FIX']['annual_turnover_two_sided']
eq = CR['exec_quantities']['30萬|FIX']
out['cap_study_30萬FIX_pure_odd_new_entry'] = eq['pure_odd_new_entry_share']
print(json.dumps(out, ensure_ascii=False, indent=1))
