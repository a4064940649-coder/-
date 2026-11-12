#!/usr/bin/env python3
"""第 1 步探索（只用挑選期 2006-04～2014-12-31 的描述統計；不碰 2015 以後任何報酬或成本）。→ explore.json
  A 會計拖累五項（o20 口徑）與 B1 分母；B 4.4pp 橋段在挑選期的樣子（cash／prop 兩種退款）與月份 bootstrap SE→MDE；
  C 實際成交價 edge vs 開盤價 edge；D 成本五等分（ĉ、實際成本）；E 敏感度格在挑選期的年化變動與順延／下單日。
python3 -I -B explore.py"""
import sys, json, pickle, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import csr_lib as L; import diag as DG; sys.path.pop(0)
SCR = L.SCR; t0 = time.time(); PER = 'SEL14'; END = L.SEL_END
OUT = dict(utc=pd.Timestamp.now('UTC').isoformat(), period='2006-04～2014-12-31（題 1 挑選期）', note='描述統計；不是正式數字')
M = pickle.load(open(SCR / 'center_meta.pkl', 'rb'))
NC = pd.read_parquet(SCR / 'navs_center.parquet'); NC.index = DG.IDX
NR = pd.read_parquet(SCR / 'navs_refund.parquet'); NR.index = DG.IDX
NK = pd.read_parquet(SCR / 'navs_coal.parquet'); NK.index = DG.IDX
cut = lambda df: df.loc[:END]
NC, NR, NK = cut(NC), cut(NR), cut(NK)          # 物理截斷：之後的程式碼看不到 2015 年以後的淨值
F = pd.read_parquet(SCR / 'fills_center_normal_100萬_std.parquet'); F['variant'] = F.variant.astype(str)
F = F[DG.IDX[F.t.values] <= END]
c0 = L.cash0(DG.E, '100萬')
# ---------- A／B ----------
A = {}; B = {}
for v in L.VARS:
    ad = []; b1 = []
    for j in range(13):
        key = f'{v}|{j}|normal|100萬|std'
        f = DG.prep_fills(F, M['rebs'][key], v, j)
        ad.append(DG.acct_drag(f, M['rebs'][key], PER, j)); b1.append(DG.b1_drag(f, NC[key], M['daycost'][key], PER, j, c0))
    A[v] = dict(acct={k: float(np.median([x[k] for x in ad])) for k in ad[0]}, b1={k: float(np.median([x[k] for x in b1])) for k in b1[0]})
    for rk in ('cash', 'prop'):
        BI = DG.bridge_inputs(PER, v, NC, NR, NK, M['rebs'], M['daycost'], F, c0, rk)
        q = DG.bridge_levels(BI); ch = DG.chain(q)
        ch['coal_vs_center_allon_maxabs'] = float(max(abs(DG.cagr_per(NK[f'TXFKIC|{v}|{j}'], j, PER) - DG.cagr_per(NC[f'{v}|{j}|normal|100萬|std'], j, PER)) for j in range(13)))
        ch['Tfirst_med'] = q['Tfirst_med']; ch['per_j'] = {k: [float(x) for x in q[k]] for k in ('c_acc', 'c_B1', 'ell', 'G', 'G_lin', 'gA', 'gR', 'gT', 'q3', 'q4', 'CCj')}
        if v == 'REAL_opt': ch['boot'] = DG.bridge_boot(BI, B=300)
        B[f'{v}|{rk}'] = ch
    print('AB', v, round(time.time() - t0), flush=True)
OUT['A_acct_drag'] = A; OUT['B_bridge'] = B
# ---------- C／D ----------
X = pd.read_parquet(SCR / 'swaps_center.parquet')
X = X[(pd.to_datetime(X.signal) <= END) & np.isfinite(X.edge_B)].copy()
X = X[X.date_e <= pd.Timestamp('2014-11-30')]
X['gap'] = X.edge_B - X.rtc_B; X['gap_fill'] = X.edge_fill - X.rtc_B; X['d_fill'] = X.edge_fill - X.edge_B
C = {}
for cell, g in X.groupby('cell'):
    o = {}
    for v, h in g.groupby('variant'):
        r = dict(n=int(len(h)), edge_open_mean=float(h.edge_B.mean()), edge_fill_mean=float(h.edge_fill.mean()), edge_allin_mean=float(h.edge_allin.mean()),
                 edge_open_med=float(h.edge_B.median()), edge_fill_med=float(h.edge_fill.median()), rtc_mean=float(h.rtc_B.mean()),
                 deferred_share=float((h.deferred > 0).mean()), d_fill_deferred=float(h.d_fill[h.deferred > 0].mean()) if (h.deferred > 0).any() else None,
                 d_fill_nondeferred=float(h.d_fill[h.deferred == 0].mean()), uniq_events=int(h.assign(m=pd.to_datetime(h.signal).dt.to_period('M')).groupby(['code', 'm']).ngroups))
        if cell in ('normal|100萬|std', 'normal|30萬|std') and v == 'REAL_opt':
            r['d_fill_neff'] = DG.neff_mean(h.d_fill.values, h.signal.values)
            for col in ('chat', 'rtc_B'):
                for gc in ('gap', 'gap_fill'):
                    tab, dd, st = DG.quint(h, col, gc)
                    r[f'quint_{col}_{gc}'] = dict(tab=tab, diff_q0_minus_q4=DG.boot(dd, st, B=500))
            r['spearman_chat_rtc'] = float(h[['chat', 'rtc_B']].corr(method='spearman').iloc[0, 1])
        o[v] = r
    C[cell] = o
OUT['C_D_edges_quintiles'] = C
print('CD', round(time.time() - t0), flush=True)
# ---------- E 敏感度（挑選期年化；順延） ----------
Egrid = {}
try:
    G = pickle.load(open(SCR / 'grid_meta.pkl', 'rb'))
    base = {k: M['pstats'][k]['SEL14']['cagr'] for k in M['pstats']}
    for c in L.CELLS:
        ck = L.CELLKEY[c]
        for (mo, mc) in L.GRID:
            if (mo, mc) == (1.0, 1.0): continue
            d = []
            for v in L.VARS:
                d.append(float(np.median([G['pstats'][f'{mo}|{mc}|1.0|{v}|{j}|{ck}']['SEL14']['cagr'] for j in range(13)]) - np.median([base[f'{v}|{j}|{ck}'] for j in range(13)])))
            sm = [G['summ'][f'{mo}|{mc}|1.0|REAL_opt|{j}|{ck}']['SEL14'] for j in range(13)]
            Egrid[f'{ck}|mo={mo}|mc={mc}'] = dict(dCAGR_SEL14_by_variant=d, fill_days_share=float(np.median([s['n_fill_days'] / s['n_tdays'] for s in sm])),
                                                   chain_td_p50=float(np.median([s['chain_td_p50'] or 0 for s in sm])))
        sm = [M['summ'][f'REAL_opt|{j}|{ck}']['SEL14'] for j in range(13)]
        Egrid[f'{ck}|center'] = dict(fill_days_share=float(np.median([s['n_fill_days'] / s['n_tdays'] for s in sm])), chain_td_p50=float(np.median([s['chain_td_p50'] or 0 for s in sm])))
except FileNotFoundError:
    Egrid = 'grid 尚未產生'
OUT['E_grid_SEL14'] = Egrid
OUT['runtime_s'] = round(time.time() - t0, 1)
(HERE / 'explore.json').write_text(json.dumps(OUT, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print('done', round(time.time() - t0))
