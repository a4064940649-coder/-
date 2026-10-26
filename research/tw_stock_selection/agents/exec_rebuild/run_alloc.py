#!/usr/bin/env python3
"""D：配置（權重不改）：選股10 50%＋處置 50%；0050 40%＋選股30 40%＋處置 20%。每個換股日收盤再平衡三個部位。
主策略部位用 M2 日頻 NAV（成本見 exlib），處置部位用固定槽位逐日模擬（槽位資金＝處置部位淨值÷10）。
部位間搬錢：主策略部位增減以其買／賣費率計，0050 以 ETF 費率計，處置部位只動閒置資金（持 0050 版本另計 ETF 費率）。
輸出 results_alloc.json。"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exlib as L, dlib as D

OUT = L.HERE
MN = pd.read_parquet(OUT / 'main_nav.parquet')
DN = pd.read_parquet(OUT / 'disposal_nav.parquet')
t0, t1 = L.pos(L.dates[0]), L.pos(L.dates[-1])
REB = set(L.pos(d) for d in L.dates[1:-1])
E50 = L.E50.to_numpy()
ALLOCS = {'選股10 50%＋處置 50%': ('n10', {'main': .5, 'disp': .5, 'b50': 0.0}),
          '0050 40%＋選股30 40%＋處置 20%': ('n30', {'main': .4, 'disp': .2, 'b50': .4})}

def simulate(main_key, w, capital, idle='cash', disp_mode='slots', slip=0.001):
    mnav = MN[main_key].to_numpy(); mb, ms = L.cost_rates(slip)
    if disp_mode == 'slots':
        ev, _ = D.build_events(); sl = D.Sleeve(ev, 10, idle, True); sl.deposit(w['disp'] * capital, t0)
    else:   # 原版處置日報酬序列（每日等權、來回 0.5%、2021-10 前現金）
        dr = DN[disp_mode].pct_change().reindex(L.IDX).fillna(0.0).to_numpy(); dval = w['disp'] * capital
    mval = w['main'] * capital * mnav[0]; bval = w['b50'] * capital * (1 - D.ETF_BUY)
    nav, held = [], []
    for t in range(t0, t1 + 1):
        if t > t0:
            mval *= mnav[t - t0] / mnav[t - t0 - 1]; bval *= E50[t] / E50[t - 1]
            if disp_mode == 'slots': sl.step(t)
            else: dval *= 1 + dr[t]
        dcur = sl.equity(t) if disp_mode == 'slots' else dval
        E = mval + bval + dcur
        if t in REB:
            Td = w['disp'] * E
            if disp_mode == 'slots':
                if Td > dcur: sl.deposit(Td - dcur, t); pool = E - Td
                else: x = sl.withdraw_idle(dcur - Td, t); pool = E - dcur + x
            else:
                dval = Td; pool = E - Td
            wm, wb = w['main'], w['b50']
            nm = pool * wm / (wm + wb); nb = pool - nm
            dm, db = nm - mval, nb - bval
            mval = nm - (dm * mb if dm > 0 else -dm * ms); bval = nb - (db * D.ETF_BUY if db > 0 else -db * D.ETF_SELL)
        dcur = sl.equity(t) if disp_mode == 'slots' else dval
        nav.append(mval + bval + dcur); held.append(len(sl.pos) if disp_mode == 'slots' else np.nan)
    nav = pd.Series(nav, index=L.IDX[t0:t1 + 1]) / capital
    # 起點以 t0 前一日收盤 = 1（t0 當日已含進場成本）
    nav = pd.concat([pd.Series([1.0], index=[L.IDX[t0 - 1]]), nav])
    return nav, held

res = {}
# ---- 原版重現（月報酬法，複製 bt_aggressive.py）----
orig = pd.read_csv(L.HERE.parents[1] / 'output' / 'disposal_daily_returns.csv', index_col=0, parse_dates=True)['0.005_cash']
v = (1 + orig).cumprod().reindex(L.DATES, method='ffill').fillna(1.0); disp = (v.shift(-1) / v - 1).iloc[:-1]
def run_orig(n, rt=0.005):
    el = L.t300_d & L.yoy3.notna() & L.s52_d.notna() & ~L.inD_d
    sc = 0.5 * L.s52_d.where(el).rank(axis=1, pct=True) + 0.5 * L.yoy3.where(el).rank(axis=1, pct=True)
    port, to, prev = {}, [], None
    for t in L.R.index:
        ok = el.loc[t] & L.R.loc[t].notna()
        if ok.sum() < 50: continue
        pick = list(sc.loc[t][ok].sort_values(ascending=False).index[:n]); port[t] = L.R.loc[t][pick].mean()
        if prev is not None: to.append(len(set(pick) - prev) / n)
        prev = set(pick)
    return pd.Series(port) - float(np.mean(to)) * rt
r30, r10 = run_orig(30), run_orig(10); idx = r30.index.intersection(disp.index)
for name, parts in {'選股10 50%＋處置 50%': {'r10': .5, 'disp': .5}, '0050 40%＋選股30 40%＋處置 20%': {'b': .4, 'r30': .4, 'disp': .2}}.items():
    src = {'b': L.B, 'r30': r30, 'r10': r10, 'disp': disp}
    r = sum(wt * src[k].reindex(idx).fillna(0) for k, wt in parts.items())
    a = L.ann(r, L.dates); ex = r - L.B.reindex(idx)
    res[f'ORIG_period_{name}'] = {'CAGR': a['CAGR'], 'maxDD': a['maxDD'], 'vol': a['vol'], 't_vs_0050': L.tstat(ex), 'yearly': L.yearly(r, L.dates)}
    print('ORIG', name, res[f'ORIG_period_{name}'], flush=True)

# ---- 拆解（100 萬、閒置現金、滑價 0.1%）----
res['decomposition_1M_cash_slip0.1pct'] = {}
for name, (nk, w) in ALLOCS.items():
    o = {}
    for lab, mk, dm in [('M0主＋原版處置序列', f'M0_{nk}_slip0.001', 'orig_0.005_cash'), ('M2主＋原版處置序列', f'M2_{nk}_slip0.001', 'orig_0.005_cash'),
                        ('M0主＋固定槽位', f'M0_{nk}_slip0.001', 'slots'), ('M2主＋固定槽位（正式）', f'M2_{nk}_slip0.001', 'slots')]:
        nav, _ = simulate(mk, w, 1e6, 'cash', dm)
        s = L.daily_stats(nav, [('2021-03-01', '2026-10-31')], L.E50_NAV)
        o[lab] = {k: s[k] for k in ('CAGR', 'maxDD', 'vol', 't_vs_0050_m')}
    res['decomposition_1M_cash_slip0.1pct'][name] = o
    print(name, o, flush=True)

# ---- 正式網格 ----
for name, (nk, w) in ALLOCS.items():
    for slip in (0.001, 0.003, 'stress'):
        for cap in (1e6, 5e6, 2e7):
            for idle in ('cash', '0050'):
                nav, held = simulate(f'M2_{nk}_slip{slip}', w, cap, idle, 'slots', slip)
                key = f'{name}｜M2_{nk}_slip{slip}｜{int(cap/1e4)}萬｜idle_{idle}'
                h = pd.Series(held); tot = h + (10 if nk == 'n10' else 30)
                res[key] = {'windows': L.all_windows(nav, L.E50_NAV),
                            'total_names_held_distribution': {int(k): round(float(c / len(tot)), 3) for k, c in tot.value_counts().sort_index().items()},
                            'final_value_NTD': round(float(nav.iloc[-1] * cap))}
                f = res[key]['windows']
                print(key, {k: (x['CAGR'], x['maxDD'], x.get('t_vs_0050_m')) for k, x in f.items() if x}, flush=True)
(OUT / 'results_alloc.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
