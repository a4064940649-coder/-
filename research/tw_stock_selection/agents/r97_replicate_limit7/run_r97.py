#!/usr/bin/env python3
"""R97：獨立重現「2015-06-01 以前漲跌幅 7%」。不讀 S97 程式。不改任何既有檔：載入面板後，只在記憶體裡覆寫 P.LIMUP／P.LIMDN 的 2015-06-01 以前各列。
用法：python3 -I -B run_r97.py base|lim7    （一次一個情境；V_BRIDGE、100 萬、位移 0；口徑＝v2：0050 銜接檔＋REDUCE_FIX）
輸出：out_<mode>.json、nav_<mode>.csv.gz、trades_<mode>.csv.gz、（lim7）events_lim7.json"""
import sys, json, time, resource
from pathlib import Path
HERE = Path(__file__).resolve().parent
S75 = HERE.parent / 's75_continuous_2008_2026'
sys.path.insert(0, str(S75))
import numpy as np, pandas as pd
import s75lib as K
from s75lib import cslib, s10lib
MODE = sys.argv[1]; assert MODE in ('base', 'lim7', 'lim7h')   # lim7h＝lim7 再加 2008-10-13～10-24 跌幅減半（跌停 3.5%；日期由原始日 K 驗證，見 extra_checks.json）
CUT = pd.Timestamp('2015-06-01')
t0 = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
# ---- v2 口徑（run_v2.py v2 模式的兩個開關）；銜接檔在記憶體重算後與既有 e50_spliced.csv 比對（不重寫）
old = pd.read_csv(cslib.E50_CSV, parse_dates=['date']).set_index('date')['adj'].sort_index()
new = pd.read_csv(cslib.E50_OFFICIAL_CSV, parse_dates=['date']).set_index('date')['adj'].sort_index()
seam = new.index[0]; k = float(new.loc[seam] / old.loc[seam])
sp = pd.concat([old.loc[:seam].iloc[:-1] * k, new])
ex = pd.read_csv(S75 / 'v2_2026-11-21' / 'e50_spliced.csv', parse_dates=['date']).set_index('date')['adj']
assert sp.index.equals(ex.index) and float((sp - ex).abs().max()) < 1e-9
cslib.E50_SOURCE = S75 / 'v2_2026-11-21' / 'e50_spliced.csv'; cslib.REDUCE_FIX = True
P, U = K.load_cont()
assert P.reduce_fix and P.exright_fix
cut = int(np.searchsorted(P.IDX.values, np.datetime64(CUT), side='left'))
assert P.IDX[cut] == CUT


def tick_r97(p):
    """上市股票升降單位（2005-03-01 起，至今未變；本人憑記憶，未上網查）：<10→0.01；10～<50→0.05；50～<100→0.1；100～<500→0.5；500～<1000→1；≥1000→5。"""
    p = np.asarray(p, float)
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)


def limits(ref, pct):
    """漲停＝floor(ref×(1+pct)÷跳動點)×跳動點；跌停＝ceil(ref×(1−pct)÷跳動點)×跳動點；跳動點取「未取整漲跌停價」所在級距；±1e-9 防浮點。"""
    up = ref * (1 + pct); t = tick_r97(up); up = np.floor(up / t + 1e-9) * t
    dn = ref * (1 - pct); t = tick_r97(dn); dn = np.ceil(dn / t - 1e-9) * t
    return up, dn


# 自己的 10% 算式先對引擎的 LIMUP／LIMDN（確認同一套取整）
u10, d10 = limits(P.REF, 0.10)
chk10 = dict(limup_max_abs_diff=float(np.nanmax(np.abs(u10 - P.LIMUP))), limdn_max_abs_diff=float(np.nanmax(np.abs(d10 - P.LIMDN))))
del u10, d10
info = dict(mode=MODE, cut_row=cut, cut_date=str(P.IDX[cut].date()), my10_vs_engine=chk10)
LU10 = P.LIMUP[:cut].copy(); LD10 = P.LIMDN[:cut].copy()
if MODE in ('lim7', 'lim7h'):
    u7, d7 = limits(P.REF[:cut], 0.07)
    P.LIMUP = P.LIMUP.copy(); P.LIMDN = P.LIMDN.copy()
    P.LIMUP[:cut] = u7; P.LIMDN[:cut] = d7
    if MODE == 'lim7h':
        h0 = int(np.searchsorted(P.IDX.values, np.datetime64('2008-10-13'), side='left')); h1 = int(np.searchsorted(P.IDX.values, np.datetime64('2008-10-24'), side='right'))
        _, d35 = limits(P.REF[h0:h1], 0.035); P.LIMDN[h0:h1] = d35; info['halved_rows'] = [str(P.IDX[h0].date()), str(P.IDX[h1 - 1].date()), h1 - h0]
    info['cells_changed'] = dict(limup=int((np.isfinite(u7) & (u7 != LU10)).sum()), limdn=int((np.isfinite(d7) & (d7 != LD10)).sum()))
    assert np.array_equal(P.LIMUP[cut:], cslib.np.floor(P.REF[cut:] * 1.10 / cslib.tick(P.REF[cut:] * 1.10) + 1e-9) * cslib.tick(P.REF[cut:] * 1.10), equal_nan=True)
D0, DE = P.IDX[P.t0], P.IDX[-1]
D19 = K.last_before(P.IDX, '2019-01-01'); D157 = pd.Timestamp('2015-07-01'); DH1 = K.last_before(P.IDX, '2015-07-01')
bench = pd.Series(P.E50, index=P.IDX).loc[D0:]; bench = bench / bench.iloc[0] * 1e6
F, RT = K.surge(P, 'B2')
sg = K.signal(P, U, RT, 0)
nav, lg, st = K.sim(P, sg, U, 1e6)
b = bench.reindex(nav.index); x = lg[lg['why'] != 'open_end']
yr = K.yearly_table(nav, b)
out = dict(info=info, window=[str(D0.date()), str(DE.date())], final_nav=float(nav.iloc[-1]), cagr=K.cagr(nav, D0, DE), maxdd=K.dd_info(nav),
           cagr_2008_2015H1=K.cagr(nav, D0, DH1), cagr_since2015_07=K.cagr(nav, D157, DE), cagr_since2019=K.cagr(nav, D19, DE),
           bases=dict(D0=str(D0.date()), DH1=str(DH1.date()), D157=str(D157.date()), D19=str(D19.date()), nav_DH1=float(nav.loc[DH1]), nav_D157=float(nav.loc[D157]), nav_D19=float(nav.loc[D19])),
           n_closed=int(len(x)), n_all=int(len(lg)), win_rate=float((x['ret'] > 0).mean()),
           final_nav_0050=float(b.iloc[-1]), cagr_0050=K.cagr(b, D0, DE), cagr_0050_2008_2015H1=K.cagr(b, D0, DH1), maxdd_0050=K.dd_info(b),
           yearly=yr, years_win=int(sum(y['strat'] > y['e0050'] for y in yr)), years_lose=int(sum(y['strat'] <= y['e0050'] for y in yr)),
           st={k_: (int(v) if isinstance(v, (int, np.integer)) else float(v)) for k_, v in st.items() if isinstance(v, (int, float, np.integer, np.floating))},
           limit7_diag_engine=K.limit7_diag(P, lg) if MODE == 'base' else None)
if MODE == 'base':
    ref = pd.read_csv(S75 / 'v2_2026-11-21' / 'nav_2008_2026_v2.csv.gz', index_col=0, parse_dates=True)['V_BRIDGE']
    same = ref.index.equals(nav.index)
    out['accept'] = dict(same_index=bool(same), max_abs_diff_vs_v2_nav=float((nav.round(2) - ref).abs().max()) if same else None,
                         trades_same=None)
    tr = pd.read_csv(S75 / 'v2_2026-11-21' / 'trades_v2.csv.gz', dtype={'code': str}); tr = tr[tr['version'] == 'V_BRIDGE']
    out['accept']['trades_same'] = bool(len(tr) == len(lg) and (tr['ent_t'].to_numpy() == lg['ent_t'].to_numpy()).all() and (tr['code'].to_numpy() == lg['code'].to_numpy().astype(str)).all()
                                        and np.allclose(tr['proceeds'].to_numpy(), lg['proceeds'].to_numpy()))
# ---- 2015-06-01 以前：每個「引擎判定點」在 7%／10% 下的狀態（買：前一日訊號名單；賣：照本情境實際的賣出與順延）
fills = st['fills']
out['fills_pre_cut'] = [(int(t), str(P.COLS[j]), s, float(o)) for t, j, s, o in fills if t < cut]
# 訊號名單（換股日隔天）開盤鎖漲停檢查——不管有沒有空位，全部列出
rows = []
for i in sg['info']:
    kk = i['k']; t = kk + 1
    if t >= cut or t >= P.T: continue
    for c in i['targets']:
        j = P.ci[c]; o = P.O[t, j]
        u7_, d7_ = limits(P.REF[t, j], 0.07)
        rows.append(dict(month=i['month'], date=str(P.IDX[t].date()), code=c, ref=float(P.REF[t, j]), open=float(o), high=float(P.Hraw[t, j]) if hasattr(P, 'Hraw') else None,
                         close=float(P.C[t, j]), limup7=float(u7_), limup10=float(LU10[t, j]), lock7=bool(np.isfinite(o) and o >= u7_ - 1e-9), lock10=bool(np.isfinite(o) and o >= LU10[t, j] - 1e-9)))
out['list_pre_cut'] = dict(n=len(rows), n_lock7=int(sum(r['lock7'] for r in rows)), n_lock10=int(sum(r['lock10'] for r in rows)), locked=[r for r in rows if r['lock7'] or r['lock10']])
out['runtime_s'] = round(time.time() - t0, 1); out['maxrss_mb'] = rss()
(HERE / f'out_{MODE}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
pd.DataFrame({'nav': nav.round(2), 'e0050': b.round(2)}).rename_axis('date').to_csv(HERE / f'nav_{MODE}.csv.gz')
lg.drop(columns=['j']).to_csv(HERE / f'trades_{MODE}.csv.gz', index=False)
print(json.dumps({k_: v for k_, v in out.items() if k_ not in ('yearly', 'fills_pre_cut', 'list_pre_cut', 'st')}, ensure_ascii=False, indent=1, default=str))
print('list_pre_cut', out['list_pre_cut']['n'], out['list_pre_cut']['n_lock7'], out['list_pre_cut']['n_lock10'])
print({k_: v for k_, v in out['st'].items() if 'blk' in k_ or 'defer' in k_})
