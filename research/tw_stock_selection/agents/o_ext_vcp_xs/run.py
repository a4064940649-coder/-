#!/usr/bin/env python3
"""VCP 突破警示（XS）忠實重做：全部回測。執行：python3 -I -B run.py（約 3 分鐘）。
前置：prereg.py 已寫入 results.json 的 PREREG（本檔檢查 sha256 後只附加結果）；
data/low.parquet、data/volume.parquet 已存在（不存在時用 vcplib.build_lowvol() 重建，需要 scratchpad/ext_vcp/k66）。"""
import json, math, hashlib, sys, time, itertools
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vcplib as L  # noqa: E402

T0 = time.time()
RES = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
PRE = RES['PREREG']
assert hashlib.sha256(json.dumps(PRE, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest() == RES['PREREG_SHA256'], 'PREREG 被改過'
RES = {'PREREG': PRE, 'PREREG_SHA256': RES['PREREG_SHA256'], 'PREREG_SHA256_method': RES['PREREG_SHA256_method']}


def rnd(o, k=6):
    if isinstance(o, dict): return {str(a): rnd(b, k) for a, b in o.items()}
    if isinstance(o, (list, tuple)): return [rnd(a, k) for a in o]
    if isinstance(o, (float, np.floating)): return None if not np.isfinite(o) else round(float(o), k)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, (pd.Timestamp, pd.Period)): return str(o)
    return o


def log(*a): print(f'[{time.time() - T0:6.1f}s]', *a, flush=True)


# ---------------------------------------------------------------- 資料閘門
if not (L.DATA / 'low.parquet').exists():
    gate = L.build_lowvol(verbose=False); (L.SCR / 'gate_lowvol.json').write_text(json.dumps(gate, indent=1))
gate = json.loads((L.SCR / 'gate_lowvol.json').read_text()) if (L.SCR / 'gate_lowvol.json').exists() else json.loads((HERE / 'data' / 'gate_lowvol.json').read_text())
(HERE / 'data' / 'gate_lowvol.json').write_text(json.dumps(gate, indent=1))
G = L.load()
IDX, COLS = G.IDX, G.COLS
yrs = IDX.year.to_numpy()
RES['data_gate'] = dict(
    panel=dict(dates=[str(IDX[0].date()), str(IDX[-1].date())], n_days=G.T, n_codes=G.N, missing_gap='2024-01-02～2024-02-14 全市場沒有任何來源（面板沒有這 28 個交易日；K 棒計數會跳過）',
               tdr_codes=int(sum(c.startswith('91') for c in COLS))),
    lowvol=gate,
    src_share_of_bars={int(k): round(float((G.SRC[G.TR] == k).mean()), 4) for k in (1, 2, 3, 4)},
    taiex=dict(file=str(L.TAIEX), missing_on_panel_days=G.TWII_raw_missing, note='缺的日子用前值'),
    disposal=dict(file=str(L.DISPF), events_in_panel=G.N_DISP, announce_range=G.DISP_RANGE),
    open_at_limit_up_share_of_bars=float(G.OPENLIM.sum() / G.TR.sum()),
    tpex='沒有做：scratchpad/tpex_panel 沒有最低價與成交股數；Asoul/tsec 的上櫃列只到 2019-05，之後沒有含最低價的來源，且櫃買面板 2023 年只剩 186 檔',
    volume_unit='張 = 成交股數 ÷ 1000（官方 STOCK_DAY 成交股數；是否含盤後定價與零股需查證）')
log('data gate done')

# ---------------------------------------------------------------- 訊號
R0 = L.compute_signal()
SIG = L.signal_table(R0)
keep_cols = ['date', 'code', 'barno', 'close', 'prev_close', 'pivot', 'pivot_prev', 'h1', 'l1', 'h2', 'l2', 'h3', 'l3', 'd1', 'd2', 'd3', 'av10', 'av50', 'vr', 'vol_lots', 'vbr',
             'ma50', 'ma150', 'ma200', 'ma200_prev', 'low52', 'high52', 'stkroc', 'idxroc', 'flag_d', 'src_d', 'open_e', 'gap_ref', 'noopen', 'openlim', 'lockup', 'disp_e', 'flag_e']
SIG[keep_cols].to_csv(float_format='%.6g', path_or_buf=HERE / 'signals.csv.gz', index=False, compression='gzip')
per = SIG.groupby('code').size()
okb = R0['okbar']
RES['signals'] = dict(
    n=int(len(SIG)), n_stocks=int(per.size), first=str(SIG.date.min().date()), last=str(SIG.date.max().date()),
    by_year={int(y): int(n) for y, n in SIG.groupby(SIG.date.dt.year).size().items()},
    per_stock_distribution={int(k): int(v) for k, v in per.value_counts().sort_index().items()},
    top_stocks=per.sort_values(ascending=False).head(8).to_dict(),
    by_src={int(k): int(v) for k, v in SIG.src_d.value_counts().items()}, by_flag_d={int(k): int(v) for k, v in SIG.flag_d.value_counts().items()},
    in_bars_260_279=int(((SIG.barno >= 260) & (SIG.barno < 280)).sum()), barno_lt_660=int((SIG.barno < 660).sum()),
    tdr=int(SIG.code.str.startswith('91').sum()),
    okbar_stock_days=int(okb.sum()),
    cond_pass_share={k: float((R0[k] & okb).sum() / okb.sum()) for k in ('trend', 'contract', 'voldry', 'rs', 'volbrk', 'minvol', 'cross')},
    funnel={'trend': int((okb & R0['trend']).sum()), 'trend&rs': int((okb & R0['trend'] & R0['rs']).sum()),
            '&contract': int((okb & R0['trend'] & R0['rs'] & R0['contract']).sum()),
            '&voldry': int((okb & R0['trend'] & R0['rs'] & R0['contract'] & R0['voldry']).sum()),
            '&cross': int((okb & R0['trend'] & R0['rs'] & R0['contract'] & R0['voldry'] & R0['cross'] & (R0['pivot'] > 0)).sum()),
            '&volbrk': int((okb & R0['trend'] & R0['rs'] & R0['contract'] & R0['voldry'] & R0['cross'] & (R0['pivot'] > 0) & R0['volbrk']).sum()),
            '&minvol(=signal)': int(R0['sig'].sum())})


def naive_check(code, date):
    """另一條程式路徑的人工核對：直接用該股的 K 棒序列與明確切片，逐行照 XS 算。"""
    j = G.CI[code]; m = G.TR[:, j]
    d = pd.DataFrame({'C': G.C[m, j], 'H': G.H[m, j], 'L': G.L[m, j], 'V': G.VOL[m, j] / 1000.0, 'I': G.TWII[m]}, index=IDX[m])
    i = d.index.get_loc(pd.Timestamp(date)); C, H, Lo, V, I = (d[k].to_numpy() for k in 'CHLVI')
    avg = lambda a, n, k=i: float(a[k - n + 1:k + 1].mean())
    o = dict(code=code, date=str(pd.Timestamp(date).date()), CurrentBar=i + 1, Close=float(C[i]), Close_1=float(C[i - 1]), Volume_lots=float(V[i]))
    o['ma50'], o['ma150'], o['ma200'] = avg(C, 50), avg(C, 150), avg(C, 200)
    o['ma200_20ago'] = avg(C, 200, i - 20) if (i + 1 - 20) >= 260 else 0.0
    o['low52'] = float(Lo[i - 249:i + 1].min()); o['high52'] = float(H[i - 249:i + 1].max())
    o['trend'] = bool(C[i] > o['ma150'] and C[i] > o['ma200'] and o['ma150'] > o['ma200'] and o['ma200'] > o['ma200_20ago'] and o['ma50'] > o['ma150'] and o['ma50'] > o['ma200']
                      and C[i] > o['ma50'] and o['low52'] > 0 and C[i] >= o['low52'] * 1.3 and o['high52'] > 0 and C[i] >= o['high52'] * 0.75)
    S = 25
    seg = lambda a, k0, fn: float(fn(a[i - k0 - S + 1:i - k0 + 1]))        # X[k0] 起往前 S 根
    o['h3'], o['l3'] = seg(H, 1, np.max), seg(Lo, 1, np.min); o['h2'], o['l2'] = seg(H, S + 1, np.max), seg(Lo, S + 1, np.min); o['h1'], o['l1'] = seg(H, 2 * S + 1, np.max), seg(Lo, 2 * S + 1, np.min)
    for q in '123': o['d' + q] = (o['h' + q] - o['l' + q]) / o['h' + q] * 100
    o['contract'] = bool(o['d1'] >= 10 and o['d1'] > o['d2'] and o['d2'] > o['d3'] and o['d3'] <= 10)
    o['avgVol10'] = float(V[i - 10:i].mean()); o['avgVol50'] = float(V[i - 50:i].mean()); o['volRatio'] = o['avgVol10'] / o['avgVol50']
    o['volDry'] = bool(o['avgVol50'] > 0 and o['volRatio'] < 0.7)
    o['pivot'] = o['h3']; o['pivot_1'] = float(H[i - 1 - S:i - 1].max()) if i >= 260 else 0.0
    o['idxROC'] = float((I[i] / I[i - 250] - 1) * 100); o['stkROC'] = float((C[i] / C[i - 250] - 1) * 100); o['rsOK'] = bool(not (o['stkROC'] < o['idxROC']))
    o['cross'] = bool(C[i] > o['pivot'] and C[i - 1] <= o['pivot_1'])
    o['volBreak'] = bool(V[i] > o['avgVol50'] * 1.5); o['vol_ratio_break'] = float(V[i] / o['avgVol50']); o['minVol'] = bool(V[i] >= 500)
    o['signal'] = bool(o['CurrentBar'] >= 260 and o['trend'] and o['contract'] and o['volDry'] and o['rsOK'] and o['cross'] and o['volBreak'] and o['minVol'])
    return o


smp = SIG.sample(5, random_state=20261114).sort_values('date')
chk = []
for _, r in smp.iterrows():
    o = naive_check(r.code, r.date)
    o['matches_vectorized'] = bool(o['signal'] and abs(o['pivot'] - r['pivot']) < 1e-9 and abs(o['d1'] - r.d1) < 1e-9 and abs(o['d3'] - r.d3) < 1e-9 and abs(o['volRatio'] - r.vr) < 1e-9
                                   and abs(o['ma200'] - r.ma200) < 1e-9 and abs(o['stkROC'] - r.stkroc) < 1e-9 and abs(o['idxROC'] - r.idxroc) < 1e-9)
    chk.append(o)
# 全部訊號都用另一條路徑重算一次；再抽 300 個非訊號（通過 cross 的）確認另一條路徑也說「不是訊號」
allok = sum(naive_check(c, d)['signal'] for c, d in zip(SIG.code, SIG.date))
bb, jj = np.where(R0['cross'] & okb & ~R0['sig']); rng = np.random.default_rng(5); pick = rng.choice(len(bb), 300, replace=False)
neg = sum(naive_check(COLS[jj[i]], IDX[G.POS[bb[i], jj[i]]])['signal'] for i in pick)
RES['signals']['manual_check_5'] = chk
RES['signals']['naive_recheck'] = dict(signals_confirmed=int(allok), of=int(len(SIG)), nonsignal_cross_days_sampled=300, falsely_flagged_by_naive=int(neg))
log('signals', len(SIG), 'naive ok', allok, 'neg', neg)

# ---------------------------------------------------------------- 事件研究
EV, EX = L.event_study(SIG)
ES = dict(exclusions=EX, all=L.event_summary(EV))
ES['excl_disposal'] = L.event_summary(EV[~EV.disp_e], boot=False)
ES['disposal_only_n'] = int(EV.disp_e.sum())
for nm, a, b in [('sel_2006_2014', '2006-01-01', '2014-12-31'), ('val_2015_2020', '2015-01-01', '2020-12-31'), ('hold_2021_2026', '2021-01-01', '2026-12-31'),
                 ('p1_2006_2012', '2006-01-01', '2012-12-31'), ('p2_2013_2018', '2013-01-01', '2018-12-31'), ('p3_2019_on', '2019-01-01', '2026-12-31')]:
    ES[nm] = L.event_summary(EV[(EV.date >= a) & (EV.date <= b)], boot=False)
gp = SIG.loc[~SIG.noopen, 'gap_ref'].to_numpy()
ES['next_open_gap_vs_ref'] = dict(n=int(len(gp)), mean=float(np.mean(gp)), median=float(np.median(gp)), p25=float(np.percentile(gp, 25)), p75=float(np.percentile(gp, 75)),
                                  p90=float(np.percentile(gp, 90)), share_gap_up=float((gp > 0).mean()), share_ge_3pct=float((gp >= 0.03).mean()),
                                  open_at_limit=int(SIG.openlim.sum()), locked_all_day=int(SIG.lockup.sum()))
gb = EV['gap_ref'].to_numpy()
ES['next_open_gap_buyable'] = dict(n=int(len(gb)), mean=float(gb.mean()), median=float(np.median(gb)))
# 同池同日的隔日開盤缺口（對照）
pg = []
for ti in np.unique(SIG.t.to_numpy()):
    ei = ti + 1
    if ei >= G.T: continue
    m = (G.VOL[ti] >= 500e3) & (G.BARNO[ti] >= 260) & np.isfinite(G.O[ei])
    with np.errstate(invalid='ignore', divide='ignore'): pg.append(np.nanmean(G.O[ei, m] / G.REF[ei, m] - 1))
ES['pool_next_open_gap_mean'] = float(np.mean(pg))
# 訊號日當天的漲幅（收盤對參考價）
tt, jj = SIG.t.to_numpy(), SIG.j.to_numpy()
dret = G.C[tt, jj] / G.REF[tt, jj] - 1
ES['signal_day_return'] = dict(mean=float(dret.mean()), median=float(np.median(dret)), share_limit_up_close=float((G.C[tt, jj] >= G.UPLIM[tt, jj] - 1e-6).mean()))
ES['breakout_margin_close_over_pivot'] = dict(mean=float((SIG['close'] / SIG['pivot'] - 1).mean()), median=float((SIG['close'] / SIG['pivot'] - 1).median()))
RES['event_study'] = ES
log('event study done')

# ---------------------------------------------------------------- 組合回測
first_sig_possible = IDX[np.where((G.BARNO >= 261).any(1))[0][0]]
nxt = IDX[(IDX.year * 12 + IDX.month) > (first_sig_possible.year * 12 + first_sig_possible.month)]
t0 = IDX.get_loc(nxt[0]) - 1          # 淨值起點＝起始日前一交易日收盤（100 萬現金），第一個買進日最早是起始日
START = IDX[t0]
E50 = pd.Series(G.E50, index=IDX)
PERIODS = {'full': (str(START.date()), '2026-10-22'), 'y2019_on': ('2019-01-01', '2026-10-22'), 'p1_2006_2012': ('2006-01-01', '2012-12-31'),
           'p2_2013_2018': ('2013-01-01', '2018-12-31'), 'sel_2006_2014': ('2006-01-01', '2014-12-31'), 'val_2015_2020': ('2015-01-01', '2020-12-31'),
           'hold_2021_2026': ('2021-01-01', '2026-10-22'), 'y2024_on': ('2024-01-01', '2026-10-22')}
RES['portfolio_setup'] = dict(first_signal_possible=str(first_sig_possible.date()), nav_start=str(START.date()), periods=PERIODS)
NAVS = {}


def run(sig, rule, name=None, keep_nav=False, **kw):
    nav, tr, info = L.simulate(sig, rule, t0=t0, **kw)
    o = {k: L.period_stats(nav, E50, a, b) for k, (a, b) in PERIODS.items()}
    o['ex_best3'] = L.ex_best3(nav, E50, *PERIODS['full'])
    o['info'] = {k: v for k, v in info.items() if k not in ('sum_pos', 'sum_invested_share', 'min_cash_note')}
    if len(tr):
        o['trades'] = dict(n=int(len(tr)), mean_ret_net=float(tr.ret_net.mean()), median_ret_net=float(tr.ret_net.median()), win=float((tr.ret_net > 0).mean()),
                           avg_hold_days=float(tr.hold_days.mean()), reasons=tr.reason.value_counts().to_dict(),
                           avg_win=float(tr.ret_net[tr.ret_net > 0].mean()), avg_loss=float(tr.ret_net[tr.ret_net <= 0].mean()),
                           top10_pnl_share=float((tr.proceeds - tr.cost).sort_values(ascending=False).head(10).sum() / (tr.proceeds - tr.cost).sum()) if (tr.proceeds - tr.cost).sum() != 0 else None)
    yrs_ = (nav.index[-1] - nav.index[0]).days / 365.25
    o['turnover_buy_per_year_x_nav'] = float(info['turnover_buy'] / yrs_ / nav.mean())
    o['cost_drag_per_year_pct_of_avg_nav'] = float((info['fee_paid'] + info['tax_paid'] + info['slip_paid']) / yrs_ / nav.mean())
    if keep_nav or name: NAVS[name or rule] = nav
    return o, nav, tr


PORT = {'candidates': {}, 'robust': {}}
TR_ALL = []
for rule in L.EXITS:
    for idle in ('CASH', 'P50'):
        nm = f'{rule}|{idle}'
        o, nav, tr = run(SIG, rule, name=nm, idle=idle)
        PORT['candidates'][nm] = o
        if idle == 'CASH' and len(tr): tr = tr.copy(); tr.insert(0, 'rule', rule); TR_ALL.append(tr)
pd.concat(TR_ALL).to_csv(HERE / 'trades.csv.gz', index=False, compression='gzip', float_format='%.8g')
PRI = 'E3_stop8_ma50'
log('12 candidates done')

rb = PORT['robust']
for rule in L.EXITS:
    rb[f'{rule}|CASH|stress'], _, _ = run(SIG, rule, cost='stress')
    rb[f'{rule}|CASH|zero_cost(回測上緣，實現不了)'], _, _ = run(SIG, rule, cost='zero')
    rb[f'{rule}|CASH|K20'], _, _ = run(SIG, rule, K=20)
    rb[f'{rule}|P50|stress'], _, _ = run(SIG, rule, cost='stress', idle='P50')
    rb[f'{rule}|P50|K20'], _, _ = run(SIG, rule, K=20, idle='P50')
    o, nav, tr = run(SIG, rule, fixed_amt=100_000.0)
    o['note'] = '每筆固定 10 萬、不限檔數；現金可為負（隱含融資、利率 0），只當對照'
    o['min_cash'] = None
    rb[f'{rule}|FIXED100k_unlimited'] = o
for cap, lab in [(300_000, '30萬'), (3_000_000, '300萬'), (10_000_000, '1000萬')]:
    for idle in ('CASH', 'P50'):
        rb[f'{PRI}|{idle}|{lab}'], _, _ = run(SIG, PRI, capital=cap, idle=idle)
        rb[f'{PRI}|{idle}|{lab}|stress'], _, _ = run(SIG, PRI, capital=cap, idle=idle, cost='stress')
for lag in (1, 2, 3):
    for idle in ('CASH', 'P50'):
        rb[f'{PRI}|{idle}|lag{lag}'], _, _ = run(SIG, PRI, lag=lag, idle=idle)
rb[f'{PRI}|CASH|buy_disposal'], _, _ = run(SIG, PRI, buy_disp=True)
rb[f'{PRI}|CASH|buy_open_limit(回測上緣，實現不了)'], _, _ = run(SIG, PRI, buy_openlim=True)
rb[f'{PRI}|CASH|bars>=660'], _, _ = run(SIG[SIG.barno >= 660], PRI)
rb[f'{PRI}|P50|bars>=660'], _, _ = run(SIG[SIG.barno >= 660], PRI, idle='P50')
rb[f'{PRI}|CASH|ex_TDR'], _, _ = run(SIG[~SIG.code.str.startswith('91')], PRI)
# 還原價訊號
RA = L.compute_signal(basis='adj'); SIGA = L.signal_table(RA, basis='adj')
ka = set(zip(SIG.code, SIG.date)); kb = set(zip(SIGA.code, SIGA.date))
only_raw = SIG[[k not in kb for k in zip(SIG.code, SIG.date)]]; only_adj = SIGA[[k not in ka for k in zip(SIGA.code, SIGA.date)]]


def exdiv_in_window(df, n=250):
    """訊號日往前 n 根內是否有除權息／參考價重設（flag 1、2、3）。"""
    out = []
    for t, j in zip(df.t.to_numpy(), df.j.to_numpy()):
        p = np.where(G.TR[:t + 1, j])[0][-n:]
        out.append(bool(np.isin(G.FLAG[p, j], [1, 2, 3]).any()))
    return np.array(out, bool)


RES['adj_basis'] = dict(n_raw=int(len(SIG)), n_adj=int(len(SIGA)), both=int(len(ka & kb)), only_raw=int(len(only_raw)), only_adj=int(len(only_adj)),
                        jaccard=float(len(ka & kb) / len(ka | kb)), by_year_adj={int(y): int(n) for y, n in SIGA.groupby(SIGA.date.dt.year).size().items()},
                        only_raw_with_exdiv_in_250bars=float(exdiv_in_window(only_raw).mean()) if len(only_raw) else None,
                        only_adj_with_exdiv_in_250bars=float(exdiv_in_window(only_adj).mean()) if len(only_adj) else None,
                        both_with_exdiv_in_250bars=float(exdiv_in_window(SIG[[k in kb for k in zip(SIG.code, SIG.date)]]).mean()),
                        raw_signals_on_exdiv_or_reset_day=int(SIG.flag_d.isin([1, 2, 3]).sum()),
                        only_adj_examples=only_adj[['date', 'code', 'close', 'pivot', 'd1', 'd2', 'd3']].head(5).astype(str).to_dict('records'))
EVA, EXA = L.event_study(SIGA)
RES['adj_basis']['event_study'] = dict(exclusions=EXA, all=L.event_summary(EVA, boot=False))
for rule in L.EXITS:
    for idle in ('CASH', 'P50'):
        rb[f'{rule}|{idle}|ADJ_signal'], _, _ = run(SIGA, rule, idle=idle)
RES['portfolio'] = PORT
yr_tab = {}
for nm in (f'{PRI}|CASH', f'{PRI}|P50', 'E6_trail10|P50', 'E5_close_below_pivot|P50'):
    v = NAVS[nm]; yr_tab[nm] = {int(y): float(g.iloc[-1] / v[v.index < g.index[0]].iloc[-1] - 1) for y, g in v.groupby(v.index.year) if len(v[v.index < g.index[0]])}
e_ = E50[E50.index >= START]; yr_tab['0050'] = {int(y): float(g.iloc[-1] / e_[e_.index < g.index[0]].iloc[-1] - 1) for y, g in e_.groupby(e_.index.year) if len(e_[e_.index < g.index[0]])}
RES['yearly_returns'] = yr_tab
last_bar = IDX.values[np.array([np.where(G.TR[:, j])[0][-1] for j in range(G.N)])]
RES['data_gate']['survivorship'] = dict(stocks_last_bar_2019_05_16=int((last_bar == np.datetime64('2019-05-16')).sum()), stocks_last_bar_before_2026_10=int((last_bar < np.datetime64('2026-10-01')).sum()),
                                        signals_in_stocks_ending_2019_05_16=int(SIG.code.isin(np.array(COLS)[last_bar == np.datetime64('2019-05-16')]).sum()),
                                        note='Asoul/tsec 到 2019-05-16；k66 只含抓取時仍上市的股票 → 2019-05～2023-12 之間下市的股票在 2019-05-16 之後沒有資料（存活者斷點）')
RES['data_gate']['signals_affected'] = dict(src3_yahoo_period=int((SIG.src_d == 3).sum()), window_spans_2024_gap=int(((SIG.date >= '2024-02-15') & (SIG.date <= '2025-03-07')).sum()),
                                            flag_e_exdiv_ret0=int((SIG.flag_e == 1).sum()))
log('robust done')

# ---------------------------------------------------------------- 參數鄰域（81 組；只看翻號，不挑參數）
NB = []
for S_, M_, V_, B_ in itertools.product((20, 25, 30), (8, 10, 12), (0.6, 0.7, 0.8), (1.3, 1.5, 2.0)):
    Rn = L.compute_signal(dict(SegLen=S_, MaxLastDepth=M_, VolDryRatio=V_, VolBreak=B_)); sg = L.signal_table(Rn, full=False)
    row = dict(SegLen=S_, MaxLastDepth=M_, VolDryRatio=V_, VolBreak=B_, n=int(len(sg)))
    for idle in ('CASH', 'P50'):
        nav, _, info = L.simulate(sg, PRI, t0=t0, idle=idle, log=False)
        for k in ('full', 'y2019_on', 'p1_2006_2012', 'p2_2013_2018'):
            st = L.period_stats(nav, E50, *PERIODS[k]); row[f'{idle}_{k}_excess'] = st['excess']; row[f'{idle}_{k}_t'] = st['t']
        row[f'{idle}_avg_invested'] = info['avg_invested']
    NB.append(row)
NBD = pd.DataFrame(NB); NBD.round(5).to_csv(HERE / 'grid_neighborhood.csv', index=False)
nbs = {}
for c in [c for c in NBD.columns if c.endswith('_excess') or c.endswith('_t') or c == 'n']:
    v = NBD[c].to_numpy()
    nbs[c] = dict(min=float(v.min()), p10=float(np.percentile(v, 10)), median=float(np.median(v)), p90=float(np.percentile(v, 90)), max=float(v.max()), share_positive=float((v > 0).mean()))
dflt = NBD[(NBD.SegLen == 25) & (NBD.MaxLastDepth == 10) & (NBD.VolDryRatio == 0.7) & (NBD.VolBreak == 1.5)].iloc[0]
one_dim = {}
for dim, vals in [('SegLen', (20, 25, 30)), ('MaxLastDepth', (8, 10, 12)), ('VolDryRatio', (0.6, 0.7, 0.8)), ('VolBreak', (1.3, 1.5, 2.0))]:
    q = NBD.copy()
    for d2, dv in [('SegLen', 25), ('MaxLastDepth', 10), ('VolDryRatio', 0.7), ('VolBreak', 1.5)]:
        if d2 != dim: q = q[q[d2] == dv]
    one_dim[dim] = {str(r[dim]): dict(n=int(r['n']), CASH_full_excess=float(r['CASH_full_excess']), P50_full_excess=float(r['P50_full_excess']),
                                      CASH_y2019_excess=float(r['CASH_y2019_on_excess']), P50_y2019_excess=float(r['P50_y2019_on_excess'])) for _, r in q.iterrows()}
RES['neighborhood'] = dict(n_combos=len(NBD), summary=nbs, one_dim=one_dim,
                           default_rank_pct={c: float((NBD[c] <= dflt[c]).mean()) for c in ('CASH_full_excess', 'P50_full_excess')},
                           sign_flip={c: bool(((NBD[c] > 0) != (dflt[c] > 0)).any()) for c in ('CASH_full_excess', 'P50_full_excess', 'CASH_y2019_on_excess', 'P50_y2019_on_excess')})
log('neighborhood done')

# ---------------------------------------------------------------- 消融（診斷；不是候選）
AB = {}
for nm, drop in [('完整（預設）', ()), ('拿掉趨勢樣板', ('trend',)), ('拿掉收縮條件', ('contract',)), ('拿掉量縮', ('voldry',)), ('拿掉突破放量', ('volbrk',)),
                 ('拿掉相對強度', ('rs',)), ('拿掉 500 張門檻', ('minvol',)), ('只留突破＋500 張（拿掉其餘五項）', ('trend', 'contract', 'voldry', 'volbrk', 'rs'))]:
    Rn = L.compute_signal(drop=drop); sg = L.signal_table(Rn, full=False)
    ev, ex = L.event_study(sg, horizons=(20, 60)); es = L.event_summary(ev, horizons=(20, 60), boot=False)
    row = dict(n=int(len(sg)), n_buyable=int(len(ev)), gap_mean=float(ev.gap_ref.mean()))
    for h in ('20', '60'):
        row[f'h{h}'] = dict(mean=es[h]['mean'], median=es[h]['median'], win=es[h]['win'], xpool=es[h]['xpool']['mean'], xpool_t=es[h]['xpool']['t'], x0050=es[h]['x0050']['mean'], x0050_t=es[h]['x0050']['t'],
                            months=es[h]['xpool']['G'], neff=es[h]['xpool']['neff'])
    for idle in ('CASH', 'P50'):
        nav, _, info = L.simulate(sg, PRI, t0=t0, idle=idle, log=False)
        for k in ('full', 'y2019_on'):
            st = L.period_stats(nav, E50, *PERIODS[k]); row[f'{idle}_{k}'] = dict(cagr=st['cagr'], excess=st['excess'], t=st['t'], maxdd=st['maxdd'])
        row[f'{idle}_avg_invested'] = info['avg_invested']
    AB[nm] = row
RES['ablation'] = AB
log('ablation done')

# ---------------------------------------------------------------- 對貼文的評語用的診斷
def seg_diag(code, date, S=25):
    j = G.CI[code]; m = G.TR[:, j]; H = G.H[m, j]; Lo = G.L[m, j]; i = IDX[m].get_loc(pd.Timestamp(date)); o = {}
    for q, k0 in (('3', 1), ('2', S + 1), ('1', 2 * S + 1)):
        h = H[i - k0 - S + 1:i - k0 + 1]; l = Lo[i - k0 - S + 1:i - k0 + 1]
        rm = np.maximum.accumulate(h); o['pb' + q] = float(((rm - l) / rm).max() * 100)          # 真正的回檔：先高後低的最大跌幅
        fh = np.maximum.accumulate(h[::-1])[::-1]; o['ru' + q] = float(((fh - np.minimum.accumulate(l)) / fh).max() * 100)   # 先低後高的最大漲幅（同樣以高點為分母，才能和 pb 比）
        o['low_before_high' + q] = bool(np.argmin(l) < np.argmax(h))
        o['rng' + q] = float((h.max() - l.min()) / h.max() * 100)
    return o


SD = pd.DataFrame([seg_diag(c, d) for c, d in zip(SIG.code, SIG.date)])
SG = pd.concat([SIG.reset_index(drop=True), SD], axis=1)
hmax12 = np.maximum(SG.h1, SG.h2)
CR = dict(
    n=int(len(SG)),
    true_pullback_contracting_share=float(((SG.pb1 > SG.pb2) & (SG.pb2 > SG.pb3)).mean()),
    true_pullback_contracting_and_first_ge10_last_le10=float(((SG.pb1 > SG.pb2) & (SG.pb2 > SG.pb3) & (SG.pb1 >= 10) & (SG.pb3 <= 10)).mean()),
    seg1_true_pullback_lt_10pct_share=float((SG.pb1 < 10).mean()),
    seg1_low_before_high_share=float(SG.low_before_high1.mean()),
    seg1_range_is_mostly_runup_share=float((SG.ru1 > SG.pb1).mean()),
    seg2_range_is_mostly_runup_share=float((SG.ru2 > SG.pb2).mean()),
    seg3_range_is_mostly_runup_share=float((SG.ru3 > SG.pb3).mean()),
    median_pb=[float(SG.pb1.median()), float(SG.pb2.median()), float(SG.pb3.median())], median_range=[float(SG.d1.median()), float(SG.d2.median()), float(SG.d3.median())],
    pivot_below_earlier_high_share=float((SG.h3 < hmax12).mean()), pivot_below_earlier_high_by_gt5pct_share=float((SG.h3 < hmax12 * 0.95).mean()),
    close_still_below_75bar_high_share=float((SG.close < hmax12).mean()),
    stairstep_up_share=float(((SG.l1 < SG.l2) & (SG.l2 < SG.l3) & (SG.h1 < SG.h2) & (SG.h2 < SG.h3)).mean()),
    net_rise_over_75_bars_median=float((SG.h3 / SG.h1 - 1).median()),
)
ex_ = SG[(SG.ru1 > SG.pb1) & (SG.pb1 < 10)].head(3); CR['typical_misclassified_examples'] = ex_[['date', 'code', 'd1', 'd2', 'd3', 'pb1', 'pb2', 'pb3', 'ru1', 'h1', 'l1', 'h2', 'l2', 'h3', 'l3']].round(2).astype(str).to_dict('records')
# 500 張門檻
V = G.VOL / 1000.0; ok260 = G.TR & (G.BARNO >= 260)
by = {}
for a, b in [(2005, 2012), (2013, 2018), (2019, 2023), (2024, 2026)]:
    m = ok260 & ((yrs >= a) & (yrs <= b))[:, None]
    medv = np.where(m, V, np.nan); med = np.nanmedian(medv, axis=0); med = med[np.isfinite(med)]
    by[f'{a}-{b}'] = dict(stock_days_ge500_share=float((m & (V >= 500)).sum() / m.sum()), stocks_median_vol_ge500_share=float((med >= 500).mean()), n_stocks=int(len(med)))
CR['minvol500'] = dict(by_period=by, signals_with_minvol=int(len(SIG)), signals_without_minvol=AB['拿掉 500 張門檻']['n'],
                       removed_share=float(1 - len(SIG) / AB['拿掉 500 張門檻']['n']),
                       median_value_ntd_at_signal=float(np.nanmedian(G.VAL[SIG.t.to_numpy(), SIG.j.to_numpy()])),
                       p10_value_ntd_at_signal=float(np.nanpercentile(G.VAL[SIG.t.to_numpy(), SIG.j.to_numpy()], 10)),
                       median_price_at_signal=float(SIG.close.median()))
# 相對強度：腳本的 250 日漲幅用原始收盤（除權息會壓低個股漲幅）
# 診斷：真正逐段變淺的訊號 vs 其餘訊號的事件報酬（不是候選）
tc = (SG.pb1 > SG.pb2) & (SG.pb2 > SG.pb3) & (SG.pb1 >= 10) & (SG.pb3 <= 10)
keyc = set(zip(SG.code[tc], SG.date[tc])); mk = np.array([k in keyc for k in zip(EV.code, EV.date)])
CR['event_true_contracting'] = {h: {kk: vv for kk, vv in L.event_summary(EV[mk], horizons=(20, 60), boot=False)[h].items() if kk in ('n', 'mean', 'median', 'win', 'xpool', 'x0050')} for h in ('20', '60')}
CR['event_not_true_contracting'] = {h: {kk: vv for kk, vv in L.event_summary(EV[~mk], horizons=(20, 60), boot=False)[h].items() if kk in ('n', 'mean', 'median', 'win', 'xpool', 'x0050')} for h in ('20', '60')}
RES['critique'] = CR
log('critique done')

# ---------------------------------------------------------------- 資料窺探（G3）
cn = list(PORT['candidates'])
MX = pd.DataFrame({k: (L.month_ret(NAVS[k]) - L.month_ret(E50)) for k in cn}).dropna()
MX = MX[MX.index >= pd.Period(START, 'M') + 1]
srs = MX.mean() / MX.std(ddof=1)
spa = L.spa_test(MX.to_numpy())
best = srs.idxmax()
RES['snooping'] = dict(N_before=PRE['N_before'], N_added=len(cn), N_after=PRE['N_before'] + len(cn), spa_family12=spa, monthly_sr={k: float(v) for k, v in srs.items()},
                       var_sr_across_12=float(srs.var(ddof=1)), best_by_full_sr=best,
                       dsr_primary=L.dsr(MX[f'{PRI}|CASH'].to_numpy(), PRE['N_before'] + len(cn), float(srs.var(ddof=1))),
                       dsr_best=L.dsr(MX[best].to_numpy(), PRE['N_before'] + len(cn), float(srs.var(ddof=1))),
                       t_full={k: L.tstat(MX[k].to_numpy()) for k in cn})
# 只用挑選期選出場規則 → 驗證期、保留期
sel_rank = sorted(cn, key=lambda k: -PORT['candidates'][k]['sel_2006_2014']['excess'])
RES['selection_protocol'] = dict(ranking_by_selection_excess=[(k, PORT['candidates'][k]['sel_2006_2014']['excess']) for k in sel_rank],
                                 chosen=sel_rank[0], chosen_val=PORT['candidates'][sel_rank[0]]['val_2015_2020'], chosen_hold=PORT['candidates'][sel_rank[0]]['hold_2021_2026'],
                                 chosen_cash_only=[k for k in sel_rank if k.endswith('CASH')][0])
log('snooping done')

# ---------------------------------------------------------------- 與主策略比較
MN = pd.read_csv(L.MAINNAV, index_col=0, parse_dates=True)
cols = [f'base|REAL_opt|{j}|normal|100萬|std' for j in range(13)]
mn = MN[cols].dropna(how='all')
main_nav = (mn / mn.bfill().iloc[0]).mean(axis=1).reindex(IDX).dropna()          # 13 個 j 等資金
a19 = main_nav.index[main_nav.index >= '2019-01-01'][0]
per_j = [L.cagr(mn[c].dropna(), mn[c].dropna().index[mn[c].dropna().index >= '2019-01-01'][0], mn[c].dropna().index[-1]) for c in cols]
VS = dict(main_series='integrate_o20/data/v7x_navs_jmean_100w.csv.gz 的 base|REAL_opt|j|normal|100萬|std，13 個 j 等資金平均淨值',
          main_cagr_2019_on_mean_nav=L.cagr(main_nav, a19, main_nav.index[-1]), main_cagr_2019_on_mean_of_j=float(np.mean(per_j)),
          official_main_2019_on=0.2327, official_0050_2019_on=0.2949, note='正式數字是逐格取 opt／p90／carry 最差並用各 j 自己的終點日；這裡的序列只用來算相關與混合')
mm = L.month_ret(main_nav); me = L.month_ret(E50)
for nm in (f'{PRI}|CASH', f'{PRI}|P50', 'E2_hold60|CASH', 'E2_hold60|P50'):
    v = NAVS[nm]; mv = L.month_ret(v)
    d = pd.concat([mv, mm, me], axis=1, keys=['v', 'm', 'e']).dropna()
    o = {}
    for lab, dd in (('common_2006-05_on', d), ('y2019_on', d[d.index >= '2019-01'])):
        bl = 0.5 * dd.v + 0.5 * dd.m; n = len(dd); ann = lambda r: float((1 + r).prod() ** (12 / len(r)) - 1)
        dd_ = lambda r: float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min())
        o[lab] = dict(n_months=n, corr_returns=float(dd.v.corr(dd.m)), corr_excess=float((dd.v - dd.e).corr(dd.m - dd.e)),
                      vcp_ann=ann(dd.v), main_ann=ann(dd.m), b0050_ann=ann(dd.e), blend_ann=ann(bl),
                      blend_excess_t=L.tstat((bl - dd.e).to_numpy()), main_excess_t=L.tstat((dd.m - dd.e).to_numpy()), vcp_excess_t=L.tstat((dd.v - dd.e).to_numpy()),
                      blend_maxdd_m=dd_(bl), main_maxdd_m=dd_(dd.m), vcp_maxdd_m=dd_(dd.v), b0050_maxdd_m=dd_(dd.e),
                      blend_sharpe=float(bl.mean() / bl.std() * math.sqrt(12)), main_sharpe=float(dd.m.mean() / dd.m.std() * math.sqrt(12)), b0050_sharpe=float(dd.e.mean() / dd.e.std() * math.sqrt(12)))
    VS[nm] = o
RES['vs_main'] = VS

# ---------------------------------------------------------------- G1～G8
c = PORT['candidates']; pr = c[f'{PRI}|CASH']; pp = c[f'{PRI}|P50']
lag_ex = [pr['full']['excess']] + [rb[f'{PRI}|CASH|lag{k}']['full']['excess'] for k in (1, 2, 3)]
lag_ex_p = [pp['full']['excess']] + [rb[f'{PRI}|P50|lag{k}']['full']['excess'] for k in (1, 2, 3)]
RES['gates'] = dict(
    G1=dict(normal=pr['full']['excess'], stress=rb[f'{PRI}|CASH|stress']['full']['excess'], lag_excess=lag_ex, lag_median=float(np.median(lag_ex)),
            P50_normal=pp['full']['excess'], P50_stress=rb[f'{PRI}|P50|stress']['full']['excess'], P50_lag_excess=lag_ex_p,
            passed=bool(pr['full']['excess'] > 0 and rb[f'{PRI}|CASH|stress']['full']['excess'] > 0 and np.median(lag_ex) > 0)),
    G2=dict(t_full=pr['full']['t'], t_2019=pr['y2019_on']['t'], P50_t_full=pp['full']['t'], P50_t_2019=pp['y2019_on']['t'], passed=bool(pr['full']['t'] >= 3 and pr['y2019_on']['t'] >= 2)),
    G3=dict(spa_p=spa['p_consistent'], dsr_primary=RES['snooping']['dsr_primary']['dsr'], dsr_best=RES['snooping']['dsr_best']['dsr'], N=PRE['N_before'] + len(cn),
            passed=bool(spa['p_consistent'] < 0.05 and RES['snooping']['dsr_primary']['dsr'] >= 0.95)),
    G4=dict(three=[pr['p1_2006_2012']['excess'], pr['p2_2013_2018']['excess'], pr['y2019_on']['excess']], t_ex_best3=pr['ex_best3']['t_ex_best3'], best3=pr['ex_best3']['best3_years'],
            neighborhood_sign_flip=RES['neighborhood']['sign_flip'],
            P50_three=[pp['p1_2006_2012']['excess'], pp['p2_2013_2018']['excess'], pp['y2019_on']['excess']], P50_t_ex_best3=pp['ex_best3']['t_ex_best3'],
            passed=bool(min(pr['p1_2006_2012']['excess'], pr['p2_2013_2018']['excess'], pr['y2019_on']['excess']) > 0 and pr['ex_best3']['t_ex_best3'] >= 2 and not RES['neighborhood']['sign_flip']['CASH_full_excess'])),
    G7=dict(excess_100w=pr['full']['excess'], excess_300w=rb[f'{PRI}|CASH|300萬']['full']['excess'], excess_1000w=rb[f'{PRI}|CASH|1000萬']['full']['excess'], excess_30w=rb[f'{PRI}|CASH|30萬']['full']['excess'],
            P50=[pp['full']['excess'], rb[f'{PRI}|P50|300萬']['full']['excess'], rb[f'{PRI}|P50|1000萬']['full']['excess'], rb[f'{PRI}|P50|30萬']['full']['excess']],
            passed=bool(min(pr['full']['excess'], rb[f'{PRI}|CASH|300萬']['full']['excess'], rb[f'{PRI}|CASH|1000萬']['full']['excess']) > 0)))
RES['runtime_sec'] = round(time.time() - T0, 1)
RES['files'] = {f: L.sha(HERE / f) for f in ('vcplib.py', 'run.py', 'prereg.py', 'signals.csv.gz', 'trades.csv.gz', 'grid_neighborhood.csv', 'raw/vcp_alert.xs', 'raw/SOURCE.txt')}
(HERE / 'results.json').write_text(json.dumps(rnd(RES), ensure_ascii=False, indent=1), encoding='utf-8')
pd.DataFrame(NAVS).round(2).to_csv(HERE / 'navs.csv.gz', compression='gzip')
log('all done')
