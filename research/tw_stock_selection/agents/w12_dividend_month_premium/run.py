#!/usr/bin/env python3
"""w12 預期配息月溢酬（Hartzmark–Solomon 2013）：依 results.json 的 PREREG 執行，不改 PREREG。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild'))
import exlib as L  # 只讀：面板、cost_rates、LTOL

RES = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert 'PREREG' in RES
OUT = {'PREREG': RES['PREREG']}
OUT['PREREG_post_note'] = ('看結果後的資料更正（不改任何參數）：PREREG why_main_differs 說 2004 年 882 個 flag 日只抓到 460 個，'
    '原因其實大半是 flag 1 也標記「前一日無成交」的復牌日（前收缺值，2004～2023 每年約 400～990 個），不是除息日；'
    '前收存在的 flag 1 日中，逐字規則只漏掉約 8.6%（當日收漲者），main 漏約 4%。兩者結論相同。')

# ---------------- 資料 ----------------
adj0 = L.adj; C0 = L.C; O0 = L.O; R0 = L.RET; V0 = L.V
F0 = pd.read_parquet(L.P / 'flag.parquet').reindex(index=adj0.index, columns=adj0.columns)
GOOD = adj0.notna().sum(axis=1) >= 300
OUT['sparse_days_removed'] = [str(d.date()) for d in adj0.index[~GOOD]]
adj = adj0[GOOD]; C = C0[GOOD]; O = O0[GOOD]; RET = R0[GOOD]; V = V0[GOOD]; FL = F0[GOOD]
IDX = adj.index; COLS = adj.columns; N = len(COLS); NT = len(IDX)
CI = {c: j for j, c in enumerate(COLS)}
is_common = np.array([not c.startswith('91') for c in COLS])
Cn = C.to_numpy(); On = O.to_numpy(); Rn = RET.to_numpy(); Fn = FL.to_numpy(); ADJn = adj.to_numpy()
Cprev = np.vstack([np.full((1, N), np.nan), Cn[:-1]])
with np.errstate(all='ignore'):
    raw = Cn / Cprev - 1
    d_fac = Rn - raw                       # ret − 原始漲跌（因子跳動）
    normal = (Fn == 0) & np.isfinite(Cprev) & np.isfinite(On)
    mgap = np.nanmedian(np.where(normal, On / Cprev - 1, np.nan), axis=1)
    E3 = 1 - On / (Cprev * (1 + mgap[:, None]))
    E1 = 1 - Cn / Cprev
isX = np.isin(Fn, [1, 2])
y_main = np.where(np.isfinite(Rn) & (Rn != 0), d_fac, np.where(np.isfinite(On), E3, E1))
EV = {
    'main': isX & (y_main > 0.003),
    'literal': (d_fac > 0.003) & ~np.isin(Fn, [3, 4, 5]),
}
# official：2021-10-07 起用 exdiv_5y
CUT = pd.Timestamp('2021-10-07')
ex = pd.read_csv(L.X / 'exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
ex = ex[(ex['market'] == 'TWSE') & ex['code'].isin(COLS) & (ex['date'] >= CUT)].copy()
exc = ex[ex['kind'].isin(['息', '權息']) & (ex['value'] / ex['prev_close'] > 0.003)]
OFF = np.zeros((NT, N), bool)
pi = IDX.searchsorted(exc['date'].values)
ok = (pi < NT) & (IDX[np.minimum(pi, NT - 1)] == exc['date'].values)   # 只保留面板有的日期（2024-01-02～02-14 缺資料期間的事件不移到 02-15）
OUT['official_events_dropped_not_in_panel'] = int((~ok).sum())
OFF[pi[ok], [CI[c] for c in exc['code'].values[ok]]] = True
post = (IDX >= CUT)[:, None]
EV['official'] = np.where(post, OFF, EV['main'])

# 校對（2021-10-07 起，exdiv_5y 為真值；同日 ±0 交易日）
ALLX = np.zeros((NT, N), bool)
pa = IDX.searchsorted(ex['date'].values); oka = pa < NT
ALLX[pa[oka], [CI[c] for c in ex['code'].values[oka]]] = True
calib = {}
for k in ('main', 'literal'):
    m = EV[k] & post; t = OFF & post
    calib[k] = {'detected': int(m.sum()), 'official_cash_events': int(t.sum()),
                'recall_vs_cash': round(float((m & t).sum() / t.sum()), 3),
                'precision_vs_cash': round(float((m & t).sum() / m.sum()), 3),
                'precision_vs_any_exright': round(float((m & ALLX).sum() / m.sum()), 3)}
    for a, b in (('2021-10-07', '2023-12-31'), ('2024-01-01', '2026-12-31')):
        w = ((IDX >= a) & (IDX <= b))[:, None]
        mm = m & w; tt = t & w
        calib[k][f'recall_{a[:4]}_{b[:4]}'] = round(float((mm & tt).sum() / max(tt.sum(), 1)), 3)
        calib[k][f'precision_{a[:4]}_{b[:4]}'] = round(float((mm & tt).sum() / max(mm.sum(), 1)), 3)
# 前期：literal 抓到的 flag 日比例
pre = (IDX < CUT)[:, None]
calib['pre2021_flag12_days'] = int((isX & pre).sum())
calib['pre2021_main_events'] = int((EV['main'] & pre).sum())
calib['pre2021_literal_events'] = int((EV['literal'] & pre).sum())
calib['pre2021_literal_also_main'] = int((EV['literal'] & EV['main'] & pre).sum())
OUT['calibration'] = calib

# ---------------- 價格（含 EXFIX 版本） ----------------
def price_arrays(fix):
    a = adj.copy()
    if fix:
        with np.errstate(all='ignore'):
            rf = Cn / Cprev - On / (Cprev * (1 + mgap[:, None]))
        m = isX & (Rn == 0) & np.isfinite(rf)
        mult = np.where(m, 1 + rf, 1.0)
        a = a * np.cumprod(mult, axis=0)
        OUT.setdefault('exfix_info', {'n_days_fixed': int(m.sum()), 'mean_fixed_ret_pct': round(float(np.nanmean(rf[m]) * 100), 3)})
    AC = a.ffill().to_numpy()
    AOr = (O * a / C).to_numpy()
    return AC, AOr
thr = np.where(IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None] - L.LTOL

V60 = V.rolling(60, min_periods=50).mean().to_numpy()

# 0050
e = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
e = e.drop(index=[d for d in (pd.Timestamp('2026-07-10'), pd.Timestamp('2026-09-20')) if d in e.index])
E50 = e.reindex(e.index.union(IDX)).ffill().reindex(IDX).to_numpy()

# ---------------- 月曆 ----------------
mon = IDX.to_period('M')
mpos = pd.Series(np.arange(NT), index=mon)
first = mpos.groupby(level=0).min(); last = mpos.groupby(level=0).max()
MONTHS = list(first.index)
allp = pd.period_range(MONTHS[0], MONTHS[-1], freq='M')
PI = {p: k for k, p in enumerate(allp)}

def month_event_matrix(evm):
    M = np.zeros((len(allp), N), bool)
    for p in MONTHS:
        M[PI[p]] = evm[first[p]:last[p] + 1].any(0)
    return M

B_BUY, B_SELL = L.cost_rates(0.001)
S_BUY, S_SELL = L.cost_rates('stress')

def run(det='main', fix=False, quarterly=True):
    EM = month_event_matrix(EV[det])
    AC, AOr = price_arrays(fix)
    ACprev = np.vstack([np.full((1, N), np.nan), AC[:-1]])
    rows = []; paths = {}
    for p in MONTHS:
        if p < pd.Period('2005-01', 'M'): continue
        en = first[p]; xt = last[p]; sg = en - 1
        if sg < 0: continue
        k = PI[p]
        def em(lag):
            return EM[k - lag] if k - lag >= 0 else np.zeros(N, bool)
        hist = np.array([em(l) for l in range(1, 13)])   # t-1..t-12
        q = hist.sum(0) >= 3
        pred = em(12) | (quarterly & q & (em(3) | em(6) | em(9)))
        payer = hist.any(0)
        uni = (is_common & (V60[sg] >= 1e7) & (Cn[sg] >= 10) & (Fn[sg] != 5)
               & np.isfinite(On[en]) & np.isfinite(ADJn[en]) & np.isfinite(AOr[en]))
        gap = (AOr[en] / ACprev[en] - 1) >= thr[en]
        buyable = uni & ~gap
        r = AC[xt] / AOr[en] - 1
        gL = buyable & pred; ga = uni & payer & ~pred; gb = uni; greal = buyable & EM[k]
        def m_(g): return float(np.nanmean(r[g])) if g.sum() else np.nan
        row = {'m': p, 'entry': IDX[en], 'exit': IDX[xt], 'nL': int(gL.sum()), 'nL_blocked': int((uni & pred & gap).sum()),
               'na': int(ga.sum()), 'nb': int(gb.sum()), 'n_real': int(greal.sum()),
               'hit': float((gL & EM[k]).sum() / gL.sum()) if gL.sum() else np.nan,
               'cover': float((gL & EM[k]).sum() / greal.sum()) if greal.sum() else np.nan,
               'rL': m_(gL), 'ra': m_(ga), 'rb': m_(gb), 'rReal': m_(greal), 'rRealNotPred': m_(buyable & EM[k] & ~pred),
               'rPredNotReal': m_(gL & ~EM[k]),
               'r50': E50[xt] / E50[sg] - 1}
        rows.append(row)
        # 日頻路徑（月內等權買入持有）
        segL = np.nanmean(AC[en:xt + 1][:, gL] / AOr[en][gL], axis=1) if gL.sum() else None
        segb = np.nanmean(AC[en:xt + 1][:, gb] / AOr[en][gb], axis=1)
        paths[p] = (np.arange(en, xt + 1), segL, segb)
    df = pd.DataFrame(rows).set_index('m')
    df['active'] = df['nL'] >= 5
    for nm, (bb, ss) in {'net': (B_BUY, B_SELL), 'stress': (S_BUY, S_SELL), 'rt05': (0.0025, 0.0025)}.items():
        df[f'rL_{nm}'] = (1 - bb) * (1 + df['rL']) * (1 - ss) - 1
    return df, paths, AC

def tstat(x):
    x = pd.Series(x).dropna()
    return round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2) if len(x) > 2 and x.std() > 0 else None

def pct(x): return round(float(x) * 100, 3) if pd.notna(x) else None

PERIODS = {'select_2006-01_2019-04': ('2006-01', '2019-04'), 'valid_2019-05_2026-10': ('2019-05', '2026-10'),
           'sub_2024-01_2025-12': ('2024-01', '2025-12'), 'full_2006-01_2026-10': ('2006-01', '2026-10'),
           'post_2021-10_2023-12': ('2021-10', '2023-12'), 'post_2024-01_2026-10': ('2024-01', '2026-10'),
           'valid_ex_src3': ('2019-05', '2026-10')}
SRC3 = (pd.Period('2024-02', 'M'), pd.Period('2025-09', 'M'))

def sel(df, a, b, name=''):
    s = df[(df.index >= pd.Period(a, 'M')) & (df.index <= pd.Period(b, 'M'))]
    if name == 'valid_ex_src3': s = s[(s.index < SRC3[0]) | (s.index > SRC3[1])]
    return s

def nav_stats(df, paths, a, b, rest='cash'):
    """只做多組合：L ≥ 5 檔月份持有 L（扣主成本），其餘持現金（或 0050）。日頻 NAV。"""
    s = sel(df, a, b)
    nav = []; v = 1.0; navb = []; vb = 1.0; n50 = []
    for p, row in s.iterrows():
        idxs, segL, segb = paths[p]
        if row['active'] and segL is not None:
            seq = v * (1 - B_BUY) * segL; seq[-1] *= (1 - B_SELL)
        elif rest == '0050':
            sg = idxs[0] - 1; seq = v * E50[idxs] / E50[sg]
        else:
            seq = np.full(len(idxs), v)
        nav.extend(zip(IDX[idxs], seq)); v = seq[-1]
        sb = vb * segb; navb.extend(zip(IDX[idxs], sb)); vb = sb[-1]
    nav = pd.Series(dict(nav)); navb = pd.Series(dict(navb))
    yrs = (s['exit'].iloc[-1] - (IDX[IDX.get_loc(s['entry'].iloc[0]) - 1])).days / 365.25
    e50 = pd.Series(E50[[IDX.get_loc(d) for d in nav.index]], index=nav.index)
    sg0 = IDX.get_loc(s['entry'].iloc[0]) - 1
    e50v = e50 / E50[sg0]
    def cg(x, base=1.0): return round(float((x.iloc[-1] / base) ** (1 / yrs) - 1), 4)
    def dd(x): x = pd.concat([pd.Series([1.0]), x.reset_index(drop=True)]); return round(float((x / x.cummax() - 1).min()), 4)
    if rest == '0050':
        mret = s.apply(lambda r: r['rL_net'] if r['active'] else r['r50'], axis=1)
    else:
        mret = s['rL_net'].where(s['active'], 0.0)
    o = {'CAGR_net': cg(nav), 'maxDD_daily': dd(nav), 'EW_pool_CAGR': cg(navb), 'EW_pool_maxDD': dd(navb),
         '0050_CAGR': cg(e50v), '0050_maxDD': dd(e50v), 'share_months_invested': round(float(s['active'].mean()), 3), 'years': round(yrs, 2),
         'month_excess_vs_EW_pct': pct((mret - s['rb']).mean()), 't_vs_EW_m': tstat(mret - s['rb']),
         'month_excess_vs_0050_pct': pct((mret - s['r50']).mean()), 't_vs_0050_m': tstat(mret - s['r50'])}
    ys = {}
    for y, g in s.groupby(s.index.year):
        mr = mret.loc[g.index]
        ys[int(y)] = {'L_net': round(float((1 + mr).prod() - 1) * 100, 1), 'EW_pool': round(float((1 + g['rb']).prod() - 1) * 100, 1),
                      '0050': round(float((1 + g['r50']).prod() - 1) * 100, 1)}
    o['yearly_pct'] = ys
    return o

def ev_stats(s):
    s = s[s['active']]
    if len(s) < 3: return {'n_months': int(len(s))}
    o = {'n_months': int(len(s)), 'n_stock_months_L': int(s['nL'].sum()), 'avg_nL': round(float(s['nL'].mean()), 1),
         'avg_na': round(float(s['na'].mean()), 1), 'avg_nb': round(float(s['nb'].mean()), 1),
         'hit_rate_pred_paid': round(float((s['hit'] * s['nL']).sum() / s['nL'].sum()), 3),
         'blocked_gap_up': int(s['nL_blocked'].sum()),
         'L_gross_pct': pct(s['rL'].mean()), 'a_pct': pct(s['ra'].mean()), 'b_pct': pct(s['rb'].mean()), '0050_pct': pct(s['r50'].mean()),
         'LS_L_minus_a_gross_pct': pct((s['rL'] - s['ra']).mean()), 't_L_minus_a': tstat(s['rL'] - s['ra']),
         'L_minus_b_gross_pct': pct((s['rL'] - s['rb']).mean()), 't_L_minus_b_gross': tstat(s['rL'] - s['rb']),
         'L_minus_b_net_pct': pct((s['rL_net'] - s['rb']).mean()), 't_L_minus_b_net': tstat(s['rL_net'] - s['rb']),
         'L_minus_b_stress_pct': pct((s['rL_stress'] - s['rb']).mean()), 't_L_minus_b_stress': tstat(s['rL_stress'] - s['rb']),
         'L_minus_b_rt05_pct': pct((s['rL_rt05'] - s['rb']).mean()), 't_L_minus_b_rt05': tstat(s['rL_rt05'] - s['rb']),
         'L_minus_0050_net_pct': pct((s['rL_net'] - s['r50']).mean()), 't_L_minus_0050_net': tstat(s['rL_net'] - s['r50']),
         'event_weighted_L_minus_b_net_pct': pct(((s['rL_net'] - s['rb']) * s['nL']).sum() / s['nL'].sum()),
         'realized_minus_b_gross_pct_DIAG': pct((s['rReal'] - s['rb']).mean()), 't_realized_minus_b_DIAG': tstat(s['rReal'] - s['rb']),
         'pred_not_paid_minus_b_gross_pct_DIAG': pct((s['rPredNotReal'] - s['rb']).mean()),
         'paid_not_pred_minus_b_gross_pct_DIAG': pct((s['rRealNotPred'] - s['rb']).mean()),
         'hit_L_beats_b': round(float((s['rL_net'] > s['rb']).mean()), 3)}
    return o

def report(df, paths, with_nav=True):
    o = {}
    for nm, (a, b) in PERIODS.items():
        s = sel(df, a, b, nm)
        r = {'all_months': ev_stats(s),
             'jun_sep': ev_stats(s[s.index.month.isin([6, 7, 8, 9])]),
             'other_months': ev_stats(s[~s.index.month.isin([6, 7, 8, 9])])}
        if with_nav and nm != 'valid_ex_src3':
            r['long_only_cash'] = nav_stats(df, paths, a, b, 'cash')
            r['long_only_rest0050'] = nav_stats(df, paths, a, b, '0050')
        o[nm] = r
    # 每月份（1～12 月）
    cm = {}
    for nm, (a, b) in list(PERIODS.items())[:2]:
        s = sel(df, a, b)
        cm[nm] = {int(mo): {'n_months': int(len(g)), 'active': int(g['active'].sum()), 'avg_nL': round(float(g['nL'].mean()), 1),
                            'avg_na': round(float(g['na'].mean()), 1),
                            'L_minus_a_gross_pct': pct((g['rL'] - g['ra'])[g['active']].mean()),
                            'L_minus_b_net_pct': pct((g['rL_net'] - g['rb'])[g['active']].mean()),
                            't_L_minus_b_net': tstat((g['rL_net'] - g['rb'])[g['active']])}
                  for mo, g in s.groupby(s.index.month)}
    o['by_calendar_month'] = cm
    # 逐年事件（L−b、L−a）
    s = sel(df, '2006-01', '2026-10'); s = s[s['active']]
    o['yearly_event'] = {int(y): {'months': int(len(g)), 'avg_nL': round(float(g['nL'].mean()), 1),
                                  'L_minus_b_net_pct_m': pct((g['rL_net'] - g['rb']).mean()),
                                  'L_minus_a_gross_pct_m': pct((g['rL'] - g['ra']).mean())}
                         for y, g in s.groupby(s.index.year)}
    return o

VARIANTS = {'main': dict(det='main'), 'literal': dict(det='literal'), 'official': dict(det='official'),
            'annual_only': dict(det='main', quarterly=False), 'exfix': dict(det='main', fix=True)}
monthly_dump = None
for vn, kw in VARIANTS.items():
    df, paths, _ = run(**kw)
    OUT[vn] = report(df, paths, with_nav=True)
    if vn == 'main':
        monthly_dump = df
    print(vn, 'done', flush=True)
    for nm in ('select_2006-01_2019-04', 'valid_2019-05_2026-10', 'sub_2024-01_2025-12'):
        x = OUT[vn][nm]['all_months']; y = OUT[vn][nm]['long_only_cash']
        print(' ', nm, 'n', x.get('n_months'), 'nL', x.get('avg_nL'), 'hit', x.get('hit_rate_pred_paid'),
              'L-a', x.get('LS_L_minus_a_gross_pct'), x.get('t_L_minus_a'),
              'L-b net', x.get('L_minus_b_net_pct'), x.get('t_L_minus_b_net'),
              'L-50 net', x.get('L_minus_0050_net_pct'), x.get('t_L_minus_0050_net'),
              '| CAGR', y['CAGR_net'], y['EW_pool_CAGR'], y['0050_CAGR'], 'DD', y['maxDD_daily'])

md = monthly_dump.copy(); md.index = md.index.astype(str)
md.to_csv(HERE / 'monthly_main.csv', float_format='%.6f')
OUT['monthly_counts_main'] = {k: {'nL': int(r['nL']), 'na': int(r['na']), 'nb': int(r['nb']), 'n_real': int(r['n_real'])} for k, r in md.iterrows()}
(HERE / 'results.json').write_text(json.dumps(OUT, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(OUT['calibration'], ensure_ascii=False))
