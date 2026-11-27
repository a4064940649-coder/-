# -*- coding: utf-8 -*-
"""R107 獨立複製：由淨值與逐筆檔重算 SURGE 完整報告的核心數字。
未閱讀 build_report.py / diagnose.py；公式依第三方規格書 §四、§六、§七、§七點五 與改編版規格書。
執行：python3 -I -B replicate.py   → 寫 my_numbers.json
"""
import json
import math
import os

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(HERE, '..', 's75_continuous_2008_2026', 'v3_2026-11-25_limit7'))
NAV = os.path.join(BASE, 'nav_2008_2026_v3.csv.gz')
TRADES = os.path.join(BASE, 'trades_v3.csv.gz')
# 路徑常數取自 agents/o_ext_canslim_xs/cslib.py 的 TWII_CSV
TWII_CSV = '/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv'
CAP = 1_000_000.0
ANN = 252


# ---------------- 工具（自寫） ----------------
def sr_daily(x):
    x = np.asarray(x, float)
    return x.mean() / x.std(ddof=1)


def sr_ann(x):
    return sr_daily(x) * math.sqrt(ANN)


def skew_kurt(x):
    x = np.asarray(x, float)
    return float(stats.skew(x, bias=False)), float(stats.kurtosis(x, bias=False))


def hac_t(x):
    """Newey-West HAC t（常數迴歸），maxlags = int(n**0.25)。statsmodels 與手算各一。"""
    x = np.asarray(x, float)
    n = len(x)
    L = int(n ** 0.25)
    m = sm.OLS(x, np.ones(n)).fit(cov_type='HAC', cov_kwds={'maxlags': L})
    # 手算（Bartlett 核，statsmodels 預設含小樣本無修正）
    e = x - x.mean()
    s = (e @ e) / n
    for k in range(1, L + 1):
        w = 1 - k / (L + 1)
        s += 2 * w * (e[k:] @ e[:-k]) / n
    t_manual = x.mean() / math.sqrt(s / n)
    return float(m.tvalues[0]), float(m.pvalues[0]), float(t_manual), L


def var_sr(sr, sk, ku, n):
    return (1 - sk * sr + ku / 4 * sr ** 2) / (n - 1)


def lobato_ci(x, alpha=0.05):
    x = np.asarray(x, float); n = len(x)
    sr = sr_daily(x); sk, ku = skew_kurt(x)
    se = math.sqrt(max(var_sr(sr, sk, ku, n), 0))
    z = stats.norm.ppf(1 - alpha / 2)
    a = math.sqrt(ANN)
    return sr * a, (sr - z * se) * a, (sr + z * se) * a


def psr(x, target=0.0):
    x = np.asarray(x, float); n = len(x)
    sr = sr_daily(x); sk, ku = skew_kurt(x)
    z = (sr - target) * math.sqrt(n - 1) / math.sqrt(1 - sk * sr + ku / 4 * sr ** 2)
    return float(stats.norm.cdf(z))


def bootstrap_sr(x, n_boot=5000, seed=42):
    x = np.asarray(x, float); n = len(x)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for i in range(n_boot):
        s = x[rng.integers(0, n, size=n)]
        sd = s.std(ddof=1)
        out[i] = s.mean() / sd * math.sqrt(ANN) if sd > 0 else 0.0
    return out


def expected_max(N):
    if N <= 1:
        return 0.0
    a = math.sqrt(2 * math.log(N))
    return a - (math.log(math.log(max(N, 3))) + math.log(4 * math.pi)) / (2 * a)


def dsr(x, N):
    x = np.asarray(x, float); n = len(x)
    sr = sr_daily(x); sk, ku = skew_kurt(x)
    em = expected_max(N)
    tgt = em / math.sqrt(n)
    z = (sr - tgt) * math.sqrt(n - 1) / math.sqrt(max(1 - sk * sr + ku / 4 * sr ** 2, 1e-12))
    return float(stats.norm.cdf(z)), em, float(z)


def mdd_add(pnl, idx):
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    i = int(np.argmin(dd))
    return float(dd[i]), str(idx[i].date()), dd


def mc_bootstrap(pnl, n_sims=5000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(pnl)
    fin = np.empty(n_sims); mdd = np.empty(n_sims); tuw = np.empty(n_sims)
    for i in range(n_sims):
        c = np.cumsum(pnl[rng.integers(0, n, size=n)])
        d = c - np.maximum.accumulate(c)
        fin[i] = c[-1]; mdd[i] = d.min(); tuw[i] = (d < 0).mean()
    return fin, mdd, tuw


def rolling_windows(pnl, w):
    n = len(pnl)
    ns = n - w
    cs = np.concatenate([[0.0], np.cumsum(pnl)])
    pn = cs[w:w + ns] - cs[:ns]          # 起點 s，持有 s..s+w-1
    md = np.empty(ns)
    for s in range(ns):
        c = np.cumsum(pnl[s:s + w])
        md[s] = (c - np.maximum.accumulate(c)).min()
    return pn, md


def capm(y, x):
    m = sm.OLS(y, sm.add_constant(x)).fit()
    a = float(m.params[0])
    return {'alpha_daily': a, 'alpha_ann': (1 + a) ** ANN - 1, 't_alpha': float(m.tvalues[0]),
            'beta': float(m.params[1]), 'r2': float(m.rsquared)}


def q(a, p):
    return float(np.percentile(a, p))


# ---------------- 主流程 ----------------
def main():
    nav = pd.read_csv(NAV, parse_dates=['date']).set_index('date').sort_index()
    V = nav['V_BRIDGE'].astype(float)
    E = nav['0050'].astype(float)
    ret = V.pct_change().iloc[1:]           # 4,617 天
    ret50 = E.pct_change().iloc[1:]
    dates = ret.index
    pnl = (ret * CAP).values
    n = len(pnl)
    out = {'meta': {'n_days': n, 'first': str(dates[0].date()), 'last': str(dates[-1].date()),
                    'nan_count': int(np.isnan(pnl).sum())}}

    # 1. 複利口徑
    days = (V.index[-1] - V.index[0]).days
    cagr = (V.iloc[-1] / V.iloc[0]) ** (365.25 / days) - 1
    cagr50 = (E.iloc[-1] / E.iloc[0]) ** (365.25 / days) - 1
    ddc = V / V.cummax() - 1
    tr = ddc.idxmin()
    pk = V.loc[:tr].idxmax()
    rec = V.loc[tr:][V.loc[tr:] >= V.loc[pk]]
    out['compound'] = {'cagr': cagr, 'mdd': float(ddc.min()), 'mdd_trough': str(tr.date()),
                       'mdd_peak': str(pk.date()), 'mdd_recover': str(rec.index[0].date()) if len(rec) else None,
                       'final': float(V.iloc[-1]), 'cagr_0050': cagr50, 'final_0050': float(E.iloc[-1])}

    # 2. 固定本金 9 核心
    total = pnl.sum()
    mdd_v, mdd_date, dd = mdd_add(pnl, dates)
    yrs252 = n / ANN
    yrs_cal = days / 365.25
    ann_pnl = total / yrs252
    win = pnl[pnl > 0]; los = pnl[pnl < 0]
    tr_df = pd.read_csv(TRADES)
    tb = tr_df[tr_df['version'] == 'V_BRIDGE'].copy()
    out['core9'] = {
        'total_pnl': float(total),
        'sharpe': sr_ann(pnl),
        'sortino': float(pnl.mean() / los.std(ddof=1) * math.sqrt(ANN)),
        'ann_pnl_252': float(ann_pnl),
        'ann_pnl_calendar': float(total / yrs_cal),
        'mdd': mdd_v, 'mdd_date': mdd_date,
        'calmar_252': float(ann_pnl / abs(mdd_v)),
        'calmar_calendar': float(total / yrs_cal / abs(mdd_v)),
        'win_rate_daily': float((pnl > 0).mean()),
        'pl_ratio_daily': float(win.mean() / -los.mean()),
        'n_trades': int(len(tb)),
        'n_trades_open_end': int((tb['why'] == 'open_end').sum()),
        'trade_win_rate': float((tb['ret'] > 0).mean()),
        'trade_pl_ratio_pnl': float(tb.loc[tb['ret'] > 0, 'pnl'].mean() / -tb.loc[tb['ret'] < 0, 'pnl'].mean()),
        'trade_pl_ratio_ret': float(tb.loc[tb['ret'] > 0, 'ret'].mean() / -tb.loc[tb['ret'] < 0, 'ret'].mean()),
    }

    # 3. 檢定
    T = {}
    t1, p1, t1m, L = hac_t(pnl)
    T['t1'] = {'t': t1, 'p': p1, 't_manual_nw': t1m, 'maxlags': L, 'pass': '✓' if p1 < 0.05 else '✗'}
    s2, lo2, hi2 = lobato_ci(pnl)
    T['t2'] = {'sr': s2, 'lo': lo2, 'hi': hi2, 'pass': '✓' if lo2 > 0 else '✗'}
    p3 = psr(pnl)
    T['t3'] = {'psr': p3, 'pass': '✓' if p3 > 0.95 else '⚠'}
    bs = bootstrap_sr(pnl)
    lo4, hi4 = q(bs, 2.5), q(bs, 97.5)
    T['t4'] = {'lo': lo4, 'hi': hi4, 'mean': float(bs.mean()), 'pass': '✓' if lo4 > 0 else '⚠'}
    # 6 子期（年份三等分）
    yrs = dates.year
    ymin, ymax = int(yrs.min()), int(yrs.max())
    seg = (ymax - ymin) // 3
    cuts = [(ymin, ymin + seg - 1), (ymin + seg, ymin + 2 * seg - 1), (ymin + 2 * seg, ymax)]
    sub = {}
    for a, b in cuts:
        m = (yrs >= a) & (yrs <= b)
        x = pnl[m]
        sub[f'{a}-{b}'] = {'sr': sr_ann(x), 'n': int(m.sum()), 'pnl': float(x.sum())}
    T['t6'] = {'sub': sub, 'pass': '✓' if all(v['sr'] > 0 for v in sub.values()) else '⚠'}
    # 7 Rolling 252
    s = pd.Series(pnl, index=dates)
    rsr = (s.rolling(252).mean() / s.rolling(252).std(ddof=1) * math.sqrt(ANN)).dropna()
    pb = float((rsr < 0).mean())
    T['t7'] = {'mean': float(rsr.mean()), 'pct_below0': pb, 'n': int(len(rsr)), 'pass': '✓' if pb < 0.15 else '⚠'}
    # 8 Drawdown
    uw = float((dd < 0).mean())
    T['t8'] = {'mdd': mdd_v, 'mdd_date': mdd_date, 'underwater': uw, 'pass': '✓' if uw < 0.85 else '⚠'}
    # 11 集中度
    srt = np.sort(pnl)[::-1]
    k = max(1, int(n * 0.05))
    contrib = srt[:k].sum() / pnl.sum()
    sk, ku = skew_kurt(pnl)
    kt = int(math.ceil(len(tb) * 0.05))
    top = tb.sort_values('ret', ascending=False).head(kt)
    trade_contrib = top['pnl'].sum() / tb['pnl'].sum()
    T['t11'] = {'daily_top5_contrib': float(contrib), 'k_days': k, 'skew': sk, 'kurt': ku,
                'trade_top5_n': kt, 'trade_top5_contrib': float(trade_contrib),
                'trade_total_pnl': float(tb['pnl'].sum()),
                'pass': '△' if abs(contrib) > 0.5 else '✓'}
    # 12 結算日（每月第三個週三）＋結算後首日（日曆 +1 天）＋換股執行日、訊號日
    wd = dates.weekday; dy = dates.day
    is3 = (wd == 2) & (dy >= 15) & (dy <= 21)
    settle = pnl[is3].sum() / total
    nxt = set((dates[is3] + pd.Timedelta(days=1)).normalize())
    ispost = np.array([d in nxt for d in dates])
    post = pnl[ispost].sum() / total
    # 依規則推訊號日：每月 10 日（含）之後第一個交易日的下一個交易日；執行日＝其下一交易日
    cal = V.index
    sig_days, exe_days = [], []
    for (yy, mm), _ in pd.Series(1, index=cal).groupby([cal.year, cal.month]):
        c = cal[(cal.year == yy) & (cal.month == mm) & (cal.day >= 10)]
        if len(c) == 0:
            continue
        i0 = cal.get_loc(c[0])
        if i0 + 1 < len(cal):
            sig_days.append(cal[i0 + 1])
        if i0 + 2 < len(cal):
            exe_days.append(cal[i0 + 2])
    # 逐筆檔顯示的訊號日若與規則推算不同（2019-02、2023-04 兩個月），以逐筆檔為準
    obs = sorted(set(pd.to_datetime(tb['sig_date'])))
    obs_by_m = {(d.year, d.month): d for d in obs}
    rule_sig = list(sig_days)
    sig_days = [obs_by_m.get((d.year, d.month), d) for d in sig_days]
    exe_days = [cal[cal.get_loc(d) + 1] for d in sig_days if cal.get_loc(d) + 1 < len(cal)]
    n_sig_adjusted = sum(1 for a, b in zip(rule_sig, sig_days) if a != b)
    sig_set, exe_set = set(sig_days), set(exe_days)
    is_sig = np.array([d in sig_set for d in dates]); is_exe = np.array([d in exe_set for d in dates])
    tb_sig = set(pd.to_datetime(tb['sig_date'])); tb_ent = set(pd.to_datetime(tb['ent_date']))
    T['t12'] = {'settle_pct': float(settle), 'post_pct': float(post), 'n_settle': int(is3.sum()),
                'exec_day_pct': float(pnl[is_exe].sum() / total), 'signal_day_pct': float(pnl[is_sig].sum() / total),
                'n_exec': int(is_exe.sum()), 'n_sig': int(is_sig.sum()), 'n_sig_adjusted_from_trades': n_sig_adjusted,
                'check_trade_sig_subset': bool(tb_sig <= sig_set),
                'check_trade_ent_subset': bool(tb_ent <= exe_set),
                'pass': '✓' if abs(settle) < 0.20 else '⚠'}
    # 13 CAPM
    tw = pd.read_csv(TWII_CSV, parse_dates=['date']).set_index('date')['close'].sort_index()
    twr = tw.pct_change().reindex(dates)
    twr_missing = int(twr.isna().sum())
    twr0 = twr.fillna(0).values
    equity = CAP + np.cumsum(pnl)
    sret = pd.Series(equity).pct_change().fillna(0).values
    c_spec = capm(sret, twr0)
    c_act_tw = capm(pnl / CAP, twr0)
    c_act_50 = capm(pnl / CAP, ret50.values)
    T['t13'] = {'spec_vs_taiex': c_spec, 'actual_vs_taiex': c_act_tw, 'actual_vs_0050': c_act_50,
                'taiex_missing_days': twr_missing,
                'pass': '✓' if c_spec['t_alpha'] > 1.96 else '⚠',
                'pass_if_actual': '✓' if c_act_tw['t_alpha'] > 1.96 else '⚠'}
    # 14 DSR
    d14 = {}
    for N in (10, 50, 200, 1000, 4500, 4040):
        pr, em, z = dsr(pnl, N)
        d14[str(N)] = {'prob': pr, 'e_max': em, 'z': z}
    T['t14'] = {'dsr': d14, 'pass': '✓' if d14['4500']['prob'] > 0.95 else '⚠'}
    out['tests'] = T

    # 4. 蒙地卡羅與滾動
    fin, mdds, tuws = mc_bootstrap(pnl)
    out['monte_carlo'] = {
        'final_p5': q(fin, 5), 'final_p50': q(fin, 50), 'final_p95': q(fin, 95),
        'mdd_p5': q(mdds, 5), 'mdd_p50': q(mdds, 50), 'mdd_p95': q(mdds, 95),
        'tuw_p50': q(tuws, 50),
        'actual_final': float(total), 'actual_mdd': mdd_v,
        'actual_final_in_5_95': bool(q(fin, 5) <= total <= q(fin, 95)),
        'actual_mdd_in_5_95': bool(q(mdds, 5) <= mdd_v <= q(mdds, 95)),
    }
    roll = {}
    for w in (63, 252, 504):
        pn, md = rolling_windows(pnl, w)
        roll[str(w)] = {'win_rate': float((pn > 0).mean()), 'median_pnl': float(np.median(pn)),
                        'p5_pnl': q(pn, 5), 'mean_pnl': float(pn.mean()), 'median_mdd': float(np.median(md)),
                        'p5_mdd': q(md, 5), 'n_starts': int(len(pn))}
    out['rolling'] = roll

    # 5. 對照組
    ctrl = {}
    for name, x in (('strategy', pnl), ('etf0050', ret50.values * CAP), ('excess', (ret.values - ret50.values) * CAP)):
        t, p, _, _ = hac_t(x)
        ctrl[name] = {'sr': sr_ann(x), 'hac_t': t, 'hac_p': p,
                      'dsr_200': dsr(x, 200)[0], 'dsr_4040': dsr(x, 4040)[0]}
    out['control'] = ctrl

    # 6. 判定：第 5、9、10 項無法由淨值重算 → 讀 results.json 的判定符號（未獨立重算）
    RES = os.path.normpath(os.path.join(HERE, '..', 'eng_full_report_2026-11-27', 'results.json'))
    borrowed = {}
    with open(RES, encoding='utf-8') as f:
        rows = json.load(f)['summary_rows']
    for i in (5, 9, 10):
        borrowed[i] = rows[i - 1]['verdict'][0]
    sym = {1: T['t1']['pass'], 2: T['t2']['pass'], 3: T['t3']['pass'], 4: T['t4']['pass'],
           5: borrowed[5], 6: T['t6']['pass'], 7: T['t7']['pass'], 8: T['t8']['pass'], 9: borrowed[9],
           10: borrowed[10], 11: T['t11']['pass'], 12: T['t12']['pass'], 13: T['t13']['pass'], 14: T['t14']['pass']}
    def counts(sy):
        v = list(sy.values())
        return {'passes': v.count('✓'), 'warns': v.count('⚠'), 'triangs': v.count('△'), 'crosses': v.count('✗')}
    c = counts(sym)
    def strength(c):
        if c['passes'] >= 12 and c['warns'] <= 2:
            return '✓ 強'
        return '✓ 顯著' if c['passes'] >= 8 else '⚠ 弱'
    pw = total / 10000
    econ = '✓ 強' if pw > 100 else ('△ 中' if pw > 50 else '⚠ 弱')
    p200 = d14['200']['prob']
    over = '✓ 低' if p200 > 0.9 else ('△ 中' if p200 > 0.7 else '⚠ 高')
    mid = list(sub.values())[1]['sr']
    stab = '✓ 良好' if mid > 0.5 else '△ 中等'
    sym_alt = dict(sym); sym_alt[13] = T['t13']['pass_if_actual']
    c_alt = counts(sym_alt)
    out['verdicts'] = {'symbols': {str(k): v for k, v in sym.items()},
                       'borrowed_not_recomputed': {str(k): v for k, v in borrowed.items()},
                       'counts': c, 'statistical': strength(c), 'economic': econ, 'overfit': over,
                       'stability': stab, 'mid_sub_sr': mid,
                       'sensitivity_t13_actual_return': {'counts': c_alt, 'statistical': strength(c_alt)}}
    with open(os.path.join(HERE, 'my_numbers.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print('my_numbers.json 已寫出；counts', c, out['verdicts']['statistical'], econ, over, stab)


if __name__ == '__main__':
    main()
