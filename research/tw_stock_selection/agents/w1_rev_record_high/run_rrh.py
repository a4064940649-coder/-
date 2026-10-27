#!/usr/bin/env python3
"""月營收創歷史新高（CMoney 被動投資實驗室版）＋ 48／13 個月新高變體，長歷史 2005～2026 可成交回測。
輸出 results.json（PREREG 先落地）、nav.parquet。執行：python3 -I run_rrh.py
成交、成本、漲停判定、60 日均值修補全部沿用 momentum_family/mlib.py → exec_rebuild/exlib.py（只讀取）。
唯一新增的引擎功能：權重向量（入選 < 5 檔時其餘放 0050），邏輯與 exlib.sim_portfolio 相同，並以等權情形逐日比對驗證。"""
import json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'momentum_family'))
import mlib as M  # noqa: E402
L = M.L
T0 = time.time()
LONG = Path('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv')
XR = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')

PREREG = {
    '寫於看結果前': True,
    '來源規則': 'CMoney 被動投資實驗室「月營收創歷史新高」：當月營收 > 該股過去所有月份營收最大值，等權買進入選股，每月換股。',
    '股池': '面板上市普通股（排除 91xx 存託憑證）；訊號日 d 收盤：60 日均成交值 ≥ NT$1,000 萬（mlib.V60，2018-01-13 稀疏日剔除後滾動）、收盤 ≥ 10 元；t 月營收有值且 > 0；歷史長度條件（見各版本）；d+1 有開盤價。不設前 300 名、不設處置濾網（教學沒有）。',
    '營收資料': 'NiklavsD/bottleneck-signals tw_revenue.csv（千元；sii 優先去重；只用面板內代碼）。',
    '訊號日': '每月第一個日期 ≥ 11 日的交易日 d（= 次月 10 日營收截止後第一個交易日）收盤計算，用 t = d 所在月的前一月營收；長期檔無公告日，一律假設 10 日前已公告。',
    '主版本 H_all（全歷史新高）': 't 月營收 > t 之前所有有值月份營收的最大值，且 t 之前至少 36 個有值月份。',
    '變體 a H48（48 個月新高，鉅亨易策略）': 't 月營收 > 前 47 個月最大值（含當月共 48 個月為最高），前 47 個月須全部有值。',
    '變體 b H13（13 個月新高，商周版，不加其他條件）': 't 月營收 > 前 12 個月最大值，前 12 個月須全部有值。',
    'IFRS 處理': '營收月份 2013-01～2013-12（合併口徑未滿 12 個月）一律不產生訊號、不入股池（三版本相同）；這 12 個持有期 100% 放 0050，並從挑選期統計中剔除。2014-01 起的比較基準已含 ≥12 個合併口徑月份。',
    '進場': 'd+1 開盤等權買進；d+1 開盤漲停（還原開盤／前收 −1 ≥ 漲停幅 −0.6%，漲停幅 2015-06-01 前 7%、後 10%）且未持有者不買（不遞補）；已持有者續抱。',
    '權重': '入選 k ≥ 5：每檔 1/k；k < 5：每檔 1/5，其餘 (5−k)/5 放 0050；k = 0：100% 0050。0050 以前一日收盤價（含息）成交（資料無 0050 開盤價），成本與個股相同（偏保守：實際 ETF 證交稅 0.1%）。',
    '出場': '下一個訊號日的隔日開盤全部換股（調回新目標權重；續選者只調整差額），持有約 1 個月；最後一期持有到面板最後一日 2026-10-22。',
    '成本': 'exlib.cost_rates：買 0.0855%+滑價、賣 0.0855%+稅 0.3%+滑價；主＝滑價 0.1%（來回約 0.67%），壓力＝買 0.5%+賣 0.5%（來回 1%）；另報未扣成本。',
    '對手': '0050 含息（日頻）；同池等權＝同一版本股池全部等權、同引擎、d+1 開盤換股、不扣成本（與 momentum_family 慣例相同，對策略較嚴格）。',
    '統計': '日頻 NAV 算年化、最大回撤、逐年；t＝換股日收盤→下一換股日收盤的月窗超額（每月一個觀察值＝按進場月份分群）。另報事件層級：每筆（股票, 月）d+1 開盤→下一 d+1 開盤報酬，扣主成本，減同池等權，先按進場月份平均再算 t。',
    '期間': {'挑選期': '2005-01～2019-04（實際訊號起點受歷史長度限制：H_all 2008-02、H48 2009-01、H13 2006-02），剔除 2013 營收月份 12 期',
             '驗證期': '2019-05～2026-10', '另報': ['2024-01～2025-12', 'A 乾淨段（起點～2013-01 訊號；無存活者偏誤、個體口徑一致）', 'B 存活者段 2014-02～2019-04 訊號', '2019-05～2023-12（價格面板只有存活者）', '2021-03～2023-12／2024-01～2026-10（對照 2021 起資料的切分）', '官方價格來源段（排除 src3 2024-02-15～2025-09-23）']},
    '判定（只用主版本 H_all、滑價 0.1%）': '挑選期與驗證期扣成本後皆：對同池等權月窗 t ≥ 2，且年化 ≥ 0050 年化 → 可用；只有一段通過 → 待觀察；否則不可用。變體只報告、不改判定。',
    '資料檢查（事前列出）': ['2013 若不剔除的訊號數（假新高規模）', '每月入選檔數、<5 檔月數', '入選股持有期公司行動 flag 3/4/5 與單月 >100% 報酬', '2026-10 訊號（營收 2026-09）用 rev_hist ann_date 檢查公告是否晚於訊號日', 'rev_hist 2024-10 起營收與長期檔比對'],
}

res = {'PREREG': PREREG}
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')

IDX, COLS, N, pos = L.IDX, L.COLS, L.N, L.pos
# ---------------- 營收 ----------------
d = pd.read_csv(LONG, dtype={'code': str, 'market': str, 'name': str})
d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
d = d.sort_values(['p', 'market']).drop_duplicates(['p', 'code'], keep='first')
d = d[d['code'].isin(COLS)]
REV = d.pivot_table(index='p', columns='code', values='revenue_k_twd', aggfunc='last').sort_index()
REV = REV.reindex(pd.period_range(REV.index.min(), REV.index.max(), freq='M')).reindex(columns=COLS)
REV = REV.where(REV > 0)
cnt_prior = REV.notna().cumsum().shift(1)
hm_all = REV.cummax().ffill().shift(1)
hm48 = REV.rolling(47, min_periods=47).max().shift(1)
hm13 = REV.rolling(12, min_periods=12).max().shift(1)
IFRS = (REV.index >= pd.Period('2013-01', 'M')) & (REV.index <= pd.Period('2013-12', 'M'))
VAR = {}
for name, hm, hist in (('H_all', hm_all, cnt_prior >= 36), ('H48', hm48, hm48.notna()), ('H13', hm13, hm13.notna())):
    base = REV.notna() & hist
    sig = base & (REV > hm)
    VAR[name] = {'base_raw': base.copy(), 'sig_raw': sig.copy()}
    base.loc[IFRS] = False; sig.loc[IFRS] = False
    VAR[name]['base'] = base; VAR[name]['sig'] = sig

# ---------------- 日曆 ----------------
D = [x for x in M.LDATES if (pd.Period(x, 'M') - 1) in REV.index]
D = D + [IDX[-1]] if IDX[-1] > D[-1] else D
DATES = pd.DatetimeIndex(D)
mm = [pd.Period(x, 'M') - 1 for x in D]
common = np.array([not c.startswith('91') for c in COLS])
v60 = M.V60.reindex(DATES); px = L.C.reindex(DATES)
liq = ((v60 >= 1e7) & (px >= 10)).to_numpy() & common[None, :]

# 0050 第 N 欄
E50 = L.E50.ffill()
AC2 = np.hstack([L.AC, E50.to_numpy()[:, None]])
AO2 = np.hstack([L.AO, E50.shift(1).bfill().to_numpy()[:, None]])
J50 = N

def build(name):
    sig = VAR[name]['sig'].reindex(mm).to_numpy(); base = VAR[name]['base'].reindex(mm).to_numpy()
    sig = np.where(np.isnan(sig.astype(float)), False, sig).astype(bool); base = np.where(np.isnan(base.astype(float)), False, base).astype(bool)
    events, pools, info = {}, {}, []
    held = set(); nblk = 0
    for k, dd in enumerate(D[:-1]):
        i = pos(dd); ex = i + 1
        trad = np.isfinite(L.OPEN_RAW[ex]) & np.isfinite(L.ADJ_RAW[ex])
        pool = liq[k] & base[k] & trad
        cand = np.where(pool & sig[k])[0]
        pick = [j for j in cand if not (M.GAP_UP[ex, j] and j not in held)]
        nblk += len(cand) - len(pick)
        kk = len(pick)
        if kk >= 5: idx = np.array(pick); w = np.full(kk, 1.0 / kk)
        else: idx = np.array(pick + [J50], dtype=int); w = np.r_[np.full(kk, 0.2), (5 - kk) * 0.2]
        events[ex] = (idx, w)
        pp = np.where(pool)[0]
        pools[ex] = (pp, np.full(len(pp), 1.0 / len(pp))) if len(pp) else (np.array([J50]), np.array([1.0]))
        info.append({'d': dd, 'ex': ex, 'n_sig': int(len(cand)), 'n_pick': kk, 'n_blocked': int(len(cand) - kk), 'pool': int(pool.sum()), 'picks': [COLS[j] for j in pick]})
        held = set(pick)
    return events, pools, pd.DataFrame(info), nblk

def sim_w(events, buy, sell, t0, t1):
    """exlib.sim_portfolio 的權重版：換股日以開盤價把漂移後持股調到目標權重，買賣金額分別乘費率（6 次固定點迭代）。"""
    NN = N + 1; u = np.zeros(NN); cash = 1.0; nav = np.empty(t1 - t0 + 1); tov = []
    for t in range(t0, t1 + 1):
        if t in events:
            pk, w = events[t]
            Pz = AO2[t]
            h = np.where(u > 0, u * np.nan_to_num(Pz), 0.0)
            Vt = h.sum() + cash; Vn = Vt; tgt = np.zeros(NN)
            for _ in range(6):
                tgt[:] = 0; tgt[pk] = w * Vn
                cost = buy * np.clip(tgt - h, 0, None).sum() + sell * np.clip(h - tgt, 0, None).sum()
                Vn = Vt - cost
            tov.append(np.clip(h - tgt, 0, None).sum() / Vt if Vt > 0 else 0)
            u = np.zeros(NN); u[pk] = tgt[pk] / Pz[pk]; cash = 0.0
        nav[t - t0] = np.nansum(u * AC2[t]) + cash
    return pd.Series(nav, index=IDX[t0:t1 + 1]), float(np.mean(tov[1:])) if len(tov) > 1 else float('nan')

def daily_stats(nav, segs, bench_nav=None, ew_nav=None):
    """同 revenue_longhistory/lhlib.daily_stats（DATES 用本檔日曆）。"""
    r = nav.pct_change(); mask = np.zeros(len(r), bool); span = 0.0
    for a, b in segs:
        m = (r.index > pd.Timestamp(a)) & (r.index <= pd.Timestamp(b)) & r.notna().to_numpy()
        if m.sum() == 0: continue
        first = np.argmax(m); last = len(m) - 1 - np.argmax(m[::-1])
        span += (r.index[last] - r.index[first - 1]).days / 365.25
        mask |= m
    rr = r[mask]
    if len(rr) < 20: return None
    eq = (1 + rr).cumprod()
    o = {'CAGR': round(float(eq.iloc[-1] ** (1 / span) - 1), 4), 'vol': round(float(rr.std() * math.sqrt(252)), 4),
         'maxDD': round(float((eq / eq.cummax() - 1).min()), 4), 'years': round(span, 2)}
    nd = nav.reindex(DATES); pr = (nd.shift(-1) / nd - 1).iloc[:-1].dropna()
    inw = np.zeros(len(pr), bool)
    for a, b in segs: inw |= (pr.index >= pd.Timestamp(a)) & (pr.index < pd.Timestamp(b))
    pr = pr[inw]; o['n_months'] = int(len(pr))
    for nm, bn in (('0050', bench_nav), ('EW', ew_nav)):
        if bn is None: continue
        bd = bn.reindex(DATES); bp = (bd.shift(-1) / bd - 1).iloc[:-1].reindex(pr.index); ex = pr - bp
        o[f'excess_vs_{nm}_pct_m'] = round(float(ex.mean() * 100), 2); o[f't_vs_{nm}_m'] = L.tstat(ex)
        o[f'hit_vs_{nm}_m'] = round(float((ex > 0).mean()), 3)
        b = bn.pct_change().reindex(rr.index); eb = (1 + b.fillna(0)).cumprod()
        o[f'{nm}_CAGR'] = round(float(eb.iloc[-1] ** (1 / span) - 1), 4); o[f'{nm}_maxDD'] = round(float((eb / eb.cummax() - 1).min()), 4)
    return o

def yearly(nav):
    r = nav.pct_change().dropna(); return {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r.groupby(r.index.year)}

# ---- 引擎驗證：等權情形與 exlib.sim_portfolio 逐日比對 ----
_ev, _pl, _, _ = build('H_all')
_t0 = pos(D[0]); _t1 = pos(D[-1])
_pe = {t: ('open', v[0]) for t, v in _pl.items() if J50 not in v[0]}
_pw = {t: v for t, v in _pl.items() if J50 not in v[0]}
b1, s1 = L.cost_rates(0.001)
na, _ = L.sim_portfolio(_pe, b1, s1, _t0, _t1); nb_, _ = sim_w(_pw, b1, s1, _t0, _t1)
res['engine_check_max_abs_nav_diff'] = float((na - nb_).abs().max())
print('engine check', res['engine_check_max_abs_nav_diff'], flush=True)

def first_on_or_after(s): return DATES[DATES >= pd.Timestamp(s)][0]
d13 = first_on_or_after('2013-02-01'); d14 = first_on_or_after('2014-02-01')

def windows(start):
    a0 = str(start.date())
    return {
        'select_official': [(a0, str(d13.date())), (str(d14.date()), '2019-04-30')],
        'valid_2019-05_2026-10': [('2019-05-01', '2026-10-31')],
        'y2024-01_2025-12': [('2024-01-01', '2025-12-31')],
        'A_clean_to_2013-01sig': [(a0, str(d13.date()))],
        'B_survivor_2014-02_2019-04': [(str(d14.date()), '2019-04-30')],
        'IFRS_block_0050_2013-02_2014-01': [(str(d13.date()), str(d14.date()))],
        'valid_survivorPanel_2019-05_2023-12': [('2019-05-01', '2023-12-31')],
        'r21_select_2021-03_2023-12': [('2021-03-01', '2023-12-31')],
        'r21_valid_2024-01_2026-10': [('2024-01-01', '2026-10-31')],
        'valid_officialPx_only': [('2019-05-01', '2024-02-14'), ('2025-09-24', '2026-10-31')],
        'full_ex_IFRS': [(a0, str(d13.date())), (str(d14.date()), '2026-10-31')],
    }

navs = {}
E50N = E50
for name in ('H_all', 'H48', 'H13'):
    ev, pools, info, nblk = build(name)
    st = info[info.n_sig > 0]
    k0 = int(st.index[0]) if len(st) else 0
    start = D[k0]                      # 版本自身第一個有訊號的換股日
    t0 = pos(start); t1 = pos(D[-1])
    ew, _ = sim_w(pools, 0.0, 0.0, t0, t1); navs[f'{name}_EW'] = ew
    W = windows(start)
    o = {'first_signal_date': str(start.date()), 'skipped_limit_up_total': int(nblk),
         'n_pick_by_year_mean': {int(y): round(float(g.n_pick.mean()), 1) for y, g in info.groupby(pd.DatetimeIndex(info.d).year)},
         'pool_by_year_mean': {int(y): round(float(g.pool.mean()), 1) for y, g in info.groupby(pd.DatetimeIndex(info.d).year)},
         'months_lt5_picks_select': int(((info.n_pick < 5) & (info.d >= start) & (info.d < pd.Timestamp('2019-05-01'))).sum()),
         'months_lt5_picks_valid': int(((info.n_pick < 5) & (info.d >= pd.Timestamp('2019-05-01'))).sum()),
         'EW_windows': {k: daily_stats(ew, s, E50N) for k, s in W.items()}, 'EW_yearly': yearly(ew)}
    for cost in ('gross', 0.001, 'stress'):
        b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
        nav, tov = sim_w(ev, b, s, t0, t1); navs[f'{name}_{cost}'] = nav
        o[f'cost_{cost}'] = {'turnover_per_rebalance': round(tov, 3), 'windows': {k: daily_stats(nav, s_, E50N, ew) for k, s_, in W.items()}, 'yearly_pct': yearly(nav)}
    # 事件層級（主成本）
    b, s = L.cost_rates(0.001)
    exs = list(info.ex) + [pos(D[-1])]
    rows = []
    for k in range(len(info)):
        e0, e1 = exs[k], exs[k + 1]
        if e1 >= len(IDX) or k < k0: continue
        p0, p1 = L.AO[e0], (L.AO[e1] if e1 < len(IDX) and e1 != pos(D[-1]) else L.AC[e1])
        rr = p1 / p0 - 1
        pl = pools[e0][0]; pl = pl[pl != J50]
        if len(pl) == 0: continue
        ewr = np.nanmean(rr[pl])
        for c in info.picks[k]:
            j = L.CI[c]
            if not np.isfinite(rr[j]): continue
            net = (1 + rr[j]) * (1 - s) / (1 + b) - 1
            rows.append({'d': info.d[k], 'code': c, 'ret': rr[j], 'net': net, 'abn': net - ewr})
    evd = pd.DataFrame(rows)
    evs = {}
    if len(evd):
        for kname, segs in W.items():
            m = np.zeros(len(evd), bool)
            for a, b_ in segs: m |= (evd.d >= pd.Timestamp(a)) & (evd.d < pd.Timestamp(b_))
            x = evd[m]
            if len(x) < 10: continue
            g = x.groupby('d').abn.mean()
            evs[kname] = {'n_events': int(len(x)), 'n_months': int(len(g)), 'mean_net_ret_pct': round(float(x.net.mean() * 100), 2),
                          'mean_abn_vs_poolEW_pct': round(float(x.abn.mean() * 100), 2), 'month_cluster_mean_abn_pct': round(float(g.mean() * 100), 2),
                          't_month_cluster': L.tstat(g), 'hit_abn': round(float((x.abn > 0).mean()), 3)}
        big = evd[evd.ret > 1.0]
        o['events_ret_gt_100pct'] = big.assign(d=big.d.astype(str)).round(3).to_dict('records')[:20]
    o['event_study'] = evs
    # 公司行動 flag（持有期內 flag 3/4/5）
    FL = pd.read_parquet(L.P / 'flag.parquet').reindex(index=IDX, columns=COLS).to_numpy()
    nfl = 0; ntot = 0
    for k in range(k0, len(info)):
        e0, e1 = exs[k], min(exs[k + 1], len(IDX) - 1)
        for c in info.picks[k]:
            j = L.CI[c]; ntot += 1
            if np.isin(FL[e0:e1 + 1, j], [3, 4, 5]).any(): nfl += 1
    o['picks_with_flag345_in_hold'] = {'n': nfl, 'of': ntot}
    o['raw_2013_signal_counts_if_not_excluded'] = {str(p): int((VAR[name]['sig_raw'].loc[p] & VAR[name]['base_raw'].loc[p]).sum()) for p in pd.period_range('2012-07', '2014-06', freq='M')}
    o['raw_signal_share_by_year'] = {int(y): round(float(VAR[name]['sig_raw'].loc[g].sum().sum() / max(VAR[name]['base_raw'].loc[g].sum().sum(), 1)), 3)
                                     for y, g in pd.Series(VAR[name]['sig_raw'].index, index=VAR[name]['sig_raw'].index).groupby(VAR[name]['sig_raw'].index.year)}
    o['last_signal'] = {'d': str(info.d.iloc[-1].date()), 'picks': info.picks.iloc[-1]}
    res[name] = o
    res[f'{name}_info_tail'] = info.tail(8).assign(d=info.d.tail(8).astype(str)).drop(columns='picks').to_dict('records')
    w = o['cost_0.001']['windows']
    print(name, start.date(), {k: (w[k]['CAGR'], w[k]['maxDD'], w[k].get('t_vs_0050_m'), w[k].get('t_vs_EW_m'), w[k].get('0050_CAGR'), w[k].get('EW_CAGR')) for k in ('select_official', 'valid_2019-05_2026-10', 'y2024-01_2025-12', 'A_clean_to_2013-01sig') if w.get(k)}, round(time.time() - T0), flush=True)
    pd.to_pickle(info, HERE / f'_info_{name}.pkl')

res['0050_yearly'] = yearly(E50N[(E50N.index >= D[0]) & (E50N.index <= D[-1])])

# ---- ann_date 檢查（只有營收 2026-09 有公告日）----
rh = pd.read_csv(XR / 'rev_hist.csv.gz', dtype={'code': str})
ad = rh[(rh.ym == '2026-09') & rh.ann_date.notna()].set_index('code')['ann_date']
chk = {}
for name in ('H_all', 'H48', 'H13'):
    info = pd.read_pickle(HERE / f'_info_{name}.pkl'); last = info.iloc[-1]
    dsig = str(last.d.date()); pk = last.picks
    a = ad.reindex(pk)
    chk[name] = {'signal_date': dsig, 'n_picks': len(pk), 'n_with_ann_date': int(a.notna().sum()), 'n_ann_after_signal_date': int((a > dsig).sum()),
                 'late_codes': list(a[a > dsig].index)}
res['ann_date_check_2026-09'] = chk
# rev_hist vs 長期檔
rh['p'] = pd.PeriodIndex(rh['ym'], freq='M')
m = rh.merge(d[['p', 'code', 'revenue_k_twd']], on=['p', 'code'], how='inner')
rel = (m.revenue / m.revenue_k_twd - 1).abs()
res['rev_hist_vs_long'] = {'n': int(len(m)), 'share_within_0.1pct': round(float((rel < 1e-3).mean()), 4)}
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
print('done', round(time.time() - T0))
