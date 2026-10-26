"""短線價格型態研究共用函式（上市股日線面板）。

設計：
* 交易日曆 CAL = 加權指數日期 ∪ 面板日期（2004-01-02～2026-10-22）。所有寬表 reindex 到 CAL，
  因此「前一日」一定是真實的前一個交易日；股票前一日無資料（停牌、2024-01-02～02-14 缺口、
  被整日刪除的 2017-08-03／2018-07-12）時，跨日報酬一律無效，不會把兩天以上當一天。
* 隔夜報酬 on_t = open_t / close_{t-1} − 1，只用 flag==0 的日子（剔除除權息 1/2、公司行動 3、缺口 4、錯誤 5），
  並剔除 |on| > 漲跌幅 + 1% 的異常值。盤中報酬 id_t = close_t / open_t − 1（剔除 flag 3/4/5、|id| > 2×漲跌幅）。
* 流動性（只用 t−1 以前資訊）：60 日均成交值 ≥ NT$1,000 萬（停牌日成交值視為 0，需滿 60 筆）且 t−1 收盤 ≥ 10 元。
* 漲跌停（無 low 欄，近似）：漲跌幅 lim = 7%（2015-06-01 前）／10%；tick 誤差容許 0.6%。
  收盤鎖漲停 ≈ close/prev − 1 ≥ lim − 0.6% 且 close == high → 收盤買不到；
  開盤漲停 ≈ open/prev − 1 ≥ lim − 0.6% → 開盤買不到；
  收盤跌停 ≈ close/prev − 1 ≤ −(lim − 0.6%) → 當沖收盤賣不掉，改以隔日開盤賣（成本改用 0.5%）。
* 資料來源：src 1/2/4 官方、3 為 Yahoo 5 分 K 近似（收盤＝13:20）。日層級來源取當日眾數。
"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd

P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
BENCH = Path('/home/user/jonatw/finlab-txd/data/raw')
OUT = Path(__file__).resolve().parent
LIMIT_CHANGE = pd.Timestamp('2015-06-01')
COST_RT = 0.005      # 一般來回
COST_RT_STRESS = 0.01
COST_DT = 0.0023     # 現股當沖來回（稅 0.15%＋手續費）
COST_DT_PRE2017 = 0.0038  # 2017-04-28 前當沖稅 0.3%（且 2014-01-06 前不能現股當沖）

WINDOWS = {
    'S0_2005_2015': ('2005-01-01', '2015-05-31', False),   # 7% 漲跌幅期（另列）
    'S1_2015_2019': ('2015-06-01', '2019-04-30', False),   # 主要挑選期（10% 漲跌幅後）
    'V_official':   ('2019-05-01', '2026-10-31', True),    # 驗證期，只用官方來源 src 1,2,4
    'V_all':        ('2019-05-01', '2026-10-31', False),   # 驗證期，含 src 3
}


def load():
    names = ['adj', 'close_raw', 'open', 'high', 'value', 'ret', 'flag', 'src']
    raw = {n: pd.read_parquet(P / f'{n}.parquet') for n in names}
    pidx = raw['adj'].index
    tx = pd.read_csv(BENCH / 'taiex_twii.csv', parse_dates=['date'])
    tdates = pd.DatetimeIndex(tx['date'])
    CAL = pidx.union(tdates[(tdates >= pidx[0]) & (tdates <= pidx[-1])])
    meta = json.loads((P / 'meta.json').read_text())
    CAL = CAL.union(pd.DatetimeIndex(meta['market_wide_corrupted_dates_removed']))   # 指數檔也缺這兩天
    CAL = CAL[(CAL.dayofweek < 5) | CAL.isin(pidx)]   # 指數檔 2026-09-20（週日）為錯誤日期；面板內的週六補班日保留
    # 流動性（面板日序上計算，避免缺口日把均值拉成 0）
    V = raw['value']
    v0 = V.fillna(0).where(V.notna().cumsum() > 0)
    liq = (v0.rolling(60, min_periods=60).mean().shift(1) >= 1e7)
    px = raw['close_raw'].shift(1) >= 10
    elig_p = (liq & px)
    D = {n: m.reindex(CAL) for n, m in raw.items()}
    # elig 只在「前一個面板日 = 前一個交易日」時有意義；其餘情形由各報酬的 prev 檢查剔除
    D['elig'] = elig_p.reindex(CAL).fillna(False).astype(bool)
    D['CAL'] = CAL
    D['in_panel'] = pd.Series(CAL.isin(pidx), index=CAL)
    lim = pd.Series(np.where(CAL < LIMIT_CHANGE, 0.07, 0.10), index=CAL)
    D['lim'] = lim
    C, O, H = D['close_raw'], D['open'], D['high']
    pc = C.shift(1)
    D['pc'] = pc
    F = D['flag']
    limv = lim.values[:, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        on = O / pc - 1
        idr = C / O - 1
        cc = C / pc - 1
    ok_on = (F == 0) & pc.notna() & O.notna() & (on.abs() <= limv + 0.01)
    D['on'] = on.where(ok_on)
    ok_id = F.isin([0, 1, 2]) & O.notna() & C.notna() & (idr.abs() <= 2 * limv + 0.01)
    D['id'] = idr.where(ok_id)
    D['cc_raw'] = cc   # 原始收盤／前收（只有 flag 0 時才代表真實報酬）
    tol = limv - 0.006
    D['lock_up_close'] = (cc >= tol) & (C >= H - 1e-9) & (F == 0)
    D['open_limit_up'] = (on >= tol) & (F == 0)
    D['close_limit_down'] = (cc <= -tol) & (F == 0)
    # 日層級來源
    daysrc = D['src'].mode(axis=1)[0].reindex(CAL)
    D['daysrc'] = daysrc
    off = daysrc.isin([1, 2, 4])
    D['off_day'] = off & off.shift(1, fill_value=False)        # t 與 t−1 都是官方
    D['all_day'] = D['in_panel'] & D['in_panel'].shift(1, fill_value=False)
    # 基準
    b = pd.read_csv(BENCH / 'etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
    D['r0050'] = b.reindex(CAL.union(b.index)).ffill().reindex(CAL).pct_change()
    R = D['ret']
    ew = R.where(D['elig'] & F.isin([0, 1, 2]) & (R.abs() <= limv + 0.03)).mean(axis=1)
    D['ew_pool'] = ew
    return D


def day_set(D, win, need_next=False):
    """回傳窗口內可用交易日（bool Series on CAL）。need_next：交易需要 t+1 的資料。"""
    a, b, official = WINDOWS[win]
    m = (D['CAL'] >= pd.Timestamp(a)) & (D['CAL'] <= pd.Timestamp(b))
    m = pd.Series(m, index=D['CAL'])
    base = D['off_day'] if official else D['all_day']
    m &= base
    if need_next:
        m &= base.shift(-1, fill_value=False)
    return m


def _perf(daily):
    daily = daily.fillna(0.0)
    n = len(daily)
    if n < 20: return {}
    eq = (1 + daily).cumprod()
    yrs = n / 252.0
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0
    vol = daily.std() * math.sqrt(252)
    return {'CAGR': round(float(cagr), 4), 'vol': round(float(vol), 4),
            'maxDD': round(float((eq / eq.cummax() - 1).min()), 4), 'days': int(n)}


def tstat(x):
    x = pd.Series(x).dropna()
    if len(x) < 3 or x.std() == 0: return float('nan')
    return round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2)


def yearly(daily):
    d = daily.fillna(0.0)
    return {int(y): round(float(((1 + g).prod() - 1) * 100), 1) for y, g in d.groupby(d.index.year)}


def evaluate(D, gross, cost, nev, daily_net, win, need_next=False, pass_test=True):
    """gross：事件日（cohort）等權毛報酬 Series（index=歸屬日）；cost：每筆來回成本（可為 Series，處理跌停改隔日賣）；
    nev：當日事件數；daily_net：組合日報酬（CAL 上，未持股=0）。回傳窗口統計。"""
    m = day_set(D, win, need_next)
    days = m[m].index
    g = gross.reindex(days).dropna()
    c = cost.reindex(g.index) if isinstance(cost, pd.Series) else pd.Series(cost, index=g.index)
    net = g - c
    o = {'event_days': int(len(g)), 'trades': int(nev.reindex(g.index).sum()) if len(g) else 0,
         'avg_names_per_event_day': round(float(nev.reindex(g.index).mean()), 1) if len(g) else 0}
    if len(g) < 3:
        o['pass'] = False; return o
    o.update({'mean_gross_pct': round(float(g.mean() * 100), 3), 't_gross': tstat(g),
              'mean_cost_pct': round(float(c.mean() * 100), 3),
              'mean_net_pct': round(float(net.mean() * 100), 3), 't_net': tstat(net),
              'hit_rate_net': round(float((net > 0).mean()), 3),
              'net_at_stress_pct': None})
    dn = daily_net.reindex(days).fillna(0.0)
    b = D['r0050'].reindex(days).fillna(0.0)
    o['portfolio'] = _perf(dn)
    o['exposure'] = round(float((daily_net.reindex(days).fillna(0) != 0).mean()), 3)
    o['bench_0050_same_days'] = _perf(b)
    o['ew_pool_same_days_gross'] = _perf(D['ew_pool'].reindex(days))
    o['t_daily_vs_0050'] = tstat(dn - b)
    o['yearly_net_pct'] = yearly(dn); o['yearly_0050_pct'] = yearly(b)
    o['years_beat_0050'] = f"{sum(o['yearly_net_pct'][y] > o['yearly_0050_pct'].get(y, -1e9) for y in o['yearly_net_pct'])}/{len(o['yearly_net_pct'])}"
    if pass_test:
        o['pass'] = bool(o['mean_net_pct'] > 0 and o['t_net'] >= 2)
    return o


def evaluate_all(D, gross, cost, nev, daily_net, need_next=False, stress=None):
    """stress：(label, cost) 另算淨報酬。"""
    out = {}
    for w in WINDOWS:
        out[w] = evaluate(D, gross, cost, nev, daily_net, w, need_next)
        if stress is not None and out[w].get('event_days', 0) >= 3:
            m = day_set(D, w, need_next); g = gross.reindex(m[m].index).dropna()
            out[w]['net_at_stress_pct'] = {stress[0]: round(float((g.mean() - stress[1]) * 100), 3)}
    out['selected'] = bool(out['S1_2015_2019'].get('pass', False))
    out['selected_S0'] = bool(out['S0_2005_2015'].get('pass', False))
    return out


def cohort_daily(gross, cost):
    """1 日型交易（持有 ≤ 1 個交易日、資金每天全部進出）：組合日報酬＝當日 cohort 淨報酬。"""
    c = cost if isinstance(cost, pd.Series) else pd.Series(cost, index=gross.index)
    return (gross - c.reindex(gross.index)).dropna()


def brief(o):
    """把窗口結果壓縮成表格列。"""
    keys = ['event_days', 'trades', 'mean_gross_pct', 'mean_net_pct', 't_net', 'pass']
    r = {k: o.get(k) for k in keys}
    pf = o.get('portfolio', {}); bb = o.get('bench_0050_same_days', {})
    r.update({'CAGR': pf.get('CAGR'), 'maxDD': pf.get('maxDD'), '0050_CAGR': bb.get('CAGR'), '0050_maxDD': bb.get('maxDD')})
    return r


def _clean(x):
    if isinstance(x, dict): return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)): return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.bool_,)): return bool(x)
    return x


def save(name, obj):
    (OUT / name).write_text(json.dumps(_clean(obj), ensure_ascii=False, indent=1))


def daytrade_exit(D):
    """開盤買、收盤賣的每筆毛報酬與成本矩陣（CAL×股票）。收盤跌停（賣不掉）→ 隔日開盤賣、成本 0.5%。"""
    O, idr = D['open'], D['id']
    nxt_open = O.shift(-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        alt = nxt_open / O - 1
    cld = D['close_limit_down'].fillna(False).astype(bool)
    g = idr.where(~cld, alt.where(alt.notna(), idr))
    cost = pd.DataFrame(COST_DT, index=idr.index, columns=idr.columns).where(~cld, COST_RT)
    return g, cost


def cohort(g, cost, mask):
    """mask：事件矩陣（bool）。回傳 (cohort 毛報酬, cohort 平均成本, 事件數)，index=CAL 中有事件的日子。"""
    m = mask.fillna(False).astype(bool) & g.notna()
    gm = g.where(m); n = m.sum(axis=1)
    keep = n > 0
    gross = gm.mean(axis=1)[keep]
    c = cost.where(m).mean(axis=1)[keep] if isinstance(cost, pd.DataFrame) else pd.Series(cost, index=gross.index)
    return gross, c, n[keep]


def run_rule(D, g, cost, mask, need_next=False, stress=None, daily=None):
    """1 日型規則：組合日報酬＝cohort 淨報酬（資金每天全進全出）。"""
    gross, c, n = cohort(g, cost, mask)
    dn = daily if daily is not None else (gross - c)
    res = evaluate_all(D, gross, c, n, dn, need_next=need_next, stress=stress)
    return res, gross, c, n
