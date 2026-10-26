"""經典技術分析回測共用函式（上市股日線面板，只做多現股）。

設計（事前固定）：
* 價格：還原收盤 AC＝adj；還原開盤 AO＝open×adj/close_raw；還原最高 AH＝high×adj/close_raw。
  面板沒有最低價：KD 的 9 日最低價用「收盤價」近似（任務指定）；跳空缺口的「當日低點」用 min(開, 收) 近似（實體下緣，偏保守）。
  成交量＝value/close_raw；本研究的「量」一律直接用成交值 value。
  flag 5（資料錯誤）日的價格視為缺值。
* 訊號：用第 t 日收盤後資訊；第 t+1 日開盤買進（t+1 必須有開盤價、且非開盤漲停（開盤/前收−1 ≥ 漲跌幅−0.6%）才買得到）。
  訊號日條件：60 日均成交值（停牌日記 0）≥ NT$1,000 萬、收盤 ≥ 10 元、flag ∈ {0,1,2}。
* 出場：出場條件在第 s 日收盤成立 → 第 s+1 日開盤賣；固定持有 N 日 → 進場後第 N 個交易日開盤賣。
  停牌日以復牌後第一個開盤價賣；下市以最後收盤價結算。跌停開盤賣不掉的情形未處理（偏樂觀，影響極小）。
* 成本：來回 0.5%（買 0.25%＋賣 0.25%，含證交稅 0.3%），壓力測試 1%。
* 組合：同時持有所有在手訊號股、每日等權（全額投入；沒有持股時為現金 0 報酬）；
  同時在手上限 30 檔：有空位才進，新訊號多於空位時依「訊號強度」取前幾名；已持有的股票不重複買。
  另列「每檔固定 1/30 資金、空位放現金」版本。
* 組合日報酬：第 d 日開盤 → 第 d+1 日開盤（還原開盤價）；基準（0050 含息、上市等權池）用收盤對收盤；
  t 值一律用「月報酬差」（月內日報酬連乘）計算，降低開/收時點錯位的影響。
* 期間：挑選期 S＝2005-01-01～2019-04-30；驗證期 V_all＝2019-05-01～2026-10-22（含 src 3 近似收盤），
  V_off＝同期間但剔除 src 3 的日子（2024-02-15～2025-09-23）。交易層級以訊號日歸屬期間；V_off 另要求持有期間不碰到 src 3。
"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd

P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
BENCH = Path('/home/user/jonatw/finlab-txd/data/raw')
OUT = Path(__file__).resolve().parent
PKL = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/technical/rules_events.pkl')   # 中間檔（事件表＋日報酬），不放研究資料夾
LIMIT_CHANGE = pd.Timestamp('2015-06-01')
RT = 0.005
RT_STRESS = 0.01
CAP = 30
SPAN = None
WIN = {'S': ('2005-01-01', '2019-04-30'), 'V_all': ('2019-05-01', '2026-10-31'), 'V_off': ('2019-05-01', '2026-10-31')}


def load():
    names = ['adj', 'close_raw', 'open', 'high', 'value', 'ret', 'flag', 'src']
    D = {n: pd.read_parquet(P / f'{n}.parquet') for n in names}
    idx = D['adj'].index
    F = D['flag']
    bad = F == 5
    fac = D['adj'] / D['close_raw']
    D['AC'] = D['adj'].mask(bad)
    D['AO'] = (D['open'] * fac).mask(bad)
    D['AH'] = (D['high'] * fac).mask(bad)
    V = D['value']
    v0 = V.fillna(0).where(V.notna().cumsum() > 0)
    D['v60'] = v0.rolling(60, min_periods=60).mean()
    lim = pd.Series(np.where(idx < LIMIT_CHANGE, 0.07, 0.10), index=idx)
    D['lim'] = lim
    pc = D['close_raw'].shift(1)   # 面板列＝交易日，前一列即前一交易日（停牌則為 NaN）
    on = D['open'] / pc - 1
    D['open_lu'] = (on >= lim.values[:, None] - 0.006) & (F == 0)
    D['sig_ok'] = (D['v60'] >= 1e7) & (D['close_raw'] >= 10) & F.isin([0, 1, 2]) & D['AC'].notna()
    # 隔日可買：t+1 有開盤價且非開盤漲停
    D['buyable_next'] = (D['AO'].shift(-1).notna() & ~D['open_lu'].shift(-1, fill_value=False))
    # 出場價矩陣 PX：當日有開盤用開盤；停牌用之後第一個開盤；下市後用最後收盤
    AO = D['AO']
    last_valid = D['AC'].notna()[::-1].cummax()[::-1]       # 之後還有資料
    PX = AO.bfill().where(last_valid)
    PX = PX.fillna(D['AC'].ffill())
    D['PX'] = PX
    D['R_oo'] = (PX.shift(-1) / PX - 1)
    # 每檔最後有資料的列
    nn = D['AC'].notna().to_numpy()
    T = len(idx)
    lv = np.where(nn.any(0), T - 1 - np.argmax(nn[::-1], axis=0), -1)
    D['last_row'] = lv
    # 日層級來源
    daysrc = D['src'].mode(axis=1)[0]
    D['daysrc'] = daysrc
    s3 = (daysrc == 3)
    D['off_day'] = ~(s3 | s3.shift(1, fill_value=False) | s3.shift(-1, fill_value=False))
    # 基準：0050 含息、上市等權池（前一日符合流動性、ret 剔除公司行動與極端值）
    b = pd.read_csv(BENCH / 'etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
    D['r0050'] = b.reindex(idx.union(b.index)).ffill().reindex(idx).pct_change()
    R = D['ret']
    elig_prev = ((D['v60'] >= 1e7) & (D['close_raw'] >= 10)).shift(1, fill_value=False)
    D['r_ew'] = R.where(elig_prev & F.isin([0, 1, 2]) & (R.abs() <= lim.values[:, None] + 0.03)).mean(axis=1)
    D['I_ew'] = (1 + D['r_ew'].fillna(0)).cumprod().to_numpy()
    D['idx'] = idx
    global SPAN
    SPAN = pd.Series(idx, index=idx).diff().dt.days.fillna(1).clip(upper=60)   # 每個日報酬涵蓋的日曆天數（2024 年初缺口記入）
    return D


# ---------------- 指標 ----------------
def sma(X, n):
    return X.rolling(n, min_periods=n).mean()


def ema(X, n):
    return X.ewm(span=n, adjust=False, min_periods=n).mean()


def cross_up(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1))


def cross_dn(a, b):
    return (a < b) & (a.shift(1) >= b.shift(1))


def kd(D, n=9):
    C, H = D['AC'], D['AH']
    lo = C.rolling(n, min_periods=n).min()          # 最低價以收盤近似
    hi = H.rolling(n, min_periods=n).max()
    rsv = ((C - lo) / (hi - lo).replace(0, np.nan) * 100).clip(0, 100)
    K = rsv.ewm(alpha=1 / 3, adjust=False, min_periods=1).mean()
    Dd = K.ewm(alpha=1 / 3, adjust=False, min_periods=1).mean()
    K = K.where(rsv.notna()); Dd = Dd.where(rsv.notna())
    return K, Dd


def macd(D):
    C = D['AC']
    dif = ema(C, 12) - ema(C, 26)
    dea = dif.ewm(span=9, adjust=False, min_periods=9).mean()
    return dif, dea


def rsi(D, n=14):
    C = D['AC']
    d = C.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    r = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    r = r.where(~((dn == 0) & (up > 0)), 100)
    return r.where(C.notna())


def boll(D, n=20, k=2):
    C = D['AC']
    m = sma(C, n); s = C.rolling(n, min_periods=n).std(ddof=0)
    return m, m + k * s, m - k * s


# ---------------- 事件 → 交易 ----------------
def next_true(E):
    """E：T×N bool ndarray。回傳 NT[t,j]＝最小的 s ≥ t 使 E[s,j]；沒有則為 T。"""
    T = E.shape[0]
    a = np.where(E, np.arange(T)[:, None], T)
    return np.minimum.accumulate(a[::-1], axis=0)[::-1]


def make_events(D, sig, strength, exit_cond=None, hold=None, name=''):
    """sig：T×N bool DataFrame（第 t 日收盤成立）。exit_cond：T×N bool（收盤成立→隔日開盤賣）。hold：最長持有交易日數。
    回傳事件表（未去重疊）：j, t, e, x, gross, strength。"""
    S = (sig.fillna(False).astype(bool) & D['sig_ok'] & D['buyable_next']).to_numpy().copy()
    T, N = S.shape
    S[-1, :] = False
    t, j = np.nonzero(S)
    e = t + 1
    x = np.full(len(t), 10 ** 9, dtype=np.int64)
    if exit_cond is not None:
        E = exit_cond.fillna(False).astype(bool).to_numpy()
        NT = next_true(E)
        x = np.minimum(x, NT[e, j] + 1)
    if hold is not None:
        x = np.minimum(x, e + hold)
    x = np.minimum(x, D['last_row'][j] + 1)
    x = np.minimum(x, T - 1)
    PX = D['PX'].to_numpy()
    p0 = D['AO'].to_numpy()[e, j]
    p1 = PX[x, j]
    g = p1 / p0 - 1
    st = strength.to_numpy()[t, j]
    ev = pd.DataFrame({'j': j, 't': t, 'e': e, 'x': x, 'gross': g, 'strength': np.nan_to_num(st, nan=-1e9)})
    ev = ev[(ev['x'] > ev['e']) & np.isfinite(ev['gross'])].reset_index(drop=True)
    ev['open_end'] = ev['x'] >= T - 1
    return ev


def dedup(ev):
    """個股層級：同一檔持有中不再進場（先到先得）。"""
    ev = ev.sort_values(['j', 'e']).reset_index(drop=True)
    keep = np.zeros(len(ev), bool)
    jj, ee, xx = ev['j'].to_numpy(), ev['e'].to_numpy(), ev['x'].to_numpy()
    lastj, lastx = -1, -1
    for i in range(len(ev)):
        if jj[i] != lastj:
            lastj, lastx = jj[i], -1
        if ee[i] >= lastx:
            keep[i] = True; lastx = xx[i]
    return ev[keep].reset_index(drop=True)


def simulate(D, ev, cap=CAP, rt=RT):
    """組合模擬。回傳 DataFrame（index=日期）：gross（全額等權）、cost、slot_gross、slot_cost、n_held。"""
    T = len(D['idx'])
    R = D['R_oo'].to_numpy()
    ev = ev.sort_values(['e', 'strength'], ascending=[True, False])
    by_e = {}
    for e, g in ev.groupby('e', sort=False):
        by_e[e] = list(zip(g['j'].to_numpy(), g['x'].to_numpy()))
    held = {}  # j -> x
    gross = np.zeros(T); cost = np.zeros(T); sg = np.zeros(T); sc = np.zeros(T); nh = np.zeros(T, int)
    n_prev = 0; ntrades = 0
    half = rt / 2
    for d in range(T):
        exits = [j for j, x in held.items() if x == d]
        for j in exits: del held[j]
        entries = 0
        if d in by_e and len(held) < cap:
            for j, x in by_e[d]:
                if len(held) >= cap: break
                if j in held: continue
                held[j] = x; entries += 1
        ntrades += entries
        n = len(held)
        c = (half * len(exits) / n_prev if n_prev else 0.0) + (half * entries / n if n else 0.0)
        cost[d] = c
        sc[d] = half * (len(exits) + entries) / cap
        if n:
            r = R[d, list(held.keys())]
            r = np.nan_to_num(r, nan=0.0)
            gross[d] = r.mean(); sg[d] = r.sum() / cap
        nh[d] = n; n_prev = n
    out = pd.DataFrame({'gross': gross, 'cost': cost, 'slot_gross': sg, 'slot_cost': sc, 'n_held': nh}, index=D['idx'])
    out.attrs['ntrades'] = ntrades
    return out


# ---------------- 統計 ----------------
def win_mask(D, w, index=None):
    a, b = WIN[w]
    idx = D['idx'] if index is None else index
    m = (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
    m = pd.Series(m, index=idx)
    if w == 'V_off':
        m &= D['off_day'].reindex(idx).fillna(False)
    return m


def tstat(x):
    x = pd.Series(x).dropna()
    if len(x) < 3 or x.std() == 0: return float('nan')
    return round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2)


def perf(daily):
    d = daily.fillna(0.0)
    n = len(d)
    if n < 20: return {}
    eq = (1 + d).cumprod()
    yrs = float(SPAN.reindex(d.index).fillna(1).sum()) / 365.25 if SPAN is not None else n / 252.0
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0
    vol = d.std() * math.sqrt(252)
    return {'CAGR': round(float(cagr), 4), 'vol': round(float(vol), 4), 'maxDD': round(float((eq / eq.cummax() - 1).min()), 4)}


def monthly(daily):
    d = daily.fillna(0.0)
    return (1 + d).groupby([d.index.year, d.index.month]).prod() - 1


def yearly(daily):
    d = daily.fillna(0.0)
    return {int(y): round(float(((1 + g).prod() - 1) * 100), 1) for y, g in d.groupby(d.index.year)}


def trade_stats(D, ev, w, rt=RT):
    m = win_mask(D, w)
    sd = m.to_numpy()
    tt = ev['t'].to_numpy()
    k = sd[tt]
    if w == 'V_off':
        off = D['off_day'].to_numpy()
        cs = np.concatenate([[0], np.cumsum(~off)])
        k &= (cs[ev['x'].to_numpy() + 1] - cs[tt]) == 0
    g = ev.loc[k, 'gross']
    if len(g) < 3: return {'n': int(len(g))}
    net = g - rt
    a, b = WIN[w]
    yrs = (min(pd.Timestamp(b), D['idx'][-1]) - pd.Timestamp(a)).days / 365.25
    if w == 'V_off': yrs -= (pd.Timestamp('2025-09-23') - pd.Timestamp('2024-02-15')).days / 365.25
    hd = (ev.loc[k, 'x'] - ev.loc[k, 'e'])
    I = D['I_ew']; ee = ev.loc[k, 'e'].to_numpy(); xx = ev.loc[k, 'x'].to_numpy()
    exg = g.to_numpy() - (I[xx - 1] / I[ee - 1] - 1)       # 同期間上市等權池報酬（收盤 e−1 → 收盤 x−1，近似開盤 e → 開盤 x）
    return {'n': int(len(g)), 'per_year': round(len(g) / yrs, 1), 'mean_gross_pct': round(float(g.mean() * 100), 2),
            'mean_net_pct': round(float(net.mean() * 100), 2), 'median_net_pct': round(float(net.median() * 100), 2),
            'win_rate': round(float((net > 0).mean()), 3), 'avg_hold_days': round(float(hd.mean()), 1),
            'net_per_hold_day_bp': round(float(net.mean() / hd.mean() * 1e4), 1), 't_trade_naive': tstat(net),
            'gross_excess_vs_EW_pct': round(float(np.mean(exg) * 100), 2), 'net_excess_vs_EW_pct': round(float((np.mean(exg) - rt) * 100), 2),
            'gross_excess_per_day_bp': round(float(np.mean(exg) / hd.mean() * 1e4), 1)}


def port_stats(D, pf, w, rt=RT):
    m = win_mask(D, w)
    days = m[m].index
    p = pf.reindex(days)
    net = p['gross'] - p['cost']
    net_stress = p['gross'] - p['cost'] * (RT_STRESS / rt)
    slot = p['slot_gross'] - p['slot_cost']
    b = D['r0050'].reindex(days).fillna(0.0); ew = D['r_ew'].reindex(days).fillna(0.0)
    mn, mb, me = monthly(net), monthly(b), monthly(ew)
    o = {'net': perf(net), 'cost_drag_ann': round(float(p['cost'].mean() * 252), 4), 'net_cost1pct': perf(net_stress).get('CAGR'), 'gross_CAGR': perf(p['gross']).get('CAGR'),
         'slot_1_30': perf(slot), 'exposure': round(float((p['n_held'] > 0).mean()), 3), 'avg_held': round(float(p['n_held'].mean()), 1),
         'cap_hit_share': round(float((p['n_held'] >= CAP).mean()), 3),
         '0050': perf(b), 'EW': perf(ew),
         'excess_vs_0050_pct_m': round(float((mn - mb).mean() * 100), 3), 't_vs_0050': tstat(mn - mb),
         'excess_vs_EW_pct_m': round(float((mn - me).mean() * 100), 3), 't_vs_EW': tstat(mn - me),
         'months': int(len(mn)),
         'yearly_net_pct': yearly(net), 'yearly_0050_pct': yearly(b), 'yearly_EW_pct': yearly(ew)}
    yn, yb, ye = o['yearly_net_pct'], o['yearly_0050_pct'], o['yearly_EW_pct']
    o['years_beat_0050'] = f"{sum(yn[y] > yb[y] for y in yn)}/{len(yn)}"
    o['years_beat_EW'] = f"{sum(yn[y] > ye[y] for y in yn)}/{len(yn)}"
    return o


def evaluate(D, ev_all, name, cap=CAP):
    ev_d = dedup(ev_all)
    pf = simulate(D, ev_all, cap)
    out = {'rule': name}
    for w in WIN:
        out[w] = {'trades_all_signals': trade_stats(D, ev_d, w), 'portfolio': port_stats(D, pf, w)}
    s = out['S']['portfolio']
    pfu = simulate(D, ev_all, cap=10 ** 6)
    for w in WIN:
        pu = port_stats(D, pfu, w)
        out[w]['portfolio_uncapped'] = {'net': pu['net'], 'gross_CAGR': pu['gross_CAGR'], 'avg_held': pu['avg_held'], 't_vs_EW': pu['t_vs_EW'], 't_vs_0050': pu['t_vs_0050']}
    out['pass_S'] = bool(s['net'].get('CAGR', -1) > 0 and (s['t_vs_EW'] if s['t_vs_EW'] == s['t_vs_EW'] else -9) >= 2)
    return out, pf, ev_d


def row(o, w):
    tr = o[w]['trades_all_signals']; p = o[w]['portfolio']
    return {'rule': o['rule'], 'n': tr.get('n'), 'per_yr': tr.get('per_year'), 'net%': tr.get('mean_net_pct'), 'win': tr.get('win_rate'),
            'hold': tr.get('avg_hold_days'), 'CAGR': p['net'].get('CAGR'), 'CAGR1%': p['net_cost1pct'], 'MDD': p['net'].get('maxDD'),
            'expo': p['exposure'], 'held': p['avg_held'], 't0050': p['t_vs_0050'], 'tEW': p['t_vs_EW'],
            '0050': p['0050'].get('CAGR'), 'EW': p['EW'].get('CAGR'), 'yrsEW': p['years_beat_EW'],
            'xsEW_g%': tr.get('gross_excess_vs_EW_pct'), 'xs_bp/d': tr.get('gross_excess_per_day_bp'), 'drag': p['cost_drag_ann'],
            'unc_CAGR': o[w]['portfolio_uncapped']['net'].get('CAGR'), 'unc_tEW': o[w]['portfolio_uncapped']['t_vs_EW']}


def _clean(x):
    if isinstance(x, dict): return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)): return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.bool_,)): return bool(x)
    return x


def save(name, obj):
    (OUT / name).write_text(json.dumps(_clean(obj), ensure_ascii=False, indent=1))
