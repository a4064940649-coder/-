"""公司事件研究共用函式（events 資料夾）。

異常報酬＝個股含息累積報酬 − 同期間對照池等權累積報酬（兩者各自複利）。
* 對照池：mkt_liq＝流動性合格上市股（60 日均成交值 ≥ 1,000 萬、股價 ≥ 10）等權；mkt_all＝所有有價上市股等權。
* 持有期間停牌／下市：以最後成交價計（ret 缺值視為 0），不剔除事件，只要求進場日有成交。
* t 值：以「進場月份」為群（同月事件先平均再做 t），降低同期事件相關造成的高估；另報「進場日」群的 t。
* 日曆時間組合（calendar-time portfolio）：同時持有所有事件股、等權、每日再平衡；
  每檔進場時扣來回成本（按其權重），閒置時持現金（或 0050）。這是「實際能不能賺」的檢驗。
"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd

S = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad'); P = S / 'panel'
X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
HERE = Path(__file__).resolve().parent
COST, COST_STRESS = 0.005, 0.01


def load():
    D = {k: pd.read_parquet(P / f'{k}.parquet') for k in ['adj', 'close_raw', 'open', 'high', 'value', 'ret', 'flag', 'src']}
    idx = D['adj'].index
    D['idx'] = idx
    C = D['close_raw']
    D['liq'] = (D['value'].rolling(60, min_periods=60).mean() >= 1e7) & (C >= 10)
    R = D['ret']
    D['mkt_liq'] = R.where(D['liq']).mean(axis=1).fillna(0)
    D['mkt_all'] = R.mean(axis=1).fillna(0)
    D['LA'] = np.log1p(R.fillna(0).clip(lower=-0.99)).cumsum()
    e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj']
    D['e50'] = e50.reindex(idx.union(e50.index)).ffill().reindex(idx).pct_change().fillna(0)
    tw = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close']
    D['twii'] = tw
    lim = np.where(idx < pd.Timestamp('2015-06-01'), 0.07, 0.10)
    D['limit_up_close'] = (R.ge(pd.Series(lim - 0.005, index=idx), axis=0)) & (C >= D['high'] * 0.999)
    return D


def cum_log(D, mkt):
    return np.log1p(D[mkt].clip(lower=-0.99)).cumsum()


class Car:
    """快速計算 (code, i0, i1)：i0 收盤進場、i1 收盤出場 的原始與異常報酬。"""
    def __init__(self, D, mkt='mkt_liq'):
        self.LA = D['LA'].to_numpy(); self.cols = {c: j for j, c in enumerate(D['adj'].columns)}
        self.LM = cum_log(D, mkt).to_numpy(); self.adj = D['adj'].to_numpy(); self.n = len(D['idx'])
        self.LE = np.log1p(D['e50']).cumsum().to_numpy()

    def __call__(self, code, i0, i1):
        j = self.cols.get(code)
        if j is None or i0 < 0 or i1 >= self.n or i1 <= i0 or not np.isfinite(self.adj[i0, j]): return (np.nan, np.nan, np.nan)
        raw = math.expm1(self.LA[i1, j] - self.LA[i0, j]); m = math.expm1(self.LM[i1] - self.LM[i0])
        e = math.expm1(self.LE[i1] - self.LE[i0])
        return (raw, raw - m, raw - e)


def tstat(x):
    x = pd.Series(x).dropna()
    return round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2) if len(x) > 2 and x.std() > 0 else None


def ev_summary(df, cost=COST, label=''):
    """df：欄位 date（進場日）、raw、ab、ab50。回傳事件統計。"""
    df = df.dropna(subset=['ab'])
    o = {'label': label, 'n': int(len(df))}
    if len(df) < 5: return o
    m = df.groupby(df['date'].dt.to_period('M'))['ab'].mean()
    d = df.groupby('date')['ab'].mean()
    o.update({'mean_raw_pct': round(df['raw'].mean() * 100, 2), 'mean_ab_pct': round(df['ab'].mean() * 100, 2),
              'median_ab_pct': round(df['ab'].median() * 100, 2), 'mean_vs0050_pct': round(df['ab50'].mean() * 100, 2),
              'hit_ab': round(float((df['ab'] > 0).mean()), 3), 'net_ab_pct': round((df['ab'].mean() - cost) * 100, 2),
              'net_ab_stress_pct': round((df['ab'].mean() - COST_STRESS) * 100, 2),
              't_month': tstat(m), 't_date': tstat(d), 'n_months': int(len(m))})
    return o


def by_period(df, splits, cost=COST, label=''):
    out = {'all': ev_summary(df, cost, label)}
    for name, (a, b) in splits.items():
        out[name] = ev_summary(df[(df['date'] >= pd.Timestamp(a)) & (df['date'] <= pd.Timestamp(b))], cost, label)
    return out


def yearly_ab(df):
    return {int(y): {'n': int(len(g)), 'ab_pct': round(g['ab'].mean() * 100, 2)} for y, g in df.dropna(subset=['ab']).groupby(df['date'].dt.year)}


def calendar_portfolio(D, trades, start, end, rt=COST, idle='cash', bench='e50'):
    """trades：list of (code, i0, i1)。i0 收盤買、i1 收盤賣。回傳績效 dict。"""
    idx = D['idx']; cols = {c: j for j, c in enumerate(D['adj'].columns)}
    R = D['ret'].fillna(0).to_numpy()
    w = np.zeros(R.shape, dtype=np.float32); ent = np.zeros(len(idx))
    for code, i0, i1 in trades:
        j = cols[code]
        if i1 <= i0: continue
        w[i0 + 1:i1 + 1, j] += 1.0; ent[i0] += 1
    n = w.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        g = (w * R).sum(axis=1) / n
        cost = np.where(n > 0, np.r_[0, ent[:-1]] / np.where(n > 0, n, 1) * rt, 0)
    r = pd.Series(np.where(n > 0, g - cost, np.nan), index=idx)
    idle_r = D['e50'] if idle == '0050' else pd.Series(0.0, index=idx)
    r = r.fillna(idle_r)
    win = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    r = r[win]; b = D[bench][win]; mk = D['mkt_liq'][win]
    return perf(r, b, mk, invested=pd.Series(n > 0, index=idx)[win])


def perf(r, b, mk, invested=None):
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    def st(x):
        eq = (1 + x).cumprod()
        return {'CAGR': round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4), 'maxDD': round(float((eq / eq.cummax() - 1).min()), 4),
                'vol': round(float(x.std() * math.sqrt(252)), 4)}
    o = {'strategy': st(r), '0050': st(b), 'EW_liquid': st(mk)}
    ex = r - b
    o['excess_vs_0050_bp_day'] = round(float(ex.mean() * 1e4), 2); o['t_vs_0050'] = tstat(ex)
    if invested is not None:
        inv = invested.astype(bool)
        o['share_days_invested'] = round(float(inv.mean()), 3)
        exi = (r - mk)[inv]
        o['excess_vs_EW_bp_on_invested_days'] = round(float(exi.mean() * 1e4), 2) if len(exi) else None
        o['t_vs_EW_invested'] = tstat(exi)
    ys = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r.groupby(r.index.year)}
    yb = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in b.groupby(b.index.year)}
    o['yearly_pct'] = ys; o['yearly_0050_pct'] = yb
    o['years_beat_0050'] = f"{sum(ys[y] > yb[y] for y in ys)}/{len(ys)}"
    return o


def save(key, obj):
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    d[key] = obj
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
