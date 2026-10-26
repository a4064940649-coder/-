"""機器學習選股共用：資料載入、價格特徵、橫斷面排名、逐期訓練、組合建構與報表。

避免前視：
* 所有價格特徵只用換股日 t 當天收盤以前的日資料（rolling 視窗以 t 為最後一天）。
* 標籤 = t → 下一換股日 的含息報酬，在可投資池內的橫斷面百分位（減 0.5）。
* 訓練樣本只收「標籤區間結束日 ≤ 預測日」的期數（呼叫端負責傳入正確的訓練期數清單）。
* 鎖漲停收盤（當日含息報酬 ≥ 漲停幅度−0.5pp 且 收盤 = 最高）視為買不到，剔除。
"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path('/home/user/-/research/tw_stock_selection')
sys.path.insert(0, str(ROOT))
from btlib import period_returns, bench_returns, ann, yearly, tstat  # noqa: E402

P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
E50 = '/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv'

PRICE_FEATS = ['mom1', 'mom3', 'mom6', 'mom12', 'mom12_1', 'hi52', 'lo52', 'vol60', 'max21',
               'ma20d', 'ma60d', 'ma200d', 'lv60', 'val5_60', 'val20_60', 'shr20_120', 'amihud60']


def load_panel():
    adj = pd.read_parquet(P / 'adj.parquet'); C = pd.read_parquet(P / 'close_raw.parquet')
    V = pd.read_parquet(P / 'value.parquet'); H = pd.read_parquet(P / 'high.parquet'); R1 = pd.read_parquet(P / 'ret.parquet')
    return adj, C, V, H, R1


def price_features(adj, C, V, H, R1, dates):
    """回傳 {name: DataFrame(index=dates)}，全部只用 ≤ t 的資料。"""
    D = pd.DatetimeIndex(dates)
    A = adj.ffill(limit=5)
    at = lambda M: M.reindex(D)
    f = {}
    for k, n in [('mom1', 21), ('mom3', 63), ('mom6', 126), ('mom12', 252)]:
        f[k] = at(A / A.shift(n) - 1)
    f['mom12_1'] = at(A.shift(21) / A.shift(252) - 1)
    f['hi52'] = at(adj / adj.rolling(252, min_periods=200).max())
    f['lo52'] = at(adj / adj.rolling(252, min_periods=200).min())
    f['vol60'] = at(R1.rolling(60, min_periods=40).std())
    f['max21'] = at(R1.rolling(21, min_periods=15).max())
    for n in [20, 60, 200]:
        f[f'ma{n}d'] = at(adj / adj.rolling(n, min_periods=int(n * 0.8)).mean() - 1)
    v5, v20, v60 = V.rolling(5).mean(), V.rolling(20).mean(), V.rolling(60).mean()
    f['lv60'] = at(np.log(v60.where(v60 > 0)))
    f['val5_60'] = at(v5 / v60); f['val20_60'] = at(v20 / v60)
    SH = V / C  # 成交股數
    f['shr20_120'] = at(SH.rolling(20).mean() / SH.rolling(120, min_periods=90).mean())
    f['amihud60'] = at((R1.abs() / V.where(V > 0)).rolling(60, min_periods=40).mean() * 1e9)
    return f


def universe(adj, C, V, H, R1, dates, top=300):
    """60 日均成交值 ≥ 1,000 萬、股價 ≥ 10、流動性前 top 名、近 252 日有 52 週高點、非鎖漲停。"""
    D = pd.DatetimeIndex(dates)
    v60 = V.rolling(60).mean().reindex(D); px = C.reindex(D)
    base = (v60 >= 1e7) & (px >= 10)
    t300 = base & (v60.where(base).rank(axis=1, ascending=False) <= top)
    s52 = (adj / adj.rolling(252).max()).reindex(D)
    lim = pd.Series(np.where(D < pd.Timestamp('2015-06-01'), 0.065, 0.095), index=D)
    locked = R1.reindex(D).ge(lim, axis=0) & (C.reindex(D) >= H.reindex(D) - 1e-9)
    return t300 & s52.notna(), locked


def xs_rank(F, elig):
    """橫斷面百分位 − 0.5（只在 elig 內排名）。"""
    return F.where(elig).rank(axis=1, pct=True) - 0.5


def stack(feats: dict, elig, names):
    """把 {name: DataFrame} 疊成長表 (date, code) × names，只保留 elig。"""
    cols = {}
    for k in names:
        cols[k] = xs_rank(feats[k], elig).stack(future_stack=True)
    df = pd.DataFrame(cols)
    m = elig.stack(future_stack=True)
    return df[m.reindex(df.index).fillna(False).astype(bool)]


def fit_predict(model, Xtr, ytr, Xte, feat_names, seed=0):
    """model: 'ridge' | 'lgbm' | 'lgbm_shallow'。回傳 (pred, importance Series)。"""
    if model == 'ridge':
        from sklearn.linear_model import Ridge
        m = Ridge(alpha=100.0).fit(Xtr.fillna(0.0).values, ytr.values)
        return m.predict(Xte.fillna(0.0).values), pd.Series(m.coef_, index=feat_names)
    import lightgbm as lgb
    if model == 'lgbm':
        prm = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=300, subsample=0.7, subsample_freq=1,
                   colsample_bytree=0.7, reg_lambda=5.0)
    elif model == 'lgbm_shallow':
        prm = dict(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=500, subsample=0.7, subsample_freq=1,
                   colsample_bytree=0.7, reg_lambda=5.0)
    elif model == 'lgbm_top20':   # 事後診斷：二元標籤「下月排名前 20%」，參數同 lgbm
        prm = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=300, subsample=0.7, subsample_freq=1,
                   colsample_bytree=0.7, reg_lambda=5.0)
        m = lgb.LGBMClassifier(objective='binary', random_state=seed, n_jobs=4, verbose=-1, **prm)
        m.fit(Xtr.values, (ytr.values > 0.3).astype(int))
        imp = pd.Series(m.booster_.feature_importance('gain'), index=feat_names)
        return m.predict_proba(Xte.values)[:, 1], imp / imp.sum()
    else:
        raise ValueError(model)
    m = lgb.LGBMRegressor(objective='regression', random_state=seed, n_jobs=4, verbose=-1, **prm)
    m.fit(Xtr.values, ytr.values)
    imp = pd.Series(m.booster_.feature_importance('gain'), index=feat_names)
    return m.predict(Xte.values), imp / imp.sum()


def build_port(score: pd.DataFrame, R: pd.DataFrame, elig: pd.DataFrame, n=30, rt=0.005, min_names=50):
    """每期取 score 最高 n 檔等權；net = gross − 平均周轉 × 來回成本（與 bt_final_rule 同法）。"""
    port, uni, to, prev, picks = {}, {}, [], None, {}
    for t in score.index:
        if t not in R.index: continue
        ok = elig.loc[t].fillna(False).astype(bool) & R.loc[t].notna() & score.loc[t].notna()
        if ok.sum() < min_names: continue
        pick = list(score.loc[t][ok].sort_values(ascending=False).index[:n])
        port[t] = R.loc[t][pick].mean(); uni[t] = R.loc[t][ok].mean(); picks[t] = pick
        if prev is not None: to.append(len(set(pick) - prev) / n)
        prev = set(pick)
    g = pd.Series(port, dtype=float); tov = float(np.mean(to)) if to else 0.0
    return {'gross': g, 'net': g - tov * rt, 'net1': g - tov * 0.01, 'uni': pd.Series(uni, dtype=float), 'turnover': tov, 'picks': picks}


def ic_series(score, R, elig):
    out = {}
    for t in score.index:
        if t not in R.index: continue
        ok = elig.loc[t].fillna(False).astype(bool) & R.loc[t].notna() & score.loc[t].notna()
        if ok.sum() < 50: continue
        out[t] = score.loc[t][ok].rank().corr(R.loc[t][ok].rank())
    return pd.Series(out, dtype=float)


def quintiles(score, R, elig):
    """每期依 score 分 5 組等權（毛報酬），回傳 DataFrame(Q1..Q5)。"""
    out = {}
    for t in score.index:
        if t not in R.index: continue
        ok = elig.loc[t].fillna(False).astype(bool) & R.loc[t].notna() & score.loc[t].notna()
        if ok.sum() < 50: continue
        q = pd.qcut(score.loc[t][ok].rank(method='first'), 5, labels=False)
        out[t] = R.loc[t][ok].groupby(q).mean().rename(lambda k: f'Q{k+1}')
    return pd.DataFrame(out).T


def cut(x, a=None, b=None):
    if a is not None: x = x[x.index >= pd.Timestamp(a)]
    if b is not None: x = x[x.index <= pd.Timestamp(b)]
    return x


def report(res, B, dates, a=None, b=None, ic=None):
    net = cut(res['net'], a, b); bb = B.reindex(net.index); uni = res['uni'].reindex(net.index)
    ex, exu = net - bb, net - uni
    o = {'net': ann(net, dates), 'net_1pct': ann(cut(res['net1'], a, b), dates), 'gross_CAGR': ann(cut(res['gross'], a, b), dates).get('CAGR'),
         '0050': ann(bb, dates), 'pool_EW': ann(uni, dates), 'turnover': round(res['turnover'], 3),
         'excess_vs_0050_pct_m': round(float(ex.mean() * 100), 3), 't_vs_0050': tstat(ex),
         'excess_vs_EW_pct_m': round(float(exu.mean() * 100), 3), 't_vs_EW': tstat(exu),
         'hit_vs_0050': round(float((ex > 0).mean()), 3),
         'yearly_net_pct': yearly(net, dates), 'yearly_0050_pct': yearly(bb, dates), 'yearly_EW_pct': yearly(uni, dates)}
    if ic is not None:
        ic = cut(ic, a, b); o['rankIC_mean'] = round(float(ic.mean()), 4); o['rankIC_t'] = tstat(ic)
    return o


def line(name, o):
    n = o['net']; return (f"{name:34s} net {n.get('CAGR', float('nan')):7.2%} MDD {n.get('maxDD', float('nan')):7.2%} SR {n.get('sharpe0')} | "
                          f"0050 {o['0050'].get('CAGR', float('nan')):6.2%} EW {o['pool_EW'].get('CAGR', float('nan')):6.2%} | to {o['turnover']:.2f} | "
                          f"ex0050 {o['excess_vs_0050_pct_m']:+.2f}%/m t={o['t_vs_0050']} exEW t={o['t_vs_EW']} | IC {o.get('rankIC_mean')} t={o.get('rankIC_t')}")
