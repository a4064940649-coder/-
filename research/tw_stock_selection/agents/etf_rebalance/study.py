"""ETF 成分股調整事件研究（事前固定規則，全部報告）。

時點定義（A＝公告日；資訊在 A 收盤後才公開。E＝被動資金收盤換股日；C＝審核資料截止日）
* 累積異常報酬（CAR，個股含息複利 − 上市等權複利；另報對 0050）：
  pre20＝close(A−20)→close(A)；cut2ann＝close(C)→close(A)；ann2effm1＝close(A)→close(E−1)；
  effday＝close(E−1)→close(E)；post20＝close(E)→close(E+20)。
* 可交易規則：
  R1  公告後隔日開盤買 open(A+1) → 生效日前一日收盤賣 close(E−1)（納入做多；刪除另報「做多」結果僅供參考）。
  R1b 同 R1 但在 close(E) 賣（賣給被動買盤；E 收盤在 2024-02-15～2025-09-23 為 13:20 近似價，另標示）。
  R0  完美預測版：截止日隔日開盤 open(C+1) 買 → close(E−1) 賣（只適用臺灣50，假設能正確算出市值排名）。
  R2  生效日收盤買 close(E) → 持有 20 日 close(E+20)（刪除股反彈；納入股另報）。
* 不可成交：買進當天開盤即漲停（開盤 ≥ 前收×(1+漲幅−0.5%) 且 開=高）或 收盤鎖漲停 → 該筆略過。
* 流動性：進場前 60 日均成交值 ≥ 1,000 萬、股價 ≥ 10，不合格者剔除並計數。
* t 值：以審核期（etf×review）為群，先群內平均再做 t；另報事件層級 t。成本：來回 0.5%、壓力 1%。
* 挑選期／驗證期：E ≤ 2019-04-30 為挑選期，其後為驗證期（本研究無參數挑選，只並列報告）。
"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
SPLIT = pd.Timestamp('2019-04-30')
SRC3 = (pd.Timestamp('2024-02-15'), pd.Timestamp('2025-09-23'))


def tstat(x):
    x = pd.Series(x, dtype=float).dropna()
    return round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2) if len(x) > 2 and x.std() > 0 else None


def load():
    D = {k: pd.read_parquet(P / f'{k}.parquet') for k in ['adj', 'close_raw', 'open', 'high', 'value', 'ret']}
    idx = D['adj'].index
    R = D['ret']
    D['mkt'] = R.mean(axis=1).fillna(0)  # 上市等權
    oc = (D['close_raw'] / D['open'] - 1).where(D['open'] > 0)
    D['mkt_oc'] = oc.clip(-0.2, 0.2).mean(axis=1).fillna(0)  # 上市等權 開→收
    e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj']
    D['e50'] = e50.reindex(idx.union(e50.index)).ffill().reindex(idx).pct_change().fillna(0)
    D['LA'] = np.log1p(R.fillna(0).clip(lower=-0.99)).cumsum()
    D['LM'] = np.log1p(D['mkt']).cumsum()
    D['LE'] = np.log1p(D['e50']).cumsum()
    D['liq'] = (D['value'].rolling(60, min_periods=40).mean() >= 1e7) & (D['close_raw'] >= 10)
    lim = pd.Series(np.where(idx < pd.Timestamp('2015-06-01'), 0.07, 0.10), index=idx)
    pc = D['close_raw'].shift(1)
    D['open_lu'] = (D['open'] >= pc.mul(1 + lim - 0.005, axis=0)) & (D['open'] >= D['high'] * 0.999)
    D['close_lu'] = (R.ge(lim - 0.005, axis=0)) & (D['close_raw'] >= D['high'] * 0.999)
    D['idx'] = idx
    return D


class Calc:
    def __init__(self, D):
        self.D = D; self.idx = D['idx']; self.n = len(self.idx)
        self.LA = D['LA']; self.LM = D['LM'].to_numpy(); self.LE = D['LE'].to_numpy()

    def pos(self, t):
        return int(self.idx.get_loc(pd.Timestamp(t)))

    def cc(self, c, i0, i1):
        """close(i0)→close(i1)：原始、對等權異常、對 0050 異常。"""
        if c not in self.LA.columns or i0 < 0 or i1 >= self.n or i1 <= i0: return (np.nan,) * 3
        a = self.D['adj'][c]
        if not (np.isfinite(a.iat[i0])): return (np.nan,) * 3
        la = self.LA[c].to_numpy()
        raw = math.expm1(la[i1] - la[i0])
        return raw, raw - math.expm1(self.LM[i1] - self.LM[i0]), raw - math.expm1(self.LE[i1] - self.LE[i0])

    def oc_trade(self, c, i_buy, i_sell):
        """open(i_buy)→close(i_sell)。"""
        if c not in self.LA.columns or i_buy < 1 or i_sell >= self.n or i_sell < i_buy: return (np.nan,) * 3
        o = self.D['open'][c].iat[i_buy]; cl = self.D['close_raw'][c].iat[i_buy]
        if not (np.isfinite(o) and o > 0 and np.isfinite(cl)): return (np.nan,) * 3
        la = self.LA[c].to_numpy()
        raw = (cl / o) * math.exp(la[i_sell] - la[i_buy]) - 1
        m = (1 + self.D['mkt_oc'].iat[i_buy]) * math.exp(self.LM[i_sell] - self.LM[i_buy]) - 1
        e = math.exp(self.LE[i_sell] - self.LE[i_buy - 1]) - 1  # 0050 無開盤價，用 close(i_buy−1) 近似
        return raw, raw - m, raw - e


def run():
    D = load(); K = Calc(D); idx = D['idx']
    ev = pd.read_csv(HERE / 'events.csv', dtype={'code': str})
    rows = []
    for _, r in ev.iterrows():
        c = r.code; o = r.to_dict()
        iA, iE = K.pos(r.ann), K.pos(r.eff)
        iC = K.pos(r.cutoff) if isinstance(r.cutoff, str) and r.cutoff else None
        o['in_panel'] = c in D['adj'].columns and np.isfinite(D['adj'][c].iat[iA]) if c in D['adj'].columns else False
        if not o['in_panel']:
            rows.append(o); continue
        win = {'pre20': (iA - 20, iA), 'ann2effm1': (iA, iE - 1), 'ann2eff': (iA, iE), 'effday': (iE - 1, iE), 'post20': (iE, iE + 20)}
        if iC is not None: win['cut2ann'] = (iC, iA)
        for k, (a, b) in win.items():
            raw, ab, ab50 = K.cc(c, a, b); o[f'{k}_raw'], o[f'{k}_ab'], o[f'{k}_ab50'] = raw, ab, ab50
        # 成交值放大倍數（驗證事件與生效日）：E 日成交值 / E−25..E−6 中位數
        v = D['value'][c]; base = v.iloc[max(0, iE - 25):iE - 5].median()
        o['vol_ratio_E'] = float(v.iat[iE] / base) if base and base > 0 else np.nan
        o['vol_ratio_Em1'] = float(v.iat[iE - 1] / base) if base and base > 0 else np.nan
        o['liq_ok_A'] = bool(D['liq'][c].iat[iA])
        o['src3_E'] = bool(SRC3[0] <= idx[iE] <= SRC3[1])
        # 交易規則
        o['R1_fill'] = not bool(D['open_lu'][c].iat[iA + 1])
        o['R1_raw'], o['R1_ab'], o['R1_ab50'] = K.oc_trade(c, iA + 1, iE - 1)
        o['R1b_raw'], o['R1b_ab'], o['R1b_ab50'] = K.oc_trade(c, iA + 1, iE)
        if iC is not None:
            o['R0_fill'] = not bool(D['open_lu'][c].iat[iC + 1])
            o['R0_raw'], o['R0_ab'], o['R0_ab50'] = K.oc_trade(c, iC + 1, iE - 1)
        o['R2_fill'] = not bool(D['close_lu'][c].iat[iE])
        o['R2_raw'], o['R2_ab'], o['R2_ab50'] = K.cc(c, iE, iE + 20)
        rows.append(o)
    X = pd.DataFrame(rows)
    X.to_csv(HERE / 'event_returns.csv', index=False)
    return X, D, K


def stats(df, col, cost=0.0, cluster=('etf', 'review')):
    s = df[[col, *cluster]].dropna(subset=[col])
    if len(s) == 0: return {'n': 0}
    x = s[col] - cost
    g = (s.assign(x=x).groupby(list(cluster))['x'].mean())
    return {'n': int(len(x)), 'n_reviews': int(len(g)), 'mean%': round(100 * x.mean(), 2), 'median%': round(100 * x.median(), 2),
            'win%': round(100 * (x > 0).mean(), 1), 't_event': tstat(x), 't_review': tstat(g)}


def summarize(X):
    out = {}
    Xp = X[X.in_panel == True].copy()
    Xp['E'] = pd.to_datetime(Xp.eff); Xp['year'] = Xp.E.dt.year
    Xp['period'] = np.where(Xp.E <= SPLIT, 'select_2015_2019Apr', 'valid_2019May_2026')
    out['coverage'] = X.groupby(['etf', 'action']).agg(n=('code', 'size'), in_panel=('in_panel', 'sum')).reset_index().to_dict('records')
    out['liquidity_excluded'] = int((Xp.liq_ok_A == False).sum())
    Xl = Xp[Xp.liq_ok_A == True]
    car_cols = ['pre20', 'cut2ann', 'ann2effm1', 'effday', 'post20']
    res = {}
    for (etf, act), g in Xl.groupby(['etf', 'action']):
        key = f'{etf}_{act}'; d = {}
        for w in car_cols:
            if f'{w}_ab' in g and g[f'{w}_ab'].notna().any():
                d[f'CAR_{w}'] = stats(g, f'{w}_ab'); d[f'CAR50_{w}'] = stats(g, f'{w}_ab50')
        for rule in ['R1', 'R1b', 'R0', 'R2']:
            if f'{rule}_ab' not in g or g[f'{rule}_ab'].isna().all(): continue
            fill = g[f'{rule}_fill'] if f'{rule}_fill' in g else pd.Series(True, index=g.index)
            gg = g[fill.fillna(False).astype(bool) | (rule == 'R1b')]
            if rule == 'R1b': gg = g[g['R1_fill'].fillna(False).astype(bool)]
            d[rule] = {'gross_ab': stats(gg, f'{rule}_ab'), 'net05_ab': stats(gg, f'{rule}_ab', 0.005), 'net1_ab': stats(gg, f'{rule}_ab', 0.01),
                       'net05_vs0050': stats(gg, f'{rule}_ab50', 0.005), 'raw_net05': stats(gg, f'{rule}_raw', 0.005),
                       'not_fillable': int((~fill.fillna(False).astype(bool)).sum()) if rule != 'R1b' else int((~g['R1_fill'].fillna(False).astype(bool)).sum())}
                # 分期與逐年（淨 0.5%）
            for rule2 in ['R1', 'R1b', 'R0', 'R2']:
                pass
        for rule in ['R1', 'R1b', 'R0', 'R2']:
            if rule not in d: continue
            fillcol = 'R1_fill' if rule == 'R1b' else f'{rule}_fill'
            gg = g[g[fillcol].fillna(False).astype(bool)]
            d[rule]['by_period_net05'] = {p: stats(h, f'{rule}_ab', 0.005) for p, h in gg.groupby('period')}
            d[rule]['by_year_net05'] = {int(y): {'n': int(len(h)), 'mean%': round(100 * (h[f'{rule}_ab'].dropna() - 0.005).mean(), 2) if h[f'{rule}_ab'].notna().any() else None}
                                         for y, h in gg.groupby('year')}
            d[rule]['excl_src3_net05'] = stats(gg[gg.src3_E == False], f'{rule}_ab', 0.005)
            d[rule]['verified_news_only_net05'] = stats(gg[gg.verify == 'news'], f'{rule}_ab', 0.005)
        d['vol_ratio_E_median'] = round(float(g.vol_ratio_E.median()), 2) if g.vol_ratio_E.notna().any() else None
        d['vol_ratio_E_median_official_src'] = round(float(g[g.src3_E == False].vol_ratio_E.median()), 2) if (g.src3_E == False).any() else None
        res[key] = d
    out['results'] = res
    return out, Xl


def detect_eff(D, X):
    """以成交值放大找出實際換股日：在 E_guess ±5 交易日內，取群組成交值倍數中位數最大日。"""
    idx = D['idx']; V = D['value']; out = []
    for (etf, rev), g in X[X.in_panel == True].groupby(['etf', 'review']):
        iE = int(idx.get_loc(pd.Timestamp(g.eff.iloc[0]))); codes = [c for c in g.code if c in V.columns]
        best = None
        for k in range(-5, 6):
            i = iE + k
            if i < 26 or i >= len(idx): continue
            r = [V[c].iat[i] / V[c].iloc[i - 25:i - 5].median() for c in codes if V[c].iloc[i - 25:i - 5].median() > 0]
            m = float(np.nanmedian(r)) if r else np.nan
            if best is None or (np.isfinite(m) and m > best[1]): best = (k, m)
        out.append({'etf': etf, 'review': rev, 'E_rule': str(g.eff.iloc[0]), 'best_offset': best[0], 'best_med_ratio': round(best[1], 2),
                    'E_detected': str(idx[iE + best[0]].date()), 'src3': bool(SRC3[0] <= idx[iE] <= SRC3[1])})
    return pd.DataFrame(out)


def predict_recent(D, X):
    """近期樣本（股本只有 2025-09 起）：截止日 C 的總市值排名 vs 實際增刪。"""
    sh = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/shares_hist.csv', dtype={'code': str})
    sl = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype=str)
    stocks = set(sl[(sl.market == 'TWSE')].code)
    out = {}
    rv = pd.read_csv(HERE / 'reviews.csv')
    rv = rv[(rv.etf == 'TW50') & (rv.review >= '2025-12')]
    prev_rank = None
    for _, r in rv.iterrows():
        C = pd.Timestamp(r.cutoff); ym = C.strftime('%Y-%m')
        s = sh[sh.ym == ym].set_index('code')['shares']
        px = D['close_raw'].loc[C]
        cols = [c for c in s.index if c in px.index and c in stocks and not c.startswith('00') and np.isfinite(px.get(c, np.nan))]
        mc = (px[cols] * s[cols]).dropna().sort_values(ascending=False)
        rank = pd.Series(np.arange(1, len(mc) + 1), index=mc.index)
        g = X[(X.etf == 'TW50') & (X.review == r.review)]
        adds = list(g[g.action == 'add'].code); dels = list(g[g.action == 'del'].code)
        top40 = list(rank[rank <= 40].index)
        # 前一期排名 >60 但本期 ≤40、卻沒被納入者（若存在 → 規則預測會誤判）
        new_top = [c for c in top40 if prev_rank is not None and prev_rank.get(c, 999) > 60]
        out[r.review] = {'cutoff': str(C.date()), 'add_ranks': {c: int(rank.get(c, -1)) for c in adds}, 'del_ranks': {c: int(rank.get(c, -1)) for c in dels},
                         'jumped_from_gt60_to_top40': {c: int(rank[c]) for c in new_top},
                         'jumped_but_not_added': [c for c in new_top if c not in adds]}
        prev_rank = rank
    return out


if __name__ == '__main__':
    X, D, K = run()
    out, Xl = summarize(X)
    out['eff_detect'] = detect_eff(D, X).to_dict('records')
    out['predict_recent_tw50'] = predict_recent(D, X)
    (HERE / 'results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k in ['coverage', 'liquidity_excluded']}, ensure_ascii=False))
