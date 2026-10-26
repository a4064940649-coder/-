"""產業研究共用函式（industry 組）。

約定（事前固定）：
* 產業別：stock_list.csv 的 industry（現行分類）。已下市、不在 stock_list 的股票：傳統產業代碼段（11~29 開頭、
  23/24 除外）依代碼前兩碼回推產業；電子股（23/24/30~/6x…）、TDR（91）無法回推 → 不列入任何產業（偏誤，報告註明）。
* 價格類訊號一律用「換股日前一個交易日」收盤資料（signal lag 1 日），在換股日收盤成交；換股日收盤鎖漲停者視為買不到。
* 合格股：前一日 60 日均成交值 ≥ NT$1,000 萬、前一日收盤 ≥ 10 元、換股日有成交、換股日非鎖漲停。
* 成本：權重周轉（單邊買入權重和）× 來回成本 0.5%（壓力 1%）。
"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path('/home/user/-/research/tw_stock_selection')
sys.path.insert(0, str(ROOT))
from btlib import month_ends, period_returns, bench_returns, ann, yearly, tstat  # noqa: E402

P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
DATA = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data')
X = DATA / 'extras'
OUT = Path(__file__).resolve().parent

PREFIX_IND = {'11': '水泥工業', '12': '食品工業', '13': '塑膠工業', '14': '紡織纖維', '15': '電機機械', '16': '電器電纜',
              '17': '化學工業', '18': '玻璃陶瓷', '19': '造紙工業', '20': '鋼鐵工業', '21': '橡膠工業', '22': '汽車工業',
              '25': '建材營造業', '26': '航運業', '27': '觀光餐旅', '28': '金融保險業', '29': '貿易百貨業', '99': '其他業'}

_cache = {}


def load():
    if _cache: return _cache
    adj = pd.read_parquet(P / 'adj.parquet'); C = pd.read_parquet(P / 'close_raw.parquet')
    H = pd.read_parquet(P / 'high.parquet'); V = pd.read_parquet(P / 'value.parquet'); RET = pd.read_parquet(P / 'ret.parquet')
    sl = pd.read_csv(DATA / 'stock_list.csv', dtype={'code': str})
    ind = sl.drop_duplicates('code').set_index('code')['industry']
    m = {}
    n_pref = 0
    for c in adj.columns:
        if c in ind.index and isinstance(ind[c], str): m[c] = ind[c]
        elif c[:2] in PREFIX_IND: m[c] = PREFIX_IND[c[:2]]; n_pref += 1
    IND = pd.Series(m)
    chg = C / C.shift(1) - 1
    thr = pd.Series(np.where(C.index < pd.Timestamp('2015-06-01'), 0.065, 0.095), index=C.index)
    LU = chg.ge(thr, axis=0) & (C >= H - 1e-9)
    V60 = V.rolling(60, min_periods=50).mean()
    e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
    _cache.update(adj=adj, C=C, V=V, RET=RET, IND=IND, LU=LU, V60=V60, e50=e50, n_prefix_mapped=n_pref,
                  n_unclassified=int(len(adj.columns) - len(IND)))
    return _cache


def prev_day(dates, index):
    """每個換股日的前一個交易日。"""
    pos = index.get_indexer(pd.DatetimeIndex(dates))
    return [index[p - 1] for p in pos]


def eligibility(dates):
    d = load(); adj, C, V60, LU = d['adj'], d['C'], d['V60'], d['LU']
    pdays = prev_day(dates, adj.index)
    v = V60.reindex(pdays).set_axis(pd.DatetimeIndex(dates)); px = C.reindex(pdays).set_axis(pd.DatetimeIndex(dates))
    trd = adj.reindex(dates).notna() & C.reindex(dates).notna()
    lu = LU.reindex(dates).fillna(False)
    el = (v >= 1e7) & (px >= 10) & trd & ~lu
    return el, v


def industry_daily_index():
    """產業等權日指數（成分＝前一日合格的股票：60 日均值 ≥1,000 萬、價 ≥10）。"""
    d = load()
    if 'IDX' in d: return d['IDX']
    RET, V60, C, IND = d['RET'], d['V60'], d['C'], d['IND']
    ok = ((V60.shift(1) >= 1e7) & (C.shift(1) >= 10)) & RET.notna()
    r = RET.where(ok).clip(-0.5, 0.5)
    cols = [c for c in r.columns if c in IND.index]
    g = r[cols].T.groupby(IND[cols]).mean().T
    n = ok[cols].T.groupby(IND[cols]).sum().T
    g = g.where(n >= 3).fillna(0.0)
    IDX = (1 + g).cumprod()
    mkt = r.mean(axis=1).fillna(0.0); MKT = (1 + mkt).cumprod()
    d['IDX'], d['IDX_N'], d['MKT_EW'] = IDX, n, MKT
    return IDX


def build_weights(t, inds, el_t, v_t, mode, n_per=10):
    """mode='liq10'：每產業流動性前 n_per 檔，全部名稱等權；'indEW'：產業等權、產業內全合格股等權。"""
    IND = load()['IND']
    w = {}
    if not inds: return w
    if mode == 'liq10':
        names = []
        for k in inds:
            mem = [c for c in el_t.index[el_t.values] if IND.get(c) == k]
            names += list(v_t[mem].sort_values(ascending=False).index[:n_per])
        for c in names: w[c] = 1.0 / len(names)
    else:
        for k in inds:
            mem = [c for c in el_t.index[el_t.values] if IND.get(c) == k]
            for c in mem: w[c] = w.get(c, 0) + 1.0 / len(inds) / len(mem)
    return w


def run_weights(W, R, extra_ret=None):
    """W: {t: {code: w}}；R: 期間報酬。extra_ret: {t: {'name': ret}} 非股票資產（如 0050）。回傳 gross, turnover series。"""
    g, to, prev = {}, {}, {}
    for t in R.index:
        if t not in W: continue
        w = W[t]
        if not w: continue
        rr = 0.0; tot = 0.0
        for c, x in w.items():
            if extra_ret is not None and c in extra_ret.get(t, {}):
                r = extra_ret[t][c]
            else:
                r = R.at[t, c] if c in R.columns else np.nan
            if pd.isna(r): r = 0.0  # 持有期停牌/下市：period_returns 已用最後成交價（ffill）；剩餘 NaN 視為 0
            rr += x * r; tot += x
        g[t] = rr / tot if tot > 0 else np.nan
        to[t] = sum(max(x - prev.get(c, 0.0), 0.0) for c, x in w.items())
        prev = w
    return pd.Series(g, dtype=float), pd.Series(to, dtype=float)


def stats(gross, to, dates, B, U=None, start=None, end=None, rt=0.005):
    net = gross - to.reindex(gross.index).fillna(0) * rt
    net1 = gross - to.reindex(gross.index).fillna(0) * 0.01
    def cut(x):
        x = x.dropna()
        if start is not None: x = x[x.index >= pd.Timestamp(start)]
        if end is not None: x = x[x.index <= pd.Timestamp(end)]
        return x
    net, net1, gross = cut(net), cut(net1), cut(gross)
    if len(net) < 6: return {'periods': int(len(net))}
    b = B.reindex(net.index); ex = net - b
    o = {'net': ann(net, dates), 'net_1pct': ann(net1, dates).get('CAGR'), 'gross_CAGR': ann(gross, dates).get('CAGR'),
         '0050': ann(b, dates), 'turnover_mean': round(float(to.reindex(net.index).mean()), 3),
         'ex_vs_0050_pct_m': round(float(ex.mean() * 100), 3), 't_vs_0050': tstat(ex)}
    if U is not None:
        u = U.reindex(net.index); o['EW_univ'] = ann(u, dates); o['t_vs_EW'] = tstat(net - u)
        o['ex_vs_EW_pct_m'] = round(float((net - u).mean() * 100), 3)
    ys, yb = yearly(net, dates), yearly(b, dates)
    o['yearly_net'] = ys; o['yearly_0050'] = yb
    o['years_beat_0050'] = f"{sum(ys[y] > yb.get(y, -1e9) for y in ys)}/{len(ys)}"
    return o


def brief(s):
    """精簡一行。"""
    if 'net' not in s: return s
    return {'CAGR': s['net'].get('CAGR'), 'CAGR_1pct': s.get('net_1pct'), 'maxDD': s['net'].get('maxDD'), 'vol': s['net'].get('vol'),
            '0050_CAGR': s['0050'].get('CAGR'), 'EW_CAGR': s.get('EW_univ', {}).get('CAGR'), 'ex_0050_m%': s['ex_vs_0050_pct_m'],
            't_0050': s['t_vs_0050'], 't_EW': s.get('t_vs_EW'), 'TO': s['turnover_mean'], 'n': s['net'].get('periods'),
            'beat': s['years_beat_0050']}


def universe_ew(R, el):
    return pd.Series({t: R.loc[t][el.loc[t] & R.loc[t].notna()].mean() for t in R.index}, dtype=float)


def save(name, obj):
    p = OUT / name
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    return p
