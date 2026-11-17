"""S66 共用：載入、含息還原、統計。只讀本地資料；不連外網。"""
import glob, hashlib, json, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s66')
NAV_CSV = '/home/user/-/xq_scripts/site/navs_official.csv.gz'
TWT49U = '/home/user/-/research/tw_stock_selection/raw/twt49u/twse-exright'
FINLAB = '/home/user/jonatw/finlab-txd/data/raw/'
EXTRAS = '/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/'
STRAT = 'E0|W10_70'
END = pd.Timestamp('2026-10-22')

MAIN = ['0050', '006208', '00692', '00850', '0051', '00733', '0056', '00878', '00919', '00929', '00713']
SUPP = ['00900', '00915', '00918', '00940', '00939', '009803']
CAT = {'0050': '市值型', '006208': '市值型', '00692': '市值型（公司治理）', '00850': '市值型（ESG）', '0051': '中小型（中型 100）',
       '00733': '中小型＋動能', '0056': '高股息', '00878': '高股息', '00919': '高股息', '00929': '高股息（科技）', '00713': '高股息＋低波',
       '00900': '高股息', '00915': '高股息', '00918': '高股息', '00940': '高股息', '00939': '高股息＋動能', '009803': '市值＋動能'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_daily():
    d = pd.read_parquet(TMP / 'etf_daily.parquet'); d['date'] = pd.to_datetime(d.date)
    return d


def load_div():
    e = pd.concat([pd.read_csv(f, dtype={'code': str}) for f in sorted(glob.glob(TWT49U + '/*/*.csv.gz'))], ignore_index=True)
    e['date'] = pd.to_datetime(e.date)
    return e


def build_tr(d, e, code):
    """含息還原日淨值（起始＝第一個有收盤的交易日收盤＝1）與不含息價格淨值。
    P＝收盤（無成交日沿用前一收盤）；除息日參考價＝前收−息值（twt49u；事件日非交易日者順延到下一個交易日）；
    分割日（|收盤÷前收−1| > 30%）參考價＝官方（收盤−漲跌價差），倍數＝前收÷參考價。日報酬＝收盤÷參考價−1。"""
    s = d[d.stock_id == code].sort_values('date').set_index('date')
    P = s.close.ffill().dropna(); chg = s.change.reindex(P.index)
    ev = e[(e.code == code) & (e.date > P.index[0])]
    D = pd.Series(0.0, index=P.index); unmapped = 0
    for t, v in zip(ev.date, ev.value):
        i = P.index.searchsorted(t)
        if i >= len(P): unmapped += 1; continue
        D.iloc[i] += float(v)
    pc = P.shift(1); ref = pc - D; splits = []
    raw = P / pc - 1
    for t in raw.index[raw.abs() > 0.30]:
        r_off = P[t] - chg[t]; k = pc[t] / r_off
        splits.append((str(t.date()), round(float(k), 4))); ref[t] = r_off
    r = (P / ref - 1).fillna(0.0)
    tr = (1 + r).cumprod()
    # 不含息：只還原分割
    r_px = raw.copy()
    for t, _ in splits: r_px[pd.Timestamp(t)] = P[pd.Timestamp(t)] / ref[pd.Timestamp(t)] - 1
    px = (1 + r_px.fillna(0.0)).cumprod()
    return tr, px, dict(n_div=int((D > 0).sum()), div_sum=float(D.sum()), splits=splits, unmapped=unmapped, first=str(P.index[0].date()), last=str(P.index[-1].date()))


def nw_t(x, lag=6):
    x = np.asarray(x, float); T = len(x); m = x.mean(); e = x - m
    lr = (e * e).sum() / T
    for i in range(1, lag + 1):
        lr += 2 * (1 - i / (lag + 1)) * (e[i:] * e[:-i]).sum() / T
    return float(m / math.sqrt(lr / T)) if lr > 0 else float('nan')


def month_points(s):
    """基底日＋每個月最後一個交易日（含期末不滿月的最後一日）。"""
    g = s.groupby(s.index.to_period('M')).tail(1)
    if g.index[0] != s.index[0]:
        g = pd.concat([s.iloc[:1], g])
    return g


def cagr(s):
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1) if yrs > 0 else float('nan')


def mdd(s):
    return float((s / s.cummax() - 1).min())


def stats(nav, bench, a, b):
    """a（基底日）～b；nav 與 bench 取共同交易日。"""
    s = nav.loc[a:b].dropna(); q = bench.reindex(s.index).ffill()
    cg, cb = cagr(s), cagr(q); dd = mdd(s)
    rs = month_points(s).pct_change().dropna(); rb = month_points(q).pct_change().dropna()
    ex = (rs - rb.reindex(rs.index)).to_numpy(); n = len(ex)
    sd = ex.std(ddof=1) if n > 1 else float('nan')
    dr = s.pct_change().dropna()
    return dict(a=str(s.index[0].date()), b=str(s.index[-1].date()), years=round((s.index[-1] - s.index[0]).days / 365.25, 2),
                cagr=cg, total=float(s.iloc[-1] / s.iloc[0] - 1), vol=float(dr.std(ddof=1) * math.sqrt(252)), maxdd=dd,
                calmar=(cg / abs(dd) if dd < 0 else float('nan')), cagr_0050=cb, maxdd_0050=mdd(q), geo_diff_pp=(cg - cb) * 100,
                months=int(n), ex_ann_pp=float(ex.mean() * 12 * 100) if n else float('nan'),
                te_ann_pp=float(sd * math.sqrt(12) * 100) if n > 1 else float('nan'),
                t_nw6=nw_t(ex, 6) if n >= 13 and sd > 0 else float('nan'))


def combo(a_nav, b_nav, w_a, start, end):
    """a 佔 w_a、b 佔 1−w_a；基底日建立，之後每年第一個交易日收盤再平衡回目標比例；不扣交易成本。"""
    idx = a_nav.loc[start:end].index.intersection(b_nav.loc[start:end].index)
    ra = a_nav.reindex(idx).pct_change().fillna(0).to_numpy(); rb = b_nav.reindex(idx).pct_change().fillna(0).to_numpy()
    va, vb = w_a, 1 - w_a; out = np.empty(len(idx)); yr = idx[0].year
    for i, t in enumerate(idx):
        va *= 1 + ra[i]; vb *= 1 + rb[i]
        if t.year != yr:
            tot = va + vb; va, vb = tot * w_a, tot * (1 - w_a); yr = t.year
        out[i] = va + vb
    return pd.Series(out, index=idx)
