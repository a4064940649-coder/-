"""事件表：股池（A 日收盤）、SUE⊥（前 4 季基準）、關注分組、事件報酬（j=0..9，60 日開盤到開盤）。"""
from pathlib import Path
import json
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
D = HERE / 'data'
P = np.load(D / 'panel.npz', allow_pickle=True)
dates = P['dates']; codes = list(P['codes'])
dt = pd.to_datetime(dates)
cidx = {c: i for i, c in enumerate(codes)}
close, opn, amt, ret, mkt = P['close'], P['open'], P['amount'], P['ret'], P['mkt']
T, N = close.shape

# 填補：無交易日價格不動
cl_ff = pd.DataFrame(close).ffill().values
r0 = np.nan_to_num(ret, nan=0.0)
I = np.cumprod(1 + r0, axis=0)
adjc = I  # 含息指數（以第一日為 1 的相對值，但同一檔內比值正確）
fac = I / cl_ff
adjo = opn * fac
amt0 = np.where(np.isfinite(close), np.nan_to_num(amt), 0.0)
listed = np.isfinite(cl_ff)
v60 = pd.DataFrame(np.where(listed, amt0, np.nan)).rolling(60, min_periods=40).mean().values


# 0050 PIT 成分
mem = json.load(open('/home/user/-/research/tw_stock_selection/agents/o_0050_tilt_pit/data/membership_pit.json'))
mk = sorted(mem)
mem_dates = pd.to_datetime(mk)
is50 = np.zeros((T, N), bool)
for i, d0 in enumerate(mk):
    a = dt.searchsorted(mem_dates[i])
    b = dt.searchsorted(mem_dates[i + 1]) if i + 1 < len(mk) else T
    cols = [cidx[c] for c in mem[d0] if c in cidx]
    is50[a:b, cols] = True

fin = pd.read_parquet(D / 'fin.parquet')
av = pd.read_parquet(D / 'avail.parquet')
ev = av.merge(fin[['stock_id', 'qi', 'industry', 'sue', 'revsue', 'eps', 'rev']], on=['stock_id', 'qi'])
ev = ev[ev.stock_id.str.fullmatch(r'[1-9]\d{3}') & ~ev.stock_id.str.startswith('91')]
ev = ev[ev.industry == 'ci']
ev['j'] = ev.stock_id.map(cidx)
ev = ev[ev.j.notna()]
ev['j'] = ev.j.astype(int)
t, j = ev.A_pos.values, ev.j.values
ev['close_A'] = close[t, j]; ev['v60'] = v60[t, j]; ev['is50'] = is50[t, j]; ev['mkt'] = mkt[t, j]
# 股池日截面（含 0050 與否兩版）、低關注三分位門檻
ok_code = np.array([c[0] != '0' and not c.startswith('91') for c in codes])
pool_day = np.isfinite(v60) & (v60 >= 1e7) & (cl_ff >= 10) & ok_code[None, :] & np.isfinite(close)
ev['in_pool'] = (ev.v60 >= 1e7) & (ev.close_A >= 10)
ev = ev[ev.in_pool & ev.sue.notna() & ev.revsue.notna()].copy()
# 日截面三分位（股池、非 0050）
thr_lo = np.full(T, np.nan); thr_hi = np.full(T, np.nan)
for tt in np.unique(ev.A_pos.values):
    m = pool_day[tt] & ~is50[tt]
    x = v60[tt, m]
    if len(x) > 30:
        thr_lo[tt], thr_hi[tt] = np.quantile(x, [1 / 3, 2 / 3])
ev['att'] = np.where(ev.is50, 'X50', np.where(ev.v60 <= thr_lo[ev.A_pos.values], 'L',
                                               np.where(ev.v60 <= thr_hi[ev.A_pos.values], 'M', 'H')))
j = ev.j.values
# 事件報酬 j=0..9：開盤 e 到開盤 e+60，含息；下市則以最後收盤
H = 60
for lag in range(10):
    e = ev.A_pos.values + 1 + lag
    x = e + H
    ok = x < T
    ee = np.minimum(e, T - 1); xx = np.minimum(x, T - 1)
    po = adjo[ee, j]
    px = adjo[xx, j]
    # 出場日無開盤 → 用之後第一個開盤；下市 → 最後收盤指數
    bad = ~np.isfinite(px)
    px = np.where(bad, adjc[xx, j], px)
    rr = px / po - 1
    rr[~ok] = np.nan
    ev[f'r{lag}'] = rr
# 累積異常報酬曲線用：第 h 日收盤
for h in [10, 20, 40, 60, 120]:
    e = ev.A_pos.values + 1
    xx = np.minimum(e + h - 1, T - 1)
    rr = adjc[xx, j] / adjo[np.minimum(e, T - 1), j] - 1
    rr[e + h - 1 >= T] = np.nan
    ev[f'c{h}'] = rr
ev['kq'] = ev.qi
# 關注分組改為：同財報季股池事件（含 0050）的 60 日均成交值三分位
q1 = ev.groupby('kq').v60.transform(lambda s: s.quantile(1 / 3)); q2 = ev.groupby('kq').v60.transform(lambda s: s.quantile(2 / 3))
ev['att'] = np.where(ev.is50, 'X50', np.where(ev.v60 <= q1, 'L', np.where(ev.v60 <= q2, 'M', 'H')))
print(len(ev), ev.groupby('period').size().head(5).to_dict(), ev.is50.sum())


def ecdf_pct(base, x):
    b = np.sort(base)
    return np.searchsorted(b, x, side='right') / len(b)


def orth(ev, pool_mask_for_base, sue_col='sue'):
    """回傳 SUE⊥、pct(SUE)、門檻（p20/p80）。"""
    out = pd.DataFrame(index=ev.index, columns=['psue', 'prev', 'sperp', 'thr80', 'thr20', 'pthr80', 'pthr20', 'nbase'], dtype=float)
    qs = sorted(ev.kq.unique())
    for k in qs:
        bq = [q for q in qs if k - 4 <= q < k]
        base = ev[ev.kq.isin(bq) & pool_mask_for_base]
        if base.kq.nunique() < 2:
            continue
        cur = ev.index[ev.kq == k]
        bs, br = base[sue_col].values, base.revsue.values
        pbs, pbr = ecdf_pct(bs, bs), ecdf_pct(br, br)
        b1, a1 = np.polyfit(pbr, pbs, 1)
        bperp = pbs - (a1 + b1 * pbr)
        ps = ecdf_pct(bs, ev.loc[cur, sue_col].values); pr = ecdf_pct(br, ev.loc[cur, 'revsue'].values)
        out.loc[cur, 'psue'] = ps; out.loc[cur, 'prev'] = pr
        out.loc[cur, 'sperp'] = ps - (a1 + b1 * pr)
        out.loc[cur, 'thr80'], out.loc[cur, 'thr20'] = np.quantile(bperp, 0.8), np.quantile(bperp, 0.2)
        out.loc[cur, 'pthr80'], out.loc[cur, 'pthr20'] = np.quantile(pbs, 0.8), np.quantile(pbs, 0.2)
        out.loc[cur, 'nbase'] = len(base)
    return out


for tag, mask in [('all', np.ones(len(ev), bool))]:
    o = orth(ev, mask)
    e2 = pd.concat([ev, o], axis=1)
    e2 = e2[e2.sperp.notna()]
    e2['top'] = e2.sperp >= e2.thr80; e2['bot'] = e2.sperp <= e2.thr20
    e2['top_raw'] = e2.psue >= e2.pthr80
    e2.to_parquet(D / f'events_{tag}.parquet', index=False)
    sel = e2.period.between('2016Q1', '2019Q1'); oos = e2.period.between('2019Q2', '2026Q2')
    print(tag, 'pool events(with perp)', len(e2), 'raw pool', int(mask.sum()), 'sel', int(sel.sum()), 'oos', int(oos.sum()),
          'selL', int((sel & (e2.att == 'L')).sum()), 'oosL', int((oos & (e2.att == 'L')).sum()),
          'oosOTC', int((oos & (e2.mkt == 2)).sum()), 'first', e2.period.min(),
          'sel ex50', int((sel & ~e2.is50).sum()), 'oos ex50', int((oos & ~e2.is50).sum()))
