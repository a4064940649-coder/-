"""組合回測（MAIN＝SUE⊥ 前 1/5、非 0050、20 槽、持有 60 日、空槽持有 0050），K5T 近似成本。
用法：python3 -I -B r6_port.py
"""
from pathlib import Path
import json, sys
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
D = HERE / 'data'
P = np.load(D / 'panel.npz', allow_pickle=True)
dates = P['dates']; codes = list(P['codes']); dt = pd.to_datetime(dates)
close, opn, amt, ret, ref = P['close'], P['open'], P['amount'], P['ret'], P['ref']
T, N = close.shape
cl_ff = pd.DataFrame(close).ffill().values
r0 = np.nan_to_num(ret, nan=0.0)
I = np.cumprod(1 + r0, axis=0)
fac = I / cl_ff
adjo = opn * fac
amt0 = np.where(np.isfinite(close), np.nan_to_num(amt), 0.0)
listed = np.isfinite(cl_ff)
V60 = pd.DataFrame(np.where(listed, amt0, np.nan)).rolling(60, min_periods=40).mean().values
SIG = pd.DataFrame(np.where(np.isfinite(close), ret, np.nan)).rolling(60, min_periods=40).std().values
del amt0
j50 = codes.index('0050')
_e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date').adj
_r50 = _e50.pct_change().reindex(dt).fillna(pd.Series(ret[:, j50], index=dt)).fillna(0).values  # 0050 含息用 etf_0050.csv（2026-10-22 後補面板）
I50 = np.cumprod(1 + _r50); adjo50 = opn[:, j50] * I50 / cl_ff[:, j50]


def tick(p):
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)


def floor_tick(x):
    t = tick(x)
    return np.floor(x / t + 1e-9) * t


LIMUP = floor_tick(ref * 1.1)
LIMDN = np.ceil(ref * 0.9 / tick(ref * 0.9) - 1e-9) * tick(ref * 0.9)

# 處置期間（2021-09 起官方）
disp = np.zeros((T, N), bool)
for f in ['official_trades_twse.csv', 'official_trades_tpex.csv']:
    x = pd.read_csv('/home/user/-/research/tw_stock_selection/agents/disposal_reconcile/' + f, dtype={'code': str})
    for _, r in x.iterrows():
        if r.code in codes:
            a = dt.searchsorted(pd.Timestamp(r.date)); b = dt.searchsorted(pd.Timestamp(r.exit))
            disp[a:b, codes.index(r.code)] = True

E = pd.read_parquet(D / 'events_all.parquet')
E = E[~E.is50]
FEE = 0.001425 * 0.28; TAX = 0.003; TAX50 = 0.001
D_INTRA = pd.Timestamp('2020-10-26')


def fee(v):
    return max(20.0, v * FEE) if v > 0 else 0.0


def odd_c(p, stress):
    c = tick(p) / p + 0.001  # 價格桶近似：一跳＋0.1%
    return c * (2 if stress else 1)


def run(cands, start, end, cap0=1e6, mode='GROW', stress=False, nslot=20, hold=60, lag=0, rec=False):
    """cands: DataFrame with A_pos, j, sperp. 回傳日 NAV Series 與 0050 NAV。"""
    t0 = dt.searchsorted(pd.Timestamp(start)); t1 = dt.searchsorted(pd.Timestamp(end), side='right') - 1
    fee_m = 0.6 if stress else 1.0
    extra = 0.004 if stress else 0.0
    # 候選依進場日分組
    c = cands.copy(); c['e'] = c.A_pos + 1 + lag
    c = c[(c.e >= t0) & (c.e <= t1)].sort_values(['e', 'sperp'], ascending=[True, False])
    byday = {}
    for e, j, s in zip(c.e.values, c.j.values, c.sperp.values):
        byday.setdefault(int(e), []).append([int(j), 0])
    u50 = cap0 / adjo50[t0]  # 起始全放 0050（開盤買，忽略首筆成本）
    pos = []  # dict(j, u (adj units), exit_t)
    nav = np.full(t1 - t0 + 1, np.nan); navprev = cap0
    stats = dict(n_buy=0, n_skip_full=0, n_skip_lim=0, n_disp=0, fill=[], part=[])
    pending = []
    for t in range(t0, t1 + 1):
        p50o = adjo50[t] if np.isfinite(adjo50[t]) else I50[t - 1] * 1.0
        # 出場
        keep = []
        for p in pos:
            j = p['j']
            if t >= p['xt']:
                o = opn[t, j]
                if (not np.isfinite(o)) or o <= LIMDN[t, j] + 1e-9:
                    if np.isfinite(cl_ff[t, j]) and t - p['xt'] < 60 and np.isfinite(close[min(t + 1, T - 1):min(t + 40, T), j]).any():
                        keep.append(p); continue
                    # 下市：以最後收盤指數清算
                    val = p['u'] * I[t, j] / 1.0
                else:
                    sh_val = p['u'] * adjo[t, j]
                    q = sh_val
                    imp = max(0.001 + 0.5 * np.nan_to_num(SIG[t - 1, j], nan=0.03) * np.sqrt(q / max(0.03 * np.nan_to_num(amt[t, j], nan=1.0), 1.0)), tick(o) / o) + extra
                    val = sh_val * (1 - imp)
                val = val - fee(val) * fee_m - TAX * val
                u50 += val * (1 - FEE * fee_m) / p50o
            else:
                keep.append(p)
        pos = keep
        # 進場（含漲停順延）
        todays = byday.get(t, []) + pending
        pending = []
        for item in todays:
            j, tries = item
            if len(pos) >= nslot:
                stats['n_skip_full'] += 1; continue
            if any(p['j'] == j for p in pos):
                continue
            o = opn[t, j]
            if not np.isfinite(o) or o >= LIMUP[t, j] - 1e-9:
                if tries < 2 and t + 1 <= t1:
                    pending.append([j, tries + 1])
                else:
                    stats['n_skip_lim'] += 1
                continue
            if disp[t, j]:
                stats['n_disp'] += 1; continue
            slot = (navprev if mode == 'GROW' else cap0) / nslot
            avail = u50 * p50o * (1 - TAX50 - FEE * fee_m)
            slot = min(slot, avail)
            if slot < 1000:
                continue
            v60 = np.nan_to_num(V60[t - 1, j])
            # 整股
            lot_cap = 0.2 * 0.03 * v60
            n_lots = int(min(slot, lot_cap) // (1000 * o))
            q_lot = n_lots * 1000 * o
            sig = np.nan_to_num(SIG[t - 1, j], nan=0.03)
            oa = max(0.03 * np.nan_to_num(amt[t, j]), 1.0)
            imp = (max(0.001 + 0.5 * sig * np.sqrt(q_lot / oa), tick(o) / o) + extra) if q_lot > 0 else 0
            # 零股
            rem = slot - q_lot
            odd_cap = 0.000193 * v60 * (1 if t >= dt.searchsorted(D_INTRA) else 0.5)
            podd = o if dt[t] >= D_INTRA else close[t, j]
            if not np.isfinite(podd):
                podd = o
            n_odd = int(min(rem, odd_cap) // (podd * (1 + odd_c(podd, stress))))
            q_odd = n_odd * podd
            cost_lot = q_lot * (1 + imp); cost_odd = q_odd * (1 + odd_c(podd, stress))
            spent = cost_lot + cost_odd
            if spent <= 0:
                stats['fill'].append(0.0); continue
            fees = (fee(q_lot) if q_lot > 0 else 0) * fee_m + (fee(q_odd) if q_odd > 0 else 0) * fee_m
            # 賣 0050 籌款
            need = spent + fees
            u50 -= need / (p50o * (1 - TAX50 - FEE * fee_m))
            # 轉為含息單位：整股按開盤、零股按成交基準
            # 股數 ÷ 當日還原因子 → 含息指數單位（價值＝u × I）
            u = (q_lot / o) / fac[t, j]
            if q_odd > 0:
                u += (q_odd / podd) / fac[t, j]
            pos.append(dict(j=j, u=u, xt=t + hold, et=t))
            stats['n_buy'] += 1; stats['fill'].append(spent / slot)
            stats['part'].append(q_lot / max(0.03 * v60, 1))
        # 收盤估值
        v = u50 * I50[t]
        for p in pos:
            v += p['u'] * I[t, p['j']]
        nav[t - t0] = v; navprev = v
    s = pd.Series(nav, index=dt[t0:t1 + 1])
    b = pd.Series(I50[t0:t1 + 1] / adjo50[t0] * cap0, index=dt[t0:t1 + 1])
    return s, b, stats


def summ(s, b):
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    a = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1
    ab = (b.iloc[-1] / b.iloc[0]) ** (1 / yrs) - 1
    ms = s.resample('ME').last(); mb = b.resample('ME').last()
    ms = pd.concat([pd.Series([s.iloc[0]], index=[s.index[0]]), ms]); mb = pd.concat([pd.Series([b.iloc[0]], index=[b.index[0]]), mb])
    ex = (ms.pct_change() - mb.pct_change()).dropna()
    t = ex.mean() / ex.std(ddof=1) * np.sqrt(len(ex))
    mdd = (s / s.cummax() - 1).min()
    return dict(ann=round(a * 100, 2), b50=round(ab * 100, 2), ex=round((a - ab) * 100, 2), t=round(t, 2), mdd=round(mdd * 100, 1))


def yearly(s, b):
    out = {}
    for y in sorted(set(s.index.year)):
        x = s[s.index.year == y]; z = b[b.index.year == y]
        prev_s = s[s.index.year < y]; prev_b = b[b.index.year < y]
        s0 = prev_s.iloc[-1] if len(prev_s) else x.iloc[0]; b0 = prev_b.iloc[-1] if len(prev_b) else z.iloc[0]
        out[y] = round((x.iloc[-1] / s0 - z.iloc[-1] / b0) * 100, 1)
    return out


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'main'
    res = {}
    top = E[E.top]
    OOS = top[top.period.between('2019Q2', '2026Q2')]
    ALL = top[top.period.between('2016Q1', '2026Q2')]
    SEL = top[top.period.between('2016Q1', '2019Q1')]
    cfgs = {
        'main': [('OOS_100_GROW', OOS, '2019-05-01', '2026-10-30', 1e6, 'GROW', False)],
        'more': [('OOS_100_STRESS', OOS, '2019-05-01', '2026-10-30', 1e6, 'GROW', True),
                 ('OOS_300_GROW', OOS, '2019-05-01', '2026-10-30', 3e6, 'GROW', False),
                 ('OOS_1000_GROW', OOS, '2019-05-01', '2026-10-30', 1e7, 'GROW', False),
                 ('OOS_100_FIX', OOS, '2019-05-01', '2026-10-30', 1e6, 'FIX', False),
                 ('SEL_100_GROW', SEL, '2016-04-01', '2019-08-30', 1e6, 'GROW', False)],
        'rep': [('OOS_R1_LOWATT', OOS[OOS.att == 'L'], '2019-05-01', '2026-10-30', 1e6, 'GROW', False),
                ('OOS_R2_RAWSUE', E[E.top_raw & E.period.between('2019Q2', '2026Q2')], '2019-05-01', '2026-10-30', 1e6, 'GROW', False)],
        'full': [('FULL_100_GROW', ALL, '2016-04-01', '2026-10-30', 1e6, 'GROW', False)],
    }[which]
    for name, cands, st, en, cap, mode, stress in cfgs:
        rows = []; navs = {}
        for lag in range(10):
            s, b, stt = run(cands, st, en, cap, mode, stress, lag=lag)
            r = summ(s, b)
            r['n_buy'] = stt['n_buy']; r['skip_full'] = stt['n_skip_full']; r['fill_med'] = round(float(np.median(stt['fill'])), 3) if stt['fill'] else None
            r['part_p90'] = round(float(np.quantile(stt['part'], 0.9)), 3) if stt['part'] else None
            rows.append(r); navs[lag] = (s, b)
        df = pd.DataFrame(rows)
        med_lag = int((df.ex - df.ex.median()).abs().idxmin())
        out = dict(ex_median=round(float(df.ex.median()), 2), ex_p10=round(float(df.ex.quantile(0.1)), 2),
                   t_median=round(float(df.t.median()), 2), ann_median=round(float(df.ann.median()), 2), b50=float(df.b50.iloc[0]),
                   mdd_median=float(df.mdd.median()), j0=rows[0], per_j=df.ex.tolist())
        s, b = navs[med_lag]
        if name.startswith('OOS'):
            for seg, (a0, a1) in {'val': ('2019-05-01', '2023-12-31'), 'hold': ('2024-01-01', '2026-10-30')}.items():
                exs = []; ts = []
                for lag in range(10):
                    ss, bb = navs[lag]
                    z = summ(ss[a0:a1], bb[a0:a1]); exs.append(z['ex']); ts.append(z['t'])
                out[seg] = dict(ex_median=round(float(np.median(exs)), 2), t_median=round(float(np.median(ts)), 2))
        if name.startswith('FULL'):
            out['yearly_medlag'] = yearly(s, b)
            for seg, (a0, a1) in {'2016_2018': ('2016-04-01', '2018-12-31'), '2019_2026': ('2019-01-01', '2026-10-30')}.items():
                exs = []; ts = []
                for lag in range(10):
                    ss, bb = navs[lag]
                    z = summ(ss[a0:a1], bb[a0:a1]); exs.append(z['ex']); ts.append(z['t'])
                out[seg] = dict(ex_median=round(float(np.median(exs)), 2), t_median=round(float(np.median(ts)), 2))
            # 拿掉最好 3 年
            ms = s.resample('ME').last().pct_change(); mb = b.resample('ME').last().pct_change()
            ex = (ms - mb).dropna()
            yr = out['yearly_medlag']; best = sorted(yr, key=lambda y: -yr[y])[:3]
            e2 = ex[~ex.index.year.isin(best)]
            out['drop_best3'] = dict(years=best, t=round(float(e2.mean() / e2.std(ddof=1) * np.sqrt(len(e2))), 2))
        res[name] = out
        print(name, json.dumps(out, default=float), flush=True)
    json.dump(res, open(HERE / f'r6_port_{which}.json', 'w'), indent=1, default=float)
