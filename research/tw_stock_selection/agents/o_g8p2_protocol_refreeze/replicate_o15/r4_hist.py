# 歷史線：自 V7（navs_exec7）與 V7X（R4 正式設定）淨值重建 J 窗超額，重算 Δ_w、較差線比例、滾動 sd、in-sample 示範
import json, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlib as L

AG = os.path.join(L.HERE, '..', '..')
V7 = pd.read_parquet(os.path.join(AG, 'o_forward_sequential_v2/raw/v7/navs_exec7.parquet'))
V7X = pd.read_parquet(os.path.join(AG, 'o_exec_engine_rewrite_v7x/out/navs_oos_R4.parquet'))
E = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj']


import bisect
PL = pd.read_csv(os.path.join(AG, 'o_forward_sequential_v2/raw/v7/plans_j7.csv'))
PL = PL[(PL.strat == 'AB_V7_B10') & (PL.seed == 0)]
JE = json.load(open(os.path.join(AG, 'o_forward_sequential_v2/raw/v7/jscan_end_days.json')))


def jmatrix(nav, col_fmt):
    """DataFrame(錨定月 × j) 的 J 窗超額 pp。窗＝名單訊號日 J_j 收盤 → 同 j 下一個訊號日收盤
    （最後一窗到 jscan_end_days）；錨定月＝不晚於訊號日的最後一個 J0 所在月；超額＝策略 − 0050。
    （此讀法逐月重現協定 resid_post2019，最大差 0）"""
    e = E.reindex(nav.index).ffill()
    j0 = sorted(pd.to_datetime(PL[PL.j == 0].signal))
    rows = {}
    for j in range(13):
        s = nav[col_fmt.format(j=j)]
        sig = sorted(pd.to_datetime(PL[PL.j == j].signal))
        for k, a in enumerate(sig):
            m = str(j0[bisect.bisect_right(j0, a) - 1].to_period('M'))
            b = sig[k + 1] if k + 1 < len(sig) else pd.Timestamp(JE[f'V7_B10|J{j}'])
            rows.setdefault(m, {})[j] = 100 * ((s.loc[b] / s.loc[a] - 1) - (e.loc[b] / e.loc[a] - 1))
    return pd.DataFrame(rows).T.sort_index()


def lines(nav, tag):
    N = jmatrix(nav, 'B10|J{j}|K5T|normal|100萬|' + tag)
    S = jmatrix(nav, 'B10|J{j}|K5T|stress|100萬|' + tag)
    W = np.minimum(N, S)
    return N, S, W


res = {}
for name, nav, tag in (('V7_FIX100', V7, 'FIX'), ('V7X_GROW100', V7X, 'GROW'), ('V7X_FIX100', V7X, 'FIX'), ('V7_GROW100', V7, 'GROW')):
    N, S, W = lines(nav, tag)
    sel = [m for m in N.index if '2006-04' <= m <= '2019-04']
    Ns, Ss, Ws = N.loc[sel], S.loc[sel], W.loc[sel]
    frac = float((Ss < Ns).values.mean())
    jmedN = N.median(axis=1); jmedW = W.median(axis=1)
    shift_jmed = float((jmedW.loc[sel] - jmedN.loc[sel]).mean())
    shift_win = float((Ws - Ns).values.mean())
    res[name] = dict(n_sel_months=len(sel), frac_stress_worse_sel=frac, shift_jmed_sel=shift_jmed, shift_window_sel=shift_win,
                     months=[N.index[0], N.index[-1], len(N)], nj_last=int(N.iloc[-1].notna().sum()))
    res[name]['_N'] = N; res[name]['_W'] = W
    print(name, {k: v for k, v in res[name].items() if not k.startswith('_')})

# V7 FIX100 一般成本 2019-05 起 JMED 去均值 vs 協定 resid_post2019
N7 = res['V7_FIX100']['_N']
j7 = N7.median(axis=1)
post = j7.loc[[m for m in j7.index if '2019-05' <= m <= '2026-09']]
dm = (post - post.mean()).values
print('resid check n', len(dm), 'max abs diff vs protocol', float(np.abs(dm - L.RESID).max()) if len(dm) == 89 else None,
      'mean', post.mean())

dw = min(abs(res['V7X_GROW100']['shift_jmed_sel']), abs(res['V7_FIX100']['shift_jmed_sel']))
dw_w = min(abs(res['V7X_GROW100']['shift_window_sel']), abs(res['V7_FIX100']['shift_window_sel']))
print('Delta_w jmed-shift floor3', np.floor(dw * 1000) / 1000, 'window-shift floor3', np.floor(dw_w * 1000) / 1000)

# 較差成本線（V7X GROW100）2019 起滾動 sd
WX = res['V7X_GROW100']['_W'].median(axis=1)
NX = res['V7X_GROW100']['_N'].median(axis=1)
w19 = WX.loc[[m for m in WX.index if m >= '2019-05']]
r24 = w19.rolling(24).std(ddof=1).dropna(); r12 = w19.rolling(12).std(ddof=1).dropna()
w19b = WX.loc[[m for m in WX.index if m >= '2019-01']]
r24b = w19b.rolling(24).std(ddof=1).dropna(); r12b = w19b.rolling(12).std(ddof=1).dropna()
print('roll24 (2019-05 start)', r24.min(), r24.max(), r24.quantile(0.75), 'roll12 max', r12.max())
print('roll24 (2019-01 start)', r24b.min(), r24b.max(), r24b.quantile(0.75), 'roll12 max', r12b.max())
# 用全序列滾動、只取窗終點 ≥ 2019 的
r24c = WX.rolling(24).std(ddof=1); r12c = WX.rolling(12).std(ddof=1)
k = [m for m in WX.index if m >= '2019-01']
print('roll24 end>=2019-01', r24c.loc[k].min(), r24c.loc[k].max(), r24c.loc[k].quantile(0.75), 'roll12', r12c.loc[k].max())


# in-sample 示範
def v1_path(x):
    x = np.asarray(x)
    S = np.cumsum(x)
    def z(n):
        sd = x[:n].std(ddof=1)
        return S[n - 1] / (max(L.C * L.SIG_D, sd) * np.sqrt(n))
    if not S[11] > 0: return 'FAILED'
    z12 = z(12)
    if z12 <= -L.CD_P: return 'FAILED'
    if z12 >= L.B_P * np.sqrt(2): return 'S1'
    if z(18) <= -L.CD_P: return 'FAILED'
    z24 = z(24)
    if z24 <= -L.CD_P: return 'FAILED'
    if z24 >= L.B_P: return 'S1'
    return 'NO_REJ'


def g8p1_path(x):
    X = np.asarray(x)[None, :]
    g = L.eval_g8p1(X)
    return 'S1' if g['up'][0] else ('FAILED' if g['fail'][0] else 'NO_REJ')


demo = {}
for nm, ser, fn in (('V1_worse', WX, v1_path), ('G8P1_normal', NX, g8p1_path), ('G8P1_worse', WX, g8p1_path)):
    ser = ser.loc[[m for m in ser.index if '2006-04' <= m <= '2026-09']]
    v = ser.values
    starts = list(ser.index[:len(v) - 23])
    st = [fn(v[i:i + 24]) for i in range(len(starts))]
    lit = [bool(np.sum(v[i:i + 12]) > 0) for i in range(len(starts))]
    demo[nm] = dict(n_months=len(v), first=ser.index[0], last=ser.index[-1], mean=float(v.mean()), sd=float(v.std(ddof=1)),
                    n=len(st), S1=st.count('S1') / len(st), n_S1=st.count('S1'), FAILED=st.count('FAILED') / len(st),
                    literal=float(np.mean(lit)), last_start=starts[-1], last_status=st[-1],
                    s1_starts=[starts[i] for i, s in enumerate(st) if s == 'S1'])
    print(nm, demo[nm])

out = {k: {kk: vv for kk, vv in v.items() if not kk.startswith('_')} for k, v in res.items()}
out['Delta_w_floor3'] = float(np.floor(dw * 1000) / 1000)
out['roll'] = dict(r24_min=float(r24.min()), r24_max=float(r24.max()), r24_q75=float(r24.quantile(0.75)), r12_max=float(r12.max()))
out['demo'] = demo
json.dump(out, open(os.path.join(L.HERE, 'r4_hist.json'), 'w'), ensure_ascii=False, indent=1, default=str)
