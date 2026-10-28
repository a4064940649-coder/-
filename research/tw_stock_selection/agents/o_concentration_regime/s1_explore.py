#!/usr/bin/env python3
"""步驟 1：探索（只用挑選期 2019-01～2022-12 的視窗；不調參）。
2023 年以後的同一組描述統計放在 PREREG 之後的 s4_oos.py（描述延伸），這裡不碰。
輸出 explore.json 的 S1 區塊。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import cr_lib as L

W = pd.read_parquet(L.DATA / 'W_sel.parquet')
END = W['end'].iloc[-1]                     # 2023-01 決策日
WS = L.sel(W, L.SEL).copy()
assert WS.index.min() >= pd.Timestamp('2019-01-01') and WS['end'].max() <= END
out = {}

# ---- 主策略（AB_V4）、B10、同池等權 的挑選期視窗報酬 ----
nv = pd.read_parquet(L.RAW / 'navs_v4.parquet')
nv = nv[nv.index <= END]
sd = list(W.index) + [END]
for c in ['AB_V4|normal', 'AB_V4_B10|normal', 'AB_V4|EW']:
    WS[c] = L.win_ret(nv[c].dropna(), sd).reindex(WS.index)
WS['alpha_vs_EW'] = np.log1p(WS['AB_V4|normal']) - np.log1p(WS['rEW'])
WS['main_minus_0050'] = np.log1p(WS['AB_V4|normal']) - np.log1p(WS['r50'])

d = WS['d']
out['d_summary_sel'] = dict(n=int(len(d)), mean_pp=float(d.mean() * 100), sd_pp=float(d.std() * 100), t=L.tstat(d), frac_pos=float((d > 0).mean()),
                            cum_log=float(d.sum()), acf={f'lag{l}': float(d.autocorr(l)) for l in range(1, 7)})
# main − 0050 ＝ (main − EW) − d 的變異拆解
x = WS[['alpha_vs_EW', 'd', 'main_minus_0050']].dropna()
out['decomp_main_minus_0050_sel'] = dict(var_main_minus_0050=float(x.main_minus_0050.var()), var_d=float(x.d.var()), var_alpha=float(x.alpha_vs_EW.var()),
                                         cov_alpha_d=float(np.cov(x.alpha_vs_EW, x.d)[0, 1]), corr_alpha_d=float(x.corr().loc['alpha_vs_EW', 'd']),
                                         mean_alpha_pp=float(x.alpha_vs_EW.mean() * 100), t_alpha=L.tstat(x.alpha_vs_EW), mean_d_pp=float(x.d.mean() * 100))

# ---- P1 預測迴歸（挑選期）----
P1 = {}
for kind, Ls in (('S1', (3, 6, 12)), ('S1skip', (12,)), ('S2', (6,))):
    for Lk in Ls:
        S = L.signal(W, kind, Lk)
        y = W['d']; xx = S
        m = (W.index >= pd.Timestamp(L.SEL[0])) & (W.index <= pd.Timestamp(L.SEL[1]))
        r = L.ols_hac(y[m].to_numpy(), xx[m].to_numpy(), lag=max(Lk, 3))
        lbl = np.sign(xx[m].to_numpy())
        # 狀態區塊 bootstrap：以訊號符號的連續區段為區塊，對 (x,y) 配對重抽
        xy = np.column_stack([xx[m].to_numpy(), y[m].to_numpy()])
        rs = L.runs(lbl); blocks = [xy[a:b + 1] for a, b, _ in rs]; rng = np.random.default_rng(0); bs = []
        for _ in range(3000):
            acc = []; n_ = 0
            while n_ < len(xy):
                bl = blocks[rng.integers(len(blocks))]; acc.append(bl); n_ += len(bl)
            z = np.concatenate(acc)[:len(xy)]
            if np.std(z[:, 0]) > 0: bs.append(np.polyfit(z[:, 0], z[:, 1], 1)[0])
        r['boot_state_block'] = dict(p5=float(np.percentile(bs, 5)), p50=float(np.percentile(bs, 50)), p95=float(np.percentile(bs, 95)), p_le0=float(np.mean(np.array(bs) <= 0)), n_blocks=len(blocks))
        nxt = y[m]; sg = xx[m]
        r['mean_next_d_when_pos_pp'] = float(nxt[sg > 0].mean() * 100); r['mean_next_d_when_neg_pp'] = float(nxt[sg < 0].mean() * 100)
        r['n_pos'] = int((sg > 0).sum()); r['n_neg'] = int((sg < 0).sum())
        P1[f'{kind}_L{Lk}'] = r
out['P1_selection'] = P1

# ---- 狀態描述：符號分段（d 的 6 個月累積）與 2 狀態高斯 HMM（只描述）----
S6 = L.signal(W, 'S1', 6); s6 = L.sel(S6, L.SEL)
lab = np.where(s6 > 0, 'C', 'D')
rs = L.runs(lab)
out['sign_segments_S1_L6_sel'] = [dict(start=str(s6.index[a].date()), end=str(s6.index[b].date()), state=l, months=int(b - a + 1)) for a, b, l in rs]
rawlab = np.where(d > 0, 'C', 'D'); rr = L.runs(rawlab)
out['run_length_raw_sign_d_sel'] = dict(n_runs=len(rr), mean_len=float(np.mean([b - a + 1 for a, b, _ in rr])), expected_iid=float(1 / (1 - ((d > 0).mean() ** 2 + (d <= 0).mean() ** 2))))

def hmm2(y, it=200):
    y = np.asarray(y, float); n = len(y)
    mu = np.array([np.percentile(y, 75), np.percentile(y, 25)]); sg = np.array([y.std(), y.std()]); A = np.array([[0.8, 0.2], [0.2, 0.8]]); pi = np.array([0.5, 0.5])
    for _ in range(it):
        B = np.exp(-0.5 * ((y[:, None] - mu) / sg) ** 2) / sg
        al = np.zeros((n, 2)); c = np.zeros(n); al[0] = pi * B[0]; c[0] = al[0].sum(); al[0] /= c[0]
        for t in range(1, n): al[t] = (al[t - 1] @ A) * B[t]; c[t] = al[t].sum(); al[t] /= c[t]
        be = np.ones((n, 2))
        for t in range(n - 2, -1, -1): be[t] = (A @ (B[t + 1] * be[t + 1])) / c[t + 1]
        g = al * be; g /= g.sum(1, keepdims=True)
        xi = np.zeros((2, 2))
        for t in range(n - 1): xi += (al[t][:, None] * A * (B[t + 1] * be[t + 1])[None, :]) / c[t + 1]
        A = xi / xi.sum(1, keepdims=True); pi = g[0]
        mu = (g * y[:, None]).sum(0) / g.sum(0); sg = np.sqrt((g * (y[:, None] - mu) ** 2).sum(0) / g.sum(0)); sg = np.maximum(sg, 1e-4)
    return dict(mu_pp=(mu * 100).tolist(), sd_pp=(sg * 100).tolist(), A=A.tolist(), expected_duration_months=(1 / (1 - np.diag(A))).tolist(), loglik=float(np.log(c).sum())), g
h, g = hmm2(d.to_numpy())
out['hmm2_d_sel'] = h
out['hmm2_d_sel']['state_hi_prob_by_month'] = {str(k.date()): round(float(v), 3) for k, v in zip(d.index, g[:, int(np.argmax(h['mu_pp']))])}

# ---- 逐月表（挑選期）----
out['monthly_table_sel'] = [dict(start=str(i.date()), r0050=round(r.r50 * 100, 2), rEW=round(r.rEW * 100, 2), d=round(r.d * 100, 2), main=round(r['AB_V4|normal'] * 100, 2),
                                 top5_share=round(r.top5 * 100, 2), top1_share=round(r.top1 * 100, 2), S1_L6=None if not np.isfinite(S6.get(i, np.nan)) else round(float(S6[i]) * 100, 2))
                            for i, r in WS.iterrows()]
out['top5_share_sel'] = dict(first=float(WS.top5.iloc[0]), last=float(WS.top5.iloc[-1]), min=float(WS.top5.min()), max=float(WS.top5.max()),
                             corr_top5_change_with_d=float(pd.Series(WS.top5.diff().to_numpy()[1:]).corr(pd.Series(WS.d.shift(1).to_numpy()[1:]))))

# ---- P4（2021-10～2022-12 的部分）：外資淨買集中度 ----
P = L.Panel(end=END)
inst = pd.read_csv(L.RAW / 'inst_5y.csv.gz', dtype={'code': str}, usecols=['date', 'code', 'foreign'], parse_dates=['date'])
inst = inst[(inst.date <= END)]
FW = inst.pivot_table(index='date', columns='code', values='foreign', aggfunc='sum')
FW = FW.reindex(columns=P.C.columns)
CR = P.C.reindex(FW.index)
FV = FW * 1000 * CR       # 外資淨買超金額（NT$）
VV = P.V.reindex(FW.index)
rows = []
for k in range(len(W)):
    a, b = W.index[k], W['end'].iloc[k]
    if a < pd.Timestamp('2021-10-08') or a > pd.Timestamp(L.SEL[1]): continue
    idx, v = P.pool(a); top = idx[np.argsort(-v)[:5]]
    m = (FV.index > a) & (FV.index <= b)
    net = FV.loc[m].iloc[:, idx].sum(); netT = FV.loc[m].iloc[:, top].sum().sum()
    val = VV.loc[m].iloc[:, idx].sum(); valT = VV.loc[m].iloc[:, top].sum().sum()
    rows.append(dict(start=str(a.date()), FC_top5_net_over_abs=float(netT / net.abs().sum()), value_share_top5=float(valT / val.sum()), d=float(W['d'].iloc[k]),
                     state_S1L6=('C' if S6.iloc[k] > 0 else 'D') if np.isfinite(S6.iloc[k]) else None, top5=[P.C.columns[j] for j in top]))
F4 = pd.DataFrame(rows)
if len(F4):
    F4['excess'] = F4.FC_top5_net_over_abs - F4.value_share_top5
    out['P4_sel_2021_10_to_2022_12'] = dict(rows=F4.drop(columns=['top5']).round(4).to_dict('records'),
                                            mean_excess_when_C=float(F4[F4.state_S1L6 == 'C'].excess.mean()) if (F4.state_S1L6 == 'C').any() else None,
                                            mean_excess_when_D=float(F4[F4.state_S1L6 == 'D'].excess.mean()) if (F4.state_S1L6 == 'D').any() else None,
                                            corr_excess_with_same_window_d=float(F4.excess.corr(F4.d)), top5_sets=sorted(set(tuple(t) for t in F4.top5)))

# ---- 贏家輪替（ETF 月報酬，2020-08～2022-12，日曆月）----
tr = pd.read_csv(L.RAW / 'etf_tr_m.csv.gz', dtype={'code': str})
tr = tr[tr.ym <= '2022-12']
codes = ['0050', '0052', '006208', '00733', '0056', '00878', '00881', '00692', '00713']
T = tr[tr.code.isin(codes)].pivot_table(index='ym', columns='code', values='tr')   # tr＝月底含息價格水準（不是報酬），先轉月報酬並修補分割
H = T / T.shift(1); H = H.where(~(H < 0.5), H * (1 / H).round()); H = (H - 1) * 100; H = H.iloc[1:]
H.index = pd.PeriodIndex(H.index, freq='M')
half = (1 + H / 100).groupby([H.index.year, (H.index.month - 1) // 6]).prod() - 1
out['etf_winner_rotation_halfyear_sel'] = {f'{y}H{h + 1}': {c: round(float(v) * 100, 1) for c, v in row.dropna().sort_values(ascending=False).items()} for (y, h), row in half.iterrows()}
out['etf_note'] = '00733＝富邦臺灣中小（中小型 A 級動能 50），代表中型動能；0052＝富邦科技（權值集中的科技籃）。日曆月口徑。安聯台灣科技等境內基金沒有可下載淨值，不列入。'

ex = L.load_json(L.HERE / 'explore.json'); ex['S1'] = out; L.save_json(ex, L.HERE / 'explore.json')
print(pd.Series(out['d_summary_sel']).to_string())
print(out['decomp_main_minus_0050_sel'])
for k, v in P1.items(): print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
print(out['sign_segments_S1_L6_sel']); print(out['run_length_raw_sign_d_sel']); print({k: v for k, v in h.items()})
if len(F4): print({k: v for k, v in out['P4_sel_2021_10_to_2022_12'].items() if k != 'rows'})
for k, v in out['etf_winner_rotation_halfyear_sel'].items(): print(k, v)
