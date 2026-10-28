#!/usr/bin/env python3
"""步驟 2：模型與迭代（只用 5 分 K 的校準期 CAL 擬合；模型選擇只看驗證期 VAL 的預測誤差；不看任何策略報酬）。
事前登記（本檔開頭即規格，round 2、3 的內容在 round 1 結果後補寫並記錄於 iterations.json）：
  目標 y：每個股日 09:05～09:55 共 11 根 5 分 K 的 |r_b| 平均（bar 層擬合、股日層評估）。
  開盤量：OA_b＝(官方日量 − Yahoo 全日量 − 收盤集合競價估計) − ρ_y × q_09:05（只用 53 根 K 棒齊全的日子）；
          收盤集合競價估計＝官方日量 × 該十分位在「09:00 K 棒有量日」的 (官方 − Yahoo)/官方 中位數（explore.json）。
          事前的 ex-ante 預測 OA_hat＝sm·V60sh·exp(β·[1, ln名次, ln價, lnσ, pick, drop])，β 用 CAL 正值 OA_b 的 OLS，
          sm 讓 CAL 上 Σ OA_hat＝Σ max(OA_b,0)（均值不偏）。
  候選：M1 κσ√(q/OA_hat)；M2 Yσ√(q/V60sh)；M3 a＋bσ(q/OA_hat)^δ（δ 格點 0.20～1.00、a、b 最小平方）；
        M4＝選定模型＋擁擠溢價項（只在換股日、綜合分數前 10% 的買單生效，c＝CAL 換股日 TOP10 的開盤溢價差）。
  估計：M1、M2 用比值估計（Σy/Σx，總量不偏）；M3 用最小平方。
  選擇：VAL 股日層中位數絕對誤差（MdAE）最小、且 VAL 各十分位 Σŷ/Σy 都在 [0.67, 1.5] 的模型；MdAE 相差 < 2% 時取參數少者（M2＜M1＜M3）。
        M4 只有在 VAL 擁擠檢定通過（換股日 TOP10 溢價差 95% 區間不含 0，且 BOT50 安慰劑不顯著）時才取代。
  κ 不確定性：CAL 以交易日為群的 bootstrap 1,000 次（seed 20261028），報 p5／p10／p50／p90／p95。
  （round 3 後補記：M3 類模型改為 (a,b,δ) 聯合 bootstrap，δ 每次在格點上重估；分位以參考點 Q/V＝1e-3 的隱含衝擊排序，
   p10／p90 欄用該分位那一次重抽的 (δ,b)。只影響區間，不影響點估計與選版。）
  閘門：CAL 股日 ≥ 2,000，每十分位 ≥ 150；VAL 各十分位比值在 [0.67, 1.5]。
用法：python3 -I -B iterate.py [round]
"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import calib_lib as C

EXP = json.load(open(HERE / 'explore.json'))
RHO = EXP['yahoo_open_volume']['rho_y_q0905_over_q0910']
CA_DEC = {int(k): v for k, v in EXP['yahoo_open_volume']['ca_y_share_by_dec'].items()}
BAD = set(EXP['step0_inventory']['5m_vs_official_volume_ratio_by_day']['anomalous_days_lt075_or_gt1'])
OA_VARS = ['const', 'lnrank', 'lnprice', 'lnsig', 'pick', 'drop']

def load():
    X = pd.read_parquet(C.DATA / 'stockdays.parquet'); R = pd.read_parquet(C.DATA / 'bars.parquet')
    assert (X.code.values == R.code.values).all() and (X.date.values == R.date.values).all()
    U = C.universe(X) & ~X.date.dt.strftime('%Y-%m-%d').isin(BAD)
    X['U'] = U
    X['ca_hat'] = X.dec.map(CA_DEC) * X.vol_off
    X['OA_b'] = np.where(X.nbar == 53, X.miss - X.ca_hat - RHO * X['q09:05'], np.nan)
    return X, R

def oa_design(d):
    return np.column_stack([np.ones(len(d)), np.log(d.vrank_l1.clip(lower=1)), np.log(d.prevc), np.log(d.sig60_l1.clip(lower=1e-4)),
                            d.pick.astype(float), d['drop'].astype(float)])

def fit_oa(X):
    c = X[X.U & (X.period == 'CAL') & np.isfinite(X.OA_b)]
    pos = c[c.OA_b > 0]
    b, *_ = np.linalg.lstsq(oa_design(pos), np.log(pos.OA_b / pos.V60sh).to_numpy(), rcond=None)
    raw = np.exp(oa_design(c) @ b) * c.V60sh
    sm = float(c.OA_b.clip(lower=0).sum() / raw.sum())
    return dict(vars=OA_VARS, coef=list(map(float, b)), smear=sm, n_fit=int(len(pos)), n_all=int(len(c)), share_nonpos=float((c.OA_b <= 0).mean()))

def oa_hat(X, spec):
    return spec['smear'] * np.exp(oa_design(X) @ np.array(spec['coef'])) * X.V60sh

def long_bars(X, R, mask, bars=C.BARS):
    L = []
    for b in bars:
        L.append(pd.DataFrame({'i': np.where(mask)[0], 'y': R.loc[mask, b].abs().to_numpy(), 'q': X.loc[mask, 'q' + b].to_numpy(),
                               'sig': X.loc[mask, 'sig60_l1'].to_numpy(), 'V': X.loc[mask, 'V60sh'].to_numpy(), 'OA': X.loc[mask, 'OA_hat'].to_numpy(),
                               'dec': X.loc[mask, 'dec'].to_numpy(), 'date': X.loc[mask, 'date'].to_numpy()}))
    Lb = pd.concat(L, ignore_index=True)
    return Lb[np.isfinite(Lb.y) & (Lb.q > 0) & np.isfinite(Lb.OA) & (Lb.OA > 0)]

def xvar(Lb, model, delta=0.5):
    if model == 'M1': return Lb.sig * np.sqrt(Lb.q / Lb.OA)
    if model == 'M2': return Lb.sig * np.sqrt(Lb.q / Lb.V)
    if model == 'M3': return Lb.sig * (Lb.q / Lb.OA) ** delta
    if model == 'M3V': return Lb.sig * (Lb.q / Lb.V) ** delta
    raise ValueError(model)

def fit(Lb, model, est='ratio'):
    if model in ('M1', 'M2'):
        x = xvar(Lb, model)
        k = float(Lb.y.sum() / x.sum()) if est == 'ratio' else float((x * Lb.y).sum() / (x * x).sum())
        return dict(model=model, est=est, k=k)
    best = None
    for dl in np.round(np.arange(0.20, 1.001, 0.01), 2):
        x = xvar(Lb, model, dl)
        A = np.column_stack([np.ones(len(x)), x])
        cf, res, *_ = np.linalg.lstsq(A, Lb.y.to_numpy(), rcond=None)
        sse = float(((A @ cf - Lb.y.to_numpy()) ** 2).sum())
        if best is None or sse < best[0]: best = (sse, float(dl), float(cf[0]), float(cf[1]))
    return dict(model=model, est='nls_grid', delta=best[1], a=best[2], b=best[3])

def predict(Lb, p, intercept=True):
    if p['model'] in ('M1', 'M2'): return p['k'] * xvar(Lb, p['model'])
    return (p['a'] if intercept else 0.0) + p['b'] * xvar(Lb, p['model'], p['delta'])

def evaluate(Lb, p):
    Lb = Lb.assign(yh=predict(Lb, p).to_numpy())
    sd = Lb.groupby('i').agg(y=('y', 'mean'), yh=('yh', 'mean'), dec=('dec', 'first'))
    dr = sd.groupby('dec').apply(lambda g: g.yh.sum() / g.y.sum())
    return dict(MdAE=float(np.median(np.abs(sd.yh - sd.y))), MdAE_rel=float(np.median(np.abs(sd.yh - sd.y) / sd.y.clip(lower=1e-5))),
                agg_ratio=float(sd.yh.sum() / sd.y.sum()), dec_ratio={int(k): float(v) for k, v in dr.items()},
                dec_ratio_in_band=bool(((dr >= 0.67) & (dr <= 1.5)).all()), n_stockdays=int(len(sd)))

def boot(Lb, p, B=1000, seed=20261028):
    days = np.array(sorted(Lb.date.unique())); rng = np.random.default_rng(seed)
    g = {d: idx for d, idx in Lb.groupby('date').indices.items()}
    out = []
    if p['model'] in ('M1', 'M2'):
        x = xvar(Lb, p['model']).to_numpy(); y = Lb.y.to_numpy()
        sy = {d: (y[i].sum(), x[i].sum(), (x[i] * y[i]).sum(), (x[i] ** 2).sum()) for d, i in g.items()}
        for _ in range(B):
            dd = rng.choice(days, len(days), replace=True)
            s = np.array([sy[d] for d in dd]).sum(axis=0)
            out.append(s[0] / s[1] if p['est'] == 'ratio' else s[2] / s[3])
        key = 'k'
    else:
        # 聯合 bootstrap：每次重抽交易日後，在 δ 格點上重新求 (a, b, δ) 的最小平方；以參考點 Q/V＝1e-3 的隱含衝擊 b·σ̄·(1e-3)^δ 排序取分位
        grid = np.round(np.arange(0.20, 1.001, 0.01), 2); y = Lb.y.to_numpy()
        dl = list(g.keys()); di = {d: k for k, d in enumerate(dl)}
        S = np.zeros((len(grid), len(dl), 6))
        for gi, de in enumerate(grid):
            x = xvar(Lb, p['model'], de).to_numpy()
            for d, i in g.items():
                S[gi, di[d]] = (len(i), x[i].sum(), y[i].sum(), (x[i] ** 2).sum(), (x[i] * y[i]).sum(), (y[i] ** 2).sum())
        reps = []
        for _ in range(B):
            idx = np.array([di[d] for d in rng.choice(days, len(days), replace=True)])
            T = S[:, idx, :].sum(axis=1)
            n, sx, sy_, sxx, sxy, syy = T.T
            bb = (sxy - sx * sy_ / n) / (sxx - sx * sx / n); aa = (sy_ - bb * sx) / n
            sse = syy - 2 * aa * sy_ - 2 * bb * sxy + n * aa * aa + 2 * aa * bb * sx + bb * bb * sxx
            k = int(np.argmin(sse)); reps.append((float(grid[k]), float(aa[k]), float(bb[k]), float(bb[k] * 1e-3 ** grid[k])))
        out = [r[3] for r in reps]
        order = np.argsort(out)
        pick = {f'p{q}': dict(zip(('delta', 'a', 'b', 'impact_ref_per_sigma'), reps[order[min(int(round(q / 100 * (B - 1))), B - 1)]])) for q in (5, 10, 50, 90, 95)}
        dq = np.percentile([r[0] for r in reps], [5, 50, 95]); bq = np.percentile([r[2] for r in reps], [5, 50, 95])
        q = np.percentile(out, [5, 10, 50, 90, 95])
        return dict(param='b·(1e-3)^δ（隱含衝擊÷σ，參考點 Q/V60＝1e-3）', p5=float(q[0]), p10=float(q[1]), p50=float(q[2]), p90=float(q[3]), p95=float(q[4]), B=B,
                    replicate_at_quantile=pick, delta_p5_p50_p95=list(map(float, dq)), b_p5_p50_p95=list(map(float, bq)),
                    point_impact_ref_per_sigma=float(p['b'] * 1e-3 ** p['delta']))
    q = np.percentile(out, [5, 10, 50, 90, 95])
    return dict(param=key, p5=float(q[0]), p10=float(q[1]), p50=float(q[2]), p90=float(q[3]), p95=float(q[4]), B=B)

# ---- round 2（round 1 結果後寫入；選擇規則不變）----
# round 1：VAL MdAE M3 0.000789 < M2 0.000847 < M1 0.001041，三者十分位比值都在帶內 → 暫選 M3。
# round 2 檢查 M3 的兩個設計選擇：分母（OA_hat 對 V60）與估計窗（只用 09:05～09:25 五根，較接近開盤），
# 以及 M2 改用最小平方估計（原點迴歸）。評估目標與規則不變（11 根平均、VAL MdAE）。
ROUND2 = [('v4', 'M3V', None), ('v5', 'M2', 'ls0'), ('v6', 'M3', 'early')]
# ---- round 3：M4＝round 1～2 選定模型＋擁擠項（c＝CAL 換股日 TOP10 溢價差）；只在 VAL 擁擠檢定通過時取代 ----
ROUND3 = [('v7', 'M4', None)]

def crowd_test(X, per, B=2000, seed=7):
    """換股日（成交日）高分股（TOP10：最近訊號綜合分數百分位 > 0.9）的開盤溢價 − 同股 ±60 日曆日內非換股日（排除換股日 ±2 日）平均；
    安慰劑 BOT50（≤ 0.5）。按換股日分群：每日平均後對日做 t 與 bootstrap 95% 區間。"""
    pick, new, drop, days = C.exec_flags()
    lo_, hi_ = (C.CAL if per == 'CAL' else C.VAL if per == 'VAL' else (C.CAL[0], C.VAL[1]))
    ex = [d for d in days if lo_ <= d <= hi_]
    A = X[X.U & (X.date >= lo_) & (X.date <= hi_)].copy()
    near = np.zeros(len(A), bool)
    for e in ex: near |= ((A.date >= e - pd.Timedelta(days=2)) & (A.date <= e + pd.Timedelta(days=2))).to_numpy()
    base = A[~near]
    out = {}; rng = np.random.default_rng(seed)
    for grp, lo, hi in (('TOP10', 0.9, 1.01), ('BOT50', -1, 0.5)):
        per_day = []
        for e in ex:
            g = A[(A.date == e) & (A.pct > lo) & (A.pct <= hi)]
            if len(g) == 0: continue
            bb = base[base.code.isin(g.code) & (base.date >= e - pd.Timedelta(days=60)) & (base.date <= e + pd.Timedelta(days=60))].groupby('code').prem.mean()
            dl = (g.set_index('code').prem - bb).dropna()
            per_day.append(float(dl.mean()))
        m = np.array(per_day)
        bs = [rng.choice(m, len(m)).mean() for _ in range(B)]
        out[grp] = dict(n_days=len(m), mean_pp=float(m.mean() * 100), t_day=float(m.mean() / (m.std(ddof=1) / np.sqrt(len(m)))),
                        ci95_pp=[float(np.percentile(bs, 2.5) * 100), float(np.percentile(bs, 97.5) * 100)], per_day_pp=[x * 100 for x in m])
    d = np.array(out['TOP10']['per_day_pp']) - np.array(out['BOT50']['per_day_pp'])
    out['TOP10_minus_BOT50'] = dict(mean_pp=float(d.mean()), t_day=float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))))
    out['MDE80_pp_TOP10'] = float(2.8 * np.std(out['TOP10']['per_day_pp'], ddof=1) / np.sqrt(len(d)))
    ci = out['TOP10']['ci95_pp']
    out['pass'] = bool((ci[0] > 0 or ci[1] < 0) and abs(out['BOT50']['t_day']) < 1.96)
    return out

if __name__ == '__main__':
    rnd = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    T0 = time.time()
    X, R = load()
    oa = fit_oa(X)
    X['OA_hat'] = oa_hat(X, oa)
    cal = (X.U & (X.period == 'CAL')).to_numpy(); val = (X.U & (X.period == 'VAL')).to_numpy()
    Lc, Lv = long_bars(X, R, cal), long_bars(X, R, val)
    IT = json.load(open(HERE / 'iterations.json')) if (HERE / 'iterations.json').exists() and rnd > 1 else {'spec': __doc__, 'rounds': []}
    IT['rounds'] = [r for r in IT['rounds'] if r['round'] < rnd]
    gate = dict(n_CAL_stockdays=int(cal.sum()), n_CAL_by_dec={int(k): int(v) for k, v in X[cal].groupby('dec').size().items()})
    gate['pass_sample'] = gate['n_CAL_stockdays'] >= 2000 and min(gate['n_CAL_by_dec'].values()) >= 150
    oa_chk = dict(OA_hat_share_of_day_CAL_median=float((X.OA_hat / X.vol_off)[cal].median()), OA_b_share_CAL_median=float((X.OA_b / X.vol_off)[cal & np.isfinite(X.OA_b).to_numpy()].median()),
                  OA_hat_share_by_dec_VAL={int(k): float(v) for k, v in (X.OA_hat / X.vol_off)[val].groupby(X.dec[val]).median().items()},
                  OA_b_share_by_dec_VAL={int(k): float(v) for k, v in (X.OA_b / X.vol_off)[val].groupby(X.dec[val]).median().items()},
                  OA_hat_over_OA_b_VAL_sum_ratio_by_dec={int(k): float(g.OA_hat.sum() / g.OA_b.clip(lower=0).sum()) for k, g in X[val & np.isfinite(X.OA_b).to_numpy()].groupby('dec')})
    if rnd == 1:
        vers = [('v1', 'M1', 'ratio'), ('v2', 'M2', 'ratio'), ('v3', 'M3', None)]
    elif rnd == 2:
        vers = ROUND2
    else:
        vers = ROUND3
    res = []
    if rnd == 3:
        prev = [v for r in IT['rounds'] for v in r['versions']]
        ok = [v for v in prev if v['VAL']['dec_ratio_in_band']]
        order = {'M2': 0, 'M1': 1, 'M3V': 2, 'M3': 3}
        mn = min(v['VAL']['MdAE'] for v in ok)
        cand = sorted([v for v in ok if v['VAL']['MdAE'] <= mn * 1.02], key=lambda v: (order[v['params']['model']], v['VAL']['MdAE']))
        sel = cand[0]
        ct_cal = crowd_test(X, 'CAL'); ct_val = crowd_test(X, 'VAL'); ct_all = crowd_test(X, 'ALL')
        p = {('base_' + k if k == 'model' else k): v for k, v in sel['params'].items()}; p['crowd_c'] = ct_cal['TOP10']['mean_pp'] / 100; p['base_version'] = sel['version']
        res.append(dict(version='v7', params=dict(model='M4', **p), crowd_CAL=ct_cal, crowd_VAL=ct_val, crowd_ALL_descriptive=ct_all,
                        replaces_base=bool(ct_val['pass']), note='M4 與基底模型在 11 根 bar 目標上相同；只在 VAL 擁擠檢定通過時取代'))
        sel_final = res[-1] if ct_val['pass'] else sel
        IT['selected'] = dict(version=sel_final['version'], params=sel_final['params'], rule='VAL MdAE 最小且十分位在帶內；2% 內取參數少者；M4 需擁擠檢定通過',
                              candidates_within_2pct=[v['version'] for v in cand])
        Lsel = Lc if sel['version'] != 'v6' else long_bars(X, R, cal, C.BARS[:5])
        IT['selected']['bootstrap_CAL'] = boot(Lsel, sel['params'])
        Lv_fit = fit(Lv if sel['version'] != 'v6' else long_bars(X, R, val, C.BARS[:5]), sel['params']['model'], sel['params'].get('est') if sel['params']['model'] in ('M1', 'M2') else None)
        IT['selected']['refit_on_VAL_stability'] = Lv_fit
        vers = []
    for name, model, est in vers:
        if est == 'early':
            p = fit(long_bars(X, R, cal, C.BARS[:5]), model); p['window'] = '09:05-09:25'
        else:
            p = fit(Lc, model, est or 'ratio')
        res.append(dict(version=name, params=p, CAL=evaluate(Lc, p), VAL=evaluate(Lv, p)))
        print(name, p, res[-1]['VAL']['MdAE'], res[-1]['VAL']['dec_ratio_in_band'], flush=True)
    IT['rounds'].append(dict(round=rnd, oa_model=oa, oa_check=oa_chk, gate_sample=gate, versions=res, runtime_s=time.time() - T0))
    json.dump(IT, open(HERE / 'iterations.json', 'w'), ensure_ascii=False, indent=1)
