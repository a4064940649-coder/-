#!/usr/bin/env python3
"""統計與判定（PREREG 後）：讀 out/run_*.npz，算 P1～P5、判定，寫 results.json['main']。
python3 -I -B s4_stats.py"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
PRE = json.loads((HERE / 'data' / 'PREREG.json').read_text(encoding='utf-8'))
OUT = HERE / 'out'
JK = [f'J{j}' for j in range(13)]

def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return float('nan')
    s = x.std(ddof=1); return float(x.mean() / (s / math.sqrt(len(x)))) if s > 0 else float('nan')

def load(tag):
    f = OUT / f'{tag}.npz'
    if not f.exists(): return None
    z = np.load(f, allow_pickle=False)
    D = {}
    for key in z.files:
        k, n = key.split('|'); D.setdefault(k, {})[n] = z[key]
    return D

def per_set(d):
    ra, Rr, r5 = d['ra'], d['Rr'], d['r5']
    x = ra - Rr.mean(axis=0)
    rnd_ex = (Rr - r5).mean(axis=1); act_ex = (ra - r5).mean()
    return dict(n=int(len(x)), mean_pp=float(x.mean() * 100), t=tstat(x), perm_p=float(np.mean(rnd_ex >= act_ex)),
                vs0050_mean_pp=float((ra - r5).mean() * 100), vs0050_t=tstat(ra - r5),
                rand_vs0050_mean_pp=float((Rr.mean(axis=0) - r5).mean() * 100), rand_vs0050_t=tstat(Rr.mean(axis=0) - r5))

def summarize(D):
    S = {k: per_set(v) for k, v in D.items()}
    ts = np.array([S[k]['t'] for k in JK]); mp = np.array([S[k]['mean_pp'] for k in JK])
    v5 = np.array([S[k]['vs0050_t'] for k in JK]); v5m = np.array([S[k]['vs0050_mean_pp'] for k in JK])
    return dict(t_jmed=float(np.median(ts)), t_p10=float(np.percentile(ts, 10)), t_min=float(ts.min()), t_max=float(ts.max()),
                mean_pp_jmed=float(np.median(mp)), mean_pp_p10=float(np.percentile(mp, 10)),
                perm_p_jmed=float(np.median([S[k]['perm_p'] for k in JK])),
                t_off=S['OFF']['t'], mean_pp_off=S['OFF']['mean_pp'], perm_p_off=S['OFF']['perm_p'],
                vs0050_t_jmed=float(np.median(v5)), vs0050_t_p10=float(np.percentile(v5, 10)), vs0050_mean_pp_jmed=float(np.median(v5m)),
                vs0050_t_off=S['OFF']['vs0050_t'], rand_vs0050_mean_pp_jmed=float(np.median([S[k]['rand_vs0050_mean_pp'] for k in JK])),
                by_set=S)

def attrib(d, codes, tiers=((1, 100), (101, 300))):
    """靜態歸因：每窗 Σ_i (w_實際 − w̄_隨機)_i ×(F_i − 1)；依股票、依訊號日市值名次層。"""
    A = (d['Wa'].astype(float) - d['Wr'].astype(float)) * (np.nan_to_num(d['F'].astype(float), nan=1.0) - 1)
    by_stock = A.sum(axis=0)
    tier = {}
    rk = d['rk']
    for a, b in tiers:
        m = (rk >= a) & (rk <= b)
        tier[f'{a}-{b}'] = (A * m).sum(axis=1)
    m = (rk > 300) | (rk == 0)
    tier['>300/無名次'] = (A * m).sum(axis=1)
    return A.sum(axis=1), by_stock, tier

def main():
    R = json.loads((HERE / 'results.json').read_text(encoding='utf-8')) if (HERE / 'results.json').exists() else {}
    M = {}
    runs = {}
    for cfg in ['o7', 'i', 'ii', 'a', 'b']:
        for per in ['MAIN', 'PRE']:
            D = load(f'run_{cfg}_{per}')
            if D is None: continue
            runs[(cfg, per)] = D
            s = summarize(D); M[f'{cfg}|{per}'] = s
            print(cfg, per, 'jmed t', round(s['t_jmed'], 2), 'p10', round(s['t_p10'], 2), 'off t', round(s['t_off'], 2), 'mean jmed', round(s['mean_pp_jmed'], 3),
                  'vs0050 jmed t', round(s['vs0050_t_jmed'], 2), 'rand vs0050', round(s['rand_vs0050_mean_pp_jmed'], 3), flush=True)
    out = {'summary': {k: {kk: vv for kk, vv in v.items() if kk != 'by_set'} for k, v in M.items()}, 'by_set': {k: v['by_set'] for k, v in M.items()}}
    # ---- P1 判定 ----
    ta, tb = M['a|MAIN']['t_jmed'], M['b|MAIN']['t_jmed']
    low = 'a' if ta <= tb else 'b'
    tlow = min(ta, tb)
    out['P1'] = dict(t_jmed_a=ta, t_jmed_b=tb, lower=low, t_lower=tlow, threshold=2.0,
                     verdict='刪除「大型股選股力」機制，0050 骨架線結案' if tlow < 2.0 else '≥ 2.0：只列待觀察，以前瞻為準',
                     researcher_prediction='< 2.0', prediction_correct=bool(tlow < 2.0))
    # ---- P2 拆解 ----
    steps = [('o7 原輸入', 'o7'), ('(i) PIT 股數', 'i'), ('(ii) ＋4763／2317 修正', 'ii'), ('(iii) ＋納入下市股＝版本 (a)', 'a')]
    P2 = []
    for nm, c in steps:
        s = M[f'{c}|MAIN']; P2.append(dict(step=nm, config=c, t_off=s['t_off'], t_jmed=s['t_jmed'], t_p10=s['t_p10'], mean_pp_off=s['mean_pp_off'], mean_pp_jmed=s['mean_pp_jmed']))
    s = M['b|MAIN']; P2.append(dict(step='版本 (b) 落後 63 日', config='b', t_off=s['t_off'], t_jmed=s['t_jmed'], t_p10=s['t_p10'], mean_pp_off=s['mean_pp_off'], mean_pp_jmed=s['mean_pp_jmed']))
    drops = {}
    prev = M['o7|MAIN']['t_off']
    for nm, c in steps[1:]:
        drops[nm] = M[f'{c}|MAIN']['t_off'] - prev; prev = M[f'{c}|MAIN']['t_off']
    drops['(iv) 官方日 → j 中位數（版本 a）'] = M['a|MAIN']['t_jmed'] - M['a|MAIN']['t_off']
    drops['(iv) 官方日 → j 中位數（o7 原輸入）'] = M['o7|MAIN']['t_jmed'] - M['o7|MAIN']['t_off']
    out['P2'] = dict(table=P2, step_changes_t_official=drops, gate=R.get('gate_reproduce'))
    # ---- P3／P4：決定版本的集中度與分層 ----
    import pcore_lite as PL
    codes = PL.COLS
    D = runs[(low, 'MAIN')]
    stock_tot = np.zeros(len(codes)); tier_rows = {}; contrib_year = {}; shipping = 0.0; tot_all = 0.0
    approx_corr = []
    for k in JK:
        d = D[k]
        tot, bys, tier = attrib(d, codes)
        x = d['ra'] - d['Rr'].mean(axis=0)
        approx_corr.append(float(np.corrcoef(tot, x)[0, 1]))
        stock_tot += bys / len(d['ra'])
        for tn, ser in tier.items(): tier_rows.setdefault(tn, []).append((float(ser.mean() * 100), tstat(ser)))
        yrs = pd.DatetimeIndex(d['sd']).year
        for y in sorted(set(yrs)): contrib_year.setdefault(int(y), []).append(float(x[yrs == y].sum() * 100))
        A = (d['Wa'].astype(float) - d['Wr'].astype(float)) * (np.nan_to_num(d['F'].astype(float), nan=1.0) - 1)
        shipmask = np.array([PL.IND.get(c) == '航運業' for c in codes])
        y21 = (yrs == 2021)
        shipping += A[y21][:, shipmask].sum() / len(d['ra']); tot_all += A.sum() / len(d['ra'])
    stock_tot /= len(JK); shipping /= len(JK); tot_all /= len(JK)
    order = np.argsort(-stock_tot)
    top = [(codes[j], float(stock_tot[j] * 100)) for j in order[:10]]
    bottom = [(codes[j], float(stock_tot[j] * 100)) for j in order[-5:]]
    named = {c: float(stock_tot[PL.CI[c]] * 100) for c in ['4763', '2308', '2603', '2609', '2615', '2330', '2317', '2454']}
    out['P4'] = {tn: dict(mean_pp_jmed=float(np.median([a for a, _ in v])), t_jmed=float(np.median([b for _, b in v])), t_p10=float(np.percentile([b for _, b in v], 10))) for tn, v in tier_rows.items()}
    out['P4']['static_vs_actual_corr_jmed'] = float(np.median(approx_corr))
    # 去掉最好 1 個月、最好 1 年
    t_drop_m, t_drop_y, best_year = [], [], []
    for k in JK:
        d = D[k]; x = d['ra'] - d['Rr'].mean(axis=0)
        i = int(np.argmax(x)); t_drop_m.append(tstat(np.delete(x, i)))
        yrs = pd.DatetimeIndex(d['sd']).year; ys = sorted(set(yrs)); sums = [x[yrs == y].sum() for y in ys]
        yb = ys[int(np.argmax(sums))]; best_year.append(int(yb)); t_drop_y.append(tstat(x[yrs != yb]))
    yr_med = {y: float(np.median(v)) for y, v in contrib_year.items()}
    tot_med = sum(yr_med.values())
    out['P3'] = dict(version=low, top_contributors_pp_per_month=top, bottom_contributors=bottom, named_pp_per_month=named,
                     static_total_pp_per_month=float(tot_all * 100), shipping_2021_pp_per_month=float(shipping * 100),
                     shipping_2021_share=float(shipping / tot_all) if tot_all else None,
                     year_contrib_cum_pp_jmed=yr_med, year_2025_share=float(yr_med.get(2025, 0) / tot_med) if tot_med else None,
                     year_2021_share=float(yr_med.get(2021, 0) / tot_med) if tot_med else None,
                     t_drop_best_month_jmed=float(np.median(t_drop_m)), t_drop_best_year_jmed=float(np.median(t_drop_y)),
                     best_year_mode=int(pd.Series(best_year).mode()[0]),
                     share_4763=float(named['4763'] / (tot_all * 100)) if tot_all else None, share_2308=float(named['2308'] / (tot_all * 100)) if tot_all else None)
    ex = load(f'run_{low}_MAIN_ex{top[0][0]}')
    if ex is not None:
        s = summarize(ex); out['P3']['drop_top_stock'] = dict(code=top[0][0], t_jmed=s['t_jmed'], t_p10=s['t_p10'], t_off=s['t_off'], mean_pp_jmed=s['mean_pp_jmed'])
    # ---- P5：2014-01～2019-04 ----
    out['P5'] = {}
    for c in ['o7', 'i', 'ii', 'a', 'b']:
        if f'{c}|PRE' in M:
            s = M[f'{c}|PRE']; out['P5'][c] = dict(t_jmed=s['t_jmed'], t_p10=s['t_p10'], t_off=s['t_off'], mean_pp_jmed=s['mean_pp_jmed'],
                                                  same_sign_as_MAIN=bool(np.sign(s['mean_pp_jmed']) == np.sign(M[f'{c}|MAIN']['mean_pp_jmed'])))
    pa = out['P5'].get(low)
    out['P5']['verdict'] = None if pa is None else ('方向一致' if pa['same_sign_as_MAIN'] else '方向相反：機制不穩')
    # 合併 2014-01～2026-10（決定版本，j 中位數）
    comb = []
    if (low, 'PRE') in runs:
        for k in JK:
            d1, d2 = runs[(low, 'PRE')][k], runs[(low, 'MAIN')][k]
            x = np.concatenate([d1['ra'] - d1['Rr'].mean(axis=0), d2['ra'] - d2['Rr'].mean(axis=0)]); comb.append(tstat(x))
        out['P5']['combined_2014_2026_t_jmed'] = float(np.median(comb))
    # 逐年（決定版本、j 中位數）
    yr = {}
    for k in JK:
        d = D[k]; x = d['ra'] - d['Rr'].mean(axis=0); yrs = pd.DatetimeIndex(d['sd']).year
        for y in sorted(set(yrs)): yr.setdefault(int(y), []).append(float(x[yrs == y].mean() * 100))
    out['yearly_mean_pp_jmed'] = {y: float(np.median(v)) for y, v in yr.items()}
    R['main'] = out
    (HERE / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ['P1', 'P3', 'P4', 'P5']}, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(out['P2'], ensure_ascii=False, indent=1, default=float))

if __name__ == '__main__':
    main()
