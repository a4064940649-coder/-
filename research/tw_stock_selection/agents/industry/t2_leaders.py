#!/usr/bin/env python3
"""檢定 2：龍頭領先效應。

事前固定：
* 龍頭：2330 台積電、2454 聯發科、2317 鴻海；目標產業：半導體業、電子零組件業、電腦及週邊設備業、其他電子業（12 組配對全報告）。
* 目標組合：該產業合格股中流動性（前一日 60 日均成交值）前 20 檔，排除三家龍頭，等權。
* A. 價格版（月底換股，訊號用前一交易日收盤）：P1 龍頭近 1 個月報酬 > 0；P2 近 3 個月報酬 > 0；P3 近 1 個月報酬 > 0050 近 1 個月。
     挑選 2005-01～2019-04、驗證 2019-05～2026-10。
* B. 營收版（每月第一個 ≥11 日交易日）：V1 龍頭上月營收年增率 > 0；V2 年增率較 3 個月前上升。挑選 2021-03～2023-12、驗證 2024～2026-10。
* 交易：訊號強 → 持有目標組合；否則持有 0050（換手一律以 0.5% 來回計，保守）。
* 另報：迴歸 r_G(t+1) − r_0050(t+1) = a + b·訊號連續值(t) + c·r_G(t)（控制自身動能），以及強／弱月份的超額報酬差。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from ilib import load, eligibility, run_weights, stats, brief, universe_ew, period_returns, bench_returns, month_ends, prev_day, save, tstat
from t1_revenue_cycle import revenue_matrix

LEAD = ['2330', '2454', '2317']
TGT = ['半導體業', '電子零組件業', '電腦及週邊設備業', '其他電子業']


def ols_t(y, X):
    """回傳各係數與 t（Newey-West 不做，月資料、殘差近似獨立）。"""
    df = pd.concat([y, X], axis=1).dropna()
    if len(df) < 24: return None
    Y = df.iloc[:, 0].values; Z = np.column_stack([np.ones(len(df)), df.iloc[:, 1:].values])
    b, *_ = np.linalg.lstsq(Z, Y, rcond=None); e = Y - Z @ b
    s2 = e @ e / (len(Y) - Z.shape[1]); se = np.sqrt(np.diag(s2 * np.linalg.inv(Z.T @ Z)))
    return {'b_signal': round(float(b[1]), 4), 't_signal': round(float(b[1] / se[1]), 2), 'b_own': round(float(b[2]), 4) if len(b) > 2 else None,
            't_own': round(float(b[2] / se[2]), 2) if len(b) > 2 else None, 'n': int(len(Y))}


def setup(dates):
    d = load(); adj, IND = d['adj'], d['IND']
    R = period_returns(adj, dates); B = bench_returns(d['e50'], dates)
    el, v = eligibility(dates); U = universe_ew(R, el)
    G = {}
    for k in TGT:
        W = {}
        for t in R.index:
            mem = [c for c in el.columns[el.loc[t].values] if IND.get(c) == k and c not in LEAD]
            top = list(v.loc[t, mem].sort_values(ascending=False).index[:20])
            if len(top) >= 10: W[t] = {c: 1 / len(top) for c in top}
        G[k] = W
    return R, B, el, v, U, G


def strat(signal_bool, Wg, R, B):
    W, ex = {}, {}
    for t in R.index:
        if t not in Wg or t not in signal_bool.index or pd.isna(signal_bool[t]): continue
        W[t] = Wg[t] if signal_bool[t] else {'0050': 1.0}
        ex[t] = {'0050': B[t]}
    return run_weights(W, R, extra_ret=ex)


def group_ret(Wg, R):
    g, _ = run_weights(Wg, R); return g


def block(kind, dates, sigs, splits):
    R, B, el, v, U, G = setup(dates)
    out = {}
    for k in TGT:
        gr = group_ret(G[k], R)
        exg = gr - B.reindex(gr.index)
        always, _ = run_weights(G[k], R)
        out[f'ALWAYS_top20|{k}'] = {s: brief(stats(*run_weights(G[k], R), dates, B, U, *rng)) for s, rng in splits.items()}
        for ld in LEAD:
            for sn, (cont, boo) in sigs[ld].items():
                cont = cont.reindex(R.index); boo = boo.reindex(R.index)
                g, to = strat(boo, G[k], R, B)
                row = {s: brief(stats(g, to, dates, B, U, *rng)) for s, rng in splits.items()}
                for s, (a, b_) in splits.items():
                    m = (exg.index >= pd.Timestamp(a)) & (exg.index <= pd.Timestamp(b_))
                    e = exg[m]; sb = boo.reindex(e.index)
                    strong, weak = e[sb == True], e[sb == False]  # noqa: E712
                    row[s]['reg'] = ols_t(e, pd.concat([cont.reindex(e.index), gr.shift(1).reindex(e.index)], axis=1))
                    row[s]['ex_strong_m%'] = round(float(strong.mean() * 100), 3) if len(strong) else None
                    row[s]['ex_weak_m%'] = round(float(weak.mean() * 100), 3) if len(weak) else None
                    row[s]['n_strong'] = int(len(strong))
                    if len(strong) > 3 and len(weak) > 3:
                        diff = strong.mean() - weak.mean(); se = np.sqrt(strong.var() / len(strong) + weak.var() / len(weak))
                        row[s]['t_strong_minus_weak'] = round(float(diff / se), 2)
                out[f'{kind}|{ld}|{sn}|{k}'] = row
    return out


def main():
    d = load(); adj, e50 = d['adj'], d['e50']
    res = {}
    # A. 價格版
    me = [x for x in month_ends(adj.index) if x >= pd.Timestamp('2004-12-01')]
    pdays = prev_day(me, adj.index)
    A = adj.ffill(limit=5).reindex(pdays).set_axis(pd.DatetimeIndex(me))
    E = e50.reindex(adj.index, method='ffill').reindex(pdays).set_axis(pd.DatetimeIndex(me))
    sigs = {}
    for ld in LEAD:
        r1 = A[ld] / A[ld].shift(1) - 1; r3 = A[ld] / A[ld].shift(3) - 1; rel = r1 - (E / E.shift(1) - 1)
        sigs[ld] = {'P1_ret1m>0': (r1, (r1 > 0).where(r1.notna())), 'P2_ret3m>0': (r3, (r3 > 0).where(r3.notna())),
                    'P3_ret1m>0050': (rel, (rel > 0).where(rel.notna()))}
    res['price'] = block('price', me, sigs, {'selection': ('2005-01-01', '2019-04-30'), 'validation': ('2019-05-01', '2026-12-31')})
    # B. 營收版
    ii = adj.index[adj.index >= pd.Timestamp('2021-03-01')]; s = pd.Series(ii, index=ii)
    ds = [g[g.dt.day >= 11].iloc[0] for _, g in s.groupby([ii.year, ii.month]) if (g.dt.day >= 11).any()]
    REVF, PREV, REVOK = revenue_matrix()
    yoy = REVOK / PREV - 1
    sigs = {}
    for ld in LEAD:
        y = pd.Series({t: yoy.loc[pd.Period(t, 'M') - 1, ld] for t in ds})
        y3 = pd.Series({t: yoy.loc[pd.Period(t, 'M') - 4, ld] for t in ds})
        acc = y - y3
        sigs[ld] = {'V1_yoy>0': (y, (y > 0).where(y.notna())), 'V2_accel>0': (acc, (acc > 0).where(acc.notna()))}
    res['revenue'] = block('revenue', ds, sigs, {'selection': ('2021-03-01', '2023-12-31'), 'validation': ('2024-01-01', '2026-12-31')})
    # 挑選
    for kind in ['price', 'revenue']:
        cand = {k: v['selection']['t_0050'] for k, v in res[kind].items() if not k.startswith('ALWAYS') and 't_0050' in v['selection']}
        best = max(cand, key=cand.get)
        res[f'{kind}_selected'] = {'config': best, 'selection': res[kind][best]['selection'], 'validation': res[kind][best]['validation']}
        top5 = sorted(cand, key=cand.get, reverse=True)[:5]
        res[f'{kind}_top5_selection'] = {k: {'sel_t': cand[k], 'val': res[kind][k]['validation']} for k in top5}
    save('results_t2.json', res)
    for kind in ['price', 'revenue']:
        print('=====', kind)
        for k, v in res[kind].items():
            sv, vv = v['selection'], v['validation']
            print(f"{k:45s} SEL ex%={sv.get('ex_0050_m%')} t={sv.get('t_0050')} reg_t={(sv.get('reg') or {}).get('t_signal')} s-w t={sv.get('t_strong_minus_weak')} | "
                  f"VAL CAGR={vv.get('CAGR')} 0050={vv.get('0050_CAGR')} ex%={vv.get('ex_0050_m%')} t={vv.get('t_0050')} reg_t={(vv.get('reg') or {}).get('t_signal')} s-w t={vv.get('t_strong_minus_weak')}")
        print('SELECTED', res[f'{kind}_selected'])


if __name__ == '__main__':
    main()
