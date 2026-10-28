#!/usr/bin/env python3
"""版本評估：給定訊號基礎表 W、主策略腿淨值、0050，產生輪動淨值、同平均權重靜態混合、統計。挑選期與驗證期共用（凍結後不得改）。"""
import numpy as np, pandas as pd
import cr_lib as L


def version_states(W, spec):
    S = L.signal(W, spec['sig'], spec['L'])
    st = L.states(S, spec['h'])
    w = st.map(lambda s: None if s is None else (spec['xC'] if s == 'C' else 0.0))
    return S, st, w


def build(W, spec, mainnav, e50, sim_start, cost='normal', wbar=None, per=None):
    """回傳 dict：rot 淨值、static 淨值（wbar 給定時用 wbar，否則用 per 期間內目標權重的平均）、狀態、權重。"""
    S, st, w = version_states(W, spec)
    dec = [d for d in W.index if d >= pd.Timestamp(sim_start) and w.get(d) is not None and np.isfinite(w.get(d, np.nan))]
    if wbar is None:
        ww = L.sel(w.dropna().astype(float), per)
        wbar = float(ww.mean())
    exm = L.exec_map(dec, [w[d] for d in dec], mainnav.index)
    rot, lg = L.rotate(mainnav, e50, exm, cost)
    exs = L.exec_map(dec, [wbar] * len(dec), mainnav.index)
    sta, lgs = L.rotate(mainnav, e50, exs, cost)
    return dict(S=S, st=st, w=w, wbar=wbar, rot=rot, sta=sta, log=lg, logs=lgs, dec=dec)


def evaluate(B, W, mainnav, e50, per, sd_end, ew_nav=None, label=''):
    """per：視窗起點所在的期間；sd_end：最後一個視窗的終點日（評價終點）。"""
    sd = [d for d in W.index] + [sd_end]
    sd = sorted(set(pd.Timestamp(x) for x in sd))
    r_rot = L.sel(L.win_ret(B['rot'], sd), per); r_sta = L.sel(L.win_ret(B['sta'], sd), per)
    ix = r_rot.index
    r_main = L.win_ret(mainnav, sd).reindex(ix)
    e = L.on_cal(e50, mainnav.index); r50 = L.win_ret(e, sd).reindex(ix)
    a, b = ix[0], min(pd.Timestamp(sd_end), B['rot'].index[-1])
    cg = {k: L.cagr_mdd(n, a, b) for k, n in (('rot', B['rot']), ('static', B['sta']), ('main', mainnav), ('e50', e))}
    x_sta = r_rot - r_sta; x50 = r_rot - r50; xm = r_rot - r_main
    st = B['st'].reindex(ix)
    lbl = st.to_numpy()
    # 轉換次數：期間內決策日狀態改變的次數（含期初相對期前一個決策日）
    full = B['st'].dropna()
    pos = [full.index.get_loc(d) for d in ix if d in full.index]
    nsw = int(sum(1 for p in pos if p > 0 and full.iloc[p] != full.iloc[p - 1]))
    yrs = (b - a).days / 365.25
    # whipsaw 拆解：c_k＝(w_k − w̄)(r50_k − rM_k)；以狀態區段加總，再扣該區段起點的移轉成本（輪動與靜態成本差）
    wk = B['w'].reindex(ix).astype(float)
    c = (wk - B['wbar']) * (r50 - r_main)
    lg = B['log']
    cost_at = {}
    if len(lg):
        for _, r in lg.iterrows(): cost_at[r['date']] = r['cost']
    ex_dates = {d: B['rot'].index[B['rot'].index.searchsorted(d, side='right')] if B['rot'].index.searchsorted(d, side='right') < len(B['rot']) else None for d in ix}
    eps = []
    for s_, e_, l_ in L.runs(list(lbl)):
        dd = ix[s_:e_ + 1]
        g = float(c.iloc[s_:e_ + 1].sum())
        cst = float(sum(cost_at.get(ex_dates[d], 0.0) for d in dd))
        eps.append(dict(start=str(dd[0].date()), months=int(e_ + 1 - s_), state=l_, gain_pp=g * 100, switch_cost_pp=cst * 100, net_pp=(g - cst) * 100))
    E = pd.DataFrame(eps)
    long_ = E[E.months >= 6]; short = E[E.months < 6]
    gain_pos = float(E.net_pp.clip(lower=0).sum())
    whip_loss = float(-short.net_pp.clip(upper=0).sum())
    out = dict(
        label=label, n=int(len(ix)), span=[str(a.date()), str(b.date())], wbar=B['wbar'], frac_C=float((st == 'C').mean()),
        switches=nsw, switches_per_year=nsw / yrs,
        CAGR_rot=cg['rot'][0], MDD_rot=cg['rot'][1], CAGR_static=cg['static'][0], MDD_static=cg['static'][1],
        CAGR_main=cg['main'][0], MDD_main=cg['main'][1], CAGR_0050=cg['e50'][0], MDD_0050=cg['e50'][1],
        vs_static=dict(mean_pp=float(x_sta.mean() * 100), sd_pp=float(x_sta.std() * 100), t=L.tstat(x_sta), t_nw3=L.nw_t(x_sta.to_numpy(), 3),
                       IR_ann=float(x_sta.mean() / x_sta.std() * np.sqrt(12)) if x_sta.std() > 0 else float('nan'),
                       boot_state=L.state_block_boot(x_sta.to_numpy(), lbl), boot_mbb6=L.mbb_boot(x_sta.to_numpy())),
        vs_0050=dict(mean_pp=float(x50.mean() * 100), t=L.tstat(x50), t_nw3=L.nw_t(x50.to_numpy(), 3)),
        vs_main=dict(mean_pp=float(xm.mean() * 100), t=L.tstat(xm), CAGR_diff_pp=(cg['rot'][0] - cg['main'][0]) * 100),
        static_vs_0050=dict(mean_pp=float((r_sta - r50).mean() * 100), t=L.tstat(r_sta - r50)),
        TE_vs_main_ann=L.te_ann(r_rot, r_main), TE_main_vs_0050_ann=L.te_ann(r_main, r50), TE_rot_vs_0050_ann=L.te_ann(r_rot, r50),
        whipsaw=dict(episodes=eps, gain_long_ge6_pp=float(long_.net_pp.sum()) if len(long_) else 0.0, net_short_lt6_pp=float(short.net_pp.sum()) if len(short) else 0.0,
                     whipsaw_loss_pp=whip_loss, positive_gain_pp=gain_pos, ratio_whip_to_gain=(whip_loss / gain_pos) if gain_pos > 0 else float('inf'),
                     approx_total_pp=float(c.sum() * 100), switch_cost_total_pp=float(sum(cost_at.values()) * 100)),
        n_cost_events=int(len(lg)),
    )
    out['_series'] = dict(r_rot=r_rot, r_sta=r_sta, r50=r50, r_main=r_main, st=st)
    return out


def strip(o):
    return {k: v for k, v in o.items() if not k.startswith('_')}
