#!/usr/bin/env python3
"""o14 反方審查（成本與可成交角度）：只讀研究員檔案，不寫入任何研究員或前瞻資料夾。
1. 若真實成本＝壓力成本，但判定線用一般成本：淨 α＝0 時的「誤升級」機率（D1 iid 常態，判定器語意）。
2. 重播 trades.csv：零股 vs 整股成交值占比；Δ_open 在重播資料上的值與 share_of_open_vol_actual 的來源。
3. 重播 forward_returns：JMED 一般 vs 壓力、MAIN_OFFICIAL vs JMED。
"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / 'G8_PROTOCOL.json').read_text())
sd, c, cd, b = (P['params'][k] for k in ('sigma_design', 'c', 'c_d', 'b'))
se = c * sd


def sim(mu, Pn=400000, seed=7, sig=sd):
    rng = np.random.default_rng(seed)
    X = rng.normal(mu, sig, size=(Pn, 24))
    S = X.cumsum(1)
    z = lambda n: S[:, n - 1] / (se * math.sqrt(n))
    fail = S[:, 11] <= 0                     # 字面 G8
    fail |= z(12) <= -cd
    up = (~fail) & (z(12) >= b * math.sqrt(2))
    fail18 = (~fail) & (~up) & (z(18) <= -cd)
    fail |= fail18
    dn24 = (~fail) & (~up) & (z(24) <= -cd)
    up |= (~fail) & (~dn24) & (z(24) >= b)
    return dict(mu=mu, s1=float(up.mean()), fwd_fail=float((fail | dn24).mean()), g8_pass=float((S[:, 11] > 0).mean()))


out = {}
# 一般成本判定線、真實成本＝壓力（挑選期平移 0.577；2019 年起描述 0.78）
out['sim_D1'] = [sim(m) for m in (0.0, 0.3, 0.577, 0.78, 1.0)]

rp = HERE.parent / 'o_forward_infra_live/replay/seq'
tr = pd.read_csv(rp / 'trades.csv', dtype={'code': str})
pp = tr[tr.kind.isin(['paper_whole', 'paper_odd']) & tr.line.str.match(r'^(J\d+|OFFICIAL)$')].copy()
pp['val'] = pp.shares * pp.fill_px
g = pp.groupby('kind').val.agg(['count', 'sum'])
out['replay_fill_mix'] = dict(rows=g['count'].to_dict(), value=g['sum'].to_dict(),
                              odd_value_share=float(g.loc['paper_odd', 'sum'] / g['sum'].sum()))
pw = pp[pp.kind == 'paper_whole'].dropna(subset=['share_of_open_vol', 'share_of_open_vol_actual', 'sigma60_prev'])
pwd = P['cost_calibration']['kappa'] * pw.sigma60_prev * (np.sqrt(pw.share_of_open_vol_actual) - np.sqrt(pw.share_of_open_vol))
# share_of_open_vol_actual / share_of_open_vol 應＝V60/V（OASH 是固定十分位參數，不是實際開盤量）
ratio = (pw.share_of_open_vol / pw.share_of_open_vol_actual)
out['replay_delta_open'] = dict(n=int(len(pw)), p50=float(np.median(pwd)), p75=float(np.percentile(pwd, 75)),
                                oa_share_dec_unique=sorted(set(np.round(pw.oa_share_dec, 6)))[:10],
                                share_pred_over_actual_eq_V_over_V60=float(np.median(ratio)))
fr = pd.read_csv(rp / 'forward_returns.csv')
fc = fr[fr.status == 'complete']
out['replay_lines'] = fc[fc.line.isin(['JMED', 'MAIN_OFFICIAL'] + [f'J{j}' for j in range(13)])][['line', 'anchor', 'excess_pp', 'excess_stress_pp']].to_dict('records')
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
