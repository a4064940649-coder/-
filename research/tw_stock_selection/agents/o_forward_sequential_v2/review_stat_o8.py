#!/usr/bin/env python3
"""反方審查（資料與統計，o8）：只讀研究員檔案，不寫 results.json。用法：python3 -I -B review_stat_o8.py"""
import sys, math, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import fslib as F, seq2 as Q
from s5_oc_final import decision_line

res = F.load_res(); oc = res['OC_FINAL']; par = dict(res['PSEUDO_FORWARD']['params'])
sig = oc['sigma_design']
line = decision_line(); post = line[line.index >= F.VAL[0]]
print('post n', len(post), 'mean', round(post.mean(), 3), 'sd', round(post.std(ddof=1), 3), 'hac', round(F.hac_sd(post), 3))
print('top 5 abs months', post.abs().sort_values(ascending=False).head(5).round(2).to_dict())
print('kurt', round(post.kurt(), 2), 'skew', round(post.skew(), 2))

def r2_fu(resid, a=0.0, P=40_000, seed=1, block=6, sig_rule=sig, pr=par):
    rng = np.random.default_rng(seed)
    X = Q.gen('D2', a, sig_rule, P, 36, rng, resid=resid, block=block)
    up, dn = Q.raw_triggers('R2', X, sig_rule, pr)
    u, d = Q.first_events(up, dn)
    return float(np.mean(u > 0)), float(np.mean(d > 0))

r = (post - post.mean()).to_numpy()
out = {}
out['D2_base'] = r2_fu(r)
# 去掉 2021-05
p2 = post.drop(pd.Period('2021-05', 'M')); out['D2_drop_2021_05'] = r2_fu((p2 - p2.mean()).to_numpy())
# 去掉最大 3 個 |x|
p3 = post.drop(post.abs().sort_values().index[-3:]); out['D2_drop_top3'] = r2_fu((p3 - p3.mean()).to_numpy())
# 縮尾 ±2.5 sd
w = post.clip(post.mean() - 2.5 * post.std(), post.mean() + 2.5 * post.std()); out['D2_winsor2.5sd'] = r2_fu((w - w.mean()).to_numpy())
# 子期間
for nm, a_, b_ in (('2019-05~2023-12', F.VAL[0], F.VAL[1]), ('2024-01~', F.HOLD[0], None)):
    s = post[(post.index >= a_) & ((post.index <= b_) if b_ is not None else True)]
    out[f'D2_{nm}'] = r2_fu((s - s.mean()).to_numpy())
# 殘差縮放到 σ_design（只看形狀）
rs = r / r.std(ddof=1) * sig; out['D2_rescaled_to_sigma'] = r2_fu(rs)
for k, v in out.items(): print(k, 'false_up36', round(v[0], 4), 'false_down36', round(v[1], 4))

# 外層 bootstrap：D2 的參考殘差本身只是一個 89 月樣本；重抽 6 月區塊產生 200 組替代殘差，看 false-up 分布
rng = np.random.default_rng(99); n = len(r); fus = []
for b in range(200):
    st = rng.integers(0, n, -(-n // 6)); idx = ((st[:, None] + np.arange(6)[None, :]) % n).ravel()[:n]
    rb = r[idx]; rb = rb - rb.mean()
    fus.append(r2_fu(rb, P=8_000, seed=b)[0])
fus = np.array(fus)
print('outer bootstrap D2 false-up: median', round(np.median(fus), 4), 'q10/q90', np.round(np.quantile(fus, [0.1, 0.9]), 4), 'P(>0.05)', round(np.mean(fus > 0.05), 3))

# 下修的判別力：α＝−0.5、0、+0.5 的 36 月下修率（D1/D2/D3）
d = pd.read_csv(F.DATA / 'oc_final.csv')
t = d[(d.rule == 'R2') & (d.h == 36)].pivot_table(index='alpha', columns='dgp', values='down')
print('R2 down36 by alpha\n', t.round(3))
print('LR down(α=-0.5)/down(α=0):', (t.loc[-0.5] / t.loc[0.0]).round(2).to_dict())

# 偽前瞻：依起點當時 σ 重新校準 c_d／c_u 後是否仍觸發
pf = res['PSEUDO_FORWARD']
for nm in ('VAL_2019-05', 'HOLD_2024-01'):
    s0 = pf[nm]['sigma']
    c = Q.calibrate(s0, 36, 0.04, 0.55, 60, P=100_000)
    z = pd.Series(pf[nm]['z_path'])
    looks = [l for l in Q.looks6(len(z))]
    trig = [(l, z.index[l - 1], round(z.iloc[l - 1], 3)) for l in looks if z.iloc[l - 1] <= -c['c_d']]
    print(nm, 'sigma_at_start', round(s0, 3), 'c_d recal', round(c['c_d'], 3), 'vs used', round(par['c_d'], 3), 'down looks', trig[:3])
    # z_path 是用 s0 算的；不同 σ 只影響門檻（z 已用起點 σ 標準化）

# P30 對 0050 的 t 與主策略 t 是否為同一統計量（VAL 無觸發：P30 超額＝0.3×主策略超額）
v = pf['VAL_2019-05']
print('VAL t P30', round(v['policies']['P30']['normal']['t_vs_0050'], 4), 'main', round(v['comparators']['MAIN_JMED_EXEC']['t_vs_0050'], 4))
h = pf['HOLD_2024-01']
print('HOLD P100 ann', round(h['policies']['P100']['normal']['ann'], 2), 'MAIN_JMED_EXEC ann', round(h['comparators']['MAIN_JMED_EXEC']['ann'], 2))
# 2019 後主策略相對 0050：去掉 2021 與去掉 2021-05 後的均值與 HAC t
for nm, s in (('all', post), ('ex2021', post[[p.year != 2021 for p in post.index]]), ('ex2021-05', post.drop(pd.Period('2021-05', 'M')))):
    print(nm, 'n', len(s), 'mean', round(s.mean(), 3), 'HAC t', round(s.mean() / (F.hac_sd(s) / math.sqrt(len(s))), 2))
