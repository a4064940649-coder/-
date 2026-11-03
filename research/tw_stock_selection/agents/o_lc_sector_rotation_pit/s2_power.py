#!/usr/bin/env python3
"""事前通過機率（o11 修正版 S2：用本題自己的追蹤誤差、周轉、成本假設；不看任何報酬）。
假設：月超額（對 0050）iid 常態，年化 IR ∈ {0.3, 0.5, 0.66}，TE ∈ {3%, 5%, 7%}；成本拖累已含在 IR（本題周轉低，V7X 大型股載體）。
D1 未通過 → 可計入的歷史只有 2014-01～2026-10（154 個月）：G2 全期 2006～2026 與 G4 的 2006～2012 段不可評估 → 必然不過。
另報「若 D1 通過」（2006-04～2026-10，247 個月）的假想值，與本題「待觀察」條件（驗證期 2017-01～2026-10 t ≥ 2）的機率。
G3：o11 實算全族群 DSR 需要 IR≈2.0、F_main 口徑需要 0.78（直接沿用，不重算）。"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lclib as X

rng = np.random.default_rng(20261103)
B = 20000
def sim(ir, te, months, seg_bounds, y19_start, val_start, drop3=True):
    mu = ir * te / 12; sd = te / np.sqrt(12)
    x = rng.normal(mu, sd, (B, months))
    def t(a): return a.mean(1) / a.std(1, ddof=1) * np.sqrt(a.shape[1])
    tall = t(x); t19 = t(x[:, y19_start:]); tval = t(x[:, val_start:])
    segs = np.all([x[:, a:b].mean(1) > 0 for a, b in seg_bounds], axis=0)
    ny = months // 12
    yr = x[:, :ny * 12].reshape(B, ny, 12).sum(2)
    best = np.argsort(-yr, axis=1)[:, :3]
    keep = np.ones((B, ny), bool); np.put_along_axis(keep, best, False, axis=1)
    xm = x[:, :ny * 12].reshape(B, ny, 12)
    td3 = np.array([t(xm[b][keep[b]].reshape(1, -1))[0] for b in range(min(B, 4000))])
    p_drop = float((td3 >= 2).mean())
    return dict(P_G2=float(((tall >= 3) & (t19 >= 2)).mean()), P_G4_segs=float(segs.mean()), P_drop3_t2=p_drop,
                P_G2_and_G4_approx=float(((tall >= 3) & (t19 >= 2) & segs).mean()) * p_drop / max(float(((tall >= 3)).mean()), 1e-9) if (tall >= 3).mean() > 0 else 0.0,
                P_val_t2=float((tval >= 2).mean()), P_t19_2=float((t19 >= 2).mean()))
out = {'doc': __doc__, 'grid': {}}
for ir in (0.3, 0.5, 0.66):
    for te in (0.03, 0.05, 0.07):
        a = sim(ir, te, 154, [(0, 60), (60, 154)], 60, 36)      # 2014-01 起：段 2013～2018 只有 2014～2018，2019～2026
        b = sim(ir, te, 247, [(0, 81), (81, 153), (153, 247)], 153, 129)
        out['grid'][f'IR{ir}_TE{int(te*100)}'] = dict(D1_fail_2014on=dict(a, P_G1_G7=0.0, note='G2 全期與 G4 第一段不可評估 → 0'),
                                                     if_D1_pass=dict(b, P_G1_G7_with_DSR=0.0 if ir < 2.0 else None,
                                                                     note='G3 全族群 DSR 需 IR≈2.0（o11）→ 0；F_main DSR 需 0.78 → IR≤0.66 時≈0'))
out['S2_verdict'] = 'S2 < 20%（所有格 G1～G7 通過機率 0%）；使用者已指定本題，不停題，照實列入判定框。'
(X.DATA / 's2_power.json').write_text(X.dumps(out), encoding='utf-8')
for k, v in out['grid'].items(): print(k, {kk: round(vv, 3) for kk, vv in v['D1_fail_2014on'].items() if isinstance(vv, float)}, '| ifpass', {kk: round(vv, 3) for kk, vv in v['if_D1_pass'].items() if isinstance(vv, float)})
