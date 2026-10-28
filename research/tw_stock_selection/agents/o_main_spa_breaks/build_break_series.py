#!/usr/bin/env python3
"""斷點序列的輸入資料（V4 凍結版；只讀 import o_v4_revenue_patch_official/v4lib.py，引擎與規格完全沿用）：
  (1) AB_V4 扣成本前（zero cost）淨值：與 run_v4.py 的 ab() 相同（mc seed 0～39 的日報酬平均）。
      同時重跑一般成本版，斷言與 navs_v4.parquet 的 AB_V4|normal 相同（引擎核對）。
  (2) 池內綜合分數五分位：每個訊號日，在 V4 的合格池（el）內依綜合分數 sc 分 5 組等權（Q5＝最高），
      隔日開盤成交、零成本與一般成本兩版；營收用 AB 最壞情境（期限日全部視為未公告 → 用前一個月），
      確保沒有前視。不做開盤漲停遞補（理論價差）。
輸出 data/break_inputs.parquet（窗報酬）與 data/break_inputs_daily.parquet（日頻淨值）。"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fam_common as C
sys.path.insert(0, str(C.V4DIR))
import v4lib as L            # noqa: E402（只讀 import）
sys.path.pop(0)
K, V, R = L.K, L.V, L.R
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)

def main():
    fz = L.check_freeze()
    XR = L.setup_px()
    YFX, Y5F = L.load_rfix()
    DMT = L.disp_mask_fn('trade')
    mk = lambda **kw: L.V4(YFX, Y5F, DMT, **kw)
    S0 = mk(); t0, t1, sd = S0.bounds()
    # ---- (1) AB_V4 零成本與一般成本 ----
    rz, rn, ew = [], [], []
    for seed in range(40):
        S = mk(unavail=V.make_unavail('mc', seed))
        pl = S.plan(10, 'official')
        nz, _ = L.sim_plan(pl, t0, t1, 'zero', XR); rz.append(nz.pct_change())
        nn, _ = L.sim_plan(pl, t0, t1, 'normal', XR); rn.append(nn.pct_change())
        if seed % 10 == 0: log('AB seed', seed)
    mean_nav = lambda L_: (1 + pd.concat(L_, axis=1).mean(axis=1).fillna(0)).cumprod()
    navz, navn = mean_nav(rz), mean_nav(rn)
    ref = pd.read_parquet(C.V4DIR / 'navs_v4.parquet')['AB_V4|normal']
    d = float(np.nanmax(np.abs(navn.reindex(ref.index).to_numpy() - ref.to_numpy())))
    log('AB_V4 normal 重跑 vs navs_v4 最大差', d)
    assert d < 1e-8, 'AB_V4 重跑不符'
    # ---- (2) 五分位 ----
    SW = mk(unavail=V.make_unavail('worst'))
    plans = {q: [] for q in range(1, 6)}
    nq = []
    for dd in SW.sig_days:
        cs = SW.cross_section(dd)
        el = cs['el']
        if el.sum() < 50: continue
        sc = cs['sc']; idx = np.where(el & np.isfinite(sc))[0]
        order = idx[np.argsort(sc[idx], kind='stable')]          # 由低到高
        parts = np.array_split(order, 5)
        for q in range(5): plans[q + 1].append((cs['e'], np.array(parts[q], int)))
        nq.append(len(order))
    log('quintile plans', len(plans[1]), 'mean pool', np.mean(nq))
    out_d = {'AB_V4|zero': navz, 'AB_V4|normal_rerun': navn}
    for q in range(1, 6):
        b, s = L.cost_rates('zero'); nav, _ = L.O3.simulate(plans[q], b, s, t0, t1, xrules=XR); out_d[f'Q{q}|zero'] = nav
        b, s = L.cost_rates('normal'); nav, _ = L.O3.simulate(plans[q], b, s, t0, t1, xrules=XR); out_d[f'Q{q}|normal'] = nav
        log('Q', q)
    D = pd.DataFrame(out_d)
    D.to_parquet(C.DATA / 'break_inputs_daily.parquet')
    W = pd.DataFrame({k: C.win_ret(v) for k, v in out_d.items()})
    W.to_parquet(C.DATA / 'break_inputs.parquet')
    log('done', W.shape, 'Q5-Q1 mean pp', float((W['Q5|zero'] - W['Q1|zero']).mean() * 100))

if __name__ == '__main__':
    main()
