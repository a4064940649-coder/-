#!/usr/bin/env python3
"""反方審查（成本與可成交角度）o_tier_shrink_sequential。只讀研究員檔案，不改寫。
用法：python3 -I -B review_exec_o7.py
檢查：
 1) 拖累 drag＝(40 seed 可執行平均 − AB_V4|normal) 是否混入「seed 平均名單 vs 官方名單」的差（非執行成本）。
 2) 拖累中各成分（KAPPA 拆解欄）與衝擊項大小，估計 o5 審查指出的 OA 前視（衝擊低估約 32%）的額外成本。
 3) 對關鍵可執行格，把事後分布整體平移 δ（−0.03、−0.06、−0.10、−0.15 pp／月），用研究員自己的 Alloc 重算 x*、勝率。
    δ≈−0.06 對應 o5 審查 ALL 情境（CASH2＋FEEMIX＋ODD0，年化約 −0.6～−0.8%）。
 4) x* 的 E[log W] 曲線平坦度：相鄰格差多少。
"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import tlib as T
import s4_formal as F

def wr(s): return T.win_ret(s)

def main():
    out = {}
    r50 = wr(T.e50_nav())
    # ---------- 1) drag 汙染
    Z = pd.read_parquet(T.RAW / 'exec' / 'zero_impact_seed_navs.parquet')
    B = pd.read_parquet(T.RAW / 'exec' / 'bench.parquet')
    Zr = Z.pct_change().mean(axis=1).fillna(0)
    zavg = (1 + Zr).cumprod()
    d_seed = (wr(zavg) - wr(B['AB_V4|normal'])) * 100
    out['seedavg_minus_official_pp_per_window'] = {'mean': float(d_seed.mean()), 'sd': float(d_seed.std()), 'maxabs': float(d_seed.abs().max())}
    # ---------- 2) KAPPA 拆解
    KA = pd.read_parquet(T.RAW / 'navs' / 'navs_abavg.parquet')
    dec = {}
    for t in ('30萬', '100萬', '300萬', '1000萬'):
        for m in ('FIX', 'GROW'):
            row = {}
            for c in ('d1_integer', 'd2_oddtime', 'd3_oddspread', 'd4_fees', 'd5_impact', 'main', 'kappa_lo', 'kappa_hi', 'stress'):
                fee = 'B' if c == 'stress' else 'A'
                k = f'{c}|{t}|{m}|{fee}'
                if k in KA.columns:
                    row[c] = float(((wr(KA[k]) - wr(B['AB_V4|normal'])) * 100).mean())
            dec[f'{t}|{m}'] = row
    out['kappa_drag_components_pp'] = dec
    # ---------- 3) 平移後重算配置
    cj = json.loads((T.DATA / 'cells.json').read_text(encoding='utf-8'))
    cells = cj['cells']
    CS = pd.read_parquet(T.DATA / 'cell_series.parquet')
    PZ = np.load(T.DATA / 'post_cells.npz')
    th = PZ['theta']; step = float(th[1] - th[0])
    A = F.Alloc(r50.values, 20000)
    keys = ['EXEC|30萬|FIX|normal', 'EXEC|30萬|GROW|normal', 'EXEC|100萬|FIX|normal', 'EXEC|100萬|GROW|normal', 'EXEC|300萬|FIX|normal']
    res = {}
    for k in keys:
        for mo in ('KAPPA', 'TICK'):
            nm = f'{k}|{mo}'
            d0 = PZ[nm.replace('|', '__')]
            for dl in (0.0, -0.03, -0.06, -0.10, -0.15):
                d = T.shift_dens(d0, th, dl)
                s = T.dens_summary(d, th)
                a = A.run(d, th, step, CS[nm].values)
                g20 = a['20y']['grid']; g10 = a['10y']['grid']
                res[f'{nm}|{dl:+.2f}'] = {'mean': round(s['mean'], 3), 'P_pos': round(s['p_pos'], 3), 'xstar20': a['20y']['xstar'], 'xstar10': a['10y']['xstar'],
                                          'Elog20_x': {x: round(g20[x]['Elog_W_per_year'] * 100, 3) for x in (0, 50, 70, 90, 100)},
                                          'win10_x100': round(g10[100]['P_win'], 3), 'win10_xstar': round(g10[a['10y']['xstar']]['P_win'], 3),
                                          'Plt075_10y_x100': round(g10[100]['P_ratio_lt_0.75'], 3)}
                print(nm, dl, res[f'{nm}|{dl:+.2f}'], flush=True)
    out['shifted_alloc'] = res
    # ---------- 4) 研究員原結果平坦度
    R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))['RESULTS']
    flat = {}
    for k in keys:
        for mo in ('KAPPA', 'TICK'):
            g = R['allocation'][f'{k}|{mo}']['20y']['grid']
            v = {int(x): g[x]['Elog_W_per_year'] * 100 for x in g}
            best = max(v.values())
            flat[f'{k}|{mo}'] = {'within_0.10pct_of_best': [x for x in sorted(v) if best - v[x] <= 0.10],
                                 'within_0.25pct_of_best': [x for x in sorted(v) if best - v[x] <= 0.25]}
    out['elog_flatness_20y'] = flat
    print(json.dumps({k: v for k, v in out.items() if k != 'shifted_alloc'}, ensure_ascii=False, indent=1))
    (Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad') / 'review_exec_o7_out.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')

if __name__ == '__main__' and '--rounding' not in sys.argv:
    main()

# ---------------------------------------------------------------- 5) 取整「紅利」來源（--rounding）
def rounding_probe(seeds=(0, 1, 2, 3)):
    """d1_integer（只開取整）相對零衝擊為正（30萬 FIX 約 +0.10pp／窗）。檢查是否來自『以前收價 PC 決定股數、以開盤價 O 成交』
    使權重 ∝ O/PC（加碼開高股）——把 D.PC 換成 D.O 只影響 d1 的股數決定（d1 關閉零股價差與參與上限）。"""
    import ecore_tick as E
    D = E.Data()
    MODEL = json.loads((E.DATA / 'model_spec.json').read_text(encoding='utf-8'))
    fl = dict(integer=True, odd_time=False, odd_spread=False, fees=False, impact='flat', cap=False)
    B = pd.read_parquet(T.RAW / 'exec' / 'bench.parquet')
    Z = pd.read_parquet(T.RAW / 'exec' / 'zero_impact_seed_navs.parquet')
    PC0 = D.PC.copy()
    out = {}
    for tier in ('30萬', '100萬', '1000萬'):
        for usePC in (True, False):
            D.PC = PC0 if usePC else np.where(np.isfinite(D.O), D.O, PC0)
            diffs = []
            for s in seeds:
                nav, _ = E.simulate(D, s, tier=E.TIERS[tier], version='FIX', fee='A', model=E.ExecModel(MODEL), flags=fl)
                z = Z[f's{s:02d}'].reindex(nav.index)
                diffs.append(float(((wr(nav) - wr(z)) * 100).mean()))
            out[f'{tier}|FIX|size_by_{"PC" if usePC else "OPEN"}'] = {'mean_pp_per_window': float(np.mean(diffs)), 'by_seed': diffs}
            print(tier, usePC, np.round(diffs, 3), flush=True)
    D.PC = PC0
    return out

if __name__ == '__main__' and '--rounding' in sys.argv:
    print(json.dumps(rounding_probe(), ensure_ascii=False, indent=1))
