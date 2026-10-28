#!/usr/bin/env python3
"""步驟 0：建立焦點序列（窗報酬）並檢查淨值跳日、缺月（o5 教訓 1）。不做任何收縮計算。
輸出：data/focus_windows.parquet（窗報酬，小數）、data/focus_meta.json（跳日清單、j 中位數平移、成本不確定）。
用法：python3 -I -B build_inputs.py
焦點欄（全部以 AB_V4 訊號日格點的窗報酬，與族群同格點）：
  r0050      0050 含息（bench.parquet，與 fam_common.e50_nav 相同）
  AB         AB_V4|normal（V4 凍結版，正式主欄）
  B10        AB_V4_B10|normal（SOP 保守口徑）
  EXEC       B10 ＋ (o_exec_capacity_tiers main|100萬|FIX|A − 零衝擊 AB_V4) 的逐窗拖累（o5 成本審查的移植公式）
  EXEC_J     EXEC ＋ j 中位數平移 δ_j（j＝0～12 的 AB_V5 對 0050 平均窗超額中位數 − 官方日 AB_V5 平均窗超額；o5 教訓 10）
  AB_stress、B10_stress、EXEC_stress（B10_stress＋stress|100萬|FIX|B 的拖累）
  B10_V5     o_v5_pit_revenue AB_V5_B10|normal（核對）
  EXEC_30F、EXEC_300F、EXEC_100G（其他級距，資訊用）
  EW         同池等權 AB_V4|EW（零成本，參考）
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sc_lib as S
FC = S.FC
AG = S.AG

def gaps(nav, name, thr=5):
    s = nav.dropna()
    d = s.index
    out = []
    for a, b in zip(d[:-1], d[1:]):
        nb = int(np.busday_count(a.date(), b.date())) - 1
        if nb > thr:
            out.append({'series': name, 'from': str(a.date()), 'to': str(b.date()), 'missing_weekdays': nb})
    return out

def main():
    V4 = pd.read_parquet(AG / 'o_v4_revenue_patch_official' / 'navs_v4.parquet')
    V5 = pd.read_parquet(AG / 'o_v5_pit_revenue' / 'navs_v5.parquet')
    JS = pd.read_parquet(AG / 'o_v5_pit_revenue' / 'navs_jscan_v5.parquet')
    EX = pd.read_parquet(AG / 'o_exec_capacity_tiers' / 'navs_abavg.parquet')
    EB = pd.read_parquet(AG / 'o_exec_capacity_tiers' / 'data' / 'bench.parquet')
    BE = pd.read_parquet(S.SPA / 'data' / 'bench.parquet')
    e50 = FC.e50_nav()
    w = FC.win_ret
    r50 = w(e50)
    assert np.allclose(r50.values, BE['r0050'].values, atol=1e-12), '0050 窗報酬與 SPA bench 不一致'
    F = pd.DataFrame(index=FC.WIN)
    F['r0050'] = r50
    F['AB'] = w(V4['AB_V4|normal']); F['B10'] = w(V4['AB_V4_B10|normal'])
    F['AB_stress'] = w(V4['AB_V4|stress']); F['B10_stress'] = w(V4['AB_V4_B10|stress'])
    F['B10_V5'] = w(V5['AB_V5_B10|normal'])
    F['EW'] = w(V4['AB_V4|EW'])
    zi = w(EB['AB_V4|normal'])
    assert float((zi - F['AB']).abs().max()) < 1e-12, '容量研究的零衝擊 AB_V4 與 navs_v4 不一致'
    drag = {k: w(EX[c]) - zi for k, c in [('100F', 'main|100萬|FIX|A'), ('30F', 'main|30萬|FIX|A'), ('300F', 'main|300萬|FIX|A'),
                                           ('100G', 'main|100萬|GROW|A'), ('100F_stressB', 'stress|100萬|FIX|B'),
                                           ('100F_d4', 'd4_fees|100萬|FIX|A'), ('100F_d5', 'd5_impact|100萬|FIX|A'),
                                           ('100G_d4', 'd4_fees|100萬|GROW|A'), ('100G_d5', 'd5_impact|100萬|GROW|A'),
                                           ('100G_klo', 'kappa_lo|100萬|GROW|A'), ('100G_khi', 'kappa_hi|100萬|GROW|A')]}
    F['EXEC'] = F['B10'] + drag['100F']
    F['EXEC_stress'] = F['B10_stress'] + drag['100F_stressB']
    F['EXEC_30F'] = F['B10'] + drag['30F']; F['EXEC_300F'] = F['B10'] + drag['300F']; F['EXEC_100G'] = F['B10'] + drag['100G']
    # j 中位數平移（AB_V5 normal、40 seed 平均；官方格點取樣）
    ab5 = w(V5['AB_V5|normal'])
    off = float((ab5 - r50).mean())
    jm = {c: float((w(JS[c]) - r50).mean()) for c in JS.columns}
    jt = {c: S.tstat((w(JS[c]) - r50).dropna()) for c in JS.columns}
    dj = float(np.median(list(jm.values())) - off)
    F['EXEC_J'] = F['EXEC'] + dj
    assert F.notna().all().all(), F.isna().sum()
    # 成本參數不確定（o5 衝擊區間）：衝擊成分 I＝d5−d4（100萬 FIX）；區間 [0.2I, 1.3I]（下界＝事後全日 V60 分母約 1/4～1/6；
    # 上界＝kappa_lo 對 GROW 衝擊的放大倍數），均勻分布 → sd＝1.1|I|/√12（pp／窗）
    I_fix = float((drag['100F_d5'] - drag['100F_d4']).mean() * 100)
    I_grow = float((drag['100G_d5'] - drag['100G_d4']).mean() * 100)
    klo = float((drag['100G_klo'] - drag['100G']).mean() * 100); khi = float((drag['100G_khi'] - drag['100G']).mean() * 100)
    up_mult = 1 + abs(klo) / abs(I_grow)
    sd_cost = (up_mult - 0.2) * abs(I_fix) / np.sqrt(12)
    gl = []
    for nm, s in [('0050', e50[e50.index >= '2006-04-11']), ('AB_V4', V4['AB_V4|normal']), ('AB_V4_B10', V4['AB_V4_B10|normal']),
                  ('EXEC_100F', EX['main|100萬|FIX|A']), ('AB_V5_B10', V5['AB_V5_B10|normal'])]:
        gl += gaps(s, nm)
    # 窗長（交易日）與跨缺口窗
    wl = []
    G = FC.GRID
    for a, b in zip(G[:-1], G[1:]):
        wl.append(int(np.busday_count(a.date(), b.date())))
    wl = pd.Series(wl, index=FC.WIN)
    long_w = [{'window_start': str(i.date()), 'weekdays': int(v)} for i, v in wl.items() if v > 27]
    meta = {'n_windows': int(len(F)), 'first': str(F.index[0].date()), 'last_start': str(F.index[-1].date()), 'end': str(FC.END_DAY.date()),
            'nav_gaps_gt5_weekdays': gl, 'long_windows_gt27_weekdays': long_w,
            'window_weekdays': {'median': float(wl.median()), 'min': int(wl.min()), 'max': int(wl.max())},
            'j_shift': {'AB_V5_official_mean_pp': off * 100, 'j_means_pp': {k: v * 100 for k, v in jm.items()}, 'j_t': jt,
                        'median_j_mean_pp': float(np.median(list(jm.values())) * 100), 'delta_j_pp': dj * 100},
            'cost_uncertainty': {'impact_100F_pp': I_fix, 'impact_100G_pp': I_grow, 'kappa_lo_extra_100G_pp': klo, 'kappa_hi_extra_100G_pp': khi,
                                 'upper_mult': up_mult, 'lower_mult': 0.2, 'sd_cost_pp': float(sd_cost)},
            'drag_mean_pp': {k: float(v.mean() * 100) for k, v in drag.items()},
            'mean_excess_pp': {c: float((F[c] - F['r0050']).mean() * 100) for c in F.columns if c != 'r0050'},
            't_excess': {c: S.tstat(F[c] - F['r0050']) for c in F.columns if c != 'r0050'},
            'check_B10_V4_vs_V5_mean_diff_pp': float(((F['B10'] - F['B10_V5'])).mean() * 100)}
    F.to_parquet(S.DATA / 'focus_windows.parquet')
    (S.DATA / 'focus_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in meta.items() if k not in ('j_shift',)}, ensure_ascii=False, indent=1)[:4000])
    print('delta_j_pp', dj * 100, 'jt', jt)

if __name__ == '__main__':
    main()
