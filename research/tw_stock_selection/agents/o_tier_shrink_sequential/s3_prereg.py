#!/usr/bin/env python3
"""步驟 3：凍結（PREREG）。依 PREREG_DESIGN 的選版規則機械選版，寫 iterations.json 與 results.json 的 PREREG（含 sha256、時間戳）。
之後 s4_formal.py 才能計算真實焦點的事後分布與配置表。
用法：python3 -I -B s3_prereg.py
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pandas as pd
import tlib as T, hmethods as H

def select(res):
    S = pd.DataFrame(res['STEP1_SYNTH']['summary'])
    P = S[(S.scen == 'P_primary') & (S.tag == 'sel')].set_index('ver')
    passed = [v for v in H.VERSIONS if bool(P.loc[v, 'gate_pass'])]
    cv = pd.DataFrame(res['STEP1_CV']['summary']); cv = cv[cv.tag == 'sel'].set_index('ver')['sum_logpd']
    if passed:
        v = max(passed, key=lambda x: cv[x]); how = '主情境閘門通過者中 CV 預測對數似然最高'
        status = '驗收通過'
    else:
        cand = [v for v in H.VERSIONS if 0.70 <= P.loc[v, 'coverage80'] <= 0.90]
        pool = cand if cand else list(H.VERSIONS)
        v = min(pool, key=lambda x: abs(P.loc[x, 'bias'])); how = '全部未過閘門：涵蓋率 ∈ [0.70,0.90] 中主情境 |偏誤| 最小'
        status = '驗收未過'
    return v, how, status, passed, P, cv

def main():
    res = T.check_block('PREREG_DESIGN', files=False); T.check_block('PREREG_DESIGN_AMEND1', files=False)
    for k in ('STEP1_O6_REPRO', 'STEP1_SYNTH', 'STEP1_CV'): assert k in res, k
    assert res['STEP1_O6_REPRO']['pass_le_0.01']
    assert 'PREREG' not in res and 'RESULTS' not in res
    v, how, status, passed, P, cv = select(res)
    # iterations.json
    S = pd.DataFrame(res['STEP1_SYNTH']['summary'])
    it = {'n_versions': len(H.VERSIONS), 'versions': {}, 'disclosures': [
        '冒煙：s2b_synth.py 主情境 r＝0 一次（合成，非真實焦點）；看到 σ_w 估計偏低，登記 PREREG_DESIGN_AMEND1 修正動差估計式後才跑全量合成。',
        'σ_w 新估計式在 8 次只含 σ_w 的模擬確認（0／0.10／0.25 → 0.03／0.095／0.25）。',
        '登記的合成主情境把焦點群真值限定在 g_true 右尾 10%：以真值為條件時，任何以資料為條件才校準的貝氏事後都會往下偏（同群隨機成員 V1／V2 偏誤約 −0.14）。'
        '這是閘門設計的缺陷；選版仍照登記規則機械執行，另跑 STEP1_SYNTH_POSTHOC（事後診斷）揭露。',
        'o6 重現（V1）：差 ≤ 1e-15，通過。',
        'CV 的 B 期含 2019～2026（已看過）；後期成員超額更分散，所有版本的預測涵蓋都偏低。'],
        'selection': {'selected': v, 'rule_applied': how, 'status': status, 'passed_primary_gate': passed}}
    for ver in H.VERSIONS:
        rows = S[S.ver == ver][['scen', 'tag', 'bias', 'bias_se', 'coverage80', 'rmse']].round(4).to_dict('records')
        it['versions'][ver] = {'definition': res['PREREG_DESIGN']['hierarchical_model']['versions_to_iterate'][ver], 'synthetic': rows,
                               'cv_sel_sum_logpd': float(cv[ver])}
    if 'STEP1_SYNTH_POSTHOC' in res:
        it['posthoc_synthetic'] = res['STEP1_SYNTH_POSTHOC']['summary']
    (HERE / 'iterations.json').write_text(json.dumps(it, ensure_ascii=False, indent=1, default=T._js), encoding='utf-8')
    spec = FROZEN_SPEC.format(ver=v, how=how, status=status)
    FWD = {'seqlib.py', 's5_seqsim.py', 'forward_monitor.py', 's6_freeze_forward.py', 's7_tier_table.py', 's8_fake_forward.py'}   # 前瞻部分由 PREREG_FORWARD 另行凍結
    files = sorted([p.name for p in HERE.glob('*.py') if p.name not in FWD]) + ['raw/manifest.json', 'iterations.json', 'explore.json',
            'data/cells.json', 'data/cell_series.parquet', 'data/lines_series.parquet', 'data/lines_meta.json', 'data/dJ_windows.parquet',
            'data/navs_tick_abavg.parquet', 'data/tick_exec_meta.json']
    files += ['raw/' + k for k in json.loads((HERE / 'raw' / 'manifest.json').read_text(encoding='utf-8'))['files']]
    PRE = {'selected_version': v, 'selection_rule_applied': how, 'validation_status': status, 'frozen_spec': spec,
           'design_sha256': res['PREREG_DESIGN_sha256'], 'amend1_sha256': res['PREREG_DESIGN_AMEND1_sha256'],
           'hashes': {f: T.sha(HERE / f) for f in files}, 'utc': T.now_utc()}
    T.save_key('PREREG', PRE); T.save_key('PREREG_sha256', T.canon_sha(PRE))
    print('PREREG', T.canon_sha(PRE), PRE['utc'], v, status, len(PRE['hashes']), 'files')

FROZEN_SPEC = """o_tier_shrink_sequential 凍結規格（PREREG）
一、資料：raw/（44 檔，manifest 含來源與 sha256）。族群 F_long 2,319 條（一般成本，缺值補 0），o6 的 161 群與 medoid；
    窗格點＝AB_V4 訊號日＋2026-10-12（245 窗）；超額＝對 0050 含息，pp／窗；全期 W_long 2006-04-11～2026-10-12。
二、選定方法：{ver}（選版規則：{how}；{status}）。
    群間：G2 Efron log-spline（df 5、c0 0.1、格點 [−8,4] 步 0.02、grid 節點）擬合 161 個 medoid 的 α̂（誤差拆成族群共同 vP＋個別 HAC L6）。
    群內：焦點 α＝α_medoid＋δ；δ 先驗 t3、sd＝√2·σ_w；σ_w 由所有 m_c ≥ 3 群的動差法估計（AMEND1 修正式）；δ 事後＝先驗 × N(d; δ, s_d²)，
    d＝焦點 − medoid 的平均差、s_d²＝差序列 HAC L6 變異；α 事後＝medoid 事後（含 vP 卷積）⊛ δ 事後（格點卷積）。
三、焦點格：AB、B10（官方日／13 個 j 等權混合 × 一般／壓力）；可執行 4 級距 × FIX／GROW × 一般／壓力 × {{KAPPA（o5 κ）、TICK（每邊 1 跳）}}。
    α_cell＝α_B10 事後 ⊛ 13 點等權平移 S_j＝mean(cell 官方日序列 − B10) ＋ mean(B10_Jj − B10 官方)（V6 jscan，缺窗補 0）。
四、配置：主策略 x＋0050 (1−x)，x∈{{0,10,…,100}}%；每窗再平衡（含成本）；平穩 bootstrap L6、20,000 路徑、10／20 年；α 由事後逆 CDF 抽（共同亂數）。
    效用 E[log W]（20 年為主）；勝率＝P(W_p>W_0050)；另報 P(財富比<0.75)、p10、MDD。
    輸出規則：兩模型 x* 差 ≤ 30pp 寫較小值，否則寫範圍。條件式：所有 GROW 可執行格 P(α>0)<0.5 → SOP 加註以 0050 為主。
五、新策略：市值匹配 11 版、權值輪動 8 版（日頻重建）各成一個新群，以同一方法（新群 medoid＝群內平均相關最高者）求事後；
    輪動登記焦點 d：N(0,0.5²) 先驗、d̂ 減 8 版最大值偏誤（bootstrap）後常態更新。
六、前瞻各線先驗與 σ：見 RESULTS.forward_priors 與 prereg_forward.json（序貫門檻在步驟 5 模擬後凍結）。
七、敏感度：c0、格點、σ_w×0.5／×2、j 點值（平均／中位數）、L12、拿掉 2021、拿掉 B10 最好 10 窗。
八、V7：只換輸入（s3_cells.py --inputs v7、s4_formal.py --inputs v7），結果存 RESULTS_V7。
"""

if __name__ == '__main__':
    main()
