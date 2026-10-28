#!/usr/bin/env python3
"""步驟 7：彙整（只讀 results.json，不計算新統計）：預測對照、判定、檔案清單。寫 results.json 的 SUMMARY，並補 explore.json 的結論欄。
用法：python3 -I -B s9_summary.py
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tlib as T

def main():
    res = T.load_res(); R = res['RESULTS']; S5 = res['STEP5_SEQSIM']
    cp = R['cells_posterior']; al = R['allocation']; tr = R['tier_rule']
    growP = {k: v['p_pos'] for k, v in cp.items() if k.startswith('EXEC') and '|GROW|normal|' in k}
    win10 = {k: al[k]['10y']['grid']['100']['P_win'] for k in al if k.startswith('EXEC')}
    diffs = {k: v['diff_pp'] for k, v in tr.items()}
    ns = R['new_strategies']
    P = {
     'P1': {'post_over_raw_AB_B10_official': [cp['AB']['post_over_raw'], cp['B10']['post_over_raw']],
            'post_over_raw_AB_J_B10_J': [cp['AB_J']['post_over_raw'], cp['B10_J']['post_over_raw']],
            'GROW_normal_P_pos': growP,
            'verdict': '不成立（官方日 AB／B10 為 0.77／0.81，與 o6 相同；群內懲罰在選定模型下幾乎為零）；j 混合與可執行格落在 0.43～0.69 是 j 與成本造成，不是群內懲罰'},
     'P2': {'xstar_30F': [tr['EXEC|30萬|FIX|normal']['xstar_KAPPA_20y'], tr['EXEC|30萬|FIX|normal']['xstar_TICK_20y']],
            'xstar_1000G': [tr['EXEC|1000萬|GROW|normal']['xstar_KAPPA_20y'], tr['EXEC|1000萬|GROW|normal']['xstar_TICK_20y']],
            'two_model_diff_pp': diffs, 'verdict': '大致成立：30 萬 FIX 寫 90%、1,000 萬 GROW 寫 0%；300 萬以上 FIX／GROW 一般成本兩模型差 40～100pp（1,000 萬 GROW 恰 30pp），100 萬 GROW 也有 80pp；否定條件（全部 ≤ 30pp）未發生'},
     'P3': {'max_win10_x100': max(win10.values()), 'min_win10_x100': min(win10.values()),
            'verdict': '部分不成立：最有利格 0.62（> 0.55）；最不利格 0.09（≤ 0.35 成立）'},
     'P4': S5['prediction4'] | {'verdict': '部分不成立：誤升級 ≤ 5%（1.9%）與短期難升級成立；α −1 時 24 個月下修只有 0.25%（預測 ≥ 70%），前瞻規則對主策略連失效都偵測不到，只有財富比審查（24 個月 58%）有反應'},
     'P5': {k: {'post_P': v['p_pos'], 'Phi_t_hac': v['P_from_raw_t_hac']} for k, v in ns.items()} | {'verdict': '成立：三個新策略焦點的事後 P(α>0) 都低於 Φ(t)'}}
    SUM = {'predictions': P,
           'acceptance': {'synthetic_gate_primary': '全部 7 版未過（登記主情境把焦點群真值限在右尾，對任何貝氏事後都必然往下偏；事後診斷在不限右尾時 V2／V5／V7 通過）',
                          'o6_repro': res['STEP1_O6_REPRO']['pass_le_0.01'], 'false_upgrade_le_5pct': S5['prediction4']['holds']['false_up_le_0.05'],
                          'failure_detection_ge_70pct_24m': S5['prediction4']['holds']['down-1_ge_0.70'],
                          'fake_forward_bitwise': res.get('FAKE_FORWARD_TEST', {}).get('identical_two_runs'), 'independent_replication': '未做（待複製員）'},
           'sop_conditional_0050_primary': R['sop_conditional_0050_primary'],
           'verdict': '待觀察', 'mechanism': '部分支持', 'utc': T.now_utc()}
    T.save_key('SUMMARY', SUM)
    ex = json.loads((HERE / 'explore.json').read_text(encoding='utf-8'))
    ex['conclusion'] = {'mechanism_visible': True,
        'note': 'SEL 內群內贏家詛咒看得到：m_c ≥ 5 的 31 群，前半群內最佳成員相對 medoid 的偏離 0.43pp 在後半只剩 0.11（保留 26%）；第 114 群最佳成員 0.50 → −0.18。'
                '主策略在 SEL 是第 114 群（228 條）的第 25 名、全族群第 28 名；第 114 群 medoid 與主策略 α̂ 幾乎相同（0.94 vs 1.01），群內離散 σ_w 只有 0.10～0.19pp。'
                '群內 228 條取最大值的雜訊偏誤（bootstrap）0.43pp（sd 0.22～0.33），約等於成員標準誤的 1.1 倍。'}
    (HERE / 'explore.json').write_text(json.dumps(ex, ensure_ascii=False, indent=1, default=T._js), encoding='utf-8')
    print(json.dumps(SUM, ensure_ascii=False, indent=1, default=T._js)[:5000])

if __name__ == '__main__':
    main()
