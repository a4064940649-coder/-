#!/usr/bin/env python3
"""缺淨值成員清單（不能在 30 分鐘內重跑者）＋合成成員規格。
來源：(1) 研究資料夾中沒有淨值檔、只報統計的規則（人工整理自各 FINDINGS.md 的結論段）；
      (2) 第 1～3 輪的早期搜尋（output/bt_factor_search.json、bt_revenue_robust.json、bt_aggressive.json、bt_final_rule.json）。
合成方法（只用於敏感度，不進主族群）：取「最相似成員」的對 0050 月超額序列，各段先去均值再加上
  μ_seg = t_seg × sd_seg / √n_seg，使該段 t 等於報告值；波動與對其他成員的相關都和最相似成員相同。
  span='long'：段 1＝窗起點 < 2019-05、段 2＝≥ 2019-05；span='recent'：只在 2021-03 起有值，段 1＝2021-03～2023-12、段 2＝2024-01 起。
  事件型規則只報「每筆 t_month」時，取其值但截在 [−3, +3]；只寫「不可用、未報 t」者記 0。
輸出 data/missing_members.csv。"""
import json, sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fam_common as C
OUTDIR = Path('/home/user/-/research/tw_stock_selection/output')

SIM = {   # 類別 → 最相似的真實成員（族群矩陣中的 member_id）
    'main': 'o_v4_revenue_patch_official/navs_v4.parquet::AB_V4|normal',
    'price': 'momentum_family/long_nav.parquet::P1_52wHigh_n30_0.001',
    'rev': 'combo_search/search_nav.parquet::C2|n30|raw',
    'flow': 'combo_search/search_nav.parquet::C9|n30|raw',
    'timing': 'strategy_ensemble/strategy_nav.parquet::E0050_MA',
    'event': 'w3_limitup_pullback/nav_primary.parquet::net',
    'disposal': 'disposal_v2/navs.parquet::TWSE_K10_0050',
    'disp_combo': 'disposal_v2/navs.parquet::combo_n10_80_20_official',
}
cap = lambda x: max(-3.0, min(3.0, x))
# (id, 來源資料夾, span, t1, t2, 類別, 說明)
CURATED = [
    ('adm_B1', 'attention_disposal_mechanism', 'recent', 0.18, -0.29, 'disposal', 'B1 連續 2 日注意，t_month'),
    ('adm_B2', 'attention_disposal_mechanism', 'recent', 0.92, -0.73, 'disposal', 'B2 第 5 日注意'),
    ('adm_B4', 'attention_disposal_mechanism', 'recent', -1.72, -1.72, 'disposal', 'B4 處置解除後'),
    ('etf_R1', 'etf_rebalance', 'long', 1.53, 1.53, 'event', '臺灣50 納入股（按審核期分群 t）'),
    ('etf_R2', 'etf_rebalance', 'long', 2.0, 1.33, 'event', '臺灣50 刪除股（對等權 t）'),
    ('ev_earnings', 'events', 'recent', 3.0, 2.80, 'rev', '財報季窗口營收強勢股（窗口 t，截 3）'),
    ('ev_capred', 'events', 'long', 0.0, 0.0, 'event', '減資後，不顯著'),
    ('ind_cycle', 'industry', 'recent', 1.15, -2.61, 'rev', '產業營收循環'),
    ('ind_lead_px', 'industry', 'recent', 0.93, 0.40, 'price', '龍頭領先（價格）'),
    ('ind_lead_rev', 'industry', 'recent', 2.13, -3.0, 'rev', '龍頭領先（營收），驗證 −3.03 截 −3'),
    ('ind_leadlag', 'industry', 'recent', 2.8, -2.6, 'price', '產業領先落後（對等權）'),
    ('ind_mom', 'industry', 'recent', 2.24, -0.1, 'price', '產業動能 RS_52wH（IC t；驗證年化略輸 0050）'),
    ('ind_mask', 'industry', 'recent', 2.04, -0.91, 'main', '主策略加產業遮罩（對主策略配對 t）'),
    ('irl_A', 'industry_revenue_long', 'long', 1.5, 1.0, 'main', '產業營收長歷史 A 組態（未報對 0050 t，取保守估計）'),
    ('insider', 'insider_public', 'recent', 0.0, 0.0, 'flow', '內部人與公開資訊，|t| < 1.4'),
    ('inst_trust', 'institutions', 'recent', 0.77, -1.85, 'flow', '投信買超（延後一天）'),
    ('inst_foreign', 'institutions', 'recent', 1.74, 0.0, 'flow', '外資買超（延後一天）'),
    ('ml_ridge', 'ml', 'long', 0.0, -1.86, 'price', 'Ridge（挑選期未報對 0050 t）'),
    ('ml_lgbm', 'ml', 'long', 0.0, -1.21, 'price', 'LightGBM'),
    ('ml_nn', 'ml', 'long', 0.0, -1.14, 'price', '淺層網路'),
    ('ml_combo', 'ml', 'long', 0.0, -1.94, 'price', '模型合成'),
    ('prepriced', 'o_prepriced_growth', 'long', 0.93, 0.0, 'rev', 'AVG|B1|n30（挑選期失敗，未開驗證期）'),
    ('ss_seasonal', 'seasonal_statarb', 'long', -0.75, -2.1, 'price', '個股同月季節性'),
    ('ss_indrev', 'seasonal_statarb', 'long', -2.24, -2.0, 'price', '產業內反轉'),
    ('ss_pairs', 'seasonal_statarb', 'long', -1.18, -1.18, 'price', '只做多配對'),
    ('ss_calendar', 'seasonal_statarb', 'long', 2.21, -1.11, 'timing', '春節前後 0050 擇時'),
    ('st_overnight', 'short_term', 'long', -3.0, -3.0, 'event', '隔夜（每筆 t −12～−17，截 −3）'),
    ('st_gapdown', 'short_term', 'long', -2.76, -3.0, 'event', '跳空下跌當沖'),
    ('tech_best', 'technical', 'long', 1.4, -0.81, 'price', '21 條經典技術規則中最好的帶量突破'),
    ('tech_comp', 'technical', 'long', 0.0, -1.68, 'price', '技術合成'),
    ('timing_daily', 'timing_daily', 'long', -0.02, -1.05, 'timing', '0050 日頻擇時開關'),
    ('w12_divmonth', 'w12_dividend_month_premium', 'long', -2.67, -3.0, 'event', '配息月溢酬'),
    ('w12_ffboll', 'w12_foreign_flow_bollinger_timing', 'recent', -0.33, -1.94, 'timing', '外資流向布林擇時'),
    ('w12_ipo2', 'w12_ipo_secondary_listing_high', 'long', 0.0, 0.0, 'event', '無 FINDINGS，記 0'),
    ('w12_preholiday', 'w12_pre_holiday_0050', 'long', 2.43, -1.27, 'timing', '節前 0050（每次 t）'),
    ('w12_tom', 'w12_turn_of_month_0050', 'long', -0.36, -0.04, 'timing', '月初效應 0050'),
    ('w1_bias', 'w1_bias_reversion', 'long', -3.0, -3.0, 'event', '乖離反轉（不可用，事件淨異常為負）'),
    ('w1_granville', 'w1_granville_pullback_ma60', 'long', -3.0, -3.0, 'event', '葛蘭碧回測'),
    ('w1_kd', 'w1_kd_high_passivation', 'long', -3.0, -3.0, 'event', 'KD 高檔鈍化'),
    ('w1_squeeze', 'w1_ma_squeeze_breakout', 'long', -3.0, -3.0, 'event', '均線糾結突破'),
    ('w1_trust_small', 'w1_trust_first_buy_small', 'recent', 0.0, 0.0, 'flow', '投信首買小型股'),
    ('w1_tuyang', 'w1_tuyang_streak', 'recent', -2.44, -0.73, 'flow', '土洋連買'),
    ('w3_adr', 'w3_adr25_breadth_timing', 'long', -2.17, -2.97, 'timing', 'ADR 廣度擇時'),
    ('w3_rsrs', 'w3_rsrs_timing', 'long', -2.54, -1.88, 'timing', 'RSRS 擇時'),
    ('w3_voltrough', 'w3_volume_trough', 'long', -3.0, -3.0, 'event', '量縮低點'),
    ('w4_ddadd', 'w4_0050_drawdown_add', 'long', 1.16, 1.13, 'timing', '0050 跌深加碼'),
    ('w4_engulf', 'w4_bullish_engulfing_low', 'long', 0.0, 0.0, 'event', '低檔吞噬'),
    ('w4_choke', 'w4_choke_volume_first_red', 'long', -2.38, 0.0, 'event', '爆量首紅'),
    ('w4_chu', 'w4_chu_jiahong_strong_bar_ma5_exit', 'long', 0.0, 0.0, 'event', '強勢長紅 5 日線出場'),
    ('w4_climax', 'w4_climax_volume_black_exit', 'long', 1.25, 1.16, 'price', '高檔爆量長黑出場（加在主策略上，對 0050 t）'),
    ('w4_dryup', 'w4_dryup_volume_rebound', 'long', 0.0, 0.0, 'event', '量縮反彈'),
    ('w4_ma20', 'w4_ma20_deduction_low', 'long', 0.0, 0.0, 'event', '月線扣抵'),
    ('w4_obv', 'w4_obv_leads_new_high', 'long', 0.0, 0.0, 'event', 'OBV 領先'),
    ('w4_tower', 'w4_tower_line_n3', 'long', 0.0, 0.0, 'event', '寶塔線'),
    ('w4_vlp', 'w4_vol_leads_price_vma_cross', 'long', 0.0, 0.0, 'event', '量先價行'),
    ('w4_vr', 'w4_vr_volume_ratio', 'long', -2.16, -3.0, 'event', 'VR 量比'),
    ('w4_wbottom', 'w4_w_bottom_neckline', 'long', -3.0, -3.0, 'event', 'W 底頸線'),
    ('w4_xiao', 'w4_xiao_volume_dimple_attack', 'long', 0.0, 0.0, 'event', '量窩攻擊'),
]

def harvest_output():
    rows = []
    fs = json.loads((OUTDIR / 'bt_factor_search.json').read_text())
    facs = sorted(set(k.split('|')[0] for k in fs))
    for f in facs:
        if 'REFERENCE' in f: continue
        a = fs.get(f + '|select_2005_2019-04'); b = fs.get(f + '|validate_2019-05_2026-10')
        if a and b and 't_top30_vs_0050' in a and 't_top30_vs_0050' in b:
            cat = 'rev' if ('rev' in f or 'r_' in f.split(':')[1][:2]) else ('flow' if f.split(':')[1].startswith('c_') else 'price')
            rows.append((f'fs_{f}', 'output/bt_factor_search', 'long', a['t_top30_vs_0050'], b['t_top30_vs_0050'], cat, '早期大範圍搜尋 top30'))
            continue
        a = fs.get(f + '|select_2021-03_2023-12'); b = fs.get(f + '|validate_2024-01_2026-10')
        if a and b and 't_top30_vs_0050' in a and 't_top30_vs_0050' in b:
            nm = f.split(':')[1]
            cat = 'rev' if nm.startswith('r_') or 'rev' in nm else ('flow' if nm.startswith('c_') else 'price')
            rows.append((f'fs_{f}', 'output/bt_factor_search', 'recent', a['t_top30_vs_0050'], b['t_top30_vs_0050'], cat, '早期大範圍搜尋 top30（2021 起）'))
    rr = json.loads((OUTDIR / 'bt_revenue_robust.json').read_text())
    for k in ['R5_base', '2_s52_lag1', '3_no_limit_up_entry', '4_n10', '4_n20', '4_n50', '4_liq_top200', '4_liq_top500']:
        t = rr[k]['t_vs_0050']; rows.append((f'rr_{k}', 'output/bt_revenue_robust', 'recent', t, t, 'main', '早期主策略穩健性（2021～2026 全期 t）'))
    ag = json.loads((OUTDIR / 'bt_aggressive.json').read_text())
    def walk(o, p, out):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, dict) and 't_vs_0050' in v: out.append((k, v['t_vs_0050']))
                else: walk(v, p + '/' + k, out)
    out = []; walk(ag, '', out)
    for k, t in out:
        if k == '0050' or t != t: continue
        cat = 'disposal' if k == '處置股' else ('disp_combo' if '處置' in k else 'main')
        rows.append((f'ag_{k}', 'output/bt_aggressive', 'recent', t, t, cat, '早期積極配置（舊資料）'))
    fr = json.loads((OUTDIR / 'bt_final_rule.json').read_text())
    out = []; walk(fr, '', out)
    for k, t in out:
        rows.append((f'fr_{k}', 'output/bt_final_rule', 'recent', t, t, 'main', '早期最終規則與敏感度'))
    return rows

def main():
    rows = [r[:3] + (cap(r[3]), cap(r[4])) + r[5:] for r in CURATED]
    rows += [r[:3] + (float(r[3]), float(r[4])) + r[5:] for r in harvest_output()]
    df = pd.DataFrame(rows, columns=['id', 'source', 'span', 't1', 't2', 'cat', 'note'])
    df['similar_member'] = df.cat.map(SIM)
    assert df.similar_member.notna().all()
    df.to_csv(C.DATA / 'missing_members.csv', index=False)
    print(df.groupby(['span', 'cat']).size()); print(len(df))
    print(df.sort_values('t2', ascending=False).head(10).to_string())

if __name__ == '__main__':
    main()
