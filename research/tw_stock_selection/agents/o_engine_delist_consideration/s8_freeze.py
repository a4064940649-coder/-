#!/usr/bin/env python3
"""凍結 SPEC（任何登記重算之前）。寫入 results.json 的 PREREG、PREREG_sha256、PREREG_utc；已存在就拒絕覆寫。
python3 -I -B s8_freeze.py"""
import sys, json, hashlib
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import cz_lib as Z
RJ = HERE / 'results.json'
R = json.loads(RJ.read_text(encoding='utf-8')) if RJ.exists() else {}
assert 'PREREG' not in R, 'PREREG 已存在，不得覆寫'
assert 'RESULTS' not in R
FILES = ['events_v2.csv', 'cz_engine.py', 'cz_v7x.py', 'cz_lib.py', 'census.json', 'explore.json', 'iterations.json', 'smoke.json', 'smoke_v7x.json',
         's1_census.py', 's3_events.py', 's4_plans.py', 's5_smoke.py', 's6_v7x_smoke.py', 'plans_ia_meta.json']
EXT = {'ia_core.py': Z.AG / 'independent_audit' / 'ia_core.py', 'x_exec.py': Z.AG / 'o_exec_engine_rewrite_v7x' / 'x_exec.py',
       'x_data.py': Z.AG / 'o_exec_engine_rewrite_v7x' / 'x_data.py', 'x_stats.py': Z.AG / 'o_exec_engine_rewrite_v7x' / 'x_stats.py',
       'navs_oos_R4.parquet': Z.AG / 'o_exec_engine_rewrite_v7x' / 'out' / 'navs_oos_R4.parquet', 'navs_gapfix.parquet': Z.AG / 'main_v2' / 'navs_gapfix.parquet',
       'v2_core.py': Z.AG / 'main_v2' / 'v2_core.py', 'fslib.py': Z.AG / 'o_forward_sequential_v2' / 'fslib.py', 'plans_ia.pkl': Z.WORK / 'plans_ia.pkl'}
S = json.loads((HERE / 'events_v2_summary.json').read_text(encoding='utf-8'))
SPEC = dict(
    topic='o_engine_delist_consideration：共用引擎下市「收購／換股對價」口徑（基礎設施，不產生訊號）',
    settlement_rules={
        'trigger': '持有中的證券在本引擎面板「最後成交日 lt」的下一個交易日轉為請求權；代碼重用時以終止上市日切開新舊證券。',
        'cash': '股數 × 每股現金對價；付款日未知（全部事件皆未知）＝終止日＋20 個交易日入帳；入帳前以 0% 現金計（主口徑）；A16 的 0050 計值為敏感度（c50）。現金對價不課證交稅、不付手續費（假設）。',
        'swap/conversion': '股數 × 換股比例 → 存續股；自存續股首個可交易日 s 起隨其日報酬（s 為換股日時以 s 開盤價計入）；空窗期（lt→s）市值固定於最後收盤；之後由一般換股流程以一般賣出成本出清（存續股若入選則照目標權重保留）。',
        'mixed': '現金部分同 cash ＋ 股票部分同 swap ＋ 特別股部分以發行價固定市值、s 起下一個換股日以一般賣出成本出清。',
        'swap_ur': '併入上市存續公司已核實、有效比例未核實：比例＝lt 當日（標的收盤 ÷ 存續收盤）×(1+δ)，其餘同 swap。',
        'acq_ur': '被併購已核實、對價形式或金額未核實（或存續公司不在面板）：最後收盤 ×(1+δ) 的固定請求權，終止日＋20 交易日轉現金、不付成本。',
        'unknown': '無公開文字證據：上界＝最後收盤固定、終止日＋20 交易日轉現金（main 與 hi 皆用上界）；下界（lo）＝0（lt 後第一個換股日沖銷，沖銷前市值固定）。兩個界限都報。',
        'distress': '主口徑 0；最後成交價為對照（本題被持有事件中沒有 distress）。',
        'price_gap': '本引擎面板在真實最後交易日前 7 天以上就沒有價格（ia 面板 2019–2023 存活者缺口）：以面板最後收盤固定，下一個換股日以一般賣出成本出清（＝nofreeze 處理）。',
        'bounds': 'δ：main＝0、lo＝−d、hi＝+d；d＝0.057（已核實事件 lt 當日最大偏離 5.66% 無條件進位到 0.1%）。',
        'not_in_table': '不在事件表的永久停牌：ia 維持 freeze；V7X 沿用原 XR／MCASH（計數 n_fallback）。',
        'old_modes': 'freeze、nofreeze、delist_zero 保留作對照；freeze 仍為 ia_core 預設。',
    },
    events_v2=dict(file='events_v2.csv', sha256=Z.sha(HERE / 'events_v2.csv'), summary=S),
    recompute_set={
        'C0': 'main_v2 gapfix 正式、排除金融 n10（s4_plans.py 重建，名單與 o14 lists_C0、FM_COUNTS 逐筆相同）',
        'U_opt': 'ER2 樂觀放回名單（FM_COUNTS C0|U_opt）', 'U_opt_F2': 'U_opt 加 FM 靜態金融遮罩 F2',
        'ia_engines': ['freeze', 'nofreeze', 'dzero', 'cons_main', 'cons_lo', 'cons_hi', 'cons_c50'], 'ia_costs': ['normal', 'stress'],
        'V7X': 'AB_V7_B10 名單、K5T、GROW、一般／壓力 × 100 萬／300 萬／1,000 萬 × 13 j × 20 seeds（R4 設定＝orig_cfg＋f1,f2,f3=0.5,f4）；引擎 orig（＝現行正式，須逐欄重現 navs_oos_R4.parquet）、cons_main、cons_lo、cons_hi、cons_c50',
    },
    judgement_fields={
        'C0 / U_opt / U_opt_F2（integrate_o16 窗口口徑，一般成本）': ['ALL 年化', 'ALL 對 0050 t', 'SEG1 2006～2012 年化超額', 'SEG2 2013～2018 年化超額', 'SEG3 2019 起年化超額', '拿掉最好 3 年 t'],
        'C0 壓力成本': ['ALL 年化', 'ALL 對 0050 t'],
        'V7X（每個資金級距 × 一般／壓力）': ['全期 j 中位數年化 CAGR_med', '全期 j 中位數年化超額 exc_med', '全期 j 中位數 t_0050_med', '全期 j p10 年化超額 exc_p10',
                                   '2019 起 j 中位數年化 CAGR_Y19on_med', '2019 起 j 中位數年化超額 exc_Y19on_med', '2019 起 j 中位數 t_Y19on_med', '2019 起 j p10 年化超額 exc_Y19on_p10'],
    },
    threshold_rule='|Δ年化| > 0.3pp 或 |Δt| > 0.05 → 加註；兩者都 ≤ → 不加註',
    threshold_detail='Δ＝consideration 減現行正式口徑（C0／U_opt：freeze；V7X：orig＝A16）。年化類欄位（年化、年化超額、p10）用 0.3pp，t 類欄位用 0.05。'
                     'unknown／distress 與 δ 界限：對 main、lo、hi、c50 四個 consideration 版本逐一算 Δ，取絕對值最大者判定（偏向加註）。'
                     '主策略正式數字指 C0 與 V7X；U_opt／U_opt_F2 只報四欄（freeze、nofreeze、consideration、dzero）並更新 o15／o16 引用，不改 ER2 的 G4 第一段判定。',
    falsification='consideration 落在 [dzero, max(freeze, nofreeze)＋事件價差] 之外先當實作錯誤查；任何事件結算值與公開對價差 > 0.5% 代表表或程式有錯（引擎逐事件斷言）。',
    predictions=dict(
        P1=dict(text='C0 從未持有下市凍結股 → consideration 與 freeze 逐日差＝0', known_before_registration=True, evidence='smoke.json (c)：main／lo／hi × 一般／壓力 maxabs 0、settled 0'),
        P2=dict(text='U_opt 的 consideration 年化落在 nofreeze ±0.5pp 內，且比 freeze 高 ≥ 3pp', known_before_registration='部分：U_opt 對 C0 的 freeze −4.18、nofreeze −0.75 已知（方向已知、幅度未知）；consideration 未算'),
        P3=dict(text='V7X 判定欄 |Δ年化| ≤ 0.1pp、|Δt| ≤ 0.02；影響集中在 2016～2018', known_before_registration='部分：普查已知 V7X B10 名單 260 份全數持有 2311／2325／3474／6286（2016～2018）與 2006～2012 多起；數值未算'),
        P4=dict(text='被持有事件中現金＋換股＋股份轉換合計 ≥ 85%、困境 ≤ 15%；困境用 0 與最後價對主策略正式數字差 < 0.3pp',
                known_before_registration='事件組成已知（登記前）：28 起中條款已核實 11、類型已核實 11、無證據 6；困境 0 起。依「類型已核實＋條款已核實」計 22/28＝78.6%，未達 85%（因 6 起無證據，不是因為困境）。差 < 0.3pp 部分未知'),
        P5=dict(text='換股比例 × 存續股價相對最後收盤的逐事件中位數 |偏差| < 3%；股份轉換空窗個別事件可到 ±10%',
                known_before_registration='已知（events_v2 欄位）：已核實換股類 lt 當日 |偏差| 2823 5.66%、2847 4.36%、3697 0.51% → 中位數 4.36% > 3%；空窗 s 日 2311 −9.8%、2448 +9.9%、3698 +12.5%'),
    ),
    stats_def=dict(ia='integrate_o16 run.py 的 win_v2／stats_one 口徑（訊號日收盤→下一訊號日收盤窗，終點 2026-10-12；ann＝窗連乘年化；t＝窗超額 t；拿掉最好 3 年＝窗超額年和最大的 3 年）',
                   v7x='x_stats.stats／jsumm（j 中位數與 p10；seeds 日報酬平均成一條淨值，同 x_run.run_grid）'),
    no_signal='本題不產生任何選股訊號，也不改主策略規則；consideration 成為共用引擎選項，是否改預設由研究中心另案決定。',
    file_sha256={**{f: Z.sha(HERE / f) for f in FILES}, **{k: Z.sha(v) for k, v in EXT.items()}},
)
txt = json.dumps(SPEC, ensure_ascii=False, sort_keys=True)
R['PREREG'] = SPEC; R['PREREG_sha256'] = hashlib.sha256(txt.encode()).hexdigest(); R['PREREG_utc'] = pd.Timestamp.now('UTC').isoformat()
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
print(R['PREREG_sha256'], R['PREREG_utc'])
