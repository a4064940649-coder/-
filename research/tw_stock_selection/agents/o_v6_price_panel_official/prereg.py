#!/usr/bin/env python3
"""PREREG：在任何 2024 年以後的策略報酬之前寫入 results.json（含時間戳與 sha256）。之後改任何登記檔案都要重新登記。"""
import sys, json, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v6common as C
res = C.load_res()
assert res.get('GATE', {}).get('PASS'), '閘門沒過'
assert 'PREREG' not in res, 'PREREG 已存在（凍結後改引擎要另存 PREREG_superseded 再重登）'
H = C.HERE; AG = C.AG
scripts = ['v6common.py', 'srcload.py', 's0_gaps.py', 's1_explore.py', 's2_build.py', 's2_smoke.py', 's3_pools.py', 's3_gate.py', 's3_freeze.py',
           'v6env.py', 'v6stats.py', 's4_run.py', 's4_blk.py', 's4_jscan.py', 's5_hold_spa.py', 'prereg.py']
engine = [AG / 'independent_audit' / 'ia_core.py', AG / 'main_v2' / 'v2_core.py', AG / 'o_disposal_rejudge' / 'drj_core.py',
          AG / 'o_infra_v3_rebuild' / 'o3_core.py', AG / 'o_v4_revenue_patch_official' / 'v4lib.py', AG / 'o_stagger_tranche' / 'st_core.py',
          AG / 'o_main_spa_breaks' / 'spa_lib.py', AG / 'o_main_spa_breaks' / 'breaks_lib.py', AG / 'o_main_spa_breaks' / 'fam_common.py']
inputs = [H / 'data_freeze_v6.json', AG / 'o_v4_revenue_patch_official' / 'data_freeze_v4.json', AG / 'o_v4_revenue_patch_official' / 'navs_v4.parquet',
          AG / 'o_v4_revenue_patch_official' / 'plan_V4.csv', AG / 'o_v4_revenue_patch_official' / 'rfix_cells.csv',
          AG / 'o_main_spa_breaks' / 'data' / 'family_excess.parquet', AG / 'o_main_spa_breaks' / 'data' / 'family_excess_ew.parquet',
          AG / 'o_main_spa_breaks' / 'data' / 'family_members.csv', AG / 'o_main_spa_breaks' / 'data' / 'bench.parquet',
          AG / 'o_exec_capacity_tiers' / 'navs_abavg.parquet', AG / 'o_exec_capacity_tiers' / 'data' / 'bench.parquet',
          C.E50F, C.TAIEX, C.XTR / 'shares_hist.csv', C.XTR / 'ex_tsmc.csv'] + [C.DATA / f'{k}.parquet' for k in C.FIELDS + ['low']]
hashes = {str(p if Path(p).is_absolute() else H / p): C.sha(H / p if not Path(p).is_absolute() else p) for p in scripts + engine + inputs}
spa_o5 = json.loads((AG / 'o_main_spa_breaks' / 'results.json').read_text(encoding='utf-8'))['PREREG']
FROZEN = (C.HERE / 'frozen_spec.txt').read_text(encoding='utf-8')
PREREG = {
    'topic': 'o_v6_price_panel_official：V6 價格面板修補（2024-01 缺口＋以官方日 K 取代 2024-02～2025-09 的 5 分 K 近似），重跑 AB、B10、保留期與 SPA 判定欄',
    'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
    'frozen_spec': FROZEN,
    'data_version': json.loads((C.DATA / 'panel_v6_meta.json').read_text(encoding='utf-8'))['version'],
    'n_data_versions_tried': 8,
    'strategy_spec': 'V4 規格一字不改（v4lib.V4：前 300 流動性池、價 ≥ 10、52 週高點比百分位 0.5＋近 3 月營收年增率百分位 0.5、排除金融與 KY／DR、處置成交日口徑、n=10、每月第一個 ≥ 11 日的面板交易日訊號、隔日開盤成交、開盤漲停遞補、gapfix 滾動窗；AB＝v2_core.make_unavail mc seed 0～39 平均；B10＝缺營收下市股營收百分位固定 1.0 放回）。成本：一般＝滑價 0.1%＋手續費 6 折＋稅 0.3%；壓力＝每邊 0.5%。期間：年化起點 2006-04-11、終點 2026-10-12。',
    'columns': ['V6', 'AB_V6（40 seeds）', 'V6_B10', 'AB_V6_B10', '一般／壓力成本', '可執行版：o5 移植公式（o_exec_capacity_tiers 可執行 AB_V4 − 零衝擊 AB_V4 的逐日拖累），30 萬／100 萬／300 萬／1,000 萬 × FIX／GROW；2024-01 新增日期拖累記 0'],
    'per_column_reports': ['逐 seed t 中位數、範圍、跌破 2.0／2.3 的 seed 數', '換股日 j＝0～12（st_core 定義）t 中位數與 p10、年化中位數', '端點：拿掉最好 1/3/5/10 個月、拿掉第一年、起點 2008、拿掉最後 1 個月、日曆月 t',
                           '成本損益兩平（滑價 0.10%～0.40%）與手續費全額', 'S1～S4、SEL／VAL／HOLD、日曆歸屬 HOLD', 'V6 − V4 配對差（V4 格點）：非零月數、拿掉最大月後的 t'],
    'hold': {'pre': ['2006-04-01', '2023-12-31'], 'definition': '主口徑＝日曆歸屬：V6 格點上窗終點 ≥ 2024-01-02 的窗全部屬於 HOLD（含 2023-12-11→2024-01-11 窗）；另報起點歸屬（V6 格點）與 V4 格點日曆歸屬',
             'block_lengths': [3, 6, 12], 'B': 10000, 'seed': 5150, 'stat': 'Σ log((1+r_main)/(1+r_0050))', 'judge': 'p_lower(L6) ∈ [0.10, 0.90]→「與雜訊相容」；< 0.10→「落後超出雜訊」（附台積電缺口占比與 MDE）；不得寫「衰退已證實」，除非同池等權超額也出現斷點'},
    'spa': {'family': 'o_main_spa_breaks 的 F_long × W_long（2,319 條、V4 格點 245 窗）照舊；只把 o_v4_revenue_patch_official 的 14 條成員換成 V6 版', 'B': spa_o5['spa']['B'],
            'block_lengths': spa_o5['spa']['block_lengths'], 'variance': spa_o5['spa']['variance'], 'seed_base': spa_o5['spa']['seed_base'],
            'focus': ['AB_V6|normal（判定）', 'AB_V6_B10|normal（判定）', 'AB_V6|stress（加入族群當第 2,320 條）'], 'stat': 'p_SPA_c_focus，L6 為判定欄'},
    'decision': {'gate': '池內股日官方 OHLC ≥ 99%、2024-01-02～02-14 100%、官方對官方收盤一致率 ≥ 99.5%、除權息前收相符 ≥ 99%、2006～2023 逐位相同 → 已通過（GATE）',
                 'replace': '閘門與獨立複製都通過 → V6 取代 V4 成為正式資料版本（資料修補，不看方向）',
                 'sop_numbers': '|Δ 年化(AB_V6 − AB_V4, ALL)| ≤ 1pp → SOP 正式數字直接換 V6；> 1pp → 先拆到逐月逐檔才能換',
                 'grade': 'AB 焦點 p ≤ 0.10 且 B10 焦點 p ≤ 0.10 → 可提議回第 2 級（需總監確認）；否則維持第 3 級',
                 'live_expect': '實盤預期以 AB_V6_B10 可執行版 j 分布中位數為準', 'researcher_cap': '可用（資料）／待觀察（策略含義）'},
    'predictions': {
        'P1_coverage': '池內股日官方覆蓋 ≥ 99%，2024-01 100%；兩官方來源重疊期收盤一致率 ≥ 99.5%（已在探索驗證：見 explore.json／GATE）',
        'P2_approx_error': '13:20 近似收盤對官方絕對差中位數 0.1～0.4%、日報酬相關 > 0.97；誤差在收盤漲跌停、52 週比 ≥ 0.95、V60 排名 200～300 較大，前 50 大最小（已在探索檢驗）',
        'P3_identity': '2006～2023 日淨值逐位相同（已驗證 smoke.json）',
        'P4_AB': 'AB_V6 對 AB_V4：|Δ 年化| ≤ 0.6pp、t 2.35～2.65；配對差非零月 ≤ 22 個且集中在 2024-01～2025-10；HOLD 年化變化 ±3pp；開盤漲停遞補次數 2024-02～2025-09 變動 ≤ 3',
        'P5_hold': '日曆歸屬 33～34 窗，p_lower 0.04～0.20，點預測 0.08～0.12；MDE 約 2.2～3.1pp／月',
        'P6_SPA': 'AB 焦點 0.20～0.30、B10 焦點 0.35～0.50、壓力焦點 0.45～0.60，三者 > 0.10、第 3 級不變；AB ≤ 0.10 且 B10 ≤ 0.15 即否定',
        'P7_tsmc': '台積電缺口解釋落後比例 ≥ 60%'},
    'effective_tests_note': '資料版本 8（D1～D8）＋j-scan 13×4 欄＋SPA 焦點 3×區塊長度 3＋保留期 3 口徑；主策略改版輪數沿用 o5 的 ≥ 7 輪',
    'hashes': hashes,
}
res['PREREG'] = PREREG; res['PREREG_sha256'] = C.canon_sha(PREREG)
C.jdump(res, C.RES_F)
print('PREREG', res['PREREG_sha256'], len(hashes), 'files hashed')
