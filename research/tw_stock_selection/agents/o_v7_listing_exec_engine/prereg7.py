#!/usr/bin/env python3
"""PREREG：在任何 2024 年以後的策略報酬、任何可執行淨值之前寫入 results.json（含時間戳與 sha256）。之後改任何登記檔案都要重新登記。"""
import sys, json, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v7common as C
res = C.load_res()
assert res.get('GATE', {}).get('PASS'), '閘門沒過'
assert 'PREREG' not in res, 'PREREG 已存在'
H = C.HERE; AG = C.AG
scripts = ['v7common.py', 'srcload7.py', 's0_gaps.py', 's1_listing.py', 's1b_listing_check.py', 's1c_explore_misc.py', 'gatepop.py', 's2_build7.py', 's2_smoke7.py',
           's3_pools7.py', 's3_gate7.py', 's3_freeze7.py', 'v7env.py', 'v7stats.py', 's4_run7.py', 's4_jscan7.py', 's4_blk7.py', 's5_hold_spa7.py',
           'gen_model7.py', 'ecore7.py', 'run_exec7.py', 'dry_plans7.py', 'prereg7.py', 'frozen_spec.txt', 'listing_manual.json']
engine = [AG / 'independent_audit' / 'ia_core.py', AG / 'main_v2' / 'v2_core.py', AG / 'o_disposal_rejudge' / 'drj_core.py',
          AG / 'o_infra_v3_rebuild' / 'o3_core.py', AG / 'o_v4_revenue_patch_official' / 'v4lib.py', AG / 'o_stagger_tranche' / 'st_core.py',
          AG / 'o_main_spa_breaks' / 'spa_lib.py', AG / 'o_main_spa_breaks' / 'breaks_lib.py', AG / 'o_main_spa_breaks' / 'fam_common.py']
inputs = [H / 'data_freeze_v7.json', H / 'listing_master.csv', C.DATA / 'listed_twse.parquet', C.DATA / 'model_spec7.json', C.DATA / 'gate_population.parquet',
          AG / 'o_v4_revenue_patch_official' / 'data_freeze_v4.json', AG / 'o_v4_revenue_patch_official' / 'navs_v4.parquet',
          AG / 'o_v4_revenue_patch_official' / 'plan_V4.csv', AG / 'o_v4_revenue_patch_official' / 'rfix_cells.csv',
          C.V6DIR / 'navs_v6.parquet', C.V6DIR / 'navs_jscan_v6.parquet', C.V6DIR / 'plan_V6.csv', C.V6DIR / 'plan_V6_B10.csv',
          AG / 'o_main_spa_breaks' / 'data' / 'family_excess.parquet', AG / 'o_main_spa_breaks' / 'data' / 'family_excess_ew.parquet',
          AG / 'o_main_spa_breaks' / 'data' / 'family_members.csv', AG / 'o_main_spa_breaks' / 'data' / 'bench.parquet',
          C.E50F, C.TAIEX, C.XTR / 'shares_hist.csv', C.XTR / 'ex_tsmc.csv'] + [C.DATA / f'{k}.parquet' for k in C.FIELDS + ['low']]
hashes = {str(p if Path(p).is_absolute() else H / p): C.sha(H / p if not Path(p).is_absolute() else p) for p in scripts + engine + inputs}
spa_o5 = json.loads((AG / 'o_main_spa_breaks' / 'results.json').read_text(encoding='utf-8'))['PREREG']
PREREG = {
    'topic': 'o_v7_listing_exec_engine：V7＝以全市場上市身分重建股池、修補颱風除權息與 2024-01 前視；可執行引擎逐名單（每個 j）模擬衝擊、處置期賣出、合併下市結算延遲；交出 V7 正式數字',
    'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
    'frozen_spec': (H / 'frozen_spec.txt').read_text(encoding='utf-8'),
    'data_version': C.jload(C.DATA / 'panel_v7_meta.json')['version'], 'n_data_versions_tried': 6,
    'gate_result': {k: res['GATE'][k] for k in ('checks', 'PASS', 'coverage_conservative', 'coverage_2024_01_02_to_02_14')},
    'seen_periods_note': '2024-01～2026-10 的報酬已被 o5、o6 看過（V4／V6 淨值），不是盲測；本題是資料與引擎修補，不改策略規格。2026-11 起前瞻盲測不受影響。',
    'run_order': ['s4_run7.py', 's4_jscan7.py', 's4_blk7.py v7', 's4_blk7.py v4', 'run_exec7.py', 's5_hold_spa7.py', 's6_summary7.py（事後彙整，只讀結果）', 'replicate/（獨立複製）'],
    'exec_grid': {'strats': ['AB', 'B10'], 'plans': {'OFF': 'seeds 0～39', 'J0..J12': 'seeds 0～19'}, 'tiers': ['30萬', '100萬', '300萬', '1000萬'], 'versions': ['FIX', 'GROW'],
                  'costs': ['normal', 'stress'], 'models': {'K5T': '上界', 'TICK': '下界', 'K5': '敏感度（官方日、一般成本、seeds 0～19）'}},
    'predictions': {
        'P1_gate': '全市場上市母體池內官方覆蓋 ≥ 99.5%；補回的上市空洞股在 2024-07～2025-03 每個訊號日有 3～6 檔進前 300，進前 10 名總次數 ≤ 6（已在閘門檢驗：1～5 檔、1 次）',
        'P2_AB': 'AB_V7 對 AB_V4（及 V6）：跨循環年化 |Δ| ≤ 0.6pp，配對差 |t| < 1.5，非零月份集中在 2024～2025；AB_V7 t 2.40～2.70；j 中位數 t 1.6～1.9',
        'P3_2019': '2019 年起對 0050 的 t 在 1.1～1.5；保留期月均超額 |·| ≤ 0.3pp',
        'P4_disp_merger': '處置期賣出建模後每年拖累 0.05～0.3pp；合併下市結算延遲使 B10 每年少 0.05～0.2pp',
        'P5_exec': 'B10 可執行 j 中位數：1,000 萬在 K5T 模型下 ≤ 0050（約 14.9%）；TICK 模型下 30 萬～100 萬高於 0050 約 1～4pp；兩模型差距隨資金規模單調擴大；若 30 萬 FIX 在 TICK 下也輸 0050，「小資金有優勢」被否定',
        'P6_SPA': 'SPA 焦點 p：AB 0.18～0.30、B10 0.30～0.45、壓力 0.45～0.60；第 3 級不變'},
    'decision': {'gate': '已通過（GATE）', 'replace': '閘門與獨立複製都通過 → V7 取代 V4（資料修補，不看方向）；|Δ年化| ≤ 1pp → SOP 數字直接換 V7；> 1pp → 先拆解',
                 'grade': 'AB 與 B10 焦點 p 都 ≤ 0.10 → 可提議回第 2 級；否則第 3 級',
                 'live_expect': '實盤預期表＝B10 可執行 j 中位數（一般成本）為主欄，並列 p10 與 K5T／TICK；兩模型對是否贏 0050 相反 → 無法判斷',
                 'replication': '面板與上市矩陣逐格 ≥ 99.9%；AB_V7、B10_V7 日淨值 ≤ 1e-8；SPA p 差 ≤ 0.02；p_lower 差 ≤ 0.02；可執行表各格年化差 ≤ 0.2pp；超過取不利方向',
                 'researcher_cap': '可用（資料與引擎）／待觀察（策略含義）'},
    'hold': {'pre': ['2006-04-01', '2023-12-31'], 'block_lengths': [3, 6, 12], 'B': 10000, 'seed': 5150, 'stat': 'Σ log((1+r_main)/(1+r_0050))', 'versions': ['raw', 'shrink50', 'zero']},
    'spa': {'B': spa_o5['spa']['B'], 'block_lengths': spa_o5['spa']['block_lengths'], 'variance': spa_o5['spa']['variance'], 'seed_base': spa_o5['spa']['seed_base'],
            'focus': ['AB_V7|normal', 'AB_V7_B10|normal', 'AB_V7|stress（第 2,320 條）']},
    'effective_tests_note': '資料版本 6（L1～L5＋V6REPRO）＋j-scan 13×4 欄×2 成本＋可執行 2 模型×4 級距×2 版本×2 成本×14 名單組＋SPA 3 焦點×3 L＋保留期 3 口徑×3 版；主策略改版鏈 ≥ 8 輪',
    'hashes': hashes,
}
res['PREREG'] = PREREG; res['PREREG_sha256'] = C.canon_sha(PREREG)
C.jdump(res, C.RES_F)
print('PREREG', res['PREREG_sha256'], len(hashes), 'files hashed', PREREG['time_utc'])
