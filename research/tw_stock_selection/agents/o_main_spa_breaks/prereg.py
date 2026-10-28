#!/usr/bin/env python3
"""事前登記：把最終規格、族群矩陣與全部腳本、輸入資料的 sha256 寫進 results.json 的 PREREG（附時間戳）。
之後 run_tests.py／run_breaks.py／run_hold.py／crosscheck_arch.py 都會先斷言 PREREG 與所有雜湊。"""
import sys, json, glob, os
from datetime import datetime, timezone
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fam_common as C

PANEL = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
EXTRAS = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')

def main():
    res = C.load_res()
    assert 'PREREG' not in res, 'PREREG 已存在；不得覆寫（若必須重登記，先把舊的移到 PREREG_superseded 並說明理由）'
    it = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
    scripts = sorted(os.path.basename(p) for p in glob.glob(str(HERE / '*.py')))
    data = sorted(os.path.relpath(p, HERE) for p in glob.glob(str(HERE / 'data' / '*')) if p.endswith(('.parquet', '.csv')))
    hashes = {f: C.sha(HERE / f) for f in scripts + data + ['explore.json', 'iterations.json']}
    ext = [C.V4DIR / 'navs_v4.parquet', C.V4DIR / 'plan_V4.csv', C.E50F, PANEL / 'adj.parquet', PANEL / 'close_raw.parquet',
           EXTRAS / 'ex_tsmc.csv', EXTRAS / 'shares_hist.csv', EXTRAS / 'etf0050_w.csv', C.AG / 'strategy_ensemble' / 'exhaustive_all.csv.gz']
    for p in ext: hashes[str(p)] = C.sha(p)
    PREREG = {
        'topic': 'o_main_spa_breaks：主策略的真實顯著性（White RC／Hansen SPA）與結構斷點，2024～2026 落後是雜訊還是衰退（方法題）',
        'time_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'main_id': 'o_v4_revenue_patch_official/navs_v4.parquet::AB_V4|normal',
        'main_spec': 'AB_V4（V4 凍結版、只用已公告營收 mc 40 seed 平均、一般成本＝滑價 0.1%＋手續費 6 折＋證交稅 0.3%）；訊號窗 t 2.5118（與 o4 正式值逐位相同）',
        'grid': '主策略訊號日（plan_V4.csv signal，每月第一個 ≥11 日的交易日）＋終點 2026-10-12；245 窗；窗報酬＝NAV(下一訊號日收盤)/NAV(本訊號日收盤)−1（簡單報酬）；年化起點 2006-04-11',
        'windows': C.WINDOWS, 'subperiods': C.SUBPER,
        'family': {
            'rule': '曾被當作候選策略報告過的多頭規則淨值（網路規則、組合、配置、改版、穩健性版本）；排除純基準、事後拆解、描述用序列、扣成本前與壓力成本版本、複製資料夾、本輪 o5 同時進行的資料夾、非淨值檔；完全相同序列只留一條；見 build_family.py 的 EXCL_RE／BENCH_RE',
            'loss': '扣該研究自己的一般成本後，對 0050 含息的窗超額（主）；對主策略同池等權 AB_V4|EW 的窗超額（副）',
            'coverage': '窗內缺值 ≤ max(2, 2.5%×窗長) 才算有覆蓋，缺值以超額 0 填補',
            'F_all': '所有有覆蓋成員（含 strategy_ensemble 窮舉 5,076 組重跑）', 'F_main': '主策略改版鏈與所有被拒絕的變體（資料夾與欄位規則見 build_family.is_main）',
            'F_long': '在 W_long 有覆蓋的成員（同一成員集合也用於 W_19、W_21）',
            'F_main_core_exclude_regex': '2330|TSMC|L_E50|E0050|0050_30|MIX_|TOP1VALUE|POOLEW|combo_n10|COMBO',
            'synthetic': '缺淨值成員 121 條（data/missing_members.csv）只做敏感度：報告值版、悲觀版（t 至少 2.0）',
        },
        'spa': {'B': 10000, 'block_lengths': [3, 6, 12], 'variance': 'pr', 'variance_note': '迭代 A 依事前規則選出 V1（Politis-Romano 核）',
                'seed_base': 20261028, 'methods': ['White RC', 'Hansen SPA lower/consistent/upper', 'Romano-Wolf StepM（標準化、逐步調整 p）', 'Deflated Sharpe（N＝Li-Ji、Nyholt、原始 N）'],
                'focus_p_definition': '焦點（主策略）p＝P*(max(max_k Z*_k, 0) ≥ t_main)，Z* 依 l/c/u 規則重新置中；若主策略是族群最大值，與標準 SPA p 相同'},
        'decision': {'spa_column': 'F_long|W_long|L6|x0050', 'spa_stat': 'p_SPA_c_focus',
                     'rule': 'p ≤ 0.05：維持第 2 級，註記「通過資料窺探校正」；0.05 < p ≤ 0.10：第 2 級，註記「邊緣」；p > 0.10：對外級別改為第 3 級「有理論支持，但未通過資料窺探校正」，SOP 預期超額只報區間下緣',
                     'breaks': '同池等權超額（xEW_net）或五分位價差（Q5mQ1_gross）有 5% 顯著斷點（Bai-Perron 序列法 m≥1，或事前 Chow 日期 p<0.05），且斷點後均值 ≤ 斷點前一半：SOP 加註「選股力已衰退」，2019 年後預期改用斷點後均值；只有對 0050 超額有斷點：加註「落後來自權值集中」；沒有偵測到斷點一律附 MDE'},
        'breaks': {'series': ['x50_net', 'x50_gross', 'xEW_net', 'xEW_gross', 'Q5mQ1_gross', 'Q5mQ1_net'], 'trim': 0.15, 'max_breaks': 3,
                   'variance': 'HAC（Bartlett，落後 floor(4(n/100)^(2/9))）為主；iid 並列（迭代 B 依規則選出 iid，但研究計畫指定 HAC）',
                   'nsim_null': 2000, 'seed_null': 777, 'chow_dates': {'2019-05': '2019-05-01', '2024-01': '2024-01-01'},
                   'ci': 'Bai（1997）90%：k̂ ± c·ω²/δ² ± 1，c 由模擬 argmax 分布的 90% 對稱分位', 'nsim_power': 400, 'seed_power': 4242,
                   'quintile': '池內綜合分數五分位（V4 合格池、AB 最壞情境營收、零成本與一般成本、隔日開盤成交、不做漲停遞補）'},
        'hold': {'pre': ['2006-04-01', '2023-12-31'], 'hold': ['2024-01-01', '2026-10-12'], 'block_lengths': [3, 6, 12], 'B': 10000, 'seed': 5150,
                 'stat': 'Σ log((1+r_main)/(1+r_0050))（主）；平均算術超額（副）', 'pass': '實際值落在預測分布 10～90% 內 → 與雜訊相容',
                 'tsmc_weight': 'ex_tsmc.csv 實際權重（2025-10 起）；之前用市值前 50 名近似＋平均校準差（build_tsmc.py）',
                 'decomp': 'r_main − r_0050 ＝ (r_main − r_ex) − w(r2330 − r_ex)；台積電缺口解釋比例＝log(W_0050/W_0050exT) ÷ log(W_0050/W_main)（落後 > 0 時）'},
        'crosscheck': {'cells': [['F_long', 'W_long', 6], ['F_main', 'W_long', 6], ['F_long', 'W_19', 6]], 'seed': 99},
        'predictions_from_plan': '族群 300～2,000、有效試驗數 30～150；AB_V4 前 3 名；F_long×W_long SPA_c p 0.03～0.20；RC p 高於 SPA_c；W_21 全族群 p > 0.20；F_main SPA p < 0.05；對 0050 Bai-Perron 0～1 個斷點、sup-F 高點 2019～2021、CUSUM 不越界；對等權與五分位無斷點；2024～2026 落在 10～90%；MDE ≥ 1.0 pp；台積電缺口解釋 ≥ 60%',
        'iterations_summary': {'n_versions': 'SPA 4 版、斷點 3 版、族群蒐集 4 版', 'selected_spa': it['A_SPA_variants']['selected'], 'selected_break_variance_by_rule': it['B_break_variants']['selected']},
        'hashes': hashes,
    }
    keep = {k: v for k, v in res.items() if k.startswith('PREREG_superseded')}   # 保留被取代的舊登記（含理由）
    res = {'PREREG': PREREG, 'PREREG_sha256': C.canon_sha(PREREG), **keep}
    C.RES_F.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written', PREREG['time_utc'], res['PREREG_sha256'], len(hashes), 'files hashed')

if __name__ == '__main__':
    main()
