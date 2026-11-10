#!/usr/bin/env python3
"""integrate_o19 事前登記（寫入 results.json 的 PREREG＋sha256；之後所有 run_* 斷言雜湊）。只跑一次。
python3 -I -B prereg.py"""
import json, hashlib, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; AG = HERE.parent
RJ = HERE / 'results.json'
assert not RJ.exists(), '只跑一次'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
FILES = ['integrate_o19/mk_pseudo.py', 'integrate_o19/pseudo.json', 'integrate_o19/data/DU_I3a.csv', 'integrate_o19/data/DU_I3b.csv', 'integrate_o19/data/DU_I3c.csv',
         'o_disposal_list_pit_audit_pkl18/data/DISP_OFFICIAL_v2.csv', 'o_disposal_list_pit_audit_pkl18/data/DISP_OFFICIAL_v2_lenient.csv',
         'o_disposal_list_pit_audit_pkl18/data/DISP_OFFICIAL_v2_posthoc.csv', 'o_disposal_list_pit_audit_pkl18/results.json',
         'o_rev_avail_ds_pit_count/raw/disposal_official_universe.csv',
         'o_delist_rev_backfill_allpit/o18lib.py', 'o_delist_rev_backfill_allpit/v7x_lib.py', 'o_delist_rev_backfill_allpit/results.json',
         'o_engine_delist_consideration/cz_v7x.py', 'o_engine_delist_consideration/cz_engine.py', 'o_engine_delist_consideration/events_v2.csv',
         'o_exec_engine_rewrite_v7x/x_stats.py', 'o_exec_engine_rewrite_v7x/x_exec.py', 'o_exec_engine_rewrite_v7x/results.json']
PREREG = dict(
    written_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    topic='o_disposal_list_pit_audit_pkl18（官方處置清單完整性與公告時點 PIT 核對＋18 組 pkl 追查；待觀察）',
    nature='題目是資料勘誤：不產生選股訊號、沒有分數、沒有獨立報酬序列。能整合的只有「處置濾網換成 DISP_OFFICIAL_v2」與「2019 不完整的影響界限」。G3 族群 N 維持 2,573（本整合不新增候選策略）。',
    main_reference=dict(
        paper='o18 正式：ALLPIT_REAL_{opt,p90,carry} × cons_main（o18lib.build／sim），一般與壓力成本；另列 C0（舊主策略，freeze 引擎）為對照',
        v7x='V7X 正式：v7x_lib.make REAL_{opt,p90,carry} × 13 j × 20 seeds（B1）× cz_v7x.CSim × K5T × GROW × cons_main，R4＝orig_cfg＋f1,f2,f3=0.5,f4；逐格取 opt／p90／carry 最差',
        disposal_mask='現行＝ia_core disposal_5y ∪ o_rev_avail_ds_pit_count/raw/disposal_official_universe.csv（≤ 2021-09-23，訊號日或成交日）'),
    combos={
        'I1_filter_strict': '濾網：官方清單換成 DISP_OFFICIAL_v2 嚴格版（＝寬鬆版，sha 8f97b304…）。斷言覆蓋集合與名單 sha 和現行相同；數字照跑（紙上全變體、V7X 全格）。主欄。',
        'I2_filter_posthoc': '濾網並列：DISP_OFFICIAL_v2_posthoc（＋2344 2020-12-10～12-23，搜尋後新增、範圍外、單一來源）。只並列，不進主欄。',
        'I3_bound_2019': '敏感度（2019 疑似不完整的影響界限）：DISP_OFFICIAL_v2 ＋ 面板代理未比對事件當擬處置（觸發日下一交易日起 10 個交易日）。'
                         'I3a＝s7b 強觸發未比對 2019（9 起，合理界）；I3b＝K_reg 代理未比對 2019 全部（143 起，極端上界，代理偽陽性率高）；I3c＝2006＋2008＋2019 全部（353 起，極端上界）。'
                         '這是搜尋後新增的分類狀態，和嚴格版並列、以嚴格版為主欄；不能用來取代正式數字。',
        'I4_score_bonus': '不適用：題目沒有分數或排序變數。',
        'I5_second_leg_blend': '固定比例 80/20、50/50（主策略／第二條腿），每個訊號日窗（紙上）或每個 j 的訊號日（V7X）再平衡，比例事前固定、不用任何期間挑選。'
                               '第二條腿：(i) I1 串流（題目的「策略」＝主策略換嚴格清單）；(ii) I3a 串流（唯一和主策略不同的合理變體）。預期 (i) 結構上等於主策略、相關 1。'
                               '和 0050 混合在「不再開」清單，不做。'},
    cells=dict(
        paper=['一般／壓力 × 全期、2006～2012、2013～2018、2019 起、2024 起、挑選期 ～2019-04、驗證期 2019-05 起：年化、對 0050 超額、對 0050 t、最大回撤（窗頻；全期另附日頻）、和主策略（同放回版）月差 pp 與 t、年化差、相關、拿掉最好 3 年 t、逐年差'],
        v7x=['100 萬一般（全期超額 j 中位數、j p10、t、2019 起年化／超額、三段超額、2024 起、最大回撤）', '100 萬壓力', '300 萬一般', '1,000 萬一般',
             '30 萬一般（V7X 標準：2020-10-26 前盤後零股 f3 上限）', '30 萬一般整股限制（2020-10-26 前不允許零股）', '30 萬壓力',
             '每格另報：和主策略（同放回版、同 j）訊號日窗月差 pp 與 t、窗報酬相關（13 j 中位數）、日報酬相關'],
        shortcut='名單與輸入完全相同的 (變體, j, seed) 直接沿用基準淨值（引擎是決定性的）；基準 780 組 × 7 格全跑；其他變體只跑名單有變動的組。'),
    decision={
        'D1': 'I1：名單 sha 全同且全部格 |Δ| = 0 → 「整合＝主策略本身，正式數字不變」；任一格 |Δ年化| > 0.3pp 或 |Δt| > 0.05 → 取代（與審查題規則相同；預期不可能，因為覆蓋集合相同）。',
        'D2': 'I3a（合理界）：任一正式格（紙上全期年化／t；V7X 100 萬一般全期超額、t、j p10、2019 起年化與超額、壓力、300 萬、1,000 萬）逐格取 opt／p90／carry 最差後 |Δ| > 0.3pp 或 |Δt| > 0.05 → 正式數字旁加註「2019 處置清單不完整的可能影響 Δ」（不取代：非嚴格來源）；否則加註「2019 不完整的合理影響 ≤ 0.3pp」。I3b／I3c 只報為極端上界。',
        'D3': 'I5：混合對主策略的窗報酬相關 ≥ 0.99 或混合超額落在兩腿之間 → 判「不構成第二條腿（自混）」，不推薦任何比例。',
        'D4': '本整合不改變題目判定（待觀察）與主策略 G1～G8 判定；G6 狀態於本整合結束時重新核對（引擎獨立重寫是否已有結果）。'},
    predictions={
        'P1': 'I1 所有格 Δ = 0（確定，結構性）',
        'P2': 'I2 紙上 Δ = 0（審查題已知 0 槽變動）；V7X 全格 |Δ| ≤ 0.1pp（審查題已報 0；本整合重跑核對）',
        'P3': 'I3a 紙上全期 |Δ年化| ≤ 0.3pp（機率約 75%），V7X 100 萬一般全期 |Δ| ≤ 0.3pp（約 70%）；方向不確定',
        'P4': 'I3b 紙上全期 |Δ年化| ≤ 1.5pp（約 70%）；2019 年單年差絕對值 ≥ I3a',
        'P5': 'I5(i) 相關 = 1、月差 0；I5(ii) 相關 ≥ 0.99',
        'P6': 'G1～G8 全部不過（主策略不可投資）'},
    o18_lessons=['不利端逐格取最差（opt／p90／carry；嚴格／寬鬆）', '凍結前不得計算正式口徑檢定統計量（本 PREREG 前只做擬處置列建構，未跑任何報酬）',
                 '搜尋後新增分類狀態（posthoc、代理擬處置）與嚴格版並列、嚴格版為主欄', 'raw/ 只放未修改原始檔（本整合不新增原始資料，故無 raw/）',
                 '區間同時報假設區間（放回 3 種 × 清單版本）與抽樣區間（±1.96 SE）', '取代正式數字須附歸因拆解（若 D1 觸發）', '紙上數字附成本兩平點（沿用審查題 s10 結果，名單相同）',
                 '可執行測到 30 萬（含整股限制）', 'G6 狀態在複製完成後重新核對'],
    readonly=dict(dirs=['live_tracking/main_forward', 'o_forward_infra_live', 'o_g8p2a_protocol_addendum'], rule='前後 sha256 快照比對；不呼叫 G8P-2a 判定器'),
    file_sha256={f: sha(AG / f) for f in FILES})
s = hashlib.sha256(json.dumps(PREREG, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
RJ.write_text(json.dumps(dict(PREREG=PREREG, PREREG_sha256=s), ensure_ascii=False, indent=1), encoding='utf-8')
print(PREREG['written_utc'], s)
