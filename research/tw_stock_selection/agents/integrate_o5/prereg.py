#!/usr/bin/env python3
"""integrate_o5 事前登記：在任何整合回測之前，把組合、比例、選版規則、判定規則、雜湊範圍寫進 results.json（含 sha256 與時間）。只能執行一次。"""
import json, hashlib
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent
RJ = HERE / 'results.json'
assert not RJ.exists(), 'results.json 已存在：PREREG 只能寫一次'
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
J1R = json.loads((AG / 'o_rev_event_trigger/results.json').read_text(encoding='utf-8'))
SPR = json.loads((AG / 'o_main_spa_breaks/results.json').read_text(encoding='utf-8'))
assert J1R['PREREG_sha256'] == '208462cf804a8d1ce2dd8e524334adb4a4c086a12b95779cdc23f9914f54880c'
assert SPR['PREREG_sha256'] == 'ab57019a7397f5e47c751792b81de5eb90a2d293361f347a49edd4d55e61d75f'

# 雜湊範圍：本資料夾全部 .py 與 data/、raw/；兩題的 results.json 與引擎；J+1 研究 PREREG 登記的全部檔案（引擎庫、V4 凍結檔、面板與輸入）；SPA 引擎與族群資料。
files = sorted(str(p) for p in HERE.glob('*.py')) + sorted(str(p) for p in (HERE / 'data').glob('*')) + [str(HERE / 'raw/access_log.json')]
files += [str(AG / 'o_rev_event_trigger/results.json'), str(AG / 'o_rev_event_trigger/navs_oos.parquet'),
          str(AG / 'o_main_spa_breaks/results.json'), str(AG / 'o_main_spa_breaks/spa_lib.py'), str(AG / 'o_main_spa_breaks/fam_common.py'),
          str(AG / 'o_main_spa_breaks/data/family_excess.parquet'), str(AG / 'o_main_spa_breaks/data/family_members.csv'),
          str(AG / 'integrate_o4/navs.parquet'), str(AG / 'o_v4_revenue_patch_official/navs_v4.parquet')]
files += list(J1R['PREREG']['sha256'].keys())
files = sorted(set(files))
HASH = {f: sha(f) for f in files}

PREREG = {
    'study': 'integrate_o5：把 o5 通過審查（待觀察）的兩題——o_rev_event_trigger（J+1 固定換股日）與 o_main_spa_breaks（主策略真實顯著性，方法題）——和主策略（AB_V4，排除金融 n10）整合',
    'context': 'o5 另外兩題（o_v5_pit_revenue、o_exec_capacity_tiers）不在通過名單，不納入。'
               'J+1 不是新選股訊號，而是同一策略的換股日定義：訊號日＝營收法定期限 D_m 之後第 1 個交易日（245 個錨定月中 155 個月與官方日相同，其餘多半晚 1 個交易日）。'
               '它沒有個股層級的分數，所以「濾網」「加分項」兩種形式不適用；可執行的整合只有「取代」與「分資金兩帳戶按固定比例並行」（每帳戶各自在自己的日子換股）。'
               'o_main_spa_breaks 是方法題，不產生淨值；它的整合方式是：(a) 用同一族群（F_long × W_long）、同一檢定（Hansen SPA_c，以候選自己的檢定量計算的焦點 p）、同一種子，'
               '替每個整合候選算資料窺探校正後的 p，套用它事前登記的級別規則；(b) 它的結論「2024～2026 落後來自權值集中」對應 SOP 既有的 0050 30%、2330 30% 配置選項，'
               '本研究在相同框架下重算這兩個參考組合（數字應與 integrate_o4 完全相同），只報告，不重新判定。',
    'base': {'AB_V4': 'o_v4_revenue_patch_official 的 AB_V4|normal（官方日、只用已公告營收、40 seeds 日報酬平均）＝主策略單獨，判定與選版基準。本研究重跑 40 seeds，斷言與 navs_v4.parquet 逐日相對差 ≤ 1e-9。'},
    'candidates': {
        'R_J1': '取代：主策略改用 J+1 換股日（o_rev_event_trigger 的選定版，fx_core.version_days(1)、day 模式 AB、seed 0～39）。本研究重跑 40 seeds，斷言平均淨值與 navs_oos.parquet 的 J+1|AB、J+1|AB_stress、J+1|EW 逐日相對差 ≤ 1e-9。',
        'M_J1_20': '分資金並行：AB_V4 帳戶 80%＋J+1 帳戶 20%。兩帳戶各自照自己的日子換股、各自扣成本；每個 AB_V4 成交日 e 收盤調回 80/20，移轉金額扣股票來回成本（賣方含證交稅 0.3%＋買方）。',
        'M_J1_50': '分資金並行：AB_V4 50%＋J+1 50%，同上。',
    },
    'references_not_judged': {
        'REF_E50_30': 'AB_V4 70%＋0050 30%（SOP 既有「追蹤誤差預算」選項）；斷言與 integrate_o4 的 L_E50_30 normal／stress 逐日相對差 ≤ 1e-9。',
        'REF_2330_30': 'AB_V4 70%＋2330 30%（SOP 事後觀察欄）；斷言與 integrate_o4 的 L_2330_30 逐日相對差 ≤ 1e-9。',
        'why': 'o_main_spa_breaks 指出落後 0050 的主因是台積電權重（缺口 log 0.24 對落後 log 0.044）。這兩個選項 o4 已判定，本研究只在同一表格與同一 SPA 族群下重列，並報它們的 SPA 焦點 p。',
    },
    'not_applicable': {
        'J+1 當濾網或加分項': 'J+1 只是換股日，沒有個股分數或名單條件可以疊在主策略上。',
        'J+1 與 AB_V4 混合 80/20 以外的比例（例如 70/30）': '事前只固定 80/20 與 50/50 兩種；不另挑比例。',
        'SPA 當濾網': 'SPA 是對整條策略的檢定，不是選股條件。',
        '斷點檢定結果當擇時開關': 'o_main_spa_breaks 沒有偵測到衰退斷點，沒有可規則化的切換日。',
        '容量': 'o_exec_capacity_tiers 沒有通過審查；本研究不做容量數字。只註記：兩帳戶並行會把開盤集合競價需求分到兩天（88 個月），屬未驗證假說，不寫進 SOP。',
    },
    'ratios_fixed': '80/20、50/50 事前固定，不在任何期間挑；HOLD（2024～2026）不參與任何選擇。',
    'cost': {'normal': '買＝手續費 0.1425%×0.6＋滑價 0.1%；賣＝同＋證交稅 0.3%（v4lib.cost_rates("normal")）',
             'stress': '每邊 0.5%（兩條腿都用 stress 淨值；袖子移轉用同費率）',
             'breakeven': '單邊滑價格點 0.10%～0.40%（每 0.02%）：AB_V4 與 J+1 兩條腿都用同一滑價重跑 40 seeds，袖子移轉也用同滑價；報 ALL 對 0050 t 跌到 2.30 與 2.00 的滑價（線性內插）。'},
    'spec_fixed': {
        'return_type': '一律簡單報酬。',
        'annualization': '全期年化從 2006-04-11 收盤起算，到 2026-10-12 收盤（AB_V4 官方評價終點；J+1 腿截到同一天）。分段年化從段內第一個面板日的前一日收盤起算（v4lib.seg → drj_core.cagr_mdd）；HOLD 因此從 2023-12-29 收盤起算（＝SOP 口徑）。',
        'windows': 't 用官方訊號窗（AB_V4 的訊號日收盤 → 下一訊號日收盤，v4lib.win_ret），窗依起點歸段；所有候選都用同一組窗，才能配對。另報日曆月 t（完整月 2006-05～2026-09，fx_core.cal_ex）。',
        'periods': 'v4lib.PER：S1 2006-04～2013、S2 2014～2020、S3 2021-03～2026-10-12、SEL 2006-04～2018、VAL 2019～2023、HOLD 2024～2026-10-12、ALL 2006-04～2026-10-12。',
        'paired': '和 AB_V4 的配對差主指標用日曆月（同 o_rev_event_trigger 的版本比較規則），訊號窗配對差為次要；都附 MDE（80% 檢定力、雙尾 5%：2.8×sd/√n）與拿掉絕對值最大月後的 t。',
        'corr': '月報酬相關（訊號窗與日曆月）與對 0050 月超額的相關。',
        'mix_official': '組合的正式淨值＝兩條腿的 40-seed 平均淨值按袖子法組合；逐 seed 分布用 seed s 的 AB_V4 腿配 seed s 的 J+1 腿。',
        'EW': '對同池等權 t 一律用 AB_V4|EW（同池等權零成本）。',
    },
    'metrics': ['各期年化、最大回撤（日頻）、對 0050 t（訊號窗與日曆月）、月均超額、追蹤誤差、對同池等權 t', '和 AB_V4 的配對差（日曆月主、訊號窗次）＋MDE', '相關係數',
                '壓力成本年化與 t', '端點：拿掉最後 1 窗、最好 1／3／5／10 窗、最好年份、第一年（2006）、起點 2007、2008、只到 2025 年底、日曆月 t 與拿掉最好 1 個日曆月',
                'AB 逐 seed 分布：t 中位數、p10、最小、最大、跌破 2.3／2.0 的 seed 數、日曆月 t 中位數、HOLD t 中位數', '成本損益兩平滑價', '逐年與輸 0050 年數',
                '實盤預期＝0050 年化＋0.5×（回測年化 − 0050 年化）；下緣＝實盤預期 − 0.5×（AB_V4 − AB_V4_B10 年化差，即 V4 研究的保守口徑折扣）',
                'SPA：F_long × W_long，區塊長度 3／6／12，B 10,000，種子＝20261028＋L（與 o_main_spa_breaks 相同）；族群＝原 2,319 條＋3 個新候選；'
                '報各候選的 SPA_c 焦點 p（判定欄 L6）、l／u 焦點 p、RC 焦點 p、StepM 焦點 p、單一檢定 p、族群排名。先斷言原族群下 AB_V4 的焦點 p 與 o_main_spa_breaks 完全相同。'],
    'selection_rule': '只用 SEL：在 R_J1、M_J1_20、M_J1_50 中，選 SEL 對 0050 t 高於 AB_V4、而且 SEL 年化比 AB_V4 少 ≤ 1pp 者裡 t 最高的一個為「建議整合版」；都不符合就維持 AB_V4。HOLD 不參與任何選擇。',
    'judgement_rule': {
        '可用': '(A) 報酬改善：ALL 日曆月配對 t ≥ 1.0，而且 SEL 與 VAL 的日曆月配對平均都 > 0；或 (B) 對 0050 t 改善：ALL t 高 ≥ 0.2、SEL 與 VAL 的 t 都較高、ALL 年化少 ≤ 1pp、ALL 最大回撤不更深。',
        '待觀察': '不符合可用，但 ALL 日曆月配對 t > −1、ALL 年化少 ≤ 2pp，而且 SEL 或 VAL 至少一段的對 0050 t 高於 AB_V4。',
        '不可用': '其餘。',
        'cap': '上限待觀察：主策略已改版 ≥ 10 次、2019 年後資料已看過、沒有盲測期；任何升級只能用 2026-11 起的前瞻資料。',
        'SPA_grade': '每個候選另依 o_main_spa_breaks 的事前規則給對外級別：L6 SPA_c 焦點 p ≤ 0.05 第 2 級（通過校正）；0.05～0.10 第 2 級（邊緣）；> 0.10 第 3 級（有理論支持，但未通過資料窺探校正），SOP 預期超額只報下緣。',
        'J+1_definition': 'J+1 研究判定「可用（只改定義）」：它證明 J+1 沒有前視、可以當正式日；但是否取代官方日只看本研究的選版與判定。不取代時，SOP 換股日不變，J+1 研究的「實盤預期下修到約 20.0%」只適用於改用 J+1 的情形。',
    },
    'honesty_notes': ['AB_V4 與 J+1 的全期、分段數字都已在兩份研究公開；兩條腿的月相關預期極高（155 個月同一天換股），所以混合的結果大致可以從已公開數字推得。本研究沒有新的樣本外證據，只是用事前固定的組合方式與判定規則機械處理。',
                      'PREREG 寫入前，我用 2 seeds、B=200 的 --smoke 模式跑過一次 run.py 只測管線（結果寫到 scratchpad，沒有印出或檢視任何績效數字）。',
                      'o_rev_event_trigger 與 o_main_spa_breaks 都尚待獨立複製（兩題的 replicate_o5 資料夾存在，但本研究不讀其結論）。'],
    'validation': ['AB_V4 normal／stress／EW 重跑與 navs_v4.parquet 逐日相對差 ≤ 1e-9。', 'J+1 normal／stress／EW 重跑與 navs_oos.parquet 逐日相對差 ≤ 1e-9。',
                   'REF_E50_30、REF_2330_30 與 integrate_o4 逐日相對差 ≤ 1e-9。', '原族群 F_long|W_long|L3／L6／L12 的 AB_V4 SPA_c 焦點 p 與排名與 o_main_spa_breaks 完全相同。',
                   'data/ 下的副本與原檔 sha256 相同。'],
    'files': {'prereg.py': '本檔', 'run.py': '重跑、組合、統計、SPA、選版、判定，寫回 RESULTS（只寫一次）', 'data/': '直接讀取的淨值與族群檔副本', 'raw/access_log.json': '沒有外部抓取'},
    'run': 'python3 -I -B prereg.py && python3 -I -B run.py',
    'inputs': {'o_rev_event_trigger_PREREG_sha256': J1R['PREREG_sha256'], 'o_main_spa_breaks_PREREG_sha256': SPR['PREREG_sha256']},
    'sha256': HASH,
}
blob = json.dumps(PREREG, ensure_ascii=False, sort_keys=True).encode()
R = {'PREREG': PREREG, 'PREREG_sha256': hashlib.sha256(blob).hexdigest(), 'PREREG_written_at': pd.Timestamp.now(tz='UTC').isoformat()}
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
print(R['PREREG_sha256'], R['PREREG_written_at'], len(HASH), 'files hashed')
