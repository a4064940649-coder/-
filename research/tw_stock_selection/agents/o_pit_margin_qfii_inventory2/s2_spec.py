#!/usr/bin/env python3
"""第 2 步：閘門規格凍結（任何閘門計算之前）。寫 GATE_SPEC.json，並把 sha256＋時間戳寫入 data/gate_spec_lock.json。"""
import json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
spec = {
 'title': 'PIT 新資料盤點第 2 批：MI_MARGN、維持率相關欄位、MI_QFIIS（只盤點不回測）',
 'frozen_before_any_gate_computation': True,
 'segments': {'S1': ['2006-01-02', '2012-12-31'], 'S2': ['2013-01-02', '2018-12-28'], 'S3': ['2019-01-02', '2026-10-31']},
 'datasets': {
  'MARGN': {'core_field': '個股融資今日餘額（張），每個交易日', 'aux_fields_reported': ['融資買進', '融資賣出', '現金償還', '融資限額', '融券餘額'],
            'denominator': 'PIT 上市普通股（4 碼、非 0 開頭）當日在市；扣除上市首日起 130 個交易日（新上市股不得信用交易約 6 個月；事前固定，另報不扣除版本）',
            'note': '融資維持率不是 MI_MARGN 欄位，不在此計'},
  'MAINT': {'core_field': '官方個股融資維持率；若無，審原始相關欄位：個股融資限額（使用率分母）與市場融資金額',
            'rule': '只審原始欄位；任何以價格／FIFO 推估的維持率標「非官方推估」，不計入覆蓋率'},
  'QFIIS': {'core_field': '外資及陸資持股比率（%），並審發行股數、外資持有股數', 'denominator': 'PIT 上市普通股當日在市（不扣新上市期）',
            'secondary_monthly_variant': '另報「月頻形成日」變體（每月一個資料日，M3 只需月頻）：分母＝該月形成日的 PIT 上市普通股；只描述，不改主判定（主判定＝日頻）'}},
 'denominator_sources': {'2006-01-02~2013-12-31': 'o_v7_listing_exec_engine/data/listed_twse.parquet（PIT 上市布林表，含已下市）',
                         '2014-01-02~2026-10-31': '封存本身之官方 MI_INDEX 日檔（joeycchen72，按日期抓全市場，含當日在市之後來下市股）；與 listed_twse 交叉並報差異',
                         'pool300': '流動性前 300 池：分母股票中，以 t−1 之 60 日均成交值排名前 300（2006～2013 用面板 value.parquet；2014 起用 MI_INDEX value）'},
 'gates': {
  'G_PIT': {'metric': '有來源給定資料日（檔名或內容日期）的紀錄占比', 'threshold': 0.98, 'availability_rule': '資料日 T 收盤後公布，T+1 開盤起可用',
            'commit_time_report': '前瞻爬取 repo 另報 commit 時間 ≤ T+1 09:00（台北）比例；回溯爬取者報 0 並註明「回溯抓官方按日查詢」',
            'backfill_rule': '覆蓋只能往回看：任何 T 的值只能來自資料日 ≤ T 的紀錄或以 ≤ T 錨點向前累加；禁止用 T 之後的快照倒推 T（o13 教訓 4）'},
  'G_SURV': {'metric': '分母含當日在市、後來下市者；覆蓋率逐年報，S1/S2/S3 分段報；下市股（最後在市日 < 2026-10-01）另報',
             'pass': '下市股段覆蓋率與全體同一門檻（≥ 0.95），否則判存活者偏誤不過'},
  'G_COV': {'metric': '股日覆蓋率＝分母股日中核心欄位非缺值的比例', 'threshold': 0.95, 'universes': ['全市場', '流動性前 300 池'], 'must_pass_each_segment': True},
  'CONSIST': {'identity': {'threshold': 0.999, 'definition': '今日餘額＝前日餘額＋買進−賣出−現金償還（±資券互抵）；欄位不全時改用可檢部分：(a) 有前日餘額欄者檢 前日餘額(t)＝今日餘額(t−1)；(b) 只有買進、賣出者檢 Δ餘額 ≤ 買進−賣出（現償 ≥ 0）；(c) 只有增減者檢與獨立來源 Δ 一致。全部不可檢＝不合格'},
              'local_overlap': {'threshold': 0.99, 'definition': '與本機 BlackBoxOP MI_MARGN（2024-01 起）、margin_hist（2025-09 起）、voidful MI_QFIIS（2025-10 起）逐格完全一致率；無重疊時改報與任一獨立來源重疊期一致率，並註明'},
              'range': '餘額 ≥ 0；比率 ∈ [0,100]；外資股數 ≤ 發行股數', 'cross_source': '兩個以上來源重疊時報來源間一致率（完全相等；容差 0）'},
 },
 'verdict_rule': {'通過': 'G_PIT、G_SURV、G_COV 在 S1、S2、S3 全部合格，且 CONSIST 合格', '部分': '列出哪些段合格；部分不得拿來回測不合格段', '不通過': '其餘',
                  'forbidden_wording': '不得寫「近似 PIT」'},
 'build_iterations': {'max_versions': 3, 'scope': '只在資料層：欄位對齊、代碼正規化、重複列、民國／西元、來源合併與 PIT 累加規則',
                      'planned': {'V1': '只用原始餘額（TEJ 存活股逐日、BlackBoxOP、margin_hist、月形成日官方快照只在當日）',
                                  'V2': 'V1＋FinBrain 日增減向前累加：錨點＝同股資料日 ≤ T 的最近原始餘額（TEJ 或官方月快照）；上市首日 ≥ 2007-04-23 者以 0 起算；不准向後倒推',
                                  'V3': 'V2＋每個官方月快照日重設錨點（只向前），並把連續 ≤ 2 個交易日的單點缺值以前值延續（只向前）'},
                      'selection': '取各閘門（三段 × 三閘門＋一致性）通過數最多的版本；同分取規則最簡單者（V1 < V2 < V3）',
                      'freeze': '選版後不再改程式；修 bug 記 AMEND 與 sha256，由複製員核對'},
 'replication_tolerance': '閘門數字小數點後 1 位（百分比）一致',
 'prereg_draft_trigger': '只有某資料「通過」才寫 PREREG_DRAFT.json；部分或不通過只存 raw/ 與 data/gate.json',
 'disk': '開工前剩餘 ≥ 1.5GB；clone 只在 scratchpad；題目結束刪除中間檔與複製員 cache 並報刪除量',
}
p = HERE / 'GATE_SPEC.json'
json.dump(spec, open(p, 'w'), ensure_ascii=False, indent=1)
h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
lock = {'file': 'GATE_SPEC.json', 'sha256': h, 'frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
json.dump(lock, open(HERE / 'data' / 'gate_spec_lock.json', 'w'), indent=1)
print(lock)
