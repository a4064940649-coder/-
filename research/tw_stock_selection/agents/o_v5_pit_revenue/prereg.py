#!/usr/bin/env python3
"""凍結：把 V5 規格、data_freeze_v5.json 的 sha256、必報項目、預測與判定規則寫進 results.json 的 PREREG（含 sha256 與時間戳）。
之後 run_v5.py／jscan_v5.py 斷言 PREREG 存在、未被改、data_freeze_v5 未變，才計算任何 V5 報酬。"""
import json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
it = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
SPEC = (HERE / 'frozen_spec.txt').read_text(encoding='utf-8')
P = {
 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
 'data_freeze_v5_sha256': sha(HERE / 'data_freeze_v5.json'),
 'frozen_spec_sha256': sha(HERE / 'frozen_spec.txt'),
 'frozen_spec': SPEC,
 'gates_computed_before_prereg': {'gap_gate': it['gap_gate'], 'first_value_gate': it['first_value_gate'], 'first_value_rule_selected': it['first_value_selected']},
 'known_before_prereg': {
   '非 AB 下首次值替換改變的前 10 名股月': 0, 'y3 被首次值改變的（訊號日×股）格數（非 AB）': 174,
   '缺口格 2006-04～2012-12': '490 格（80 個換股日、65 檔），其中 52 週比百分位 ≥ 0.7 者 137 格；PIT 合法代理覆蓋 0',
   '缺口股個股層級下一窗超額（描述）': '全部 −0.86pp（t −1.15）；52 週比 ≥ 0.7 者 +0.63pp（t 0.67）；< 0.7 者 −1.83pp（t −2.06）',
   '新聞與面板不一致率': '原始 1.74%（6,513 格）；有佐證者 0.55%（F3，36 格）；前 30 名本期格原始 1.30%、比值 0.75',
   'smoke': 'fv=None 的 V5 類別重現 V4 與 AB_V4 淨值逐位相同（smoke.json）；run／jscan 以 2 seed 做過程式冒煙（輸出不印不存）'},
 'decision_rule': {
   'grade_by': 'AB_V5 一般成本、跨循環對 0050 t',
   'rules': 't ≥ 2.58 第 1 級；2.30～2.58 第 1 級邊緣，但「拿掉最好 1 個月」「起點 2008-01」「壓力成本」三項中有兩項 < 2.0，或 40 seed 中超過 20 個 t < 2.30 → 第 2 級；t < 2.30 第 2 級',
   'sop_expectation': '第 2 級時 SOP 預期年化改用 V5 保守口徑：缺口閘門未過 → AB_V5_B10（＝V5 資料下的 AB＋B10）；實盤預期＝0050＋（策略−0050）×0.5',
   'data_grade': '首次值面板、缺口代理各判 A 或 A−（缺口代理閘門未過則不適用）'},
 'columns_to_report': ['V5（非 AB，一般／壓力）', 'AB_V5（判定主欄，一般／壓力）', 'AB_V5_B10（缺口取最壞＝正式保守口徑）', 'AB_V5_B05', 'ABW_V5（期限日全部未公告）',
                       '只做首次值＝AB_V5（缺口代理未建）', '只做缺口代理＝無（閘門未過；等於 AB_V4）', 'AB_V5 首次值界限：FV MC s0～s19、固定方向 neg／pos 的最壞／最好', 'AB_V5 基期用最新值（敏感度）',
                       'AB_V4 重現（fv=None）'],
 'stats_to_report': ['逐 seed：t 中位數、範圍、p10、t < 2.30 的 seed 數、拿掉最好 1 個月後 < 2.30 的 seed 數',
                     '端點：拿掉最後 1 個月；拿掉最好 1／3／5／10 個月；拿掉最好年份；拿掉 2006、2007 年；起點 2007-01、2008-01；日曆月 t',
                     '換股日分布：st_core j＝0～12（13 版，每版 AB 40 seed），t 中位數與 p10', '成本損益兩平：單邊滑價 0.10%～0.40%（0.02% 格點）t＝2.3、2.0 的滑價；手續費全額＋0.1% 滑價',
                     'NW t（3／6／12）、12 個月區塊 bootstrap 單尾 p（10,000 次）、逐年對 0050', 'Bonferroni p（m＝10／20／50；正式口徑以 o_main_spa_breaks 為準）',
                     '分段 S1 2006-04～2013、S2 2014～2020、S3 2021-03～2026-10-12、S4／HOLD 2024～2026-10-12、SEL 2006-04～2018、VAL 2019～2023',
                     '配對：AB_V5 − AB_V4（分段）、V5 − V4、AB_V5 − main_v2、AB_V5_B10 − AB_V5；附 MDE（80% 檢定力、雙尾 5%）',
                     '機制：首次值換進 vs 被換出名額、B10 缺口股換進 vs 被取代名額（AB 逐 seed，按訊號月平均後 t）＋ MDE', '分產業貢獻、最大貢獻股、首次值換股名額的電子／非電子分布'],
 'predictions_from_director': {
   'P1': '不一致率 0.5～1.0%；前 30 名不一致率 ≥ 全體 1.5 倍（< 1.2 即否定「更正集中在極端成長股」）；因更正改變前 10 名的股月 ≤ 15；AB_V5−AB_V4 配對差在 −0.05～0pp／月，|Δ年化| ≤ 0.3pp',
   'P2': '缺口代理（閘門通過時）：入選前 10 缺口股月 10～40，對同月其他入選股 −1～−3pp；S1 年化下修 0.3～1.5pp、S1 t 降到 1.7～2.0 —— 閘門未過，不檢定',
   'P3': 'AB_V5 一般成本 25.5～26.8%、t 2.25～2.48；> AB_V4 要逐格核對；< AB＋B10 先查資料錯誤',
   'P4': '偏誤集中在 S1，S3／S4 ≈ 0；非電子偏誤 > 電子',
   'P5': '起點 2008-01 時 AB_V5 t < 2.0'},
 'versions_tried_count_for_multiple_testing': {'data_rule_versions': it['n_versions_total'], 'strategy_spec_versions': 1,
                                                'main_strategy_prior_revisions': 'main_refine、main_v2、gapfix、integrate_o1／o2、o3、o4（≥ 7 輪）→ 報 m＝10／20／50'},
 'replication': '複製者不讀研究員 .py，只讀 data_freeze_v5.json 所列檔案與 frozen_spec；V5|normal、AB_V5|normal 淨值逐日最大相對差 < 1e-9；逐 seed t 與 j 分布要重現；落差逐項解釋，否則不得發布',
}
canon = json.dumps(P, ensure_ascii=False, sort_keys=True).encode()
f = HERE / 'results.json'
assert not f.exists() or 'PREREG' not in json.loads(f.read_text(encoding='utf-8')), 'PREREG 已存在，不得覆寫'
f.write_text(json.dumps({'PREREG': P, 'PREREG_sha256': hashlib.sha256(canon).hexdigest()}, ensure_ascii=False, indent=1), encoding='utf-8')
print(hashlib.sha256(canon).hexdigest(), P['created_utc'])
