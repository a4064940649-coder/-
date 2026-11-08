#!/usr/bin/env python3
"""第 3 步：凍結 SPEC（sha256＋UTC），之後才可以算任何判定比例。只寫 SPEC.json 與 results.json 的 PREREG。python3 -I -B s3_freeze.py"""
import sys, json, hashlib
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (HERE / 'SPEC.json').exists(), 'SPEC 已凍結，不得覆寫'
C = json.loads((HERE / 'census.json').read_text(encoding='utf-8'))
cn = C['counts']
FILES = ['pclib.py', 'avail.py', 's1_census.py', 's0_sources.py', 's2_iterate.py', 'census.json', 'explore.json', 'iterations.json',
         'data/census_days.csv', 'data/c0_slots.csv', 'data/jscan_slots_by_month_seed0.csv',
         'raw/ts_tasi_upstream_firstseen.csv.gz', 'raw/ts_mops_ann_date.csv', 'raw/ts_news_mentions.csv.gz', 'raw/tasi_commit_times.txt',
         'raw/disposal_official_universe.csv', 'raw/disposal_hist_rejudge.csv', 'raw/step0_connectivity.txt', 'raw/manifest.json', 's4_count.py', 's4b_v7surv.py']
SPEC = {
 'title': '主策略 PIT 勘誤普查（只計數，不重算報酬）：營收可得時點（D=S／D>S 前視）、2013 年後營收存活者缺口、處置期槽位（o_rev_avail_ds_pit_count，o17 題 1）',
 'no_returns': '只計數名單槽位；不計算任何報酬、淨值或 t 值；不改主策略規則；不產生選股訊號',
 'C0': 'main_v2/v2_core.V2(gapfix, YL, Y5, exfin=True, latest=False).plan(10, official)：正式 245 個訊號日 2006-04-11～2026-09-11；2006～2013 與 o_g8 lists_C0 逐日逐檔相同（s1 斷言）。2026-10-12（latest 版，D=S）只作附表，不進判定分母。',
 'denominators': {
   'C0_2006_2013': {'slots': cn['2006_2013']['slots'], 'DS_slots': cn['2006_2013']['DS_slots'], 'DS_days': cn['2006_2013']['DS_days']},
   'C0_ALL_2006_2026': {'slots': cn['ALL']['slots'], 'DS_slots': cn['ALL']['DS_slots'], 'DgtS_slots': cn['ALL']['DgtS_slots']},
   'C0_2019_2026': {'slots': cn['2019_2026']['slots'], 'DS_slots': cn['2019_2026']['DS_slots'], 'DgtS_slots': cn['2019_2026']['DgtS_slots'], 'DS_DgtS_days': cn['2019_2026']['DS_days'] + cn['2019_2026']['DgtS_days']},
   'C0_2013_2018': {'slots': cn['2013_2018']['slots']}, 'C0_2013_2026': {'slots': cn['2013_2026']['slots']},
   'C0_2014_2018': {'slots': cn['2014_2018']['slots']}, 'C0_2014_2021-09-23': {'slots': cn['2014_2021-09-23']['slots']},
   'jscan_AB_V7_seed0': C['jscan'],
   'note': 'D=S／D>S 依 v2_core.deadline_case（2020-02-11 v2_core 判 D=S，法定期限延到 2020-02-15 → legal_vs_S＝D>S；兩種標籤都在分子內）'},
 'classification': {
   'version': 'V6（iterations.json）',
   'cutoff': '訊號日 S 的 13:30（台北），D>S 日也用訊號日 S（不用期限日）',
   'AVAIL（確定可得）': 'MOPS ann_date < S；或任一快照（Tasi／上游 git）首見時刻 ≤ S 13:30；或新聞提及（數值通過嚴格有效位數閘門）日期 < S（含時刻者 ≤ S 13:30）',
   'UNDET（未定）': 'MOPS ann_date＝S（沒有時刻）；或快照首見 > S 13:30 而前一張快照日期 ≤ S（或沒有前一張）；或新聞日期＝S（無時刻）；或來源互相衝突（同時有 AVAIL 與 UNAVAIL）',
   'UNAVAIL（確定不可得）': 'MOPS ann_date > S；或快照「前一張快照的日期 > S」時該檔仍不在（Tasi 營收表落後公告日 1 天，S 當晚不在不算證據）',
   'NONE（查不到）': '以上都沒有（含：只有晚於 S 的新聞、只有上游首次匯入列）',
   'upper': '上界＝UNAVAIL＋UNDET＋NONE（槽位數）', 'lower': '下界＝UNAVAIL（槽位數）',
   'second_cutoff': 'S 24:00／隔日開盤前只作描述（SEEN_BY_24／ABSENT_ON_LATER_DAY／ANN_AFTER_S／UNKNOWN），不能推翻判定',
   'scope': '只判 D=S 與 D>S 訊號日的槽位；D<S 日的槽位期限已過，視為可得（不檢查逾期申報）。'},
 'rules_verbatim': {
   'threshold_direction': '> 0.05 觸發、≤ 0.05 不觸發',
   'R1': 'R1：以 C0 槽為分母，2006～2013 的 D=S 槽上界 > 5%，或全期 D=S＋D>S 槽上界 > 5%，G5 加「前視未排除」。2019 起子集的上界 > 5%，主策略可執行正式數字（V7X、j 中位數）旁加同一條。下界也 > 5% 時，改寫「前視已證實」，列為最優先處理。J0 另外報兩件事：占全部 j 槽的比例，以及從既有 j-scan 結果（只讀）看，J0 是不是 p10 或中位數落點的那個 j。',
   'R2': 'R2：2013 年後存活者缺口的 U_opt 上界以 2013～2026 的槽為分母，> 5% 時 G4 第三段加「存活者偏誤未排除」。另外分段報 2013～2018 與 2019～2026；2013～2018 段自身 > 5% 時，G4 第二段也加同一條。這一條是本輪總監在登記前新增的。',
   'R3': 'R3：處置期槽位 ≤ 5%，只更新 o16 第 5 節註記的數字；> 5%，G5 加「處置排除未 PIT」。',
   'census': '普查口徑：不報二項上界。報受影響的訊號日數和發行人數（有效獨立事件數），並報拿掉最大貢獻股之後的比例。'},
 'R1_operational': {
   'r1a': '2006～2013：上界(D=S 槽) ÷ 930', 'r1b': '全期：上界(D=S＋D>S 槽) ÷ 2,450', 'r1c': '2019 起：上界(D=S＋D>S 槽) ÷ 920',
   'trigger_G5': 'r1a > 0.05 或 r1b > 0.05', 'trigger_exec_note': 'r1c > 0.05',
   'proven': '對應的下界 > 0.05 → 改寫「前視已證實」（r1a／r1b 的下界 → G5；r1c 的下界 → 可執行註記）',
   'J0': 'J0 槽占全部 j 槽比例（AB_V7 seed 0，2019 起與全期）；J0 槽（seed 0，2019 起）用同一分類器的上下界÷全部 j 槽（描述）；J0 在 o_exec_engine_rewrite_v7x/out/navs_oos_R4.parquet（B10|J{j}|K5T|normal|100萬|GROW）13 個 j 的年化超額（全期、2019 起）排序位置：是否為中位數（第 7 名）或 p10 落點（線性內插的兩端 j）'},
 'survivorship': {
   'pool': '同 ER2：前 300 流動性 ∩ 52 週有值 ∩ 非處置（disposal_5y）∩ 隔日可成交 ∩ 非 FINMASK；y3 缺值',
   'cause': 'ER2 erlib 定義延伸：tw_revenue 期間（S < 2021-03-01）TRUNC > IFRS13（S ≤ 2013-12-31）／POST13（S ≥ 2014-01-01；x ≥ 2013-01、x > 最後營收月、最後營收月 ≥ 2012-12）> PRE12 > HOLE > NEWLIST > BASE_LE0；rev_5y 期間（S ≥ 2021-03-01）rev_5y 完全沒有該檔且 tw_revenue 有 → POST13_R5；兩者都沒有 → NEVER_DOM；rev_5y 有但缺月：x > 最後月 → POST13_R5、x < 第一月 → NEWLIST、其他 HOLE。FOREIGN（91xx／KY／DR／F-）不算。',
   'governing_causes': ['TRUNC', 'IFRS13', 'POST13', 'POST13_R5', 'PRE12', 'HOLE', 'NEVER_DOM'],
   'U_opt': 'ER2 定義（pclib.VU，2006～2013 重現 ER2 162 槽、槽位清單逐筆相同）：el′＝el ∪ A，52 週百分位在 el′ 重算；A 營收百分位＝1.0，其他＝el′ 內非 A 股 y3 的 average rank ÷ |el′|；同分序 A 股 y3＝+∞；照 V2.plan 選 10 檔（開盤漲停遞補、held）。放回期間＝訊號日 2013-01-01～2026-09-11。',
   'count': 'U_opt 槽位＝Σ_d |新名單 ∩ A(d)|',
   'ratios': {'R2_main': 'U_opt(2013～2026) ÷ 1,640', 'R2_seg2': 'U_opt(2013～2018) ÷ 720', 'R2_seg3': 'U_opt(2019～2026) ÷ 920（描述）'},
   'panels': '面板 A＝共用面板＋未補 tw_revenue＋rev_5y（紙上正式 C0）；面板 B＝V7 面板（o_v7_listing_exec_engine/data，含 2019～2023 下市股價）＋V4 營收（R_FIX 含 o2 補值、Y5_FIX）＋官方處置 trade 口徑＋排除 KY/DR 與 FIN2，V7 正式名單＝plan_V7.csv（s4b 斷言重現）。判定取兩面板的較大者；兩面板都報。',
   'descriptive': 'U_p90、U_carry、U_opt_ALL（所有缺值原因）只描述，不得推翻判定'},
 'disposal': {
   'list': 'raw/disposal_official_universe.csv（官方全歷史處置，普通股，start～end）；交叉核對 raw/disposal_hist_rejudge.csv',
   'signal_day': 'start ≤ S ≤ end', 'trade_day': 'start ≤ 成交日 ≤ end',
   'segments': {'designated': '2006～2013（訊號日）', 'extension': '2014-01-01～2021-09-23（C0 沒有處置排除的其餘期間）', 'after': '2021-09-24 起（C0 已用 disposal_5y 排除；只描述殘留）'},
   'R3_governing': 'max(2006～2013 訊號日、2006～2013 成交日、2014～2021-09-23 訊號日、2014～2021-09-23 成交日 的槽位數 ÷ 各段 C0 槽位數)'},
 'effective_events': '每個比例另報：受影響訊號日數、發行人數、拿掉貢獻最多槽位的那一檔之後的比例',
 'descriptive_only': ['子樣本：入選槽 vs 同日股池平均（AVAIL 率）、營收百分位 ≥ 0.9／< 0.5、D>S 對 D=S、分段四段', '第二截止 S 24:00',
                      'main_v2 已發布的分段最大貢獻前 5 名（results.json RUN.gapfix.official.EXFIN.10.normal.contrib，各段 top5 聯集）是否落在上界不可得槽位'],
 'pre_known_before_registration': {
   'disposal_2006_2013_4_slots': 'o16 可執行審查已知 4 槽（2009-05-11 2383、2009-07-13 2545、2009-11-11 1702、2010-12-13 3406）——不算證據',
   'DS_330_of_930': '33 日／330 槽（35.5%）——o16 E4 已知',
   'o2_coverage_39pct': 'o2 補值覆蓋 39%、o4 1,194 缺值股月只補 6 個——已知',
   'tasi_10th_most': 'Tasi 顯示 10 日當天公告最多（1,461 筆）——已知',
   'upper_gt5_2006_2013': '探索已知 2006～2013 D=S 股池股月的時間戳覆蓋只有 1.76%（只有 2013 年的新聞），所以 r1a 上界 > 5% 在登記前幾乎確定，不算證據',
   'upper_gt5_2019': '探索已知 2019-01～2024-05 只有新聞（13.4%）、沒有快照，所以 r1c 上界 > 5% 在登記前幾乎確定，不算證據',
   'tasi_lag_1day': '閘門已知 Tasi 營收表落後 MOPS 公告日 1 天，2024-06 起的 D=S 槽多數會落在 UNDET',
   'ER2_port_162': 'pclib.VU 2006～2013 重現 ER2 162 槽（只是程式正確性檢查）',
   'informative': '有資訊量的是：下界（UNAVAIL）大小、UNDET 占比、子樣本方向、2013 年後存活者缺口兩段比例、處置延伸段槽位數'},
 'predictions': {
   'A': '2006～2013：確定可得 60～85%、未定 15～40%、確定不可得 ≤ 2%（本題預期：因覆蓋近 0，多數會落在「查不到」，此預測將無法檢驗）；上界 > 5%、下界 ≤ 5%；入選槽 AVAIL 率 ≥ 同日池平均；營收百分位 ≥ 0.9 更明顯；< 0.5 較弱；D>S 不可得率高於 D=S',
   'B': '2019 起 D=S＋D>S 日 25～35 日（普查已知正式 35 日、含 2026-10-12 共 36 日）；2024-06 起下界 ≤ 2%；J0 約占 7.7%',
   'C': '2013～2018 U_opt > 5%（日月光、矽品、華亞科、立錡等）；2019～2026 ≤ 5%；U_carry ≤ 2%',
   'D': '2006～2013＝4 槽（已知）；2014～2021-09-23 延伸段 5～20 槽、占比 < 2%',
   'falsifiers': '入選槽 AVAIL 率比池平均低 10pp 以上 →「好消息早公告」在入選股不成立；下界本身 > 5% → 逾期申報普遍、與 MOPS 法規不符，先查資料錯誤'},
 'files_sha256': {f: sha(HERE / f) for f in FILES},
 'engine_sha256': C['engine_sha'],
 'frozen_utc': pd.Timestamp.now('UTC').isoformat(),
}
body = json.dumps(SPEC, ensure_ascii=False, indent=1, sort_keys=True)
(HERE / 'SPEC.json').write_text(body, encoding='utf-8')
h = sha(HERE / 'SPEC.json')
R = {'PREREG': {'file': 'SPEC.json', 'sha256': h, 'frozen_utc': SPEC['frozen_utc'], 'note': '凍結時未算任何 D=S 槽位可得性比例、U_opt 2013～2026 槽位、處置槽位'}}
(HERE / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
print(h, SPEC['frozen_utc'])
