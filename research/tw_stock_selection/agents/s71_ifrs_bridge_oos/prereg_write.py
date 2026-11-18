#!/usr/bin/env python3
"""S71 事前登記：寫 SPEC.json、PREREG.json 與 sha256、code.sha256（在跑任何候選規則的回測之前）。
主規則依 step0.json 的資料特徵機械決定（不看報酬）。python3 -I -B prereg_write.py"""
import json, hashlib, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
assert not (HERE / 'results.json').exists()
s0 = json.load(open(HERE / 'step0.json')); s0b = json.load(open(HERE / 'step0b.json')); s0c = json.load(open(HERE / 'step0c.json'))
acc = json.load(open(HERE / 'accept.json')); prep = json.load(open(HERE / 'prep_check.json'))
di = s0['decision_inputs']
b1_primary = bool(abs(di['median_gap_pp']) < 5 and di['share_gt50_ratio'] < 1.5)
primary = 'B1' if b1_primary else 'B2'

SPEC = {
    'id': 'S71', 'title': '跨 IFRS 斷點的 SURGE 規則，把 S55／R55 的樣本外延伸到 2013-02～2015-06（29 個月）',
    'question': '凍結現行規則在 2013-02～2015-06 沒有訊號（2013 年營收年增量被設成缺值）。事前定一條跨斷點算 SURGE 的規則後，這 29 個月對 0050 有沒有正超額？',
    'dedup': '2026-11-18 grep BACKLOG.md 與 agents/ 的「IFRS、斷點、卡住期、2013」：所有既有研究都是遮罩或剔除 2013（s17lib／s2lib／stlib／s20lib 同一行；'
             'w3_revenue_sue、main_refine、industry_revenue_long 另報剔除 IFRS 段）；revenue_longhistory 描述過斷點（2013 年增率中位數 25%、台泥例）；'
             'S55 FINDINGS 建議「訂 IFRS 斷點期規則需另題事前登記」。沒有人做過跨斷點規則。',
    'frozen_current': {
        'engine': 's47lib.sim（→ s10lib.simulate_s10：W10_70、E0、嚴成本 strict、100 萬）', 'universe': 'stlib.universe(twse) & ~創新板（A10 門檻在 make_signal_s10 內，原樣）',
        'signal': 'F1＝s10lib.make_signal_s10(shift=0)', 'switches': 'cslib.EXRIGHT_FIX＝DEADLINE_FIX＝INS15_MASK＝True（正式開關）；s17lib.e50_fix_suspension',
        'surge_std': 'D＝月營收−12 個月前；SURGE＝(D−前 24 個月 D 平均)÷前 24 個月 D 標準差（24 個都要有值、σ>0）；s17lib.zscore 原函式',
        'panel': '自建 2007-01-02～2015-06-30 上市面板，R55 主做法（agents/r55_replicate_s55/build_panel_r55.py 逐行沿用，只改暫存路徑）；stlib.START＝2008-01-02、訊號月份 ≥ 2008-01',
        'not_changed': '共用引擎檔一律不改；只在 s71lib.rev_rule 換掉「2013 年 D 設缺值」這一步'},
    'candidate_rules': {
        'B1': '不遮罩：2013 年的 D 照算（營收檔原值）。',
        'B2': {
            'name': '水準校正',
            'formula': [
                'j_n(y)＝ln( y 年前 n 個月營收合計 ÷ y−1 年最後 n 個月營收合計 )，n 個月都要有值；',
                'a_n＝j_n(2013) − 中位數{ j_n(y)，y＝2008…2012 }（至少 3 年有值，否則 a_n 缺值）；',
                'k_n＝max(1, exp(a_n))（合併營收不小於個體營收；不設其他不校正帶）；',
                '校正：該公司 2012-12（含）以前所有月營收 × k_n，再照標準定義算 D 與 SURGE；k_n 缺值的公司 2013 年 D 缺值（等同現行遮罩）；',
                '時點（只用當時已公布的營收）：營收月 2013-01 的 SURGE 用 n＝1、2013-02 用 n＝2、2013-03 起一律 n＝3；營收月 ≤ 2012-12 的列照現行（整段同乘 k 不改變 z 值）。'],
            'design_from_data_only': '帶寬與 n 的選擇只用月營收資料（step0b.json、step0c.json），準則是「校正後 2013 年未參與估計的月份裡，整年持續偏高／偏低的公司比例最接近 2012、2014」；'
                                     '沒有載入價格、沒有算任何報酬。單月 n＝1 的版本校正過頭（校正後年增率中位數 −9.9%，因 2013-01 無春節、單月雜訊大），累計 3 個月為 −0.6%。'}},
    'max_candidates': 2, 'run_once': True,
    'windows': {'ext': '2013-02～2015-06（29 個月；判定用）', 'merged': '2008-01～2015-06（90 個月，連續一條淨值；報超額、資訊比率、t）',
                'old_primary': '2008-01～2013-01（61 個月；應與驗收 A2 逐月相同，因為規則只影響營收月 ≥ 2013-01）'},
    'stats': 'r55lib：月底＝每個日曆月最後交易日；月超額＝策略月報酬−0050 含息月報酬；年化超額＝月平均×12（個百分點）；資訊比率＝月平均÷月標準差×√12；Newey-West t（Bartlett，滯後 6）',
    'data': {'revenue': s0['source'], 'prices': 'raw/pre2015/asoul_twse_2007_2015H1.csv.gz＋data_panel（R55 主做法）；0050 含息淨值 jonatw etf_0050.csv',
             'no_network': True, 'git': '不呼叫 git（快取先放好；樣本內快取由 S55 scratchpad 複製，驗收逐位通過）'},
    'known_limits': ['營收檔 2013 起只有現在還上市的公司（存活者偏誤，對策略有利）', '2014 以前除權息日報酬記 0（TWT49U 只有 2014 起）', '0050 原始開盤 2015 起，之前開盤÷收盤比＝1',
                     'B2 的 k 是估計值：約 5% 的公司校正後整年偏低（被校正過頭，等於這些公司 2013 年不會被選到）；k 缺值的公司（約四分之一欄位，多為 2008 後才有營收者）到 2016-01 前沒有 SURGE',
                     '29 個月檢定力低：即使真實資訊比率＝樣本內的 0.72，預期 t 只有約 0.72×√(29÷12)＝1.12，過 1.65 的機率約三成；「不顯著」不能解讀成「沒有」'],
    'discipline': 'python3 -I -B；峰值記憶體 ≤ 3GB；不改共用引擎、既有研究、raw/；不寫 git',
}
PREREG = {
    'id': 'S71', 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
    'written_before': '任何候選規則（B1、B2）的回測。此前只跑過：驗收（現行遮罩規則，accept.json）、第 0 步營收描述（step0／0b／0c.json）、SURGE 涵蓋檢查（prep_check.json，不含報酬）。',
    'already_known_returns': '現行遮罩規則 2013-02～2015-06（卡住期）對 0050 −5.99 個百分點（R55）；0050 自身報酬。未看過任何 B1／B2 的選股名單或報酬。',
    'step0_decision_inputs': di,
    'primary_rule_selection': {
        'rule': 'B1 為主的條件：2013 年 12 個月年增率截面中位數的平均，與 2012、2014 兩年平均相差 < 5 個百分點，且「年增率 > +50%」的公司比例 < 2012、2014 平均的 1.5 倍；否則 B2 為主。',
        'observed': f"中位數差 {di['median_gap_pp']:+.1f} 個百分點；比例 {di['share_gt50_2013']:.1%} 對 {di['share_gt50_ref']:.1%}＝{di['share_gt50_ratio']:.2f} 倍",
        'local_file_2012_basis': '原始個體口徑，沒有回溯成合併（2012-12→2013-01 的對數跳動中位數 +0.26、37% 公司跳 1.5 倍以上；其他年 3～6%）',
        'result': primary},
    'primary_rule': primary, 'run_rules': [primary] + [r for r in ('B1', 'B2') if r != primary],
    'secondary_rule_role': '另一條照跑、只報告，不參與判定；不因結果互換主次。',
    'b2_parameters': {'prior_years': [2008, 2009, 2010, 2011, 2012], 'min_prior': 3, 'k_floor': 1.0, 'n_by_revenue_month': {'2013-01': 1, '2013-02': 2, '>=2013-03': 3},
                      'chosen_by': {'k_floor(ZERO)': s0b['variants'], 'n': s0c['variants']}},
    'decision': {'window': '2013-02～2015-06（29 個月）', 'benchmark': '0050 含息',
                 'support': '年化超額 > 0 且 Newey-West t（滯後 6）> 1.65 → 支持', 'weak': '年化超額 > 0 但 t ≤ 1.65 → 方向一致但不顯著', 'no': '年化超額 ≤ 0 → 不支持'},
    'also_report': ['合併 2008-01～2015-06（90 個月）年化超額、資訊比率、NW t', '逐年超額', '延伸段訊號月數、平均入選檔數、進場筆數、勝率', '對現行卡住期的配對差（描述）',
                    '延伸段進場股票中 k_3 > 1 的比例（描述）', '次要規則同表'],
    'acceptance': {'A1_official_bitwise_48.76pct_381': acc['A1_official']['bitwise_vs_ACC'], 'A1_since2019': acc['A1_official']['since2019'], 'A1_n_closed': acc['A1_official']['n_closed'],
                   'A2_uses': 'R55 主做法（+1.34 個百分點／NW t 0.52）', 'A2_primary_ex_ann_pp': acc['A2_R55_main']['primary']['ex_ann_pp'], 'A2_primary_t_nw6': acc['A2_R55_main']['primary']['t_nw6'],
                   'A2_max_abs_diff_vs_R55': acc['A2_R55_main']['max_abs_diff_vs_R55'], 'A3_mask_rule_bitwise_equals_engine': acc['A3_mask_equals_engine']['bitwise'],
                   'A4_pre2013_surge_rows_identical': prep['pre2013_rows_identical']},
    'run_once': 'run.py 只跑一次（results.json 存在即拒絕）；不重跑、不改規則後再跑。',
    'no_changes_whatever_result': '本題是樣本外檢定，結果不改現行規則；若要把跨斷點規則納入正式引擎需另題。',
}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name, obj in (('SPEC', SPEC), ('PREREG', PREREG)):
    (HERE / f'{name}.json').write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    (HERE / f'{name}.sha256').write_text(f'{sha(HERE / (name + ".json"))}  {name}.json\n')
files = ['s71lib.py', 'run.py', 'build_panel.py', 'accept.py', 'step0.py', 'step0b.py', 'step0c.py', 'prep.py', 'accept.json', 'step0.json', 'step0b.json', 'step0c.json', 'prep_check.json']
(HERE / 'code.sha256').write_text(''.join(f'{sha(HERE / f)}  {f}\n' for f in files))
print(primary, (HERE / 'SPEC.sha256').read_text().strip(), (HERE / 'PREREG.sha256').read_text().strip())
