#!/usr/bin/env python3
"""S75 事前登記：寫 SPEC.json、PREREG.json 與 sha256、code.sha256（在跑任何連續帳戶之前；此時只做過面板接縫比對、正式引擎驗收 A1、訊號驗收 A2／A3）。
python3 -I -B prereg_write.py"""
import json, hashlib, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
assert not (HERE / 'results.json').exists()
pc = json.load(open(HERE / 'panel_check.json')); a1 = json.load(open(HERE / 'accept_is.json')); a2 = json.load(open(HERE / 'accept_sig.json'))
assert a1['pass'] and a2['pass']
ov = pc['overlap_2015H1']

SPEC = {
    'id': 'S75', 'title': '現行策略 2008-01～2026-10 連續帳戶長期成績單（產品擁有者：「幫我測一下 2008～2026」）',
    'question': '把現行策略（月營收驚奇 SURGE、F1 進場、E0 出場、A10、W10_70、嚴成本）做成一個連續帳戶，100 萬從 2008-01-02 跑到 2026-10-22，長期成績單長什麼樣？',
    'nature': '把已登記的規則接起來跑；不新增可調參數、不挑窗、不做判定門檻。描述性成績單。',
    'frozen_current': {
        'engine': 's47lib.sim（→ s10lib.simulate_s10：W10_70＝衛星 70%、每檔＝衛星÷10、最多 10 檔、空位與底倉買 0050；E0 出場；嚴成本 strict；零股 2020-10-26 起）',
        'universe': 'stlib.universe(twse) & ~創新板', 'signal': 'F1＝s10lib.make_signal_s10(shift)；A10 換股日＝營收期限日後第一個交易日的下一個交易日（cslib.rev_avail_days）',
        'switches': 'cslib.EXRIGHT_FIX＝DEADLINE_FIX＝INS15_MASK＝True（正式開關）；s17lib.e50_fix_suspension',
        'surge_std': 'D＝月營收−12 個月前；SURGE＝(D−前 24 個月 D 平均)÷前 24 個月 D 標準差（s17lib.zscore 原函式）；訊號月份 ≥ 2008-01',
        'not_changed': '共用引擎、既有研究、raw/ 一律不改'},
    'versions': {
        'V_MASK': '完全照現行凍結規則：2013 年營收年增量遮罩（s71lib.rev_rule("MASK")，S71 驗收 A3 已證與 s17lib.rev_family 逐格相同）→ 營收月 2013-01～2015-12 沒有 SURGE，'
                  '訊號日 2013-02～2016-01 沒有換股，帳戶停在 2013-01 那批持股與 0050。',
        'V_BRIDGE': '2013 年改用 S71 事前登記的 B2 水準校正（s71lib.rev_rule("B2")，公式見 agents/s71_ifrs_bridge_oos/SPEC.json），其餘完全相同。主要呈現版本（訊號連續）。',
        'identical_parts': '兩版在營收月 ≤ 2012-12 與 ≥ 2016-01 的 SURGE 逐格相同（accept_sig.json A2）；差別只在 2013-02～2016-01 這 36 個訊號月，以及之後因帳戶狀態不同造成的路徑差。'},
    'panel': {
        'range': pc['range'], 'T_loaded': pc['T_loaded'], 'N': pc['N'],
        'before_2015_01_05': 'R55 主做法面板（agents/r55_replicate_s55/build_panel_r55.py 主做法逐行沿用）：Asoul/tsec 證交所原始行情長表＋data_panel 補格（補格漲跌價差＝收−收÷(1＋ret)，'
                             '使 2014 以前除權息日報酬記 0）；2007-01-02～2014-12-31 共 1,989 個交易日',
        'from_2015_01_05': '官方面板（jb7k6295r7 官方日 K 的樣本內快取，＝正式引擎逐位重現 48.76%／381 筆用的那一份），原值不動',
        'seam': '2015 上半年（2015-01-05～06-30，116 個交易日）兩邊重疊，一律用「官方面板」那一邊；重疊期逐格比對（官方上市格）：'
                f'收盤 {ov["C"]["n_both"]} 格相符率 {ov["C"]["match"]}、開盤 {ov["O"]["n_both"]} 格 {ov["O"]["match"]}、成交金額 {ov["AMT"]["n_both"]} 格 {ov["AMT"]["match"]}、'
                f'漲跌價差 {ov["CHG"]["n_both"]} 格 {ov["CHG"]["match"]}（官方有而 pre 無 {ov["CHG"]["n_official_only"]} 格）；官方上市而 pre 面板沒有的代號：{ov["MK"]["official_twse_codes_missing_in_pre"]}',
        'columns': f'pre 面板全部 {pc["n_pre_codes"]} 檔 ∪ 官方面板曾經上市的 {pc["n_official_codes_ever_twse"]} 檔＝{pc["N"]} 檔；官方面板中從未上市的純上櫃 {pc["n_official_codes_dropped_never_twse"]} 檔不放'
                   '（股池＝上市，永遠不會入選；只為了把峰值記憶體壓在 3GB 內）。驗收：2016-02 起每月入選名單與正式引擎逐月相同（accept_sig.json A3）。',
        'e50': '0050 含息淨值 jonatw etf_0050.csv（2003 起）；0050 原始開高低收只有 2015-01-05 起 → 之前開盤÷收盤＝1（用當日收盤代替開盤），跳動點滑價用引擎退路價 100 元',
        'innov': '創新板旗標：官方日 K 名稱欄（2015 起）；2015 前全 False（當時沒有創新板）'},
    'account': '單一連續帳戶：100 萬元，2008-01-02 起算（stlib.START），第一個訊號日 2008-02-13；中途不重設、不加減碼；到 2026-10-22',
    'benchmark': '同日（2008-01-02）100 萬買 0050 含息、不動、不計成本：P.E50 ÷ P.E50[起日] × 100 萬',
    'stats': {
        'cagr': 'stlib.cagr：(期末÷期初)^(365.25÷日曆天數)−1', 'maxdd': '日淨值對先前最高點的最大跌幅，報高點日、低點日、收復日',
        'vol': '日報酬標準差×√252', 'excess': 'r55lib：月底＝每個日曆月最後交易日；月超額＝策略月報酬−0050 含息月報酬；年化超額＝月平均×12（個百分點）；'
               '資訊比率＝月平均÷月標準差×√12；Newey-West t（Bartlett，滯後 6）；另報年化報酬相減（幾何差）',
        'yearly': '年底÷前一年底−1；2008 以 2008-01-02 為基底；2026 到 10-22（未滿一年，照樣列入「勝過 0050 的年數／19」，並另報不含 2026）',
        'periods': '2008～2012、2013～2015、2016～2018、2019～2023、2024～2026（2026-10 止）：各段年化報酬（基底＝段前最後一個交易日）、月超額統計、段內最大回撤',
        'bears': '段內最大跌幅（高低點都在段內；同 s17lib.seg_dd）：2008（整年）、2011（整年）、2015（整年）、2018（整年）、2020-03（2020-01-01～06-30）、2022（整年）、2024-07～2025-04；'
                 '後四段是 s17lib.SEGMENTS 原定義，前三段本題照「整個日曆年」補上',
        'paths': '20 條路徑＝本金 60／80／100／150／200 萬 × 換股日晚 0～3 個交易日（make_signal_s10 的 shift）：全期年化的中位數、最小、最大；另報最大回撤與 2019 起年化的同樣統計',
        'trades': '已平倉筆數、勝率、平均每筆報酬、平均持有交易日數、平均持股檔數、閒置資金比例（現金＋0050 占淨值；含 30% 底倉）；依進場日分 2008～2015、2016～2026'},
    'acceptance': {
        'A1': '正式引擎單獨跑官方面板（2015-07-01 起）：2019 起年化、全期年化、最大回撤、已平倉筆數與 L.ACC 逐位相同（48.76%／381 筆），淨值與 xq_scripts/site/navs_official.csv.gz 逐日相同 → 已通過（accept_is.json）',
        'A2': '連續面板 SURGE：兩版在營收月 ≥ 2016-01 與 ≤ 2012-12 逐格相同；MASK 版 2013～2015 全缺值 → 已通過',
        'A3': '連續面板訊號：兩版 2016-02 起 129 個訊號月的訊號日與入選名單與正式引擎逐月相同；V_BRIDGE 在 2015-06 以前 89 個月與 S71 的 B2 名單逐月相同 → 已通過',
        'A4_in_run': '連續帳戶淨值到 2014-12-31（接縫前）應與 S71 的 B2 淨值逐日相同、V_MASK 月報酬應與 S71 存的 MASK 月報酬相同（報最大差；到 2015-06-30 的差另報）',
        'A5_in_run': '連續帳戶在 2015-07-01 重新基準化後：2019 起年化「應接近」48.76%、2015-07 起年化對照 34.40%。因為 2015-07-01 當天的持股與資金規模不同（正式引擎從 100% 現金起跑；'
                     '連續帳戶帶著既有持股、淨值不是 100 萬 → 整股時代的可買股數、5% 成交值上限、名額占用都不同），不會逐位相同；報差多少（不設及格線），並逐年對照。'},
    'data': {'revenue': '/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv（sha256 見 S71 SPEC）',
             'prices': 'raw/pre2015/asoul_twse_2007_2015H1.csv.gz＋data_panel（2015-01-05 前）；官方日 K 快取（之後）；0050 jonatw etf_0050.csv',
             'no_network': True, 'git': '不呼叫 git（快取先放好；樣本內快取取自 eng/cache，A1 逐位通過）'},
    'outputs': ['results.json', 'nav_2008_2026.csv.gz（date、V_MASK、V_BRIDGE、0050）', 'yearly.csv', 'trades.csv.gz', 'signals.csv', 'paths.csv'],
    'discipline': 'python3 -I -B；峰值記憶體 ≤ 3GB；暫存 scratchpad/s75/ 結束刪除；不改共用引擎、既有研究、raw/；不寫 git；不連外網',
}
PREREG = {
    'written_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'run_once': 'run.py 只跑一次（results.json 存在就拒跑）；結果不論好壞照登記的口徑全部報出，不調任何參數、不換窗、不換版本',
    'state_before_run': '尚未跑過任何連續帳戶、沒有看過任何 2008～2026 連續淨值；已知的只有既有研究的數字（S55／R55：2008-01～2013-01 +1.34 個百分點；S71 B2：2013-02～2015-06 +3.68、'
                        '合併 2008-01～2015-06 +2.10；正式引擎 2015-07 起 34.40%、2019 起 48.76%）',
    'primary_version': 'V_BRIDGE（主要呈現）；V_MASK 同表並列（完全照現行凍結規則）',
    'standard_path': '100 萬、位移 0；成績單的所有主數字都用這一條；20 條路徑只報中位數與範圍',
    'no_decision_threshold': '本題是成績單，不設「支持／不支持」門檻；Newey-West t 照報，但事前聲明：2015-07 以後是規則的樣本內（規則在這段資料上選出來），'
                             '全期 t 值大半來自樣本內，不能當成樣本外證據；真正樣本外只有 2008-01～2013-01（現行規則）與 2013-02～2015-06（B2，2026 年才訂的規則）',
    'how_to_read_declared_before_run': [
        '全期年化會被 2015 年以前拉低（已知那段年化超額只有 +1～2 個百分點、閒置資金約 75%）；',
        '「2015-07 起 34.4%／2019 起 48.8%」是樣本內數字；連續帳戶 2019 起的年化應與 48.8% 接近（同樣的訊號），差異來自起始持股與資金規模；',
        '白話結論固定三句：2008 起連續做到現在的結果；哪幾段賺、哪幾段不賺；和 34.4%／48.8% 比該怎麼看。'],
    'known_biases_declared_before_run': [
        '2014 以前除權息日報酬記 0（TWT49U 只有 2014 起）→ 對策略偏保守（data_exright_fix 量過約 +0.86 個百分點／年）；0050 用含息淨值，不受影響 → 超額偏保守',
        '0050 原始開盤價 2015-01-05 前沒有 → 用當日收盤代替開盤成交（方向不定、影響小）',
        '下市股月營收在最後交易日前約 13 個月就斷 → 下市前一年不可能入選（存活者偏誤，偏樂觀）',
        '月營收檔 2013 年起只有現在還存在的公司（存活者偏誤，偏樂觀）',
        '2015-06-01 以前實際漲跌停幅度 7%，引擎固定用 10% → 可能把實際漲停鎖死（買不到）當成買得到、跌停（賣不掉）當成賣得掉；事後報告受影響筆數（不改結果）',
        '2020-10-26 以前只能買整張：100 萬本金、每檔約 7 萬 → 股價超過約 70 元的入選股買不起（空下的錢留在 0050）；早年持股檔數少、閒置資金高，是本金假設造成的，不是訊號少',
        '成本假設＝現行嚴成本（手續費 2.8 折、證交稅 0.3%、跳動點滑價、5% 成交值上限）套用到 2008 年：當年實際手續費折扣較少、流動性較差 → 偏樂觀',
        'B2 規則是 2026 年才訂的（事前登記、設計時只看營收資料、沒看報酬），不是 2013 年當時就有的規則；k 是估計值',
        '2015 以前沒有營收期限特例登錄（一律法定 10 日）；保險業 15 日名單照 ins15_list.csv 的起始月',
        'Asoul 長表缺部分 2008～2012 下市股的行情（已知多為低價股，多半不在股池）'],
    'diagnostics_preregistered': ['7% 漲跌停下開盤即漲停的買進筆數、開盤即跌停的賣出筆數', '2015-07-01 重新基準化後與正式引擎的逐年差', '兩版在 2016 年以後的日報酬相關與年化差',
                                  '引擎統計：買不起整張（blk_lot）、漲停買不到、下市筆數'],
    'accept_before_run': dict(A1_bitwise=a1['bitwise_vs_ACC'], A1_nav_vs_site_max_abs=a1['max_abs_diff_vs_navs_official_csv'], A2=a2['A2_surge'],
                              A3_post2016_identical=dict(V_MASK=a2['A3_signals']['V_MASK']['post2016_identical'], V_BRIDGE=a2['A3_signals']['V_BRIDGE']['post2016_identical']),
                              A3_bridge_vs_S71=a2['A3_signals']['bridge_vs_S71_B2']),
}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name, obj in (('SPEC', SPEC), ('PREREG', PREREG)):
    (HERE / f'{name}.json').write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    (HERE / f'{name}.sha256').write_text(f'{sha(HERE / f"{name}.json")}  {name}.json\n')
    print(name, sha(HERE / f'{name}.json'))
files = ['s75lib.py', 'run.py', 'build_panel.py', 'accept_is.py', 'accept_sig.py', 'panel_check.json', 'accept_is.json', 'accept_sig.json']
(HERE / 'code.sha256').write_text(''.join(f'{sha(HERE / f)}  {f}\n' for f in files))
