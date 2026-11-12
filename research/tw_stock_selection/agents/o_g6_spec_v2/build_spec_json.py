#!/usr/bin/env python3
"""把 SPEC v2 的條文寫成機器可讀 JSON（每條規則一個 key）。只讀資料檔的參數與 verify_inputs.json；不算任何報酬。
python3 -I -B build_spec_json.py"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
VI = json.loads((HERE / 'verify_inputs.json').read_text(encoding='utf-8'))
MS = json.loads((AG / 'o_v7_listing_exec_engine/data/model_spec7.json').read_text(encoding='utf-8'))
CPI = json.loads((AG / 'o_v7_listing_exec_engine/raw/exec_inputs/cpi_tw.json').read_text(encoding='utf-8'))['index_2006_100']
ENDS = json.loads((AG / 'o_v7_listing_exec_engine/data/jscan_end_days.json').read_text(encoding='utf-8'))

def R(title, pipeline, definition, code, v1, task=None):
    o = dict(title=title, pipeline=pipeline, definition=definition, code_ref=code, diff_vs_v1=v1)
    if task: o['task_item'] = task
    return o

rules = {}
# ---------------- G
rules['G-01'] = R('兩條管線', 'both', [
    'PAPER：名單 O18.build→PC.VU→V2.V2；面板 D-01b；訊號日＝每月第一個日>=11 的面板日；營收 YLp/Y5p（P-03）；unavail＝A_UB；外國股只在放回層判定（P-07）；金融 FINMASK；放回＝GOV；營收百分位分母＝|el|；遮罩現行正式＝M1_paper；引擎 CE.simulate consideration',
    'V7X：名單 VL.make→JV7R→V4→V2；面板 D-01a＋P；訊號日＝j 日（j=0..12）；營收 R_FIX/Y5_FIX 改寫版（L-03）；unavail＝MC(seed) OR A_UB；外國股 FORMASK；金融 FIN2MASK；放回＝GAP OR GOV；營收百分位分母＝|el_rev|；遮罩現行正式＝DM_BASE；引擎 CV.CSim consideration；seeds 0..19'],
    ['O18:92-124', 'VL:147-167'], 'v1 只寫「見 v7x_lib.py docstring」')
rules['G-02'] = R('日期記號', 'both', [
    'd＝訊號日（面板日），t＝d 的面板列號', 'e＝列號 t+1 的面板日（下一個面板列；不是下一個 GOOD 日）',
    'm＝錨定月（V7X）；紙上 m＝Period(d)', 'j∈{0..12}', '日期比較皆為日，除 A_UB 分類用 13:30'],
    ['V2:158', 'V4:157'], 'v1 未定義')
rules['G-03'] = R('欄位順序', 'both', ['兩面板欄位相同：1249 檔、4 碼、字串遞增、無 00 開頭', '「欄位序」與所有長度 N 陣列（含亂數）依此順序；N=1249'],
                  ['check_panel.json'], 'v1 未定義')
# ---------------- C
rules['C-01'] = R('TRADED 與 GOOD', 'both', ['TRADED[t,c]＝close_raw 有限', 'GOOD[t]＝TRADED 檔數 >= 300', '唯一非 GOOD 日：2018-01-13', 'V7X 管線在補 P 後重算'],
                  ['IA:41', 'IA:53-54', 'O3:53'], 'v1 未定義')
rules['C-02'] = R('下市價補值 P', 'V7X', ['delist_prices 只寫入 close_raw 原為 NaN 的格：close、open、high、value、ret 照寫，flag=0', '之後重算 TRADED、GROW、MYADJ、GOOD、PREVC', '紙上管線不套 P'],
                  ['O3:34-57', 'XD:38-43'], 'v1 未定義')
rules['C-03'] = R('GROW 與 MYADJ', 'both', ['GROW[t,c]＝TRADED ? 1+ret（NaN 當 0） : 1', 'MYADJ[t,c]＝TRADED ? prod_{s<=t} GROW[s,c] : NaN'], ['IA:43', 'IA:46'], 'v1 未定義')
rules['C-04'] = R('名單側 V60 與 S52（gapfix）', 'both', [
    '只在 GOOD 日列上滾動；非 GOOD 日＝NaN、不補值', 'EX＝{2017-08-03, 2018-07-12} ∪ [2024-02-15, 2024-03-15]',
    'V60[t,c]：往回 60 個 GOOD 日（含 t）；窗不足 60 列或窗內有非 EX 日的 value 為 NaN → NaN；否則＝窗內非 NaN 的平均',
    'S52[t,c]＝MYADJ[t,c] / max(窗內 MYADJ 非 NaN)；窗＝252 個 GOOD 日（含 t）；窗不足或窗內有非 EX 日的 MYADJ 為 NaN → NaN'],
    ['V2:31-58'], 'v1 未定義')
rules['C-05'] = R('跳動點與開盤漲停', 'both', [
    'tick(p)：p<10→0.01；<50→0.05；<100→0.1；<500→0.5；<1000→1；否則 5', 'lim＝成交日 < 2015-06-01 ? 0.07 : 0.10',
    'ref[e,c]＝close_raw[e]/(1+ret[e])；flag[e]==1 → e 之前最後一筆收盤', 'lu＝floor(ref*(1+lim)/tick(ref*(1+lim)) + 1e-9) * tick(ref*(1+lim))',
    'gap_up[e,c]＝open[e] 有限 且 ref 有限 且 open[e] >= lu - 1e-6'], ['IA:143-164'], 'v1 未定義')
rules['C-06'] = R('y3', 'both', ['y3(Y,m)[c]＝mean(Y[m-1,c], Y[m-2,c], Y[m-3,c])；任一 NaN 或任一月不在 Y 索引 → NaN；單位 %', 'd >= 2021-03-01 用 5 年表，否則長表'],
                  ['IA:124-131', 'V2:22', 'V2:151-155'], 'v1 未定義')
rules['C-07'] = R('年增率表基本版本', 'both', [
    'YL：tw_revenue 同月同代碼 sii 優先去重→水準表 lv（月×代碼，索引補成連續月）→ (lv/lv.shift(12)-1)*100，僅 lv.shift(12)>0 有值',
    'Y5：rev_5y 的 yoy 欄；同月同代碼取最後一筆；索引補成連續月'], ['IA:100-110', 'IA:112-122'], 'v1 未定義')
rules['C-08'] = R('FINMASK', 'both', ["stock_list.industry=='金融保險業' ∪ 代碼前兩碼 '28' ∪ {'5876','5880'}；用現行 stock_list"], ['V2:84-87'], 'v1 未定義')
rules['C-09'] = R('股池：先篩後取前 300', 'both', [
    'base[c]＝V60[t,c] 有限且 >= 1e7 且 close_raw[t,c] 有限且 >= 10', '在 base 內依 V60 由大到小（穩定排序，同值依欄位序）取前 min(300,|base|)＝top',
    '排序母體＝面板全部 1249 檔（金融、外國、處置股先佔名額）'], ['V2:159-165', 'V4:158-163', 'PC:182-186'],
    'v1 未寫順序；另一讀法（先取前 300 再篩）o20 複製員實測 Jaccard 0.896', task=5)
rules['C-10'] = R('排序、同分與選股', 'both', [
    'sc 四捨五入到 10 位小數（NaN 當 -9）', '排序鍵：sc 大→小；y3tie 大→小（NaN 當 -1e18；+inf 最大）；欄位序小→大',
    '候選＝排序前 |el| 名；依序：gap_up[e,c] 且 c 不在 held → 跳過；否則入選；滿 10 檔停',
    'held＝上一個有產生名單的訊號日的名單（非引擎實際持股）；第一期為空', '|el| < 50 → 該訊號日不產生名單、held 不更新', '名單內順序＝入選順序'],
    ['V2:185-215'], 'v1 未定義')
# ---------------- P
rules['P-01'] = R('紙上訊號日', 'PAPER', [
    '面板 D-01b；每個日曆月第一個 day>=11 的面板日（起算 2005-01-01）',
    '保留條件：Period(d)-1 <= last_full_month(Y)；y3(Y,Period(d)) 有限檔數 > 100；d < 2026-10-12',
    'Y＝d>=2021-03-01 ? Y5p : YLp（P-03 補值後）；last_full_month(Y)＝非 NaN 檔數 >= 0.8*各月非 NaN 檔數中位數 的最後一個月',
    '結果 245 日：2006-04-11～2026-09-11（無 2024-01）；END＝2026-10-12'], ['IA:55-64', 'V2:130-149', 'V2:79-81'], 'v1 未定義')
rules['P-02'] = R('紙上成交日', 'PAPER', ['e＝d 的下一個面板列'], ['V2:158'], 'v1 未定義')
rules['P-03'] = R('紙上營收表（B1 只補缺）', 'PAPER', ['YLp＝YL 複本、Y5p＝Y5 複本', 'B1 每列 (code, ym, yoy_pct 有限)：ym 在索引內且該格為 NaN（無此欄先新增）→ 填入；不覆蓋既有值'],
                  ['O18:73-85', 'O18:99-103'], 'v1 只寫「B1 補回值」', task=1)
rules['P-04'] = R('A_UB', 'PAPER', [
    "觸發：deadline_case(d) ∈ {'D=S','D>S'}",
    "deadline_case(d)：rm＝Period(d)-1；rm∈{2019-01,2023-03}→'D>S'；rm==2020-01→'D=S'；否則 D＝TCAL2 中第一個 >= d 所在月 10 日；D<d→'D<S'；D==d→'D=S'；D>d→'D>S'；TCAL2＝TAIEX 日期 ∪ 面板日期",
    "觸發時 ym＝Period(d)-1（d 的日曆月前一月）；U[c]＝classify(d,c,ym,'V6') 狀態 != 'AVAIL'；未觸發 U 全 False",
    'U[c] 為真 → y3[c] 改用 y3(Y, Period(d)-1)[c]', '紙上無資料邊界豁免'],
    ['O18:41-52', 'V2:101-106', 'V2:170-176'], '附註 5 只寫觸發與月份；未寫 classify 與 !=AVAIL', task=6)
rules['P-04a'] = R('classify V6', 'both', [
    'S＝d；C1330＝S+13:30；C24＝S+1 日；時間戳轉台北時間',
    'MOPS(ts_mops_ann_date)：ann<S→AVAIL；==S→UNDET；>S→UNAVAIL',
    "快照(ts_tasi_upstream_firstseen；T 為 NaT 跳過)：T<=C1330→AVAIL；否則 T_prev 有值且其日期 > S→UNAVAIL；否則 UNDET；覆寫：source in {'upstream','tasi'} 且 T_prev 空 且 T > C24+3 日 → NONE",
    '新聞(ts_news_mentions，需過數值閘門)：has_time：ts<=C1330→AVAIL；ts<C24→UNDET；否則 NONE；無時間：日期<S→AVAIL；==S→UNDET；否則 NONE',
    "數值閘門 'sig'：v＝rev_k 有限且>0；tv＝tw_revenue.revenue_k_twd（同 code,ym 第一筆）與 rev_5y.revenue 的有限值；z＝str(int(round(v))) 尾端 0 的個數；tol＝0.5*10^z+1.0；存在 t∈tv 使 |v-t|<=tol",
    '合併：AVAIL 與 UNAVAIL 並存→UNDET；有 AVAIL→AVAIL；有 UNAVAIL→UNAVAIL；有 UNDET→UNDET；否則 NONE', 'A_UB 只看 == AVAIL；UNDET、UNAVAIL、NONE 都算未公告'],
    ['AV:22-48', 'AV:50-93'], 'v1 未定義', task=6)
rules['P-05'] = R('紙上處置遮罩 M0／M1_paper／M2', 'PAPER', [
    'e＝d 的下一個面板列日期（最後一列時 e=d）',
    'M0＝{disposal_5y: start<=d<=end}（不看公告日）∪ (d<=2021-09-23 ? {official_universe: (start<=d<=end) or (start<=e<=end)} : ∅)',
    'M1_paper（現行正式）＝M0 ∪ (d>2021-09-23 ? {disposal_5y: date<=d and start<=e<=end} ∪ {universe_announce: announce<=d and start<=e<=end} : ∅)',
    'M2＝{universe_announce: announce<=d and start<=e<=end}'], ['IA:134-140', 'O18:55-70', 'I20P:24-42'], 'v1 只有 M0', task=7)
rules['P-06'] = R('紙上合格集合 el_base', 'PAPER', ['el_base＝top ∩ y3 有限（已套 P-04；FINMASK 的 y3＝NaN）∩ S52[t] 有限 ∩ 非遮罩 ∩ open[e] 有限 ∩ TRADED[e]', '無明文外國股條件'], ['V2:157-183'], 'v1 未定義')
rules['P-07'] = R('is_foreign（紙上放回層）', 'PAPER', [
    "is_foreign(c)＝c[:2]=='91' 或 名稱集合中任一 n：'KY' in n.upper() 或 n.endswith('DR') 或 n.startswith('F-')",
    '名稱集合＝{NAME[c]} ∪ SNAPN[c]', 'NAME[c]＝tw_revenue（依 月,代碼,sii 優先 排序去重）中該代碼最早一個月的名稱；tw_revenue 無此代碼才用 stock_list.name；皆無＝空字串',
    'SNAPN[c]＝三個 industry_snap 檔中第一欄為 4 位數字的列的第二欄（去空白與雙引號）', '不使用 delist_official、merger_consideration 的名稱'],
    ['PC:98-107', 'PC:118-120'], 'v1 未寫名稱來源；delist_official 讀法在面板上多判 6 檔外國（1262,1592,2936,6422,6452,8480），不是程式行為', task=1)
rules['P-08'] = R('放回集合 A 與 cause', 'PAPER', [
    'el0＝top ∩ S52 有限 ∩ 非遮罩 ∩ open[e] 有限 ∩ TRADED[e] ∩ 非 FINMASK', 'A＝{c ∈ el0 : y3[c] 為 NaN 且 cause(c,d) ∈ GOV}',
    'GOV＝{TRUNC, IFRS13, POST13, POST13_R5, PRE12, HOLE, NEVER_DOM}',
    "cause(c,d)，m＝Period(d)：(1) is_foreign→FOREIGN",
    "(2) d<2021-03-01：c 不在 tw_revenue 水準表→NEVER_DOM；對 x∈{m-1,m-2,m-3,m-13,m-14,m-15}：lv[x,c] NaN→month_cause(c,x,d)；有限且<=0 且 x<=m-13→BASE_LE0",
    "month_cause：lr/fr＝該檔最後/最早有營收月；ltm＝面板最後成交日所在月；x>lr：2009-12<=ltm<=2012-12→TRUNC；否則 x>=2013-01 且 lr>=2012-12→(d<=2013-12-31 ? IFRS13 : POST13)；否則 PRE12；x<fr→NEWLIST；其餘→HOLE",
    "(3) d>=2021-03-01：c 不在 rev_5y yoy 表→(c 在 tw_revenue ? POST13_R5 : NEVER_DOM)；對 x∈{m-1,m-2,m-3}：yoy NaN→x>最後有 yoy 月→POST13_R5；x<最早有 yoy 月→NEWLIST；否則 HOLE",
    '(4) 優先序 [TRUNC, IFRS13, POST13, POST13_R5, PRE12, HOLE, NEWLIST, BASE_LE0] 第一個；皆無→UNKNOWN', '原因判定用未補值的原始 tw_revenue 水準與 rev_5y yoy；不看 A_UB 回退'],
    ['PC:126-165', 'PC:176-194'], 'v1 只寫「pclib.VU 的定義」', task=1)
rules['P-09'] = R('紙上三種放回值與分數', 'PAPER', [
    'el＝el_base ∪ A；n＝|el|；s52p＝S52 在 el 內平均名次/n',
    'opt：A 的 rp＝1.0；norm＝el∖A 的 rp＝（y3 在 norm 內平均名次）/n；A 的 y3tie＝+inf',
    'p90：A 的 rp＝0.9；norm 同上；A 的 y3tie＝norm 的 y3 第 90 百分位（線性內插）',
    'carry：A 每檔 b=1..24 取 y3(Y, Period(d)-b)[c] 第一個有限值；無→移出 A；rp＝y3 在 el 內平均名次/n；y3tie＝y3',
    'sc＝0.5*s52p+0.5*rp；非 el＝NaN；之後 C-10'], ['PC:195-222'], 'v1 未寫分母、p90 次排序值、24 個月上限', task=8)
rules['P-10'] = R('紙上名單輸出', 'PAPER', ['每訊號日：signal(d), exec(e), picks（10 檔，排名順序）'], ['O18:105-124'], '—')
# ---------------- L
rules['L-01'] = R('錨定月與 j 日', 'V7X', [
    'ANCH＝以 D-01a＋P、R_FIX/Y5_FIX（未改寫）照 P-01 規則得到的官方訊號日所屬月份（246 個月：2006-04..2026-09）',
    'D(m)：m==2019-02→2019-02-15；m==2023-04→2023-04-12；否則 m 月 10 日；取 TAIEX 日曆（不含面板獨有日）中第一個 >= 該日',
    'day_for(m,j)＝TAIEX 日曆中 D(m) 之後第 j 個日期；不在面板、非 GOOD、或其後無 GOOD 面板日 → 無',
    'd 的下一個面板列非 GOOD → 該月跳過', 'de(j)＝day_for(2026-10,j)；無或 > 面板最後一日 → 面板最後一日；只保留 d < de(j)', 'e＝d 的下一個面板列',
    '特例：j0 首日 2006-04-10；j2 無 2018-01；j7 無 2026-09；農曆年附近 j 日可落到下一日曆月（例 錨定月 2020-01、j=9 → 2020-02-03）'],
    ['ST:91-121', 'VL:26-39'], 'v1 只寫「13 j」', task=8)
rules['L-02'] = R('V7X 的 y3 月份', 'V7X', ['y3(Y, m-back)，m＝錨定月；Y 依 d>=2021-03-01 切換；FINMASK 的 y3＝NaN'], ['VL:104-109'], 'v1 未定義')
rules['L-03'] = R('V7X 營收表（REAL）', 'V7X', [
    '長表缺口格＝YL（C-07，重新索引到 R_FIX）為 NaN 且 R_FIX 有值：B1 有 (code,ym)→改成 B1.yoy_pct；無→設 NaN',
    'B1 其餘 (code,ym)：在索引內且該格 NaN→填入', 'Y5_FIX 同法（缺口格＝rev_5y yoy 為 NaN 且 Y5_FIX 有值）',
    'o18 正式計數：長表缺口 2052（改 961、設缺 1091、新增 0）；5 年表缺口 0'], ['VL:63-94'], 'v1 一句話帶過', task=1)
rules['L-04'] = R('MC(seed)', 'V7X', [
    'd < D(m)：全 True', 'd == D(m) 或 (m==2020-02 且 d<=2020-02-14)：u＝numpy.random.default_rng([seed, YYYYMMDD(d), 0]).random(N) < PCOL', '其餘：全 False',
    'm-1 ∈ [2026-01,2026-08]：D15＝TAIEX 中第一個 >= m 月 15 日；d<D15→u|=IS15；d==D15→u|=IS15 & (default_rng([seed, YYYYMMDD(d), 15]).random(N) < PCOL)',
    'PCOL[c]＝c∈FIN ? 1.0 : p_by_industry.get(stock_list.industry, p_overall)；IS15[c]＝c∈INS15', 'N=1249、欄位序；numpy 2.5.3（PCG64）'],
    ['ST:137-152', 'V2:91-97'], 'v1 只寫「MC(seed)」', task=8)
rules['L-05'] = R('V7X 的 A_UB 與邊界豁免', 'V7X', [
    'ub(d)＝P-04（用 d 的日曆月：deadline_case(d)、分類月份 Period(d)-1；非錨定月）', 'U(d)＝u_mc(m,d,seed) OR ub(d)',
    '例外：ub(d) 有任何 True 且 y3(Y,m-1) 有限檔數 < 100 → U(d)＝u_mc（實際：2006-04-10、2021-03-10）', 'U[c] 為真 → y3 用 y3(Y,m-1)'],
    ['VL:42-51', 'VL:157-162'], '附註 5 未寫豁免判斷式', task=6)
rules['L-06'] = R('V7X 處置遮罩 DM_BASE／M2', 'V7X', [
    'T_V4＝rejudge[in_panel]（公告日 announce）∪ disposal_5y 中代碼在面板且 (code,start) 不在前者的列（公告日 date）',
    'DMT(d)＝{T_V4: 公告日<=d and start<=e<=end}', 'DM_BASE(d)（現行正式）＝DMT(d) ∪ (d<=2021-09-23 ? {official_universe: (start<=d<=end) or (start<=e<=end)} : ∅)',
    'M2(d)＝{universe_announce: announce<=d and start<=e<=end}'], ['DRJ:37-47', 'V4:92-117', 'VL:52-62', 'I20G:22-31'], 'v1 寫「DMT ∪ 官方全歷史」', task=7)
rules['L-07'] = R('FORMASK', 'V7X', ["FOREIGN＝foreign_codes（o_rev_survivorship/data_freeze.json）∪ {面板欄位以 '91' 開頭} ∪ {delist_official 名稱以 'F-' 開頭或大寫含 'KY'}"], ['V4:66-70', 'V4:79'], 'v1 未定義；與 P-07 相差 6 檔', task=1)
rules['L-08'] = R('FIN2MASK', 'V7X', ["c∈FIN 或 delist_official 名稱含任一：'銀','證','金控','證金','產保','壽','票券','產險','人壽','投信','合庫','信託'", 'y3 的 NaN 化仍用 FINMASK'], ['V4:71-75'], 'v1 未定義')
rules['L-09'] = R('V7X 合格與放回集合', 'V7X', [
    'ok＝top ∩ S52 有限 ∩ 非遮罩 ∩ open[e] 有限 ∩ TRADED[e] ∩ 非 FIN2MASK ∩ 非 FORMASK', 'el_rev＝ok ∩ y3 有限；miss＝ok ∩ y3 NaN',
    'A＝{c∈miss : (c∈GAP 且 d<=GAP[c]) 或 cause(c,d)∈GOV}（cause 的 m＝d 的日曆月）', "GAP＝{delist_official four=='True' 的代碼→date}，排除 FOREIGN 與 is_fin"],
    ['V4:155-173', 'VL:110-120', 'V4:76-78', 'O3:205-209'], 'v1 寫「GAP ∪ GOV」', task=1)
rules['L-10'] = R('V7X 三種放回值與分數', 'V7X', [
    'opt/p90：el＝el_rev ∪ A；pr52＝S52 在 el 內平均名次/|el|；prr＝y3 在 el_rev 內平均名次/|el_rev|；A 的 prr＝1.0 或 0.9；A 的 y3tie＝+inf（opt）或 el_rev 的 y3 第 90 百分位（p90）',
    'carry：A 每檔 b=1..24 取 y3(Y,m-b)[c] 第一個有限值；無→移出 A；el＝ok ∩ y3 有限；pr52、prr 皆為 el 內百分位；y3tie＝y3', 'sc＝0.5*pr52+0.5*prr；之後 C-10'],
    ['VL:121-145'], 'v1 未定義；分母與 P-09 不同', task=8)
rules['L-11'] = R('V7X 名單輸出', 'V7X', ['欄：variant,j,seed,signal,exec,exec_idx,picks,inserted；exec_idx＝e 的 V7 面板列號（0 起）；picks 排名順序；inserted＝picks ∩ A', '每個 REAL 變體 63820 列（o18 正式）'], ['VL:163-167'], '—')
# ---------------- E
rules['E-00'] = R('引擎設定', 'V7X', ["cfg＝{tier, mode:'GROW', model:'K5T', cost:'normal'|'stress', stress_odd_extra:true, rank_right:true, f1:true, f2:true, f3:0.5, f4:true}", "settle＝E-20（bound='main'）；cmode='consideration'"], ['I20V:18-37'], 'v1 只寫 grid')
rules['E-01'] = R('引擎基本陣列', 'V7X', ['C,O,V,R,F＝補 P 後的 close_raw,open,value,ret,flag', 'TR＝C 有限；G＝TR ? 1+R(NaN→0) : 1；CF＝C 前值補；PC[t]＝CF[t-1]；ADJ[t]＝TR ? G*PC/C : 1（非有限→1）；GOOD＝TR 檔數>=300'], ['XD:31-51'], 'v1 未定義')
rules['E-02'] = R('引擎側 V60', 'V7X', ['GOOD 日列上同 C-04 的 V60', '放回全日曆後逐欄 ffill(limit=3)（對所有 NaN 生效，含 GOOD 日因窗內缺值的 NaN）'], ['XD:52-57', 'XD:61'], '程式行為偏離 V7X frozen_spec「窗內不得缺值」（o19 G1）', task=2)
rules['E-03'] = R('σ60', 'V7X', ['GOOD 日列上 R 的 60 列滾動標準差（ddof=1，min 40 個非 NaN），逐欄 ffill；放回全日曆後再 ffill'], ['XD:58-62'], 'v1 未定義')
rules['E-04'] = R('vrank', 'V7X', [
    '每列 t：母體＝全部 1249 檔中 V60[t]（E-02）有限且>=1e7 且 CF[t] 有限且>=10', '依 V60 大→小（穩定排序，同值依欄位序）名次 1..；不在母體→NaN',
    '成交日 t 用前一列：VRp[t]=VR[t-1]；V60p[t]=V60[t-1]；SIGp[t]=σ60[t-1]'], ['XD:63-73'], 'v1 未定義；frozen_spec 寫「收盤>=10」，程式用前值補收盤 CF（o19 G2）', task=2)
rules['E-05'] = R('分桶', 'V7X', [
    'DEC＝VRp 有限且<=300 ? clip(ceil(VRp/30),1,10) : 10', 'OASH＝oa_share_by_dec[DEC]；ZSH＝VRp 有限且<=300 ? zcap.share_by_dec[DEC] : zcap.share_default',
    '零股名次桶＝(VRp>=150 或 NaN) ? 1 : 0', '零股價格桶（PC[t]；NaN 當 0）：<50→0；[50,100)→1；[100,300)→2；>=300→3', 'OCB＝odd.buy[價格桶][名次桶]；OCS＝max(odd.sell[價格桶][名次桶],0)'],
    ['XE:33-50'], 'v1 未定義', task=2)
rules['E-06'] = R('漲跌停價', 'V7X', ['REF[t]＝F==1 ? PC : C/G；lim＝t<2015-06-01 ? 0.07 : 0.10', 'LIMUP＝floor(REF*(1+lim)/tick+1e-9)*tick；LIMDN＝ceil(REF*(1-lim)/tick-1e-9)*tick；tick 取 REF*(1±lim) 所在級距'], ['XE:27-31', 'XD:75'], 'v1 未定義')
rules['E-07'] = R('零股成交價與整股／零股邊界', 'V7X', [
    't < 2020-10-26（盤後零股）：ODDP[t]＝C[t]', "t >= 2020-10-26（盤中零股）：ODDP[t]＝O[t]*(1+drift)；gap＝O/PC-1（無效→0）；drift＝drift.vals[searchsorted(drift.edges, gap, side='right')]"],
    ['XE:38-44', 'XD:17'], 'v1 未定義', task=8)
rules['E-08'] = R('引擎 DISP', 'V7X', ['DISP[t,c]＝T_V4（L-06）中存在 start<=IDX[t]<=end 的列（不看公告日）', '與名單遮罩無關；M2 整合未改'], ['XD:93-107'], 'v1 未定義', task=7)
rules['E-09'] = R('起始資金 GROW', 'V7X', ['t0＝名單第一列的訊號日列號；t0 收盤現金＝tier*CPI[2006]/CPI[2026]（係數 0.7649078503404548）；無持股；無現金流', 'tier：30 萬 3e5；100 萬 1e6；300 萬 3e6；1000 萬 1e7（2026 幣值）'], ['CV:33', 'XD:143-145', 'I20V:28'], 'v1 只寫 GROW', task=8)
rules['E-10'] = R('每日順序', 'V7X', [
    '1 股數*=ADJ[t]', '2 cash_late 併入現金', '3 tr[c]＝TR[t,c] 且 O[t,c] 有限', '4 對價結算觸發與到期（E-21）', '5 換股日→E-11；否則有順延單→E-17 順延日',
    '6 權益＝現金+cash_late+Σ股數*CF[t]+待撥款+Σ請求權；淨值＝權益/起始現金', '引擎跑到面板最後一列；統計截到 e(j)'], ['CV:38-131'], 'v1 未定義')
rules['E-11'] = R('換股日開盤前', 'V7X', ['清空全部順延單', '凍結持股（持股>0 且 tr 假）→E-22', '待撥款到期併入現金', '可交易持股小數股 S-floor(S+1e-9) 以 O[t] 轉現金，不計成本', '→E-12'], ['CV:75-120'], 'v1 未定義', task=3)
rules['E-12'] = R('目標股數', 'V7X', [
    'cur＝持股>0 且 tr 真；U＝cur ∪ pk；k＝|pk|；凍結的非名單持股不在 U', 'avail＝現金+Σ_cur 股數*O[t]；hi＝avail/k*(1-0.02)',
    'p＝PC[t]；L＝floor(T/(1000p)+1e-12)；q＝floor((T-1000Lp)/p+1e-9)；名單股目標＝1000L+max(q,0)；非名單 0；delta＝目標-floor(持股+1e-9)',
    'need(T)＝Σ_U cost_est(delta)；非名單且在 DISP 的賣出不計', 'need(hi)<=現金→T=hi；否則 [0,hi] 二分 40 次取 need<=現金 的下界'], ['XE:272-305'], 'v1 未定義', task=8)
rules['E-13'] = R('cost_est', 'V7X', [
    'a=|delta|；L=a//1000；q=a%1000；fr＝E-16', '處置賣出：價＝C[t]-tick(C[t])；整股、零股部分各一筆：-v+max(20,fr*v)+0.003*v',
    '整股：v＝1000*L*O[t]*(1+side*imp(L))；side*v+max(20,fr*v)+(賣 ? 0.003*v : 0)；imp 用全部 L 張、不套上限', '零股：v＝q*ODDP[t]*(1+side*oddc)；同式'], ['XE:122-140'], 'v1 未定義')
rules['E-14'] = R('整股成交：K5T 與參與上限', 'V7X', [
    '參與上限：oa_pred＝OASH*V60p；非有限或<=0→1 張；否則 max(1, floor(0.20*oa_pred/(1000*PC[t])))；超過部分順延',
    'imp＝max(0.001+0.5*SIGp*sqrt(Q/OA), tick(O[t])/O[t])；Q＝成交張數*1000*PC[t]；OA＝OASH*V[t]（當日實際成交值）；張數=0、OA 無效或 SIGp 無效→省略平方根項；壓力 +0.004（加在 max 之後）',
    '成交價＝O[t]*(1±imp)；v＝1000*張數*成交價；手續費 max(20,fr*v)；賣出稅 0.003*v'], ['XE:73-86', 'XE:114-120', 'XE:143-157'], 'v1 只寫 K5T', task=8)
rules['E-15'] = R('零股成交', 'V7X', [
    '量上限：t>=2020-10-26：floor(1.0*ZSH*V60p/ODDP)；更早：floor(0.5*ZSH*V60p/C[t])；V60p 無效→0；超過→只成交上限，其餘順延',
    'oddc：買＝OCB（壓力 *2 再 +0.004）；賣＝OCS（壓力 +0.004）', 'v＝股數*ODDP*(1±oddc)；手續費 max(20,fr*v)；賣出稅 0.003*v'], ['XE:91-112', 'XE:159-175'], 'v1 未定義', task=8)
rules['E-16'] = R('手續費與稅', 'V7X', ['一般：t<2015-01-01→0.001425*0.6；之後 0.001425*0.28；壓力：全期 0.001425*0.6', '每筆最低 20 元；一筆＝每檔每日整股一筆、零股一筆（處置賣出亦各一筆）', '證交稅＝賣出金額*0.003'], ['XE:69-71'], 'v1 未定義', task=8)
rules['E-17'] = R('執行順序與順延', 'V7X', [
    '換股日：U 依 delta 小→大（穩定排序，同值依欄位序）；非名單且在 DISP 的賣出→E-18，未賣完順延；漲跌停擋住（買 O>=LIMUP-1e-6；賣 O<=LIMDN+1e-6）→整筆順延；買單 cost_est(完整 delta)>現金+1e-9→整筆順延；先零股後整股；上限未成交部分順延',
    '順延日：依待成交股數小→大；tr 假者跳過續留；賣量＝min(待賣, round(持股))，<=0 取消；在 DISP→E-18；跌停擋住→當日跳過；否則先零股後整股',
    '順延日買：漲停擋住→跳過；Lmax＝floor(現金/(O*1000*1.05))（現金<=0→0）；L＝min(待買張數,Lmax)；q＝待買%1000；零股金額+手續費 > 現金-L*O*1050→當日不下零股；整股 L 張仍套參與上限',
    '未成交續留到下一個換股日清空'], ['XE:305-359'], 'v1 未定義', task=8)
rules['E-17a'] = R('換股日停牌的非名單持股（讀法 a）', 'V7X', ['tr 假的持股不進入 U：不產生賣單、不產生順延單', '留在帳上（CF 計價），到之後某個換股日當天可成交且仍不在名單時才賣；非換股日復牌不賣', '符合 E-22 時於換股日轉現金'],
                    ['XE:274-276', 'CV:77'], 'v1 與 frozen_spec 都未寫', task=3)
rules['E-18'] = R('處置期賣出', 'V7X', ['適用：賣出日在 DISP 且（換股日）非名單；順延日只看 DISP', '量上限 floor(0.20*V[t]/C[t])；超過順延', '價＝C[t]-tick(C[t])；無衝擊、無零股價差、不受漲跌停', '整股、零股部分各一筆手續費（最低 20）與稅', '款項隔日入帳（cash_late）'], ['XE:177-196'], 'v1 未定義')
rules['E-19'] = R('30 萬 wholelot 格', 'V7X', ['唯一差別：t<2020-10-26 零股量上限＝0（買賣皆不成交、列入順延）', '目標股數仍含零股；need 仍估零股成本；處置賣出仍可賣零股'], ['I20V:24-27'], 'v1 無此格', task=8)
rules['E-20'] = R('對價結算表 cons_main', 'V7X', [
    'events_v2 每列（last_trade 空白跳過；代碼不在面板跳過）：ltU＝last_trade；term＝terminate_date 或 ltU+14 日曆天',
    'lt＝面板中日期<=max(term,ltU) 的最後成交列（無或為最後一列→跳過）；lc＝C[lt]；trig＝lt+1；ti＝第一個日期>=term 的列；pay＝min(ti+20, 最後列)',
    '面板 lt 日期 < ltU-7 天 → price_gap：固定賣出（每股 lc，列 lt+1）',
    'cash：現金請求權（每股 cash，列 pay）', 'swap/conversion/mixed（存續股在面板且有 s）：s＝存續股在 succ_first_trade 當日起第一個有成交且有開盤的列；ratio 有限→股票請求權（比例 ratio，列 s，空窗期每股 lc）；mixed 另加現金（每股 cash，列 pay）與特別股固定賣出（每股 pref_ratio*pref_value，列 s）',
    'swap_ur（存續股在面板、有 s、lt 當日存續股有收盤）：股票請求權，比例＝lc/存續股 C[lt]', 'acq_ur 或股票型但存續股不在面板/無 s：現金請求權（每股 lc，列 pay）', 'unknown：現金請求權（每股 lc，列 pay）；distress：固定賣出歸零（列 lt+1）'],
    ['CE:36-87'], 'v1 只寫 build_settlement(events_v2, main, d=0.057)', task=4)
rules['E-21'] = R('結算流程', 'V7X', [
    '觸發（t==trig 且持股>0）：持股歸零、該檔順延單取消；各部分建立請求權：價值＝股數*每股值；股票型空窗期價值＝股數*lc、q＝股數*比例；可結算列＝max(規則列,t)',
    't>=可結算列：現金型→併入現金（任何日；不計息、不課稅、無手續費）；股票型→存續股持股+q（任何日）；固定賣出型→只在換股日：入帳＝價值*(1-0.003-當日手續費率)（歸零型 0）', '未結算請求權以其價值計入權益'],
    ['CV:48-74'], 'v1 未定義', task=4)
rules['E-22'] = R('表外永久停牌（E23）', 'V7X', [
    '對象：換股日凍結持股且未被 E-21 轉成請求權（不在 events_v2、last_trade 空白、或 E-20 跳過）',
    "lt[c]＝面板最後成交列；代碼重用：delist_official four=='True' 且終止日後仍有成交→lt＝終止日（含）前最後成交列、uu＝終止日後第一個成交列；否則 uu＝∞；只對 lt 日期<=面板末日-30 天者建規則；lc＝C[lt]；ltd＝lt 日期",
    "在 merger_consideration：dd＝delist_date 或 ltd+14 日曆天；type=='cash' 且 cash 有值→f＝cash/lc、rel＝pay_date 或 dd+14 天；type=='failure'→f=0、rel=ltd；其餘 f=1、rel=dd；type=='cash'→MCASH＝(di＝第一個日期>=dd 的列, pi＝min(di+20,最後列), uu)",
    '不在 merger_consideration：f=1、rel＝ltd+14 日曆天', 'XR[c]＝(f, ri＝第一個日期>=rel 的列, uu)',
    '換股日 t：在 MCASH 且 di<=t<uu→轉待撥款＝股數*CF[t-1]*f*(當日 0050 報酬因子)，逐日隨 0050，至 pi<t 的換股日併入現金（當日不乘 0050）；在 MCASH 者不再看 XR',
    '否則在 XR 且 ri<=t<uu→現金+=股數*CF[t-1]*f、持股歸零、無成本；否則續留'], ['XD:110-141', 'CV:80-100'],
    'v1 未定義；o3/V7X 文字「終止日之後第一個換股日」，程式對表外者用「最後成交+14 日曆天」', task=4)
rules['E-23'] = R('壓力成本', 'V7X', ['整股 imp+0.004；零股買 oddc*2+0.004、賣 +0.004；手續費全期 0.001425*0.6'], ['XE:64-71', 'XE:91-99'], 'v1 未定義')
# ---------------- X
rules['X-01'] = R('紙上成本率', 'PAPER', ['一般：買 0.001425*0.6+0.001＝0.001855；賣 0.001425*0.6+0.003+0.001＝0.004855', '壓力：買賣各 0.005', '比例成本；無最低手續費、無整股限制'], ['IA:24-28', 'O18:126'], 'v1 未定義')
rules['X-02'] = R('紙上引擎流程', 'PAPER', [
    't0＝第一個訊號日列；t1＝2026-10-12 列；t0 收盤現金 1.0', '非換股日：市值*=GROW[t]',
    '換股日：tr＝TRADED[t] 且 open 有限；持股到開盤 *=GROW[t]*open/close（tr 假不變）；凍結＝市值>0 且 tr 假（保留）；Vt＝可交易市值+現金；解 T：T*k+買率*Σmax(T_c-h_c,0)+賣率*Σmax(h_c-T_c,0)＝Vt（[0,Vt/k] 二分 60 次取下界）；名單股設 T；現金歸零；開盤→收盤 *=close/open',
    '結算表照 E-20（面板 D-01b，不補 P）；流程同 E-21，差別：股票請求權在換股日以 open 計入、非換股日以收盤計入；固定賣出入帳＝價值*(1-賣率)；表外永久停牌一直凍結'],
    ['CE:99-109', 'CE:143-212'], 'v1 只寫引擎名')
# ---------------- S
rules['S-01'] = R('seeds 平均', 'V7X', ['每格 20 seeds 的日報酬逐日算術平均（忽略 NaN；全為 NaN 當 0）後累乘，起點 1'], ['XS:95-97'], 'v1 未定義', task=8)
rules['S-02'] = R('年化與超額（日曆年化）', 'V7X', [
    'e(j)＝jscan_end_days["AB_V7_B10|J{j}"]；a＝該 j 第一個訊號日', 'CAGR＝(nav[e]/nav[a])^(1/yrs)-1；yrs＝(e-a) 日曆天/365.25；0050 同日期；exc＝CAGR-CAGR_0050',
    'Y19on（主欄）：起點＝第一個 >=2019-01-01 的日（2019-01-02）收盤；終點 e(j)', '三段 S0612/S1318/S19on：起點＝區間內第一個交易日的前一交易日收盤；終點＝區間內最後交易日'],
    ['XS:33-35', 'XS:46-67', 'I20V:41-58'], 'v1 未定義', task=8)
rules['S-03'] = R('t 值', 'V7X', ["窗＝訊號日收盤→下一訊號日收盤，最後一窗到 e(j)；訊號日取自 plans_j7.csv（strat=='AB_V7_B10', seed==0, 同 j）", 'x＝策略窗報酬-0050 同窗報酬；t＝mean/(std(ddof=1)/sqrt(n))；2019 起：窗起點>=2019-01-01'], ['XS:12-18', 'XS:27-30', 'XS:38-43'], 'v1 未定義')
rules['S-04'] = R('j 彙總與逐格取最差', 'V7X', ['med＝13 個 j 的中位數；p10＝numpy.percentile(.,10) 線性內插；超額＝每個 j 的超額再取中位數', '正式數字＝每格每指標取 REAL_opt/p90/carry 的最小值；最差變體可逐格逐指標不同'], ['XS:83-92', 'I20D:13-14'], 'v1 附註 1', task=8)
rules['S-05'] = R('紙上統計', 'PAPER', [
    '窗 245 個（訊號日收盤→下一訊號日收盤；最後到 2026-10-12）；淨值與 0050 取 <= 該日的最後一筆', '日曆年化（主口徑）：prod(1+r)^(365.25/days)-1；days＝末窗終點-首窗起點',
    'ia 口徑（並列）：prod(1+r)^(12/窗數)-1', 't＝mean(r-r50)/std(ddof=1)*sqrt(n)', '分段依窗起點月：SEG1 2006-01~2012-12；SEG2 2013-01~2018-12；SEG3 2019-01 起；HOLD 2024-01 起；SEL ~2019-04；VAL 2019-05 起',
    '拿掉最好 3 年：依窗起點年加總超額，去掉最大 3 年後重算 t', '逐格取最差：每指標取 opt/p90/carry 最小'], ['O18:138-172', 'I20P:70-77', 'I20D:11-12', 'FS:88-90', 'FS:133-136'], 'v1 只有 ia 文字', task=8)
rules['S-06'] = R('報表格', 'V7X', ['100 萬一般；100 萬壓力；300 萬一般；1000 萬一般；30 萬一般（標準）；30 萬一般（wholelot）；30 萬壓力'], ['I20V:29-30'], 'v1 只有 3 級距 × 2 成本')
# ---------------- N
N = [
 ('N-01', '紙上與 V7X 外國股定義不同', '紙上 is_foreign（P-07）只用於放回層；V7X 用 FORMASK（L-07）；6 檔不同：1262,1592,2936,6422,6452,8480；紙上正式名單含 6452/1262/6422', ['PC:118-120', 'V4:66-79']),
 ('N-02', '營收百分位分母不同', '紙上 opt/p90 ÷|el|；V7X ÷|el_rev|', ['PC:211-213', 'VL:135-137']),
 ('N-03', '營收底表不同', '紙上 tw_revenue 年增率+B1 只補缺；V7X R_FIX 缺口格改 B1 或設缺', ['O18:73-85', 'VL:63-94']),
 ('N-04', '金融遮罩不同', '紙上 FINMASK；V7X 的 ok 用 FIN2MASK', ['V2:84-87', 'V4:71-75']),
 ('N-05', '紙上面板只在 scratchpad', 'D-01b 不在研究資料夾；缺 2024-01-02~02-05；不補 P', ['IA:18']),
 ('N-06', '兩個 0050 檔', '紙上統計用 fslib 副本；V7X 用 jonatw 檔；sha 不同', ['FS:88-90', 'XD:76']),
 ('N-07', 'TAIEX 日曆與面板不一致', '2026-09-20 只在 TAIEX→j7 無 2026-09、j>=8 的 2026-09 訊號日比面板日計數早一列；2018-01-13 只在面板且非 GOOD→j2 無 2018-01', ['ST:112-121', 'VL:35']),
 ('N-08', 'cause 的月份', '用 d 的日曆月、不看回退；回退後才缺值者可能為 UNKNOWN 而不放回', ['PC:141-165']),
 ('N-09', 'A_UB 非 AVAIL 含無證據', '沒有任何時間戳證據者在觸發日視為未公告', ['AV:88-93', 'O18:49']),
 ('N-10', '「2019 起」兩個起點', 'Y19on＝2019-01-02 收盤；S19on＝2018 最後交易日收盤', ['XS:50-56', 'I20V:41-44']),
 ('N-11', 't 的訊號日來源', 'REAL 名單的 t 用 orig 名單 seed 0 的訊號日', ['XS:13-18']),
 ('N-12', '同 delta 執行順序', '穩定排序，同值依欄位序（o19 G3）', ['XE:306']),
 ('N-13', '順延日零股現金判定的張數', 'L＝min(待買張數, 現金張數上限)，未先套參與上限（o19 G4）', ['XE:350-355']),
 ('N-14', '股數浮點殘差', '順延單股數可為浮點；q＝d%1000 未取整（o19 G5）', ['XE:351']),
 ('N-15', 'need(T) 估計口徑', '全部張數、不套上限、用當日實際成交值；漲跌停股仍估成本', ['XE:284-296']),
 ('N-16', '衝擊用當日成交值', 'F4 只改參與上限；衝擊 OA 仍用 V[t]', ['XE:81', 'XE:116']),
 ('N-17', 'wholelot 實際語意', '目標仍含零股；處置賣出仍可賣零股', ['I20V:24-27']),
 ('N-18', 'MC 亂數相依性', '依 numpy default_rng（PCG64）與欄位序；PCOL 用現行 stock_list 產業', ['ST:137-152']),
 ('N-19', 'held 定義', '看上一期名單，非實際持股；|el|<50 不更新', ['V2:199-215']),
 ('N-20', '引擎 V60 補值', 'ffill(limit=3) 連 GOOD 日 NaN 也補（o19 G1）', ['XD:61']),
 ('N-21', 'vrank 收盤條件', '用前值補收盤 CF（o19 G2）', ['XD:64']),
 ('N-22', '引擎讀快取', 'x_cache.npz 存在即直接讀，不重算', ['XD:82-87']),
 ('N-23', 'M1_paper 的組成', '2021-09-23 後聯集 disposal_5y（date 當公告日）與 universe_announce', ['I20P:24-33']),
 ('N-24', '名單股當日不可成交', '引擎未防護（名單規則保證 e 有開盤）', ['XE:276']),
 ('N-25', '現金請求權入帳後閒置', '入帳日非換股日時閒置到下一換股日', ['CV:65-66']),
 ('N-26', '換股日清空順延單', '上一期未成交單一律作廢', ['CV:77', 'XE:222']),
 ('N-27', '處置賣出用引擎 DISP', '不看公告日、不隨 M2 改變', ['XD:93-107']),
 ('N-28', '紙上壓力成本', '每邊固定 0.5%，不另計稅費', ['IA:26-28']),
]
for i, t, d, c in N:
    rules[i] = R(t, 'new_undefined', [d], c, 'v1 與既有文字規格都未寫')

data_files = {k: dict(path=v['path'], exists=v['exists'], sha256=v.get('sha256'), bytes=v.get('bytes'), header=v.get('header')) for k, v in VI.items()}
code_files = {
 'IA': 'independent_audit/ia_core.py', 'V2': 'main_v2/v2_core.py', 'PC': 'o_rev_avail_ds_pit_count/pclib.py', 'AV': 'o_rev_avail_ds_pit_count/avail.py',
 'O18': 'o_delist_rev_backfill_allpit/o18lib.py', 'VL': 'o_delist_rev_backfill_allpit/v7x_lib.py', 'V4': 'o_v4_revenue_patch_official/v4lib.py',
 'DRJ': 'o_disposal_rejudge/drj_core.py', 'O3': 'o_infra_v3_rebuild/o3_core.py', 'ST': 'o_stagger_tranche/st_core.py', 'ENV': 'o_v7_listing_exec_engine/v7env.py',
 'XD': 'o_exec_engine_rewrite_v7x/x_data.py', 'XE': 'o_exec_engine_rewrite_v7x/x_exec.py', 'XS': 'o_exec_engine_rewrite_v7x/x_stats.py',
 'CE': 'o_engine_delist_consideration/cz_engine.py', 'CV': 'o_engine_delist_consideration/cz_v7x.py', 'I20V': 'integrate_o20/run_v7x.py',
 'I20P': 'integrate_o20/run_paper.py', 'I20G': 'integrate_o20/gen_v7x_plans.py', 'I20D': 'integrate_o20/decide.py', 'FS': 'o_forward_sequential_v2/fslib.py'}
import hashlib
code = {k: dict(path=p, sha256=hashlib.sha256((AG / p).read_bytes()).hexdigest()) for k, p in code_files.items()}
out = dict(
    spec='SPEC v2（草案，待總監凍結）：ALLPIT_REAL 名單層＋V7X 可執行引擎', written_taipei='2026-11-12', human_readable='SPEC_v2.md',
    supersedes=dict(file='o_delist_rev_backfill_allpit/SPEC.json', sha256='f5e141b072053791e24fec904ee1f3d30a95d0b331dbe0924c36192046ffc404', plus='FINDINGS.md 開頭附註 1～6'),
    scope='描述正式程式的實際行為；不含新算的績效數字；未重跑程式驗證',
    task_item_map={'1': ['P-03', 'P-07', 'P-08', 'P-09', 'L-03', 'L-07', 'L-09', 'L-10'], '2': ['E-02', 'E-04', 'E-05'], '3': ['E-17a', 'E-11'], '4': ['E-22', 'E-20', 'E-21'],
                   '5': ['C-09'], '6': ['P-04', 'P-04a', 'L-05'], '7': ['P-05', 'L-06', 'E-08'],
                   '8': ['S-01', 'S-02', 'S-03', 'S-04', 'S-05', 'S-06', 'E-07', 'E-09', 'E-12', 'E-14', 'E-15', 'E-16', 'E-17', 'E-19', 'E-23', 'L-01', 'L-04', 'P-09', 'L-10']},
    constants=dict(
        N_columns=1249, switch_rev_table='2021-03-01', paper_end='2026-10-12', limit_switch='2015-06-01', fee_switch='2015-01-01', odd_switch='2020-10-26', disposal_hi='2021-09-23',
        exempt_days=['2017-08-03', '2018-07-12'], exempt_range=['2024-02-15', '2024-03-15'], special_deadlines={'2019-02': '2019-02-15', '2023-04': '2023-04-12'},
        deadline_case_special={'2019-01': 'D>S', '2023-03': 'D>S', '2020-01': 'D=S'},
        GOV=['TRUNC', 'IFRS13', 'POST13', 'POST13_R5', 'PRE12', 'HOLE', 'NEVER_DOM'], cause_priority=['TRUNC', 'IFRS13', 'POST13', 'POST13_R5', 'PRE12', 'HOLE', 'NEWLIST', 'BASE_LE0'],
        fin_words=['銀', '證', '金控', '證金', '產保', '壽', '票券', '產險', '人壽', '投信', '合庫', '信託'],
        tiers={'30萬': 3e5, '100萬': 1e6, '300萬': 3e6, '1000萬': 1e7}, cpi_2006=CPI['2006'], cpi_2026=CPI['2026'], cpi_ratio=CPI['2006'] / CPI['2026'],
        engine_cfg=dict(mode='GROW', model='K5T', stress_odd_extra=True, rank_right=True, f1=True, f2=True, f3=0.5, f4=True),
        fee=dict(base=0.001425, discount_pre2015=0.6, discount_post2015=0.28, discount_stress=0.6, min_per_fill=20.0), tax_sell=0.003,
        paper_cost=dict(normal_buy=0.001425 * 0.6 + 0.001, normal_sell=0.001425 * 0.6 + 0.003 + 0.001, stress_each=0.005),
        model_spec7=MS, jscan_end_days_AB_V7_B10={k.split('|')[1]: v for k, v in ENDS.items() if k.startswith('AB_V7_B10|')},
        seeds=list(range(20)), js=list(range(13)), variants_paper=['ALLPIT_REAL_opt', 'ALLPIT_REAL_p90', 'ALLPIT_REAL_carry'], variants_v7x=['REAL_opt', 'REAL_p90', 'REAL_carry'],
        numpy_version_official_run='2.5.3', pandas_version_official_run='3.0.5', python_version='3.13.17'),
    data_files=data_files, code_files=code, rules=rules,
    known_limits=['未重跑程式驗證「照規格重建＝程式輸出」', 'V7X REAL 名單驗收目標不在磁碟上，需保管員重產封存', '紙上面板 D-01b 只在 scratchpad', 'N-01～N-04 為兩條管線間的不一致，未裁定'])
(HERE / 'SPEC_v2.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print('rules', len(rules), sorted(rules)[:5], 'bytes', (HERE / 'SPEC_v2.json').stat().st_size)
