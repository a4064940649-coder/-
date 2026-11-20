#!/usr/bin/env python3
"""把 8 項的結果、方案、建議指令彙整成 plan.json（數字都從本資料夾的結果檔讀，不手抄）。python3 -I -B make_plan.py"""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
J = lambda p: json.loads((H / p).read_text(encoding='utf-8'))
tv = J('tables/twt49u_verify.json'); man = J('twt49u_candidate/twse-exright/MANIFEST.json'); dr = J('tables/drill_2026-11_list.json')
r10 = J('tables/revenue_ready_2026-10.json'); e5 = J('e50/e50_compare.json'); ew = J('e50/whatif_official.json'); ev = J('e50/e50_events.json')
rw = J('reduce/whatif_reduce.json'); gp = J('gaps/gap_scan.json'); pt = J('patches.json')
OLD = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/tsd'
NEW = '/home/user/-/research/tw_stock_selection/raw/tsd_daily'
A = 'research/tw_stock_selection/agents/'
pct = lambda x: round(x * 100, 4)
E0 = 'E0|W10_70'
plan = {
 '說明': '工程 eng_audit_fixes_2026-11：數據總稽核待修清單的方案與驗證結果。正式檔、raw/、引擎程式都沒有改；本資料夾以外沒有新增或修改任何檔案。',
 '日期': '2026-11-21', 'tsd_head': tv['tsd_head'],
 '1_TWT49U補檔': {
  '結論': '可以用官方日 K 倉庫事件檔產生同格式月檔。候選檔已產生，沒有寫進 raw/twt49u、沒有動 SHA256SUMS。',
  '轉換程式': 'convert_exright.py（只做字串處理：欄名對應 stock_id→code、pre_close→prev_close、kind→type；數字去掉尾端 0；依日期、代號排序；gzip 固定 mtime＝0，同輸入永遠同位元組）',
  '重疊期驗證': {'比對期間': f"{tv['repo_first_readable']}～{tv['archive_last']}", '比對天數': tv['overlap_days_compared'], '封存列數': tv['archive_rows_on_compared_days'],
              '逐欄相同列數': tv['rows_identical'], '不同列數': tv['rows_different'], '只在倉庫的日子': tv['days_only_in_repo'], '只在封存的日子': tv['days_only_in_archive_within_repo_range'],
              '整月檔解壓後逐位元相同': tv['month_files_bytes_identical_after_gunzip'], '整月檔不同': tv['month_files_different'],
              '說明': '11,781 筆含 ETF 等非四碼代號；稽核說的 9,870 筆是四碼普通股口徑。',
              '沒比到的': f"倉庫事件檔本機讀不到內容 {len(tv['repo_unreadable_days'])} 天（2015-01-05～2015-07-01 共 40 天、2019-09-25 一天；部分複製沒抓這些檔），這些日子無法比對；封存 2014-01-16～2015-07-01 也沒有對照。和要補的 2026-10-30 以後無關。"},
  '候選檔': {'位置': A + 'eng_audit_fixes_2026-11/twt49u_candidate/twse-exright/2026/2026-11.csv.gz', **man['files'][0],
          '四碼事件': '6165（11-02）、8467（11-04）、2414（11-06）、2344（11-18）、5434（11-18）、6924（11-19）、4562（11-20）；前 6 個就是稽核列的 6 個，4562 是 11-20 新增',
          '2026-10 月檔': '不用改：倉庫在 2026-10-30 沒有事件檔，10 月最後一個事件日是 10-29，和封存相同'},
  '名單演練（暫存，不是正式名單）': {'做法': '2026-11 名單用同一份資料重跑三次：a＝現行封存；c＝封存＋候選月檔',
                       'a 等於正式 2026-11 名單（前 20 名、F1 全欄位）': dr['a_equals_official_list'], 'c 的前 20 名、F1 與 a 全欄位相同': dr['a_equals_c_top20_f1'], '換股計畫相同': dr['plan_a_equals_c'],
                       '差別': f"修正的除權息格 {dr['exright_a']['n_twt49u']} → {dr['exright_c']['n_twt49u']}；「封存還沒涵蓋的日子」{len(dr['exright_a']['days_after_twt49u_in_window'])} 天 → {len(dr['exright_c']['days_after_twt49u_in_window'])} 天"},
  'make_list.py 缺資料時會不會自動改讀倉庫事件檔': {
   '答案': '不會。',
   '依據': ['make_list.py 第 260～268 行 load_twt49u()：只讀 --twt49u-dir 底下的 */*.csv.gz；資料夾空的就停（第 263～264 行）',
          '第 368～375 行 compute()：ev = load_twt49u(twt49u_dir)，沒有任何讀 data/universe/exright 的程式；封存最後一天之後的交易日只記在 meta.exright.days_after_twt49u_in_window',
          '第 436～448 行：前 20 名、F1、持股近 60 日的可疑格子標「TWT49U 封存還沒涵蓋這天，未修正」——只提示、不修正',
          '會自動改讀倉庫事件檔的是另一支：xq_scripts/site/live_tr.py 第 64～80 行 load_events()（倉庫優先、封存備援），只用在網站 live.json 的含息帳戶，不影響名單'],
   '後果': '12/11 出名單前若沒補封存，10-30 以後的除權息日報酬記 0，季線與 RET20 會漏這幾筆股利（2026-11 名單驗過沒影響；12 月名單不保證）。'},
  '若採用要執行的指令': [
   '# 0. 先把官方日 K 倉庫更新到最新（連網步驟，由每日排程做）',
   f'cd /home/user/- && python3 -I -B {A}eng_audit_fixes_2026-11/convert_exright.py verify <官方日K倉庫> research/tw_stock_selection/raw/twt49u/twse-exright /tmp/twt49u_verify.json   # rows_different 必須是 0',
   f'python3 -I -B {A}eng_audit_fixes_2026-11/convert_exright.py build <官方日K倉庫> <暫存資料夾> 2026-10-30   # 會產生 2026/2026-11.csv.gz（12/11 當天還會有 2026-12.csv.gz）',
   'cp <暫存資料夾>/2026/2026-1[12].csv.gz research/tw_stock_selection/raw/twt49u/twse-exright/2026/   # 只複製 .csv.gz，不要複製 MANIFEST.json',
   "cd research/tw_stock_selection/raw/twt49u && find twse-exright -name '*.csv.gz' | sort | xargs sha256sum > SHA256SUMS",
   '# 套用 patches.json 類別「1-TWT49U 補檔」那一筆（README 的涵蓋日期）；之後跑一次每日回測檢查（agents/audit_daily/daily_check.py）確認 48.76%／381 筆不變'],
  '採用前要知道': ['正式回測到 2026-10-22，封存多了 11 月檔不會改正式數字（面板只到 END＝2026-10-22）；本次沒有實際放進 raw 後重跑驗證，請採用後跑每日回測檢查確認',
             '月檔是「到產生當天為止」的部分月份；每次出名單前要重產覆蓋，SHA256SUMS 也要重算',
             '倉庫「沒有事件的日子沒有檔」，所以無法從檔案本身分辨「沒事件」和「還沒抓」；倉庫的 _fetched.json 本機讀不到。以倉庫最新日 K 日期 ≥ 訊號日當作已更新的依據',
             '來源變成單一來源（官方日 K 倉庫）；原封存來源是另一個公開鏡像。兩者在重疊期逐筆相同，但 11 月以後沒有第二來源可對']},
 '2_tsd固定路徑方案（只交方案，沒有搬）': {
  '現況': {'路徑': OLD, '大小': '105MB', '性質': 'jb7k6295r7-bot/tw-stock-data 的部分複製（blob:none、淺層），每日排程更新；每月名單、網站資料、月營收新月份都讀它'},
  '建議路徑': NEW,
  '理由': '和同一倉庫的舊封存 raw/jb7k6295r7-bot_tw-stock-data 放一起、同樣不進版本控制；程式都用參數收路徑，改指令就好。',
  '.gitignore 要加': 'research/tw_stock_selection/raw/tsd_daily/   （比照 .gitignore 第 32 行的 research/tw_stock_selection/raw/jb7k6295r7-bot_tw-stock-data/）',
  '建議步驟': [f'1. 每日排程沒在跑的時候：cp -a {OLD} {NEW}',
           f'2. 核對：git -C {NEW} rev-parse HEAD 與舊的相同；python3 -I -B check_revenue_ready.py --month 2026-10 --tsd {NEW} 印【可以出名單】',
           f'3. 舊路徑改成符號連結，每日排程不用改也照常運作：mv {OLD} {OLD}.bak && ln -s {NEW} {OLD}',
           '4. .gitignore 加一行；每日排程的提示文字改用新路徑（排程文字不在倉庫內，要總監改）；確認跑過一天後刪 .bak',
           '5. 套用 patches.json 類別「2-tsd 固定路徑」',
           '6. restore_env.sh 加一段：新路徑不存在時 git clone --filter=blob:none --no-checkout --depth=1 https://github.com/jb7k6295r7-bot/tw-stock-data 到新路徑（需要連網）'],
  '已驗證': '在暫存區做過「cp -a 複製＋符號連結」：HEAD 相同、本機物件清單相同、透過連結跑 check_revenue_ready 與 convert_exright 結果相同（候選月檔 sha256 相同）。沒有動真正的 tsd。',
  '寫死舊路徑的腳本（要一起改）': [
   {'檔案': A + 'audit_live_2026-11/audit_0050.py', '行': 10, '內容': "TSD = '<舊路徑>'"},
   {'檔案': A + 'audit_live_2026-11/audit_list.py', '行': 24, '內容': "TSD = '<舊路徑>'"},
   {'檔案': A + 'audit_published_numbers/audit_core.py', '行': 18, '內容': "TSD = '<舊路徑>'"},
   {'檔案': A + 's36_announce_entry/prep.py', '行': 14, '內容': "TSD = Path('<舊路徑>')"},
   {'檔案': A + 'audit_data_integrity/common.py', '行': '10、14', '內容': "SCR_OLD = Path('…/scratchpad')；TSD = SCR_OLD / 'tsd'"}],
  '寫「<暫存>/tsd」的文件': [
   {'檔案': 'xq_scripts/site/state/README.md', '行': '12、19、121、127、130'},
   {'檔案': A + 's18_live_list/FINDINGS.md', '行': '51、55、60、63'}],
  '用參數收路徑、程式不用改的': ['xq_scripts/site/monthly_rebalance.py 第 277 行 --tsd', 'xq_scripts/site/build_site_data.py 第 34 行（第 9 個參數）', 'xq_scripts/site/test_live_tr.py 第 22 行', 'make_list.py --repo'],
  '不用改的': '各研究的 json 紀錄檔（lists/2026-11*.json、data_manifest.json 等）裡的舊路徑是當時的來源紀錄，保留原樣。',
  '找不到的': '倉庫裡沒有任何檔案把每日排程的 tsd 路徑寫死（o_forward_infra_live/daily_step0.sh 不讀 tsd）。排程寫死的路徑在排程的提示文字裡，不在倉庫內，我看不到。',
  '重要限制': '固定路徑只防「暫存區被清」。容器整個回收時，不進版本控制的資料夾一樣會消失，要靠重新 clone（需要連網、上游倉庫還在）。真正的保險是把出名單必需的小檔存進版本控制：見第 3 項「快照」。'},
 '3_月營收人工備援': {
  '2026-10 彙總表': {'讀得到': True, '位置': 'data/mops/revenue_hist/2026-10_twse.csv（git HEAD）', '欄位': r10['columns'], '列數': r10['n_rows'], '四碼普通股': r10['n_4digit'], '營收大於 0': r10['n_revenue_positive'],
                '單位': '千元（和營收主檔 tw_revenue.csv 的 revenue_k_twd 相同）', '首見紀錄': r10['seen_snapshots'], '倉庫 HEAD commit 時間': r10['tsd_head_time'],
                '涵蓋率A（名單產生器口徑，截止 2026-11-11 22:00 台北）': r10['coverage_A']['value'], '缺': r10['coverage_A']['missing'],
                '和正式 2026-11 名單 meta 對照': '截止前 1,075 家、涵蓋率 0.9839，與 lists/2026-11_exright.json 的 n_used、coverage 相同',
                '缺的是誰': '15 家金融保險股在截止時間後才出現在首見紀錄（2816、2832、2850、2851、2880～2887、2891、2892、5880）；另一個「0」是營收主檔裡的髒代號。金融股申報期限較晚，12/11 可能一樣',
                '涵蓋率B（對最新日 K 上市四碼）': r10['coverage_B']['value']},
  '2026-11 彙總表': '倉庫 HEAD 還沒有（最新是 2026-10）；12 月初上游才會開始有。check_revenue_ready.py --month 2026-11 目前回「檔案不存在」、結束碼 2（已試）。',
  '工具': {'check_revenue_ready.py': '給營收月份，回報：檔案在不在、欄位、家數、上市涵蓋率（兩種口徑）、缺哪些公司、營收 ≤ 0 的公司、首見紀錄第一份／最後一份時間、倉庫 commit 時間、持股有沒有缺。結束碼 0＝可以出名單、1＝還不行、2＝檔案不存在或讀不到。只讀、不連網。',
         'manual_revenue_to_repo.py': '把人工下載的彙總表 csv 轉成名單產生器讀得懂的小 git 倉庫（make_list.py 只會從 --repo 的 git HEAD 讀彙總表，同月份後面的倉庫覆蓋前面的）。'},
  '12/11 當天先跑': f'python3 -I -B {A}eng_audit_fixes_2026-11/check_revenue_ready.py --month 2026-11 --tsd <官方日K倉庫> --cutoff 2026-12-11T14:00:00Z --holdings xq_scripts/site/state/holdings_after_2026-11.csv',
  '人工備援演練結果（暫存，不是正式名單）': {
   '做法': '把倉庫的 2026-10 彙總表改成公開資訊觀測站開放資料的欄名（公司代號、公司名稱、營業收入-當月營收、資料年月＝民國 11510）當成「人工下載檔」，只留截止前已看到的 1,075 家 → manual_revenue_to_repo.py 轉成小倉庫 → make_list.py 加第三個 --repo 重跑 2026-11 名單',
   '結果': f"名單讀的是人工小倉庫（meta.rev.supplement 的 source 指向它）；前 20 名、F1 全欄位與正式 2026-11 名單相同：{dr['a_equals_b']}；月份填錯（拿 10 月檔當 11 月）會被擋下",
   '沒演練到的': ['真的從公開資訊觀測站下載：這個環境連不到；欄名是照開放資料格式寫的，沒有拿真檔測過。真檔若欄名不同，工具會列出實際欄位並停下，不會亂猜',
              'monthly_rebalance.py 只接受兩個倉庫（第 290 行 a.repos = [raw, a.tsd]），不能直接帶人工小倉庫']},
  '人工備援要走的流程': ['1. 到公開資訊觀測站下載上市公司 2026-11 月營收彙總表 csv（人工、在能連網的電腦），放進環境',
                '2. python3 -I -B manual_revenue_to_repo.py <下載檔> 2026-11 <新的空資料夾>',
                '3. python3 -I -B check_revenue_ready.py --month 2026-11 --tsd <官方日K倉庫> --manual-csv <新資料夾>/data/mops/revenue_hist/2026-11_twse.csv   # 要印【可以出名單】',
                '4. 出名單：目前只能直接呼叫 make_list.py list … --repo <raw> --repo <tsd> --repo <新資料夾>（state/README.md 第 12 行的手動寫法再多一個 --repo）',
                '5. 建議改 monthly_rebalance.py（兩行）：第 277 行附近加 p.add_argument("--extra-repo", action="append", default=[])；第 290 行改 a.repos = [raw, a.tsd] + a.extra_repo。改完 plan／settle 都要帶同一個 --extra-repo'],
  '注意': ['人工檔沒有首見紀錄，--rev-cutoff 對這個月不會做時點檢查；請在截止時間前下載',
         '另一個缺口：2026-10 營收也不在營收主檔（tw_revenue.csv 只到 2026-09），12 月名單同時需要彙總表的 2026-10 與 2026-11 兩個月；人工備援時兩個月都要有（2026-10 倉庫已有）',
         '快照建議（待辦「每月名單定稿時存營收彙總表快照」）：名單定稿時把用到的彙總表與首見紀錄從倉庫取出存進 raw/（進版本控制、附 sha256），約 100KB／月；指令：git -C <tsd> show HEAD:data/mops/revenue_hist/2026-11_twse.csv > research/tw_stock_selection/raw/revenue_hist_snapshots/2026-11_twse.csv（本次沒做，raw/ 不能動）']},
 '4_0050含息基準': {
  '做法': 'build_e50_tr.py：兩個官方日 K 倉庫的 0050 原始收盤＋上市除權息事件檔＋ETF 分割事件檔，事件日倍數＝前收 ÷ 參考價（分割取整數倍），含息淨值＝收盤 × 累乘倍數。不用 jonatw、不用倉庫現成的還原因子。',
  '事件': {'用到': ev['n_events_used'], '說明': '22 次除息＋2025-06-18 一拆四；每個事件的前收都和日 K 前一成交日收盤相符', '和 TWT49U 封存的 0050 事件比': e5['events_vs_twt49u_archive'],
         '讀不到的事件檔': ev['event_files_unreadable'], '讀不到的影響': '這兩天 TWT49U 封存都沒有 0050 事件（封存 22 筆與自建 22 筆完全相同），不影響'},
  '逐日比對（2015-07-01～2026-10-22，2,760 天）': {
   '逐年累計差最大值_百分比': {y: v['max_abs_cum_diff_pct'] for y, v in e5['daily']['by_year'].items()},
   '結論': '2015～2024 逐日相同（累計差最大 0.00008%，是四位小數的進位）；2025-07-21 除息日起差 −0.02%；2026-07-21 起開始有差，2026-08-12 起連續漂移',
   '當日報酬差超過 0.02 個百分點的天數': e5['daily']['n_days_ret_diff_gt_0p02pp'], '第一天': e5['daily']['first_such_day'],
   '累計差（現用 ÷ 自建 − 1）_百分比': e5['cum_diff_pct_at'], '說明': '2026-08-11 到 10-22 之間多漂 0.58%（−0.04 → +0.54），和稽核的 +0.58% 相同；差的那些天多半剛好是一個跳動點（0.05 元），像是來源取到的不是收盤價',
   'jonatw 檔在期間內的非交易日列': e5['jonatw_non_trading_rows_in_period']},
  '逐年（%）': {y: {'現用': v['now'], '自建': v['new'], '差': v['diff_pp']} for y, v in e5['yearly'].items()},
  '對 navs_official 的 0050 欄': {'期末現在': e5['navs_official_0050_col']['end_now'], '期末改用後': e5['navs_official_0050_col']['end_new'], '差': e5['navs_official_0050_col']['end_diff'],
                          '全期年化%': [round(e5['navs_official_0050_col']['cagr_full_now'], 2), round(e5['navs_official_0050_col']['cagr_full_new'], 2)],
                          '2019起年化%': [round(e5['navs_official_0050_col']['cagr_2019_now'], 2), round(e5['navs_official_0050_col']['cagr_2019_new'], 2)],
                          '最大回撤%': [round(e5['navs_official_0050_col']['maxdd_now'], 2), round(e5['navs_official_0050_col']['maxdd_new'], 2)],
                          '有變的天數（差超過 1 元）': e5['navs_official_0050_col']['n_days_changed_gt_1']},
  '對 S75 的 0050 期末': {'現在': round(e5['s75_0050']['end_now']), '改用後': round(e5['s75_0050']['end_new']), '年化%': [round(e5['s75_0050']['cagr_now'], 2), round(e5['s75_0050']['cagr_new'], 2)],
                    '說明': e5['s75_0050']['note']},
  '稽核沒提、但要一起知道的：策略淨值也會變': {
   '原因': '引擎的 0050 底倉與閒置資金也是用同一條 0050 含息序列算的（s10lib.py 第 302～307、359～360 行），不是只有對照欄。',
   '試算做法': 'run_official_e50_whatif.py：正式管線原樣複製到暫存區重跑，只換 0050 序列；對照組（不換）與正式檔逐日差 0、結果檔相同',
   '對照組通過': ew['control'],
   '現行策略（單一路徑）': {k: [pct(v['now']) if abs(v['now']) < 5 else round(v['now']), pct(v['new']) if abs(v['new']) < 5 else round(v['new'])] for k, v in ew['E0_W10_70'].items()},
   '20 條路徑中位數_2019起%': [pct(ew['E0_paths_summary']['now']['cagr19_median']), pct(ew['E0_paths_summary']['new']['cagr19_median'])],
   '各欄期末差_百分比': {k: round(v['end_diff_pct'], 3) for k, v in ew['columns'].items()},
   '白話': '改用後現行策略 2019 起 48.76% → 48.71%、全期 34.40% → 34.37%、期末 2,832 萬 → 2,825 萬；0050 本身 2019 起 28.91% → 28.82%。兩邊都往下，策略對 0050 的超額幾乎不變。',
   '候選檔': 'e50/navs_official_e50fix.csv.gz、e50/res_official_e50fix.json（不是正式檔）'},
  '要改哪些程式哪幾行（只交方案）': [
   {'檔案': A + 'o_ext_canslim_xs/cslib.py', '行': '15、181～183', '改法': '第 15 行 E50_CSV 保持預設不變（舊研究逐位重現）；正式管線用開關指到新檔。第 181 行讀 date、adj 兩欄，新檔照同格式給就不用改程式'},
   {'檔案': A + 'official_nav/run_official.py', '行': '37～39 之後加一行；146～149', '改法': "加 cslib.E50_CSV = '<新檔>'（和三個開關放一起）；第 146 行驗收目標 0.4876 與第 149 行門檻要換成新數字（試算 0.4871），第 147～148 行對 deadline_engine_fix 的比對會不再相同，要改成對新的基準檔"},
   {'檔案': A + 'audit_daily/daily_check.py', '行': '4、55', '改法': '每日回測檢查寫死 48.76／34.40／46.40／381，要換成新數字（試算 48.71／34.37／46.35／381）'},
   {'檔案': 'xq_scripts/site/make_profit.py', '行': '20、53', '改法': '0050 來源預設值改成新檔；來源說明文字跟著改'},
   {'檔案': A + 's1_shortterm_leaderboard/stlib.py', '行': '63～68', '改法': '不用改（0050 還原開盤＝含息收盤 × 原始開盤 ÷ 原始收盤，比值不變）'},
   {'檔案': '新資料檔', '行': '—', '改法': '建議放 research/tw_stock_selection/raw/e50_official_tr/etf_0050_tr.csv（date,adj）＋sha256＋README：2015-07-01 起用 build_e50_tr.py 的 tr_close；之前沿用 jonatw 並在接點等比例銜接（凍結一份快照，不再跟上游變動）。每日排程更新官方日 K 後重產'},
   {'檔案': '要跟著改的已公布數字', '行': '—', '改法': 'README 第 12 節正式表、網站 risk／yearly／nav、S75 的 0050 欄與 1,291 萬、每日回測檢查的基準。屬於改正式數字，要產品擁有者同意、獨立複製後才換'}],
  '限制': ['2008-01～2015-06 沒有官方日 K 事件檔可用，仍是 jonatw 單一來源', 'S75 的策略淨值（V_BRIDGE 5,308 萬）沒有重跑，只換了 0050 對照欄；依正式引擎試算的比例推估會少約 0.2%，未驗證',
         'cslib.END＝2026-10-22 是因為 jonatw 檔只到那天；改用自建序列後 0050 可以到最新一天，但加權指數檔（taiex_twii.csv）也只到那天，END 要延後得另外處理']},
 '5_減資恢復買賣日': {
  '引擎現在怎麼處理': [
   'cslib.py 第 160～169 行：參考價＝收盤 − 漲跌價差；當日報酬＝收盤 ÷ 參考價 − 1。官方日 K 在減資恢復買賣日的漲跌價差是 0 → 參考價＝當天收盤、報酬＝0',
   'cslib.py 第 101～132 行 exright_fix_chg()：只處理 (a) TWT49U 除權息日（第 121～127 行）與 (b) 前一天沒成交、漲跌 ≤ 10.5% 的日子用前一成交收盤（第 128～129 行）。沒有讀減資恢復買賣參考價（TWTAUU）',
   '規則 (b) 的副作用：減資後參考價和減資前收盤差不到 10.5% 時（多半是退還股款的小比例減資），引擎把「前收 → 收盤」整段當成漲跌，等於把參考價調整也算成報酬',
   's10lib.py 第 162 行 val_now()：開盤賣出市值＝前收市值 × 開盤 ÷ 參考價（REF）；第 364、367 行每日市值 ×（1＋R）；第 185、247、322 行漲跌停判斷也用同一個參考價。所以減資日當天開盤賣出、漲跌停判斷都會受影響',
   's1_shortterm_leaderboard/stlib.py、s2lib～s9lib 是同一段寫法；make_list.py 第 271～296 行是逐行複製（名單的季線、RET20 用同一個還原序列）'],
  '事件盤點（2015-07-01～2026-10-22，上市、當天有成交）': {
   '事件數': rw['events_in_period']['n'], '種類': rw['events_in_period']['by_kind'], '引擎的處理': rw['events_in_period']['by_engine_case'],
   '事件前收與面板前一成交收盤不符': rw['events_in_period']['n_pre_close_mismatch'], '正確報酬超過 ±10.1%': rw['events_in_period']['n_true_R_beyond_10pct'],
   '平均偏差（正確 ÷ 引擎 − 1，%）': {k: round(v, 2) for k, v in rw['events_in_period']['mean_close_factor_pct_by_case'].items()},
   '白話': '312 個事件裡 227 個報酬記 0（平均少算 0.6%，有正有負）；85 個被當成一般漲跌（平均多算 4.7%，因為減資後參考價通常比減資前收盤高）。',
   '明細': 'reduce/event_cells.csv（每個事件：引擎報酬、正確報酬、參考價）；事件表 tables/reduce_parvalue_events.csv'},
  '正式策略（E0｜W10_70 標準路徑）持股期間碰到的': {
   '筆數': len(rw['held_events_E0_standard']), '明細': rw['held_events_E0_standard'],
   '白話': '2015-07 起 389 筆交易只碰到 1 筆：8249 菱光 2024-10-21 減資（退還股款）恢復買賣，當天開盤賣出。引擎用「當天收盤」當參考價，賣出市值等於前一天市值；正確是前收 48.95 → 參考價 53.27 → 開盤 58.50（漲停），多 9.82%。這筆賣出實得約少算 7.9 萬（當時帳戶 1,030 萬，約 0.76%）。稽核寫的約 15 萬是 2008 起連續帳戶（規模較大）的同一筆。',
   '明細檔': 'reduce/held_events.csv、reduce/trades_E0.csv.gz（389 筆交易明細，正式結果檔原本沒有）'},
  '試算：改用事件參考價後的正式數字': {
   '做法': 'run_official_reduce_whatif.py：正式管線複製到暫存區重跑，在 exright_fix_chg 之後多一步「上市、當天有成交、在減資或面額變更事件表 → 漲跌價差＝收盤 − 事件參考價」；對照組與正式檔逐日差 0',
   '對照組通過': rw['control'], '套用格數': rw['whatif']['n_cells_patched'],
   '現行策略（單一路徑）': {k: [pct(v['now']) if abs(v['now']) < 5 else round(v['now']), pct(v['new']) if abs(v['new']) < 5 else round(v['new'])] for k, v in rw['whatif'][E0]['std'].items()},
   '逐年有變的（%）': rw['whatif'][E0]['yearly'],
   '20 條路徑中位數_2019起%': [pct(rw['whatif'][E0]['paths_summary']['now']['cagr19_median']), pct(rw['whatif'][E0]['paths_summary']['new']['cagr19_median'])],
   '每月名單': f"129 個訊號月的買進名單全部相同（有變的月份：{rw['whatif']['targets_changed_months']}）", '交易': f"筆數相同；1 筆出場日晚一天（{rw['whatif']['trades_only_now']} → {rw['whatif']['trades_only_new']}，帳戶變大後碰到 5% 成交值上限分批賣）",
   'XQ 版（網站 XQ版 欄，A10）': {k: [pct(v['now']) if abs(v['now']) < 5 else round(v['now']), pct(v['new']) if abs(v['new']) < 5 else round(v['new'])] for k, v in rw['whatif']['XT|W10_70']['std'].items()},
   'XQ 版說明': 'XQ 版影響較大（2019 起 45.09% → 45.61%，2021 年 90.5% → 96.1%），它持有過別的減資事件；本次沒有逐筆列 XQ 版',
   '白話': '現行策略 2019 起 48.76% → 48.91%（+0.15）、全期 34.40% → 34.49%、2024 起 70.46% → 70.93%、最大回撤 −32.12% → −31.61%、期末 2,832 萬 → 2,854 萬。偏保守的方向，和稽核判斷一致。',
   '候選檔': 'reduce/navs_official_reducefix.csv.gz、reduce/res_official_reducefix.json（不是正式檔）'},
  '修正方案（只交方案）': [
   '1. 資料：比照 TWT49U 封存，建 raw/twtauu/（上市減資恢復買賣）與面額變更事件封存，欄位 date,code,prev_close,ref_price,reason；來源＝官方日 K 倉庫 data/universe/reduce、parvalue（本次整理成 tables/reduce_parvalue_events.csv，上市 2015-04-17 以後的事件檔本機讀得到），附 sha256 與 README',
   '2. 引擎：cslib.py 加第四個開關 REDUCE_FIX（預設 False，舊研究逐位重現），開啟時在 exright_fix_chg() 第 129 行之後加規則 (c)：上市、當天有成交、當天在事件表 → CHG＝收盤 − 事件參考價（蓋掉 (b) 的結果）；info 加 n_reduce_parvalue',
   '3. 名單產生器 make_list.py 第 271～296 行是 exright_fix_chg 的逐行複製，要同步加（否則名單的季線／RET20 和回測不一致）',
   '4. 正式管線 run_official.py 第 37～39 行加 cslib.REDUCE_FIX = True；驗收目標、每日回測檢查基準、README 與網站數字跟著換（同第 4 項，屬於改正式數字）',
   '5. 建議和第 4 項（0050 序列）合併成一次改版、一次獨立複製；兩項合計的試算沒有跑（分開跑是 −0.04 與 +0.15，方向相反、大小不同，合計約 +0.1，未驗證）'],
  '限制': ['「正確報酬＝收盤 ÷ 事件參考價 − 1」把退還的股款當成在恢復買賣日以參考價再投入（和現行權息、現增的近似相同），不是把現金另外記帳',
         '股數沒有跟著減資縮（引擎用市值記帳，股數只影響 5% 上限與零股費用，影響很小）',
         '2015-05 以前的上市減資事件檔本機讀不到（4 天）；2008～2015 連續面板（S75 等）沒有事件參考價來源，本次沒有處理、也沒有重跑 S75',
         '上櫃減資（otcreduce）不在範圍（股池只有上市）；上櫃轉上市的股票在上櫃期間的事件沒有套用']},
 '6_代號重用與無價格': {
  '確認': {'2432': '連續面板裡 2008-08-25 是舊公司最後一天（收 56.8），2023-05-31 另一家公司用同代號掛牌（28.6），中斷 3,633 個交易日。正式面板（2015 起）只有新公司，沒有這個問題',
         '11 檔': 'S77 列的 11 檔在連續面板都找到中斷，全部到 2010-01-04 才恢復；起點各不同：6176（2007-05-14）、8033（2007-06-20）、6205／6283／8021／9955（2008-01-18）、5533（2008-04-29）、2059（2008-06-24）、4108（2008-07-15）、6226（2008-11-07）、8103（2009-09-28）',
         '原因': '來源資料的洞：raw/pre2015 的價格檔沒有這段，本機另一份面板 data_panel 在 2008～2009 也只有 0～5 天有值。不是公司停牌',
         '為什麼會凍結': 's10lib.py 第 314 行（stlib 第 197 行、s2lib～s9lib、s37lib 同一句）：下市＝「這一天不在面板，而且之後再也沒出現」。代號之後又出現就不算下市，部位停在最後市值、沒有開盤價不能賣，一直卡到代號再出現那天，再用新價格÷舊參考價接上'},
  '另外掃到的中斷（≥ 60 個交易日後又出現）': {
   '連續面板': {'中斷數': gp['連續面板_2007_2026（S75～S83、R75 用）']['n_gaps'], 'S77 沒列的代號': gp['連續面板_2007_2026（S75～S83、R75 用）']['new_codes']},
   '正式面板': {'中斷數': gp['正式面板_2015_2026（正式引擎用）']['n_gaps'], '代號': gp['正式面板_2015_2026（正式引擎用）']['new_codes']},
   '說明': '新掃到的多半是真的長期停止買賣後恢復（例如 6131 2018-05 → 2019-03、1435、2025），引擎「卡住到恢復」對這種情況是合理的；2491（2008-09 → 2009-03）、2361（2010-12 → 2011-04）在 data_panel 有價格，是不是來源的洞沒有查清。明細 gaps/panel_gaps_cont.csv、panel_gaps_official.csv'},
  '影響範圍（哪些研究的哪些交易碰到）': {
   '結果': f"有存交易明細的 8 份檔案裡，持有期間跨到任何一段中斷的交易：{gp['n_affected_trades_total']} 筆",
   '查了哪些': {k: (v if isinstance(v, str) else {'交易筆數': v['n_trades'], '跨中斷': v['n_trades_crossing_gap'], '買過 2432': v['n_trades_2432_any'], '2010 前買過那 11 檔': v['n_trades_known11_before_2010']}) for k, v in gp['studies'].items()},
   '白話': 'S75 兩個版本、S76、S77（現行與新規則）、S79 凍結格、S81 重點格、S82、稽核逐筆重算、正式引擎，都沒有任何一筆交易碰到。已公布的數字（S75 的 5,308 萬／3,883 萬、正式 48.76%）不受影響。',
   '沒查到的': 'S77（147 格）、S78（288 格）、S79（110 格）、S80（66 格）、S81（192 格）的搜尋格子大多沒有存交易明細，個別格子有沒有碰到無法從檔案確認（S77 的 FINDINGS 說「持有中遇到會被凍結」，表示搜尋格子裡發生過）。這些題目都判定不通過、沒有採用任何格子，不影響現行規則。要確認得重跑各題，這次沒做。'},
  '處理方案（只交方案）': [
   '1. 代號重用：在連續面板的組裝程式（s75_continuous_2008_2026/build_panel.py 第 104～106 行組 MK 的地方；S77／S78／S80 的 build 都是複製它）加一張明列的「代號重用表」（目前一列：2432，舊公司最後一天 2008-08-25、新公司第一天 2023-05-31），把舊公司那一段搬到另一欄（例如代號 2432 保留給新公司、舊公司用內部欄名），兩家公司各自一欄。舊公司那欄在 2008-08-25 之後不再出現 → 引擎自然判成下市、以最後市值轉現金。不改共用引擎',
   '2. 11 檔無價格：這是資料洞，補不到價格前不該讓策略買進後卡兩年。方案 A（建議）：組面板時把這 11 檔在中斷之前的列也拿掉（等於 2010-01-04 才進股池），損失是 2007～2008 少 11 檔可選、偏保守、說明文件註記。方案 B：找到 2008～2009 的價格補進 raw/pre2015（官方來源這個環境連不到；本機兩個來源都沒有）',
   '3. 防呆：連續面板的 panel_check.json 加一項「中斷 ≥ 60 個交易日後又出現的代號清單」，和審過的白名單（gaps/panel_gaps_cont.csv）比對，多出來就停；正式面板比照（每日回測檢查可順便跑 code_gap_scan.py）',
   '4. 不建議直接改 s10lib 第 314 行的下市判斷（例如「消失超過 N 天就算下市」）：真的長期停牌後恢復的股票（6131 等）會被提前用停牌前價格變現，而且 s2lib～s9lib、stlib、s37lib 有同一句，改一處會讓舊研究不能逐位重現'],
  '預估對數字的影響': '已存明細的交易 0 筆受影響 → 修正後 S75、正式引擎的已公布數字預期不變（沒有實際改面板重跑驗證）。'},
 '7_文件修改建議': {'檔案': 'patches.json', '筆數': pt['筆數'], '各類筆數': {}, '說明': '每筆有 檔案、原文片段、建議改成、理由；產生時逐筆檢查原文片段在檔案裡剛好出現一次。3 筆有前提（總監決定採用後才套）。build_site_data.py 的 4 筆是引擎檔，工程沒有改，列給總監；改完要重產 data/risk.json。',
              '沒有給 patch 的': ['xq_scripts/策略利潤圖.html：用修正前淨值畫的（期末 2,758 萬）。不是改幾個字能修的，要用 navs_official.csv.gz 重產整張圖，或在標題下加一行「本圖是 2026-11-15 修正前的淨值，正式數字見網站」；倉庫裡沒找到產生它的程式，請總監決定',
                            'README 第 12 節正式表的 9.30 vs 9.31、「差」欄 −0.14／−0.29（精確 −0.15／−0.30）：稽核建議可不改，沒有給',
                            'nav.json meta「每週取最後一個交易日」說明不精確：文字在 build_site_data.py／make_profit.py 裡（引擎檔），沒有給；建議改「每週取週一到週五的最後一個交易日」',
                            'README 第 30 節警戒線表其他格（E 列 5 年 10%→9%、86%→85%，B 列 9%→8%、35%／49%→34%／48%，A 歷史最差 −3.0→−3.5）：節內下方段落已用文字說明，表格本身沒改；稽核只點了 E 列 1 年那一格，本次只改那一格，其餘請總監決定要不要一併改到表上'],
              '請總監確認的一處': '規格補充第 6 條寫「5% 成交值上限買進、賣出都適用，賣出超過會分批賣」，是照程式寫的（s10lib.py 第 325～327 行）；稽核摘要寫的是「只限買進」，兩者不同。'},
 '8_風險表加一列（只交方案，沒有改程式）': {
  '位置': 'xq_scripts/site/build_site_data.py 的 RISK 表（第 596～632 行），建議插在「賠的平均」之後、「晚1天少」之前（第 608 行後），和其他「少」類的列放一起',
  '要加的那一列': ['成本_0050也加滑價少', 0.7, '個百分點',
             '正式回測對 0050 的買賣只收半個跳動點、手續費（2.8 折、最低 20 元）和 ETF 證交稅 0.1%，沒有另外加個股用的每邊 0.3% 滑價（設計如此：0050 流動性高）。如果 0050 的買賣也加 0.3% 滑價，2019 年起每年約少 0.7 個百分點。這是清潔室重寫引擎（R0）的敏感度：48.81% → 48.06%（單一路徑，精確差 0.75；不是用正式引擎算的）',
             'agents/r0_cleanroom_engine/results.json（sensitivity_from_final、sens_etf_noslip_vs_official；2026-11-20 數據稽核部 R0）'],
  'Python 寫法': '    ("成本_0050也加滑價少", 0.7, "個百分點", "正式回測對 0050 的買賣只收半個跳動點、手續費（2.8 折、最低 20 元）和 ETF 證交稅 0.1%，沒有另外加個股用的每邊 0.3% 滑價（設計如此：0050 流動性高）。如果 0050 的買賣也加 0.3% 滑價，2019 年起每年約少 0.7 個百分點。這是清潔室重寫引擎（R0）的敏感度：48.81% → 48.06%（單一路徑，精確差 0.75；不是用正式引擎算的）", "agents/r0_cleanroom_engine/results.json（sensitivity_from_final、sens_etf_noslip_vs_official；2026-11-20 數據稽核部 R0）"),',
  '數字核對': '用 R0 的淨值檔重算：out/final_navs.csv 2019 起 48.062%、out/sens_etf_noslip_navs.csv 48.811%，差 0.749 個百分點（四捨五入到一位是 0.7）。正式 48.756% 對 R0 加滑價版 48.062% 差 0.694。',
  '加了之後要一起做的': ['重產 data/risk.json（列數 35 → 36）', 'README 第 12 節或第 32 節如有「風險表 35 列」字樣要改', '稽核部的風險表對帳（audit_published_numbers）下次會多一列，狀態是「未用正式引擎重現」',
                'index.html 的風險表若是照項目名稱寫死順序或分組，要確認新列會顯示（本次沒有檢查頁面）'],
  '提醒': '數值欄慣例：其他「少」類的列用正數（晚1天少 3.7）；這裡照慣例寫 0.7，不寫 −0.7。'},
}
import collections
plan['7_文件修改建議']['各類筆數'] = dict(collections.Counter(p['類別'] for p in pt['patches']))
(H / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding='utf-8')
print('ok', len(json.dumps(plan, ensure_ascii=False)))
