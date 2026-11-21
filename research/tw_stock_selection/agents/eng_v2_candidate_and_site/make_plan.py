#!/usr/bin/env python3
"""產生 plan.json 與 patches.json（工程師不能寫 .md，文件修改一律交 patches.json；每筆「原文片段」寫檔前檢查在檔案中剛好出現一次）。
用法：python3 -I -B make_plan.py"""
import json, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent; RES = AG.parent; REPO = RES.parents[1]
SITE = REPO / 'xq_scripts' / 'site'
V = HERE / 'verify'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
J = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
base = J(V / 'baseline_check.json'); arch = J(V / 'archives_check_vs_audit.json'); bits = J(V / 'panel_bits.json'); ml = J(V / 'make_list_check.json')
mle = J(V / 'make_list_reduce_effect.json'); num = J(V / 'site_numbers_now_vs_v2.json'); th1 = J(V / 'thermo_check_asof_2026-10-22.json'); th2 = J(V / 'thermo_check_latest.json')
thr = J(V / 'thermo_reduce_fix_effect.json'); pg = J(HERE / 'page_test' / 'page_test_result.json'); rl = J(V / 'rules_check.json')
v2 = J(AG / 'official_nav' / 'out_v2' / 'res_official_v2.json'); acc = v2['acceptance']
thermo = J(SITE / 'data' / 'thermo.json'); e50i = J(RES / 'raw' / 'e50_tr_official' / 'build_info.json'); rdi = J(RES / 'raw' / 'twse_reduce' / 'build_info.json')
t_rows = thermo['rows']; t_oct = [r for r in t_rows if r['月份'] == '2026-10'][0]; t_nov = t_rows[-1]
risk = {r['項目']: r for r in num['risk_rows']}
TSD = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/tsd'

README_E50 = f"""# 0050 含息序列（自建，候選）

- 內容：`etf_0050_tr.csv`，欄位 `date,adj`（交易日、含息還原收盤）。{e50i['first']}～{e50i['last']}，{e50i['n_days']} 個交易日；只列 0050 有成交的日子（2025-06-11～17 分割換發停牌不列，引擎讀檔時沿用前一天）。
- 算法：含息還原收盤＝官方日 K 原始收盤 × 事件日累乘倍數。除息日倍數＝前收 ÷（前收 − 息值）；分割日倍數＝前收 ÷ 分割後參考價，取最接近的整數。等於股利在除息日以參考價再投入。
- 事件：`etf_0050_events.csv`（{e50i['n_events']} 個：{e50i['n_dividend']} 次除息＋2025-06-18 一拆四），每個事件的前收都和日 K 前一成交日收盤相符。除息事件是 TWT49U 封存（`raw/twt49u`）與官方日 K 倉庫 `data/universe/exright` 的聯集（兩邊 22 筆完全相同）；分割來自倉庫的 `data/universe/etfsplit`。
- 來源：日 K 以研究中心封存 `raw/jb7k6295r7-bot_tw-stock-data`（git HEAD `{e50i['heads']['raw'][:12]}`）為主，它沒有的日子用官方日 K 倉庫 tsd（git HEAD `{e50i['heads']['tsd'][:12]}`）；兩邊重疊的日子收盤完全相同。不連網、不用 jonatw、不用倉庫現成的還原因子。
- 產生程式：`agents/eng_v2_candidate_and_site/build_raw_archives.py`（`python3 -I -B build_raw_archives.py [輸出根目錄] [tsd 路徑]`；加 `--check` 會和稽核對照檔 `agents/r_fix_e50_reduce/data/e50_tr_mine.csv` 逐筆比對）。重產兩次結果逐位元相同。`build_info.json` 記錄來源與筆數。
- 驗證：與稽核獨立重建的序列 2,890 天逐筆相同（差 0）；稽核另和工程第一版互相核對（`agents/r_fix_e50_reduce/FINDINGS.md`）。
- 為什麼要換：現用的 `/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv` 在 2015-07～2025-07-18 與本檔逐日相同，但 2025-07-21、2026-07-21 兩次除息各少算約 0.02%，2026-08-12～10-14 有 20 天與官方收盤差 1～4 個跳動點，累積多出 +0.58%（根因查不到）。
- 誰在讀：`agents/o_ext_canslim_xs/cslib.py` 的 `E50_SOURCE`（**預設 None＝不讀，仍用現用檔，舊結果逐位相同**）；候選正式管線 `agents/official_nav/run_official_v2.py` 設成 `cslib.E50_OFFICIAL_CSV` 才讀。**產品擁有者尚未採用，正式數字還沒有換。**
- 限制：只從 2015-01-05 起（官方日 K 的起點）；2008～2014 沒有官方日 K 與事件檔，連續帳戶（S75 等）那一段仍是舊來源。回測引擎的最後一天仍是 `cslib.END＝2026-10-22`（加權指數檔只到那天），本檔雖然到 {e50i['last']}，引擎只用到 END。
- 更新：官方日 K 倉庫更新後重跑產生程式（會整檔重產並重寫 `SHA256SUMS`）；重跑前後 `etf_0050_tr.csv` 舊日期的值不應改變。
"""
README_RED = f"""# 上市減資／面額變更恢復買賣事件（封存，候選）

- 內容：`twse_reduce_events.csv`，欄位 `date,code,prev_close,ref_price,reason`（恢復買賣日、代號、停止買賣前收盤、恢復買賣參考價、原因）。{rdi['first']}～{rdi['last']}，{rdi['n_rows']} 列：彌補虧損 {rdi['by_reason'].get('彌補虧損')}、退還股款 {rdi['by_reason'].get('退還股款')}、面額變更 {rdi['by_reason'].get('面額變更')}。只有上市，沒有上櫃。
- 來源：官方日 K 倉庫 tsd（git HEAD `{rdi['heads']['tsd'][:12]}`）的 `data/universe/reduce`（減資恢復買賣）與 `data/universe/parvalue`（面額變更；原檔 reason 空白，這裡填「面額變更」）。原始 {rdi['n_rows_raw']} 列，同一天同一代號重複的 {rdi['n_dropped_duplicates']} 列已去掉（留最後一筆）。
- 讀不到的事件檔：2015-01-23、2015-03-20、2015-04-10、2015-04-16 四天（本機沒有這幾個檔的內容），都在回測起點 2015-07-01 之前。
- 產生程式：`agents/eng_v2_candidate_and_site/build_raw_archives.py`（同 0050 含息序列；`--check` 會和稽核抽出的 `agents/r_fix_e50_reduce/data/ev_reduce_tsd.csv`、`ev_parvalue_tsd.csv` 逐筆比對：320 列逐欄相同）。
- 用途：官方日 K 在減資／面額變更恢復買賣日的漲跌價差是 0，引擎把當日報酬記成 0，或（前一天沒成交、差距在 10.5% 內時）把參考價調整當成一般漲跌。開啟修正後，上市、當天有成交、當天在本表的格子 → 漲跌價差＝收盤 − 本表參考價。2015-07～2026-10-22 會套用 311 格。
- 誰在讀：`cslib.py` 的 `REDUCE_FIX`（**預設 False＝不讀，舊結果逐位相同**；`cslib.load_reduce()`、`cslib.reduce_fix_chg()`）；候選正式管線 `run_official_v2.py`；名單產生器候選版 `agents/s18_live_list/live_surge/make_list_v2.py --reduce-fix`（預設關）；網站溫度計 `xq_scripts/site/make_thermo.py --reduce-fix`（預設關）。**產品擁有者尚未採用。**
- 已知限制：退還股款視為在恢復買賣日以參考價再投入（近似；173 個事件平均差 −0.02 個百分點）；恢復買賣日當天沒成交的 3 個事件（1418 2017-11-27、1529 2018-12-22、1538 2024-12-30）與因颱風停市順延的 3 個事件（2314、3055 2016-09-28；2323 2019-09-30）不會套用；上櫃減資不在範圍。
- 更新：官方日 K 倉庫更新後重跑產生程式。每月出名單前若有新的減資事件要先重產（和 TWT49U 月檔同一個時間點）。
"""

plan = {
    '說明': '工程：把兩項已獨立複製通過的修正（A＝0050 含息序列自建、B＝減資／面額變更恢復買賣日用事件參考價）做成可一鍵切換的候選版，正式數字沒有換；網站新增「環境溫度計」「照做須知」兩個區塊，檔案準備好、沒有上傳。',
    '日期': '2026-11-21',
    '一句話': '候選版全部驗收通過（48.86%／34.46%／−31.61%／381 筆／46.42%，與稽核 A＋B 逐日差 0）；開關全關時正式結果逐位不變；2026-11 名單開不開減資修正完全一樣；網站兩個新區塊的資料檔與頁面已備好、本機測試通過，等總監上傳。',
    '任務一_候選版': {
        '1_新資料檔': {
            '0050 含息序列': dict(資料夾='research/tw_stock_selection/raw/e50_tr_official/', 檔案=['etf_0050_tr.csv（date,adj）', 'etf_0050_events.csv（23 個事件與倍數）', 'build_info.json', 'SHA256SUMS'],
                                sha256={f: sha(RES / 'raw' / 'e50_tr_official' / f) for f in ('etf_0050_tr.csv', 'etf_0050_events.csv')}, 期間=f"{e50i['first']}～{e50i['last']}（{e50i['n_days']} 天）"),
            '減資／面額變更事件': dict(資料夾='research/tw_stock_selection/raw/twse_reduce/', 檔案=['twse_reduce_events.csv（date,code,prev_close,ref_price,reason）', 'build_info.json', 'SHA256SUMS'],
                                  sha256={'twse_reduce_events.csv': sha(RES / 'raw' / 'twse_reduce' / 'twse_reduce_events.csv')}, 列數=rdi['n_rows'], 去掉的重複列=rdi['n_dropped_duplicates']),
            '產生程式': 'research/tw_stock_selection/agents/eng_v2_candidate_and_site/build_raw_archives.py',
            '重產指令': f'python3 -I -B research/tw_stock_selection/agents/eng_v2_candidate_and_site/build_raw_archives.py research/tw_stock_selection/raw {TSD} --check',
            '驗證': dict(與稽核對照檔逐筆比對=arch, 重產兩次逐位元相同=True, 說明='0050：2,890 天的 adj 與 r_fix_e50_reduce/data/e50_tr_mine.csv 完全相同（差 0）；事件：320 列與稽核的 ev_reduce_tsd.csv＋ev_parvalue_tsd.csv（去重複後）逐欄相同。'),
            '注意': ['「只讀 tsd」做不到：tsd 是部分下載的倉庫，2,895 個日 K 檔裡有 1,577 個本機沒有內容，所以日 K 以既有封存 raw/jb7k6295r7-bot_tw-stock-data 為主、tsd 補最新的 15 天（和稽核的做法相同）。',
                     'tsd 的除權息事件檔有 41 個本機讀不到，0050 的除息事件用 TWT49U 封存補齊（封存 22 筆＝倉庫讀得到的 22 筆）。',
                     '任務說明寫的欄位是 5 欄，面額變更在原檔沒有原因，reason 欄我填了「面額變更」。'],
            'README 內容（請總監存成 raw/e50_tr_official/README.md）': README_E50,
            'README 內容（請總監存成 raw/twse_reduce/README.md）': README_RED,
        },
        '2_引擎開關': {
            '檔案': 'research/tw_stock_selection/agents/o_ext_canslim_xs/cslib.py', 'sha256_改之後': sha(AG / 'o_ext_canslim_xs' / 'cslib.py'), 'sha256_改之前': 'c808e43eb5e46c4e4787774fbce2f74486eda52f7eafc3130a113c61a33f06e4',
            '加了什麼': ['REDUCE_FIX = False、REDUCE_CSV（事件封存路徑）', 'E50_SOURCE = None（None＝現用的 E50_CSV）、E50_OFFICIAL_CSV（自建檔路徑）', 'load_reduce()、reduce_fix_chg()（規則 c：上市、當天有成交、在事件表 → 漲跌價差＝收盤 − 事件參考價，接在除權息修正之後）',
                       'Panel(reduce_fix=None)：None＝照 REDUCE_FIX；Panel 多三個屬性 reduce_fix、reduce_info、e50_source（只是紀錄，不影響計算）'],
            '怎麼開': 'cslib.REDUCE_FIX = True；cslib.E50_SOURCE = cslib.E50_OFFICIAL_CSV（要在 load 面板之前設）',
            '開關全關的檢查（做法同 daily_check.py，但不寫 log.csv）': base,
            '面板陣列逐位比對（改之前 對 改之後，各 22 個陣列的 sha256）': bits,
            '結論': '開關全關：48.76%／34.40%／381 筆／與 navs_official.csv.gz 逐日差 0／20 條路徑中位數 46.40%／結果檔與 res_official.json 相同，全部通過；沒有還原修改。',
            '程式': ['verify_baseline.py', 'verify_panel_bits.py'],
        },
        '3_候選正式管線': {
            '檔案': 'research/tw_stock_selection/agents/official_nav/run_official_v2.py', '指令': 'cd research/tw_stock_selection/agents/official_nav && GIT_NO_LAZY_FETCH=1 python3 -I -B run_official_v2.py',
            '輸出': ['official_nav/out_v2/navs_official_v2.csv.gz', 'official_nav/out_v2/res_official_v2.json', 'official_nav/out_v2/trades_E0.csv、trades_XT.csv、trades_XT_B11.csv（標準路徑交易明細）'],
            '驗收（程式內建，不符會結束碼非 0）': acc,
            '白話': f"2019 起 {acc['since2019']*100:.2f}%、全期 {acc['full']*100:.2f}%、最大回撤 {acc['maxdd']*100:.2f}%、已出場 {acc['n_trades_closed']} 筆、20 條路徑中位數 {acc['paths_median']*100:.2f}%、0050 的 2019 起 {acc['e50_since2019']*100:.2f}%；"
                    '與稽核 r_fix_e50_reduce/out/AB/navs.csv.gz 四欄逐日差 0，每月名單、標準路徑指標、20 條路徑、附帶數字都與稽核結果檔相同。',
            '沒有動到正式檔': dict(檢查='跑完後 xq_scripts/site/navs_official.csv.gz 與 official_nav/out/res_official.json 的 sha256 不變', sha256=(V / 'official_files_before.sha256').read_text().split('\n')[:2]),
            '執行時間與記憶體': f"{v2['secs']} 秒、峰值 {v2['peak_mb']} MB",
        },
        '4_名單產生器同步': {
            '做法': 'make_list.py 沒有改。新增 live_surge/make_list_v2.py（＝make_list.py 原樣複製＋ --reduce-fix、--reduce-csv，預設關），另附 diff 檔 eng_v2_candidate_and_site/make_list_reduce_fix.diff（套到 make_list.py 上就等於 make_list_v2.py，已驗證）。',
            'sha256': dict(make_list_py=ml['make_list_sha256'], make_list_v2_py=ml['make_list_v2_sha256']),
            '測試輸入': f"--month 2026-11 --repo <raw 封存> --repo <tsd> --rev-cutoff {ml['rev_cutoff']}；兩種情境：空倉 100 萬、策略帳戶（上月回測持股，底倉 1,995 股，由已發布的 2026-11_backtest_holdings_exright.json 還原）",
            '開關關': {k: v['off_identical_files'] for k, v in ml['cases'].items()},
            '開關關_結論': '兩種情境的 .csv／.json／.md 三個輸出檔都與現行 make_list.py 逐位元相同。',
            '開關開': {k: dict(前20名代號相同=v['top20_codes_same'], F1代號相同=v['f1_codes_same'], 前20名欄位差異=v['top20_field_diffs'], F1欄位差異=v['f1_field_diffs'], 換股計畫差異=v['plan_diffs'], 其他差異=v['other_top_level_diffs'],
                             賣出=v['sells'], 續抱=v['keeps'], 買進=v['buys'], 視窗內事件格=v['reduce_meta']['n_reduce_parvalue'], 落在前20名或F1或持股的事件格=v['reduce_meta']['cells_in_top20_f1_holdings']) for k, v in ml['cases'].items()},
            '開關開_結論': '2026-11 名單的前 20 名、F1（只有 1802）、換股計畫（賣出、續抱、買進、0050、現金）逐欄都沒有變；差異清單是空的。輸出只多一個 meta.reduce 區塊與報告裡一行說明。',
            '開關確實有作用': dict(說明='視窗（2025-07-18～2026-11-12）內 34 個事件格所屬的股票，開了之後還原價有變（31 檔的距季線數值不同），但沒有任何一檔翻過季線、名次也沒變；所以 2026-11 名單不受影響是「這個月剛好沒碰到」，不是開關沒作用。',
                             n_cells=mle['n_cells'], n_stocks_changed=mle['n_stocks_dist_ma60_changed'], n_above_ma60_flipped=mle['n_above_ma60_flipped'], n_rank_changed=mle['n_rank_changed']),
            '和已發布的 2026-11 名單比': ml['vs_published_2026-11'],
            '程式': ['verify_make_list.py', 'verify/make_list_check.json', 'verify/make_list_reduce_effect.json'],
        },
        '5_切換清單（產品擁有者說「採用」之後，總監照順序做；每一步附驗證）': [
            dict(步驟=1, 做什麼='確認兩份封存是最新的並重驗', 指令=f'python3 -I -B research/tw_stock_selection/agents/eng_v2_candidate_and_site/build_raw_archives.py research/tw_stock_selection/raw {TSD} --check；把本檔「README 內容」兩段存成兩個 README.md',
                 驗證='最後一行「比對」pass＝True；(cd raw/e50_tr_official && sha256sum -c SHA256SUMS)、(cd raw/twse_reduce && sha256sum -c SHA256SUMS) 都 OK'),
            dict(步驟=2, 做什麼='正式管線換成候選版', 檔案='research/tw_stock_selection/agents/official_nav/run_official.py',
                 改法=['最省事：用 run_official_v2.py 的內容取代 run_official.py，再改回正式輸出位置——',
                       'run_official_v2.py 第 26 行 OUT = HERE / \'out_v2\' → HERE / \'out\'',
                       '第 152、159 行 OUT / \'navs_official_v2.csv.gz\' → SITE / \'navs_official.csv.gz\'（並把 run_official.py 第 21 行的 SITE = Path(\'/home/user/-/xq_scripts/site\') 加回來）',
                       '第 172 行 res_official_v2.json → res_official.json；檔頭說明的「候選」字樣與第 105 行附近 res 的 candidate／note 兩個欄位拿掉',
                       '若只想在原檔上改：run_official.py 第 39 行之後加 cslib.REDUCE_FIX = True、cslib.E50_SOURCE = cslib.E50_OFFICIAL_CSV；第 105 行 flags 加這兩項；第 14 行說明與第 143～149 行驗收（目標 0.4876、對 deadline_engine_fix 的比對）換成 v2 的驗收段（目標 0.4886／0.3446／−0.3161／381／0.4642，對 r_fix_e50_reduce/out/AB）'],
                 指令='cd research/tw_stock_selection/agents/official_nav && GIT_NO_LAZY_FETCH=1 python3 -I -B run_official.py',
                 驗證='印出 acceptance pass＝True；新的 xq_scripts/site/navs_official.csv.gz 解壓後與 official_nav/out_v2/navs_official_v2.csv.gz 逐日相同；先備份舊的 navs_official.csv.gz（建議另存成 navs_official_pre_v2.csv.gz，比照 navs_exright.csv.gz 保留舊版）'),
            dict(步驟=3, 做什麼='每日回測檢查的基準', 檔案='research/tw_stock_selection/agents/audit_daily/daily_check.py', 改法=['第 4 行說明：48.76%→48.86%、34.40%→34.46%、46.40%→46.42%（381 筆不變）', '第 55 行 nums_ok：48.76→48.86、34.40→34.46、46.40→46.42'],
                 指令='python3 -I -B research/tw_stock_selection/agents/audit_daily/daily_check.py <當天日期>', 驗證='輸出 result＝PASS、nav_diff_max＝0（這一步會寫 log.csv，是正式紀錄）'),
            dict(步驟=4, 做什麼='名單產生器同步（否則名單的季線／RET20 和回測口徑不一致）', 檔案='research/tw_stock_selection/agents/s18_live_list/live_surge/make_list.py',
                 改法=['套 diff：cd live_surge && patch make_list.py < ../../eng_v2_candidate_and_site/make_list_reduce_fix.diff（結果＝make_list_v2.py；檔頭「這份檔（make_list_v2.py）」那句改成一般說明）',
                       '把預設改成開：--reduce-fix 的 default=False 改 True，並加一個 --no-reduce-fix（照 --exright-fix 的寫法）',
                       'xq_scripts/site/monthly_rebalance.py 第 175 行 ml.compute(D, a.month, exright=True, …, deadline=True) 加 reduce=True（第 52 行的命令列走 CLI 預設，不用改）',
                       'xq_scripts/site/build_site_data.py 第 118 行加 rd_on = bool((M.get("reduce") or {}).get("on"))，第 122、129 行兩個 ml.compute(…) 加 reduce=rd_on',
                       's18_live_list/code.sha256 第 1 行換成新的 make_list.py 雜湊'],
                 驗證='python3 -I -B eng_v2_candidate_and_site/verify_make_list.py 的邏輯改成「新 make_list.py --no-reduce-fix 對舊版逐位元相同」；12/11 出名單前先重產減資事件封存（步驟 1）；s18_live_list/validate.py 重跑通過'),
            dict(步驟=5, 做什麼='網站利潤圖與逐年的 0050 來源', 檔案='xq_scripts/site/make_profit.py', 改法=['第 20 行預設值 /home/user/jonatw/finlab-txd/data/raw/etf_0050.csv → /home/user/-/research/tw_stock_selection/raw/e50_tr_official/etf_0050_tr.csv', '第 4 行說明、第 53 行來源文字跟著改'],
                 指令='python3 -I -B xq_scripts/site/make_profit.py', 驗證='輸出的 profit.json 與 eng_v2_candidate_and_site/verify/profit_v2_candidate.json 相同（我已用參數指定新來源試產過）；0050 逐年 2025 年 36.8→36.9、2026 年 67.2→66.3'),
            dict(步驟=6, 做什麼='網站風險表（build_site_data.py 的 RISK 常數）', 檔案='xq_scripts/site/build_site_data.py',
                 會變的列與新值=[dict(行=ln, 項目=k, 現在=risk[k]['現行'], 新值=risk[k]['v2']) for ln, k in ((597, '回撤_單一路徑'), (598, '回撤_20條中位數'), (601, '落後0050最多'), (602, '落後時間比例'), (604, '年化_全部'), (605, '年化_不買最賺5檔'), (607, '賺的平均'))],
                 不變的列=[dict(行=ln, 項目=k, 值=risk[k]['v2']) for ln, k in ((600, '0050_回撤'), (603, '最賺5筆占比'), (606, '勝率'), (608, '賠的平均'))],
                 說明文字要跟著改的列=['第 597～608 行各列說明裡的「除權息＋期限修正後」要加上這次修正與舊值', '第 621 行（實盤可預期_保守）說明裡的「網站上的 48.76%」→ 48.86%', '第 623 行（期限前視高估）說明裡的「年化 48.76%」→ 48.86%',
                              '第 627～630 行（照做_*）說明裡的「46.4%」基準、第 631～632 行（小資金_*）說明裡的「0050 的 28.9%」：要等 S64、S62 用新口徑重跑才知道', '第 634 行 risk 的 meta 說明'],
                 要另外重跑才有新值的列=['第 599、610～614 行（回撤_壞的時候、晚 1／2 天、準時／有時晚可預期）：official_nav/resync 用新引擎重跑', '第 615～618、624 行（警戒線現值、正常範圍）：S27', '第 620～621 行（實盤可預期）：S42',
                               '第 625～626 行（參數挑選）：S46', '第 627～632 行：S64、S62', '第 609 行（0050 也加滑價）：R0 敏感度'],
                 數字出處='eng_v2_candidate_and_site/verify/site_numbers_now_vs_v2.json（由 out/res_official.json 與 out_v2/res_official_v2.json 算出；我沒有改 build_site_data.py）',
                 驗證='改完後跑 build_site_data.py（參數同現在的排程）產生 risk.json，逐列對 site_numbers_now_vs_v2.json'),
            dict(步驟=7, 做什麼='重產網站 nav／yearly（步驟 2、5 之後）', 指令='python3 -I -B xq_scripts/site/build_site_data.py <同現在排程的參數>',
                 預期變化=dict(逐年_現行策略={'2024': '63.4→64.6', '2026': '75.2→74.8'}, 逐年_XQ版={'2016': '17.9→18.3', '2017': '22.7→22.8', '2020': '75.7→75.5', '2021': '90.5→96.1', '2026': '54.6→54.5'}, 逐年_0050={'2025': '36.8→36.9', '2026': '67.2→66.3'},
                           週淨值有變的點數=num['weekly_nav_points_changed'], 總週數=num['n_weeks']),
                 驗證='yearly.json 的 12 列與上面預期相同；nav.json 最後一筆現行策略＝28,477,599 四捨五入；頁面「回測：100 萬會變多少」說明裡的「中位數約 46%」不用改（46.40→46.42）'),
            dict(步驟=8, 做什麼='README 第 12 節與其他文件', 檔案='xq_scripts/README.md',
                 改法=['第 332～349 行「2026-11-17 起正式數字」之後新增一小節「<採用日> 起正式數字＝再加 0050 自建含息序列＋減資恢復買賣日修正」，表格用下面「正式數字對照」',
                       '第 338 行驗收說明（48.76%，對 deadline_engine_fix）改成新的驗收', '第 389 行（引擎細節第 3 點）改寫成「已修正（REDUCE_FIX）」', '第 784 行「回測 48.76%」',
                       '第 143、177、218、264、407、450、484、509 行的 0050 28.91% 是舊節的對照欄：建議不逐格改，在第 12 節新小節註明「0050 2019 起 28.91%→28.82%、全期 21.14%→21.08%、2024 起 54.48%→54.18%」（比照 2026-11-21 對舊節加註的做法）',
                       '其他寫著 48.76 或 28.91 的文件：xq_scripts/底倉衛星_操作說明.md 第 18 行、xq_scripts/每檔金額怎麼算_補充說明.md 第 87 行、根目錄 README.md 第 43 行（約 48.8%→約 48.9%）、research/tw_stock_selection/BACKLOG.md（7 處）'],
                 正式數字對照={'2019 起年化': '48.76% → 48.86%', '2015-07 起年化': '34.40% → 34.46%', '2024 起年化': '70.46% → 70.79%', '最大回撤': '−32.12% → −31.61%（一樣是 2024-07-17 到 2025-04-09）', '期末淨值': '28,319,037 → 28,477,599',
                           '已出場筆數': '381 → 381', '20 條路徑（最少／中位數／最多）': '42.01／46.40／49.29 → 41.96／46.42／49.40', '20 條路徑最深回撤': '−39.4% → −39.0%', '0050（2019 起／全期／2024 起）': '28.91／21.14／54.48 → 28.82／21.08／54.18',
                           'XQ 版（A10）2019 起': '45.09% → 45.60%（337 → 338 筆；20 條中位數 43.90 → 43.84）', '成本加倍 2019 起': '43.11% → 43.20%'},
                 驗證='數字逐格對 official_nav/out_v2/res_official_v2.json 與 verify/site_numbers_now_vs_v2.json'),
            dict(步驟=9, 做什麼='連續帳戶 S75 用新序列重跑（研究題，不是工程切換的一部分）', 說明='2008-01～2014 沒有官方日 K 與事件檔：0050 那一段仍要用舊來源，在 2015-01-05 接點等比例銜接；減資事件 2008～2015-07 沒有來源。S75、S82、S83 等用連續帳戶的數字（例如 1,291 萬、23.5%）在重跑前維持舊口徑並註明。',
                 驗證='重跑後由稽核獨立複製'),
            dict(步驟=10, 做什麼='網站溫度計要不要跟著開減資修正（可選）', 指令='make_thermo.py … --reduce-fix',
                 影響=f"2015-05 起只有 {thr['n_months_changed']} 個月的讀數會變（最大 {thr['max_abs_change_pp']} 個百分點），冷溫熱沒有任何一個月改變（verify/thermo_reduce_fix_effect.json）。開了之後 --check 對研究存檔會有那 1 個月不符（研究存檔是舊口徑），要先請研究部用新口徑重產 thermometer_monthly.csv。", 建議='先不開，等 S83 重跑'),
            dict(步驟=11, 做什麼='收尾', 改法=['raw/twt49u/README.md「誰在讀」加上兩份新封存的關係（可選）', 'cslib.py 檔頭註解「候選…尚未採用」改成採用日期', '每日排程：官方日 K 更新後多跑一次 build_raw_archives.py'],
                 驗證='隔天 daily_check.py PASS'),
        ],
    },
    '任務二_網站兩個新區塊': {
        '6_環境溫度計資料': {
            '程式': 'xq_scripts/site/make_thermo.py', '輸出': 'xq_scripts/site/data/thermo.json', 'sha256': dict(make_thermo_py=sha(SITE / 'make_thermo.py'), thermo_json=sha(SITE / 'data' / 'thermo.json')),
            '指令': f'python3 -I -B xq_scripts/site/make_thermo.py xq_scripts/site/data/thermo.json {TSD}（約 35 秒、峰值約 1 GB；需要 numpy／pandas）',
            '內容': f"rows {len(t_rows)} 列（{t_rows[0]['月份']}～{t_nov['月份']}）：月份、資料日、完整月、三個讀數（%）、各自冷／溫／熱與門檻、高於熱／低於冷的項數、合成狀態、股池檔數、來源、連續月數（只有最新一列）；meta：資料最後一天、最新月份是否完整月、用途限制（只描述當下，不能拿來進出場）、定義、門檻、合成規則、歷史對照、資料來源、程式全文。",
            '驗證_算到 2026-10-22 對研究存檔': th1,
            '驗證_白話': f"程式用官方日 K 算得出來的 {th1['months_compared']} 個月（{th1['first']}～{th1['last']}，2026-10 算到 10-22）與 thermometer_monthly.csv 逐月相同：三個讀數最大差 {max(th1['max_abs_diff'].values()):.1e}（< 1e-9），冷溫熱全部相同。",
            '做不到的部分': f"2008-01～2015-04 共 {th1['n_months_only_in_hist']} 個月無法由官方日 K 重算（官方日 K 2015-01-05 才開始；股池要 60 天、飆股密度要再往前 63 天）。這 88 個月在 thermo.json 裡是直接沿用研究存檔 thermometer_monthly.csv 的值（來源欄標「研究存檔（S83）」），不是我重算的；研究用的 2008 起連續面板快取（scratchpad/s75/cache_cont）這台機器上已經不在。",
            '最新讀數（另外列出，不在比對範圍）': {
                '2026-10（完整月，月底 10-30）': {k: t_oct[k] for k in ('資料日', '飆股密度%', '大漲家數比例%', '右尾厚度%', '飆股密度_狀態', '大漲家數比例_狀態', '右尾厚度_狀態', '合成狀態')},
                '2026-11（未結束，到 11-20）': {k: t_nov[k] for k in ('資料日', '完整月', '飆股密度%', '大漲家數比例%', '右尾厚度%', '飆股密度_狀態', '大漲家數比例_狀態', '右尾厚度_狀態', '合成狀態', '連續月數')},
                '提醒': 'README 第 32.2 節表上的 3.91%／3.15%／21.0% 是 2026-10-22（當時資料最後一天）的讀數；10 月底完整月是 5.45%／3.14%／16.88%（右尾厚度落回「溫」，合成仍是熱）。11 月到 11-20：飆股密度掉到 0.93%（低於冷門檻），其餘兩項仍熱，合成仍是熱；連續月數 17 含還沒結束的 11 月。',
            },
            '除權息': '含息日報酬直接呼叫 cslib.exright_fix_chg（TWT49U 封存到 2026-11-20）；預設不開減資修正（和研究存檔同口徑）。',
        },
        '7_照做須知資料': {
            '輸出': 'xq_scripts/site/data/rules.json', 'sha256': sha(SITE / 'data' / 'rules.json'), '產生程式': 'research/tw_stock_selection/agents/eng_v2_candidate_and_site/make_rules.py（直接從 README 第 32.1 節的表格抽，不是手打）',
            '內容': '7 列：編號、須知、依據、重現狀態、已獨立重現（是／否）；meta：摘要（632 筆那一段）、補充、重現說明、出處。', '檢查': rl,
            '標註': '第 6、7 條標「未獨立重現」（README 原文的括號註記移到「重現狀態」欄）；第 1～5 條標「主要數字已獨立重現（R82）」，依據是第 32.1 節標題的說法——R82 沒有逐條複製每一個數字（例如第 1 條的「一年 2～3 筆」）。',
        },
        '8_頁面': {
            '檔案': 'xq_scripts/site/index.html', 'sha256_改之後': sha(SITE / 'index.html'), 'sha256_改之前': '4d362da6a79214559bbe06cd3c2ecb35a4d6022753293689738f380ab4cc04ac',
            '改了什麼': ['樣式：檔尾加一段「環境溫度計、照做須知」', '區塊：在「風險提示」之後、「本月排名前 20」之前加 <section id="rules">、<section id="thermo">', '腳本：在「圖」之前加一段（一個 dash.calc＝thermo_now、一個 dash.onData、8 個 dash.draw）；既有程式一行都沒改',
                       '完整差異：page_test/index_html.diff；可重放的修改程式：page_test/apply_page_patch.py'],
            '照頁面規範': '頁面開頭有 dash-one-writer 標記，我照它的規範寫：文字在 HTML、資料決定的字放 data-draw 元素（存檔時只有「—」或空的）、drawing 以外只設 hidden／class、整句切換用 hidden、每個數字標 data-source／data-field（最新一個月的值走 thermo_now 這個 calc 的第 0 列）。',
            '資料缺的時候': '資料集不存在或沒有列 → 區塊只顯示「尚無資料」；載入失敗 → 「無法載入」；欄位缺或不是數字 → 該格「—」。其他區塊不受影響。',
            '本機測試': dict(做法='page_test/test.js（Chromium／Playwright）＋ page_test/build.py、tpl.html：把 index.html 包進假的 dash 執行環境（沿用 2026-11-16 的 harness 樣板，加上 meta 與載入失敗模擬），直接餵 data/*.json',
                         情境='正常資料 × 1280／400／360 寬 × 淺色／深色；兩個資料集都缺、都載入失敗、空資料、壞資料（缺欄位、非數字）；合成＝冷＋完整月', 全部通過=pg['pass'],
                         每個情境={k: dict(腳本錯誤=c['errors'], 越界修改=c['faults'], 頁面橫向捲動=c['hScroll'], 新區塊內超出畫面的元素=c['overflowing'], 既有區塊文字與改之前相同=c.get('oldSectionsTextSameAsBefore'), 標記檢查=[c['marksChecked'], len(c['marksMismatch'])]) for k, c in pg['cases'].items()},
                         截圖='page_test/shots/（normal_1280_light／dark、normal_400_light／dark、normal_360_light 各兩張；missing、error、empty、bad、cold 各兩張；fullpage_400_light.png）', 靜態檢查='node --check 抽出的 <script>：通過'),
            '沒辦法在本機驗證的': '真正的網站執行環境（datasets/<id> → source.url 載入、來源面板的標記比對、載入中的狀態）本機沒有，只能用假的 dash 物件模擬；上傳後請總監看一次網站回報的 runs／page 有沒有列出 fault 或對不上的標記。',
            '手機': '400 與 360 寬度頁面都沒有橫向捲動；「最近 12 個月」表格在自己的框裡可以左右滑（和頁面其他表格相同做法），合成欄排在月份旁邊，不用滑就看得到。',
        },
        '9_總監上傳步驟': [
            dict(步驟=1, 做什麼=f'重產資料（上傳當天）：python3 -I -B xq_scripts/site/make_thermo.py xq_scripts/site/data/thermo.json {TSD} --check <暫存>/thermo_check.json', 驗證='印出的比對 pass＝true；最後一行的「資料最後一天」＝官方日 K 最新一天'),
            dict(步驟=2, 做什麼='（README 第 32.1 節有改才需要）python3 -I -B research/tw_stock_selection/agents/eng_v2_candidate_and_site/make_rules.py', 驗證='印出 pass＝True'),
            dict(步驟=3, 做什麼='上傳兩個資料檔取得網址：xq_scripts/site/data/thermo.json、xq_scripts/site/data/rules.json（和其他資料集一樣用上傳附件的方式，各得到一個 /_blob/<id>）', 注意='thermo.json 約 160 KB、rules.json 約 7 KB，都含產生程式全文（和既有資料檔相同做法），所有看得到網站的人都讀得到'),
            dict(步驟=4, 做什麼='在網站資料庫新增兩個 datasets 文件（欄位照既有的 datasets/trades）',
                 文件={'datasets/thermo': dict(title='環境溫度計（每月）', description='三個市場環境讀數與冷／溫／熱；只描述當下，不能拿來進出場', source=dict(kind='file', url='<上傳 thermo.json 得到的 /_blob/…>', name='thermo.json', format='json'),
                                              updated=dict(at='<上傳時間 UTC ISO>', by='<同既有文件的寫法>'), edit=dict(via='claude', at='<UTC ISO>')),
                       'datasets/rules': dict(title='照做須知（7 條）', description='這個策略靠少數大贏家：7 條操作須知與是否已獨立重現', source=dict(kind='file', url='<上傳 rules.json 得到的 /_blob/…>', name='rules.json', format='json'),
                                             updated=dict(at='<上傳時間 UTC ISO>', by='<同既有文件的寫法>'), fixed=True, edit=dict(via='claude', at='<UTC ISO>'))},
                 注意='updated、edit、fixed 這幾欄我是照儀表板類型的說明文件寫的，沒看過網站資料庫裡既有文件的實際內容；請以既有的 datasets/trades 為準，欄位不同就照它的。rules 是不會每天變的檔，thermo 每天（或每月）重產後要更新 source.url 與 updated。'),
            dict(步驟=5, 做什麼='上傳頁面 files/index.html（xq_scripts/site/index.html 全文；約 106 KB，低於單一文件 256 KB 上限）', 注意='網站上的頁面若有人用鉛筆改過字，先讀回線上版本比對，不要直接蓋掉；本機這份同時包含 2026-11-18、11-19 還沒上傳的修改與 datasets/trades 的需求（見 state/README.md）'),
            dict(步驟=6, 做什麼='開網站檢查', 驗證=['「照做須知」7 條都在、第 6、7 條有「未獨立重現」標籤', f"「環境溫度計」最新月份 {t_nov['月份']}、資料日期 {t_nov['資料日']}、三個讀數 {t_nov['飆股密度%']}%／{t_nov['大漲家數比例%']}%／{t_nov['右尾厚度%']}%、合成「{t_nov['合成狀態']}」", '其他區塊數字不變；網站回報沒有 fault、沒有對不上的標記；手機寬度不會整頁左右滑']),
            dict(步驟=7, 做什麼='之後的更新', 說明='每天排程跑完 build_site_data.py 後多跑一次 make_thermo.py 並更新 datasets/thermo 的 source.url 與 updated；月中看到的最新月份是「未結束」，月底最後一個交易日收盤後才是完整月。套用 patches.json 的文件修改。'),
        ],
    },
    '做不到的與原因': [
        '0050 序列與減資事件「只從 tsd 重產」做不到：tsd 是部分下載，舊日 K 本機沒有內容；改成「既有封存為主、tsd 補最新」，結果與稽核對照檔逐筆相同。',
        '溫度計 2008-01～2015-04（88 個月）無法重算，只能沿用研究存檔；「2008-01～2026-10 逐月相同」的驗證實際只涵蓋 2015-05～2026-10 的 138 個月。',
        '真正的網站執行環境本機沒有，頁面只做了假環境測試與靜態檢查。',
        '風險表裡要靠其他研究程式（resync、S27、S42、S46、S62、S64、R0）才算得出來的列，切換後的新值我沒有算；S75 沒有重跑。',
        '沒有上傳、沒有改網站資料庫、沒有 git commit（但工作中發現有別的程序在 07:07 左右把我當時已寫好的檔案一起提交了，含 cslib.py 的開關；不是我提交的）。',
    ],
    '風險': [
        'cslib.py 是所有研究共用的引擎：這次只加預設關的開關，已驗證面板 22 個陣列逐位相同、正式管線逐日差 0；但 Panel 物件多了 3 個屬性，若有程式把整個面板物件拿去雜湊或序列化，結果會不同（我沒有找到這種用法）。',
        'run_official_v2.py 的驗收對的是稽核 A＋B 的存檔；之後官方日 K 封存或 TWT49U 補檔若改到 2026-10-22 以前的資料，候選版和正式版會一起變，要以每日回測檢查為準。',
        'make_list_v2.py 放在線上名單資料夾 live_surge/ 裡，檔名不同、沒有任何程式呼叫它；怕混淆可以搬走，diff 檔是獨立的。',
        '溫度計月中讀數會跳：2026-11 到 11-20 飆股密度已低於冷門檻，若月底再有一項轉冷，合成就不是熱；頁面已標「這個月還沒結束」。這只是描述，不能拿來進出場。',
        '溫度計讀數存到小數 2 位、冷溫熱用原值判斷：讀數剛好貼著門檻時，畫面上可能看起來相等但狀態不同。',
        '減資修正的近似（退還股款當成以參考價再投入）與 6 個沒套用的事件，見 r_fix_e50_reduce/FINDINGS.md 的限制；上櫃與 2015-07 以前不在範圍。',
        'thermo.json 每天重產要約 35 秒、1 GB 記憶體，並依賴研究中心的 raw 封存、TWT49U 封存與 cslib.py 的路徑；路徑搬動（例如 tsd 搬家）時 make_thermo.py 的預設值要一起改。',
    ],
    '檔案清單': {
        '新增': ['research/tw_stock_selection/raw/e50_tr_official/（4 檔）', 'research/tw_stock_selection/raw/twse_reduce/（3 檔）', 'research/tw_stock_selection/agents/official_nav/run_official_v2.py、out_v2/（5 檔）',
               'research/tw_stock_selection/agents/s18_live_list/live_surge/make_list_v2.py', 'xq_scripts/site/make_thermo.py', 'xq_scripts/site/data/thermo.json、rules.json',
               'research/tw_stock_selection/agents/eng_v2_candidate_and_site/（build_raw_archives.py、verify_*.py、make_rules.py、make_plan.py、make_list_reduce_fix.diff、plan.json、patches.json、verify/、page_test/）'],
        '修改': ['research/tw_stock_selection/agents/o_ext_canslim_xs/cslib.py（加預設關的開關）', 'xq_scripts/site/index.html（加兩個區塊）'],
        '沒有動': ['run_official.py、make_list.py、build_site_data.py、navs_official.csv.gz、site/data 既有 json、site/state、raw 既有檔、audit_daily/log.csv、其他研究資料夾'],
    },
}
(HERE / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding='utf-8')

# ---------------------------------------------------------------- patches.json
P = [
    dict(檔案='xq_scripts/site/state/README.md', 原文片段='；上傳新檔後更新 datasets/<id> 的 source.url 與 updated）',
         建議改成='；上傳新檔後更新 datasets/<id> 的 source.url 與 updated；2026-11-21 的新頁面另需 thermo＝`data/thermo.json`（環境溫度計，`make_thermo.py <輸出> <官方日K倉庫>` 產生，每天重產）、rules＝`data/rules.json`（照做須知 7 條，`agents/eng_v2_candidate_and_site/make_rules.py` 由 README 第 32.1 節產生），要在網站資料庫新增 `datasets/thermo`、`datasets/rules`；頁面在「風險提示」後面多「照做須知」「環境溫度計」兩個區塊，資料集不存在時只顯示「尚無資料」）',
         理由='網站多兩個資料集與兩個區塊，上傳與每日更新步驟要記在同一行'),
    dict(檔案='xq_scripts/README.md', 原文片段='- 出處：`agents/s83_environment_thermometer/`、`agents/r82_r83_replicate/`。',
         建議改成='- 出處：`agents/s83_environment_thermometer/`、`agents/r82_r83_replicate/`。\n- 網站「環境溫度計」區塊（2026-11-21 加）：`xq_scripts/site/make_thermo.py` 每天用官方日 K 重算到最新股價日，輸出 `site/data/thermo.json`。2015-05 起由程式計算（算到 2026-10-22 時與研究存檔 138 個月逐月相同）；2008-01～2015-04 官方日 K 不夠，沿用研究存檔。'
                  f"上表的 2026-10-22 欄是當時的資料最後一天；10 月完整月（10-30）是 {t_oct['飆股密度%']}%／{t_oct['大漲家數比例%']}%／{t_oct['右尾厚度%']}%（右尾厚度回到溫，合成仍是熱）；2026-11 到 11-20 是 {t_nov['飆股密度%']}%／{t_nov['大漲家數比例%']}%／{t_nov['右尾厚度%']}%（飆股密度低於冷門檻，合成仍是熱，月份還沒結束）。",
         理由='說明網站讀數的來源、驗證範圍，以及 10-22 與月底讀數不同的原因'),
    dict(檔案='xq_scripts/README.md', 原文片段='- 出處：`agents/s82_big_winner_dependence/`、`agents/r82_r83_replicate/`。',
         建議改成='- 出處：`agents/s82_big_winner_dependence/`、`agents/r82_r83_replicate/`。\n- 網站「照做須知」區塊（2026-11-21 加）直接讀上面這張表：`agents/eng_v2_candidate_and_site/make_rules.py` 逐字抽成 `site/data/rules.json`（第 6、7 條標「未獨立重現」）。**這張表的文字或數字有改，要重跑 make_rules.py 並重新上傳。**',
         理由='表格現在是網站資料的來源，改表要記得重產'),
    dict(檔案='research/tw_stock_selection/agents/s18_live_list/FINDINGS.md', 原文片段='| `live_surge/make_list.py` | 名單產生器：list、fill、holdings 三個子命令 |',
         建議改成='| `live_surge/make_list.py` | 名單產生器：list、fill、holdings 三個子命令 |\n| `live_surge/make_list_v2.py` | 候選版（2026-11-21）：make_list.py 原樣複製＋ `--reduce-fix`（減資／面額變更恢復買賣日用事件參考價，預設關）。關閉時輸出與 make_list.py 逐位元相同；2026-11 名單開了也完全沒變。線上仍用 make_list.py；採用步驟見 `agents/eng_v2_candidate_and_site/plan.json` |',
         理由='資料夾裡多了一個檔，要說明它不是線上用的'),
    dict(檔案='xq_scripts/README.md', 原文片段='修正方案見 `research/tw_stock_selection/agents/eng_audit_fixes_2026-11/plan.json` 第 5 項（改用事件參考價的試算：48.76% → 48.91%）。',
         建議改成='修正方案見 `research/tw_stock_selection/agents/eng_audit_fixes_2026-11/plan.json` 第 5 項（改用事件參考價的試算：48.76% → 48.91%）。2026-11-21 已做成候選版（**尚未採用，正式數字沒有換**）：引擎加了預設關的開關 `cslib.REDUCE_FIX` 與 `cslib.E50_SOURCE`（0050 含息序列改自建），候選管線 `agents/official_nav/run_official_v2.py` 兩項都開是 2019 起 48.86%、全期 34.46%、最大回撤 −31.61%、381 筆、20 條路徑中位數 46.42%，與稽核獨立複製逐日相同；切換步驟見 `agents/eng_v2_candidate_and_site/plan.json`。',
         理由='引擎細節第 3 點補上候選版現況，免得讀者以為還只有方案'),
    dict(檔案='research/tw_stock_selection/agents/r_fix_e50_reduce/FINDINGS.md', 原文片段='## 採用清單（若同意）',
         建議改成='## 採用清單（若同意）\n（2026-11-21 工程已備好候選版：新資料檔 `raw/e50_tr_official/`、`raw/twse_reduce/`，cslib 開關，`official_nav/run_official_v2.py`，`live_surge/make_list_v2.py`；逐步切換清單與每一步的驗證見 `agents/eng_v2_candidate_and_site/plan.json`。以下是原清單。）',
         理由='稽核報告的採用清單指到已備好的候選版與切換步驟'),
    dict(檔案='research/tw_stock_selection/raw/twt49u/README.md', 原文片段='- 誰在讀：',
         建議改成='- 相關封存（2026-11-21，候選，尚未採用）：`raw/twse_reduce/`（上市減資／面額變更恢復買賣參考價）、`raw/e50_tr_official/`（0050 含息序列，除息事件取自本封存與官方日 K 倉庫的聯集）。\n- 誰在讀：',
         理由='三份事件／序列封存互相指路'),
]
for p in P:
    f = REPO / p['檔案']; n = f.read_text(encoding='utf-8').count(p['原文片段'])
    assert n == 1, (p['檔案'], p['原文片段'][:30], n)
(HERE / 'patches.json').write_text(json.dumps(dict(說明='文件修改建議（工程師不能寫 .md）。每筆的「原文片段」在該檔案中剛好出現一次（產生時已檢查）。新檔 README 的全文在 plan.json「任務一_候選版 → 1_新資料檔」。', 筆數=len(P), patches=P), ensure_ascii=False, indent=1), encoding='utf-8')
print('plan.json', (HERE / 'plan.json').stat().st_size, 'patches', len(P))
