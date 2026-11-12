# SPEC v2：主策略 ALLPIT_REAL 名單層＋V7X 可執行引擎（草案，待總監凍結）

> **盲版**：本檔由 K 保管員自 `o_g6_spec_v2/SPEC_v2.md`（sha256 `5cfa7e5cd026896da4029ef267e8fa262e86c9a21bd49c20f8e1000afb0f81b6`）以 `k_make_blind.py` 產生。標「〔已遮蔽〕」處為刪去的名單摘要、比對結果與路徑提示（清單見 `BLIND_DIFF.md`）；規則、公式、欄位、資料檔路徑與 sha256 逐字未動。

- 撰寫：規格撰寫員（o21 前置作業），2026-11-12（台北）。
- 定位：把正式程式**實際的行為**寫成不看程式也能唯一重建的文字。凡與 SPEC v1（`o_delist_rev_backfill_allpit/SPEC.json`，sha256 `f5e141b0…c404`，以及其 FINDINGS.md 開頭附註 1～6）文字不一致處，逐條標「程式行為 vs v1 文字」。
- 本文件不含任何新算的績效數字；引用的影響量全部來自 o18～o20 既有報告，並註明出處。
- 小型驗證（只讀資料檔欄位、日期、名稱；不算報酬）腳本與輸出在本資料夾：`verify_inputs.py/.json`、`check_names.py`、〔已遮蔽〕、`check_panel.py/.json`、〔已遮蔽〕、〔已遮蔽〕（皆以 `python3 -I -B` 執行）。
- 機器可讀版：〔已遮蔽〕。模糊點清單：〔已遮蔽〕。驗收規則：`ACCEPTANCE.md`。

**程式出處縮寫**（括號內為檔案 sha256 前 8 碼；行號以該檔現況為準）

| 縮寫 | 檔案 |
|---|---|
| IA | `independent_audit/ia_core.py`（0e4a3265） |
| V2 | `main_v2/v2_core.py`（30741b58） |
| PC | `o_rev_avail_ds_pit_count/pclib.py`（0e4a84b7） |
| AV | `o_rev_avail_ds_pit_count/avail.py`（be18f65f） |
| O18 | `o_delist_rev_backfill_allpit/o18lib.py`（1ace0061） |
| VL | `o_delist_rev_backfill_allpit/v7x_lib.py`（c5746d6a） |
| V4 | `o_v4_revenue_patch_official/v4lib.py`（eb200e68） |
| DRJ | `o_disposal_rejudge/drj_core.py`（3980a698） |
| O3 | `o_infra_v3_rebuild/o3_core.py`（1919c0f3） |
| ST | `o_stagger_tranche/st_core.py`（42783dae） |
| ENV | `o_v7_listing_exec_engine/v7env.py`（5c51d9fe） |
| XD | `o_exec_engine_rewrite_v7x/x_data.py`（0cb307fb） |
| XE | `o_exec_engine_rewrite_v7x/x_exec.py`（7e5b62d8） |
| XS | `o_exec_engine_rewrite_v7x/x_stats.py`（4e82856e） |
| CE | `o_engine_delist_consideration/cz_engine.py`（509ab232） |
| CV | `o_engine_delist_consideration/cz_v7x.py`（8bb07300） |
| I20V | `integrate_o20/run_v7x.py`（4fc5297b） |
| I20P | `integrate_o20/run_paper.py`（87488f82） |
| I20G | `integrate_o20/gen_v7x_plans.py`（5b1fd54d） |
| I20D | `integrate_o20/decide.py` |
| FS | `o_forward_sequential_v2/fslib.py`（df36da29） |

以下路徑若無前綴，皆相對於 `/home/user/-/research/tw_stock_selection/agents/`。

---

## 0. 兩條管線與日期對齊

**G-01 兩條管線（v1 未分開寫）**
正式數字來自兩條**不同**的管線，共用概念但資料表、面板、外國股定義、百分位分母都不同，複製時不得混用：

| | 紙上（PAPER） | 可執行（V7X） |
|---|---|---|
| 名單程式 | O18.build → PC.VU → V2.V2 | VL.make → VL.JV7R → V4.V4 → V2.V2 |
| 價格面板 | 舊面板（D-01b；5,586 日，缺 2024-01-02～02-05） | V7 面板（D-01a；5,611 日）＋下市價補值 P（C-02） |
| 訊號日 | 每月第一個日 ≥ 11 的面板日（〔已遮蔽〕） | 13 個錯開日 j＝0…12（〔已遮蔽〕） |
| 營收表 | tw_revenue 年增率＋rev_5y 年增率欄，B1 只補缺格（P-03） | R_FIX／Y5_FIX，存活者缺口格改 B1 值或設缺（L-03） |
| 未公告（unavail） | 只有 A_UB（P-04） | MC(seed) ∪ A_UB（L-04、L-05） |
| 外國股 | 不明文排除；只在放回層用 PC.is_foreign（P-07） | 明文排除 FORMASK（L-07） |
| 金融 | FINMASK（C-08） | FIN2MASK（L-08） |
| 放回集合 | GOV 原因（P-08） | GAP ∪ GOV（L-09） |
| 營收百分位分母 | 池內全部（含放回股）（P-09） | 只算有營收者（L-10） |
| 處置遮罩（現行正式） | M1_paper（P-05） | DM_BASE（L-06） |
| 引擎 | CE.simulate consideration（紙上，等權、比例成本） | CV.CSim consideration（K5T、整股／零股、順延） |
| seeds | 無 | 20（0…19） |

程式出處：O18:92-124、VL:147-167。v1 差異：v1 只寫「見 v7x_lib.py docstring」。

**G-02 日期記號**
- `d`：訊號日（面板日）。訊號用 d 收盤後可得的資料。
- `t`：d 在該管線面板索引 IDX 的列號；`e`＝列號 t+1 的面板日＝成交日（**下一個面板列**，不是「下一個 GOOD 日」，也不是日曆隔日）。出處：V2:158、V4:157。
- `m`：錨定月（V7X）＝該次換股所屬的月份（L-01）；紙上的 m＝`Period(d,'M')`。
- `j`：V7X 錯開日序號，j∈{0,…,12}。
- 營收月：y3 用 m−1、m−2、m−3 三個月（C-06）。
- 所有「≤、≥」比較皆為日期（無時分），除 A_UB 證據分類（P-04）用到 13:30。

**G-03 欄位順序**
兩個面板的欄位（股票代碼）完全相同：1,249 檔、皆為 4 碼、已按字串遞增排序、沒有 00 開頭的 ETF（check_panel.json）。本文件所有「欄位序」「同分依欄位序」都指這個順序；所有長度 N 的陣列（含亂數）都按這個順序。N＝1,249。

---

## 1. 資料檔（路徑、欄位、單位、sha256）

sha256 全文見 `verify_inputs.json`；此處列前 16 碼。

**D-01a V7 價格面板**（V7X 名單與引擎用）
`o_v7_listing_exec_engine/data/{close_raw,open,high,value,ret,flag,adj,src}.parquet`。index＝交易日（2004-01-02～2026-10-22，5,611 列），columns＝1,249 個代碼。
- `close_raw`、`open`、`high`：元／股，未還原；無成交＝NaN。
- `value`：當日成交值（元）。
- `ret`：當日含息報酬（小數；0.01＝1%）；除權息日已含權息；flag 1 日 ret＝0。
- `flag`：1＝除權息日且 ret 被設為 0（參考價無法由 ret 反推）；2、5 等其他值本規格不使用（程式只判斷 `==1`）。
- sha：close_raw `2a2f7b99b44f19ba`、open `5f49fdec4b6df098`、value `e9c6236ba7aee17b`、ret `6e70541ac9375683`、flag `014587ef662e724e`。

**D-01b 舊價格面板**（紙上管線用）
`/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/*.parquet`（IA:18）。5,586 列，與 D-01a 相比**缺 2024-01-02～2024-02-05 共 25 個交易日**，其餘日期相同。sha：close_raw `f3b158eba8671846`、open `9ff204fb1c521732`、value `d7280bb511d215ae`、ret `e32ffeba2c7ac16b`、flag `3f22b23084c26101`（與 `o_v4_revenue_patch_official/data_freeze_v4.json` 登記相同）。
**注意：此面板只存在 scratchpad，不在研究資料夾內**（見 N-05）。

**D-02 0050 含息**
- V7X 統計與引擎：`/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv`（欄 `date,adj`；sha `771fabfbe0a3165c`）。
- 紙上統計：`o_forward_sequential_v2/raw/finlab/data/raw/etf_0050.csv`（同欄；sha `dace65be24b9a9e6`，**與上一個檔不同**，見 N-06）。

**D-03 加權指數日曆**
`/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv`（欄 `date,open,high,low,close`；1999-01-05～2026-10-22，6,866 列；sha `648adb5c8c60cf20`）。只用 `date`。與 V7 面板的差異：TAIEX 有 2026-09-20、面板沒有；面板有 2018-01-13（非 GOOD 日）、TAIEX 沒有。

**D-04 營收**
- `tw_revenue`：`/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv`，欄 `month`（YYYY-MM-DD，取年月）、`market`（sii／otc…）、`code`、`name`、`revenue_k_twd`（千元）；sha `02657e7c41578302`。
- `rev_5y`：`/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/rev_5y.csv.gz`，欄 `code,ym,revenue,yoy,mom`（yoy 單位＝%）；sha `6ee92a2f5c4e469d`。
- `R_FIX`：`o_v4_revenue_patch_official/yoy_long_R_FIX.parquet`（index＝月、columns＝代碼、值＝年增率 %）；sha `759a6831f18954f3`。
- `Y5_FIX`：`o_v4_revenue_patch_official/yoy_rev5y_FIX.parquet`（同格式）；sha `216941c6c0e6da83`。
- `B1`：`o_delist_rev_backfill_allpit/data/rev_backfill_B1.csv`，欄 `code,ym,rev_k,rev_src,yoy_pct,yoy_src,yoy_method,news_first_date`；1,325 列、38 檔、2011-01～2021-07，其中 `yoy_pct` 有值 961 列；只用 `code,ym,yoy_pct`（%）且只取 yoy_pct 有限值的列；sha `c2672cfa81546d46`。

**D-05 處置清單**
- `disposal_5y`：`…/extras/disposal_5y.csv.gz`，欄 `date`（公告日）、`code,flag,start,end`；2,144 列，date 2021-09-23～2026-10-05；sha `be06c807e4dfcc26`。
- `rejudge`：`o_disposal_rejudge/disposal_hist_rejudge.csv`，欄含 `announce,code,start,end,in_panel`；3,179 列（in_panel＝True 2,367 列）；sha `2c171924096ecf4c`。
- `official_universe`：`o_rev_avail_ds_pit_count/raw/disposal_official_universe.csv`，欄 `code,ordr,start,end,…`（**沒有公告日欄**）；2,218 列；sha `2842f45d825861bf`。
- `universe_announce`：`o_disposal_list_pit_audit_pkl18/data/universe_announce.csv`，欄 `code,ordr,announce,start,end,…`；2,218 列，announce 2002-12-25～2026-10-20；sha `e15c1b02690d663b`。

**D-06 下市／併購**
- `delist_official`：`o_infra_v3_rebuild/delist_official.csv`，欄 `roc,name,code,date,four,in_panel,panel_last`；sha `41f51cb7e1aadbf1`。
- `merger_consideration`：`o_infra_v3_rebuild/merger_consideration.csv`，欄 `code,name,announce_date,delist_date,last_trade,last_close,pay_date,type,cash,cash_over_last_close,ratio,successor,verified,note`；sha `262e92568a0162e9`。
- `delist_prices`（P）：`o_infra_v3_rebuild/delist_prices.parquet`，欄 `date,code,open,high,close,value,ret`；sha `bc4526b4424eaf9b`。
- `events_v2`：`o_engine_delist_consideration/events_v2.csv`，154 列（unknown 132、acq_ur 6、cash 5、swap_ur 5、conversion 3、mixed 3；`last_trade` 空白 13 列）；用到的欄 `code,type,last_trade,terminate_date,cash,ratio,pref_ratio,pref_value,successor,succ_first_trade`；sha `66e80725a4e48387`。

**D-07 名稱與產業**
- `stock_list`：`/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv`，欄 `code,name,market,industry`；sha `06111252bee0f002`。
- `industry_snap` 三檔：`o_pit_shares_industry_2006_2013/raw/industry_snap/{grs_568e6685_tools_twse_list.csv, grs_893c8772_grs_stock_no.csv, goristock_c3c718ab_grs_stock_no.csv}`；sha `00fa81b5d85a1768`、`52c68861590dfaef`、`df09d5a32b700dfc`。
- `foreign_codes`：`o_rev_survivorship/data_freeze.json` 的 `foreign_codes` 陣列（114 個代碼）；sha `01ae47d8793ddf97`。

**D-08 營收公告時間戳（A_UB 用）**
`o_rev_avail_ds_pit_count/raw/`：
- `ts_mops_ann_date.csv`（`code,ym,ann_date`）；sha `4e1079850e4f3e15`。
- `ts_tasi_upstream_firstseen.csv.gz`（`source,code,ym,T,T_prev,file,commit,valid,reason`；T、T_prev 為 UTC 時間戳字串）；sha `460d47489c006fc3`。
- `ts_news_mentions.csv.gz`（`src,art,ts,has_time,code,ym,rev_k,yoy,mom,snip`）；sha `32f7548fc6884e23`。

**D-09 延後機率（MC 用）**
`rev_announce/p_delay.json`：鍵 `p_by_industry`（35 個產業→機率）、`p_overall`（0.17078528281164196）、`INS15`（18 個代碼）；sha `5c447f7434539bca`。

**D-10 引擎參數**
- `o_v7_listing_exec_engine/data/model_spec7.json`（sha `d88c6ea77ceda04b`）：數值全文收在〔已遮蔽〕。
- `o_v7_listing_exec_engine/raw/exec_inputs/cpi_tw.json` 的 `index_2006_100`（2006＝100.0、2026＝130.73470216770653）；sha `afb556ae5b72b7ba`。
- `o_v7_listing_exec_engine/data/jscan_end_days.json`（統計終點日）；sha `3c561e7cfe4ee3a3`。
- `o_exec_engine_rewrite_v7x/results.json` 的 `PREREG.orig_cfg`＝`{stress_odd_extra: true, rank_right: true}`。
- `o_engine_delist_consideration/results.json` 的 `PREREG.events_v2.summary.d.d_used`＝0.057（只影響 cons_lo／cons_hi，主口徑 cons_main 不用）。

**D-11 既有名單輸出（只能當驗收目標，複製員不得讀）**
- 紙上正式名單：〔已遮蔽〕
- V7X orig 名單：〔已遮蔽〕
- V7X REAL 名單（13 j × 20 seeds × 3 變體）：〔已遮蔽〕

---

## 2. 共用衍生量

**C-01 成交與 GOOD 日**
`TRADED[t,c]`＝close_raw 有限值。`GOOD[t]`＝當日 TRADED 檔數 ≥ 300。兩面板唯一的非 GOOD 日是 2018-01-13。出處：IA:41、IA:53-54。V7X 管線的 GOOD 在補上 P 之後重算（O3:53）。

**C-02 下市價補值 P（只有 V7X 管線）**
把 `delist_prices` 的列寫進面板，**只寫 close_raw 原為 NaN 的格**：close、open、high、value、ret 照寫，flag 設 0。寫完後重算 TRADED、GROW、MYADJ、GOOD、PREVC。紙上管線不套 P。出處：O3:34-57（名單側）、XD:38-43（引擎側）。

**C-03 日成長與自算還原價**
`GROW[t,c]`＝若 TRADED 則 1+ret（ret 為 NaN 當 0），否則 1。`MYADJ[t,c]`＝若 TRADED 則 ∏_{s≤t} GROW[s,c]，否則 NaN。出處：IA:43、IA:46。

**C-04 名單側 V60 與 52 週比（gapfix 版）**
只在 GOOD 日列上計算（把非 GOOD 日整列抽掉後做滾動；非 GOOD 日結果＝NaN，不補值）：
- 豁免日集合 EX＝{2017-08-03, 2018-07-12} ∪ [2024-02-15, 2024-03-15]。
- `V60[t,c]`：往回 60 個 GOOD 日（含 t）內，若有任何一日 value 為 NaN 且該日不在 EX → NaN；否則＝窗內 value 非 NaN 值的算術平均。窗不足 60 列 → NaN。
- `S52[t,c]`＝MYADJ[t,c] ÷ max(窗內 MYADJ 非 NaN 值)，窗＝往回 252 個 GOOD 日（含 t）；窗內有任何非 EX 日的 MYADJ 為 NaN，或窗不足 252 列 → NaN。
出處：V2:31-58。v1 差異：v1 無此定義。

**C-05 跳動點與開盤漲停**
- `tick(p)`：p<10→0.01；<50→0.05；<100→0.1；<500→0.5；<1000→1；其餘 5。
- 漲跌幅 lim：成交日 < 2015-06-01 為 0.07，否則 0.10。
- 參考價 `ref[e,c]`＝close_raw[e]/(1+ret[e])；若 flag[e]==1 → 改用「e 之前最後一筆收盤」（跨停牌往前找）。
- 漲停價 `lu`＝floor(ref×(1+lim)/tick(ref×(1+lim)) + 1e-9) × tick(ref×(1+lim))。
- `gap_up[e,c]`＝open[e] 有限 且 ref 有限 且 open[e] ≥ lu − 1e-6。
出處：IA:143-164。

**C-06 y3（3 個月平均年增率）**
`y3(Y, m)[c]`＝Y[m−1,c]、Y[m−2,c]、Y[m−3,c] 的算術平均；三者任一 NaN、或三個月有任一不在 Y 的索引內 → NaN。Y 的單位為 %。出處：IA:124-131。
營收表切換：d ≥ 2021-03-01 用「5 年表」，否則用「長表」（V2:22、V2:151-155）。

**C-07 年增率表的基本版本**
- 長表 `YL`（tw_revenue）：同月同代碼重複列時 `market=='sii'` 優先，其餘取排序後第一筆；pivot 成水準表 lv（月×代碼），索引補成連續月份；`YL = (lv/lv.shift(12) − 1)×100`，只在 lv.shift(12) > 0 時有值。出處：IA:100-110。
- 5 年表 `Y5`（rev_5y）：直接用檔案的 `yoy` 欄；同月同代碼重複取最後一筆；索引補成連續月份。出處：IA:112-122。

**C-08 金融遮罩 FINMASK**
代碼屬於：stock_list 中 `industry == '金融保險業'` 者 ∪ 代碼前兩碼為 `28` ∪ {`5876`,`5880`}。用現行 stock_list（非 PIT）。出處：V2:84-87。

**C-09 股池（兩條管線相同）**
在訊號日 d（列 t）：
1. `base[c]`＝V60[t,c] 有限 且 ≥ 1e7（元） 且 close_raw[t,c] 有限 且 ≥ 10（元）。
2. 在 base 為真的股票中依 V60 由大到小排序（穩定排序；同值依欄位序），取前 min(300, |base|) 檔＝`top`。
即「**先篩 V60 ≥ 1e7 且收盤 ≥ 10，再取前 300**」。排序母體是面板全部 1,249 檔（金融、外國、處置股都佔名額，之後才剔除）。
出處：V2:159-165、V4:158-163、PC:182-186。v1 差異：v1 未寫順序；〔已遮蔽〕。

**C-10 排序與選股（兩條管線相同）**
- 分數 `sc` 先四捨五入到小數 10 位（NaN 當 −9）。
- 排序鍵（優先序由高到低）：sc 由大到小 → 次排序值 `y3tie` 由大到小（NaN 當 −1e18；+∞ 視為最大）→ 欄位序由小到大。
- 取排序前 |el| 名為候選；依序挑選：若 `gap_up[e,c]` 為真且 c **不在上一期名單**（held）→ 跳過；否則入選；滿 10 檔停止。
- `held`＝上一個**有產生名單的**訊號日的 10 檔名單（不是引擎實際持股）。第一期 held 為空。
- 若 |el| < 50：該訊號日不產生名單、held 不更新（續抱上一期）。
- 名單內順序＝入選順序（排名順序）。
出處：V2:185-215。

---

## 3. 紙上名單層：ALLPIT_REAL_{opt,p90,carry}（PAPER）

變體參數（I20P:43-44；v1 `paper.variants`）：`(fill, lo, rev, dsp, patch)`＝`(opt|p90|carry, '2006-01-01', 'UB', True, 'B1')`。

**P-01 訊號日**
從面板日（D-01b）中，取每個日曆月第一個「日 ≥ 11」的面板日（起算 2005-01-01）。保留條件（全部成立）：
1. `Period(d)−1 ≤ last_full_month(Y)`，Y＝該日適用的營收表（d ≥ 2021-03-01 用 Y5p，否則 YLp；都是 P-03 補值**後**的表）；`last_full_month(Y)`＝「該月非 NaN 檔數 ≥ 0.8 × 各月非 NaN 檔數的中位數」的最後一個月。
2. `y3(Y, Period(d))` 有限值檔數 > 100。
3. d < 2026-10-12。
結果：〔已遮蔽〕。評價終點 END＝2026-10-12。
出處：IA:55-64、V2:130-149、V2:79-81。v1 差異：v1 未定義。

**P-02 成交日** e＝d 的下一個面板列（G-02）。

**P-03 營收表（B1 補值）**
`YLp`＝YL 的複本、`Y5p`＝Y5 的複本。對 B1 每一列 (code, ym, yoy_pct)：若 ym 在表的索引內，且該格原為 NaN（代碼不在欄位則先新增全 NaN 欄）→ 填入 yoy_pct。**只填缺格，不覆蓋既有值**。兩張表都這樣處理。之後 y3 一律用 YLp／Y5p。出處：O18:73-85、O18:99-103。v1 差異：v1 只寫「B1 補回值，無則缺」。

**P-04 A_UB（未公告回退）**
對訊號日 d：
- 觸發條件：`deadline_case(d) ∈ {'D=S','D>S'}`。`deadline_case(d)`：令 rm＝Period(d)−1；rm ∈ {2019-01, 2023-03} → 'D>S'；rm＝2020-01 → 'D=S'；否則 D＝TCAL2 中第一個 ≥「d 所在月的 10 日」的日期，D<d → 'D<S'、D==d → 'D=S'、D>d → 'D>S'。TCAL2＝TAIEX 日期 ∪ 面板日期（V2:98-106）。
- 觸發時，分類月份 `ym`＝**Period(d)−1（d 所在日曆月的前一月）**；對每一檔 c：`U[c]`＝（`classify(d, c, ym, 'V6')` 的狀態 ≠ 'AVAIL'）。未觸發時 U 全為 False。
- 套用：U[c] 為真者，`y3[c]` 改用 `y3(Y, Period(d)−1)[c]`（即 m−2、m−3、m−4）。
- 紙上管線**沒有**資料邊界豁免（V7X 有，見 L-05）。
出處：O18:41-52、V2:170-176。v1 差異：v1 附註 5 只寫「deadline_case 觸發、分類月份＝Period(d)−1」，未寫 classify 規則與 ≠'AVAIL' 的判定。

**P-04a classify V6（證據分類器）**
輸入：S＝d（日）、code、ym。C1330＝S＋13:30、C24＝S＋1 日。時間戳一律轉成台北時間（無時區）。
逐來源給狀態：
1. MOPS（`ts_mops_ann_date`，鍵 (code, ym)）：ann_date < S → AVAIL；== S → UNDET；> S → UNAVAIL。
2. 快照（`ts_tasi_upstream_firstseen` 每一列，T 為 NaT 者跳過）：T ≤ C1330 → AVAIL；否則若 T_prev 有值且 T_prev 的日期 > S → UNAVAIL；否則 UNDET。之後覆寫：`source=='upstream'` 且 T_prev 為空且 T > C24＋3 日 → NONE；`source=='tasi'` 且 T_prev 為空且 T > C24＋3 日 → NONE。
3. 新聞（`ts_news_mentions` 中通過數值閘門的列）：has_time 為真：ts ≤ C1330 → AVAIL；ts < C24 → UNDET；否則 NONE。has_time 為假：ts 的日期 < S → AVAIL；== S → UNDET；否則 NONE。
   數值閘門（'sig'）：令 v＝rev_k；比較值 tv＝tw_revenue 的 revenue_k_twd（同 code、ym 第一筆）與 rev_5y 的 revenue 中有限者；v 有限且 > 0 且 tv 非空時，z＝`str(int(round(v)))` 尾端連續 0 的個數，tol＝0.5×10^z＋1.0；存在 t∈tv 使 |v−t| ≤ tol → 通過。
合併：同時有 AVAIL 與 UNAVAIL → UNDET；有 AVAIL → AVAIL；有 UNAVAIL → UNAVAIL；有 UNDET → UNDET；否則 NONE。
A_UB 只看是否 == 'AVAIL'：**UNDET、UNAVAIL、NONE（完全沒有證據）都算未公告**。
出處：AV:22-48、AV:50-93。

**P-05 處置遮罩（紙上）**
令 e＝d 的下一個面板列日期（d 為最後一列時 e＝d）。
- `M0`（o18 原管線）：{disposal_5y 中 start ≤ d ≤ end 的代碼}（**不檢查公告日**）∪（若 d ≤ 2021-09-23）{official_universe 中 (start ≤ d ≤ end) 或 (start ≤ e ≤ end) 的代碼}。出處：IA:134-140、O18:55-70。
- `M1_paper`（**現行正式紙上口徑**）：M0 ∪（若 d > 2021-09-23）{disposal_5y 中 date ≤ d 且 start ≤ e ≤ end} ∪（若 d > 2021-09-23）{universe_announce 中 announce ≤ d 且 start ≤ e ≤ end}。出處：I20P:24-33。
- `M2`（全期統一；待複製後改採）：{universe_announce 中 announce ≤ d 且 start ≤ e ≤ end}，不與任何舊遮罩聯集。出處：I20P:34-42。
不在面板欄位的代碼忽略。v1 差異：v1 無 M1、M2；v1 的 `dsp` 只對應 M0。

**P-06 合格集合（放回前）**
`el_base`＝top（C-09）∩ y3 有限（已套 P-04，且 FINMASK 的 y3 設為 NaN）∩ S52[t] 有限 ∩ 非處置遮罩 ∩ open[e] 有限 ∩ TRADED[e]。
外國股**沒有**明文排除條件（紙上管線靠營收表沒有外國股資料而自然排除；見 P-07 與 N-01）。出處：V2:157-183。

**P-07 外國股判定 is_foreign（只用於放回層）**
`is_foreign(c)`＝代碼前兩碼為 `91`，或下列名稱集合中任一名稱 n 滿足「n 轉大寫後含 `KY`」或「以 `DR` 結尾」或「以 `F-` 開頭」：
- `NAME[c]`：tw_revenue（依 月、代碼、sii 優先 排序去重後）中該代碼**最早一個月**的名稱；tw_revenue 沒有該代碼時，才用 stock_list 的名稱；兩者都沒有 → 空字串。
- `SNAPN[c]`：三個 industry_snap 檔中，第一欄為 4 位數字的列，其第二欄（去除空白與雙引號）名稱的集合。
**不使用 delist_official、merger_consideration 的名稱。**
出處：PC:98-107、PC:118-120。
v1 差異／重要：o20 複製員的「事後修正」是加入 delist_official 名稱；那**不是**程式行為。〔已遮蔽〕在面板欄位上兩種讀法相差 6 檔：1262、1592、2936、6422、6452、8480（程式判「非外國」，delist_official 讀法判「外國」；〔已遮蔽〕）。〔已遮蔽〕

**P-08 放回集合 A 與缺值原因**
僅當 `2006-01-01 ≤ d ≤ 2026-12-31`（全部訊號日都符合）：
- `el0`＝top ∩ S52 有限 ∩ 非處置遮罩 ∩ open[e] 有限 ∩ TRADED[e] ∩ 非 FINMASK。
- 對 el0 中 y3 為 NaN 的每一檔 c 算 `cause(c, d)`；cause ∈ GOV → c ∈ A。
- GOV＝{TRUNC, IFRS13, POST13, POST13_R5, PRE12, HOLE, NEVER_DOM}。

`cause(c, d)`（m＝Period(d)；**不看 A_UB 是否回退，一律檢查 m−1…m−3**）：
1. is_foreign(c) → 'FOREIGN'。
2. d < 2021-03-01：
   - c 不在 tw_revenue 水準表欄位 → 'NEVER_DOM'。
   - 對 x ∈ {m−1,m−2,m−3,m−13,m−14,m−15}：水準 lv[x,c] 為 NaN → 收集 `month_cause(c,x,d)`；有限且 ≤ 0 且 x ≤ m−13 → 收集 'BASE_LE0'。
   - `month_cause`：lr＝該檔最後一個有營收的月、fr＝第一個；ltm＝該檔在面板最後成交日所在月。x > lr 時：若 2009-12 ≤ ltm ≤ 2012-12 → 'TRUNC'；否則若 x ≥ 2013-01 且 lr ≥ 2012-12 →（d ≤ 2013-12-31 ? 'IFRS13' : 'POST13'）；否則 'PRE12'。x < fr → 'NEWLIST'。其餘 → 'HOLE'。
3. d ≥ 2021-03-01：
   - c 不在 rev_5y 的 yoy 表欄位 →（c 在 tw_revenue 欄位 ? 'POST13_R5' : 'NEVER_DOM'）。
   - 對 x ∈ {m−1,m−2,m−3}：yoy 為 NaN → x > 該檔最後有 yoy 的月 → 'POST13_R5'；x < 第一個有 yoy 的月 → 'NEWLIST'；否則 'HOLE'。
4. 依優先序 [TRUNC, IFRS13, POST13, POST13_R5, PRE12, HOLE, NEWLIST, BASE_LE0] 回傳收集到的第一個；都沒有 → 'UNKNOWN'。
原因判定用的是**未補值的原始** tw_revenue 水準與 rev_5y yoy 欄，不是 P-03 補值後的表。
出處：PC:126-165、PC:176-194。v1 差異：v1 只寫「pclib.VU 的定義」。

**P-09 三種放回值與分數**
令 el＝el_base ∪ A，n＝|el|；`s52p`＝S52 在 el 內的百分位（平均名次 ÷ n）。
- **opt**：A 的營收百分位 rp＝1.0；非 A（norm＝el∖A）的 rp＝（y3 在 norm 內的平均名次）÷ **n**（分母含 A）；A 的次排序值 y3tie＝+∞。
- **p90**：A 的 rp＝0.9；norm 同上；A 的 y3tie＝norm 的 y3 第 90 百分位（線性內插，忽略 NaN）。
- **carry**：對 A 每檔，b＝1…24 依序取 `y3(Y, Period(d)−b)[c]`，第一個有限值即為其 y3；24 個月內都沒有 → 從 A 移除。之後 A 與 norm 一起排名：rp＝y3 在 el 內的百分位（平均名次 ÷ n），y3tie＝y3。
- `sc`＝0.5×s52p＋0.5×rp；非 el 設 NaN。
- 之後照 C-10 選 10 檔。
出處：PC:195-222。v1 差異：v1 未寫百分位分母、p90 的次排序值、carry 的 24 個月上限。

**P-10 輸出**
每個訊號日一列：`signal`（d）、`exec`（e）、`picks`（10 檔，逗號分隔，排名順序）。

---

## 4. V7X 名單層：REAL_{opt,p90,carry} × j × seed

每個變體 × j∈{0…12} × seed∈{0…19} 產生一份名單（共 3×13×20＝780 份）。

**L-01 錨定月與 j 日**
- 官方訊號日：照 P-01 的規則，但面板用 D-01a（補 P 後）、營收表用 R_FIX／Y5_FIX（未改寫版）；得〔已遮蔽〕＝錨定月集合 ANCH。出處：VL:26-27。
- 期限日 `D(m)`：m ∈ {2019-02 → 2019-02-15, 2023-04 → 2023-04-12} 用該特例日，否則用 m 月 10 日；取 **TAIEX 日曆**（D-03，不含面板獨有日）中第一個 ≥ 該日的日期。
- `day_for(m, j)`＝TAIEX 日曆中 D(m) 之後第 j 個日期（j＝0 即 D(m)）。若該日不在面板、或不是 GOOD 日、或之後已無 GOOD 面板日 → 該月對這個 j 無訊號日。
- 另一條件：d 的下一個面板列必須是 GOOD 日，否則該月跳過。
- 終點 `de(j)`＝day_for(2026-10, j)；若不存在或晚於面板最後一日 → 面板最後一日（2026-10-22）。只保留 d < de(j) 的訊號日。
- e＝d 的下一個面板列。
- 已核對的特例（〔已遮蔽〕）：〔已遮蔽〕；農曆年前後 j 日會落到**下一個日曆月**（例：〔已遮蔽〕）。
出處：ST:91-121、VL:29-39。v1 差異：v1 只寫「13 j」。

**L-02 y3 的月份**
V7X 的 y3 用**錨定月** m（不是 d 的日曆月）：`y3(Y, m − back)`，Y 依 d ≥ 2021-03-01 切換；FINMASK 的 y3 設 NaN。出處：VL:104-109。

**L-03 營收表（REAL）**
從 R_FIX（長表）與 Y5_FIX（5 年表）出發：
1. 「存活者缺口格」＝tw_revenue 年增率 YL（C-07，重新索引到 R_FIX 的月×代碼）為 NaN、但 R_FIX 有值的格。對每一格：B1 有該 (code, ym) → 改成 B1 的 yoy_pct；沒有 → **設為 NaN**。
2. B1 其餘的 (code, ym)：若在表的索引內且該格為 NaN → 填入。
3. Y5_FIX 同法（缺口格＝rev_5y yoy 為 NaN、Y5_FIX 有值）。
o18 正式執行的計數：〔已遮蔽〕
出處：VL:63-94。v1 差異：v1 `v7x.definition` 一句話帶過。

**L-04 MC(seed) 未公告抽樣**
`u_mc(m, d, seed)`（長度 N 的布林陣列）：
- d < D(m)：全部 True。
- d == D(m)，或（m＝2020-02 且 d ≤ 2020-02-14）：`rng = numpy.random.default_rng([seed, YYYYMMDD(d), 0])`；`u = rng.random(N) < PCOL`（N＝1,249，依欄位序）。
- 其餘：全部 False。
- 另外，若 m−1 ∈ [2026-01, 2026-08]：D15＝TAIEX 日曆中第一個 ≥ m 月 15 日的日期；d < D15 → u |= IS15；d == D15 → u |= IS15 ∧（`default_rng([seed, YYYYMMDD(d), 15]).random(N) < PCOL`）。
- `PCOL[c]`＝c∈FIN（C-08 的集合）→ 1.0；否則 p_by_industry[stock_list 的 industry]，查不到產業或產業不在表內 → p_overall。`IS15[c]`＝c ∈ INS15。
- YYYYMMDD(d) 為整數，例如 20060410。亂數產生器為 numpy 的 PCG64（default_rng），種子為整數序列；正式執行環境 numpy 2.5.3。
出處：ST:137-152、V2:91-97。v1 差異：v1 只寫「MC(seed)」。

**L-05 A_UB（V7X）與資料邊界豁免**
`ub(d)` 照 P-04（**用 d 的日曆月**：觸發用 deadline_case(d)、分類月份 Period(d)−1；不是錨定月）。
`U(d)`＝u_mc(m, d, seed) ∨ ub(d)；例外：若 ub(d) 有任何 True **且** `y3(Y, m−1)` 的有限值檔數 < 100 → U(d)＝u_mc（該日不套 A_UB）。實際發生於〔已遮蔽〕。
U[c] 為真者 y3 改用 `y3(Y, m−1)`。
出處：VL:42-51、VL:157-162。v1 差異：附註 5 有寫定義與〔已遮蔽〕豁免日，但未寫豁免的判斷式（< 100 檔）。

**L-06 處置遮罩（V7X）**
令 e＝d 的下一個面板列日期。
- 表 `T_V4`＝rejudge 中 in_panel 為真的列（公告日＝announce）∪ disposal_5y 中「代碼在面板欄位、且 (code, start) 不在前者」的列（公告日＝date）。
- `DMT(d)`＝{T_V4 中 公告日 ≤ d 且 start ≤ e ≤ end}。
- `DM_BASE(d)`（**現行正式 V7X 口徑**）＝DMT(d) ∪（若 d ≤ 2021-09-23）{official_universe 中 (start ≤ d ≤ end) 或 (start ≤ e ≤ end)}。
- `M2(d)`＝{universe_announce 中 announce ≤ d 且 start ≤ e ≤ end}（與 P-05 同式；面板換成 V7）。
出處：DRJ:37-47、V4:92-117、VL:52-62、I20G:22-31。v1 差異：v1 寫「DMT ∪ 官方全歷史」，未寫表的組成與比較式。

**L-07 外國股遮罩 FORMASK（V7X）**
FOREIGN＝D-07 的 foreign_codes ∪ {面板欄位中以 `91` 開頭者} ∪ {delist_official 中名稱以 `F-` 開頭或轉大寫後含 `KY` 的代碼}。出處：V4:66-70、V4:79。
（與 P-07 不同；面板上 FOREIGN 比 is_foreign 多 6 檔：1262、1592、2936、6422、6452、8480。）

**L-08 金融遮罩 FIN2MASK（V7X）**
c ∈ FIN（C-08）或 delist_official 的名稱含下列任一字串：`銀`、`證`、`金控`、`證金`、`產保`、`壽`、`票券`、`產險`、`人壽`、`投信`、`合庫`、`信託`。出處：V4:71-75。
（y3 的 NaN 化仍用 FINMASK；FIN2MASK 用在 ok 條件。）

**L-09 合格與放回集合（V7X）**
- `ok`＝top（C-09）∩ S52 有限 ∩ 非處置遮罩 ∩ open[e] 有限 ∩ TRADED[e] ∩ 非 FIN2MASK ∩ 非 FORMASK。
- `el_rev`＝ok ∩ y3 有限。
- `miss`＝ok ∩ y3 為 NaN。對 miss 每一檔 c：c ∈ A 當且僅當（c ∈ GAP 且 d ≤ GAP[c]）或 `cause(c, d)` ∈ GOV（P-08 的 cause；m 用 **d 的日曆月**）。
- `GAP`＝{delist_official 中 four=='True' 的代碼 → 終止上市日 date}，排除 FOREIGN（L-07）與 is_fin（L-08）者。
出處：V4:155-173、VL:110-120、V4:76-78、O3:205-209。v1 差異：v1 寫「GAP ∪ GOV」，未寫 ok 的組成。

**L-10 三種放回值與分數（V7X）**
- **opt／p90**：el＝el_rev ∪ A；`pr52`＝S52 在 el 內的百分位（÷|el|）；`prr`＝y3 在 **el_rev** 內的百分位（平均名次 ÷ **|el_rev|**，分母不含 A）；A 的 prr＝1.0（opt）或 0.9（p90）；A 的次排序值＝+∞（opt）或 el_rev 的 y3 第 90 百分位（p90）。
- **carry**：對 A 每檔，b＝1…24 取 `y3(Y, m−b)[c]`（錨定月；FINMASK 為 NaN）第一個有限值；都沒有 → 移出 A。el＝ok ∩ y3 有限（含 carry 值）；pr52、prr 都是 el 內百分位；次排序值＝y3。
- `sc`＝0.5×pr52＋0.5×prr。之後照 C-10。
出處：VL:121-145。v1 差異：v1 未寫；**與紙上 P-09 的分母不同**（N-02）。

**L-11 輸出**
每份名單每個訊號日一列：`variant,j,seed,signal,exec,exec_idx,picks,inserted`；`exec_idx`＝e 在 V7 面板的列號（0 起算）；`picks` 依排名順序；`inserted`＝picks 中屬於 A 者。〔已遮蔽〕出處：VL:163-167。

---

## 5. V7X 可執行引擎（CSim，cons_main）

**E-00 引擎設定**
`cfg = {tier, mode:'GROW', model:'K5T', cost:'normal'|'stress', stress_odd_extra:true, rank_right:true, f1:true, f2:true, f3:0.5, f4:true}`；`settle`＝E-20 的結算表（bound='main'）；`cmode='consideration'`。出處：I20V:18-37。v1 差異：v1 只寫 grid。

### 5.1 引擎資料層（V7 面板＋P；XD:31-79）

**E-01 基本陣列**
C、O、V、R、F＝close_raw、open、value、ret、flag（補 P 後）。`TR`＝C 有限。`G`＝TR ? 1+R（NaN 當 0）: 1。`CF`＝C 逐欄前值補。`PC[t]`＝CF[t−1]（前一筆收盤；第一列 NaN）。`ADJ[t]`＝TR ? G×PC/C : 1；非有限 → 1。`GOOD`＝TR 檔數 ≥ 300。

**E-02 引擎側 V60（與 C-04 不同）**
1. 在 GOOD 日列上：同 C-04 的 V60（60 列、非 EX 日不得缺值、窗內非 NaN 平均）。
2. 放回全日曆後，**逐欄前值補，最多連續補 3 列**（`ffill(limit=3)`，對所有 NaN 生效：非 GOOD 日，以及 GOOD 日因窗內缺值而為 NaN 的格）。
出處：XD:52-57、XD:61。程式行為 vs 規格文字：V7X frozen_spec 寫「窗內不得缺值」，程式註解寫「非 GOOD 日沿用前值」，實際連 GOOD 日的 NaN 也補 3 列（o19 J_RULINGS G1 判「原引擎偏離規格」）。

**E-03 σ60**
在 GOOD 日列上，R 的 60 列滾動樣本標準差（ddof＝1，窗內至少 40 個非 NaN），逐欄前值補；放回全日曆後再逐欄前值補（無上限）。出處：XD:58-62。

**E-04 vrank（成交值名次）**
對每個面板列 t：母體＝面板全部 1,249 檔中 `V60[t]`（E-02）有限且 ≥ 1e7、且 **CF[t]（前值補收盤）** 有限且 ≥ 10 者；依 V60 由大到小（穩定排序，同值依欄位序）給名次 1、2、…；不在母體 → NaN。
成交日 t 使用的是前一列的值：`VRp[t]＝VR[t−1]`、`V60p[t]＝V60[t−1]`、`SIGp[t]＝σ60[t−1]`。
出處：XD:63-73。程式行為 vs 規格文字：frozen_spec 寫「收盤 ≥ 10」；程式用前值補收盤（當日無成交的股票仍排名），且 V60 是 E-02 的補值版；母體含金融、外國股。v1 差異：v1 未定義。

**E-05 由 vrank 衍生的分桶**（XE:33-50）
- `DEC[t,c]`＝VRp 有限且 ≤ 300 → clip(ceil(VRp/30), 1, 10)；否則 10。
- `OASH`＝oa_share_by_dec[DEC]。`ZSH`＝VRp 有限且 ≤ 300 → zcap.share_by_dec[DEC]；否則 zcap.share_default。
- 零股名次桶（rank_right＝true）：VRp ≥ 150 或 NaN → 桶 1；否則桶 0。
- 零股價格桶：依 PC[t]：<50 → 0；[50,100) → 1；[100,300) → 2；≥300 → 3（PC 為 NaN 當 0）。
- `OCB`＝odd.buy[價格桶][名次桶]；`OCS`＝max(odd.sell[價格桶][名次桶], 0)。

**E-06 漲跌停價**（XE:27-31、XD:75）
`REF[t]`＝F==1 ? PC : C/G。lim＝t 早於 2015-06-01 ? 0.07 : 0.10。
LIMUP＝floor(REF×(1+lim)/tick + 1e-9)×tick；LIMDN＝ceil(REF×(1−lim)/tick − 1e-9)×tick；tick 取 REF×(1±lim) 所在級距（C-05 的 tick 表）。

**E-07 零股成交價**（XE:38-44）
- t 早於 2020-10-26（盤後零股）：`ODDP[t]`＝C[t]（當日收盤）。
- 之後（盤中零股）：`ODDP[t]`＝O[t]×(1＋drift)，gap＝O/PC−1（PC 無效或結果非有限 → 0），drift＝drift.vals[k]，k＝gap 在 drift.edges＝[−0.03,−0.01,0,0.01,0.03] 中 `searchsorted(…, side='right')` 的位置。

**E-08 引擎的處置日矩陣 DISP**（XD:93-107）
用 L-06 的 T_V4（不看公告日）：`DISP[t,c]`＝存在一列使 start ≤ IDX[t] ≤ end。與名單遮罩（DM_BASE／M2）無關，M2 整合也沒有改它。

### 5.2 帳戶與逐日流程（CV:30-132）

**E-09 起始資金（GROW）**
`t0`＝名單第一列的訊號日列號。t0 收盤時持有現金 `tier × CPI[2006]/CPI[2026]`（＝tier × 0.7649078503404548），無持股；之後不提領不加碼。tier：30 萬＝3e5、100 萬＝1e6、300 萬＝3e6、1,000 萬＝1e7（2026 幣值）。出處：CV:33、XD:143-145、I20V:28。

**E-10 每日順序**（t＝t0+1 … 面板最後一列）
1. 持股股數 × ADJ[t]（除權息以股數調整；可能產生小數股）。
2. 前一日的處置賣出款（cash_late）併入現金。
3. `tr[c]`＝TR[t,c] 且 O[t,c] 有限。
4. 對價結算的觸發與到期（E-21）。
5. 若 t 是換股日（名單的 exec_idx）→ E-11；否則若有順延單 → E-17。
6. 收盤權益＝現金＋cash_late＋Σ 股數×CF[t]＋待撥款＋Σ 請求權價值。單位淨值＝權益 ÷ 起始現金（GROW 無外部現金流，時間加權等同此式）。
引擎一律跑到面板最後一列；統計時才截到 e(j)（S-02）。名單只含 d < de(j) 的訊號日（L-01），所以 2026-10 不換股。

**E-11 換股日開盤前**
1. **清空全部順延單**（上一期沒成交完的委託作廢）。
2. 「凍結持股」＝持股 > 0 且 tr 為假者 → 照 E-22（表外停牌）檢查是否轉現金。
3. 待撥款到期者併入現金（E-22）。
4. 可交易持股的小數股（S − floor(S+1e-9)）以 O[t] 轉現金，不計成本。
5. 進入 E-12。

**E-12 目標股數**（XE:272-305）
- `cur`＝持股 > 0 且 tr 為真者；`U`＝cur ∪ 名單 pk（**凍結的非名單持股不在 U 內**）；k＝|pk|。
- `avail`＝現金＋Σ_{cur} 股數×O[t]；`hi`＝avail/k×(1−0.02)。
- 給定每檔目標金額 T：p＝PC[t]；L＝floor(T/(1000p)+1e-12)；q＝floor((T−1000Lp)/p+1e-9)；名單股目標股數＝1000L＋max(q,0)，非名單股目標 0；`delta`＝目標 − floor(現持股+1e-9)。
- `need(T)`＝Σ_{U} cost_est(delta)（E-13）；但「非名單且當日在 DISP」的賣出不計入（f2：處置賣出款當日不可用）。
- need(hi) ≤ 現金 → T＝hi；否則在 [0, hi] 二分 40 次，取使 need ≤ 現金的下界。

**E-13 單檔成本估計 cost_est(t,c,delta,disp)**（XE:122-140）
回傳現金淨支出（買為正、賣為負）。a＝|delta|，L＝a//1000 張，q＝a%1000 股，fr＝手續費率（E-16）。
- 處置賣出（disp 且 delta<0）：價＝C[t]−tick(C[t])；整股部分與零股部分**各自**計一筆：−v＋max(20, fr×v)＋0.003×v。
- 整股部分（L>0）：v＝1000×L×O[t]×(1＋side×imp(L))（imp 見 E-14，用全部 L 張、不套參與上限）；支出＝side×v＋max(20, fr×v)＋（賣出時 0.003×v）。
- 零股部分（q>0）：v＝q×ODDP[t]×(1＋side×oddc)（E-15）；同式。

**E-14 整股成交（K5T 衝擊與參與上限）**（XE:73-86、114-120、143-157）
- 參與上限（f4）：`oa_pred`＝OASH[t,c]×V60p[t,c]；非有限或 ≤ 0 → 上限 1 張；否則上限＝max(1, floor(0.20×oa_pred/(1000×PC[t])))。委託張數超過上限 → 只成交上限張數，其餘順延。
- 衝擊（每邊）：`imp`＝max(0.001＋0.5×SIGp[t,c]×sqrt(Q/OA), tick(O[t])/O[t])，Q＝成交張數×1000×PC[t]，OA＝OASH[t,c]×**V[t,c]（當日實際成交值）**；若張數＝0、OA 非有限或 ≤ 0、SIGp 非有限 → 平方根項省略（0.001 與 tick/O 取大）。壓力成本：imp 再加 0.004（加在 max 之後）。
- 成交價＝O[t]×(1±imp)；成交金額 v＝1000×張數×成交價。
- 手續費＝max(20, fr×v)；賣出另扣證交稅 0.003×v。

**E-15 零股成交**（XE:91-112、159-175）
- 量上限：t ≥ 2020-10-26：floor(1.0×ZSH×V60p/ODDP)；t 更早（f3）：floor(0.5×ZSH×V60p/C[t])。V60p 非有限 → 0。委託股數 > 上限 → 只成交上限股數（上限 0 則不成交），其餘順延。
- 成本率 oddc：買＝OCB，壓力時 ×2 再 ＋0.004；賣＝OCS，壓力時 ＋0.004。
- 成交金額 v＝股數×ODDP×(1±oddc)；手續費＝max(20, fr×v)；賣出證交稅 0.003×v。
- 順延日買單另有現金限制（E-17）。

**E-16 手續費率與稅**（XE:69-71）
- 一般成本：t 早於 2015-01-01 → 0.001425×0.6；之後 → 0.001425×0.28（2.8 折）。壓力成本：全期 0.001425×0.6。
- **每筆最低 20 元**；「一筆」＝每檔每日的整股成交一筆、零股成交一筆（處置賣出的整股部分、零股部分也各一筆）。
- 證交稅：賣出金額 × 0.003（整股、零股、處置賣出都課）。

**E-17 執行順序、順延規則**（XE:305-359）
換股日：
1. U 內各檔依 delta 由小到大（穩定排序；同值依 U 的順序＝欄位序）逐檔執行（賣先、買後）。
2. 非名單且在 DISP 的賣出 → 處置賣出（E-18）；沒賣完的部分順延。
3. 其餘：若開盤被漲跌停擋住（買：O ≥ LIMUP−1e-6；賣：O ≤ LIMDN+1e-6）→ 整筆順延。
4. 買單：若 cost_est(完整 delta) > 現金＋1e-9 → **整筆順延**（不部分成交）。
5. 先成交零股部分、再成交整股部分；未成交的股數（上限造成）順延。
順延日（非換股日且有順延單）：
1. 順延單依待成交股數由小到大（賣先）。當日無成交（tr 為假）的股票跳過、續留。
2. 賣：量＝min(待賣, round(持股))；量 ≤ 0 → 取消。當日在 DISP → 處置賣出。否則被跌停擋住 → 當日跳過。否則先零股後整股，套各自上限。
3. 買：被漲停擋住 → 當日跳過。Lmax＝floor(現金/(O×1000×1.05))（現金 ≤ 0 → 0）；L＝min(待買張數, Lmax)；q＝待買股數 % 1000；若 q>0：零股成交金額＋手續費 > 現金 − L×O×1050 → 當日不下零股；之後整股 L 張（仍套參與上限）。
4. 沒成交完的續留，直到**下一個換股日一律清空**（E-11）。

**E-17a 換股日停牌的非名單持股（讀法 a）**
程式行為：換股日開盤無成交（tr 為假）的持股不進入 U，**不產生賣單、也不產生順延單**；它留在帳上（以前值補收盤計價），要等到之後某個**換股日**當天可成交時，才會被當成非名單持股賣出（若屆時仍不在名單）。期間恢復交易的非換股日不會賣。若它符合 E-22 則在換股日轉現金。
出處：XE:274-276（cur 只含 tr 為真者）、CV:77（清空順延單）。v1 差異：v1 與 V7X frozen_spec 都未寫；〔已遮蔽〕。

**E-18 處置期賣出**（XE:177-196）
適用：賣出日 t 該股在 DISP（E-08），且（換股日）該股不在名單；順延日則只看 DISP。
- 量上限（f2）：floor(0.20×V[t]/C[t])（V 為 NaN 當 0）；超過 → 只賣上限，其餘順延。
- 價＝C[t]−tick(C[t])；不計衝擊、不計零股價差、不受漲跌停限制。
- 整股部分與零股部分各一筆手續費（最低 20）與證交稅。
- 款項記入 cash_late，**隔日**才併入現金。

**E-19 30 萬整股限制格（wholelot）**
與標準引擎唯一的差別：t 早於 2020-10-26 時零股量上限＝0（買賣皆不成交、列入順延）。目標股數仍照 E-12 計算（含零股部分）、need(T) 仍估零股成本；處置賣出不經零股上限，仍可賣出零股。出處：I20V:24-27。

### 5.3 對價結算（cons_main）與表外停牌

**E-20 結算表**（CE:36-87；面板＝V7＋P）
對 events_v2 每一列（`last_trade` 空白者跳過；代碼不在面板者跳過）：
- ltU＝last_trade；term＝terminate_date，空白則 ltU＋14 個日曆天。
- `lt`＝該代碼在面板中日期 ≤ max(term, ltU) 的最後成交列；無 → 跳過；lt 為面板最後一列 → 跳過。`lc`＝C[lt]。
- `trig`＝lt＋1（列號）。`ti`＝面板中第一個日期 ≥ term 的列號；`pay`＝min(ti＋20, 最後一列)。
- 若面板 lt 的日期早於 ltU−7 天 → 類型改 `price_gap`：一個「固定賣出」請求權（每股 lc，可結算列＝lt+1）。
- 否則依 type：
  - `cash`：現金請求權，每股 cash，可結算列＝pay。
  - `swap`／`conversion`／`mixed`（存續股在面板且 succ_first_trade 有值）：s＝存續股在 succ_first_trade 當日或之後第一個「有成交且有開盤」的列。股票請求權（ratio 為有限值時才建立）：比例 ratio、可結算列 s、空窗期每股以 lc 計值。mixed 另加：現金部分（每股 cash，列 pay）、特別股部分（固定賣出，每股 pref_ratio×pref_value，列 s）。
  - `swap_ur`（存續股在面板、s 存在、lt 當日存續股有收盤）：股票請求權，比例＝lc ÷ 存續股 C[lt]（bound=main，δ＝0）。
  - `acq_ur`，或上述股票型但存續股不在面板／無 s：現金請求權，每股 lc，列 pay。
  - `unknown`（main）：現金請求權，每股 lc，列 pay。`distress`（main）：固定賣出、歸零，列 lt+1。

**E-21 結算流程**（CV:48-74）
- 觸發：t＝trig 且當時持股 > 0 → 持股歸零、該檔順延單取消；每個部分建立請求權：價值＝股數×每股值（股票型空窗期價值＝股數×lc，另記存續股數 q＝股數×比例）；可結算列＝max(規則列, t)。
- 每日：t ≥ 可結算列時——現金型：價值併入現金（任何日；入帳前不計息、不課稅、不付手續費）；股票型：存續股持股加 q（任何日；之後照一般流程持有或賣出）；固定賣出型：**只在換股日**結算，入帳＝價值×(1−0.003−當日手續費率)（歸零型入帳 0）。
- 未結算的請求權以其價值計入權益。

**E-22 表外永久停牌（E23 規則）**（XD:110-141、CV:80-100）
適用對象：換股日的凍結持股（持股 > 0、當日無成交），且沒有被 E-21 轉成請求權者（即代碼不在 events_v2、或其 last_trade 空白、或 E-20 被跳過）。
先建兩張表（面板＝V7＋P）：
- `lt[c]`＝面板最後成交列；代碼重用修正：delist_official 中 four=='True' 的代碼，若面板在其終止日之後仍有成交，則 lt＝終止日（含）以前最後成交列、`uu`＝終止日之後第一個成交列；否則 uu＝無限大。
- 只對 lt 的日期 ≤ 面板最後一日 − 30 天者建規則。lc＝C[lt]、ltd＝lt 的日期。
- 代碼在 merger_consideration：dd＝delist_date（空白則 ltd＋14 個日曆天）。type=='cash' 且 cash 有值 → 係數 f＝cash/lc、釋放日 rel＝pay_date（空白則 dd＋14 天）；type=='failure' → f＝0、rel＝ltd；其餘 → f＝1、rel＝dd。另外 type=='cash' 者建 `MCASH`＝(f 或 1, di＝第一個日期 ≥ dd 的列, pi＝min(di＋20, 最後列), uu)。
- 代碼**不在** merger_consideration → f＝1、rel＝**ltd＋14 個日曆天**（不是 delist_official 的終止日）。
- `XR[c]`＝(f, ri＝第一個日期 ≥ rel 的列, uu)。
換股日 t 對每一檔凍結持股：
1. 若在 MCASH：di ≤ t < uu → 持股轉為「待撥款」＝股數×CF[t−1]×f×(當日 0050 報酬因子)，之後每日隨 0050 含息報酬變動，直到「pi < t 的換股日」才併入現金（該日不再乘 0050）；不論是否成立都不再看 XR。
2. 否則若在 XR 且 ri ≤ t < uu → 現金＋＝股數×CF[t−1]×f、持股歸零（不計成本）。
3. 否則續留（凍結）。
程式行為 vs 規格文字：o3／V7X 規格寫「unknown ×1，終止日之後第一個換股日轉現金」；程式對表外者用「最後成交＋14 個日曆天之後的第一個換股日」。〔已遮蔽〕
v1 差異：v1 未寫。

### 5.4 壓力成本

**E-23 壓力格**
cost='stress'：整股衝擊＋0.004；零股買成本率 ×2 再＋0.004、賣＋0.004；手續費全期 0.001425×0.6；其餘相同。出處：XE:64-71、91-99。

---

## 6. 紙上引擎（CE.simulate，mode='consideration'）

**X-01 成本率**
一般：買 0.001425×0.6＋0.001＝0.001855；賣 0.001425×0.6＋0.003＋0.001＝0.004855。壓力：買賣各 0.005。皆為成交金額的比例，無最低手續費、無整股限制。出處：IA:24-28、O18:126。

**X-02 流程**
t0＝第一個訊號日列、t1＝2026-10-12 的列。t0 收盤現金 1.0。
- 非換股日：各持股市值 × GROW[t]。
- 換股日 t（＝e）：tr＝TRADED[t] 且 open 有限；持股由前一日收盤到今日開盤：×GROW[t]×open/close（tr 為假者不變）；凍結＝市值 > 0 且 tr 為假（保留不賣）；Vt＝可交易持股市值＋現金；k＝名單檔數；解每檔目標市值 T：T×k＋買率×Σmax(T_c−h_c,0)＋賣率×Σmax(h_c−T_c,0)＝Vt（在 [0, Vt/k] 二分 60 次取下界）；名單股市值設為 T（凍結股保留原值），現金歸零；開盤到收盤：×close/open。
- 對價結算：結算表照 E-20，但面板用 D-01b（不補 P）；流程同 E-21，差別：股票請求權在換股日以 open 計入持股、非換股日以收盤計入；固定賣出在換股日入帳＝價值×(1−賣率)；不在結算表的永久停牌持股一直凍結。
出處：CE:99-109、CE:143-212。v1 差異：v1 只寫引擎名稱。

---

## 7. 統計口徑

**S-01 seeds 平均（V7X）**
每格（變體, j, 成本, 級距, 模式）：20 個 seed 的日報酬（單位淨值的 pct_change）逐日取算術平均（忽略 NaN；全部為 NaN 的日子當 0），累乘成一條淨值（起點 1）。出處：XS:95-97。

**S-02 年化與超額（V7X，日曆年化）**
- 終點 `e(j)`＝jscan_end_days.json 的 `AB_V7_B10|J{j}`（j0…j8＝2026-10-12、13、14、15、16、19、20、21、22；j9…j12＝2026-10-22）。
- 起點 a＝該 j 第一個訊號日。
- `CAGR`＝(nav[e]/nav[a])^(1/yrs)−1，yrs＝(e−a) 的日曆天數 ÷ 365.25。0050 用同一對日期。`exc`＝CAGR−CAGR_0050（百分點）。
- **2019 起（主欄 Y19on）**：起點＝淨值索引中第一個 ≥ 2019-01-01 的日（2019-01-02）的**收盤**，終點 e(j)。
- 三段（S0612／S1318／S19on，報表的「三段超額」）：區間內第一個交易日的**前一個交易日收盤**為起點（例：S19on 起點＝2018 最後交易日），區間內最後一個交易日為終點。**此口徑與 Y19on 的起點差一天**（N-10）。
出處：XS:33-35、XS:46-67、I20V:41-58。

**S-03 t 值（V7X）**
窗＝訊號日收盤 → 下一訊號日收盤，最後一窗到 e(j)；訊號日序列取自〔已遮蔽〕中 strat=='AB_V7_B10'、seed==0、同一 j 的 signal（**orig 名單的訊號日**，REAL 名單也用這一組）。x＝策略窗報酬 − 0050 同窗報酬；t＝mean(x)/(std(x, ddof=1)/sqrt(n))。2019 起的 t：窗起點 ≥ 2019-01-01。出處：XS:12-18、XS:38-43、XS:27-30。

**S-04 j 彙總與逐格取最差（V7X）**
- 每個指標先算 13 個 j 的值；`*_med`＝中位數（13 個值的第 7 大）；`*_p10`＝numpy.percentile(…, 10)（線性內插）。超額＝「每個 j 的超額」的中位數，不是兩個中位數相減。
- 正式數字＝對每一格每一指標，取 REAL_opt、REAL_p90、REAL_carry 三者 j 中位數（或 p10）的**最小值**；最差變體可逐格逐指標不同。
出處：XS:83-92、I20D:13-14。

**S-05 紙上統計**
- 窗：〔已遮蔽〕個（訊號日收盤 → 下一訊號日收盤；最後一窗到 2026-10-12）；淨值與 0050 取「≤ 該日的最後一筆」。
- **日曆年化（主口徑，o20 裁定）**：`(∏(1+r))^(365.25/days) − 1`，days＝最後一窗終點 − 第一窗起點的日曆天數。
- ia 口徑（並列）：`(∏(1+r))^(12/窗數) − 1`。
- t＝mean(r−r50)/std(ddof=1)×sqrt(n)。
- 分段依窗起點所在月：SEG1 2006-01～2012-12、SEG2 2013-01～2018-12、SEG3 2019-01 起、HOLD 2024-01 起、SEL ～2019-04、VAL 2019-05 起。
- 拿掉最好 3 年：依窗起點年份加總超額，去掉最大的 3 年後重算 t。
- 逐格取最差：對每一指標取 opt／p90／carry 的最小值。
出處：O18:138-172、I20P:70-77、I20D:11-12、FS:88-90、FS:133-136。v1 差異：v1 只有 ia 口徑的文字。

**S-06 報表格**
V7X 七格：100 萬一般、100 萬壓力、300 萬一般、1,000 萬一般、30 萬一般（標準）、30 萬一般（wholelot，E-19）、30 萬壓力。出處：I20V:29-30。

---

## 8. 新發現的未定義處（讀程式時發現、v1 與既有文字規格都沒寫）

| 條號 | 內容 | 程式行為 | 出處 |
|---|---|---|---|
| N-01 | 紙上與 V7X 的外國股定義不同 | 紙上只在放回層用 is_foreign（P-07），V7X 用 FORMASK（L-07）。6 檔不同：1262、1592、2936、6422、6452、8480。〔已遮蔽〕 | PC:118-120、V4:66-79 |
| N-02 | 營收百分位分母不同 | 紙上 opt／p90：÷|el|（含放回股）；V7X：÷|el_rev|（不含放回股） | PC:211-213、VL:135-137 |
| N-03 | 營收底表不同 | 紙上：tw_revenue 年增率＋B1 只補缺；V7X：R_FIX，缺口格改 B1 或設缺 | O18:73-85、VL:63-94 |
| N-04 | 金融遮罩不同 | 紙上 FINMASK；V7X 的 ok 用 FIN2MASK（多了以名稱字串判定的已下市金融股） | V2:84-87、V4:71-75 |
| N-05 | 紙上面板只在 scratchpad | D-01b 不在研究資料夾；缺 2024-01-02～02-05；不補 P | IA:18 |
| N-06 | 兩個 0050 檔 | 紙上統計用 fslib 的副本，V7X 用 jonatw 的檔；sha 不同、檔案大小差 100 位元組 | FS:88-90、XD:76 |
| N-07 | TAIEX 日曆與面板不一致 | 2026-09-20 只在 TAIEX → 〔已遮蔽〕；2018-01-13 只在面板且非 GOOD → 〔已遮蔽〕 | ST:112-121、VL:35 |
| N-08 | cause 的月份 | 用 d 的日曆月、不看 A_UB／MC 回退；回退後才缺值的股票可能得到 'UNKNOWN' 而不放回 | PC:141-165 |
| N-09 | A_UB 的「非 AVAIL」含無證據 | 沒有任何時間戳證據的 (股票, 月) 在觸發日一律視為未公告 | AV:88-93、O18:49 |
| N-10 | 「2019 起」有兩個起點 | Y19on＝2019-01-02 收盤；三段 S19on＝2018 最後交易日收盤 | XS:50-56、I20V:41-44 |
| N-11 | t 的訊號日來源 | REAL 名單的 t 用 orig 名單 seed 0 的訊號日 | XS:13-18 |
| N-12 | 同 delta 的執行順序 | 穩定排序，同值依欄位序（o19 G3） | XE:306 |
| N-13 | 順延日零股現金判定的張數 | L＝min(待買張數, 現金張數上限)，未先套參與上限（o19 G4） | XE:350-355 |
| N-14 | 股數浮點殘差 | 順延單股數可能是浮點（例 125.99999999999994），q＝d%1000 未取整（o19 G5） | XE:351 |
| N-15 | need(T) 的估計口徑 | 用全部張數、不套上限、用當日實際成交值算衝擊；被漲跌停擋住的股票仍估成本 | XE:284-296 |
| N-16 | 衝擊用當日成交值 | F4 只把參與上限改成事前值；衝擊的 OA 仍用 V[t]（事後值） | XE:81、XE:116 |
| N-17 | wholelot 的實際語意 | 見 E-19：目標仍含零股、處置賣出仍可賣零股 | I20V:24-27 |
| N-18 | MC 亂數的相依性 | 依 numpy default_rng（PCG64）與欄位序；PCOL 用現行 stock_list 產業 | ST:137-152 |
| N-19 | held 的定義 | 開盤漲停豁免看「上一期名單」，不是實際持股；|el|<50 的月份不更新 | V2:199-215 |
| N-20 | 引擎 V60 補值 | ffill(limit=3) 連 GOOD 日的 NaN 也補（o19 G1） | XD:61 |
| N-21 | vrank 的收盤條件 | 用前值補收盤 CF（o19 G2） | XD:64 |
| N-22 | 引擎讀快取 | XD 啟動時若 `data/x_cache.npz` 存在就直接讀（sha `12860cde6b0df415`），不重算 | XD:82-87 |
| N-23 | M1_paper 的組成 | 2021-09-23 之後聯集了 disposal_5y（以 date 當公告日）與 universe_announce | I20P:24-33 |
| N-24 | 名單股當日不可成交 | 引擎未防護（名單規則保證 e 有開盤，故未發生） | XE:276 |
| N-25 | 現金請求權入帳後閒置 | 入帳日非換股日時，現金閒置到下一個換股日 | CV:65-66 |
| N-26 | 換股日清空順延單 | 上一期未成交的買賣單一律作廢，重新以當期名單計算 | CV:77、XE:222 |
| N-27 | 處置賣出判定用引擎自己的 DISP | 不看公告日、不隨 M2 改變 | XD:93-107 |
| N-28 | 紙上壓力成本 | 每邊固定 0.5%，不另計稅費 | IA:26-28 |

---

## 9. 與 SPEC v1 的差異總表

v1 的規則性內容只有：`paper.variants`（五元組）、`v7x.definition`（一句話指向 docstring）、`v7x.grid`、`decision`、附註第 5 點（A_UB）。v2 新增或釐清的條文：

| 任務編號 | v2 條號 | v1 狀態 | v2 |
|---|---|---|---|
| (1) REAL 放回名單 | P-03、P-07、P-08、P-09、L-03、L-07、L-09、L-10 | 只有程式定義 | 逐式寫出；外國股名稱來源定為 tw_revenue 最早名稱／stock_list 備援／三個 industry_snap，**不是** delist_official |
| (2) vrank | E-02、E-04、E-05 | 未定義 | 補值版 V60、前值補收盤、全母體、t−1 |
| (3) 停牌順延 | E-17a、E-11 | 未定義 | 不下順延賣單；下一個可成交的換股日才賣 |
| (4) E23 | E-22、E-20、E-21 | 未定義 | 表外＝最後成交＋14 日曆天後的第一個換股日；表內 unknown＝終止日＋20 面板日入帳 |
| (5) 股池 | C-09 | 未定義 | 先篩後取前 300，母體為全面板 |
| (6) A_UB 月份 | P-04、P-04a、L-05 | 附註 5 | 加上 classify 全規則、≠AVAIL 判定、V7X 豁免式；明寫用 d 的日曆月 |
| (7) 遮罩 | P-05、L-06、E-08 | 只有 M0 | M0／M1_paper／DM_BASE／M2 四式；引擎 DISP 另列 |
| (8) 年化與成本等 | S-01～S-06、E-09、E-12～E-19、E-23、L-01、L-04 | 只有 ia 文字 | 全部寫出 |
| 其他 | G-01～G-03、C-01～C-10、D-01～D-11、X-01～X-02、N-01～N-28 | 無 | 新增 |

---

## 10. 已知限制

1. 本文件描述程式現況，**沒有重跑任何程式驗證「照本文件重建＝程式輸出」**；是否足以唯一重建，要由 ACCEPTANCE.md 的三位複製員實測。
2. V7X REAL 名單的驗收目標不在磁碟上（D-11），需由保管員以原程式重產並封存。
3. 紙上面板 D-01b 只在 scratchpad。
4. N-01～N-04 是兩條管線之間的不一致，本文件照實分開描述，未做裁定。
