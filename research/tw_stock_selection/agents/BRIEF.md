# 研究代理共同規範（台股策略研究中心）

## 資料（唯讀）
- 上市股日線面板（2004-01-02～2026-10-22，1,249 檔，含已下市股；2019–2023 只有存活者）：
  `/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/{adj,close_raw,open,high,value,ret,flag,src}.parquet`
  - 寬表：index=日期、columns=股票代碼。adj＝含息還原指數、ret＝含息日報酬、value＝成交值 NT$、open/high＝原始價。
  - 沒有 low 欄、沒有成交量欄（成交量＝value/close_raw）。src：1/2＝證交所官方、3＝Yahoo 5 分 K 近似（2024-02-15～2025-09-23，收盤為 13:20 近似）、4＝官方（2025-09-24 起）。
  - 2024-01-02～2024-02-14 無資料。
- 其他（2021 起為主）：`/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/` 下 history.csv.gz（上市＋上櫃官方日 K，2025-09-24 起）、stock_list.csv（code,name,market,industry）、extras/ 的 rev_5y（月營收）、inst_5y（三大法人，單位張）、pe_5y（月初估值）、exdiv_5y（除權息參考價）、disposal_5y、attention_5y、margin_hist（2025-09 起）、shares_hist。
- 當沖比率：`/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/scr_extra/daytrade_5y.csv.gz`（date,code,dt_vol,dt_ratio）
- 0050 含息：`/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv`（date,adj）；加權指數 `taiex_twii.csv`（同資料夾）
- 共用函式：`/home/user/-/research/tw_stock_selection/btlib.py`（month_ends, period_returns, bench_returns, run_sort, summarize, ann, yearly, tstat）

## 方法紀律（一定要遵守）
1. 事前寫下要測的規則與參數，全部報告，不只報最好的。
2. 挑選期／驗證期切分：價格類 2005-01～2019-04 挑選、2019-05～2026-10 驗證；2021 起才有的資料用 2021～2023 挑選、2024～2026 驗證。只能用挑選期決定參數。
3. 只用「當時已知」的資訊與實際可成交的時點：盤後資料最早隔日開盤成交；鎖漲停收盤視為買不到。
4. 成本：一般來回 0.5%（壓力 1%）；現股當沖來回約 0.23%（稅 0.15%＋手續費）。
5. 流動性：60 日均成交值 ≥ NT$1,000 萬、股價 ≥ 10 元。
6. 對手：0050 含息與同池等權；報告年化、最大回撤、t 值、逐年。
7. 看到太好的結果先懷疑資料（極端值、公司行動 flag 3/4/5、來源 3 的近似收盤）。

## 產出
- 程式與結果放在 `/home/user/-/research/tw_stock_selection/agents/<你的資料夾>/`：`*.py`、`results.json`、`FINDINGS.md`（繁體中文、條列、關鍵數字、結論：可用／不可用／待觀察）。
- Python 一律用 `python3 -I` 執行。不要修改其他資料夾的檔案，不要 git commit（主研究員會統一提交）。

## 第 2 輪後補充
- **面板缺漏：** 2018-01-13（週六）只有 46 檔有值。算 rolling 訊號（52 週高、60 日均量等）前，先把「有值檔數 < 300」的日子剔除再滾動，否則 2018 年大段訊號會變 NaN（作法見 revenue_longhistory/lhlib.py）。
- **長期月營收：** /home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv（2005-01～2026-09，上市＋上櫃，營收單位千元；2013 起只剩存活公司，2013 年有 IFRS 口徑斷點）。讀法見 revenue_longhistory/lhlib.py。
- **可成交回測引擎：** exec_rebuild/exlib.py（M2：換股日收盤算訊號、隔日開盤成交、開盤漲停遞補、成本模型）；長歷史版見 momentum_family/mlib.py。新研究優先 import 這些，不要自己重寫成本與成交規則。
- **處置股正式版事件表：** disposal_reconcile/official_trades_twse.csv、official_trades_tpex.csv。
- **統計口徑：** 一律報扣成本後的 t（月份分群），挑選期與驗證期分開；不要只報全期或 t_date。
