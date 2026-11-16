# 網站狀態

- `holdings_after_2026-11.csv`：2026-11 換股（11/12 開盤）後的策略持股（100 萬比例縮放的回測持股＋新買 1802），0050 共 5,965 股、其中底倉 1,995 股。
- `cash_after_2026-11.txt`：同一時點的策略現金 **12,126 元**（2026-11-16 稽核更正：原本寫「現金約 0」是錯的）。
  - 原因：換股後持股是用訊號日收盤估的，沒有照成交日開盤重算；賣出股開盤都比收盤高，賣出實得較多，照上面的股數會剩約 12,126 元現金（研究中心嚴成本）。
  - 照 11/12 開盤重算，1802 應為 882 股（計 0.3% 滑價為 880 股）、0050 應為 6,067 股；**上線起點沿用已發布的股數（1802 862 股、0050 5,965 股），不改**，差額以現金記在這個檔。
  - `build_site_data.py` 會自動讀同月份的 `cash_after_YYYY-MM.txt`（和 `holdings_after_YYYY-MM.csv` 放同一個資料夾），計入 live.json 的策略淨值（起點和每日都加上現金）。
  - 出處：`research/tw_stock_selection/agents/audit_live_2026-11/AUDIT_REPORT.md` 問題 3、audit.json 的 `cash_left_implied_by_state`。
- 下次（2026-12）產生名單：
  `python3 -I -B research/tw_stock_selection/agents/s18_live_list/live_surge/make_list.py list --month 2026-12 --holdings xq_scripts/site/state/holdings_after_2026-11.csv --cash 12126 --base-shares 1995 --repo research/tw_stock_selection/raw/jb7k6295r7-bot_tw-stock-data --repo <暫存>/tsd --out-prefix research/tw_stock_selection/agents/s18_live_list/lists/2026-12_strategy`
  （2026-11-16 起 make_list.py 預設開啟除權息參考價修正，輸出 meta.exright 會標明；TWT49U 封存 `research/tw_stock_selection/raw/twt49u/` 要每月補，見該資料夾 README。）
  然後**照成交日開盤的實際價格**算換股後持股（照輸出的賣出／續抱／買進與股數），存成 `holdings_after_2026-12.csv`，剩下的現金存成 `cash_after_2026-12.txt`（沒有現金也寫 0）。
- 回測淨值與逐年（profit.json）：`python3 -I -B xq_scripts/site/make_profit.py`（2026-11-16 起預設讀 `xq_scripts/site/navs_exright.csv.gz`＝除權息修正後、XQ版用統一後換股日，由 `research/tw_stock_selection/agents/exright_engine_fix/run_engine.py 1` 產生；修正前是 s19 navs.csv.gz，可當第一個參數給；0050 用精確含息還原價，不用 navs 的整數欄；週標籤＝當週最後一個交易日）。
- 網站資料：`python3 -I -B xq_scripts/site/build_site_data.py <月份 csv> <策略持股 csv> <月份 json> xq_scripts/site/profit.json <今天> <下次訊號日> <下次執行日> xq_scripts/site/data [<官方日K倉庫> <換股後持股 csv> <上線起始日>]`（後三個參數讓 holdings 加股數、起始價與 0050 一列，並產生 live.json；持股表「本月以來%」＝最新收盤 ÷ 起始價 − 1，換股清單 actions 的欄位叫「訊號日以來%」＝對訊號日收盤）
- 網站：https://claude.ai/artifact/BB8g52evYQx8gSKGXyECnk（資料集 status、actions、holdings、top20、nav、yearly、live、risk；上傳新檔後更新 datasets/<id> 的 source.url 與 updated）

## 2026-11-16 產品擁有者的選擇
- 目前沒有持股，選 **A：不在月中進場，等 2026-12-11 收盤後的名單，2026-12-14 開盤照名單買進**（從空手開始）。
- 網站的「策略持股」仍是 100 萬比例的策略持股（用來追蹤策略本身），不是他的實際持股。
- 策略提案已放進趨勢雷達 repo（a4064940649-coder/trend-radar）分支 proposal/revenue-surge-strategy 的
  trade-force-radar/docs/策略提案_月營收驚奇SURGE/，給那邊的總監審；未改那邊的程式。
- 研究中心的每日研究照常進行。
