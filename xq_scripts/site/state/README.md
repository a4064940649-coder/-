# 網站狀態

- `holdings_after_2026-11.csv`：2026-11 換股（11/12 開盤）後的策略持股（100 萬比例縮放的回測持股＋新買 1802），0050 共 5,965 股、其中底倉 1,995 股、現金約 0。
- 下次（2026-12）產生名單：
  `python3 -I -B research/tw_stock_selection/agents/s18_live_list/live_surge/make_list.py list --month 2026-12 --holdings xq_scripts/site/state/holdings_after_2026-11.csv --cash 0 --base-shares 1995 --repo research/tw_stock_selection/raw/jb7k6295r7-bot_tw-stock-data --repo <暫存>/tsd --out-prefix research/tw_stock_selection/agents/s18_live_list/lists/2026-12_strategy`
  然後算換股後持股（照輸出的賣出／續抱／買進與股數），存成 `holdings_after_2026-12.csv`。
- 網站資料：`python3 -I -B xq_scripts/site/build_site_data.py <月份 csv> <策略持股 csv> <月份 json> xq_scripts/site/profit.json <今天> <下次訊號日> <下次執行日> xq_scripts/site/data [<官方日K倉庫> <換股後持股 csv> <上線起始日>]`（後三個參數讓 holdings 加股數、起始價與 0050 一列，並產生 live.json）
- 網站：https://claude.ai/artifact/BB8g52evYQx8gSKGXyECnk（資料集 status、actions、holdings、top20、nav、yearly、live、risk；上傳新檔後更新 datasets/<id> 的 source.url 與 updated）

## 2026-11-16 產品擁有者的選擇
- 目前沒有持股，選 **A：不在月中進場，等 2026-12-11 收盤後的名單，2026-12-14 開盤照名單買進**（從空手開始）。
- 網站的「策略持股」仍是 100 萬比例的策略持股（用來追蹤策略本身），不是他的實際持股。
- 策略提案已放進趨勢雷達 repo（a4064940649-coder/trend-radar）分支 proposal/revenue-surge-strategy 的
  trade-force-radar/docs/策略提案_月營收驚奇SURGE/，給那邊的總監審；未改那邊的程式。
- 研究中心的每日研究照常進行。
