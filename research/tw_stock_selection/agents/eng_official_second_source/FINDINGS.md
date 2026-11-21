# FINDINGS：資料第二來源小組——官方公開資料第二來源與換股日保險（2026-11-21）

（工程師不能寫 .md，由總監依其最後回覆存檔；欄位、端點全表、時程範本見 manifest.json，交叉核對見 crosscheck_2026-11-21.json。）

## 一句話
官方第二來源已接上並實測：上市月營收兩來源 1,091 家完全一致、除權息 480 筆三方逐欄相同、用官方營收演練出的 2026-11 名單與正式名單前 20 名／F1／換股計畫全同。兩個限制：證交所主站一天第 4 次請求就回 307（已照規則停手）；官方月營收在訊號日當晚是否已切到新月份還不知道，那晚官方備援不保證能用。

## 交叉核對
| 項目 | 官方 | GitHub 倉庫 | 不一致 |
|---|---|---|---|
| 上市營收 2026-10 | 1,091 家 | 1,091 家 | 0 |
| 上櫃營收 2026-10 | 898 家 | 898 家 | 1 家（7782：71,466 對 76,697 千元，原因推測為事後更正；不在策略股池） |
| 除權息 11/1～11/20 | 55 筆 | 55 筆（封存 55） | 0 |
| 除權息 2026-07 整月 | 425 筆 | 425 筆（封存 425） | 0 |
備援演練：官方營收出的 2026-11 名單，前 20 名每一欄、F1、換股計畫與正式名單全同；差別只在官方檔沒有首見紀錄、不做截止檢查（多算 15 檔截止後才公布的金融股），拿掉後 csv 逐位元相同。

## 可用官方端點
已接：上市月營收（openapi.twse.com.tw t187ap05_L，只給最新一個月）、上櫃月營收（www.tpex.org.tw mopsfin_t187ap05_O）、上市除權息 TWT49U 與減資 TWTAUU（www.twse.com.tw，一天最多 4 次、同站間隔 20 秒）。可用未接：上市每日收盤 STOCK_DAY_ALL（日 K 第二來源候選）、上市／上櫃最新一季財報、重大訊息。不可用：公開資訊觀測站靜態頁（307 安全頁，不繞）。

## 程式（agents/eng_official_second_source/）
fetch_official.py（revenue／exright／reduce／probe／convert-*／budget；請求上限寫死、每次請求記 _requests.csv、被擋整天停；UA 不含個資）、cross_check.py、rebalance_readiness.py（總檢查 (a)～(f)，印【可以出名單】或【還不能】與缺什麼）、official_to_repo.py（官方快照轉小倉庫當備援）、compare_lists.py、daily_source_check.py（每日比對，寫 agents/audit_daily/source_check_log.csv）。快照在 raw/official_snapshots/（gzip＋index.csv，內容相同不重存；一年估 45MB）。

## 換股日時程（台北時間；2026-12）
1. 12/1～12/10：每日例行跑 daily_source_check.py，觀察官方資料年月哪天變 2026-11。
2. 12/11 18:30：更新倉庫後跑 rebalance_readiness.py --signal-date 2026-12-11，看兩來源營收家數。
3. 12/11 21:00 前：fetch_official.py revenue 抓備援快照（必須在 22:00 截止前）。
4. 12/11 21:30：更新倉庫 → 重產 TWT49U 月檔與減資封存 → 總檢查加 --refetch-events。【可以出名單】→ monthly_rebalance.py plan；「改用官方備援」→ official_to_repo.py 2026-11 <固定資料夾>，plan 與之後 settle 都加 --extra-repo；兩來源算出的前 20 名或 F1 不同 → 人工確認後加 --ack-mismatch 才放行。
5. 12/11 23:15：21:30 沒過才做第三次。
6. 最晚：目標 12/11 23:30；12/12 12:00 前要有名單，否則通知產品擁有者；絕對底線 12/14 07:30。
7. 來不及：照實說原因、已公布幾家、還缺什麼、下次檢查時間；名單出來前不換股；不用上月營收、涵蓋率不到 95% 的營收或估計值頂替。
8. 12/13 20:00：--no-fetch 再確認；營收彙總表若換版本，在暫存區重出並比對，不自動改名單。
9. 12/14 07:30：每日回測 PASS；fetch_official.py exright --start 2026-12-14 --end 2026-12-14 查執行日有無除權息落在要買賣的股票（有的話掛單漲跌停價要重算）。

## 總監已做
套用 5 筆文件 patches；套用 monthly_rebalance.py 的 --extra-repo diff（只驗過語法與 --help，plan／settle 尚未用它實跑）；另排兩個提醒（12/11 18:30 第一次檢查、20:55 抓官方備援快照）。每日例行加：`python3 -I -B research/tw_stock_selection/agents/eng_official_second_source/daily_source_check.py --tsd <GitHub 日K倉庫>`（接在每日回測檢查後，約 10～15 秒、3 次請求）。**不要另外手動 curl 官方網站重試。**

## 沒做到
主站被擋後官方除權息只用稍早快照離線驗過；減資沒有完整官方回應可逐筆比；官方日 K、面額變更、上櫃除權息第二來源未接；「GitHub 沒有當月營收、官方有」只用假快照測過。
