# 研究中心待辦與每日紀錄

專案期間：2026-10-27～2027-01-27（3 個月）。每日排程：「台股研究中心每日研究」，每天台北時間 06:49（trigger id trig_018uxTz588Aup718qjUU2SD3），到期後自動停用。每天的排程依這份清單派工；做完的題目移到「已完成」，新想到的題目加到「待辦」。

## 每日流程
0. 每月換股日（每月第一個 ≥ 11 日的交易日）收盤後：執行 `python3 -I research/tw_stock_selection/main_picks_today.py --data /home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data --n 10` 產生主策略名單，記錄到 agents/live_tracking/（真實樣本外追蹤），並回報使用者。資料要先更新（tw-stock-screener repo `git pull`）。
1. 環境檢查：`/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/adj.parquet` 不存在時，先執行 `bash research/tw_stock_selection/restore_env.sh`。
2. 收回昨天還在跑的工作：FINDINGS 存進各資料夾，提交推送。
3. 開一輪網路教學工作流 `.claude/workflows/tw-web-strategy-harvest.js`，scout_set 依序輪流 tw → global → posts；上一輪剩下的 leftover 優先。
4. 從下方待辦挑 1～3 題，派研究代理。
5. 把結論寫進 `台股個股選股規則實證研究報告.md` 與線上報告（https://claude.ai/artifact/5u7fyL7i9QrVj2w6nwiNMZ），更新本檔，提交推送，回報使用者。

## 規則
- 只用公開資料與公開貼文；不找、不用未公開重大消息（證交法 157-1 條）。
- 一律事前登記參數、扣成本、挑選期與驗證期分開；新規則必須和主策略比較。

## 待辦（依優先順序）
- 上櫃面板（277MB，scratchpad/tpex_panel）不在 repo：容器回收後先跑 restore_env.sh 再執行 agents/tpex_main/build_tpex_panel.py 重建。合併版 n10（待觀察）待補齊 2023-01～09、2025-04～09 上櫃日 K 後重測。
- 營收公告日：蒐集 rev_hist 以外的歷史公告日（GitHub），確認金控是否常態在 10 日後公告；若是，引擎改成「只用訊號日前已公告的營收」，並事前登記重測「營收腿排除金融保險業」。
- 面板成交值異常（2025-04-30～07-04 約 10 倍）：檢查主策略流動性前 300 在這段是否受影響（各股同倍率則排名不變），必要時用 daytrade_5y 還原官方量修正。
- w3 偵查剩下的 11 條（agents/WEB_w3_leftover.json）：布林壓縮、ConnorsRSI、Double 7、R3、TPS、Radge 週末趨勢、漲跌家數擇時、PSY、高股息低波、量能谷底、RSRS 擇時。
- 營收驚喜 SURGE（w3 待觀察）：每月追蹤真實樣本外。
- 集保千張大戶週資料：每週一 `git -C /home/user/wirelessr/tdcc-opendata-archive pull`，累積滿 2 年後用 agents/insider_public/tdcc_study.py 原規則重跑（目前「排除大戶減、散戶增」待觀察）。
- PTT 空方文避開名單（forum_gurus 待觀察）：每月用 a607ernie/ptt-stock-data 新快照補資料重跑。
- 主策略改 10～15 檔（concentration_equity 待觀察）：每月追蹤 10 檔與 30 檔的真實樣本外差異。
- w1 偵查剩下的規則（scratchpad/w2_leftover.json）：寶塔線、朱家泓做多 SOP、爆量長紅後量縮、費波那契回檔、營收雙增、融資減法人買、殖利率法、EPS 成長、魔術公式、低基期起漲、投信連買、多頭排列＋移動停損。
- 處置股紙上交易：每天用 disposal_today.py 記錄，累積 12 個月。
- 上櫃股長期資料：找 2005 年起的上櫃日 K（目前只有 2025-09 起），把主策略延伸到上櫃。
- 「排除 G2 處置股」（處置股中營收衰退、離 52 週高較遠者）：需新的事前登記。
- 主策略 2026-11 起的真實樣本外追蹤：每月換股日記錄名單與報酬。

## 已完成（摘要見報告第 11～13 節）
- 網路教學 w7、w8（9 條全部不可用；累計 51 條）、上櫃延伸、營收公告時點（實盤預期下修）。
- 網路教學 w3～w6（共 42 條，只有 SURGE 待觀察）、多策略綜合投組（5,076 組合）、主策略細部改良（13 版本無一通過）、每月選股工具 main_picks_today.py（--ex-fin）。
- 第 3 輪：網路教學 w2（8 條全部不可用）、集中度與淨值曲線濾網、PTT 大神貼文、合法公開內部人申報（多數資料不足）。
- 第 1 輪交叉審查、第 2 輪（長期月營收樣本外、處置對帳、動能家族、緩衝區、ETF 成分股）。
- 交叉搜尋 129 組合、處置 × 主策略前 100、產業營收循環長期、擇時開關日頻、網路教學 w1。
