# 研究資料夾索引

每個資料夾是一題研究，通常有事前登記（PREREG／SPEC）、程式、結果檔與結論 `FINDINGS.md`。研究紀錄總表在上一層 `BACKLOG.md`，章程在 `RESEARCH_CENTER_ORG.md`。

| 資料夾開頭 | 是什麼 |
|---|---|
| `s1_`～`s80_` | **月營收驚奇策略的研究**（現行策略由此而來；S45 以後多為穩健性與散戶做法） |
| `r46_`、`r51_`…`r75_`、`r0_` | 獨立複製：第二位研究員不看原程式重算對應的 S 題 |
| `audit_*` | 數據稽核部：已公布數字、原始資料、資料完整性、每日回測紀錄（`audit_daily/`） |
| `official_nav/` | 正式回測管線（產生網站用的正式淨值） |
| `s18_live_list/` | 每月名單產生程式與歷次名單 |
| `s69_forward_validation_plan/` | 已採用的十年實盤驗證計畫 |
| `s75_continuous_2008_2026/` | 2008～2026 連續帳戶回測 |
| `o_*` | 較早的「主策略」研究與前瞻追蹤基礎設施（`o_forward_infra_live/` 每日自動更新，請勿手動改） |
| `w*_` | 網路教學與書上方法的收集回測 |
| `original_*`、`integrate_*` | 早期原創訊號搜尋與整合 |
| 其他（`disposal_*`、`timing_*`、`industry_*` 等） | 早期單題研究 |

回測引擎鏈：`s20lib` → `s17lib` → `s10lib` → `o_ext_canslim_xs/cslib.py`。
