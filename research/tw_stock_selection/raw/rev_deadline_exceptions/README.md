# 月營收申報期限例外表（封存）

給共用引擎 `agents/o_ext_canslim_xs/cslib.py` 的兩個開關讀（**預設都關＝不讀，舊結果逐位相同**）：

| 檔案 | 開關 | 內容 |
|---|---|---|
| `deadline_exceptions.csv` | `cslib.DEADLINE_FIX` | 主管機關特例延長的營收月份。`apply=1` 才改可得日（期限換成 `actual_deadline`，可得日＝該日（含）以後第一個交易日的下一個交易日＝換股日）。 |
| `ins15_list.csv` | `cslib.INS15_MASK` | 適用「次月 15 日」營收期限的公司（保險業及具保險子公司）。自 `start_month` 起，訊號日（含位移）早於 15 日期限日（次月 15 日（含）以後第一個交易日）時，該股當月 SURGE 當缺值。 |

## deadline_exceptions.csv
- 欄位：`rev_month`（營收月份）、`statutory_deadline`（法定 10 日、遇假日順延，以交易日近似）、`actual_deadline`（延後後的期限）、`apply`（1＝引擎採用；0＝只註記）、`scope`、`note`、`source`。
- 2019-01 → 2019-02-15（全體；換股日 2019-02-12 → 2019-02-18）。
- 2023-03 → 2023-04-12（全體；換股日 2023-04-11 → 2023-04-13）。
- 2020-01（疫情，只限經申請的受影響公司，沒有逐檔名單）：`apply=0`，引擎不改；S40 最壞情境（整月延後）2019 起約 −0.81 個百分點。
- 來源：ETtoday 2019（https://finance.ettoday.net/news/1346406）、中央社 2020-02-03（https://www.cna.com.tw/news/afe/202002030173.aspx）、工商時報 2023-02-22（https://www.ctee.com.tw/news/20230222700065-430298）；整理見 `agents/rev_announce/FINDINGS.md` 第 2 節、`agents/s40_revenue_deadline_pit/data/deadline_calendar.csv`。金管會／證交所原文從本環境讀不到，以上依公開新聞。

## ins15_list.csv
- 欄位：`code`、`name`（僅供辨識）、`category`、`start_month`（從哪個營收月份起適用）、`end_month`（空白＝仍適用）、`deadline_day`（15）、`source`。
- 名單＝S40 `s40lib.INS15` 的 18 檔（上市金控中具保險子公司者、上市保險、2905、2207）＋2884 玉山金（S40 註明 2026-09 實際於 15 日公告；比照 S41 從 2026-01 起算）＝S41 的 GROUP。
- 依據：金管會修正「公開發行公司財務報告及營運情形公告申報特殊適用範圍辦法」第 3 條（金管證審字第 1140346436 號，2025-07-01；聯合報 https://money.udn.com/money/story/5607/9162636、優分析 https://uanalyze.com.tw/articles/9242343173 轉述），2026-02 申報 1 月營收起適用。原始條文讀不到，以上依新聞。
- **每年依金管會公告更新**：適用對象有增減（新掛牌／下市的保險業、金控取得或處分保險子公司、合併，例如玉山金與三商美邦）時，
  - 新增：加一列，`start_month` 填開始適用的營收月份；
  - 不再適用：在原列 `end_month` 填最後一個適用的營收月份（不要刪列，回測要能重現當時名單）；
  - 期限再改：改 `deadline_day` 或另加一列分段。
  - 改完更新本檔「更新紀錄」與下方 SHA256。
- 注意：實盤 `agents/s18_live_list/live_surge/make_list.py` 不讀這份名單（實盤本來就看不到還沒公告的營收），這份只給回測引擎用。

## 誰在讀
- `cslib.deadline_exceptions()`、`cslib.rev_avail_days()`（`DEADLINE_FIX=True` 且 pit）。
- `cslib.ins15_list()`、`cslib.ins15_hidden_cols()`、`cslib.ins15_mask_row()`（`INS15_MASK=True`）；目前接在 `agents/s10_exit_details/s10lib.make_signal_s10`（現行 E0 引擎）。
- 驗證：`agents/deadline_engine_fix/`（兩個都關＝舊結果逐位相同；都開＝S40 ALL 版逐位相同）。

## 更新紀錄
- 2026-11-17 建立（deadline_engine_fix）。

## SHA256
```
c537ad07524d1f0f26de82a6a4d90589081497d1f0653e33aeb9698257520e33  deadline_exceptions.csv
deca13fcfee6c5206c6c96cc2ff0d72034cc46e0e52213a0b89936edbae9dc16  ins15_list.csv
```
