#!/usr/bin/env python3
"""integrate_o17 事前登記（只寫一次）。o17 通過審查的兩題都不產生選股訊號、沒有自己的報酬序列：
  (1) o_rev_avail_ds_pit_count（題 1，可用）：主策略 PIT 勘誤普查（只計數）：D=S／D>S 營收前視上界 31.71%（下界 0）、2013 年後存活者缺口 U_opt 15.30%、處置 1.18%。
  (2) o_engine_delist_consideration（題 2，待觀察）：共用引擎下市「收購／換股對價」口徑（consideration；freeze 仍是預設）。
本整合把兩題當成主策略（排除金融 n10）的「勘誤濾網」，量化報酬敏感度。用法：python3 -I -B prereg.py"""
import json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
RA = AG / 'o_rev_avail_ds_pit_count'
CZ = AG / 'o_engine_delist_consideration'
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()

FILES = [HERE / 'run.py', HERE / 'smoke' / 'smoke.out',
         RA / 'SPEC.json', RA / 'results.json', RA / 'pclib.py', RA / 'avail.py', RA / 'data' / 'c0_slots.csv', RA / 'data' / 'slot_avail.csv',
         RA / 'raw' / 'disposal_official_universe.csv', RA / 'raw' / 'ts_mops_ann_date.csv', RA / 'raw' / 'ts_tasi_upstream_firstseen.csv.gz', RA / 'raw' / 'ts_news_mentions.csv.gz',
         CZ / 'results.json', CZ / 'cz_engine.py', CZ / 'events_v2.csv',
         AG / 'o_rev_trunc_2009_erratum' / 'ERRATUM2.json', AG / 'o_g8_protocol_fin_erratum' / 'ERRATUM.json',
         AG / 'main_v2' / 'v2_core.py', AG / 'main_v2' / 'navs_gapfix.parquet', AG / 'independent_audit' / 'ia_core.py',
         AG / 'o_forward_sequential_v2' / 'fslib.py',
         Path('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv'),
         Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/rev_5y.csv.gz')]

PREREG = dict(
    written_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    topic='o17 整合：營收可得時點／2013 後存活者缺口／處置 PIT 勘誤（題 1）＋ 下市對價引擎（題 2）× 主策略 C0（52 週高 0.5＋營收 3 月年增 0.5、前 300 池、月換 10 檔、排除金融；main_v2 gapfix 官方名單）',
    what_is_integrated=('兩題都沒有個股分數也沒有報酬序列，依 S1 不產生新選股訊號。可整合的是把勘誤當成主策略的濾網，做報酬敏感度；不取代主策略正式數字，也不回寫兩題，'
                        '兩題的事前計數判定（G5「前視未排除」、主策略可執行正式數字旁同一條、G4 第二／三段「存活者偏誤未排除」、處置只更新 o16 第 5 節數字、V7X「下市對價口徑差」加註）一律不因整合結果改變。'),
    variants=dict(
        C0='V2(gapfix, YL, Y5, exfin=True).plan(10, official)；必須逐日重現 main_v2/navs_gapfix.parquet EXFIN|n10|{normal,stress}（差 < 1e-9），cz_engine freeze 與 ia_core.simulate 差 < 1e-12。',
        A_UB=('營收可得時點上界濾網：在 v2_core.deadline_case 為 D=S 或 D>S 的訊號日（正式 91 日），凡題 1 的凍結分類器 avail.Evidence(\'sig\').classify(S, code, m−1, \'V6\') 不是 AVAIL 的股票，'
              '營收分數改用前一個月（V2 的 unavail 機制：y3(d, back=1)，與 main_v2 AB 相同）。核對：C0 被選槽中被回退者＝題 1 上界 777 槽，逐筆相同。'),
        A_LB='下界濾網：只有 UNAVAIL 的股票回退（題 1 下界 0；煙霧測試顯示股池內也是 0 → 名單與 C0 完全相同，只描述）。',
        A_WORST='D=S／D>S 訊號日全部回退（v2_core.make_unavail(\'worst\')），描述。',
        U13='存活者缺口放回：pclib.VU(\'opt\', lo=2013-01-01, hi=2026-12-31, GOV)（題 1 面板 A 的判定定義）。核對：放回槽 2013～2026 251、2013～2018 227、2019～2026 24，與題 1 相同。',
        UFULL='放回期間 2006～2026（lo＝2006-01-01）。核對：2006～2013 放回槽位與 ER2 的 162 筆逐筆相同。',
        C_DSP='官方處置期濾網：訊號日 ≤ 2021-09-23 時，題 1 raw/disposal_official_universe.csv 的處置期涵蓋訊號日或成交日者加入排除（與 ia_core 原遮罩取聯集）。核對：C0 槽中被涵蓋者 2006～2013 4 槽、2014～2021-09-23 11 槽，與題 1 R3（d_sig ∪ d_trd）逐筆相同。',
        ALLPIT='合併：UFULL 放回 ＋ A_UB 回退 ＋ C_DSP 官方處置（三個勘誤一起套，「已知 PIT 問題全部往不利方向處理」的壓力版）。'),
    engines=('題 2 的 cz_engine.simulate：freeze（現行正式）、nofreeze、dzero、cons_main、cons_lo、cons_hi、cons_c50（events_v2.csv sha 66e80725…、d＝0.057）。'
             'C0、U13、UFULL、ALLPIT 跑全部 7 種；A_*、C_DSP 跑 freeze＋cons_main／lo／hi。壓力成本只跑 freeze、cons_main、cons_lo。'),
    periods='ALL（245 窗，2006-04～2026-09 起點）、SEG1 2006～2012、SEG2 2013～2018、SEG3 2019 起、SEL ～2019-04、VAL 2019-05 起、HOLD 2024 起、Y0613、Y14ON。窗＝訊號日收盤到下一訊號日收盤，依起點月份歸期；0050 含息（fslib.e50）。',
    stats='每個變體 × 成本 × 引擎 × 期間：年化（窗幾何）、窗頻與日頻最大回撤、對 0050 年化超額與 t、和 C0（同成本 freeze）的月差均值、t、年化差、相關、超額相關、不同窗數、拿掉最好 3 年 t；放回類另報對自己 freeze 的差（只看引擎效果）。',
    decision=('判定對（vs C0 同成本 freeze）：A_UB、A_LB、C_DSP 用 freeze；U13、UFULL、ALLPIT 用 cons_main／cons_lo／cons_hi／cons_c50，四者都算（依題 2 規則取絕對差最大者，偏向加註）。'
              'R1 可忽略＝一般成本下所有列 |ALL 年化差| ≤ 0.3pp 且 |ALL 對 0050 t 差| ≤ 0.05，且 SEG1／SEG2／SEG3 對 0050 幾何超額都不變號；'
              'R2 方向不變＝不符 R1，但所有列（含壓力）三段超額都不變號；R3 可能翻號＝其他。門檻方向：「> 觸發、≤ 不觸發」。'
              '後果：任何一級都不推翻兩題的事前計數判定；R1／R2 只決定加註文字（附年化差、t 差區間），R3 產生「建議下一輪以該口徑重算主策略正式數字」。不改 SOP、不改主策略規則。'
              '另報 ALLPIT×cons_lo 的全期 t 與 2019 起 t 是否達 G2（3.0／2.0），只描述。'),
    E_mix=('不做。理由：(i) 兩題都沒有可當第二條腿的報酬序列；勘誤版主策略和 C0 只差少數槽位，當第二條腿等於自己混自己；'
           '(ii) 主策略×0050 固定比例混合在「不再開」清單；integrate_o14 已事前登記並跑過 w＝0.8／0.7／0.5／0.3，全部「不可用」。FINDINGS 引用 o14 數字。'),
    V7X=('不重跑 V7X 可執行引擎：V7X 的 AB_V7 名單本來就用 MC 延後機率模型處理未公告營收（不是 C0 的「全部當已公告」），'
         '把 A_UB 濾網接到 V7X 需重建 13 j × 20 seeds 的名單（題 2 的 V7X 重算 25 分鐘），超出整合範圍；以 A_UB 的 SEG3 紙上差描述方向，並建議研究中心另案。'),
    predictions=('事前預測（寫在看任何變體報酬之前）：'
                 'A_UB：D=S 日幾乎全部股票回退（煙霧測試：股池前 300 平均 269 檔回退，C0 名單變 147 槽），效果接近 main_v2 的 AB_worst；'
                 '預測全期年化差在 −2.0～−0.3pp、t 差 −0.25～−0.05，三段不變號 → R2；A_LB 與 C0 完全相同 → R1；A_WORST 與 A_UB 相近（±0.5pp）。'
                 'U13（cons_main）：2013～2018 放回 227 槽，主要是 2325 矽品、2311 日月光、1704 榮化、3454 等後來被併購或下市的公司；預測全期年化差 −3～−0.5pp，SEG2 超額變小但仍 > 0 → R2；'
                 'freeze 會比 cons_main 差（凍結死錢），cons_lo 比 cons_main 差。UFULL：比 U13 再差 2～4pp（ER2 段的 −4.18pp freeze／對價口徑約 −0.7pp），三段不變號 → R2。'
                 'C_DSP：15 槽，|年化差| ≤ 0.3pp → R1。ALLPIT：年化差 −5～−1.5pp，全期 t 降到約 2.0～2.4，三段可能有一段（SEG2）接近 0 → R2 或 R3；G2 不過。'),
    honesty=('寫 PREREG 前我已讀過：題 1 的 SPEC、results.json（headline：上界 31.71%／下界 0、U_opt 15.30%、處置 1.18%；R2 放回的發行人與年度分布）、pclib.py、avail.py、s4_count.py、data/*.csv；'
             '題 2 的 results.json（C0 Δ＝0；U_opt 對價口徑 27.61%、t 2.536，freeze 24.10%；V7X 界限 −1.55～+0.47pp）、cz_engine.py、cz_lib.py、s9_ia.py；'
             'main_v2 FINDINGS 與 AB 數字（EXFIN n10 AB 27.81%、t 2.58，MC 平均）、integrate_o16 FINDINGS／run.py（C0 28.28%、t 2.643、三段 13.17／9.71／16.62pp）。'
             '登記前的煙霧測試（smoke/smoke.out）只核對：C0 重現（差 0；C0 報酬早已公開）、名單層級的計數與兩題公開計數逐筆相同（777／0 槽、251／227／24、162、4／11）、各變體名單異動槽數（A_UB 147、A_WORST 165、U13 251、UFULL 370、C_DSP 15、ALLPIT 468、A_LB 0）、'
             '持有事件表代碼的槽數、結算規則數 141；沒有算任何變體的報酬、t 值或淨值。2006～2026 資料在主策略各研究都看過，沒有樣本外。'
             '本 PREREG 沒有外部錨定（子代理不得 git commit）；請研究中心以 written_utc 早於 RUN_UTC 為準。'),
    tests='判定用：6 個判定對（A_UB、A_LB、C_DSP、U13、UFULL、ALLPIT）× 三級規則；其餘描述。',
    sha256={str(f): sha(f) for f in FILES},
)
if __name__ == '__main__':
    RJ = HERE / 'results.json'
    assert not RJ.exists(), 'results.json 已存在：PREREG 只寫一次'
    h = hashlib.sha256(json.dumps(PREREG, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    RJ.write_text(json.dumps(dict(PREREG=PREREG, PREREG_sha256=h), ensure_ascii=False, indent=1), encoding='utf-8')
    print(h, PREREG['written_utc'])
