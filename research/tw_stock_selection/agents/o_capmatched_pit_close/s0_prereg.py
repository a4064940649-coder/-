#!/usr/bin/env python3
"""第 0 步：事前登記（算任何報酬之前）。寫 data/PREREG.json，並把 sha256 與時間戳記進 results.json['prereg']。
python3 -I -B s0_prereg.py"""
import sys, json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
F = HERE / 'data' / 'PREREG.json'
assert not F.exists(), 'PREREG 已存在，不得覆寫'
O7 = AG / 'o_capmatched_selection'; O8 = AG / 'o_0050_tilt_pit'
SPEC = r"""
題目：o7「大型股選股力 t 4.02」的 PIT 重驗與結案（確認性檢定；不探索、不迭代模型；只換輸入、不換載體）。

一、策略（逐字照抄 o7 凍結版，程式直接唯讀 import o7 的 cmlib.py／cmstrat.py，不改一行）
- F1_K5_N20_sqrt：前 5 大成分（必含台積電）與全部金融成分照 0050 自由流通複製權重 R 持有；非持有成分若 R > 5pp 至少保留 R−5pp；
  其餘權重給「市值名次 > 5、不在持有集合」的合格股中分數最高的 20 檔，權重 ∝ 市值^0.5；單檔主動權重 ≤ 5pp，觸頂往下加檔；
  開盤漲停且未持有者跳過、下一名遞補；其他成分權重 0；台積電主動權重 0。
- 分數＝0.5×52 週比百分位＋0.5×近 3 月營收年增率百分位（合格集合內排名）；合格：市值名次 ≤ 300、非金融、非 KY／DR、
  60 日均成交值 ≥ NT$1,000 萬、收盤 ≥ 10、52 週比有值、營收（V4 已公告，最壞口徑）有值、非處置中、成交日有開盤成交。
- 訊號日 d 收盤算、d+1 開盤成交；引擎 cmlib.simulate_w（凍結、二分法成本、下市轉現金 v2），日報酬 exfix（grow_exfix）。
- 成本：一般（買 0.0855%＋0.1%；賣 0.0855%＋0.3%＋0.1%）。本題快速引擎 pcore.sim_fast 與 simulate_w 數學相同，s1_gate 逐窗核對。

二、隨機分數（安慰劑）
- 1,000 次；每次每個訊號日抽 rng.random(N)，只在合格集合內有值（同層、同合格池、同權重規則、同漲停遞補）。
- 種子：np.random.default_rng(10000 + s)，s＝0..999；每個設定、每個日期集合都用同一組種子（共同隨機數）。
- 檢定量：每窗超額 x_k＝實際窗報酬 − 1,000 次隨機窗報酬平均；t＝mean/(sd/√n)（月序列、OLS）；另報排列 p＝P(隨機的平均對 0050 超額 ≥ 實際)。

三、日期
- 13 個 j（J0～J12）一律用 o_v7_listing_exec_engine/s4_jscan7.py（st_core.day_for）的定義：錨定月 m 的法定期限 D_m＝m 月 10 日起第一個
  加權指數交易日（2019-02 用 2019-02-15、2023-04 用 2023-04-12 起算）；J_j＝D_m 之後第 j 個加權指數交易日；該日須是面板 GOOD 日（有值 ≥ 300 檔）
  且下一面板日也是 GOOD 日，否則該月跳過。營收可用性仍用 o7 cmlib.unavail（期限日 ≥ 訊號日 → 改用 m−2..m−4）。
- 官方日：o7 定義（每月第一個日 ≥ 11 的面板交易日）。
- MAIN（P1）：錨定月 2019-05～2026-09，最後一窗終點＝2026-10 的同 j 日（不存在則面板最後一日 2026-10-22）；官方日終點 2026-10-12。
- PRE（P5）：錨定月 2014-01～2019-04，終點＝2019-05 的同 j 日（官方日：2019-05 官方日）。

四、輸入版本（P2 拆解的替換順序，事前寫死）
- o7：shares_est（o7 回推股數）、o7 面板、o7 規則重建 2015-09 前臺灣50 名單。
- (i) PIT 股數：o_0050_tilt_pit/data/shares_pit.parquet，只用在 shares_est 有值的格子（下市股仍排除）；2015-09-18 前臺灣50 名單改用 o8 membership_pit.json。
- (ii) ＋修正：4763 用 TDCC 真值（2019-09-04～2023-07-26 ÷12.65；2023-07-27～2025-06-29 ÷10；同 replicate_o8/rcore.py）；
  2317：2018-06-29 之後到 2018-10-25 固定為 2018-06-29 股數，2018-10-26 起減資後股數。
- (iii) ＋納入下市股＝PIT 版本 (a)：shares_pit 全部格子（含下市股）＋修正；o_infra_v3_rebuild/delist_prices.parquet 補 2019-05-17～2023 下市股價格
  （只補 o7 面板缺值格，同 o8 tlib）；V7 上市身分（o_v7_listing_exec_engine/data/listed_twse.parquet）為 False 的格子市值設為缺值。
- PIT 版本 (b)：版本 (a) 的股數整體落後 63 個面板交易日（模擬季報公告延遲），其餘同 (a)。
- shares_pit 的已知限制（下游引用必註明）：季內對數內插用到下一季股本（前視）；4763 錯誤（已修正）；2317 減資提前內插（已修正）。
- 已知覆蓋限制：月營收資料（V4 R_FIX／Y5_FIX）對 2019-09 以後下市的公司沒有值，所以下市股可進入市值名次、0050 複製權重，但幾乎不能當衛星
  （實際與隨機一樣）。

五、判定（P1，決定性，事前寫死）
- 主欄：MAIN 期間、13 個 j 的「實際−隨機」t 的中位數。版本 (a)、(b) 取較低者 t_low。
- t_low < 2.0 → 刪除「大型股選股力」機制，0050 骨架線結案；o7 FINDINGS 與報告第 13 節「站得住的反證」改寫；live_tracking 的 capmatched F1、C50、C70 紙上線停止（已記錄資料保留）。
- t_low ≥ 2.0 → 只列待觀察，以前瞻為準；不改 SOP、不重啟 0050 骨架研究。
- 研究員事前預測：t_low < 2.0。

六、其他預測（描述，不參與判定）
- P2：先用 o7 原輸入、官方日、2019-01～2026-10-12、50 次隨機（seed 5000+，review_stat_o7.py G 段）重現 t 4.02，|差| ≤ 0.10 才往下做；沒過先追差異、不改判定規則。
  之後依序替換 (i)、(ii)、(iii)、(iv)（官方日 → j 中位數），每步報 j 中位數 t、p10、官方日 t。預測 (i)+(ii) 降幅最大，(iv) 再降約 1.5～2.4。
- P3（決定版本）：去掉最大 1 檔（靜態歸因：Σ (w_實際 − w̄_隨機)×(窗因子−1)，13 個 j 平均最大者；實際與 1,000 次隨機都從合格集合剔除後重跑）、
  最好 1 個月（每個 j 刪掉 x 最大的窗）、最好 1 年（每個 j 刪掉 x 合計最大的日曆年）後的 j 中位數 t。預測 4763、2308、2021 航運、2025 年占比高。
- P4：靜態歸因依訊號日市值名次分 1～100、101～300、其他；若還有選股力應集中在 1～100；101～300 > 0 只記錄。
- P5：PRE 期間 2014-01～2019-04，版本 (a)、(b)；方向（j 中位數平均超額的正負）應與 MAIN 一致；相反就判機制不穩。
- 和主策略比較：報告 PIT 版在 0050 骨架上的選股力（對隨機），對照主策略在前 300 池內對同池等權的選股力（main_v2，t 3.86，引用不重算）。

七、G1～G8
- 本題是機制結案，不產生可投資策略。G1：K5T 可執行版（s6_exec.py，100 萬／300 萬／1,000 萬 GROW 與 FIX，另加每邊 0.5% 壓力），對 0050，j 中位數與 p10。
- G2：不適用，改報對隨機的 t；同時報對 0050 的 t。G5：本題核心。G6：獨立複製（replicate/，複製者只讀本規格，從原始股本來源自己重建 PIT 股數，不用 shares_pit）。
- 其他：不適用（機制檢定）。

八、檢定總數（揭露）
- 決定性檢定 1 個（兩版本取小，等於對兩版本做交集檢定，不需多重校正）。
- 描述性：5 個輸入設定 × 2 期間 × 14 個日期集合 = 140 個 t；P3 3 個、P4 3 層、G1 12 格；全部報告，不做挑選。
"""
CODE = ['pcore.py', 'pcore_lite.py', 's1_gate.py', 's2_explore.py', 's3_run.py', 's4_stats.py', 's6_exec.py', 's0_prereg.py']
DATA_FILES = {
    'o7_cmlib.py': O7 / 'cmlib.py', 'o7_cmstrat.py': O7 / 'cmstrat.py', 'o7_shares_est.parquet': O7 / 'data' / 'shares_est.parquet',
    'o7_membership_pre2015.json': O7 / 'data' / 'membership_pre2015.json', 'o7_float_phi.parquet': O7 / 'data' / 'float_phi.parquet',
    'o7_review_stat_o7.py': O7 / 'review_stat_o7.py',
    'o8_shares_pit.parquet': O8 / 'data' / 'shares_pit.parquet', 'o8_membership_pit.json': O8 / 'data' / 'membership_pit.json',
    'o8_replicate_rcore.py': O8 / 'replicate_o8' / 'rcore.py',
    'delist_prices.parquet': AG / 'o_infra_v3_rebuild' / 'delist_prices.parquet',
    'listed_twse.parquet': AG / 'o_v7_listing_exec_engine' / 'data' / 'listed_twse.parquet',
    's4_jscan7.py': AG / 'o_v7_listing_exec_engine' / 's4_jscan7.py', 'st_core.py': AG / 'o_stagger_tranche' / 'st_core.py',
    'model_spec7.json': AG / 'o_v7_listing_exec_engine' / 'data' / 'model_spec7.json',
}
P = dict(title='o_capmatched_pit_close PREREG', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
         spec=SPEC, n_random=1000, seed0=10000, decision_threshold=2.0, frozen_version='F1_K5_N20_sqrt',
         code_sha256={f: sha(HERE / f) for f in CODE}, input_sha256={k: sha(v) for k, v in DATA_FILES.items()},
         explore_sha256=sha(HERE / 'explore.json'),
         note='explore.json（股數與名次描述，無報酬）在 PREREG 前完成。快速引擎在 PREREG 前只用 o7 原輸入、2019-05～2020-07 的隨機名單核對過與 simulate_w 一致（最大相對差 4e-16），沒有看任何 PIT 版本的報酬。')
F.write_text(json.dumps(P, ensure_ascii=False, indent=1), encoding='utf-8')
h = sha(F)
R = {}
R['prereg'] = dict(file='data/PREREG.json', sha256=h, created_utc=P['created_utc'])
(HERE / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
print('PREREG', h, P['created_utc'])
