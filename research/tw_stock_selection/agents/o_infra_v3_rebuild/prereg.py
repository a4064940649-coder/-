#!/usr/bin/env python3
"""事前登記：python3 -I -B prereg.py → results.json 的 PREREG（含 sha256、時間）。已存在就拒絕覆寫。
在任何策略淨值計算之前執行；run.py 斷言 PREREG 存在且 data_freeze.json 的 sha256 相符。
另外計算（只用 2021-09-24～2023 的真實處置名單與合格池，不看報酬）all_plus 需要的「每個訊號日漏網名額」。"""
import json, hashlib, datetime, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
RES = HERE / 'results.json'
if RES.exists() and json.loads(RES.read_text(encoding='utf-8')).get('PREREG'):
    raise SystemExit('PREREG 已存在，不覆寫')
import o3_core as O
V, K = O.V, O.K
fz = O.freeze()
for f in fz['files']: O.check(f)
hm = json.loads((HERE / 'disposal_hist_meta.json').read_text(encoding='utf-8'))
O.apply_patch()
YL, Y5, _ = V.load_revenue()
# 漏網名額：挑選期每個訊號日，合格池（不排除處置）內「真實處置中」的檔數 × (1 − 召回率)
O.set_disposal('none')
S = V.V2('gapfix', YL, Y5, exfin=True)
real_in_pool = []
for d in S.sig_days:
    if not (O.D5_START <= d <= pd.Timestamp('2023-12-31')): continue
    cs = S.cross_section(d)
    real_in_pool.append(int((O._mask_from(O.DSP5, d) & cs['el']).sum()))
rec = [x for x in hm['iterations'] if x['version'] == hm['chosen']][0]['select_2021-09-24_2023']['recall']
miss_per_sig = float(np.mean(real_in_pool) * (1 - rec))
extra = {}; acc = 0.0
for d in S.sig_days:
    if d >= O.D5_START: break
    acc += miss_per_sig; k = int(np.floor(acc + 1e-9)); acc -= k
    extra[str(d.date())] = k

PREREG = f"""【事前登記：o_infra_v3_rebuild（回測基礎設施三缺口修補，重算主策略正式數字 V3）】
時間：見 PREREG_time_utc。資料表在此之前凍結（data_freeze.json，{fz['timestamp']}）：
  delist_prices.parquet {fz['files']['delist_prices.parquet']['sha256']}
  merger_consideration.csv {fz['files']['merger_consideration.csv']['sha256']}
  disposal_hist.csv {fz['files']['disposal_hist.csv']['sha256']}
  delist_official.csv {fz['files']['delist_official.csv']['sha256']}
在此之前沒有計算任何策略淨值或報酬統計（探索 explore.json 只有個股層描述、名單與池內計數）。

一、基準（V2）＝main_v2 正式版：gapfix 窗、正式次排序（分數→營收 3 月年增率→代碼）、BASE 營收、跳動點漲停遞補、
  一般成本（買 0.0855%+0.1% 滑價、賣再加稅 0.3%）與壓力成本（買賣各 0.5%）、評價到 2026-10-12、期間 S1／S2／S3／S4／P1／ALL。
  引擎＝ia_core.simulate（freeze=True）的等價重寫 o3_core.simulate（xrules=None 時逐日相同，run.py 內斷言）。
二、修正開關（全部報告，不選版）
  R：營收底稿 tw_revenue → o_rev_survivorship 凍結的 R_DEL（下市公司 2015-01～2019-08 營收）。價格修補的股票要有營收才可能入選，所以 V3 含 R。
  X：持有中永久停止交易的部位照 merger_consideration.csv 出清（現金：×現金對價/最後收盤，付款日未知＝終止日＋14 天之後第一個換股日轉回現金；
     換股、控股公司轉換、型態未知：×1（最後收盤），終止日之後第一個換股日；經營惡化：歸零）。代碼重用（2301、2432）舊證券在終止日前最後一筆下市。
  P：delist_prices.parquet 寫入面板（32 檔 2019-05-17～下市，15,375 個股票日）。
  D：處置濾網全期一致。2021-09-24 以前用近似名單 {hm['chosen']}（挑選期精確率 {rec:.3f} 召回率見 data_freeze）。
     D_lo＝只排除近似事件；D_hi＝近似事件＋依召回率推估的漏網名額（每個訊號日平均 {miss_per_sig:.3f} 檔＝挑選期池內真實處置檔數平均 {np.mean(real_in_pool):.3f}×(1−召回率)，
     以累積進位取整，排除同訊號日合格池中 52 週比最高者）。D_none＝全期一律不排除（V3−D，只供第 2 題參考，本題不判定）。
  全因子：R∈{{0,1}} × X∈{{0,1}} × P∈{{0,1}} × D∈{{v2, lo, hi, none}}，全市場／排除金融 × n10／n30 × 一般／壓力成本。等權基準（同池等權）在每個版本用同一個合格池與同一個 X 規則重算。
三、V3 定義（看到 V3 數字以前固定）：V3＝R＋X＋P＋D_worse；D_worse＝D_lo、D_hi 中「排除金融 n10 一般成本 ALL 年化」較低者（處置覆蓋率＝召回率 < 90%，依門檻用較差端）；
  其他 univ／n／成本／期間都跟著同一個 D 端。X 的覆蓋率以 V3 各版本實際凍結事件計；若 < 90%，正式數字改用 X_lower（最後收盤 −5%、延到終止日＋60 天）。
  P 的股票日覆蓋率 91.2% ≥ 90%，正式數字用資料本身。
四、界限（以 V3 為底，全部報告；不進正式數字，但寫進 SOP 加註）
  X_upper（全部以最後收盤、終止日後出清，經營惡化也以最後收盤）、X_lower（−5%、延 60 天；經營惡化歸零）。
  缺營收界限（終止上市名單內 4 碼、國內股、訊號日 ≤ 終止日，營收以外條件合格但缺營收）：B05（營收百分位 0.5）、B10（1.0），都含 X。
  期望拖累（task 定義的下界）：每月 Δ＝Σ_s(該股持有期報酬 − 同池等權)/300，s＝上述缺營收的合格股（P 補到價格者）；把 Δ 加到 V3 每月報酬。
  最壞情境：每年挑 1 檔「最可能入選」者（B10 分數最高的訊號日×股票），該月以 1/n 權重取代平均持股，報酬＝實際持有期報酬（含下市 X 規則：經營惡化歸零、現金對價）。
  價格補不到的 1,483 個股票日（6251、6289、9188、3682 等）：上界＝現況；下界＝同上期望拖累法，報酬未知時經營惡化 −50%／併購 0%。
五、統計：年化、最大回撤（日頻）；t 對 0050、t 對等權（訊號窗，月份分群，扣成本）；日曆月 t；Calmar；逐年；最大單股貢獻與剔除；周轉；凍結事件數與最大凍結占比；
  t 變動拆成月均超額（均值）與超額標準差（波動）。主判定版本：排除金融 n10、一般成本、ALL（2006-04～2026-10-12）訊號窗 t 對 0050。
六、判定門檻：V3 t ≥ 2.3 → 第 1 級（維持跨循環顯著）；2.0～2.3 → 第 2 級（SOP 加註）；< 2.0 → 第 3 級（主策略降為邊緣，實盤預期依界限下修）。
  V3 取代 v2 必須由獨立複製逐日重現淨值（誤差 < 0.1pp），本題只交出淨值檔與規格，複製另案。
七、事前預測（寫在任務書，原樣登記）：P1 X：排除金融 n10 ALL +0.1～+0.8pp，集中 S2；凍結事件 125→0；若下降 > 0.3pp 先查資料。
  P2 P：S3 −1.0～0pp，等權降更多（對等權 t 降幅 > 對 0050 t 降幅）。P3 D：S1＋S2（P1 段）−3～+1pp，點估計為負。P4 V3 ALL 25～28.5%、t 2.2～2.7。
  子樣本：P1 在 2014～2018 最強；P2 在 2020～2022 與中小型股；P3 在 2015-06 以後。
八、SOP 預期表：ALL、P1（2006～2020）、S3（2021～2026）、S4（2024～2026），預期年化＝0050＋0.5×(V3−0050)，一般與壓力成本。
九、多重檢定揭露：處置近似規則 7 版（iterations.json）；本題 32 組全因子版本全部報告、不選版。
十、期間已被看過（data_freeze.json periods_seen_by）：全部期間都被看過，沒有盲測期；本題的「樣本外」只是同一份規格跑全期。
"""
h = hashlib.sha256(PREREG.encode('utf-8')).hexdigest()
res = {'PREREG': PREREG, 'PREREG_sha256': h, 'PREREG_time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
       'PREREG_params': {'D_hi_extra': extra, 'miss_per_sig': miss_per_sig, 'real_in_pool_mean_select': float(np.mean(real_in_pool)), 'recall': rec,
                         'iterations_json_sha256': hashlib.sha256((HERE / 'iterations.json').read_bytes()).hexdigest(),
                         'explore_json_sha256': hashlib.sha256((HERE / 'explore.json').read_bytes()).hexdigest(),
                         'o3_core_sha256': hashlib.sha256((HERE / 'o3_core.py').read_bytes()).hexdigest()}}
RES.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print(h, res['PREREG_time_utc'], 'miss/sig', round(miss_per_sig, 3), 'extra total', sum(extra.values()), 'of', len(extra))
