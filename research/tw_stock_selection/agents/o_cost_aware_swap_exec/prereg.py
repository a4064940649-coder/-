#!/usr/bin/env python3
"""PREREG（事前登記）：寫入 results.json['PREREG'] 與 PREREG_sha256。寫入前只讀已公布數字；未跑任何變體或安慰劑報酬
（此前只跑了 gen_cands.py 的名單重建與 selftest.py 的「全開關關閉＝CSim」逐位檢查，皆為主策略本身）。
python3 -I -B prereg.py"""
import sys, json
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import cas_lib as CL; sys.path.pop(0)
RES = HERE / 'results.json'
assert not RES.exists() or 'PREREG' not in json.loads(RES.read_text(encoding='utf-8')), 'PREREG 已存在，不得覆寫'
P = dict(
 topic='o_cost_aware_swap_exec：成本感知換股與執行層減單（不延長持有）；S1 已由使用者 2026-11-12 11:07 針對本題解除；以 G1～G8 為目標',
 written_utc=pd.Timestamp.now('UTC').isoformat(),
 read_before=['BRIEF.md', '台股個股選股規則實證研究報告.md §0、11～13（只讀已公布數字）', '選股SOP.md', 'BACKLOG.md', 'main_v2/FINDINGS.md',
              'original_o20/DIRECTOR_REPORT.md §3、5～7', 'o_turnover_fixedcost_diagnosis/FINDINGS.md', 'o_hold_buffer_fresh/FINDINGS.md', 'turnover_buffer/FINDINGS.md',
              'integrate_o20/FINDINGS.md', 'original_o1/DIRECTOR_REPORT.md §5.3'],
 runs_before_prereg=['gen_cands.py：官方名單候選序列重建（斷言 63,820×3 列 picks 與 o20 plans_real_allseeds 逐列相同）',
                     'selftest.py：CASim 全部開關關閉時對 CSim 淨值逐位相同（主策略本身，非變體）'],
 published_numbers=dict(main_v7x_100n_normal_y19on_cagr=23.27, e50_y19on=29.49, full_exc_worst=0.86, full_t_worst=0.45, jp10_worst=-2.25,
                        n300=-1.00, n1000=-3.32, stress=-6.28, paper_upper_calendar=21.95, paper_upper_ia=22.06,
                        turnover_oneway=7.89, edge_per_swap=0.36, rtc_per_swap=1.17, shapley_cost=13.67, n_family=2573),
 feasibility_statement=('即使成本全部歸零（加回 13.67pp），2019 起超額也只有約 +7.45pp；以現行 2019 起標準誤約 17pp 推算 t≈0.43，遠低於 G2 的 2.0；'
                        '全期由 t 0.45 升到 3.0 約需 +4.9pp／年；o_g2_feasibility_stoprule 已觸發。事前預期 G2、G3（DSR）幾乎確定不過，'
                        '最可能結論是「執行層改良（或否）＋不可投資」。仍照 G1～G8 完整報告。'),
 data=dict(engine='o_engine_delist_consideration/cz_v7x.CSim（consideration × cons_main）× K5T × GROW；R4＝orig_cfg＋f1,f2,f3=0.5,f4',
           data_layer='o_exec_engine_rewrite_v7x/data/x_cache.npz', plans='v7x_lib.make 邏輯；ALLPIT_REAL_{opt,p90,carry} × 13 j × 20 seeds；data/cands_allseeds.csv.gz（官方序列前 40 名＋開盤漲停旗標）',
           engine_sha256=CL.check_engine(), e50='x_cache.npz E50（0050 含息）',
           regulation='raw/local_copies/：model_spec7.json（V7X 成本模型：跳動點表、K5T、零股價差、上限）、cpi_tw.json、regulation_sources（盤中零股 2020-10-26 等，搜尋摘要）；'
                      '證交所網站經 proxy 403，未另抓原文；手續費 0.1425%（2015 前 6 折、之後 2.8 折）、最低 20 元、證交稅 0.3% 沿用 V7X 引擎內建參數；'
                      'ETF 交易稅 0.1%（0050，只用於 DC5E 殘餘資金）為法定稅率，未抓原文，列為假設。'),
 split=dict(selection=['2006-04（各 j 首個訊號日）', '2014-12-31'], validation=['2015-01-01', '2020-12-31'], holdout=['2021-01-01', '各 j 終點（jscan_end_days）'],
            honest_note='2006～2026 全期都在主策略研發時看過；真正的樣本外只有 G8 前瞻。',
            selection_runs='挑選期模擬一律以 run(t1＝2014-12-31 前最後交易日) 截斷，不產生 2015 後淨值。'),
 c_hat=('ĉ＝事前預估來回成本率，只用訊號日 d 收盤前資料（x_cache 的 C[d]、V60p[e]＝V60[d]、SIGp[e]＝σ[d]、VRp[e]＝名次[d]、PC[e]＝C[d]）：'
        '部位 P＝d 收盤淨值×(1−2%)／10；p＝d 收盤價；L＝floor(P/1000p)、q＝floor((P−1000Lp)/p)；每邊整股率＝max(BASE＋κ·σ·√(1000Lp／(OA 比例×V60)), tick(p)/p)；'
        '零股價差＝OCB1／OCS1（rank_right）；手續費＝max(20, 費率×名目)，整股與零股各算；賣出加證交稅 0.3%；一律一般成本口徑（壓力格、成本全關格的決策規則相同）。'
        '不使用題 2（o_cost_model_sensitivity_recon）任何結果。'),
 variants=dict(
   CA80='新進名額（不在前月實際名單者）依官方排名逐一檢查 ĉ；ĉ>τ 則改選官方序列（已套開盤漲停跳過）第 11～20 名中不在前月名單、ĉ≤τ 的下一名；找不到保留原候選。'
        'τ＝過去 36 個月全部訊號日前 10 名候選的 ĉ（各自用當時資料、以目前部位規模 P 重算）之第 80 百分位；歷史 < 12 個換股日時不啟動（各 j 約 2007-04 起）。換出、強制出場照常；不續抱舊股。',
   CA90='同 CA80，τ 取第 90 百分位。',
   MT='F*＝20 元 ÷ 當時手續費率（2015 前 23,392 元、之後 50,125 元；壓力格依壓力費率 23,392 元）。續抱股再平衡單名目 < F* 不下；有 ≥1 張時零股尾數名目 < F* 不下（買進與續抱調整皆然）；換出全部賣出；剩餘資金留現金。',
   NR='不做等權再平衡：續抱股股數不動；新進股目標＝(可用資金×(1−2%)−續抱股開盤市值)／新進檔數。',
   DC5C='買進順延單在換股後第 6 個交易日起取消（換股日＋5 個交易日內可成交），資金留現金到下次換股；賣出順延照常。',
   DC5E='同 DC5C，但取消的資金（≤ 現金）當日收盤買 0050（E50 含息計值），下次換股日以前一日收盤賣出；0050 成本＝手續費（最低 20 元）＋每邊價差 0.05%＋賣出 ETF 稅 0.1%。這只是殘餘資金處理，不是 0050 固定比例混合。',
   CB1='CA（CA80、CA90 中挑選期主指標較高者）＋MT＋NR。',
   CB2='CB1＋DC（DC5C、DC5E 中挑選期主指標較高者）。',
   A20_reference='參照臂：前月名單中官方序列前 20 名者續抱，其餘依官方序列補滿（o_hold_buffer_fresh 已依事前規則永久關閉；只報告、不參與選擇、不重複計入 N）。'),
 family_N=dict(before=2573, added=8, after=2581, note='模型迭代若改變變體定義（不只修錯），舊版照樣計入 N。'),
 cells=dict(selection=['100萬 一般（主）', '100萬 成本全關（tax/fee/tick/imp 關；成交時點與上限不變）', '100萬 壓力', '30萬 一般（挑選期全在 2020-10 前：盤後零股＋整股）'],
            seeds='全部格 20 seeds × 13 j × 3 放回版本；20 seeds 平均淨值＝x_stats.avg_nav 同式', ),
 metric=dict(main='挑選期主指標＝日曆年化（(b−a).days/365.25）配對差（變體 − 主策略，同 j、同 seed 集合的 20 seeds 平均淨值），13 j 中位數，opt／p90／carry 取最小值（不利端）。',
             also='ia 口徑（交易日 252 年化）並列；三層比對（j 中位、逐 j 平均、逐 j 最大）並報中位 j 與最差變體是否換人。',
             monthly='月配對差＝各 j 月底淨值月報酬差的 13 j 平均（每個放回版本一條）；t 與月份區塊 bootstrap（每月為一區塊，2,000 次，seed 20261112）90／95% CI。'),
 selection_rule=('候選須同時滿足：(a) 主指標 ≥ +0.5pp／年且 ≥ MDE 的一半；(b) 成本前配對差 ≥ −0.3pp／年；(c) 壓力成本配對差 > 0；(d) 30 萬配對差不翻號（> 0）。'
                 '符合者取主指標最高；差距 ≤ 0.2pp 時取組件較少者（單一臂 1、CB1 3、CB2 4）。沒有任何變體符合 → 判「不可用」，驗證期與保留期仍以 CA80 以外的事前固定代表（主指標最高者）各跑一次，只作描述紀錄。'),
 predictions=dict(
   P1='CA80／CA90 挑選期成本前配對差在 ±0.5pp／年內；100 萬一般扣成本後 ≥ +0.5pp／年。成本前 < −0.5pp → 判「右尾集中在高成本候選」，機制否定。',
   P2='主策略 2006～2014 持股槽窗報酬最好 5% 的 ĉ 分布與其他槽無差異（KS p > 0.10，平均 ĉ 差 < 0.15%）；右尾 ĉ 顯著偏高則與 P1 一起否定。',
   P3='CA 改善：股價 < 50 元、跳動點占價 > 0.5% 的子樣本，以及 300 萬、1,000 萬格大於 100 萬全樣本；30 萬格（2020-10 前）不翻號。',
   P4='MT：30 萬與 100 萬在 2020-10-26 後效果較大；1,000 萬接近 0；下單日占交易日比例由約 35% 降到 ≤ 25%。',
   P5='NR：省下的成本 ≤ 0.4pp／年；若低於 MDE 只能寫「無資訊」，不得寫「無效」。',
   P6='DC5C：成本前配對差 ≥ −0.3pp；DC5E 不比 DC5C 差超過 0.3pp。',
   P7='CB1／CB2 成本前配對差 ≥ −0.5pp；扣成本後 ≥ CA 單一臂。',
   weaker='2006～2012 MT 效果小；換股檔數 ≤ 2 的月份 CA 效果小。'),
 power=('先做（寫進 PREREG_POWER，於任何變體報酬之前）：挑選期、100 萬一般、seed 0、opt、13 j；安慰劑 PR20（新進名額 20% 隨機改成下一名新候選）與 PS（再平衡單與零股尾數以 MT 事前可得的比例隨機不下）各 200 次；'
        'σ_d＝13 j 平均月配對差的月標準差；n＝月數；同時報「j 合併、每事件約重複 4 次」的有效樣本數口徑；80% 檢定力、雙尾 5%：MDE＝2.80×σ_d／√n×12（pp／年），三段各報；'
        '另報 200 次安慰劑的主指標分布（j 中位年化配對差）之標準差×2.80。MDE 大於預期效果時，結果只能是「改善」或「無資訊」。'),
 oos=dict(main_table='凍結後一次：V7X 可執行、100 萬一般、13 j 中位數與 j p10、opt／p90／carry 逐格取最差，2019 起與 0050（29.49%）並列；',
          also=['30 萬（2020-10 後盤中零股；之前另列）、300 萬、1,000 萬（GROW）、壓力成本', '三段 2006～2012／2013～2018／2019～2026 超額', '拿掉最好 3 年的 t', '拿掉貢獻最大一窗',
                '參數鄰域（τ 75／85／95、F*×0.5／×2、DC 3／10 日；只報不選）', '配對差 t、bootstrap 90／95% CI、三層比對、成本前與成本後並列', '會計成本拖累（稅、手續費、跳動點、衝擊、零股）、周轉、下單日占比、成本兩平滑價',
                '紙上上緣（不另跑 ia 引擎；註明回測上緣實現不了，沿用正式 21.95%）'],
          improvement_rule='驗證期與保留期配對差都 > 0，且全期配對 t ≥ 2.0、bootstrap 95% CI 不含 0，才能寫「相對主策略改良成立」。'),
 gates=('G1 j p10 > 0 且壓力 > 0；G2 全期 t ≥ 3.0 且 2019 起 t ≥ 2.0；G3 2,581 條 Hansen SPA p < 0.05、DSR ≥ 0.95（報 Li-Ji 有效 N）；G4 三段 > 0、拿掉最好 3 年 t ≥ 2.0、鄰域不翻號；'
        'G5 PIT；G6 由工作流層分派不同子代理重寫遞補邏輯與包裝器，關鍵數字差 ≤ 0.1pp／t 0.05（研究員不得兼任）；G7 100 萬、300 萬、1,000 萬都成立；G8 前瞻 12 個月。任何一項不過→不可用，SOP 與正式數字不改。'),
 lessons=['不利端逐格取最差（opt／p90／carry × 一般／壓力）', '凍結前不算正式統計量（對 0050 t、G 表只在凍結後）', '事後新增狀態與嚴格版並列，嚴格版為主欄', 'raw/ 只放未修改原始檔',
          '假設區間與抽樣區間並報', '取代正式數字要附歸因（本題不取代）', '紙上附成本兩平點', '可執行測到 30 萬', '比對報三層（j 中位、逐 j 平均、逐 j 最大）並報中位 j 與最差變體是否換人',
          '偏態分布用平均數口徑（中位數並列）', '重疊樣本報有效樣本數', '年化口徑：日曆年化為主（ia 並列）', '任何 ≤ 0.2pp 改善要報拿掉最大一窗後的值', '事後讀法不得寫成建議',
          '關鍵中間檔（名單、逐 j 淨值、成交統計）壓縮存進 data/，不只放 scratchpad', 'o1 §5.3：事件 t 按事件日分群、驗證期污染、格點邊緣贏家詛咒、成交可行性事前寫入、t 提升拆均值與波動、存活者偏誤'],
 iteration_policy='挑選期內允許修正 ĉ／包裝器實作錯誤（寫進 iterations.json，每版一筆）；不得新增第 9 個變體、不得用 2015 後資料；變體定義若改變，舊版計入 N。',
 readonly='live_tracking/main_forward、o_forward_infra_live、o_g8p2a_protocol_addendum 只讀（ro_snapshot_start/end 比 sha256）；不呼叫 G8P-2a 判定器。',
)
r = json.loads(RES.read_text(encoding='utf-8')) if RES.exists() else {}
r['PREREG'] = P; r['PREREG_sha256'] = CL.canon(P)
RES.write_text(CL.dumps(r), encoding='utf-8')
print(r['PREREG_sha256'], P['written_utc'])
