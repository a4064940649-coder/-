# S69 前瞻驗證計畫：事前寫 SPEC.json／PREREG.json 與 sha256（在任何模擬、界線、檢定力計算之前執行一次）
# 用法：python3 -I -B prereg_write.py
import json, hashlib, datetime
from pathlib import Path
H = Path(__file__).resolve().parent
assert not (H / 'SPEC.json').exists() and not (H / 'results.json').exists(), '已登記，不得重寫'
now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

SPEC = {
 "id": "S69",
 "title": "前瞻驗證計畫：2026-11-12 起實盤紀錄的檢定力、序貫判定界線、貝氏更新與白話判讀表",
 "written": now + "，在任何檢定力、界線、模擬計算之前",
 "nature": "方法／治理題；不改策略、不挑規則、不增加資料窺探族群 N。",
 "dedupe": {
  "already_done": [
   "S27 s27_expectation_ranges：正常範圍與失效警戒線 A（滾動 36 個月年化超額 < −8 個百分點）、E（策略÷0050 對前高 < −24%）、B、C、D，黃／紅燈；誤報率是相對於「策略照樣本內 +11.7 個百分點有效（EFF）」算的。本題不重做、不改。",
   "S61 s61_health_light：健康度燈號（勝率／單位部位超額／選出檔數），5 年誤報 68.5%，只當資料檢查提示。本題不重做。",
   "o_forward_infra_live（o9，只讀）：前瞻產生器 v3，訊號日當晚凍結名單、時戳早於開盤、V7X K5T 成本；是紀錄基礎建設，不含統計判定。附帶 G8 檢定力一行（σ 6.14 個百分點／月、12 個月加總 > 0 的機率）。",
   "o_tier_shrink_sequential（o7）、o_forward_sequential_v2（o8）：階層收縮先驗＋升級／下修序貫規則，月頻、24～36 個月、決策線為 o 系列主策略 J 窗中位數（V7X 可執行）；o8 結論「不可用」（D2 下誤升級 6.2～6.8%）。",
   "o_g8_protocol_fin_erratum／o_g8p2_protocol_refreeze／o_g8p2a_protocol_addendum（G8P-1／2／2a，不准動）：o 系列主策略（52 週高點＋營收、J0..J12 中位數、V7X K5T GROW 100 萬）前 12／24 個有效錨定月的升級判定協定，含作廢規則與雜湊鏈。",
   "o_g2_feasibility_stoprule（o11）：G1～G7 門檻可達性（真實 IR 0.5 全過機率 0%）與研究中心停損規則。",
   "live_tracking/LIVE_TEST_PLAN.md：12 個月紙上追蹤的粗略判定（落後 > 10 個百分點降到 0%），未經統計設計。",
   "S52／S55／S60：資料窺探校正、樣本外、因子歸因（本題只引用其數字當先驗與追蹤誤差）。"
  ],
  "new_in_s69": [
   "對象是網站正式產品 E0|W10_70 的實盤逐日淨值（xq_scripts/site/data/live.json，2026-11-12 開盤＝100 萬）對 0050，不是 o 系列 J 窗決策線",
   "期限 10 年、每年檢視一次（共 10 次）的教科書 group sequential 設計（O'Brien-Fleming 型 alpha spending），整體型一誤差 5%",
   "真實年化超額 0／3／5／8／12 個百分點 × 1／2／3／5／7／10 年的檢定力表與「運氣」機率",
   "常態共軛貝氏後驗表（描述）",
   "給產品擁有者的白話判讀表與 forward_check.py"
  ]
 },
 "known_before_writing": "S52 results.json：正式淨值 2015-07～2026-10 月超額 136 個月，月平均 0.9737%、月標準差 4.6794%（年化追蹤誤差 16.21 個百分點）、偏態 1.48、峰度 8.12、一階自相關 −0.117、資訊比率 0.72。S55：樣本外主窗追蹤誤差 5.63 個百分點（平均只選 3.3 檔、74% 資金閒置在 0050）。由此心算：10 年標準誤約 16.2/√10＝5.1 個百分點，真實 +5 時 10 年單尾 5% 檢定力約 25～30%；這是寫規格時已知的粗估，界線與規則不依模擬結果調整（唯一例外見 size_contingency）。",
 "data": {
  "official_nav": "/home/user/-/xq_scripts/site/navs_official.csv.gz（只讀）欄 E0|W10_70 與 0050；月底＝每月最後一個交易日，2015-07 以 2015-07-01 為基期，2026-10 為部分月（至 10-22）→ 136 個月；月超額 x＝策略月報酬 − 0050 月報酬（簡單報酬相減，同 S52）。驗收：月平均、月標準差與 S52 results.json 相差 < 1e-9。",
  "oos": "agents/s55_pre2015_oos/oos_monthly.csv 主窗 2008-01～2013-01（61 個月）的 excess 欄。",
  "live": "/home/user/-/xq_scripts/site/data/live.json（只讀；rows：日期、策略、0050）。"
 },
 "definitions": {
  "annualized_excess": "累積年化超額 m＝12 × 月超額算術平均（個百分點）；真實年化超額 δ＝12 × 月超額期望值。",
  "t": "t＝月超額平均 ÷（月超額樣本標準差(ddof=1) ÷ √n），n＝月數。",
  "forward_months": "前瞻月＝日曆月；第 1 個月＝2026-11-12 開盤（淨值 100 萬）到 2026-11 最後一個交易日（部分月）；之後每個日曆月以月底最後交易日淨值計。",
  "looks": "第 k 次正式檢視（k＝1..10）＝(2026+k) 年 10 月最後一個交易日收盤後，用 2026-11～(2026+k)-10 共 12k 個月。資訊比例 t_k＝k/10。"
 },
 "part1_power": {
  "dgp": "x*_t＝e_b(t)＋δ/12；e＝歷史月超額減其平均（去均值殘差）；循環區塊 bootstrap，固定區塊長 L。",
  "variants": {
   "D1_main": "正式淨值 136 個月殘差，L＝6（主）",
   "D1_L1／D1_L12": "同 D1，L＝1（獨立同分布）／12",
   "D2_te56_scaled": "D1 殘差等比例縮到年化追蹤誤差 5.63 個百分點（形狀不變），L＝6",
   "D3_s55_resid": "S55 主窗 61 個月殘差直接抽，L＝6",
   "D4_since2019": "正式淨值 2019-01 起 94 個月殘差，L＝6（追蹤誤差較大的不利情境）",
   "D5_normal": "常態獨立、追蹤誤差同 D1（解析式對照）"
  },
  "grid": {"delta_pp": [0, 3, 5, 8, 12], "years": [1, 2, 3, 5, 7, 10]},
  "outputs": ["P(單尾 5% 顯著)：t ≥ t 分布(n−1) 的 95% 分位數（單次、固定期限檢定）", "P(累積年化超額 ≥ +5 個百分點)（δ＝0 時＝運氣）", "P(累積年化超額 < 0)", "固定期限達 80% 檢定力所需年數（常態近似 ((1.645+0.842)·TE/δ)²）"],
  "n_sim": 200000, "seed": 20261112
 },
 "part2_sequential": {
  "references": ["O'Brien & Fleming (1979) Biometrics 35(3):549-556", "Lan & DeMets (1983) Biometrika 70(3):659-663", "Kim & DeMets (1987) Biometrika 74(1):149-154（power-family spending）", "Armitage, McPherson & Rowe (1969) JRSS-A 132(2):235-244（遞迴數值積分）", "Jennison & Turnbull (2000) Group Sequential Methods, Chapman & Hall（t 統計量用名目顯著水準換算；non-binding futility）", "Pampallona & Tsiatis (1994) JSPI 42:19-35（同時可提早接受或拒絕）"],
  "efficacy": "H0：δ ≤ 0 對 H1：δ > 0，單尾整體 α＝5%，K＝10、等距資訊。Lan-DeMets O'Brien-Fleming 型 spending：α(t)＝2−2Φ(z_{0.975}/√t)。以遞迴數值積分求每次檢視的 z 界線 u_k；再以相同名目單尾 p 值換成 t 分布(12k−1) 的界線 ut_k。判定有效：t_k ≥ ut_k。另列原始 O'Brien-Fleming（Z_k ≥ c√(10/k)）作對照，不採用。",
  "futility_formal": "「正式判定無效」＝顯著低於『值得做的最低超額』δ1＝+3 個百分點／年：檢定 H：δ ≥ 3，統計量 t3_k＝(月超額平均 − 3/12 個百分點) ÷ 標準誤；單尾整體 γ＝10%，Kim-DeMets 線性 spending γ(t)＝0.10·t；界線 −f_k（同樣換成 t 分布）。判定無效：t3_k ≤ −ft_k。理由：誤停用的代價只是之後拿 0050 報酬，比誤判有效便宜，所以容許較大的誤差（10%）與較平均的 spending（早期也有機會停）。non-binding：有效界線計算時不考慮無效界線，所以不論是否照無效界線停用，型一誤差都 ≤ 5%。",
  "pragmatic_stop": "「務實停用線」（非統計顯著，預設動作）：第 5 次檢視起（k ≥ 5），累積年化超額 m_k < 0 → 建議停用改 0050。理由（寫規格時的推導）：先驗 N(2, 5²)、追蹤誤差 16.2 時，後驗平均低於『值得花工夫的 +1 個百分點』的門檻約為 m < 1 − 10.5/k（第 5 年 −1.1、第 10 年 −0.05），取整為『滿 5 年後 < 0』。k < 5 時 m_k < 0 只降低期望、不停用。停用後紙上紀錄照常做到第 10 年，之後若 t_k ≥ ut_k 仍可判定有效並恢復。",
  "terminal": "第 10 次檢視仍未判定：m_10 ≥ +3 → 「未證實、點估計尚可」，由擁有者決定、降低期望；m_10 < +3 → 停用改 0050。",
  "middle_zones": "未觸任何界線時的描述性分區：m_k ≥ +3「續做」；0 ≤ m_k < 3「續做、降低期望」；m_k < 0 且 k < 5「續做、降低期望，並看 S27 警戒線」。",
  "simulation": "D1_main（另 D1_L1、D1_L12、D2、D3、D4）下 δ ∈ {−5, 0, 3, 5, 8, 12}：各年「判定有效／正式無效／務實停用」首次發生的機率、10 年內未判定的機率、判定年份中位數；n_sim 200,000。型一誤差＝δ＝0 時 10 年內曾判定有效的機率（不論是否已停用，即 non-binding 口徑）；另報「照規則停用後不再判定有效」口徑的檢定力。",
  "size_contingency": "界線由理論決定，不依歷史資料調整。唯一例外（事前寫死）：D1_main 下若 δ＝0 的模擬型一誤差 > 5.5%，把全部 z 界線 u_k 乘同一個常數 c ≥ 1（二分法求最小 c 使模擬型一誤差 ≤ 5.0%）後再換成 t 界線；無效界線同理（δ＝3 時誤判無效 > 11% → 乘常數至 ≤ 10%）。有無觸發都要寫進 results.json。",
  "validation": "遞迴積分程式先重現文獻常數：原始 O'Brien-Fleming 雙尾 5%（以單尾 2.5% 近似）K＝5 的 c＝2.040、K＝10 的 c＝2.087（Jennison & Turnbull 表 2.3），容許差 0.002。"
 },
 "part3_bayes": {
  "model": "常態共軛：δ ~ N(μ0, τ0²)；觀察到 T 年累積年化超額 m ~ N(δ, TE²/T)；後驗常態。描述用，不是判定規則。",
  "prior_main": {"mu": 2.0, "sd": 5.0, "reason": "S52 BHY 打折後約 +0.5；S55 樣本外 +1.27（到 2015-06 為 −1.10）；S60 四因子＋營收 α +5.16（t 1.18）、情境 +1.0～+6.6；o11 收縮後真實資訊比率 0.07～0.29（× 追蹤誤差 16 ≈ +1～+5）。中心取 +2；標準差 5 使 95% 區間約 −8～+12，樣本內的 +11.7 落在約 +2 個標準差。"},
  "prior_sens": {"conservative": {"mu": 0.0, "sd": 4.0, "reason": "XQ 版扣因子後 α≈0、S52 DSR≈0、LIVE_TEST_PLAN『約等於 0050 到每年少 5』"}, "optimistic": {"mu": 5.0, "sd": 6.0, "reason": "S60 四因子＋營收 α +5.16、近親族群 SPA 顯著"}},
  "grid": {"years": [1, 3, 5, 10], "observed_m_pp": [-10, -5, 0, 3, 5, 8, 12, 15]},
  "te": {"main": "D1 追蹤誤差（約 16.2）", "sens": 5.63},
  "outputs": ["後驗 P(δ > 0)", "後驗 P(δ > 3)", "後驗平均"]
 },
 "part4_plain_table": "依 part2 的分區寫成「幾年後看到什麼 → 代表什麼 → 建議動作」；門檻以累積年化超額表示時用 D1 追蹤誤差換算並註明實際以 t 為準。",
 "part5_forward_check": "forward_check.py：讀 live.json（或 date,strategy,0050 的 csv）與本題 PLAN_FROZEN.json，輸出累積超額、年化、t、t3、所在分區、下次檢視日；不改 xq_scripts。以目前 live.json（4 個交易日）示範一次並註明無意義。",
 "governance": {
  "start": "計畫自 2026-11-12（實盤起始日）起算；本規格寫於 2026-11-18，當時實盤只有 4 個交易日、距第 1 次檢視（2027-10 底）約 11 個月，界線未參考任何前瞻資料。",
  "no_change": "PLAN_FROZEN.json 產出後，界線、檢視日、δ1、γ、務實停用線不得更改；要改必須另題事前登記，舊版檔案與 sha256 保留並在 BACKLOG 記錄，且新版只對登記日之後的檢視生效。",
  "no_extra_looks": "正式判定只在 10 個檢視日做；期中執行 forward_check.py 只供參考，不得據以判定有效（多看會吃掉 α）。遲做的檢視只能用到排定月底為止的資料。",
  "record": "淨值必須是含息總報酬口徑（策略與 0050 都要含現金股利），且每日紀錄不可回改；規則若經另題登記而變更，前瞻紀錄照實延續並註記變更日（本計畫檢定的是『研究中心實際交付的產品』）。",
  "relation": "S27 警戒線 A／E 照舊每月檢查，觸發＝提前檢討（不是本計畫的判定）；G8P-2a 協定管 o 系列主策略決策線，與本計畫互不取代。"
 },
 "resources": "只用本地資料、不連外網；python3 -I -B；峰值記憶體 ≤ 3GB；暫存 /tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s69/。",
 "limits": ["bootstrap 假設未來月超額的波動形狀像過去；平移平均而不改追蹤誤差（若未來選出檔數少，超額與追蹤誤差會一起變小，t 檢定自動適應，但以個百分點表示的門檻會變）", "一年一次檢視、10 年只有 10 次；產品可能中途改規則", "第 1 個前瞻月是部分月"]
}
PREREG = {
 "title": "S69 事前登記：前瞻驗證計畫的判定規則與預期",
 "written": now + "，在任何模擬與界線計算之前；搭配 SPEC.json",
 "registered_rules": {
  "efficacy": "t_k ≥ ut_k（Lan-DeMets O'Brien-Fleming 型，單尾整體 5%，10 次）→ 判定有效",
  "futility_formal": "t3_k ≤ −ft_k（對 δ ≥ +3 的單尾 10%，線性 spending，non-binding）→ 正式判定無效，停用改 0050",
  "pragmatic_stop": "k ≥ 5 且累積年化超額 < 0 → 建議停用改 0050（非統計顯著的預設動作）",
  "terminal": "第 10 年未判定：m ≥ +3 擁有者決定；m < +3 停用改 0050",
  "bayes": "只描述，不作判定"
 },
 "pre_stated_expectations": {
  "note": "寫規格時的預期，只供事後對照，不是判定標準。",
  "E1": "D1 下真實 +5 個百分點、10 年固定期限單尾 5% 檢定力約 25～35%；真實 +12 約 70～80%；真實 0 時 1 年看起來 ≥ +5 的機率約 35～40%、10 年約 15～18%。",
  "E2": "序貫設計在 δ＝0 的模擬型一誤差 4～5.5%（正偏態使上尾 t 檢定略保守）；δ＝+12 時判定有效的年份中位數約 8～10 年，δ ≤ +5 時多數路徑 10 年內不會判定有效。",
  "E3": "正式無效界線很難碰到：δ＝0 時 10 年內觸發 < 25%；多數『停用』來自務實停用線與第 10 年終局規則。δ＝+5 時被務實停用線誤停的機率約 25～35%。",
  "E4": "追蹤誤差 5.6 的敏感度下檢定力大幅提高（δ＝+5、10 年 > 80%）。",
  "E5": "貝氏：主先驗、追蹤誤差 16.2 下，5 年觀察到 −5 個百分點／年時 P(δ>0) 仍約 45～50%。"
 },
 "reporting_commitments": "全部數字照實報告（含『10 年也判不出來』這種不利結論）；程式錯誤修正寫入 spec_deviations；size_contingency 有無觸發照實寫。",
 "N": "不增加資料窺探族群 N。"
}
for name, obj in (('SPEC', SPEC), ('PREREG', PREREG)):
    p = H / f'{name}.json'
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    (H / f'{name}.sha256').write_text(f'{h}  {name}.json\n')
    print(name, h)
