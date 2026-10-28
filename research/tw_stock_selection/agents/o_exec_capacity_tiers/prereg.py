#!/usr/bin/env python3
"""凍結：把最終規格寫成 PREREG（含全部腳本、引擎庫、data/、raw/ 的 sha256 與時間戳）存進 results.json。
之後 run_oos.py／replicate 會斷言 PREREG 存在、未被改過、所有檔案雜湊相同，才計算。
用法：python3 -I -B prereg.py
"""
import json, hashlib, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()

FROZEN_SPEC = r"""
o_exec_capacity_tiers 凍結規格（PREREG）

〇、範圍
- 目的：用同一張 AB_V4 名單（逐 seed），模擬 30 萬、100 萬、300 萬、1,000 萬四個資金級距的可執行回測，交出 SOP 容量表。不改任何選股規則。
- 名單：data/plans_ab.csv（seed 0～39，欄 seed,signal,exec,exec_idx,picks）。由 gen_plans.py 以 o_v4_revenue_patch_official/v4lib.py 產生：
  V4(R_FIX, Y5_FIX, 處置 trade 口徑, unavail=v2_core.make_unavail('mc', seed)).plan(10, 'official')；零衝擊引擎（O3.simulate、一般成本）逐 seed 重現、40 seed 平均日報酬淨值與 navs_v4.parquet 的 AB_V4|normal 逐日相差 0。另有 data/plan_v4.csv（非 AB 主名單，seed −1，只報告）。
- 期間：t0＝2006-04-11（第一個訊號日）收盤起算，t1＝2026-10-12 收盤。年化＝(淨值_t1/淨值_t0)^(365.25/天數)−1；簡單報酬；現金不計息。

一、資料（全部在本題 data/、raw/，雜湊見 PREREG.file_sha256）
- px_open、px_close（原始價，P 下市股價格已補）、px_grow（1＋ret，無成交＝1）、px_traded、prevclose（前一筆收盤，跨停牌）：只含曾入選的 521 檔，面板全期日曆。
- CF＝close 前值補。ADJ_t＝有成交時 G_t×PC_t/C_t（非有限→1），無成交＝1。GAP_t＝O_t/PC_t−1（有成交且有開盤；否則缺值，用到時當 0）。
- feat_daily：V60（gapfix 60 GOOD 日平均成交值）、sig60（60 個 GOOD 日 ret 標準差，≥40 筆，向後補值）、vrank（V60≥1e7 且收盤≥10 的上市股中 V60 由大到小名次）。成交日 t 一律用 t−1 的值。
- xrules.json：X 出清（factor, release, type, valid_until）。
- model_spec.json：iterate.py 選定的 v4 執行量模型（見三）。
- bench.parquet：0050 含息、AB_V4|normal（零衝擊 AB）、AB_V4|EW（同池等權）、AB_V4|stress。

二、帳戶與級距
- 級距 tier（2026 幣值）：30 萬、100 萬、300 萬、1,000 萬。換成當年幣值：× CPI[年]/CPI[2026]（raw/cpi_tw.json，2006→2026 ×0.765）。
- GROW（判定用）：t0 收盤現金＝tier×CPI[2006]/CPI[2026]，之後隨自身淨值成長、不提領不加碼。
- FIX（另報）：每個換股日開盤前，把全部股數與現金乘 sc＝tier×CPI[該年]/CPI[2026] ÷ 權益（權益＝可交易持股×開盤＋凍結持股×前收＋現金），無成本；之後才換股。淨值用時間加權：該日 unit ×＝(權益_pre/前日權益)×(收盤權益/(權益_pre＋flow))，flow＝target−權益_pre。
- 其他日 unit ×＝收盤權益/前日收盤權益；收盤權益＝現金＋Σ 股數×CF_t。

三、執行量模型（model_spec.json；iterations.json 選 v4，5 版中第 4 版）
- OA（開盤集合競價成交值）＝V60_{t−1} × κ × exp(b·x)，κ＝0.7789（vol_curve 中 OA_central÷09:05–09:10 量的中位數）。
  x＝[1, ln max(vrank,1), ln PC_t, ln max(σ,1e−4), |GAP_t|, pick, drop]；b＝[−3.05646, −0.05438, 0.03689, 0.21549, 20.5129, 0.76624, −0.05387]（以 model_spec.json 全精度為準）。
  pick＝換股日當天的入選股（含續抱）；drop＝換股日被剔除的持股；順延日 pick＝drop＝0。
- drift（9:10 價÷開盤−1）：依 GAP_t 分桶 [−3%, −1%, 0, 1%, 3%]（右閉 searchsorted 'right'），各桶平均（model_spec.json）。
- 零股成本 c（相對同時點整股價；正＝不利）：依 PC_t 分桶 [50,100,300] × vrank_{t−1} 分桶 [150]（vrank 缺值當 1e9）的中位數，買、賣各一表。

四、逐日流程（t＝t0+1…t1）
1) 股數 S ← S × ADJ_t。
2) 若 t 是某換股日的成交日 e：
   a) 清掉尚未成交的順延委託。
   b) 凍結＝持有（S>0）且當日無成交或無開盤。凍結且有 X 規則、release ≤ t < valid_until → 現金＋＝S×CF_{t−1}×factor，S＝0。
   c) FIX：縮放（二）。
   d) 可交易持股的小數股 S−floor(S+1e−9) 以開盤價 O_t 轉成現金（無成本；代表現金股利入帳）。
   e) 換股（五）。
3) 否則若有順延委託：執行（六）。
4) 計算收盤權益、淨值。

五、換股（成交日 t＝e，入選 pk，k＝檔數）
- cur＝可交易且 S>0 的持股（股數取整）。
- 目標股數：入選股 p＝PC_t（前一筆收盤）；張數 L＝floor(T/(1000p)+1e−12)；零股 q＝floor((T−1000Lp)/p+1e−9)；目標＝1000L＋max(q,0)（不足 1 股留現金）。非入選的可交易持股目標＝0。
- delta＝目標−cur；整股張數＝|delta|//1000，零股股數＝|delta| mod 1000。
- 單檔成本函數 cost(j,delta)（買＝現金支出、賣＝負的淨收入；side＝±1）：
  整股（張數 L>0）：imp＝0.001＋0.5×σ_{t−1}×sqrt(1000·L·PC_t / OA)（OA 有限且 >0、σ 有限才加第二項）；val＝1000·L·O_t·(1＋side·imp)；cost＋＝side·val＋fee(val, 整股)＋(賣：0.003·val)。
  零股（q>0）：基準價 b＝ t<2020-10-26：C_t（盤後零股，以成交日收盤代理）；t≥2020-10-26：O_t×(1＋drift(GAP_t))（盤中零股 9:10 首次撮合）；val＝q·b·(1＋side·c)；cost＋＝side·val＋fee(val, 零股)＋(賣：0.003·val)。
- 每檔目標金額 T：hi＝(Σ_cur S×O_t＋現金)/k；need(T)＝Σ_j cost(j, delta_j(T))。若 need(hi) ≤ 現金，T＝hi；否則在 [0,hi] 二分 40 次（need(mid) ≤ 現金 → lo＝mid，否則 hi＝mid），T＝lo。（用當日實際開盤／收盤價估計，與引擎同為「已知開盤價」的理想化。）
- 執行順序：delta 由小到大（賣單先）。買單若 cost(j,delta) > 現金＋1e−9 → 整筆列入順延。否則先下零股、再下整股。
- 整股參與上限：當日可成交張數 ≤ max(1, floor(0.20×OA/(1000·PC_t)))；超出張數列入順延。衝擊用實際成交張數計算。
- 手續費 fee(val)：情境 A（主）：max(20, 0.001425×0.28×val)，整股、零股都最低 20 元；B：0.6 折、最低 20；C：0.28 折、整股最低 20、零股最低 1。證交稅 0.3%（賣）。每一筆（每檔每日的整股、零股各一筆）各自計最低手續費。
- KEEP（敏感度，SOP 第 6 步）：已持有的入選股不下單，只有新進股與剔除股。

六、順延日（換股日之後、下一換股日之前）
- 依待成交股數由小到大（賣先）。當日無成交或無開盤 → 原量留著。
- 賣：數量＝min(待賣, round(S))。買：張數上限 floor(現金/(O_t×1000×1.05))；若零股成本 > 現金−張數×O_t×1050，零股當日不下。
- 先零股、再整股（整股同樣受參與上限與衝擊；pick＝drop＝0）。剩餘留到下一日；到下一個換股日清掉並記錄未成交金額。

七、變體（全部事前登記，一次跑完）
- 主格：40 seeds × 4 級距 × {GROW, FIX} × 手續費 {A, B, C}。
- 拆解（A）：d1 取整（整股＋零股、全部在開盤、引擎手續費 0.6 折無最低、零股無價差、整股 0.1%）→ d2 零股時點（2020-10-26 前收盤、之後 9:10）→ d3 零股價差 → d4 手續費情境 A（最低 20 元）→ d5 平方根衝擊 → 主格（＋20% 參與上限與順延）。
- 敏感度（A、GROW）：κ 低（p25＝0.31）、κ 高（V(09:01) 口徑＝1.04）、2020-03-23 逐筆交易前 OA×0.5、零股壓力（買成本×2、賣成本≤0 者改 0）、KEEP。
- 壓力（GROW、FIX）：手續費 B＋κ 低＋零股壓力。
- 非 AB 主名單 V4（seed −1）：A、GROW／FIX，只報告。

八、統計（每個變體都以 40 seed 的日報酬平均連乘成一條「可執行 AB」淨值；另報逐 seed）
- 對 0050 的 t：訊號窗報酬（訊號日收盤→下一訊號日收盤，最後一窗到 2026-10-12）減 0050 含息同窗，窗起點落在區間內，一般 t。
- 分段：S1 2006-04～2013、S2 2014～2020、S3 2021-03～2026-10-12、SEL 2006-04～2018、VAL 2019～2023、HOLD 2024～2026-10-12、FROM2008 2008～、PRE_ODD 2006-04～2020-10-25、POST_ODD 2020-10-26～。
- 端點：拿掉最好 1／3／5／10 個月、最後 1 個月、第一年；起點改 2007、2008。
- 逐 seed：t 中位數、範圍、t<2.0 與 <2.3 的 seed 數；年化中位數與範圍。
- 與零衝擊 AB_V4 的配對差（訊號窗月差）：平均、t、80% 檢定力 MDE＝2.8×SE（月、×12 年）。
- 年化成本拆解：證交稅、手續費、衝擊（扣掉 0.1% 基礎）、基礎 0.1%、零股價差、零股時點（相對開盤），各為「金額÷當日開盤前權益」逐筆加總÷年數；現金拖累＝平均現金比例與拆解步驟 d1 的 ΔCAGR。
- 成本損益兩平：在可執行成本上再加單邊滑價 x（日報酬 − x×當日成交金額÷權益），40 seed 平均後 t 跌到 2.3、2.0 的 x。
- 執行量：新進股名額中只能零股（張數 0）的比例；換股日整股委託中預定張數超過 OA 10%、20% 的比例（全期、2006～2018、2019～2026、名次 201～300）；衝擊成本落在名次 201～300、|跳空|>3% 的比例；零股成本依價位（>300）與制度期分段。
- 取整追蹤誤差：d1 與零衝擊逐 seed 日報酬差的年化標準差中位數。

九、判定（每個級距一格；主欄＝GROW、手續費 A 的可執行 AB；歷史回測判定上限為待觀察）
- 建議：t ≥ 2.3 且年化比零衝擊 AB_V4（27.00%）少 ≤ 1.5pp。
- 可行但打折：t ≥ 2.0。
- 不建議：t < 2.0，或對 0050 年化超額 < 2pp。
- 另報 FIX、手續費 B、壓力下的判定。
- 可證偽：30 萬與 300 萬（FIX，級距意義固定）年化差 < 0.5pp（附配對 MDE）→「小帳戶被零股制度懲罰」不成立。GROW 另報。

十、SOP 容量表（只寫事前規格結果）
- 每級距：預期年化（GROW 與 FIX）、t、零股名額比例、下單方式：整股＝成交日開盤市價（集合競價）；零股＝2020-10-26 起盤中 9:10 首次撮合（現行），之前盤後零股。
- 事後想到的執行建議（分 3 個開盤、限價單、只用整股忽略零股餘額…）不進 SOP，只列為假說交給 live_tracking。
- 事前登記的校準規則：2026-11 起 live_tracking 滿 6 個換股日後，若實際成交價相對開盤價的價差中位數比模型（整股：0.1%＋0.5σ√(Q/OA)；零股：drift＋c）高 > 0.05%，容量表重新發布。若盤中零股改為 9:00 開盤（報導 2026-12-07 或 2027-07），零股時點參數也要重估。
"""

PREDICTIONS = {
    'P1_30萬': '零股名額 75～85%；2020-10 前盤後零股（收盤代理）使 AB t 降 0.15～0.3；最低 20 元、2.8 折每年多付 0.5～1.5pp；年化比零衝擊 AB 少 1.5～3pp',
    'P2_100萬': '零股名額 40～60%；年化少 0.5～1.5pp',
    'P3_300萬': '年化少 0.3～1.0pp；t 2.3～2.45，最接近回測',
    'P4_1000萬': '超過 OA 10% 的名額 15～35%；計入衝擊後 t 2.0～2.25；2019 年後超標比例低於 2006～2018',
    'P5_子樣本': '零股成本在 >300 元與 2020-10 前最大；衝擊集中在名次 201～300 與跳空大的日子',
    'P6_校準': '開盤集合競價量占全日 2～6%，中小型股較低',
    'falsify': '30 萬與 300 萬年化差 < 0.5pp（附 MDE）→ 小帳戶被零股制度懲罰不成立',
}

files = []
for p in sorted(HERE.glob('*.py')): files.append(p)
for p in sorted((HERE / 'replicate').glob('*.py')): files.append(p)
for p in ('explore.json', 'iterations.json'): files.append(HERE / p)
for p in sorted((HERE / 'data').glob('*')):
    if p.suffix in ('.log',): continue
    files.append(p)
for p in sorted((HERE / 'raw').rglob('*')):
    if p.is_file(): files.append(p)
ENGINE = [AG / 'exec_rebuild' / 'exlib.py', AG / 'o_v4_revenue_patch_official' / 'v4lib.py', AG / 'main_v2' / 'v2_core.py',
          AG / 'independent_audit' / 'ia_core.py', AG / 'o_infra_v3_rebuild' / 'o3_core.py', AG / 'o_disposal_rejudge' / 'drj_core.py',
          AG / 'o_v4_revenue_patch_official' / 'data_freeze_v4.json', AG / 'o_v4_revenue_patch_official' / 'navs_v4.parquet',
          AG / 'o_v4_revenue_patch_official' / 'plan_V4.csv']
files += ENGINE
it = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
PREREG = {
    'topic': 'o_exec_capacity_tiers：實盤執行規格（30 萬～1,000 萬可執行回測、零股制度史、開盤集合競價量）',
    'frozen_spec': FROZEN_SPEC, 'predictions': PREDICTIONS,
    'model_version_selected': it['selected'], 'n_model_versions': it['n_versions'],
    'file_sha256': {str(f): sha(f) for f in files},
    'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'disclosure': ('PREREG 前看過的東西：探索與迭代只用執行量／價格資料；模擬器除錯時曾用佔位參數（OA＝5% ADV 常數）跑過 seed 0 的 30 萬／300 萬／1,000 萬 GROW 終值，'
                   '以及 seed 3 的四項成本比例（不含報酬）。這些沒有用來選任何參數；執行量模型只依 iterations.json 的校準損失選版。'),
}
res_f = HERE / 'results.json'
res = json.loads(res_f.read_text(encoding='utf-8')) if res_f.exists() else {}
assert 'RESULTS' not in res, '已經跑過樣本外，不得重寫 PREREG'
res['PREREG'] = PREREG
res['PREREG_sha256'] = hashlib.sha256(json.dumps(PREREG, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
res_f.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print('PREREG', res['PREREG_sha256'], PREREG['timestamp_utc'], 'files', len(files))
