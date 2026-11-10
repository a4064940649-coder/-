# R1 第一步：在算任何市場數字之前寫 interp_register.json（每一個規格模糊處的選擇、不利端、出處）
import json, hashlib, datetime, os
HERE = os.path.dirname(os.path.abspath(__file__))
S_V7X = 'o_exec_engine_rewrite_v7x/frozen_spec.txt'
S_IR = 'o_exec_engine_rewrite_v7x/interp_register.json'
S_O5 = 'o_exec_capacity_tiers/FINDINGS.md 凍結規格'
S_M7 = 'o_v7_listing_exec_engine/data/model_spec7.json'
S_V7 = 'o_v7_listing_exec_engine/frozen_spec.txt'
S_CZ = 'o_engine_delist_consideration/FINDINGS.md frozen_spec'
S_O3 = 'o_infra_v3_rebuild/FINDINGS.md 凍結規格'
S_O4 = 'o_v4_revenue_patch_official/results.json PREREG.V4_spec＋FINDINGS 第 380～400 行'
items = []
def add(i, topic, choice, adverse, src, gap=False):
    items.append({'id': i, 'topic': topic, 'choice': choice, 'adverse_side': adverse, 'source': src, 'spec_gap': gap})

add('E00','範圍','本題角色 A／K／J／V 由同一個子代理依序扮演（本執行環境沒有可呼叫的子代理工具）：A 階段（重寫、凍結）在讀任何原引擎 .py 之前完成；K 階段（重跑原引擎、讀原碼）只在凍結之後。四角色分離不成立 → G6 依事前規則最多只能判「部分進展」，不能判通過。','—','任務文字 R0～R5；BRIEF', gap=True)
add('E01','名單範圍','可執行引擎驗證用原名單（K 於凍結後產生 orig／REAL_opt／REAL_p90／REAL_carry 的 AB_V7_B10 j 名單）；名單層另做 clean-room 重建：官方日確定性 V7_B10 名單（逐槽比對 plan_V7_B10.csv）與 AB_V7_B10 13 j×20 seeds（自訂亂數，與原 RNG 不可能逐位一致）。ALLPIT 放回（ER2 原因分類、B1 補值、A_UB）文字規格以程式函式定義（pclib.VU、v2_core.deadline_case、v7x_lib docstring），本題不重建，登記為規格缺口。','—','SPEC.json v7x.definition「見 v7x_lib.py docstring」', gap=True)
add('E02','價格面板','V7 凍結面板 close_raw/open/value/ret/flag；delist_prices.parquet 只補面板缺值格（close/open/value/ret）。TR＝收盤有值；G＝1＋ret（無成交＝1；ret 缺而有成交＝1）；CF＝收盤前值補；PC_t＝t 之前最後一筆收盤；ADJ_t＝G·PC/C（無成交＝1，非有限＝1）。',' —',S_V7X+' 一')
add('E03','GOOD 日','GOOD＝當日收盤有值檔數 ≥ 300（以 V7 面板＋P 補值後計數）。','—',S_V7X+' 一；BRIEF 第 2 輪補充')
add('E04','V60','60 個 GOOD 日滾動平均成交值；窗內任一 GOOD 日缺值 → NaN，但豁免日 2017-08-03、2018-07-12、2024-02-15～2024-03-15 的缺值不算（以窗內非缺值平均）。非 GOOD 日不計入窗、V60 沿用前一個 GOOD 日值。','豁免期以平均代替：與 NaN 相比較有利（多一些股票有 V60）→ 規格明文豁免，照做',S_V7X+' 一')
add('E05','σ60','60 個 GOOD 日 ret 標準差（ddof=1，至少 40 筆非缺值），之後向後補值（ffill）。','—',S_V7X+' 一')
add('E06','vrank','t−1 日：全部面板股票（含金融）中 V60≥1e7 且當日收盤（close_raw，非補值）≥10 者依 V60 由大到小名次（同值依欄位序）。dec＝ceil(vrank/30)，>300 或缺值＝10。','—',S_V7X+' 一；'+S_IR+' A20')
add('E07','跳動點','台股現行升降單位（2005-03 起）：<10 0.01、<50 0.05、<100 0.1、<500 0.5、<1000 1、其餘 5；以價格所在級距決定。','—','BRIEF「跳動點漲停」；'+S_IR+' A14')
add('E08','漲跌停','參考價＝C_t/G_t（flag==1 用 PC）；漲停＝floor(ref×(1+L)/tick)×tick，跌停＝ceil(ref×(1−L)/tick)×tick，tick 用 ref×(1±L) 所在級距；L＝7%（2015-06-01 前）／10%。買單 O ≥ 漲停−1e−9、賣單 O ≤ 跌停＋1e−9 → 該檔當日不成交、順延（整股＋零股）。','—',S_V7X+' 三；'+S_IR+' A14')
add('E09','OA','OA_t＝share[dec_{t−1}]×value_t（衝擊用）；OA_pred＝share[dec_{t−1}]×V60_{t−1}（F4 參與上限用）。OA_pred 缺值 → 參與上限取 1 張（不利端）。OA 缺值或 ≤0 → 省略平方根項。','OA_pred 缺值時上限 1 張（不利）',S_V7X+' 三、四 F4；'+S_M7, gap=True)
add('E10','K5T','整股每邊 imp＝max(0.001＋0.5·σ60_{t−1}·√(Q/OA_t), tick(O_t)/O_t)，Q＝成交張數×1000×PC_t；壓力：imp＋0.004（加在 max 之後）。成交價＝O_t×(1±imp)。','—',S_V7X+' 三')
add('E11','零股價','2020-10-26 前＝C_t（盤後零股）；之後＝O_t×(1＋drift(GAP_t))，GAP＝O_t/PC_t−1（缺＝0），drift 邊界 searchsorted right。成本 c：buy[price_bucket(PC_t)][rank_bucket(vrank_{t−1})]，price edges [50,100,300]、rank edges [150]，searchsorted right（缺 vrank 當 1e9）；賣出 c＝max(c,0)（ODD0）。成交價＝b×(1±c)。','—',S_V7X+' 三；'+S_M7)
add('E12','零股量上限','2020-10-26 起 ≤ floor(1.0×zcap[dec]×V60_{t−1}/零股價)；之前（F3）≤ floor(0.5×zcap[dec]×V60_{t−1}/C_t)；名次 >300 或缺值用 share_default；V60 缺值 → 0 股（不利端）。超過順延。','V60 缺值 0 股（不利）',S_V7X+' 三、四 F3；'+S_IR+' A06', gap=True)
add('E13','手續費／稅','一般：2015-01-01 前 0.1425%×0.6，之後 0.1425%×0.28；每檔每日整股、零股各一筆，各自 max(20, rate×val)；賣出稅 0.3%（整股、零股、處置賣出）。壓力：全期 0.1425%×0.6、最低 20。','—',S_V7X+' 三；'+S_O5+' 五')
add('E14','壓力零股','壓力：零股買 c×2 再 +0.004；賣 max(c,0)+0.004；整股 imp＋0.004。','—',S_V7X+' 三（iterations v2）')
add('E15','2% 現金','hi＝(Σ可交易持股 S×O_t＋現金)/k×0.98；k＝名單檔數（10）。可交易＝當日有成交且有開盤。','—',S_V7X+' 三；'+S_IR+' A17')
add('E16','目標股數','p＝PC_t；L＝floor(T/(1000p)+1e−12)；q＝floor((T−1000Lp)/p+1e−9)；目標＝1000L＋max(q,0)。30 萬整股主欄：2020-10-26 前目標＝1000L（不下零股）。','—',S_O5+' 五；任務 P3')
add('E17','need(T)','Σ 全部名單股與非名單可交易持股的 cost(delta)：整股成本用全部張數、不套參與上限、K5T 用實際 OA；零股照價；處置期賣出以 C−tick、無衝擊，且 F2 下其款項不計入（need 中處置賣出貢獻 0）；need(hi)≤現金 → T＝hi，否則 [0,hi] 二分 40 次取 lo。LIM 或無成交的股不在 need 中排除（照常估計）。','LIM 股仍估計成本（略保守）',S_V7X+' 三；'+S_O5+' 五', gap=True)
add('E18','執行順序','delta 由小到大（賣先）；每檔先零股後整股；買單若 cost（以實際可成交量計）> 現金＋1e−9 整筆順延；整股超過參與上限的張數順延；換股日無成交／無開盤／LIM → 整筆順延。','—',S_V7X+' 三；'+S_O5+' 五')
add('E19','順延日','依待成交股數由小到大（賣先）；賣量＝min(待賣, round(S))；買張數上限 floor(現金/(O×1000×1.05))；零股成本 > 現金−張數×O×1050 → 當日不下零股；先零股後整股；整股受參與上限與 K5T；到下一換股日清掉。','—',S_V7X+' 三；'+S_IR+' A03')
add('E20','處置賣出','賣出日 t 在 V4 凍結官方處置名單（disposal_hist_rejudge in_panel ∪ disposal_5y 補缺）start≤t≤end：整股＋零股以 C_t−tick(C_t) 成交，無衝擊、無零股價差、不受 LIM；F2：當日合計 ≤ floor(0.20×value_t/C_t) 股，超過順延；款項隔日才可用（記入隔日現金）。換股日處置賣出在開盤階段之後以收盤成交。','—',S_V7X+' 三、四 F2；'+S_IR+' A15')
add('E21','GROW','j 名單第一個訊號日收盤持有現金 tier×CPI[2006]/CPI[2026]；不提領不加碼；FIX 只作原設定，不跑。','—',S_V7X+' 二')
add('E22','小數股','換股日把可交易持股小數股 S−floor(S+1e−9) 以 O_t 轉現金、不計成本（ADJ 造成的小數股）。','—',S_IR+' A21')
add('E23','合併下市 orig（A16）','merger_consideration type=cash（金額缺 → 比例 1）：換股日 t ≥ 下市日（delist_date；缺＝last_trade＋14 天）且持股凍結時，S×PC×比例 轉為 0050 計值待撥款（當日 0050 報酬計入）；撥款日＝下市日起第一個面板日之後第 20 個面板日；撥款日之後第一個換股日以前一日收盤值併入現金（當日 0050 不計）。其他 X（swap／unknown／cash 無額→×1 於下市日後第一個換股日轉現金；failure→最後交易日後第一個換股日歸零），不付成本。','—',S_V7X+' 三；'+S_IR+' A16；'+S_O3+' X')
add('E24','合併下市 cons_main','events_v2.csv：觸發＝最後成交日 lt 的下一個面板日，部位轉請求權。cash：S×cash，終止日＋20 面板日入帳前以 0% 現金計；swap／conversion：S×ratio 轉存續股，s＝存續股第一個可交易日（≥ 觸發日）起跟其報酬，空窗期固定最後收盤，之後由一般換股流程賣出（若存續股剛好入選則保留）；mixed：現金＋股票分別處理，特別股以 pref_value 固定，s 後下一換股日以一般賣出成本出清（本題以 pref_value 現金化不付成本，規格缺口）；swap_ur：ratio＝lt 日標的收盤/存續公司收盤；acq_ur、unknown（main）：最後收盤固定，終止日＋20 面板日轉現金；distress：0；price_gap：面板最後收盤固定，下一換股日轉現金（以 0 成本，規格缺口：原文為一般賣出成本，但無價可成交）。不在事件表的永久停牌：沿用 E23。','price_gap 與特別股處置採 0 成本（偏有利，影響極小）',S_CZ+' 一', gap=True)
add('E25','淨值','收盤權益＝現金＋隔日現金＋Σ S×CF_t＋待撥款＋請求權；20 seeds 日報酬平均成一條淨值；年化＝第一個訊號日收盤→終點（min(該 j 的 2026-10 訊號日, 2026-10-22)），365.25 日。','—',S_V7X+' 五；jscan_end_days.json')
add('E26','t 與分段','訊號日窗（訊號日收盤→下一訊號日收盤，最後一窗到終點）超額＝策略窗報酬−0050 同窗報酬，t＝mean/(sd(ddof=1)/√n)。全期窗全部；2019 起：窗起點 ≥ 2019-01-01，年化區間 [2019-01-02, 終點]（基期 2018 最後交易日收盤）。三段：2006～2012（年化至 2012 最後交易日）、2013～2018、2019 起；段界 2013-01-01、2019-01-01 起第一個交易日。超額＝每個 j 的（年化−同期 0050 年化）再取 13 j 中位數；p10＝np.percentile(…,10)（線性）。','—',S_V7X+' 五；任務 R1')
add('E27','0050','etf_0050.csv adj；對齊面板日期（缺值前值補）；日報酬＝adj_t/adj_{t−1}−1。','—','BRIEF')
add('E28','處置名單','V4 凍結官方名單＝o_disposal_rejudge/disposal_hist_rejudge.csv 的 in_panel 列 ∪ disposal_5y 中前者沒有的（代碼, 起日）列；start≤t≤end。','—',S_V7X+' 一；'+S_O4)
add('E29','T+2','V7X 規格沒有一般 T+2 現金時差（同日賣出款可用於同日買入，台股 T+2 雙邊同日交割軋差），唯一的現金時差是 F2 處置賣出款隔日可用與合併撥款延遲。任務文字「T+2」依此解讀。','—',S_V7X+' 四 F2', gap=True)
add('E30','30 萬整股主欄','2020-10-26 前不下任何零股（買賣都只做整股；持有的零股留著，賣出時零股部分順延到 2020-10-26 後或不賣；本題：賣出零股照盤後零股規則可成交——買不到不等於賣不掉，取保守：賣出零股也禁止）。敏感度：30 萬照 V7X（盤後零股＋F3 0.5）。','賣出零股也禁止（不利端）','任務 P3', gap=True)
add('E31','attribution 階梯','2019 起落差拆解在本引擎上逐項開關：S0 訊號日收盤成交、零成本、可分割、無 2% 現金；S1 改隔日開盤；S2＋LIM 順延；S3＋證交稅；S4＋手續費（含最低 20）；S5＋K5T 衝擊與參與上限；S6＋整股／零股（價差、時點、量上限）；S7＋2% 現金＋處置規則＋順延＝V7X。名單差另以「官方日 vs j 中位數」與「紙上名單 vs AB_B10」報。','—','任務 P4')
add('E32','右尾捕捉','持股槽＝(j, seed, 換股日, 股票)；紙上槽報酬＝成交日開盤→下一成交日開盤（G 連乘含息）；可執行槽報酬＝該槽實際持股在同期的損益 ÷ 目標金額 T（含成本、未成交以 0 計）。前 5%＝紙上槽報酬前 5%；捕捉率＝Σ 可執行／Σ 紙上。','—','任務 P5')
reg = {'version': 'v0', 'written_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'note': '在任何市場數字（報酬、年化、t）計算之前寫定；之後若修改，以新版本號另存並記錄原因（iterations.json）。',
       'sources_read_before_v0': ['BRIEF.md','選股SOP.md 步驟與執行注意', S_V7X, S_IR, S_M7, S_V7, S_O5, S_CZ+'（文字）', S_O3+'（文字）', S_O4+'（文字）', 'o_delist_rev_backfill_allpit/SPEC.json＋FINDINGS 開頭附註', 'original_o18/DIRECTOR_REPORT.md 3～7 節', 'o_rev_avail_ds_pit_count/SPEC.json（文字）'],
       'not_read': ['任何研究員或複製員 .py（含 x_exec、x_data、x_ideal、x_stats、pclib、cz_engine、cz_v7x、v7env、ia_core、btlib、v2_core、v4lib、o3_core、st_core）', 'plans_j7.csv、plans_ab7.csv、navs_oos_R*.parquet（密封）'],
       'items': items}
s = json.dumps(reg, ensure_ascii=False, indent=1)
open(os.path.join(HERE, 'interp_register.json'), 'w').write(s)
print(len(items), hashlib.sha256(s.encode()).hexdigest())
