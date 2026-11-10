# interp_register v1：加入名單層與統計層的選擇（仍在任何市場數字之前；不變量檢查只輸出結構量）
import json, hashlib, datetime, os
HERE = os.path.dirname(os.path.abspath(__file__))
reg = json.load(open(os.path.join(HERE, 'interp_register.json')))
S_O4 = 'o_v4_revenue_patch_official/results.json PREREG.V4_spec＋FINDINGS 380～400 行'
S_ER2 = 'o_rev_avail_ds_pit_count/SPEC.json survivorship.U_opt（ER2 文字）'
S_V7 = 'o_v7_listing_exec_engine/frozen_spec.txt 二'
add = lambda i, topic, choice, adverse, src, gap=False: reg['items'].append({'id': i, 'topic': topic, 'choice': choice, 'adverse_side': adverse, 'source': src, 'spec_gap': gap})
add('L01','訊號日','官方日＝營收月 m−1 的次月第一個 ≥ 11 日的面板交易日（2006-04-11～2026-09-11，245 日；≥ 2026-10-12 不交易）；成交日＝次一面板日。j 日：J_j＝deadline_calendar.csv 的 deadline_D（10 日起第一個交易日）在面板上的位置＋j 個面板日。','—',S_O4+'；o_v6 FINDINGS 449 行（day_for 文字）', gap=True)
add('L02','股池','訊號日：全部面板股票中 V60≥1e7 且收盤≥10，依 V60 取前 300（先排名後排除）；再要求 52 週比有值、成交日有成交且有開盤、非金融、非外國（KY／DR／F-／91xx）、非處置（成交日口徑：公告日 ≤ d 且 start ≤ e ≤ end）、y3 有值。V60 用 E04 定義。','—',S_O4+' pool', gap=True)
add('L03','52 週比','還原指數 adj（P 期間以 G 連乘延伸）在 GOOD 日上滾動 252 日最大值（min_periods=252），比值＝adj/max。','min_periods 未明文',S_O4+' pool；SOP 步驟 3', gap=True)
add('L04','金融','stock_list.csv 產業含「金融」∪ 代碼 28xx ∪ 5876、5880 ∪ 終止上市金融股 5854、6004、6012。','—','o3 凍結規格；SOP 2026-11-06 註記', gap=True)
add('L05','外國股','代碼 91xx，或任一名稱（stock_list、tw_revenue、delist_official、merger_consideration）含 KY、DR 或以 F- 開頭。','—',S_ER2+' FOREIGN；o15 複製員差異 1', gap=True)
add('L06','營收','訊號日 < 2021-03-01 用 yoy_long_R_FIX.parquet，之後用 yoy_rev5y_FIX.parquet（皆為凍結資料檔，非程式）；y3＝m−1、m−2、m−3 年增率平均，三個都要有；未公告者改用 m−2、m−3、m−4。名單層的營收修補（R_FIX 新聞修補 45 格、o2 補值）沿用凍結資料檔，未重建。','—',S_O4+' revenue', gap=True)
add('L07','分數與同分','池內百分位＝average rank ÷ N；分數四捨五入到 1e−10；同分序：分數高 → y3 高 → 面板欄位序；B10 放回股營收百分位 1.0、同分序 y3＝+∞，52 週百分位在含放回股的池內重算；非放回股 y3 百分位＝非放回股間 average rank ÷ N（N 含放回股）。','—',S_ER2+'；'+S_O4+' bounds')
add('L08','開盤漲停遞補','成交日開盤 ≥ 漲停（E08 跳動點價）且不在前一期名單內者跳過，由下一名遞補（前一期已持有者照留）。','held 判定以前一期名單代理',S_V7+'；o3 凍結規格「開盤漲停（跳動點）由下一名遞補」', gap=True)
add('L09','B10','gap＝delist_official.csv 中 4 碼、非外國、非金融者；訊號日 ≤ 終止上市日、營收以外條件都合格但 y3 缺 → 放回（L07）。','—',S_O4+' bounds；'+S_V7)
add('L10','AB（mc）','D<S：全部已公告；J＝D（含 j0 每個月）：各股依 p_delay.json 產業延後機率（stock_list 產業，查不到用 p_overall，金融 1）以自訂亂數 numpy default_rng(10000+seed) 抽未公告；J < D（2019-01 延至 02-15、2023-03 延至 04-12）全部未公告；2020-01 疫情月 D < J ≤ 2020-02-14 也抽 p；2026-09 營收月以 firstseen A_before 判定；2026-01～08 營收月 J < D15 時 INS15 名單未公告。原 RNG 不可重現 → 逐 seed 名單不可能逐位一致（規格缺口）。','—',S_O4+' AB；rev_announce/deadline_calendar.csv', gap=True)
add('L11','ALLPIT／A_UB／官方處置 union','不重建（E01）。可執行驗證改用 K 在凍結後以原程式產生的 REAL_opt／p90／carry 名單。','—','E01', gap=True)
add('L12','紙上 C0 對照','紙上零衝擊引擎不重寫成 ia 版（成本 0.1% 滑價＋0.0855% 手續費、二分 60 次、cons_main 結算）：本題以 V7X 引擎開關「無整股、無衝擊、滑價 0.1%、手續費 0.1425%×0.6、無最低手續費、無 2% 現金、無上限」近似紙上；紙上 22.22%／t 1.64 以原 ALLPIT_REAL_opt 紙上名單（K 產生）跑此近似引擎比對，並列為規格缺口近似。','—','任務 R3；o4 FINDINGS 380 行', gap=True)
add('S01','dev 切片','2006-04-10～2008-12-31、j 0～2、seeds 0～2、100 萬一般 GROW、orig 合併口徑、我的 AB_V7_B10 名單；K 只回傳每個 j 的年化差（K 端是 20 seeds 平均，seed 不同造成雜訊），不回傳 t。','—','任務 R1')
add('E33','round(S) 賣量','順延日賣量 round(S) 可能比持有多不到 1 股；成交後持股若在 (−0.5,0) 設為 0。','—',xengine_src if (xengine_src:='E19') else '')
reg['version'] = 'v1'; reg['written_utc_v1'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
reg['v1_note'] = 'v1 在第一次 dev 切片之前寫入；不變量檢查（logs/invariants_v1.json）只輸出結構量。'
s = json.dumps(reg, ensure_ascii=False, indent=1)
open(os.path.join(HERE, 'interp_register.json'), 'w').write(s)
print(len(reg['items']), hashlib.sha256(s.encode()).hexdigest())
