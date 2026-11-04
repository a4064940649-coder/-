#!/usr/bin/env python3
"""凍結：寫 EXEC_SPEC.json、frozen_spec.txt，並把 PREREG（含雜湊與時戳）存進 results.json。之後才可跑 s4_oos.py（只跑一次）。
用法：python3 -I -B s3b_prereg.py"""
import sys, json, datetime
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dlib as D

cal = json.loads((HERE / 'calib5m.json').read_text())['frozen_shares']
prm = D.default_prm()
SPEC = {
    'topic': 'o_disposal_d0_exec_close：處置腿 d0（只做上市）可執行重建與結案判定',
    'event_rules': 'o_disposal_official_fullhist/frozen_spec.txt（disposal_v2 V_OFF）逐字；只把 R3、R5 改成成交規則（見 fill_rules）',
    'event_rules_sha256_of_o9_spec': D.sha(HERE / 'raw' / 'o9_frozen_spec.txt'),
    'list': {'file': 'raw/disposal_hist_rejudge.csv', 'sha256': D.sha(HERE / 'raw' / 'disposal_hist_rejudge.csv'), 'universe': '上市普通股（4 碼、不以 0 開頭、名稱不含 DR）'},
    'info_rule': '所有決策只用進場前一日收盤前已知資訊：公告日 ≤ 進場前一日；R4 用 V60p、PC（前一日）；R2 用進場（s）前一個指數交易日；同檔持有中跳過',
    'fill_rules': {
        'buy': '以限價掛單：主欄限價＝漲停價（LIMUP，REF×(1+漲跌幅) 取跳動下緣）；該盤價格（最後一盤＝收盤、第一盤＝開盤）≥ 限價 → 未成交，事件放棄（不遞補）',
        'sell': '以跌停價掛該盤：收盤 ≤ 跌停價（鎖跌停）→ 未成交，順延到下一個未鎖跌停日的最後一盤；下市則以最後有價日收盤出清',
        'sizing': '股數＝floor(預算 ÷ 限價)；整張＝股數//1000，零股＝股數%1000（兩張單）',
        'participation_cap': '整張部分 ≤ cap × Qa，Qa＝當日 V × 該盤占比（處置期內分盤撮合依撮合分鐘；只預收期與處置期外用一般收盤／開盤占比）；src 3 日 V ÷ (1−收盤占比) 還原；買超過 → 縮整張；賣超過 → 剩餘整張順延下一日最後一盤',
        'oddlot': '零股全部成交於該盤價格 ×(1 ± odd_spread)；2020-10-26 前開盤點的零股改用當日盤後零股（收盤價）',
    },
    'cost': {'fee_rate': '0.1425%×0.6（2015-01-01 前）／×0.28（之後）；壓力一律 ×0.6', 'min_fee_per_order': 20, 'tax_sell': 0.003,
             'lot_impact': 'max(一跳/價, 0.1%) ＋ κ·σ60(前一日)·√(q/Qa)', 'kappa': prm['kappa'], 'odd_spread_main': 0.004, 'odd_spread_grid': [0.002, 0.004, 0.008],
             'stress': '每邊再加 0.4%、手續費 6 折，且每邊總成本（賣含稅）不低於 1.0%（來回 ≥ 2%）',
             'etf_0050': '買 0.0855%、賣 0.0855%＋稅 0.1%'},
    'auction_shares': {k: v for k, v in cal.items()}, 'cap': prm['cap'],
    'timings': {'entries': {'O1': 's 第一盤', 'C1': 's 最後一盤', 'O2': 's+1 第一盤'}, 'exits': {'CL1': 'e−1 最後一盤', 'CL': 'e 最後一盤', 'OA': 'e+1 第一盤'},
                'main_column': '9 種時點取中位數，報 p10（np.percentile 線性內插）'},
    'sleeve': {'slots': 10, 'tiers_FIX': [3e5, 1e6, 3e6, 1e7], 'tiers_GROW': [1e6, 3e6, 1e7], 'main': '100 萬 FIX、一般成本、嚴格預收、現金池 k=2',
               'FIX': '每月第一個交易日前把帳戶縮放回 tier×CPI(年)/CPI(2026)（同 o9 修正版）；每槽＝tier×CPI/10', 'GROW': '每槽＝權益/10',
               'prepay_main': 'strict：每筆買進都需進場日已交割現金 ≥ 下單預算；0050 賣出款與處置股出場款 T+2 入帳',
               'prepay_sens': ['threshold（只預收期全收、第二次以上全收、第一次單筆 ≥ 10 張全收）', 'none（o9 口徑：當日賣 0050 支應）'],
               'pool_k_main': 2, 'pool_k_sens': [1, 3], 'min_fill_frac': 0.5,
               'cash_rule': '每日收盤：已交割現金＋應收 < 現金池目標（k 槽）＋已保留待進場金額＋（公告日 ≤ t−1、進場日 ≥ t+2 的已知事件槽數 × 每槽）→ 收盤賣 0050 補足（T+2 入帳）；已交割現金 > 目標＋1 槽 → 多餘買回 0050',
               'cash_skip': '進場日已交割現金 < 0.5 槽 → 跳過（cash_skip）；介於 0.5～1 槽 → 以可用現金下單',
               'order': '同日依 (announce, 檔案順序)；滿槽或同檔持有中 → 跳過', 'idle': '0050 含息', 'start': '2006-04-01'},
    'event_level': {'slot_2026': {'30萬': 3e4, '100萬': 1e5, '300萬': 3e5, '1000萬': 1e6}, 'main_slot': 1e5,
                    'net_x': '可執行報酬 − 同窗流動性合格上市等權（EW）；net50_x 對 0050', 't': '同進場月份平均後的 t（t_month）'},
    'periods': {'SEL': D.PER['SEL'], 'VAL': D.PER['VAL'], 'FULL': D.PER['FULL'], 'NEWREG': D.PER['NEWREG'],
                'segments': [D.PER['P1_2006_2012'], D.PER['P2_2013_2018'], D.PER['P3_2019_2026']],
                'FULL_label': '描述段：已看過 55%（o9 已用 2006-04～2026-10 全部事件）'},
    'decision': {
        'K1': '驗證期主欄（100 萬 FIX、一般成本、嚴格預收 k=2）9 時點對 0050 月超額 t 的中位數 < 1.0 → 結案',
        'K2': '全期 2006-04～2026-10 逐筆可執行淨超額（net_x，每槽 10 萬）9 時點平均值的中位數 ≤ 0，或 p10 ≤ 0 → 結案',
        'K3': '驗證期壓力成本逐筆 net_x 9 時點中位數 ≤ 0 → 結案',
        'usable': 'K1～K3 都不成立，且驗證期每槽 3 萬／10 萬／30 萬（30 萬、100 萬、300 萬級距）逐筆 net_x 9 時點中位數都 > 0 → 可用（只前瞻紙上，真錢 0）',
        'close_action': '結案：SOP 永久刪除處置配置列；前瞻只保留事件時戳存檔；不再開處置變體'},
    'predictions': {
        'P1': '驗證期主欄月 t（9 時點中位數）< 1.0（預期成立）；≥ 1.0 → P1 否證',
        'P2': '相對 o9 V7X 層（同為 C1×CL、同每槽金額），逐筆 net_x 全期平均再少 ≥ 0.5pp；報 3 萬／10 萬／100 萬',
        'P3': '主欄嚴格預收下，cash_skip ÷ (entered＋cash_skip) ≥ 15%（全期，9 時點中位數）',
        'P4': 'C1×CL、每槽 10 萬、全期 net_x 子群差（月群聚 t）> 0：第一次處置、2014 起、10% 漲跌幅期、整張可買（10 萬可買 ≥ 1 張）、late−early>0',
        'P5': '子群差 < 0：第二次以上（second）、20 分鐘撮合（對 5 分鐘）、run6 最高五分位、2006～2013、vdrop ≤ 中位數',
        'P6': '主欄 9 時點中位數月超額序列與主策略可執行（V7X R4 B10 K5T normal 100 萬 FIX，13 個 j 平均）月超額相關 ≥ 0.4；< 0.3 才重新考慮',
        'P7': '新制（進場 ≥ 2026-08-10）逐筆 net_x（C1×CL 與 9 時點中位數）≤ 0（只描述）',
        'mechanism_falsified': '全期逐筆 net_x 9 時點中位數 ≤ 0，或壓力成本下 ≤ 0'},
    'G_tests': {
        'G1': '全期逐筆 net_x 9 時點中位數與 p10 > 0（一般與壓力）；主欄衛星全期月超額 9 時點中位數與 p10 > 0（一般與壓力）',
        'G2': '主欄衛星 9 時點中位數 t：全期 ≥ 3.0 且 2019 起 ≥ 2.0',
        'G3': '焦點＝9 時點日報酬逐日中位數累積的 NAV；在 AB_V4 訊號日格點的窗對 0050 超額；族群＝o_main_spa_breaks F_long（W_long）2,319 條＋o9 46 條＋本題全部衛星變體；SPA_c（平穩 bootstrap L=6、B=10,000、PR 核、種子 20261103）焦點 p < 0.05；DSR（Li-Ji 有效試驗數）≥ 0.95',
        'G4': '三段（2006～2012、2013～2018、2019～2026）主欄月超額均值 > 0；去掉月超額年合計最好 3 年後 t ≥ 2.0；鄰域（限價 漲跌幅−2%／−0.5%、流動性 500／2,000 萬、槽數 5／20）全期月超額均值不翻號（> 0）',
        'G5': '第 0 步閘門通過；下市股價已併入；公告日全部 ≤ 進場前一日',
        'G6': '另一位研究員只看 frozen_spec.txt 與 EXEC_SPEC.json 從頭重寫：全期逐筆 C1×CL net_x 平均差 ≤ 0.1pp、主欄驗證期 t 差 ≤ 0.2（本題不自評，交審查）',
        'G7': 'GROW 100 萬／300 萬／1,000 萬 主欄口徑全期月超額（9 時點中位數）> 0 且 t ≥ 2.0',
        'G8': '前瞻紙上 12 個月（新制滿 12 個月 2027-08）對 0050 超額 > 0 且逐筆可執行淨超額 > 0'},
    'comparisons': ['可執行月超額相關（主欄 vs 主策略 V7X）', '主策略 80＋d0 20 對主策略 100：IR 與配對 t（月再平衡近似）', '0050 80＋d0 20 對 0050 的 t', '同池等權（EW）對 0050'],
    'variants_counted': '9 時點 × {100 萬 FIX 一般／壓力、30 萬／300 萬／1,000 萬 FIX、100／300／1,000 萬 GROW、k1、k3、threshold、none、零股 0.2%／0.8%、κ 0.25、鄰域 6 種}',
}
spec_txt = """處置腿 d0 可執行重建（o_disposal_d0_exec_close）凍結規格
（PREREG 前寫定；複製員只看本檔與 EXEC_SPEC.json 即可重做。時戳與 sha256 見 results.json 的 PREREG。）

一、事件規則：逐字沿用 raw/o9_frozen_spec.txt（o_disposal_official_fullhist，disposal_v2 V_OFF）第一、二節，
   名單 raw/disposal_hist_rejudge.csv（只取上市普通股），價格 o_exec_engine_rewrite_v7x/data/x_cache.npz，
   鎖漲停描述另用共用面板 high。唯二改動：
   R3（s 日收盤漲幅 ≥ 漲跌幅−1% 不買）→ 成交規則：以漲停價掛單於進場盤；該盤價格 ≥ 漲停價視為未成交，放棄該事件。
   R5（e 日跌幅 ≤ −(漲跌幅−1%) 順延）→ 成交規則：以跌停價掛單；收盤 ≤ 跌停價視為未成交，順延到下一個未鎖跌停日最後一盤。
   其餘（R1 鏈、R2 空頭開關、R4 流動性 V60p ≥ 1,000 萬且 PC ≥ 10、同檔持有中跳過、R6 再處置旗標）不變。
   資訊：公告日 ≤ 進場前一日（全部事件 s−公告日＝1 個交易日）。

二、進出場時點（9 種）：進場 O1＝s 第一盤（開盤價）、C1＝s 最後一盤（收盤價）、O2＝s+1 第一盤；
   出場 CL1＝e−1 最後一盤、CL＝e 最後一盤、OA＝e+1 第一盤（開盤 ≤ 跌停 → 改當日收盤並套 R5 成交規則）。
   開盤水準＝收盤水準 − log(C/O)（含息累積對數指數 LA）。主欄＝9 時點的中位數，報 p10。

三、可執行模型
   股數＝floor(預算 ÷ 限價)，整張＝股數//1000、零股＝股數%1000，兩張單各自付手續費 max(20 元, 費率×金額)。
   費率 0.1425%×0.6（2015 前）／×0.28（2015 起）。賣出證交稅 0.3%。
   整張：成本率＝max(一跳/價, 0.1%)＋0.5×σ60(前一日)×√(q/Qa)；Qa＝當日成交值 V×該盤占比；q ≤ 20%×Qa（買超過縮張數；賣超過剩餘順延下一日最後一盤）。
   該盤占比（calib5m.json，Yahoo 5 分 K 與官方日 K 重疊 2025-09-24～2026-03-31 估）：處置期內分盤撮合
   最後一盤 5 分 0.1071、20 分 0.1342、其他（2／10／長間隔）0.1153；第一盤 5 分 0.0301、20 分 0.0802、其他 0.0393；
   一般（處置期外與 2010-12 前只預收期）收盤 0.1296、開盤 0.0437。src 3 日 V ÷ (1−占比) 還原。
   零股：成交價＝該盤價格×(1 ± 0.4%)（網格 0.2%／0.8%）；2020-10-26 前開盤點的零股走當日盤後零股（收盤價）。
   壓力：每邊 +0.4%、手續費一律 6 折，且每邊總成本（賣含稅）≥ 1.0%。

四、衛星組合（10 槽，閒置 0050 含息，起點 2006-04-01）
   主欄：100 萬 FIX（每月初縮放回 100 萬×CPI(年)/CPI(2026)；每槽＝十分之一）、一般成本、嚴格預收、現金池 k=2 槽現金（0 報酬）。
   嚴格預收：進場日的已交割現金 ≥ 預算才全額下單；0.5～1 槽以可用現金下單；< 0.5 槽 → cash_skip。
   0050 賣出款與處置股出場款 T+2 入帳。每日收盤依前一日已知事件補足現金池（賣 0050，T+2）、多餘現金買回 0050。
   決策時點：進場前一日收盤後；同日依 (announce, 檔案順序)；滿槽或同檔持有中跳過。
   敏感度：k=1／3；預收 threshold／none；零股 0.2%／0.8%；κ 0.25；級距 30 萬／300 萬／1,000 萬 FIX、100／300／1,000 萬 GROW；
   鄰域：限價 REF×(1＋漲跌幅−2%)／(−0.5%)、流動性 500／2,000 萬、槽數 5／20。

五、期間：挑選 2021-10-01～2023-12-31；驗證 2024-01-02～2026-10-22；全期 2006-04-01～2026-10-22（描述，已看過 55%）；新制 2026-08-10 起另報。

六、事前門檻（任一成立即結案）
   K1：驗證期主欄 9 時點對 0050 月 t 的中位數 < 1.0。
   K2：全期逐筆 net_x（每槽 10 萬，對 EW）9 時點平均的中位數 ≤ 0，或 p10 ≤ 0。
   K3：驗證期壓力成本逐筆 net_x 9 時點中位數 ≤ 0。
   可用（僅前瞻紙上）：K1～K3 皆不成立，且驗證期每槽 3 萬／10 萬／30 萬逐筆 net_x 9 時點中位數 > 0。
七、預測 P1～P7、G1～G8 定義見 EXEC_SPEC.json（predictions、G_tests）。
"""
(HERE / 'EXEC_SPEC.json').write_text(json.dumps(SPEC, ensure_ascii=False, indent=1))
(HERE / 'frozen_spec.txt').write_text(spec_txt)
files = ['dlib.py', 's4_oos.py', 'EXEC_SPEC.json', 'frozen_spec.txt', 'calib5m.json', 'iterations.json', 'explore.json', 'data_gate.json',
         'raw/disposal_hist_rejudge.csv', 'raw/o9_frozen_spec.txt', 'raw/official_trades_twse.csv', 'raw/yahoo5m_daily_features.parquet',
         str(D.AG / 'o_disposal_official_fullhist' / 'core.py'), str(D.AG / 'o_exec_engine_rewrite_v7x' / 'data' / 'x_cache.npz')]
PRE = {'timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z', 'selected_version': 'E4', 'n_iteration_versions': 4,
       'hashes': {f: D.sha(f if f.startswith('/') else HERE / f) for f in files}, 'spec_summary': SPEC['decision']}
res_f = HERE / 'results.json'
res = json.loads(res_f.read_text()) if res_f.exists() else {}
assert 'PREREG' not in res, 'PREREG 已存在，不可覆寫'
res['PREREG'] = PRE
res['PREREG_sha256'] = D.canon_sha(PRE)
res_f.write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(res['PREREG_sha256'], PRE['timestamp_utc'])
