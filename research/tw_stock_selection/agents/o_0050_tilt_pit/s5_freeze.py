#!/usr/bin/env python3
"""凍結（PREREG）：寫 data/PREREG.json 與 results.json['PREREG']（規格全文、sha256、時間戳、程式 sha256）。之後 s6/s7/s8 斷言此檔存在且程式未改。"""
import sys, json, hashlib, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tlib as L
assert not (L.DATA / 'PREREG.json').exists(), 'PREREG 已存在，不得覆寫'
IT = json.loads((L.HERE / 'iterations.json').read_text(encoding='utf-8'))
EX = json.loads((L.HERE / 'explore.json').read_text(encoding='utf-8'))
G = json.loads((L.HERE / 'results.json').read_text(encoding='utf-8'))['gate']
spec = {
 'title': 'o_0050_tilt_pit：0050 骨架選股 2.0（PIT 股數、層內權重中性 52 週×營收傾斜）',
 'frozen_versions': {'T1': IT['selected_T1'], 'T2': IT['selected_T2'], 'primary': 'T1'},
 'universe': {'T1': 'e 日開盤適用的臺灣50 成分（2015-09-18 起公開名單 tw50_members.json；2013-12～2015-09 用 PIT 股數依規則重建 membership_pit.json）',
              'T2': 'T1 ∪ 訊號日 d 收盤 PIT 市值前 100 名（非 DR、近 6 日有成交）'},
 'market_cap': 'PIT 股數（data/shares_pit.parquet，s1_shares.py）× 前值補收盤；全市值（不做自由流通調整）',
 'base_B': '同一宇宙 PIT 市值加權（同引擎、同成本）',
 'tiers': '宇宙內市值名次：2330 固定｜2～5｜6～50｜51 以後（T2）；金融保險股不論名次照權重不傾斜',
 'tilt': "w'＝w×exp(λz)，縮放回層原始總權重；z＝層內分數排名標準化 (r−(n+1)/2)/√((n²−1)/12)；分數＝0.5×層內 52 週比百分位＋0.5×層內近 3 月營收年增率（V4 已公告值）百分位；缺值 z＝0；同分比營收再比代碼；凍結 λ＝1.5、無上下限",
 'timing': '訊號日 d：j 平移＝營收期限日 D_m 後第 j 個面板交易日（j＝0..12）；官方日＝每月第一個 ≥ 11 日的交易日（只列參考）。d 收盤算權重、d+1 開盤成交；評估窗 d_t 收盤 → d_{t+1} 收盤；0050 含息（finlab etf_0050.csv）同窗',
 'costs': {'normal': '買 0.0855%＋0.1% 滑價；賣 0.0855%＋0.3% 稅＋0.1%', 'stress': '每邊 0.5%', 'executable': 's7_exec.py：K5T＋零股價差＋最低手續費 20 元＋FIX 縮放成本＋0.25% 不交易帶'},
 'periods': {'SEL': '2014-01～2018-12（原計畫 2006-04 起；2006～2013 不過 PIT 閘門）', 'VAL': '2019-01～2023-12（已被看過）', 'HOLD': '2024-01～2026-10（已被看過）', 'POST2019': 'VAL∪HOLD'},
 'statistics': {
   'main_column': '13 個 j 的中位數（每個 j 先算統計量，再取中位數）；另報 p10、p90 與官方日',
   'placebo': '每月每層把 z 在可傾斜股之間隨機排列（1,000 組，種子 7000+j），零成本毛報酬；序列 x_t＝平均_p(R_p,t)−R_B,t（B 也零成本）；報均值與 t',
   'vs_random': '序列 y_t＝R_T,t（零成本）−平均_p(R_p,t)；報均值與 t',
   'excess': '一般成本淨報酬：T−B、T−0050；t＝月序列均值/標準誤；IR＝均值×12/(sd×√12)',
   'drop_best5': 'T−B 一般成本序列刪除最好 5 個月後的 t',
   'max_year_share': 'T−B 各日曆年（以訊號窗起點年）加總 ÷ 全期加總（全期 ≤ 0 時記 NaN，視為不通過）'},
 'gates': {
   'a': '安慰劑 |t| < 1（SEL 與 POST2019 皆須）',
   'b': 'POST2019：T−B 與 T−0050 的 j 中位數月均 ≥ +0.08pp 且 j 中位數 t ≥ 2.0',
   'c': 'POST2019：vs_random j 中位數 t ≥ 2.0',
   'd': 'POST2019：drop_best5 j 中位數 t ≥ 1.0 且 max_year_share j 中位數 ≤ 40%',
   'e': 'SEL：T−B j 中位數月均 ≥ 0',
   'f': 'POST2019 可執行 K5T：30 萬、100 萬、300 萬、1,000 萬 FIX 與 1,000 萬 GROW 每格 j 中位數對 0050 超額 > 0；2020-10-26 前零股以收盤價成交，標示不可執行並另報 2020-10 後',
   'g': '比較：IR（對 0050）「T1 70%＋主策略 30%」對「0050 70%＋主策略 30%」，及同帳戶配對 t；主策略＝V7 AB（navs_jscan_v7 的 AB_V7|normal|J{j}）',
   'verdict': '(a)～(f) 全部成立 → 待觀察（上限，因 2019～2026 已被看過）；(a) 或 (e) 不成立 → 不可用；(b)～(d) 任一不成立 → 不可用、機制「部分支持」；(f) 不成立 → 不可用（可執行性）',
   'mechanism': '(c) 成立且 (b) 對 B 成立 → 支持；只有 (c) 成立 → 部分支持；(c) 不成立 → 不支持'},
 'predictions_registered': {
   'P1': '安慰劑 |均值| < 0.03pp、|t| < 1', 'P2': 'POST2019 對 B 與 0050 > 0、對隨機 t ≥ 2', 'P3': 'SEL 超額 ≥ 0',
   'P4': '追蹤誤差 2～4%、單邊月周轉 ≤ 15%、成本 ≤ 0.03pp／月；可執行 K5T 1,000 萬 GROW 仍為正',
   'P5': '應更強：營收年增率分散度高（T1 IQR > %.2f）、6～50 名層、電子權重高（> %.4f）；應更弱：2330 相對 0050 大漲月（> %.4f），金融層主動權重恆為 0' % (EX['E6_subsample_thresholds']['rev_yoy_IQR_T1_median'], EX['E6_subsample_thresholds']['elec_weight_T1_median'], EX['E6_subsample_thresholds']['tsmc_minus_0050_top_tercile']),
   'P6': 'T1 主動報酬（對 0050）與主策略對 0050 超額相關 < 0.5；「T1 70%＋主策略 30%」IR 高於「0050 70%＋主策略 30%」',
   'P7': '否證：PIT 股數下對隨機 t 在 POST2019 < 1.5 → o7 的 t 4.02 來自股數前視或存活者偏誤'},
 'selection_period_result_known_at_freeze': {k: IT['versions'][IT[f'selected_{k}']]['jmedian'] for k in ('T1', 'T2')},
 'disclosures': ['選中的 λ＝1.5 無上限位於格點邊緣（贏家詛咒風險）', '挑選期 12 個設定全部 |t| < 0.7；選版幾乎等於挑最大 λ',
                 '挑選期成本 0.10～0.11pp／月、周轉 15～17%，已違反 P4 的成本預測（≤ 0.03pp）', '2019～2026 已被 o7 看過，判定上限待觀察',
                 '金融股 2013～2015Q1 與部分無報表股以 TWT49U 價格比回推（權息事件扣同年息值殖利率中位數），屬近似'],
 'gate_frozen': {'shares_pit_sha256': G['shares_pit_sha256'], 'membership_pit_sha256': G['membership_pit_sha256'], 'frozen_utc': G['frozen_utc']},
}
code = {f: L.sha(L.HERE / f) for f in ['tlib.py', 'tcore.py', 's6_oos.py', 's7_exec.py', 's8_robust.py']}
spec_sha = hashlib.sha256(json.dumps(spec, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
PRE = {'spec': spec, 'spec_sha256': spec_sha, 'code_sha256': code, 'frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
(L.DATA / 'PREREG.json').write_text(json.dumps(PRE, ensure_ascii=False, indent=1), encoding='utf-8')
R = json.loads((L.HERE / 'results.json').read_text(encoding='utf-8')); R['PREREG'] = PRE
(L.HERE / 'results.json').write_text(L.dumps(R), encoding='utf-8')
print(spec_sha, PRE['frozen_utc']); print(json.dumps(code, indent=1))
