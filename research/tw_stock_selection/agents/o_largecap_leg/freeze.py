#!/usr/bin/env python3
"""凍結：讀 iterations.json 的機械選版結果，把最終規格與判定程序寫成 results.json 的 PREREG（sha256＋時間戳）。
之後 oos.py 斷言 PREREG 存在且未被改過，才計算 2019 年以後。"""
import json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
RJ = HERE / 'results.json'
assert not RJ.exists(), 'results.json 已存在：只凍結一次'
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
PL = json.loads((HERE / 'plan.json').read_text(encoding='utf-8'))
IT = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
pick = IT['selection']['picked']; spec = IT['versions'][pick]['spec']
PREREG = {
 'frozen_at_rule': '2018-12 底凍結；本檔寫入前沒有任何 2019 年以後的報酬被計算（explore.py、iterate.py 的淨值只模擬到 2019-01-11 收盤）',
 'plan_sha256': PL['PLAN_sha256'], 'iterations_sha256': sha(HERE / 'iterations.json'), 'lcore_sha256': sha(HERE / 'lcore.py'),
 'explore_sha256': sha(HERE / 'explore.json'), 'tw50_members_sha256': sha(HERE / 'tw50_members.json'),
 'selection': {'picked': pick, 'note': IT['selection']['note'], 'eligible': IT['selection']['eligible'], 'n_versions_tested': 12, 'n_controls': 7},
 'frozen_spec': {
  'name': f'L 腿 {pick}', **spec,
  'pool': '訊號日（每月第一個日 ≥ 11 的交易日）上市股中：60 日均成交值（gapfix 窗）≥ 1,000 萬、收盤 ≥ 10、非金融保險（含 28xx、5876、5880）、非 KY／DR、近 3 月營收年增率與 52 週比有值、非處置（2021-09 起名單）、隔日有開盤成交；依 60 日均成交值取前 P 名',
  'score': '池內百分位：0.5×52 週比（還原收盤 ÷ 252 日最高）＋0.5×近 3 月營收年增率（m−1～m−3 平均；2021-03 以前 R_DEL，之後 rev_5y yoy）；同分：營收高者→代碼小者' if spec['score'] == '55' else '池內 52 週比百分位',
  'execution': '隔日開盤成交；開盤漲停且原本未持有者遞補下一名；停牌凍結；每月換股',
  'combo': f"單一帳戶：主策略（MAIN_R，排除金融 n10）10 檔各 {(1-spec['w_leg'])/10:.3f}、L 腿 {spec['n']} 檔各 {spec['w_leg']/spec['n']:.3f}（共同持股相加），每次換股成交日回到目標權重，換股之間不再平衡",
  'costs': {'normal': '買 0.0855%＋滑價 0.1%；賣 0.0855%＋稅 0.3%＋滑價 0.1%', 'stress3': '每邊再加 0.3% 滑價', 'stress_old': '買賣各 0.5%（研究中心舊壓力）'},
 },
 'periods': {'SEL': '訊號窗起點 2006-04～2018-12', 'VAL': '2019-01～2026-09 訊號（日頻到 2026-10-12）', 'VAL_A': '2019-01～2023-12', 'HOLD': '2024-01～2026-10-12', 'ALL': '2006-04～2026-10-12'},
 'controls': ['MAIN', 'TSMC30（＝integrate_o2 TSMC30_R，程式內斷言）', 'COMBO＝MAIN70＋L 腿 30%', 'MAIN70＋0050 30%（袖子層）', 'MAIN70＋池等權 30%（安慰劑）', 'MAIN70＋成交值第 1 名 30%', 'L 腿單獨'],
 'judgement': {
  '(1)': 'SEL：L 腿（一般成本淨值）對同池等權（零成本）訊號窗月超額 t ≥ 1.5；VAL：月均 > 0',
  '(2)': 'COMBO 在 VAL：對 MAIN 的月配對差平均 ≥ −0.1pp；HOLD 年化 ≥ MAIN；VAL_A 與 HOLD 的追蹤誤差（月超額年化標準差）都 ≤ MAIN − 3pp',
  '(3)': 'SEL：COMBO 對 0050 t ≥ TSMC30；2330 漂移歸零反事實（日報酬減去該段平均日超額，SEL＝2006-04-11～2019-01-11、VAL＝之後到 2026-10-12 兩段各自）下，COMBO 對 0050 t 在 SEL 與 VAL 都 > TSMC30；池剔除 2330 後條件 (1) 仍成立（SEL t ≥ 1.5、VAL 月均 > 0）',
  '(4)': 'VAL：L 腿貢獻最大的 5 檔（貢獻＝Σ 權重 ×（個股期報酬 − 池等權期報酬））剔除後（其期報酬換成池等權），L 腿對池等權的期平均超額（毛）減去成本拖累（＝剔除前毛平均 − 淨值訊號窗平均）> 0',
  '(5)': '全部通過 → 待觀察（上限）；P1（(1) 的 SEL 部分）不成立，或池剔除 2330 後 SEL 或 VAL 的 L 腿對池等權月均 ≤ 0 → 不可用（L 腿只能當降低追蹤誤差的配置，和 0050 沒有差別）；(3) 不過 → 維持 TSMC30_R 現狀；其餘未全過 → 不可用',
  'predictions_check': {'P2': 'L 腿對 0050 超額與 MAIN 對 0050 超額月相關 ≤ 0.5（SEL、VAL）；L 腿超額對（2330−0050）β ≥ 0',
                        'P3': 'COMBO TE 比 MAIN 低 ≥ 3pp、月均超額不低於 MAIN −0.15pp；Δt 的 Shapley 拆解中波動占一半以上時明寫「改善的是追蹤誤差，不是報酬」'},
  'multiple_testing': '本題共 12 個模型版本＋7 個對照組＋4 條判定門檻（含 8 個子條件）；版本選擇只用 SEL 一次，判定只用凍結版',
  'cap': '待觀察（2019 年以後不是盲測）',
 },
 'SEL_facts_known_at_freeze': {'COMBO_SEL_t_0050': IT['versions'][pick]['SEL_normal']['t_0050'], 'MAIN_SEL_t_0050': IT['MAIN_SEL']['t_0050'],
                               'TSMC30_SEL_t_0050': IT['controls']['TSMC30']['SEL_normal']['t_0050'], 'L_vs_pool_SEL': IT['versions'][pick]['leg']['L_vs_poolEW_net']},
}
txt = json.dumps(PREREG, ensure_ascii=False, sort_keys=True)
R = {'PREREG': PREREG, 'PREREG_sha256': hashlib.sha256(txt.encode()).hexdigest(), 'PREREG_time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
print(R['PREREG_sha256'], R['PREREG_time_utc'], pick)
