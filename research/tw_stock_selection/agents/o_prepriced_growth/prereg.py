#!/usr/bin/env python3
"""凍結：python3 -I -B prereg.py → results.json 的 PREREG（sha256＋時間戳）。

依研究計畫事前寫死的選擇規則，從 iterations.json（只含挑選期）機械式選出單一版本，並寫下：
  * 單一版本與完整規格
  * 單一合成方式：主策略 n10（排除金融）與第二腿 n10 各半資金、每月再平衡（不搜尋權重）
  * 閘門：挑選期對同池等權 t ≥ 1.5 才可以打開驗證期（2014～2020）與保留期（2021～2026-10）；否則判不可用。
已經存在 PREREG 時拒絕覆寫。
"""
import json, hashlib, datetime, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
RES = HERE / 'results.json'
res = json.loads(RES.read_text(encoding='utf-8')) if RES.exists() else {}
if res.get('PREREG'):
    sys.exit('PREREG 已存在，不覆寫')
it = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
assert it['period'][1] <= '2013-12-31'
vers = it['versions']
elig = {v: r for v, r in vers.items() if r['normal']['corr_main_exEW'] is not None and r['normal']['corr_main_exEW'] < 0.6}
best = max(elig, key=lambda v: elig[v]['normal']['t_EW'])
tb = elig[best]['normal']['t_EW']
gate_open = tb >= 1.5
rev, band, n = best.split('|')
SPEC = {
    'topic': 'o_prepriced_growth：搶在價格之前的成長股（第二條腿）',
    'selected_version': best,
    'selection_rule': '挑選期 2006-07～2013-12（模擬截在 2013-12-31）對同池等權 t（訊號窗、一般成本）最高，且與主策略 n10 排除金融的月超額（對同池等權）相關 < 0.6',
    'selection_t_EW': tb, 'selection_corr_main': elig[best]['normal']['corr_main_exEW'],
    'n_versions_tried': it['n_versions'], 'maxt_bootstrap_selection': it['maxt_bootstrap_selection'],
    'rule': {
        'pool': '同主策略 v2：gapfix 滾動窗、前 300 名 60 日均成交值（≥ NT$1,000 萬、收盤 ≥ 10）、排除金融保險業（stock_list 金融保險業＋28xx、5876、5880）、非處置、隔日有開盤成交、52 週比與營收 3 月年增率有值',
        'band': f'52 週比（還原收盤 ÷ 252 日最高）在股池內百分位 ∈ {it["space"]["bands"][band]}（含端點）',
        'revenue_signal': {'ACC': '營收加速度＝yoy3(m)−yoy3(m−3)；yoy3＝最近 3 個月單月年增率平均',
                           'NH': '12 月新高比＝最近一月營收 ÷ 前 11 個月最大值',
                           'AVG': 'ACC 與 NH 在股池內百分位的平均'}[rev],
        'rank': '區間內依營收分數由高到低，同分比營收 3 月年增率、再比代碼',
        'n': int(n[1:]),
        'execution': '每月第一個 ≥ 11 日的交易日收盤算訊號，隔日開盤等權成交；開盤漲停（跳動點）且未持有者遞補；停牌凍結；成本一般＝手續費 0.0855%＋滑價 0.1%（賣再加稅 0.3%），壓力＝買賣各 0.5%',
        'revenue_source': '訊號日 < 2021-03-01 用 tw_revenue.csv；之後年增率用 rev_5y yoy（落後 3 個月那段缺值時用長期營收補），營收水準 rev_5y 優先',
        'revenue_timing': '主結果 BASE（同主策略正式 2.64 的口徑）；AB（只用已公告營收，rev_announce 延後機率 MC）為敏感度'},
    'combination': '主策略 n10（排除金融）與第二腿 n10 各半資金、每月換股日再平衡、重疊持股權重相加、成本只算淨交易；不搜尋權重',
    'note_on_n': '合成方式事前固定為兩條腿各 n10；若選中版本為 n30，第二腿單獨檢定用選中的 n30，合成仍用同訊號同區間的 n10（計畫原文）',
    'periods': {'selection': '2006-01～2013-12（實際 2006-07 起）', 'validation': '2014-01～2020-12', 'holdout': '2021-01～2026-10-12'},
    'gate': {'rule': '挑選期對同池等權 t ≥ 1.5 才打開後兩段；否則判不可用', 'best_t': tb, 'open': gate_open},
    'final_criteria': '(i) 驗證期與保留期第二腿對同池等權扣成本 t 都 ≥ 1.5，合併 2014～2026 t ≥ 2；(ii) 與主策略月超額相關 < 0.5；(iii) 各半合成跨循環 2006～2026 對 0050 t > 主策略單獨（2.64）、年化 ≥ 主策略 −1.5pp、最大回撤不深於主策略；(iv) 合成在 2006～2013、2021～2026 年化 ≥ 主策略 −3pp；(v) 壓力成本下 (iii) 方向不變；(vi) 安聯 2014～2020 對第二腿載荷 > 0',
    'decision_if_gate_closed': '不可用（P5 失敗條件：挑選期對等權 t < 1.5），不打開驗證期與保留期，P3 安聯 2014～2020 不檢定（屬驗證期資料）',
}
body = json.dumps(SPEC, ensure_ascii=False, sort_keys=True)
res['PREREG'] = SPEC
res['PREREG_sha256'] = hashlib.sha256(body.encode('utf-8')).hexdigest()
res['PREREG_time_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
res['iterations_json_sha256'] = hashlib.sha256((HERE / 'iterations.json').read_bytes()).hexdigest()
RES.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print(best, tb, 'gate_open', gate_open, res['PREREG_sha256'], res['PREREG_time_utc'])
