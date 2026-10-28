#!/usr/bin/env python3
"""步驟 3：凍結。把最終規格、對照、判定門檻、驗證與前瞻計畫寫進 results.json 的 PREREG（含 sha256 與時間戳）；
同時寫 live_tracking/o7_regime/registration.json。此時尚未做任何 2023 年以後的計算（W_sel 截到 2023-01-11；s1、s2 的輸入都截在該日）。"""
import sys, json, time, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import cr_lib as L

RES = L.load_json()
assert 'OOS' not in RES, '已有樣本外結果，不得重新登記'
it = json.loads((L.HERE / 'iterations.json').read_text(encoding='utf-8'))
fin = it['final_pick']; assert fin['id'] == 'V5'
full = {o['label']: o for r in it['rounds'] for o in r['full']}
wbar = float(full[fin['id']]['wbar'])
versions = [o['spec'] for r in it['rounds'] for o in r['full']]
LIVE = L.HERE.parent / 'live_tracking' / 'o7_regime'

seq = dict(
    primary='若 o_tier_shrink_sequential 在 2026-11-11（第一個前瞻決策日）之前交出並凍結序貫規則，本題改用該規則（先驗用其事後分布、焦點＝本題輪動對同平均權重靜態混合的月配對差）；否則用下列備援規則，且之後不得再換。',
    fallback=dict(type='mSPRT（常態混合序貫機率比，任意時點有效）', stat='x_k＝輪動月報酬 − 靜態混合月報酬（凍結 w̄，同日調回、扣成本）', sigma=0.04, tau=0.01,
                  upgrade='Λ_n ≥ 20（α＝0.05）且 n ≥ 12 → 前瞻支持：判定升為「待觀察（前瞻）」，再累積到 n ≥ 36 且對 0050 月均超額 > 0 才可討論「可用」',
                  futility='n ≥ 24 時累積平均 ≤ 0 → 不可用（結案）；n＝60 仍未跨上界 → 不可用（無證據）',
                  min_obs=12, max_obs=60, mechanism_forward='每滿 12 個月重估 P1：d 對過去 6／12 個月累積 d 的係數（HAC）；36 個月後若仍 ≤ 0，機制判不支持並結案'),
    note='序貫規則只看 2026-11 以後的前瞻資料；不得引用 2026-10 以前的報酬調整 σ、τ 或門檻。')
reg = dict(source='agents/o_concentration_regime', version_id=fin['id'], spec=fin, live_start='2026-11-01', wbar_frozen=wbar,
           costs_normal=list(L.COST['normal']), sequential_rule=seq,
           rule_text='S2＝前 5 大 60 日均成交值占比（前 300 池）本次決策日減 6 個決策日前；S2 > 0 → 0050 100%；否則主策略 100%；決策日隔一交易日收盤移轉。')
LIVE.mkdir(parents=True, exist_ok=True)
(LIVE / 'registration.json').write_text(json.dumps(reg, ensure_ascii=False, indent=1), encoding='utf-8')

spec = dict(
    topic='權值集中狀態的持續性：0050 對同池等權的相對強弱，在 0050 與主策略之間輪動',
    frozen=dict(main_version=fin['id'], main_spec=fin, wbar_frozen=wbar,
                rule='決策日 T＝主策略官方訊號日（每月第一個日期 ≥ 11 的 GOOD 交易日）。S2(T)＝前 5 大 60 日均成交值 ÷ 前 300 池總和，本次減 6 個決策日前。'
                     'S2 > 0（h＝0）→ 集中期：0050 100%；否則擴散期：主策略（AB_V4 腿）100%。移轉在 T+1（主策略 exec 日）收盤；主腿出清與回場各付完整單邊成本，0050 單邊 0.1% 稅（賣）＋手續費 0.1425%×0.6＋滑價 0.05%。'
                     '壓力：主腿每邊 0.5%（AB_V4|stress 淨值）、0050 每邊滑價加 0.3%。',
                controls=['STATIC_FROZEN：0050 權重＝挑選期平均 w̄（%.4f），每月 exec 日收盤調回' % wbar,
                          'STATIC_REAL：0050 權重＝同一評估期間輪動版的實際平均目標權重（純擇時價值的對照）',
                          'MIX30：0050 30%＋主策略 70%（本題引擎），另列 integrate_o6 的 M_E50_30（AB_V6 基底）',
                          '0050、AB_V4、AB_V4_B10、同池等權（AB_V4|EW）'],
                derived=['CORE50_SAT：0050 50% 核心＋主版本 50% 衛星（w＝0.5＋0.5×w_rot），只報告不判定']),
    versions=versions,
    periods=dict(SEL=L.SEL, VAL=L.VAL, PRE=L.PRE, note='期間以視窗起點（決策日）所在月份歸屬；驗證期最後一窗 2026-09-11→2026-10-12（主策略淨值終點）。'),
    seen_ledger='驗證期不是盲測：integrate_o6 看過 2019～2026 的 0050 混合；o_capacity_tsmc_gap、o_main_spa_breaks 看過 2024～2026 的台積電缺口；timing_switch、breadth_derisk、concentration_equity、w3_size_rotation_28（0050 對中小籃 20 日相對動能週輪動）看過 2006～2026 的擇時。判定上限＝待觀察。',
    verdict_rules=dict(
        watch='待觀察（歷史上限）需全部成立：(a) 挑選期對靜態混合 t ≥ 1.0 且對 0050 t ≥ 1.5；(b) 驗證期對靜態混合月均 > 0（STATIC_REAL 與 STATIC_FROZEN 取較差者）、對 0050 月均 > 0、驗證期轉換 ≥ 2 次；'
              '(c) 壓力成本與 j 中位數下 (b) 仍成立；(d) 驗證期拿掉最好 3 個月後對靜態混合月均 ≥ 0。',
        unusable='驗證期對靜態混合 ≤ 0 或對 0050 ≤ 0 → 不可用。',
        undecidable='任一期轉換 < 2 → 無法判斷。',
        gap='以上皆不符（例如 (a) 不過但驗證期為正）→ 不可用（未達待觀察門檻）；驗證期的正值只當事後觀察。',
        mechanism='P1：挑選期 S1 L6 或 L12 的係數 > 0 且 t_HAC ≥ 1.5（另一個也 > 0）才算支持；結構說法另需 2006～2018 係數 < 挑選期的一半。挑選期係數 ≤ 0 → P1 不支持。'
                  'P5：驗證期對靜態 ≤ 0 或轉換 < 2 → 本題不成立。P1／P5 只寫進機制判定，不改策略判定。',
        note_selection_already_failed='登記時已知：8 版在挑選期對靜態混合全部為負（最佳 V5 IR −0.13、t −0.26），(a) 已不成立；P1 挑選期係數為負。驗證期照登記跑完，判定最高只能是「不可用（未達待觀察門檻）」或「不可用」或「無法判斷」。'),
    validation_plan=['全部 8 版：挑選期重算（必須與 iterations 相同）、驗證期（兩種靜態對照）',
                     '主版本：一般／壓力成本、B10 主腿、V6 主腿、零移轉成本；端點（拿掉最好 1／3 個月、拿掉 2024-01 那一窗）、逐年',
                     '換股日 j＝0～12：主腿 navs_jscan_v6 AB_V6|normal|Jj（與 stress），訊號在 J 日重算，報中位數',
                     '可執行 4 級距 × FIX／GROW（o_exec_capacity_tiers navs_abavg 的 main|級距|GROW/FIX|A）',
                     '0052、006208 取代 0050 持有腿（訊號不變）',
                     '結構前對照 2006～2018：P1 係數與 P5 方向（凍結參數）',
                     'P3 行為、P4 外資（與投信）集中度、P6 平均權重與追蹤誤差、2023～2026 描述延伸、MDE、有效檢定數'],
    live=dict(folder='agents/live_tracking/o7_regime', script='o7_regime_live.py', registration='registration.json', sequential_rule=seq),
    mde_note='配對差月標準差約 3～4%，46 個月的 MDE80 約 1.2～1.7pp／月，檢定力不足。',
)
spec_sha = hashlib.sha256(json.dumps(spec, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
files = [L.HERE / f for f in ('cr_lib.py', 'cr_eval.py', 's0_build.py', 's1_explore.py', 's2_iterate.py', 's3_prereg.py', 's4_oos.py', 'explore.json', 'iterations.json')]
files += sorted(L.RAW.glob('*')) + [L.DATA / 'W_sel.parquet']
files += [L.PANEL / f for f in ('adj.parquet', 'close_raw.parquet', 'value.parquet')]
files += [LIVE / 'o7_regime_live.py', LIVE / 'registration.json']
hashes = {str(f): L.sha(f) for f in files}
RES['PREREG'] = dict(spec, spec_sha256=spec_sha, hashes=hashes, timestamp_utc=pd.Timestamp.utcnow().isoformat(),
                     statement='登記前沒有任何 2023 年以後的報酬、狀態或訊號計算：s0 只建到 2023-01-11 決策日；s1、s2 的主策略淨值、0050、面板全部截在 2023-01-11。')
L.save_json(RES)
print('PREREG', RES['PREREG']['timestamp_utc'], spec_sha, len(hashes), 'files; wbar', wbar)
