#!/usr/bin/env python3
"""步驟 8：寫 G8_PROTOCOL.json（協定全文、規則參數、OC 表、程式與資料 sha256），並在 results.json 記 G8_FREEZE（檔案 sha256＋UTC）。
--draft：只寫到 scratchpad（測試判定器用），不登記。用法：python3 -I -B s8_protocol.py [--draft]"""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import g8lib as G

DRAFT = '--draft' in sys.argv
PRE = G.check_reg('PREREG'); D = G.check_reg('PREREG_DESIGN')
R = G.load_res(); OC = R['OC_FINAL']; EX = R['EXPLORE']; IT = R['ITERATIONS']
sel = OC['selected']; V = OC['versions'][sel]
par = IT['params'][sel]
DEADLINE = '2026-11-10T05:30:00+00:00'      # 2026-11-10 13:30 台北（J0 收盤）
resid = np.load(G.SCR / 'resid_post2019.npy'); resid24 = np.load(G.SCR / 'resid_2024.npy')


def octab(oc, dgps, alphas=(-1.0, -0.5, 0.0, 0.5, 1.0, 2.0)):
    t = {}
    for g in dgps:
        for a in alphas:
            o = oc[f'{g}|{a:+.1f}']
            t[f'{g}|{a:+.1f}'] = {str(h): dict(up=round(o[str(h)]['up'] if str(h) in o else o[h]['up'], 5), down=round(o[str(h)]['down'] if str(h) in o else o[h]['down'], 5))
                                  for h in (12, 18, 24, 36, 48)}
            t[f'{g}|{a:+.1f}']['literal_g8_pass'] = round(o['literal_g8_pass'], 5)
            cb = o['combined']; t[f'{g}|{a:+.1f}']['combined_24'] = {k: round(v, 5) for k, v in (cb['24'] if '24' in cb else cb[24]).items()}
    return t


proto = dict(
    title='主策略 G8 前瞻序貫檢定協定（o14 凍結版）',
    protocol_version='G8P-1',
    frozen_utc=G.now_utc(), deadline_utc=DEADLINE,
    scope=dict(strategy='主策略：52 週高點百分位 0.5＋近 3 月營收年增率百分位 0.5、前 300 流動性池、月換股 10 檔、排除金融（main_picks_today.py --ex-fin，sha256 0897787f…）',
               benchmark='0050 含息（finlab-txd etf_0050 adj）', data='只用 2026-11-10（J0）收盤之後產生、快照時戳早於成交日 09:00 的前瞻紀錄；歷史資料只用來給設計 σ 與殘差形狀',
               not_changed='本協定不改選股規則、前瞻產生器（o_forward_infra_live）、帳本或成本模型；只規定怎麼讀它們的輸出與怎麼判定'),
    timeline=dict(
        signal_days='不需要假日表：錨定月 m 的 J0＝m 月 10 日起第一個「實際資料中有成交」的交易日；J_j＝J0 後第 j 個交易日（j＝0..12）；官方日＝m 月第一個 ≥ 11 日的交易日；成交＝訊號日隔一交易日開盤。2027 假日表只作交叉核對',
        first_window='2026-11 錨定月：J0＝2026-11-10、官方日＝2026-11-11、官方成交 2026-11-12；第一個窗從 J0 收盤（13:30 台北）起算',
        freeze_rule=f'本檔必須在 {DEADLINE}（2026-11-10 13:30 台北）之前寫好並記 sha256；若晚於此時，計數改從凍結後第一個完整錨定月開始',
        g8_window='2026-11 起前 12 個「有效」錨定月（無作廢時＝2026-11～2027-10；JMED 在 2027-10 錨定月 J12 窗結束、約 2027-11-26 定稿）',
        s1_window='前 24 個有效錨定月（無作廢時＝2026-11～2028-10，約 2028-11-27 定稿）'),
    lines=dict(
        decision='JMED：forward_returns.csv line＝JMED 的定義（13 個 J 窗超額中位數），但本協定改由判定器自 J0..J12 列重算：只取有效 J 窗（作廢規則），有效 J 窗 ≥ 7 才算有效錨定月；每窗超額先扣成本校準調整（見 cost_calibration）。引擎＝V7X K5T GROW 100 萬、一般成本',
        reported=dict(JMED_stress='J0..J12 excess_stress_pp 的中位數（描述）', JMED_cap10='excess_cap10_pp（描述）', j_p10='有效 J 窗超額的 p10（G1 前瞻版描述）',
                      MAIN_OFFICIAL='line＝MAIN_OFFICIAL excess_pp（描述）', E50='line＝E50 ret', C_E50_70='line＝MIX_C70（0050 70%＋MAIN_OFFICIAL 30%）',
                      M_E50_30='0.7×MAIN_OFFICIAL.ret＋0.3×E50.ret − |w′−w|×0.5%（窗初再平衡，同 MIX_C70 算法），離線計算'),
        inventory=EX['inventory']['lines'],
        gaps=['GAP-1 100 萬 FIX：前瞻引擎只跑 GROW 100 萬；以 GROW 100 萬為主欄（歷史 GROW−FIX 挑選期平均 −0.12pp／月、相關 0.998）',
              'GAP-2 300 萬 GROW、GAP-3 1,000 萬 GROW（G7 前瞻版）：forward_returns.csv 沒有欄位；只能離線以 x_exec.py 讀凍結 paper_lists 與 .exec.json 重算，屬描述線，不得作為 G8 判定線；要加進產生器須先經使用者同意'],
        family=dict(decision_lines=['JMED'], rule='S1 翻案只看 JMED。M_E50_30、C_E50_70（MIX_C70）共用同一套窗、作廢與成本規則，但只描述：混合線對 0050 的超額＝權重×主策略超額（再平衡成本除外），z 值與主策略幾乎相同，不是獨立證據',
                    multiplicity='若使用者日後指定 M_E50_30、C_E50_70 也當決策線：事前擇定 Holm（族群 K＝3：JMED、M_E50_30、C_E50_70；24 個月時以 p＝1−Φ(z24)、依序與 0.05/3、0.05/2、0.05 比較）。不用 e 值相乘：三條線高度相依，乘積不是有效 e 值')),
    layer1_literal_g8=dict(rule='依使用者定義原樣判定：前 12 個有效錨定月的 JMED 超額加總 > 0 → 字面 G8 通過；≤ 0 → 未過（降級為前瞻失敗）',
                           warning='σ≈6.14pp／月時 α＝0 也有約 50% 機率通過（OC：D1 0.499），所以字面 G8 只是必要條件，不是 α>0 的證據',
                           o9_relation='沿用 o9 PREREG_FORWARD.G8 的線與窗；o9「≥ 2 個月時戳不合格 → 不判定」照樣另報，與本協定 (c) 不一致時兩者並列'),
    layer2_sequential=dict(H0='α ≤ 0（JMED 對 0050 的月超額期望值）', rule=PRE['upgrade_rule'], down_rule=PRE['down_rule'],
                           statistic='S_n＝前 n 個有效月超額和；z_n＝S_n／(σ_eff·√n)；σ_eff＝c·σ_d（refit 後見下）',
                           upgrade='z_12 ≥ b·√2 或 z_24 ≥ b → 升級（拒絕 H0）；在第 24 個有效月期滿時才生效為「S1 例外 (i) 證據成立」；24 個月後不再判升級',
                           down='第 12、18、24 個有效月（之後每 6 個月）z_n ≤ −c_d → 下修＝前瞻失敗',
                           absorbing='先觸發者吸收；前瞻失敗不得回復；字面 G8 未過即關閉升級路徑'),
    params=dict(sigma_design=G.SIGMA_D, c=par['c'], c_d=par['c_d'], b=par['b'], up_looks=par['up_looks'],
                b_note='單位布朗運動、檢視 {12,24}、OBF 形，穿越機率 0.05；P＝2,000,000、seed 20261105',
                c_d_note='D1（σ_d）、α＝−1、統計量含 c，24 個月內下修機率 0.55；P＝400,000、seed 20261106',
                c_note='c＝sqrt(max(1, s²/σ_d², LRV_L3/σ_d², LRV_L6/σ_d²))，2019-05～2026-09 殘差 → s²/σ_d²＝1.1228 為最大'),
    rule=dict(jmed_min_valid_j=7, refit_ratio=1.5, g8_max_void_in_12=2, g8_window_last_anchor='2027-10', up_looks=par['up_looks'], dn_looks_first=[12, 18, 24]),
    refit='(b) 唯一允許的 refit：檢視點 n ≥ 12 時，前 n 個有效月樣本 sd ＞ 1.5σ_d → 自該月起 σ_eff＝max(c·σ_d, 樣本 sd)，兩側同時套用（下修界限只會放寬、升級只會收緊）；只觸發一次',
    void_rules=dict(rev_coverage_min=0.95,
                    rules=['V0 快照 meta 不存在（上游過期 rc＝2 且 fallback 未在成交日 09:00 前完成，或未凍結）→ 該 J 窗作廢',
                           'V1 meta.ts_ok＝false 或 frozen_ts ≥ exec_open_ts（成交日 09:00 台北）→ 作廢',
                           'V2 meta.rev_gate 不存在或 coverage < 0.95 → 作廢',
                           '有效 J 窗 < 7 → 該錨定月作廢；作廢月不計入 n',
                           '(c) 前 12 個有效月之前累計作廢錨定月 > 2 → 字面 G8「無法判定」，延長到第 24 個有效月以前 24 個有效月加總判一次；累計作廢 > 4 → 永久無法判定（前瞻失敗）'],
                    principle='作廢只能由判定器依時戳與覆蓋率機械判定，不得看報酬'),
    degrade_ladder=dict(states={'FORWARD_VALIDATING': '前瞻驗證中', 'UPGRADE_PENDING_24': '已碰升級界限，待第 24 個有效月生效',
                                'S1_EXCEPTION_EVIDENCE': 'S1 例外 (i) 證據成立（前瞻 24 個月且序貫檢定拒絕虛無）',
                                'FORWARD_VALIDATING_NO_REJECTION_24': '24 個月未拒絕虛無：S1 翻案路徑關閉，繼續紙上追蹤與下修檢視',
                                'FORWARD_FAILED': '前瞻失敗：停止引用主策略任何歷史數字，只保留紀錄'},
                        rule='(a) 字面 G8 未過，或碰到下修界限 → 前瞻失敗（吸收）'),
    cost_calibration=dict(
        data_start='2026-11-12', forbidden='驗證期（≤ 2026-10）任何資料一律不得用於校準（o13 方法教訓 1）',
        sources=['實際成交（trades.csv kind＝real，來自 fills_input.csv）：距開盤跳數、相對開盤價滑價、零股成交價差',
                 '紙上整股成交（kind＝paper_whole）只用市場觀察值：實際開盤成交值占比 share_of_open_vol_actual 對模型預測 share_of_open_vol；紙上成交價本身是模型輸出，不得拿來校準（循環）'],
        min_rows=40, min_months=3, rule_min='合格列 ≥ 40 筆且距 2026-11-12 ≥ 3 個月，取較晚者',
        estimators='中位數與 p75（固定）；Δ_real＝max(0, p75(實際滑價) − 中位數(同日同檔同向紙上模型滑價))；Δ_open＝max(0, p75(κσ(√share_actual − √share_pred)))，κ＝0.5；Δ＝max(Δ_real, Δ_open)',
        kappa=0.5,
        apply='主欄取「事前 V7X K5T」與「校準 p75」較不利者：每窗超額 − 100·Δ·換手（買＋賣成交值／窗初淨值）；只套用在訊號日晚於校準時戳的窗（只往後）',
        frequency='每 12 個月最多重校一次；每次校準結果另存 JSON 並記 sha256 與 UTC（g8_evaluate.py calibrate），之後判定以 --calib 帶入',
        real_money_note='若沒有真錢成交，Δ_real 無法估計；主欄仍不得比事前 K5T 有利'),
    judgement_language=dict(
        states={'FORWARD_VALIDATING': '前瞻驗證中', 'UPGRADE_PENDING_24': '前瞻驗證中（已碰升級界限，24 個月期滿才生效）',
                'S1_EXCEPTION_EVIDENCE': 'S1 例外 (i) 證據成立：可重新評估主策略；仍須 G1～G7 歷史門檻全過才可寫「可投資」',
                'FORWARD_VALIDATING_NO_REJECTION_24': '24 個月前瞻未拒絕虛無：S1 翻案路徑關閉，不可投資',
                'FORWARD_FAILED': '前瞻失敗：不可投資；停止引用歷史數字'},
        historical_gate='(e) G1～G7 歷史部分不過時，即使字面 G8 過了，也只能寫「前瞻 12 個月超額 > 0，但歷史門檻未過，不可投資」；G1～G7 全過且字面 G8 過，才可寫「可投資」；只有字面 G8 過不得寫 α>0'),
    oc=dict(selected=sel, P=OC['P'], seed_base=OC['seed_base'], horizon_sim=OC['H'],
            criteria=D['pass_criteria'], pass_final=OC['selected_pass_final'],
            summary_primary=V['primary'], summary_sensitivity=V['sensitivity'], summary_stress=V['stress'],
            table_primary=octab(V['oc_primary'], ['D1', 'D2', 'D3']),
            other_versions={v: dict(rule=OC['versions'][v]['rule'], pass_=OC['versions'][v]['pass_primary_within_2se'],
                                    primary={g: {k: round(x, 4) for k, x in d.items() if k in ('false_up', 'down_power', 'false_down', 'up_p1', 'up_p2')} for g, d in OC['versions'][v]['primary'].items()})
                            for v in OC['versions']},
            predictions=OC['predictions'], n_versions_tried=4,
            dgp=dict(D1='iid N(α, σ_d²)', D2='α＋σ_d·t4/√2', D3='定態區塊 bootstrap（幾何區塊平均 6、環狀）2019-05～2026-09 去均值殘差，不縮放',
                     seeds='每格 seed＝[seed_base, round((α+3)×100), DGP 序號 D1＝1,D2＝2,D3＝3,D3_2024＝4,D3_L3＝5,D3_L12＝6,D4_o8＝7]；敏感度 seed_base＋1、壓力 seed_base＋2')),
    replication=dict(how='只看本檔即可重做：以 params 的 σ_d、c、b、c_d 與 rule，對 D1～D3（D3 用 data.resid_post2019）各 α 模擬 ≥ 20,000 條 24～48 個月路徑，套 layer2 與 refit，結果應在 2 倍 MC 標準誤內'),
    data=dict(resid_post2019=[round(float(x), 10) for x in resid], resid_post2019_sha256=G.canon_sha([round(float(x), 10) for x in resid]),
              resid_2024=[round(float(x), 10) for x in resid24], resid_2024_sha256=G.canon_sha([round(float(x), 10) for x in resid24]),
              resid_source='o_forward_sequential_v2/fslib.py cell_jmatrix(100萬, FIX, normal) 的 JMED，2019-05 起去均值（V7 navs_exec7 sha256 見 raw/manifest.json）',
              mixture_prior_sha256=D['mixture_prior']['source_sha256']),
    code_sha256={f: G.sha(G.HERE / f) for f in ('g8lib.py', 'g8_evaluate.py', 's1_explore.py', 's2_prereg_design.py', 's3_iterate.py', 's4_freeze.py', 's5_oc_final.py', 's8_protocol.py')},
    registrations={k: R[k + '_sha256'] for k in ('PREREG_DESIGN', 'PREREG')},
    forward_files_at_freeze={f: G.sha(G.AG / 'live_tracking/main_forward' / f) for f in ('forward_returns.csv', 'trades.csv', 'fills_input.csv', 'runs.jsonl')},
)
if DRAFT:
    p = G.SCR / 'G8_PROTOCOL_draft.json'
    p.write_text(json.dumps(proto, ensure_ascii=False, indent=1, default=G._js), encoding='utf-8')
    print('draft', p)
else:
    p = G.HERE / 'G8_PROTOCOL.json'
    assert not p.exists(), 'G8_PROTOCOL.json 已存在：不得覆寫'
    p.write_text(json.dumps(proto, ensure_ascii=False, indent=1, default=G._js), encoding='utf-8')
    fz = dict(file='G8_PROTOCOL.json', sha256=G.sha(p), utc=G.now_utc(), deadline_utc=DEADLINE)
    fz['before_deadline'] = fz['utc'] < DEADLINE
    G.save_key('G8_FREEZE', fz, need=['OC_FINAL'], overwrite=False)
    print(fz)
