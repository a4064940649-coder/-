#!/usr/bin/env python3
"""步驟 8：寫 G8_PROTOCOL_v2a.json 與 K_ALIGNMENT.json（判定器程式定稿後執行；協定記錄判定器 sha）。
凍結（GENESIS、FREEZE_MANIFEST_v2a、錨定行）另由 s9_freeze.py 在判定器測試全過後執行。
用法：python3 -I -B s8_protocol.py"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g8alib as L

R = L.load_res()
PRg = L.check_reg('PREREG', files=False)
assert 'OC_FINAL' in R and 'HIST_DEMO' in R and not (HERE / 'FREEZE_MANIFEST_v2a.json').exists()
OC = json.loads((HERE / 'oc_final.json').read_text(encoding='utf-8'))
assert L.sha(HERE / 'oc_final.json') == R['OC_FINAL']['sha256']
S = OC['summary']; sel = S['selected']; assert sel == 'A1'
P2 = L.v2_protocol()
q = S['qualification'][sel]
fm = {k: v[sel]['up'] for k, v in OC['main'].items()}
v2m = {k: v['v2']['up'] for k, v in OC['main'].items()}
d5 = json.loads((HERE / 'd5_resid_sel.json').read_text(encoding='utf-8'))
H = R['HIST_DEMO']; dn = S['down'][sel]; pw = S['power'][sel]

unc = [t if 'stress_odd_extra' not in t else
       'stress_odd_extra（壓力情境零股加碼 0.4%）：前瞻產生器 o_forward_infra_live/xlive.py CFG_V7X 與 V7X 正式 R4（PREREG orig_cfg）實際為 True；'
       'G8P-2 協定原文「預設關閉」有誤（只對 x_exec.py 未給設定時的程式預設成立）'
       for t in P2['cost_calibration']['uncalibratable_without_real_fills']]

K = [
 f"K1 嚴格口徑（真實成本比壓力成本再多 s，s 直接加在判定線上）不在合格條件內：A1 最高 {S['strict'][sel]['max']:.4f}（{S['strict'][sel]['cell']}），{S['strict'][sel]['n_gt_05']} 格 > 0.05（495 格）；主口徑只保護到「真實成本 ≤ 壓力成本＋0.253pp／月」。",
 "K2 σ > 10.4 不在 OC 格內：升級觸發時 σ_checked（樣本 sd、滾動 12／24 個月 sd 的最大值）> 10.4 → OUT_OF_GRID，升級證據附缺陷（A1 不擋升級，只標記）。",
 f"K3 D3（2019 年起殘差區塊 bootstrap）偏樂觀：保護來自 lag 6 負自相關（−0.28，24 個月和變異比 0.40～0.45）與 2021-05（+26.1pp）離群月；A1 的 D3 最差格 {q['max_by_dgp']['D3']:.4f}，挑選期 D5 最差格 {q['max_by_dgp']['D5']:.4f}。G8P-2 口徑對照：去掉 2021-05 的 D3x，V1 最差 0.0120；G8P-1 原樣 D3x 45 格最高 0.1032（σ 7.0、s 0.78），s＝0 時最高 0.0318（σ 7.5）。",
 f"K4 檢定力極低：α_net＝+1pp／月、σ 6.51、真實成本＝壓力成本時，24 個月升級 D1 {S['power_D1_651_DW_a1'][sel]:.4f}（v2 同路徑 {S['power_D1_651_DW_a1']['v2']:.4f}）；11 種 DGP 最低 {q['power_min']:.4f}（{q['power_min_dgp']}）、最高 {max(q['power_a1_s651_sDW'].values()):.4f}；α＝+1 全 σ×DGP {pw['a1.0|DW'][0]:.4f}～{pw['a1.0|DW'][1]:.4f}，α＝+2 最高 {pw['a2.0|DW'][1]:.4f}。S1 翻案路徑實務上不可行，前瞻主要只能做否定。",
 f"K5 下修 c_d 在 s＝0 校準：σ_d、s＝0、α＝−1 的下修 {min(dn['dn_m1_s0_sigd'].values()):.3f}～{max(dn['dn_m1_s0_sigd'].values()):.3f}；真實成本＝壓力成本（s＝Δ_w）時 α＝−1 下修只有 {dn['dn_m1_sDW_range'][0]:.3f}～{dn['dn_m1_sDW_range'][1]:.3f}（9σ×11 DGP）。",
 f"K6 α_net＝0 的誤下修：σ_d、s＝0 時 D1 {dn['fdn_0_s0_sigd']['D1']:.3f}，MA θ 0.2／0.3 為 {dn['fdn_0_s0_sigd']['MA2']:.3f}／{dn['fdn_0_s0_sigd']['MA3']:.3f}（> 0.30）；全網格（9σ×11 DGP×s∈{{0, Δ_w}}）{dn['n_fdn0_gt_030']} 格 > 0.30，最高 {dn['fdn_0_max']['max']:.3f}（{dn['fdn_0_max']['cell']}）；D5 上 {dn['n_fdn0_gt_030_D5']} 格 > 0.30。",
 "K7 Δ_w＝0.527（V7X GROW100 挑選期；V7 FIX100 為 0.578）；V7X 只有 91.0% 的 J 窗是壓力線較差。",
 "K8 J0 窗：官方月營收抓取修好以前，J0 的 rev_gate 為 J0_UNVERIFIED、沒有 coverage → V2 機械作廢；判定線實際是 J1～J12 的中位數。",
 "K9 沒有真錢成交時，cost_calibration.uncalibratable_without_real_fills 的參數永遠校準不到；S1 證據沒有 REAL_CALIB → NO_REAL_CALIB、不得引用（v2a 新增）。",
 "K10 S1 證據只適用 100 萬 GROW；100 萬 FIX、300 萬、1,000 萬、30 萬都不涵蓋。",
 "K11 外部錨定要研究中心在期限前 commit＋push FREEZE_MANIFEST_v2a.json 並把錨定行推上 origin 的 BACKLOG.md；判定器以 GitHub 伺服器端 push 時間判定，晚於 2026-11-10T05:30Z → v2a 自 push 後第一個完整錨定月起生效，之前照 G8P-2。",
 "K12 b_a 是蒙地卡羅校準（集 A，P＝200,000／格，495 格最大 q_0.9525）；G6 獨立複製差距 > 2 倍 MC 標準誤 → 缺陷；任何格合格判定翻轉 → 一票否決。",
 f"K13 主口徑邊際薄：A1 最差格 {q['max_false_up']:.5f}（{q['worst_cell']}），距 0.05 約 2.1 SE（集 B），未觸發集 C；> 0.045 的格 {S['selected_detail']['n_gt_045']} 個，全在 MA θ 0.3、s 0.78。",
 "K14 V3（open 逾期）仍是報酬可挑選的作廢管道（前 12 個有效月內最多 2 個）：V3 計作廢且不得插回；任何 V3 存在時 S1 證據加旗標 V3_PRESENT、不得引用；但字面 G8 本身仍可能因 V3 剔除較差月份而受益。",
 "K15 校準登錄簿的 input_trades_sha256 只能證明登錄當時的帳本內容；帳本事後改寫由 status 雜湊鏈、月份記錄與鎖定偵測（RETRO_CHANGE 取較不利），但兩次正式呼叫之間的改寫只在下一次呼叫時才看得到。",
 "K16 遠端錨定依賴 GitHub activity API（push 事件保留期未公開）；查不到或無法連線 → REMOTE_UNVERIFIED、NON_OFFICIAL。API 回應存 raw/remote_anchor/。",
 "K17 open 月份的見證要靠研究中心定期（至少每月一次）正式呼叫並把 status_chain.jsonl 推上 origin；漏做會讓已完成的月份在期限後變成 V3（偏嚴，不會偏寬）。",
 f"K18 in-sample 規則示範（不是證據）：歷史 V7X 較差成本 JMED 2006-04～2026-09，223 個 24 個月起點，v2a（A1）升級 {H['v2a']['S1']:.4f}、前瞻失敗 {H['v2a']['FAILED']:.4f}（與 v2 相同）。",
 f"K19 版本選擇：A1～A3 全部合格；最低檢定力 A3 {S['qualification']['A3']['power_min']:.5f} 高於 A1 {q['power_min']:.5f}，但差距 {S['selection_trail'][0]['gap']:.5f} < 2√2·SE {S['selection_trail'][0]['thr']:.5f}，依事前規則取較簡單的 A1；版本數 4（另有 A1b0 診斷欄）。",
 f"K20 DGP 選擇：AR 用 t5 創新、MA 用常態創新；常態創新 AR（敏感度）A1 最高 {S['sens'][sel]['ARN10']:.4f}，t5 創新 MA θ 0.3 最高 {S['sens'][sel]['MAT3']:.4f}，都 ≤ 0.05。網格外的更強相依（ρ > 0.1、θ > 0.3）不在 OC 內。",
]

PA = dict(
 title='主策略 G8 前瞻序貫檢定協定 v2a（G8P-2a，只能更嚴的補遺，o16）',
 protocol_version='G8P-2a',
 frozen_utc=L.now_utc(),
 deadline_utc='2026-11-10T05:30:00+00:00',
 relation=dict(
  addendum_to=dict(protocol='G8P-2', sha256=L.V2_PROTO_SHA, manifest_sha256=L.V2_MANIFEST_SHA, evaluator_sha256=L.V2_EVAL_SHA, path='o_g8p2_protocol_refreeze/G8_PROTOCOL_v2.json'),
  g8p1=dict(sha256=L.G8P1_SHA, path='o_g8_protocol_fin_erratum/G8_PROTOCOL.json'),
  rule='G8P-1、G8P-2 保留不刪。G8P-2a 是 G8P-2 的補遺：判定＝G8P-2 判定器原樣（C1）、G8P-2 規則套在 v2a 序列（C2）、新規則（C3）三者的較不利者。'
       '生效錨定月：FREEZE_MANIFEST_v2a.json 首次 commit 與 BACKLOG 錨定行首次 commit 的 origin（GitHub 伺服器端）push 時間取較晚者；早於 2026-11-10T05:30Z → 2026-11；'
       '否則＝push 後第一個完整錨定月（push 早於該月 10 日 13:30 台北者），之前的錨定月（含檢視點）照 G8P-2 判定。判定器自動計算，不得手動指定。',
  only_stricter='構造保證：v2a 升級 ⇒ G8P-2 升級；G8P-2 失敗 ⇒ v2a 失敗；字面 G8 不變。CRN 下 OC 每格誤升級 ≤ G8P-2 同格（逐路徑，viol 0）。'),
 scope=P2['scope'],
 timeline=P2['timeline'],
 decision_line=dict(P2['decision_line'], adj_v2a='adj_w＝100·max(全部適用 REAL_CALIB 的 Δ)·換手_w（修正 G8P-2 迴圈中最後一筆覆寫）；回溯與只往後兩種算法取較不利者。',
                    stress_odd_extra_correction=unc[-1] if 'stress_odd_extra' in unc[-1] else None),
 void_rules=dict(P2['void_rules'], V3='open 月份時限：錨定月 m 的期限＝(m+3) 月 26 日 00:00 台北（最後一窗最早可能結束日 (m+1) 月 26 日後 2 個月）。期限前沒有任何「已推上 origin 的正式鏈條目（utc 與 push 時間都 ≤ 期限）」記錄該月完成 → V3 作廢，計入 g8_max_void_in_12 與累計作廢，之後補完也不得插回。帳本整月缺列也視為 open。期限內 open 的月份暫不計入，且新規則（C2、C3）在它之後的檢視延後到它補完或轉 V3。'),
 rule=P2['rule'],
 layer1_literal_g8=P2['layer1_literal_g8'],
 layer2_sequential=dict(
  version='A1',
  combination='狀態＝min_ORDER(C1 回溯, C1 只往後, C2 回溯, C2 只往後, C3 回溯, C3 只往後)；ORDER：FORWARD_FAILED < FORWARD_VALIDATING_NO_REJECTION_24 < FORWARD_VALIDATING < UPGRADE_PENDING_24 < S1_EXCEPTION_EVIDENCE。',
  C1='G8P-2 判定器 g8_evaluate_v2.py 的 month_rows／sequential 原樣（sha 斷言），生效錨定月依 G8P-2 manifest 的 origin push 時間計',
  C2='G8P-2 規則（z_n＝S_n／(max(c·σ_d, s_n)·√n)、b＝1.93376、c_d）套在 v2a 序列',
  C3='新規則 A1：z^A_n＝S_n／(c·max(σ_d, s_n)·√n)；升級 n＝12：z^A ≥ b_a·√2；n＝24：z^A ≥ b_a；下修 n＝12、18、24（之後每 6 個月）z^A ≤ −c_d；順序同 G8P-2（12：字面 G8 → 下修 → 升級；18：下修；24：下修 → 升級；先觸發者吸收）',
  s1='第 24 個有效月期滿且三元件都達 S1 才寫「S1 證據成立」；另須附 REAL_CALIB entry_sha（否則 NO_REAL_CALIB）且序列內沒有 V3（否則 V3_PRESENT），任一旗標 → 不得引用。',
  trigger_report='升級觸發時輸出 σ_checked、所在 σ 格、該格 v2a OC 最大誤升級；> 10.4 → OUT_OF_GRID'),
 params=dict(selected=sel, sigma_design=L.SIGMA_D, c=L.C_INFL, c_d=L.CD_V2, b_v2=L.B_V2, b_a=PRg['params']['b_a'][sel], delta_w=L.DW,
             g8_max_void_in_12=2, jmed_min_valid_j=7, sig_max=L.SIG_MAX, b_a_all_versions=PRg['params']['b_a'],
             b_a_rule='b_a＝max(1.9337576513（v2 b）, max over 495 格 q_{0.9525}(B*_comb))；B*_comb＝v2 升級的路徑上取 A1「剛好升級所需的 b」＝max(z^A_12/√2, z^A_24（18 月 A1 未下修時）)，字面 G8 未過或 12 月 A1 下修 → −∞；v2 未升級 → −∞。集 A：cell_rng(2026110701, DGP_ID, σ 序, s 序, α 序＝2, mode 0)，每格 P＝200,000，numpy quantile linear。',
             c_d_note='c_d 不重校（G8P-2 值）；新規則下修失敗與 G8P-2 失敗取聯集',
             calibration_driving_cells=PRg['calibration_summary'][sel]['driving_cells']),
 data=dict(resid_post2019=P2['data']['resid_post2019'], resid_post2019_sha256=P2['data']['resid_post2019_sha256'],
           resid_sel_D5=d5['z'], resid_sel_D5_desc=d5['desc'], resid_sel_D5_file_sha256=L.sha(HERE / 'd5_resid_sel.json'),
           acf_note='挑選期 acf1 0.099、acf6 +0.163、24 個月和變異比 1.22（截斷 11）；2019 年起 acf6 −0.28、變異比 0.40～0.45（explore E1）'),
 oc=dict(
  sigma_grid=L.SIG_GRID, sigma_label=L.SIG_LABEL, s_grid=L.S_GRID, mapping_main='μ＝α + s − Δ_w', mapping_strict='μ＝α + s',
  dgps=PRg['versions'] and json.loads(json.dumps(L.check_reg('PREREG_DESIGN', files=False)['oc_grid']['dgps'], ensure_ascii=False)),
  dgp_id=L.DGP_ID, seeds=dict(A=2026110701, B=2026110702, C=2026110703, rule='cell_rng(base, DGP_ID, σ 序, s 序（Δ_w＝5）, α 序（−1,−0.5,0,+0.5,+1,+2＝0..5）, mode)；SeedSequence 依序；同一格各版本與 v2 共用路徑'),
  P=dict(A=200_000, B=100_000, C=400_000),
  generator_details='D3／D5：殘差去均值、除以 sd（ddof＝1），環狀區塊 bootstrap：首期均勻抽起點，之後每期以 1/6 機率重抽、否則接續下一期（環狀）。AR：e_0＝u_0、e_t＝ρe_{t−1}＋√(1−ρ²)u_t，u＝t5/√(5/3)。MA：e_t＝(u_t＋θu_{t−1})/√(1＋θ²)，u 常態。D2：t4/√2。',
  criteria=dict(false_up='主口徑 495 格，24 個月內誤升級（S1 證據，含字面 G8 閘）≤ 0.05（集 B；|p−0.05| < 2SE 的格用集 C）', pathwise='全部 OC 格 viol_up＝viol_fail＝0',
                down_power='照實報，不設門檻'),
  qualification={v: {k: S['qualification'][v][k] for k in ('max_false_up', 'worst_cell', 'n_gt_05', 'viol_total', 'qualified', 'max_by_dgp', 'power_min', 'power_min_dgp')}
                 for v in ('A1', 'A2', 'A3', 'A4')},
  selected=sel, selection_trail=S['selection_trail'], rerun_C=S['cells_rerun_C'],
  false_up_main={k: round(v, 5) for k, v in fm.items()},
  false_up_max_by_sigma=S['selected_detail']['max_by_sigma'], false_up_max_by_dgp=S['selected_detail']['max_by_dgp'],
  v2_same_paths=dict(main_max=S['v2_main']['max'], worst=S['v2_main']['worst'], n_gt_05=S['v2_main']['n_gt_05'], max_by_dgp=S['v2_main']['max_by_dgp'],
                     primary135_max=S['v2_main']['primary135_max'], false_up_main={k: round(v, 5) for k, v in v2m.items()}),
  diag_A1b0=S['A1b0_main'], strict=S['strict'], sens=S['sens'], down=S['down'][sel], power=dict(selected=S['power'][sel], v2_same_paths=S['power_v2_same_paths'],
  D1_651_DW_a1=S['power_D1_651_DW_a1'], a1_s651_sDW_by_dgp=q['power_a1_s651_sDW']),
  oc_final_json_sha256=R['OC_FINAL']['sha256']),
 governance=dict(L.check_reg('PREREG_DESIGN', files=False)['governance'],
  ii_b_v3_present='（PREREG 後追加，只影響 S1 可引用性，更嚴）序列內任何 V3 → S1 證據加旗標 V3_PRESENT、不得引用。'),
 cost_calibration=dict(P2['cost_calibration'], uncalibratable_without_real_fills=unc,
  note_v2a='校準登錄沿用 G8P-2 的 calib_registry.jsonl 與 g8_evaluate_v2.py calibrate --official；v2a 判定器另加 (vi)：條目 utc 晚於 sched_date＋30 天、或同一排定日重複 → INVALID；adj 取最大 Δ。'),
 evaluator=dict(file='g8_evaluate_v2a.py', official_status='python3 -I -B g8_evaluate_v2a.py status --official',
  official_calibrate='python3 -I -B ../o_g8p2_protocol_refreeze/g8_evaluate_v2.py calibrate --official（沿用 G8P-2）',
  hardcoded=dict(protocol='o_g8p2a_protocol_addendum/G8_PROTOCOL_v2a.json', manifest='o_g8p2a_protocol_addendum/FREEZE_MANIFEST_v2a.json',
                 chain='o_g8p2a_protocol_addendum/status_chain.jsonl', v2=['o_g8p2_protocol_refreeze/G8_PROTOCOL_v2.json', 'FREEZE_MANIFEST_v2.json', 'g8_evaluate_v2.py', 'calib_registry.jsonl'],
                 ledger='live_tracking/main_forward', backlog='agents/BACKLOG.md', repo='github.com/a4064940649-coder/-（origin）', nominal_first_anchor='2026-11',
                 output='o_g8p2a_protocol_addendum/status_runs/status_<UTC>_{OFFICIAL|INVALID|NON_OFFICIAL}.json', raw='o_g8p2a_protocol_addendum/raw/remote_anchor/'),
  integrity='INVALID：協定／判定器 sha 與 FREEZE_MANIFEST_v2a 不符；G8P-2 協定／清單／判定器 sha 不符；登錄簿鏈斷、漏登、utc 逾 30 天、重複排定日；鏈斷、seq 不連續、utc 倒退、GENESIS 不符、本機鏈不是遠端鏈的延伸；manifest 不在 origin、內容不符、push 查不到；BACKLOG 錨定行不在 origin 分支頭；G8P-2 manifest 的 origin push 查不到。',
  non_official='任何其他引數 → NON_OFFICIAL（不得引用、不寫鏈）；遠端無法連線或查詢失敗 → REMOTE_UNVERIFIED → NON_OFFICIAL。',
  chain_write='只有 label＝OFFICIAL 的呼叫追加 status_chain.jsonl；研究中心每次正式呼叫後要 commit＋push 該檔（見證 open 月份的必要條件）。',
  output_fields='OFFICIAL、label、argv、旗標、協定／清單／判定器、G8P-2 三檔、登錄簿、鏈、帳本三檔 sha256、遠端錨定（heads、首次 commit、local %cI 輔助、push 時間、BACKLOG 遠端、API 回應檔 sha）、生效錨定月（v2a、v2）、problems、flags（RETRO_CHANGE、NO_REAL_CALIB、V3_PRESENT、ANCHOR_AFTER_DEADLINE、REMOTE_UNVERIFIED）、狀態（C1／C2／C3 × 回溯／只往後與合併）、字面 G8、逐月列'),
 family=P2['family'],
 judgement_language=dict(P2['judgement_language'], not_citable='S1 例外 (i) 證據在統計上成立，但不得引用', retro='RETRO_CHANGE：已判定的狀態或已記錄月份事後改變，狀態取較不利者'),
 known_defects=K,
 replication=dict(how='只看本檔即可重做：以 params 的 σ_d、c、c_d、b_v2、Δ_w 與 layer2_sequential 的 A1 定義，oc.dgps／generator_details／data（resid_post2019、resid_sel_D5）在 9 σ × 5 s × 11 DGP 格各模擬 ≥ 100,000 條 24 個月路徑，'
                       '同一路徑算 v2 與 A1；依 params.b_a_rule 重算 b_a，並重算 oc.false_up_main、down、power；結果應在 2 倍 MC 標準誤內；任何格合格判定翻轉 → 一票否決。',
                  v2_rule_ref='v2 的升級／下修語意見 G8P-2 協定 layer2_sequential'),
 hist_demo_in_sample=H,
 registrations=dict(PREREG_DESIGN=R['PREREG_DESIGN_sha256'], PREREG_DESIGN_utc=R['PREREG_DESIGN_utc'], PREREG=R['PREREG_sha256'], PREREG_utc=R['PREREG_utc']),
 code_sha256={f: L.sha(HERE / f) for f in ('g8_evaluate_v2a.py', 'g8alib.py', 'refimpl.py', 'synth.py', 's1_explore.py', 's2_prereg_design.py', 's3_iterate.py', 's4_prereg.py',
                                           's5_oc_final.py', 's6_hist_demo.py', 's8_protocol.py', 'ro_snapshot.py')},
 smoke_manifest_sha256=L.sha(HERE / 'smoke' / 'SMOKE_MANIFEST_v2a.json'),
)
(HERE / 'G8_PROTOCOL_v2a.json').write_text(json.dumps(PA, ensure_ascii=False, indent=1), encoding='utf-8')

# ---------------------------------------------------------------- K_ALIGNMENT
v2k = P2['known_defects']
KA = dict(
 canonical='G8_PROTOCOL_v2a.json known_defects（K1～K20）是唯一正本',
 v2_protocol_to_v2a={f'K{i+1}': dict(v2_text=t, v2a=f'K{[1,2,3,4,5,6,7,8,9,10,11,12,18][i]}') for i, t in enumerate(v2k)},
 v2_findings_to_v2a={
  'K1': dict(v2_findings='嚴格成本口徑最高 8.5%', v2a='K1', note='一致（v2 V1 0.0852）'),
  'K2': dict(v2_findings='σ > 10.4 判 OUT_OF_GRID', v2a='K2', note='一致'),
  'K3': dict(v2_findings='D3 的保護部分來自離群值（正文 D3x：G8P-1 原樣在 s＝0 最高 3.2%）', v2a='K3',
             correction='口徑不同而非矛盾：FINDINGS 3.2% 是 G8P-1 原樣 D3x 在 s＝0 的最高（σ 7.5，0.0318）；協定 10.32% 是 G8P-1 原樣 D3x 45 格最高（σ 7.0、s 0.78，0.1032）；V1 D3x 最差 0.0120。v2a K3 兩個口徑並列。'),
  'K4': dict(v2_findings='檢定力極低', v2a='K4', note='v2a 以新數字更新'),
  'K5': dict(v2_findings='真實成本＝壓力成本時下修 0.38～0.43（協定 K5 寫 0.379～0.411）', v2a='K5',
             correction='依 G8P-2 oc_final.json down 表重算：s＝Δ_w、α＝−1、V1 在 9σ×D1～D3 的下修為 0.343～0.431（D1 σ 10.4 0.343；D3 σ 7.5 0.431）；FINDINGS 與協定兩個下限都錯，統計審查的 0.343～0.431 正確。'),
  'K6': dict(v2_findings='D3 在 σ ≥ 7 時 α＝0 誤下修 0.31～0.33（協定 K6 寫 σ ≥ 7.5 時 0.332）', v2a='K6',
             correction='依 G8P-2 oc_final.json：D3、s＝0、α＝0 的 V1 誤下修 σ 7.0 為 0.308、σ 7.5～10.4 為 0.319～0.332；> 0.30 從 σ 7.0 開始，FINDINGS 的 σ ≥ 7 正確、協定的 σ ≥ 7.5 偏窄。'),
  'K7': dict(v2_findings='Δ_w 取兩條線中較小的 0.527', v2a='K7', note='一致'),
  'K8': dict(v2_findings='J0 窗機械作廢', v2a='K8', note='一致'),
  'K9': dict(v2_findings='7 項參數永遠校準不到（含 stress_odd_extra）', v2a='K9',
             correction='stress_odd_extra 描述更正：前瞻產生器 xlive.py CFG_V7X 與 R4 實際為 True（v2 協定「預設關閉」有誤）；v2a 另加 NO_REAL_CALIB'),
  'K10': dict(v2_findings='證據只涵蓋 100 萬 GROW', v2a='K10', note='一致'),
  'K11': dict(v2_findings='外部錨定靠主研究員（git commit 時間）', v2a='K11', correction='v2a 改用 origin 伺服器端 push 時間；本機 %cI 只作輔助'),
  'K12': dict(v2_findings='b 是蒙地卡羅校準', v2a='K12', note='v2a 指 b_a'),
  'K13': dict(v2_findings='校準登錄簿 input_trades_sha256 只能證明登錄當時的帳本', v2a='K15',
              correction='v2 FINDINGS K13 與 v2 協定 K13（in-sample 示範）內容不同；v2a 把前者編為 K15、後者編為 K18'),
 },
 machine_check='凍結前由 s9_freeze.py 比對：v2a FINDINGS 引用的每個數字都要能在協定（known_defects、oc、params、hist_demo_in_sample）或 oc_final.json summary 找到',
)
(HERE / 'K_ALIGNMENT.json').write_text(json.dumps(KA, ensure_ascii=False, indent=1), encoding='utf-8')
print('protocol sha', L.sha(HERE / 'G8_PROTOCOL_v2a.json'), 'K', len(K))
for k in K: print(k[:200])
