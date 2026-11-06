#!/usr/bin/env python3
"""步驟 8：寫 G8_PROTOCOL_v2.json、calib_registry.jsonl（GENESIS）、FREEZE_MANIFEST_v2.json，記 sha256 與 UTC。
  python3 -I -B s8_protocol.py --draft DIR   草稿（寫到 DIR，給判定器測試用；不登記）
  python3 -I -B s8_protocol.py --freeze      正式凍結（只能一次；寫到本資料夾並登記 results.json G8_FREEZE）"""
import sys, json, argparse, importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import g8lib2 as L

DEADLINE = '2026-11-10T05:30:00+00:00'


def evaluator_module():
    spec = importlib.util.spec_from_file_location('g8ev2', HERE / 'g8_evaluate_v2.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def build(frozen_utc):
    PR = L.check_reg('PREREG'); R = L.load_res()
    OCF = R['OC_FINAL']; HD = R['HIST_DEMO']; EX = R['EXPLORE']
    OCT = json.loads((HERE / 'oc_final.json').read_text(encoding='utf-8'))
    assert L.sha(HERE / 'oc_final.json') == OCF['oc_final_json_sha256']
    sel = OCF['selected']; V = PR['versions'][sel]; T = OCT['tables']; Q = OCF['qualification']
    pick = lambda tab, v=sel, k='up': {c: r[v][k] for c, r in tab.items()}
    main_fu = pick(T['main'])
    fu_by_sig = {lab: max(x for c, x in main_fu.items() if c.split('|')[1] == lab) for lab in L.SIG_LABEL}
    strict_fu = pick(T['strict'])
    P1 = L.g8p1_protocol()
    sens_v = {c: r[sel]['up'] for c, r in T['sens'].items() if '|a' not in c}
    known = [
        f"K1 嚴格口徑（真實成本比壓力成本再多 s，s 直接加在判定線上）不在合格條件內：{sel} 最差格誤升級 {max(strict_fu.values()):.4f}（{max(strict_fu, key=strict_fu.get)}）。主口徑只保護到「真實成本 ≤ 一般成本＋0.78pp／月」＝壓力成本＋{0.78 - PR['delta_w']:.3f}pp／月。",
        'K2 σ > 10.4 不在 OC 格內：判定器在升級觸發時若樣本 sd 或滾動 12／24 個月 sd > 10.4 → 標 OUT_OF_GRID，升級證據附缺陷。',
        f"K3 D3（2019 年起殘差區塊 bootstrap）的保護一部分來自 2021-05（+26.1pp）離群月與 lag 6 負自相關；去掉 2021-05（D3x）後 {sel} 最差格 {max(v for c, v in sens_v.items() if c.startswith('D3x')):.4f}，G8P1 原樣最高 {max(T['sens_g8p1'][c]['up'] for c in T['sens_g8p1'] if c.startswith('D3x')):.4f}。",
        f"K4 檢定力極低：α_net＝+1pp／月、σ 6.51、真實成本＝壓力成本時，24 個月升級 D1／D2／D3＝{Q[sel]['power_a1_s651_sDW']['D1']:.3f}／{Q[sel]['power_a1_s651_sDW']['D2']:.3f}／{Q[sel]['power_a1_s651_sDW']['D3']:.3f}。S1 翻案路徑實務上不可行；前瞻主要只能做否定。",
        f"K5 下修 c_d 在 s＝0（真實成本＝一般成本，判定線漂移 −1 − Δ_w）校準；若真實成本＝壓力成本，α_net＝−1 的下修只有 {min(T['down'][f'{dg}|6.14|a-1.0|sDW'][sel]['dn_seq'] for dg in L.DGP_PRIMARY):.3f}～{max(T['down'][f'{dg}|6.14|a-1.0|sDW'][sel]['dn_seq'] for dg in L.DGP_PRIMARY):.3f}；σ 8.4 以上、s＝0 時 D1 降到 0.50 以下。",
        f"K6 α_net＝0 的誤下修只在 σ_d 受 ≤ 0.30 約束；D3 在 σ ≥ 7.5 時 {max(T['down'][f'D3|{l}|a0.0|s0'][sel]['dn_seq'] for l in L.SIG_LABEL):.3f}（> 0.30）。",
        f"K7 Δ_w＝{PR['delta_w']}（V7X GROW100 挑選期；V7 FIX100 為 {abs(EX['hist']['V7_FIX100']['sel']['shift_worse']):.3f}）。V7X 只有 {EX['hist']['V7X_GROW100']['sel']['frac_stress_worse']:.1%} 的 J 窗是壓力線較差（GROW 路徑相依），所以逐窗取較差者有意義。",
        'K8 J0 窗：官方月營收抓取目前 0 列，J0（非官方日）的 rev_gate 會是 J0_UNVERIFIED、沒有 coverage → V2 機械作廢；修好以前判定線實際是 J1～J12 的中位數。',
        'K9 沒有真錢成交時，下列參數永遠校準不到（見 cost_calibration.uncalibratable_without_real_fills）；成本保護只靠「一般與壓力取較差」與 OC 的 s 格。',
        'K10 S1 證據只適用 100 萬 GROW；100 萬 FIX、300 萬、1,000 萬、30 萬都不涵蓋。',
        'K11 外部錨定要靠主研究員：BACKLOG.md 追加錨定行並 git commit FREEZE_MANIFEST_v2.json；兩者缺一，判定器 --official 輸出 INVALID。git 首次 commit 晚於 2026-11-10T05:30Z → v2 自之後第一個完整錨定月起生效，之前照 G8P-1。',
        'K12 b、c_d 是蒙地卡羅校準（集 A，P＝200,000／400,000）；獨立複製應在 2 倍 MC 標準誤內。b 不一致 → 列為缺陷。',
        f"K13 in-sample 示範：歷史 V7X 較差成本線 2006-04～2026-09 月均 {HD['V_sel_worse']['mean_pp']:.3f}pp（一般成本線 {HD['G8P1_normal']['mean_pp']:.3f}）；{sel} 在 223 個 24 個月起點升級 {HD['V_sel_worse']['all']['S1']:.4f}、前瞻失敗 {HD['V_sel_worse']['all']['FAILED']:.3f}。",
    ]
    proto = dict(
        title='主策略 G8 前瞻序貫檢定協定 v2（G8P-2，o15 凍結版）', protocol_version='G8P-2', frozen_utc=frozen_utc, deadline_utc=DEADLINE,
        supersedes=dict(protocol='G8P-1', sha256=L.G8P1_SHA, path='o_g8_protocol_fin_erratum/G8_PROTOCOL.json',
                        rule='G8P-1 保留不刪。v2 自「生效錨定月」起取代 G8P-1；生效錨定月＝2026-11（若 FREEZE_MANIFEST_v2.json 的 git 首次 commit 早於 2026-11-10T05:30Z），'
                             '否則＝該 commit 之後第一個完整錨定月（commit 早於該月 10 日 13:30 台北者），之前的錨定月照 G8P-1 判定。判定器自動計算，不得手動指定。'),
        scope=dict(strategy=P1['scope']['strategy'], benchmark=P1['scope']['benchmark'], data=P1['scope']['data'],
                   s1_evidence_scope='S1 例外 (i) 證據只適用「100 萬 GROW」這一個資金點；100 萬 FIX、300 萬、1,000 萬、30 萬都不涵蓋，不得外推。',
                   not_changed='本協定不改選股規則、前瞻產生器（o_forward_infra_live）、帳本或成本模型；只規定怎麼讀它們的輸出與怎麼判定。'),
        timeline=dict(signal_days=P1['timeline']['signal_days'], first_window=P1['timeline']['first_window'],
                      g8_window='生效錨定月起前 12 個有效錨定月', s1_window='生效錨定月起前 24 個有效錨定月'),
        decision_line=dict(
            definition='每個錨定月 J0..J12 共 13 個窗；判定器自 forward_returns.csv 的 J 列重算：逐有效 J 窗 x_w＝min(excess_pp_w − adj_w, excess_stress_pp_w)，有效 J 窗 ≥ 7 取中位數＝x_n；有效 J 窗 < 7 → 該月作廢、不計入 n',
            adj='adj_w＝100·Δ·換手_w（換手＝窗內紙上整股＋零股買賣成交值／窗初淨值，GROW 100 萬起算）；Δ 只來自登錄簿中 type＝REAL_CALIB 的條目',
            engine='V7X K5T GROW 100 萬（一般成本 excess_pp、壓力成本 excess_stress_pp）',
            reported='另報 x_normal_med（一般成本中位數）、x_stress_med、j_p10（較差成本 J 窗 p10，G1 前瞻版描述）'),
        void_rules=P1['void_rules'], rule=dict(jmed_min_valid_j=7, g8_max_void_in_12=2),
        layer1_literal_g8=dict(rule='前 12 個有效錨定月 Σx_n > 0 → 字面 G8 通過；≤ 0 → 前瞻失敗（吸收，升級路徑關閉）。x_n 是較差成本決策線',
                               void_ext='前 12 個有效月之前累計作廢錨定月 > 2 → 字面 G8 無法判定，延長到第 24 個有效月以前 24 個有效月加總判一次；累計作廢 > 4 → 前瞻失敗',
                               warning='α＝0 時字面 G8 本身約 4～5 成通過，只是必要條件，不是 α>0 的證據'),
        layer2_sequential=dict(
            version=sel, H0='α_net ≤ 0（JMED 較差成本決策線對 0050 的月超額期望值）',
            statistic='z_n＝(S_n − δn)／(σ_eff,n·√n)，S_n＝前 n 個有效月 x 和，σ_eff,n＝max(c·σ_d, s_n)，s_n＝前 n 個有效月 (x−δ) 樣本 sd（ddof＝1）；每個檢視點重算，沒有 refit 門檻、沒有空窗',
            upgrade='n＝12：z ≥ b·√2；n＝24：z ≥ b → 升級待生效；第 24 個有效月期滿才寫「S1 例外 (i) 證據成立」；24 個月後不再判升級',
            down='n＝12、18、24（之後每 6 個月）：z ≤ −c_d → 前瞻失敗',
            order='第 12 個有效月：先字面 G8、再下修、再升級；18：只下修；24：先下修、再升級；先觸發者吸收；前瞻失敗不得回復',
            trigger_report='升級觸發時輸出：樣本 sd、滾動 12／24 個月 sd、σ_checked＝三者最大、σ 格（≥ σ_checked 的最小格點；> 10.4＝OUT_OF_GRID）、該格 max over DGP×s 的 OC 誤升級'),
        params=dict(sigma_design=L.SIGMA_D, c=L.C_INFL, c_d=V['c_d'], b=V['b'], delta=V['delta'], upgrade=V['upgrade'], stat=V['stat'], up_looks=[12, 24],
                    dn_looks_first=[12, 18, 24], delta_w=PR['delta_w'],
                    b_rule='b＝max(1.6787026（G8P-1 b）, max over 主口徑 135 格 q_{0.9525}(B*))；B*＝路徑「剛好升級所需的 b」＝max(z12/√2, z24（若 18 月未下修）)，字面 G8 未過或 12 月下修 → −∞；集 A（SeedSequence([2026110601, DGP_ID, σ 序, s 序, α 序＝2, 0])）每格 P＝200,000',
                    c_d_rule='D1、σ_d、μ＝−1 − Δ_w、P＝400,000、default_rng(2026110602)：c_d＝−q_{0.55}(min_{n∈12,18,24} z_n)（不含字面閘與升級吸收）',
                    dgp_ids=L.DGP_ID),
        oc=dict(seed_B=PR['oc_final']['seed_B'], P_B=PR['oc_final']['P_B'], sigma_grid=L.SIG_GRID, sigma_label=L.SIG_LABEL, s_grid=L.S_GRID,
                mapping_main='μ＝α_net + s − Δ_w − δ（δ 由統計量內扣）', mapping_strict='μ＝α_net + s', dgp=PR['oc_final'],
                criteria=PR['criteria'], all_cells_pass=OCF['oc_all_cells_pass'], qualification=Q, selected=sel, tie_set=OCF['tie_set'],
                false_up_main=main_fu, false_up_max_by_sigma=fu_by_sig, false_up_strict=strict_fu,
                up12_main={c: r[sel]['up12'] for c, r in T['main'].items()},
                down={c: dict(dn=r[sel]['dn_seq'], fail=r[sel]['fail'], up=r[sel]['up']) for c, r in T['down'].items()},
                power={c: r[sel]['up'] for c, r in T['power'].items()}, sens=sens_v,
                g8p1_on_new_grid={c: r['up'] for c, r in T['g8p1'].items()}, rerun_C=OCF['rerun_C'],
                oc_final_json_sha256=OCF['oc_final_json_sha256']),
        cost_calibration=dict(
            data_start='2026-11-12', forbidden='驗證期（≤ 2026-10）任何資料一律不得用於校準',
            registry='calib_registry.jsonl：每行一筆 JSON（canonical：sort_keys、無空白），欄位 seq、type、sched_date、utc、valid_from、counts_toward_slot、applied、delta_per_side、'
                     'delta_real、delta_open、n_real、n_real_months、n_paper_whole、input_trades_sha256、protocol_sha256、evaluator_sha256、official_call、argv、prev_sha、entry_sha；'
                     'entry_sha＝sha256(去掉 entry_sha 的 canonical JSON)；prev_sha＝前一筆 entry_sha（GENESIS＝64 個 0）。只能追加；判定器強制讀全簿，鏈斷或 seq 不連續 → INVALID',
            schedule='排定日＝2027-02-12 起每 3 個月的 12 日；只有 calibrate --official 能追加，每次處理最早一個未登記的到期排定日；排定日過 30 天仍未登記 → status INVALID（漏登）',
            types=dict(NO_REAL_FILLS='合格真錢成交不足、紙上整股也 < 40 列：不占名額、不套用',
                       DELTA_OPEN_ONLY='沒有合格真錢成交、只有紙上整股 Δ_open：只記錄、不占名額、不套用（Δ_open 是模型內部比較，不是市場觀察值）',
                       REAL_CALIB='真錢成交（trades.csv kind＝real）≥ 40 筆、分布在 ≥ 3 個月、首筆距排定日 ≥ 90 天，且距上一次 REAL_CALIB ≥ 365 天：占 12 個月名額；Δ＝max(Δ_real, Δ_open) 套用',
                       REAL_SLOT_LOCKED='真錢成交合格但 12 個月內已有 REAL_CALIB：只記錄、不套用'),
            apply='REAL_CALIB 的 Δ 同時以「回溯（自生效錨定月所有窗）」與「只往後（窗起點晚於 valid_from）」兩種方式計算，正式狀態取兩者較不利者',
            min_rows=40, min_months=3, kappa=0.5,
            estimators='Δ_real＝max(0, p75(真錢相對開盤滑價) − 中位數(同日同檔同向紙上模型滑價))；Δ_open＝max(0, p75(κ·σ60·(√share_actual − √share_pred)))',
            delta_open_desc='Δ_open 是模型內部比較（OASH 固定十分位 × 當日量；share_of_open_vol_actual＝q/(OASH×當日成交量)），不是市場觀察值；它不能校準開盤衝擊',
            uncalibratable_without_real_fills=['零股價差與零股成交機率（100 萬時零股占紙上成交值約 84%）', 'K5T 衝擊係數', 'OASH 開盤參與率本身',
                                               '整股開盤滑價（距開盤跳數）', '開盤漲停遞補股（M2）的實際成交價', '手續費實際折讓（2.8 折假設）', 'stress_odd_extra（壓力情境零股加碼，預設關閉）'],
            real_money_note='沒有真錢成交時，24 個月內登錄簿只會出現 NO_REAL_FILLS 或 DELTA_OPEN_ONLY，沒有任何計入名額的校準（P9）'),
        evaluator=dict(
            file='g8_evaluate_v2.py', official_status='python3 -I -B g8_evaluate_v2.py status --official',
            official_calibrate='python3 -I -B g8_evaluate_v2.py calibrate --official',
            hardcoded=dict(protocol='o_g8p2_protocol_refreeze/G8_PROTOCOL_v2.json', manifest='o_g8p2_protocol_refreeze/FREEZE_MANIFEST_v2.json',
                           registry='o_g8p2_protocol_refreeze/calib_registry.jsonl', ledger='live_tracking/main_forward', nominal_first_anchor='2026-11',
                           backlog='agents/BACKLOG.md', output='o_g8p2_protocol_refreeze/status_runs/status_<UTC>_{OFFICIAL|INVALID}.json'),
            integrity='協定 sha 或判定器 sha 與 FREEZE_MANIFEST_v2.json 不符、登錄簿鏈斷／漏登、BACKLOG 沒有 manifest sha、manifest 未 commit 或 commit 內容不符 → INVALID（不得引用）',
            non_official='任何其他引數（含 --quiet）→ 輸出標 NON_OFFICIAL，不得引用；--test-* 旗標只在 NON_OFFICIAL 生效',
            output_fields='OFFICIAL、label、argv、全部旗標、protocol／manifest／evaluator／registry sha256、帳本三檔 sha256、meta 合併 sha、外部錨定檢查、生效錨定月、problems、'
                          '狀態（回溯／只往後兩種與取較不利者）、事件（升級時含 σ 檢查）、逐月列'),
        family=dict(decision_lines=['JMED（較差成本）'],
                    rule='o8 候選線 M_E50_30、C_E50_70（MIX_C70）事前擇定「不共用為決策線，只描述」，族群 K＝1，不做多重檢定',
                    if_user_designates='若使用者日後指定它們為決策線：Holm K＝3（JMED、M_E50_30、C_E50_70；p＝1−Φ(z24)，依序比 0.05/3、0.05/2、0.05），只能從新協定凍結後第一個完整錨定月起算，不得回溯'),
        judgement_language=dict(
            states={'FORWARD_VALIDATING': '前瞻驗證中', 'UPGRADE_PENDING_24': '前瞻驗證中（已碰升級界限，24 個月期滿才生效）',
                    'S1_EXCEPTION_EVIDENCE': 'S1 例外 (i) 證據成立（只適用 100 萬 GROW）：可重新評估主策略；仍須 G1～G7 歷史門檻全過才可寫「可投資」',
                    'FORWARD_VALIDATING_NO_REJECTION_24': '24 個月前瞻未拒絕虛無：S1 翻案路徑關閉，不可投資',
                    'FORWARD_FAILED': '前瞻失敗：不可投資；停止引用歷史數字'},
            invalid='INVALID：完整性或外部錨定檢查未過，本輸出不得引用',
            historical_gate=P1['judgement_language']['historical_gate']),
        known_defects=known,
        replication=dict(how='只看本檔即可重做：以 params 的 σ_d、c、Δ_w 與 statistic，D1／D2／D3（D3 用 data.resid_post2019 等比縮放到 σ）在 9 σ × 5 s 格各模擬 ≥ 100,000 條 24 個月路徑，'
                             '重算 c_d（c_d_rule）、b（b_rule）與 oc.false_up_main、oc.down、oc.power；結果應在 2 倍 MC 標準誤內'),
        data=dict(resid_post2019=P1['data']['resid_post2019'], resid_post2019_sha256=P1['data']['resid_post2019_sha256'],
                  resid_source=P1['data']['resid_source']),
        hist_demo_in_sample={k: HD[k] for k in ('V_sel_worse', 'G8P1_normal', 'G8P1_worse')},
        registrations=dict(PREREG_DESIGN=R['PREREG_DESIGN_sha256'], PREREG=R['PREREG_sha256']),
        code_sha256={f: L.sha(HERE / f) for f in ('g8_evaluate_v2.py', 'g8lib2.py', 's3_explore.py', 's4_iterate.py', 's5_prereg.py', 's6_oc_final.py', 's7_hist_demo.py', 's8_protocol.py', 'ro_snapshot.py')},
        smoke_manifest_sha256=L.sha(HERE / 'smoke/SMOKE_MANIFEST.json'),
    )
    return proto


def write_all(outdir, frozen_utc):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    EV = evaluator_module()
    proto = build(frozen_utc)
    pp = outdir / 'G8_PROTOCOL_v2.json'
    pp.write_text(json.dumps(proto, ensure_ascii=False, indent=1, default=L._js), encoding='utf-8')
    g = dict(seq=0, type='GENESIS', sched_date=None, utc=frozen_utc, protocol_sha256=L.sha(pp), evaluator_sha256=L.sha(HERE / 'g8_evaluate_v2.py'),
             counts_toward_slot=False, applied=False, delta_per_side=0.0, note='G8P-2 校準登錄簿起點（只能追加）', prev_sha=EV.ZERO)
    g['entry_sha'] = EV.entry_sha(g)
    rp = outdir / 'calib_registry.jsonl'
    rp.write_text(EV.canon(g) + '\n', encoding='utf-8')
    man = dict(protocol='G8P-2', frozen_utc=frozen_utc, deadline_utc=DEADLINE, protocol_sha256=L.sha(pp), evaluator_sha256=L.sha(HERE / 'g8_evaluate_v2.py'),
               registry_genesis_entry_sha=g['entry_sha'], registry_genesis_file_sha256=L.sha(rp), g8lib2_sha256=L.sha(HERE / 'g8lib2.py'),
               results_registrations=proto['registrations'], supersedes_g8p1_sha256=L.G8P1_SHA)
    mp = outdir / 'FREEZE_MANIFEST_v2.json'
    mp.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf-8')
    return pp, rp, mp, man


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--draft'); ap.add_argument('--freeze', action='store_true'); a = ap.parse_args()
    if a.draft:
        pp, rp, mp, man = write_all(a.draft, L.now_utc()); print('draft', json.dumps(man, ensure_ascii=False))
    elif a.freeze:
        assert not (HERE / 'G8_PROTOCOL_v2.json').exists(), '已凍結，不可覆寫'
        t = L.now_utc()
        assert t < DEADLINE, f'晚於期限 {t}'
        pp, rp, mp, man = write_all(HERE, t)
        ms = L.sha(mp)
        line = (f"- [G8P-2 凍結錨定] o_g8p2_protocol_refreeze：FREEZE_MANIFEST_v2.json sha256 {ms}；G8_PROTOCOL_v2.json sha256 {man['protocol_sha256']}；"
                f"g8_evaluate_v2.py sha256 {man['evaluator_sha256']}；凍結 UTC {t}（早於 2026-11-10T05:30Z）。自生效錨定月（2026-11，以 git 首次 commit 為準）起取代 G8P-1（799f8b84…113fa）。")
        (HERE / 'BACKLOG_ANCHOR_LINE.txt').write_text(line + '\n', encoding='utf-8')
        L.save_key('G8_FREEZE', dict(frozen_utc=t, manifest_sha256=ms, protocol_sha256=man['protocol_sha256'], evaluator_sha256=man['evaluator_sha256'],
                                     registry_genesis_entry_sha=man['registry_genesis_entry_sha'], backlog_line=line, before_deadline=t < DEADLINE), need=('PREREG', 'OC_FINAL', 'HIST_DEMO'), overwrite=False)
        print(json.dumps(dict(manifest_sha256=ms, **man), ensure_ascii=False, indent=1)); print(line)
