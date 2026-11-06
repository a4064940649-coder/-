#!/usr/bin/env python3
"""步驟 2：PREREG_DESIGN（只寫一次；版本、OC 格、合格條件、選版規則、seed、預測、協定條款）。
引用 smoke/ 的程式與輸出 sha256（登記前煙霧測試，P ≤ 20,000）。用法：python3 -I -B s2_prereg_design.py"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g8lib2 as L

smoke = {f'smoke/{p.name}': L.sha(p) for p in sorted((HERE / 'smoke').iterdir()) if p.is_file()}
(HERE / 'smoke/SMOKE_MANIFEST.json').write_text(json.dumps(dict(
    utc=L.now_utc(), files=smoke,
    note='smoke0.py 第 1 次執行（smoke0_run1_before_fix.out）發現 t→z 轉換在負尾不對稱（isf(sf(−50)) 精度不足），已改成以 |t| 計算再補號；'
         '第 2 次執行（smoke0.out）全部一致。G8P1 原樣在 σ_d、P＝20,000 的誤升級 0.0392（與 G8P-1 公開的 0.0389 一致）。'
         '煙霧測試沒有計算 V1～V4 的任何誤升級或檢定力。'), ensure_ascii=False, indent=1), encoding='utf-8')
smoke['smoke/SMOKE_MANIFEST.json'] = L.sha(HERE / 'smoke/SMOKE_MANIFEST.json')

D = dict(
    topic='G8P-2：主策略 G8 前瞻序貫協定修正並重凍結（較差成本決策線、消除 σ 空窗、σ×成本全格 OC、只能追加的校準登錄簿、單一正式判定器呼叫、外部錨定）',
    deadline_utc='2026-11-10T05:30:00+00:00', target_freeze_utc='2026-11-09T12:00:00+00:00',
    S1_note='本題不產生選股訊號、不讀前瞻報酬；對象是主策略本身（main_picks_today.py --ex-fin sha256 0897787f…）。',
    smoke_sha256=smoke,
    inputs=dict(g8p1_protocol=f'o_g8_protocol_fin_erratum/G8_PROTOCOL.json sha256 {L.G8P1_SHA}',
                sigma_d=L.SIGMA_D, c=L.C_INFL, resid='G8P-1 data.resid_post2019（89 個月，去均值，sha256 057770ae…）',
                hist_lines='V7X navs_oos_R4.parquet B10|Jj|K5T|{normal,stress}|100萬|GROW（決策引擎）；交叉核對 V7 navs_exec7 100 萬 FIX（o8 fslib.cell_jmatrix）'),
    decision_line='逐 J 窗 x_w＝min(excess_pp_w − adj_w, excess_stress_pp_w)（adj＝校準登錄簿中計入名額的 REAL_CALIB 調整，無則 0），有效 J 窗 ≥ 7 取中位數＝該錨定月 x_n',
    delta_w_rule='Δ_w（較差成本決策線相對一般成本 JMED 的平移）＝min(V7X GROW100, V7 FIX100) 兩條歷史線在挑選期 2006-04～2019-04 的 mean_t[median_j min(n,s) − median_j n] 的絕對值，'
                 '取到小數 3 位（無條件捨去）。2019 年起的平移只描述。Δ_w 取較小者＝OC 最差格漂移較大（保守）。',
    versions={
        'V1': '較差成本線＋每個檢視點 σ_eff＝max(c·σ_d, s_n)（s_n＝前 n 個有效月樣本 sd，ddof＝1）＋b 依最差格校準',
        'V2': '較差成本線＋studentized：t_n＝S_n／(√n·ŝ_HAC6,n)（Bartlett L＝6、去均值、γ_k/n），z̃_n＝Φ⁻¹(F_{t,n−1}(t_n))，OBF 形界限 z̃_n ≥ b√(24/n)；b 依最差格校準',
        'V3': 'V1，序貫統計量改用 x_n − δ（δ＝0.20pp／月）；字面 G8 仍用 x_n',
        'V4': '只有字面 G8＋V1 下修模組（放棄升級路徑），作為退路',
        'G8P1': 'G8P-1 R-FIX 原樣（一般成本線、σ_eff＝c·σ_d、refit 1.5σ_d 一次性）：只作重現與對照，不是候選版本'},
    common=dict(up_looks=[12, 24], up_boundary='z_n ≥ b·√(24/n)', dn_looks='12、18、24（之後每 6 個月）', dn_boundary='z_n ≤ −c_d（版本自己的統計量）',
                literal_g8='前 12 個有效月 Σx_n > 0；未過＝前瞻失敗（吸收、關閉升級）', absorbing='先觸發者吸收；升級在第 24 個有效月期滿才生效',
                trigger_report='升級觸發時輸出當時樣本 sd、滾動 12／24 個月 sd、所在 σ 格（≥ 樣本 sd 的最小格點；> 10.4＝OUT_OF_GRID）與該格（max over DGP×s）的 OC 誤升級'),
    calibration=dict(
        c_d='每版：D1、σ_d、s＝0、α_net＝−1（判定線漂移＝−1 − Δ_w − δ_v），P(min_{n∈12,18,24} z_n ≤ −c_d)＝0.55；P＝400,000、seed 2026110602（V4 用 V1 的 c_d）',
        b='每版：α_net＝0，主口徑 135 格（D1/D2/D3 × 9 σ × 5 s），每格 P＝200,000（seed 集 A，base 2026110601）路徑的 B*（剛好升級所需的 b）之 1−0.0475 分位數；'
          'b＝max(G8P-1 的 b＝1.6787, 各格最大值)。0.0475 是事前留給 MC 誤差的邊際（比 0.05 嚴），不得因檢定力改動',
        order='先 c_d 再 b（b 的 B* 用到 18 月下修）'),
    oc_grid=dict(sigma=L.SIG_LABEL, sigma_first='σ_d＝6.141373712658076（標示 6.14）', s=L.S_GRID,
                 main_mapping='主口徑：判定線漂移 μ＝α_net + s − Δ_w − δ_v（δ_v：V3＝0.20，其他 0）',
                 strict_mapping='嚴格口徑（敏感度）：μ＝α_net + s − δ_v（s 直接加在判定線上，不扣 Δ_w）',
                 g8p1_mapping='G8P1 對照：μ＝α_net + s（一般成本線）',
                 dgp=dict(D1='iid N(μ, σ²)', D2='μ＋σ·t4/√2', D3='μ＋σ·e/sd(e)，e＝resid_post2019 幾何區塊（平均 6）環狀 bootstrap',
                          sens=dict(D3x='D3 但殘差去掉 2021-05（+26.14）後重新去均值、等比縮放', D4='o8 波動狀態切換（0.7／1.4，t4）× σ')),
                 power='α_net ∈ {+0.5, +1, +2} × 9 σ × D1～D3：主檢定力口徑 s＝Δ_w（真實成本＝壓力成本 → μ＝α_net − δ_v）；另報 s＝0（μ＝α_net − Δ_w − δ_v）',
                 down='α_net ∈ {−1, −0.5, 0} × 9 σ × D1～D3，s＝0 與 s＝Δ_w',
                 seeds=dict(A='2026110601（校準，P＝200,000）', B='2026110711（正式 OC，P＝100,000）', C='2026110821（重跑，P＝400,000）',
                            explore='2026110631', cell='SeedSequence([base, DGP_ID, σ 序, s 序（99＝非格點）, α 序, 口徑 mode])；同一格各版本共用路徑')),
    criteria=dict(false_up='主口徑 135 格，24 個月內誤升級（S1 證據，含字面 G8 閘）點估計 ≤ 0.05（集 B）；|p−0.05| < 2SE 的格用集 C（P＝400,000）重跑，以 C 為準',
                  down='σ_d、s＝0、D1～D3：α_net＝−1 序貫下修 ≥ 0.50；α_net＝0 序貫誤下修 ≤ 0.30',
                  power='全部照實報，不設下限，不得為檢定力調參'),
    selection=dict(rule='全格合格（false_up 與 down 都過）的版本中，取 α_net＝+1、σ＝6.51、s＝Δ_w、D1～D3 中最低者的檢定力最高者；差距在 2√2 SE 內視為同分，取較簡單者（簡單順序 V1 < V3 < V2）',
                   fallback='V1～V3 都不合格 → 凍結 V4（若 V4 合格）並宣告「OC 無法全格合格（升級路徑）」，附已知缺陷清單，S1 翻案路徑標「實務上不可行」；'
                            '若連 V4 都不合格（下修條件）→ 凍結最差格誤升級最低的版本並列缺陷'),
    iterations='只允許 V1～V4 四個版本；集 A 上做 c_d、b 校準與初步 OC（iterations.json）。程式錯誤可修（記錄），規格不得改。',
    predictions={
        'P1': 'G8P1 原樣（μ＝s）D1：σ＝8.4 誤升級 ≥ 7%、σ＝10.4 ≥ 7.5%；σ 偏高時較強、σ ≤ 6.5 較弱',
        'P2': '逐窗較差成本 ≥ 95% 的 J 窗是壓力線較差；挑選期決策線對一般成本線平移 −0.58pp／月',
        'P3': 'V1、V3 在 D1、D3 全格誤升級 ≤ 5%；D2 最可能超標，超標較常在第 12 月檢視點',
        'P4': '最差格＝σ 最高 × s＝0.78；b 由 1.679 提高到約 1.75～1.95',
        'P5': 'α＝+1 時 24 個月升級 ≤ 12%；α＝+2 ≤ 35%；任一版本 α＝+1 > 15% 先懷疑程式',
        'P6': 'G8P1 在 D3x（去 2021-05、等比放大）誤升級由 2.2% 升到 ≥ 3%',
        'P7': '選定版本 α＝−1 下修：σ＝8.4 < 0.50；σ_d ≥ 0.50',
        'P8': 'v2 套到歷史 V7X JMED（較差成本線，in-sample 規則示範）升級頻率 ≤ G8P1 的 1.35%；起點 2024-10 的最後完整路徑判前瞻失敗',
        'P9': '沒有真錢成交時，24 個月內校準登錄簿只會有 NO_REAL_FILLS 或 DELTA_OPEN_ONLY 條目，沒有任何計入名額的校準'},
    protocol_clauses=[
        '校準登錄簿 calib_registry.jsonl：每筆含 seq、prev_sha（前一筆整行 sha256）、entry_sha、UTC，只能追加；判定器強制讀全簿，鏈斷、漏登（排定日已過 30 天未登）→ INVALID',
        '排定校準日：2027-02-12 起每 3 個月（2027-05-12、2027-08-12 …）；只有 calibrate --official 能追加',
        'NO_REAL_FILLS、DELTA_OPEN_ONLY 不占 12 個月名額、不套用；只有真錢成交 ≥ 40 筆且跨 ≥ 3 個月的 REAL_CALIB 計入名額並套用（自 2026-11 起所有窗；判定取「回溯套用」與「只往後套用」兩者較不利的狀態）',
        'Δ_open 改寫為「模型內部比較（OASH 固定十分位 × 當日量），不是市場觀察值」',
        '列出沒有真錢成交時永遠校準不到的參數',
        '判定器只有一種正式呼叫：python3 -I -B g8_evaluate_v2.py status --official；協定、登錄簿、帳本路徑與起算月寫死；協定或程式 sha 與凍結清單不符 → 拒絕；其他旗標 → NON_OFFICIAL；輸出記錄全部旗標與 sha256',
        'S1 證據只適用 100 萬 GROW；100 萬 FIX、300 萬、1,000 萬、30 萬都不涵蓋',
        'o8 候選線 M_E50_30、C_E50_70 事前擇定「不共用為決策線，只描述」，族群 K＝1；若使用者日後指定為決策線 → Holm K＝3，只能從新協定凍結後第一個完整錨定月起算',
        'G8P-1 保留不刪；v2 自生效錨定月起取代 G8P-1；凍結晚於 05:30Z 或外部錨定（git 首次 commit）晚於期限 → 生效月延後到之後第一個完整錨定月，之前的錨定月照 G8P-1',
        '作廢規則沿用 G8P-1（V0／V1／V2，只看時戳與覆蓋率）'],
    readonly=dict(folders=['live_tracking/main_forward', 'o_forward_infra_live'],
                  g8p1_baseline='719 檔、清單 sha 639dd8b0…（2026-11-04）；本題開工清單見 ro_start.json（736 檔，差異須由每日排程日誌與 mtime 解釋）'),
)
h = L.register('PREREG_DESIGN', D)
print('PREREG_DESIGN sha256', h, L.load_res()['PREREG_DESIGN_utc'])
