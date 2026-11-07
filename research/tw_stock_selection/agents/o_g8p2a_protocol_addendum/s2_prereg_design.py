#!/usr/bin/env python3
"""步驟 2：寫煙霧測試清單 SMOKE_MANIFEST_v2a.json，並登記 PREREG_DESIGN（results.json；sha256＋UTC）。
PREREG_DESIGN 之後才跑集 A（校準 b_a）；PREREG 之後才跑集 B／C、歷史示範與判定器正式測試。
用法：python3 -I -B s2_prereg_design.py"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g8alib as L

res = L.load_res()
assert 'EXPLORE' in res, '先跑 s1_explore.py'
smoke = dict(purpose='登記前煙霧測試：只做管線核對（P＝20,000），不輸出任何版本的誤升級或檢定力',
             runs=[dict(program='smoke/smoke0.py', output='smoke/smoke0_run1.out', result='n_bad＝0（第 1 次即全部一致，未修改程式）')],
             file_sha256={f: L.sha(HERE / f) for f in ('smoke/smoke0.py', 'smoke/smoke0_run1.out', 'g8alib.py', 'refimpl.py')},
             utc=L.now_utc())
(HERE / 'smoke' / 'SMOKE_MANIFEST_v2a.json').write_text(json.dumps(smoke, ensure_ascii=False, indent=1), encoding='utf-8')

SIG = L.SIG_LABEL; SG = L.S_GRID
D = dict(
 title='G8P-2a：G8P-2 只能更嚴的補遺（PREREG_DESIGN）',
 question='在第一個前瞻窗（2026-11-10 J0）之前，能否凍結一份對 G8P-2 逐路徑更嚴的補遺：在 iid、正序列相關（AR／MA）與挑選期區塊 bootstrap 的 495 格裡，24 個月內誤升級全部 ≤ 0.05，並把判定輸出、open 月份、外部錨定、S1 證據的自由度改成機械規則？檢定力代價照實報。',
 mechanism='證據機制（非報酬機制）：序貫檢定型一錯誤要在所有合理世界受控。v2 分母 max(c·σ_d, s_n) 在 σ 偏高時丟掉相依膨脹 c；OC 只有 iid 與 2019 年起殘差；status 可回溯改寫、open 月份可靜默剔除、錨定讀可偽造的本機 committer date、S1 不需真錢校準。',
 construction=dict(
  rule='v2a 升級＝v2 升級（G8P-2 判定器原樣）∧ 新規則升級；v2a 前瞻失敗＝v2 失敗 ∨ 新規則失敗；字面 G8 不變。判定器以「較不利者」合併（ORDER：FORWARD_FAILED < NO_REJECTION_24 < VALIDATING < UPGRADE_PENDING_24 < S1）。',
  pathwise='同一路徑上 v2a 升級 ⇒ v2 升級、v2 失敗 ⇒ v2a 失敗；CRN 下每格誤升級 ≤ v2 同格（逐路徑，不只期望值）。OC 每格輸出 viol_up＝#(up_v2a ∧ ¬up_v2)、viol_fail＝#(fail_v2 ∧ ¬fail_v2a)，任何 > 0 → 實作錯誤，不得凍結。',
  c_d='c_d＝1.2098465851（G8P-2）不重校；新規則下修也用 c_d，失敗取聯集。',
  c='c＝1.0596378780（G8P-1／G8P-2 協定值）不重估；σ_d＝6.1413737127。'),
 versions=dict(
  A1='z^A_n＝S_n／(c·max(σ_d, s_n)·√n)；升級 n＝12：z^A ≥ b_a·√2；n＝24：z^A ≥ b_a；下修 n＝12、18、24（之後每 6 個月）z^A ≤ −c_d；順序同 v2。',
  A2='A1，且升級時 σ_checked＝max(s_n, 滾動 12 個月 sd, 滾動 24 個月 sd)（原始 x，ddof＝1；不足 12／24 個月者不計）≤ 10.4，且所在 σ 格（≥ σ_checked 的最小格點）的 v2a OC 最大誤升級（over DGP×s）≤ 0.05；不成立 → 該檢視點不升級（24 月仍可升級）。',
  A3='z^H_n＝S_n／(max(c·σ_d, ŝ_HAC,n)·√n)；ŝ_HAC＝Newey-West Bartlett L＝6、去均值、γ_k 除以 n 的長期 sd；其餘同 A1。',
  A4='沒有升級路徑：字面 G8＋v2 下修＋A1 下修聯集（可達成的最嚴底線）。',
  n_versions='4（A1～A4）；另有診斷欄 A1b0（A1 分母、b＝v2 的 1.93376，只供 P1，非候選）'),
 b_rule='每個有升級路徑的版本 ver∈{A1,A2,A3}：b_a(ver)＝max(1.9337576513（v2 b）, max over 495 格 q_{0.9525}(B*_comb))；B*_comb＝路徑上 v2 升級時取新規則「剛好升級所需的 b」（max(z12/√2〔A2 需 σ 條件〕, z24〔18 月未下修且 A2 σ 條件〕)，字面 G8 未過或 12 月新規則下修 → −∞），v2 未升級 → −∞；numpy quantile（linear）。集 A：SeedSequence([2026110701, DGP_ID, σ 序, s 序, α 序＝2, mode 0])，每格 P＝200,000。b_a 不得低於 v2 的 b。',
 oc_grid=dict(
  sigma=SIG, s=SG, alpha_index='α 序：0..5＝−1、−0.5、0、+0.5、+1、+2', mapping_main='μ＝α + s − Δ_w，Δ_w＝0.527', mapping_strict='μ＝α + s（嚴格口徑，只作敏感度）',
  dgps=dict(D1='iid 常態', D2='iid t(4)/√2', D3='G8P-2 協定內嵌 resid_post2019（89 個月，sha 057770ae…）去均值標準化，環狀區塊 bootstrap（幾何平均區塊 6）',
            AR03='AR(1) ρ＝0.03，t5 創新（標準化到單位變異），平穩起點，邊際 sd＝σ', AR057='AR(1) ρ＝0.057（c＝1.06 隱含程度），t5 創新', AR08='AR(1) ρ＝0.08，t5', AR10='AR(1) ρ＝0.10，t5',
            MA1='MA(1) θ＝0.1，常態創新（較 t5 保守），邊際 sd＝σ', MA2='MA(1) θ＝0.2，常態', MA3='MA(1) θ＝0.3，常態（涵蓋 J12／J0 重疊理論 θ≈0.14，見 explore E2）',
            D5='挑選期 2006-04～2019-04 V7X GROW100 較差成本 JMED 殘差（157 個月，d5_resid_sel.json）去均值標準化，環狀區塊 bootstrap（幾何平均區塊 6）'),
  n_cells=495, dgp_id=L.DGP_ID,
  sens='敏感度（不入合格條件）：ARN057／ARN08／ARN10（常態創新 AR）、MAT3（t5 創新 MA θ 0.3），主口徑 9σ×5s；嚴格口徑 495 格'),
 seeds=dict(A=2026110701, B=2026110702, C=2026110703, rule='cell_rng(base, DGP_ID, σ 序, s 序（Δ_w＝5）, α 序, mode)；同一格各版本與 v2 共用路徑（CRN）'),
 oc_sets=dict(
  B='PREREG 之後：P＝100,000／格。main（α＝0、mode 0）495 格；strict（mode 1）495 格；down α∈{−1,−0.5,0} × 9σ × 11 DGP × s∈{0, Δ_w}；power α∈{+0.5,+1,+2} × 9σ × 11 DGP × s∈{Δ_w, 0}；sens 180 格；A1b0 診斷（main 495 格）',
  C='主口徑 (版本, 格) 若 |p_B − 0.05| < 2·SE_B → 集 C P＝400,000 重跑，以 C 為準'),
 qualification='合格＝主口徑 495 格誤升級（24 個月內 S1 證據，含字面 G8 閘）全部 ≤ 0.05（集 B／C），且全部 OC 格（main、strict、down、power、sens）逐路徑子集檢查 viol_up＝viol_fail＝0。下修、誤下修、檢定力不入合格條件，照實報。',
 selection='只在合格版本中選：取「11 種 DGP 中最低的檢定力（α＝+1、σ 6.51、s＝Δ_w，集 B）」最高者；與較簡單版本差距 < 2√2·SE（SE＝√(p(1−p)/P)，取兩者較大）時取較簡單版本（A1 > A2 > A3）。全部不合格 → A4。不得為檢定力放寬任何門檻。',
 predictions=dict(
  P1='A1b0（只改分母、b 不變）：主口徑 135 格（D1／D2／D3）誤升級最高 ≤ 4.3%；但 AR(1) ρ 0.1、s 0.78、σ 8.4～10.4 至少 1 格 > 5%（預期 5.3～6.0%）。',
  P2='新增 DGP 中最差的是 MA(1) θ 0.3，不是 AR(1) ρ 0.1；D5 的最差格（A1b0 口徑）高於 D3 最差格至少 2 倍（D3 約 1.0%，D5 預期 3.5～5.5%）。',
  P3='重校後 b_a(A1) 落在 2.05～2.40；驅動格在 MA θ 0.3、s 0.78、σ 8～10.4 的高原上；σ 6.14、s 0 的格子誤升級（選定版本）≤ 1.5%。',
  P4='檢定力（α＝+1、真實成本＝壓力成本、σ 6.51）：D1 從 v2 的 0.106 降到 0.05～0.08；11 種 DGP 的最低值 ≤ 0.04；α＝+2 最高 ≤ 0.25。若 α＝+1 的檢定力高於 v2 同格（同路徑），代表實作有錯，一票否決。',
  P5='逐路徑子集檢查：OC 全部路徑（集 B）與 ≥ 300 條合成帳本路徑上，「v2a 升級但 v2 沒升級」＝0、「v2 失敗但 v2a 沒失敗」＝0；任何 > 0 → 實作錯誤，不得凍結。',
  P6='同一份 in-sample 歷史示範（V7X 較差成本 JMED，2006-04～2026-09，223 個 24 個月起點）：選定版本升級 0，前瞻失敗 ≥ 65.5%（不得低於 v2）。',
  P7='下修側：α＝0 的誤下修在 D5 上 > 0.30 的格數 ≥ 1；照實列為缺陷，不為此放寬。',
  subsample='誤升級最強在 s 0.78 × σ 8～10.4 × MA／AR 正相關；最弱在 s 0、D3。治理測試：open 月份與回溯更正兩種攻擊在 v2 下改變判定 ≥ 1 種（explore E4 已確認 2 種），在 v2a 下改變判定次數＝0。'),
 governance=dict(
  i_chain='每次正式呼叫（label＝OFFICIAL）把 canonical JSON（sort_keys、無空白）追加到 status_chain.jsonl：seq、prev_sha、entry_sha（去掉 entry_sha 的 canonical JSON 的 sha256）、utc、帳本三檔 sha、逐有效月 x_n 與狀態、合併狀態、v2／新規則狀態、字面 G8、遠端錨定摘要、status_runs 輸出檔 sha。GENESIS（seq 0）在凍結時建立並寫進 FREEZE_MANIFEST_v2a。鏈斷、seq 不連續、utc 倒退、GENESIS 不符、或本機鏈不是遠端（origin）鏈的延伸（截尾）→ INVALID。',
  i_lock='吸收狀態（FORWARD_FAILED、S1_EXCEPTION_EVIDENCE）與字面 G8（PASS／FAIL）一經正式輸出即鎖定；之後重算不同 → RETRO_CHANGE 旗標，狀態取較不利者。已記錄月份的 x_n 或有效／作廢狀態事後改變 → RETRO_CHANGE，並以「鏈上首次記錄值」重算一次，取兩者較不利者。',
  ii_open='錨定月 m 的期限＝(m+3) 月 26 日 00:00 台北（＝最後一窗最早可能結束日 (m+1) 月 26 日後 2 個月）。期限前沒有任何「已推上 origin 的正式鏈條目」見證該月完成（13 個 J 列全部 complete）→ 計作廢 V3，計入 g8_max_void_in_12 與累計作廢，之後補完也不得插回。期限內 open 的月份暫不計入，且其後的檢視點延後到它補完或轉 V3 為止（只延後新規則；v2 原樣元件照 v2）。',
  iii_remote='判定器以 git ls-remote 取 origin 分支頭，必要時 git fetch；確認 FREEZE_MANIFEST_v2a 的首次 commit 是 origin 分支頭的祖先、內容相同；以 GitHub 伺服器端 push 事件（GET /repos/{owner}/{repo}/activity，activity_type＝push 的 timestamp）取「after 含該 commit 的最早 push 時間」。只有 push 時間 < 期限才算錨定成立；本機 %cI 只作輔助記錄。BACKLOG 錨定行要出現在 origin 分支頭的 BACKLOG.md，且以同法取其首次 push 時間；生效判定用兩者較晚者。無法連線或查不到 → REMOTE_UNVERIFIED，該次輸出 NON_OFFICIAL（不得引用），不寫鏈。API 回應存 raw/remote_anchor/（sha＋抓取 UTC）。',
  iv_s1='S1_EXCEPTION_EVIDENCE 輸出必須附至少一筆 type＝REAL_CALIB 且 applied 的登錄簿條目 entry_sha；沒有 → 旗標 NO_REAL_CALIB，judgement_language 寫「不得引用」。',
  v_adj='adj_w＝100·max(全部適用 REAL_CALIB 的 Δ)·換手_w（修正 v2 迴圈覆寫）；回溯與只往後兩種算法取較不利者。',
  vi_registry='登錄簿（沿用 G8P-2 calib_registry.jsonl，校準仍以 g8_evaluate_v2.py calibrate --official 登記）：條目 utc 晚於 sched_date＋30 天，或同一排定日重複登記 → INVALID。',
  vii_text='stress_odd_extra 更正：前瞻產生器 xlive.py CFG_V7X 與 V7X 正式 R4 實際為 True（v2 協定原文「預設關閉」有誤）。',
  viii_K='以 v2a 協定 known_defects 為唯一正本；K_ALIGNMENT.json 逐條對照 v2 協定 K1～K13、v2 FINDINGS K1～K13 到 v2a 編號並更正 K3、K5、K6、K13；凍結前機器比對：FINDINGS 每個數字都要能在協定或 oc 摘要中找到。',
  ix_supersede='G8P-1、G8P-2 保留不刪。v2a 生效錨定月：FREEZE_MANIFEST_v2a（與 BACKLOG 錨定行）的 origin push 時間 < 2026-11-10T05:30Z → 2026-11；否則＝push 後第一個完整錨定月（push 早於該月 10 日 13:30 台北者）；之前的錨定月照 G8P-2 判定。判定器自動計算，不得手動指定。'),
 evaluator_tests='≥ 300 條合成帳本路徑（含 36 個有效月、作廢延長、open 逾期、回溯更正攻擊、REAL_CALIB 有／無、登錄簿竄改、遠端未推送、本機 committer date 偽造：GIT_COMMITTER_DATE 早於期限但 push 晚於期限 → 判錨定失敗）；判定器 vs OC 向量化程式 vs 純量參考實作 0 不一致；逐路徑子集 0 違反；治理攻擊在 v2a 下 0 次改變判定。遠端以暫存 bare repo＋假 push 事件表模擬（測試旗標只在 NON_OFFICIAL 生效）。',
 hist_demo='PREREG 後：歷史 V7X 較差成本 JMED 線 2006-04～2026-09（246 個月），223 個 24 個月起點，選定版本與 v2 各算 S1／失敗比例（in-sample 規則示範，不是證據）。',
 readonly='開工 ro_start.json（已做）與結束 ro_end.json：兩個前瞻資料夾逐檔 sha256＋mtime；差異檔 mtime 只能落在每日排程日誌時段，無法歸因 0 檔；本題 0 寫入。',
 fallback='11-10 03:30Z 前若相依 DGP 的 OC 未跑完：凍結 v2a-G＝第 4 段全部治理規則＋v2 原參數＋b_a＝max(已算完格子所需值, v2 b)，照實宣告「相依 DGP 未全格驗證」；仍來不及 → 照 G8P-2 執行並附缺陷清單。若 A1～A3 全部不合格 → 凍結 A4（可達成的最嚴版本）並照實宣告。',
 final_gate='「v2a 生效」需全部成立：G8_PROTOCOL_v2a.json sha256 與凍結 UTC 寫入 FREEZE_MANIFEST_v2a.json；commit 已 push 到 origin 且伺服器端 push 時間 < 2026-11-10T05:30Z；BACKLOG 錨定行在遠端可見；判定器正式呼叫 OFFICIAL、problems 為空；495 格全部 ≤ 0.05 且 viol 0；判定器測試 0 不一致、治理攻擊 0 次改變判定；唯讀無法歸因 0 檔。子代理不能 commit／push：前三項由研究中心於本輪結束時完成並以正式呼叫確認。',
 g6='複製員只看 G8_PROTOCOL_v2a.json 從頭重寫並重算 b_a 與 OC；b_a 差距 > 2 倍 MC 標準誤 → 缺陷；任何格合格判定翻轉 → 一票否決，退回 v2a-G 或 G8P-2。',
 data='全部既有資料，不下載新資料；raw/ 只放判定器的遠端錨定 API 回應。2019 年起殘差只用於 D3 既有格；挑選期 D5 只取形狀（平均超額未輸出、未用於任何設計步驟）。',
 file_sha256={f: L.sha(HERE / f) for f in ('g8alib.py', 'refimpl.py', 'synth.py', 's1_explore.py', 'explore.json', 'd5_resid_sel.json', 'ro_snapshot.py', 'ro_start.json',
                                           'smoke/smoke0.py', 'smoke/smoke0_run1.out', 'smoke/SMOKE_MANIFEST_v2a.json')},
 smoke_manifest_sha256=L.sha(HERE / 'smoke' / 'SMOKE_MANIFEST_v2a.json'),
 v2_refs=dict(protocol=L.V2_PROTO_SHA, manifest=L.V2_MANIFEST_SHA, evaluator=L.V2_EVAL_SHA, g8p1=L.G8P1_SHA),
)
h = L.register('PREREG_DESIGN', D)
print('PREREG_DESIGN', h, L.load_res()['PREREG_DESIGN_utc'])
