export const meta = {
  name: 'tw-research-center',
  description: '台股策略研究中心一輪：研究組結論交叉審查 → 實際重測 → 角色討論 → 彙整與下一輪研究題目（可選：直接執行新研究）',
  whenToUse: '各研究組在 research/tw_stock_selection/agents/<組>/ 寫好 FINDINGS.md 之後，執行一輪審查與討論；args.run_new_research=true 時會再執行完整性審查員提出的新題目',
  phases: [
    { title: '審查', detail: '每組兩位反方審查員（資料與前視偏誤、成本與可成交性）' },
    { title: '重測', detail: '被質疑的結論由重測員寫程式重跑' },
    { title: '討論', detail: '基金經理、風控長、散戶實戰派：提案與交叉回應' },
    { title: '彙整', detail: '主研究員寫中心報告；完整性審查員提出下一輪題目' },
    { title: '新研究', detail: '（可選）執行下一輪題目，產出新的 FINDINGS.md' },
  ],
}

const ROOT = '/home/user/-/research/tw_stock_selection'
const AG = ROOT + '/agents'
const BRIEF = AG + '/BRIEF.md'
const RUN = (args && args.round) ? String(args.round) : 'r1'
const MAX_RECHECK = (args && args.max_rechecks) || 6
const MAX_TOPICS = (args && args.max_topics) || 4
const TEAMS = (args && args.teams) || ['short_term', 'events', 'ml', 'seasonal_statarb', 'industry', 'technical', 'institutions']
const CORE = `中心目前採用的策略（主研究員已驗證，見 ${ROOT}/output/ 與 /home/user/-/台股個股選股規則實證研究報告.md）：
（第 1 輪審查後版本，見 ${AG}/CENTER_REPORT_r1.md 與報告第 11 節）
(1) 52 週高點 × 月營收動能月度選股，可成交版 M2（換股日收盤算訊號、隔日開盤買；agents/exec_rebuild/）：30 檔淨年化 47.3%、日頻最大回撤 −35%、對 0050 t 1.91、對等權 t 3.99；10 檔 57.5%、t 1.99；2024～25 對 0050 t −0.26～0.41
(2) 處置股 10 槽固定金額、嚴格版（去重疊、進場 ret ≥ 9% 不買、跌停延後出場、成本 0.8%／1.2%）：2021～2026 真實名單每筆淨 +2.1～3.2%、t_month 2.7；但 2005～2021 規則重建淨 +0.35%、t_month −0.89（agents/disposal_bear_longhistory/），機制檢定不成立（agents/attention_disposal_mechanism/）→ 上限 20%、指數 < 200 日均線不進場
(3) 建議配置（agents/exec_rebuild/run_alloc2.py，閒置持 0050）：選股 10 檔 80%＋處置 20% 年化 55.8%、日頻回撤 −33%、對 0050 t 2.29；0050 30%＋選股 30 檔 50%＋處置 20% 年化 41.4%、t 2.47。bt_aggressive.py 的 92.9% 與第三輪 40/40/20 的 54.2% 已作廢。使用者目標：打敗大盤，可接受大回撤。`

const CLAIMS = {
  type: 'object',
  properties: {
    claims: { type: 'array', items: { type: 'object', properties: {
      claim: { type: 'string' },
      verdict: { type: 'string', enum: ['holds', 'doubtful', 'refuted'] },
      issues: { type: 'array', items: { type: 'string' } },
      recheck: { type: 'string', description: '若 doubtful/refuted：具體要怎麼重跑才能確認（檔案、參數、期間）；否則空字串' },
    }, required: ['claim', 'verdict', 'issues', 'recheck'] } },
  },
  required: ['claims'],
}
const RECHECK = {
  type: 'object',
  properties: {
    claim: { type: 'string' },
    final: { type: 'string', enum: ['holds', 'weakened', 'refuted'] },
    evidence: { type: 'string', description: '重跑的數字與檔案路徑' },
  },
  required: ['claim', 'final', 'evidence'],
}
const VIEW = {
  type: 'object',
  properties: {
    adopt: { type: 'array', items: { type: 'object', properties: {
      strategy: { type: 'string' }, weight_pct: { type: 'number' }, reason: { type: 'string' } }, required: ['strategy', 'weight_pct', 'reason'] } },
    reject: { type: 'array', items: { type: 'string' } },
    concerns: { type: 'array', items: { type: 'string' } },
  },
  required: ['adopt', 'reject', 'concerns'],
}
const AGENDA = {
  type: 'object',
  properties: {
    topics: { type: 'array', items: { type: 'object', properties: {
      folder: { type: 'string', description: '新資料夾名（英數底線）' },
      hypothesis: { type: 'string' },
      design: { type: 'string', description: '事前固定的檢定設計、資料、挑選期/驗證期' },
      why: { type: 'string' },
    }, required: ['folder', 'hypothesis', 'design', 'why'] } },
  },
  required: ['topics'],
}

const LENSES = [
  { key: 'data', text: '資料與統計偏誤：資料錯誤與極端值、公司行動、前視偏誤（盤後資料當天使用、未來資訊滲入特徵）、存活者偏誤、多重檢定與事後挑選、樣本期間太順（2021–2026 多頭）。' },
  { key: 'exec', text: '成本與可成交性：來回成本是否低估、鎖漲停買不到、處置股預收款券與撮合間隔、零股與整股、容量、稅、滑價、隔日才看得到的資料。' },
]

// 1–2：審查 → 重測（每組獨立流水線，不設柵欄）
phase('審查')
const reviewed = await pipeline(TEAMS,
  async (team) => {
    const votes = await parallel(LENSES.map(l => () => agent(
      `你是台股策略研究中心的反方審查員，任務是挑戰並推翻研究組的結論；不確定時判 doubtful。審查角度：${l.text}
請閱讀 ${AG}/${team}/FINDINGS.md 以及同資料夾的程式與 results*.json（需要時讀 ${BRIEF} 的資料說明，可以自己寫小程式驗證，用 python3 -I 執行；不要修改該組檔案）。
逐條列出該組的主要結論（尤其是宣稱「可用」或數字很好的），給出 verdict 與具體問題；若需要重跑，在 recheck 寫清楚怎麼做。若 FINDINGS.md 不存在，回傳 claims 為空陣列。`,
      { label: `審查:${team}:${l.key}`, phase: '審查', schema: CLAIMS })))
    const all = votes.filter(Boolean).flatMap(v => v.claims)
    return { team, claims: all }
  },
  async (rev) => {
    const todo = rev.claims.filter(c => c.verdict !== 'holds' && c.recheck)
    if (!todo.length) return { ...rev, rechecks: [] }
    const rechecks = await parallel(todo.slice(0, MAX_RECHECK).map((c, i) => () => agent(
      `你是台股策略研究中心的重測員。研究組「${rev.team}」的結論受到質疑，請實際寫程式重跑確認，不接受口頭辯護。
結論：${c.claim}
質疑：${c.issues.join('；')}
重跑方式：${c.recheck}
資料與紀律見 ${BRIEF}。程式存到 ${AG}/${rev.team}/recheck_${RUN}_${i}.py（python3 -I 執行），結果寫在回傳的 evidence（數字＋檔案路徑）。`,
      { label: `重測:${rev.team}:${i}`, phase: '重測', schema: RECHECK })))
    if (todo.length > MAX_RECHECK) log(`${rev.team}：${todo.length} 條受質疑，只重測前 ${MAX_RECHECK} 條，其餘列入下一輪`)
    return { ...rev, rechecks: rechecks.filter(Boolean) }
  })
const dossier = JSON.stringify(reviewed.filter(Boolean), null, 1)

// 3：討論（需要全部審查結果 → 柵欄）
phase('討論')
const ROLES = [
  { key: 'pm', text: '量化基金經理：追求打敗大盤、重視 t 值與可複製性，願意接受大回撤。' },
  { key: 'risk', text: '風控長：重視最壞情境、制度改變風險、容量與尾部虧損，專挑最可能失效的策略。' },
  { key: 'retail', text: '散戶實戰派：只用現股、資金 30 萬～1,000 萬、要可以每天每月實際執行、在意手續費與零股。' },
]
const views = (await parallel(ROLES.map(r => () => agent(
  `你是台股策略研究中心的${r.text}
${CORE}
以下是本輪各研究組結論經反方審查與重測後的紀錄（JSON）：
${dossier}
請提出你的配置建議（策略與權重百分比，加總 100），以及應該淘汰的策略與主要疑慮。只能採用審查後仍成立（holds 或重測 holds/weakened 且說明理由）的策略。`,
  { label: `提案:${r.key}`, phase: '討論', schema: VIEW })))).filter(Boolean)
const rebuttals = (await parallel(ROLES.map((r, i) => () => agent(
  `你是${r.text} 以下是三位角色的提案（JSON）：${JSON.stringify(views)}
請針對其他兩位的提案逐點回應：同意什麼、反對什麼、為什麼，最後給出你修正後的配置（策略：權重%）。用繁體中文條列，300 字內。`,
  { label: `回應:${r.key}`, phase: '討論' })))).filter(Boolean)

// 4：彙整＋完整性審查
phase('彙整')
const report = await agent(
  `你是台股策略研究中心的主研究員。請寫出本輪的中心報告（繁體中文、研究報告風格、條列、關鍵數字加粗），內容：
1. 本輪結論一句話；2. 各組結論審查結果表（組別、結論、審查判定、重測結果）；3. 三位角色的提案與交叉回應摘要；4. 最終採用的配置（策略、權重、預期打折後的超額報酬與最大回撤、對 0050 t 值）；5. 淘汰清單與理由；6. 風險與反證。
${CORE}
審查與重測紀錄：${dossier}
角色提案：${JSON.stringify(views)}
交叉回應：${JSON.stringify(rebuttals)}
子代理不能寫報告檔：請直接回傳完整 Markdown 全文，由主研究員存成 ${AG}/CENTER_REPORT_${RUN}.md。不要 git commit。`,
  { label: '主研究員報告', phase: '彙整' })
const agenda = await agent(
  `你是台股策略研究中心的完整性審查員。問：還有什麼沒測？哪些主張只靠一個資料來源或一段期間？哪些現股可執行的台股制度或市場結構還沒研究（例如：處置股的變形、注意股門檻、借券、ETF 申購買回與成分股調整、現金增資、可轉債、除權息、財報、法說會、集保股權分散、融資融券）？
${CORE}
本輪紀錄：${dossier}
請提出最多 4 個下一輪研究題目，每個都要有事前固定的檢定設計（資料要在 ${BRIEF} 列出的範圍內，或明確說明要從 GitHub 公開資料取得），並說明為什麼值得做。`,
  { label: '完整性審查', phase: '彙整', schema: AGENDA })

// 5：（可選）執行新研究
let newResearch = []
if (args && args.run_new_research && agenda && agenda.topics.length) {
  phase('新研究')
  const topics = agenda.topics.slice(0, MAX_TOPICS)
  if (agenda.topics.length > MAX_TOPICS) log(`完整性審查員提出 ${agenda.topics.length} 題，本輪只執行前 ${MAX_TOPICS} 題`)
  newResearch = (await parallel(topics.map(t => () => agent(
    `你是台股策略研究中心的研究員。先完整閱讀並遵守 ${BRIEF}。你的資料夾：${AG}/${t.folder}/
假說：${t.hypothesis}
事前固定的檢定設計：${t.design}
程式與 results.json 放在你的資料夾。子代理不能寫報告檔：請把完整 FINDINGS 全文（結論：可用／不可用／待觀察，附數字表）放在回傳內容，由主研究員存成 FINDINGS.md。`,
    { label: `新研究:${t.folder}`, phase: '新研究' })))).filter(Boolean)
}
return { round: RUN, reviewed, views, rebuttals, report, agenda, newResearch: newResearch.map((txt, i) => ({ folder: agenda.topics[i] && agenda.topics[i].folder, findings: txt })) }
