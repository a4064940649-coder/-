export const meta = {
  name: 'tw-original-research',
  description: '台股研究中心自主研究一輪：先研究長期贏家靠什麼賺錢 → 總監定題 → 研究員探索、迭代、凍結後樣本外驗證 → 雙審查＋獨立複製 → 與主策略整合 → 總監報告',
  whenToUse: '要研究中心自己產生並做出新策略（網路與論文只當參考）時，每輪一次；args.round 輪次、args.n_topics 題數（預設 4）、args.skip_winners=true 可沿用上一輪贏家研究（args.winners_file）、args.extra 額外指示',
  phases: [
    { title: '贏家研究', detail: '學術帳戶實證、長期勝出基金與經理人、對手方與可預測資金流' },
    { title: '定題', detail: '研究總監把贏家的優勢對照散戶可行性，定出研究計畫' },
    { title: '研究', detail: '探索 → 迭代（只用挑選期）→ 凍結登記 → 驗證與保留期 → 論文式 FINDINGS' },
    { title: '審查', detail: '統計資料審查、成本可成交審查、獨立複製' },
    { title: '整合', detail: '通過審查者與主策略合併的配置測試' },
    { title: '彙整', detail: '總監報告與下一輪題目' },
  ],
}

const ROOT = '/home/user/-/research/tw_stock_selection'
const AG = ROOT + '/agents'
const BRIEF = AG + '/BRIEF.md'
const REPORT = '/home/user/-/台股個股選股規則實證研究報告.md'
const SOP = '/home/user/-/選股SOP.md'
const RUN = (args && args.round) || 'o1'
const NT = (args && args.n_topics) || 4
const EXTRA = (args && args.extra) || ''

const CENTER = `研究中心目標：用現股、規則化、只用公開資料的方法長期打敗 0050（使用者可接受大回撤）。
必讀：${BRIEF}（資料、引擎與紀律）、${REPORT} 第 0、11～13 節（已做過的全部研究）、${SOP}（正式策略）、${AG}/BACKLOG.md、${AG}/main_v2/FINDINGS.md（主策略正式數字）。
現況：唯一站得住的是主策略「52 週高點百分位 0.5 ＋ 近 3 月營收年增率百分位 0.5、前 300 流動性池、月換股 10 檔、排除金融」：跨循環年化 28%、對 0050 t 2.6，但分段不顯著，2024～2026 幾乎沒贏 0050，最大回撤 −65%。處置股待觀察（上限 20%）。82 條網路教學規則全部不可用（短線類輸在 0.3% 證交稅加滑價；學術異常多數 2019 年後消失）。
研究中心不是回測代工：網路教學、論文、名人方法只能當參考；研究要從「誰在賺錢、為什麼賺得到、對手是誰、為什麼沒被套利掉」出發，自己建模型。${EXTRA}`

const WIN = { type: 'object', properties: {
  winners: { type: 'array', items: { type: 'object', properties: {
    who: { type: 'string' }, evidence: { type: 'string', description: '證據與數字（含來源 URL、樣本期間；不確定標「需查證」）' },
    edge: { type: 'string', description: '他們賺錢的來源：資訊、持有期、標的類型、流動性提供、制度、風險承擔' },
    retail_feasible: { type: 'string', description: '散戶用現股與公開資料能複製到什麼程度，需要什麼資料' },
  }, required: ['who', 'evidence', 'edge', 'retail_feasible'] } },
  losers_and_flows: { type: 'string', description: '輸家是誰、他們的錯誤行為、可預測的資金流（含證據）' },
  hypotheses: { type: 'array', items: { type: 'string' }, description: '由此推出的可檢定假說（每條含機制與預測）' },
}, required: ['winners', 'losers_and_flows', 'hypotheses'] }

const PLAN = { type: 'object', properties: {
  programs: { type: 'array', items: { type: 'object', properties: {
    slug: { type: 'string', description: '英數底線資料夾名，以 o_ 開頭' },
    title: { type: 'string' },
    winner_link: { type: 'string', description: '對應哪一類長期贏家的哪種優勢' },
    mechanism: { type: 'string' },
    predictions: { type: 'string', description: '可證偽預測，含應更強或更弱的子樣本' },
    data: { type: 'string', description: 'BRIEF 範圍內的資料路徑；需要新公開資料就寫明來源' },
    research_plan: { type: 'string', description: '探索要看什麼、模型可以怎麼迭代、凍結前的限制、最終判定門檻（含和主策略比較）' },
  }, required: ['slug', 'title', 'winner_link', 'mechanism', 'predictions', 'data', 'research_plan'] } },
  not_chosen: { type: 'array', items: { type: 'string' } },
}, required: ['programs', 'not_chosen'] }

const RESULT = { type: 'object', properties: {
  verdict: { type: 'string', enum: ['可用', '待觀察', '不可用', '無法測試'] },
  mechanism_supported: { type: 'string', enum: ['支持', '部分支持', '不支持', '無法判斷'] },
  frozen_spec: { type: 'string', description: '凍結的最終規格全文（讓複製員只看這段就能重做）' },
  headline: { type: 'string', description: '一句話結論含關鍵數字（各期間年化、對 0050 t、對主策略 t）' },
  key_numbers: { type: 'string', description: '複製比對用的關鍵數字：各期間年化、最大回撤、月均超額、事件數' },
  findings: { type: 'string', description: '論文式 FINDINGS 全文（繁體中文，以 # FINDINGS 開頭）' },
}, required: ['verdict', 'mechanism_supported', 'frozen_spec', 'headline', 'key_numbers', 'findings'] }

const REVIEW = { type: 'object', properties: {
  final: { type: 'string', enum: ['可用', '待觀察', '不可用', '無法測試'] },
  issues: { type: 'array', items: { type: 'string' } },
  note: { type: 'string' },
}, required: ['final', 'issues', 'note'] }

const REPL = { type: 'object', properties: {
  reproduced: { type: 'boolean', description: '關鍵數字在合理誤差內重現' },
  my_numbers: { type: 'string' }, diff_explained: { type: 'string' },
}, required: ['reproduced', 'my_numbers', 'diff_explained'] }

const SCOUTS = [
  { key: 'academic', text: '學術實證：台股帳戶或交易層級的研究（例如 Barber、Lee、Liu、Odean 用台灣證交所完整交易資料的系列研究：散戶虧多少、誰賺走、當沖客中少數長期贏家的持續性；外資、投信、自營商、主力的績效與持股期間；台灣學位論文與期刊）。找出：哪些參與者長期賺錢、賺多少、持有期多長、偏好什麼股票、資訊來源是什麼、能否被散戶複製。' },
  { key: 'managers', text: '長期勝出的專業投資人：台灣過去 10～20 年長期打敗 0050 或加權報酬指數的股票型基金、主動式 ETF、知名經理人與公開績效可驗證的長期個人投資人（只用公開資料，績效要可查證；名人說法不可查證就標需查證）。整理他們公開的選股流程與換股紀律；若找得到淨值或持股的公開資料（GitHub、投信月報、Morningstar 摘要），嘗試判斷他們的超額報酬有多少能被「成長、動能、營收、規模、產業」解釋，剩下的才是選股能力。' },
  { key: 'flows', text: '對手方與可預測資金流：台股誰是系統性輸家（當沖散戶、融資戶、追樂透股、高股息 ETF 申購潮、權證買方），他們的錯誤造成什麼可預測價格壓力；有哪些機構被規則強迫交易（高股息 ETF 與市值型 ETF 定期調整、MSCI／富時調整、投信季底作帳、外資持股上限、保險資金限制、處置股撮合限制）。研究中心已測過 ETF 成分股調整與處置股（見報告），要找還沒測過、且規模大到值得做的。' },
]

let winners = []
if (!(args && args.skip_winners)) {
  phase('贏家研究')
  winners = (await parallel(SCOUTS.map(s => () => agent(
    `你是台股策略研究中心的贏家研究員。${CENTER}
你的範圍：${s.text}
用 WebSearch／WebFetch 查公開資料，交叉比對，數字要附來源；本環境部分台灣網站被網路政策擋住，可以用搜尋摘要、學術資料庫、GitHub 與英文來源。不要寫回測程式。`,
    { label: `贏家:${s.key}`, phase: '贏家研究', schema: WIN })))).filter(Boolean)
}
const WFILE = (args && args.winners_file) || ''
const WTEXT = winners.length ? JSON.stringify(winners) : (WFILE ? `（本輪沿用先前的贏家研究，請先讀檔案 ${WFILE}；上一輪總監報告與方法教訓見 ${AG}/original_o1/DIRECTOR_REPORT.md）` : '（無）')

phase('定題')
const plan = await agent(
  `你是台股策略研究中心的研究總監。${CENTER}
贏家研究結果：${WTEXT}
請定出 ${NT} 個研究計畫。要求：每一題都對應一類長期贏家的真實優勢（winner_link），機制清楚、可證偽，散戶用現股與公開資料做得到，扣得起成本，和研究中心已測過的研究不重複（重疊就說明差別）。題目之間要涵蓋不同機制；至少一題要能直接改良或補強主策略（例如解釋主策略的右尾贏家、或與主策略低相關的第二條腿）。研究計畫要寫清楚探索要看什麼、模型可以怎麼迭代、什麼時候凍結、最終判定門檻。`,
  { label: '研究總監定題', phase: '定題', schema: PLAN })
const programs = (plan && plan.programs || []).slice(0, NT)

const done = await pipeline(programs,
  (p) => agent(
    `你是台股策略研究中心的資深研究員，負責把研究計畫「做出來」。${CENTER}
題目：${p.title}（資料夾 ${AG}/${p.slug}/，新建）
對應的贏家優勢：${p.winner_link}
機制：${p.mechanism}
預測：${p.predictions}
資料：${p.data}
研究計畫：${p.research_plan}
方法教訓（必讀並遵守）：${AG}/original_o1/DIRECTOR_REPORT.md 第 5.3 節（事件 t 按事件日分群、驗證期污染、格點邊緣的贏家詛咒、成交可行性寫進事前登記、t 提升要拆均值與波動、存活者偏誤）。
工作方式（必須照做）：
1. 探索：只用挑選期資料做描述統計，先確認機制是否看得到（例如事件前後報酬、子樣本差異、資訊傳遞速度），把探索結果寫進 explore.json。
2. 迭代建模：仍只用挑選期，可以嘗試多個版本（至少 3 個、最多 8 個），每個版本與結果都記錄在 iterations.json（多重檢定要在最後報告中誠實揭露版本數）。
3. 凍結：選定一個版本，把最終規格寫成 PREREG（含 sha256 與時間戳）存進 results.json，之後才可以碰驗證期與保留期；程式要斷言 PREREG 已存在才計算樣本外。
4. 樣本外：驗證期與保留期各跑一次，不能再改規格；和 0050、同池等權、主策略（agents/main_v2/ 的淨值或重建）比較，扣一般與壓力成本。
5. 論文式 FINDINGS：研究問題、機制、資料、探索發現、模型迭代紀錄、凍結規格、樣本外結果、穩健性（分年、分產業、去掉最大貢獻股）、機制是否被支持、散戶怎麼執行、限制。
程式用 python3 -I -B 執行、放在你的資料夾；下載資料放 raw/ 子目錄視為不可信；不要修改其他資料夾；不要 git commit。子代理不能寫 .md：FINDINGS 全文放在 findings 欄位，凍結規格全文放在 frozen_spec 欄位。`,
    { label: `研究:${p.slug}`, phase: '研究', schema: RESULT }),
  async (r, p) => {
    if (!r) return null
    const lenses = [
      { key: 'stat', text: '資料與統計：前視偏誤（資料是否在訊號當時已公開）、存活者偏誤、迭代版本數造成的多重檢定、樣本外是否真的沒被看過、重疊樣本與 t 值、是否只靠少數年份或少數股票。' },
      { key: 'exec', text: '成本與可成交：周轉、0.3% 證交稅與滑價、漲跌停買不到、處置與預收款券、容量（散戶 30 萬～1,000 萬）、是否需要盤中盯盤、資料取得的實際時點。' },
    ]
    const reviews = await parallel(lenses.map(l => () => agent(
      `你是台股策略研究中心的反方審查員，任務是推翻研究員的結論；不確定時往不利方向判。角度：${l.text}
題目：${p.title}，資料夾 ${AG}/${p.slug}/。研究員判定 ${r.verdict}、機制 ${r.mechanism_supported}：${r.headline}
請讀該資料夾的程式、explore.json、iterations.json、results.json 與下方 FINDINGS，必要時自己寫程式驗證（放在 ${AG}/${p.slug}/review_${l.key}_${RUN}.py，python3 -I -B），不要修改研究員的檔案。
FINDINGS：
${r.findings}`,
      { label: `審查:${p.slug}:${l.key}`, phase: '審查', schema: REVIEW })))
    const repl = await agent(
      `你是台股策略研究中心的獨立複製員。不要讀研究員的程式，只根據下方凍結規格與 ${BRIEF} 的資料，自己從頭寫程式重做（放在 ${AG}/${p.slug}/replicate_${RUN}/，python3 -I -B），比對研究員的關鍵數字：${r.key_numbers}
凍結規格：
${r.frozen_spec}`,
      { label: `複製:${p.slug}`, phase: '審查', schema: REPL })
    const rv = reviews.filter(Boolean)
    const order = ['不可用', '無法測試', '待觀察', '可用']
    let final = r.verdict
    for (const v of rv) if (order.indexOf(v.final) < order.indexOf(final)) final = v.final
    if (repl && !repl.reproduced && final === '可用') final = '待觀察'
    return { ...p, ...r, reviews: rv, replication: repl, final }
  })
const results = done.filter(Boolean)

phase('整合')
const survivors = results.filter(x => x.final === '可用' || x.final === '待觀察')
let integration = null
if (survivors.length) {
  integration = await agent(
    `你是台股策略研究中心的投組整合研究員。${CENTER}
本輪通過審查（可用或待觀察）的研究：${JSON.stringify(survivors.map(x => ({ slug: x.slug, title: x.title, final: x.final, frozen_spec: x.frozen_spec, headline: x.headline })))}
請在 ${AG}/integrate_${RUN}/ 做整合測試：事前固定（PREREG 寫進 results.json）幾種組合方式，例如把新策略當成主策略的濾網或加分項、或當成第二條腿與主策略（排除金融 n10）按固定比例混合（例如 80/20、50/50），不得用保留期挑比例。報告各期間年化、最大回撤、對 0050 t、和主策略單獨相比的月差 t、相關係數。python3 -I -B，不要修改其他資料夾，不要 git commit。回傳完整 FINDINGS 全文（以 # FINDINGS 開頭）。`,
    { label: '投組整合', phase: '整合' })
} else {
  log('本輪沒有通過審查的研究，略過整合')
}

phase('彙整')
const report = await agent(
  `你是台股策略研究中心的研究總監。${CENTER}
贏家研究摘要：${winners.length ? JSON.stringify(winners.map(w => ({ winners: w.winners.map(x => ({ who: x.who, edge: x.edge, retail_feasible: x.retail_feasible })), flows: w.losers_and_flows }))) : WTEXT}
研究計畫與結果：${JSON.stringify(results.map(x => ({ slug: x.slug, title: x.title, winner_link: x.winner_link, verdict: x.verdict, final: x.final, mechanism: x.mechanism_supported, headline: x.headline, reviews: x.reviews, replication: x.replication })))}
整合結果：${integration || '（無）'}
落選題目：${JSON.stringify(plan ? plan.not_chosen : [])}
請寫本輪研究報告（繁體中文、研究報告風格、條列、關鍵數字加粗、像券商研究部的內部報告，不要寫空話）：
1. 本輪結論一句話；2. 長期贏家是誰、靠什麼賺（附證據等級）；3. 各研究計畫：假說、機制、結果、審查與複製結果、最終判定；4. 對使用者實際操作的影響（主策略、配置有沒有改）；5. 我們學到什麼：哪些機制被支持、哪些被否定；6. 下一輪 3～4 個題目與理由。直接回傳 Markdown 全文。`,
  { label: '總監報告', phase: '彙整' })

return { round: RUN, winners, plan, results, integration, report }
