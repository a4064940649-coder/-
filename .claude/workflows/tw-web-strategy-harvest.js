export const meta = {
  name: 'tw-web-strategy-harvest',
  description: '收集網路上台股技術分析、產業研究、籌碼、基本面教學的具體規則，逐條用台股資料回測並由反方審查',
  whenToUse: '研究中心要持續從網路教學找新策略時，每輪執行一次；args.round 輪次、args.max_tests 每輪最多測幾條、args.tested 已測過的規則清單',
  phases: [
    { title: '搜集', detail: '多位偵查員從不同類型的網路來源抽出可回測規則' },
    { title: '測試', detail: '每條規則一位研究員寫程式回測，一位反方審查' },
    { title: '交叉', detail: '審查後仍是可用／待觀察的規則，和主策略交叉組合測試' },
    { title: '彙整', detail: '主研究員整理本輪結果與下一輪方向' },
  ],
}

const AG = '/home/user/-/research/tw_stock_selection/agents'
const BRIEF = AG + '/BRIEF.md'
const RUN = (args && args.round) || 'w1'
const MAXT = (args && args.max_tests) || 6
const TESTED = (args && args.tested) || []
const GIVEN = (args && args.rules) || null   // 直接給規則清單（例如上一輪的 leftover）就跳過偵查

const RULES = { type: 'object', properties: { rules: { type: 'array', items: { type: 'object', properties: {
  slug: { type: 'string', description: '英數底線的資料夾名' },
  name: { type: 'string' },
  category: { type: 'string', enum: ['technical', 'industry', 'chips', 'fundamental', 'event', 'other'] },
  rule: { type: 'string', description: '完全可程式化的進出場規則，含所有參數、持有期、股池' },
  sources: { type: 'array', items: { type: 'string' }, description: '教學來源 URL' },
  popularity: { type: 'string', description: '在台灣網路上有多常被推薦（依據）' },
  data_ok: { type: 'boolean', description: '是否能用 BRIEF 列出的資料回測' },
  data_note: { type: 'string' },
}, required: ['slug', 'name', 'category', 'rule', 'sources', 'popularity', 'data_ok', 'data_note'] } } }, required: ['rules'] }
const RESULT = { type: 'object', properties: {
  verdict: { type: 'string', enum: ['可用', '待觀察', '不可用', '無法測試'] },
  headline: { type: 'string', description: '一句話結論含關鍵數字（挑選期／驗證期淨年化、對 0050 t、對等權 t）' },
  findings: { type: 'string', description: '完整 FINDINGS.md 全文（繁體中文，以 # FINDINGS 開頭）' },
}, required: ['verdict', 'headline', 'findings'] }
const REVIEW = { type: 'object', properties: {
  final: { type: 'string', enum: ['可用', '待觀察', '不可用', '無法測試'] },
  issues: { type: 'array', items: { type: 'string' } },
  note: { type: 'string' },
}, required: ['final', 'issues', 'note'] }

const SCOUTS_TW = [
  { key: 'ta', text: '台灣網路上的技術分析教學：PTT Stock 板精華、Mobile01、Dcard 理財、知名部落格（例如：股市阿水、雷浩斯、春哥、財報狗、CMoney 教學）、YouTube 教學影片的文字摘要、券商學院、技術分析書籍（葛蘭碧八大法則、K 線型態、量價關係、均線糾結、KD 鈍化、W 底／頭肩底突破、海龜、通道、跳空缺口、波浪、費波那契等）。' },
  { key: 'fund', text: '台灣網路上的產業研究、基本面與籌碼教學：營收創新高、EPS／毛利率成長、本益比河流圖、殖利率法、產業循環（記憶體、面板、航運、被動元件報價）、庫存循環、法說會、董監持股與大戶持股、投信認養、外資連買、融資融券（融資減券增、券資比）、主力分點、當沖比率、注意股、現金增資、庫藏股、股東會紀念品等。' },
  { key: 'pop', text: '台灣散戶最常被推薦、或是近年最紅的「選股公式／存股法／飆股法則」：例如 CMoney 與財經 YouTuber 常見的選股條件組合、「三率三升」、「月營收連續 N 月成長」、「均線多頭排列＋量增」、「投信首買」、「低基期」、「起漲點」等；也包括國外流行但台灣常引用的方法（CAN SLIM、Minervini 趨勢模板、Piotroski F-score、魔術公式）。' },
]
const SCOUTS_GLOBAL = [
  { key: 'academic', text: '全球學術與量化資料庫整理的股市異常與策略：Quantpedia（策略清單與 SSRN 論文摘要）、SSRN／arXiv 上的實證論文、Alpha Architect、AQR、Research Affiliates、Two Sigma 文章、《151 Trading Strategies》、Robeco 因子研究；特別找「在台灣或新興市場也被驗證過」、而且研究中心還沒測過的（例如：盈餘公告漂移 PEAD、分析師修正、股票回購、內部人買進、季節性 SUE、短期反轉、Amihud 非流動性、波動率擠壓、相對成交量、價格延遲 price delay、連續上漲天數、52 週低點反彈、異常成交量突破等）。' },
  { key: 'en_ta', text: '英文世界的交易教學網站與影片：Investopedia、StockCharts ChartSchool、TradingView 熱門腳本與教學、QuantConnect／Quantopian 社群策略、r/algotrading、r/RealDayTrading、Larry Connors（RSI(2)、Connors RSI、雙重 7 日）、Linda Raschke、Mark Minervini 以外的 VCP、Stan Weinstein 階段分析、Nick Radge、Andreas Clenow 動能、Tom Hougaard 等；YouTube 熱門交易影片（Rayner Teo、The Trading Channel、Trading Rush、Ross Cameron、Tom Bowley 等）只能讀其逐字稿、文字摘要或影片說明欄，請找這些文字版本。' },
  { key: 'asia', text: '日本、中國大陸、韓國、香港的股市教學與量化社群：日本（株探、みんかぶ、ストップ高翌日、新高値ブレイク投資術、移動平均乖離、信用倍率、IPO セカンダリー）、中國大陸（聚寬 JoinQuant、優礦、米筐、BigQuant 公開策略、A 股常見的漲停板戰法、首板二板、龍虎榜、北向資金、小市值輪動、高股息低波）、韓國（외국인 순매수、상한가 따라잡기）。抽出在台股制度下（10% 漲跌幅、T+2、現股當沖、處置與注意股、無放空限制差異）仍可執行的規則。' },
]
const SCOUTS = (args && args.scout_set === 'global') ? SCOUTS_GLOBAL : SCOUTS_TW

phase('搜集')
const scouted = GIVEN ? [{ rules: GIVEN }] : await parallel(SCOUTS.map(s => () => agent(
  `你是台股策略研究中心的偵查員。任務：用 WebSearch 與 WebFetch（必要時用 ToolSearch 載入）在網路上找 ${s.text}
從教學內容中抽出「可以完全寫成程式」的具體規則：進場條件、出場條件、所有參數（若教學沒給參數，採用教學最常見的預設值並註明）、股池與持有期。每條附來源 URL 與它多常被推薦的依據。
先閱讀 ${BRIEF} 的資料清單，判斷能不能用現有資料回測（data_ok）：面板沒有最低價、沒有分點資料；融資融券與股本只有 2025-09 起；三大法人、月營收 2021 起（長期月營收 2005 起）；有當沖比率、注意／處置名單、除權息。
以下規則研究中心已經測過，不要重複（除非教學的版本明顯不同，並說明差異）：${JSON.stringify(TESTED)}
回傳 6～10 條，優先挑「很多人在用、而且能測」的。slug 用英數底線且以 ${RUN}_ 開頭。`,
  { label: `偵查:${s.key}`, phase: '搜集', schema: RULES })))
const seen = new Set()
const pool = scouted.filter(Boolean).flatMap(r => r.rules).filter(r => r.data_ok).filter(r => {
  const k = r.slug.toLowerCase(); if (seen.has(k)) return false; seen.add(k); return true })
// 輪流從三類取，避免全部是同一類
const byCat = {}
pool.forEach(r => { (byCat[r.category] = byCat[r.category] || []).push(r) })
const picks = []
while (picks.length < MAXT && Object.values(byCat).some(a => a.length)) for (const c of Object.keys(byCat)) { if (byCat[c].length && picks.length < MAXT) picks.push(byCat[c].shift()) }
log(`偵查到 ${scouted.filter(Boolean).flatMap(r => r.rules).length} 條，可測 ${pool.length} 條，本輪測 ${picks.length} 條；其餘 ${pool.length - picks.length} 條留給下一輪`)
const leftover = pool.filter(r => !picks.includes(r))

phase('測試')
const results = await pipeline(picks,
  (r) => agent(
    `你是台股策略研究中心的研究員。先完整閱讀並遵守 ${BRIEF}（特別是「第 2 輪後補充」：可成交引擎、2018-01-13 缺漏、統計口徑）。你的資料夾：${AG}/${r.slug}/（自己建立；python3 -I 執行；不要改其他資料夾；不要 git commit）。
要測的網路教學規則：「${r.name}」（${r.category}）
規則：${r.rule}
來源：${r.sources.join(' ')}
資料備註：${r.data_note}
要求：
1. 先把規則逐字轉成程式碼前的事前設定（參數、股池、進出場時點、持有期）寫在 results.json 的 PREREG，看結果前固定；教學沒給的參數用最常見預設，不准看結果後調參。
2. 盤後資料最早隔日開盤成交；鎖漲停買不到；成本照 exlib（滑價 0.1% 與壓力情境）。事件型規則以事件報酬（扣成本、對上市等權的異常報酬、t 按進場月份分群）報告；選股型規則以組合報告（年化、日頻最大回撤、對 0050 t、對同池等權 t、逐年）。
3. 期間依 BRIEF：價格類 2005-01～2019-04 挑選、2019-05～2026-10 驗證；2021 起資料 2021～2023 挑選、2024～2026 驗證。兩段都要報；另報 2024～2025。
4. 判定：挑選期與驗證期扣成本後都勝同池等權（t ≥ 2）且至少不輸 0050 → 可用；只有一段過 → 待觀察；否則不可用。資料不足 → 無法測試。
回傳 verdict、headline（一句話含數字），以及 findings（完整 FINDINGS 全文，以「# FINDINGS」開頭；子代理不能寫 .md 檔，由主研究員存檔）。`,
    { label: `測試:${r.slug}`, phase: '測試', schema: RESULT }),
  (res, r) => res ? agent(
    `你是台股策略研究中心的反方審查員。研究員回測了網路教學規則「${r.name}」，資料夾 ${AG}/${r.slug}/（讀程式與 results.json，可自己寫小程式驗證，用 python3 -I；不要改檔）。
研究員的判定：${res.verdict}；${res.headline}
請專挑毛病：前視偏誤（盤後資料當天用、未來資訊）、成本與鎖漲停、樣本太小、只靠少數極端事件、2019～2023 存活者偏誤、src=3 近似收盤、事後調參、t 值口徑。不確定就往保守判。回傳最終判定與問題清單。`,
    { label: `審查:${r.slug}`, phase: '測試', schema: REVIEW }).then(v => ({ rule: r, result: res, review: v })) : null)
const done = results.filter(Boolean)

phase('交叉')
const keep = done.filter(d => d.review && ['可用', '待觀察'].includes(d.review.final))
const combos = keep.length ? (await parallel(keep.map(d => () => agent(
  `你是台股策略研究中心的研究員，負責「交叉運用」。先讀 ${BRIEF}。網路教學規則「${d.rule.name}」回測後判為 ${d.review.final}（${d.result.headline}；程式在 ${AG}/${d.rule.slug}/）。
把它和中心主策略交叉，事前固定以下三種方式並全部報告（程式放 ${AG}/${d.rule.slug}/cross_*.py，python3 -I，不要改其他資料夾）：
(a) 第二層篩選：主策略（52 週高點 × 月營收 3 個月年增率，引擎 ${AG}/revenue_longhistory/lhlib.py 或 ${AG}/exec_rebuild/exlib.py 的 M2）先選前 60 名，再用這條規則的訊號排序取前 30／前 10；
(b) 排名合成：把這條規則的訊號轉成百分位，和主策略兩條腿等權平均（三腿各 1/3）；
(c) 狀態切換：只在這條規則認定的有利狀態下持有主策略，其餘持現金或 0050（若規則不是市場層級訊號，就略過並說明）。
期間：若資料能回到 2006 年，挑選 2006-04～2013-12、驗證 2014～2020、保留期 2021-03～2026-10；否則挑選 2021～2023、驗證 2024～2026。和原主策略做配對 t（月窗）。
回傳 verdict（可用＝驗證期與保留期都勝原主策略且配對 t ≥ 2）、headline 與完整 findings（以「# FINDINGS」開頭的交叉結果全文）。`,
  { label: `交叉:${d.rule.slug}`, phase: '交叉', schema: RESULT }).then(c => c ? { slug: d.rule.slug, ...c } : null)))).filter(Boolean) : []
if (!keep.length) log('本輪沒有可用／待觀察的規則，略過交叉')

phase('彙整')
const report = await agent(
  `你是台股策略研究中心的主研究員。本輪（${RUN}）從網路教學收集規則並回測，結果如下（JSON）：${JSON.stringify(done.map(d => ({ slug: d.rule.slug, name: d.rule.name, category: d.rule.category, sources: d.rule.sources, verdict: d.result.verdict, headline: d.result.headline, review: d.review })))}
與主策略交叉的結果：${JSON.stringify(combos.map(c => ({ slug: c.slug, verdict: c.verdict, headline: c.headline })))}
尚未測試、留到下一輪的規則：${JSON.stringify(leftover.map(r => r.name))}
寫一份繁體中文、研究報告風格的本輪摘要（條列、關鍵數字加粗）：1) 一句話結論；2) 每條規則一列的結果表（規則、來源、研究員判定、審查後判定、關鍵數字）；3) 有沒有任何規則值得加入目前配置（目前配置：52 週高點 × 月營收選股＋處置股 20%＋0050，見 /home/user/-/台股個股選股規則實證研究報告.md 第 0 節）；4) 為什麼網路教學大多無效或有效的共同原因；5) 下一輪建議。直接回傳 Markdown 全文。`,
  { label: '彙整', phase: '彙整' })
return { round: RUN, picks: picks.map(r => r.slug), leftover, combos, results: done.map(d => ({ slug: d.rule.slug, name: d.rule.name, category: d.rule.category, sources: d.rule.sources, verdict: d.result.verdict, headline: d.result.headline, findings: d.result.findings, review: d.review })), report }
