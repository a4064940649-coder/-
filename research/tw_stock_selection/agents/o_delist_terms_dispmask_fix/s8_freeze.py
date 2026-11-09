#!/usr/bin/env python3
"""步驟 8：凍結 SPEC.json 與 results.json PREREG（sha256＋written_utc）。凍結前沒有用 v3 分類或新遮罩計算任何報酬
（唯一例外：SPEC 要求的事前可行性檢查 feasibility.json 與重現已公開數字的煙霧測試）。python3 -I -B s8_freeze.py"""
import json, hashlib, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; AG = HERE.parent
SC = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/o18_delistterms')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (HERE / 'results.json').exists(), '只凍結一次'
DR = json.loads((HERE / 'SPEC_draft.json').read_text(encoding='utf-8'))
CL = json.loads((HERE / 'classify.json').read_text(encoding='utf-8'))
FE = json.loads((HERE / 'feasibility.json').read_text(encoding='utf-8'))
SI = json.loads((HERE / 'smoke_ia.json').read_text(encoding='utf-8'))
SV = json.loads((HERE / 'smoke_v7x.json').read_text(encoding='utf-8'))
MS = json.loads((HERE / 'market_screen.json').read_text(encoding='utf-8'))
files = ['SPEC_draft.json', 'event_list_frozen.csv', 'market_screen.json', 'search_log_web.csv', 'search_log_local.csv', 'events_v3.csv', 'classify.json',
         'feasibility.json', 'smoke_ia.json', 'smoke_v7x.json', 'mkt.py', 'cz3.py', 'dispmask_official.py', 'ia_lib.py', 'v7plans.py', 'ia_core_dispmask_official.patch',
         's2_event_list.py', 's3_market.py', 's4_grep.py', 's5_classify.py', 's6_feasibility.py', 's7_smoke_ia.py', 's7b_smoke_v7x.py', 's8_freeze.py']
ext = {'ia_core.py': AG / 'independent_audit' / 'ia_core.py', 'cz_engine.py': AG / 'o_engine_delist_consideration' / 'cz_engine.py',
       'cz_v7x.py': AG / 'o_engine_delist_consideration' / 'cz_v7x.py', 'events_v2.csv': AG / 'o_engine_delist_consideration' / 'events_v2.csv',
       'x_exec.py': AG / 'o_exec_engine_rewrite_v7x' / 'x_exec.py', 'x_data.py': AG / 'o_exec_engine_rewrite_v7x' / 'x_data.py',
       'x_stats.py': AG / 'o_exec_engine_rewrite_v7x' / 'x_stats.py', 'navs_oos_R4.parquet': AG / 'o_exec_engine_rewrite_v7x' / 'out' / 'navs_oos_R4.parquet',
       'disposal_official_universe.csv': AG / 'o_rev_avail_ds_pit_count' / 'raw' / 'disposal_official_universe.csv',
       'navs_gapfix.parquet': AG / 'main_v2' / 'navs_gapfix.parquet', 'v2_core.py': AG / 'main_v2' / 'v2_core.py',
       'integrate_o17_results.json': AG / 'integrate_o17' / 'results.json',
       'plans_orig.csv': SC / 'plans_orig.csv', 'plans_union.csv': SC / 'plans_union.csv', 'plans_replace.csv': SC / 'plans_replace.csv', 'plans_union_noovl.csv': SC / 'plans_union_noovl.csv'}
P1_lock = {c: MS[c]['market'] for c in ['3007', '6280', '1520', '1716', '8008', '8078', '3559']}
PRE = dict(
 topic='o18 題 2：下市對價條款補核（市場鎖價初篩＋公開條款原文）＋ia_core 處置遮罩改官方全歷史清單（基礎設施勘誤；不產生選股訊號，S1）',
 spec_draft=dict(file='SPEC_draft.json', sha256=sha(HERE / 'SPEC_draft.json'), written_utc=DR['written_utc'], note='判準與狀態表在計算市場證據與搜尋之前寫下'),
 event_list=DR['event_list'],
 market_criteria=DR['market_criteria'],
 text_search=dict(DR['text_search'], hosts_unreachable=['mops.twse.com.tw', 'mopsov.twse.com.tw', 'emops.twse.com.tw', 'www.twse.com.tw', 'web.archive.org', 'zh.wikipedia.org', 'news.cnyes.com', 'www.cna.com.tw', 'money.udn.com'],
                  hosts_reachable=['raw.githubusercontent.com（200）', 'api.github.com 403（未認證限流）', 'WebSearch 工具（伺服器端）'], n_web_queries=33, n_local_sources=10),
 statuses_final={
   'VERIFIED_TEXT': '數字化對價（最終下市步驟）有可指名出處 → 依條款；lo／hi 不加 δ',
   'TENDER_TEXT': '只有前一階段公開收購價有數字，最終合併對價無數字 → 固定請求權 lc×(1+δb)（與 MARKET_LOCK 同）',
   'TYPE_TEXT': '只有類型（現金收購／換股／轉換）→ 依市場證據：TRACK→比例×(1+δb)換存續股；換股但無法估比例或 NONE→固定 lc×(1+δb)；轉換新設控股（1716→3705）→空窗期固定 lc，s 日以 lc×(1+δb)÷存續首日開盤換股',
   'SEARCHED_NOT_FOUND': '有市場證據→同上；無→lo＝0、main＝hi＝lc（舊 unknown）',
   'NOT_SEARCHED': '期末 0 起（斷言）',
   'conflict': '文字隱含 lt 價值與 lc 偏離 >10% → 採不利方向；本題 0 起（最大 2394 −9.5%、2833 −9.5%，皆依文字＝較不利）'},
 delta=dict(rule=DR['delta_rule'], value=CL['delta'], pre2013=CL['delta_pre2013'], dmax_code=CL['dmax_code'], devs=CL['devs'],
            note='2394／2833 的 −9.5% 來自文字比例早於存續公司除權（未調整）；照規則納入，δ 由 0.057 放寬為 0.096（2013 前 0.12）'),
 events_v3=dict(file='events_v3.csv', sha256=CL['events_v3_sha256'], counts=CL['counts'], text_counts=CL['text_counts'],
                v7x20_verified=CL['v7x20_verified'], uopt6_verified=CL['uopt6_verified'], not_searched=CL['not_searched']),
 dispmask=dict(module='dispmask_official.py', patch='ia_core_dispmask_official.patch', patch_sha256=sha(HERE / 'ia_core_dispmask_official.patch'),
               ia_modes={'orig': 'DISP_OFFICIAL=None（＝現行 ia_core）', 'union': '＝integrate_o17 C_DSP（2006-01-01～2021-09-23 原遮罩 ∪ 官方訊號日或成交日）', 'replace': '2014-01-01～2021-09-23 只用官方（訊號日或成交日）'},
               ia_extra={'union_noovl': '只用 overlap_proxy=False 的官方列（P5 對照；monkeypatch）', 'union14': '視窗 2014-01-01～2021-09-23'},
               v7x='v7plans.py 重建 AB_V7_B10 13 j × 20 seeds；orig＝DMT（V4 官方名單 trade 口徑，公告日 ≤ 訊號日）逐筆重現 plans_j7；union＝DMT ∪ 官方訊號日或成交日（2006～2021-09-23）；replace＝2014～2021-09-23 只用官方',
               v7x_note='研究員事前提醒：V7X 名單本來就用 V4 官方處置名單（成交日口徑）排除，與 ia_core 不同；V7X Δ 只來自「訊號日在處置、成交日不在」與 DMT 漏列，預期遠小於紙上'),
 smoke=dict(ia=dict(flag_off_maxabs=SI['flag_off_vs_main_v2_maxabs'], C_DSP=SI['C_DSP_reproduce'], n_union_out=SI['n_union_out'], n_replace_out=SI['n_replace_out'],
                     C0_official_sig_2014_2021=len(SI['C0_slots_in_official_sig_2014_2021']), C0_official_trd_2014_2021=len(SI['C0_slots_in_official_trd_2014_2021'])),
            v7x=SV['table'], v7x_plans_orig_equal_plans_j7=True),
 feasibility=dict(file='feasibility.json', best_lo=FE['table']['best_lo'], worst_lo=FE['table']['worst_lo'], triggerable=FE['triggerable'],
                  text='最佳情境 lo（未核實事件全部固定 lc×(1−δ_evt)、不歸零）V7X 100 萬一般 j 中位數全期超額 +1.04pp > 0 → 下界回到 > 0「可以觸發」，列為檢驗；最壞情境（非 VERIFIED_TEXT 全歸零）−1.00pp'),
 formal_run=dict(
   ia=dict(lists=['C0×{orig, union, replace, union_noovl, union14}', 'U_opt×{orig, union}'], engines=['freeze', 'v3_main', 'v3_lo', 'v3_hi', 'o17 cons_main（對照）'], costs=['normal', 'stress'],
           stats='integrate_o17 win()（245 窗、END 2026-10-12）：全期年化、對 0050 超額與 t、三段、2019 起、2014～2019、拿掉最好 3 年 t、逐年差與逐年絕對差'),
   v7x=dict(plans=['orig', 'union', 'replace'], engines={'orig': ['orig', 'v3_main', 'v3_lo', 'v3_hi', 'o17_cons_main(100萬一般)'], 'union': ['orig', 'v3_main', 'v3_lo', 'v3_hi'], 'replace': ['orig', 'v3_main']},
            cells=['normal|100萬', 'stress|100萬', 'normal|300萬', 'normal|1000萬'], seeds=20, js=13, fields=['CAGR_med', 'exc_med', 't_0050_med', 'exc_p10', 'exc_Y19on_med', 't_Y19on_med', 'exc_Y19on_p10'],
            yearly='逐年 j 中位數差與逐年絕對差'),
   once='s9a_ia.py、s9b_v7x.py 各只跑一次（斷言 PREREG 雜湊、RESULTS 段不存在）'),
 predictions=dict(
   P1=dict(text='7 起 unknown 至少 5 起鎖價；2432／2403／2452 不鎖價且 R²≥0.6 判換股；5 起未搜尋全部有紀錄、期末未搜尋 0',
           evaluated_before_freeze=True, result=dict(unknown7=P1_lock, n_lock=sum(v == 'MARKET_LOCK' for v in P1_lock.values()),
           swap3={c: (MS[c]['market'], round(MS[c]['track_R2'], 3)) for c in ['2432', '2403', '2452']}, not_searched_end=CL['not_searched']),
           verdict='部分不成立：鎖價 4／7（3007 0.7／0.9%、6280 0.7／2.6%、1716 0.5 未過 80%）；2452 R² 0.569 未過 0.6；未搜尋 0（成立）'),
   P2=dict(text='V7X 20 起核實 4→≥10；U_opt 6 起 0→≥3；V7X 核實 <8 → 來源不足、待觀察', evaluated_before_freeze=True,
           result=dict(v7x=len(CL['v7x20_verified']), uopt=len(CL['uopt6_verified'])), verdict='成立（10／20、4／6），但新核實 6 起全部來自 WebSearch 摘要（非逐字原文），見 sensitivity'),
   P3=dict(text='V7X 100 萬一般 j 中位數主口徑全期超額 1.30～1.65pp；下界 −0.50～+0.10pp；P(下界 > 0) ≤ 30%'),
   P4=dict(text='官方清單 2014～2021-09-23 多出 C0 訊號日槽 11、成交日槽 10（逐槽重現 o17，已在 smoke 成立）；移出槽下一期平均報酬約 +12%；移入−移出合計約 −12.5pp、2019 起約 −45pp；聯集版 C0 全期 −0.68pp（≤0.01）、2019 起 −2.65pp；替換版與聯集版差 ≤0.2pp；V7X 13 j 全期 Δ −1.6～0pp、2019 起 −4.5～0pp（集中 2020-08、2021-06）、2014～2019 |Δ| ≤ 0.3pp；300 萬、1,000 萬方向相同'),
   P5=dict(text='2014～2021 處置列中代理推得比例 ≥ 40%；只用官方起訖日列時 C0 |Δ| 縮小（不利端：不變）；若 |Δ| 變大＝代理期間低估覆蓋',
           note='事前已知（smoke）：overlap_proxy=True 比例 60.5%，但 overlap_proxy 不是「代理推得」，全部列的起訖都是官方公告；本題把 P5 改讀為「排除與代理事件重疊的列」的對照，並標明前提不成立'),
   P6=dict(text='26 個大型 pkl 中 ≥ 20 個含永久下市部位；V7X run_R0～R4c 凍結資產最大占比 ≥ 5%；普查總數 2,075 組上修為完整數')),
 decision=dict(
   a='V7X 加註：v3_lo（100 萬一般 j 中位數全期超額）> 0 且 v3_hi > 0 → 改寫為「下市對價口徑差：主口徑 X、界限 lo～hi，均 > 0」；否則只更新數字並寫明界限由哪些事件決定',
   b='處置遮罩：union 對 orig，主策略任一正式數字（C0 紙上全期／三段／2019 起年化與 t、V7X 各欄）|Δ年化| > 0.3pp 或 |Δt| > 0.05 → 加註「處置遮罩官方化：Δ…，集中在…訊號日」；題 1 若取代正式數字則附在新數字旁；題 1 C 濾網須＝dispmask_official union（提供 assert_union_equal 斷言函式）',
   c='分級：可用＝VERIFIED_TEXT ≥ 50% V7X 結算事件 且 NOT_SEARCHED＝0 且 獨立複製一致；不可用＝patch 煙霧測試失敗；其他待觀察。另報敏感度：VERIFIED_TEXT 只計逐字原文（o17 新聞＋本地語料）時的比例',
   d='G1～G8 表更新 V7X 欄與 G5 處置欄；主策略預期仍不可投資'),
 replication=dict(must='不同子代理；不讀研究員程式與 events_v3 標籤；從 raw/（ws_*、lc_*、market_*.csv）與價格盲分類；類別一致率 ≥90%（κ ≥0.8）；lo／hi 差 ≤0.1pp；處置多出槽位逐槽一致；V7X Δ 差 ≤0.05pp；分歧逐筆列出、最終採不利方向',
                  status='本子代理沒有派生子代理的工具；交由工作流另派複製員（研究員自寫檢查不算 G6）'),
 input_sha256={**{f: sha(HERE / f) for f in files if f != 's8_freeze.py'}, **{k: sha(v) for k, v in ext.items()}},
)
utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
h = hashlib.sha256(json.dumps(PRE, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
(HERE / 'SPEC.json').write_text(json.dumps(dict(PREREG=PRE, sha256=h, written_utc=utc), ensure_ascii=False, indent=1), encoding='utf-8')
(HERE / 'results.json').write_text(json.dumps(dict(PREREG=PRE, PREREG_sha256=h, PREREG_utc=utc), ensure_ascii=False, indent=1), encoding='utf-8')
print('PREREG', h, utc, 'SPEC.json sha', sha(HERE / 'SPEC.json'))
