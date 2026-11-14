#!/usr/bin/env python3
"""K 保管員：由 o_g6_spec_v2/SPEC_v2.md 產生盲版 SPEC_v2_blind.md 與 BLIND_DIFF.md。
做法：逐項「原文片段 → 〔已遮蔽〕」的精確取代（每個片段必須在原文恰好出現 1 次，否則中止）；其餘文字逐字不動。
BLIND_DIFF.md 只列條號與一句描述，不重述被遮蔽的數字。python3 -I -B k_make_blind.py"""
import hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent; AG = HERE.parent
SRC = AG / 'o_g6_spec_v2/SPEC_v2.md'
SRC_SHA = '5cfa7e5cd026896da4029ef267e8fa262e86c9a21bd49c20f8e1000afb0f81b6'
M = '〔已遮蔽〕'
# (條號, 類別, 描述, 原文片段, 取代後)
# 類別：L＝官方名單摘要（訊號日數、列數、槽數、個別訊號日）；J＝Jaccard 或名單比對結果；P＝路徑提示（官方名單檔／由官方名單或正式結果導出的檔）；
#       C＝正式執行的中間計數（ACCEPTANCE 第 8 節第 1 點列舉）；H＝正式執行的持股／個案事實（保守加遮）
EDITS = [
 ('文件開頭', 'P', '指向五個小型驗證腳本／輸出的檔名（內容含官方名單檔的路徑與 sha256、官方名單的訊號日、列數、槽位核對，或遮蔽條文所涉的個別代碼）',
  '`verify_inputs.py/.json`、`check_names.py`、`check_foreign.py/.json`、`check_panel.py/.json`、`check_plans.py/.json`、`check_tables.py/.json`',
  f'{M}、{M}、{M}、`check_panel.py/.json`、{M}、{M}'),
 ('文件開頭', 'P', '指向未遮蔽的機器可讀版規格與模糊點清單（後者含正式績效數字）',
  '機器可讀版：`SPEC_v2.json`（條號相同；由 `build_spec_json.py` 產生；D-xx 資料檔在其 `data_files`，參數全文在 `constants`）。模糊點清單：`AMBIGUITY_LOG.md`。',
  f'機器可讀版：{M}。模糊點清單：{M}。'),
 ('文件開頭', 'P', '指向驗收規則檔（該檔第 3 節與第 8 節第 1 點含紙上訊號日數、V7X 每變體列數、缺口格數與官方名單內的個別代碼）',
  '驗收規則：`ACCEPTANCE.md`。', f'驗收規則：{M}。'),
 ('第 1 節開頭', 'P', '資料檔 sha256 全文的出處（原檔同時登記了官方名單檔、正式結果檔與快取的路徑；改指向 K 摘出的資料檔清單）',
  'sha256 全文見 `verify_inputs.json`；此處列前 16 碼。',
  'sha256 全文見 `k_blind_data_sha256.json`（與本檔同資料夾；K 保管員自原清單摘出的資料檔條目）；此處列前 16 碼。'),
 ('G-01', 'L', '紙上訊號日個數',
  '| 訊號日 | 每月第一個日 ≥ 11 的面板日（245 個） |', f'| 訊號日 | 每月第一個日 ≥ 11 的面板日（{M}） |'),
 ('G-01', 'L', 'V7X 每個 j 的訊號日個數',
  '13 個錯開日 j＝0…12（每個 j 245～246 個） |', f'13 個錯開日 j＝0…12（{M}） |'),
 ('D-10', 'P', '指向未遮蔽機器可讀版規格的欄位（參數值仍在所列資料檔本身）',
  '數值全文收在 SPEC_v2.json 的 `constants.model_spec7`。', f'數值全文收在{M}。'),
 ('D-11', 'P', '紙上正式名單檔的路徑、欄位與列數',
  '- 紙上正式名單：`o_g6_allpit_list_trace_stage2/out/k_paper_lists.csv.gz`（欄 `mask,variant,signal,picks`；M0／M1／M2 × C0／ALLPIT_REAL_{opt,p90,carry}，2,450 列）。',
  f'- 紙上正式名單：{M}'),
 ('D-11', 'P', 'V7X orig 名單檔的路徑與欄位',
  '- V7X orig 名單：`o_v7_listing_exec_engine/data/plans_j7.csv`（欄 `strat,j,seed,signal,exec,exec_idx,picks`）。',
  f'- V7X orig 名單：{M}'),
 ('D-11', 'P', 'V7X REAL 名單的存放狀況、已刪檔路徑、M2 差異檔路徑與其列數',
  '- **V7X REAL 名單（13 j × 20 seeds × 3 變體）目前不存在於磁碟**：scratchpad 的 `o18_revbackfill/plans_v7x.csv`、`integrate_o19/plans_strict.csv`、`integrate_o20/plans_base.csv` 都已刪除；只剩 `integrate_o20/data/v7x_plan_diff_M2.csv.gz`（M2 與 base 不同的 2,816 列）。',
  f'- V7X REAL 名單（13 j × 20 seeds × 3 變體）：{M}'),
 ('C-09', 'J', '另一種讀法對官方名單的 Jaccard 實測值',
  '；o20 複製員實測「先取前 300 再篩」只有 Jaccard 0.896（C0，2006～2013）。', f'；{M}。'),
 ('P-01', 'L', '紙上訊號日個數、首末訊號日、缺訊號日的月份',
  '結果：245 個訊號日，2006-04-11～2026-09-11（2024-01 因面板缺日而沒有訊號日）。', f'結果：{M}。'),
 ('P-07', 'J', '兩種外國股讀法在早期名單上的比對結果與涉及代碼',
  '在 2006～2013 兩者結果相同（3697、3638、4144 在 industry_snap 即為 F- 開頭），但在面板欄位上', f'{M}在面板欄位上'),
 ('P-07', 'P', '指向外國股核對輸出的檔名（內容含官方名單 KY 槽位）',
  'delist_official 讀法判「外國」；check_foreign.json）。', f'delist_official 讀法判「外國」；{M}）。'),
 ('P-07', 'L', '官方紙上名單內 KY 股各佔幾槽，及由此推得的名單層結論',
  '正式紙上名單 M1 的 ALLPIT_REAL_opt 含 6452（7 槽）、1262（4 槽）、6422（1 槽），所以 delist_official 讀法無法重現 2013 年後的紙上 REAL 名單。', M),
 ('L-01', 'L', '錨定月個數與起訖月',
  '得 246 個月（2006-04…2026-09）＝錨定月集合 ANCH', f'得{M}＝錨定月集合 ANCH'),
 ('L-01', 'L', '對官方名單核對過的個別訊號日特例（首個訊號日、缺訊號日的 j 與月份、跨月範例日）及其核對檔名',
  '- 已核對的特例（check_plans.json）：j＝0 第一個訊號日 2006-04-10；j＝2 無 2018-01（2018-01-12 的次一列 2018-01-13 非 GOOD）；j＝7 無 2026-09（TAIEX 的 2026-09-20 不在面板）；農曆年前後 j 日會落到**下一個日曆月**（例：錨定月 2020-01、j＝9 → 2020-02-03）。',
  f'- 已核對的特例（{M}）：{M}；農曆年前後 j 日會落到**下一個日曆月**（例：{M}）。'),
 ('L-03', 'C', '正式執行的營收表缺口格計數（改值、設缺、新增各幾格）',
  'o18 正式執行的計數：長表缺口格 2,052（改成 B1 值 961、設缺 1,091、另新增 0）；5 年表缺口格 0。', f'o18 正式執行的計數：{M}'),
 ('L-04', 'L', '亂數種子的整數日期範例（該日即 L-01／L-05 已遮蔽的 j＝0 首個訊號日）',
  'YYYYMMDD(d) 為整數，例如 20060410。', f'YYYYMMDD(d) 為整數，例如{M}。'),
 ('L-05', 'L', '資料邊界豁免實際發生的訊號日',
  '實際發生於 2006-04-10 與 2021-03-10 兩個 j＝0 日。', f'實際發生於{M}。'),
 ('L-05', 'L', '豁免日個數',
  '附註 5 有寫定義與兩個豁免日，', f'附註 5 有寫定義與{M}豁免日，'),
 ('L-11', 'L', '正式執行每個變體的名單列數',
  'o18 正式執行每個 REAL 變體 63,820 列。', M),
 ('E-17a', 'H', '原引擎在官方名單上的單一持股個案（代碼與日期）',
  '；o20 個案 3057（2013-09 停牌，10-01 復牌，原引擎 10-22 才賣）。', f'；{M}。'),
 ('E-22', 'H', '正式執行的抽樣結算計數（含指向正式結果檔的路徑）與單一持股個案',
  "o18 正式執行抽存的 40 組 (變體, j, seed) 中 n_fallback 全為 0（`o_delist_rev_backfill_allpit/results.json` 的 settle_info_sample；全部 780 組未保存）。o20 的 3007 個案是在 cmode='orig' 下量到的；在正式 cons_main 下 3007 在 events_v2（unknown），走 E-20／E-21。",
  M),
 ('S-03', 'P', 't 值訊號日來源的官方名單檔名（規則文字保留）',
  "訊號日序列取自 `plans_j7.csv` 中 strat=='AB_V7_B10'", f"訊號日序列取自{M}中 strat=='AB_V7_B10'"),
 ('S-05', 'L', '紙上統計窗個數',
  '- 窗：245 個（訊號日收盤', f'- 窗：{M}個（訊號日收盤'),
 ('N-01', 'L', '官方紙上名單含哪些 KY 股',
  '紙上正式名單含 6452／1262／6422；V7X 名單不會有 |', f'{M} |'),
 ('第 10 節', 'P', '驗收規則檔的檔名（同上）',
  '要由 ACCEPTANCE.md 的三位複製員實測。', f'要由{M}的三位複製員實測。'),
 ('N-07', 'L', '哪個 j 缺哪個月的訊號日、個別訊號日範例',
  '2026-09-20 只在 TAIEX → j＝7 沒有 2026-09 訊號日、j ≥ 8 的 2026-09 訊號日比「面板日計數」早一列（例：j＝8 為 2026-09-21）；2018-01-13 只在面板且非 GOOD → j＝2 沒有 2018-01 |',
  f'2026-09-20 只在 TAIEX → {M}；2018-01-13 只在面板且非 GOOD → {M} |'),
]
src = SRC.read_bytes(); assert hashlib.sha256(src).hexdigest() == SRC_SHA, 'SPEC_v2.md 已變動'
txt = src.decode('utf-8'); out = txt
for cl, cat, desc, old, new in EDITS:
    assert txt.count(old) == 1, (cl, desc, txt.count(old))
    assert out.count(old) == 1, (cl, desc, 'overlap')
    out = out.replace(old, new)
head = ('> **盲版**：本檔由 K 保管員自 SPEC v2 原檔（sha256 `%s`）產生。'
        '標「〔已遮蔽〕」處為刪去的名單摘要、比對結果與路徑提示；規則、公式、欄位、資料檔路徑與 sha256 逐字未動。'
        '複製員 A 只可讀本檔、同資料夾的 `k_blind_data_sha256.json` 與本檔第 1 節列出的資料檔（D-01～D-10）；'
        '不得開啟未遮蔽版規格、本資料夾其他任何檔案或子資料夾，以及「出處」所列的任何程式與結果檔。\n\n' % SRC_SHA)
lines = out.split('\n', 2); assert lines[0].startswith('# SPEC v2') and lines[1] == ''
blind = lines[0] + '\n\n' + head + lines[2]
(HERE / 'SPEC_v2_blind.md').write_text(blind, encoding='utf-8')
# 殘留檢查：不得再出現這些字樣
for bad in ('245', '246', '63,820', '2,816', '2,450', '2,052', '1,091', '0.896', 'Jaccard', 'k_paper_lists', 'plans_j7', 'plans_base', 'plans_strict', 'plans_v7x', 'plan_diff',
            'check_plans', 'check_tables', 'check_foreign', 'SPEC_v2.json', 'AMBIGUITY_LOG', '7 槽', '4 槽', '1 槽', '3057', '3007', '2006-04-10', '2006-04-11', '2021-03-10', '2026-09-11', '2026-09-21', '2020-02-03', 'sealed',
            '20060410', '20210310', 'verify_inputs', 'check_names', 'ACCEPTANCE', 'k_make_blind', 'BLIND_DIFF', 'SPEC_v2.md', 'o_g6_spec_v2', 'integrate_o20/data', '/out/'):
    assert bad not in blind, bad
import re, json
assert not re.search(r'\d(\.\d+)?\s*(pp|％)', blind), '盲版內有 pp／％ 數字'
assert not re.search(r'Jaccard|[0-9０-９]\s*槽|[0-9０-９]\s*個訊號日|訊號日數', blind)
# 資料檔 sha256 全文：自 o_g6_spec_v2/verify_inputs.json 摘出，去掉官方名單、正式結果檔、研究員快取、已刪 scratchpad 名單的條目（DROP），
# 以及規格第 1 節沒有列為輸入的 5 個檔（DROP2；避免暗示額外輸入）
VI = json.loads((AG / 'o_g6_spec_v2/verify_inputs.json').read_text(encoding='utf-8'))
DROP = {'cz_results', 'v7x_results', 'plans_j7', 'x_cache', 'k_paper_lists', 'v7x_plan_diff_M2', 'plans_v7x_o18_scratch', 'plans_o19_scratch', 'plans_o20_scratch'}
DROP2 = {'rev_hist', 'DISP_OFFICIAL_v2', 'listing_master', 'firstseen_2609', 'o3_data_freeze'}
assert (DROP | DROP2) <= set(VI), (DROP | DROP2) - set(VI)
for k in DROP2: assert '/'.join(Path(VI[k]['path']).parts[-2:]) not in blind, k
KEEP = {k: {q: v[q] for q in ('path', 'bytes', 'sha256', 'header') if q in v} for k, v in VI.items() if k not in DROP | DROP2}
for k, v in KEEP.items():
    assert v.get('sha256') and not any(b in v['path'] for b in ('plans', 'k_paper', 'results.json', 'x_cache', '/out/', 'sealed', 'navs')), k
    assert Path(v['path']).stat().st_size == v['bytes'] and hashlib.sha256(Path(v['path']).read_bytes()).hexdigest() == v['sha256'], ('資料檔已變動', k)
ds = json.dumps(dict(note='SPEC v2 第 1 節資料檔的路徑、位元組數、sha256 全文與欄位列（K 保管員摘錄；只含資料檔）。', files=KEEP), ensure_ascii=False, indent=1)
(HERE / 'k_blind_data_sha256.json').write_text(ds, encoding='utf-8')
CAT = {'L': '官方名單摘要', 'J': 'Jaccard／名單比對結果', 'P': '路徑提示', 'C': '正式執行的中間計數', 'H': '正式執行的持股個案（保守加遮）'}
n = len(EDITS); bycat = {c: sum(1 for e in EDITS if e[1] == c) for c in CAT}
md = ['# BLIND_DIFF：SPEC_v2_blind.md 相對於 SPEC_v2.md 的遮蔽清單', '',
      f'- 來源：`o_g6_spec_v2/SPEC_v2.md`，sha256 `{SRC_SHA}`。', f'- 盲版：`SPEC_v2_blind.md`，sha256 `{hashlib.sha256(blind.encode("utf-8")).hexdigest()}`。',
      f'- 產生程式：`k_make_blind.py`（逐片段精確取代；每個片段在原文恰好出現一次。除「第 1 節開頭」一項改成指向 `k_blind_data_sha256.json` 外，其餘都取代成「{M}」）。',
      f'- 附檔：`k_blind_data_sha256.json`（sha256 `{hashlib.sha256(ds.encode("utf-8")).hexdigest()}`；{len(KEEP)} 個資料檔條目，自 `o_g6_spec_v2/verify_inputs.json` 的 {len(VI)} 條去掉 {len(DROP)} 條（官方名單檔、正式結果檔、研究員快取、已刪的 scratchpad 名單）與 {len(DROP2)} 條（規格第 1 節沒有列為輸入的檔）。2026-11-14 逐檔重算，{len(KEEP)} 個檔的位元組數與 sha256 都與清單相同）。',
      f'- 共 {n} 處（分布在 {len(dict.fromkeys(e[0] for e in EDITS))} 個條號）：' + '、'.join(f'{CAT[c]} {bycat[c]}' for c in CAT) + '。',
      '- 另在標題下加一行盲版說明（含複製員 A 可讀範圍）；其餘文字逐字相同。本清單只列條號與一句描述，不重述被遮蔽的內容。',
      '- **本清單、`k_make_blind.py`（內含被遮蔽的原文）、未遮蔽版規格與 `o_g6_spec_v2/ACCEPTANCE.md` 都不得交給複製員 A。**', '',
      '| # | 條號 | 類別 | 遮蔽了什麼 |', '|---|---|---|---|']
for i, (cl, cat, desc, old, new) in enumerate(EDITS, 1): md.append(f'| {i} | {cl} | {CAT[cat]} | {desc} |')
md += ['', '## 沒有遮蔽、但複製員 A 仍不得開啟的參照', '',
       '盲版保留了下列「出處／參數來源」文字（屬規則或資料檔說明），其所指文件含目標或正式數字，依 ACCEPTANCE 第 2.2 節 A 不得讀：',
       '- 文件開頭：SPEC v1 的 `SPEC.json` 與其 FINDINGS.md 附註。',
       '- D-10：兩個 `results.json` 的 PREREG 欄位（所需參數值已直接寫在條文內，不需要開檔）。',
       '- N-22：研究員引擎快取 `x_cache.npz` 的檔名與 sha（引擎行為說明；依第 2.2 節不得讀）。',
       '- 程式出處縮寫表與各條「出處：」的研究員 .py 檔名與行號。', '',
       '## 保留的理由（易被誤認為名單摘要者）', '',
       '- 面板日數、欄位數、資料檔列數與日期範圍、B1 表列數：資料檔描述，保留。',
       '- S-02 的統計終點日 e(j)：出自資料檔 `jscan_end_days.json`，屬規則參數，保留。',
       '- P-07／L-07／N-01 的「兩種外國股判定在面板欄位上相差的 6 個代碼」：是規則＋資料檔的性質，不是名單內容，保留。',
       '- 「3×13×20＝780 份」「seeds 0…19」「每格取三變體最小值」：規則，保留。',
       '- D-01b／D-03／N-05／N-07 的「舊面板缺 2024-01-02～02-05」「2026-09-20 只在 TAIEX」「2018-01-13 只在面板且非 GOOD」：資料檔之間的差異，保留（由此可依規則自行推得哪些月份沒有訊號日，但那是規則＋資料的結果，不是名單摘要）。', '',
       '## 2026-11-14 接手保管員的修訂（相對於 2026-11-12 版，盲版 sha256 `090caeb8…b732`）', '',
       '- 新增 4 處（25 → 29）：遮蔽 3 處（文件開頭的驗收規則檔名、L-04 的整數日期範例、第 10 節的驗收規則檔名）＋改指向 1 處（第 1 節開頭的 sha256 全文出處，改指向 `k_blind_data_sha256.json`）。',
       '- 既有第 1 項擴大：多遮兩個腳本／輸出檔名（其一登記了官方名單檔的路徑與 sha256，其一列出遮蔽條文所涉的個別代碼）。',
       '- 盲版說明列改寫：不再寫出未遮蔽版規格的路徑、產生程式與本清單的檔名，並加上複製員 A 的可讀範圍。', '']
(HERE / 'BLIND_DIFF.md').write_text('\n'.join(md), encoding='utf-8')
print('edits', n, bycat, 'blind sha', hashlib.sha256(blind.encode('utf-8')).hexdigest())
