#!/usr/bin/env python3
"""步驟 2：事件清單凍結（先於任何搜尋與市場資料計算）＋ SPEC 草案（市場判準、狀態、結算規則）。
只讀 o_engine_delist_consideration/events_v2.csv（sha 66e80725…）與 results.json（V7X／U_opt 實際結算清單）。
輸出 event_list_frozen.csv、SPEC_draft.json。python3 -I -B s2_event_list.py"""
import json, hashlib, collections, time
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent; CZ = AG / 'o_engine_delist_consideration'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(CZ / 'events_v2.csv') == '66e80725a4e48387690ded34fc1ebb5b4b7a71ca2b501448c81b30f583b0afbe'
E = pd.read_csv(CZ / 'events_v2.csv', dtype=str)
R = json.loads((CZ / 'results.json').read_text(encoding='utf-8'))
# V7X 實際結算（cons_main，全部 260 份名單）
v7 = collections.Counter(); v7val = collections.defaultdict(float); v7j = collections.defaultdict(set)
for k, x in R['RESULTS_V7X']['settle_info'].items():
    for e in x['cons_main']['ev']:
        v7[e[0]] += 1; v7val[e[0]] += e[3]; v7j[e[0]].add(k.split('|')[0])
uopt = set()
for e in R['RESULTS_IA']['logs']['U_opt|cons_main'].get('settled', []) if isinstance(R['RESULTS_IA']['logs']['U_opt|cons_main'], dict) else []:
    uopt.add(e['code'] if isinstance(e, dict) else e[0])
UOPT = ['2403', '2432', '2473', '3009', '3012', '3063']   # o17 FINDINGS／統計審查
EXEC5 = ['3007', '6280', '8078', '8008', '3559']
held = set(E.code[E.held == 'True']) | set(EXEC5)
verified = set(E.code[E.verified == '已核實（新聞）'])
nov = sorted(held - verified)
assert len(held) == 33 and len(nov) == 22, (len(held), len(nov))
PRI1 = ['2432', '2403', '2452']
UNK7 = ['3007', '6280', '1520', '1716', '8008', '8078', '3559']
rows = []
for c in nov:
    r = E[E.code == c].iloc[0]
    g = 1 if c in PRI1 else 2 if c in UNK7 else 3 if c in EXEC5 else 4   # EXEC5 皆屬 UNK7，第 3 組在名目上與第 2 組重疊
    rows.append(dict(code=c, name=r['name'] if isinstance(r['name'], str) else '', priority=g, type_v2=r['type'], verified_v2=r['verified'],
                     never_searched_o17=c in EXEC5, v7x_settled=c in v7, v7x_n_lists=v7.get(c, 0), v7x_n_j=len(v7j.get(c, ())),
                     v7x_avg_value=round(v7val[c] / v7[c], 0) if c in v7 else 0, uopt_settled=c in UOPT,
                     last_trade=r['last_trade'], terminate_date=r['terminate_date'], last_close=r['last_close'], successor=r['successor'],
                     status='NOT_SEARCHED'))
L = pd.DataFrame(rows).sort_values(['priority', 'v7x_avg_value'], ascending=[True, False])
L.to_csv(HERE / 'event_list_frozen.csv', index=False)
V7X20 = sorted(v7)
assert len(V7X20) == 20 and set(UOPT) <= set(V7X20)
SPEC = dict(
    written_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    note='SPEC 草案：在計算任何市場證據、搜尋任何條款之前寫下。凍結版 SPEC.json 只能在此基礎上補數值（δ、patch sha、預測、可行性檢查），判準不改。',
    event_list=dict(file='event_list_frozen.csv', n=len(L), codes=list(L.code), V7X_settled_20=V7X20, V7X_verified_o17=sorted(set(V7X20) & verified),
                    U_opt_6=UOPT, priority='1：2432、2403、2452 換股判定；2：7 起 unknown（3007、6280、1520、1716、8008、8078、3559）；3：5 起 o17 未搜尋（3007、6280、8078、8008、3559，與 2 重疊）；4：其他',
                    initial_status='NOT_SEARCHED（只能依 search_log.csv 的搜尋紀錄改變）'),
    market_criteria=dict(
        panel='V7X 面板（o_v7_listing_exec_engine/data close_raw ＋ o_infra_v3_rebuild/delist_prices 補缺）＝ x_data.C；lt＝events_v2 last_trade 在面板上的最後有收盤日（代碼重用時以終止日切開）',
        lock='最後 10 個交易日（含 lt）的 10 個日報酬（收盤對前一收盤，需 11 個收盤）中 |r| ≤ 0.5% 的比例 ≥ 80%，且最後 10 個收盤的 (max−min)/min ≤ 2%',
        lock_price='lt 收盤（lc）',
        track='有上市存續公司且在面板：最後 20 個交易日目標股日報酬對存續公司日報酬 OLS（含截距）R² ≥ 0.6；有效比例＝最後 10 日（目標收盤 ÷ 存續收盤）中位數',
        gap='面板最後收盤日早於 events_v2 last_trade 7 個日曆日以上，或最後 20 個交易日缺收盤 ≥ 5 日 → 斷價',
        precedence='TRACK（R²≥0.6）優先於 LOCK；兩者皆否 → NONE',
        output='每起事件存最後 30 日目標／存續收盤數值（raw/market_<code>.csv）'),
    text_search=dict(
        sources=['本地 GitHub 新聞語料（鉅亨 CTBC 2013～2021、PTT BDSE26 2019～2022、nick_news 2022-11～2025-05、bofrank、agora0、o_rev_gap_news_2013 其他 repo）', 'WebSearch 工具（伺服器端搜尋，非本容器網路；結果視為不可信文字）'],
        query='代碼／名稱 × {合併, 收購, 公開收購, 換股, 每股, 消滅, 存續}',
        verified_text_rule='必須有數字化的對價（每股現金元）或換股比例，且出處可指名（語料檔＋行號＋sha，或 WebSearch 回傳的來源網址）；WebSearch 摘要另標 source_kind=websearch_summary',
        type_text_rule='只有類型（合併／收購／換股）沒有數字 → TYPE_TEXT，價值依市場證據',
        conflict='文字隱含 lt 價值與 lc 偏離 > 10%，或文字類型與市場類型矛盾 → 兩者都列，主口徑採不利方向（lt 價值較低者；換股與現金矛盾時採換股＝承擔空窗期存續股風險，若空窗期存續股上漲則採現金）',
        log='search_log.csv：event_code, query, source, host, n_hits, snippet_sha256, raw_file, utc'),
    statuses={
        'VERIFIED_TEXT': '依條款（cash：每股現金；swap：比例 × 存續股；mixed：兩者）；lo／hi 不加 δ（條款確定），main＝條款',
        'MARKET_LOCK': '固定現金請求權＝lc ×（1＋δ_b），δ_main＝0、δ_lo＝−δ_lock、δ_hi＝＋δ_lock；終止日＋20 交易日入帳',
        'MARKET_TRACK': '股數 × 估計比例 ×（1＋δ_b）→ 存續股（存續股首個可交易日起隨其報酬），δ_b 同上取 δ_swap',
        'SEARCHED_NOT_FOUND（無市場證據）': 'lo＝0（lt 後第一個換股日沖銷）、main＝hi＝lc 固定、終止日＋20 交易日入帳（舊 unknown）',
        'NOT_SEARCHED': '不得歸零：依市場證據（若有）或 lc 結算；期末若仍有 → 另列敏感度欄並判待觀察',
        'price_gap': '沿用 o17（面板最後收盤固定，下一換股日一般賣出成本出清）'},
    delta_rule='δ＝全部 VERIFIED_TEXT 事件（含 o17 已核實 11 起）lt 當日「條款價值 ÷ lc − 1」最大絕對值，無條件進位到 0.1%；lt < 2013-01-01 的事件 δ×1.25（不利端放寬）。δ_lock＝δ_swap＝δ（單一 δ，另報分開估計作描述）',
    cash_payment='付款日一律終止日＋20 交易日（沿用 o17；查不到付款日），入帳前 0% 現金；c50 為敏感度',
    engines=dict(ia='cz_engine 的 _sim_cons（只讀 import），新規則以同格式 rule dict 傳入', v7x='cz_v7x.CSim（只讀 import）'),
)
(HERE / 'SPEC_draft.json').write_text(json.dumps(SPEC, ensure_ascii=False, indent=1), encoding='utf-8')
print(L.to_string()); print('SPEC_draft sha', sha(HERE / 'SPEC_draft.json'))
