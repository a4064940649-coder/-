#!/usr/bin/env python3
"""S3e：逐筆判類（依 SPEC_draft 狀態表）→ notes/s3_review.json；產生 DISP_OFFICIAL_v2（嚴格）、_lenient（加 SINGLE_SOURCE，PREREG 範圍內）、
_posthoc（另加範圍外事後發現的 SINGLE_SOURCE，只作並列敏感度）。寫 PREREG_ADDENDUM（sha、written_utc）。
python3 -I -B s9_s3_judge.py"""
import json, hashlib, time
from pathlib import Path
import pandas as pd, numpy as np
HERE = Path(__file__).resolve().parent
RJ = HERE / 'results.json'; R = json.loads(RJ.read_text(encoding='utf-8'))
assert hashlib.sha256(json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256']
assert 'PREREG_ADDENDUM' not in R
E = pd.read_csv(HERE / 'data/unmatched_events.csv', dtype={'code': str})
E19 = E[E.year == 2019]
# 研究員逐筆閱讀 snippet 的判讀（理由寫在 reason；狀態只能是 SPEC_draft 狀態表內的值，加上 NOT_DISPOSAL 細分記在 reason）
manual = {
 '1445': ('NOT_FOUND', '大宇（名稱誤配：大宇資 6111 上櫃），處置字樣屬泰碩（已在名單）'),
 '1736': ('NOT_FOUND', 'PTT 推文「準備要關禁閉了」為推測，非陳述；WebSearch ws_01/ws_05 未找到'),
 '2009': ('NOT_FOUND', '代碼誤配（年份 2009 被當代碼）'), '2010': ('NOT_FOUND', '代碼誤配（年份）'), '2013': ('NOT_FOUND', '代碼誤配（年份）'),
 '2015': ('NOT_FOUND', '代碼誤配（年份）'), '2020': ('NOT_FOUND', '代碼誤配（年份）'),
 '2364': ('NOT_FOUND', '倫飛：文中「被列為處置股」指 2015 年事件（2015 年名單另核）'),
 '2101': ('NOT_FOUND', '名稱誤配（南港土地；處置字樣屬國產 2504，已在名單）'), '2102': ('NOT_FOUND', '鄰近股誤配（國產 2504 已在名單）'),
 '9902': ('NOT_FOUND', '鄰近股誤配（國產 2504 已在名單）'), '2330': ('NOT_FOUND', '鄰近股誤配（愛普 6531／國產，已在名單）'),
 '2316': ('NOT_FOUND', 'PTT「被關禁閉了 車拋錨無法發動」：面板 2019-09-26～10-04 無成交、前三日收盤 44.00 不動 → 暫停交易，非注意-處置（NOT_DISPOSAL）'),
 '2367': ('NOT_FOUND', 'PTT「很快被處置了啦」為推測；WebSearch ws_03 只找到 2026-03 處置（已在名單）'),
 '2371': ('NOT_FOUND', '鄰近股誤配（華映全額交割／變更交易方法）'),
 '2475': ('NOT_FOUND', '華映 2019-01／03「全額交割並採分盤交易」屬變更交易方法（NOT_DISPOSAL）；2019-04 注意-處置已在名單'),
 '2387': ('NOT_FOUND', '推文假設語氣'), '2390': ('NOT_FOUND', '鄰近股誤配（泰碩）'), '2498': ('NOT_FOUND', '鄰近股誤配（泰碩）'),
 '3027': ('NOT_FOUND', '鄰近股誤配（泰碩）'), '3149': ('NOT_FOUND', '鄰近股誤配（泰碩）'), '3167': ('NOT_FOUND', '名稱誤配（「大量」為一般詞）'),
 '2605': ('NOT_FOUND', '鄰近股誤配（立積 4968 已在名單）'), '2606': ('NOT_FOUND', '同上'), '2612': ('NOT_FOUND', '同上'), '2637': ('NOT_FOUND', '同上'),
 '3017': ('NOT_FOUND', '鄰近股誤配（力致 3483 上櫃）'), '3653': ('NOT_FOUND', '同上'), '6230': ('NOT_FOUND', '同上'), '6591': ('NOT_FOUND', '同上'),
 '3406': ('NOT_FOUND', 'PTT 疑問句「不是處置股嗎?」，非陳述；面板 2019-08～09 無觸發'),
 '3443': ('NOT_FOUND', '名稱誤配（樂巧創意）；處置字樣屬鈊象 3293（上櫃）'),
 '3519': ('NOT_FOUND', '綠能 2019-03 淨值為負 → 變更交易方法分盤（NOT_DISPOSAL）；注意-處置 2019-01、04 已在名單'),
 '5259': ('NOT_FOUND', '清惠 2019-03 退票 → 變更交易方法分盤（NOT_DISPOSAL）；注意-處置 2019-07 起已在名單'),
 '6278': ('NOT_FOUND', '鄰近股誤配（雍智 6683 上櫃）'),
 '6533': ('NOT_FOUND', 'PTT「要準備分盤交易惹」為推測；WebSearch ws_07 未找到 2019 處置'),
}
rev = []
for r in E19.itertuples():
    st, why = manual.get(r.code, ('UNREVIEWED', ''))
    rev.append(dict(code=r.code, first=r.first, srcs=r.srcs, status=st, reason=why))
assert all(x['status'] != 'UNREVIEWED' for x in rev), [x for x in rev if x['status'] == 'UNREVIEWED']
proxy_ws = [dict(code='1736', trig='2019-05-23', status='NOT_FOUND', ws=['ws_01', 'ws_05']), dict(code='1541', trig='2019-06-18', status='NOT_FOUND', ws=['ws_02', 'ws_06']),
            dict(code='2367', trig='2019-09-05', status='NOT_FOUND', ws=['ws_03']), dict(code='6533', trig='2019-01-16', status='NOT_FOUND', ws=['ws_07']),
            dict(code='3708', trig='2019-02-11', status='NOT_FOUND', ws=['ws_11']), dict(code='4935', trig='2019-07-08', status='NOT_FOUND', ws=['ws_12'], note='2019-11 全額交割（變更交易方法，非處置）')]
posthoc = [dict(code='2344', announce='2020-12-09', start='2020-12-10', status='SINGLE_SOURCE', scope='範圍外（2020，PREREG 搜尋範圍只有 2019 與 2004～2008）',
                evidence='PTT 2020-12-09 同一串兩則推文「●2344華崩 (9日)近6天漲福29.7% 連3天達標 明分盤」「●2344華崩店明天開始分盤交易10天」（同一來源）；面板 12-04～12-09 r6 0.33～0.44；WebSearch ws_13 未找到；TWSE 2020 年檔無此列',
                caveat='2022～2025（有 disposal_5y 交叉驗證的完整年份）面板強觸發卻無處置的比例 25～35%，面板強觸發本身不是缺列證據')]
out = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), scope_years=[2019, 2004, 2005, 2006, 2008],
           local_2019_unmatched_events=rev, proxy_candidates_websearched=proxy_ws,
           pre2013_years='2004～2008 無本地語料覆蓋（鉅亨鏡像 2013 起、PTT 2019 起、ELD 2016～2018）；WebSearch 對 2019 個案已證實無效（13 次 0 筆可用），不再對 2004～2008 逐案搜尋 → 這些年份狀態＝NOT_SEARCHABLE（不可驗證，非 NOT_FOUND）',
           counts=dict(CONFIRMED_MISSING=0, SINGLE_SOURCE_in_scope=0, SINGLE_SOURCE_posthoc_out_of_scope=1,
                       NOT_FOUND_2019_events=sum(x['status'] == 'NOT_FOUND' for x in rev)), posthoc=posthoc)
(HERE / 'notes').mkdir(exist_ok=True)
(HERE / 'notes/s3_review.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
# DISP_OFFICIAL_v2
U = pd.read_csv(HERE / 'data/universe_announce.csv', dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
V2 = U.copy(); V2['src'] = 'TWSE_punish_archive'
V2.to_csv(HERE / 'data/DISP_OFFICIAL_v2.csv', index=False)
V2.to_csv(HERE / 'data/DISP_OFFICIAL_v2_lenient.csv', index=False)      # 範圍內 SINGLE_SOURCE＝0 → 與嚴格相同
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
TD = pd.read_parquet(P / 'close_raw.parquet').index
s = pd.Timestamp('2020-12-10'); e = TD[TD.searchsorted(s) + 9]
ph = pd.DataFrame([dict(code='2344', ordr=-1, announce=pd.Timestamp('2020-12-09'), start=s, end=e, overlap_proxy=False, second30=False, take_R1=True, kind='regular',
                         regime='periodic_match', measure='(posthoc SINGLE_SOURCE)', condition='PTT', nth=1, lag_td=1, src='POSTHOC_SINGLE_SOURCE_PTT')])
pd.concat([V2, ph], ignore_index=True).to_csv(HERE / 'data/DISP_OFFICIAL_v2_posthoc.csv', index=False)
sha = lambda f: hashlib.sha256((HERE / f).read_bytes()).hexdigest()
ADD = dict(written_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           files={f: sha(f) for f in ['data/DISP_OFFICIAL_v2.csv', 'data/DISP_OFFICIAL_v2_lenient.csv', 'data/DISP_OFFICIAL_v2_posthoc.csv', 'notes/s3_review.json',
                                      'search_log_web.csv', 'search_log_local.csv', 'data/local_mentions.csv', 's7_proxy_match.json', 's7b_strong_unmatched.json']},
           strict_additions=0, lenient_additions=0, posthoc_additions=1,
           P1_counts='2218/2218 接回、0 列 start ≤ announce（S1；凍結前已計數）',
           recompute='依 PREREG.S4_recompute.identity_shortcut：嚴格＝寬鬆＝現行覆蓋 → 名單 sha 必須與官方相同（斷言）；另跑 posthoc（+2344 2020-12-10～12-23）作並列敏感度、30 萬兩格、100 萬一般三放回重現',
           note='posthoc 變體是搜尋後新增、範圍外、單一來源；只並列，不進主欄，也不進判定格的「嚴格／寬鬆」不利端取值（不利端在嚴格＝寬鬆時即嚴格本身）；另報「若把 posthoc 當寬鬆」的不利端數字')
R['PREREG_ADDENDUM'] = ADD; R['PREREG_ADDENDUM_sha256'] = hashlib.sha256(json.dumps(ADD, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(out['counts'], ensure_ascii=False), R['PREREG_ADDENDUM_sha256'], ADD['written_utc'], str(e.date()))
