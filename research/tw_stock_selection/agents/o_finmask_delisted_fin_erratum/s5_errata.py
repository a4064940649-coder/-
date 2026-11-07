#!/usr/bin/env python3
"""第 5 步（凍結後）：E1～E6 重新計數並寫 ERRATA_TEXT.json（不改任何原檔）。只計數，不計算報酬。
用法：python3 -I -B s5_errata.py"""
import sys, json, time, csv
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import freg as R
SPEC, REG = R.check_reg('ERRATUM_SPEC_FM')
assert R.load_res().get('PREREG', {}).get('sha256') == REG['sha256']
for k, h in SPEC['files']['code'].items():
    if k != 's3_freeze.py': assert R.sha(HERE / k) == h, k
import fclass as F
import flib as L
import numpy as np, pandas as pd
E, V, K = L.E, L.V, L.K
AG = R.AG
T0 = time.time()
FM = json.loads((HERE / 'FM_COUNTS.json').read_text(encoding='utf-8'))
ER2 = json.loads((AG / 'o_rev_trunc_2009_erratum/ERRATUM2.json').read_text(encoding='utf-8'))
CL = F.Classifier(SPEC['classifiers']['keywords'], SPEC['classifiers']['f2_manual_exclusion'], SPEC['classifiers']['name_exclusion'])
C0 = L.C0MASK
F2V = np.array([CL.f2(c) for c in L.COLS])
_m = {}
def mF1(d):
    if d not in _m: _m[d] = np.array([CL.f1_value(CL.f1(c, d)[0], C0[j]) for j, c in enumerate(L.COLS)])
    return _m[d]
MASK = {'C0': lambda d: C0, 'F1': mF1, 'F2': lambda d: F2V}
FULL = ('TRUNC', 'IFRS13', 'PRE12', 'HOLE', 'NEVER_DOM')
def cnt(R_):
    ins = L.inserted(R_)
    u1 = sum(1 for d, c in ins if d < '2013-01-01'); u2 = len(ins)
    return dict(u_810=u1, share_810=round(u1 / 810, 5), u_930=u2, share_930=round(u2 / 930, 5)), ins
def by_cause(ins):
    o = {}
    for d, c in ins:
        k = E.cause(c, pd.Period(d, 'M')); o[k] = o.get(k, 0) + 1
    return o
FILES = {'ER2_FINDINGS': AG / 'o_rev_trunc_2009_erratum/FINDINGS.md', 'ER2_SPEC': AG / 'o_rev_trunc_2009_erratum/ERRATUM2_SPEC.json',
         'ER2_JSON': AG / 'o_rev_trunc_2009_erratum/ERRATUM2.json', 'o14_FINDINGS': AG / 'o_g8_protocol_fin_erratum/FINDINGS.md',
         'o14_ERRATUM': AG / 'o_g8_protocol_fin_erratum/ERRATUM.json', 'listing_master': F.LM_PATH, 'stock_list_2026': F.SL_PATH,
         'tw_revenue': F.REV_PATH, 'disposal_5y': K.X / 'disposal_5y.csv.gz', 'v2_core': AG / 'main_v2/v2_core.py', 'FM_COUNTS': HERE / 'FM_COUNTS.json',
         'ERRATUM_SPEC_FM': HERE / 'ERRATUM_SPEC_FM.json', 'main_v2_FINDINGS': AG / 'main_v2/FINDINGS.md',
         'disposal_fullhist_flags': AG / 'o_disposal_official_fullhist/universe_flags.csv'}
SHA = {k: R.sha(p) for k, p in FILES.items()}
TXT = {k: FILES[k].read_text(encoding='utf-8') for k in ('ER2_FINDINGS', 'o14_FINDINGS')}
def jfield(txt, block_idx=0, field='findings'):
    """FINDINGS.md 內第 block_idx 個 ```json 區塊的欄位（解碼後文字）。"""
    blocks = txt.split('```json')[1:]
    b = blocks[block_idx].split('```')[0]
    return json.loads(b)
ER2R = jfield(TXT['ER2_FINDINGS'], 0); ER2S = jfield(TXT['ER2_FINDINGS'], 1); ER2X = jfield(TXT['ER2_FINDINGS'], 2)
O14R = jfield(TXT['o14_FINDINGS'], 0)
def excerpt(s, key, a=0, b=0):
    i = s.find(key); assert i >= 0, key
    return s[max(0, i - a): i + len(key) + b]

# ---------------- E1：只放回 TRUNC ----------------
e1 = {}
e1['C0_TRUNC'], ins_t = cnt(L.run(MASK['C0'], 'opt', surv=('TRUNC',)))
e1['C0_TRUNC_no5854'], _ = cnt(L.run(MASK['C0'], 'opt', surv=('TRUNC',), drop={'5854'}))
e1['C0_TRUNC_no5854_2473'], _ = cnt(L.run(MASK['C0'], 'opt', surv=('TRUNC',), drop={'5854', '2473'}))
e1['F1_TRUNC'], _ = cnt(L.run(MASK['F1'], 'opt', surv=('TRUNC',)))
e1['F2_TRUNC'], _ = cnt(L.run(MASK['F2'], 'opt', surv=('TRUNC',)))
e1['TRUNC_5854_slots'] = sum(1 for d, c in ins_t if c == '5854')
e1['C0_TRUNC_PRE12_no5854'], _ = cnt(L.run(MASK['C0'], 'opt', surv=('TRUNC', 'PRE12'), drop={'5854'}))
e1['C0_full_U_opt'], ins_full = cnt(L.run(MASK['C0'], 'opt'))
e1['by_cause_full'] = by_cause(ins_full); e1['by_cause_full_2006_2012'] = by_cause([x for x in ins_full if x[0] < '2013'])
print('E1', e1, round(time.time() - T0), flush=True)
rev = dict(TRUNC_810=0.0654, no5854_810=0.0531, no5854_930=0.0462, TRUNC_930=0.057)
e1_match = dict(TRUNC_810=e1['C0_TRUNC']['u_810'] == 53, TRUNC_930=e1['C0_TRUNC']['u_930'] == 53,
                no5854_810=e1['C0_TRUNC_no5854']['u_810'] == 43, no5854_930=e1['C0_TRUNC_no5854']['u_930'] == 43)

# ---------------- E2：IFRS13／HOLE 標籤 ----------------
LM = F.LM
def still_listed(c): return (LM.get(c, {}).get('delist_date', '') == '') and (c in F.SL_CODES)
tab = []
for r in ER2['U_opt_inserted']:
    c = r['code']; lm = LM.get(c, {})
    tab.append(dict(sig=r['sig'], code=c, name=r['name'], cause_ER2=r['cause'], listed=lm.get('listed_date', ''), delist_lm=lm.get('delist_date', ''),
                    in_stock_list_2026=c in F.SL_CODES, last_rev=str(E.LASTREV.get(c, '')), last_trade=r['last_trade'],
                    cause_new=('LISTED_GAP' if still_listed(c) else r['cause'])))
T = pd.DataFrame(tab)
relab = T[T.cause_new != T.cause_ER2]
aff_codes = set()
RU = L.run(MASK['C0'], 'opt')
for d in RU: aff_codes |= set(RU[d]['aff'])
drop_listed = sorted(c for c in aff_codes if still_listed(c))
e2_rerun, ins_rel = cnt(L.run(MASK['C0'], 'opt', drop=set(drop_listed)))
spot = {c: dict(name=E.NAME.get(c, ''), listed=LM.get(c, {}).get('listed_date'), delist=LM.get(c, {}).get('delist_date'), in_sl2026=c in F.SL_CODES,
                last_rev=str(E.LASTREV.get(c, '')), first_rev=str(E.FIRSTREV.get(c, '')), last_trade=str(E.LT[c].date()),
                ER2_rows=[(r['sig'], r['cause']) for r in ER2['U_opt_inserted'] if r['code'] == c]) for c in ('3454', '2915')}
hole_months = {}
for r in ER2['U_opt_inserted']:
    if r['cause'] == 'HOLE':
        m = pd.Period(r['sig'], 'M')
        for x in [m - k for k in (1, 2, 3, 13, 14, 15)]:
            if not np.isfinite(E.LV[r['code']].get(x, np.nan)): hole_months[str(x)] = hole_months.get(str(x), 0) + 1
e2 = dict(relabeled_rows=relab[['sig', 'code', 'name', 'cause_ER2', 'cause_new', 'delist_lm', 'in_stock_list_2026', 'last_rev']].to_dict(orient='records'),
          n_relabeled_rows=len(relab), n_relabeled_rows_2006_2012=int((relab.sig < '2013-01-01').sum()), relabeled_codes=sorted(set(relab.code)),
          by_cause_ER2=T.cause_ER2.value_counts().to_dict(), by_cause_relabeled_direct=T.cause_new.value_counts().to_dict(),
          by_cause_relabeled_direct_2006_2012=T[T.sig < '2013-01-01'].cause_new.value_counts().to_dict(),
          drop_listed_affected_codes=drop_listed, rerun_without_listed_gap=e2_rerun, rerun_by_cause=by_cause(ins_rel),
          rerun_change_930=e2_rerun['u_930'] - 162, rerun_change_810=e2_rerun['u_810'] - 119, spot=spot, HOLE_missing_months_of_inserted=hole_months,
          table=tab)
print('E2', {k: v for k, v in e2.items() if k not in ('table', 'relabeled_rows')}, round(time.time() - T0), flush=True)

# ---------------- E3：處置資料起日 ----------------
dsp = pd.read_csv(FILES['disposal_5y'], dtype={'code': str}, parse_dates=['date', 'start', 'end'])
uf = pd.read_csv(FILES['disposal_fullhist_flags'], dtype={'code': str}, parse_dates=['start', 'end'])
L0 = FM['lists']['C0|C0']
def in_disp(c, d):
    d = pd.Timestamp(d); x = uf[(uf.code == c) & (uf.start <= d) & (uf.end >= d)]
    return len(x) > 0
dslots = [(d, c) for d, pk in L0.items() for c in pk if in_disp(c, d)]
e3 = dict(pipeline_file=str(FILES['disposal_5y']), pipeline_min_start=str(dsp['start'].min().date()), pipeline_min_date=str(dsp['date'].min().date()),
          mask_hits_2006_2013=int(sum(K.disposal_mask(pd.Timestamp(d)).sum() for d in L0)),
          other_center_file=str(FILES['disposal_fullhist_flags']), other_min_start=str(uf['start'].min().date()),
          descriptive_C0_slots_in_fullhist_disposal_window=len(dslots), descriptive_rows=dslots,
          note='descriptive_* 只描述（事後，非判定）：若用研究中心另一份 2003 起的官方處置事件期，C0 名單在訊號日當天正處於處置期的槽位數')
print('E3', {k: v for k, v in e3.items() if k != 'descriptive_rows'}, flush=True)

# ---------------- E4：營收期限＝訊號當天 ----------------
days = sorted(L0)
DS = [d for d in days if V.deadline_case(pd.Timestamp(d)) == 'D=S']
e4 = dict(n_DS_days=len(DS), DS_days=DS, n_DS_2006_2012=sum(1 for d in DS if d < '2013'), C0_slots=10 * len(DS), C0_slots_2006_2012=10 * sum(1 for d in DS if d < '2013'),
          U_opt_inserted_on_DS=sum(1 for d, c in ins_full if d in DS), deadline_rule='v2_core.deadline_case：期限＝訊號月 10 日起第一個交易日（TCAL＝taiex ∪ 面板日）；D=S＝期限即訊號日')

# ---------------- E6：全部金融代碼剔除 ----------------
finlike = {c for c in L.COLS if c[:2] in ('28', '58', '60')}
e6 = {}
e6['reviewer_method_drop_28_58_60'], _ = cnt(L.run(MASK['C0'], 'opt', drop=finlike))
e6['reviewer_direct_subtraction'] = dict(u_810=119 - sum(1 for d, c in ins_full if d < '2013' and c in finlike), u_930=162 - sum(1 for d, c in ins_full if c in finlike))
for nm in ('F1', 'F2'):
    Rb = L.run(MASK[nm], 'opt')
    e6[f'U_opt_inserted_under_{nm}'], _ = cnt(Rb)
e6['U_opt_vs_U_opt_F2_changed'] = dict(slots_930=FM['blocks']['U_opt|F2']['slots_930'], slots_810=FM['blocks']['U_opt|F2']['slots_810'])
print('E6', e6, round(time.time() - T0), flush=True)

# ---------------- 文字 ----------------
J = FM['judgement']; B = FM['blocks']
mpct = lambda x: f'{100 * x:.2f}%'
need1 = not all(e1_match.values())
items = []
items.append(dict(
    id='E1', target_file=str(FILES['ER2_FINDINGS']), target_sha256=SHA['ER2_FINDINGS'], location='研究員 JSON findings 欄位「## 8. 穩健性」第 2 條（分原因）',
    original_excerpt=excerpt(ER2R['findings'], '分原因：TRUNC 53、PRE12 41、IFRS13 35、HOLE 33。只看 2009-11 截斷本身（TRUNC 53／810＝6.5%），就已經超過 5%。'),
    corrected_text=(f"分原因：TRUNC 53、PRE12 41、IFRS13 35、HOLE 33。只看 2009-11 截斷本身：只放回 TRUNC 重跑，2006～2012 是 {e1['C0_TRUNC']['u_810']}／810＝{mpct(e1['C0_TRUNC']['share_810'])}，"
                    f"2006～2013 是 {e1['C0_TRUNC']['u_930']}／930＝{mpct(e1['C0_TRUNC']['share_930'])}；其中 {e1['TRUNC_5854_slots']} 槽是 5854 合庫（PIT 下應屬金融、應被排除）。"
                    f"扣 5854（不放回、重跑）後是 {e1['C0_TRUNC_no5854']['u_810']}／810＝{mpct(e1['C0_TRUNC_no5854']['share_810'])}、2006～2013 為 {e1['C0_TRUNC_no5854']['u_930']}／930＝{mpct(e1['C0_TRUNC_no5854']['share_930'])}，剛好壓在 5% 門檻上；"
                    f"以 PIT 金融遮罩 F1／F2 只放回 TRUNC，2006～2012 為 {e1['F1_TRUNC']['u_810']}／{e1['F2_TRUNC']['u_810']}（810 槽）。"
                    "超過 5% 主要來自 PRE12、HOLE 兩類，它們是依資料形狀推論的，沒有比對 MOPS 原頁。"),
    evidence=dict(counts=e1, review_numbers=rev, match_review=e1_match, files={k: SHA[k] for k in ('ER2_FINDINGS', 'ER2_JSON', 'tw_revenue', 'v2_core')},
                  method='erlib.SURV＝(TRUNC,)；5854 不放回＝原因改 DROPPED；比例＝放回股進榜槽位 ÷ 810／930（flib.run，ERRATUM_SPEC_FM E1）'),
    proposer='o15 統計審查（數字）；本題重新計數並改寫', review_numbers_need_correction=need1))
e2_txt = (f"IFRS13／HOLE 標籤更正：U_opt 162 個放回槽位逐檔比對 listing_master 上市日、下市日與營收最後月份。仍上市（listing_master 下市日空白且在 2026 stock_list）的 {len(e2['relabeled_codes'])} 檔"
          f"（{'、'.join(f'{c} {E.NAME.get(c, "")}' for c in e2['relabeled_codes'])}）共 {e2['n_relabeled_rows']} 槽（2006～2012 有 {e2['n_relabeled_rows_2006_2012']} 槽），缺值不是存活者型，改標 LISTED_GAP（一般資料洞）。"
          f"晶睿 3454：listing_master 下市日空白、仍在 2026 stock_list，營收最後月份 {spot['3454']['last_rev']}，原「2026-03 下市」為誤記；潤泰全球 2915：下市日空白、仍上市，2013-10 缺值屬一般資料洞。"
          f"改標後依原因（直接改標）：{json.dumps(e2['by_cause_relabeled_direct'], ensure_ascii=False)}。把 LISTED_GAP 股不放回重跑，U_opt 是 {e2_rerun['u_810']}／810＝{mpct(e2_rerun['share_810'])}、{e2_rerun['u_930']}／930＝{mpct(e2_rerun['share_930'])}"
          f"（相對原 119／162 變動 {e2['rerun_change_810']}／{e2['rerun_change_930']} 槽），仍 > 5%，ER2 判定不變。")
items.append(dict(
    id='E2', target_file=str(FILES['ER2_FINDINGS']), target_sha256=SHA['ER2_FINDINGS'], location='研究員 JSON findings「## 7. 結果」主要受影響股（晶睿）、「## 結論」斷崖分類（HOLE）、key_numbers【U_opt 依原因】',
    original_excerpt=[excerpt(ER2R['findings'], '晶睿 5 次：2026-03 下市，原因未查。'), excerpt(ER2R['findings'], '2011-01、2011-02、2012-01 三個月的洞（HOLE）：只缺 2013 年以後才下市的公司。'),
                      excerpt(ER2R['key_numbers'], '全期：TRUNC 53、PRE12 41、IFRS13 35、HOLE 33（IFRS13 全部在 2013）。')],
    corrected_text=e2_txt,
    evidence=dict(counts={k: v for k, v in e2.items() if k != 'table'}, files={k: SHA[k] for k in ('ER2_FINDINGS', 'ER2_JSON', 'listing_master', 'stock_list_2026', 'tw_revenue')},
                  method='受影響股若 listing_master 下市日空白且在 2026 stock_list → 任何 SURV 原因改 LISTED_GAP；以 drop＝這些代號（在任一訊號日屬受影響集合者）重跑 U_opt'),
    proposer='o15 統計審查（晶睿、潤泰全球矛盾）；本題逐檔比對並重跑', review_numbers_need_correction=False,
    prediction_P5_note=f"改標後 U_opt 總數變動 {abs(e2['rerun_change_930'])} 槽（2006～2013）、{abs(e2['rerun_change_810'])} 槽（2006～2012）；P5 預測 ≤ 2 槽" ))
items.append(dict(
    id='E3', target_file=str(FILES['ER2_FINDINGS']), target_sha256=SHA['ER2_FINDINGS'], secondary_targets=[dict(file=str(FILES['ER2_SPEC']), sha256=SHA['ER2_SPEC']), dict(file=str(FILES['main_v2_FINDINGS']), sha256=SHA['main_v2_FINDINGS'])],
    location='研究員 JSON frozen_spec「2. 受影響集合 A(d)」股池條件；ERRATUM2_SPEC.affected_set.pool',
    original_excerpt=[excerpt(ER2R['frozen_spec'], '  - 非處置；'), excerpt(json.dumps(json.loads(FILES['ER2_SPEC'].read_text(encoding='utf-8'))['affected_set']['pool'], ensure_ascii=False), '非處置')],
    corrected_text=(f"主策略管線的處置濾網（ia_core.disposal_mask）只讀 disposal_5y.csv.gz，最早處置起日 {e3['pipeline_min_start']}（最早資料日 {e3['pipeline_min_date']}）。"
                    f"2006～2013 的 93 個訊號日處置遮罩命中 {e3['mask_hits_2006_2013']} 檔次，也就是 2006～2013 沒有處置股排除；凍結規格中「非處置／排除處置股」的描述對該期不適用（基準 C0 與 U_opt 一樣，不改變計數）。"
                    f"研究中心另有 2003 年起的官方處置事件期（o_disposal_official_fullhist，最早 {e3['other_min_start']}），主策略管線沒有使用。"),
    evidence=dict(counts={k: v for k, v in e3.items() if k != 'descriptive_rows'}, descriptive_rows=e3['descriptive_rows'], files={k: SHA[k] for k in ('disposal_5y', 'disposal_fullhist_flags', 'ER2_SPEC', 'ER2_FINDINGS')}),
    proposer='o15 可執行審查', review_numbers_need_correction=False))
items.append(dict(
    id='E4', target_file=str(FILES['ER2_FINDINGS']), target_sha256=SHA['ER2_FINDINGS'], location='研究員 JSON frozen_spec「1. 基準」（V2(…, exfin=True) 未設 unavail＝假設訊號日已取得全部 m−1 營收）；FINDINGS 未交代',
    original_excerpt=excerpt(ER2R['frozen_spec'], 'main_v2/v2_core.V2(', 0, 60),
    corrected_text=(f"營收公告期限＝訊號當天（v2_core.deadline_case＝D=S）的訊號日 2006～2013 共 {e4['n_DS_days']} 個（2006～2012 有 {e4['n_DS_2006_2012']} 個），受影響 C0 槽位 {e4['C0_slots']}（2006～2012 為 {e4['C0_slots_2006_2012']}），"
                    f"U_opt 放回槽位落在這些日子的有 {e4['U_opt_inserted_on_DS']} 個。基準 C0 與 U_opt 都假設訊號日已取得全部 m−1 月營收；當日公告是否在訊號計算前可得，無法由現有資料證明，屬 PIT 風險（只標註，不重算）。"
                    f"清單：{'、'.join(e4['DS_days'])}。"),
    evidence=dict(counts=e4, files={k: SHA[k] for k in ('v2_core', 'ER2_FINDINGS')}), proposer='o15 可執行審查', review_numbers_need_correction=(e4['n_DS_days'] != 33)))
dC = FM['direct']
items.append(dict(
    id='E5', target_file=str(FILES['o14_FINDINGS']), target_sha256=SHA['o14_FINDINGS'], secondary_targets=[dict(file=str(FILES['o14_ERRATUM']), sha256=SHA['o14_ERRATUM'])],
    location="研究員 JSON headline 與 findings「(B) 金融排除勘誤」「建議勘誤文字」；統計審查 issues（5854 出現 0 次）",
    original_excerpt=[excerpt(O14R['headline'], '5854 合庫在 2006～2011 進入 10 檔名單 0 次（最好名次 17）'),
                      excerpt(O14R['findings'], '5854 合庫從未進入 10 檔名單；非 PIT 金融股只有 6012 金鼎證在 2008-04 出現 1 次'),
                      excerpt(TXT['o14_FINDINGS'], '5854 出現 0 次', 30, 40)],
    corrected_text=("補註：「5854 出現 0 次」在 C0／C1 遮罩下成立，但 C0 本身沒有遮掉 5854 合庫、6004 元京證、6012 金鼎證（2026 年 stock_list 的事後清單；三檔都不是 28xx）；C1 在 2011-07-28 前沿用 C0，同樣繼承此缺口。"
                    f"PIT 遮罩下的正確計數（本題 o_finmask_delisted_fin_erratum）：F1（dated 快照只往回看）判 5854 全部 68 個池內月為金融（2011-07-28 起 PIT、之前準 PIT），6004、6012 無任何機器可讀分類 → UNKNOWN；F2（28／58／60 前綴＋名稱）三檔都判金融。"
                    f"C0 名單中的直接污染：F1 {len(dC['F1']['C0'])} 槽、F2 {len(dC['F2']['C0'])} 槽（6012，2008-04-11 第 4 名）；ER2 U_opt 名單中：F1 {len(dC['F1']['U_opt'])} 槽（全是 5854）、F2 {len(dC['F2']['U_opt'])} 槽（5854 10、6004 2、6012 1）。"
                    f"以 F 遮罩重跑後 C0 名單總異動：F1 {B['C0|F1']['slots_930']:.0f}／930、F2 {B['C0|F2']['slots_930']:.0f}／930（F2 的 4 槽與 o14 C2 逐筆相同）。結論方向不變（5854 在 C0 名單 0 次），但這個結論本身是在非 PIT 遮罩下得到的。"),
    evidence=dict(counts=dict(direct=dC, blocks={k: {kk: vv for kk, vv in v.items() if kk != 'rows'} for k, v in B.items()}, pool_months=FM['pool_months']),
                  files={k: SHA[k] for k in ('o14_FINDINGS', 'o14_ERRATUM', 'FM_COUNTS', 'ERRATUM_SPEC_FM')}),
    proposer='o15 兩位審查（FINMASK 漏遮）；本題計數', review_numbers_need_correction=False))
ok6 = (e6['reviewer_method_drop_28_58_60']['u_810'] == 107 and e6['reviewer_method_drop_28_58_60']['u_930'] == 150)
items.append(dict(
    id='E6', target_file=str(FILES['ER2_FINDINGS']), target_sha256=SHA['ER2_FINDINGS'], location='統計審查 issues 第 2 條、可執行審查 issues 第 4 條（全部金融代碼剔除後）',
    original_excerpt=[excerpt(ER2S['issues'][1], '我把 28xx、58xx、60xx 全部剔除後重跑，是 107／810（13.2%）、150／930（16.1%），判定不變。'),
                      excerpt(ER2X['issues'][3], '兩檔都扣掉後大約是 2006～2012 107／810＝13.2%、2006～2013 150／930＝16.1%（直接相減，沒有重排），判定不變。')],
    corrected_text=(f"重新計數：28／58／60 開頭代號全部不放回並重跑，U_opt 是 {e6['reviewer_method_drop_28_58_60']['u_810']}／810＝{mpct(e6['reviewer_method_drop_28_58_60']['share_810'])}、"
                    f"{e6['reviewer_method_drop_28_58_60']['u_930']}／930＝{mpct(e6['reviewer_method_drop_28_58_60']['share_930'])}；直接相減（5854 10 槽＋6004 2 槽）是 {e6['reviewer_direct_subtraction']['u_810']}／{e6['reviewer_direct_subtraction']['u_930']}。"
                    f"與本題 F2 對齊：F2 遮罩（28／58／60 前綴，金融股不進合格集合也不放回，基準名單同時改用 C0_F2）下 U_opt 放回槽位是 {e6['U_opt_inserted_under_F2']['u_810']}／810、{e6['U_opt_inserted_under_F2']['u_930']}／930；"
                    f"F1 下是 {e6['U_opt_inserted_under_F1']['u_810']}／810、{e6['U_opt_inserted_under_F1']['u_930']}／930。F2 與審查作法的差別只在 6012：審查作法只是不放回，6012 仍以 C0 未遮的身分留在基準池（2008-04-11 入選），F2 則把它遮掉。判定（> 5%）不變。"),
    evidence=dict(counts=e6, files={k: SHA[k] for k in ('ER2_FINDINGS', 'ER2_JSON', 'FM_COUNTS')}), proposer='o15 統計審查與可執行審查（數字）；本題重新計數並與 F2 對齊',
    review_numbers_need_correction=not ok6))
annot = (f"【金融遮罩 PIT 勘誤（o_finmask_delisted_fin_erratum，只計數）】主策略排除金融用的 FINMASK 是 2026 年的事後清單，2006～2013 漏遮已下市的 5854 合庫、6004 元京證、6012 金鼎證。"
         f"改用 PIT 遮罩重跑名單：C0 異動 F1 {B['C0|F1']['slots_930']:.0f}／930、F2 {B['C0|F2']['slots_930']:.0f}／930；ER2 放回名單 U_opt 異動 F1 {B['U_opt|F1']['slots_930']:.0f}、F2 {B['U_opt|F2']['slots_930']:.0f}／930；"
         f"8 個判定比例最大 {mpct(J['M'])}（{J['M_at']}）≤ 5%，依事前規則只加註，G5 第一段不新增項目。直接違反「排除金融」的 C0 槽位只有 1 個（6012，2008-04-11）。")
OUT = dict(utc=R.now_utc(), spec_sha256=REG['sha256'], no_returns='只計數，未計算任何報酬', note='不修改任何原檔；以下更正由研究中心決定是否貼入原文旁', items=items,
           annotation_for_center=dict(targets=[dict(file=str(FILES['main_v2_FINDINGS']), sha256=SHA['main_v2_FINDINGS'], where='正式數字旁'), dict(file=str(AG.parent.parent.parent / '台股個股選股規則實證研究報告.md'), where='第 0 節')], text=annot),
           file_sha256=SHA)
(HERE / 'ERRATA_TEXT.json').write_text(json.dumps(OUT, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
RES = R.load_res(); RES['ERRATA'] = dict(utc=OUT['utc'], sha256=R.sha(HERE / 'ERRATA_TEXT.json'), E1=e1_match, E6_match_review=ok6, E4_n=e4['n_DS_days'], E2_change=[e2['rerun_change_810'], e2['rerun_change_930']]); R.save_res(RES)
for it in items: print(it['id'], it['corrected_text'][:900], '\n')
print(annot)
