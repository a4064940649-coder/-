#!/usr/bin/env python3
"""第 5b 步（事後，未事前登記；只補充 E2）：凍結的 E2 改標規則用了「listing_master 下市日空白 且 在 2026 stock_list」當作仍上市。
s5 執行後發現 3454 晶睿的這兩個欄位可能過時（listing_master last_listed_day＝2026-03-18、面板最後交易 2026-03-18、stock_list 產業欄空白）。
本程式只做描述：盤點 listing_master 下市日空白但最後上市日早於面板終點的代號，並以「只改標 2915」重跑 U_opt。凍結規則的結果保留不刪。
用法：python3 -I -B s5b_errata_posthoc.py"""
import sys, json, csv
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import freg as R
SPEC, REG = R.check_reg('ERRATUM_SPEC_FM')
import fclass as F
import flib as L
import numpy as np, pandas as pd
E, K = L.E, L.K
ET = json.loads((HERE / 'ERRATA_TEXT.json').read_text(encoding='utf-8'))
panel_end = str(K.IDX[-1].date())
SLI = {r['code']: r for r in csv.DictReader(open(F.SL_PATH, encoding='utf-8'))}
stale = []
for c, r in F.LM.items():
    if r.get('in_panel_cols') == 'True' and r.get('delist_date', '') == '' and r.get('last_listed_day', '') and r['last_listed_day'] < '2026-10-01':
        stale.append(dict(code=c, name=E.NAME.get(c, ''), last_listed_day=r['last_listed_day'], in_stock_list_2026=c in SLI,
                          stock_list_industry=(SLI.get(c, {}).get('industry') or '')))
j = L.CI['3454']; w = np.where(K.TRADED[:, j])[0]
px = [(str(K.IDX[t].date()), float(K.Cn[t, j])) for t in w[-10:]]
ev3454 = dict(listing_master={k: F.LM['3454'][k] for k in ('listed_date', 'delist_date', 'segments', 'last_listed_day', 'last_TW_evidence', 'check_status')},
              stock_list_2026=SLI.get('3454'), panel_last10_close=px, panel_end=panel_end, tw_revenue_last=str(E.LASTREV.get('3454')),
              note='最後 10 個交易日收盤 99.4 左右、幾乎不動（常見於公開收購或換股前的釘價）；2013 年起 tw_revenue 完全沒有 3454（該檔是 2026-10 抓取、2013 起只列抓取時仍存在的公司），與「抓取時已下市」一致')
RU0 = L.run(lambda d: L.C0MASK, 'opt')
R2915 = L.run(lambda d: L.C0MASK, 'opt', drop={'2915'})
ins = L.inserted(R2915)
alt = dict(u_810=sum(1 for d, c in ins if d < '2013-01-01'), u_930=len(ins))
alt.update(share_810=alt['u_810'] / 810, share_930=alt['u_930'] / 930, change_810=alt['u_810'] - 119, change_930=alt['u_930'] - 162)
bc = {}
for d, c in ins:
    k = E.cause(c, pd.Period(d, 'M')); bc[k] = bc.get(k, 0) + 1
alt['by_cause'] = bc
it = next(x for x in ET['items'] if x['id'] == 'E2')
fr = dict(rule='凍結規則（ERRATUM_SPEC_FM E2）：listing_master 下市日空白且在 2026 stock_list → LISTED_GAP', relabeled=it['evidence']['counts']['relabeled_codes'],
          rerun=it['evidence']['counts']['rerun_without_listed_gap'], change_810=it['evidence']['counts']['rerun_change_810'], change_930=it['evidence']['counts']['rerun_change_930'])
it['evidence']['frozen_rule_result'] = fr
it['evidence']['posthoc'] = dict(label='事後補充（未事前登記），只描述', listing_master_blank_delist_but_last_listed_before_2026_10=stale, evidence_3454=ev3454,
                                 rerun_relabel_2915_only=alt)
it['corrected_text'] = (
    "IFRS13／HOLE 標籤更正：U_opt 162 個放回槽位逐檔比對 listing_master 上市日、下市日、最後上市日與營收最後月份。"
    "(1) 潤泰全球 2915（1 槽，2013-12-11，原標 HOLE）：listing_master 下市日空白、最後上市日＝面板終點、營收到 2026-09 都有，2013-10 的缺值是一般資料洞，不是存活者型 → 改標 LISTED_GAP。"
    "(2) 晶睿 3454（5 槽，2013，原標 IFRS13）：listing_master 下市日欄空白、2026 stock_list 仍列此代號（產業欄空白），這是審查指出的矛盾；"
    f"但 listing_master 的最後上市日是 {ev3454['listing_master']['last_listed_day']}、面板最後交易同日、最後 10 日收盤釘在約 99.4，且 tw_revenue（2026-10 抓取）2013 年起完全沒有 3454，"
    "這些都和「抓取前已下市」一致，所以 IFRS13（存活者型）標籤很可能是對的，錯的是 listing_master 下市日欄與 stock_list 過時。ER2 原文「2026-03 下市，原因未查」應改寫為「面板最後交易 2026-03-18（listing_master 下市日欄缺漏），下市原因未查證」。"
    f"依凍結規則（下市日空白且在 stock_list 即視為仍上市）兩檔都改標，U_opt 變成 {fr['rerun']['u_810']}／810、{fr['rerun']['u_930']}／930（變動 {fr['change_810']}／{fr['change_930']} 槽）；"
    f"依上述證據只改標 2915（事後、未登記），是 {alt['u_810']}／810、{alt['u_930']}／930（變動 {alt['change_810']}／{alt['change_930']} 槽）。兩種都 > 5%，ER2 判定不變；2006～2012 的 119／810 兩種都不變。"
    f"另：listing_master 有 {len(stale)} 檔「下市日空白但最後上市日早於 2026-10」，下市日欄不能單獨當作仍上市的證據。")
it['review_numbers_need_correction'] = True
it['review_correction_note'] = 'o15 統計審查說「仍上市股（晶睿）的 2013 年營收缺值不能算進存活者型缺口」，依 last_listed_day 與面板證據，晶睿很可能在 2026-03 已下市，此前提需更正；潤泰全球的部分成立'
it['prediction_P5_note'] = f"凍結規則下 U_opt 總數變動 {abs(fr['change_930'])} 槽（P5 預測 ≤ 2 → 不成立）；事後證據口徑變動 {abs(alt['change_930'])} 槽"
ET['posthoc_utc'] = R.now_utc()
(HERE / 'ERRATA_TEXT.json').write_text(json.dumps(ET, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
RES = R.load_res(); RES['ERRATA']['sha256_after_posthoc'] = R.sha(HERE / 'ERRATA_TEXT.json'); RES['ERRATA']['E2_posthoc_2915_only'] = [alt['change_810'], alt['change_930']]; R.save_res(RES)
print(len(stale), stale[:30]); print(ev3454); print(alt); print(it['corrected_text'])
