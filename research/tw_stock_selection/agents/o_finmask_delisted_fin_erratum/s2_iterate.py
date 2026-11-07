#!/usr/bin/env python3
"""第 2 步：分類器迭代紀錄（iterations.json）。只看分類清單、名稱與池內股月，不看名次、不算任何 F 遮罩下的名單。
規則（PREREG_DESIGN）：只允許修正明確錯誤（資料錯、關鍵字誤判），最多 3 版。用法：python3 -I -B s2_iterate.py"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import freg as R
R.check_reg('PREREG_DESIGN')
import fclass as F
import flib as L
import numpy as np, pandas as pd

KW = ['銀行', '商銀', '金控', '金融', '證券', '證', '票券', '保險', '人壽', '產險', '期貨', '投信']
days = [str(d.date()) for d in L.VM(lambda d: L.C0MASK).sig_days if L.E.LO <= d <= L.E.HI]
assert len(days) == 93
POOL = {d: L.top300(pd.Timestamp(d)) for d in days}

def inventory(CL, manual_fallback=False):
    """遮罩差異（相對 C0）的 (代號, 訊號日) 清單、UNKNOWN 股月（分有／無證據）、F2 集合。"""
    IP = {}
    if manual_fallback:
        T = pd.read_csv(R.AG / 'o_pit_shares_industry_2006_2013/raw/center/industry_pit_o12.csv', dtype=str)
        IP = {c: i for c, i, s in zip(T['code'], T['industry'], T['source']) if isinstance(s, str) and '人工推定' in s}
    f1d, f2d, unk = {}, {}, {}
    for d in days:
        for j in np.where(POOL[d])[0]:
            c = L.COLS[j]; c0 = bool(L.C0MASK[j])
            s, src = CL.f1(c, d)
            if manual_fallback and s == 'UNKNOWN' and src == 'no_source' and c in IP:
                s = 'FIN_Q' if (IP[c] == '金融保險業' and CL.evidence(c)) else s
            v1 = CL.f1_value(s, c0)
            if v1 != c0: f1d.setdefault(c, []).append(d)
            if CL.f2(c) != c0: f2d.setdefault(c, []).append(d)
            if s == 'UNKNOWN':
                k = 'UNKNOWN_EVID' if CL.evidence(c) else 'UNKNOWN_NOEVID'
                unk.setdefault(k, {}).setdefault(c, 0); unk[k][c] += 1
    summ = lambda D: {c: dict(n_pool_months=len(v), first=v[0], last=v[-1], delisted=F.is_delisted(c), name='|'.join(sorted(F.NAMES.get(c, [])))) for c, v in D.items()}
    return dict(F1_mask_diff=summ(f1d), F2_mask_diff=summ(f2d),
                unknown={k: dict(n_codes=len(v), n_pool_months=sum(v.values()), codes=v) for k, v in unk.items()})

its = []
# v1：登記設計原樣
C1 = F.Classifier(KW)
its.append(dict(version='v1', utc=R.now_utc(), classifier=C1.describe(), f1_fallback='機器可讀來源（o13 快照→center StockList→ISIN 2023→stock_list 2026），人工推定不採用',
                inventory=inventory(C1),
                findings=['關鍵字在面板 1,249 檔中沒有任何「無前綴卻命中」的代號（無關鍵字誤判）；F2＝28／58／60 前綴 51 檔',
                          'F2 比 C0 多 5854、5871、6004、6012；C0 比 F2 多 0 檔',
                          'F1 與 C0 不同的只有 5854（C0 漏遮，2006-04～2011-11 池內 68 個月）與 2841（2011-08～2012-04 池內 9 個月 F1＝非金融）',
                          '6004、6012 在 2011-07-28 前已下市，沒有任何機器可讀分類 → F1＝UNKNOWN（遮罩沿用 C0＝不遮）；只有 F2 會遮',
                          '大量 2011 年前下市的非金融股也沒有分類來源 → UNKNOWN，但沒有前綴或關鍵字證據，遮罩值與 C0 相同（不影響名單）'],
                decision='進入 v2 檢查 F2 人工排除清單'))
# v2：F2 人工排除清單（前綴命中但名稱明確非金融）
C2 = F.Classifier(KW, excl_f2=())
its.append(dict(version='v2', utc=R.now_utc(), classifier=C2.describe(),
                considered=[dict(code='5871', names=sorted(F.NAMES.get('5871', [])), stock_list_2026_industry=F.SL2026['fin'].get('5871'),
                                 decision='不排除：融資租賃屬非銀行金融，名稱不是「明確非金融」；保留在 F2 是往不利方向'),
                            dict(code='2841', names=sorted(F.NAMES.get('2841', [])),
                                 decision='不排除：名稱「台灣開發」與開發金／中華開發同型，28xx 代碼源自信託投資業，名稱不能判定為明確非金融；PIT 改判交給 F1')],
                inventory=inventory(C2),
                result='排除清單維持空集合；遮罩與 v1 完全相同',
                decision='進入 v3 檢查 F1 退路'))
# v3：F1 退路是否採用 industry_pit 人工推定（候選，依設計否決）；報表改分 UNKNOWN_EVID／UNKNOWN_NOEVID
C3 = F.Classifier(KW)
alt = inventory(C3, manual_fallback=True)
its.append(dict(version='v3', utc=R.now_utc(), classifier=C3.describe(),
                candidate='F1 退路改用 industry_pit_o12「人工推定」列（6004＝金融保險業；6012＝未分類／無資料）',
                candidate_inventory_F1_mask_diff=alt['F1_mask_diff'],
                decision='否決：PREREG_DESIGN 明定人工推定不是資料、不採用；且這不是「資料錯或關鍵字誤判」。6004 留在 UNKNOWN，由 F2 涵蓋',
                accepted_change='只改報表：UNKNOWN 拆成 UNKNOWN_EVID（有前綴或關鍵字證據）與 UNKNOWN_NOEVID；遮罩與 v1 相同',
                inventory=inventory(C3),
                final='採用 v3（遮罩＝v1）'))
out = dict(utc=R.now_utc(), n_versions=len(its), note='三版的 F1、F2 遮罩完全相同；版本只記錄排除清單與退路的決定。迭代期間沒有計算任何 F 遮罩下的名單或異動。', versions=its)
(HERE / 'iterations.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
for v in its:
    print(v['version'], json.dumps({k: (list(x.keys()) if isinstance(x, dict) else x) for k, x in v['inventory'].items()}, ensure_ascii=False))
print(json.dumps(its[2]['inventory']['unknown'], ensure_ascii=False)[:1500])
print('alt', json.dumps(alt['F1_mask_diff'], ensure_ascii=False))
