"""步驟 1：事件表。凍結規格把 events_v2.csv 定為事件來源；這裡獨立重算衍生欄位並抽查原始轉錄。
- 重算 in_panel / src3_period / src_E / vol_ratio_E（E 日成交值 ÷ E-25..E-6 中位數）/ period，逐欄比對。
- 用 lists_transcribed.tsv、hd_etf_transcribed.tsv 的名稱清單（stock_list 名稱對代碼）抽查事件表涵蓋率。
- 資料關卡：MSCI 刪除筆數、生效日來源、富時筆數、各審核日成交值放大倍數。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import rlib as L

d = L.panel(); idx = d['idx']
e = L.load_events()
out = {'n_rows': int(len(e))}

# ---- 重算衍生欄位 ----
cols = set(d['close_raw'].columns)
def pos(day):
    i = idx.searchsorted(day)
    return i if i < len(idx) and idx[i] == day else -1
rows = []
for r in e.itertuples():
    inp = r.code in cols if isinstance(r.code, str) else False
    t = pos(r.eff)
    vr = srcE = np.nan
    if inp and t >= 0:
        v = d['value'][r.code]
        m = d['volmed'][r.code].iloc[t]
        vr = v.iloc[t] / m if m and m > 0 else np.nan
        srcE = d['src'][r.code].iloc[t]
    s3 = bool(L.SRC3[0] <= r.eff <= L.SRC3[1])
    per = 'select' if r.eff <= L.SEL[1] else 'validate'
    rows.append((inp, s3, round(vr, 2) if np.isfinite(vr) else np.nan, srcE, per, t))
R = pd.DataFrame(rows, columns=['in_panel_r', 'src3_r', 'vr_r', 'srcE_r', 'period_r', 't'], index=e.index)
cmp = {
    'in_panel': int((R.in_panel_r == e.in_panel).sum()),
    'src3_period': int((R.src3_r == e.src3_period).sum()),
    'period': int((R.period_r == e.period).sum()),
    'src_E': int(((R.srcE_r == e.src_E) | (R.srcE_r.isna() & e.src_E.isna())).sum()),
    'vol_ratio_E_abs<0.015': int(((R.vr_r - e.vol_ratio_E).abs() < 0.015).sum() + (R.vr_r.isna() & e.vol_ratio_E.isna()).sum()),
}
out['column_agree_of_rows'] = cmp
out['in_panel_n'] = int(R.in_panel_r.sum())

# ---- 原始轉錄抽查 ----
SL = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype={'code': str})
nm2c = dict(zip(SL.name, SL.code))
lt = pd.read_csv(L.TOP / 'lists_transcribed.tsv', sep='\t', dtype=str)
hd = pd.read_csv(L.TOP / 'hd_etf_transcribed.tsv', sep='\t', dtype=str)
ek = set(zip(e.type, e.name)); ekc = set(zip(e.type, e.code.astype(str)))
tot = hit = 0; miss = []
for r in lt.itertuples():
    typ = f"{r.index}_{r.action.upper()}"
    for nm in str(r.names).replace('，', '、').split('、'):
        nm = nm.strip()
        if not nm: continue
        tot += 1
        c = nm2c.get(nm)
        if (typ, nm) in ek or (c and (typ, c) in ekc): hit += 1
        else: miss.append((r.review, typ, nm))
for r in hd.itertuples():
    typ = f"{r.etf}_{r.action.upper()}"
    for nm in str(r.names).replace('，', '、').split('、'):
        nm = nm.strip()
        if not nm: continue
        tot += 1
        c = nm2c.get(nm)
        if (typ, nm) in ek or (c and (typ, c) in ekc): hit += 1
        else: miss.append((r.review, typ, nm))
out['transcribed_check'] = {'names': tot, 'found_in_events': hit, 'missing_sample': miss[:30], 'n_missing': len(miss)}

# ---- 資料關卡 ----
ms = e[(e.family == 'MSCI') & e.type.str.endswith('_DEL')]
ms_ok = ms[~ms.src3_period]
out['gate'] = {
    'msci_del_all': int(len(ms)), 'msci_del_ex_src3': int(len(ms_ok)),
    'msci_del_ex_src3_in_panel': int(ms_ok.in_panel.sum()),
    'msci_del_range': [str(ms.eff.min().date()), str(ms.eff.max().date())],
    'msci_eff_stated_share': float((ms.verify == 'eff_stated_in_source').mean()),
    'msci_eff_equals_stated': float((pd.to_datetime(ms.eff_stated) == ms.eff).mean()),
    'ftse_rows': int((e['index'].astype(str).str.contains('FTSE')).sum()),
}
# 各審核生效日（MSCI 刪除、面板內、非 src3）E 日成交值倍數中位數
dd = e.join(R)
g = dd[(dd.family == 'MSCI') & dd.type.str.endswith('_DEL') & dd.in_panel_r & ~dd.src3_r & dd.vr_r.notna()]
gm = g.groupby('eff').vr_r.median()
gm_all = dd[(dd.family == 'MSCI') & dd.type.str.endswith('_DEL') & dd.in_panel_r & dd.vr_r.notna()].groupby('eff').vr_r.median()
out['gate']['review_dates_ex_src3'] = int(len(gm)); out['gate']['vr_median_range_ex_src3'] = [round(gm.min(), 1), round(gm.max(), 1)]
out['gate']['n_dates_vr_gt1_ex_src3'] = int((gm > 1).sum())
out['gate']['review_dates_all'] = int(len(gm_all)); out['gate']['vr_median_range_all'] = [round(gm_all.min(), 1), round(gm_all.max(), 1)]
out['gate']['n_dates_vr_gt1_all'] = int((gm_all > 1).sum())
out['gate']['vr_by_date'] = {str(k.date()): round(v, 1) for k, v in gm_all.items()}
e2 = e.copy(); e2['t'] = R.t
e2.to_pickle(L.HERE / 'events_r.pkl')
L.dump(out, 'build_events.json')
for k, v in out.items():
    if k != 'gate': print(k, v)
print({k: v for k, v in out['gate'].items() if k != 'vr_by_date'})
