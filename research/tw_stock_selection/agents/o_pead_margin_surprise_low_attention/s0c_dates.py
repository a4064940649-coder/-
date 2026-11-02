#!/usr/bin/env python3
"""第 0 步 C：季報「市場可得日」A（PIT）。
來源：證交所 BWIBBU／櫃買每日本益比（公開封存），當天盤後公布。
方法優先序（事前寫死）：
  1. fsq：上市 2017 年起 BWIBBU 有「財報年/季」欄，第一個顯示本季的日子＝A。
  2. ttm：隱含近四季 EPS＝收盤÷本益比；以 MOPS 單季 EPS 算出「公告前 TTM」與「公告後 TTM」，
     隱含值第一次越過兩者中點且之後 2 個觀測都維持＝A（|ΔTTM| ≥ max(0.02, 1%|TTM|) 才可用）。
  3. bv：隱含每股淨值＝收盤÷淨值比，前 3 觀測中位數 vs 後 3 觀測中位數相對變動 ≥ max(1.5%, 2×0.005/PB) 的第一天。
  4. deadline：法定期限（保守）。
進場：A 的下一個交易日開盤（j0）。搜尋窗：季底＋14 天～法定期限＋30 天。
驗證：上市有 fsq 的事件，用 ttm／bv 獨立推算，±1 交易日內比例（門檻 ≥ 90%）。"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from pcore import DATA, deadline, qend, load_panel

fin = pd.read_parquet(DATA / 'fin_long.parquet')
fin = fin[fin['code'].str.fullmatch(r'\d{4}')]
per = pd.read_parquet(DATA / 'per_long.parquet')
per = per[per['code'].str.fullmatch(r'\d{4}')]
P = load_panel()
IDX = pd.DatetimeIndex(P['IDX']); COLS = list(P['COLS']); CI = {c: j for j, c in enumerate(COLS)}
C = P['C']
TD = IDX  # 交易日曆（2015 起）

# 單季 EPS 寬表（季序號 k = 4*y + q - 1）
fin['k'] = fin['y'] * 4 + fin['q'] - 1
E = fin.pivot(index='code', columns='k', values='eps_s')
ks = np.arange(E.columns.min(), E.columns.max() + 1)
E = E.reindex(columns=ks)
TTM = E.rolling(4, axis=1, min_periods=4).sum() if False else E.T.rolling(4, min_periods=4).sum().T

rows = []
per = per.sort_values(['code', 'date'])
for code, g in per.groupby('code', sort=False):
    if code not in E.index:
        continue
    dts = pd.DatetimeIndex(g['date'].to_numpy())
    pe = g['per'].to_numpy(float); pb = g['pbr'].to_numpy(float)
    cl = g['close'].to_numpy(float)
    if code in CI:  # 收盤缺時用面板收盤
        pc = pd.Series(C[:, CI[code]], index=IDX).reindex(dts).to_numpy()
        cl = np.where(np.isfinite(cl), cl, pc)
    fsq = g['fs_quarter'].fillna('').to_numpy(str)
    ttm_imp = np.where((pe > 0) & np.isfinite(cl), cl / pe, np.nan)
    bv_imp = np.where((pb > 0) & np.isfinite(cl), cl / pb, np.nan)
    eks = [k for k in E.columns if np.isfinite(E.at[code, k]) and k // 4 >= 2015]
    for k in eks:
        y, q = k // 4, k % 4 + 1
        w0 = qend(y, q) + pd.Timedelta(days=14); dl = deadline(y, q); w1 = dl + pd.Timedelta(days=30)
        m = (dts >= w0) & (dts <= w1)
        if m.sum() < 3:
            rows.append(dict(code=code, y=y, q=q, A_fsq=pd.NaT, A_ttm=pd.NaT, A_bv=pd.NaT, nobs=int(m.sum())))
            continue
        wd = dts[m]
        # 1. fsq
        tag = f'{y - 1911}/{q}'
        hit = np.where(fsq[m] == tag)[0]
        # 有效條件：命中前一個觀測已有（不同的）財報季標記，才算觀察到「切換」
        okf = bool(len(hit) and hit[0] > 0 and fsq[m][hit[0] - 1] not in ('', tag))
        A_fsq = wd[hit[0]] if okf else pd.NaT
        early_fsq = bool(len(hit) and not okf)
        # 2. ttm
        A_ttm = pd.NaT; early_ttm = False
        b = TTM.at[code, k - 1] if (k - 1) in TTM.columns else np.nan
        a = TTM.at[code, k]
        ti = ttm_imp[m]
        if np.isfinite(a) and np.isfinite(b) and abs(a - b) >= max(0.02, 0.01 * abs(b)):
            z = (ti - b) / (a - b)
            ok = np.isfinite(z)
            zi = np.where(ok)[0]
            for ii, j in enumerate(zi):
                if z[j] > 0.5 and all(z[jj] > 0.5 for jj in zi[ii + 1:ii + 3]):
                    # 前面要有至少一個 z<0.5 的觀測，否則標記為「窗口開始即已更新」
                    early_ttm = not any(z[jj] < 0.5 for jj in zi[:ii])
                    A_ttm = wd[j] if not early_ttm else pd.NaT   # 窗口一開始就已更新：無法判定切換日，不用
                    break
        # 3. bv
        A_bv = pd.NaT
        bi = bv_imp[m]; pbi = pb[m]
        okb = np.where(np.isfinite(bi))[0]
        for ii in range(3, len(okb) - 2):
            before = np.median(bi[okb[ii - 3:ii]]); after = np.median(bi[okb[ii:ii + 3]])
            thr = max(0.015, 2 * 0.005 / max(pbi[okb[ii]], 0.05))
            if before > 0 and abs(after / before - 1) >= thr and abs(bi[okb[ii]] / before - 1) >= thr:
                A_bv = wd[okb[ii]]
                break
        rows.append(dict(code=code, y=y, q=q, A_fsq=A_fsq, A_ttm=A_ttm, A_bv=A_bv, early_fsq=early_fsq, early_ttm=early_ttm,
                         nobs=int(m.sum()), mkt=g['mkt'].iloc[-1]))

D = pd.DataFrame(rows)


def tdiff(a, b):
    ok = a.notna() & b.notna()
    ia = TD.searchsorted(a[ok]); ib = TD.searchsorted(b[ok])
    return pd.Series(ia - ib, index=a[ok].index)


val = {}
for meth in ['A_ttm', 'A_bv']:
    dd = tdiff(D[meth], D['A_fsq'])
    val[meth] = {'n': int(len(dd)), 'within1': float((dd.abs() <= 1).mean()) if len(dd) else None,
                 'exact': float((dd == 0).mean()) if len(dd) else None, 'median_diff': float(dd.median()) if len(dd) else None,
                 'early_share': float((dd < 0).mean()) if len(dd) else None, 'hist': {int(k): int(v) for k, v in dd.clip(-5, 5).value_counts().sort_index().items()}}
# 最終 A
D['A'] = D['A_fsq']; D['A_method'] = np.where(D['A_fsq'].notna(), 'fsq', '')
# 沒有 fsq 時：ttm 與 bv 都有 → 取較晚者（保守，降低前視風險）
both = D['A'].isna() & D['A_ttm'].notna() & D['A_bv'].notna()
D.loc[both, 'A'] = D.loc[both, ['A_ttm', 'A_bv']].max(axis=1); D.loc[both, 'A_method'] = 'ttm_bv_max'
for meth, nm in [('A_ttm', 'ttm'), ('A_bv', 'bv')]:
    sel = D['A'].isna() & D[meth].notna()
    D.loc[sel, 'A'] = D.loc[sel, meth]; D.loc[sel, 'A_method'] = nm
D['deadline'] = [deadline(y, q) for y, q in zip(D['y'], D['q'])]
sel = D['A'].isna()
D.loc[sel, 'A'] = D.loc[sel, 'deadline']; D.loc[sel, 'A_method'] = 'deadline'
# A 對齊到交易日（非交易日→下一交易日）
ia = TD.searchsorted(pd.DatetimeIndex(D['A']))
ia = np.minimum(ia, len(TD) - 1)
D['A'] = TD[ia]
D['late'] = D['A'] > D['deadline'] + pd.Timedelta(days=1)
# 期限遵守：A ≤ 期限＋1 個交易日的比例（fsq 法）
f = D[D['A_method'] == 'fsq']
ddl = TD.searchsorted(pd.DatetimeIndex(f['deadline']))
comp = float(((TD.searchsorted(pd.DatetimeIndex(f['A'])) - ddl) <= 1).mean()) if len(f) else None
D.to_parquet(DATA / 'avail_dates.parquet', index=False)
summ = {'validation_vs_fsq': val, 'fsq_deadline_compliance_within1td': comp,
        'method_counts': D['A_method'].value_counts().to_dict(),
        'method_by_year_mkt': D.groupby(['y', 'mkt'])['A_method'].value_counts().unstack(fill_value=0).reset_index().to_dict('records'),
        'fsq_first_date': str(per.loc[per['fs_quarter'].notna() & (per['fs_quarter'] != ''), 'date'].min())}
json.dump(summ, open(DATA / 'avail_dates_summary.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: summ[k] for k in ['validation_vs_fsq', 'fsq_deadline_compliance_within1td', 'method_counts', 'fsq_first_date']}, ensure_ascii=False, default=str))
