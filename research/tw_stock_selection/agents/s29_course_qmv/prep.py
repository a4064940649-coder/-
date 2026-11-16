#!/usr/bin/env python3
"""S29 第 0 步（不算任何報酬）：面板（EXRIGHT_FIX＝True）、SURGE、F1／L_MA 前 20 名上的 Q、M、V 涵蓋率、分位數、彼此相關、
季報落後季數、可得日來源，以及各候選規則會改動多少名單（只數檔數）。輸出 data_check.json。python3 -I -B prep.py"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s29lib
from s29lib import s10lib, s4lib, stlib

H = Path(__file__).resolve().parent
t = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s29lib.load()
A, _ = s4lib.innov_panel(P)
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
U = stlib.universe(P, 'twse') & ~A
del A
RT = s29lib.surge(P)
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
QM = s29lib.QMV(P, U)
df = s29lib.cand_table(P, U, RT, S0, QM)
df['mper'] = pd.PeriodIndex(df['month'], freq='M')
df['period'] = np.where(df['mper'] <= s29lib.TRAIN_LAST_MONTH, 'train', np.where(df['mper'] >= s29lib.VALID_FIRST_MONTH, 'valid', 'pre'))
print('載入', f'{time.time()-t:.0f}s', rss(), 'MB', flush=True)

chk = dict(panel=dict(T=P.T, N=P.N, first=str(P.IDX[0].date()), last=str(P.IDX[-1].date()), exright_fix=bool(P.exright_fix), exright_info=P.exright_info),
           definitions=dict(Q='ROE＝TTM 歸屬母公司淨利（連續 4 季）÷ 最近一季歸屬母公司權益；季報取市場可得日 A ≤ 訊號日的最新一季（A 缺 → 法定期限）',
                            M='12-1 動能＝G[k−21] ÷ G[k−252] − 1（含息）',
                            V='E/P＝TTM 歸屬母公司淨利 ÷ PIT 市值（股數落後 63 交易日）',
                            pct='同期百分位：訊號列 k 的股池 U[k]（只上市、流動性、非創新板）內有值者排名，0～1，越大越好',
                            not_available='ROIC、EV/EBIT 需要有息負債與現金，本機季報（fin_long）沒有這兩欄 → 改用 ROE（Hou-Xue-Zhang q 因子）與 E/P（S24 已用）'))
fsum = QM.F
chk['fin_table'] = dict(n_rows=int(len(fsum)), n_codes=int(fsum['code'].nunique()), avail_from_A_share=float(fsum['avail_from_A'].mean()),
                        ttm_cov=float(fsum['ttm'].notna().mean()), eq_cov=float(fsum['eq'].notna().mean()))
# 股池涵蓋率（每 12 個訊號列取一個）
cov = {}
for x in S0['info'][::12]:
    k = x['k']; r = QM.row(k); u = U[k]
    cov[x['month']] = dict(n_U=int(u.sum()), **{f'{c}_cov': float(np.isfinite(r[c][u]).mean()) for c in ('Q', 'M', 'V')})
chk['universe_coverage_every12'] = cov
qs = (0.1, 1 / 3, 0.5, 2 / 3, 0.9)
for per in ('train', 'valid'):
    o = {}
    for grp in ('F1', 'R11_20'):
        d = df[(df['period'] == per) & (df['grp'] == grp)]
        o[grp] = dict(n=int(len(d)), n_months=int(d['month'].nunique()),
                      coverage={c: float(np.isfinite(d[c]).mean()) for c in ('Q', 'M', 'V')},
                      pct_quantiles={c: [float(np.nanquantile(d[c], q)) for q in qs] for c in ('pQ', 'pM', 'pV', 'QMV')},
                      share_pV_lt_0_2=float((d['pV'] < 0.2).mean()), share_negative_EP=float((d['V'] < 0).mean()),
                      share_Q_negative=float((d['Q'] < 0).mean()),
                      lagq_hist={str(int(k)): int(v) for k, v in d['lagq'].dropna().value_counts().sort_index().items()},
                      spearman={f'{a}~{b}': float(d[[a, b]].corr(method='spearman').iloc[0, 1]) for a, b in
                                (('pQ', 'pM'), ('pQ', 'pV'), ('pM', 'pV'), ('pQ', 'surge'), ('pM', 'surge'), ('pV', 'surge'), ('pM', 'ret20'))})
    f1 = df[(df['period'] == per) & (df['grp'] == 'F1')]
    o['F1_count_per_month'] = {int(k): int(v) for k, v in f1.groupby('month').size().value_counts().sort_index().items()}
    chk[per] = o
# 規則改動名單（只數檔數）
eff = {}
for r in s29lib.RULES + [s29lib.CONTROL]:
    _, lg = s29lib.build_variant(P, U, RT, S0, QM, r)
    o = {}
    for per in ('train', 'valid'):
        L = [x for x in lg if (per == 'train' and pd.Period(x['month'], 'M') <= s29lib.TRAIN_LAST_MONTH) or (per == 'valid' and pd.Period(x['month'], 'M') >= s29lib.VALID_FIRST_MONTH)]
        o[per] = dict(n_orig=sum(len(x['orig']) for x in L), n_new=sum(len(x['new']) for x in L),
                      removed=sum(len(set(x['orig']) - set(x['new'])) for x in L), added=sum(len(set(x['new']) - set(x['orig'])) for x in L),
                      months_changed=sum(1 for x in L if x['orig'] != x['new']), n_months=len(L))
    eff[r['id']] = o
chk['rule_effect_preview'] = eff
chk['data_sufficiency'] = dict(rule='F1 上 Q、M、V 任一涵蓋率 < 80% → 該分數判資料不足，不進規則', ok={c: bool(min(chk[p]['F1']['coverage'][c] for p in ('train', 'valid')) >= 0.8) for c in ('Q', 'M', 'V')})
chk['runtime_s'] = round(time.time() - t, 1); chk['maxrss_mb'] = rss()
(H / 'data_check.json').write_text(json.dumps(chk, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in chk.items() if k not in ('universe_coverage_every12',)}, ensure_ascii=False, indent=1)[:9000])
