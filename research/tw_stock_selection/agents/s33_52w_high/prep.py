#!/usr/bin/env python3
"""S33 第 0 步（不算任何報酬）：面板（EXRIGHT_FIX＝True）、SURGE、F1／L_MA 前 20 名上的 PTH 涵蓋率、分位數、與 SURGE／RET20 的相關、
以及各候選規則會改動多少名單（只數檔數）。輸出 data_check.json。python3 -I -B prep.py"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s33lib
from s33lib import s10lib, s4lib, stlib

H = Path(__file__).resolve().parent
t = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s33lib.load()
A, _ = s4lib.innov_panel(P)
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
U = stlib.universe(P, 'twse') & ~A
del A
RT = s33lib.surge(P)
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
PS = s33lib.PTHS(P, U)
df = s33lib.cand_table(P, U, RT, S0, PS)
df['mper'] = pd.PeriodIndex(df['month'], freq='M')
df['period'] = np.where(df['mper'] <= s33lib.TRAIN_LAST_MONTH, 'train', np.where(df['mper'] >= s33lib.VALID_FIRST_MONTH, 'valid', 'pre'))
print('載入', f'{time.time()-t:.0f}s', rss(), 'MB', flush=True)

# PIT 自檢：PTH[k] 只用 k 當日以前收盤（把 k+1 之後的 G 改掉，PTH[k] 不變）
G0 = P.G.copy(); kk = 1500
P.G[kk + 1:] *= 3.0
pth2 = s33lib.pth_panel(P)
pit_ok = bool(np.array_equal(np.nan_to_num(pth2[:kk + 1], nan=-9), np.nan_to_num(P.PTH[:kk + 1], nan=-9)))
P.G = G0; del pth2, G0

chk = dict(panel=dict(T=P.T, N=P.N, first=str(P.IDX[0].date()), last=str(P.IDX[-1].date()), exright_fix=bool(P.exright_fix), exright_info=P.exright_info),
           definition='PTH＝還原收盤 G[k] ÷ 過去 252 個交易日（含訊號日）最高還原收盤；上市滿 252 根才有值。pPTH＝訊號列股池 U[k] 內同期百分位',
           pit_selfcheck=pit_ok)
qs = (0.1, 0.25, 1 / 3, 0.5, 2 / 3, 0.75, 0.9)
for per in ('train', 'valid'):
    o = {}
    for grp in ('F1', 'R11_20', 'POOL_NOMOM'):
        d = df[(df['period'] == per) & (df['grp'] == grp)]
        if len(d) == 0:
            continue
        o[grp] = dict(n=int(len(d)), n_months=int(d['month'].nunique()), coverage=float(np.isfinite(d['PTH']).mean()),
                      PTH_quantiles=[float(np.nanquantile(d['PTH'], q)) for q in qs], pPTH_quantiles=[float(np.nanquantile(d['pPTH'], q)) for q in qs],
                      share_PTH_ge_0_9=float((d['PTH'] >= 0.9).mean()), share_PTH_eq_1=float((d['PTH'] >= 0.99999).mean()),
                      spearman={f'PTH~{b}': float(d[['PTH', b]].corr(method='spearman').iloc[0, 1]) for b in ('surge', 'ret20', 'lma_rank')})
    f1 = df[(df['period'] == per) & (df['grp'] == 'F1')]
    o['F1_count_per_month'] = {int(k): int(v) for k, v in f1.groupby('month').size().value_counts().sort_index().items()}
    top10 = df[(df['period'] == per) & (df['lma_rank'] <= 10)]
    o['top10_gate_compare'] = dict(n=int(len(top10)), ret20_gt_10=int((top10['ret20'] > 0.1).sum()), pth_ge_0_9=int((top10['PTH'] >= 0.9).sum()),
                                   both=int(((top10['ret20'] > 0.1) & (top10['PTH'] >= 0.9)).sum()), pth_nan=int(top10['PTH'].isna().sum()))
    chk[per] = o
eff = {}
for r in s33lib.RULES + [s33lib.CONTROL]:
    _, lg = s33lib.build_variant(P, U, RT, S0, PS, r)
    o = {}
    for per in ('train', 'valid'):
        L = [x for x in lg if (per == 'train' and pd.Period(x['month'], 'M') <= s33lib.TRAIN_LAST_MONTH) or (per == 'valid' and pd.Period(x['month'], 'M') >= s33lib.VALID_FIRST_MONTH)]
        o[per] = dict(n_orig=sum(len(x['orig']) for x in L), n_new=sum(len(x['new']) for x in L),
                      removed=sum(len(set(x['orig']) - set(x['new'])) for x in L), added=sum(len(set(x['new']) - set(x['orig'])) for x in L),
                      months_changed=sum(1 for x in L if x['orig'] != x['new']), n_months=len(L))
    eff[r['id']] = o
# FILL_PTH 與 FILL_ALL 名單差異（檔數）
_, la = s33lib.build_variant(P, U, RT, S0, PS, s33lib.RULES[2]); _, lb = s33lib.build_variant(P, U, RT, S0, PS, s33lib.CONTROL)
eff['FILL_PTH_vs_FILL_ALL'] = dict(months_diff=sum(1 for a, b in zip(la, lb) if set(a['new']) != set(b['new'])),
                                   names_diff=sum(len(set(a['new']) ^ set(b['new'])) for a, b in zip(la, lb)))
chk['rule_effect_preview'] = eff
chk['data_sufficiency'] = dict(rule='F1 上 PTH 涵蓋率 < 80% → 資料不足', ok=bool(min(chk[p]['F1']['coverage'] for p in ('train', 'valid')) >= 0.8))
chk['runtime_s'] = round(time.time() - t, 1); chk['maxrss_mb'] = rss()
(H / 'data_check.json').write_text(json.dumps(chk, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in chk.items() if k != 'panel'}, ensure_ascii=False, indent=1)[:9000])
