# -*- coding: utf-8 -*-
"""Adversarial execution/cost review (exec o2) of o_issuance_avoid. Read-only on researcher files; prints only.
Run: python3 -I -B review_exec_o2.py"""
import sys, json, math, time
from pathlib import Path
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import numpy as np, pandas as pd
import oi_lib as L
K, IDX, N, CI = L.K, L.IDX, L.N, L.CI

# ---------- A. completion conditioning: filings that were NOT effective (withdrawn / returned) ----------
F = pd.read_csv(H / 'fsc_cases_all.csv', dtype={'code': str}, parse_dates=['recv', 'eff'])
F['item_n'] = F['item'].astype(str).str.replace(r'\s+', '', regex=True)
F = F[F['ctype'] == '上市']
isCB = F['item_n'].str.contains('轉換公司債') & ~F['item_n'].str.contains('海外|交換|附認股權|私募')
isSEO = F['item_n'].str.startswith('現金') & ~F['item_n'].str.contains('特')
print('=== A. 上市 filings by result ===')
for nm, m in (('CB', isCB), ('SEO', isSEO)):
    print(nm, F[m]['result'].value_counts().to_dict())
END = L.tpos(pd.Timestamp('2026-10-22')) - 1
def cars(df, a, b, anchor='recv'):
    v = []; ks = []
    for r in df.itertuples():
        if r.code not in CI or pd.isna(getattr(r, anchor)): continue
        t = L.tpos(getattr(r, anchor))
        if t is None or t + b > END: continue
        v.append(L.car(CI[r.code], t, a, b)); ks.append(str(pd.Timestamp(getattr(r, anchor)).to_period('M')))
    return L.cluster_t(v, ks)
for nm, m in (('CB', isCB), ('SEO', isSEO)):
    ok = F[m & F['result'].isin(['生效', '核減'])]; bad = F[m & ~F['result'].isin(['生效', '核減'])]
    for lab, d in (('effective', ok), ('NOT effective', bad)):
        for per, (pa, pb) in (('SEL', ('2005-01-01', '2019-02-01')), ('VAL', ('2019-05-01', '2026-08-01'))):
            s = d[(d['recv'] >= pa) & (d['recv'] <= pb)]
            c = cars(s, 1, 60)
            print(f'{nm} {lab:14s} {per} recv[+1,+60]: n={c["n"]} mean={c["mean"]*100 if c["n"] else float("nan"):.2f}% t_cl={c.get("t_cl", float("nan")):.2f}')

# ---------- B. CB recv->pricing run-up: tradability for a retail account ----------
E = L.load_events()
cb = E[(E['etype'] == 'CB') & E['recv'].notna()].copy()
rows = []
for r in cb.itertuples():
    t = L.tpos(r.recv)
    if t is None or t + 60 > END or t < 130: continue
    j = r.j
    c = L.car(j, t, 1, 60)
    if not np.isfinite(c): continue
    v = L.V60[t, j]; px = K.Cn[t, j]
    # buy at next open: was it a limit-up open (cannot buy)?
    gu = False
    try:
        gu = bool(K.gap_up_open(t + 1)[j])
    except Exception:
        gu = np.nan
    rows.append(dict(y=pd.Timestamp(r.recv).year, car=c, adv60=v, px=px, inpool=bool(L.INPOOL[t, j]), gu=gu,
                     ym=str(pd.Timestamp(r.recv).to_period('M'))))
B = pd.DataFrame(rows)
print('\n=== B. CB recv[+1,+60] tradability ===  n=', len(B))
print('ADV60 (NT$m) p10/p50/p90:', np.nanpercentile(B['adv60'] / 1e6, [10, 50, 90]).round(1))
print('share in top-300 pool:', round(B['inpool'].mean(), 3), ' share ADV60 < 30m:', round((B['adv60'] < 3e7).mean(), 3))
print('limit-up open next day share:', B['gu'].mean())
rt = 0.003 + 2 * 0.001425 * 0.6 + 2 * 0.002  # tax + discounted fee + 0.2% slippage each side
for lab, s in (('ALL', B), ('SEL<=2018', B[B.y <= 2018]), ('VAL>=2019', B[B.y >= 2019]), ('inpool', B[B.inpool]), ('outpool', B[~B.inpool]),
               ('ADV60>=50m', B[B.adv60 >= 5e7])):
    c = L.cluster_t(s['car'].to_numpy(), s['ym'].to_numpy())
    print(f'{lab:12s} n={c["n"]} mean={c["mean"]*100:.2f}% med={c.get("median", np.nan)*100:.2f}% t_cl={c["t_cl"]:.2f} net(rt {rt*100:.2f}%)={(c["mean"]-rt)*100:.2f}%')
# capacity: position as % of ADV for 1m / 10m (NT$) account, 5 equal names
for acct in (3e5, 1e6, 1e7):
    pos = acct / 5
    print(f'acct {acct/1e4:.0f}萬 per-name {pos/1e4:.0f}萬: median %ADV60 = {np.nanmedian(pos / B["adv60"]) * 100:.3f}%')

# ---------- C. avoid list turnover / extra trades ----------
import iterate as IT
S = L.strategy(exfin=True)
t0, t1, sd = S.bounds()
for n in (10, 30):
    bpl, _ = L.plan_avoid(S, n)
    vpl, skipped = L.plan_avoid(S, n, IT.make_avoid(IT.VERSIONS['V1_both_90d']), sell_held=False)
    nb, lgb = L.run_nav(S, bpl); nv, lgv = L.run_nav(S, vpl)
    def names(pl):
        tot = 0; prev = set()
        for p in pl:
            cur = set(p[1]); tot += len(cur - prev); prev = cur
        return tot
    print(f'\n=== C. n{n} turnover base={lgb.get("turnover")} V1={lgv.get("turnover")}  new buys base={names(bpl)} V1={names(vpl)} skipped={len(skipped)}')
    # how often the avoid list fires on a stock that the ORIGINAL plan already held (kept) vs fresh buy blocked
