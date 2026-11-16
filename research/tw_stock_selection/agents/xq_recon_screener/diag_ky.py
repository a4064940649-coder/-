#!/usr/bin/env python3
"""對帳診斷 2：營收檔沒有的公司（KY、部分已下市；來源＝o_rev_survivorship/src_taiwanan.csv，2015-01～2019-08，含去年同月營收）補進後，
只上市、還原值、H21、strict、0.6% 的訊號與交易會多幾筆。可算 SURGE 的訊號月只有 2017-02～2019-09（需要 24 個月的年增金額）。
python3 -I -B diag_ky.py → diag_ky.json。"""
import sys, json, gc
from pathlib import Path
import numpy as np, pandas as pd

H = Path(__file__).resolve().parent
AG = H.parent
XT = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/xqrecon')
sys.path.insert(0, str(AG / 's19_xq_threshold'))
import s19lib                                                       # noqa: E402
from s19lib import s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib   # noqa: E402
for mod, a in [(s19lib, 'S19'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
               (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')]:
    setattr(mod, a, XT)
F = 0.006
P = s19lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22')
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
UX, Cadj, liqX = s19lib.script_universe(P, INNOV)
del Cadj, liqX
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'NEARHI', 'V60', 'liq', 'LIMUP', 'LIMDN', 'REF', 'F_MA20', 'AMT', 'VOLF'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
S0, _ = s19lib.rev_std(P)
r = pd.read_csv(stlib.REV_CSV, dtype={'code': str})
s = pd.read_csv(AG / 'o_rev_survivorship' / 'src_taiwanan.csv', dtype={'code': str})
miss = sorted(set(s['code']) - set(r['code']))
s = s[s['code'].isin(miss)]
a1 = pd.DataFrame(dict(month=s['ym'] + '-01', market='sii', code=s['code'], name=s['name'], revenue_k_twd=s['rev']))
ly = pd.PeriodIndex(s['ym'], freq='M') - 12
a2 = pd.DataFrame(dict(month=ly.strftime('%Y-%m-01'), market='sii', code=s['code'].to_numpy(), name=s['name'].to_numpy(), revenue_k_twd=s['rev_last_year'].to_numpy()))
add = pd.concat([a1, a2]).drop_duplicates(['month', 'code'], keep='first')
f = XT / 'rev_ky.csv'
pd.concat([r, add], ignore_index=True).to_csv(f, index=False)
orig = stlib.REV_CSV; stlib.REV_CSV = str(f)
S1, _ = s19lib.rev_std(P)
stlib.REV_CSV = orig
months = list(S0.index)
avB, _ = s19lib.b11_rows(P, months, 11)
names = dict(zip(add['code'], add['name']))


def sigs(S):
    Sg = s19lib.make_signal_thr(P, U, UX, None, S, avB, 2.5, 2.0)
    out = [(x['k'], P.ci[c], float(S.loc[pd.Period(x['month'], 'M')].iloc[P.ci[c]]), x['month']) for x in Sg['info'] for c in x['targets']]
    del Sg; gc.collect()
    return out


BARS = {}


def trades(sig):
    last = {}; out = []
    for k, j, sv, m in sorted(sig, key=lambda z: (z[0], -z[2])):
        b = BARS.setdefault(j, np.flatnonzero(P.has[:, j] & np.isfinite(P.O[:, j]) & (P.O[:, j] > 0)))
        p = int(np.searchsorted(b, k, side='right'))
        if p >= len(b) or last.get(j, -1) > k:
            continue
        e = int(b[p])
        if p + 21 < len(b):
            x = int(b[p + 21]); px = P.AO[x, j]; why = 'hold'
        else:
            x = int(b[-1]); px = P.G[x, j]; why = 'delist' if not P.MK[x + 1:, j].any() and x < P.T - 1 else 'open_end'
        out.append((m, P.COLS[j], why, (px / P.AO[e, j]) * (1 - F) / (1 + F) - 1))
        last[j] = x
    return pd.DataFrame(out, columns=['month', 'code', 'why', 'r'])


win = lambda d: d[(d['month'] >= '2017-01') & (d['month'] <= '2019-08') & (d['why'] != 'open_end')]
t0, t1 = win(trades(sigs(S0))), win(trades(sigs(S1)))
st = lambda t: dict(n=int(len(t)), win=float((t['r'] > 0).mean()), mean=float(t['r'].mean()), sum=float(t['r'].sum()))
newc = t1[t1['code'].isin(miss)]
res = dict(window='訊號 2017-02～2019-09（營收月 2017-01～2019-08）', n_months=int(t1['month'].nunique()), base=st(t0), augmented=st(t1), added_only=st(newc),
           added_list=[f'{m} {c} {names.get(c, "?")} r={x:.4f}' for m, c, x in zip(newc['month'], newc['code'], newc['r'])])
(H / 'diag_ky.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(json.dumps(res, ensure_ascii=False, indent=1))
