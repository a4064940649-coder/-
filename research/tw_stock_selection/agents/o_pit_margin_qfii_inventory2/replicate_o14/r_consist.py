import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import numpy as np, pandas as pd
from r_lib import *
import r_src as S

out = {}
t = S.tej().sort_values(['code', 'date'])
b = S.bb().sort_values(['code', 'date'])
h = S.mhist()
sm = S.snaps('margin-twse')
fb, fs = S.finbrain()

# FinBrain vs TEJ
fl = fb.stack().rename('fb').reset_index()
fl.columns = ['date', 'code', 'fb']
t['net'] = t.mbuy - t.msell
t['dbal'] = t.groupby('code').mbal.diff()
for allcodes in (True, False):
    x = t if allcodes else t[t.code.str.match(COMMON)]
    m = x.merge(fl, on=['date', 'code'])
    a = m.dropna(subset=['net', 'fb'])
    c = m.dropna(subset=['dbal', 'fb'])
    out[f'fb_eq_net_{allcodes}'] = (round(100 * (a.net == a.fb).mean(), 3), len(a))
    out[f'fb_eq_dbal_{allcodes}'] = (round(100 * (c.dbal == c.fb).mean(), 3), len(c))
# FinBrain sell vs TEJ short
fsl = fs.stack().rename('fs').reset_index(); fsl.columns = ['date', 'code', 'fs']
m = t.merge(fsl, on=['date', 'code']); m['snet'] = m.ssell - m.sbuy
a = m.dropna(subset=['snet', 'fs'])
out['fs_eq_snet'] = (round(100 * (a.snet == a.fs).mean(), 3), len(a))

# BB prev balance identity
b['lag'] = b.groupby('code').Margin_Balance.shift()
b['lagd'] = b.groupby('code').date.shift()
for nm, x in [('all', b), ('common', b[b.code.str.match(COMMON)])]:
    y = x.dropna(subset=['lag'])
    out[f'bb_prev_{nm}'] = (round(100 * (y.Margin_Prev_Balance == y.lag).mean(), 3), len(y))

# TEJ dbal <= net
for nm, x in [('all', t), ('common', t[t.code.str.match(COMMON)])]:
    y = x.dropna(subset=['dbal', 'net'])
    out[f'tej_dbal_le_net_{nm}'] = (round(100 * (y.dbal <= y.net).mean(), 3), len(y))
    y2 = y[y.groupby('code').date.shift().notna()]

# cross source cell agreement
def agree(a, ka, bb_, kb, nm, common=False):
    m = a[['date', 'code', ka]].merge(bb_[['date', 'code', kb]], on=['date', 'code']).dropna()
    if common:
        m = m[m.code.str.match(COMMON)]
    out[nm] = (round(100 * (m[ka] == m[kb]).mean(), 3), len(m), str(m.date.min())[:10], str(m.date.max())[:10])

for c in (False, True):
    sfx = '_common' if c else ''
    agree(t, 'mbal', b, 'Margin_Balance', 'tej_vs_bb' + sfx, c)
    agree(sm, 'margin_lots', t, 'mbal', 'snap_vs_tej' + sfx, c)
    agree(sm, 'margin_lots', b, 'Margin_Balance', 'snap_vs_bb' + sfx, c)
    agree(sm, 'margin_lots', h, 'margin_bal', 'snap_vs_mh' + sfx, c)
    agree(b, 'Margin_Balance', h, 'margin_bal', 'bb_vs_mh' + sfx, c)
    agree(b, 'Short_Balance', h, 'short_bal', 'bb_short_vs_mh' + sfx, c)
    agree(sm, 'short_lots', t, 'sbal', 'snap_short_vs_tej' + sfx, c)
# range
out['neg_bal'] = {'tej': int((t.mbal < 0).sum()), 'bb': int((b.Margin_Balance < 0).sum()),
                  'mh': int((h.margin_bal < 0).sum()), 'snap': int((sm.margin_lots < 0).sum())}
print(json.dumps(out, ensure_ascii=False, indent=0))
json.dump(out, open(f'{HERE}/my_consist.json', 'w'), ensure_ascii=False, indent=1)
