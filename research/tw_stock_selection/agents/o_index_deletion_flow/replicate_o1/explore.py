"""步驟 3：挑選期（2008～2018）事件研究。異常＝個股 − 全體上市等權；可交易＝面板內、流動性、E 有成交、非 src3（事件研究不排除漲停）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import rlib as L

x = L.event_frame()
s = x[x.explorable & (x.win == 'select')].copy()
out = {'n_explorable_select': int(len(s)), 'by_class_n': s.cls.value_counts().to_dict()}

def blk(g):
    o = {'n': int(len(g))}
    for k in ['pre10', 'abE', 'abE1', 'car5', 'car20', 'car2_20', 'sc5', 'sc20', 'sp10']:
        v = g[k].dropna()
        o[k] = round(v.mean() * 100, 2)
        o[k + '_t_evt'] = round(L.tstat(v), 2)
        o[k + '_t_month'] = round(L.cl_t(g.dropna(subset=[k]), k)[0], 2)
    o['vr_median'] = round(g.vr.median(), 1)
    return o
out['classes'] = {c: blk(g) for c, g in s.groupby('cls')}

dl = s[s.is_del & s.pressure.notna()].copy()
out['pressure_median_select_del_explorable'] = round(float(dl.pressure.median()), 4)
tr = x[x.tradable & (x.win == 'select') & x.is_del & x.pressure.notna()]
out['pressure_median_select_del_tradable'] = round(float(tr.pressure.median()), 4)
dl['ter'] = pd.qcut(dl.pressure, 3, labels=['low', 'mid', 'high'])
out['tertiles'] = {str(k): {'n': int(len(g)), 'vr_median': round(g.vr.median(), 1), 'abE': round(g.abE.mean() * 100, 2),
                            'car20_gross': round(g.car20.mean() * 100, 2), 'sc20': round(g.sc20.mean() * 100, 2), 'pressure_range': [round(g.pressure.min(), 2), round(g.pressure.max(), 2)]}
                   for k, g in dl.groupby('ter', observed=True)}
L.dump(out, 'explore.json')
import pprint; pprint.pprint(out, width=160)
