"""步驟 6：樣本外（驗證 2019～2023、保留 2025-09-24～2026-10、合併判定窗）＋ P1／P3（全部刪除事件）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import rlib as L

x = L.event_frame()
D = x[x.tradable & x.is_del].copy()   # 刪除類 pressure 都有值
D['yr'] = D.eff.dt.year
out = {}
W = {'validate': D[D.win == 'validate'], 'holdout': D[D.win == 'holdout'], 'oos': D[D.win.isin(['validate', 'holdout'])]}
for k, g in W.items():
    hi = g[g.high]
    out[k] = {'high': L.summ(hi, 'net20'), 'low': L.summ(g[~g.high], 'net20'), 'HL': L.hl(g, 'net20'),
              'high_stress': L.summ(hi, 'net20s'), 'high_vs0050': L.summ(hi, 'net20_50')}
o = W['oos']; hi = o[o.high]
out['oos_segments'] = {'2019_2020': L.summ(hi[hi.yr <= 2020], 'net20'), '2021+': L.summ(hi[hi.yr >= 2021], 'net20')}
out['oos_high_list'] = hi[['code', 'name', 'type', 'eff', 'pressure', 'net20']].assign(eff=lambda z: z.eff.dt.date.astype(str)).round(4).to_dict('records')
r = out['oos']
c1 = r['high']['mean'] > 0 and r['high']['t_month'] >= 2 and r['high']['median'] > 0 and r['high']['trim5'] > 0
c2 = (r['HL']['paired'] or -1) > 0 and (r['HL']['t_paired'] or -9) >= 1.5
c3 = out['oos_segments']['2019_2020'].get('mean', -1) > 0 and out['oos_segments']['2021+'].get('mean', -1) > 0
out['verdict'] = {'i': bool(c1), 'ii': bool(c2), 'iii': bool(c3), 'validate_net_le0': bool(out['validate']['high']['mean'] <= 0),
                  'call': 'usable?' if c1 else '不可用'}

# ---- P1／P3：全部刪除事件（可交易），淨 20 日 ----
p = {}
sel = D[D.win == 'select']
p['select_all_del'] = L.summ(sel, 'net20')
p['oos_all_del'] = L.summ(o, 'net20')
p['oos_by_class'] = {c: L.summ(g, 'net20') for c, g in o.groupby('cls')}
p['oos_by_class_HD_pooled'] = L.summ(o[o.cls.str.startswith('HD_')], 'net20')
# P3：成交值分組（挑選期 adv20 中位數為切點；另報 OOS 內部中位數）
m_sel = sel.adv20.median(); m_oos = o.adv20.median()
p['adv20_cut_select_median'] = float(m_sel)
p['oos_lowADV_selcut'] = L.summ(o[o.adv20 < m_sel], 'net20'); p['oos_highADV_selcut'] = L.summ(o[o.adv20 >= m_sel], 'net20')
p['oos_lowADV_ooscut'] = L.summ(o[o.adv20 < m_oos], 'net20'); p['oos_highADV_ooscut'] = L.summ(o[o.adv20 >= m_oos], 'net20')
q = o.adv20.quantile([1 / 3, 2 / 3]).to_numpy()
p['oos_lowADV_tertile'] = L.summ(o[o.adv20 < q[0]], 'net20'); p['oos_highADV_tertile'] = L.summ(o[o.adv20 >= q[1]], 'net20')
p['select_lowADV'] = L.summ(sel[sel.adv20 < m_sel], 'net20'); p['select_highADV'] = L.summ(sel[sel.adv20 >= m_sel], 'net20')
out['P1_P3'] = p
L.dump(out, 'oos.json')
import pprint; pprint.pprint({k: v for k, v in out.items() if k != 'oos_high_list'}, width=170)
for z in out['oos_high_list']: print(z)
