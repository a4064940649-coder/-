#!/usr/bin/env python3
"""探索 (1)：2006-04～2012-12 缺口格清單（V4 規格：R_FIX、成交日處置、fin2、排除 KY）。只看資料，不看報酬。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import v5lib as L
L4, K = L.L4, L.K
L4.check_freeze(); L4.setup_px()
YFX, Y5F = L4.load_rfix()
S = L4.V4(YFX, Y5F, L4.disp_mask_fn('trade'))
G = L.gap_cells(S)
G['risk'] = G.s52_pct >= 0.7
# 每格缺的是哪些水準月（tw_revenue sii 與 taiwanan 都沒有）
d = pd.read_csv(K.LONG, dtype={'code': str, 'market': str}); d = d[d.market == 'sii']
d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
LV = d.pivot_table(index='p', columns='code', values='revenue_k_twd', aggfunc='first')
miss = []
for s, c in zip(G.sig, G.code):
    ks = L.needed_levels(s, c)
    mm = [str(k) for k in ks if not (c in LV.columns and k in LV.index and np.isfinite(LV.at[k, c]) and LV.at[k, c] > 0)]
    miss.append(';'.join(mm))
G['miss_levels'] = miss
G.to_csv(HERE / 'data' / 'gap_cells_2006_2012.csv', index=False)
need = sorted({(c, m) for c, ms in zip(G.code, G.miss_levels) for m in ms.split(';') if m})
pd.DataFrame(need, columns=['code', 'ym']).to_csv(HERE / 'data' / 'gap_needed_levels.csv', index=False)
dl = pd.read_csv(L4.V3_INPUTS['delist_official'][0], dtype=str); nm = dict(zip(dl.code, dl.name))
by = G.groupby('code').agg(n=('sig', 'size'), n_risk=('risk', 'sum'), first=('sig', 'min'), last=('sig', 'max'), delisted=('delisted', 'max')).sort_values('n', ascending=False)
by['name'] = [nm.get(c, K.COLS.get_loc(c) and '') for c in by.index]
by['name'] = [nm.get(c) or (L.V.SL['name'].get(c) if c in L.V.SL.index else '') for c in by.index]
summ = {'n_cells': int(len(G)), 'n_sig_days': int(S.sig_days.__len__()), 'n_days_in_range': int(G.sig.nunique()),
        'cells_per_day': float(len(G) / len([x for x in S.sig_days if pd.Timestamp('2006-04-01') <= x <= pd.Timestamp('2012-12-31')])),
        'n_risk_cells': int(G.risk.sum()), 'n_codes': int(G.code.nunique()), 'n_delisted_cells': int(G.delisted.sum()),
        'by_year': G.groupby(G.sig.str[:4]).size().to_dict(), 'risk_by_year': G[G.risk].groupby(G[G.risk].sig.str[:4]).size().to_dict(),
        'n_needed_levels': len(need)}
print(json.dumps(summ, ensure_ascii=False, indent=1))
print(by.head(80).to_string())
(HERE / 'data' / 'gap_codes_2006_2012.csv').write_text(by.to_csv(), encoding='utf-8')
json.dump(summ, open(HERE / 'data' / 'gap_summary.json', 'w'), ensure_ascii=False, indent=1)
