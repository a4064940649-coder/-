#!/usr/bin/env python3
"""複製員：彙整各版本關鍵數字、名單差異、新進股持有期超額、配對 t。輸出 my_compare.json。"""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from r_lib import *

def L(tag): return json.loads((H / f'my_run_{tag}.json').read_text())
VERS = ['ORIG', 'R_DEL', 'R_ALL', 'B05_DEL', 'B10_DEL', 'B05_ALL', 'B10_ALL', 'PX_ORIG', 'PX_DEL',
        'B05_DEL_dl', 'B10_DEL_dl', 'B05_ALL_dl', 'B10_ALL_dl']
R = {v: L(v) for v in VERS}
out = {'table': {}}
for v in VERS:
    r = R[v]; e = r['EXFIN']['n10']['normal']
    out['table'][v] = {
        'EXFIN_n10': {k: (round(e[k]['CAGR'], 2), round(e[k]['t_0050'], 2)) for k in ('ALL', 'S1', 'S2', 'S3', 'S4', 'P1', 'SEL', 'VAL', 'HOLD')},
        'EXFIN_n10_ALL_extra': {'t_EW': round(e['ALL']['t_EW'], 2), 'maxDD': round(e['ALL']['maxDD'], 1), 'Calmar': round(e['ALL']['Calmar'], 2),
                                't_0050_cal': round(e['ALL']['t_0050_cal'], 2), 'ex50_m_mean_pp': round(e['ALL']['ex50_m_mean_pp'], 3),
                                'ex50_m_sd_pp': round(e['ALL']['ex50_m_sd_pp'], 2)},
        'EXFIN_n10_stress_ALL': (round(r['EXFIN']['n10']['stress']['ALL']['CAGR'], 2), round(r['EXFIN']['n10']['stress']['ALL']['t_0050'], 2)),
        'ALL_n10': (round(r['ALL']['n10']['normal']['ALL']['CAGR'], 2), round(r['ALL']['n10']['normal']['ALL']['t_0050'], 2)),
        'EXFIN_n30': (round(r['EXFIN']['n30']['normal']['ALL']['CAGR'], 2), round(r['EXFIN']['n30']['normal']['ALL']['t_0050'], 2)),
        'AB_n10': (round(r['AB']['n10']['normal']['ALL']['CAGR'], 2), round(r['AB']['n10']['normal']['ALL']['t_0050'], 2)),
        'live_n10': {k: round(r['AB']['n10']['normal'][k]['live_exp'], 1) for k in ('ALL', 'P1', 'S3', 'S4')},
        'live_n10_stress_ALL': round(r['AB']['n10']['stress']['ALL']['live_exp'], 1),
        'live_n30_ALL': round(r['AB']['n30']['normal']['ALL']['live_exp'], 1),
    }

# ---------- 名單差異與新進股 ----------
def plans(tag): return json.loads((H / f'my_plans_{tag}.json').read_text())
P0 = plans('ORIG')
N0 = pd.read_parquet(H / 'my_navs_ORIG.parquet')
def hp_returns(pl_codes, sd_all):
    """每個訊號日的持有期（成交日開盤→下一成交日開盤）每檔報酬（無成本），用 V.stock_contrib。"""
    pl = []
    for d, cs in pl_codes:
        t = IDX.get_loc(pd.Timestamp(d)) + 1
        pl.append((t, np.array([K.CI[c] for c in cs], int), None, pd.Timestamp(d)))
    _, per = V.stock_contrib(pl, IDX.get_loc(V.END1012))
    return {str(d.date()): dict(zip([COLS[j] for j in pk], r)) for d, pk, r in per}
diff = {}
for v in ('R_DEL', 'R_ALL', 'B05_DEL', 'B10_DEL'):
    P1 = plans(v)
    for key in ('EXFIN|n10', 'ALL|n10', 'EXFIN|n30'):
        a = dict((d, set(c)) for d, c in P0[key]); b = dict((d, set(c)) for d, c in P1[key])
        slots = {d: len(b[d] - a.get(d, set())) for d in b}
        tot = sum(slots.values())
        by_year = {}
        for d, s in slots.items():
            if s: by_year[d[:4]] = by_year.get(d[:4], 0) + s
        rec = {'slots': tot, 'by_year': by_year, 'months_diff': int(sum(1 for s in slots.values() if s))}
        if v == 'R_DEL':
            hp = hp_returns(P1[key], None)
            ex = []; months = set()
            for d in b:
                new = b[d] - a.get(d, set())
                if not new: continue
                oth = [hp[d][c] for c in b[d] if c not in new]
                for c in new:
                    ex.append((hp[d][c] - np.mean(oth)) * 100); months.add(d)
            ex = np.array(ex)
            rec['entrants'] = {'n': len(ex), 'months': len(months), 'mean_pp': round(float(ex.mean()), 2),
                               't': round(float(ex.mean() / (ex.std(ddof=1) / math.sqrt(len(ex)))), 2) if len(ex) > 2 else None}
        diff.setdefault(v, {})[key] = rec
out['diff'] = diff

# ---------- 配對 R_DEL − ORIG（訊號窗） ----------
N1 = pd.read_parquet(H / 'my_navs_R_DEL.parquet')
sd = pd.DatetimeIndex(R['ORIG']['EXFIN']['sd'])
def winret(nav):
    D = pd.DatetimeIndex([d for d in sd if d in nav.index]); x = nav.reindex(D)
    return (x.shift(-1) / x - 1).iloc[:-1]
d = (winret(N1['EXFIN|n10|normal']) - winret(N0['EXFIN|n10|normal'])).dropna() * 100
d = d[(d.index >= '2006-04-01') & (d.index <= '2026-10-12')]
out['paired_RDEL_minus_ORIG'] = {'mean_pp': round(float(d.mean()), 4), 't': round(float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))), 2),
                                 'n_nonzero': int((d.abs() > 1e-9).sum()), 'n': int(len(d))}
(H / 'my_compare.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=1))
