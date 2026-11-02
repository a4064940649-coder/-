"""Statistics from series.parquet: per j, then j-median."""
import sys, json
import numpy as np, pandas as pd
OUT = '/home/user/-/research/tw_stock_selection/agents/o_0050_tilt_pit/replicate_o8'
fn = sys.argv[1] if len(sys.argv) > 1 else 'series.parquet'
S = pd.read_parquet(OUT + '/' + fn)
PER = {'SEL': ('2014-01', '2018-12'), 'VAL': ('2019-01', '2023-12'), 'HOLD': ('2024-01', '2026-09'),
       'POST2019': ('2019-01', '2026-09')}


def tt(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return x.mean() * 100, x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


def cagr(r):
    return float(np.prod(1 + r) ** (12 / len(r)) - 1)


def mdd(r):
    v = np.cumprod(1 + np.asarray(r)); return float((v / np.maximum.accumulate(v) - 1).min())


def stats(g):
    a = g['T'] - g['B']; b = g['T'] - g['E50']
    m, t = tt(a); m2, t2 = tt(b)
    sd = a.std(ddof=1)
    srt = np.sort(a.values)[:-5]
    o = dict(n=len(g), vsB=m, vsB_t=t, vsB_IR=a.mean() * 12 / (sd * np.sqrt(12)), TE=sd * np.sqrt(12) * 100,
             vs0050=m2, vs0050_t=t2, st_vs0050=tt(g['T_st'] - g['E50'])[0],
             plc=tt(g['P_g'] - g['B_g'])[0], plc_t=tt(g['P_g'] - g['B_g'])[1],
             rnd=tt(g['T_g'] - g['P_g'])[0], rnd_t=tt(g['T_g'] - g['P_g'])[1],
             drop5_t=tt(srt)[1], CAGR_T=cagr(g['T']), CAGR_B=cagr(g['B']), CAGR_0050=cagr(g['E50']),
             MDD_T=mdd(g['T']), MDD_0050=mdd(g['E50']), turn=g['turn'].iloc[1:].mean(), cost_pp=g['cost'].iloc[1:].mean() * 100)
    yr = a.groupby(g['per'].str[:4]).sum()
    o['max_year_share'] = float(yr.max() / a.sum()) if a.sum() > 0 else np.nan
    return o


res = {}
for ver in ('T1', 'T2'):
    for pn, (s, e) in PER.items():
        per_j = {}
        for j in [str(x) for x in range(13)] + ['off']:
            g = S[(S.ver == ver) & (S.j == j) & (S.per >= s) & (S.per <= e)].sort_values('per')
            if len(g):
                per_j[j] = stats(g)
        df = pd.DataFrame(per_j).T
        nums = df.loc[[str(x) for x in range(13) if str(x) in df.index]].astype(float)
        res[f'{ver}|{pn}'] = {'jmed': nums.median().to_dict(), 'p10': nums.quantile(.1).to_dict(),
                              'off': df.loc['off'].to_dict() if 'off' in df.index else None}
json.dump(res, open(OUT + '/stats_' + fn.replace('.parquet', '.json'), 'w'), indent=1, default=float)
keys = ['n', 'vsB', 'vsB_t', 'vsB_IR', 'TE', 'vs0050', 'vs0050_t', 'st_vs0050', 'plc', 'plc_t', 'rnd', 'rnd_t',
        'drop5_t', 'CAGR_T', 'CAGR_B', 'CAGR_0050', 'MDD_T', 'MDD_0050', 'turn', 'cost_pp', 'max_year_share']
for k, v in res.items():
    print(k, ' '.join(f"{x}={v['jmed'][x]:.4g}" for x in keys))
    print('   p10 vsB=%.4g vs0050=%.4g' % (v['p10']['vsB'], v['p10']['vs0050']),
          '| off vsB=%.4g t=%.3g vs0050=%.4g t=%.3g' % (v['off']['vsB'], v['off']['vsB_t'], v['off']['vs0050'], v['off']['vs0050_t']) if v['off'] else '')
