#!/usr/bin/env python3
"""integrate_o9 正式計算：照 results.json 的 PREREG，C（o_capmatched_pit_close）× M（AB_V7 排除金融 n10）的第二條腿固定比例混合。
只讀：C 的 out/*.npz、本資料夾 out/cm_exec_{a,b}.npz（cm_exec.py 產生）、V7 的 navs_jscan_v7／navs_v7／navs_exec7。RESULTS 只寫一次。
python3 -I -B run.py"""
import json, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent
CM = AG / 'o_capmatched_pit_close'
V7 = AG / 'o_v7_listing_exec_engine'
RJ = HERE / 'results.json'
RES = json.loads(RJ.read_text(encoding='utf-8'))
PRE = RES['PREREG']
assert hashlib.sha256(json.dumps(PRE, ensure_ascii=False, sort_keys=True).encode()).hexdigest() == RES['PREREG_sha256'], 'PREREG 被改過'
assert 'RESULTS' not in RES, 'RESULTS 已存在：只跑一次'
for f, h in PRE['sha256'].items():
    assert hashlib.sha256(Path(f).read_bytes()).hexdigest() == h, f
for c in ('a', 'b'):
    assert json.loads((HERE / 'out' / f'cm_exec_{c}_check.json').read_text())['reproduce_maxdiff'] <= 1e-9

JK = [f'J{j}' for j in range(13)]
WS = [0.8, 0.5, 0.2]
COST = {'normal': (0.001855, 0.004855), 'stress': (0.005, 0.005)}
TIERS = [('100萬', 'FIX'), ('100萬', 'GROW'), ('300萬', 'FIX'), ('300萬', 'GROW'), ('1000萬', 'FIX'), ('1000萬', 'GROW')]

def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return float('nan')
    s = x.std(ddof=1); return float(x.mean() / (s / math.sqrt(len(x)))) if s > 0 else float('nan')
def wr(nav, days):
    nav = nav.dropna(); ds = pd.DatetimeIndex([str(x) for x in days])
    assert ds[-1] <= nav.index[-1] and ds[0] >= nav.index[0], (ds[0], ds[-1], nav.index[0], nav.index[-1])
    v = nav.reindex(nav.index.union(ds)).ffill().reindex(ds).to_numpy(); r = v[1:] / v[:-1] - 1
    assert np.isfinite(r).all(); return r
def mix(rA, rB, w, cost, kindB='stock'):
    """窗頻袖子：每窗末調回比例 w，移轉成本記在該窗；最後一窗之後不調。"""
    if w == 1.0: return rA.copy()
    if w == 0.0: return rB.copy()
    buy, sell = COST[cost]
    sB = sell if kindB == 'stock' else buy + 0.001
    out = np.empty(len(rA)); a, b = w, 1 - w
    for i in range(len(rA)):
        V0 = a + b; a *= 1 + rA[i]; b *= 1 + rB[i]; V1 = a + b
        if i < len(rA) - 1:
            ta = w * V1
            c = (ta - a) * (sB + buy) if a < ta else (a - ta) * (sell + buy)
            V1 -= c; a, b = w * V1, (1 - w) * V1
        out[i] = V1 / V0 - 1
    return out
def cagr(r, d0, d1): return float(np.prod(1 + r) ** (365.25 / (pd.Timestamp(str(d1)) - pd.Timestamp(str(d0))).days) - 1)
def mdd(r): v = np.cumprod(1 + r); return float((v / np.maximum.accumulate(v) - 1).min())
def drop3(x, yrs):
    s = pd.Series(x).groupby(yrs).sum().sort_values(ascending=False); keep = ~np.isin(yrs, s.index[:3]); return tstat(x[keep])
def corr(x, y): return float(np.corrcoef(x, y)[0, 1])

# ---------- 載入 ----------
NJ = pd.read_parquet(V7 / 'navs_jscan_v7.parquet')
NO = pd.read_parquet(V7 / 'navs_v7.parquet')
NX = pd.read_parquet(V7 / 'navs_exec7.parquet')
def npz(tag):
    z = np.load(CM / 'out' / f'{tag}.npz'); D = {}
    for k in ['OFF'] + JK: D[k] = dict(sd=z[f'{k}|sd'], de=str(z[f'{k}|de']), ra=z[f'{k}|ra'], r5=z[f'{k}|r5'])
    return D
CMD = {(c, p): npz(f'run_{c}_{p}') for c in ('a', 'b') for p in ('MAIN', 'PRE')}
CMD[('a', 'MAIN_stress')] = npz('run_a_MAIN_stress')
def mnav(k, cost): return NO[f'AB_V7|{cost}'] if k == 'OFF' else NJ[f'AB_V7|{cost}|{k}']

def series(cfg, per, k, cost='normal'):
    """回傳 dict(sd 起日陣列、終點、rM、rC、r5)；per ∈ PRE／MAIN／ALL／MAIN_2024。"""
    if per == 'ALL':
        A, B = series(cfg, 'PRE', k, cost), series(cfg, 'MAIN', k, cost)
        assert A['days'][-1] <= B['days'][0]
        return dict(days=np.concatenate([A['days'], B['days']]), end=B['end'], rM=np.concatenate([A['rM'], B['rM']]),
                    rC=np.concatenate([A['rC'], B['rC']]), r5=np.concatenate([A['r5'], B['r5']]))
    if per == 'MAIN_2024':
        S = series(cfg, 'MAIN', k, cost); m = S['days'] >= '2024-01-01'
        d0 = S['days'][m]
        return dict(days=d0, end=S['end'], rM=S['rM'][m], rC=S['rC'][m], r5=S['r5'][m])
    src = CMD[(cfg, per if cost == 'normal' else per + '_stress')][k]
    days = list(src['sd']) + [src['de']]
    return dict(days=np.array([str(x) for x in src['sd']]), end=src['de'], rM=wr(mnav(k, cost), days), rC=src['ra'], r5=src['r5'])

VAL = {}
# V2：C 單獨重現 C 的 post
P = json.loads((CM / 'results.json').read_text(encoding='utf-8'))['post']
for cfg in ('a', 'b'):
    for per in ('MAIN', 'PRE'):
        t5 = np.median([tstat(series(cfg, per, k)['rC'] - series(cfg, per, k)['r5']) for k in JK])
        cg = np.median([cagr(series(cfg, per, k)['rC'], series(cfg, per, k)['days'][0], series(cfg, per, k)['end']) for k in JK])
        VAL[f'C_{cfg}_{per}_t5_diff'] = abs(t5 - P[f'{cfg}|{per}']['vs0050_t_jmed'])
        VAL[f'C_{cfg}_{per}_cagr_diff'] = abs(cg - P[f'{cfg}|{per}']['CAGR_jmed'])
    tA = np.median([tstat(series(cfg, 'ALL', k)['rC'] - series(cfg, 'ALL', k)['r5']) for k in JK])
    VAL[f'C_{cfg}_ALL_t5_diff'] = abs(tA - P[f'{cfg}|ALL_2014_2026']['vs0050_t_jmed'])
assert max(VAL.values()) < 1e-9, VAL
# V1：w＝1／0 重現單腿
S = series('a', 'MAIN', 'J3')
VAL['w1_eq_M'] = float(np.abs(mix(S['rM'], S['rC'], 1.0, 'normal') - S['rM']).max())
VAL['w0_eq_C'] = float(np.abs(mix(S['rM'], S['rC'], 0.0, 'normal') - S['rC']).max())
VAL['w_tiny_vs_M'] = float(np.abs(mix(S['rM'], S['rC'], 1 - 1e-12, 'normal') - S['rM']).max())
# V4：M 官方日資料身分
VAL['M_official_AB_V7_full_CAGR_daily'] = float(NO['AB_V7|normal'].iloc[-1] ** (365.25 / (NO.index[-1] - NO.index[0]).days) - 1)
# 訊號日差異（描述）：M 與 C 的訊號日
pl = pd.read_csv(V7 / 'data' / 'plans_j7.csv', usecols=['strat', 'j', 'seed', 'signal'])
pl = pl[(pl.strat == 'AB_V7') & (pl.seed == pl.seed.min())]
VAL['signal_day_mismatch'] = {f'{per}|{k}': sorted(set(map(str, CMD[('a', per)][k]['sd'])) - set(pl[pl.j == int(k[1:])].signal))
                              for per in ('MAIN', 'PRE') for k in JK if set(map(str, CMD[('a', per)][k]['sd'])) - set(pl[pl.j == int(k[1:])].signal)}

COMBOS = [('M', 1.0, 'C'), ('C', 0.0, 'C')] + [(f'MIX{int(w*100)}', w, 'C') for w in WS] + [(f'ME{int(w*100)}', w, 'E') for w in WS]
def stats_one(S, w, legB, cost):
    rB = S['rC'] if legB == 'C' else S['r5']
    r = mix(S['rM'], rB, w, cost, 'stock' if legB == 'C' else 'etf')
    ex5 = r - S['r5']; exM = r - S['rM']; yrs = np.array([d[:4] for d in S['days']])
    return dict(cagr=cagr(r, S['days'][0], S['end']), mdd=mdd(r), m5=float(ex5.mean() * 100), t5=tstat(ex5), mM=float(exM.mean() * 100), tM=tstat(exM),
                t5_drop3y=drop3(ex5, yrs), corr_withM=corr(r, S['rM']), corr_MC=corr(S['rM'], S['rC']), corr_exMC=corr(S['rM'] - S['r5'], S['rC'] - S['r5']),
                cagr5=cagr(S['r5'], S['days'][0], S['end']), mdd5=mdd(S['r5']), n=len(r))
def agg(rows, off):
    J = pd.DataFrame(rows)
    o = dict(CAGR_jmed=J.cagr.median(), CAGR_0050_jmed=J.cagr5.median(), maxDD_jmed=J.mdd.median(), maxDD_0050_jmed=J.mdd5.median(),
             vs0050_mean_pp_jmed=J.m5.median(), vs0050_mean_pp_p10=float(np.percentile(J.m5, 10)), vs0050_t_jmed=J.t5.median(), vs0050_t_p10=float(np.percentile(J.t5, 10)),
             vsM_mean_pp_jmed=J.mM.median(), vsM_mean_pp_p10=float(np.percentile(J.mM, 10)) if J.mM.notna().all() else None,
             vsM_t_jmed=J.tM.median(), vsM_t_p10=float(np.percentile(J.tM, 10)) if J.tM.notna().all() else None,
             vs0050_t_drop_best3y_jmed=J.t5_drop3y.median(), corr_combo_M_jmed=J.corr_withM.median(), corr_M_C_jmed=J.corr_MC.median(),
             corr_exM_exC_jmed=J.corr_exMC.median(), n_windows_jmed=int(J.n.median()))
    if off is not None: o['OFF'] = {k: off[k] for k in ('cagr', 'mdd', 't5', 'm5', 'tM', 'mM')}
    return {k: (float(v) if isinstance(v, (np.floating, float, int)) and v is not None else v) for k, v in o.items()}

PAPER = {}
for cfg in ('a', 'b'):
    for per in ('PRE', 'MAIN', 'ALL', 'MAIN_2024'):
        SS = {k: series(cfg, per, k) for k in ['OFF'] + JK}
        for name, w, legB in COMBOS:
            rows = [stats_one(SS[k], w, legB, 'normal') for k in JK]
            PAPER[f'{cfg}|{per}|normal|{name}'] = agg(rows, stats_one(SS['OFF'], w, legB, 'normal'))
for per in ('MAIN', 'MAIN_2024'):
    SS = {k: series('a', per if per == 'MAIN' else 'MAIN', k, 'stress') for k in ['OFF'] + JK}
    if per == 'MAIN_2024':
        SS = {k: {**{kk: vv[SS[k]['days'] >= '2024-01-01'] for kk, vv in SS[k].items() if kk in ('days', 'rM', 'rC', 'r5')}, 'end': SS[k]['end']} for k in SS}
    for name, w, legB in COMBOS:
        rows = [stats_one(SS[k], w, legB, 'stress') for k in JK]
        PAPER[f'a|{per}|stress|{name}'] = agg(rows, stats_one(SS['OFF'], w, legB, 'stress'))

# ---------- 可執行（MAIN） ----------
EXEC = {}
for cfg in ('a', 'b'):
    z = np.load(HERE / 'out' / f'cm_exec_{cfg}.npz')
    for tn, mode in TIERS:
        for cost, ctag in (('normal', 'K5T'), ('stress', 'K5T+0.5%')):
            for name, w, legB in COMBOS:
                rows = []
                for k in JK:
                    days = list(z[f'{k}|days']); rC = z[f'{k}|{tn}|{mode}|{ctag}']; r5 = z[f'{k}|r5']
                    rM = wr(NX[f'AB|{k}|K5T|{cost}|{tn}|{mode}'], days)
                    SX = dict(days=np.array(days[:-1]), end=days[-1], rM=rM, rC=rC, r5=r5)
                    rows.append(stats_one(SX, w, legB, cost))
                EXEC[f'{cfg}|{tn}|{mode}|{cost}|{name}'] = agg(rows, None)

# ---------- 判定 ----------
def worst(key_fmt, field, lower_is_worse=True):
    v = [PAPER[key_fmt.format(c)][field] for c in ('a', 'b')]; return min(v) if lower_is_worse else max(v)
VERD = {}
Mref = dict(ALL=PAPER['a|ALL|normal|M']['vs0050_t_jmed'], MAIN=PAPER['a|MAIN|normal|M']['vs0050_t_jmed'])
for name, w, legB in COMBOS:
    if not name.startswith('MIX'): continue
    c1 = worst('{}|ALL|normal|' + name, 'vs0050_t_jmed'); c2 = worst('{}|MAIN|normal|' + name, 'vs0050_t_jmed')
    c3 = worst('{}|PRE|normal|' + name, 'vs0050_mean_pp_jmed')
    c4 = min(min(EXEC[f'{c}|{tn}|{mode}|normal|{name}']['vs0050_mean_pp_jmed'], EXEC[f'{c}|{tn}|{mode}|normal|{name}']['vs0050_mean_pp_p10'])
             for c in ('a', 'b') for tn, mode in TIERS)
    c5 = min(EXEC[f'{c}|100萬|{mode}|stress|{name}']['vs0050_mean_pp_jmed'] for c in ('a', 'b') for mode in ('FIX', 'GROW'))
    ok = dict(c1_ALL_t_ge2=c1 >= 2.0, c2_MAIN_t_ge2=c2 >= 2.0, c3_PRE_excess_gt0=c3 > 0, c4_exec_normal_all_jmed_p10_gt0=c4 > 0, c5_exec_stress_100w_gt0=c5 > 0)
    beat = c1 > Mref['ALL'] and c2 > Mref['MAIN']
    v = '不可用' if not ok['c1_ALL_t_ge2'] else ('可用' if all(ok.values()) and beat else '待觀察')
    VERD[name] = dict(values=dict(c1=c1, c2=c2, c3=c3, c4_min=c4, c5_min=c5, M_alone_ALL_t=Mref['ALL'], M_alone_MAIN_t=Mref['MAIN']), conditions=ok, beats_M_alone=beat, verdict=v)

RES['RESULTS'] = dict(validation=VAL, paper=PAPER, exec=EXEC, verdict=VERD)
RJ.write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print(json.dumps(dict(VAL={k: v for k, v in VAL.items()}, VERD=VERD), ensure_ascii=False, indent=1, default=float))
