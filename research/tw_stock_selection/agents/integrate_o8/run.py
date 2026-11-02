#!/usr/bin/env python3
"""integrate_o8 回測：照 results.json 的 PREREG，把 o8 唯一通過審查的 o_exec_engine_rewrite_v7x（V7X：V7 可執行引擎獨立重寫＋四項修正；
引擎可用、策略待觀察）和主策略（排除金融 n10，B10 保守口徑）整合。
只讀 import：o_exec_engine_rewrite_v7x/x_data.py、x_stats.py（V7X 統計口徑，已對原表 912 格 ≤1e-13）。
本檔只新寫：袖子混合（與 integrate_o4～o7 同一函式，先重現 integrate_o7 的可執行混合表）、分期統計、月差配對、相關、同帳戶配對、判定。
用法：python3 -I -B run.py（正式，寫回 results.json，只能一次）；python3 -I -B run.py --smoke <輸出檔>（j 只跑 0、6，只測管線，只印驗證差，不印績效）。"""
import sys, json, time, math, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
XD = AG / 'o_exec_engine_rewrite_v7x'
O7 = AG / 'o_v7_listing_exec_engine'
I7 = AG / 'integrate_o7'
DATA = HERE / 'data'
SMOKE = '--smoke' in sys.argv
import numpy as np, pandas as pd
from multiprocessing import get_context


def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()


RJ = HERE / 'results.json'
if not SMOKE:
    RES = json.loads(RJ.read_text(encoding='utf-8'))
    assert hashlib.sha256(json.dumps(RES['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == RES['PREREG_sha256'], 'PREREG 被改過'
    assert 'RESULTS' not in RES, 'RESULTS 已存在：只跑一次'
    for f, h in RES['PREREG']['sha256'].items():
        assert sha(f) == h, f'{f} 與 PREREG 登記的 sha256 不符'
for k, f in {'plans_j7.csv': O7 / 'data/plans_j7.csv', 'jscan_end_days.json': O7 / 'data/jscan_end_days.json',
             'etf_0050.csv': Path('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv')}.items():
    assert sha(DATA / k) == sha(f), k
XR_ = json.loads((XD / 'results.json').read_text(encoding='utf-8'))
assert XR_['PREREG_sha256'] == 'faacf2bde2f13242dec52ca5ba5efa8e1eb0c000939cf584f27334b59f94a956'
I7R = json.loads((I7 / 'results.json').read_text(encoding='utf-8'))

sys.path.insert(0, str(XD))
import x_data as D, x_stats as S
sys.path.pop(0)
T0_ = time.time()


def log(*a):
    if SMOKE: print(f'[{time.time() - T0_:6.0f}s]', *[x for x in a if isinstance(x, str)], flush=True)
    else: print(f'[{time.time() - T0_:6.0f}s]', *a, flush=True)


VAL = {}
E50 = pd.Series(D.E50, index=D.IDX)
E50raw = pd.read_csv(DATA / 'etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
e50c = E50raw.reindex(D.IDX.union(E50raw.index)).ffill().reindex(D.IDX)
VAL['E50_cache_vs_csv_maxrel'] = float((E50 / e50c - 1).abs().max()); assert VAL['E50_cache_vs_csv_maxrel'] < 1e-12

FEE = 0.001425 * 0.6
COST = {'normal': (FEE + 0.001, FEE + 0.003 + 0.001), 'stress': (0.005, 0.005)}   # = v4lib.cost_rates（K.rates）
PJ = pd.read_csv(DATA / 'plans_j7.csv', usecols=['strat', 'j', 'seed', 'signal', 'exec'])
PJ = PJ[(PJ.seed == 0) & (PJ.strat == 'AB_V7_B10')]
EXECD = {int(j): pd.DatetimeIndex(pd.to_datetime(g['exec'])) for j, g in PJ.groupby('j')}
for j, g in PJ.groupby('j'): assert pd.DatetimeIndex(pd.to_datetime(g.signal)).equals(S.SIGD[('AB_V7_B10', int(j))])

JS = [0, 6] if SMOKE else list(range(13))
TIERS = ['30萬', '100萬', '300萬', '1000萬']; VERS = ['FIX', 'GROW']; MODELS = ['K5T', 'TICK']; COSTS = ['normal', 'stress']
WS = [1.0, 0.8, 0.7, 0.5, 0.3]
WNAME = {1.0: 'ALONE', 0.8: 'M_E50_20', 0.7: 'M_E50_30', 0.5: 'M_E50_50', 0.3: 'C_E50_70'}
RUNS = ['R4'] if SMOKE else ['R4', 'R4c', 'R0']
PER = {'ALL': (None, None), 'SEL': (None, '2018-12-31'), 'VAL': ('2019-05-01', '2023-12-31'), 'HOLD': ('2024-01-01', None),
       'Y19on': ('2019-01-01', None), 'P19_22': ('2019-01-01', '2022-12-31'), 'P23_26': ('2023-01-01', None)}
CAL_A, CAL_B = pd.Period('2006-05', 'M'), pd.Period('2026-09', 'M')


def sleeve(navA, rB, w, cost, kindB, ix, exi):
    """與 integrate_o4～o7 同一函式。navA：主策略腿淨值；rB：0050 日報酬；w：主策略腿比例；主策略成交日收盤調回比例，移轉扣成本。"""
    buy, sell = cost
    bB, sB = {'etf': (buy, buy + 0.001), 'stock': (buy, sell)}[kindB]
    ra = navA.reindex(ix).pct_change().fillna(0).to_numpy()
    a, b = w, 1 - w; out = np.empty(len(ix))
    for i in range(len(ix)):
        if i > 0: a *= 1 + ra[i]; b *= 1 + rB[i]
        if i in exi and i > 0:
            tot = a + b; ta = w * tot
            if a < ta: m = ta - a; c = m * (sB + buy)
            else: m = a - ta; c = m * (sell + bB)
            tot -= c; a, b = w * tot, (1 - w) * tot
        out[i] = a + b
    return pd.Series(out, index=ix)


def mixnav(nav_full, j, w, cost):
    e = S.end_day('AB_V7_B10', j); nav = nav_full.loc[:e].dropna()
    if w == 1.0: return nav
    ix = nav.index; exi = set(ix.get_loc(d) for d in EXECD[j] if d in ix)
    rE = E50.reindex(ix).pct_change().fillna(0).to_numpy()
    return sleeve(nav, rE, w, COST[cost], 'etf', ix, exi)


def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return float('nan')
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / math.sqrt(len(x)))) if sd > 0 else float('nan')


def bounds(nav, key):
    a, b = PER[key]; ix = nav.index
    ia = ix[0] if a is None else ix[ix >= pd.Timestamp(a)][0]
    ib = ix[-1] if b is None else ix[ix <= pd.Timestamp(b)][-1]
    return ia, ib


def mdd(nav, a, b):
    s = nav.loc[a:b]; return float((s / s.cummax() - 1).min() * 100)


def mret(nav):
    me = nav.groupby(nav.index.to_period('M')).last().pct_change()
    return me[(me.index >= CAL_A) & (me.index <= CAL_B)].dropna()


E50M = mret(E50)


def msel(r, key):
    a, b = PER[key]
    pa = CAL_A if a is None else max(pd.Period(a, 'M'), CAL_A); pb = CAL_B if b is None else min(pd.Period(b, 'M'), CAL_B)
    return r[(r.index >= pa) & (r.index <= pb)]


def nav_stats(nn, j):
    """單一淨值、單一 j 的全部分期統計（V7X 口徑：S.stats；另加 P19_22／P23_26、分期回撤、日曆月 t、追蹤誤差）。"""
    s = S.stats(nn, 'AB_V7_B10', j)
    x = S.windows(nn.loc[:S.end_day('AB_V7_B10', j)].dropna(), 'AB_V7_B10', j)
    o = {k: s[k] for k in ('CAGR', 'E50_CAGR', 'exc', 't_0050', 'maxDD', 'CAGR_SEL', 'E50_CAGR_SEL', 'exc_SEL', 't_SEL', 'CAGR_VAL', 'E50_CAGR_VAL', 'exc_VAL', 't_VAL',
                           'CAGR_HOLD', 'E50_CAGR_HOLD', 'exc_HOLD', 't_HOLD', 'CAGR_Y19on', 'E50_CAGR_Y19on', 'exc_Y19on', 't_Y19on')}
    for key in ('P19_22', 'P23_26'):
        a, b = bounds(nn, key)
        o['CAGR_' + key] = S.cagr(nn, a, b); o['E50_CAGR_' + key] = S.cagr(E50, a, b); o['exc_' + key] = o['CAGR_' + key] - o['E50_CAGR_' + key]
        pa, pb = PER[key]; o['t_' + key] = tstat(x[(x.index >= pd.Timestamp(pa)) & (x.index <= pd.Timestamp(pb or '2100-01-01'))])
    for key in ('SEL', 'VAL', 'HOLD', 'Y19on', 'P19_22', 'P23_26'):
        a, b = bounds(nn, key); o['maxDD_' + key] = mdd(nn, a, b); o['E50_maxDD_' + key] = mdd(E50, a, b)
    a, b = bounds(nn, 'ALL'); o['E50_maxDD'] = mdd(E50, a, b)
    mr = mret(nn); ex = mr - E50M.reindex(mr.index)
    o['t_cal'] = tstat(ex); o['TE'] = float(ex.std(ddof=1) * math.sqrt(12) * 100)
    o['corr_0050_m'] = float(mr.corr(E50M.reindex(mr.index)))
    return o, mr


def job(arg):
    run, j = arg
    df = pd.read_parquet(XD / 'out' / f'navs_oos_{run}.parquet') if run != 'V7ORIG' else pd.read_parquet(O7 / 'navs_exec7.parquet')
    out = {}; mons = {}
    for model in MODELS:
        for cost in COSTS:
            for tier in TIERS:
                for ver in VERS:
                    col = f'B10|J{j}|{model}|{cost}|{tier}|{ver}'
                    for w in WS:
                        nn = mixnav(df[col], j, w, cost)
                        o, mr = nav_stats(nn, j)
                        out[(run, j, model, cost, tier, ver, w)] = o; mons[(run, j, model, cost, tier, ver, w)] = mr
    # 配對：同腿級距（混合 − 單獨）與同帳戶（混合 w0.3 小級距 − 單獨大級距）
    pair = {}
    def pstats(d):
        o = {}
        for key in PER:
            z = msel(d, key).dropna(); o['mean_pp_' + key] = float(z.mean() * 100); o['t_' + key] = tstat(z); o['n_' + key] = int(len(z))
        z = d.dropna(); o['MDE80_pp_ALL'] = float(2.8 * z.std(ddof=1) / math.sqrt(len(z)) * 100)
        return o
    def corr(a, b, key='ALL', excess=False):
        a, b = msel(a, key), msel(b, key)
        if excess: e = E50M; a, b = a - e.reindex(a.index), b - e.reindex(b.index)
        z = pd.concat([a, b], axis=1).dropna(); return float(z.iloc[:, 0].corr(z.iloc[:, 1]))
    for model in MODELS:
        for cost in COSTS:
            for tier in TIERS:
                for ver in VERS:
                    base = mons[(run, j, model, cost, tier, ver, 1.0)]
                    for w in WS[1:]:
                        mm = mons[(run, j, model, cost, tier, ver, w)]
                        o = pstats(mm - base.reindex(mm.index))
                        o.update(corr_ret_ALL=corr(mm, base), corr_exc_ALL=corr(mm, base, excess=True), corr_ret_Y19on=corr(mm, base, 'Y19on'),
                                 corr_exc_Y19on=corr(mm, base, 'Y19on', excess=True))
                        pair[('same', run, j, model, cost, tier, ver, w)] = o
                        pair[('diffser', run, j, model, cost, tier, ver, w)] = (mm - base.reindex(mm.index))
                for acct, (ta, tb) in MATCHED.items():
                    for ver in VERS:
                        al = mons[(run, j, model, cost, ta, ver, 1.0)]; mx = mons[(run, j, model, cost, tb, ver, 0.3)]
                        d = mx - al.reindex(mx.index)
                        o = pstats(d); o.update(corr_ret_ALL=corr(mx, al), corr_exc_ALL=corr(mx, al, excess=True))
                        pair[('acct', run, j, model, cost, acct, ver)] = o; pair[('acctser', run, j, model, cost, acct, ver)] = d
    return out, pair


MATCHED = {'帳戶100萬': ('100萬', '30萬'), '帳戶1000萬': ('1000萬', '300萬')}


def q(v, p): return float(np.nanpercentile(np.asarray(v, float), p))


if __name__ == '__main__':
    jobs = [(r, j) for r in RUNS + ['V7ORIG'] for j in JS]
    with get_context('fork').Pool(4) as P:
        parts = P.map(job, jobs, chunksize=1)
    ST = {}; PR = {}
    for o, p in parts: ST.update(o); PR.update(p)
    log('per-j done')

    # ---------- 驗證 ----------
    # (V1) 本檔袖子＋統計在 V7 原表（navs_exec7）上重現 integrate_o7 可執行混合表（B10，全部 w、模型、成本、級距、版本）
    LIVE7 = I7R['RESULTS']['live_exec_j']; mx = 0.0
    for model in MODELS:
        for cost in COSTS:
            for tier in TIERS:
                for ver in VERS:
                    for w in WS:
                        rows = [ST[('V7ORIG', j, model, cost, tier, ver, w)] for j in JS]
                        r7 = LIVE7[f'B10|{model}|{cost}|{tier}|{ver}|w{w}']
                        for mine, theirs in (('CAGR', 'CAGR_median'), ('t_0050', 't_median'), ('maxDD', 'maxDD_median'), ('E50_CAGR', 'E50_CAGR_median'),
                                             ('t_Y19on', 't_Y19on_median'), ('CAGR_Y19on', 'CAGR_Y19on_median'), ('CAGR_HOLD', 'CAGR_HOLD_median')):
                            if SMOKE and theirs not in ('CAGR_median',): continue
                            mx = max(mx, abs(q([r[mine] for r in rows], 50) - r7[theirs])) if not SMOKE else mx
    VAL['V1_sleeve_on_V7orig_vs_integrate_o7_live_maxabs'] = mx
    if not SMOKE: assert mx < 1e-9, mx
    # (V2) w＝1 在 R4 上重現 V7X STEP_TABLES['R4']（32 格 j 中位數）
    STR4 = XR_['RESULTS_OOS']['STEP_TABLES']['R4']; mx = 0.0
    for k, ref in STR4.items():
        model, cost, tier, ver = k.split('|')
        rows = [ST[('R4', j, model, cost, tier, ver, 1.0)] for j in JS]
        for kk in ('CAGR', 't_0050', 'maxDD', 'exc_Y19on', 'CAGR_SEL', 'exc_VAL', 't_VAL', 'CAGR_HOLD', 'E50_CAGR'):
            if not SMOKE: mx = max(mx, abs(q([r[kk] for r in rows], 50) - ref[kk]))
    VAL['V2_w1_R4_vs_V7X_STEP_TABLES_maxabs'] = mx
    if not SMOKE: assert mx < 1e-9, mx
    # (V3) R4c 正式表亦重現
    if not SMOKE:
        F4c = XR_['RESULTS_OOS']['FORMAL_V7Xc_posthoc']; mx = 0.0
        for k, ref in F4c.items():
            tier, ver = k.split('|'); rows = [ST[('R4c', j, 'K5T', 'normal', tier, ver, 1.0)] for j in JS]
            mx = max(mx, abs(q([r['CAGR'] for r in rows], 50) - ref['CAGR_K5T']), abs(q([r['t_0050'] for r in rows], 50) - ref['t_med']))
        VAL['V3_w1_R4c_vs_V7X_FORMAL_posthoc_maxabs'] = mx; assert mx < 1e-9, mx
    log('validation', VAL)
    if SMOKE:
        Path(sys.argv[-1]).write_text(json.dumps({'VAL': VAL, 'n_ST': len(ST), 'n_PR': len(PR)}, ensure_ascii=False))
        print('smoke ok', len(ST), len(PR)); sys.exit(0)

    # ---------- j 彙整 ----------
    KEYS = list(next(iter(ST.values())).keys())
    TABLE = {}
    for run in RUNS + ['V7ORIG']:
        for model in MODELS:
            for cost in COSTS:
                for tier in TIERS:
                    for ver in VERS:
                        for w in WS:
                            rows = [ST[(run, j, model, cost, tier, ver, w)] for j in JS]
                            o = {k + '_med': q([r[k] for r in rows], 50) for k in KEYS}
                            o.update({k + '_p10': q([r[k] for r in rows], 10) for k in ('CAGR', 'exc', 'exc_Y19on', 'maxDD')})
                            o['n_j_exc_pos'] = int(sum(r['exc'] > 0 for r in rows)); o['n_j_exc_Y19on_pos'] = int(sum(r['exc_Y19on'] > 0 for r in rows))
                            o['account_capital_wan'] = float(int(tier.replace('萬', '')) / w)
                            if w != 1.0:
                                pr = [PR[('same', run, j, model, cost, tier, ver, w)] for j in JS]
                                for kk in pr[0]: o['vsAlone_' + kk + '_med'] = q([p[kk] for p in pr], 50)
                                avg = pd.concat([PR[('diffser', run, j, model, cost, tier, ver, w)] for j in JS], axis=1).mean(axis=1)
                                for key in PER:
                                    z = msel(avg, key).dropna(); o[f'vsAlone_jpool_t_{key}'] = tstat(z); o[f'vsAlone_jpool_mean_pp_{key}'] = float(z.mean() * 100)
                            TABLE[f'{run}|{model}|{cost}|{tier}|{ver}|{WNAME[w]}'] = o
    log('table done')

    # ---------- 同帳戶配對（事前固定兩組）----------
    MATCH = {}
    for run in RUNS + ['V7ORIG']:
        for cost in COSTS:
            for acct, (ta, tb) in MATCHED.items():
                for ver in VERS:
                    row = {}
                    for model in MODELS:
                        A = [ST[(run, j, model, cost, ta, ver, 1.0)] for j in JS]; B = [ST[(run, j, model, cost, tb, ver, 0.3)] for j in JS]
                        r = {}
                        for key in ('', '_SEL', '_VAL', '_HOLD', '_Y19on', '_P19_22', '_P23_26'):
                            dd = np.array([b['CAGR' + key] - a['CAGR' + key] for a, b in zip(A, B)])
                            r['diff_CAGR' + key + '_med_pp'] = float(np.median(dd)); r['n_j_mix_higher' + key] = int((dd > 0).sum())
                        r.update(alone_CAGR_med=q([a['CAGR'] for a in A], 50), mix30_CAGR_med=q([b['CAGR'] for b in B], 50),
                                 alone_maxDD_med=q([a['maxDD'] for a in A], 50), mix30_maxDD_med=q([b['maxDD'] for b in B], 50),
                                 alone_t0050_med=q([a['t_0050'] for a in A], 50), mix30_t0050_med=q([b['t_0050'] for b in B], 50),
                                 alone_CAGR_Y19on_med=q([a['CAGR_Y19on'] for a in A], 50), mix30_CAGR_Y19on_med=q([b['CAGR_Y19on'] for b in B], 50),
                                 E50_CAGR_med=q([a['E50_CAGR'] for a in A], 50), E50_CAGR_Y19on_med=q([a['E50_CAGR_Y19on'] for a in A], 50))
                        pr = [PR[('acct', run, j, model, cost, acct, ver)] for j in JS]
                        for kk in pr[0]: r['paired_' + kk + '_med'] = q([p[kk] for p in pr], 50)
                        avg = pd.concat([PR[('acctser', run, j, model, cost, acct, ver)] for j in JS], axis=1).mean(axis=1)
                        for key in PER: r[f'paired_jpool_t_{key}'] = tstat(msel(avg, key).dropna())
                        row[model] = r
                    su, sl_ = row['K5T']['diff_CAGR_med_pp'] > 0, row['TICK']['diff_CAGR_med_pp'] > 0
                    row['conclusion'] = '主策略 30% 較高' if (su and sl_) else ('全押較高' if (not su and not sl_) else '無法判斷')
                    sig = all(row[m]['paired_t_ALL_med'] >= 2.0 for m in MODELS) or all(row[m]['paired_t_ALL_med'] <= -2.0 for m in MODELS)
                    row['conclusion_strength'] = '兩模型月差配對 t 同號且 |t| ≥ 2' if sig else '月差配對 t 未達 |2|（只是點估計）'
                    MATCH[f'{run}|{cost}|{acct}|{ver}'] = row
    log('matched done')

    # ---------- 判定（R4、B10、K5T、一般成本，8 格；j 中位數）----------
    def verdict50(o):
        if o['exc_med'] > 0 and o['t_0050_med'] >= 1.5 and o['exc_Y19on_med'] >= 0: return '贏'
        if o['exc_med'] <= 0: return '輸'
        return '無可辨識差異'
    BEAT = {k: verdict50(v) for k, v in TABLE.items()}
    RANK = {'不可用': 0, '待觀察': 1, '可用': 2}
    JUDGE = {}
    for w in WS[1:]:
        nm = WNAME[w]; cells = {}
        for tier in TIERS:
            for ver in VERS:
                A = TABLE[f'R4|K5T|normal|{tier}|{ver}|ALONE']; M = TABLE[f'R4|K5T|normal|{tier}|{ver}|{nm}']
                pa = M['vsAlone_t_ALL_med']; ms, mv = M['vsAlone_mean_pp_SEL_med'], M['vsAlone_mean_pp_VAL_med']
                cA = pa >= 1.0 and ms > 0 and mv > 0
                cB = (M['t_0050_med'] - A['t_0050_med'] >= 0.2 and M['t_SEL_med'] > A['t_SEL_med'] and M['t_VAL_med'] > A['t_VAL_med']
                      and A['CAGR_med'] - M['CAGR_med'] <= 1.0 and M['maxDD_med'] >= A['maxDD_med'])
                cW = pa > -1.0 and A['CAGR_med'] - M['CAGR_med'] <= 2.0 and (M['t_SEL_med'] > A['t_SEL_med'] or M['t_VAL_med'] > A['t_VAL_med'])
                v = '可用' if (cA or cB) else ('待觀察' if cW else '不可用')
                cells[f'{tier}|{ver}'] = dict(verdict=v, path_A=bool(cA), path_B=bool(cB), watch=bool(cW), paired_t_ALL=pa, paired_mean_SEL=ms, paired_mean_VAL=mv,
                                             dCAGR=M['CAGR_med'] - A['CAGR_med'], dt0050=M['t_0050_med'] - A['t_0050_med'], dmaxDD=M['maxDD_med'] - A['maxDD_med'])
        overall = '不可用'
        for v in ('可用', '待觀察'):
            if sum(RANK[c['verdict']] >= RANK[v] for c in cells.values()) >= 5: overall = v; break
        raw = overall
        if overall == '可用': overall = '待觀察'
        JUDGE[nm] = dict(cells=cells, raw=raw, final=overall)
    # 選版（只用 SEL）：每格在 4 個混合中挑 SEL t 高於單獨且 SEL 年化少 ≤1pp 者中 t 最高
    SELECT = {}
    for tier in TIERS:
        for ver in VERS:
            A = TABLE[f'R4|K5T|normal|{tier}|{ver}|ALONE']; el = []
            for w in WS[1:]:
                M = TABLE[f'R4|K5T|normal|{tier}|{ver}|{WNAME[w]}']
                ok = M['t_SEL_med'] > A['t_SEL_med'] and A['CAGR_SEL_med'] - M['CAGR_SEL_med'] <= 1.0
                el.append(dict(name=WNAME[w], SEL_t=M['t_SEL_med'], SEL_t_alone=A['t_SEL_med'], SEL_CAGR_loss_pp=A['CAGR_SEL_med'] - M['CAGR_SEL_med'], eligible=bool(ok)))
            ok = [e for e in el if e['eligible']]
            SELECT[f'{tier}|{ver}'] = dict(candidates=el, selected=max(ok, key=lambda e: e['SEL_t'])['name'] if ok else 'ALONE（主策略單獨）')
    # R_V7X：SOP 實盤預期表換成 V7X 正式表
    RECON = XR_['RESULTS_OOS']['RECON']['cells']
    sel_recon = []
    for model in MODELS:
        for cost in ('normal',):
            for tier in TIERS:
                for ver in VERS:
                    a = np.median([ST[('R0', j, model, cost, tier, ver, 1.0)]['CAGR_SEL'] for j in JS]); b = np.median([ST[('V7ORIG', j, model, cost, tier, ver, 1.0)]['CAGR_SEL'] for j in JS])
                    sel_recon.append(abs(a - b))
    gate = json.loads((XD / 'out' / 'gate_zero.json').read_text())
    def _mx(d):
        m = 0.0
        for v in d.values():
            m = max(m, _mx(v)) if isinstance(v, dict) else (max(m, abs(v)) if isinstance(v, (int, float)) else m)
        return m
    cond = {'my_w1_reproduces_V7X_R4_table_le_1e-9': VAL['V2_w1_R4_vs_V7X_STEP_TABLES_maxabs'] < 1e-9,
            'V7X_zero_impact_gate_le_1e-9': _mx(gate) <= 1e-9, 'V7X_zero_impact_gate_max': _mx(gate),
            'SEL_recon_R0_vs_V7orig_normal16_max_le_0p10pp': float(max(sel_recon)) <= 0.10, 'SEL_recon_max_pp': float(max(sel_recon)),
            'V7X_engine_verdict_usable': XR_['VERDICT']['engine'].startswith('可用')}
    replace = all(v for k, v in cond.items() if isinstance(v, bool))
    SOPT = {}
    for tier in TIERS:
        for ver in VERS:
            k = f'K5T|normal|{tier}|{ver}'; kt = f'TICK|normal|{tier}|{ver}'
            A = TABLE[f'R4|K5T|normal|{tier}|{ver}|ALONE']; At = TABLE[f'R4|TICK|normal|{tier}|{ver}|ALONE']
            adj = max(0.0, RECON[k]['diff_pp']); adjt = max(0.0, RECON[kt]['diff_pp'])
            SOPT[f'{tier}|{ver}'] = dict(CAGR_K5T=A['CAGR_med'], CAGR_TICK=At['CAGR_med'], CAGR_K5T_unfav=A['CAGR_med'] - adj, CAGR_TICK_unfav=At['CAGR_med'] - adjt,
                                         unfav_adj_pp_K5T=adj, unfav_adj_pp_TICK=adjt, t_K5T=A['t_0050_med'], t_TICK=At['t_0050_med'],
                                         CAGR_Y19on_K5T=A['CAGR_Y19on_med'], CAGR_Y19on_TICK=At['CAGR_Y19on_med'], E50_Y19on=A['E50_CAGR_Y19on_med'],
                                         maxDD_K5T=A['maxDD_med'], verdict_K5T=BEAT[f'R4|K5T|normal|{tier}|{ver}|ALONE'], verdict_TICK=BEAT[f'R4|TICK|normal|{tier}|{ver}|ALONE'],
                                         stress_CAGR_K5T=TABLE[f'R4|K5T|stress|{tier}|{ver}|ALONE']['CAGR_med'],
                                         V7orig_CAGR_K5T=TABLE[f'V7ORIG|K5T|normal|{tier}|{ver}|ALONE']['CAGR_med'], V7orig_CAGR_TICK=TABLE[f'V7ORIG|TICK|normal|{tier}|{ver}|ALONE']['CAGR_med'])
    RV = dict(conditions=cond, decision='取代：SOP 實盤預期表換成 V7X 正式表（R4，K5T 主欄、TICK 敏感度並列；不利值欄並列）' if replace else '不取代', table=SOPT)
    # 零衝擊參考（integrate_o7，B10 一般成本，不重算）
    REF = {nm: {p: {k: I7R['RESULTS']['stats'][f'{nm}|B10']['periods'][p].get(k) for k in ('CAGR', 'maxDD', 't_0050', 'TE')}
                for p in ('ALL', 'SEL', 'VAL', 'HOLD', 'Y19on', 'P19_22', 'P23_26') if p in I7R['RESULTS']['stats'][f'{nm}|B10']['periods']}
           for nm in ('AB_V7', 'M_E50_20', 'M_E50_30', 'M_E50_50', 'C_E50_70')}

    def js(o):
        if isinstance(o, dict): return {str(k): js(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [js(v) for v in o]
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, float) and not math.isfinite(o): return None
        return o
    RES['RESULTS'] = js({'validation': VAL, 'table': TABLE, 'beat_0050_verdict': BEAT, 'matched_accounts': MATCH, 'judgement': JUDGE, 'selection_SEL_only': SELECT,
                         'R_V7X': RV, 'ref_zero_impact_integrate_o7': REF, 'run_seconds': time.time() - T0_})
    RES['RESULTS_written_at'] = pd.Timestamp.now(tz='UTC').isoformat()
    RJ.write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding='utf-8')
    log('written', {k: v['final'] for k, v in JUDGE.items()}, RV['decision'])
