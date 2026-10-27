#!/usr/bin/env python3
"""主回測：python3 -I -B run.py <P0|P1>（價格修補開關一個行程；兩個行程可平行）。
輸出 run_<P>.json、navs_<P>.parquet。斷言 PREREG 已存在、資料 sha256 相符；P0 的 R0X0Dv2 必須和 main_v2/navs_gapfix.parquet 逐日相同。"""
import sys, json, time, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import o3_core as O
V, K = O.V, O.K
PX = sys.argv[1]; assert PX in ('P0', 'P1')
RES = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert RES.get('PREREG') and RES.get('PREREG_sha256') == hashlib.sha256(RES['PREREG'].encode('utf-8')).hexdigest(), '先執行 prereg.py'
for f in O.freeze()['files']: O.check(f)
assert hashlib.md5((K.P / 'value.parquet').read_bytes()).hexdigest().startswith('8272e253')
T0 = time.time()
def log(*a): print(f'[{PX} {time.time() - T0:5.0f}s]', *a, flush=True)
if PX == 'P1': log('patched cells', O.apply_patch())
E = O.load_events(); H = O.load_disposal_hist()
EXTRA = {pd.Timestamp(k): v for k, v in RES['PREREG_params']['D_hi_extra'].items()}
YL, Y5, _ = V.load_revenue(); YR = O.load_rdel()
GC = O.delist_codes()
FOREIGN = set(json.loads((O.ORS / 'data_freeze.json').read_text(encoding='utf-8'))['foreign_codes'])
COST = {'normal': K.rates(0.001), 'stress': K.rates('stress')}
DMODES = {'v2': ('v2', None), 'lo': ('all', None), 'hi': ('all_plus', EXTRA), 'none': ('none', None)}
XR = {'main': O.exit_rules(E, 'main'), 'upper': O.exit_rules(E, 'upper'), 'lower': O.exit_rules(E, 'lower')}

def r4(o): return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in o.items()}
def segstats(nav, ew, sd, yearly=False):
    return {k: r4(V.seg_stats(nav, ew, a, b, sd, yearly=yearly, ad=ad)) for k, (a, b, ad) in V.SEGS.items()}
def excess_series(nav, sd):
    D = pd.DatetimeIndex([d for d in sd if d in nav.index])
    nd = nav.reindex(D); pr = (nd.shift(-1) / nd - 1).iloc[:-1]
    bd = K.E50.reindex(D); bp = (bd.shift(-1) / bd - 1).iloc[:-1]
    x = (pr - bp).dropna(); return x[(x.index >= '2006-04-01') & (x.index <= '2026-10-12')]
def msd(x): return {'mean_pp': float(x.mean() * 100), 'sd_pp': float(x.std(ddof=1) * 100), 'n': int(len(x)), 't': K.tstat(x)}

out = {'px': PX, 'runs': {}, 'extra': {}}
navs = {}
validated = False
for R in (0, 1):
    Yl = YR if R else YL
    for dkey, (dmode, ext) in DMODES.items():
        O.set_disposal(dmode, H, ext)
        for univ in ('ALL', 'EXFIN'):
            S = O.V3('gapfix', Yl, Y5, exfin=(univ == 'EXFIN'))
            t0, t1, sd = S.bounds()
            pls = {n: S.plan(n, 'official') for n in (10, 30)}
            plus_n = int(sum(len(v) for v in S.plus_drop.values()))
            for X in (0, 1):
                xr = XR['main'] if X else None
                ew, lge = O.simulate([(p[0], p[2]) for p in pls[10]], 0, 0, t0, t1, xrules=xr)
                key0 = f'{PX}|R{R}|X{X}|D{dkey}|{univ}'
                navs[key0 + '|EW'] = ew
                for n in (10, 30):
                    pl = pls[n]
                    for c, (bb, ss) in COST.items():
                        nav, lg = O.simulate([(p[0], p[1]) for p in pl], bb, ss, t0, t1, xrules=xr)
                        key = f'{key0}|n{n}|{c}'
                        if not validated and PX == 'P0' and R == 0 and X == 0 and dkey == 'v2':
                            ref = pd.read_parquet(HERE.parent / 'main_v2' / 'navs_gapfix.parquet')[f'{univ}|n{n}|{c}'].dropna()
                            dd = float((nav.reindex(ref.index) / ref - 1).abs().max())
                            assert dd < 1e-9, (key, dd)
                            refe = pd.read_parquet(HERE.parent / 'main_v2' / 'navs_gapfix.parquet')[f'{univ}|EW'].dropna()
                            assert float((ew.reindex(refe.index) / refe - 1).abs().max()) < 1e-9
                            if univ == 'EXFIN' and n == 30 and c == 'stress': validated = True
                            out['extra'].setdefault('validate_vs_main_v2', {})[key] = dd
                        rec = {'seg': segstats(nav, ew, sd, yearly=(c == 'normal')), 'turnover': lg['turnover'], 'n_frozen': len(lg['frozen']),
                               'max_frozen_share': max([f[2] for f in lg['frozen']], default=0.0), 'n_exits': len(lg['exits']),
                               'exits': lg['exits'][:30], 'ew_n_frozen': len(lge['frozen']), 'ew_max_frozen_share': max([f[2] for f in lge['frozen']], default=0.0),
                               'ew_exits': len(lge['exits']), 'plus_dropped': plus_n, 'msd': msd(excess_series(nav, sd))}
                        if c == 'normal':
                            rec['contrib'] = {k: V.top_contrib(pl, t1, a, b) for k, (a, b, ad) in V.SEGS.items() if k in ('S2', 'S3', 'ALL')}
                            navs[key] = nav
                            if univ == 'EXFIN' and n == 10:
                                rec['picks'] = {str(p[3].date()): [K.COLS[j] for j in p[1]] for p in pl}
                        else:
                            navs[key] = nav
                        out['runs'][key] = rec
                log(key0, {n: out['runs'][f'{key0}|n{n}|normal']['seg']['ALL']['CAGR'] for n in (10, 30)},
                    't', out['runs'][f'{key0}|n10|normal']['seg']['ALL']['t_0050'], 'frz', out['runs'][f'{key0}|n10|normal']['n_frozen'], out['runs'][f'{key0}|n30|normal']['n_frozen'], 'ewfrz', len(lge['frozen']))
            # ---------- 界限與 X 變體：只在 R1、P1、D∈{lo,hi,none} ----------
            if PX == 'P1' and R == 1 and dkey in ('lo', 'hi', 'none'):
                key0 = f'{PX}|R1|D{dkey}|{univ}'
                for xm in ('upper', 'lower'):
                    ew, _ = O.simulate([(p[0], p[2]) for p in pls[10]], 0, 0, t0, t1, xrules=XR[xm])
                    for n in (10, 30):
                        nav, lg = O.simulate([(p[0], p[1]) for p in pls[n]], *COST['normal'], t0, t1, xrules=XR[xm])
                        out['runs'][f'{key0}|X_{xm}|n{n}|normal'] = {'seg': segstats(nav, ew, sd), 'n_frozen': len(lg['frozen']), 'n_exits': len(lg['exits'])}
                # 缺營收界限 B05／B10（含 X main）
                ewm, _ = O.simulate([(p[0], p[2]) for p in pls[10]], 0, 0, t0, t1, xrules=XR['main'])
                navm = {n: O.simulate([(p[0], p[1]) for p in pls[n]], *COST['normal'], t0, t1, xrules=XR['main'])[0] for n in (10, 30)}
                for pf in (0.5, 1.0):
                    SB = O.V3('gapfix', Yl, Y5, exfin=(univ == 'EXFIN'), gapcodes=GC, pfix=pf, foreign=FOREIGN)
                    for n in (10, 30):
                        plb = SB.plan(n, 'official')
                        ewb, _ = O.simulate([(p[0], p[2]) for p in plb], 0, 0, t0, t1, xrules=XR['main'])
                        nav, lg = O.simulate([(p[0], p[1]) for p in plb], *COST['normal'], t0, t1, xrules=XR['main'])
                        nav0, lg0 = O.simulate([(p[0], p[1]) for p in plb], *COST['normal'], t0, t1)
                        out['runs'][f'{key0}|B{int(pf * 10):02d}|n{n}|normal'] = {'seg': segstats(nav, ewb, sd), 'n_frozen': len(lg['frozen']), 'n_exits': len(lg['exits']),
                                                                                  'noX': {'seg_ALL': r4(V.seg_stats(nav0, ewb, *V.SEGS['ALL'][:2], sd, ad=V.SEGS['ALL'][2])), 'n_frozen': len(lg0['frozen']),
                                                                                          'max_frozen_share': max([f[2] for f in lg0['frozen']], default=0.0)},
                                                                                  'gap_used_mean': float(np.mean([len(v) for v in SB.ngap_used.values()])),
                                                                                  'n_gap_picks': int(sum(len(set(p[1]) & set(np.where(SB.cross_section(p[3])['gm'])[0])) for p in plb))}
                        navs[f'{key0}|B{int(pf * 10):02d}|n{n}|normal'] = nav
                    if pf == 1.0: SB10 = SB
                # 期望拖累與最壞情境（解析式，加在 X main 淨值上）
                SB5 = O.V3('gapfix', Yl, Y5, exfin=(univ == 'EXFIN'), gapcodes=GC, pfix=0.5, foreign=FOREIGN)
                lt = O.last_trade_idx(); xrm = XR['main']
                ex = [p[0] for p in pls[10]]
                def per_ret(j, t, t2):
                    g = K.Cn[t, j] / K.On[t, j] if np.isfinite(K.On[t, j]) and K.On[t, j] > 0 else 1.0
                    g *= float(np.prod(K.GROW[t + 1:t2 + 1, j]))
                    if K.TRADED[t2, j] and np.isfinite(K.On[t2, j]): g *= K.On[t2, j] / K.Cn[t2, j]
                    elif j in xrm and lt[j] < t2: g *= xrm[j][0]
                    return g - 1
                for n in (10, 30):
                    nav = navm[n]; adj = nav.copy(); wst = nav.copy()
                    drag_tot = 0.0; worst = {}
                    for i, p in enumerate(pls[10][:-1]):
                        t, t2 = p[0], ex[i + 1]
                        d = p[3]
                        cs5 = SB5.cross_section(d); gm = np.where(cs5['gm'])[0]
                        rew = float(ewm.iloc[t2 - t0] / ewm.iloc[t - t0] - 1)
                        dl = sum(per_ret(j, t, t2) - rew for j in gm) / 300.0
                        adj.iloc[t2 - t0:] *= (1 + dl); drag_tot += dl
                        cs10 = SB10.cross_section(d); g10 = np.where(cs10['gm'])[0]
                        if len(g10):
                            j = g10[np.argmax(cs10['sc'][g10])]
                            y = d.year; sc = float(cs10['sc'][j])
                            if y not in worst or sc > worst[y][0]: worst[y] = (sc, i, j)
                    for y, (sc, i, j) in worst.items():
                        t, t2 = pls[10][i][0], ex[i + 1]
                        rp = float(nav.iloc[t2 - t0] / nav.iloc[t - t0] - 1)
                        dl = (per_ret(j, t, t2) - rp) / n
                        wst.iloc[t2 - t0:] *= (1 + dl)
                    # 價格補不到的 6251（2021-04-30～2022-08-15，換股：報酬 0）：期望拖累 −同池等權/300
                    a6251 = adj.copy()
                    for i, p in enumerate(pls[10][:-1]):
                        if pd.Timestamp('2021-04-30') <= p[3] <= pd.Timestamp('2022-08-15'):
                            t, t2 = p[0], ex[i + 1]
                            rew = float(ewm.iloc[t2 - t0] / ewm.iloc[t - t0] - 1)
                            a6251.iloc[t2 - t0:] *= (1 + (0.0 - rew) / 300.0)
                    out['runs'][f'{key0}|DRAG|n{n}|normal'] = {'seg': segstats(adj, ewm, sd), 'drag_sum': drag_tot}
                    out['runs'][f'{key0}|DRAG6251|n{n}|normal'] = {'seg': segstats(a6251, ewm, sd)}
                    out['runs'][f'{key0}|WORST1|n{n}|normal'] = {'seg': segstats(wst, ewm, sd),
                                                                 'picks': {int(y): [K.COLS[j], str(pls[10][i][3].date())] for y, (sc, i, j) in worst.items()}}
                log(key0, 'bounds done', {k.split('|')[-3]: v['seg']['ALL']['CAGR'] for k, v in out['runs'].items() if k.startswith(key0) and k.endswith('n10|normal') and '|X_' not in k},
                    {k.split('|')[-3]: round(v['seg']['ALL']['CAGR'], 2) for k, v in out['runs'].items() if k.startswith(key0 + '|X_') and k.endswith('n10|normal')})
assert PX == 'P1' or validated
(HERE / f'run_{PX}.json').write_text(json.dumps(out, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, 'item') else str(x)), encoding='utf-8')
pd.DataFrame(navs).to_parquet(HERE / f'navs_{PX}.parquet')
log('done')
