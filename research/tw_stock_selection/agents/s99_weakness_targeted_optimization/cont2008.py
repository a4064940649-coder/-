#!/usr/bin/env python3
"""S99 加做 (a)：凍結格放進 2008 起連續引擎（S75 v3 口徑：7% 漲跌幅＋2008-10 跌幅 3.5%、0050 銜接檔、REDUCE_FIX），回答「有沒有改善 2008～2015 年中的盤整期」。
做法照 agents/s75_continuous_2008_2026/v3_2026-11-25_limit7/run_v3.py（只讀 import s75lib；漲跌停價只在記憶體覆寫；不改任何既有檔）。快取先由該資料夾的 rebuild_cache.py 重建。
先原樣重現 v3：V_BRIDGE／V_MASK 標準路徑淨值與 nav_2008_2026_v3.csv.gz 逐日差 0、交易 627 筆（V_BRIDGE），不符就中止。
再跑：現行 vs 凍結格 × 2 版本 × 20 條路徑（一次一格）；隨機對照（V_BRIDGE 標準路徑、同月中列同檔數、MA_MOM 池、20 次、seed 99）。
輸出 results_2008.json、nav_2008_frozen_vs_current.csv.gz、yearly_2008.csv。已有 results_2008.json 就拒絕重跑。用法：python3 -I -B cont2008.py [--dry]（--dry：凍結格＝現行，只驗基準）"""
import sys, json, time, resource, gc, math, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
S75 = AG / 's75_continuous_2008_2026'; V3 = S75 / 'v3_2026-11-25_limit7'; V2 = S75 / 'v2_2026-11-21'
DRY = '--dry' in sys.argv
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
if not DRY:
    assert not (HERE / 'results_2008.json').exists(), 'results_2008.json 已存在'
    PSHA = sha(HERE / 'PREREG.json'); assert PSHA == (HERE / 'PREREG.sha256').read_text().split()[0]
    for ln in (HERE / 'code.sha256').read_text().splitlines():
        h, f = ln.split()
        assert sha(HERE / f) == h, f'{f} 與登記時不同'
    assert sha(HERE / 'FROZEN.json') == (HERE / 'FROZEN.sha256').read_text().split()[0]
    FZID = json.loads((HERE / 'FROZEN.json').read_text())['frozen_for_validation']
else:
    FZID = 'CUR'
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(S75))
import numpy as np, pandas as pd
import s99cells as CE
import s75lib as K
from s75lib import R, cslib

t0 = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
assert cslib.REDUCE_FIX is False and cslib.E50_SOURCE is None
old = pd.read_csv(cslib.E50_CSV, parse_dates=['date']).set_index('date')['adj'].sort_index()
new = pd.read_csv(cslib.E50_OFFICIAL_CSV, parse_dates=['date']).set_index('date')['adj'].sort_index()
seam = new.index[0]; assert str(seam.date()) == '2015-01-05' and seam in old.index
sp = pd.concat([old.loc[:seam].iloc[:-1] * float(new.loc[seam] / old.loc[seam]), new])
ex = pd.read_csv(V2 / 'e50_spliced.csv', parse_dates=['date']).set_index('date')['adj']
assert sp.index.equals(ex.index) and float((sp - ex).abs().max()) < 1e-9, '0050 銜接檔與 v2 不同'
cslib.E50_SOURCE = V2 / 'e50_spliced.csv'; cslib.REDUCE_FIX = True
P, U = K.load_cont()
assert P.reduce_fix and P.exright_fix
D0, DE = P.IDX[P.t0], P.IDX[-1]
assert (str(D0.date()), str(DE.date())) == ('2008-01-02', '2026-10-22')
D19 = K.last_before(P.IDX, '2019-01-01'); DH1 = K.last_before(P.IDX, '2015-07-01'); D16 = K.last_before(P.IDX, '2016-01-01')
bench = pd.Series(P.E50, index=P.IDX).loc[D0:]; bench = bench / bench.iloc[0] * 1e6


def limits(ref, up_pct, dn_pct):                              # 逐行同 run_v3.py
    u = ref * (1 + up_pct); t = cslib.tick(u); u = np.floor(u / t + 1e-9) * t
    d = ref * (1 - dn_pct); t = cslib.tick(d); d = np.ceil(d / t - 1e-9) * t
    return u, d


iv = P.IDX.values
cut = int(np.searchsorted(iv, np.datetime64('2015-06-01'), side='left')); assert str(P.IDX[cut].date()) == '2015-06-01'
h0 = int(np.searchsorted(iv, np.datetime64('2008-10-13'), side='left')); h1 = int(np.searchsorted(iv, np.datetime64('2008-10-24'), side='right'))
u, d = limits(P.REF, 0.10, 0.10)
assert np.array_equal(u, P.LIMUP, equal_nan=True) and np.array_equal(d, P.LIMDN, equal_nan=True)
u7, d7 = limits(P.REF[:cut], 0.07, 0.07)
LU = P.LIMUP.copy(); LU[:cut] = u7; LD = P.LIMDN.copy(); LD[:cut] = d7
_, d35 = limits(P.REF[h0:h1], 0.07, 0.035); LD[h0:h1] = d35
P.LIMUP, P.LIMDN = LU, LD
del u, d, u7, d7, d35; gc.collect()

ref_nav = pd.read_csv(V3 / 'nav_2008_2026_v3.csv.gz', index_col=0, parse_dates=True)
ref_res = json.loads((V3 / 'results_v3.json').read_text())
STD = '100萬|+0'
WINS = dict(full=(D0, DE), y2008_2015H1=(D0, DH1), y2015H2_2018=(DH1, D19), since2019=(D19, DE))
LOSE_YEARS = (2011, 2012, 2014, 2016, 2019)


def top_share(lg):
    r = lg['ret'].to_numpy(float); pnl = lg['pnl'].to_numpy(float); n = len(r); m = int(math.ceil(n * 0.05)); o = np.argsort(-r, kind='stable')[:m]
    return dict(n_all=int(n), n_top=m, share_of_trade_pnl=float(pnl[o].sum() / pnl.sum()), rest_mean_ret=float(np.delete(r, o).mean()), rest_pnl_sum=float(np.delete(pnl, o).sum()), mean_ret=float(r.mean()),
                median_ret=float(np.median(r)), win=float((r > 0).mean()))


def metrics(nav, lg, st, mid_rows):
    o = {k: K.cagr(nav, a, b) for k, (a, b) in WINS.items()}
    o.update(maxdd=R.maxdd(nav), maxdd_2008_2015H1=R.maxdd(nav.loc[D0:DH1]), n_all=int(len(lg)), n_mid=int(lg['sig_t'].isin(list(mid_rows)).sum()),
             stockfrac_2008_2015H1=float(np.nanmean(st['stockfrac'].loc[D0:DH1].to_numpy())), stockfrac_full=float(np.nanmean(st['stockfrac'].to_numpy())))
    return o


res = dict(frozen=FZID, engine='S75 v3（7% 漲跌幅＋2008-10 跌幅 3.5%）', versions={}, e0050={k: K.cagr(bench, a, b) for k, (a, b) in WINS.items()})
NAVOUT = {}; YR = []
for v, rule in K.VERSIONS.items():
    F, RT = K.surge(P, rule)
    PATH = {'CUR': {}, 'FZ': {}}; std = {}
    for s in K.SHIFTS:
        sg = K.signal(P, U, RT, s)
        for nm, cell in (('CUR', 'CUR'), ('FZ', FZID)):
            S2, rec = CE.apply_cell(P, U, RT, sg, cell, cslib.ins15_mask_row)
            for cap in K.CAPS:
                nav, lg, st = K.sim(P, S2, U, cap)
                key = f'{int(cap / 1e4)}萬|+{s}'
                PATH[nm][key] = metrics(nav, lg, st, S2['mid_rows'])
                if key == STD:
                    std[nm] = (nav, lg, st, S2['mid_rows'])
                    if nm == 'FZ':
                        std['counts'] = CE.mid_buy_counts(lg, S2['mid_rows']); std['sg'] = sg
                del nav, lg, st
            del S2, rec; gc.collect()
        if s != 0:
            del sg
        gc.collect()
    # ---- 基準重現（現行標準路徑對 v3 既有檔）
    navC, lgC, stC, _ = std['CUR']
    base = dict(nav_max_abs_diff=float((navC.round(2) - ref_nav[v]).abs().max()), n_all=int(len(lgC)), n_all_ref=int(ref_res['versions'][v]['n_all']) if 'versions' in ref_res else None,
                cagr=K.cagr(navC, D0, DE), cagr_2008_2015H1=K.cagr(navC, D0, DH1))
    base['pass'] = bool(base['nav_max_abs_diff'] == 0 and (base['n_all_ref'] is None or base['n_all'] == base['n_all_ref']))
    print(v, '基準重現', base, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
    assert base['pass'], 'v3 基準重現失敗'
    navF, lgF, stF, midF = std['FZ']
    keys = list(PATH['CUR'])

    def pair(f):
        a = np.array([PATH['FZ'][k][f] for k in keys]); b = np.array([PATH['CUR'][k][f] for k in keys]); dd = (a - b) * 100
        return dict(median=float(np.median(dd)), npos=int((dd > 0).sum()), min=float(dd.min()), max=float(dd.max()), fz_median=float(np.median(a)), cur_median=float(np.median(b)))
    b = bench.reindex(navC.index)
    yC, yF = K.yearly_table(navC, b), K.yearly_table(navF, b)
    for a_, f_ in zip(yC, yF):
        YR.append(dict(version=v, year=a_['year'], current=a_['strat'], frozen=f_['strat'], e0050=a_['e0050'], frozen_minus_current_pp=(f_['strat'] - a_['strat']) * 100,
                       current_minus_0050_pp=a_['diff_pp'], frozen_minus_0050_pp=f_['diff_pp']))
    yd = {r['year']: r for r in YR if r['version'] == v}
    mid = lgF[lgF['sig_t'].isin(list(midF))]; pre = mid[mid['ent_date'] <= DH1]
    R_ = dict(baseline=base, std=dict(CUR=PATH['CUR'][STD], FZ=PATH['FZ'][STD]), pairs={f: pair(f) for f in ('full', 'y2008_2015H1', 'y2015H2_2018', 'since2019', 'maxdd', 'maxdd_2008_2015H1', 'stockfrac_2008_2015H1', 'stockfrac_full')},
              lose_years={y: dict(current_minus_0050_pp=yd[y]['current_minus_0050_pp'], frozen_minus_0050_pp=yd[y]['frozen_minus_0050_pp'], frozen_minus_current_pp=yd[y]['frozen_minus_current_pp']) for y in LOSE_YEARS},
              years_beat_0050=dict(CUR=int(sum(r['current'] > r['e0050'] for r in yd.values())), FZ=int(sum(r['frozen'] > r['e0050'] for r in yd.values())), n=len(yd)),
              top5pct=dict(CUR=top_share(lgC), FZ=top_share(lgF)),
              mid_trades=dict(n=int(len(mid)), mean_ret=float(mid['ret'].mean()) if len(mid) else None, median_ret=float(mid['ret'].median()) if len(mid) else None, win=float((mid['ret'] > 0).mean()) if len(mid) else None,
                              pnl_sum=float(mid['pnl'].sum()), n_2008_2015H1=int(len(pre)), mean_ret_2008_2015H1=float(pre['ret'].mean()) if len(pre) else None,
                              median_ret_2008_2015H1=float(pre['ret'].median()) if len(pre) else None, win_2008_2015H1=float((pre['ret'] > 0).mean()) if len(pre) else None, pnl_sum_2008_2015H1=float(pre['pnl'].sum())),
              paths=PATH)
    # ---- 隨機對照（只做 V_BRIDGE；標準路徑）
    if v == 'V_BRIDGE' and not DRY:
        counts = std['counts']; sg = std['sg']
        held = {r: [int(j) for j, e, xx in zip(lgC['j'].to_numpy(), lgC['ent_t'].to_numpy(), lgC['exit_t'].to_numpy()) if e <= r and xx >= r + 1] for r in counts}
        base_S = CE.apply_cell(P, U, RT, sg, 'CUR', cslib.ins15_mask_row)[0]
        rng = np.random.default_rng(99); dr = []
        for _ in range(20):
            Sr = CE.random_overlay(P, U, base_S, counts, midF, rng, pool='MA_MOM', avoid=held)
            nav, lg, st = K.sim(P, Sr, U, 1e6)
            dr.append(metrics(nav, lg, st, Sr['mid_rows'])); del Sr, nav, lg, st
        fz = PATH['FZ'][STD]; cu = PATH['CUR'][STD]
        R_['random_control'] = dict(note='同樣的月中列、同樣檔數、MA_MOM 池（股池內、季線上、近 20 日漲幅 > 10%，不看 SURGE），20 次、seed 99；V_BRIDGE 標準路徑',
                                    frozen_mid_buys=int(sum(counts.values())),
                                    **{f: dict(frozen=fz[f], current=cu[f], random_median=float(np.median([x[f] for x in dr])), random_min=float(min(x[f] for x in dr)), random_max=float(max(x[f] for x in dr)),
                                               n_random_below_frozen=int(sum(x[f] < fz[f] for x in dr)), n_random_above_current=int(sum(x[f] > cu[f] for x in dr)))
                                       for f in ('y2008_2015H1', 'full', 'since2019', 'stockfrac_2008_2015H1')}, draws=dr)
    res['versions'][v] = R_
    NAVOUT[v + '|現行'] = navC; NAVOUT[v + '|凍結格'] = navF
    print(v, json.dumps({k: R_['pairs'][k] for k in ('y2008_2015H1', 'full', 'since2019', 'maxdd')}, ensure_ascii=False) if not DRY else 'DRY', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
    del std, F, RT, PATH; gc.collect()
vb = res['versions']['V_BRIDGE']; p = vb['pairs']['y2008_2015H1']
ly = vb['lose_years']
res['answer_2008_2015（事前登記的讀法）'] = dict(
    paired_median_pp=p['median'], npos=p['npos'],
    lose_years_2011_2012_2014_improved=int(sum(ly[y]['frozen_minus_current_pp'] > 0 for y in (2011, 2012, 2014))),
    random_n_below_frozen=vb.get('random_control', {}).get('y2008_2015H1', {}).get('n_random_below_frozen'),
    improved=bool(p['median'] > 0 and p['npos'] >= 15 and sum(ly[y]['frozen_minus_current_pp'] > 0 for y in (2011, 2012, 2014)) >= 2
                  and (vb.get('random_control', {}).get('y2008_2015H1', {}).get('n_random_below_frozen') or 0) >= 18))
res['meta'] = dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=rss(), dry=DRY)
if DRY:
    assert all(abs(res['versions'][v]['pairs'][f]['max']) == 0 and res['versions'][v]['pairs'][f]['min'] == 0 for v in res['versions'] for f in ('full', 'y2008_2015H1'))
    print('DRY OK', {v: res['versions'][v]['baseline'] for v in res['versions']}, res['meta'])
else:
    (HERE / 'results_2008.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    df = pd.DataFrame(NAVOUT).round(2); df['0050'] = bench.reindex(df.index).round(2); df.index.name = 'date'; df.to_csv(HERE / 'nav_2008_frozen_vs_current.csv.gz')
    pd.DataFrame(YR).round(6).to_csv(HERE / 'yearly_2008.csv', index=False)
    print(json.dumps(res['answer_2008_2015（事前登記的讀法）'], ensure_ascii=False)); print(json.dumps(vb['lose_years'], ensure_ascii=False)); print(json.dumps(vb['top5pct'], ensure_ascii=False))
    print(json.dumps(vb['mid_trades'], ensure_ascii=False)); print(json.dumps({k: v for k, v in vb.get('random_control', {}).items() if k != 'draws'}, ensure_ascii=False)); print(res['meta'])
