#!/usr/bin/env python3
"""S100 第 5、6 步：凍結格放進 2008 起連續引擎（S75 v3 口徑：7% 漲跌幅＋2008-10 跌幅 3.5%、0050 銜接檔、REDUCE_FIX）。
做法照 agents/s75_continuous_2008_2026/v3_2026-11-25_limit7/run_v3.py（只讀 import s75lib；漲跌停價只在記憶體覆寫；不改任何既有檔）。快取先由該資料夾的 rebuild_cache.py 重建。
先原樣重現 v3：V_BRIDGE／V_MASK 標準路徑淨值與 nav_2008_2026_v3.csv.gz 逐日差 0、交易 627 筆（V_BRIDGE）、R98 的「其餘 595 筆平均超額 −0.3745%」，不符就中止。
再跑：現行 vs 凍結格 × 2 版本 × 20 條路徑；V_BRIDGE 另做：現行縮放到同曝險（20 條）、同曝險隨機對照（標準路徑 20 次、seed 100）、全期可買零股（20 條；描述）、
集中度（前 5% 占比、其餘 95% 平均超額）、第一次換股檢查的描述統計（第 6 步，只用現行交易）。
輸出 results_2008.json、nav_2008_frozen_vs_current.csv.gz、yearly_2008.csv、first_check_current_trades.csv。已有 results_2008.json 就拒絕重跑。
用法：python3 -I -B cont2008.py [--dry｜--dry2]（--dry：凍結格＝「1 份→1 份」只驗基準與流程；--dry2：A|1/2→1@1 走完全程，不寫檔不印數字）"""
import sys, json, time, resource, gc, math, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
S75 = AG / 's75_continuous_2008_2026'; V3 = S75 / 'v3_2026-11-25_limit7'; V2 = S75 / 'v2_2026-11-21'
DRY2 = '--dry2' in sys.argv
DRY = '--dry' in sys.argv or DRY2
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(S75))
import numpy as np, pandas as pd
import s100util as UT
if not DRY:
    assert not (HERE / 'results_2008.json').exists(), 'results_2008.json 已存在'
    PSHA = sha(HERE / 'PREREG.json'); assert PSHA == (HERE / 'PREREG.sha256').read_text().split()[0]
    for ln in (HERE / 'code.sha256').read_text().splitlines():
        h, f = ln.split()
        assert sha(HERE / f) == h, f'{f} 與登記時不同'
    assert sha(HERE / 'FROZEN.json') == (HERE / 'FROZEN.sha256').read_text().split()[0]
    FZID = json.loads((HERE / 'FROZEN.json').read_text())['frozen_for_validation']; FZC = UT.CELLS[FZID]
else:
    FZID = 'A|1/2→1@1' if DRY2 else 'DRY'; FZC = UT.CELLS[FZID] if DRY2 else dict(init=1.0, final=1.0, check=1)
import s75lib as K
from s75lib import R, cslib, s10lib
import s100sim
from s10lib import COSTS

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
D19 = K.last_before(P.IDX, '2019-01-01'); DH1 = K.last_before(P.IDX, '2015-07-01')
benchE = pd.Series(P.E50, index=P.IDX)
bench = benchE.loc[D0:]; bench = bench / bench.iloc[0] * 1e6


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
KW = lambda cap: dict(sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'], capital=cap, cap_pct=0.05)   # 同 s47lib.sim 傳給 simulate_s10 的參數


def sim_fz(sg, cap, rand=None):
    stg = dict(FZC)
    if rand is not None:
        stg['rand'] = rand
    return s100sim.simulate_staged(P, s10lib.variant_S(sg, 'E0')[0], U, stage=stg, **KW(cap))


def metrics(nav, lg, st):
    o = {k: K.cagr(nav, a, b) for k, (a, b) in WINS.items()}
    sf = st['stockfrac']
    o.update(maxdd=R.maxdd(nav), maxdd_2008_2015H1=R.maxdd(nav.loc[D0:DH1]), n_all=int(len(lg)), n_topups=int(len(st['topups'])) if 'topups' in st else 0,
             **{'stockfrac_' + k: float(np.nanmean(sf.loc[a:b].to_numpy())) for k, (a, b) in WINS.items()})
    return o


def scaled(navC, mC, mF):
    o = {}
    for k, (a, b) in WINS.items():
        lam = mF['stockfrac_' + k] / mC['stockfrac_' + k]
        o[k] = K.cagr(UT.scaled_current(navC, benchE, lam, a, b)); o['lam_' + k] = float(lam)
    return o


def as_lots(lg):
    x = lg[['j', 'sig_t', 'ent_t', 'exit_t', 'cost', 'proceeds', 'why', 'ret', 'pnl', 'ent_date', 'exit_date']].copy(); x['lot'] = 0; x['n_lots'] = 1
    return x


res = dict(frozen=FZID, frozen_params=FZC, engine='S75 v3（7% 漲跌幅＋2008-10 跌幅 3.5%）', versions={}, e0050={k: K.cagr(bench, a, b) for k, (a, b) in WINS.items()})
NAVOUT = {}; YR = []; FC = None
odd0 = cslib.ODD_DATE
for v, rule in K.VERSIONS.items():
    F, RT = K.surge(P, rule)
    PATH = {'CUR': {}, 'FZ': {}, 'SCALED': {}, 'CUR_odd': {}, 'FZ_odd': {}}; std = {}
    for s in K.SHIFTS:
        sg = K.signal(P, U, RT, s)
        for cap in K.CAPS:
            key = f'{int(cap / 1e4)}萬|+{s}'
            navC, lgC, stC = K.sim(P, sg, U, cap); PATH['CUR'][key] = metrics(navC, lgC, stC)
            navF, lgF, stF = sim_fz(sg, cap); PATH['FZ'][key] = metrics(navF, lgF, stF)
            PATH['SCALED'][key] = scaled(navC, PATH['CUR'][key], PATH['FZ'][key])
            if key == STD:
                std = dict(CUR=(navC, lgC, stC), FZ=(navF, lgF, stF), sg=sg)
            del navC, lgC, stC, navF, lgF, stF
            if v == 'V_BRIDGE':                                # 描述：全期可買零股（只在記憶體改；跑完還原）
                cslib.ODD_DATE = pd.Timestamp('2000-01-01')
                n1, l1, s1 = K.sim(P, sg, U, cap); PATH['CUR_odd'][key] = metrics(n1, l1, s1); del n1, l1, s1
                n1, l1, s1 = sim_fz(sg, cap); PATH['FZ_odd'][key] = metrics(n1, l1, s1); del n1, l1, s1
                cslib.ODD_DATE = odd0
        if s != 0:
            del sg
        gc.collect()
    navC, lgC, stC = std['CUR']; navF, lgF, stF = std['FZ']; sg = std['sg']
    base = dict(nav_max_abs_diff=float((navC.round(2) - ref_nav[v]).abs().max()), n_all=int(len(lgC)), n_all_ref=int(ref_res['versions'][v]['n_all']) if 'versions' in ref_res else None,
                cagr=K.cagr(navC, D0, DE), cagr_2008_2015H1=K.cagr(navC, D0, DH1))
    cC = UT.conc_stats(lgC, as_lots(lgC), benchE); cF = UT.conc_stats(lgF, stF['lots'], benchE)
    base['pass'] = bool(base['nav_max_abs_diff'] == 0 and (base['n_all_ref'] is None or base['n_all'] == base['n_all_ref']))
    if v == 'V_BRIDGE':
        base['r98_rest595_mean_ex'] = cC['rest_mean_ex']; base['r98_all_mean_ex'] = cC['all_mean_ex']
        base['pass'] = bool(base['pass'] and base['n_all'] == 627 and abs(cC['rest_mean_ex'] + 0.003745387608196045) < 1e-9 and abs(cC['all_mean_ex'] - 0.0403228499127219) < 1e-9 and cC['n_top'] == 32)
    print(v, '基準重現', base, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
    assert base['pass'], 'v3 基準重現失敗'
    keys = list(PATH['CUR'])

    def pair(f, a='FZ', b='CUR'):
        x = np.array([PATH[a][k][f] for k in keys]); y = np.array([PATH[b][k][f] for k in keys]); dd = (x - y) * 100
        return dict(median=float(np.median(dd)), npos=int((dd > 0).sum()), min=float(dd.min()), max=float(dd.max()), a_median=float(np.median(x)), b_median=float(np.median(y)))
    b = bench.reindex(navC.index)
    yC, yF = K.yearly_table(navC, b), K.yearly_table(navF, b)
    for a_, f_ in zip(yC, yF):
        YR.append(dict(version=v, year=a_['year'], current=a_['strat'], frozen=f_['strat'], e0050=a_['e0050'], frozen_minus_current_pp=(f_['strat'] - a_['strat']) * 100,
                       current_minus_0050_pp=a_['diff_pp'], frozen_minus_0050_pp=f_['diff_pp']))
    yd = {r['year']: r for r in YR if r['version'] == v}
    tp = stF['topups']; lt = stF['lots']
    tst = lambda y: dict(n=int(len(y)), mean_ret=float(y['ret'].mean()) if len(y) else None, median_ret=float(y['ret'].median()) if len(y) else None, win=float((y['ret'] > 0).mean()) if len(y) else None,
                         pnl_sum=float(y['pnl'].sum()), cost_sum=float(y['cost'].sum()))
    R_ = dict(baseline=base, std=dict(CUR=PATH['CUR'][STD], FZ=PATH['FZ'][STD], SCALED=PATH['SCALED'][STD]),
              pairs={f: pair(f) for f in ('full', 'y2008_2015H1', 'y2015H2_2018', 'since2019', 'maxdd', 'maxdd_2008_2015H1', 'stockfrac_full', 'stockfrac_y2008_2015H1', 'stockfrac_since2019')},
              pairs_vs_scaled={f: pair(f, 'FZ', 'SCALED') for f in WINS}, pairs_scaled_vs_current={f: pair(f, 'SCALED', 'CUR') for f in WINS},
              years_beat_0050=dict(CUR=int(sum(r['current'] > r['e0050'] for r in yd.values())), FZ=int(sum(r['frozen'] > r['e0050'] for r in yd.values())), n=len(yd)),
              concentration=dict(CUR=cC, FZ=cF),
              frozen_counts=dict(topups=int(len(tp)), pend=int(stF['top_pend']), top_blk_limup=int(stF['top_blk_limup']), top_blk_min=int(stF['top_blk_min']), top_blk_noopen=int(stF['top_blk_noopen']), top_room_bind=int(stF['top_room_bind']),
                                 blk_lot=int(stF['blk_lot']), blk_lot_current=int(stC['blk_lot']), gain_since_entry_mean=float(tp['gain_since_entry'].mean()) if len(tp) else None,
                                 gain_since_entry_median=float(tp['gain_since_entry'].median()) if len(tp) else None),
              frozen_lots=dict(first=tst(lt[lt['lot'] == 0]), first_of_topped=tst(lt[(lt['lot'] == 0) & (lt['n_lots'] > 1)]), first_never_topped=tst(lt[(lt['lot'] == 0) & (lt['n_lots'] == 1)]), topup=tst(lt[lt['lot'] > 0])),
              paths=PATH)
    if v == 'V_BRIDGE':
        R_['pairs_odd_all'] = {f: pair(f, 'FZ_odd', 'CUR_odd') for f in ('full', 'y2008_2015H1', 'since2019', 'stockfrac_full', 'n_all')}
        fc, fcd = UT.first_check_stats(P, lgC, sg, benchE)
        R_['first_check_current_trades'] = fc; FC = fcd.assign(code=P.COLS[fcd['j'].to_numpy()], ent_date=P.IDX[fcd['ent_t'].to_numpy()]).drop(columns=['j', 'ent_t'])
        # 同曝險隨機對照（標準路徑；描述）
        Kc = UT.topup_K(stF); rng = np.random.default_rng(100); dr = []
        for _ in range(20):
            nav, lg, st = sim_fz(sg, 1e6, rand=dict(K=Kc, rng=rng))
            m = metrics(nav, lg, st); m['conc_top_share'] = UT.conc_stats(lg, st['lots'], benchE)['top_share_of_pnl']; dr.append(m); del nav, lg, st
        fz = PATH['FZ'][STD]; cu = PATH['CUR'][STD]
        R_['random_control'] = dict(note='每個執行日同樣個數的補買額度，隨機分給（未補過的續抱股 ∪ 當天新買名單）；20 次、seed 100；V_BRIDGE 標準路徑', frozen_topups=int(sum(Kc.values())),
                                    **{f: dict(frozen=fz[f], current=cu[f], random_median=float(np.median([x[f] for x in dr])), random_min=float(min(x[f] for x in dr)), random_max=float(max(x[f] for x in dr)),
                                               n_random_below_frozen=int(sum(x[f] < fz[f] for x in dr))) for f in ('y2008_2015H1', 'full', 'since2019', 'stockfrac_full', 'n_all', 'n_topups')},
                                    conc_top_share=dict(frozen=cF['top_share_of_pnl'], current=cC['top_share_of_pnl'], random_median=float(np.median([x['conc_top_share'] for x in dr]))), draws=dr)
    res['versions'][v] = R_
    NAVOUT[v + '|現行'] = navC; NAVOUT[v + '|凍結格'] = navF
    if v == 'V_BRIDGE':
        TRF = lgF.drop(columns=['j']); LTF = lt.drop(columns=['j'])
    print(v, json.dumps({k: R_['pairs'][k] for k in ('y2008_2015H1', 'full', 'since2019', 'maxdd')}, ensure_ascii=False) if not DRY else 'DRY', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
    del std, F, RT, PATH; gc.collect()
assert cslib.ODD_DATE == odd0
vb = res['versions']['V_BRIDGE']; c = vb['concentration']
res['answers（事前登記的讀法；V_BRIDGE 標準路徑）'] = dict(
    a_top5pct_share_current=c['CUR']['top_share_of_pnl'], a_top5pct_share_frozen=c['FZ']['top_share_of_pnl'], a_share_decreased=bool(c['FZ']['top_share_of_pnl'] < c['CUR']['top_share_of_pnl']),
    b_rest_cw_ex_current=c['CUR']['rest_cw_ex'], b_rest_cw_ex_frozen=c['FZ']['rest_cw_ex'], b_rest_mean_ex_current=c['CUR']['rest_mean_ex'], b_rest_mean_ex_frozen=c['FZ']['rest_mean_ex'],
    b_improved_cost_weighted=bool(c['FZ']['rest_cw_ex'] > c['CUR']['rest_cw_ex']), b_improved_equal_weight=bool(c['FZ']['rest_mean_ex'] > c['CUR']['rest_mean_ex']),
    b_rest_positive_after=bool(c['FZ']['rest_cw_ex'] > 0 and c['FZ']['rest_mean_ex'] > 0))
res['meta'] = dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=rss(), dry=DRY)
if DRY2:
    json.dumps(res, ensure_ascii=False, default=str); print('DRY2 OK（未印任何數字、未寫檔）', res['meta'])
elif DRY:
    assert all(res['versions'][v]['pairs'][f]['max'] == 0 and res['versions'][v]['pairs'][f]['min'] == 0 for v in res['versions'] for f in ('full', 'y2008_2015H1', 'since2019'))
    assert abs(c['FZ']['rest_mean_ex'] - c['CUR']['rest_mean_ex']) < 1e-12 and abs(c['FZ']['top_share_of_pnl'] - c['CUR']['top_share_of_pnl']) < 1e-12
    print('DRY OK', {v: res['versions'][v]['baseline'] for v in res['versions']}, res['meta'])
else:
    (HERE / 'results_2008.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    df = pd.DataFrame(NAVOUT).round(2); df['0050'] = bench.reindex(df.index).round(2); df.index.name = 'date'; df.to_csv(HERE / 'nav_2008_frozen_vs_current.csv.gz')
    pd.DataFrame(YR).round(6).to_csv(HERE / 'yearly_2008.csv', index=False)
    FC.round(6).to_csv(HERE / 'first_check_current_trades.csv', index=False)
    TRF.round(6).to_csv(HERE / 'trades_2008_frozen.csv.gz', index=False); LTF.round(6).to_csv(HERE / 'lots_2008_frozen.csv.gz', index=False)
    print(json.dumps(res['answers（事前登記的讀法；V_BRIDGE 標準路徑）'], ensure_ascii=False))
    for k in ('pairs', 'pairs_vs_scaled', 'pairs_scaled_vs_current', 'pairs_odd_all', 'concentration', 'frozen_counts', 'frozen_lots', 'first_check_current_trades', 'years_beat_0050'):
        print(k, json.dumps(vb[k], ensure_ascii=False))
    print(json.dumps({k: v for k, v in vb['random_control'].items() if k != 'draws'}, ensure_ascii=False)); print(res['meta'])
