#!/usr/bin/env python3
"""S100 驗證期（只跑一次）：凍結格 vs 現行，重新起跑（訊號日 2022-01 起；起始資金按現行 2021-12-30 淨值縮放）× 20 條路徑，驗證窗 2021-12-30 → 2026-10-22；S25 的 V1～V4。
另：R1 同曝險隨機對照（20 次、seed 100）、R2 現行線性縮放到同曝險（20 條路徑）、補買逐筆、曝險、成本加倍、全期連續（描述）、全期可買零股（描述）。
results.json 已存在就拒絕重跑。用法：python3 -I -B validate.py [--dry｜--dry2]
  --dry ：凍結格換成「1 份→1 份」（＝現行），只檢查流程，配對差應全為 0；不寫檔。
  --dry2：凍結格換成 A|1/2→1@1 走完全部程式路徑；不寫檔、不印任何數字。"""
import sys, json, time, resource, gc, hashlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
DRY2 = '--dry2' in sys.argv
DRY = '--dry' in sys.argv or DRY2
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
PSHA = None
if not DRY:
    PSHA = sha(HERE / 'PREREG.json'); assert PSHA == (HERE / 'PREREG.sha256').read_text().split()[0]
    for ln in (HERE / 'code.sha256').read_text().splitlines():
        h, f = ln.split()
        assert sha(HERE / f) == h, f'{f} 與登記時不同'
sys.path.insert(0, str(HERE))
import s100lib as L
from s100lib import stlib, s2lib, cslib
if DRY:
    FZID, T0, SELECTED = ('A|1/2→1@1' if DRY2 else 'DRY'), False, 'DRY'
    FZC = L.CELLS[FZID] if DRY2 else dict(init=1.0, final=1.0, check=1)
else:
    assert not (HERE / 'results.json').exists(), 'results.json 已存在：驗證期只跑一次'
    assert sha(HERE / 'FROZEN.json') == (HERE / 'FROZEN.sha256').read_text().split()[0]
    FZ = json.loads((HERE / 'FROZEN.json').read_text()); assert FZ['PREREG_sha256'] == PSHA
    FZID, T0, SELECTED = FZ['frozen_for_validation'], bool(FZ['T0_pass']), FZ['selected']
    FZC = L.CELLS[FZID]
t_all = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT = L.load()
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19, TA, TB, T23 = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01'), lastb('2016-02-01'), lastb('2022-01-01'), lastb('2024-01-01')
WIN = dict(valid=(TB, DE), y2022_2023=(TB, T23), since2024=(T23, DE))
cg = lambda nav, k: stlib.cagr(nav, *WIN[k])
med = lambda v: float(np.median(v))
bench = pd.Series(P.E50, index=P.IDX)
sfm = lambda st, a=TB, b=DE: float(np.nanmean(st['stockfrac'].loc[a:b].to_numpy()))

S0 = L.signal(P, U, RT, 0)
ncur, lcur, stcur = L.sim(P, S0, U, cap=1e6)
ratio = float(ncur.loc[TB] / 1e6)
FCAP = {cap: cap / ratio for cap in L.CAPS}
nfz_c, lfz_c, stfz_c = L.sim(P, S0, U, cap=1e6, cell=FZC)
expo_full = dict(current=sfm(stcur, D0, DE), frozen=sfm(stfz_c, D0, DE))
expo_full['diff_pp'] = (expo_full['frozen'] - expo_full['current']) * 100
R2_TRIGGER = bool(abs(expo_full['diff_pp']) > 3.0)
del ncur, lcur, stcur, nfz_c, lfz_c, stfz_c
F0 = L.fresh_from(S0)
nC, lC, stC = L.sim(P, F0, U, cap=FCAP[1e6])
nF, lF, stF = L.sim(P, F0, U, cap=FCAP[1e6], cell=FZC)
acc = dict(A3_no_entry_before_2022=bool((lC['ent_date'] >= pd.Timestamp('2022-01-01')).all() and (lF['ent_date'] >= pd.Timestamp('2022-01-01')).all()), ratio_nav_TB=ratio)
assert acc['A3_no_entry_before_2022']
res = dict(PREREG_sha256=PSHA, frozen=FZID, frozen_params=FZC, selected=SELECTED, T0_pass=T0, acceptance=acc, main={}, e50_valid=stlib.cagr(bench, TB, DE), exposure_full_period_std=expo_full, R2_triggered=R2_TRIGGER)


def tstats(y):
    return dict(n=int(len(y)), win=float((y['ret'] > 0).mean()) if len(y) else None, mean_ret=float(y['ret'].mean()) if len(y) else None, median_ret=float(y['ret'].median()) if len(y) else None,
                pnl_sum=float(y['pnl'].sum()), cost_sum=float(y['cost'].sum()), pnl_per_cost=float(y['pnl'].sum() / y['cost'].sum()) if len(y) and y['cost'].sum() > 0 else None)


for nm, nav, lg, st in (('CUR', nC, lC, stC), ('FZ', nF, lF, stF)):
    cl = lg[lg['why'] != 'open_end']
    top5 = cl.sort_values('pnl', ascending=False).head(5)
    r = cl['ret'].to_numpy(); pn = cl['pnl'].to_numpy(); n5 = int(np.ceil(len(cl) * 0.05)); o5 = np.argsort(-r, kind='stable')[:n5]
    o = dict(cagr={k: cg(nav, k) for k in WIN}, maxdd=stlib.maxdd(nav, TB, DE), seg=L.seg_dd(nav), n_closed=int(len(cl)), n_all=int(len(lg)), trades=tstats(cl),
             avg_stockfrac=sfm(st), final_nav=float(nav.iloc[-1]), top5_pnl_share=float(top5['pnl'].sum() / cl['pnl'].sum()), top5=[str(c) for c in top5['code']],
             top5pct_by_ret=dict(n=n5, pnl_share=float(pn[o5].sum() / pn.sum()), rest_mean_ret=float(np.delete(r, o5).mean()), rest_pnl_sum=float(np.delete(pn, o5).sum())),
             buy_val=float(st['buy_val']), noslot=int(st['noslot']), blk_limup=int(st['blk_limup']), blk_min=int(st['blk_min']), blk_lot=int(st['blk_lot']), blk_nocash=int(st['blk_nocash']))
    if nm == 'FZ':
        tp = st['topups']; lt = st['lots']; ltc = lt[lt['why'] != 'open_end']
        o['topups'] = dict(n=int(len(tp)), pend=int(st['top_pend']), blk_limup=int(st['top_blk_limup']), blk_min=int(st['top_blk_min']), blk_noopen=int(st['top_blk_noopen']), room_bind=int(st['top_room_bind']),
                           add_val_sum=float(tp['add_val'].sum()), gain_since_entry_mean=float(tp['gain_since_entry'].mean()) if len(tp) else None, gain_since_entry_median=float(tp['gain_since_entry'].median()) if len(tp) else None)
        o['lots'] = dict(first=tstats(ltc[ltc['lot'] == 0]), first_of_topped=tstats(ltc[(ltc['lot'] == 0) & (ltc['n_lots'] > 1)]), first_never_topped=tstats(ltc[(ltc['lot'] == 0) & (ltc['n_lots'] == 1)]), topup=tstats(ltc[ltc['lot'] > 0]))
    res['main'][nm] = o
res['main']['CUR']['costx2_valid'] = cg(L.sim(P, F0, U, cap=FCAP[1e6], cost='strict_x2')[0], 'valid')
res['main']['FZ']['costx2_valid'] = cg(L.sim(P, F0, U, cap=FCAP[1e6], cost='strict_x2', cell=FZC)[0], 'valid')
yr = lambda nav: {int(y): float(v) for y, v in (nav.loc[TB:DE].groupby(nav.loc[TB:DE].index.year).last().pct_change().dropna()).items()}
res['yearly_valid'] = dict(CUR=yr(nC), FZ=yr(nF), e0050=yr(bench))
if not DRY:
    x = lF.drop(columns=['j']).copy(); x['top_date'] = [str(P.IDX[t].date()) if t >= 0 else '' for t in x['top_t']]
    x.round(6).to_csv(HERE / 'valid_trades_frozen.csv.gz', index=False)
    stF['lots'].drop(columns=['j']).round(6).to_csv(HERE / 'valid_lots_frozen.csv.gz', index=False)
    pd.DataFrame({'CUR': nC.loc[TB:], 'FZ': nF.loc[TB:], '0050': bench.loc[TB:] / bench.loc[TB] * FCAP[1e6]}).round(2).to_csv(HERE / 'nav_valid_frozen_vs_current.csv.gz')

# ---- R1 同曝險隨機對照（標準路徑、重新起跑；每個執行日同樣個數的補買額度，隨機分給「未補過的續抱股 ∪ 當天新買名單」；20 次，seed 100）
K = L.topup_K(stF)
rng = np.random.default_rng(100); dr = []; sf = []; ntp = []; nfull = []; npos = []
vF = res['main']['FZ']['cagr']['valid']; vC = res['main']['CUR']['cagr']['valid']
for _ in range(20):
    nav, lg, st = L.sim(P, F0, U, cap=FCAP[1e6], cell=FZC, rand=dict(K=K, rng=rng))
    dr.append(cg(nav, 'valid')); sf.append(sfm(st)); ntp.append(int(len(st['topups']))); nfull.append(int(st['new_full'])); npos.append(int(len(lg))); del nav, lg, st
res['R1_random_control'] = dict(note='每個換股執行日，把凍結格當天實際補買成功的檔數 K 當成「補買額度」，隨機分給（未補過的續抱股 ∪ 當天新買名單）；抽到的新買股直接買補滿後的大小；名單、出場、成本同凍結格。',
                                frozen_valid=vF, current_valid=vC, draws=dr, median=med(dr), min=float(min(dr)), max=float(max(dr)), n_random_below_frozen=int(sum(d < vF for d in dr)),
                                frozen_topups=int(sum(K.values())), random_topups_median=med(ntp), random_new_full_median=med(nfull), random_positions_median=med(npos), frozen_positions=int(len(lF)),
                                frozen_avg_stockfrac=res['main']['FZ']['avg_stockfrac'], random_avg_stockfrac_median=med(sf), random_avg_stockfrac_min=float(min(sf)), random_avg_stockfrac_max=float(max(sf)),
                                R1_pass=bool(sum(d < vF for d in dr) >= 18))

# ---- V3 禁買聯集
BLK = {j: list(v) for j, v in s2lib.block_from(lC, 5)[0].items()}
for j, v in s2lib.block_from(lF, 5)[0].items():
    BLK.setdefault(j, []).extend(v)
res['V3_block_union'] = {str(P.COLS[j]): [[str(P.IDX[a].date()), str(P.IDX[b].date())] for a, b in v] for j, v in BLK.items()}
del S0, F0; gc.collect()

# ---- 20 條路徑（一次一格）
NAMES = ('CUR|fresh', 'FZ|fresh', 'SCALED|fresh', 'CUR|blk', 'FZ|blk', 'CUR|cont', 'FZ|cont', 'CUR|cont_odd', 'FZ|cont_odd')
PATH = {k: {} for k in NAMES}
odd0 = cslib.ODD_DATE
for s in L.SHIFTS:
    S = L.signal(P, U, RT, s); Fx = L.fresh_from(S)
    for cap in L.CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        navs = {}
        for nm, cell in (('CUR', None), ('FZ', FZC)):
            nav, lg, st = L.sim(P, Fx, U, cap=FCAP[cap], cell=cell)
            PATH[f'{nm}|fresh'][key] = dict(valid=cg(nav, 'valid'), valid_dd=stlib.maxdd(nav, TB, DE), y2022_2023=cg(nav, 'y2022_2023'), since2024=cg(nav, 'since2024'), seg=L.seg_dd(nav), avg_stockfrac=sfm(st),
                                            n_pos=int(len(lg)), n_topups=int(len(st['topups'])) if 'topups' in st else 0)
            navs[nm] = nav; del lg, st
            PATH[f'{nm}|blk'][key] = dict(valid=cg(L.sim(P, Fx, U, cap=FCAP[cap], block=BLK, cell=cell)[0], 'valid'))
            nav, lg, st = L.sim(P, S, U, cap=cap, cell=cell)
            PATH[f'{nm}|cont'][key] = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE), train=stlib.cagr(nav, TA, TB), avg_stockfrac=sfm(st, D0, DE)); del nav, lg, st
            cslib.ODD_DATE = pd.Timestamp('2000-01-01')         # 描述：全期可買零股（只在記憶體改；跑完還原）
            nav, lg, st = L.sim(P, S, U, cap=cap, cell=cell)
            cslib.ODD_DATE = odd0
            PATH[f'{nm}|cont_odd'][key] = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), train=stlib.cagr(nav, TA, TB), avg_stockfrac=sfm(st, D0, DE), n_pos=int(len(lg))); del nav, lg, st
        lam = PATH['FZ|fresh'][key]['avg_stockfrac'] / PATH['CUR|fresh'][key]['avg_stockfrac']
        sc = L.scaled_current(navs['CUR'], bench, lam, TB, DE)
        PATH['SCALED|fresh'][key] = dict(valid=stlib.cagr(sc, TB, DE), valid_dd=stlib.maxdd(sc, TB, DE), lam=float(lam)); del navs, sc
    del S, Fx; gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
assert cslib.ODD_DATE == odd0
res['paths'] = PATH
keys = list(PATH['CUR|fresh'])


def pair(a, b, f, sub=None):
    ga = np.array([PATH[a][k][f][sub] if sub else PATH[a][k][f] for k in keys]); gb = np.array([PATH[b][k][f][sub] if sub else PATH[b][k][f] for k in keys])
    d = (ga - gb) * 100
    return dict(median=float(np.median(d)), npos=int((d > 0).sum()), min=float(d.min()), max=float(d.max()), a_median=float(np.median(ga)), b_median=float(np.median(gb)))


pv = dict(valid=pair('FZ|fresh', 'CUR|fresh', 'valid'), valid_dd=pair('FZ|fresh', 'CUR|fresh', 'valid_dd'), y2022_2023=pair('FZ|fresh', 'CUR|fresh', 'y2022_2023'), since2024=pair('FZ|fresh', 'CUR|fresh', 'since2024'),
          seg={sg: pair('FZ|fresh', 'CUR|fresh', 'seg', sg) for sg in L.SEGV}, blk=pair('FZ|blk', 'CUR|blk', 'valid'), stockfrac=pair('FZ|fresh', 'CUR|fresh', 'avg_stockfrac'),
          R2_vs_scaled=pair('FZ|fresh', 'SCALED|fresh', 'valid'), R2_vs_scaled_dd=pair('FZ|fresh', 'SCALED|fresh', 'valid_dd'), scaled_vs_current=pair('SCALED|fresh', 'CUR|fresh', 'valid'),
          cont={f: pair('FZ|cont', 'CUR|cont', f) for f in ('since2019', 'full', 'maxdd', 'train', 'avg_stockfrac')},
          cont_odd={f: pair('FZ|cont_odd', 'CUR|cont_odd', f) for f in ('since2019', 'full', 'train', 'avg_stockfrac')})
res['pairs'] = pv
V = dict(V1=pv['valid']['npos'] >= 15, V2=pv['valid']['median'] >= 2.0, V3=pv['blk']['median'] > 0, V4=all(pv['seg'][sg]['median'] >= -2.0 for sg in L.SEGV))
V = {k: bool(v) for k, v in V.items()}
R1 = bool(res['R1_random_control']['R1_pass'])
R2raw = bool(pv['R2_vs_scaled']['median'] > 0 and pv['R2_vs_scaled']['npos'] >= 15)
R2 = bool(R2raw or not R2_TRIGGER)
passed = bool(SELECTED not in ('不改', 'DRY') and SELECTED == FZID and T0 and all(V.values()) and R1 and R2)
res['verdict'] = dict(selected=SELECTED, frozen=FZID, T0=T0, **V, R1_random_control=R1, R2_triggered=R2_TRIGGER, R2_beats_scaled_current=R2raw, R2=R2, n_pass_V=int(sum(V.values())), overall='通過（僅為候選）' if passed else '不通過')
res['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss(), dry=DRY)
if DRY2:
    json.dumps(res, ensure_ascii=False, default=str); print('DRY2 OK（未印任何數字、未寫檔）', res['meta'])
elif DRY:
    assert all(abs(pv[k]['median']) == 0 and pv[k]['min'] == 0 and pv[k]['max'] == 0 for k in ('valid', 'valid_dd', 'blk')), {k: pv[k] for k in ('valid', 'valid_dd', 'blk')}
    assert max(abs(pv['R2_vs_scaled']['min']), abs(pv['R2_vs_scaled']['max'])) < 1e-9, pv['R2_vs_scaled']      # lam＝1 時縮放版＝現行（只差浮點誤差）
    assert abs(pv['cont']['since2019']['b_median'] - L.TARGET['paths_median']) < 1e-12, pv['cont']['since2019']
    assert all(d == vC for d in dr)
    print('DRY OK', res['main']['CUR']['n_closed'], res['meta'])
else:
    (HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print('判定', json.dumps(res['verdict'], ensure_ascii=False))
    print(json.dumps(pv, ensure_ascii=False))
    print(json.dumps(res['main'], ensure_ascii=False))
    print(json.dumps({k: v for k, v in res['R1_random_control'].items() if k not in ('draws', 'note')}, ensure_ascii=False)); print(expo_full, res['meta'])
