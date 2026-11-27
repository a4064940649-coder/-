#!/usr/bin/env python3
"""S102 驗證期（只跑一次；U1、U2 兩題同一次執行、各自判定）：凍結格 vs 現行，重新起跑（訊號日 2022-01 起；起始資金按現行 2021-12-30 淨值縮放）× 20 條路徑，
驗證窗 2021-12-30 → 2026-10-22；S25 的 V1～V4；R2 現行線性縮放到同曝險（觸發時為必要條件）；另報新增交易逐筆、成本加倍、逐年、全期連續（描述）、集中度與滾動一年 SR<0 比例。
results.json 已存在就拒絕重跑。用法：python3 -I -B validate.py [--dry0｜--dry]
  --dry0：凍結格換成現行（CUR），只檢查流程，配對差應全為 0；不寫檔。
  --dry ：凍結格換成 U1|r20>0、U2|any 走完全部程式路徑；不寫檔、不印任何數字。"""
import sys, json, time, resource, gc, hashlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
DRY0 = '--dry0' in sys.argv
DRY = '--dry' in sys.argv or DRY0
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
PSHA = None
if not DRY:
    PSHA = sha(HERE / 'PREREG.json'); assert PSHA == (HERE / 'PREREG.sha256').read_text().split()[0]
    for ln in (HERE / 'code.sha256').read_text().splitlines():
        h, f = ln.split()
        assert sha(HERE / f) == h, f'{f} 與登記時不同'
sys.path.insert(0, str(HERE))
import s102lib as L
from s102lib import stlib, s2lib
if DRY:
    TOP = {tp: dict(frozen_for_validation=('CUR' if DRY0 else c), selected='DRY', T0_pass=False) for tp, c in (('U1', 'U1|r20>0'), ('U2', 'U2|any'))}
else:
    assert not (HERE / 'results.json').exists(), 'results.json 已存在：驗證期只跑一次'
    assert sha(HERE / 'FROZEN.json') == (HERE / 'FROZEN.sha256').read_text().split()[0]
    FZ = json.loads((HERE / 'FROZEN.json').read_text()); assert FZ['PREREG_sha256'] == PSHA
    TOP = FZ['topics']
t_all = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT = L.load()
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19, TA, TB, T23 = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01'), lastb('2016-02-01'), lastb('2022-01-01'), lastb('2024-01-01')
WIN = dict(valid=(TB, DE), y2022_2023=(TB, T23), since2024=(T23, DE))
cg = lambda nav, k: stlib.cagr(nav, *WIN[k])
bench = pd.Series(P.E50, index=P.IDX)
sfm = lambda st, a=TB, b=DE: float(np.nanmean(st['stockfrac'].loc[a:b].to_numpy()))
FZC = {tp: TOP[tp]['frozen_for_validation'] for tp in TOP}
NM = ['CUR'] + [f'FZ_{tp}' for tp in TOP]
CELL = dict(CUR='CUR', **{f'FZ_{tp}': FZC[tp] for tp in TOP})


def tstats(y):
    return dict(n=int(len(y)), win=float((y['ret'] > 0).mean()) if len(y) else None, mean_ret=float(y['ret'].mean()) if len(y) else None, median_ret=float(y['ret'].median()) if len(y) else None,
                pnl_sum=float(y['pnl'].sum()), cost_sum=float(y['cost'].sum()), pnl_per_cost=float(y['pnl'].sum() / y['cost'].sum()) if len(y) and y['cost'].sum() > 0 else None)


def extra_mask(lg, S, st, cell):
    """新增的交易：U1＝遞補股買進；U2＝買回。"""
    if cell == 'CUR':
        return np.zeros(len(lg), bool)
    if L.CELLS[cell]['topic'] == 'U1':
        fk = S['FILL']; return np.array([int(j) in set(fk.get(int(k), [])) for j, k in zip(lg['j'], lg['sig_t'])], bool)
    return lg['rb'].to_numpy(bool) if 'rb' in lg else np.zeros(len(lg), bool)


def ex_0050(lg):
    return lg['ret'].to_numpy() - (bench.reindex(pd.DatetimeIndex(lg['exit_date'])).to_numpy() / bench.reindex(pd.DatetimeIndex(lg['ent_date'])).to_numpy() - 1)


res = dict(PREREG_sha256=PSHA, frozen=FZC, selected={tp: TOP[tp]['selected'] for tp in TOP}, T0_pass={tp: bool(TOP[tp]['T0_pass']) for tp in TOP}, e50_valid=stlib.cagr(bench, TB, DE), topics={})

# ---- 全期連續標準路徑：曝險（R2 觸發）、集中度、滾動一年 SR<0
SG0 = {nm: L.signal(P, U, RT, 0, CELL[nm]) for nm in NM}
cont_std = {}
for nm in NM:
    nav, lg, st = L.sim(P, SG0[nm], U, cap=1e6, cell=CELL[nm])
    cont_std[nm] = dict(avg_stockfrac=sfm(st, D0, DE), since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE),
                        roll252_sr_neg=L.roll_sr_neg(nav), top5pct=L.top5_share(lg), n_extra=int(extra_mask(lg, SG0[nm], st, CELL[nm]).sum()))
    if nm == 'CUR':
        ratio = float(nav.loc[TB] / 1e6)
    del nav, lg, st
FCAP = {cap: cap / ratio for cap in L.CAPS}
res['cont_std_2015'] = cont_std; res['ratio_nav_TB'] = ratio

# ---- 重新起跑主跑（100 萬縮放、位移 0）
F0 = {nm: L.fresh_from(SG0[nm]) for nm in NM}
MAIN = {}; LG = {}
for nm in NM:
    nav, lg, st = L.sim(P, F0[nm], U, cap=FCAP[1e6], cell=CELL[nm])
    assert (lg['ent_date'] >= pd.Timestamp('2022-01-01')).all()
    cl = lg[lg['why'] != 'open_end']
    top5 = cl.sort_values('pnl', ascending=False).head(5)
    em = extra_mask(lg, F0[nm], st, CELL[nm]); x = lg[em].copy()
    o = dict(cagr={k: cg(nav, k) for k in WIN}, maxdd=stlib.maxdd(nav, TB, DE), seg=L.seg_dd(nav), n_closed=int(len(cl)), n_all=int(len(lg)), trades=tstats(cl),
             avg_stockfrac=sfm(st), final_nav=float(nav.iloc[-1]), top5_pnl_share=float(top5['pnl'].sum() / cl['pnl'].sum()), top5=[str(c) for c in top5['code']],
             top5pct=L.top5_share(lg), roll252_sr_neg_valid=L.roll_sr_neg(nav.loc[TB:DE]), noslot=int(st['noslot']), blk_limup=int(st['blk_limup']), blk_min=int(st['blk_min']), blk_lot=int(st['blk_lot']))
    if nm != 'CUR':
        x['ex'] = ex_0050(x) if len(x) else []
        o['extra_trades'] = dict(**tstats(x), mean_ex_0050=float(x['ex'].mean()) if len(x) else None, median_ex_0050=float(x['ex'].median()) if len(x) else None,
                                 list=[dict(code=str(r.code), ent=str(r.ent_date.date()), exit=str(r.exit_date.date()), ret=float(r.ret), ex=float(r.ex), why=str(r.why)) for r in x.itertuples()])
        if L.CELLS.get(CELL[nm], {}).get('topic') == 'U2':
            o['rb_counts'] = {k: int(st[k]) for k in ('rb_flag', 'rb_win', 'rb_trig', 'rb_buy', 'rb_fail', 'rb_noslot')}
    o['costx2_valid'] = cg(L.sim(P, F0[nm], U, cap=FCAP[1e6], cost='strict_x2', cell=CELL[nm])[0], 'valid')
    MAIN[nm] = o; LG[nm] = (nav, lg)
yr = lambda nav: {int(y): float(v) for y, v in (nav.loc[TB:DE].groupby(nav.loc[TB:DE].index.year).last().pct_change().dropna()).items()}
res['yearly_valid'] = dict(e0050=yr(bench), **{nm: yr(LG[nm][0]) for nm in NM})
if not DRY:
    df = pd.DataFrame({nm: LG[nm][0].loc[TB:] for nm in NM}); df['0050'] = bench.loc[TB:] / bench.loc[TB] * FCAP[1e6]
    df.round(2).to_csv(HERE / 'nav_valid_frozen_vs_current.csv.gz')
    for nm in NM[1:]:
        x = LG[nm][1].drop(columns=['j']).copy(); x.round(6).to_csv(HERE / f'valid_trades_{nm}.csv.gz', index=False)
# ---- V3 禁買聯集（各題：現行與該題凍結格各自最賺 5 筆期間）
BLK = {}
for tp in TOP:
    b = {j: list(v) for j, v in s2lib.block_from(LG['CUR'][1], 5)[0].items()}
    for j, v in s2lib.block_from(LG[f'FZ_{tp}'][1], 5)[0].items():
        b.setdefault(j, []).extend(v)
    BLK[tp] = b
res['V3_block_union'] = {tp: {str(P.COLS[j]): [[str(P.IDX[a].date()), str(P.IDX[c].date())] for a, c in v] for j, v in BLK[tp].items()} for tp in TOP}
del LG, SG0, F0; gc.collect()

# ---- 20 條路徑
PATH = {}
for s in L.SHIFTS:
    SG = {nm: L.signal(P, U, RT, s, CELL[nm]) for nm in NM}
    FX = {nm: L.fresh_from(SG[nm]) for nm in NM}
    for cap in L.CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        navs = {}
        for nm in NM:
            nav, lg, st = L.sim(P, FX[nm], U, cap=FCAP[cap], cell=CELL[nm])
            PATH.setdefault(f'{nm}|fresh', {})[key] = dict(valid=cg(nav, 'valid'), valid_dd=stlib.maxdd(nav, TB, DE), y2022_2023=cg(nav, 'y2022_2023'), since2024=cg(nav, 'since2024'),
                                                           seg=L.seg_dd(nav), avg_stockfrac=sfm(st), n_pos=int(len(lg)))
            navs[nm] = nav; del lg, st
            for tp in TOP:
                if nm == 'CUR' or nm == f'FZ_{tp}':
                    PATH.setdefault(f'{nm}|blk_{tp}', {})[key] = dict(valid=cg(L.sim(P, FX[nm], U, cap=FCAP[cap], block=BLK[tp], cell=CELL[nm])[0], 'valid'))
            nav, lg, st = L.sim(P, SG[nm], U, cap=cap, cell=CELL[nm])
            PATH.setdefault(f'{nm}|cont', {})[key] = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE), train=stlib.cagr(nav, TA, TB),
                                                          avg_stockfrac=sfm(st, D0, DE), roll252_sr_neg=L.roll_sr_neg(nav), top5_share=L.top5_share(lg)['share_of_trade_pnl']); del nav, lg, st
        for tp in TOP:
            lam = PATH[f'FZ_{tp}|fresh'][key]['avg_stockfrac'] / PATH['CUR|fresh'][key]['avg_stockfrac']
            sc = L.scaled_current(navs['CUR'], bench, lam, TB, DE)
            PATH.setdefault(f'SCALED_{tp}|fresh', {})[key] = dict(valid=stlib.cagr(sc, TB, DE), valid_dd=stlib.maxdd(sc, TB, DE), lam=float(lam)); del sc
        del navs
    del SG, FX; gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
res['paths'] = PATH
keys = list(PATH['CUR|fresh'])


def pair(a, b, f, sub=None):
    ga = np.array([PATH[a][k][f][sub] if sub else PATH[a][k][f] for k in keys]); gb = np.array([PATH[b][k][f][sub] if sub else PATH[b][k][f] for k in keys])
    d = (ga - gb) * 100
    return dict(median=float(np.median(d)), npos=int((d > 0).sum()), min=float(d.min()), max=float(d.max()), a_median=float(np.median(ga)), b_median=float(np.median(gb)))


for tp in TOP:
    F = f'FZ_{tp}'
    expo = dict(current=cont_std['CUR']['avg_stockfrac'], frozen=cont_std[F]['avg_stockfrac']); expo['diff_pp'] = (expo['frozen'] - expo['current']) * 100
    trig = bool(abs(expo['diff_pp']) > 3.0)
    pv = dict(valid=pair(f'{F}|fresh', 'CUR|fresh', 'valid'), valid_dd=pair(f'{F}|fresh', 'CUR|fresh', 'valid_dd'), y2022_2023=pair(f'{F}|fresh', 'CUR|fresh', 'y2022_2023'),
              since2024=pair(f'{F}|fresh', 'CUR|fresh', 'since2024'), seg={sg: pair(f'{F}|fresh', 'CUR|fresh', 'seg', sg) for sg in L.SEGV},
              blk=pair(f'{F}|blk_{tp}', f'CUR|blk_{tp}', 'valid'), stockfrac=pair(f'{F}|fresh', 'CUR|fresh', 'avg_stockfrac'),
              R2_vs_scaled=pair(f'{F}|fresh', f'SCALED_{tp}|fresh', 'valid'), scaled_vs_current=pair(f'SCALED_{tp}|fresh', 'CUR|fresh', 'valid'),
              cont={f: pair(f'{F}|cont', 'CUR|cont', f) for f in ('since2019', 'full', 'maxdd', 'train', 'avg_stockfrac', 'roll252_sr_neg', 'top5_share')})
    V = dict(V1=pv['valid']['npos'] >= 15, V2=pv['valid']['median'] >= 2.0, V3=pv['blk']['median'] > 0, V4=all(pv['seg'][sg]['median'] >= -2.0 for sg in L.SEGV))
    V = {k: bool(v) for k, v in V.items()}
    R2raw = bool(pv['R2_vs_scaled']['median'] > 0 and pv['R2_vs_scaled']['npos'] >= 15)
    R2 = bool(R2raw or not trig)
    sel = TOP[tp]['selected']; T0 = bool(TOP[tp]['T0_pass'])
    s25 = bool(sel not in ('不改', 'DRY') and sel == FZC[tp] and T0 and all(V.values()))
    passed = bool(s25 and R2)
    res['topics'][tp] = dict(frozen=FZC[tp], selected=sel, T0=T0, main=dict(CUR=MAIN['CUR'], FZ=MAIN[F]), exposure_full_period_std=expo, R2_triggered=trig, pairs=pv,
                             verdict=dict(**V, S25_pass=s25, R2_triggered=trig, R2_beats_scaled_current=R2raw, R2=R2, n_pass_V=int(sum(V.values())),
                                          overall='通過（僅為候選）' if passed else '不通過'))
res['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss(), dry=DRY)
if DRY0:
    for tp in TOP:
        pv = res['topics'][tp]['pairs']
        assert all(pv[k]['min'] == 0 and pv[k]['max'] == 0 for k in ('valid', 'valid_dd', 'blk')), {k: pv[k] for k in ('valid', 'valid_dd', 'blk')}
    assert abs(np.median([PATH['CUR|cont'][k]['since2019'] for k in keys]) - L.TARGET['paths_median']) < 1e-12
    print('DRY0 OK', res['meta'])
elif DRY:
    json.dumps(res, ensure_ascii=False, default=str); print('DRY OK（未印任何數字、未寫檔）', res['meta'])
else:
    (HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    for tp in TOP:
        r = res['topics'][tp]
        print(tp, '判定', json.dumps(r['verdict'], ensure_ascii=False))
        print(json.dumps({k: v for k, v in r['pairs'].items()}, ensure_ascii=False))
    print(json.dumps(res['cont_std_2015'], ensure_ascii=False)); print(res['meta'])
