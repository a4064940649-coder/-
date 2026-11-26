#!/usr/bin/env python3
"""S99 驗證期（只跑一次）：凍結格 vs 現行，重新起跑（訊號日 2022-01 起；起始資金按現行 2021-12-30 淨值縮放）× 20 條路徑，驗證窗 2021-12-30 → 2026-10-22；S25 的 V1～V4。
另：隨機對照（同樣的月中日期、同樣檔數、隨機挑；兩種池 × 20 次，seed 99）、月中交易逐筆清單、曝險（持股比例、檔數）、成本加倍、全期連續 2019 起年化（描述）。
results.json 已存在就拒絕重跑。用法：python3 -I -B validate.py [--dry]（--dry：凍結格換成現行，只檢查流程，配對差應全為 0；不寫檔）"""
import sys, json, time, resource, gc, hashlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
DRY = '--dry' in sys.argv or '--dry2' in sys.argv
DRY2 = '--dry2' in sys.argv                                 # 用「安慰劑」（每月訊號日後第 5 日隨機加 1 檔）當凍結格，走完全部程式路徑；不寫檔、不印任何數字
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
PSHA = None
if not DRY:
    PSHA = sha(HERE / 'PREREG.json'); assert PSHA == (HERE / 'PREREG.sha256').read_text().split()[0]
    for ln in (HERE / 'code.sha256').read_text().splitlines():
        h, f = ln.split()
        assert sha(HERE / f) == h, f'{f} 與登記時不同'
sys.path.insert(0, str(HERE))
import s99lib as L
import s99cells as CE
from s99lib import stlib, s2lib
if DRY:
    FZID, T0, SELECTED = 'CUR', False, 'DRY'
else:
    assert not (HERE / 'results.json').exists(), 'results.json 已存在：驗證期只跑一次'
    assert sha(HERE / 'FROZEN.json') == (HERE / 'FROZEN.sha256').read_text().split()[0]
    FZ = json.loads((HERE / 'FROZEN.json').read_text()); assert FZ['PREREG_sha256'] == PSHA
    FZID, T0, SELECTED = FZ['frozen_for_validation'], bool(FZ['T0_pass']), FZ['selected']
t_all = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT = L.load()
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19, TB, T23 = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01'), lastb('2022-01-01'), lastb('2024-01-01')
WIN = dict(valid=(TB, DE), y2022_2023=(TB, T23), since2024=(T23, DE))
cg = lambda nav, k: stlib.cagr(nav, *WIN[k])
med = lambda v: float(np.median(v))
bench = pd.Series(P.E50, index=P.IDX)

S0 = L.signal(P, U, RT, 0)
ncur, lcur, _ = L.sim(P, S0, U, cap=1e6)
ratio = float(ncur.loc[TB] / 1e6)
FCAP = {cap: cap / ratio for cap in L.CAPS}
def make(S, cell):
    if cell == 'PLACEBO':
        base = L.apply_cell(P, U, RT, S, 'CUR')[0]; ks = [x['k'] for x in S['info']]
        return CE.random_overlay(P, U, base, {k + 5: 1 for k in ks if k + 6 < P.T}, {k + 5: x['month'] for k, x in zip(ks, S['info'])}, np.random.default_rng(1), pool='ALLMA'), []
    return L.apply_cell(P, U, RT, S, cell)


if DRY2:
    FZID = 'PLACEBO'
SZ0, rec0 = make(S0, FZID)
F0 = L.fresh_from(L.apply_cell(P, U, RT, S0, 'CUR')[0]); FZ0 = L.fresh_from(SZ0)
nC, lC, stC = L.sim(P, F0, U, cap=FCAP[1e6])
nF, lF, stF = L.sim(P, FZ0, U, cap=FCAP[1e6])
acc = dict(A3_no_entry_before_2022=bool((lC['ent_date'] >= pd.Timestamp('2022-01-01')).all() and (lF['ent_date'] >= pd.Timestamp('2022-01-01')).all()), ratio_nav_TB=ratio)
assert acc['A3_no_entry_before_2022']
res = dict(PREREG_sha256=PSHA, frozen=FZID, selected=SELECTED, T0_pass=T0, acceptance=acc, main={}, e50_valid=stlib.cagr(bench, TB, DE))


def npos_mean(lg):
    d = np.zeros(P.T + 1)
    for e, x, w in zip(lg['ent_t'].to_numpy(), lg['exit_t'].to_numpy(), lg['why'].to_numpy()):
        d[e] += 1; d[(x + 1) if w == 'open_end' else x] -= 1
    return float(pd.Series(np.cumsum(d)[:P.T], index=P.IDX).loc[TB:DE].iloc[1:].mean())


def block_stats(lg, mid_rows):
    cl = lg[lg['why'] != 'open_end']; mid = cl[cl['sig_t'].isin(list(mid_rows))]; reg = cl[~cl['sig_t'].isin(list(mid_rows))]
    f = lambda y: dict(n=int(len(y)), win=float((y['ret'] > 0).mean()) if len(y) else None, mean_ret=float(y['ret'].mean()) if len(y) else None,
                       median_ret=float(y['ret'].median()) if len(y) else None, pnl_sum=float(y['pnl'].sum()), avg_hold_days=float(y['hold_days'].mean()) if len(y) else None)
    return dict(all=f(cl), regular=f(reg), mid=f(mid))


for nm, nav, lg, st, Sx in (('CUR', nC, lC, stC, F0), ('FZ', nF, lF, stF, FZ0)):
    cl = lg[lg['why'] != 'open_end']
    top5 = cl.sort_values('pnl', ascending=False).head(5)
    r = cl['ret'].to_numpy(); pn = cl['pnl'].to_numpy(); n5 = int(np.ceil(len(cl) * 0.05)); o5 = np.argsort(-r, kind='stable')[:n5]
    res['main'][nm] = dict(cagr={k: cg(nav, k) for k in WIN}, maxdd=stlib.maxdd(nav, TB, DE), seg=L.seg_dd(nav), n_closed=int(len(cl)), n_all=int(len(lg)), trades=block_stats(lg, Sx['mid_rows']),
                           avg_stockfrac=float(np.nanmean(st['stockfrac'].loc[TB:DE].to_numpy())), avg_npos=npos_mean(lg), final_nav=float(nav.iloc[-1]),
                           top5_pnl_share=float(top5['pnl'].sum() / cl['pnl'].sum()), top5=[str(c) for c in top5['code']],
                           top5pct_by_ret=dict(n=n5, pnl_share=float(pn[o5].sum() / pn.sum()), rest_mean_ret=float(np.delete(r, o5).mean()), rest_pnl_sum=float(np.delete(pn, o5).sum())),
                           buy_val=float(st['buy_val']), noslot=int(st['noslot']), blk_limup=int(st['blk_limup']), blk_min=int(st['blk_min']), blk_lot=int(st['blk_lot']), blk_nocash=int(st['blk_nocash']))
res['main']['CUR']['costx2_valid'] = cg(L.sim(P, F0, U, cap=FCAP[1e6], cost='strict_x2')[0], 'valid')
res['main']['FZ']['costx2_valid'] = cg(L.sim(P, FZ0, U, cap=FCAP[1e6], cost='strict_x2')[0], 'valid')
yr = lambda nav: {int(y): float(v) for y, v in (nav.loc[TB:DE].groupby(nav.loc[TB:DE].index.year).last().pct_change().dropna()).items()}
ny = nav = None
res['yearly_valid'] = dict(CUR=yr(nC), FZ=yr(nF), e0050=yr(bench))

# ---- 月中交易逐筆（標準路徑；描述）
if not DRY or DRY2:
    x = lF.copy(); x['mid'] = x['sig_t'].isin(list(FZ0['mid_rows']))
    keyC = {(int(r.j), int(r.ent_t)) for r in lC.itertuples()}
    x['also_in_current'] = [(int(r.j), int(r.ent_t)) in keyC for r in x.itertuples()]
    if not DRY:
        x.drop(columns=['j']).round(6).to_csv(HERE / 'valid_trades_frozen.csv.gz', index=False)
    keyF = {(int(r.j), int(r.ent_t)) for r in lF.itertuples()}
    lost = lC[[(int(r.j), int(r.ent_t)) not in keyF for r in lC.itertuples()]]
    res['list_diff'] = dict(only_frozen_n=int((~x['also_in_current']).sum()), only_current_n=int(len(lost)), only_current_pnl=float(lost['pnl'].sum()),
                            only_current=[dict(code=str(r.code), ent_date=str(r.ent_date.date()), ret=float(r.ret), pnl=float(r.pnl)) for r in lost.itertuples()],
                            mid_top10_by_pnl=[dict(code=str(r.code), ent_date=str(r.ent_date.date()), exit_date=str(r.exit_date.date()), ret=float(r.ret), pnl=float(r.pnl), why=str(r.why))
                                              for r in x[x['mid']].sort_values('pnl', ascending=False).head(10).itertuples()],
                            mid_bottom10_by_pnl=[dict(code=str(r.code), ent_date=str(r.ent_date.date()), exit_date=str(r.exit_date.date()), ret=float(r.ret), pnl=float(r.pnl), why=str(r.why))
                                                 for r in x[x['mid']].sort_values('pnl').head(10).itertuples()])
    if not DRY:
        pd.DataFrame({'CUR': nC.loc[TB:], 'FZ': nF.loc[TB:], '0050': bench.loc[TB:] / bench.loc[TB] * FCAP[1e6]}).round(2).to_csv(HERE / 'nav_valid_frozen_vs_current.csv.gz')

# ---- 隨機對照（標準路徑、重新起跑；同樣的月中列、同樣檔數；兩種池 × 20 次，seed 99）
if not DRY or DRY2:
    counts = CE.mid_buy_counts(lF, FZ0['mid_rows'])
    held = {r: [int(j) for j, e, xx in zip(lC['j'].to_numpy(), lC['ent_t'].to_numpy(), lC['exit_t'].to_numpy()) if e <= r and xx >= r + 1] for r in counts}
    rng = np.random.default_rng(99); RC = {}
    vF = res['main']['FZ']['cagr']['valid']; vC = res['main']['CUR']['cagr']['valid']
    for pool in ('MA_MOM', 'ALLMA'):
        dr = []; nb = []; sf = []; mr_ret = []
        for _ in range(20):
            Sr = CE.random_overlay(P, U, F0, counts, FZ0['mid_rows'], rng, pool=pool, avoid=held)
            nav, lg, st = L.sim(P, Sr, U, cap=FCAP[1e6])
            dr.append(cg(nav, 'valid')); m = lg[lg['sig_t'].isin(list(Sr['mid_rows']))]; nb.append(int(len(m))); sf.append(float(np.nanmean(st['stockfrac'].loc[TB:DE].to_numpy())))
            mr_ret.extend(m.loc[m['why'] != 'open_end', 'ret'].tolist()); del Sr, nav, lg, st
        RC[pool] = dict(draws=dr, median=med(dr), min=float(min(dr)), max=float(max(dr)), n_random_below_frozen=int(sum(d < vF for d in dr)), n_random_above_current=int(sum(d > vC for d in dr)),
                        mid_buys_median=med(nb), avg_stockfrac_median=med(sf), random_mid_trades_mean_ret=float(np.mean(mr_ret)), random_mid_trades_median_ret=float(np.median(mr_ret)),
                        random_mid_trades_win=float(np.mean(np.array(mr_ret) > 0)))
    res['random_control'] = dict(note='在凍結格實際有月中買進的列，各加同檔數的隨機股票（不放回；避開現行當時已持有者）；其餘同現行。MA_MOM＝股池內、季線上、近 20 日漲幅 > 10%（不看 SURGE）；ALLMA＝股池內、季線上。',
                                 frozen_valid=vF, current_valid=vC, frozen_mid_buys=int(sum(counts.values())), frozen_avg_stockfrac=res['main']['FZ']['avg_stockfrac'], **RC,
                                 R1_frozen_beats_ge18_of_20_MA_MOM=bool(RC['MA_MOM']['n_random_below_frozen'] >= 18))

# ---- V3 禁買聯集
BLK = {j: list(v) for j, v in s2lib.block_from(lC, 5)[0].items()}
for j, v in s2lib.block_from(lF, 5)[0].items():
    BLK.setdefault(j, []).extend(v)
res['V3_block_union'] = {str(P.COLS[j]): [[str(P.IDX[a].date()), str(P.IDX[b].date())] for a, b in v] for j, v in BLK.items()}
del S0, F0, FZ0, SZ0; gc.collect()

# ---- 20 條路徑（一次一格）
PATH = {k: {} for k in ('CUR|fresh', 'FZ|fresh', 'CUR|blk', 'FZ|blk', 'CUR|cont', 'FZ|cont')}
for s in L.SHIFTS:
    S = L.signal(P, U, RT, s)
    for nm, cell in (('CUR', 'CUR'), ('FZ', FZID)):
        Sx, _r = make(S, cell); Fx = L.fresh_from(Sx)
        for cap in L.CAPS:
            key = f'{int(cap/1e4)}萬|+{s}'
            nav, lg, st = L.sim(P, Fx, U, cap=FCAP[cap])
            PATH[f'{nm}|fresh'][key] = dict(valid=cg(nav, 'valid'), valid_dd=stlib.maxdd(nav, TB, DE), y2022_2023=cg(nav, 'y2022_2023'), since2024=cg(nav, 'since2024'), seg=L.seg_dd(nav),
                                            avg_stockfrac=float(np.nanmean(st['stockfrac'].loc[TB:DE].to_numpy())), n_mid=int(lg['sig_t'].isin(list(Fx['mid_rows'])).sum()))
            del nav, lg, st
            PATH[f'{nm}|blk'][key] = dict(valid=cg(L.sim(P, Fx, U, cap=FCAP[cap], block=BLK)[0], 'valid'))
            nav = L.sim(P, Sx, U, cap=cap)[0]
            PATH[f'{nm}|cont'][key] = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE)); del nav
        del Sx, Fx, _r; gc.collect()
    del S; gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
res['paths'] = PATH
keys = list(PATH['CUR|fresh'])


def pair(a, b, f, sub=None):
    ga = np.array([PATH[a][k][f][sub] if sub else PATH[a][k][f] for k in keys]); gb = np.array([PATH[b][k][f][sub] if sub else PATH[b][k][f] for k in keys])
    d = (ga - gb) * 100
    return dict(median=float(np.median(d)), npos=int((d > 0).sum()), min=float(d.min()), max=float(d.max()), a_median=float(np.median(ga)), b_median=float(np.median(gb)))


pv = dict(valid=pair('FZ|fresh', 'CUR|fresh', 'valid'), valid_dd=pair('FZ|fresh', 'CUR|fresh', 'valid_dd'), y2022_2023=pair('FZ|fresh', 'CUR|fresh', 'y2022_2023'), since2024=pair('FZ|fresh', 'CUR|fresh', 'since2024'),
          seg={sg: pair('FZ|fresh', 'CUR|fresh', 'seg', sg) for sg in L.SEGV}, blk=pair('FZ|blk', 'CUR|blk', 'valid'), stockfrac=pair('FZ|fresh', 'CUR|fresh', 'avg_stockfrac'),
          cont={f: pair('FZ|cont', 'CUR|cont', f) for f in ('since2019', 'full', 'maxdd')})
res['pairs'] = pv
V = dict(V1=pv['valid']['npos'] >= 15, V2=pv['valid']['median'] >= 2.0, V3=pv['blk']['median'] > 0, V4=all(pv['seg'][sg]['median'] >= -2.0 for sg in L.SEGV))
V = {k: bool(v) for k, v in V.items()}
R1 = bool(res.get('random_control', {}).get('R1_frozen_beats_ge18_of_20_MA_MOM', False))
passed = bool(SELECTED not in ('不改', 'DRY') and SELECTED == FZID and T0 and all(V.values()) and R1)
res['verdict'] = dict(selected=SELECTED, frozen=FZID, T0=T0, **V, R1_random_control=R1, n_pass_V=int(sum(V.values())), overall='通過（僅為候選）' if passed else '不通過')
res['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss(), dry=DRY)
if DRY2:
    json.dumps(res, ensure_ascii=False, default=str); print('DRY2 OK（未印任何數字、未寫檔）', res['meta'])
elif DRY:
    assert all(abs(pv[k]['median']) == 0 and pv[k]['min'] == 0 and pv[k]['max'] == 0 for k in ('valid', 'valid_dd', 'blk'))
    assert abs(pv['cont']['since2019']['b_median'] - L.TARGET['paths_median']) < 1e-12, pv['cont']['since2019']
    print('DRY OK', res['main']['CUR']['n_closed'], res['meta'])
else:
    (HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print('判定', json.dumps(res['verdict'], ensure_ascii=False))
    print(json.dumps({k: pv[k] for k in ('valid', 'valid_dd', 'y2022_2023', 'since2024', 'seg', 'blk', 'stockfrac', 'cont')}, ensure_ascii=False))
    print(json.dumps({k: {kk: vv for kk, vv in v.items()} for k, v in res['main'].items()}, ensure_ascii=False))
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != 'draws'} if isinstance(v, dict) else v) for k, v in res.get('random_control', {}).items()}, ensure_ascii=False)); print(res['meta'])
