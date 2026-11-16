#!/usr/bin/env python3
"""S30 驗證期（凍結後只跑一次）：凍結規則 FZ 對現行 CUR。
(1) 判定用：重新起跑（只保留營收月份 ≥ 2021-12 的訊號；起始資金依各設定自己 2021-12-30 的淨值縮放成 60～200 萬）× 20 條路徑，驗證窗 2021-12-30 → 2026-10-22；V1～V4。
(2) 描述用：全期連續主跑與 20 條路徑；CAPM β（日、月）；逐年；四段空頭；最賺 5 筆；成本加倍；平均股票占比。
輸出 results.json、trades.csv.gz。PREREG.json、FROZEN.json 的 sha256 必須相符；results.json 已存在就拒絕重跑。python3 -I -B validate.py"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s30lib
from s30lib import s17lib, s10lib, s4lib, s2lib, stlib

H = Path(__file__).resolve().parent
pre = (H / 'PREREG.json').read_bytes(); sha = hashlib.sha256(pre).hexdigest()
assert sha == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與 PREREG.sha256 不符'
fz = (H / 'FROZEN.json').read_bytes(); fsha = hashlib.sha256(fz).hexdigest()
assert fsha == (H / 'FROZEN.sha256').read_text().split()[0], 'FROZEN.json 與 FROZEN.sha256 不符'
assert not (H / 'results.json').exists(), 'results.json 已存在：驗證期只跑一次，拒絕重跑'
FZ = json.loads(fz)
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s30lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
RT = s30lib.surge(P)
s17lib.e50_fix_suspension(P)
Hd, info, FR = s30lib.holdings(P, INNOV)
IDXS = {b: s30lib.basket_index(P, Hd[b], FR) for b in s30lib.BASKETS}
IDXS['MIX'] = s30lib.mix_index(P, IDXS['LBQ'])
CFG = {'CUR': None, 'FZ': s30lib.base_arrays(P, FZ['frozen_basket'], IDXS)}
del INNOV
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
print('載入完成', FZ['frozen_rule'], f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

bench = pd.Series(P.E50, index=P.IDX)
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE = P.IDX[P.t0], P.IDX[-1]
D19, D24, TB = lastb('2019-01-01'), lastb('2024-01-01'), lastb('2022-01-01')
WIN = {'since2019': (D19, DE), 'full': (D0, DE), 'since2024': (D24, DE), 'valid': (TB, DE), 'y2022_2023': (TB, D24), 'train': (lastb('2016-02-01'), TB)}
cg = lambda nav, k: stlib.cagr(nav, *WIN[k])
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
SEGV = ('2022', '2024-07~2025-04')
sim = lambda S, c, cap=1e6, **kw: s30lib.sim(P, S, U, B=CFG[c], cap=cap, **kw)

S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
FCAP = {}; acc = {}
for c in CFG:
    nav1, lg1, _ = sim(S0, c, 1e6, fresh=True)
    ratio = float(nav1.loc[TB] / 1e6)
    FCAP[c] = {cap: cap / ratio for cap in CAPS}
    acc[c] = dict(no_trade_before_2022=bool((lg1['ent_date'] >= pd.Timestamp('2022-01-01')).all()), ratio_nav_TB=ratio)
    assert acc[c]['no_trade_before_2022']
del nav1, lg1

res = {'PREREG_sha256': sha, 'FROZEN_sha256': fsha, 'frozen_rule': FZ['frozen_rule'], 'frozen_basket': FZ['frozen_basket'], 'T0_pass': FZ['T0_pass'], 'acceptance': acc,
       'bench_0050': {k: dict(cagr=stlib.cagr(bench, *WIN[k]), maxdd=stlib.maxdd(bench, *WIN[k])) for k in WIN}, 'main': {}}
res['bench_0050']['seg'] = s17lib.seg_dd(bench)
trades = []; mainlg = {}
for mode in ('fresh', 'cont'):
    fresh = mode == 'fresh'
    for c in CFG:
        cap = FCAP[c][1e6] if fresh else 1e6
        nav, lg, st = sim(S0, c, cap, fresh=fresh)
        nm = f'{c}|{mode}'; mainlg[nm] = lg
        a = '2022-01-01' if fresh else None
        b0 = TB if fresh else D0
        ks = ('valid', 'y2022_2023', 'since2024') if fresh else tuple(WIN)
        w = {k: dict(cagr=cg(nav, k), maxdd=stlib.maxdd(nav, *WIN[k]), cagr_0050=stlib.cagr(bench, *WIN[k])) for k in ks}
        lga = lg[lg['ent_date'] >= pd.Timestamp(a or '1900-01-01')]
        blk, top = s2lib.block_from(lga, 5)
        nb = sim(S0, c, cap, fresh=fresh, block=blk)[0]
        nx = sim(S0, c, cap, fresh=fresh, cost='strict_x2')[0]
        kk = ('valid',) if fresh else ('since2019', 'full', 'valid')
        R = dict(windows=w, maxdd=stlib.maxdd(nav, b0, DE), dd_dates=s10lib.dd_dates(nav.loc[b0:]), seg=s17lib.seg_dd(nav), yearly=stlib.yearly(nav.loc[b0:], bench),
                 capm_daily=s30lib.capm_beta(nav, bench, b0, DE), capm_monthly=s30lib.capm_beta(nav, bench, b0, DE, 'M'),
                 n_closed=int((lga['why'] != 'open_end').sum()), avg_stockfrac=float(np.nanmean(st['stockfrac'].loc[b0:].to_numpy())),
                 block5={k: cg(nb, k) for k in kk}, block5_top=[dict(code=y.code, ent=str(y.ent_date.date()), ret=float(y.ret), pnl=float(y.pnl)) for y in top.itertuples()],
                 costx2={k: cg(nx, k) for k in kk}, top5_share=s17lib.top5_share(lga, nav.loc[b0:], float(nav.loc[b0])), e50_cost=float(st['e50_cost']), e50_orders=int(st['e50_orders']))
        if not fresh:
            R['capm_daily_valid'] = s30lib.capm_beta(nav, bench, TB, DE); R['capm_daily_train'] = s30lib.capm_beta(nav, bench, *WIN['train'])
        res['main'][nm] = R
        x = lg.copy(); x['run'] = nm
        trades.append(x[['run', 'code', 'sig_date', 'ent_date', 'exit_date', 'hold_days', 'pnl', 'ret', 'why']])
        del nav, nb, nx, st
    gc.collect()
print('主跑完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

blkU = {}
for nm in ('CUR|fresh', 'FZ|fresh'):
    b, _ = s2lib.block_from(mainlg[nm], 5)
    for j, v in b.items():
        blkU.setdefault(j, []).extend(v)
res['V3_block_union'] = {str(P.COLS[j]): [[str(P.IDX[a].date()), str(P.IDX[b].date())] for a, b in v] for j, v in blkU.items()}
del mainlg

PATH = {f'{c}|{m}': {} for c in CFG for m in ('fresh', 'cont', 'fresh_blk')}
for s in range(4):
    S = S0 if s == 0 else s10lib.make_signal_s10(P, U, RT, shift=s)
    for cap in CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        for c in CFG:
            nav = sim(S, c, FCAP[c][cap], fresh=True)[0]
            sg = s17lib.seg_dd(nav)
            PATH[f'{c}|fresh'][key] = dict(valid=cg(nav, 'valid'), valid_dd=stlib.maxdd(nav, TB, DE), since2024=cg(nav, 'since2024'), seg={k: sg[k] for k in SEGV},
                                           beta=s30lib.capm_beta(nav, bench, TB, DE)['beta'])
            nav = sim(S, c, FCAP[c][cap], fresh=True, block=blkU)[0]
            PATH[f'{c}|fresh_blk'][key] = dict(valid=cg(nav, 'valid'), valid_dd=stlib.maxdd(nav, TB, DE))
            nav, lg, _ = sim(S, c, cap)
            PATH[f'{c}|cont'][key] = dict(since2019=cg(nav, 'since2019'), full=cg(nav, 'full'), valid=cg(nav, 'valid'), maxdd=stlib.maxdd(nav, D0, DE), seg=s17lib.seg_dd(nav),
                                          beta=s30lib.capm_beta(nav, bench, D0, DE)['beta'], top5=s17lib.top5_share(lg, nav, cap))
            del nav, lg
    if s:
        del S
    gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
res['paths'] = PATH
keys = list(PATH['CUR|fresh'])
med = lambda v: float(np.median(v))


def pair(a, b, f, sub=None):
    g = (lambda d: d[f]) if sub is None else (lambda d: d[f][sub])
    d = np.array([g(PATH[a][k]) - g(PATH[b][k]) for k in keys]) * 100
    return dict(npos=int((d > 0).sum()), n_ge05=int((d >= 0.5).sum()), median_pp=med(d), min_pp=float(d.min()), max_pp=float(d.max()))


summ = {}
for nm, D in PATH.items():
    summ[nm] = {f: med([D[k][f] for k in keys]) for f in D[keys[0]] if f != 'seg'}
    if 'seg' in D[keys[0]]:
        summ[nm]['seg'] = {sg: med([D[k]['seg'][sg] for k in keys]) for sg in D[keys[0]]['seg']}
res['paths_median'] = summ
pv = {'fresh': dict(valid=pair('FZ|fresh', 'CUR|fresh', 'valid'), valid_dd=pair('FZ|fresh', 'CUR|fresh', 'valid_dd'), since2024=pair('FZ|fresh', 'CUR|fresh', 'since2024'),
                    seg={sg: pair('FZ|fresh', 'CUR|fresh', 'seg', sg) for sg in SEGV}, beta=pair('FZ|fresh', 'CUR|fresh', 'beta')),
      'fresh_blk': dict(valid=pair('FZ|fresh_blk', 'CUR|fresh_blk', 'valid'), valid_dd=pair('FZ|fresh_blk', 'CUR|fresh_blk', 'valid_dd')),
      'cont': {f: pair('FZ|cont', 'CUR|cont', f) for f in ('since2019', 'full', 'valid', 'maxdd', 'beta', 'top5')}}
pv['cont']['seg'] = {sg: pair('FZ|cont', 'CUR|cont', 'seg', sg) for sg in s17lib.SEGMENTS}
res['pairs'] = pv

p = pv['fresh']
V = dict(V1=p['valid_dd']['n_ge05'] >= 15, V2=p['valid']['median_pp'] >= -2.0, V3=pv['fresh_blk']['valid']['median_pp'] >= -2.0,
         V4=all(p['seg'][sg]['median_pp'] >= 0.5 for sg in SEGV))
V = {k: bool(v) for k, v in V.items()}
passed = bool(FZ['T0_pass'] and all(V.values()))
res['verdict'] = dict(T0=FZ['T0_pass'], **V, n_pass_V=int(sum(V.values())), overall='通過' if passed else '不通過',
                      text=('凍結規則在驗證期有用' if passed else '不通過：停在描述，不換組合、不換規則重跑、不寫操作建議'))
res['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss())
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
tr = pd.concat(trades, ignore_index=True)
tr['pnl'] = tr['pnl'].round(0); tr['ret'] = tr['ret'].round(4)
for c in ('sig_date', 'ent_date', 'exit_date'):
    tr[c] = pd.to_datetime(tr[c]).dt.strftime('%Y-%m-%d')
tr.to_csv(H / 'trades.csv.gz', index=False)
print('判定', json.dumps(res['verdict'], ensure_ascii=False))
print(json.dumps(pv, ensure_ascii=False))
print(json.dumps(summ, ensure_ascii=False)[:3000])
print(res['meta'])
