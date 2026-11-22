#!/usr/bin/env python3
"""S93 驗證期（只跑一次）：凍結格 vs 現行，重新起跑（訊號日 2023-01 起；起始資金縮放）× 20 條路徑，驗證窗 2022-12-30 → 2026-10-22；V1～V4。
另（描述）：名單差異逐筆、提早賣掉逐筆、隨機對照 20 次（seed 93）、較寬鬆時點版、全部 2019-08 起交易逐筆集保特徵分組、全期連續 2019 起年化。results.json 已存在就拒絕重跑。
用法：python3 -I -B validate.py [--dry]（--dry：凍結格換成現行，只檢查流程，配對差應全為 0；不寫檔、不算任何集保條件）"""
import sys, json, time, resource, gc, hashlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
DRY = '--dry' in sys.argv
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
PSHA = None
if not DRY:
    PSHA = sha(HERE / 'PREREG.json'); assert PSHA == (HERE / 'PREREG.sha256').read_text().split()[0]
    for ln in (HERE / 'code.sha256').read_text().splitlines():
        h, f = ln.split()
        assert sha(HERE / f) == h, f'{f} 與登記時不同'
sys.path.insert(0, str(HERE))
import s93lib as L
from s93lib import stlib, s2lib
if DRY:
    FZID, T0, SELECTED = 'CUR', False, 'DRY'
else:
    assert not (HERE / 'results.json').exists(), 'results.json 已存在：驗證期只跑一次'
    assert sha(HERE / 'FROZEN.json') == (HERE / 'FROZEN.sha256').read_text().split()[0]
    FZ = json.loads((HERE / 'FROZEN.json').read_text()); assert FZ['PREREG_sha256'] == PSHA
    FZID, T0, SELECTED = FZ['frozen_for_validation'], bool(FZ['T0_pass']), FZ['selected']
t_all = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT = L.load()
TD = L.TDCC(P)
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19, TB, T25 = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01'), lastb('2023-01-01'), lastb('2025-01-01')
WIN = dict(valid=(TB, DE), y2023_2024=(TB, T25), since2025=(T25, DE))
cg = lambda nav, k: stlib.cagr(nav, *WIN[k])
med = lambda v: float(np.median(v))
SEGV = tuple(L.SEGMENTS)

S0 = L.signal(P, U, RT, 0)
ncur, lcur, _ = L.sim(P, S0, U, cap=1e6)
ratio = float(ncur.loc[TB] / 1e6)
FCAP = {cap: cap / ratio for cap in L.CAPS}
SZ0, rec0 = L.apply_cell(P, U, RT, TD, S0, FZID)
F0, FZ0 = L.fresh_from(S0), L.fresh_from(SZ0)
nC, lC, stC = L.sim(P, F0, U, cap=FCAP[1e6])
nF, lF, stF = L.sim(P, FZ0, U, cap=FCAP[1e6])
acc = dict(A3_no_entry_before_2023=bool((lC['ent_date'] >= pd.Timestamp('2023-01-01')).all() and (lF['ent_date'] >= pd.Timestamp('2023-01-01')).all()), ratio_nav_TB=ratio)
assert acc['A3_no_entry_before_2023']
res = dict(PREREG_sha256=PSHA, frozen=FZID, kind=(L.kind(FZID) if FZID != 'CUR' else None), selected=SELECTED, T0_pass=T0, acceptance=acc, main={})
for nm, nav, lg, st in (('CUR', nC, lC, stC), ('FZ', nF, lF, stF)):
    cl = lg[lg['why'] != 'open_end']
    res['main'][nm] = dict(cagr={k: cg(nav, k) for k in WIN}, maxdd=stlib.maxdd(nav, TB, DE), seg=L.seg_dd(nav), n_closed=int(len(cl)), n_all=int(len(lg)), win=float((cl['ret'] > 0).mean()),
                           avg_ret=float(cl['ret'].mean()), median_ret=float(cl['ret'].median()), avg_stockfrac=float(np.nanmean(st['stockfrac'].loc[TB:DE].to_numpy())), final_nav=float(nav.iloc[-1]))
res['main']['CUR']['costx2_valid'] = cg(L.sim(P, F0, U, cap=FCAP[1e6], cost='strict_x2')[0], 'valid')
res['main']['FZ']['costx2_valid'] = cg(L.sim(P, FZ0, U, cap=FCAP[1e6], cost='strict_x2')[0], 'valid')
if not DRY:                                                # 較寬鬆時點（描述）
    SL, _ = L.apply_cell(P, U, RT, TD, S0, FZID, lag='loose')
    res['main']['FZ_loose'] = dict(valid=cg(L.sim(P, L.fresh_from(SL), U, cap=FCAP[1e6])[0], 'valid')); del SL

# ---- 名單差異與提早賣掉逐筆（標準路徑；描述）
clC = lC[lC['why'] != 'open_end'].copy()
n5 = int(round(len(clC) * 0.05))
clC['bigwin'] = clC['ret'].rank(ascending=False, method='first') <= n5
clC['biglose'] = clC['ret'].rank(ascending=True, method='first') <= n5
bw = {(int(r.j), int(r.ent_t)) for r in clC[clC['bigwin']].itertuples()}; bl = {(int(r.j), int(r.ent_t)) for r in clC[clC['biglose']].itertuples()}
thr_w = float(clC.loc[clC['bigwin'], 'ret'].min()) if n5 else None; thr_l = float(clC.loc[clC['biglose'], 'ret'].max()) if n5 else None
mapF = {(int(r.j), int(r.ent_t)): r for r in lF.itertuples()}; keyC = {(int(r.j), int(r.ent_t)) for r in lC.itertuples()}
rm_by_k = {r['k']: set(r['removed']) for r in rec0}; ad_by_k = {r['k']: set(r['added']) for r in rec0}
row = lambda r, extra: dict(code=str(r.code), sig_date=str(r.sig_date.date()), ent_date=str(r.ent_date.date()), exit_date=str(r.exit_date.date()), why=str(r.why), ret=float(r.ret), pnl=float(r.pnl), **extra)
tag = lambda r: dict(bigwin=(int(r.j), int(r.ent_t)) in bw, biglose=(int(r.j), int(r.ent_t)) in bl)
only_cur = [row(r, dict(tag(r), removed_by_rule=str(r.code) in rm_by_k.get(int(r.sig_t), set()))) for r in lC.itertuples() if (int(r.j), int(r.ent_t)) not in mapF]
only_fz = [row(r, dict(added_by_rule=str(r.code) in ad_by_k.get(int(r.sig_t), set()), would_be_bigwin=bool(thr_w is not None and r.ret >= thr_w), would_be_biglose=bool(thr_l is not None and r.ret <= thr_l)))
           for r in lF.itertuples() if (int(r.j), int(r.ent_t)) not in keyC]
early = []
for r in lC.itertuples():
    q = mapF.get((int(r.j), int(r.ent_t)))
    if q is not None and int(q.exit_t) != int(r.exit_t):
        early.append(row(r, dict(tag(r), fz_exit_date=str(q.exit_date.date()), fz_ret=float(q.ret), fz_pnl=float(q.pnl), fz_why=str(q.why), pnl_diff=float(q.pnl - r.pnl), cur_open_end=bool(r.why == 'open_end'))))
sm = lambda rows, f='pnl': dict(n=len(rows), pnl_sum=float(sum(x[f] for x in rows)), mean_ret=float(np.mean([x['ret'] for x in rows])) if rows else None, win=float(np.mean([x['ret'] > 0 for x in rows])) if rows else None)
recv = [r for r in rec0 if pd.Period(r['month'], 'M') >= L.VALID_FIRST_MONTH]
bwall = lambda df: [dict(code=str(r.code), ent_date=str(r.ent_date.date()), exit_date=str(r.exit_date.date()), ret=float(r.ret), pnl=float(r.pnl), in_frozen=(int(r.j), int(r.ent_t)) in mapF,
                         fz_exit_date=(str(mapF[(int(r.j), int(r.ent_t))].exit_date.date()) if (int(r.j), int(r.ent_t)) in mapF else None),
                         fz_ret=(float(mapF[(int(r.j), int(r.ent_t))].ret) if (int(r.j), int(r.ent_t)) in mapF else None)) for r in df.itertuples()]
res['list_diff'] = dict(n5=n5, thr_bigwin_ret=thr_w, thr_biglose_ret=thr_l, n_cur_closed=int(len(clC)),
                        list_changes_valid=dict(n_months=len(recv), n_orig=sum(len(r['orig']) for r in recv), n_removed=sum(len(r['removed']) for r in recv), n_added=sum(len(r['added']) for r in recv), n_xflag_top20=sum(len(r['xflag']) for r in recv)),
                        only_cur=sm(only_cur), only_fz=sm(only_fz), early_exit=dict(sm(early), pnl_diff_sum=float(sum(x['pnl_diff'] for x in early))),
                        only_cur_bigwin=[x for x in only_cur if x['bigwin']], only_cur_biglose=[x for x in only_cur if x['biglose']],
                        early_bigwin=[x for x in early if x['bigwin']], early_biglose=[x for x in early if x['biglose']], early_all=early,
                        only_fz_bigwin=[x for x in only_fz if x['would_be_bigwin']], only_fz_biglose=[x for x in only_fz if x['would_be_biglose']],
                        cur_bigwin_all=bwall(clC[clC['bigwin']].sort_values('ret', ascending=False)), cur_biglose_all=bwall(clC[clC['biglose']].sort_values('ret')),
                        nav_diff_main=float(nF.iloc[-1] - nC.iloc[-1]), nav_start=float(FCAP[1e6]))
if not DRY:
    pd.DataFrame([dict(side='只有現行', **x) for x in only_cur] + [dict(side='只有凍結格', **x) for x in only_fz] + [dict(side='凍結格換了出場日', **x) for x in early]).to_csv(HERE / 'valid_list_diff.csv', index=False)

# ---- 隨機對照（20 次，seed 93）
if not DRY:
    rng = np.random.default_rng(93); draws = []
    for _ in range(20):
        Sr, _r = L.apply_cell(P, U, RT, TD, S0, FZID, rng=rng)
        draws.append(cg(L.sim(P, L.fresh_from(Sr), U, cap=FCAP[1e6])[0], 'valid')); del Sr, _r
    v = res['main']['FZ']['cagr']['valid']
    res['random_control'] = dict(note='進場格→每月隨機剔除同檔數（F 照樣遞補、不套條件）；出場格→每月從季線上 SURGE 前 20 名隨機標同檔數出場。20 次、seed 93；重新起跑標準路徑驗證窗年化',
                                 cand_valid=v, cur_valid=res['main']['CUR']['cagr']['valid'], random_median=med(draws), random_min=float(min(draws)), random_max=float(max(draws)),
                                 n_random_below_cand=int(sum(d < v for d in draws)), n_random_above_cur=int(sum(d > res['main']['CUR']['cagr']['valid'] for d in draws)), draws=draws)

# ---- V3 禁買聯集
BLK = {j: list(v) for j, v in s2lib.block_from(lC, 5)[0].items()}
for j, v in s2lib.block_from(lF, 5)[0].items():
    BLK.setdefault(j, []).extend(v)
res['V3_block_union'] = {str(P.COLS[j]): [[str(P.IDX[a].date()), str(P.IDX[b].date())] for a, b in v] for j, v in BLK.items()}

# ---- 2019-08 起全部交易逐筆集保特徵分組（描述；現行全期連續標準路徑）
if not DRY:
    x = L.trade_features(TD, lcur[lcur['sig_date'] >= pd.Timestamp('2019-08-01')])
    x['period'] = np.where(x['sig_date'] <= TB, '訓練段', '驗證段')
    c = x[x['why'] != 'open_end'].copy()
    G = {}
    for per, y in c.groupby('period'):
        y = y.copy(); k5 = int(round(len(y) * 0.05))
        y['bigwin'] = y['ret'].rank(ascending=False, method='first') <= k5; y['biglose'] = y['ret'].rank(ascending=True, method='first') <= k5
        y['g_dB4'] = pd.cut(y['dB4'], [-np.inf, -1.5, 0, 1.5, np.inf], labels=['≤−1.5', '−1.5~0', '0~+1.5', '>+1.5']).astype(str)
        y['g_dB8'] = pd.cut(y['dB8'], [-np.inf, -2.5, 0, 2.5, np.inf], labels=['≤−2.5', '−2.5~0', '0~+2.5', '>+2.5']).astype(str)
        y['g_dH4'] = pd.cut(y['dH4'], [-np.inf, -0.05, 0, 0.10, np.inf], labels=['≤−5%', '−5~0%', '0~10%', '≥10%']).astype(str)
        y['g_dR4'] = pd.cut(y['dR4'], [-np.inf, -2, 0, 2, np.inf], labels=['≤−2', '−2~0', '0~+2', '≥+2']).astype(str)
        y['g_conc4'] = np.where((y['dB4'] > 0) & (y['dH4'] < 0), '集中', np.where((y['dB4'] < 0) & (y['dH4'] > 0), '發散', '其他'))
        G[per] = dict(n=int(len(y)), n5=k5, groups={lab: {str(g): dict(n=int(len(z)), win=float((z['ret'] > 0).mean()), mean=float(z['ret'].mean()), median=float(z['ret'].median()), bigwin=int(z['bigwin'].sum()), biglose=int(z['biglose'].sum()))
                                                          for g, z in y.groupby(lab)} for lab in ('g_dB4', 'g_dB8', 'g_dH4', 'g_dR4', 'g_conc4')},
                      rank_corr_ret={f: float(y[[f, 'ret']].corr(method='spearman').iloc[0, 1]) for f in ('dB4', 'dB8', 'dH4', 'dH8', 'dR4', 'dR8', 'ndec')},
                      bigwin_list=[dict(code=str(r.code), sig=str(r.sig_date.date()), ret=float(r.ret), dB4=float(r.dB4), dB8=float(r.dB8), dH4=float(r.dH4), dR4=float(r.dR4)) for r in y[y['bigwin']].sort_values('ret', ascending=False).itertuples()],
                      biglose_list=[dict(code=str(r.code), sig=str(r.sig_date.date()), ret=float(r.ret), dB4=float(r.dB4), dB8=float(r.dB8), dH4=float(r.dH4), dR4=float(r.dR4)) for r in y[y['biglose']].sort_values('ret').itertuples()])
    res['describe'] = G
    x.drop(columns=['j']).round(6).to_csv(HERE / 'trades_features.csv.gz', index=False)
    del x, c
del S0, F0, FZ0, SZ0; gc.collect()

# ---- 20 條路徑
PATH = {k: {} for k in ('CUR|fresh', 'FZ|fresh', 'CUR|blk', 'FZ|blk', 'CUR|cont', 'FZ|cont')}
for s in L.SHIFTS:
    S = L.signal(P, U, RT, s); TD._c.clear()
    SZ, _r = L.apply_cell(P, U, RT, TD, S, FZID)
    for cap in L.CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        for nm, Sx in (('CUR', S), ('FZ', SZ)):
            Fx = L.fresh_from(Sx)
            nav = L.sim(P, Fx, U, cap=FCAP[cap])[0]
            PATH[f'{nm}|fresh'][key] = dict(valid=cg(nav, 'valid'), valid_dd=stlib.maxdd(nav, TB, DE), y2023_2024=cg(nav, 'y2023_2024'), since2025=cg(nav, 'since2025'), seg=L.seg_dd(nav))
            PATH[f'{nm}|blk'][key] = dict(valid=cg(L.sim(P, Fx, U, cap=FCAP[cap], block=BLK)[0], 'valid'))
            nav = L.sim(P, Sx, U, cap=cap)[0]
            PATH[f'{nm}|cont'][key] = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE))
            del Fx
    del S, SZ, _r; gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
res['paths'] = PATH
keys = list(PATH['CUR|fresh'])


def pair(a, b, f, sub=None):
    ga = np.array([PATH[a][k][f][sub] if sub else PATH[a][k][f] for k in keys]); gb = np.array([PATH[b][k][f][sub] if sub else PATH[b][k][f] for k in keys])
    d = (ga - gb) * 100
    return dict(median=float(np.median(d)), npos=int((d > 0).sum()), min=float(d.min()), max=float(d.max()), a_median=float(np.median(ga)), b_median=float(np.median(gb)))


pv = dict(valid=pair('FZ|fresh', 'CUR|fresh', 'valid'), valid_dd=pair('FZ|fresh', 'CUR|fresh', 'valid_dd'), y2023_2024=pair('FZ|fresh', 'CUR|fresh', 'y2023_2024'), since2025=pair('FZ|fresh', 'CUR|fresh', 'since2025'),
          seg={sg: pair('FZ|fresh', 'CUR|fresh', 'seg', sg) for sg in SEGV}, blk=pair('FZ|blk', 'CUR|blk', 'valid'),
          cont={f: pair('FZ|cont', 'CUR|cont', f) for f in ('since2019', 'full', 'maxdd')})
res['pairs'] = pv
V = dict(V1=pv['valid']['npos'] >= 15, V2=pv['valid']['median'] >= 2.0, V3=pv['blk']['median'] > 0, V4=all(pv['seg'][sg]['median'] >= -2.0 for sg in SEGV))
V = {k: bool(v) for k, v in V.items()}
passed = bool(SELECTED not in ('不改', 'DRY') and SELECTED == FZID and T0 and all(V.values()))
res['verdict'] = dict(selected=SELECTED, frozen=FZID, T0=T0, **V, n_pass_V=int(sum(V.values())), overall='通過' if passed else '不通過')
res['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss(), dry=DRY)
if DRY:
    assert all(abs(pv[k]['median']) == 0 and pv[k]['min'] == 0 and pv[k]['max'] == 0 for k in ('valid', 'valid_dd', 'blk')) and not res['list_diff']['only_cur'].get('n') and not res['list_diff']['only_fz'].get('n') and not early
    assert abs(pv['cont']['since2019']['b_median'] - L.TARGET['paths_median']) < 1e-12, pv['cont']['since2019']
    print('DRY OK', res['main']['CUR'], res['meta'])
else:
    (HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print('判定', json.dumps(res['verdict'], ensure_ascii=False))
    print(json.dumps({k: pv[k] for k in ('valid', 'valid_dd', 'y2023_2024', 'since2025', 'seg', 'blk', 'cont')}, ensure_ascii=False))
    print(json.dumps(res['main'], ensure_ascii=False)); print(json.dumps({k: v for k, v in res.get('random_control', {}).items() if k != 'draws'}, ensure_ascii=False)); print(res['meta'])
