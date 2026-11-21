#!/usr/bin/env python3
"""S89 驗證期（只跑一次）：凍結格 vs 現行，重新起跑（訊號日 2022-01 起；起始資金縮放）× 20 條路徑，驗證窗 2021-12-30 → 2026-10-22；V1～V4。
另（描述）：標準路徑逐筆觸發清單（誤殺／救到／金額相抵）、全期連續 2019 起年化、成本加倍。results.json 已存在就拒絕重跑。
用法：python3 -I -B validate.py [--dry]（--dry：凍結格換成現行，只檢查程式流程，配對差應全為 0；不寫 results.json）"""
import sys, json, time, resource, gc, hashlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
DRY = '--dry' in sys.argv
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
PSHA = sha(HERE / 'PREREG.json') if (HERE / 'PREREG.json').exists() else None
if not DRY:                                                 # --dry 可在登記前跑（只跑現行）
    assert PSHA == (HERE / 'PREREG.sha256').read_text().split()[0]
    for ln in (HERE / 'code.sha256').read_text().splitlines():
        h, f = ln.split()
        assert sha(HERE / f) == h, f'{f} 與登記時不同'
sys.path.insert(0, str(HERE))
import s89lib as L
from s89lib import stlib, s2lib
if DRY:
    FZID, T0, SELECTED = None, False, 'DRY'
else:
    assert not (HERE / 'out' / 'results.json').exists(), 'results.json 已存在：驗證期只跑一次'
    assert sha(HERE / 'FROZEN.json') == (HERE / 'FROZEN.sha256').read_text().split()[0]
    FZ = json.loads((HERE / 'FROZEN.json').read_text()); assert FZ['PREREG_sha256'] == PSHA
    FZID, T0, SELECTED = FZ['frozen_for_validation'], bool(FZ['T0_pass']), FZ['selected']
t_all = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT = L.load()
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19, TB, T23 = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01'), lastb('2022-01-01'), lastb('2024-01-01')
WIN = dict(valid=(TB, DE), y2022_2023=(TB, T23), since2024=(T23, DE))
cg = lambda nav, k: stlib.cagr(nav, *WIN[k])
CELL = L.CELLS[FZID] if FZID else None

S0 = L.signal(P, U, RT, 0)
ncur, _, _ = L.sim(P, S0, U, cap=1e6)
ratio = float(ncur.loc[TB] / 1e6)
FCAP = {cap: cap / ratio for cap in L.CAPS}
F0 = L.fresh_from(S0)
nC, lC, stC = L.sim(P, F0, U, cap=FCAP[1e6])
nF, lF, stF = L.sim(P, F0, U, cell=FZID, cap=FCAP[1e6])
acc = dict(A3_no_entry_before_2022=bool((lC['ent_date'] >= pd.Timestamp('2022-01-01')).all() and (lF['ent_date'] >= pd.Timestamp('2022-01-01')).all()), ratio_nav_TB=ratio)
assert acc['A3_no_entry_before_2022']
res = dict(PREREG_sha256=PSHA, frozen=FZID, frozen_cell=CELL, selected=SELECTED, T0_pass=T0, acceptance=acc, main={}, src_sha=L.SRC_SHA)
for nm, nav, lg, st in (('CUR', nC, lC, stC), ('FZ', nF, lF, stF)):
    cl = lg[lg['why'] != 'open_end']
    res['main'][nm] = dict(cagr={k: cg(nav, k) for k in WIN}, maxdd=stlib.maxdd(nav, TB, DE), seg=L.seg_dd(nav, L.SEGV), n_closed=int(len(cl)), n_all=int(len(lg)), win=float((cl['ret'] > 0).mean()),
                           avg_ret=float(cl['ret'].mean()), n_tstop_exit=int((cl['why'] == 'tstop').sum()), n_trig=len(st.get('ts_events', [])), half_sold=int(st.get('half_sold', 0)), half_skip_lot=int(st.get('half_skip_lot', 0)),
                           avg_stockfrac=float(np.nanmean(st['stockfrac'].loc[TB:DE].to_numpy())), cost_total=float(st['fee'] + st['tax'] + st['slip'] + st['e50_cost']), final_nav=float(nav.iloc[-1]))
res['main']['CUR']['costx2_valid'] = cg(L.sim(P, F0, U, cap=FCAP[1e6], cost='strict_x2')[0], 'valid')
res['main']['FZ']['costx2_valid'] = cg(L.sim(P, F0, U, cell=FZID, cap=FCAP[1e6], cost='strict_x2')[0], 'valid')

# ---- 逐筆觸發清單（標準路徑；描述）
clC = lC[lC['why'] != 'open_end'].copy()
n5 = int(round(len(clC) * 0.05))
clC['bigwin'] = clC['ret'].rank(ascending=False, method='first') <= n5
clC['biglose'] = clC['ret'].rank(ascending=True, method='first') <= n5
thr_w, thr_l = float(clC.loc[clC['bigwin'], 'ret'].min()), float(clC.loc[clC['biglose'], 'ret'].max())
keyC = {(int(r.j), int(r.ent_t)): r for r in lC.itertuples()}
bw = {(int(r.j), int(r.ent_t)) for r in clC[clC['bigwin']].itertuples()}; bl = {(int(r.j), int(r.ent_t)) for r in clC[clC['biglose']].itertuples()}
keyF = {(int(r.j), int(r.ent_t)): r for r in lF.itertuples()}
fills = {(j, e): (t, fr) for t, j, e, fr, _ in stF.get('ts_fills', [])}
EV = []
for t, j, e, g, half in stF.get('ts_events', []):
    rF = keyF[(j, e)]; rC = keyC.get((j, e))
    if half:
        f = fills.get((j, e)); ds, frac = (f[0], f[1]) if f else (None, 0.0)
    else:
        ds, frac = (int(rF.exit_t), 1.0) if rF.why == 'tstop' else (None, 0.0)
    dc, g_cur = L.current_exit(P, F0, j, e)
    if ds is None or (dc is not None and ds >= dc):
        g_stop, d_stock, d_e50 = g_cur, 0.0, 0.0
    else:
        g_stop = L.gross_at_sell(P, j, e, ds)
        e_end = float(P.E50O[dc]) if dc is not None else float(P.E50[P.T - 1])
        d_stock = float(rF.cost) * frac * (g_stop - g_cur); d_e50 = float(rF.cost) * frac * (1 + g_stop) * (e_end / float(P.E50O[ds]) - 1)
    is_w = ((j, e) in bw) if rC is not None else bool(g_cur >= thr_w)
    is_l = ((j, e) in bl) if rC is not None else bool(g_cur <= thr_l)
    EV.append(dict(code=str(P.COLS[j]), ent_date=str(P.IDX[e].date()), ent_month=str(P.IDX[e].date())[:7], trig_date=str(P.IDX[t].date()), r_trig=float(g - 1), sell_date=str(P.IDX[ds].date()) if ds is not None else None,
                   frac_sold=float(frac), g_stop=float(g_stop), cur_exit_date=str(P.IDX[dc].date()) if dc is not None else '未平倉', g_cur=float(g_cur),
                   cur_ret_net=float(rC.ret) if rC is not None else None, in_cur_log=rC is not None, fz_ret_net=float(rF.ret), cost=float(rF.cost),
                   d_stock=float(d_stock), d_e50=float(d_e50), d_total=float(d_stock + d_e50), bigwin=bool(is_w), biglose=bool(is_l)))
E = pd.DataFrame(EV)
if len(E):
    E.to_csv(HERE / 'out' / 'valid_trigger_events.csv', index=False)
    sm = lambda y: dict(n=int(len(y)), d_stock=float(y['d_stock'].sum()), d_e50=float(y['d_e50'].sum()), d_total=float(y['d_total'].sum()))
    res['events'] = dict(n=int(len(E)), thr_bigwin_ret=thr_w, thr_biglose_ret=thr_l, n5=n5, n_cur_closed=int(len(clC)), all=sm(E), bigwin=sm(E[E['bigwin']]), biglose=sm(E[E['biglose']]),
                         others=sm(E[~E['bigwin'] & ~E['biglose']]), hurt=sm(E[E['d_total'] < 0]), helped=sm(E[E['d_total'] > 0]),
                         median_g_stop=float(E['g_stop'].median()), median_g_cur=float(E['g_cur'].median()), mean_g_stop=float(E['g_stop'].mean()), mean_g_cur=float(E['g_cur'].mean()),
                         frac_cur_better=float((E['g_cur'] > E['g_stop']).mean()),
                         bigwin_list=E[E['bigwin']].to_dict('records'), biglose_list=E[E['biglose']].to_dict('records'),
                         top5_hurt=E.sort_values('d_total').head(5).to_dict('records'), top5_helped=E.sort_values('d_total', ascending=False).head(5).to_dict('records'),
                         nav_diff_main=float(nF.iloc[-1] - nC.iloc[-1]), nav_start=float(FCAP[1e6]))
    res['cur_bigwin_all'] = [dict(code=str(r.code), ent_date=str(r.ent_date.date()), ret=float(r.ret), pnl=float(r.pnl), triggered=bool(((E['code'] == str(r.code)) & (E['ent_date'] == str(r.ent_date.date()))).any()))
                             for r in clC[clC['bigwin']].sort_values('ret', ascending=False).itertuples()]
    res['cur_biglose_all'] = [dict(code=str(r.code), ent_date=str(r.ent_date.date()), ret=float(r.ret), pnl=float(r.pnl), triggered=bool(((E['code'] == str(r.code)) & (E['ent_date'] == str(r.ent_date.date()))).any()))
                              for r in clC[clC['biglose']].sort_values('ret').itertuples()]

# ---- V3 禁買聯集（兩邊標準路徑各自最賺 5 筆）
BLK = {j: list(v) for j, v in s2lib.block_from(lC, 5)[0].items()}
for j, v in s2lib.block_from(lF, 5)[0].items():
    BLK.setdefault(j, []).extend(v)
res['V3_block_union'] = {str(P.COLS[j]): [[str(P.IDX[a].date()), str(P.IDX[b].date())] for a, b in v] for j, v in BLK.items()}
del S0, F0; gc.collect()

# ---- 20 條路徑
PATH = {k: {} for k in ('CUR|fresh', 'FZ|fresh', 'CUR|blk', 'FZ|blk', 'CUR|cont', 'FZ|cont')}
for s in L.SHIFTS:
    S = L.signal(P, U, RT, s); F = L.fresh_from(S)
    for cap in L.CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        for nm, c in (('CUR', None), ('FZ', FZID)):
            nav = L.sim(P, F, U, cell=c, cap=FCAP[cap])[0]
            v, vdd = cg(nav, 'valid'), stlib.maxdd(nav, TB, DE)
            PATH[f'{nm}|fresh'][key] = dict(valid=v, valid_dd=vdd, y2022_2023=cg(nav, 'y2022_2023'), since2024=cg(nav, 'since2024'), seg=L.seg_dd(nav, L.SEGV))
            PATH[f'{nm}|blk'][key] = dict(valid=cg(L.sim(P, F, U, cell=c, cap=FCAP[cap], block=BLK)[0], 'valid'))
            nav = L.sim(P, S, U, cell=c, cap=cap)[0]
            PATH[f'{nm}|cont'][key] = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE))
    del S, F; gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
res['paths'] = PATH
keys = list(PATH['CUR|fresh'])


def pair(a, b, f, sub=None):
    ga = np.array([PATH[a][k][f][sub] if sub else PATH[a][k][f] for k in keys]); gb = np.array([PATH[b][k][f][sub] if sub else PATH[b][k][f] for k in keys])
    d = (ga - gb) * 100
    return dict(median=float(np.median(d)), npos=int((d > 0).sum()), min=float(d.min()), max=float(d.max()), a_median=float(np.median(ga)), b_median=float(np.median(gb)))


pv = dict(valid=pair('FZ|fresh', 'CUR|fresh', 'valid'), valid_dd=pair('FZ|fresh', 'CUR|fresh', 'valid_dd'), y2022_2023=pair('FZ|fresh', 'CUR|fresh', 'y2022_2023'), since2024=pair('FZ|fresh', 'CUR|fresh', 'since2024'),
          seg={sg: pair('FZ|fresh', 'CUR|fresh', 'seg', sg) for sg in L.SEGV}, blk=pair('FZ|blk', 'CUR|blk', 'valid'),
          cont={f: pair('FZ|cont', 'CUR|cont', f) for f in ('since2019', 'full', 'maxdd')})
res['pairs'] = pv
V = dict(V1=pv['valid']['npos'] >= 15, V2=pv['valid']['median'] >= 2.0, V3=pv['blk']['median'] > 0, V4=all(pv['seg'][sg]['median'] >= -2.0 for sg in L.SEGV))
V = {k: bool(v) for k, v in V.items()}
passed = bool(SELECTED not in ('不改', 'DRY') and SELECTED == FZID and T0 and all(V.values()))
res['verdict'] = dict(selected=SELECTED, frozen=FZID, T0=T0, **V, n_pass_V=int(sum(V.values())), overall='通過' if passed else '不通過')
res['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss(), dry=DRY)
if DRY:
    assert all(abs(pv[k]['median']) == 0 and pv[k]['min'] == 0 and pv[k]['max'] == 0 for k in ('valid', 'valid_dd', 'blk')) and 'events' not in res
    print('DRY OK', res['main']['CUR']['n_closed'], res['meta'], {k: pv['cont'][k]['b_median'] for k in pv['cont']})
else:
    (HERE / 'out' / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print('判定', json.dumps(res['verdict'], ensure_ascii=False))
    print(json.dumps({k: pv[k] for k in ('valid', 'valid_dd', 'y2022_2023', 'since2024', 'seg', 'blk', 'cont')}, ensure_ascii=False))
    print(json.dumps(res['main'], ensure_ascii=False)); print(res['meta'])
