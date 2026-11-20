#!/usr/bin/env python3
"""S81 驗證段（只跑一次；valid_results.json 已存在就拒絕）。先核對 PREREG／FROZEN／程式 sha256。
python3 -I -B run_valid.py → valid_results.json、valid_grid.csv、nav_key.csv.gz、paths.csv、yearly.csv、trades_key.csv.gz"""
import sys, json, time, hashlib, resource
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (HERE / 'valid_results.json').exists(), 'valid_results.json 已存在：驗證段只跑一次'
for f in ('PREREG', 'FROZEN'):
    assert sha(HERE / f'{f}.json') == (HERE / f'{f}.sha256').read_text().split()[0], f
for line in (HERE / 'code.sha256').read_text().strip().split('\n'):
    h, f = line.split(); assert sha(HERE / f) == h, f
import numpy as np, pandas as pd
import s81lib as Q
from s81lib import K, R
t0 = time.time()
FZ = json.loads((HERE / 'FROZEN.json').read_text()); FID = FZ['frozen']
P, U = K.load_cont()
F, RT = K.surge(P, 'B2')
D0 = P.IDX[P.t0]
bench = pd.Series(P.E50, index=P.IDX).loc[D0:]; bench = bench / bench.iloc[0] * 1e6
cells = Q.grid(); byid = {c['id']: c for c in cells}
VERS = (('LOT', False), ('ODD', True))
W = Q.WINDOWS


def stats(nav):
    rs, rb = R.monthly_returns(nav, bench.reindex(nav.index))
    return rs, rb, {w: Q.block(rs, rb, nav, a, b) for w, (a, b) in W.items()}


# ---- 1 全體格子 × 2 版本
rows = []; sgc = {}; KEY = {}
for i, c in enumerate(cells):
    key = (c['depth'], c['mom'], c['keep'], c['hold'])
    if key not in sgc:
        sgc = {key: Q.cell_signal(P, U, RT, c)}
    sg = sgc[key]
    r = dict(id=c['id'], idx=i, cap=c['cap'], depth=c['depth'], mom=c['mom'], exit=c['exit'])
    for ver, odd in VERS:
        nav, lg, st = Q.simulate(P, sg, U, c['cap'], odd=odd)
        rs, rb, S = stats(nav)
        for w in W:
            for k_ in ('ex_ann_pp', 't_nw6', 'IR', 'maxdd', 'cagr'):
                r[f'{ver}|{w}|{k_}'] = S[w][k_]
        for w in ('前段 2008-2015', '後段 2016-2026'):
            ab = Q.acct_block(P, lg, st, nav, *W[w])
            r[f'{ver}|{w}|avg_npos'] = ab['avg_npos']; r[f'{ver}|{w}|stockfrac'] = ab['stockfrac']
        if c['id'] in (Q.CUR, FID):
            KEY[(c['id'], ver)] = (sg, nav, lg, st, rs, rb, S)
    rows.append(r)
G = pd.DataFrame(rows); G.to_csv(HERE / 'valid_grid.csv', index=False)
print('grid', f'{time.time() - t0:.0f}s', flush=True)
out = dict(task='S81', frozen=FID, sha=dict(PREREG=sha(HERE / 'PREREG.json'), FROZEN=sha(HERE / 'FROZEN.json')), rules={})
bs = {w: dict(cagr_0050=KEY[(Q.CUR, 'LOT')][6][w]['cagr_0050'], maxdd_0050=Q.dd_window(bench, *W[w])) for w in W}
out['e0050'] = bs
# ---- 2 現行與凍結規則的明細
navs = {'0050': bench}; trades = []; prow = []; yrows = []
for cid in dict.fromkeys((Q.CUR, FID)):
    c = byid[cid]; out['rules'][cid] = {}
    for ver, odd in VERS:
        sg, nav, lg, st, rs, rb, S = KEY[(cid, ver)]
        navs[f'{cid}#{ver}'] = nav
        trades.append(lg.drop(columns=['j']).assign(rule=cid, version=ver))
        o = dict(final_nav=float(nav.iloc[-1]), windows={})
        nz, lz, sz = Q.simulate(P, sg, U, c['cap'], odd=odd, cost='zero')
        _, _, SZ = stats(nz)
        for w, (a, b) in W.items():
            ab = Q.acct_block(P, lg, st, nav, a, b)
            g, n = SZ[w]['ex_ann_pp'], S[w]['ex_ann_pp']
            o['windows'][w] = dict(S[w], **ab, gross_ex_ann_pp=g, cost_pp=g - n, cost_share_of_gross=((g - n) / g if g > 0 else None))
        for y, v in S['全期 2008-2026']['yearly_ex_pp'].items():
            yrows.append(dict(rule=cid, version=ver, year=y, ex_pp=v))
        o['engine'] = {k: int(st[k]) for k in ('n_sig', 'n_buy', 'blk_limup', 'blk_noopen', 'blk_lot', 'blk_cap', 'blk_nocash', 'blk_min', 'noslot', 'odd_buys')}
        # 去掉報酬率前 5% 交易（各段各自）
        x = lg[lg['why'] != 'open_end']; rem = {}
        for w in ('前段 2008-2015', '後段 2016-2026'):
            a, b = W[w]; A_ = pd.Period(a, 'M').to_timestamp(); B_ = pd.Period(b, 'M').to_timestamp(how='end')
            y = x[(x['ent_date'] >= A_) & (x['ent_date'] <= B_)].sort_values('ret', ascending=False)
            pk = y.head(int(round(len(y) * 0.05)))
            blk = {}
            for j, e_, x_ in zip(pk['j'], pk['ent_t'], pk['exit_t']):
                blk.setdefault(int(j), []).append((int(e_), int(x_)))
            nb_, _, _ = Q.simulate(P, sg, U, c['cap'], odd=odd, block=blk)
            _, _, SB = stats(nb_)
            rem[w] = dict(n_trades=int(len(y)), n_removed=int(len(pk)), min_ret_removed=float(pk['ret'].min()) if len(pk) else None, ex_before=S[w]['ex_ann_pp'], ex_after=SB[w]['ex_ann_pp'], t_after=SB[w]['t_nw6'],
                          n_ret_over_100pct=int((y['ret'] > 1.0).sum()))
        o['remove_top5pct'] = rem
        # 20 條路徑
        pp = []
        for s in K.SHIFTS:
            sgs = Q.cell_signal(P, U, RT, c, shift=s)
            for cap in K.CAPS:
                n2, _, _ = Q.simulate(P, sgs, U, c['cap'], odd=odd, capital=cap)
                _, _, S2 = stats(n2)
                d = dict(rule=cid, version=ver, shift=s, capital=int(cap), cagr_full=S2['全期 2008-2026']['cagr'], maxdd_full=S2['全期 2008-2026']['maxdd'],
                         **{f'ex_{w}': S2[w]['ex_ann_pp'] for w in W})
                pp.append(d); prow.append(d)
        pdf = pd.DataFrame(pp)
        o['paths20'] = dict(median={k: float(pdf[k].median()) for k in pdf.columns if k not in ('rule', 'version', 'shift', 'capital')},
                            min={k: float(pdf[k].min()) for k in ('cagr_full', 'maxdd_full', 'ex_前段 2008-2015', 'ex_後段 2016-2026')},
                            max={k: float(pdf[k].max()) for k in ('cagr_full', 'maxdd_full', 'ex_前段 2008-2015', 'ex_後段 2016-2026')})
        if odd:
            n3, _, _ = Q.simulate(P, sg, U, c['cap'], odd=True, cost='oddlot')
            _, _, S3 = stats(n3)
            o['oddlot_cost_sensitivity'] = {w: dict(ex_ann_pp=S3[w]['ex_ann_pp'], cagr=S3[w]['cagr']) for w in W}
        out['rules'][cid][ver] = o
        print(cid, ver, f'{time.time() - t0:.0f}s', flush=True)
# ---- 3 凍結 vs 現行、判定
ver_out = {}
if FID and FID != Q.CUR:
    for ver, _ in VERS:
        f_, c_ = out['rules'][FID][ver], out['rules'][Q.CUR][ver]
        fw, cw = f_['windows'], c_['windows']
        diff = KEY[(FID, ver)][4] - KEY[(Q.CUR, ver)][4]
        dt = {w: dict(diff_ann_pp=float(Q.win(diff, *W[w]).mean() * 1200), t_nw6=R.nw_t(Q.win(diff, *W[w]).to_numpy(), 6)) for w in W}
        tr, va, fu = '前段 2008-2015', '後段 2016-2026', '全期 2008-2026'
        p1 = fw[va]['ex_ann_pp'] >= cw[va]['ex_ann_pp'] - 3
        p2 = (fw[tr]['ex_ann_pp'] >= cw[tr]['ex_ann_pp'] + 3) and (fw[tr]['t_nw6'] >= 1.5)
        p3 = fw[fu]['maxdd'] >= cw[fu]['maxdd'] - 0.03
        ver_out[ver] = dict(P1=bool(p1), P2=bool(p2), P3=bool(p3), all=bool(p1 and p2 and p3),
                            valid_ex=[fw[va]['ex_ann_pp'], cw[va]['ex_ann_pp']], train_ex=[fw[tr]['ex_ann_pp'], cw[tr]['ex_ann_pp']], train_t=fw[tr]['t_nw6'],
                            maxdd_full=[fw[fu]['maxdd'], cw[fu]['maxdd']], frozen_minus_current=dt)
    ver_out['P4_same_conclusion'] = bool(ver_out['LOT']['all'] == ver_out['ODD']['all'])
    ver_out['PASS'] = bool(ver_out['LOT']['all'] and ver_out['ODD']['all'])
out['verdict'] = ver_out
# ---- 4 搜尋空間在驗證段的分佈
dist = {}
T = pd.read_csv(HERE / 'train_grid.csv')
for ver, _ in VERS:
    d = {}
    for w in W:
        v = G[f'{ver}|{w}|ex_ann_pp']
        cur = float(v[G['id'] == Q.CUR].iloc[0]); fz = float(v[G['id'] == FID].iloc[0]) if FID else None
        d[w] = dict(min=float(v.min()), q25=float(v.quantile(.25)), median=float(v.median()), q75=float(v.quantile(.75)), max=float(v.max()), current=cur, frozen=fz,
                    n_beat_current=int((v > cur).sum()), frozen_rank_from_top=(int((v > fz).sum()) + 1) if FID else None, best_id=G['id'][v.idxmax()])
    va = G[f'{ver}|後段 2016-2026|ex_ann_pp']; tr = G[f'{ver}|前段 2008-2015|ex_ann_pp']
    cva = float(va[G['id'] == Q.CUR].iloc[0]); ctr = float(tr[G['id'] == Q.CUR].iloc[0])
    d['rank_corr_train_vs_valid_ex'] = float(tr.rank().corr(va.rank()))
    d['rank_corr_train_score_vs_valid_ex'] = float(T['score'].rank().corr(va.rank()))
    d['n_train_plus3_and_valid_minus3'] = int(((tr >= ctr + 3) & (va >= cva - 3)).sum())
    d['ids_train_plus3_and_valid_minus3'] = G['id'][(tr >= ctr + 3) & (va >= cva - 3)].tolist()
    dd = G[f'{ver}|全期 2008-2026|maxdd']; cdd = float(dd[G['id'] == Q.CUR].iloc[0])
    d['n_all_three_posthoc'] = int(((tr >= ctr + 3) & (va >= cva - 3) & (dd >= cdd - 0.03) & (G[f'{ver}|前段 2008-2015|t_nw6'] >= 1.5)).sum())
    d['ids_all_three_posthoc'] = G['id'][(tr >= ctr + 3) & (va >= cva - 3) & (dd >= cdd - 0.03) & (G[f'{ver}|前段 2008-2015|t_nw6'] >= 1.5)].tolist()
    d['by_dim_median'] = {dim: {str(k): dict(train=float(g[f'{ver}|前段 2008-2015|ex_ann_pp'].median()), valid=float(g[f'{ver}|後段 2016-2026|ex_ann_pp'].median()),
                                             sub4=float(g[f'{ver}|2019-2026|ex_ann_pp'].median()), maxdd=float(g[f'{ver}|全期 2008-2026|maxdd'].median()),
                                             npos_tr=float(g[f'{ver}|前段 2008-2015|avg_npos'].median()), npos_va=float(g[f'{ver}|後段 2016-2026|avg_npos'].median()),
                                             sf_tr=float(g[f'{ver}|前段 2008-2015|stockfrac'].median()), sf_va=float(g[f'{ver}|後段 2016-2026|stockfrac'].median()))
                                for k, g in G.groupby(dim)} for dim in ('cap', 'depth', 'mom', 'exit')}
    dist[ver] = d
both = set(dist['LOT']['ids_all_three_posthoc']) & set(dist['ODD']['ids_all_three_posthoc'])
dist['ids_all_posthoc_both_versions'] = sorted(both)
out['grid_distribution'] = dist
out['runtime_s'] = round(time.time() - t0, 1); out['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
(HERE / 'valid_results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
df = pd.DataFrame(navs).round(2); df.index.name = 'date'; df.to_csv(HERE / 'nav_key.csv.gz')
pd.concat(trades).to_csv(HERE / 'trades_key.csv.gz', index=False)
pd.DataFrame(prow).to_csv(HERE / 'paths.csv', index=False); pd.DataFrame(yrows).to_csv(HERE / 'yearly.csv', index=False)
print(json.dumps(out['verdict'], ensure_ascii=False, indent=1)); print('done', out['runtime_s'], out['maxrss_mb'])
