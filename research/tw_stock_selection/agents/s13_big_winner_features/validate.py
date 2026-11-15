#!/usr/bin/env python3
"""S13 驗證期：核對 PREREG 與 FROZEN 的 sha256 後，對驗證期（訊號日 2022-01～2026-10）候選只跑一次凍結規則，依 PREREG 四條判定；
只有四條全過才跑策略格 R13 對 CUR（W10_70、20 條路徑、四段空頭）。results.json 已存在就拒絕重跑。
python3 -I -B validate.py → results.json、cand_valid.csv.gz、（通過時）trades.csv.gz、navs.csv.gz"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s13lib
from s13lib import s10lib, s4lib, s2lib, stlib, COSTS, FEATURES

H = Path(__file__).resolve().parent
pre = (H / 'PREREG.json').read_bytes(); sha = hashlib.sha256(pre).hexdigest()
assert sha == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與 PREREG.sha256 不符'
fz = (H / 'FROZEN.json').read_bytes()
assert hashlib.sha256(fz).hexdigest() == (H / 'FROZEN.sha256').read_text().split()[0], 'FROZEN.json 與 FROZEN.sha256 不符'
DRY = len(sys.argv) > 1 and sys.argv[1] == '--dry'   # 只測策略程式路徑（規則＝全部通過、位移 0、100 萬），不算驗證期任何東西
assert DRY or not (H / 'results.json').exists(), 'results.json 已存在：驗證期只跑一次'
FZ = json.loads(fz); assert FZ['PREREG_sha256'] == sha
TRAIN = json.loads((H / 'results_train.json').read_text())
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s13lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
U0 = stlib.universe(P, 'twse'); INNOV, _ = s4lib.innov_panel(P); U = U0 & ~INNOV; del U0, INNOV
RT = s2lib.rev_tables(P)
Rv = s13lib.rev_matrix(P)
FG = s13lib.fin_gm()
F, D, THR, BW = FZ['feature'], FZ['direction'], FZ['threshold'], FZ['big_winner_threshold_excess']
SIG = {0: s10lib.make_signal_s10(P, U, RT, shift=0)}
S0 = SIG[0]
cand = s13lib.add_features(P, RT, Rv, FG, s13lib.candidates(P, U, RT, 0))
if not DRY:
    va = cand[cand['sig_date'] > pd.Timestamp('2021-12-31')].copy()
    o = s13lib.outcomes(P, S0['X'], [x['k'] for x in S0['info']], va, P.T - 1)
    va = va.join(o)
    n_noopen = int((va['how'] == 'noopen').sum())
    va = va[np.isfinite(va['excess'])].copy()
    va['bw'] = va['excess'] >= BW
    va['keep'] = s13lib.rule_keep(va, F, D, THR)
    va['year'] = va['sig_date'].dt.year

    # ---------------------------------------------------------------- 判定
    a = s13lib.auc(va[F] * 1.0, va['bw']); a_dir = a if D > 0 else 1 - a
    mc = s13lib.month_cluster_diff(va, va['keep'])
    bw_kept, bw_all = float(va.loc[va['keep'], 'bw'].mean()), float(va['bw'].mean())
    top5 = va[va['keep']].nlargest(5, 'excess').index
    v4 = float(va.loc[va['keep'] & ~va.index.isin(top5), 'excess'].mean() - va.loc[~va.index.isin(top5), 'excess'].mean())
    V = dict(V1=dict(auc_dir=a_dir, need='≥ 0.55', ok=bool(a_dir >= 0.55)),
             V2=dict(bw_share_kept=bw_kept, bw_share_all=bw_all, ok=bool(bw_kept > bw_all)),
             V3=dict(d_mean=mc['mean'], t=mc['t'], n_months=mc['n_months'], need='平均 > 0 且 t ≥ 2', ok=bool(mc['mean'] > 0 and mc['t'] >= 2.0)),
             V4=dict(diff_ex_top5=v4, top5_removed=va.loc[top5, ['sig_date', 'code', 'excess']].astype(str).values.tolist(), ok=bool(v4 > 0)))
    PASS = all(V[k]['ok'] for k in V)
    res = {'PREREG_sha256': sha, 'FROZEN': FZ, 'training': TRAIN,
           'validation': dict(n=int(len(va)), n_months=int(va['month'].nunique()), noopen_dropped=n_noopen, n_bw=int(va['bw'].sum()), how=va['how'].value_counts().to_dict(),
                              limup_open=int(va['limup_open'].sum()), keep_share=float(va['keep'].mean()), mean_excess_all=float(va['excess'].mean()),
                              mean_excess_kept=float(va.loc[va['keep'], 'excess'].mean()), mean_excess_dropped=float(va.loc[~va['keep'], 'excess'].mean()),
                              median_excess_kept=float(va.loc[va['keep'], 'excess'].median()), median_excess_all=float(va['excess'].median()),
                              checks=V, PASS=PASS)}
    R = res['validation']
    # ---- 描述（不影響判定）
    R['by_year'] = {}
    for y, g in va.groupby('year'):
        m = s13lib.month_cluster_diff(g, g['keep'])
        R['by_year'][int(y)] = dict(n=int(len(g)), keep_share=float(g['keep'].mean()), mean_kept=float(g.loc[g['keep'], 'excess'].mean()),
                                    mean_all=float(g['excess'].mean()), bw_kept=float(g.loc[g['keep'], 'bw'].mean()), bw_all=float(g['bw'].mean()), d_mean=m['mean'], t=m['t'])
    f1 = va[va['f1']]
    R['f1_subset'] = dict(n=int(len(f1)), keep_share=float(f1['keep'].mean()), mean_kept=float(f1.loc[f1['keep'], 'excess'].mean()),
                          mean_dropped=float(f1.loc[~f1['keep'], 'excess'].mean()), mean_all=float(f1['excess'].mean()),
                          bw_kept=float(f1.loc[f1['keep'], 'bw'].mean()), bw_dropped=float(f1.loc[~f1['keep'], 'bw'].mean()),
                          auc_dir=(lambda x: x if D > 0 else 1 - x)(s13lib.auc(f1[F], f1['bw'])),
                          d_month=s13lib.month_cluster_diff(f1, f1['keep']))
    R['univariate_all_features'] = {}
    for f in FEATURES:
        g = va[np.isfinite(va[f])]
        q = pd.qcut(g[f], 5, labels=False, duplicates='drop')
        tab = [dict(q=int(b) + 1, n=int((q == b).sum()), bw_share=float(g.loc[q == b, 'bw'].mean()), mean_excess=float(g.loc[q == b, 'excess'].mean()))
               for b in sorted(q.unique())]
        R['univariate_all_features'][f] = dict(auc=s13lib.auc(g[f], g['bw']), train_direction=TRAIN['univariate'][f]['direction'],
                                               spearman_excess=float(g[[f, 'excess']].corr(method='spearman').iloc[0, 1]), quintiles=tab)
    R['top_bw'] = va.sort_values('excess', ascending=False).head(20)[['sig_date', 'code', 'rank', 'f1', F, 'keep', 'excess', 'hold_days', 'how']].astype(str).values.tolist()
    R['bw_kept_count'] = dict(kept=int(va.loc[va['bw'], 'keep'].sum()), total=int(va['bw'].sum()))
    R['top5_excess_share_of_sum'] = float(va['excess'].nlargest(5).sum() / va['excess'].sum())
    res['N'] = dict(N_before=2779, training=24, N_after=2803 + (1 if PASS else 0),
                    note='訓練期 6 個單變量＋18 條規則；驗證期同一條凍結規則不加；' + ('通過，策略格 R13 +1。' if PASS else '沒通過，不跑策略格。'))
    print('驗證', json.dumps(V, ensure_ascii=False, default=str), 'PASS', PASS, flush=True)
else:
    PASS = True; THR = -np.inf if D > 0 else np.inf; res = {}   # 乾跑：規則＝全部通過（ADVCHG 涵蓋 100%），結果應與 CUR 相同


# ---------------------------------------------------------------- 策略（只有通過才跑）
if PASS:
    bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
    D0, DE = P.IDX[P.t0], P.IDX[-1]
    D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]; D24 = P.IDX[P.IDX < pd.Timestamp('2024-01-01')][-1]
    D22 = P.IDX[P.IDX < pd.Timestamp('2022-01-01')][-1]
    WIN = {'since2019': (D19, DE), 'full': (D0, DE), 'y2019_2023': (D19, D24), 'since2024': (D24, DE), 'valid': (D22, DE)}
    cg = lambda nav, k: stlib.cagr(nav, *WIN[k])
    mdd = lambda nav: stlib.maxdd(nav, D0, DE)
    CAPS = (1e6,) if DRY else (6e5, 8e5, 1e6, 1.5e6, 2e6)

    def filt(S, cd, kp=None):
        """S 的買進名單只留通過凍結規則的；cd＝同位移的候選＋特徵；kp 給定時用它（驗收 A3：全部通過）。"""
        if kp is None:
            kp = s13lib.rule_keep(cd, F, D, THR)
        bad = cd[cd['f1'] & ~kp]
        S2 = dict(S); sig = S['sig'].copy(); rank = S['rank'].copy()
        sig[bad['k'].to_numpy(), bad['j'].to_numpy()] = False; rank[bad['k'].to_numpy(), bad['j'].to_numpy()] = np.nan
        S2['sig'] = sig; S2['rank'] = rank
        S2['info'] = [dict(x, n_tgt=int(sig[x['k']].sum())) for x in S['info']]
        return S2, int(len(bad)), int(cd['f1'].sum())

    def run(S, cap=1e6, cost='strict', block=None):
        return s10lib.simulate_s10(P, S, U, sat_w=0.7, sizing='W10', lock=False, cost=COSTS[cost], capital=cap, block=block)

    def summ(nav, lg, cap):
        return dict(r19=cg(nav, 'since2019'), full=cg(nav, 'full'), r1923=cg(nav, 'y2019_2023'), r24=cg(nav, 'since2024'), rval=cg(nav, 'valid'),
                    maxdd=mdd(nav), seg=s13lib.seg_dd(nav), top5=s13lib.top5_share(lg, nav, cap), n_trades=int((lg['why'] != 'open_end').sum()))
    st = {'CUR': {'paths': {}}, 'R13': {'paths': {}}}
    acc = {}
    navs = {'0050': bench}; trades = []
    for s in (range(1) if DRY else range(4)):
        S = SIG.pop(s) if s in SIG else s10lib.make_signal_s10(P, U, RT, shift=s)
        S = dict(S); S['X'] = S['XV']['E0']; S.pop('XV', None)
        cd = s13lib.add_features(P, RT, Rv, FG, s13lib.candidates(P, U, RT, s))
        SR, nbad, nf1 = filt(S, cd)
        st['R13'].setdefault('filtered', {})[s] = dict(removed=nbad, f1_total=nf1)
        for cap in CAPS:
            key = f'{int(cap / 1e4)}萬|+{s}'
            for cid, SS in (('CUR', S), ('R13', SR)):
                nav, lg, _ = run(SS, cap)
                st[cid]['paths'][key] = summ(nav, lg, cap)
                if s == 0 and cap == 1e6:
                    st[cid]['main'] = st[cid]['paths'][key]
                    navs[cid] = nav
                    lg2 = lg.copy(); lg2['cell'] = cid; trades.append(lg2)
                    nx, lx, _ = run(SS, cap, 'strict_x2'); st[cid]['cost_x2'] = dict(r19=cg(nx, 'since2019'), full=cg(nx, 'full'), rval=cg(nx, 'valid'))
                    blk, top = s2lib.block_from(lg, 5)
                    nb, lb, _ = run(SS, cap, block=blk); st[cid]['ban_top5'] = dict(r19=cg(nb, 'since2019'), full=cg(nb, 'full'), rval=cg(nb, 'valid'), codes=top['code'].tolist())
                    if cid == 'CUR':
                        acc['A3_CUR'] = dict(r19=st[cid]['main']['r19'], full=st[cid]['main']['full'], maxdd=st[cid]['main']['maxdd'],
                                             ok=bool(abs(st[cid]['main']['r19'] - 0.4813) < 0.00025 and abs(st[cid]['main']['full'] - 0.3408) < 0.00025 and abs(st[cid]['main']['maxdd'] + 0.3340) < 0.00025))
                        S_all, nb0, _ = filt(SS, cd, pd.Series(True, index=cd.index))
                        nA, lA, _ = run(S_all)
                        acc['A3_rule_all_pass_identical'] = bool(nb0 == 0 and np.allclose(nA.to_numpy(), nav.to_numpy(), atol=1e-6) and len(lA) == len(lg))
                        assert acc['A3_CUR']['ok'] and acc['A3_rule_all_pass_identical'], ('驗收 A3 失敗', acc)
                gc.collect()
        del S, SR, cd; gc.collect()
        print('位移', s, '完成', f'{time.time() - t0:.0f}s', rss(), 'MB', flush=True)
    keys = list(st['CUR']['paths'])
    def arr(c, m): return np.array([st[c]['paths'][k][m] for k in keys])
    pair = {}
    for m in ('r19', 'full', 'r1923', 'r24', 'rval', 'maxdd', 'top5'):
        d = arr('R13', m) - arr('CUR', m)
        pair[m] = dict(n_pos=int((d > 0).sum()), median=float(np.median(d)), min=float(d.min()), max=float(d.max()))
    segp = {}
    for sg in s13lib.SEGMENTS:
        d = np.array([st['R13']['paths'][k]['seg'][sg] - st['CUR']['paths'][k]['seg'][sg] for k in keys])
        segp[sg] = dict(n_shallower=int((d > 0).sum()), median=float(np.median(d)))
    for c in st:
        st[c]['paths20'] = {m: dict(min=float(arr(c, m).min()), median=float(np.median(arr(c, m))), max=float(arr(c, m).max())) for m in ('r19', 'rval', 'r1923', 'r24', 'maxdd', 'top5')}
    pv = pair['rval']
    if pv['n_pos'] >= 15 and pv['median'] > 0 and pair['r19']['median'] > 0:
        verdict = '放進策略有改善'
    elif pv['n_pos'] <= 5:
        verdict = '放進策略變差'
    else:
        verdict = '差不多'
    res['strategy'] = dict(cells=st, paired_R13_minus_CUR=pair, segments=segp, verdict=verdict,
                           concentration=('降低集中' if st['R13']['paths20']['top5']['median'] < st['CUR']['paths20']['top5']['median'] else '沒有降低集中'),
                           acceptance=acc, bench_0050={k: stlib.cagr(bench, *WIN[k]) for k in WIN})
    if DRY:
        print('乾跑', acc, res['strategy']['paired_R13_minus_CUR'], st['R13']['filtered']); sys.exit(0)
    pd.concat(trades)[['cell', 'code', 'sig_date', 'ent_date', 'exit_date', 'why', 'ret', 'pnl', 'hold_days']].to_csv(H / 'trades.csv.gz', index=False, float_format='%.6g')
    nv = pd.DataFrame(navs); nv = nv.groupby(nv.index.to_period('W')).last(); nv.index = nv.index.astype(str)
    nv.to_csv(H / 'navs.csv.gz', float_format='%.6g')
    print('策略', verdict, json.dumps(pair, ensure_ascii=False), flush=True)

res['runtime_s'] = round(time.time() - t0, 1); res['maxrss_mb'] = rss()
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
cols = ['month', 'sig_date', 'code', 'rank', 'f1', 'surge', 'ret20'] + FEATURES + ['gm_q', 'ret', 'r50', 'excess', 'bw', 'keep', 'how', 'hold_days', 'limup_open']
va[cols].to_csv(H / 'cand_valid.csv.gz', index=False, float_format='%.6g')
print(json.dumps({k: v for k, v in R.items() if k not in ('univariate_all_features',)}, ensure_ascii=False, indent=1, default=str))
for f in FEATURES:
    u = R['univariate_all_features'][f]
    print(f, 'AUC', round(u['auc'], 3), 'train_dir', u['train_direction'], 'ρ', round(u['spearman_excess'], 3), [(t['q'], t['n'], round(t['bw_share'] * 100, 1), round(t['mean_excess'] * 100, 2)) for t in u['quintiles']])
print('runtime', res['runtime_s'], 'maxrss', res['maxrss_mb'])
