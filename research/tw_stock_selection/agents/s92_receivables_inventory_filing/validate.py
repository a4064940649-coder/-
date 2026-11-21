#!/usr/bin/env python3
"""S92 驗證期（凍結後只跑一次）。重新起跑（只保留營收月份 ≥ 2021-12 的訊號；起始資金縮放）× 20 條路徑，驗證窗 2021-12-30 → 2026-10-22。
(1) 判定：凍結新格 FZ 對現行 CUR：T0（取自 FROZEN）＋ V1～V4（同 S25／S88）。
(2) S88 時點檢驗：R09FD|Z、R09FD|F（實際公告日）與參考 R09S88|Z（S88 原估法、新口徑），同樣算 V1～V4（描述；不參與選格）。
(3) 描述：其餘新格驗證窗 20 條路徑年化差（不做 V3）；全期連續 2019 起；現行全部交易分組（全期／訓練／驗證；三分位切點取自訓練期）；
    財報新舊分組；每格在驗證段（重新起跑主跑）擋掉的大贏家（報酬前 5%）與大輸家（後 5%）清單；凍結格與 R09FD|Z 的隨機剔除對照（20 次，seed 92）。
逐格產生訊號、跑完一格就釋放。results.json 已存在就拒絕。--dry：所有格換成恆等，只測流程，不寫倉庫檔。python3 -I -B validate.py [--dry]"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
DRY = '--dry' in sys.argv
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
import s92rules as R
if not DRY:
    PSHA = sha(H / 'PREREG.json'); assert PSHA == (H / 'PREREG.sha256').read_text().split()[0]
    for n, v in json.loads((H / 'PREREG.json').read_text())['code_sha256_at_write'].items():
        assert sha(H / n) == v, f'{n} 在登記後被改過'
    assert sha(H / 'FROZEN.json') == (H / 'FROZEN.sha256').read_text().split()[0]
    assert not (H / 'results.json').exists(), 'results.json 已存在：驗證期只跑一次，拒絕重跑'
    FZ = json.loads((H / 'FROZEN.json').read_text()); assert FZ['PREREG_sha256'] == PSHA
    FZID, T0, SELECTED = FZ['frozen_for_validation'], bool(FZ['T0_pass']), FZ['selected']
    assert FZID in R.NEW_CELLS
else:
    PSHA, FZID, T0, SELECTED = 'DRY', R.NEW_CELLS[1], False, 'DRY'
import s92lib as L
from s92lib import stlib, s2lib

t_all = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT = L.load()
FN = L.Fin(P)
RU = R.Rules(P, U, FN, dry=DRY)
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19, D24, TB = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01'), lastb('2024-01-01'), lastb('2022-01-01')
WIN = {'valid': (TB, DE), 'y2022_2023': (TB, D24), 'since2024': (D24, DE), 'since2019': (D19, DE), 'full': (D0, DE)}
cg = lambda nav, k: stlib.cagr(nav, *WIN[k])
med = lambda v: float(np.median(v))
VCELLS = (FZID,) + R.S88_CELLS + R.REF_CELLS                 # 要做 V1～V4（含禁買）的格
CELLS = R.ALL_CELLS

S0 = L.signal(P, U, RT, 0)
F0 = L.fresh_from(S0)
n1, lC, stC = L.sim(P, F0, U, cap=1e6)
ratio = float(n1.loc[TB] / 1e6)
assert bool((lC['ent_date'] >= pd.Timestamp('2022-01-01')).all())
FCAP = {cap: cap / ratio for cap in L.CAPS}
nC, lC, stC = L.sim(P, F0, U, cap=FCAP[1e6])
lC['code'] = lC['code'].astype(str)
navc, lgc, _ = L.sim(P, S0, U, cap=1e6)                      # 全期連續現行（描述用）
lgc['code'] = lgc['code'].astype(str)
res = dict(PREREG_sha256=PSHA, FROZEN_sha256=(H / 'FROZEN.sha256').read_text().split()[0] if not DRY else 'DRY', dry=DRY, frozen=FZID, selected=SELECTED, T0_pass=T0,
           acceptance=dict(ratio_nav_TB=ratio, cont_since2019=cg(navc, 'since2019'), cont_n_closed=int((lgc['why'] != 'open_end').sum())),
           meta=dict(start=str(D0.date()), valid_from=str(TB.date()), end=str(DE.date())), main={})
assert abs(res['acceptance']['cont_since2019'] - L.TARGET['since2019']) < 1e-9 and res['acceptance']['cont_n_closed'] == 381


def mstat(nav, lg, st):
    cl = lg[lg['why'] != 'open_end']
    return dict(cagr={k: cg(nav, k) for k in ('valid', 'y2022_2023', 'since2024')}, maxdd=stlib.maxdd(nav, TB, DE), seg=L.seg_dd(nav), n_closed=int(len(cl)), n_all=int(len(lg)),
                win=float((cl['ret'] > 0).mean()), avg_ret=float(cl['ret'].mean()), median_ret=float(cl['ret'].median()), avg_stockfrac=float(np.nanmean(st['stockfrac'].loc[TB:DE].to_numpy())))


res['main']['CUR'] = mstat(nC, lC, stC)
res['main']['CUR']['costx2_valid'] = cg(L.sim(P, F0, U, cap=FCAP[1e6], cost='strict_x2')[0], 'valid')
# 驗證段現行（重新起跑主跑）的大贏家／大輸家
clC = lC[lC['why'] != 'open_end'].copy()
n5 = int(round(len(clC) * 0.05))
clC['bigwin'] = clC['ret'].rank(ascending=False, method='first') <= n5
clC['biglose'] = clC['ret'].rank(ascending=True, method='first') <= n5
clC['pnl_rank'] = clC['pnl'].rank(ascending=False, method='first').astype(int)
res['valid_cur_trades'] = dict(n_closed=int(len(clC)), n5=n5, bigwin_ret_ge=float(clC.loc[clC['bigwin'], 'ret'].min()), biglose_ret_le=float(clC.loc[clC['biglose'], 'ret'].max()),
                               bigwin=[dict(code=y.code, sig=str(y.sig_date.date()), ret=float(y.ret), pnl=float(y.pnl), pnl_rank=int(y.pnl_rank)) for y in clC[clC['bigwin']].sort_values('ret', ascending=False).itertuples()],
                               biglose=[dict(code=y.code, sig=str(y.sig_date.date()), ret=float(y.ret), pnl=float(y.pnl)) for y in clC[clC['biglose']].sort_values('ret').itertuples()])

# ---- 位移 0 主跑：每一格（逐格）
BLK = {}; BL = {}; REC0 = {}
keyset = lambda lg: set(zip(lg['sig_t'].astype(int), lg['code'].astype(str)))
kC_all = keyset(lC)
for c in CELLS:
    S2, rec = RU.apply(RT, S0, c)
    REC0[c] = rec
    S2F = L.fresh_from(S2); del S2
    nav, lg, st = L.sim(P, S2F, U, cap=FCAP[1e6])
    lg['code'] = lg['code'].astype(str)
    res['main'][c] = mstat(nav, lg, st)
    kx = keyset(lg)
    gone = clC[[(int(a), b) not in kx for a, b in zip(clC['sig_t'], clC['code'])]]
    new = lg[[(int(a), b) not in kC_all for a, b in zip(lg['sig_t'], lg['code'])]]
    rr = [r for r in rec if r['month'] >= '2021-12']
    BL[c] = dict(list_removed=sum(len(r['removed']) for r in rr), list_added=sum(len(r['added']) for r in rr), months_changed=sum(1 for r in rr if r['removed']),
                 n_blocked_trades=int(len(gone)), blocked_mean_ret=float(gone['ret'].mean()) if len(gone) else None, blocked_median_ret=float(gone['ret'].median()) if len(gone) else None,
                 blocked_win=float((gone['ret'] > 0).mean()) if len(gone) else None, blocked_pnl_sum=float(gone['pnl'].sum()), cur_pnl_sum=float(clC['pnl'].sum()),
                 n_blocked_bigwin=int(gone['bigwin'].sum()), n_blocked_biglose=int(gone['biglose'].sum()), n_blocked_top10_pnl=int((gone['pnl_rank'] <= 10).sum()),
                 blocked_bigwin=[dict(code=y.code, sig=str(y.sig_date.date()), ret=float(y.ret), pnl=float(y.pnl), pnl_rank=int(y.pnl_rank)) for y in gone[gone['bigwin']].sort_values('ret', ascending=False).itertuples()],
                 blocked_biglose=[dict(code=y.code, sig=str(y.sig_date.date()), ret=float(y.ret), pnl=float(y.pnl)) for y in gone[gone['biglose']].sort_values('ret').itertuples()],
                 n_new_trades=int(len(new)), new_pnl_sum=float(new['pnl'].sum()), new_mean_ret=float(new['ret'].mean()) if len(new) else None)
    if c in VCELLS:
        u = {j: list(v) for j, v in s2lib.block_from(lC, 5)[0].items()}
        for j, v in s2lib.block_from(lg, 5)[0].items():
            u.setdefault(j, []).extend(v)
        BLK[c] = u
        res['main'][c]['costx2_valid'] = cg(L.sim(P, S2F, U, cap=FCAP[1e6], cost='strict_x2')[0], 'valid')
        res['main'][c]['V3_block_union'] = {str(P.COLS[j]): [[str(P.IDX[a].date()), str(P.IDX[b].date())] for a, b in v] for j, v in u.items()}
    del S2F, nav, lg, st; gc.collect()
res['blocked_valid'] = BL
print('主跑完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

# ---- 隨機剔除對照（凍結格與 R09FD|Z；每月剔除檔數相同、對象隨機；F 模式照樣遞補不套條件）
RC = {}
for c in (FZID, 'R09FD|Z'):
    rng = np.random.default_rng(92); draws = []
    for _ in range(20):
        Sr, _r = RU.apply(RT, S0, c, rng=rng)
        SrF = L.fresh_from(Sr); del Sr
        draws.append(cg(L.sim(P, SrF, U, cap=FCAP[1e6])[0], 'valid')); del SrF
    v = res['main'][c]['cagr']['valid']
    RC[c] = dict(cand_valid=v, cur_valid=res['main']['CUR']['cagr']['valid'], random_median=med(draws), random_min=float(min(draws)), random_max=float(max(draws)),
                 n_random_below_cand=int(sum(d < v for d in draws)), n_random_above_cand=int(sum(d > v for d in draws)), draws=draws)
    gc.collect()
res['random_removal_control'] = dict(note='每月從名單隨機剔除與該格相同檔數，20 次，seed 92；重新起跑主跑（位移 0、100 萬）驗證窗年化', **RC)

# ---- 描述：現行全期連續交易分組（三分位切點取自訓練期）
rows = []
for k, jx in zip(lgc['sig_t'].to_numpy(), lgc['j'].to_numpy()):
    k = int(k); jx = int(jx)
    X = L.featx(P, FN, k, 'fd'); A = L.feat88(P, U, FN, k, 'fd'); B = L.feat88(P, U, FN, k, 's88')
    r = {n: float(X[n][jx]) for n in ('ar_g', 'inv_g', 'rev_g', 'ar_gap', 'inv_gap', 'dso', 'dio', 'dso_chg', 'dio_chg', 'kq', 'age_days', 'since_filing')}
    r.update(tpl=str(FN.tpl.iloc[jx]), kq88_fd=float(A['kq'][jx]), kq88_s88=float(B['kq'][jx]), age88_s88=float(B['age_days'][jx]),
             R09_fd=bool(A['ACC_HI'][jx] | A['OCF_NEG'][jx]), R09_s88=bool(B['ACC_HI'][jx] | B['OCF_NEG'][jx]))
    for rn, fn in R.NEW_RULES.items():
        with np.errstate(invalid='ignore'):
            r[rn] = bool(np.asarray(fn(X), bool)[jx])
    rows.append(r)
T = pd.concat([lgc.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
T['per'] = np.where(T['sig_date'] < pd.Timestamp('2016-02-01'), 'pre', np.where(T['sig_date'] < pd.Timestamp('2022-01-01'), 'train', 'valid'))
hi, lo = T['ret'].quantile(0.95), T['ret'].quantile(0.05)
T['big_win'] = T['ret'] >= hi; T['big_lose'] = T['ret'] <= lo
T['pnl_rank'] = T['pnl'].rank(ascending=False, method='first').astype(int)
T['newer_q'] = T['kq88_fd'] > T['kq88_s88']
if not DRY:
    T.drop(columns=['j']).to_csv(H / 'trades_features.csv.gz', index=False)


def gstat(z, tp):
    if not len(z):
        return dict(n=0)
    return dict(n=int(len(z)), win=float((z['ret'] > 0).mean()), mean=float(z['ret'].mean()), median=float(z['ret'].median()), n_big_win=int(z['big_win'].sum()), n_big_lose=int(z['big_lose'].sum()),
                n_top10_pnl=int((z['pnl_rank'] <= 10).sum()), pnl_sum=float(z['pnl'].sum()), pnl_share=float(z['pnl'].sum() / tp))


ZT = T[T['per'] == 'train']
CUT = {c: [float(ZT[c].quantile(1 / 3)), float(ZT[c].quantile(2 / 3))] for c in ('ar_gap', 'inv_gap', 'dso_chg', 'dio_chg')}


def gtable(Z):
    tp = float(Z['pnl'].sum()); G = dict(ALL=gstat(Z, tp))
    for c, (q1, q2) in CUT.items():
        v = Z[c]
        G[c] = dict(low=gstat(Z[v <= q1], tp), mid=gstat(Z[(v > q1) & (v <= q2)], tp), high=gstat(Z[v > q2], tp), na=gstat(Z[v.isna()], tp))
    for rn in R.NEW_RULES:
        G[rn + '=1'] = gstat(Z[Z[rn]], tp)
    G['R09_fd=1'] = gstat(Z[Z['R09_fd']], tp); G['R09_s88=1'] = gstat(Z[Z['R09_s88']], tp)
    sf = Z['since_filing']
    G['since_filing'] = {'<=10': gstat(Z[sf <= 10], tp), '11-30': gstat(Z[(sf > 10) & (sf <= 30)], tp), '31-60': gstat(Z[(sf > 30) & (sf <= 60)], tp), '>60': gstat(Z[sf > 60], tp)}
    G['newer_quarter_with_fd'] = gstat(Z[Z['newer_q']], tp); G['same_quarter'] = gstat(Z[~Z['newer_q']], tp)
    return G


res['groups'] = dict(note='現行主跑（全期連續、100 萬、位移 0）389 筆；報酬已扣成本；大贏家／大輸家＝全部交易報酬前／後 5%；top10＝損益最大 10 筆；三分位切點取自訓練期交易',
                     cuts_from_train=CUT, thresholds=dict(big_win_ret_ge=float(hi), big_lose_ret_le=float(lo)), full=gtable(T), train=gtable(ZT), valid=gtable(T[T['per'] == 'valid']))
tc = ['code', 'sig_date', 'ret', 'pnl', 'ar_gap', 'inv_gap', 'dso_chg', 'dio_chg', 'since_filing', 'R09_fd', 'R09_s88'] + list(R.NEW_RULES)
jl = lambda z: json.loads(z[tc].assign(sig_date=z['sig_date'].astype(str)).to_json(orient='records'))
res['cur_top10_pnl'] = jl(T.sort_values('pnl', ascending=False).head(10)); res['cur_worst10_ret'] = jl(T.sort_values('ret').head(10))
res['R09_flag_flips'] = jl(T[T['R09_fd'] != T['R09_s88']])
del T, ZT

# ---- 20 條路徑（逐位移、逐格）
PATH = {f'{c}|{m}': {} for c in ('CUR',) + CELLS for m in ('fresh', 'cont')}
for c in VCELLS:
    PATH[f'{c}|fresh_blk'] = {}; PATH[f'CUR@{c}|fresh_blk'] = {}


def pm(nav):
    sg = L.seg_dd(nav); vdd = stlib.maxdd(nav, TB, DE); v = cg(nav, 'valid')
    return dict(valid=v, valid_dd=vdd, y2022_2023=cg(nav, 'y2022_2023'), since2024=cg(nav, 'since2024'), seg=sg)


for s in L.SHIFTS:
    S = S0 if s == 0 else L.signal(P, U, RT, s)
    SF = L.fresh_from(S)
    for cap in L.CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        PATH['CUR|fresh'][key] = pm(L.sim(P, SF, U, cap=FCAP[cap])[0])
        nav = L.sim(P, S, U, cap=cap)[0]
        PATH['CUR|cont'][key] = dict(since2019=cg(nav, 'since2019'), full=cg(nav, 'full'), maxdd=stlib.maxdd(nav, D0, DE))
        for c in VCELLS:
            PATH[f'CUR@{c}|fresh_blk'][key] = dict(valid=cg(L.sim(P, SF, U, cap=FCAP[cap], block=BLK[c])[0], 'valid'))
    for c in CELLS:
        S2, _r = RU.apply(RT, S, c); del _r
        S2F = L.fresh_from(S2)
        for cap in L.CAPS:
            key = f'{int(cap/1e4)}萬|+{s}'
            PATH[f'{c}|fresh'][key] = pm(L.sim(P, S2F, U, cap=FCAP[cap])[0])
            nav = L.sim(P, S2, U, cap=cap)[0]
            PATH[f'{c}|cont'][key] = dict(since2019=cg(nav, 'since2019'), full=cg(nav, 'full'), maxdd=stlib.maxdd(nav, D0, DE))
            if c in VCELLS:
                PATH[f'{c}|fresh_blk'][key] = dict(valid=cg(L.sim(P, S2F, U, cap=FCAP[cap], block=BLK[c])[0], 'valid'))
        del S2, S2F; gc.collect()
    del SF
    if s:
        del S
    RU.clear(); gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
res['paths'] = PATH
keys = list(PATH['CUR|fresh']); assert len(keys) == 20


def pair(a, b, f, sub=None):
    gg = (lambda d: d[f]) if sub is None else (lambda d: d[f][sub])
    d = np.array([gg(PATH[a][k]) - gg(PATH[b][k]) for k in keys]) * 100.0
    return dict(npos=int((d > 0).sum()), n_ge025=int((d >= 0.25).sum()), median=med(d), min=float(d.min()), max=float(d.max()))


res['paths_median'] = {nm: {f: med([D[k][f] for k in keys]) for f in D[keys[0]] if f != 'seg'} for nm, D in PATH.items()}
pv = {}; VV = {}
for c in CELLS:
    pv[c] = dict(fresh=dict(valid=pair(f'{c}|fresh', 'CUR|fresh', 'valid'), valid_dd=pair(f'{c}|fresh', 'CUR|fresh', 'valid_dd'), y2022_2023=pair(f'{c}|fresh', 'CUR|fresh', 'y2022_2023'),
                            since2024=pair(f'{c}|fresh', 'CUR|fresh', 'since2024'), seg={sg: pair(f'{c}|fresh', 'CUR|fresh', 'seg', sg) for sg in L.SEGV}),
                 cont={f: pair(f'{c}|cont', 'CUR|cont', f) for f in ('since2019', 'full', 'maxdd')})
    if c in VCELLS:
        p = pv[c]; p['fresh_blk'] = dict(valid=pair(f'{c}|fresh_blk', f'CUR@{c}|fresh_blk', 'valid'))
        V = dict(V1=p['fresh']['valid']['npos'] >= 15, V2=p['fresh']['valid']['median'] >= 2.0, V3=p['fresh_blk']['valid']['median'] > 0, V4=all(p['fresh']['seg'][sg]['median'] >= -2.0 for sg in L.SEGV))
        VV[c] = {k: bool(v) for k, v in V.items()}
res['pairs'] = pv; res['V_by_cell'] = VV
V = VV[FZID]
passed = bool(SELECTED != '不改' and SELECTED == FZID and T0 and all(V.values()))
res['verdict'] = dict(selected=SELECTED, frozen=FZID, T0=T0, **V, n_pass_V=int(sum(V.values())), overall='通過' if passed else '不通過',
                      text=('凍結格通過訓練選擇、T0 與 V1～V4：建議改' if passed else '不通過：維持現行（營收驚奇名單不加應收／存貨過濾）'))
a, b = pv['R09FD|Z']['fresh']['valid'], pv['R09S88|Z']['fresh']['valid']
res['s88_timing_check'] = dict(R09FD_Z=a, R09FD_F=pv['R09FD|F']['fresh']['valid'], R09S88_Z_ref=b, V={c: VV[c] for c in R.S88_CELLS + R.REF_CELLS},
                               fd_minus_s88_paired=pair('R09FD|Z|fresh', 'R09S88|Z|fresh', 'valid'),
                               timing_is_cause=bool(all(VV['R09FD|Z'].values())),
                               text='PREREG 判讀：R09FD|Z 的 V1～V4 全過才算「時點估計是 S88 失敗的原因」；否則不是。')
res['meta'].update(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss(), fresh_capital_scale=ratio)
if DRY:
    assert all(pv[c]['fresh']['valid']['min'] == 0 and pv[c]['fresh']['valid']['max'] == 0 and pv[c]['cont']['since2019']['max'] == 0 for c in CELLS), '乾跑應全為 0'
    assert all(BL[c]['n_blocked_trades'] == 0 for c in CELLS)
    print('乾跑通過（所有格與現行完全相同）', res['meta'])
else:
    (H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print('判定', json.dumps(res['verdict'], ensure_ascii=False))
    p = pv[FZID]
    print(json.dumps(dict(valid=p['fresh']['valid'], dd=p['fresh']['valid_dd'], blk=p['fresh_blk']['valid'], seg={k: v['median'] for k, v in p['fresh']['seg'].items()}, cont=p['cont']['since2019']), ensure_ascii=False))
    print(json.dumps(res['s88_timing_check'], ensure_ascii=False))
    print(res['meta'])
