#!/usr/bin/env python3
"""S91 驗證期（凍結後只跑一次）：凍結格 FZ 對現行 CUR。由 S88 validate.py 改寫（判定 V1～V4 同 S25／S70／S88）。
(1) 判定用：重新起跑（只保留營收月份 ≥ 2021-12 的訊號；起始資金縮放）× 20 條路徑，驗證窗 2021-12-30 → 2026-10-22；T0（取自 FROZEN）＋ V1～V4。
(2) 描述用（PREREG 登記、不作判定）：
    a. 現行主跑（全期連續、100 萬、位移 0）全部交易依融資／融券／借券特徵分組（全期／訓練／驗證）；
    b. 每一格擋掉的現行交易：筆數、平均報酬、損益合計、其中屬於現行「損益最大 10 筆」「報酬前 5%」「報酬後 5%」的筆數；凍結格在驗證段擋掉的每一筆明細；
    c. 其餘 27 格在驗證窗重新起跑 20 條路徑的年化差（不做 V3、不作判定）；全期連續 20 條路徑；
    d. 凍結格的隨機剔除對照（每月剔除檔數相同、對象隨機，F 模式照樣遞補；20 次，seed 91）；
    e. 凍結格的「延遲 0 日」版（用到訊號日當天晚上公布的餘額；<格>|L0）：驗證窗重新起跑 20 條路徑成對差、全期連續主跑。
輸出 results.json、trades_features.csv.gz。results.json 已存在就拒絕重跑。乾跑 --dry：所有格換成現行，不需 FROZEN。python3 -I -B validate.py [--dry]"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s91lib as L
from s91lib import s10lib, s2lib, stlib

H = Path(__file__).resolve().parent
DRY = '--dry' in sys.argv
if not DRY:
    for n in ('PREREG', 'FROZEN'):
        assert hashlib.sha256((H / f'{n}.json').read_bytes()).hexdigest() == (H / f'{n}.sha256').read_text().split()[0], f'{n}.json 與 sha256 不符'
    PR = json.loads((H / 'PREREG.json').read_text())
    for n, v in PR['code_sha256_at_write'].items():
        assert hashlib.sha256((H / n).read_bytes()).hexdigest() == v, f'{n} 在登記後被改過'
    assert not (H / 'results.json').exists(), 'results.json 已存在：驗證期只跑一次，拒絕重跑'
    FZ = json.loads((H / 'FROZEN.json').read_text())
    FZID = FZ['frozen_for_validation']; T0 = bool(FZ['T0_pass']); SELECTED = FZ['selected']
    assert FZID in L.CELLS
else:
    FZID = L.CELLS[1]; T0 = False; SELECTED = 'DRY'
OUT = L.TMP.parent / 'dry' if DRY else H
OUT.mkdir(parents=True, exist_ok=True)
L0ID = FZID + '|L0'
CAND = {c: ('CUR' if DRY else c) for c in L.CELLS}
CFG = dict(CUR='CUR', **CAND)
CFG[L0ID] = 'CUR' if DRY else L0ID                          # 凍結格的延遲 0 日版（乾跑一律換成現行，不碰驗證期）
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT = L.load()
CH = L.Chips(P)
print('載入完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE = P.IDX[P.t0], P.IDX[-1]
D19, D24, TB = lastb('2019-01-01'), lastb('2024-01-01'), lastb('2022-01-01')
WIN = {'valid': (TB, DE), 'y2022_2023': (TB, D24), 'since2024': (D24, DE), 'since2019': (D19, DE), 'full': (D0, DE)}
cg = lambda nav, k: stlib.cagr(nav, *WIN[k])
med = lambda v: float(np.median(v))
prep_S = lambda S, s: {c: L.apply_cell(P, U, RT, CH, S, cell, shift=s) for c, cell in CFG.items()}

S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
SS0 = prep_S(S0, 0)
n1, l1, _ = L.sim(P, S0, U, cap=1e6, fresh=True)
ratio = float(n1.loc[TB] / 1e6)
acc = dict(A3_no_trade_before_2022=bool((l1['ent_date'] >= pd.Timestamp('2022-01-01')).all()), ratio_nav_TB=ratio, first_entry=str(l1['ent_date'].min().date()))
assert acc['A3_no_trade_before_2022'], acc
del n1, l1
FCAP = {cap: cap / ratio for cap in L.CAPS}

res = dict(PREREG_sha256=(H / 'PREREG.sha256').read_text().split()[0] if not DRY else 'DRY', FROZEN_sha256=(H / 'FROZEN.sha256').read_text().split()[0] if not DRY else 'DRY',
           dry=DRY, frozen=FZID, selected=SELECTED, T0_pass=T0, acceptance=acc, meta=dict(start=str(D0.date()), valid_from=str(TB.date()), end=str(DE.date())),
           bench_0050={k: dict(cagr=cg(bench, k), maxdd=stlib.maxdd(bench, *WIN[k])) for k in WIN}, main={})

# ---- 主跑（位移 0）：現行與每一格，重新起跑＋全期連續
mainlg = {}
for mode in ('fresh', 'cont'):
    fresh = mode == 'fresh'
    cap = FCAP[1e6] if fresh else 1e6
    b0 = TB if fresh else D0
    for c in CFG:
        nav, lg, st = L.sim(P, SS0[c][0], U, cap=cap, fresh=fresh)
        lg['code'] = lg['code'].astype(str)
        mainlg[f'{c}|{mode}'] = lg
        x = lg[lg['ent_date'] >= pd.Timestamp('2022-01-01')] if fresh else lg
        ks = ('valid', 'y2022_2023', 'since2024') if fresh else tuple(WIN)
        mdd = stlib.maxdd(nav, b0, DE)
        R = dict(cagr={k: cg(nav, k) for k in ks}, maxdd=mdd, seg=L.seg_dd(nav), n_closed=int((x['why'] != 'open_end').sum()), n_all=int(len(x)),
                 avg_stockfrac=float(np.nanmean(st['stockfrac'].loc[b0:].to_numpy())), top5_share=L.top5_share(x, nav.loc[b0:], float(nav.loc[b0])), end_nav=float(nav.iloc[-1]))
        if c in ('CUR', FZID):
            R['costx2'] = {k: cg(L.sim(P, SS0[c][0], U, cap=cap, cost='strict_x2', fresh=fresh)[0], k) for k in ks}
            R['yearly'] = stlib.yearly(nav.loc[b0:], bench)
        res['main'][f'{c}|{mode}'] = R
        del nav, st
    gc.collect()
gc_ = res['main']['CUR|cont']
assert abs(gc_['cagr']['since2019'] - L.TARGET['since2019']) < 1e-9 and gc_['n_closed'] == L.TARGET['n_closed'] and gc_['n_all'] == L.TARGET['n_all'], '現行主跑與正式數字不符'

# ---- 描述 a：現行全期交易分組
T = pd.concat([mainlg['CUR|cont'].reset_index(drop=True), L.trade_features(P, U, CH, mainlg['CUR|cont'], 1), L.trade_features(P, U, CH, mainlg['CUR|cont'], 0, '_L0')], axis=1)
T['ret20'] = P.RET20[T['sig_t'].to_numpy(), T['j'].to_numpy()]
T['ex'] = T['ret'].to_numpy() - (P.E50[T['exit_t'].to_numpy()] / P.E50O[T['ent_t'].to_numpy()] - 1)
T['per'] = np.where(T['sig_date'] < pd.Timestamp('2016-02-01'), 'pre', np.where(T['sig_date'] < pd.Timestamp('2022-01-01'), 'train', 'valid'))
hi, lo = T['ret'].quantile(0.95), T['ret'].quantile(0.05)
T['big_win'] = T['ret'] >= hi; T['big_lose'] = T['ret'] <= lo
T['pnl_rank'] = T['pnl'].rank(ascending=False, method='first').astype(int)
for r, fn in L.RULES.items():
    with np.errstate(invalid='ignore'):
        T[r] = np.asarray(fn({c: T[c].to_numpy() for c in T.columns if not c.endswith('_L0')}), bool)
        T[r + '_L0'] = np.asarray(fn({c[:-3]: T[c].to_numpy() for c in T.columns if c.endswith('_L0')}), bool)
T.drop(columns=['j']).to_csv(OUT / 'trades_features.csv.gz', index=False)


def gstat(z, tp, nall):
    if not len(z):
        return dict(n=0)
    return dict(n=int(len(z)), win=float((z['ret'] > 0).mean()), mean=float(z['ret'].mean()), median=float(z['ret'].median()), mean_ex=float(z['ex'].mean()), median_ex=float(z['ex'].median()),
                n_big_win=int(z['big_win'].sum()), big_win=float(z['big_win'].mean()), n_big_lose=int(z['big_lose'].sum()), big_lose=float(z['big_lose'].mean()),
                n_top10_pnl=int((z['pnl_rank'] <= 10).sum()), n_bottom10_pnl=int((z['pnl_rank'] > nall - 10).sum()), pnl_sum=float(z['pnl'].sum()), pnl_share=float(z['pnl'].sum() / tp))


BINS = (('p00_33', 0, 1 / 3), ('p33_67', 1 / 3, 2 / 3), ('p67_100', 2 / 3, 1.0001), ('top20', 0.8, 1.0001), ('top10', 0.9, 1.0001), ('bot20', 0, 0.2))


def gtable(Z):
    tp = float(Z['pnl'].sum()); na = len(T)
    G = dict(ALL=gstat(Z, tp, na), no_margin=gstat(Z[Z['no_margin']], tp, na))
    for c in L.FEATS:
        p = Z[c + '_pct']
        G[c] = {b: gstat(Z[(p >= a) & (p < e)], tp, na) for b, a, e in BINS}
        G[c]['NaN'] = gstat(Z[~np.isfinite(p)], tp, na)
    for r in L.RULES:
        G[r + '=1'] = gstat(Z[Z[r]], tp, na); G[r + '=0'] = gstat(Z[~Z[r]], tp, na)
        G[r + '_L0=1'] = gstat(Z[Z[r + '_L0']], tp, na)
    G['chase50'] = gstat(Z[Z['ret20'] > 0.5], tp, na)
    return G


res['groups'] = dict(note='現行主跑（全期連續、100 萬、位移 0）全部交易；報酬已扣成本；ex＝減同持有期 0050；大贏家／大輸家＝全部交易報酬前／後 5%；top10／bottom10＝全期損益金額最大／最小 10 筆；依訊號日分段；特徵延遲 1 日（_L0＝延遲 0）',
                     thresholds=dict(big_win_ret_ge=float(hi), big_lose_ret_le=float(lo)), full=gtable(T), train=gtable(T[T['per'] == 'train']), valid=gtable(T[T['per'] == 'valid']))
res['big_lose_rule_rate'] = {r: float(T.loc[T['big_lose'], r].mean()) for r in L.RULES}
res['big_win_rule_rate'] = {r: float(T.loc[T['big_win'], r].mean()) for r in L.RULES}
res['all_rule_rate'] = {r: float(T[r].mean()) for r in L.RULES}
fp = lambda y: {c: (None if not np.isfinite(getattr(y, c + '_pct')) else round(float(getattr(y, c + '_pct')), 3)) for c in L.FEATS}
res['cur_top10_pnl'] = [dict(code=str(y.code), sig=str(y.sig_date.date()), ret=float(y.ret), pnl=float(y.pnl), **fp(y)) for y in T.sort_values('pnl', ascending=False).head(10).itertuples()]
res['cur_bottom10_pnl'] = [dict(code=str(y.code), sig=str(y.sig_date.date()), ret=float(y.ret), pnl=float(y.pnl), **fp(y)) for y in T.sort_values('pnl').head(10).itertuples()]

# ---- 描述 b：每格擋掉的現行交易（全期連續主跑）
lc = T
kc = set(zip(lc['sig_t'], lc['code']))
BL = {}
row = lambda y: dict(code=str(y.code), sig=str(y.sig_date.date()), ent_month=str(y.ent_date)[:7], exit=str(y.exit_date.date()), why=str(y.why), ret=float(y.ret), pnl=float(y.pnl), pnl_rank=int(y.pnl_rank),
                     big_win=bool(y.big_win), big_lose=bool(y.big_lose))
for c in CFG:
    if c == 'CUR':
        continue
    lx = mainlg[f'{c}|cont']
    kx = set(zip(lx['sig_t'], lx['code']))
    gm = np.array([(a, b) not in kx for a, b in zip(lc['sig_t'], lc['code'])])
    gone = lc[gm]
    new = lx[[(a, b) not in kc for a, b in zip(lx['sig_t'], lx['code'])]]
    o = dict(n_cur=int(len(lc)), n_cand=int(len(lx)), n_blocked=int(len(gone)), blocked_mean_ret=float(gone['ret'].mean()) if len(gone) else None, blocked_median_ret=float(gone['ret'].median()) if len(gone) else None,
             blocked_pnl_sum=float(gone['pnl'].sum()), blocked_big_win=int(gone['big_win'].sum()), blocked_big_lose=int(gone['big_lose'].sum()), blocked_top10_pnl=int((gone['pnl_rank'] <= 10).sum()),
             n_new=int(len(new)), new_pnl_sum=float(new['pnl'].sum()), new_mean_ret=(float(new['ret'].mean()) if len(new) else None), new_median_ret=(float(new['ret'].median()) if len(new) else None))
    for lab in ('train', 'valid'):
        g2 = gone[gone['per'] == lab]
        o[lab] = dict(n_blocked=int(len(g2)), blocked_mean_ret=float(g2['ret'].mean()) if len(g2) else None, blocked_pnl_sum=float(g2['pnl'].sum()), blocked_big_win=int(g2['big_win'].sum()),
                      blocked_big_lose=int(g2['big_lose'].sum()), blocked_top10_pnl=int((g2['pnl_rank'] <= 10).sum()))
    o['blocked_top5'] = [row(y) for y in gone.sort_values('pnl', ascending=False).head(5).itertuples()]
    o['blocked_worst5'] = [row(y) for y in gone.sort_values('pnl').head(5).itertuples()]
    if c in (FZID, L0ID):
        gv = gone[gone['per'] == 'valid']
        o['valid_blocked_all'] = [row(y) for y in gv.sort_values('pnl', ascending=False).itertuples()]
        nx = lx.copy(); nx = nx[[(a, b) not in kc for a, b in zip(nx['sig_t'], nx['code'])]]
        o['valid_new_all'] = [dict(code=str(y.code), sig=str(y.sig_date.date()), ret=float(y.ret), pnl=float(y.pnl), why=str(y.why)) for y in nx[nx['sig_date'] >= pd.Timestamp('2022-01-01')].sort_values('pnl', ascending=False).itertuples()]
    BL[c] = o
res['blocked_trades_cont'] = BL
print('主跑與描述完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

# ---- V3 禁買聯集（只對凍結格）
u = {j: list(v) for j, v in s2lib.block_from(mainlg['CUR|fresh'], 5)[0].items()}
for j, v in s2lib.block_from(mainlg[f'{FZID}|fresh'], 5)[0].items():
    u.setdefault(j, []).extend(v)
BLK = u
res['V3_block_union'] = {str(P.COLS[j]): [[str(P.IDX[a].date()), str(P.IDX[b].date())] for a, b in v] for j, v in u.items()}
del mainlg

# ---- 描述 d：凍結格的隨機剔除對照
rng = np.random.default_rng(91)
rec = SS0[FZID][1]; fmode = FZID.split('|')[1]
info_by_k = {x['k']: x for x in S0['info']}
draws = []
for _ in range(20):
    sig = S0['sig'].copy(); rank = S0['rank'].copy()
    for r in rec:
        nrm = len(r['removed'])
        if not nrm:
            continue
        k = r['k']; x = info_by_k[k]; tg = np.flatnonzero(S0['sig'][k])
        rm = rng.choice(tg, size=nrm, replace=False)
        sig[k, rm] = False; rank[k, rm] = np.nan
        if fmode == 'F':
            o, s = L.order_ma(P, U, RT, x, 0); nadd = 0
            for j in o[10:L.FILL_DEPTH]:
                if nadd >= nrm:
                    break
                if P.RET20[k][j] > 0.10:
                    sig[k, j] = True; rank[k, j] = s[j]; nadd += 1
    Sr = dict(S0); Sr['sig'] = sig; Sr['rank'] = rank
    draws.append(cg(L.sim(P, Sr, U, cap=FCAP[1e6], fresh=True)[0], 'valid'))
v = res['main'][f'{FZID}|fresh']['cagr']['valid']
res['random_removal_control'] = dict(note='每月從名單隨機剔除與凍結格相同檔數（F 模式照樣由第 11～20 名遞補，不套籌碼條件），20 次，seed 91；重新起跑主跑（位移 0、100 萬）驗證窗年化',
                                     cand_valid=v, cur_valid=res['main']['CUR|fresh']['cagr']['valid'], random_median=med(draws), random_min=float(min(draws)), random_max=float(max(draws)),
                                     n_random_below_cand=int(sum(d < v for d in draws)), draws=draws)

# ---- 20 條路徑
PATH = {f'{c}|{m}': {} for c in CFG for m in ('fresh', 'cont')}
PATH[f'{FZID}|fresh_blk'] = {}; PATH['CUR|fresh_blk'] = {}
for s in L.SHIFTS:
    S = S0 if s == 0 else s10lib.make_signal_s10(P, U, RT, shift=s)
    SS = SS0 if s == 0 else prep_S(S, s)
    for cap in L.CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        for c in CFG:
            S2 = SS[c][0]
            nav = L.sim(P, S2, U, cap=FCAP[cap], fresh=True)[0]
            sg = L.seg_dd(nav); vdd = stlib.maxdd(nav, TB, DE); v = cg(nav, 'valid')
            PATH[f'{c}|fresh'][key] = dict(valid=v, valid_dd=vdd, valid_calmar=v / abs(vdd), y2022_2023=cg(nav, 'y2022_2023'), since2024=cg(nav, 'since2024'), seg={k: sg[k] for k in L.SEGV})
            nav = L.sim(P, S2, U, cap=cap)[0]
            mdd = stlib.maxdd(nav, D0, DE); s19 = cg(nav, 'since2019')
            PATH[f'{c}|cont'][key] = dict(since2019=s19, full=cg(nav, 'full'), maxdd=mdd, calmar=s19 / abs(mdd), seg=L.seg_dd(nav))
            del nav
        PATH[f'{FZID}|fresh_blk'][key] = dict(valid=cg(L.sim(P, SS[FZID][0], U, cap=FCAP[cap], fresh=True, block=BLK)[0], 'valid'))
        PATH['CUR|fresh_blk'][key] = dict(valid=cg(L.sim(P, SS['CUR'][0], U, cap=FCAP[cap], fresh=True, block=BLK)[0], 'valid'))
    if s:
        del S, SS
    gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
res['paths'] = PATH
keys = list(PATH['CUR|fresh'])
assert len(keys) == 20


def pair(a, b, f, sub=None):
    gg = (lambda d: d[f]) if sub is None else (lambda d: d[f][sub])
    sc = 1.0 if 'calmar' in f else 100.0
    d = np.array([gg(PATH[a][k]) - gg(PATH[b][k]) for k in keys]) * sc
    return dict(npos=int((d > 0).sum()), n_ge025=int((d >= 0.25).sum()), median=med(d), min=float(d.min()), max=float(d.max()))


res['paths_median'] = {nm: {f: med([D[k][f] for k in keys]) for f in D[keys[0]] if f != 'seg'} for nm, D in PATH.items()}
pv = {}
for c in CFG:
    if c == 'CUR':
        continue
    pv[c] = dict(fresh=dict(valid=pair(f'{c}|fresh', 'CUR|fresh', 'valid'), valid_dd=pair(f'{c}|fresh', 'CUR|fresh', 'valid_dd'), valid_calmar=pair(f'{c}|fresh', 'CUR|fresh', 'valid_calmar'),
                            y2022_2023=pair(f'{c}|fresh', 'CUR|fresh', 'y2022_2023'), since2024=pair(f'{c}|fresh', 'CUR|fresh', 'since2024'),
                            seg={sg: pair(f'{c}|fresh', 'CUR|fresh', 'seg', sg) for sg in L.SEGV}),
                 cont=dict({f: pair(f'{c}|cont', 'CUR|cont', f) for f in ('since2019', 'full', 'maxdd', 'calmar')}, seg={sg: pair(f'{c}|cont', 'CUR|cont', 'seg', sg) for sg in L.SEGMENTS}))
res['pairs'] = pv
p = pv[FZID]
p['fresh_blk'] = dict(valid=pair(f'{FZID}|fresh_blk', 'CUR|fresh_blk', 'valid'))
V = dict(V1=p['fresh']['valid']['npos'] >= 15, V2=p['fresh']['valid']['median'] >= 2.0, V3=p['fresh_blk']['valid']['median'] > 0, V4=all(p['fresh']['seg'][sg]['median'] >= -2.0 for sg in L.SEGV))
V = {k: bool(v) for k, v in V.items()}
passed = bool(SELECTED != '不改' and SELECTED == FZID and T0 and all(V.values()))
res['verdict'] = dict(selected=SELECTED, frozen=FZID, T0=T0, **V, n_pass_V=int(sum(V.values())), overall='通過' if passed else '不通過',
                      text=('凍結格通過訓練選擇、T0 與 V1～V4：建議改' if passed else '不通過：維持現行（營收驚奇名單不加融資／融券／借券過濾）'))
res['meta'].update(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss(), fresh_capital_scale=ratio)
(OUT / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print('判定', json.dumps(res['verdict'], ensure_ascii=False))
print(json.dumps(dict(valid=p['fresh']['valid'], dd=p['fresh']['valid_dd'], blk=p['fresh_blk']['valid'], seg={k: v['median'] for k, v in p['fresh']['seg'].items()}, cont=p['cont']['since2019'],
                      L0_valid=pv[L0ID]['fresh']['valid'], random=res['random_removal_control']), ensure_ascii=False))
print(res['meta'])
