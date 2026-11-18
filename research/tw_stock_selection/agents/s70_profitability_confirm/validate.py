#!/usr/bin/env python3
"""S70 驗證期（凍結後只跑一次）：凍結候選 FZ 對現行 CUR。由 S67 validate.py 改寫（判定段 V1～V4 逐字相同）。
(1) 判定用：重新起跑（只保留營收月份 ≥ 2021-12 的訊號；起始資金縮放使 2021-12-30 淨值約＝60～200 萬）× 20 條路徑，驗證窗 2021-12-30 → 2026-10-22；
    S25 標準 T0（訓練期，取自 FROZEN）＋ V1～V4；另報驗證窗回撤與 Calmar。
(2) 描述用（PREREG 登記、不作判定）：另兩個候選在驗證期的同樣統計；全期連續主跑與 20 條路徑；各位移；
    第 0 步的報酬分組（現行主跑交易：虧損 vs 不虧損、毛利／資產三組）全樣本／驗證期版；被候選擋掉的現行交易；
    隨機剔除對照（C1、C2：每月從名單隨機剔除與該候選相同檔數，20 次，重新起跑主跑）。
輸出 results.json。PREREG／SPEC／FROZEN 的 sha256 與登記時程式碼 sha256 必須相符；results.json 已存在就拒絕重跑。
乾跑 --dry：所有候選換成現行（恆等），不需 FROZEN，輸出到 scratchpad/s70/dry。python3 -I -B validate.py [--dry]"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s70lib as L
from s70lib import s10lib, s2lib, stlib

H = Path(__file__).resolve().parent
DRY = '--dry' in sys.argv
if not DRY:
    for n in ('SPEC', 'PREREG', 'FROZEN'):
        assert hashlib.sha256((H / f'{n}.json').read_bytes()).hexdigest() == (H / f'{n}.sha256').read_text().split()[0], f'{n}.json 與 sha256 不符'
    SPEC = json.loads((H / 'SPEC.json').read_text())
    for n, v in SPEC['code_sha256_at_write'].items():
        assert hashlib.sha256((H / n).read_bytes()).hexdigest() == v, f'{n} 在登記後被改過'
    assert not (H / 'results.json').exists(), 'results.json 已存在：驗證期只跑一次，拒絕重跑'
    FZ = json.loads((H / 'FROZEN.json').read_text())
    FZID = FZ['frozen_for_validation']; T0 = bool(FZ['T0_pass']); SELECTED = FZ['selected']
    assert FZID in L.CANDS
else:
    FZID = 'C1'; T0 = False; SELECTED = 'DRY'
OUT = L.TMP / 'dry' if DRY else H
OUT.mkdir(parents=True, exist_ok=True)
CAND = {c: ('CUR' if DRY else c) for c in L.CANDS}
CFG = dict(CUR='CUR', **CAND)
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = L.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
U, RT = L.base(P)
L.s17lib.e50_fix_suspension(P)
L.slim(P); gc.collect()
FN = L.Fin(P)
print('載入完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE = P.IDX[P.t0], P.IDX[-1]
D19, D24, TB = lastb('2019-01-01'), lastb('2024-01-01'), lastb('2022-01-01')
WIN = {'valid': (TB, DE), 'y2022_2023': (TB, D24), 'since2024': (D24, DE), 'since2019': (D19, DE), 'full': (D0, DE)}
cg = lambda nav, k: stlib.cagr(nav, *WIN[k])
med = lambda v: float(np.median(v))


def prep_S(S):
    return {c: L.apply_cand(P, U, FN, S, cell) for c, cell in CFG.items()}


# ---- 起始資金縮放（重新起跑；2022 前全在 0050，各候選相同）
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
SS0 = prep_S(S0)
n1, l1, _ = L.sim(P, S0, U, cap=1e6, fresh=True)
ratio = float(n1.loc[TB] / 1e6)
acc = dict(A3_no_trade_before_2022=bool((l1['ent_date'] >= pd.Timestamp('2022-01-01')).all()), ratio_nav_TB=ratio, first_entry=str(l1['ent_date'].min().date()))
assert acc['A3_no_trade_before_2022'], acc
del n1, l1
FCAP = {cap: cap / ratio for cap in L.CAPS}

# 名單逐檔特徵（位移 0）
FEAT = {}
for x in S0['info']:
    r = L.features(P, U, FN, x['k'])
    for c in x['targets']:
        j = P.ci[c]
        FEAT[(int(x['k']), str(c))] = dict(L.flags(r, j), gpa_pct=float(r['gpa_pct'][j]), opi_na=bool(not np.isfinite(r['opi_q'][j])), ni_na=bool(not np.isfinite(r['ni_ttm'][j])))


def ret_stats(z, total_pnl):
    if not len(z):
        return dict(n=0)
    mm = z.groupby('sig_t')['ex'].mean()
    return dict(n=int(len(z)), n_months=int(z['sig_t'].nunique()), mean_ret=float(z['ret'].mean()), median_ret=float(z['ret'].median()), win=float((z['ret'] > 0).mean()),
                mean_ex=float(z['ex'].mean()), median_ex=float(z['ex'].median()), win_vs_0050=float((z['ex'] > 0).mean()),
                month_t=(float(mm.mean() / (mm.std(ddof=1) / np.sqrt(len(mm)))) if len(mm) > 2 else None),
                pnl_sum=float(z['pnl'].sum()), pnl_share=float(z['pnl'].sum() / total_pnl), best=float(z['ret'].max()),
                top3=[dict(code=str(a), sig=str(b.date()), ret=float(c), pnl=float(e)) for a, b, c, e in z.sort_values('pnl', ascending=False).head(3)[['code', 'sig_date', 'ret', 'pnl']].itertuples(index=False)])


def grp_trades(lg):
    y = lg.copy()
    f = [FEAT[(int(a), str(b))] for a, b in zip(y['sig_t'], y['code'])]
    for c in ('OPLOSS', 'GPA_LOW', 'EPS_TTM', 'gpa_pct', 'opi_na', 'ni_na'):
        y[c] = [v[c] for v in f]
    y['ex'] = y['ret'].to_numpy() - (P.E50[y['exit_t'].to_numpy()] / P.E50O[y['ent_t'].to_numpy()] - 1)
    tp = float(y['pnl'].sum())
    g = {'營業利益<0（最近一季）': y['OPLOSS'], '營業利益≥0': ~y['OPLOSS'] & ~y['opi_na'], '近四季淨利≤0': y['EPS_TTM'], '近四季淨利>0': ~y['EPS_TTM'] & ~y['ni_na'],
         '毛利／資產最低30%': y['GPA_LOW'], '毛利／資產中間40%': (y['gpa_pct'] >= 0.3) & (y['gpa_pct'] < 0.7), '毛利／資產最高30%': y['gpa_pct'] >= 0.7}
    return dict(ALL=ret_stats(y, tp), **{k: ret_stats(y[m], tp) for k, m in g.items()})


# ---- 主跑（位移 0）：重新起跑＋全期連續
res = dict(PREREG_sha256=(H / 'PREREG.sha256').read_text().split()[0] if not DRY else 'DRY',
           FROZEN_sha256=(H / 'FROZEN.sha256').read_text().split()[0] if not DRY else 'DRY',
           dry=DRY, cells=CFG, frozen=FZID, selected=SELECTED, T0_pass=T0, acceptance=acc,
           meta=dict(start=str(D0.date()), valid_from=str(TB.date()), end=str(DE.date())),
           bench_0050={k: dict(cagr=cg(bench, k), maxdd=stlib.maxdd(bench, *WIN[k])) for k in WIN}, main={}, blocked_trades={})
res['bench_0050']['seg'] = L.s17lib.seg_dd(bench)
mainlg = {}
for mode in ('fresh', 'cont'):
    fresh = mode == 'fresh'
    cap = FCAP[1e6] if fresh else 1e6
    b0 = TB if fresh else D0
    for c in CFG:
        S2, rec = SS0[c]
        nav, lg, st = L.sim(P, S2, U, cap=cap, fresh=fresh)
        lg['code'] = lg['code'].astype(str)
        mainlg[f'{c}|{mode}'] = lg
        x = lg[lg['ent_date'] >= pd.Timestamp('2022-01-01')] if fresh else lg
        ks = ('valid', 'y2022_2023', 'since2024') if fresh else tuple(WIN)
        nx = L.sim(P, S2, U, cap=cap, cost='strict_x2', fresh=fresh)[0]
        mdd = stlib.maxdd(nav, b0, DE)
        R = dict(cagr={k: cg(nav, k) for k in ks}, maxdd=mdd, calmar=(cg(nav, 'valid' if fresh else 'since2019') / abs(mdd)), seg=L.s17lib.seg_dd(nav),
                 yearly=stlib.yearly(nav.loc[b0:], bench), n_closed=int((x['why'] != 'open_end').sum()), n_buy=int(st['n_buy']), noslot=int(st['noslot']),
                 avg_npos=float(st['avg_npos']), avg_stockfrac=float(np.nanmean(st['stockfrac'].loc[b0:].to_numpy())),
                 top5_share=L.s17lib.top5_share(x, nav.loc[b0:], float(nav.loc[b0])),
                 costx2={k: cg(nx, k) for k in ks}, end_nav=float(nav.iloc[-1]))
        if fresh:
            top = s2lib.block_from(x, 5)[1]
            R['top5_trades'] = [dict(code=str(y.code), ent=str(y.ent_date.date()), exit=str(y.exit_date.date()), ret=float(y.ret), pnl=float(y.pnl)) for y in top.itertuples()]
        res['main'][f'{c}|{mode}'] = R
        del nav, nx, st
    gc.collect()
# 被候選擋掉／多買的現行交易（描述）：以現行主跑交易為準
for mode in ('fresh', 'cont'):
    lc = mainlg[f'CUR|{mode}']
    exc = lc['ret'].to_numpy() - (P.E50[lc['exit_t'].to_numpy()] / P.E50O[lc['ent_t'].to_numpy()] - 1)
    lc = lc.assign(ex=exc)
    tp = float(lc['pnl'].sum())
    for c in CAND:
        lx = mainlg[f'{c}|{mode}']
        kc = set(zip(lc['sig_t'], lc['code'])); kx = set(zip(lx['sig_t'], lx['code']))
        gone = lc[[(a, b) not in kx for a, b in zip(lc['sig_t'], lc['code'])]]
        new = lx[[(a, b) not in kc for a, b in zip(lx['sig_t'], lx['code'])]]
        o = dict(n_cur=len(lc), n_cand=len(lx), not_bought_by_cand=ret_stats(gone, tp), n_new_in_cand=int(len(new)),
                 new_in_cand_pnl=float(new['pnl'].sum()), new_in_cand_mean_ret=(float(new['ret'].mean()) if len(new) else None))
        if mode == 'cont':
            for lab, m in (('train', (gone['sig_date'] >= '2016-02-01') & (gone['sig_date'] < '2022-01-01')), ('valid', gone['sig_date'] >= '2022-01-01')):
                o[f'not_bought_{lab}'] = ret_stats(gone[m], tp)
        o['not_bought_top5'] = [dict(code=str(y.code), sig=str(y.sig_date.date()), ret=float(y.ret), pnl=float(y.pnl)) for y in gone.sort_values('pnl', ascending=False).head(5).itertuples()]
        o['not_bought_worst5'] = [dict(code=str(y.code), sig=str(y.sig_date.date()), ret=float(y.ret), pnl=float(y.pnl)) for y in gone.sort_values('pnl').head(5).itertuples()]
        res['blocked_trades'][f'{c}|{mode}'] = o
lgc = mainlg['CUR|cont']
res['step0_returns_by_group'] = dict(note='現行口徑、全期連續主跑（100 萬、位移 0）交易；報酬已扣成本；ex＝減同持有期 0050；依訊號日分段',
                                     full=grp_trades(lgc), valid=grp_trades(lgc[lgc['sig_date'] >= pd.Timestamp('2022-01-01')]),
                                     train=grp_trades(lgc[(lgc['sig_date'] >= pd.Timestamp('2016-02-01')) & (lgc['sig_date'] < pd.Timestamp('2022-01-01'))]))
print('主跑完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

# V3 禁買聯集：CUR 與各候選（重新起跑主跑）各自最賺 5 筆
BLK = {}
b_cur = s2lib.block_from(mainlg['CUR|fresh'], 5)[0]
for c in CAND:
    u = {j: list(v) for j, v in b_cur.items()}
    for j, v in s2lib.block_from(mainlg[f'{c}|fresh'], 5)[0].items():
        u.setdefault(j, []).extend(v)
    BLK[c] = u
res['V3_block_union'] = {c: {str(P.COLS[j]): [[str(P.IDX[a].date()), str(P.IDX[b].date())] for a, b in v] for j, v in u.items()} for c, u in BLK.items()}
del mainlg

# ---- 隨機剔除對照（描述）：C1、C2 每月剔除檔數相同、對象隨機；重新起跑主跑（位移 0、100 萬換算）
rng = np.random.default_rng(70)
RC = {}
for c in ('C1', 'C2'):
    rec = SS0[c][1]
    draws = []
    for _ in range(20):
        sig = S0['sig'].copy(); rank = S0['rank'].copy()
        for r in rec:
            nrm = len(r['removed'])
            if nrm:
                tg = np.flatnonzero(S0['sig'][r['k']])
                rm = rng.choice(tg, size=nrm, replace=False)
                sig[r['k'], rm] = False; rank[r['k'], rm] = np.nan
        Sr = dict(S0); Sr['sig'] = sig; Sr['rank'] = rank
        draws.append(cg(L.sim(P, Sr, U, cap=FCAP[1e6], fresh=True)[0], 'valid'))
    v = res['main'][f'{c}|fresh']['cagr']['valid']
    RC[c] = dict(cand_valid=v, cur_valid=res['main']['CUR|fresh']['cagr']['valid'], random_median=med(draws), random_p10=float(np.percentile(draws, 10)),
                 random_p90=float(np.percentile(draws, 90)), n_random_below_cand=int(sum(d < v for d in draws)), draws=draws)
res['random_removal_control'] = dict(note='每月從名單隨機剔除與候選相同檔數（不遞補），20 次，seed 70；重新起跑主跑驗證窗年化', runs=RC)
print('隨機對照完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

# ---- 20 條路徑
PATH = {f'{c}|{m}': {} for c in CFG for m in ('fresh', 'cont')}
for c in CAND:
    PATH[f'{c}|fresh_blk'] = {}; PATH[f'CUR|fresh_blk_{c}'] = {}
for s in L.SHIFTS:
    S = S0 if s == 0 else s10lib.make_signal_s10(P, U, RT, shift=s)
    SS = SS0 if s == 0 else prep_S(S)
    for cap in L.CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        for c in CFG:
            S2, _ = SS[c]
            nav = L.sim(P, S2, U, cap=FCAP[cap], fresh=True)[0]
            sg = L.s17lib.seg_dd(nav); vdd = stlib.maxdd(nav, TB, DE); v = cg(nav, 'valid')
            PATH[f'{c}|fresh'][key] = dict(valid=v, valid_dd=vdd, valid_calmar=v / abs(vdd), y2022_2023=cg(nav, 'y2022_2023'), since2024=cg(nav, 'since2024'),
                                           seg={k: sg[k] for k in L.SEGV})
            nav = L.sim(P, S2, U, cap=cap)[0]
            mdd = stlib.maxdd(nav, D0, DE); s19 = cg(nav, 'since2019')
            PATH[f'{c}|cont'][key] = dict(since2019=s19, full=cg(nav, 'full'), valid=cg(nav, 'valid'), maxdd=mdd, calmar=s19 / abs(mdd), seg=L.s17lib.seg_dd(nav))
            if c != 'CUR':
                PATH[f'{c}|fresh_blk'][key] = dict(valid=cg(L.sim(P, S2, U, cap=FCAP[cap], fresh=True, block=BLK[c])[0], 'valid'))
                PATH[f'CUR|fresh_blk_{c}'][key] = dict(valid=cg(L.sim(P, SS['CUR'][0], U, cap=FCAP[cap], fresh=True, block=BLK[c])[0], 'valid'))
            del nav
    if s:
        del S, SS
    gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
res['paths'] = PATH
keys = list(PATH['CUR|fresh'])
assert len(keys) == 20


def pair(a, b, f, sub=None, ks=None):
    gg = (lambda d: d[f]) if sub is None else (lambda d: d[f][sub])
    sc = 1.0 if 'calmar' in f else 100.0
    d = np.array([gg(PATH[a][k]) - gg(PATH[b][k]) for k in (ks or keys)]) * sc
    return dict(npos=int((d > 0).sum()), n_ge025=int((d >= 0.25).sum()), median=med(d), min=float(d.min()), max=float(d.max()))


summ = {}
for nm, D in PATH.items():
    summ[nm] = {f: med([D[k][f] for k in keys]) for f in D[keys[0]] if f != 'seg'}
    if 'seg' in D[keys[0]]:
        summ[nm]['seg'] = {sg: med([D[k]['seg'][sg] for k in keys]) for sg in D[keys[0]]['seg']}
res['paths_median'] = summ
pv = {}
for c in CAND:
    pv[c] = dict(fresh=dict(valid=pair(f'{c}|fresh', 'CUR|fresh', 'valid'), valid_dd=pair(f'{c}|fresh', 'CUR|fresh', 'valid_dd'),
                            valid_calmar=pair(f'{c}|fresh', 'CUR|fresh', 'valid_calmar'), y2022_2023=pair(f'{c}|fresh', 'CUR|fresh', 'y2022_2023'),
                            since2024=pair(f'{c}|fresh', 'CUR|fresh', 'since2024'), seg={sg: pair(f'{c}|fresh', 'CUR|fresh', 'seg', sg) for sg in L.SEGV}),
                 fresh_blk=dict(valid=pair(f'{c}|fresh_blk', f'CUR|fresh_blk_{c}', 'valid')),
                 cont=dict({f: pair(f'{c}|cont', 'CUR|cont', f) for f in ('since2019', 'full', 'valid', 'maxdd', 'calmar')},
                           seg={sg: pair(f'{c}|cont', 'CUR|cont', 'seg', sg) for sg in L.s17lib.SEGMENTS}),
                 by_shift_valid_pp={f'+{s}': pair(f'{c}|fresh', 'CUR|fresh', 'valid', ks=[k for k in keys if k.endswith(f'+{s}')])['median'] for s in L.SHIFTS})
    p = pv[c]
    V = dict(V1=p['fresh']['valid']['npos'] >= 15, V2=p['fresh']['valid']['median'] >= 2.0, V3=p['fresh_blk']['valid']['median'] > 0,
             V4=all(p['fresh']['seg'][sg]['median'] >= -2.0 for sg in L.SEGV))
    pv[c]['V'] = {k: bool(v) for k, v in V.items()}
res['pairs'] = pv

# ---- 判定（只用凍結候選）
V = pv[FZID]['V']
passed = bool(SELECTED != '不改' and SELECTED == FZID and T0 and all(V.values()))
res['verdict'] = dict(selected=SELECTED, T0=T0, **V, n_pass_V=int(sum(V.values())), overall='通過' if passed else '不通過',
                      text=('凍結候選通過訓練選擇、T0 與 V1～V4：建議改' if passed else '不通過：維持現行（營收驚奇名單不加獲利確認）'))
res['meta'].update(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss(), fresh_capital_scale=ratio)
(OUT / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print('判定', json.dumps(res['verdict'], ensure_ascii=False))
print(json.dumps({c: dict(V=pv[c]['V'], valid=pv[c]['fresh']['valid'], dd=pv[c]['fresh']['valid_dd'], cal=pv[c]['fresh']['valid_calmar'], blk=pv[c]['fresh_blk']['valid'],
                          seg={k: v['median'] for k, v in pv[c]['fresh']['seg'].items()}) for c in CAND}, ensure_ascii=False))
print(res['meta'])
