#!/usr/bin/env python3
"""S67 驗證期（凍結後只跑一次）：凍結候選 FZ 對現行 CUR。由 S51 validate.py 改寫（「配置格」換成「名單太少月份的處理方式」）。
(1) 判定用：重新起跑（只保留營收月份 ≥ 2021-12 的訊號；起始資金縮放使 2021-12-30 淨值約＝60～200 萬）× 20 條路徑，驗證窗 2021-12-30 → 2026-10-22；
    S25 標準 T0（訓練期，取自 FROZEN）＋ V1～V4；另報驗證窗回撤與 Calmar。
(2) 描述用（PREREG 登記、不作判定）：另兩個候選在驗證期的同樣統計；全期連續主跑與 20 條路徑；
    容量（1,000 萬、5% 上限、全期連續、位移 0～3，S37 口徑 capped_rate）；「晚 2 天」（位移 2 對位移 0）；
    第 0 步的全樣本／驗證期版（現行口徑：名單 ≤2 檔月份買進的交易 vs ≥5 檔）；各候選在名單太少月份實際多買／少買／加倍的交易。
輸出 results.json。PREREG／SPEC／FROZEN 的 sha256 與登記時程式碼 sha256 必須相符；results.json 已存在就拒絕重跑。
乾跑 --dry：所有候選換成現行（恆等），不需 FROZEN，輸出到 scratchpad/s67/dry。python3 -I -B validate.py [--dry]"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s67lib as L
from s67lib import s10lib, s2lib, stlib

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
print('載入完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE = P.IDX[P.t0], P.IDX[-1]
D19, D24, TB = lastb('2019-01-01'), lastb('2024-01-01'), lastb('2022-01-01')
WIN = {'valid': (TB, DE), 'y2022_2023': (TB, D24), 'since2024': (D24, DE), 'since2019': (D19, DE), 'full': (D0, DE)}
cg = lambda nav, k: stlib.cagr(nav, *WIN[k])
med = lambda v: float(np.median(v))


def prep_S(S):
    return {c: L.apply_cand(P, U, RT, S, cell) for c, cell in CFG.items()}


# ---- 起始資金縮放（重新起跑；2022 前全在 0050，各候選相同）
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
SS0 = prep_S(S0)
n1, l1, _ = L.sim(P, S0, U, cap=1e6, fresh=True)
ratio = float(n1.loc[TB] / 1e6)
acc = dict(A3_no_trade_before_2022=bool((l1['ent_date'] >= pd.Timestamp('2022-01-01')).all()), ratio_nav_TB=ratio, first_entry=str(l1['ent_date'].min().date()))
assert acc['A3_no_trade_before_2022'], acc
del n1, l1
FCAP = {cap: cap / ratio for cap in L.CAPS}
NT = {int(x['k']): int(x['n_tgt']) for x in S0['info']}


def few_trades(lg, sub=None):
    """名單太少月份（現行名單 n_tgt ≤ 2）買進的交易統計；sub＝{(sig_t, 代號)} 時只算這些。"""
    x = lg[lg['sig_t'].map(NT).fillna(99) <= L.FEW]
    if sub is not None:
        x = x[[(int(a), str(b)) in sub for a, b in zip(x['sig_t'], x['code'])]]
    if not len(x):
        return dict(n=0)
    r50 = P.E50[x['exit_t'].to_numpy()] / P.E50O[x['ent_t'].to_numpy()] - 1
    ex = x['ret'].to_numpy() - r50
    return dict(n=int(len(x)), n_months=int(x['sig_t'].nunique()), mean_ret=float(x['ret'].mean()), median_ret=float(x['ret'].median()), win=float((x['ret'] > 0).mean()),
                mean_r50=float(r50.mean()), mean_ex=float(ex.mean()), median_ex=float(np.median(ex)), win_vs_0050=float((ex > 0).mean()),
                avg_cost=float(x['cost'].mean()), pnl_sum=float(x['pnl'].sum()))


def grp_trades(lg):
    o = {}
    n = lg['sig_t'].map(NT)
    r50 = P.E50[lg['exit_t'].to_numpy()] / P.E50O[lg['ent_t'].to_numpy()] - 1
    y = lg.assign(n=n, ex=lg['ret'].to_numpy() - r50, r50=r50)
    for g, m in (('≤2', y['n'] <= L.FEW), ('3～4', (y['n'] > L.FEW) & (y['n'] < 5)), ('≥5', y['n'] >= 5)):
        z = y[m]
        if not len(z):
            o[g] = dict(n=0); continue
        mm = z.groupby('sig_t')['ex'].mean()
        o[g] = dict(n=int(len(z)), n_months=int(z['sig_t'].nunique()), mean_ret=float(z['ret'].mean()), median_ret=float(z['ret'].median()), win=float((z['ret'] > 0).mean()),
                    mean_r50=float(z['r50'].mean()), mean_ex=float(z['ex'].mean()), median_ex=float(z['ex'].median()), win_vs_0050=float((z['ex'] > 0).mean()),
                    month_mean_ex=float(mm.mean()), month_t=(float(mm.mean() / (mm.std(ddof=1) / np.sqrt(len(mm)))) if len(mm) > 2 else None),
                    avg_hold_days=float(z['hold_days'].mean()), pnl_share=float(z['pnl'].sum() / y['pnl'].sum()))
    return o


# ---- 主跑（位移 0）：重新起跑＋全期連續
res = dict(PREREG_sha256=(H / 'PREREG.sha256').read_text().split()[0] if not DRY else 'DRY',
           FROZEN_sha256=(H / 'FROZEN.sha256').read_text().split()[0] if not DRY else 'DRY',
           dry=DRY, cells=CFG, frozen=FZID, selected=SELECTED, T0_pass=T0, acceptance=acc,
           meta=dict(start=str(D0.date()), valid_from=str(TB.date()), end=str(DE.date())),
           bench_0050={k: dict(cagr=cg(bench, k), maxdd=stlib.maxdd(bench, *WIN[k])) for k in WIN}, main={}, few_months={})
res['bench_0050']['seg'] = L.s17lib.seg_dd(bench)
mainlg = {}
for mode in ('fresh', 'cont'):
    fresh = mode == 'fresh'
    cap = FCAP[1e6] if fresh else 1e6
    b0 = TB if fresh else D0
    for c in CFG:
        S2, mult, rec = SS0[c]
        nav, lg, st = L.sim(P, S2, U, mult=mult, cap=cap, fresh=fresh)
        mainlg[f'{c}|{mode}'] = lg
        x = lg[lg['ent_date'] >= pd.Timestamp('2022-01-01')] if fresh else lg
        ks = ('valid', 'y2022_2023', 'since2024') if fresh else tuple(WIN)
        nx = L.sim(P, S2, U, mult=mult, cap=cap, cost='strict_x2', fresh=fresh)[0]
        mdd = stlib.maxdd(nav, b0, DE)
        R = dict(cagr={k: cg(nav, k) for k in ks}, maxdd=mdd, calmar=(cg(nav, 'valid' if fresh else 'since2019') / abs(mdd)), seg=L.s17lib.seg_dd(nav),
                 yearly=stlib.yearly(nav.loc[b0:], bench), n_closed=int((x['why'] != 'open_end').sum()), n_buy=int(st['n_buy']),
                 avg_npos=float(st['avg_npos']), avg_stockfrac=float(np.nanmean(st['stockfrac'].loc[b0:].to_numpy())),
                 max_single_cost_frac=None,
                 top5_share=L.s17lib.top5_share(x, nav.loc[b0:], float(nav.loc[b0])),
                 costx2={k: cg(nx, k) for k in ks}, end_nav=float(nav.iloc[-1]))
        if len(x):
            R['max_single_cost_frac'] = float((x['cost'].to_numpy() / nav.reindex(x['ent_date']).to_numpy()).max())    # 單筆買進成本 ÷ 進場日收盤淨值 的最大值
        if fresh:
            top = s2lib.block_from(x, 5)[1]
            R['top5_trades'] = [dict(code=y.code, ent=str(y.ent_date.date()), exit=str(y.exit_date.date()), ret=float(y.ret), pnl=float(y.pnl)) for y in top.itertuples()]
        res['main'][f'{c}|{mode}'] = R
        # 名單太少月份的交易（描述）
        added = {(r['k'], a) for r in rec for a in r['added']}
        fm = dict(all=few_trades(x))
        if added:
            fm['added_by_C3'] = few_trades(x, added)
        res['few_months'][f'{c}|{mode}'] = fm
        del nav, nx, st
    gc.collect()
res['few_months']['list_months'] = dict(
    valid=[dict(month=r['month'], n_tgt=r['n_tgt'], added=r['added']) for r in SS0['C3'][2] if r['month'] >= '2021-12'],
    all_n=len(SS0['C3'][2]), valid_n=sum(1 for r in SS0['C3'][2] if r['month'] >= '2021-12'))
lgc = mainlg['CUR|cont']
res['step0_full_and_valid'] = dict(note='現行口徑、全期連續主跑（100 萬、位移 0）；依訊號日分段',
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

# ---- 20 條路徑
PATH = {f'{c}|{m}': {} for c in CFG for m in ('fresh', 'cont')}
for c in CAND:
    PATH[f'{c}|fresh_blk'] = {}; PATH[f'CUR|fresh_blk_{c}'] = {}
CAPAC = {c: {} for c in CFG}
for s in L.SHIFTS:
    S = S0 if s == 0 else s10lib.make_signal_s10(P, U, RT, shift=s)
    SS = SS0 if s == 0 else prep_S(S)
    for cap in L.CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        for c in CFG:
            S2, mult, _ = SS[c]
            nav = L.sim(P, S2, U, mult=mult, cap=FCAP[cap], fresh=True)[0]
            sg = L.s17lib.seg_dd(nav); vdd = stlib.maxdd(nav, TB, DE); v = cg(nav, 'valid')
            PATH[f'{c}|fresh'][key] = dict(valid=v, valid_dd=vdd, valid_calmar=v / abs(vdd), y2022_2023=cg(nav, 'y2022_2023'), since2024=cg(nav, 'since2024'),
                                           seg={k: sg[k] for k in L.SEGV})
            nav = L.sim(P, S2, U, mult=mult, cap=cap)[0]
            mdd = stlib.maxdd(nav, D0, DE); s19 = cg(nav, 'since2019')
            PATH[f'{c}|cont'][key] = dict(since2019=s19, full=cg(nav, 'full'), valid=cg(nav, 'valid'), maxdd=mdd, calmar=s19 / abs(mdd), seg=L.s17lib.seg_dd(nav))
            if c != 'CUR':
                PATH[f'{c}|fresh_blk'][key] = dict(valid=cg(L.sim(P, S2, U, mult=mult, cap=FCAP[cap], fresh=True, block=BLK[c])[0], 'valid'))
                PATH[f'CUR|fresh_blk_{c}'][key] = dict(valid=cg(L.sim(P, SS['CUR'][0], U, cap=FCAP[cap], fresh=True, block=BLK[c])[0], 'valid'))
            del nav
    for c in CFG:                                              # 容量：1,000 萬、5% 上限、全期連續
        S2, mult, _ = SS[c]
        nav, _, st = L.sim(P, S2, U, mult=mult, cap=1e7)
        CAPAC[c][f'1000萬|+{s}'] = dict(L.capped_rate(st), since2019=cg(nav, 'since2019'), maxdd=stlib.maxdd(nav, D0, DE))
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
                           seg={sg: pair(f'{c}|cont', 'CUR|cont', 'seg', sg) for sg in L.s17lib.SEGMENTS}))
    p = pv[c]
    V = dict(V1=p['fresh']['valid']['npos'] >= 15, V2=p['fresh']['valid']['median'] >= 2.0, V3=p['fresh_blk']['valid']['median'] > 0,
             V4=all(p['fresh']['seg'][sg]['median'] >= -2.0 for sg in L.SEGV))
    pv[c]['V'] = {k: bool(v) for k, v in V.items()}
res['pairs'] = pv

# ---- 「晚 2 天」（描述）：位移 2 對位移 0（各 5 種資金的中位數）
late = {}
for c in CFG:
    o = {}
    for mode, f in (('fresh', 'valid'), ('cont', 'since2019')):
        by = {s: med([PATH[f'{c}|{mode}'][k][f] for k in keys if k.endswith(f'+{s}')]) for s in L.SHIFTS}
        o[mode] = dict(by_shift={f'+{s}': by[s] for s in L.SHIFTS}, late2_minus_0_pp=(by[2] - by[0]) * 100)
        if c != 'CUR':
            o[mode]['minus_cur_by_shift_pp'] = {f'+{s}': pair(f'{c}|{mode}', f'CUR|{mode}', f, ks=[k for k in keys if k.endswith(f'+{s}')])['median'] for s in L.SHIFTS}
    late[c] = o
res['late2'] = dict(by_cell=late, note='位移 s＝訊號日延後 s 個交易日（名單、名次、RET20 都用延後那天的資料）；fresh＝驗證窗年化、cont＝2019 起年化；各 5 種資金中位數。')

cap_s = {}
for c in CFG:
    r = [CAPAC[c][f'1000萬|+{s}']['capped_rate'] for s in L.SHIFTS]
    cap_s[c] = dict(shift0=r[0], median4=med(r), min4=float(min(r)), max4=float(max(r)),
                    since2019_median4=med([CAPAC[c][f'1000萬|+{s}']['since2019'] for s in L.SHIFTS]),
                    sell_partial_median4=med([CAPAC[c][f'1000萬|+{s}']['sell_partial_cap'] for s in L.SHIFTS]))
res['capacity_1000w'] = dict(runs=CAPAC, summary=cap_s, note='S37 口徑：capped_rate＝被 5% 當日成交值上限縮量的買進嘗試 ÷ 算出股數的買進嘗試；全期連續、嚴成本、位移 0～3。')

# ---- 判定（只用凍結候選）
V = pv[FZID]['V']
passed = bool(SELECTED != '不改' and SELECTED == FZID and T0 and all(V.values()))
res['verdict'] = dict(selected=SELECTED, T0=T0, **V, n_pass_V=int(sum(V.values())), overall='通過' if passed else '不通過',
                      text=('凍結候選通過訓練選擇、T0 與 V1～V4：建議改' if passed else '不通過：維持現行（名單幾檔就買幾檔、每檔金額不變、其餘放 0050）'))
res['meta'].update(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss(), fresh_capital_scale=ratio)
(OUT / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print('判定', json.dumps(res['verdict'], ensure_ascii=False))
print(json.dumps({c: dict(V=pv[c]['V'], valid=pv[c]['fresh']['valid'], dd=pv[c]['fresh']['valid_dd'], cal=pv[c]['fresh']['valid_calmar'], blk=pv[c]['fresh_blk']['valid'],
                          seg={k: v['median'] for k, v in pv[c]['fresh']['seg'].items()}) for c in CAND}, ensure_ascii=False))
print(json.dumps(cap_s, ensure_ascii=False))
print(res['meta'])
