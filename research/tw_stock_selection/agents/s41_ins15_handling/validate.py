#!/usr/bin/env python3
"""S41 驗證期（凍結後只跑一次）：反事實下凍結處理 FZ 對 A。
(1) 判定：重新起跑（營收月份 ≥ 2021-12；起始資金縮放）× 20 條路徑，驗證窗 2021-12-30 → 2026-10-22；V1～V4。
(2) 描述：A、B、C、CUR 全期連續（20 條路徑）的 2019 起、全期、驗證窗年化與回撤；與 CUR 的距離（建議規則）；交易差異（標準路徑）；成本加倍。
(3) 描述：實際情境（只有營收月份 ≥ 2026-01 看不到）A、B、C 對 CUR。
輸出 results.json、data/trades_diff.csv。PREREG、FROZEN sha256 須相符；results.json 已存在就拒絕。python3 -I -B validate.py"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s41lib as L
from s41lib import s17lib, s10lib, s2lib, stlib

H = Path(__file__).resolve().parent
pre = (H / 'PREREG.json').read_bytes(); sha = hashlib.sha256(pre).hexdigest()
assert sha == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與 PREREG.sha256 不符'
fz = (H / 'FROZEN.json').read_bytes(); fsha = hashlib.sha256(fz).hexdigest()
assert fsha == (H / 'FROZEN.sha256').read_text().split()[0], 'FROZEN.json 與 FROZEN.sha256 不符'
assert not (H / 'results.json').exists(), 'results.json 已存在：驗證期只跑一次，拒絕重跑'
FZ = json.loads(fz); FT = FZ['frozen']
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT = L.prep_panel()
S0, av = RT['S'], RT['av']
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE = P.IDX[P.t0], P.IDX[-1]
D19, D24, TB = lastb('2019-01-01'), lastb('2024-01-01'), lastb('2022-01-01')
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
SEGV = ('2022', '2024-07~2025-04')
CFG = ['CUR', 'A', 'B', 'C']
cg = stlib.cagr; md = stlib.maxdd


def signals(s, first_month=None):
    cells = L.hidden_cells(P, av, S0, s, first_month=first_month)
    SG = {'CUR': s10lib.make_signal_s10(P, U, RT, shift=s)}
    for c in 'ABC':
        SG[c] = L.build(P, U, RT, c, cells, s)[0]
    return SG, len(cells)


# 起始資金縮放（同 S25）
SG0, nc0 = signals(0)
nav1, lg1, _ = L.sim(P, U, SG0['A'], 1e6, fresh=True)
ratio = float(nav1.loc[TB] / 1e6)
acc = dict(no_trade_before_2022=bool((lg1['ent_date'] >= pd.Timestamp('2022-01-01')).all()), ratio_nav_TB=ratio)
assert acc['no_trade_before_2022'], acc
FCAP = {cap: cap / ratio for cap in CAPS}
res = {'PREREG_sha256': sha, 'FROZEN_sha256': fsha, 'frozen': FT, 'T0_pass': FZ['T0_pass'], 'acceptance': acc,
       'meta': dict(start=str(D0.date()), valid_from=str(TB.date()), end=str(DE.date()))}

# ---------------------------------------------------------------- 主跑（位移 0、100 萬）：判定 V3 用的最賺 5 筆＋描述
main = {}; key = lambda d: set(zip(d['code'], d['ent_t'], d['exit_t']))
tdiff = []
for mode in ('fresh', 'cont'):
    fresh = mode == 'fresh'
    for c in CFG:
        cap = FCAP[1e6] if fresh else 1e6
        nav, lg, st = L.sim(P, U, SG0[c], cap, fresh=fresh)
        navx = L.sim(P, U, SG0[c], cap, cost='strict_x2', fresh=fresh)[0]
        b0 = TB if fresh else D0
        main[f'{c}|{mode}'] = dict(lg=lg, m=dict(valid=cg(nav, TB, DE), valid_dd=md(nav, TB, DE), since2019=cg(nav, D19, DE) if not fresh else None,
                                                full=cg(nav, D0, DE) if not fresh else None, maxdd=md(nav, b0, DE), seg=s17lib.seg_dd(nav),
                                                valid_costx2=cg(navx, TB, DE), since2019_costx2=cg(navx, D19, DE) if not fresh else None,
                                                n_closed=int((lg['why'] != 'open_end').sum()), n_buy=int(st['n_buy']),
                                                yearly=stlib.yearly(nav.loc[b0:], pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:]))))
for mode in ('fresh', 'cont'):
    lc = main[f'CUR|{mode}']['lg']
    for c in 'ABC':
        a = main[f'{c}|{mode}']['lg']; ka, kb = key(a), key(lc)
        main[f'{c}|{mode}']['m'].update(trades_only_here=len(ka - kb), trades_only_CUR=len(kb - ka), jaccard_vs_CUR=len(ka & kb) / max(len(ka | kb), 1))
        if mode == 'cont':
            for nm, d, ks in ((c, a, ka - kb), ('CUR', lc, kb - ka)):
                for y in d.itertuples():
                    if (y.code, y.ent_t, y.exit_t) in ks:
                        tdiff.append(dict(treat=c, side=nm, code=y.code, in_group=y.code in L.GROUP, ent=str(y.ent_date.date()), exit=str(y.exit_date.date()),
                                          why=y.why, ret=round(float(y.ret), 4), pnl=round(float(y.pnl))))
blkU = {}
for nm in ('A|fresh', f'{FT}|fresh'):
    b, _ = s2lib.block_from(main[nm]['lg'], 5)
    for j, v in b.items():
        blkU.setdefault(j, []).extend(v)
res['V3_block_union'] = {str(P.COLS[j]): [[str(P.IDX[a].date()), str(P.IDX[b].date())] for a, b in v] for j, v in blkU.items()}
res['main_shift0_1m'] = {k: v['m'] for k, v in main.items()}
del main
(H / 'data').mkdir(exist_ok=True)
pd.DataFrame(tdiff).to_csv(H / 'data' / 'trades_diff.csv', index=False)
print('主跑完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

# ---------------------------------------------------------------- 20 條路徑（反事實）
PATH = {f'{c}|{m}': {} for c in CFG for m in ('fresh', 'cont')}
PATH['A|fresh_blk'] = {}; PATH[f'{FT}|fresh_blk'] = {}
ncells = {}
for s in range(4):
    SG, ncells[f'+{s}'] = (SG0, nc0) if s == 0 else signals(s)
    for cap in CAPS:
        k = f'{int(cap/1e4)}萬|+{s}'
        for c in CFG:
            nav = L.sim(P, U, SG[c], FCAP[cap], fresh=True)[0]
            sg = s17lib.seg_dd(nav)
            PATH[f'{c}|fresh'][k] = dict(valid=cg(nav, TB, DE), valid_dd=md(nav, TB, DE), since2024=cg(nav, D24, DE), seg={x: sg[x] for x in SEGV})
            nav = L.sim(P, U, SG[c], cap)[0]
            PATH[f'{c}|cont'][k] = dict(since2019=cg(nav, D19, DE), full=cg(nav, D0, DE), valid=cg(nav, TB, DE), since2024=cg(nav, D24, DE), maxdd=md(nav, D0, DE))
        for c in ('A', FT):
            PATH[f'{c}|fresh_blk'][k] = dict(valid=cg(L.sim(P, U, SG[c], FCAP[cap], fresh=True, block=blkU)[0], TB, DE))
    if s:
        del SG
    gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
keys = list(PATH['A|fresh'])


def pair(a, b, f, sub=None):
    g = (lambda d: d[f]) if sub is None else (lambda d: d[f][sub])
    d = np.array([g(PATH[a][k]) - g(PATH[b][k]) for k in keys]) * 100
    return dict(npos=int((d > 1e-12).sum()), nneg=int((d < -1e-12).sum()), median_pp=float(np.median(d)), min_pp=float(d.min()), max_pp=float(d.max()),
                abs_median_pp=float(np.median(np.abs(d))))


p = dict(valid=pair(f'{FT}|fresh', 'A|fresh', 'valid'), valid_dd=pair(f'{FT}|fresh', 'A|fresh', 'valid_dd'), since2024=pair(f'{FT}|fresh', 'A|fresh', 'since2024'),
         seg={x: pair(f'{FT}|fresh', 'A|fresh', 'seg', x) for x in SEGV}, blk=pair(f'{FT}|fresh_blk', 'A|fresh_blk', 'valid'))
V = dict(V1=p['valid']['npos'] >= 15, V2=p['valid']['median_pp'] >= 2.0, V3=p['blk']['median_pp'] > 0, V4=all(p['seg'][x]['median_pp'] >= -2.0 for x in SEGV))
V = {k: bool(v) for k, v in V.items()}
passed = bool(FZ['T0_pass'] and all(V.values()))
res['pairs_FZ_vs_A'] = p
res['verdict'] = dict(T0=FZ['T0_pass'], **V, n_pass_V=int(sum(V.values())), overall='通過' if passed else '不通過',
                      text=f'凍結處理 {FT} 在驗證期明顯優於現狀 A' if passed else '不通過：各處理差異視為雜訊內，改用事前登記的建議規則')
print('判定', json.dumps(res['verdict'], ensure_ascii=False), json.dumps(p, ensure_ascii=False)[:1500], flush=True)

# 描述：所有處理兩兩、對 CUR
desc = {}
for c in 'ABC':
    for b in ('CUR', 'A'):
        if c == b:
            continue
        desc[f'{c}_vs_{b}'] = dict(fresh={f: pair(f'{c}|fresh', f'{b}|fresh', f) for f in ('valid', 'valid_dd', 'since2024')},
                                   cont={f: pair(f'{c}|cont', f'{b}|cont', f) for f in ('since2019', 'full', 'valid', 'since2024', 'maxdd')})
res['pairs_desc'] = desc
res['paths_median'] = {nm: {f: float(np.median([D[k][f] for k in keys])) for f in D[keys[0]] if f != 'seg'} for nm, D in PATH.items()}
res['paths'] = PATH; res['n_cells_by_shift'] = ncells

# 建議規則（事前登記）
dist = {c: desc[f'{c}_vs_CUR']['cont']['since2019']['abs_median_pp'] for c in 'ABC'}
best = min(dist, key=lambda c: dist[c])
simple = ['A', 'C', 'B']
pick = next(c for c in simple if dist[c] - dist[best] <= 0.10 + 1e-12)
res['recommendation_rule'] = dict(distance_to_CUR_since2019_abs_median_pp=dist,
                                  distance_to_CUR_valid_fresh_abs_median_pp={c: desc[f'{c}_vs_CUR']['fresh']['valid']['abs_median_pp'] for c in 'ABC'},
                                  closest=best, pick=FT if passed else pick, applied='判定通過 → 採凍結處理' if passed else '不通過 → 距離最小者，與較簡單者相差 ≤ 0.10 個百分點時選較簡單者')
print('建議規則', json.dumps(res['recommendation_rule'], ensure_ascii=False), flush=True)

# ---------------------------------------------------------------- (3) 實際情境：只有營收月份 ≥ 2026-01 看不到
REAL = {c: {} for c in CFG}
for s in range(4):
    SG, n = signals(s, first_month=L.REAL_START)
    for cap in CAPS:
        k = f'{int(cap/1e4)}萬|+{s}'
        for c in CFG:
            nav, lg, _ = L.sim(P, U, SG[c], cap)
            REAL[c][k] = dict(since2019=cg(nav, D19, DE), since2024=cg(nav, D24, DE), full=cg(nav, D0, DE), maxdd=md(nav, D0, DE),
                              y2026=float(nav.iloc[-1] / nav.loc[lastb('2026-01-01')] - 1))
            if s == 0 and cap == 1e6 and c != 'CUR':
                g = lg[lg['code'].isin(L.GROUP) & (lg['ent_date'] >= pd.Timestamp('2025-06-01'))]
                REAL[c]['_group_trades_since_2025-06'] = [dict(code=y.code, ent=str(y.ent_date.date()), exit=str(y.exit_date.date()), why=y.why, ret=round(float(y.ret), 4)) for y in g.itertuples()]
            if s == 0 and cap == 1e6 and c == 'CUR':
                g = lg[lg['code'].isin(L.GROUP) & (lg['ent_date'] >= pd.Timestamp('2025-06-01'))]
                REAL[c]['_group_trades_since_2025-06'] = [dict(code=y.code, ent=str(y.ent_date.date()), exit=str(y.exit_date.date()), why=y.why, ret=round(float(y.ret), 4)) for y in g.itertuples()]
    del SG
    gc.collect()
rk = [k for k in REAL['CUR'] if not k.startswith('_')]
realsum = {}
for c in 'ABC':
    o = {}
    for f in ('since2019', 'since2024', 'full', 'maxdd', 'y2026'):
        d = np.array([REAL[c][k][f] - REAL['CUR'][k][f] for k in rk]) * 100
        o[f] = dict(median_pp=float(np.median(d)), min_pp=float(d.min()), max_pp=float(d.max()), nneg=int((d < -1e-12).sum()), npos=int((d > 1e-12).sum()))
    o['standard_path_pp'] = {f: (REAL[c]['100萬|+0'][f] - REAL['CUR']['100萬|+0'][f]) * 100 for f in ('since2019', 'since2024', 'y2026')}
    realsum[f'{c}_vs_CUR'] = o
res['real_2026'] = dict(paths=REAL, summary=realsum)
print('實際情境', json.dumps(realsum, ensure_ascii=False)[:2500], flush=True)
res['meta'].update(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss(), fresh_capital_scale=ratio)
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print(res['meta'])
