#!/usr/bin/env python3
"""S51 驗證期（凍結後只跑一次）：凍結候選 FZ 對現行 CUR（S70_D10）。由 S44 validate.py 改寫（只換「規則」為「配置格」）。
(1) 判定用：重新起跑（只保留營收月份 ≥ 2021-12 的訊號；起始資金縮放使 2021-12-30 淨值約＝60～200 萬）× 20 條路徑，驗證窗 2021-12-30 → 2026-10-22；
    S25 標準 T0（訓練期，取自 FROZEN）＋ V1～V4；另報驗證窗回撤與 Calmar（驗證窗年化 ÷ |驗證窗最大回撤|）。
(2) 描述用（PREREG 登記、不作判定）：另兩個候選在驗證期的同樣統計；全期連續主跑與 20 條路徑；容量（1,000 萬、5% 上限、全期連續、位移 0～3，S37 口徑 capped_rate）。
輸出 results.json。PREREG／SPEC／FROZEN 的 sha256 與登記時程式碼 sha256 必須相符；results.json 已存在就拒絕重跑。
乾跑 --dry：所有候選換成現行格（恆等），不需 FROZEN，輸出到 scratchpad/s51/dry。python3 -I -B validate.py [--dry]"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s51lib as L
from s51lib import s10lib, s2lib, stlib

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
    FZID = FZ['frozen_for_validation']
    assert FZ['frozen_cell'] == L.CANDS[FZID]
    T0 = bool(FZ['T0_pass']); SELECTED = FZ['selected']
    CAND = dict(L.CANDS)
else:
    FZ = dict(frozen_for_validation='C1', T0_pass=False, selected='DRY'); FZID = 'C1'; T0 = False; SELECTED = 'DRY'
    CAND = {k: L.CUR for k in L.CANDS}
OUT = L.TMP / 'dry' if DRY else H
OUT.mkdir(parents=True, exist_ok=True)
CFG = dict(CUR=L.CUR, **CAND)
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

# ---- 起始資金縮放（重新起跑；2022 前全在 0050，各格相同）
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
n1, l1, _ = L.sim(P, S0, U, L.CUR, cap=1e6, fresh=True)
ratio = float(n1.loc[TB] / 1e6)
acc = dict(A3_no_trade_before_2022=bool((l1['ent_date'] >= pd.Timestamp('2022-01-01')).all()), ratio_nav_TB=ratio, first_entry=str(l1['ent_date'].min().date()))
assert acc['A3_no_trade_before_2022'], acc
del n1, l1
FCAP = {cap: cap / ratio for cap in L.CAPS}

# ---- 主跑（位移 0）：重新起跑＋全期連續
res = dict(PREREG_sha256=(H / 'PREREG.sha256').read_text().split()[0] if not DRY else 'DRY',
           FROZEN_sha256=(H / 'FROZEN.sha256').read_text().split()[0] if not DRY else 'DRY',
           dry=DRY, cells=CFG, frozen=FZID, frozen_cell=CFG[FZID], selected=SELECTED, T0_pass=T0, acceptance=acc,
           meta=dict(start=str(D0.date()), valid_from=str(TB.date()), end=str(DE.date())),
           bench_0050={k: dict(cagr=cg(bench, k), maxdd=stlib.maxdd(bench, *WIN[k])) for k in WIN}, main={})
res['bench_0050']['seg'] = L.s17lib.seg_dd(bench)
mainlg = {}
for mode in ('fresh', 'cont'):
    fresh = mode == 'fresh'
    cap = FCAP[1e6] if fresh else 1e6
    b0 = TB if fresh else D0
    for c, cell in CFG.items():
        nav, lg, st = L.sim(P, S0, U, cell, cap=cap, fresh=fresh)
        mainlg[f'{c}|{mode}'] = lg
        x = lg[lg['ent_date'] >= pd.Timestamp('2022-01-01')] if fresh else lg
        ks = ('valid', 'y2022_2023', 'since2024') if fresh else tuple(WIN)
        nx = L.sim(P, S0, U, cell, cap=cap, cost='strict_x2', fresh=fresh)[0]
        mdd = stlib.maxdd(nav, b0, DE)
        R = dict(cagr={k: cg(nav, k) for k in ks}, maxdd=mdd, calmar=(cg(nav, 'valid' if fresh else 'since2019') / abs(mdd)), seg=L.s17lib.seg_dd(nav),
                 yearly=stlib.yearly(nav.loc[b0:], bench), n_closed=int((x['why'] != 'open_end').sum()), n_buy=int(st['n_buy']),
                 avg_npos=float(st['avg_npos']), avg_stockfrac=float(np.nanmean(st['stockfrac'].loc[b0:].to_numpy())),
                 top5_share=L.s17lib.top5_share(x, nav.loc[b0:], float(nav.loc[b0])),
                 costx2={k: cg(nx, k) for k in ks}, end_nav=float(nav.iloc[-1]))
        if fresh:
            top = s2lib.block_from(x, 5)[1]
            R['top5_trades'] = [dict(code=y.code, ent=str(y.ent_date.date()), exit=str(y.exit_date.date()), ret=float(y.ret), pnl=float(y.pnl)) for y in top.itertuples()]
        res['main'][f'{c}|{mode}'] = R
        del nav, nx, st
    gc.collect()
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
    for cap in L.CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        for c, cell in CFG.items():
            nav = L.sim(P, S, U, cell, cap=FCAP[cap], fresh=True)[0]
            sg = L.s17lib.seg_dd(nav); vdd = stlib.maxdd(nav, TB, DE); v = cg(nav, 'valid')
            PATH[f'{c}|fresh'][key] = dict(valid=v, valid_dd=vdd, valid_calmar=v / abs(vdd), y2022_2023=cg(nav, 'y2022_2023'), since2024=cg(nav, 'since2024'),
                                           seg={k: sg[k] for k in L.SEGV})
            nav = L.sim(P, S, U, cell, cap=cap)[0]
            mdd = stlib.maxdd(nav, D0, DE); s19 = cg(nav, 'since2019')
            PATH[f'{c}|cont'][key] = dict(since2019=s19, full=cg(nav, 'full'), valid=cg(nav, 'valid'), maxdd=mdd, calmar=s19 / abs(mdd), seg=L.s17lib.seg_dd(nav))
            if c != 'CUR':
                PATH[f'{c}|fresh_blk'][key] = dict(valid=cg(L.sim(P, S, U, cell, cap=FCAP[cap], fresh=True, block=BLK[c])[0], 'valid'))
                PATH[f'CUR|fresh_blk_{c}'][key] = dict(valid=cg(L.sim(P, S, U, L.CUR, cap=FCAP[cap], fresh=True, block=BLK[c])[0], 'valid'))
            del nav
    for c, cell in CFG.items():                                # 容量：1,000 萬、5% 上限、全期連續
        _, _, st = L.sim(P, S, U, cell, cap=1e7)
        CAPAC[c][f'1000萬|+{s}'] = L.capped_rate(st)
    if s:
        del S
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
cap_s = {}
for c in CFG:
    r = [CAPAC[c][f'1000萬|+{s}']['capped_rate'] for s in L.SHIFTS]
    cap_s[c] = dict(shift0=r[0], median4=med(r), min4=float(min(r)), max4=float(max(r)),
                    sell_partial_median4=med([CAPAC[c][f'1000萬|+{s}']['sell_partial_cap'] for s in L.SHIFTS]))
res['capacity_1000w'] = dict(runs=CAPAC, summary=cap_s, note='S37 口徑：capped_rate＝被 5% 當日成交值上限縮量的買進嘗試 ÷ 算出股數的買進嘗試；全期連續、嚴成本、位移 0～3。')

# ---- 判定（只用凍結候選）
V = pv[FZID]['V']
passed = bool(SELECTED != '不改' and SELECTED == FZID and T0 and all(V.values()))
res['verdict'] = dict(selected=SELECTED, T0=T0, **V, n_pass_V=int(sum(V.values())), overall='通過' if passed else '不通過',
                      text=('凍結候選通過訓練選擇、T0 與 V1～V4：建議改' if passed else '不通過：維持現行 S70_D10，不改每檔比例'))
res['meta'].update(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss(), fresh_capital_scale=ratio)
(OUT / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print('判定', json.dumps(res['verdict'], ensure_ascii=False))
print(json.dumps({c: dict(V=pv[c]['V'], valid=pv[c]['fresh']['valid'], dd=pv[c]['fresh']['valid_dd'], cal=pv[c]['fresh']['valid_calmar'], blk=pv[c]['fresh_blk']['valid'],
                          seg={k: v['median'] for k, v in pv[c]['fresh']['seg'].items()}) for c in CAND}, ensure_ascii=False))
print(json.dumps(cap_s, ensure_ascii=False))
print(res['meta'])
