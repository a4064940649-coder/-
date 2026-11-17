#!/usr/bin/env python3
"""S50 驗證期（凍結後只跑一次）：凍結候選（或「不改」時訓練分數最高的候選，只描述）對現行，重新起跑 × 20 條，S25 的 V1～V4；
另做事後描述（不影響判定）：全期連續四個設定 × 20 條的 2016-03～2018-12、2024 起等分段；其餘候選的驗證窗配對差。
輸出 results.json。SPEC／PREREG／FROZEN 的 sha256 必須相符；results.json 已存在就拒絕重跑。
python3 -I -B validate.py [--dry]（乾跑：目標換成現行本身、只印配對差（應全為 0）、不做事後描述、不寫 results.json）"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s50lib as L
from s50lib import s17lib, s2lib, stlib

H = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for f in ('SPEC', 'PREREG', 'FROZEN'):
    assert sha(H / f'{f}.json') == (H / f'{f}.sha256').read_text().split()[0], f'{f}.json 與登記的 sha256 不符'
DRY = '--dry' in sys.argv
assert DRY or not (H / 'results.json').exists(), 'results.json 已存在：驗證期只跑一次，拒絕重跑'
FZ = json.loads((H / 'FROZEN.json').read_text())
TGT = 'CUR' if DRY else FZ['validate_target']
assert TGT in L.CANDS
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT, FMA = L.setup()
print('載入', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE = P.IDX[P.t0], P.IDX[-1]
D16, D19, D24, TB = lastb('2016-03-01'), lastb('2019-01-01'), lastb('2024-01-01'), lastb('2022-01-01')
SEGV = ('2022', '2024-07~2025-04')
cg = stlib.cagr
OTHERS = [c for c in ('C1', 'C2', 'C3') if c != TGT]

# ---- 起始資金縮放、驗收 A3
SS0, a1 = L.signals(P, U, RT, FMA, 0)
assert a1
nav1, lg1, _ = L.sim(P, U, SS0['CUR'], 1e6, fresh=True)
ratio = float(nav1.loc[TB] / 1e6)
acc = dict(A1_shift0=a1, A3_no_trade_before_2022=bool((lg1['ent_date'] >= pd.Timestamp('2022-01-01')).all()), ratio_nav_TB=ratio, first_entry=str(lg1['ent_date'].min().date()))
assert acc['A3_no_trade_before_2022'], acc
del nav1, lg1
FCAP = {cap: cap / ratio for cap in L.CAPS}

# ---- 主跑（重新起跑，100 萬、位移 0）：描述＋V3 禁買聯集
main = {}; blkU = {}; tops = {}
for c in dict.fromkeys(['CUR', TGT]):
    nav, lg, st = L.sim(P, U, SS0[c], FCAP[1e6], fresh=True)
    b, x = s2lib.block_from(lg, 5)
    for j, v in b.items():
        blkU.setdefault(j, []).extend(v)
    tops[c] = [dict(code=str(r.code), ent=str(pd.Timestamp(r.ent_date).date()), exit=str(pd.Timestamp(r.exit_date).date()), ret=float(r.ret), pnl=float(r.pnl)) for r in x.itertuples()]
    cl = lg[lg['why'] != 'open_end']
    main[c] = dict(valid=cg(nav, TB, DE), valid_dd=stlib.maxdd(nav, TB, DE), y2022_2023=cg(nav, TB, D24), since2024=cg(nav, D24, DE), seg=s17lib.seg_dd(nav),
                   n_closed=int(len(cl)), n_buy=int(st['n_buy']), avg_npos=float(st['avg_npos']), hold_mean=float(cl['hold_days'].mean()),
                   why_counts={str(k): int(v) for k, v in lg['why'].value_counts().items()},
                   top5_share=s17lib.top5_share(lg, nav.loc[TB:], float(nav.loc[TB])))
    del nav, lg, st
blk_txt = {str(P.COLS[j]): [[str(P.IDX[a].date()), str(P.IDX[b].date())] for a, b in v] for j, v in blkU.items()}
print('主跑完成', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

# ---- 20 條路徑
PATH = {k: {} for k in ['CUR|fresh', 'TGT|fresh', 'CUR|fresh_blk', 'TGT|fresh_blk']}
if not DRY:
    PATH.update({f'{c}|cont': {} for c in L.CANDS})
    PATH.update({f'{c}|fresh': {} for c in OTHERS})
for s in range(4):
    SS = SS0 if s == 0 else L.signals(P, U, RT, FMA, s)[0]
    for cap in L.CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        runs = [('CUR', 'CUR'), ('TGT', TGT)] + ([] if DRY else [(c, c) for c in OTHERS])
        for nm, c in runs:
            nav = L.sim(P, U, SS[c], FCAP[cap], fresh=True)[0]
            sg = s17lib.seg_dd(nav)
            PATH[f'{nm}|fresh'][key] = dict(valid=cg(nav, TB, DE), valid_dd=stlib.maxdd(nav, TB, DE), y2022_2023=cg(nav, TB, D24), since2024=cg(nav, D24, DE),
                                            seg={k: sg[k] for k in SEGV})
            if nm in ('CUR', 'TGT'):
                nav = L.sim(P, U, SS[c], FCAP[cap], fresh=True, block=blkU)[0]
                PATH[f'{nm}|fresh_blk'][key] = dict(valid=cg(nav, TB, DE))
        if not DRY:
            for c in L.CANDS:
                nav, lg, st = L.sim(P, U, SS[c], cap)
                PATH[f'{c}|cont'][key] = dict(seg16_18=cg(nav, D16, D19), since2024=cg(nav, D24, DE), since2019=cg(nav, D19, DE), y2019_2021=cg(nav, D19, TB),
                                              y2022_2023=cg(nav, TB, D24), full=cg(nav, D0, DE), maxdd_full=stlib.maxdd(nav, D0, DE))
                del lg, st
        del nav
    if s:
        del SS
    gc.collect()
    print('位移', s, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

keys = list(PATH['CUR|fresh'])


def pair(a, b, f, sub=None):
    g = (lambda d: d[f]) if sub is None else (lambda d: d[f][sub])
    d = np.array([g(PATH[a][k]) - g(PATH[b][k]) for k in keys]) * 100
    return dict(npos=int((d > 1e-9).sum()), n_ge025=int((d >= 0.25).sum()), median_pp=float(np.median(d)), min_pp=float(d.min()), max_pp=float(d.max()))


pv = dict(valid=pair('TGT|fresh', 'CUR|fresh', 'valid'), valid_dd=pair('TGT|fresh', 'CUR|fresh', 'valid_dd'),
          y2022_2023=pair('TGT|fresh', 'CUR|fresh', 'y2022_2023'), since2024=pair('TGT|fresh', 'CUR|fresh', 'since2024'),
          seg={sg: pair('TGT|fresh', 'CUR|fresh', 'seg', sg) for sg in SEGV}, blk=pair('TGT|fresh_blk', 'CUR|fresh_blk', 'valid'))
V = dict(V1=pv['valid']['npos'] >= 15, V2=pv['valid']['median_pp'] >= 2.0, V3=pv['blk']['median_pp'] > 0,
         V4=all(pv['seg'][sg]['median_pp'] >= -2.0 for sg in SEGV))
V = {k: bool(v) for k, v in V.items()}
if DRY:
    print('乾跑配對差（應全為 0）', json.dumps(pv, ensure_ascii=False))
    print('乾跑 OK' if all(abs(x) < 1e-12 for x in (pv['valid']['median_pp'], pv['blk']['median_pp'], pv['valid']['max_pp'], pv['valid']['min_pp'])) else '乾跑失敗',
          f'{time.time()-t0:.0f}s', rss(), 'MB')
    sys.exit(0)

med = lambda v: float(np.median(v))
summ = {}
for nm, D in PATH.items():
    summ[nm] = {f: med([D[k][f] for k in keys]) for f in D[keys[0]] if f != 'seg'}
    if 'seg' in D[keys[0]]:
        summ[nm]['seg'] = {sg: med([D[k]['seg'][sg] for k in keys]) for sg in SEGV}
adopt = bool(FZ['frozen'] != 'CUR' and FZ['T0_pass'] and all(V.values()))
verdict = dict(frozen=FZ['frozen'], validate_target=TGT, T0=FZ['T0_pass'], **V, n_pass_V=int(sum(V.values())),
               overall='建議改出場' if adopt else '不改',
               text=(f'{TGT} 通過 T0 與 V1～V4' if adopt else ('訓練期選「不改」，驗證只描述' if FZ['frozen'] == 'CUR' else '任一項不過 → 不改；不換候選重跑')))
posthoc = dict(note='事後描述，不影響判定',
               cont_vs_CUR={c: {f: pair(f'{c}|cont', 'CUR|cont', f) for f in ('seg16_18', 'since2024', 'since2019', 'y2019_2021', 'y2022_2023', 'full', 'maxdd_full')}
                            for c in ('C1', 'C2', 'C3')},
               others_valid_fresh={c: dict(valid=pair(f'{c}|fresh', 'CUR|fresh', 'valid'), y2022_2023=pair(f'{c}|fresh', 'CUR|fresh', 'y2022_2023'),
                                           since2024=pair(f'{c}|fresh', 'CUR|fresh', 'since2024'),
                                           seg={sg: pair(f'{c}|fresh', 'CUR|fresh', 'seg', sg) for sg in SEGV}) for c in OTHERS})
res = dict(SPEC_sha256=sha(H / 'SPEC.json'), PREREG_sha256=sha(H / 'PREREG.json'), FROZEN_sha256=sha(H / 'FROZEN.json'), validate_sha256=sha(Path(__file__)),
           frozen=FZ, acceptance=acc, window=dict(valid=[str(TB.date()), str(DE.date())], seg16_18=[str(D16.date()), str(D19.date())], since2024=[str(D24.date()), str(DE.date())]),
           verdict=verdict, pairs_TGT_vs_CUR=pv, main_fresh=main, V3_top5=tops, V3_block_union=blk_txt, paths_median=summ, posthoc=posthoc, paths=PATH, path_keys=keys,
           meta=dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=rss(), fresh_capital_scale=ratio))
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print('判定', json.dumps(verdict, ensure_ascii=False))
print('配對', json.dumps(pv, ensure_ascii=False))
print('事後', json.dumps(posthoc, ensure_ascii=False))
print(res['meta'])
