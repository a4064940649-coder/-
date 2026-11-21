#!/usr/bin/env python3
"""S87 訓練期：現行＋16 格（池 × 回看 × 規則 × 範圍），各 20 條路徑（5 種資金 × 4 個位移），連續模擬、只量訓練窗
（2016-01-30 → 2021-12-30＝訊號 2016-02～2021-12）。載具的換倉決定只做到 2021-12-30 之前的月底（之後一律維持不動，訓練窗量測不受影響）。
依 PREREG 規則選一格或「不改」，寫 FROZEN.json＋sha256。PREREG.json 的 sha256 與登記時程式碼 sha256 必須相符；FROZEN.json 已存在就拒絕重跑。
乾跑 --dry：16 格都換成現行（V=None），只測程式路徑，輸出到 scratchpad/s87/dry。python3 -I -B train.py [--dry]"""
import sys, json, hashlib, time, resource, gc, datetime
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s87lib as L
from s87lib import s10lib, stlib

H = Path(__file__).resolve().parent
DRY = '--dry' in sys.argv
if not DRY:
    assert hashlib.sha256((H / 'PREREG.json').read_bytes()).hexdigest() == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與 sha256 不符'
    PRE = json.loads((H / 'PREREG.json').read_text())
    for n, v in PRE['code_sha256_at_write'].items():
        assert hashlib.sha256((H / n).read_bytes()).hexdigest() == v, f'{n} 在登記後被改過'
    assert not (H / 'FROZEN.json').exists(), 'FROZEN.json 已存在，拒絕重跑'
    PSHA = (H / 'PREREG.sha256').read_text().split()[0]
else:
    PSHA = 'DRY'
OUT = L.TMP / 'dry' if DRY else H
OUT.mkdir(parents=True, exist_ok=True)
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = L.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
U, RT = L.base(P)
L.s17lib.e50_fix_suspension(P)
L.slim(P); gc.collect()
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TA, TB, T18 = lastb('2016-02-01'), lastb('2022-01-01'), lastb('2019-01-01')
iTB = int(np.flatnonzero(P.IDX == TB)[0])
E = L.etf_panel(P)
IDS = [c['id'] for c in L.CELLS]
VEH = {'CUR': None}; VINFO = {}
e50 = pd.Series(P.E50, index=P.IDX)
for cell in L.CELLS:
    if DRY:
        VEH[cell['id']] = None; continue
    dec, rec = L.decisions(P, E, cell)
    V = L.vehicle(P, E, {t: w for t, w in dec.items() if t <= iTB})          # 換倉列 ≤ 2021-12-30（決定日 ≤ 2021-12-29 之前的月底）
    VEH[cell['id']] = V
    vs = pd.Series(V['Dc'], index=P.IDX)
    VINFO[cell['id']] = dict(vehicle_train_cagr=stlib.cagr(vs, TA, TB), e50_train_cagr=stlib.cagr(e50, TA, TB), vehicle_train_dd=stlib.maxdd(vs, TA, TB),
                             e50_train_dd=stlib.maxdd(e50, TA, TB), n_switch=V['n_trades'], cost_sum=V['cost_sum'])
print('載入完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)


def tr_metrics(nav, lg, st):
    return dict(train=stlib.cagr(nav, TA, TB), train_dd=stlib.maxdd(nav, TA, TB), y1618=stlib.cagr(nav, TA, T18), y1921=stlib.cagr(nav, T18, TB))


M = {c: {} for c in VEH}
for s in L.SHIFTS:
    S = s10lib.make_signal_s10(P, U, RT, shift=s)
    for c, V in VEH.items():
        for cap in L.CAPS:
            M[c][f'{int(cap/1e4)}萬|+{s}'] = tr_metrics(*L.sim(P, S, U, V=V, cap=cap))
    del S; gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

keys = list(M['CUR'])
assert len(keys) == 20
g = lambda c, f: np.array([M[c][k][f] for k in keys])
res = dict(PREREG_sha256=PSHA, dry=DRY, window=dict(train=[str(TA.date()), str(TB.date())]), paths=M, summary={}, vehicle=VINFO)
for c in VEH:
    o = dict(train_median=float(np.median(g(c, 'train'))), train_min=float(g(c, 'train').min()), train_max=float(g(c, 'train').max()),
             train_dd_median=float(np.median(g(c, 'train_dd'))))
    if c != 'CUR':
        d = (g(c, 'train') - g('CUR', 'train')) * 100
        dd = (g(c, 'train_dd') - g('CUR', 'train_dd')) * 100
        d1 = (g(c, 'y1618') - g('CUR', 'y1618')) * 100
        d2 = (g(c, 'y1921') - g('CUR', 'y1921')) * 100
        o.update(median_minus_cur_pp=(o['train_median'] - float(np.median(g('CUR', 'train')))) * 100,
                 dd_median_minus_cur_pp=(o['train_dd_median'] - float(np.median(g('CUR', 'train_dd')))) * 100,
                 paired_median_pp=float(np.median(d)), npos=int((d > 0).sum()), n_ge025=int((d >= 0.25).sum()), paired_min_pp=float(d.min()), paired_max_pp=float(d.max()),
                 paired_dd_median_pp=float(np.median(dd)), y1618_median_pp=float(np.median(d1)), y1921_median_pp=float(np.median(d2)))
    res['summary'][c] = o

order = sorted(IDS, key=lambda c: (res['summary'][c]['train_median'], res['summary'][c]['n_ge025'], -IDS.index(c)), reverse=True)
top = order[0]; ts = res['summary'][top]
cond_ret = bool(ts['median_minus_cur_pp'] >= 1.0)
cond_dd = bool(ts['dd_median_minus_cur_pp'] >= -2.0)
selected = top if (cond_ret and cond_dd) else '不改'
t0 = bool(ts['paired_median_pp'] >= 1.0 and ts['n_ge025'] >= 14)
n_ok = int(sum(1 for c in IDS if res['summary'][c]['paired_median_pp'] >= 1.0 and res['summary'][c]['n_ge025'] >= 14))
res['selection'] = dict(order=order, top=top, cond_return_ge_cur_plus1=cond_ret, cond_dd_not_deeper_than_cur_minus2=cond_dd, selected=selected,
                        frozen_for_validation=top, T0_pass=t0, n_cells_pass_T0=n_ok,
                        T0_text=f"T0：成對中位數 {ts['paired_median_pp']:+.2f} 個百分點、≥ +0.25 的路徑 {ts['n_ge025']}/20")
res['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss())
(OUT / 'results_train.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
for c in ['CUR'] + order:
    print(c, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res['summary'][c].items()}, flush=True)
print('選擇', json.dumps(res['selection'], ensure_ascii=False))
if not DRY:
    FZ = dict(written_at=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), PREREG_sha256=PSHA, selected=selected, frozen_for_validation=top,
              frozen_cell=[c for c in L.CELLS if c['id'] == top][0], T0_pass=t0, selection=res['selection'], train_summary=ts,
              train_summary_cur=res['summary']['CUR'],
              note='凍結後 validate.py 只跑一次；不得換格。selected＝「不改」時仍以訓練期中位數最高的格跑驗證期，只作描述，總判定為不通過。')
    txt = json.dumps(FZ, ensure_ascii=False, indent=1, default=str)
    (H / 'FROZEN.json').write_text(txt)
    fsha = hashlib.sha256(txt.encode()).hexdigest()
    (H / 'FROZEN.sha256').write_text(f'{fsha}  FROZEN.json\n')
    code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(H.glob('*.py'))}
    (H / 'code.sha256').write_text(''.join(f'{v}  {k}\n' for k, v in code.items()))
    print('凍結', top, 'selected', selected, 'T0', t0, fsha)
print(res['meta'])
