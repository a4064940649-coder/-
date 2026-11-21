#!/usr/bin/env python3
"""S90 訓練期：30 格 × 20 條路徑（全期連續模擬，取訓練窗 2016-01-29 → 2021-12-30）→ 依 PREREG 選擇 → 寫 FROZEN.json＋sha256。
PREREG.json 與 code.sha256 必須相符；FROZEN.json 已存在就拒絕重跑。只輸出訓練窗的數字。用法：python3 -I -B train.py [--dry]（--dry：只跑位移 0、100 萬、前 3 格，不寫檔）"""
import sys, json, time, resource, gc, hashlib, datetime
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
DRY = '--dry' in sys.argv
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
if not DRY:
    PSHA = sha(HERE / 'PREREG.json')
    assert PSHA == (HERE / 'PREREG.sha256').read_text().split()[0], 'PREREG 不符'
    for ln in (HERE / 'code.sha256').read_text().splitlines():
        h, f = ln.split()
        assert sha(HERE / f) == h, f'{f} 與登記時不同'
    assert not (HERE / 'FROZEN.json').exists(), 'FROZEN.json 已存在'
sys.path.insert(0, str(HERE))
import s90lib as L
from s90lib import stlib
t_all = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT = L.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22')
FW = L.Flow(P, U)
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TA, TB, T18 = lastb('2016-02-01'), lastb('2022-01-01'), lastb('2019-01-01')
kTA, kTB = int(P.IDX.get_loc(TA)), int(P.IDX.get_loc(TB))
CELLS = list(L.CELLS)[:3] + ['RK_A20', 'WD_TFB20'] if DRY else list(L.CELLS)
CFG = ['CUR'] + CELLS


def tr_metrics(nav, lg, st):
    x = lg[(lg['sig_date'] > TA) & (lg['sig_date'] <= TB)]
    return dict(train=stlib.cagr(nav, TA, TB), train_dd=stlib.maxdd(nav, TA, TB), y1618=stlib.cagr(nav, TA, T18), y1921=stlib.cagr(nav, T18, TB),
                n_trades=int(len(x)), avg_stockfrac=float(np.nanmean(st['stockfrac'].loc[TA:TB].to_numpy())))


M = {c: {} for c in CFG}; REC = {}
for s in ((0,) if DRY else L.SHIFTS):
    S = L.signal(P, U, RT, s); FW._c.clear()
    for c in CFG:
        S2, rec = L.apply_cell(P, U, RT, FW, S, c)
        if s == 0 and c != 'CUR':
            rt = [r for r in rec if kTA < r['k'] <= kTB]
            REC[c] = dict(n_months=len(rt), n_orig=sum(len(r['orig']) for r in rt), n_removed=sum(len(r['removed']) for r in rt), n_added=sum(len(r['added']) for r in rt))
        for cap in ((1e6,) if DRY else L.CAPS):
            M[c][f'{int(cap/1e4)}萬|+{s}'] = tr_metrics(*L.sim(P, S2, U, cap=cap))
        del S2, rec; gc.collect()
    del S; gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
keys = list(M['CUR'])
g = lambda c, f: np.array([M[c][k][f] for k in keys])
res = dict(cells=CELLS, window=dict(train=[str(TA.date()), str(TB.date())], y1618=[str(TA.date()), str(T18.date())], y1921=[str(T18.date()), str(TB.date())]), paths=M, list_changes_shift0_train=REC, summary={})
for c in CFG:
    o = dict(train_median=float(np.median(g(c, 'train'))), train_min=float(g(c, 'train').min()), train_max=float(g(c, 'train').max()), train_dd_median=float(np.median(g(c, 'train_dd'))),
             n_trades_median=float(np.median(g(c, 'n_trades'))), avg_stockfrac_median=float(np.median(g(c, 'avg_stockfrac'))))
    if c != 'CUR':
        d = (g(c, 'train') - g('CUR', 'train')) * 100; dd = (g(c, 'train_dd') - g('CUR', 'train_dd')) * 100
        d1 = (g(c, 'y1618') - g('CUR', 'y1618')) * 100; d2 = (g(c, 'y1921') - g('CUR', 'y1921')) * 100
        o.update(median_minus_cur_pp=(o['train_median'] - float(np.median(g('CUR', 'train')))) * 100, dd_median_minus_cur_pp=(o['train_dd_median'] - float(np.median(g('CUR', 'train_dd')))) * 100,
                 paired_median_pp=float(np.median(d)), npos=int((d > 0).sum()), n_ge025=int((d >= 0.25).sum()), paired_min_pp=float(d.min()), paired_max_pp=float(d.max()),
                 paired_dd_median_pp=float(np.median(dd)), y1618_median_pp=float(np.median(d1)), y1921_median_pp=float(np.median(d2)), **REC.get(c, {}))
    res['summary'][c] = o
    print(c, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items() if k in ('train_median', 'train_dd_median', 'median_minus_cur_pp', 'paired_median_pp', 'npos', 'n_ge025', 'y1618_median_pp', 'y1921_median_pp', 'n_removed', 'n_added', 'n_orig')}, flush=True)
if DRY:
    print('DRY OK', round(time.time() - t_all), 's', rss(), 'MB'); sys.exit(0)
assert len(keys) == 20
order = sorted(CELLS, key=lambda c: (res['summary'][c]['train_median'], res['summary'][c]['n_ge025'], -CELLS.index(c)), reverse=True)
top = order[0]; ts = res['summary'][top]
cond_ret = bool(ts['median_minus_cur_pp'] >= 1.0); cond_dd = bool(ts['dd_median_minus_cur_pp'] >= -2.0); cond_half = bool(ts['y1618_median_pp'] > 0 and ts['y1921_median_pp'] > 0)
selected = top if (cond_ret and cond_dd and cond_half) else '不改'
t0 = bool(ts['paired_median_pp'] >= 1.0 and ts['n_ge025'] >= 14)
res['PREREG_sha256'] = PSHA
res['selection'] = dict(order=order, top=top, cond_return_ge_cur_plus1=cond_ret, cond_dd_not_deeper_than_cur_minus2=cond_dd, cond_both_halves_positive=cond_half, selected=selected,
                        frozen_for_validation=top, T0_pass=t0, T0_text=f"T0：成對中位數 {ts['paired_median_pp']:+.2f} 個百分點、≥ +0.25 的路徑 {ts['n_ge025']}/20")
res['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss())
(HERE / 'results_train.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
FZ = dict(written_at=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), PREREG_sha256=PSHA, selected=selected, frozen_for_validation=top, T0_pass=t0,
          selection=res['selection'], train_summary=ts, train_summary_cur=res['summary']['CUR'],
          note='凍結後 validate.py 只跑一次；不得換格。selected＝「不改」時，仍以訓練期中位數最高的格跑驗證期，只作描述，總判定為不通過。')
(HERE / 'FROZEN.json').write_text(json.dumps(FZ, ensure_ascii=False, indent=1))
(HERE / 'FROZEN.sha256').write_text(f"{sha(HERE / 'FROZEN.json')}  FROZEN.json\n")
print('選擇', json.dumps(res['selection'], ensure_ascii=False)); print('凍結', (HERE / 'FROZEN.sha256').read_text(), res['meta'])
