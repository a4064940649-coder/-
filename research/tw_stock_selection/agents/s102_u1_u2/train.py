#!/usr/bin/env python3
"""S102 訓練期：U1 3 格、U2 2 格 × 20 條路徑（全期連續模擬，取訓練窗 2016-01-30 → 2021-12-30）→ 兩題各自依 PREREG 選擇 → 寫 FROZEN.json＋sha256。
PREREG.json 與 code.sha256 必須相符；FROZEN.json 已存在就拒絕重跑。只輸出訓練窗的數字（驗證窗一個都不算、不印）。
用法：python3 -I -B train.py [--dry]（--dry：只跑位移 0、100 萬，不寫檔、不印任何報酬）"""
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
import s102lib as L
from s102lib import stlib
t_all = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT = L.load()
assert (P.T, str(P.IDX[-1].date())) == (2876, '2026-10-22')
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TA, TB, T18 = lastb('2016-02-01'), lastb('2022-01-01'), lastb('2019-01-01')
assert (str(TA.date()), str(TB.date())) == ('2016-01-30', '2021-12-30')
TOPICS = {'U1': list(L.U1_CELLS), 'U2': list(L.U2_CELLS)}
CELLS = list(L.CELLS); CFG = ['CUR'] + CELLS
bench = pd.Series(P.E50, index=P.IDX)


def tr_metrics(nav, lg, st, S, cell):
    x = lg[(lg['sig_date'] > TA) & (lg['sig_date'] <= TB)]
    o = dict(train=stlib.cagr(nav, TA, TB), train_dd=stlib.maxdd(nav, TA, TB), y1618=stlib.cagr(nav, TA, T18), y1921=stlib.cagr(nav, T18, TB),
             n_trades=int(len(x)), avg_stockfrac=float(np.nanmean(st['stockfrac'].loc[TA:TB].to_numpy())))
    if cell != 'CUR' and L.CELLS[cell]['topic'] == 'U1':
        fk = S['FILL']; o['n_extra'] = int(sum(int(j) in set(fk.get(int(k), [])) for j, k in zip(x['j'], x['sig_t'])))
    elif cell != 'CUR':
        rb = st['rebuys']; d = P.IDX[rb['t'].to_numpy()] if len(rb) else pd.DatetimeIndex([])
        o['n_extra'] = int(((d > TA) & (d <= TB)).sum())
    return o


M = {c: {} for c in CFG}; SC = {c: {} for c in CELLS}
for s in ((0,) if DRY else L.SHIFTS):
    SG = {c: L.signal(P, U, RT, s, c) for c in ['CUR'] + TOPICS['U1']}
    for cap in ((1e6,) if DRY else L.CAPS):
        key = f'{int(cap/1e4)}萬|+{s}'
        navC, lg, st = L.sim(P, SG['CUR'], U, cap=cap); M['CUR'][key] = tr_metrics(navC, lg, st, SG['CUR'], 'CUR'); sfC = M['CUR'][key]['avg_stockfrac']; del lg, st
        for c in CELLS:                                        # 一次一格：跑 → 取數 → 釋放
            S = SG[c] if c in SG else SG['CUR']
            nav, lg, st = L.sim(P, S, U, cap=cap, cell=c)
            M[c][key] = tr_metrics(nav, lg, st, S, c)
            lam = M[c][key]['avg_stockfrac'] / sfC             # 描述：現行線性縮放到同曝險（訓練窗）
            sc = L.scaled_current(navC, bench, lam, TA, TB)
            SC[c][key] = dict(lam=float(lam), scaled_train=stlib.cagr(sc, TA, TB)); del nav, lg, st, sc
        del navC; gc.collect()
    del SG; gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
keys = list(M['CUR'])
g = lambda c, f: np.array([M[c][k][f] for k in keys])
res = dict(cells=L.CELLS, window=dict(train=[str(TA.date()), str(TB.date())], y1618=[str(TA.date()), str(T18.date())], y1921=[str(T18.date()), str(TB.date())]), paths=M, scaled_current=SC, summary={})
for c in CFG:
    o = dict(train_median=float(np.median(g(c, 'train'))), train_min=float(g(c, 'train').min()), train_max=float(g(c, 'train').max()), train_dd_median=float(np.median(g(c, 'train_dd'))),
             n_trades_median=float(np.median(g(c, 'n_trades'))), avg_stockfrac_median=float(np.median(g(c, 'avg_stockfrac'))))
    if c != 'CUR':
        d = (g(c, 'train') - g('CUR', 'train')) * 100; dd = (g(c, 'train_dd') - g('CUR', 'train_dd')) * 100
        d1 = (g(c, 'y1618') - g('CUR', 'y1618')) * 100; d2 = (g(c, 'y1921') - g('CUR', 'y1921')) * 100
        ds = (g(c, 'train') - np.array([SC[c][k]['scaled_train'] for k in keys])) * 100
        o.update(score_paired_median_pp=float(np.median(d)), npos=int((d > 0).sum()), n_ge025=int((d >= 0.25).sum()), paired_min_pp=float(d.min()), paired_max_pp=float(d.max()),
                 paired_dd_median_pp=float(np.median(dd)), y1618_median_pp=float(np.median(d1)), y1921_median_pp=float(np.median(d2)), n_extra_median=float(np.median(g(c, 'n_extra'))),
                 stockfrac_minus_cur_pp=float(np.median(g(c, 'avg_stockfrac') - g('CUR', 'avg_stockfrac')) * 100),
                 vs_scaled_current_median_pp=float(np.median(ds)), vs_scaled_current_npos=int((ds > 0).sum()), lam_median=float(np.median([SC[c][k]['lam'] for k in keys])))
    res['summary'][c] = o
    if DRY:
        print(c, {k: v for k, v in o.items() if k in ('n_trades_median', 'n_extra_median', 'avg_stockfrac_median')}, flush=True)   # 試跑不印任何報酬數字
        continue
    print(c, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()}, flush=True)
if DRY:
    print('DRY OK', round(time.time() - t_all), 's', rss(), 'MB'); sys.exit(0)
assert len(keys) == 20
res['PREREG_sha256'] = PSHA; res['selection'] = {}
FZ = dict(written_at=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), PREREG_sha256=PSHA, topics={},
          note='兩題各自凍結；凍結後 validate.py 只跑一次；不得換格、不得換題。selected＝「不改」時，仍以訓練期分數最高的格跑驗證期一次，只作描述，該題總判定必為不通過。')
for tp, cells in TOPICS.items():
    order = sorted(cells, key=lambda c: (res['summary'][c]['score_paired_median_pp'], res['summary'][c]['n_ge025'], -cells.index(c)), reverse=True)
    top = order[0]; ts = res['summary'][top]
    t0 = bool(ts['score_paired_median_pp'] >= 1.0 and ts['n_ge025'] >= 14)
    selected = top if t0 else '不改'
    sel = dict(order=order, top=top, T0_pass=t0, selected=selected, frozen_for_validation=top,
               T0_text=f"T0：分數（成對差中位數）{ts['score_paired_median_pp']:+.2f} 個百分點、≥ +0.25 的路徑 {ts['n_ge025']}/20")
    res['selection'][tp] = sel
    FZ['topics'][tp] = dict(selected=selected, frozen_for_validation=top, frozen_params=L.CELLS[top], T0_pass=t0, selection=sel, train_summary=ts)
FZ['train_summary_cur'] = res['summary']['CUR']
res['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss())
(HERE / 'results_train.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
(HERE / 'FROZEN.json').write_text(json.dumps(FZ, ensure_ascii=False, indent=1))
(HERE / 'FROZEN.sha256').write_text(f"{sha(HERE / 'FROZEN.json')}  FROZEN.json\n")
print('選擇', json.dumps(res['selection'], ensure_ascii=False)); print('凍結', (HERE / 'FROZEN.sha256').read_text(), res['meta'])
