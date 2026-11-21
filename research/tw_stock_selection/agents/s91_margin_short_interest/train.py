#!/usr/bin/env python3
"""S91 訓練期：現行＋28 格（14 條融資／融券／借券規則 × Z 放 0050／F 往下遞補；延遲 1 日），各 20 條路徑（5 種資金 × 4 個位移），
連續模擬、只量訓練窗（2016-01-30 前最後交易日 → 2021-12-30，＝訊號 2016-02～2021-12）；依 PREREG 的選擇規則選一格或「不改」，寫 FROZEN.json。
由 S88 train.py 改寫（選擇規則、T0 同 S25 標準）。PREREG.json 的 sha256 與登記時程式碼 sha256 必須相符；FROZEN.json 已存在就拒絕重跑。
乾跑 --dry：所有格子換成現行（恆等），只測程式路徑，輸出到 scratchpad，不寫 FROZEN。python3 -I -B train.py [--dry]"""
import sys, json, hashlib, time, resource, gc, datetime
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s91lib as L
from s91lib import s10lib, stlib

H = Path(__file__).resolve().parent
DRY = '--dry' in sys.argv
if not DRY:
    assert hashlib.sha256((H / 'PREREG.json').read_bytes()).hexdigest() == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與 sha256 不符'
    PR = json.loads((H / 'PREREG.json').read_text())
    for n, v in PR['code_sha256_at_write'].items():
        assert hashlib.sha256((H / n).read_bytes()).hexdigest() == v, f'{n} 在登記後被改過'
    assert not (H / 'FROZEN.json').exists(), 'FROZEN.json 已存在，拒絕重跑'
    PSHA = (H / 'PREREG.sha256').read_text().split()[0]
else:
    PSHA = 'DRY'
OUT = L.TMP.parent / 'dry' if DRY else H
OUT.mkdir(parents=True, exist_ok=True)
assert len(L.CELLS) == 28
CFG = dict(CUR='CUR', **{c: ('CUR' if DRY else c) for c in L.CELLS})
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT = L.load()
CH = L.Chips(P)
print('載入完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TA, TB, T18 = lastb('2016-02-01'), lastb('2022-01-01'), lastb('2019-01-01')


def tr_metrics(nav, lg, st):
    x = lg[(lg['sig_date'] > TA) & (lg['sig_date'] <= TB)]
    return dict(train=stlib.cagr(nav, TA, TB), train_dd=stlib.maxdd(nav, TA, TB), y1618=stlib.cagr(nav, TA, T18), y1921=stlib.cagr(nav, T18, TB),
                n_trades=int(len(x)), avg_stockfrac=float(np.nanmean(st['stockfrac'].loc[TA:TB].to_numpy())))


M = {c: {} for c in CFG}
REC = {}
for s in L.SHIFTS:
    S = s10lib.make_signal_s10(P, U, RT, shift=s)
    for c, cell in CFG.items():
        S2, rec = L.apply_cell(P, U, RT, CH, S, cell, shift=s)
        if s == 0:
            rr = [r for r in rec if '2016-01' <= r['month'] <= '2021-11']
            REC[c] = dict(n_orig=sum(len(r['orig']) for r in rr), n_removed=sum(len(r['removed']) for r in rr), n_added=sum(len(r['added']) for r in rr),
                          months_changed=sum(1 for r in rr if r['removed']))
        for cap in L.CAPS:
            M[c][f'{int(cap/1e4)}萬|+{s}'] = tr_metrics(*L.sim(P, S2, U, cap=cap))
        del S2
    del S; gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

keys = list(M['CUR'])
assert len(keys) == 20
g = lambda c, f: np.array([M[c][k][f] for k in keys])
res = dict(PREREG_sha256=PSHA, dry=DRY, cells=CFG, window=dict(train=[str(TA.date()), str(TB.date())], y1618=[str(TA.date()), str(T18.date())],
           y1921=[str(T18.date()), str(TB.date())]), paths=M, summary={}, rule_effect_shift0=REC)
for c in CFG:
    o = dict(train_median=float(np.median(g(c, 'train'))), train_min=float(g(c, 'train').min()), train_max=float(g(c, 'train').max()),
             train_dd_median=float(np.median(g(c, 'train_dd'))), n_trades_median=float(np.median(g(c, 'n_trades'))), avg_stockfrac_median=float(np.median(g(c, 'avg_stockfrac'))))
    o['train_calmar_median'] = float(np.median(g(c, 'train') / np.abs(g(c, 'train_dd'))))
    if c != 'CUR':
        d = (g(c, 'train') - g('CUR', 'train')) * 100
        dd = (g(c, 'train_dd') - g('CUR', 'train_dd')) * 100
        d1 = (g(c, 'y1618') - g('CUR', 'y1618')) * 100
        d2 = (g(c, 'y1921') - g('CUR', 'y1921')) * 100
        o.update(median_minus_cur_pp=(o['train_median'] - float(np.median(g('CUR', 'train')))) * 100,
                 dd_median_minus_cur_pp=(o['train_dd_median'] - float(np.median(g('CUR', 'train_dd')))) * 100,
                 paired_median_pp=float(np.median(d)), npos=int((d > 0).sum()), n_ge025=int((d >= 0.25).sum()), paired_min_pp=float(d.min()), paired_max_pp=float(d.max()),
                 paired_dd_median_pp=float(np.median(dd)), n_dd_shallower=int((dd > 0).sum()),
                 y1618_median_pp=float(np.median(d1)), y1618_npos=int((d1 > 0).sum()), y1921_median_pp=float(np.median(d2)), y1921_npos=int((d2 > 0).sum()),
                 by_shift_median_pp={f'+{s}': float(np.median([(M[c][k]['train'] - M['CUR'][k]['train']) * 100 for k in keys if k.endswith(f'+{s}')])) for s in L.SHIFTS})
    res['summary'][c] = o

# ---- 選擇（PREREG 規則）
order = sorted(L.CELLS, key=lambda c: (res['summary'][c]['train_median'], res['summary'][c]['n_ge025'], -L.CELLS.index(c)), reverse=True)
top = order[0]; ts = res['summary'][top]
cond_ret = bool(ts['median_minus_cur_pp'] >= 1.0)
cond_dd = bool(ts['dd_median_minus_cur_pp'] >= -2.0)
cond_half = bool(ts['y1618_median_pp'] > 0 and ts['y1921_median_pp'] > 0)
selected = top if (cond_ret and cond_dd and cond_half) else '不改'
t0 = bool(ts['paired_median_pp'] >= 1.0 and ts['n_ge025'] >= 14)
res['selection'] = dict(order=order, top=top, cond_return_ge_cur_plus1=cond_ret, cond_dd_not_deeper_than_cur_minus2=cond_dd, cond_both_halves_positive=cond_half,
                        selected=selected, frozen_for_validation=top, T0_pass=t0,
                        T0_text=f"T0：成對中位數 {ts['paired_median_pp']:+.2f} 個百分點、≥ +0.25 的路徑 {ts['n_ge025']}/20")
res['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss())
(OUT / 'results_train.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
for c in CFG:
    o = res['summary'][c]
    print(c, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items() if k in ('train_median', 'train_dd_median', 'median_minus_cur_pp', 'paired_median_pp', 'npos', 'n_ge025', 'dd_median_minus_cur_pp', 'y1618_median_pp', 'y1921_median_pp', 'n_trades_median')}, REC[c], flush=True)
print('選擇', json.dumps(res['selection'], ensure_ascii=False))
if not DRY:
    FZ = dict(written_at=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), PREREG_sha256=PSHA, selected=selected, frozen_for_validation=top,
              T0_pass=t0, selection=res['selection'], train_summary=ts, train_summary_cur=res['summary']['CUR'],
              note='凍結後 validate.py 只跑一次；不得換格。selected＝「不改」時，仍以訓練期中位數最高的格跑驗證期，只作描述，總判定為不通過。')
    txt = json.dumps(FZ, ensure_ascii=False, indent=1, default=str)
    (H / 'FROZEN.json').write_text(txt)
    fsha = hashlib.sha256(txt.encode()).hexdigest()
    (H / 'FROZEN.sha256').write_text(f'{fsha}  FROZEN.json\n')
    print('凍結', top, 'selected', selected, 'T0', t0, fsha)
print(res['meta'])
