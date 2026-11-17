#!/usr/bin/env python3
"""S54 大盤濾網家族：0050 趨勢弱時暫停衛星新買進（空位買 0050／空位留現金），E0 出場不動。
規格 SPEC.json、判定 PREREG.json（寫於任何非現行格報酬計算之前，sha256 啟動時核對）。
用法：python3 -I -B run.py [--quick]（--quick 只跑驗收 A2／A3）
載入與訊號逐行照 agents/s48_exit_rule_family/run.py；不改任何既有檔案。暫存 scratchpad/s54/（結束刪除）。"""
import sys, json, time, resource, gc, hashlib, inspect
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s54')
CACHE = TMP / 'cache'
CACHE.mkdir(parents=True, exist_ok=True)
QUICK = '--quick' in sys.argv
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for f in ('SPEC', 'PREREG'):
    assert sha(HERE / f'{f}.json') == (HERE / f'{f}.sha256').read_text().split()[0], f'{f}.json 與登記的 sha256 不符'

sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402
for mod, attr in ((s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                  (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(mod, attr, CACHE)
assert cslib.DEADLINE_FIX is False and cslib.INS15_MASK is False
cslib.EXRIGHT_FIX = True

# ---- 引擎副本：空位留現金（cashmode[t]＝True 的日子），照 S47 idle='cash' 寫法；原檔不動
_SRC0 = inspect.getsource(s10lib.simulate_s10)
_REPL = [
    ("def simulate_s10(", "def simulate_s54("),
    ("exec_mode='open', trail=None, trail_daily=True, requal=None):",
     "exec_mode='open', trail=None, trail_daily=True, requal=None, cashmode=None):"),
    ("""        elif cash >= sweep_min:
            f = max(minfee, cash * fee)""",
     """        elif cashmode is not None and cashmode[t]:
            gap = B - v50
            if gap >= sweep_min and cash >= sweep_min:
                x = min(cash, gap)
                f = max(minfee, x * fee); val = (x - f) / (1 + sl50)
                st['e50_cost'] += x - val; st['e50_orders'] += 1
                v50 += val; cash -= x
            elif -gap >= sweep_min:
                y = -gap
                g = y * (1 - sl50); f = max(minfee, g * fee); net = g - f - g * TAX_ETF * cm
                v50 -= y; cash += net; st['e50_cost'] += y - net; st['e50_orders'] += 1
        elif cash >= sweep_min:
            f = max(minfee, cash * fee)"""),
]
_SRC = _SRC0
for _a, _b in _REPL:
    assert _SRC.count(_a) == 1, f'替換點不唯一或不存在：{_a[:40]}'
    _SRC = _SRC.replace(_a, _b)
_G = dict(s10lib.__dict__)
exec(compile(_SRC, '<s54 simulate_s54>', 'exec'), _G)
simulate_s54 = _G['simulate_s54']
SRC_SHA = dict(simulate_s10=hashlib.sha256(_SRC0.encode()).hexdigest(), simulate_s54=hashlib.sha256(_SRC.encode()).hexdigest())

P = s20lib.load()
assert P.exright_fix
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), '面板不符'
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
F = s17lib.rev_family(P, defs=['STD'])
s17lib.e50_fix_suspension(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
S0 = F['S']['STD']; months = list(S0.index)
del F; gc.collect()
cslib.DEADLINE_FIX = True; cslib.INS15_MASK = True
AV = cslib.rev_avail_days(P, months, True)
RT = dict(S=S0, av=AV)
print('載入', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

E = np.asarray(P.E50, float)


def weak(ind, k):
    """只用 ≤ k 的 0050 含息還原收盤。True＝弱（暫停新買）。"""
    if ind == 'R60':
        if k - 60 < 0 or not (np.isfinite(E[k]) and np.isfinite(E[k - 60])):
            return False
        return bool(E[k] / E[k - 60] - 1 <= 0)
    n = int(ind[2:])
    w = E[max(0, k - n + 1):k + 1]
    if len(w) < n or np.isfinite(w).sum() < n:
        return False
    return bool(not (E[k] > w.mean()))


lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TB = P.IDX[-1]; D19 = lastb('2019-01-01'); FULL0 = P.IDX[P.t0]
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
CUR = 'NONE'
RUN = [('MA60', 'E50'), ('MA120', 'E50'), ('R60', 'E50'), ('MA60', 'CASH'), ('MA120', 'CASH'), ('MA200', 'CASH'), ('R60', 'CASH')]
cells = [CUR] if QUICK else [CUR] + [f'{i}_{a}' for i, a in RUN]
SEGK = list(s17lib.SEGMENTS)


def metrics(nav, lg, st):
    cl = lg[lg['why'] != 'open_end']
    m = dict(since2019=stlib.cagr(nav, D19, TB), full=stlib.cagr(nav, FULL0, TB), maxdd_full=stlib.maxdd(nav, FULL0, TB),
             maxdd_2019=stlib.maxdd(nav, D19, TB), n_closed=int(len(cl)), n_buy=int(st['n_buy']), avg_npos=float(st['avg_npos']),
             avg_idle_frac=float(st['avg_idle_frac']))
    m['calmar'] = m['since2019'] / abs(m['maxdd_full'])
    m['seg_dd'] = s17lib.seg_dd(nav)
    return m


R = {c: dict(paths={}) for c in cells}
TRIG = {}
A3 = None
for sft in range(4):
    sg = s10lib.make_signal_s10(P, U, RT, shift=sft, depth=10, mom_lo=0.10, nsel=10)
    S2base, ekw = s10lib.variant_S(sg, 'E0')
    ks = [int(x['k']) for x in sg['info']]
    for c in cells:
        if c == CUR:
            S2 = S2base; cmode = None
        else:
            ind, act = c.split('_')
            wk = [weak(ind, k) for k in ks]
            if sft == 0:
                TRIG[c] = dict(n_rows=len(ks), n_weak=int(sum(wk)),
                               n_weak_since2019=int(sum(w for w, k in zip(wk, ks) if P.IDX[k] > D19)),
                               months=[str(P.IDX[k].date()) for w, k in zip(wk, ks) if w])
            sig = S2base['sig'].copy(); rank = S2base['rank'].copy()
            for w, k in zip(wk, ks):
                if w:
                    sig[k, :] = False; rank[k, :] = np.nan
            S2 = dict(S2base); S2['sig'] = sig; S2['rank'] = rank
            cmode = None
            if act == 'CASH':
                cmode = np.zeros(P.T, bool)
                for i, (w, k) in enumerate(zip(wk, ks)):
                    a = k + 1; b = (ks[i + 1] + 1) if i + 1 < len(ks) else P.T
                    if w:
                        cmode[a:b] = True
        for cap in CAPS:
            kw = dict(cost=COSTS['strict'], capital=cap, **CELL, **ekw)
            if cmode is None:
                nav, lg, st = s10lib.simulate_s10(P, S2, U, **kw)
            else:
                nav, lg, st = simulate_s54(P, S2, U, cashmode=cmode, **kw)
            m = metrics(nav, lg, st)
            R[c]['paths'][f'{int(cap/1e4)}萬|+{sft}'] = m
            if sft == 0 and cap == 1e6:
                R[c]['main'] = m
                if c == CUR:                               # 驗收 A3：cashmode 全 False 的副本與原引擎逐位相同
                    nav2, lg2, _ = simulate_s54(P, S2, U, cashmode=np.zeros(P.T, bool), **kw)
                    A3 = bool(np.array_equal(nav.to_numpy(), nav2.to_numpy(), equal_nan=True) and len(lg) == len(lg2))
                    print('A3', A3, flush=True)
                    assert A3, '驗收 A3 失敗'
    del sg, S2base; gc.collect()
    print('位移', sft, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

ref = json.loads((AG / 'deadline_engine_fix' / 'out' / 'results.json').read_text())
cur = R[CUR]
acc = dict(A3_cash_copy_bitwise=A3, main_since2019=cur['main']['since2019'], ref_since2019=ref['standard']['ALL']['since2019'],
           main_n_closed=cur['main']['n_closed'], ref_n_closed=ref['standard']['ALL']['n_closed'])
acc['main_bitwise'] = bool(acc['main_since2019'] == acc['ref_since2019'] and acc['main_n_closed'] == acc['ref_n_closed']
                           and cur['main']['full'] == ref['standard']['ALL']['full'] and cur['main']['maxdd_full'] == ref['standard']['ALL']['maxdd_full'])
acc['paths_bitwise'] = bool(all(cur['paths'][p]['since2019'] == ref['paths']['ALL'][p]['since2019'] for p in ref['paths']['ALL']))
acc['paths_median'] = float(np.median([cur['paths'][p]['since2019'] for p in cur['paths']]))
acc['ok'] = acc['main_bitwise'] and acc['paths_bitwise'] and bool(A3)
print('驗收', json.dumps(acc, ensure_ascii=False), flush=True)
assert acc['ok'], '驗收失敗'
if QUICK:
    print('quick ok', f'{time.time()-t0:.0f}s', rss(), 'MB'); sys.exit(0)

keys = list(cur['paths'])
for c, o in R.items():
    g = lambda f: np.array([o['paths'][p][f] for p in keys])
    gs = lambda s: np.array([o['paths'][p]['seg_dd'][s] for p in keys])
    o['path'] = dict(median=float(np.median(g('since2019'))), min=float(g('since2019').min()), max=float(g('since2019').max()),
                     maxdd_full_median=float(np.median(g('maxdd_full'))), maxdd_2019_median=float(np.median(g('maxdd_2019'))),
                     calmar_median=float(np.median(g('calmar'))), n_closed_median=float(np.median(g('n_closed'))),
                     avg_npos_median=float(np.median(g('avg_npos'))), avg_idle_frac_median=float(np.median(g('avg_idle_frac'))),
                     seg_dd_median={s: float(np.median(gs(s))) for s in SEGK})
    if c == CUR:
        o['label'] = '現行'; continue
    gc_ = lambda f: np.array([cur['paths'][p][f] for p in keys])
    dC = g('calmar') - gc_('calmar')
    dR = (g('since2019') - gc_('since2019')) * 100
    dD = (g('maxdd_full') - gc_('maxdd_full')) * 100
    dS = {s: (gs(s) - np.array([cur['paths'][p]['seg_dd'][s] for p in keys])) * 100 for s in SEGK}
    nC, mC = int((dC > 1e-12).sum()), float(np.median(dC))
    lab = '較好' if (mC >= 0.10 and nC >= 15) else ('較差' if (mC <= -0.10 and nC <= 5) else '差不多')
    mR, nR = float(np.median(dR)), int((dR > 1e-9).sum())
    s24 = dS['2024-07~2025-04']
    clear = bool(lab == '較好' and mR >= -3 and int((s24 > 1e-9).sum()) >= 15)
    labR = '比現行好' if (nR >= 15 and mR >= 2) else ('比現行差' if (nR <= 5 and mR <= -2) else '差不多')
    o['vs_current'] = dict(calmar_median_diff=mC, calmar_n_pos=nC, label=lab, clear_better_candidate=clear,
                           since2019_median_pp=mR, since2019_n_pos=nR, return_label_S17=labR,
                           maxdd_full_median_pp=float(np.median(dD)), maxdd_n_shallower=int((dD > 1e-9).sum()),
                           seg_dd={s: dict(median_pp=float(np.median(v)), n_shallower=int((v > 1e-9).sum())) for s, v in dS.items()})
    o['trigger_shift0'] = TRIG[c]

fam = [CUR] + [f'{i}_{a}' for i, a in RUN]
meds = {c: R[c]['path']['median'] for c in fam}
fm = float(np.median(list(meds.values())))
p3 = (meds[CUR] - fm) * 100
s17 = dict(family=fam, family_median=fm, current_median=meds[CUR], current_minus_median_pp=p3,
           current_rank_return=int(1 + sum(v > meds[CUR] for v in meds.values())),
           current_rank_calmar=int(1 + sum(R[c]['path']['calmar_median'] > R[CUR]['path']['calmar_median'] for c in fam)),
           verdict='現行偏樂觀（被挑過）' if p3 >= 3 else '沒被挑過')
act_cmp = {}
for ind in ('MA60', 'MA120', 'R60'):
    a, b = R[f'{ind}_CASH'], R[f'{ind}_E50']
    act_cmp[ind] = dict(calmar=float(np.median([a['paths'][p]['calmar'] - b['paths'][p]['calmar'] for p in keys])),
                        since2019_pp=float(np.median([(a['paths'][p]['since2019'] - b['paths'][p]['since2019']) * 100 for p in keys])),
                        maxdd_pp=float(np.median([(a['paths'][p]['maxdd_full'] - b['paths'][p]['maxdd_full']) * 100 for p in keys])))
bench = pd.Series(E[P.t0:], index=P.IDX[P.t0:])
b0050 = dict(since2019=stlib.cagr(bench, D19, TB), maxdd_full=stlib.maxdd(bench, FULL0, TB), seg_dd=s17lib.seg_dd(bench))
b0050['calmar'] = b0050['since2019'] / abs(b0050['maxdd_full'])
res = dict(spec_sha256=sha(HERE / 'SPEC.json'), prereg_sha256=sha(HERE / 'PREREG.json'), run_sha256=sha(Path(__file__)), src_sha256=SRC_SHA,
           flags=dict(EXRIGHT_FIX=cslib.EXRIGHT_FIX, DEADLINE_FIX=cslib.DEADLINE_FIX, INS15_MASK=cslib.INS15_MASK),
           acceptance=acc, s17_return_rule=s17, cash_minus_e50=act_cmp, bench_0050=b0050,
           cited_MA200_E50='S23 MA_STOP（舊引擎）：2019 起 −5.82 個百分點（0／20）、全期回撤淺 0.40（15／20）、2018 淺 1.77、2020-03 深 1.02、2022 深 1.47、2024-07～2025-04 淺 1.19（20／20）；本題未重跑',
           cells=R, path_keys=keys, runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
for c in fam:
    o = R[c]; pth = o['path']; v = o.get('vs_current', {})
    print(c, f"主跑 {o['main']['since2019']*100:.2f} 中位 {pth['median']*100:.2f} DD {pth['maxdd_full_median']*100:.2f} Cal {pth['calmar_median']:.3f}",
          {s: round(x * 100, 2) for s, x in pth['seg_dd_median'].items()}, json.dumps({k: v[k] for k in v if k != 'seg_dd'}, ensure_ascii=False),
          {s: (round(x['median_pp'], 2), x['n_shallower']) for s, x in v.get('seg_dd', {}).items()}, o.get('trigger_shift0', {}).get('n_weak'))
print(json.dumps(s17, ensure_ascii=False)); print(json.dumps(act_cmp, ensure_ascii=False)); print(json.dumps(b0050))
print('done', res['runtime_s'], 's', res['maxrss_mb'], 'MB')
