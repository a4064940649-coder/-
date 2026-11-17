#!/usr/bin/env python3
"""S57 月中出場檢查家族：檢查頻率 {每月（現行）, 每週, 每日} × 月中觸發 {MA, MA2, SL15}；月中只賣不補買（錢掃 0050），E0 換股日出場不動。
規格 SPEC.json、判定 PREREG.json（寫於任何非現行格報酬計算之前，sha256 啟動時核對）。
用法：python3 -I -B run.py [--quick]（--quick 只跑驗收 A1～A4）
載入與訊號逐行照 agents/s56_industry_cap_family/run.py（＝s48／s54）；−15% 用 inspect 副本＋字串替換（原檔不動）。暫存 scratchpad/s57/（結束刪除）。"""
import sys, json, time, resource, gc, hashlib, inspect
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s57')
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

# ---- 引擎副本：月中檢查日 chk 上，持有期含息報酬 g < 1 − stop → 標記 'stop'（次日開盤賣）；原檔不動
_SRC0 = inspect.getsource(s10lib.simulate_s10)
_REPL = [
    ("def simulate_s10(", "def simulate_s57("),
    ("exec_mode='open', trail=None, trail_daily=True, requal=None):",
     "exec_mode='open', trail=None, trail_daily=True, requal=None, stop=None, chk=None):"),
    ("""                elif requal is not None and is_sig and p['nm'] >= requal and j not in RQ.get(t, ()):
""",
     """                elif stop is not None and chk[t] and not is_sig and p['g'] < 1.0 - stop:
                    p['flag'] = True; p['why'] = 'stop'
                elif requal is not None and is_sig and p['nm'] >= requal and j not in RQ.get(t, ()):
"""),
]
_SRC = _SRC0
for _a, _b in _REPL:
    assert _SRC.count(_a) == 1, f'替換點不唯一或不存在：{_a[:40]}'
    _SRC = _SRC.replace(_a, _b)
_G = dict(s10lib.__dict__)
exec(compile(_SRC, '<s57 simulate_s57>', 'exec'), _G)
simulate_s57 = _G['simulate_s57']
SRC_SHA = dict(simulate_s10=hashlib.sha256(_SRC0.encode()).hexdigest(), simulate_s57=hashlib.sha256(_SRC.encode()).hexdigest())

P = s20lib.load()
assert P.exright_fix
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), '面板不符'
# ---- 月中觸發條件（G 與 60 日均線都要有值）
with np.errstate(invalid='ignore'):
    _MA = stlib.rmean(P.G, 60)
    MA1 = np.isfinite(P.G) & np.isfinite(_MA) & (P.G <= _MA)
del _MA
MA2 = MA1.copy(); MA2[0] = False; MA2[1:] &= MA1[:-1]
DIFF_S10 = int((MA1 != ~P.F_MA60).sum())                  # 與 S10 DMA60 定義（~F_MA60）不同的格數（只差在無值）
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

# ---- 檢查日：每週＝ISO 週的最後一個交易日；每日＝全部
iso = P.IDX.isocalendar()
wk = (iso['year'].astype(int) * 100 + iso['week'].astype(int)).to_numpy()
WEEK = np.zeros(P.T, bool); WEEK[:-1] = wk[:-1] != wk[1:]; WEEK[-1] = True
DAY = np.ones(P.T, bool)
A4 = bool(all(WEEK[np.flatnonzero(wk == w)].sum() == 1 and WEEK[np.flatnonzero(wk == w)[-1]] for w in np.unique(wk)))
assert A4, '驗收 A4 失敗'

lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TB = P.IDX[-1]; D19 = lastb('2019-01-01'); FULL0 = P.IDX[P.t0]
YRS = (TB - FULL0).days / 365.25
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
CUR = 'CUR'
DEF = {'W_MA': (WEEK, 'MA'), 'W_MA2': (WEEK, 'MA2'), 'W_SL15': (WEEK, 'SL'), 'D_MA2': (DAY, 'MA2'), 'D_SL15': (DAY, 'SL'), 'D_MA': (DAY, 'MA')}
FAM = [CUR, 'W_MA', 'W_MA2', 'W_SL15', 'D_MA2', 'D_SL15']
REFC = ['D_MA']
cells = [CUR] if QUICK else FAM + REFC
SEGK = list(s17lib.SEGMENTS)
E50 = np.asarray(P.E50, float)


def post_sell(lg, sigrows):
    """事後描述：月中被賣（why ∈ ma_daily／stop）股票，賣出日 t 收盤 → 下一個訊號列收盤的還原報酬，減同期 0050。"""
    sk = np.flatnonzero(sigrows)
    x = lg[lg['why'].isin(['ma_daily', 'stop'])]
    ex = []
    for j, t in x[['j', 'exit_t']].itertuples(index=False):
        nk = sk[sk > t]
        if not len(nk):
            continue
        k = int(nk[0]); a, b = P.G[t, j], P.G[k, j]
        if not (np.isfinite(a) and np.isfinite(b) and a > 0):
            continue
        ex.append(b / a - E50[k] / E50[t])
    ex = np.array(ex)
    return dict(n=int(len(ex)), mean=float(ex.mean()) if len(ex) else None, median=float(np.median(ex)) if len(ex) else None,
                frac_pos=float((ex > 0).mean()) if len(ex) else None)


def metrics(nav, lg, st):
    cl = lg[lg['why'] != 'open_end']
    mnav = float(nav.mean())
    m = dict(since2019=stlib.cagr(nav, D19, TB), full=stlib.cagr(nav, FULL0, TB), maxdd_full=stlib.maxdd(nav, FULL0, TB),
             maxdd_2019=stlib.maxdd(nav, D19, TB), n_closed=int(len(cl)), n_buy=int(st['n_buy']), avg_npos=float(st['avg_npos']),
             n_mid=int(cl['why'].isin(['ma_daily', 'stop']).sum()),
             turnover=float((st['buy_val'] + st['sell_val']) / 2 / mnav / YRS),
             cost_pct=float((st['fee'] + st['tax'] + st['slip'] + st['e50_cost']) / mnav / YRS),
             avg_hold=float(cl['hold_days'].mean()) if len(cl) else None)
    m['calmar'] = m['since2019'] / abs(m['maxdd_full'])
    m['seg_dd'] = s17lib.seg_dd(nav)
    return m


R = {c: dict(paths={}) for c in cells}
ACC = dict(A2={}, A3={}, A4_week=A4, n_week_checkdays=int(WEEK.sum()), diff_vs_S10_DMA60_cells=DIFF_S10)
PS = {}
for sft in range(4):
    sg = s10lib.make_signal_s10(P, U, RT, shift=sft, depth=10, mom_lo=0.10, nsel=10)
    S2, ekw = s10lib.variant_S(sg, 'E0')
    sr = sg['sigrows']
    XS = {}
    for c in cells:
        if c == CUR:
            continue
        chk, trig = DEF[c]
        chk_ns = chk & ~sr
        if trig != 'SL':
            XS[c] = S2['X'] | (chk_ns[:, None] & (MA1 if trig == 'MA' else MA2))
    for c in cells:
        for cap in CAPS:
            kw = dict(cost=COSTS['strict'], capital=cap, **CELL, **ekw)
            if c == CUR:
                nav, lg, st = s10lib.simulate_s10(P, S2, U, **kw)
            else:
                chk, trig = DEF[c]
                if trig == 'SL':
                    nav, lg, st = simulate_s57(P, S2, U, stop=0.15, chk=chk, **kw)
                else:
                    S3 = dict(S2); S3['X'] = XS[c]
                    nav, lg, st = s10lib.simulate_s10(P, S3, U, **kw)
            m = metrics(nav, lg, st)
            R[c]['paths'][f'{int(cap/1e4)}萬|+{sft}'] = m
            if c == CUR and cap == 1e6:                    # 驗收 A2／A3（每個位移都驗）
                nav2, lg2, _ = simulate_s57(P, S2, U, **kw)
                ACC['A2'][sft] = bool(np.array_equal(nav.to_numpy(), nav2.to_numpy(), equal_nan=True) and len(lg) == len(lg2))
                nav3, lg3, _ = simulate_s57(P, S2, U, stop=1.0, chk=DAY, **kw)
                ACC['A3'][sft] = bool(np.array_equal(nav.to_numpy(), nav3.to_numpy(), equal_nan=True) and len(lg) == len(lg3))
                assert ACC['A2'][sft] and ACC['A3'][sft], f'驗收 A2／A3 失敗（位移 {sft}）'
                del nav2, lg2, nav3, lg3
            if sft == 0 and cap == 1e6:
                R[c]['main'] = m
                if c != CUR:
                    PS[c] = post_sell(lg, sr)
                    PS[c]['why_counts'] = lg['why'].value_counts().to_dict()
                else:
                    PS[c] = dict(why_counts=lg['why'].value_counts().to_dict())
            del nav, lg, st
    del sg, S2, XS; gc.collect()
    print('位移', sft, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

ref = json.loads((AG / 'deadline_engine_fix' / 'out' / 'results.json').read_text())
cur = R[CUR]
acc = dict(ACC, main_since2019=cur['main']['since2019'], ref_since2019=ref['standard']['ALL']['since2019'],
           main_n_closed=cur['main']['n_closed'], ref_n_closed=ref['standard']['ALL']['n_closed'])
acc['A1_main_bitwise'] = bool(acc['main_since2019'] == acc['ref_since2019'] and acc['main_n_closed'] == acc['ref_n_closed']
                              and cur['main']['full'] == ref['standard']['ALL']['full'] and cur['main']['maxdd_full'] == ref['standard']['ALL']['maxdd_full'])
acc['A1_paths_bitwise'] = bool(all(cur['paths'][p]['since2019'] == ref['paths']['ALL'][p]['since2019'] for p in ref['paths']['ALL']))
acc['paths_median'] = float(np.median([cur['paths'][p]['since2019'] for p in cur['paths']]))
acc['ok'] = acc['A1_main_bitwise'] and acc['A1_paths_bitwise'] and all(ACC['A2'].values()) and all(ACC['A3'].values()) and A4
print('驗收', json.dumps(acc, ensure_ascii=False), flush=True)
assert acc['ok'], '驗收失敗'
if QUICK:
    print('quick ok', f'{time.time()-t0:.0f}s', rss(), 'MB'); sys.exit(0)

keys = list(cur['paths'])
for c, o in R.items():
    g = lambda f: np.array([o['paths'][p][f] for p in keys], float)
    gs = lambda s: np.array([o['paths'][p]['seg_dd'][s] for p in keys])
    o['path'] = dict(median=float(np.median(g('since2019'))), min=float(g('since2019').min()), max=float(g('since2019').max()),
                     maxdd_full_median=float(np.median(g('maxdd_full'))), maxdd_2019_median=float(np.median(g('maxdd_2019'))),
                     calmar_median=float(np.median(g('calmar'))), n_closed_median=float(np.median(g('n_closed'))),
                     n_buy_median=float(np.median(g('n_buy'))), n_mid_median=float(np.median(g('n_mid'))),
                     turnover_median=float(np.median(g('turnover'))), cost_pct_median=float(np.median(g('cost_pct'))),
                     avg_npos_median=float(np.median(g('avg_npos'))), avg_hold_median=float(np.median(g('avg_hold'))),
                     seg_dd_median={s: float(np.median(gs(s))) for s in SEGK})
    o['post_sell_main'] = PS[c]
    o['role'] = '現行' if c == CUR else ('參考（S10 已做，不判定）' if c in REFC else '家族')
    if c == CUR:
        continue
    gc_ = lambda f: np.array([cur['paths'][p][f] for p in keys], float)
    dC = g('calmar') - gc_('calmar')
    dR = (g('since2019') - gc_('since2019')) * 100
    dD = (g('maxdd_full') - gc_('maxdd_full')) * 100
    dS = {s: (gs(s) - np.array([cur['paths'][p]['seg_dd'][s] for p in keys])) * 100 for s in SEGK}
    nC, mC = int((dC > 1e-12).sum()), float(np.median(dC))
    lab = '較好' if (mC >= 0.10 and nC >= 15) else ('較差' if (mC <= -0.10 and nC <= 5) else '差不多')
    mR, nR = float(np.median(dR)), int((dR > 1e-9).sum())
    clear = bool(lab == '較好' and mR >= -3)
    labR = '比現行好' if (nR >= 15 and mR >= 2) else ('比現行差' if (nR <= 5 and mR <= -2) else '差不多')
    o['vs_current'] = dict(calmar_median_diff=mC, calmar_n_pos=nC, label=lab, clear_better=clear,
                           since2019_median_pp=mR, since2019_n_pos=nR, return_label_S17=labR,
                           maxdd_full_median_pp=float(np.median(dD)), maxdd_n_shallower=int((dD > 1e-9).sum()),
                           turnover_median_diff=float(np.median(g('turnover') - gc_('turnover'))),
                           n_closed_median_diff=float(np.median(g('n_closed') - gc_('n_closed'))),
                           seg_dd={s: dict(median_pp=float(np.median(v)), n_shallower=int((v > 1e-9).sum())) for s, v in dS.items()})
    if c in REFC:
        o['vs_current']['note'] = '參考格：數字照算，不屬於本題判定'


def s17rule(fam):
    meds = {c: R[c]['path']['median'] for c in fam}
    fm = float(np.median(list(meds.values())))
    p3 = (meds[CUR] - fm) * 100
    return dict(family=fam, family_median=fm, current_median=meds[CUR], current_minus_median_pp=p3,
                current_rank_return=int(1 + sum(v > meds[CUR] for v in meds.values())),
                current_rank_calmar=int(1 + sum(R[c]['path']['calmar_median'] > R[CUR]['path']['calmar_median'] for c in fam)),
                verdict='現行偏樂觀（被挑過）' if p3 >= 3 else '沒被挑過')


s17 = dict(main=s17rule(FAM), with_D_MA=s17rule(FAM + REFC))
res = dict(spec_sha256=sha(HERE / 'SPEC.json'), prereg_sha256=sha(HERE / 'PREREG.json'), run_sha256=sha(Path(__file__)), src_sha256=SRC_SHA,
           flags=dict(EXRIGHT_FIX=cslib.EXRIGHT_FIX, DEADLINE_FIX=cslib.DEADLINE_FIX, INS15_MASK=cslib.INS15_MASK),
           acceptance=acc, s17_return_rule=s17, cells=R, path_keys=keys, runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
for c in FAM + REFC:
    o = R[c]; pth = o['path']; v = o.get('vs_current', {}); mn = o['main']
    print(c, f"主跑 {mn['since2019']*100:.2f}/DD {mn['maxdd_full']*100:.2f}/Cal {mn['calmar']:.3f}/n {mn['n_closed']}/mid {mn['n_mid']}/TO {mn['turnover']:.2f}",
          f"| 中位 {pth['median']*100:.2f} DD {pth['maxdd_full_median']*100:.2f} Cal {pth['calmar_median']:.3f} n {pth['n_closed_median']} mid {pth['n_mid_median']} TO {pth['turnover_median']:.2f} cost {pth['cost_pct_median']*100:.2f}",
          {s: round(x * 100, 2) for s, x in pth['seg_dd_median'].items()}, json.dumps({k: v[k] for k in v if k != 'seg_dd'}, ensure_ascii=False),
          {s: (round(x['median_pp'], 2), x['n_shallower']) for s, x in v.get('seg_dd', {}).items()}, json.dumps(o['post_sell_main'], ensure_ascii=False))
print(json.dumps(s17, ensure_ascii=False))
print('done', res['runtime_s'], 's', res['maxrss_mb'], 'MB')
