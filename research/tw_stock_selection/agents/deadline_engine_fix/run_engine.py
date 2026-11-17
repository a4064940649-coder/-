#!/usr/bin/env python3
"""營收申報期限修正接進共用引擎（cslib.DEADLINE_FIX／cslib.INS15_MASK，預設都關）後的驗證重跑。
依據：agents/s40_revenue_deadline_pit（建議 1、2）、agents/s41_ins15_handling（處理 A）。
用法：python3 -I -B run_engine.py [組合,逗號分隔，預設 OFF,DL,INS,ALL] [--save-ref 檔] [--ref 檔] [--no-paths]
  OFF＝兩個開關都關（應與舊結果逐位相同）；DL＝只開 DEADLINE_FIX；INS＝只開 INS15_MASK；ALL＝兩個都開。
  --save-ref：把 OFF 的標準路徑淨值與交易存成 npz（改引擎前先跑一次當參考）；--ref：拿參考檔逐位比對 OFF。
鏈：s20lib→s17lib→s10lib；cslib.EXRIGHT_FIX＝True；s17lib.e50_fix_suspension；股池＝stlib.universe(P,'twse') & ~創新板；
SURGE＝s17lib.rev_family STD；make_signal_s10(shift)；variant_S('E0')；simulate_s10(嚴成本, sat_w=0.7, W10, lock=False)（＝S40 第 3 步）。
標準路徑＝100 萬、位移 0；20 條路徑＝本金 {60,80,100,150,200} 萬 × 位移 {0,1,2,3}。
結果：out/results.json、out/nav_std.csv.gz、out/trades_diff.csv、out/changed_cells.csv。暫存放 scratchpad/dlfix/。"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/dlfix')
CACHE = TMP / 'cache'
CACHE.mkdir(parents=True, exist_ok=True)
OUT = HERE / 'out'
args = [a for a in sys.argv[1:]]
opt = lambda k: args[args.index(k) + 1] if k in args else None
pos = [a for i, a in enumerate(args) if not a.startswith('--') and (i == 0 or args[i - 1] not in ('--save-ref', '--ref'))]
CONFIGS = (pos[0] if pos else 'OFF,DL,INS,ALL').split(',')
SAVE_REF, REF = opt('--save-ref'), opt('--ref')
PATHS = '--no-paths' not in args
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402
for mod, attr in ((s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                  (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(mod, attr, CACHE)
HAS_FLAGS = hasattr(cslib, 'DEADLINE_FIX') and hasattr(cslib, 'INS15_MASK')
FLAGS = {'OFF': (False, False), 'DL': (True, False), 'INS': (False, True), 'ALL': (True, True)}
assert HAS_FLAGS or CONFIGS == ['OFF'], '引擎還沒有 DEADLINE_FIX／INS15_MASK，只能跑 OFF'
if HAS_FLAGS:
    assert cslib.DEADLINE_FIX is False and cslib.INS15_MASK is False, '開關預設必須是關閉'
cslib.EXRIGHT_FIX = True


def set_flags(cfg):
    if HAS_FLAGS:
        cslib.DEADLINE_FIX, cslib.INS15_MASK = FLAGS[cfg]


set_flags('OFF')
P = s20lib.load()
assert P.exright_fix
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), '面板不符'
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
F = s17lib.rev_family(P, defs=['STD'])                     # 在 OFF 下建：F['av']＝舊規則
s17lib.e50_fix_suspension(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
S0, av_off = F['S']['STD'], F['av']
months = list(S0.index)
del F
print('載入', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TA, TB = lastb('2024-07-01'), P.IDX[-1]
D19 = lastb('2019-01-01')
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)


def RT(cfg):
    """依開關重算可得日（cslib.rev_avail_days 讀 DEADLINE_FIX）；INS15_MASK 在 make_signal_s10 裡讀。"""
    set_flags(cfg)
    av = cslib.rev_avail_days(P, months, True)
    return dict(S=S0, av=av)


def signal(cfg, shift):
    rt = RT(cfg)
    sg = s10lib.make_signal_s10(P, U, rt, shift=shift)
    set_flags('OFF')
    return sg, rt['av']


def sim(sg, cap=1e6, cost='strict'):
    S2, ekw = s10lib.variant_S(sg, 'E0')
    return s10lib.simulate_s10(P, S2, U, cost=COSTS[cost], capital=cap, **CELL, **ekw)


def metrics(nav):
    return dict(since2019=stlib.cagr(nav, D19, TB), full=stlib.cagr(nav, P.IDX[P.t0], TB), maxdd_full=stlib.maxdd(nav, P.IDX[P.t0], TB),
                cagr_window=stlib.cagr(nav, TA, TB), maxdd_window=stlib.maxdd(nav, TA, TB))


res = dict(configs={c: dict(DEADLINE_FIX=FLAGS[c][0], INS15_MASK=FLAGS[c][1]) for c in CONFIGS}, engine_has_flags=HAS_FLAGS,
           chain='s20lib→s17lib→s10lib；EXRIGHT_FIX=True；e50_fix_suspension；twse & ~創新板；STD SURGE；make_signal_s10；E0；simulate_s10(strict, sat_w=0.7, W10, lock=False)')
if HAS_FLAGS:
    res['flag_defaults'] = dict(DEADLINE_FIX=False, INS15_MASK=False, checked=True)
main = {}
for c in CONFIGS:
    sg, av = signal(c, 0)
    nav, lg, _ = sim(sg)
    navx = sim(sg, cost='strict_x2')[0]
    lg = lg.copy(); lg['code'] = [P.COLS[j] for j in lg['j']]
    main[c] = dict(nav=nav, lg=lg, navx=navx, av=av, info={x['month']: x for x in sg['info']}, X=sg['X'], sig=sg['sig'])
    print('main', c, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

# ---- 驗收 A：OFF 與舊結果
acc = {}
if 'OFF' in main:
    nc, lc = main['OFF']['nav'], main['OFF']['lg']
    a0 = dict(since2019=stlib.cagr(nc, D19, TB), full=stlib.cagr(nc, P.IDX[P.t0], TB), maxdd=stlib.maxdd(nc, P.IDX[P.t0], TB),
              n_trades=int((lc['why'] != 'open_end').sum()))
    a0['ok_vs_48.99/34.54/-32.11/381'] = bool(abs(a0['since2019'] - 0.4899) < 0.00025 and abs(a0['full'] - 0.3454) < 0.00025
                                               and abs(a0['maxdd'] + 0.3211) < 0.00025 and a0['n_trades'] == 381)
    s40 = json.loads((AG / 's40_revenue_deadline_pit' / 'results.json').read_text())['acceptance']['A0_engine_current']
    a0['bitwise_vs_S40_CUR_metrics'] = bool(a0['since2019'] == s40['since2019'] and a0['full'] == s40['full'] and a0['maxdd'] == s40['maxdd'])
    acc['A0_OFF'] = a0
    if HAS_FLAGS:
        set_flags('OFF')
        acc['OFF_av_equal_old_rule'] = cslib.rev_avail_days(P, months, True) == av_off
    if SAVE_REF:
        np.savez_compressed(SAVE_REF, nav=nc.to_numpy(), navx=main['OFF']['navx'].to_numpy(), X=main['OFF']['X'], sig=main['OFF']['sig'],
                            trades=lc[['j', 'ent_t', 'exit_t', 'why']].astype(str).to_numpy().astype('U'), av=np.array([av_off[m] for m in months]))
        print('參考檔已存', SAVE_REF, flush=True)
    if REF:
        z = np.load(REF, allow_pickle=True)                   # 參考檔是本程式自己寫在 scratchpad 的
        acc['OFF_bitwise_vs_preedit_ref'] = dict(
            nav=bool(np.array_equal(z['nav'], nc.to_numpy(), equal_nan=True)),
            navx=bool(np.array_equal(z['navx'], main['OFF']['navx'].to_numpy(), equal_nan=True)),
            X=bool(np.array_equal(z['X'], main['OFF']['X'])), sig=bool(np.array_equal(z['sig'], main['OFF']['sig'])),
            trades=bool(np.array_equal(z['trades'].astype('U'), lc[['j', 'ent_t', 'exit_t', 'why']].astype(str).to_numpy().astype('U'))),
            av=bool(np.array_equal(z['av'], np.array([av_off[m] for m in months]))))
    assert a0['ok_vs_48.99/34.54/-32.11/381'], f'OFF 驗收失敗 {a0}'
print(json.dumps(acc, ensure_ascii=False, default=str), flush=True)
if not HAS_FLAGS:
    OUT.mkdir(exist_ok=True)
    print('改引擎前的參考跑完（不寫 out/）', f'{time.time()-t0:.0f}s', rss(), 'MB')
    sys.exit(0)

# ---- 改動的可得日與遮罩格（位移 0～3）
chg = []
for c in CONFIGS:
    if c == 'OFF':
        continue
    av = main[c]['av']
    for m in months:
        if av[m] != av_off[m]:
            chg.append(dict(config=c, kind='av', shift='', rev_month=str(m), code='',
                            old_signal_day=str(P.IDX[av_off[m]].date()) if av_off[m] < P.T else None,
                            new_signal_day=str(P.IDX[av[m]].date()) if av[m] < P.T else None))
    if FLAGS[c][1]:
        cslib.INS15_MASK = True
        for s in range(4):
            for m in months:
                k = av[m] + s
                if av[m] >= P.T or k >= P.T or av[m] < 61:
                    continue
                for j in cslib.ins15_hidden_cols(P, m, k):
                    chg.append(dict(config=c, kind='ins15_mask', shift=s, rev_month=str(m), code=str(P.COLS[j]),
                                    old_signal_day=None, new_signal_day=str(P.IDX[k].date())))
        cslib.INS15_MASK = False
chg = pd.DataFrame(chg)
OUT.mkdir(exist_ok=True)
chg.to_csv(OUT / 'changed_cells.csv', index=False)
ncell = chg[chg['kind'] == 'ins15_mask'].groupby(['config', 'shift']).size() if len(chg) else pd.Series(dtype=int)
res['n_ins15_cells'] = {f'{c}|+{s}': int(n) for (c, s), n in ncell.items()}
res['av_changes'] = chg[chg['kind'] == 'av'].to_dict('records') if len(chg) else []
res['acceptance'] = acc

# ---- 標準路徑
key = lambda d: set(zip(d['code'], d['ent_t'], d['exit_t']))
std, tdiff, sigdiff, firstdiff = {}, [], {}, {}
for c in CONFIGS:
    m = metrics(main[c]['nav']); m['since2019_costx2'] = stlib.cagr(main[c]['navx'], D19, TB)
    lg = main[c]['lg']; m['n_closed'] = int((lg['why'] != 'open_end').sum()); m['end_nav'] = float(main[c]['nav'].iloc[-1])
    std[c] = m
for c in CONFIGS:
    if c == 'OFF':
        continue
    for f in ('since2019', 'full', 'maxdd_full', 'cagr_window', 'maxdd_window', 'since2019_costx2'):
        std[c][f'd_{f}_pp'] = (std[c][f] - std['OFF'][f]) * 100
    dd = np.flatnonzero(main[c]['nav'].to_numpy() != main['OFF']['nav'].to_numpy())
    firstdiff[c] = str(main[c]['nav'].index[dd[0]].date()) if len(dd) else None
    a, b = main[c]['lg'], main['OFF']['lg']
    ka, kb = key(a), key(b)
    for nm, d, ks in ((c, a, ka - kb), ('OFF', b, kb - ka)):
        for y in d.itertuples():
            if (y.code, y.ent_t, y.exit_t) in ks:
                tdiff.append(dict(config=c, side=nm, code=y.code, ent=str(P.IDX[y.ent_t].date()), exit=str(P.IDX[y.exit_t].date()), why=y.why,
                                  ret=float(y.proceeds / y.cost - 1) if y.cost else None))
    sd = {}
    for mo, x in main[c]['info'].items():
        y = main['OFF']['info'].get(mo)
        if y is None or x['targets'] != y['targets'] or x['sig_date'] != y['sig_date']:
            sd[mo] = dict(sig_date=x['sig_date'], off_sig_date=y['sig_date'] if y else None, f1=x['targets'], off_f1=y['targets'] if y else None)
    sigdiff[c] = sd
res['standard'] = std
res['first_nav_diff_date'] = firstdiff
res['f1_diff_months'] = sigdiff
pd.DataFrame(tdiff).to_csv(OUT / 'trades_diff.csv', index=False)
pd.DataFrame({c: main[c]['nav'] for c in CONFIGS}).rename_axis('date').to_csv(OUT / 'nav_std.csv.gz', float_format='%.6f')
for c in list(main):
    for k in ('X', 'sig'):
        main[c].pop(k, None)
gc.collect()
print('standard', json.dumps({c: {k: round(x, 4) for k, x in d.items() if k.startswith('d_')} for c, d in std.items()}, ensure_ascii=False), flush=True)

# ---- 20 條路徑
if PATHS:
    PATH = {c: {} for c in CONFIGS}
    for s in range(4):
        for c in CONFIGS:
            sg, _ = signal(c, s)
            for cap in CAPS:
                nav, _, _ = sim(sg, cap)
                PATH[c][f'{int(cap/1e4)}萬|+{s}'] = metrics(nav)
            del sg
        gc.collect()
        print('shift', s, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
    keys = list(PATH['OFF'])
    pairs = {}
    for c in CONFIGS:
        if c == 'OFF':
            continue
        o = {}
        for f in ('since2019', 'full', 'maxdd_full', 'cagr_window', 'maxdd_window'):
            d = np.array([PATH[c][k][f] - PATH['OFF'][k][f] for k in keys]) * 100
            o[f] = dict(median_pp=float(np.median(d)), mean_pp=float(d.mean()), min_pp=float(d.min()), max_pp=float(d.max()),
                        n_lower=int((d < -1e-9).sum()), n_higher=int((d > 1e-9).sum()), n_equal=int((np.abs(d) <= 1e-9).sum()))
        pairs[c] = o
    res['paths'] = PATH
    res['paths_pairs'] = pairs

# ---- 驗收 B：對 S40（ALL／DL／PM）
s40e = json.loads((AG / 's40_revenue_deadline_pit' / 'results.json').read_text())['engine']
MAP = {'DL': 'DL', 'INS': 'PM', 'ALL': 'ALL'}
cmp = {}
for c, v in MAP.items():
    if c not in std:
        continue
    r = dict(s40_variant=v, d_since2019_pp=std[c]['d_since2019_pp'], s40_d_since2019_pp=s40e['standard'][v]['d_since2019_pp'],
             since2019=std[c]['since2019'], s40_since2019=s40e['standard'][v]['since2019'])
    r['gap_std_pp'] = r['d_since2019_pp'] - r['s40_d_since2019_pp']
    if PATHS:
        r['median20_pp'] = res['paths_pairs'][c]['since2019']['median_pp']
        r['s40_median20_pp'] = s40e['paths_pairs'][v]['since2019']['median_pp']
        r['gap_median20_pp'] = r['median20_pp'] - r['s40_median20_pp']
        r['per_path_max_abs_gap_pp'] = float(max(abs((res['paths'][c][k]['since2019'] - s40e['paths'][v][k]['since2019']) * 100) for k in keys))
    r['pass_le_0.02pp'] = bool(abs(r['gap_std_pp']) <= 0.02 and (not PATHS or abs(r['gap_median20_pp']) <= 0.02))
    cmp[c] = r
res['acceptance']['vs_S40'] = cmp
res['runtime_s'] = round(time.time() - t0, 1); res['maxrss_mb'] = rss()
(OUT / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print(json.dumps(dict(acc=res['acceptance'], cells=res['n_ins15_cells']), ensure_ascii=False, indent=1, default=str))
print('done', res['runtime_s'], 's', res['maxrss_mb'], 'MB')
