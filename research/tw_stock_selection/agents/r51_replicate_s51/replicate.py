#!/usr/bin/env python3
"""R51：獨立重現 S51「每檔多押一點」訓練→凍結→驗證（不讀 S51 的 train.py／validate.py／s51lib.py）。
只依 S51 的 SPEC／PREREG／FROZEN／FINDINGS 規格文字，自己寫。
引擎：agents/s47_allocation_family/s47lib.py 的 simulate_s47（S47 已驗收副本；import 不改檔）＋本檔自己的計數副本
     （s47lib._SRC 在買進迴圈 5% 成交值上限檢查前加 st['n_sized'] 計數，只計數不影響計算）。
正式開關：cslib.EXRIGHT_FIX／DEADLINE_FIX／INS15_MASK 全開（s47lib import 時已開），s17lib.e50_fix_suspension。
股池 twse & ~創新板；SURGE＝s17lib.rev_family STD；make_signal_s10(shift)；variant_S('E0')；嚴成本。
用法：python3 -I -B replicate.py   輸出：results.json、compare.csv（本資料夾）。暫存 scratchpad/r51/（結束刪除）。"""
import sys, json, time, resource, gc, shutil
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/r51')
CACHE = TMP / 'cache'
CACHE.mkdir(parents=True, exist_ok=True)
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
log = lambda *a: print(f'[{time.time()-t_all:6.0f}s {rss():5d}MB]', *a, flush=True)

sys.path.insert(0, str(AG / 's47_allocation_family'))
import s47lib                                              # noqa: E402  （import 時開三個正式開關）
from s47lib import s20lib, s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402
for _m, _a in ((s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
               (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(_m, _a, CACHE)
assert cslib.EXRIGHT_FIX and cslib.DEADLINE_FIX and cslib.INS15_MASK

# ---- 計數副本（自己寫；只加計數）
_A = "            if cap_pct and np.isfinite(AMT[t, j]):\n                mx = int(cap_pct * AMT[t, j] / o)"
assert s47lib._SRC.count(_A) == 1
_SRC = s47lib._SRC.replace(_A, "            st['n_sized'] = st.get('n_sized', 0) + 1\n" + _A).replace('def simulate_s47(', 'def simulate_r51(')
_G = dict(s10lib.__dict__)
exec(compile(_SRC, '<r51 simulate_r51>', 'exec'), _G)
simulate_r51 = _G['simulate_r51']

CELLS = {'CUR': (0.7, 10), 'C1': (0.7, 8), 'C2': (0.8, 10), 'C3': (1.0, 12)}
CELLNAME = {'CUR': 'S70_D10', 'C1': 'S70_D8', 'C2': 'S80_D10', 'C3': 'S100_D12'}
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
SHIFTS = (0, 1, 2, 3)
KEYS = [f'{int(c/1e4)}萬|+{s}' for s in SHIFTS for c in CAPS]

P = s20lib.load()
assert P.exright_fix and (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22')
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
F = s17lib.rev_family(P, defs=['STD'])
s17lib.e50_fix_suspension(P)
RT = dict(S=F['S']['STD'], av=F['av'])
del F
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
log('載入完成')

lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
# 訓練窗起點＝2016-02 前最後一個交易日＝2016-01-30（週六補班日，在 P.IDX 內）；SPEC 文字寫 2016-01-29，FINDINGS 寫 01-30。
# 兩者都算（TR_A_ALT＝SPEC 字面 01-29），主結果用 01-30（＝「訊號 2016-02 起」的窗定義）。
TR_A, TR_B = lastb('2016-02-01'), pd.Timestamp('2021-12-30')
TR_A_ALT = pd.Timestamp('2016-01-29')
VA_A, VA_B = pd.Timestamp('2021-12-30'), P.IDX[-1]
D19 = lastb('2019-01-01')
assert str(TR_A.date()) == '2016-01-30' and TR_A_ALT in P.IDX and TR_B in P.IDX


def sim(sg, cell, cap, cost='strict', block=None, counter=False):
    sat, d = CELLS[cell]
    S2, ekw = s10lib.variant_S(sg, 'E0')
    kw = dict(cost=COSTS[cost], capital=cap, cap_pct=0.05, sat_w=sat, sizing='W10', lock=False, block=block, **ekw)
    if counter:
        return simulate_r51(P, S2, U, ndiv=d, idle='e50', **kw)
    return s47lib.simulate_s47(P, S2, U, ndiv=d, idle='e50', **kw)


def dd_of(s):
    s = s.dropna()
    return float((s / s.cummax() - 1).min())


def ex2020(nav):
    s = nav.loc[TR_A:TR_B].dropna()
    r = s.pct_change().iloc[1:]
    r = r[r.index.year != 2020]
    idx = (1 + r).cumprod()
    lv = pd.concat([pd.Series([1.0], index=[s.index[0]]), idx])
    days = (s.index[-1] - s.index[0]).days
    d20 = (min(s.index[-1], pd.Timestamp('2020-12-31')) - max(s.index[0], pd.Timestamp('2020-01-01'))).days + 1
    yrs = (days - d20) / 365.25
    return float(lv.iloc[-1] ** (1 / yrs) - 1), dd_of(lv)


def train_m(nav):
    c, d = stlib.cagr(nav, TR_A, TR_B), stlib.maxdd(nav, TR_A, TR_B)
    e, ed = ex2020(nav)
    return dict(cagr=c, dd=d, calmar=c / abs(d), ex20=e, ex20_dd=ed, cagr_0129=stlib.cagr(nav, TR_A_ALT, TR_B), dd_0129=stlib.maxdd(nav, TR_A_ALT, TR_B),
                since2019=stlib.cagr(nav, D19, VA_B), full=stlib.cagr(nav, P.IDX[P.t0], VA_B), full_dd=stlib.maxdd(nav, P.IDX[P.t0], VA_B))


def valid_m(nav, a=VA_A):
    c, d = stlib.cagr(nav, a, VA_B), stlib.maxdd(nav, a, VA_B)
    sd = s17lib.seg_dd(nav)
    return dict(cagr=c, dd=d, calmar=c / abs(d), seg2022=sd['2022'], seg2407=sd['2024-07~2025-04'],
                y2223=stlib.cagr(nav, a, lastb('2024-01-01')), y24=stlib.cagr(nav, lastb('2024-01-01'), VA_B),
                nav_start=float(nav.loc[a]))


def pairs(A, B, f):
    d = np.array([A[k][f] - B[k][f] for k in KEYS]) * 100
    return dict(median_pp=float(np.median(d)), npos=int((d > 0).sum()), n_ge025=int((d >= 0.25).sum()), min_pp=float(d.min()), max_pp=float(d.max()))


med = lambda D, f: float(np.median([D[k][f] for k in KEYS]))
res = dict(meta=dict(engine='s47lib.simulate_s47（S47 驗收副本）＋本檔計數副本 simulate_r51', flags=dict(EXRIGHT_FIX=True, DEADLINE_FIX=True, INS15_MASK=True)))

# ---- 訊號（位移 0～3）
SG = {s: s10lib.make_signal_s10(P, U, RT, shift=s) for s in SHIFTS}
for s in SHIFTS:
    for k in ('XV',):
        SG[s]['XV'] = {'E0': SG[s]['XV']['E0']}
gc.collect()
log('訊號完成')

# ---- 驗收：現行 100 萬位移 0 全期主跑
nav0, lg0, _ = sim(SG[0], 'CUR', 1e6)
_S2, _ek = s10lib.variant_S(SG[0], 'E0')
nav0b, lg0b, _ = s10lib.simulate_s10(P, _S2, U, cost=COSTS['strict'], capital=1e6, sat_w=0.7, sizing='W10', lock=False, **_ek)
ref = json.loads((AG / 'deadline_engine_fix' / 'out' / 'results.json').read_text())
A0 = dict(since2019=stlib.cagr(nav0, D19, VA_B), full=stlib.cagr(nav0, P.IDX[P.t0], VA_B), maxdd=stlib.maxdd(nav0, P.IDX[P.t0], VA_B),
          n_closed=int((lg0['why'] != 'open_end').sum()))
A0['bitwise_vs_deadline_ALL'] = bool(A0['since2019'] == ref['standard']['ALL']['since2019'] and A0['full'] == ref['standard']['ALL']['full']
                                     and A0['n_closed'] == ref['standard']['ALL']['n_closed'])
A0['copy_vs_simulate_s10_bitwise'] = bool(np.array_equal(nav0.to_numpy(), nav0b.to_numpy(), equal_nan=True))
res['A0'] = A0
log('A0', A0)
del nav0b, lg0b

# ---- 全期連續 20 條 × 4 格（訓練窗從這裡切）
TRN = {c: {} for c in CELLS}
for s in SHIFTS:
    for cap in CAPS:
        k = f'{int(cap/1e4)}萬|+{s}'
        for c in CELLS:
            nav, _, _ = sim(SG[s], c, cap)
            TRN[c][k] = train_m(nav)
    log('訓練 位移', s)
A2 = max(abs(TRN['CUR'][k]['since2019'] - ref['paths']['ALL'][k]['since2019']) for k in KEYS)
res['A2_max_abs_gap_vs_deadline_ALL_pp'] = A2 * 100
res['train_window'] = dict(start=str(TR_A.date()), end=str(TR_B.date()), note='2016-01-30 為週六補班交易日；SPEC 字面 2016-01-29 的結果另列 *_0129')

train = {}
for c in CELLS:
    train[c] = dict(cell=CELLNAME[c], median=med(TRN[c], 'cagr'), median_spec_literal_0129=med(TRN[c], 'cagr_0129'), dd_median_0129=med(TRN[c], 'dd_0129'), dd_median=med(TRN[c], 'dd'), calmar_median=med(TRN[c], 'calmar'),
                    ex20_median=med(TRN[c], 'ex20'), ex20_dd_median=med(TRN[c], 'ex20_dd'),
                    full_since2019_median=med(TRN[c], 'since2019'), full_median=med(TRN[c], 'full'), full_dd_median=med(TRN[c], 'full_dd'))
    if c != 'CUR':
        train[c].update(median_minus_cur_pp=(train[c]['median'] - train['CUR']['median']) * 100,
                        dd_median_minus_cur_pp=(train[c]['dd_median'] - train['CUR']['dd_median']) * 100,
                        paired=pairs(TRN[c], TRN['CUR'], 'cagr'), paired_0129=pairs(TRN[c], TRN['CUR'], 'cagr_0129'), paired_dd=pairs(TRN[c], TRN['CUR'], 'dd'),
                        ex20_paired=pairs(TRN[c], TRN['CUR'], 'ex20'), ex20_dd_paired=pairs(TRN[c], TRN['CUR'], 'ex20_dd'),
                        ex20_median_minus_cur_pp=(train[c]['ex20_median'] - train['CUR']['ex20_median']) * 100,
                        ex20_dd_median_minus_cur_pp=(train[c]['ex20_dd_median'] - train['CUR']['ex20_dd_median']) * 100)
res['train'] = train
# 選擇規則（PREREG step1）
cand = ['C1', 'C2', 'C3']
top = sorted(cand, key=lambda c: (-round(train[c]['median'], 12), -train[c]['paired']['n_ge025'], cand.index(c)))[0]
okr = train[top]['median_minus_cur_pp'] >= 1.0
okd = train[top]['dd_median_minus_cur_pp'] >= -2.0
sel = top if (okr and okd) else '不改'
T0 = dict(paired_median_pp=train[top]['paired']['median_pp'], n_ge025=train[top]['paired']['n_ge025'])
T0['pass'] = bool(T0['paired_median_pp'] >= 1.0 and T0['n_ge025'] >= 14)
ex_sel = {c: bool(train[c]['ex20_median_minus_cur_pp'] >= 1.0 and train[c]['ex20_dd_median_minus_cur_pp'] >= -2.0) for c in cand}
res['selection'] = dict(top=top, cond_ret=bool(okr), cond_dd=bool(okd), selected=sel, frozen=top, T0=T0, ex2020_would_pass=ex_sel)
log('選擇', res['selection'])
FZ = top

# ---- 驗證：重新起跑
SGV = {s: s20lib.fresh_from(SG[s], s20lib.VALID_FIRST_MONTH) for s in SHIFTS}
probe, lgp, _ = sim(SGV[0], 'CUR', 1e6)
ratio = float(probe.loc[VA_A] / 1e6)
first_ent = str(P.IDX[int(lgp['ent_t'].min())].date()) if len(lgp) else None
res['valid_meta'] = dict(scale_ratio=ratio, first_entry_restart=first_ent, start_capital_rule='資金 ÷ ratio（現行主跑 100 萬位移 0 的 2021-12-30 淨值 ÷ 100 萬）')
log('ratio', ratio, '首筆進場', first_ent)
assert first_ent is None or first_ent >= '2022-01-01'
del probe, lgp

VAL = {c: {} for c in CELLS}
VALX = {c: {} for c in ('CUR', FZ)}
mainv = {}
for s in SHIFTS:
    for cap in CAPS:
        k = f'{int(cap/1e4)}萬|+{s}'
        for c in CELLS:
            nav, lg, _ = sim(SGV[s], c, cap / ratio)
            VAL[c][k] = valid_m(nav)
            if s == 0 and cap == 1e6:
                mainv[c] = (nav, lg)
        for c in ('CUR', FZ):
            navx, _, _ = sim(SGV[s], c, cap / ratio, cost='strict_x2')
            VALX[c][k] = valid_m(navx)
    log('驗證 位移', s)


def vstats(c, base='CUR', V=VAL):
    o = dict(V1=pairs(V[c], V[base], 'cagr'), dd=pairs(V[c], V[base], 'dd'), calmar=pairs(V[c], V[base], 'calmar'),
             seg2022=pairs(V[c], V[base], 'seg2022'), seg2407=pairs(V[c], V[base], 'seg2407'),
             y2223=pairs(V[c], V[base], 'y2223'), y24=pairs(V[c], V[base], 'y24'),
             median=med(V[c], 'cagr'), base_median=med(V[base], 'cagr'), dd_median=med(V[c], 'dd'), base_dd_median=med(V[base], 'dd'),
             calmar_median=med(V[c], 'calmar'), base_calmar_median=med(V[base], 'calmar'))
    return o


def top5_block(lg_list):
    blk, codes = {}, []
    for lg in lg_list:
        x = lg[lg['why'] != 'open_end'].sort_values('pnl', ascending=False).head(5)
        for y in x.itertuples():
            blk.setdefault(int(y.j), []).append((int(y.ent_t), int(y.exit_t)))
            codes.append(str(P.COLS[y.j]))
    return blk, sorted(set(codes))


def v3(c):
    blk, codes = top5_block([mainv['CUR'][1], mainv[c][1]])
    A, B = {}, {}
    for s in SHIFTS:
        for cap in CAPS:
            k = f'{int(cap/1e4)}萬|+{s}'
            A[k] = valid_m(sim(SGV[s], c, cap / ratio, block=blk)[0])
            B[k] = valid_m(sim(SGV[s], 'CUR', cap / ratio, block=blk)[0])
    return dict(codes=codes, paired=pairs(A, B, 'cagr'))


valid = {}
for c in cand:
    v = vstats(c)
    v['V3'] = v3(c)
    v['pass'] = dict(V1=v['V1']['npos'] >= 15, V2=v['V1']['median_pp'] >= 2.0, V3=v['V3']['paired']['median_pp'] > 0,
                     V4=v['seg2022']['median_pp'] >= -2.0 and v['seg2407']['median_pp'] >= -2.0)
    valid[c] = v
    log('驗證', c, v['V1'], v['V3']['paired'])
valid[FZ]['costx2'] = pairs(VALX[FZ], VALX['CUR'], 'cagr')
res['valid'] = valid
res['overall_pass'] = bool(sel != '不改' and T0['pass'] and all(valid[FZ]['pass'].values()))
res['main_restart_100w_s0'] = {c: dict(cagr=stlib.cagr(mainv[c][0], VA_A, VA_B), n_closed=int((mainv[c][1]['why'] != 'open_end').sum())) for c in CELLS}

# ---- V2 敏感度（起跑資金、驗證窗起點）
sens = {}
def v2_alt(tag, capf, a=VA_A, sg=SGV):
    A, B = {}, {}
    for s in SHIFTS:
        for cap in CAPS:
            k = f'{int(cap/1e4)}萬|+{s}'
            A[k] = valid_m(sim(sg[s], FZ, capf(cap))[0], a)
            B[k] = valid_m(sim(sg[s], 'CUR', capf(cap))[0], a)
    sens[tag] = dict(V2=pairs(A, B, 'cagr'), fz_median=med(A, 'cagr'), cur_median=med(B, 'cagr'))
    log('敏感度', tag, sens[tag])
v2_alt('no_scale（起始資金不縮放，2015-07-01 就是 60～200 萬）', lambda c: c)
v2_alt('scale_2.558_rounded', lambda c: c / 2.558)
v2_alt('window_from_2021-12-31_next_trading_day', lambda c: c / ratio, a=P.IDX[P.IDX > VA_A][0])
res['V2_sensitivity'] = sens
# 連續模擬切驗證窗（非重新起跑，只供對照）
A, B = {}, {}
for s in SHIFTS:
    for cap in CAPS:
        k = f'{int(cap/1e4)}萬|+{s}'
        A[k] = valid_m(sim(SG[s], FZ, cap)[0]); B[k] = valid_m(sim(SG[s], 'CUR', cap)[0])
res['V2_sensitivity']['continuous_slice（非重新起跑）'] = dict(V2=pairs(A, B, 'cagr'), fz_median=med(A, 'cagr'), cur_median=med(B, 'cagr'))
log('連續切窗', res['V2_sensitivity']['continuous_slice（非重新起跑）'])

# ---- 容量：1,000 萬全期連續、位移 0～3
capres = {}
for c in CELLS:
    r = {}
    for s in SHIFTS:
        _, _, st = sim(SG[s], c, 1e7, counter=True)
        r[f'+{s}'] = dict(n_cap=int(st['blk_cap']), n_sized=int(st['n_sized']), capped_rate=st['blk_cap'] / st['n_sized'])
    r['mean'] = float(np.mean([r[f'+{s}']['capped_rate'] for s in SHIFTS]))
    capres[c] = r
    log('容量', c, r['+0'], r['mean'])
res['capacity_1e7'] = capres
res['runtime_s'] = round(time.time() - t_all, 1); res['maxrss_mb'] = rss()

# ---- 對照表
S = []
def row(item, s51, r51, unit='%'):
    S.append(dict(item=item, s51=s51, r51=round(r51, 4) if r51 is not None else None, diff=round(r51 - s51, 4) if (r51 is not None and s51 is not None) else None,
                  ok=bool(abs(r51 - s51) <= 0.05 + 1e-9) if (r51 is not None and s51 is not None and unit != 'n') else (r51 == s51 if unit == 'n' else None), unit=unit))
for c, a, b in (('CUR', 33.27, -26.36), ('C1', 36.92, -25.77), ('C2', 35.80, -25.75), ('C3', 35.67, -26.14)):
    row(f'訓練 中位數 {CELLNAME[c]}', a, train[c]['median'] * 100)
    row(f'訓練 回撤中位數 {CELLNAME[c]}', b, train[c]['dd_median'] * 100)
for c, a in (('CUR', 33.27), ('C1', 36.92), ('C2', 35.80), ('C3', 35.67)):
    row(f'（對照）訓練 中位數 {CELLNAME[c]}，起點照 SPEC 字面 2016-01-29', a, train[c]['median_spec_literal_0129'] * 100)
row('訓練 C1 減現行（中位數差）', 3.65, train['C1']['median_minus_cur_pp'], 'pp')
row('去掉2020 C1 成對差', 1.30, train['C1']['ex20_paired']['median_pp'], 'pp')
row('去掉2020 C1 回撤中位數差', -2.28, train['C1']['ex20_dd_median_minus_cur_pp'], 'pp')
row('T0 成對中位數', 3.61, T0['paired_median_pp'], 'pp')
row('T0 ≥+0.25 路徑數', 20, T0['n_ge025'], 'n')
row('V1 為正路徑數', 20, valid[FZ]['V1']['npos'], 'n')
row('V2 驗證窗成對差中位數', 2.26, valid[FZ]['V1']['median_pp'], 'pp')
row('V3 禁買後成對差中位數', 1.52, valid[FZ]['V3']['paired']['median_pp'], 'pp')
row('V4 2022 段跌幅差', 1.08, valid[FZ]['seg2022']['median_pp'], 'pp')
row('V4 2024-07～2025-04 段跌幅差', -1.34, valid[FZ]['seg2407']['median_pp'], 'pp')
row('驗證窗中位數 C1', 42.84, valid[FZ]['median'] * 100)
row('驗證窗中位數 現行', 40.50, valid[FZ]['base_median'] * 100)
row('驗證窗回撤中位數 C1', -35.18, valid[FZ]['dd_median'] * 100)
row('驗證窗回撤中位數 現行', -33.47, valid[FZ]['base_dd_median'] * 100)
row('Calmar 中位數 C1', 1.221, valid[FZ]['calmar_median'], 'x')
row('Calmar 中位數 現行', 1.174, valid[FZ]['base_calmar_median'], 'x')
row('2022～2023 年化差', 2.83, valid[FZ]['y2223']['median_pp'], 'pp')
row('2024 起年化差', 1.63, valid[FZ]['y24']['median_pp'], 'pp')
row('容量 現行 位移0', 13.2, capres['CUR']['+0']['capped_rate'] * 100)
row('容量 C1 位移0', 18.9, capres['C1']['+0']['capped_rate'] * 100)
row('容量 C2 位移0', 16.8, capres['C2']['+0']['capped_rate'] * 100)
row('容量 C3 位移0', 17.0, capres['C3']['+0']['capped_rate'] * 100)
row('全期現行主跑 2019起', 48.76, A0['since2019'] * 100)
row('全期現行主跑 已平倉筆數', 381, A0['n_closed'], 'n')
row('全期 20 條 2019 起中位數 現行', 46.40, train['CUR']['full_since2019_median'] * 100)
row('全期 20 條 2019 起中位數 C1', 50.77, train['C1']['full_since2019_median'] * 100)
row('全期 20 條全期回撤中位數 現行', -33.39, train['CUR']['full_dd_median'] * 100)
row('全期 20 條全期回撤中位數 C1', -35.18, train['C1']['full_dd_median'] * 100)
row('全期 主跑 C1 2019 起', 53.86, TRN['C1']['100萬|+0']['since2019'] * 100)
row('全期 主跑 現行 2019 起（20 條中 100萬+0）', 48.76, TRN['CUR']['100萬|+0']['since2019'] * 100)
res['compare'] = S
pd.DataFrame(S).to_csv(HERE / 'compare.csv', index=False)
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
log('完成')
shutil.rmtree(TMP, ignore_errors=True)
