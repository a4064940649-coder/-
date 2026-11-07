#!/usr/bin/env python3
"""integrate_o16 正式執行（只跑一次）：驗 PREREG 雜湊與 sha 後，跑
  A 濾網（金融遮罩 PIT 化）：把 o_finmask_delisted_fin_erratum（FM，可用）的 F1（PIT 主口徑）／F2（靜態敏感度）／F3（描述）遮罩
    換進主策略 C0（排除金融 n10，main_v2 gapfix 官方名單）與 ER2 的 U_opt 名單，只改 2006～2013 訊號日；量化名單異動對報酬的影響
    （FM 本身只計數；本塊是整合階段的報酬敏感度，不取代主策略正式數字，也不回寫 FM）。
    引擎四口徑：freeze 一般（主）、freeze 壓力、dzero 一般（下市最壞）、nofreeze 一般（換股日以最後收盤出清＝近似收購／換股對價，o15 教訓要求同時報）。
  C 協定歷史套用：獨立重寫 G8P-2a（G8_PROTOCOL_v2a.json，選定 A1：z^A＝S_n／(c·max(σ_d,s_n)·√n)、b_a）與 G8P-2 V1，
    以「較不利者」合併，套到 V7X 可執行歷史線，對照 G8P-2a 自報 HIST_DEMO 與 G8P-2 HIST_DEMO（一致性一票否決）。
  B（第二條腿固定比例混合）依 PREREG 不做。
輸出寫入 results.json 的 RESULTS／DECISION（只寫一次）。
python3 -I -B run.py                  正式
INTEG_SMOKE=1 python3 -I -B run.py    登記前管線核對（C0 重現、名單與 FM_COUNTS 逐筆相同、序列長度與均值；不算任何變體報酬或協定結果，不寫 results.json）"""
import os, sys, json, math, hashlib, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
FM = AG / 'o_finmask_delisted_fin_erratum'
GA = AG / 'o_g8p2a_protocol_addendum'
G2 = AG / 'o_g8p2_protocol_refreeze'
ER = AG / 'o_rev_trunc_2009_erratum'
RJ = HERE / 'results.json'
SMOKE = bool(os.environ.get('INTEG_SMOKE'))
T0 = time.time()
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)
def now(): return pd.Timestamp.now('UTC').isoformat()

RO_DIRS = [AG / 'live_tracking' / 'main_forward', AG / 'o_forward_infra_live']
def ro_snapshot():
    rows = []
    for d in RO_DIRS:
        for p in sorted(d.rglob('*')):
            if p.is_file():
                st = p.stat(); rows.append([str(p), sha(p), st.st_size, st.st_mtime])
    return dict(utc=now(), n=len(rows), sha=hashlib.sha256(json.dumps([r[:3] for r in rows]).encode()).hexdigest(), rows=rows)

if not SMOKE:
    R = json.loads(RJ.read_text(encoding='utf-8'))
    assert 'RESULTS' not in R, 'RESULTS 已存在：只跑一次'
    P = R['PREREG']
    assert hashlib.sha256(json.dumps(P, ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256'], 'PREREG 被改'
    for f, h in P['sha256'].items():
        assert sha(f) == h, f'{f} 雜湊不符'
    RO0 = ro_snapshot()

# =============================================================== A：FM 遮罩 × 主策略
SPEC = json.loads((FM / 'ERRATUM_SPEC_FM.json').read_text(encoding='utf-8'))
assert sha(FM / 'ERRATUM_SPEC_FM.json') == '9f2d0622bb1754c1d13e1411760fd8fad9cc5f0fac40fc9281a688ce992b077d'
FMC = json.loads((FM / 'FM_COUNTS.json').read_text(encoding='utf-8'))
sys.path.insert(0, str(FM))
import fclass as FCL      # 只讀 import：分類器（F1 PIT／F2 靜態）
import flib as FL         # 只讀 import：VM（每訊號日替換 v2_core.FINMASK）、sym_slots、inserted
sys.path.pop(0)
EL, V, K = FL.E, FL.V, FL.K
COLS = FL.COLS
LO, HI = EL.LO, EL.HI
assert sha(AG / 'main_v2' / 'v2_core.py') == SPEC['files']['engine']['v2_core']
assert sha(AG / 'independent_audit' / 'ia_core.py') == SPEC['files']['engine']['ia_core']
assert sha(ER / 'erlib.py') == SPEC['files']['engine']['erlib']
cls = SPEC['classifiers']
CL = FCL.Classifier(cls['keywords'], cls['f2_manual_exclusion'], cls['name_exclusion'])
C0M = FL.C0MASK.copy()
F2V = np.array([CL.f2(c) for c in COLS])
_m = {}
def mF1(d):
    d = pd.Timestamp(d)
    if d not in _m: _m[d] = np.array([CL.f1_value(CL.f1(c, d)[0], C0M[j]) for j, c in enumerate(COLS)])
    return _m[d]
def scoped(fn):   # 只改 2006～2013 訊號日（FM 規格範圍）；之後一律 C0 遮罩
    return lambda d: fn(d) if LO <= pd.Timestamp(d) <= HI else C0M
MASKF = {'C0': lambda d: C0M, 'F1': scoped(mF1), 'F2': scoped(lambda d: F2V), 'F3': scoped(lambda d: mF1(d) | F2V)}
VARS = {'C0': ('C0', 'base'), 'C0_F1': ('F1', 'base'), 'C0_F2': ('F2', 'base'), 'C0_F3': ('F3', 'base'),
        'U_opt': ('C0', 'opt'), 'U_opt_F1': ('F1', 'opt'), 'U_opt_F2': ('F2', 'opt'), 'U_opt_F3': ('F3', 'opt')}
REFV = {'C0_F1': 'C0', 'C0_F2': 'C0', 'C0_F3': 'C0', 'U_opt_F1': 'U_opt', 'U_opt_F2': 'U_opt', 'U_opt_F3': 'U_opt'}
FMKEY = {'C0': 'C0|C0', 'C0_F1': 'F1|C0', 'C0_F2': 'F2|C0', 'C0_F3': 'F3|C0', 'U_opt': 'C0|U_opt', 'U_opt_F1': 'F1|U_opt', 'U_opt_F2': 'F2|U_opt', 'U_opt_F3': 'F3|U_opt'}
COST = {'normal': K.rates(0.001), 'stress': K.rates('stress')}
ENGS = [('normal', 'freeze', dict()), ('stress', 'freeze', dict()), ('normal', 'dzero', dict(delist_zero=True)), ('normal', 'nofreeze', dict(freeze=False))]
PL, NAV, AFF, SDS, FRZ = {}, {}, {}, {}, {}
for k, (mk, mode) in VARS.items():
    S = FL.VM(MASKF[mk], mode, 'SURV')
    pl = S.plan(10, 'official'); PL[k] = pl
    t0, t1, sd = S.bounds(); SDS[k] = list(sd)
    AFF[k] = {d: set(v) for d, v in S.aff.items()}
    if not SMOKE or k == 'C0':
        for c, eg, kw in ENGS:
            if SMOKE and eg != 'freeze': continue
            bb, ss = COST[c]
            NAV[(k, c, eg)], lg = K.simulate([(p[0], p[1]) for p in pl], bb, ss, t0, t1, **kw)
            if (c, eg) == ('normal', 'freeze'):
                FRZ[k] = dict(n_events=len(lg['frozen']), codes=sorted({x for _, cs, _ in lg['frozen'] for x in cs}))
    log('A', k)
assert all(SDS[k] == SDS['C0'] for k in VARS), '訊號日不一致'
SD = pd.DatetimeIndex(SDS['C0'])
OFF = pd.read_parquet(AG / 'main_v2' / 'navs_gapfix.parquet')
C0DIFF = {}
for c in COST:
    ref = OFF[f'EXFIN|n10|{c}'].dropna(); mine = NAV[('C0', c, 'freeze')].reindex(ref.index)
    C0DIFF[c] = float(np.nanmax(np.abs(mine.to_numpy() - ref.to_numpy())))
    assert C0DIFF[c] < 1e-9, f'C0 未重現 main_v2 {c}：{C0DIFF[c]}'
PK = {k: {p[3]: [COLS[j] for j in p[1]] for p in PL[k]} for k in PL}
days13 = [d for d in PK['C0'] if LO <= d <= HI]
assert len(days13) == 93
# 名單核對：2006～2013 與 FM_COUNTS.lists 逐筆相同；2014 起各變體與其基準（C0／U_opt）相同；U_opt 2014 起也與 C0 相同
LISTCHK = {}
for k in VARS:
    ref = FMC['lists'][FMKEY[k]]
    assert all(PK[k][d] == ref[str(d.date())] for d in days13), f'{k} 2006～2013 名單與 FM_COUNTS 不同'
    after = [d for d in PK[k] if d > HI]
    assert all(PK[k][d] == PK['C0'][d] for d in after), f'{k} 2014 起名單與 C0 不同'
    LISTCHK[k] = dict(days13_equal_FM=True, after_equal_C0=True, n_after=len(after))
L0 = json.loads((AG / 'o_g8_protocol_fin_erratum' / 'ERRATUM.json').read_text(encoding='utf-8'))['lists_C0']
assert all(PK['C0'][d] == L0[str(d.date())] for d in days13), 'C0 與 o14 lists_C0 不同'
ER2 = json.loads((ER / 'ERRATUM2.json').read_text(encoding='utf-8'))
ins_u = sorted([str(d.date()), c] for d in days13 for c in PK['U_opt'][d] if c in AFF['U_opt'].get(d, set()))
assert ins_u == sorted(map(list, ER2['U_opt_slot_list'])), 'U_opt 槽位清單與 ER2 不同'
def sym(a, b):
    rows = []
    for d in days13:
        A_, B_ = set(PK[a][d]), set(PK[b][d])
        if A_ != B_: rows.append(dict(sig=str(d.date()), out=sorted(A_ - B_), in_=sorted(B_ - A_), slots=len(A_ ^ B_) / 2))
    return rows
SLOTS = {}
for k, r in REFV.items():
    rows = sym(r, k); tot = sum(x['slots'] for x in rows)
    SLOTS[k] = dict(slots_930=tot, slots_810=sum(x['slots'] for x in rows if x['sig'] < '2013-01-01'), rows=rows)
    fmk = f"{'C0' if r == 'C0' else 'U_opt'}|{VARS[k][0]}"
    assert tot == FMC['blocks'][fmk]['slots_930'], (k, tot, FMC['blocks'][fmk]['slots_930'])
log('A 核對通過', {k: v['slots_930'] for k, v in SLOTS.items()})

# =============================================================== 共用：fslib（窗、錨定月、0050）
sys.path.insert(0, str(AG / 'o_forward_sequential_v2'))
import fslib as F
sys.path.pop(0)
E = F.e50(); Dm = F.deadlines()
JS = list(range(13))

def win_v2(nav):
    pts = list(SD[SD < V.END1012]) + [pd.Timestamp(V.END1012)]
    rows = []
    for a, b in zip(pts[:-1], pts[1:]):
        rows.append((pd.Period(a, 'M'), F._at(nav, b) / F._at(nav, a) - 1, F._at(E, b) / F._at(E, a) - 1, a.year))
    D = pd.DataFrame(rows, columns=['per', 'r', 'r50', 'yr']).set_index('per')
    return D[~D.index.duplicated(keep='last')].dropna()

TIERS = ['100萬', '300萬', '1000萬']
XC = [f'B10|J{j}|K5T|{c}|{t}|GROW' for j in JS for c in ('normal', 'stress') for t in TIERS]
X = pd.read_parquet(AG / 'o_exec_engine_rewrite_v7x' / 'out' / 'navs_oos_R4.parquet', columns=XC)
MX = {}
for t in TIERS:
    for c in ('normal', 'stress'):
        M = {}
        for j in JS:
            W = F.window_excess(X[f'B10|J{j}|K5T|{c}|{t}|GROW'], F.j_signals(j, 'AB_V7_B10'), F.j_end(j, 'AB_V7_B10'), E, Dm)
            W = W[~W.index.duplicated(keep='last')].dropna(subset=['r', 'r50'])
            W = W[[x is not None for x in W.index]]
            W.index = pd.PeriodIndex(W.index, freq='M')
            M[j] = (W['r'] - W['r50']) * 100.0
        MX[(t, c)] = pd.DataFrame(M).sort_index()
del X

def line(t, worse, min_j=7):
    Mn = MX[(t, 'normal')]
    M = np.minimum(Mn, MX[(t, 'stress')].reindex_like(Mn)) if worse else Mn
    ok = M.notna().sum(axis=1) >= min_j
    return M.median(axis=1, skipna=True).where(ok).dropna()

LINES = {f'V7X_{t}_{"worse" if w else "normal"}': line(t, w) for t in TIERS for w in (True, False)}
HD2 = json.loads((G2 / 'results.json').read_text(encoding='utf-8'))['HIST_DEMO']
GAR = json.loads((GA / 'results.json').read_text(encoding='utf-8'))
HDA = GAR['HIST_DEMO']

if SMOKE:
    w = win_v2(NAV[('C0', 'normal', 'freeze')])
    out = dict(utc=now(), C0_maxabs_vs_main_v2=C0DIFF, v2_windows=len(w), v2_first=str(w.index[0]), v2_last=str(w.index[-1]),
               list_check=LISTCHK, slots={k: dict(slots_930=v['slots_930'], slots_810=v['slots_810']) for k, v in SLOTS.items()},
               slot_rows_sha=hashlib.sha256(json.dumps({k: v['rows'] for k, v in SLOTS.items()}, sort_keys=True).encode()).hexdigest(),
               lines={k: dict(n=len(s), first=str(s.index[0]), last=str(s.index[-1]), mean=float(s.mean()), sd=float(s.std(ddof=1))) for k, s in LINES.items()},
               hist_demo_ref=dict(worse_mean_g8p2=HD2['V_sel_worse']['mean_pp'], worse_sd_g8p2=HD2['V_sel_worse']['sd_pp'], n_starts_g8p2a=HDA['n_starts']),
               note='只做管線核對；沒有計算任何變體的報酬、t 值或協定結果（C0 只算 freeze 一般／壓力以核對 main_v2）。')
    print(json.dumps(out, ensure_ascii=False, indent=1))
    sys.exit(0)

# =============================================================== 統計工具
PERS = {'ALL': (None, None), 'SEG1': ('2006-01', '2012-12'), 'SEG2': ('2013-01', '2018-12'), 'SEG3': ('2019-01', None),
        'SEL': (None, '2019-04'), 'VAL': ('2019-05', None), 'HOLD': ('2024-01', None),
        'Y0613': ('2006-01', '2013-12'), 'Y14ON': ('2014-01', None)}
def pmask(idx, k):
    a, b = PERS[k]; m = np.ones(len(idx), bool)
    if a: m &= idx >= pd.Period(a, 'M')
    if b: m &= idx <= pd.Period(b, 'M')
    return m
def tt(x):
    x = np.asarray(x, float); s = x.std(ddof=1)
    return float(x.mean() / s * math.sqrt(len(x))) if len(x) > 2 and s > 1e-15 else None
def ann(r): r = np.asarray(r, float); return float(np.prod(1 + r) ** (12 / len(r)) - 1) * 100
def mdd(r):
    nav = np.cumprod(1 + np.asarray(r, float)); pk_ = np.maximum.accumulate(np.concatenate([[1.0], nav]))[1:]
    return float((nav / pk_ - 1).min()) * 100
def corr(a, b): return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None

def stats_one(rv, r50, rM, idx, yrs):
    o = {}
    for k in PERS:
        m = pmask(idx, k)
        if m.sum() < 6: continue
        a, b, c = rv[m], r50[m], rM[m]
        o[k] = dict(n=int(m.sum()), ann=ann(a), ann_0050=ann(b), gex=ann(a) - ann(b), mdd=mdd(a), t_vs_0050=tt(a - b),
                    mean_ex_pp=float(np.mean(a - b) * 100), diff_vs_ref_pp=float(np.mean(a - c) * 100), t_diff_vs_ref=tt(a - c),
                    ann_diff_vs_ref=ann(a) - ann(c), corr_ref=corr(a, c), corr_0050=corr(a, b), corr_ex=corr(a - b, c - b),
                    n_win_diff=int(np.sum(np.abs(a - c) > 1e-12)))
    s = pd.Series(rv - r50).groupby(np.asarray(yrs)).sum()
    b3 = list(s.sort_values(ascending=False).index[:3])
    m = ~np.isin(yrs, b3)
    o['ALL']['t_drop_best3'] = tt(rv[m] - r50[m]); o['ALL']['best3'] = [int(x) for x in b3]
    yr = {}
    for y in range(2006, 2014):
        mm = yrs == y
        if mm.any(): yr[str(y)] = dict(ann=float((np.prod(1 + rv[mm]) - 1) * 100), ex_vs_0050=float((np.prod(1 + rv[mm]) - np.prod(1 + r50[mm])) * 100),
                                      ex_vs_ref=float((np.prod(1 + rv[mm]) - np.prod(1 + rM[mm])) * 100))
    o['yearly_2006_2013'] = yr
    return o

RES = {}
# =============================================================== A 結果
A = dict(C0_maxabs_vs_main_v2=C0DIFF, list_check=LISTCHK, slots=SLOTS, frozen_normal_freeze=FRZ)
for c, eg, _ in ENGS:
    WIN = {k: win_v2(NAV[(k, c, eg)]) for k in VARS}
    for k in VARS:
        for refn in (['C0'] + ([REFV[k]] if REFV.get(k, 'C0') != 'C0' else [])):
            W = WIN[refn].join(WIN[k][['r']], rsuffix='_k', how='inner')
            st = stats_one(W['r_k'].to_numpy(), W['r50'].to_numpy(), W['r'].to_numpy(), W.index, W['yr'].to_numpy())
            nav = NAV[(k, c, eg)]; st['ALL']['mdd_daily'] = float((nav / nav.cummax() - 1).min() * 100)
            A[f'{k}|{c}|{eg}|vs_{refn}'] = st
# 描述：每個異動槽位，被移出（遮罩後成為金融而排除，或被擠出）與移入股的「訊號日收盤→下一訊號日收盤」紙上報酬（含息還原、下市後持平）
ADJ = K.ADJ.ffill()
def fwd(d, codes):
    d = pd.Timestamp(d); i = SD.get_loc(d); b = SD[i + 1] if i + 1 < len(SD) else pd.Timestamp(V.END1012)
    a0 = ADJ.loc[:d].iloc[-1]; b0 = ADJ.loc[:b].iloc[-1]
    return {c: float(b0[c] / a0[c] - 1) for c in codes if np.isfinite(a0[c]) and np.isfinite(b0[c])}
DESC = {}
for k in REFV:
    rows = []
    for r in SLOTS[k]['rows']:
        o_ = fwd(r['sig'], r['out']); i_ = fwd(r['sig'], r['in_'])
        rows.append(dict(sig=r['sig'], out=o_, in_=i_, diff_pp=(float(np.mean(list(i_.values())) - np.mean(list(o_.values()))) * 100 if o_ and i_ else None)))
    dd = [x['diff_pp'] for x in rows if x['diff_pp'] is not None]
    DESC[k] = dict(rows=rows, n=len(dd), mean_in_minus_out_pp=float(np.mean(dd)) if dd else None, sum_in_minus_out_pp=float(np.sum(dd)) if dd else None)
A['desc_slot_returns'] = DESC
RES['A_filter'] = A
log('A 完成')

# =============================================================== C 協定歷史套用（獨立重寫 G8P-2a A1 ＋ G8P-2 V1，較不利者合併）
assert sha(GA / 'G8_PROTOCOL_v2a.json') == '13972450a2b21938fe892b4e8659c89c8ade56372ce3a2fff678300642ad9d33'
assert sha(G2 / 'G8_PROTOCOL_v2.json') == 'dcab4dbd255fb11e0ad8f4508397ae109cae9a47090eee17ab2ec7952981b993'
PA = json.loads((GA / 'G8_PROTOCOL_v2a.json').read_text(encoding='utf-8'))['params']
P2 = json.loads((G2 / 'G8_PROTOCOL_v2.json').read_text(encoding='utf-8'))['params']
assert PA['selected'] == 'A1'
SIG, CI_, CD = PA['sigma_design'], PA['c'], PA['c_d']
assert abs(SIG - P2['sigma_design']) < 1e-12 and abs(CI_ - P2['c']) < 1e-12 and abs(CD - P2['c_d']) < 1e-12 and abs(PA['b_v2'] - P2['b']) < 1e-12
BV2, BA = P2['b'], PA['b_a']
ORDER = ['FORWARD_FAILED', 'FORWARD_VALIDATING_NO_REJECTION_24', 'FORWARD_VALIDATING', 'UPGRADE_PENDING_24', 'S1_EXCEPTION_EVIDENCE']

def zst(x, n, kind):
    S = float(np.sum(x[:n])); s_n = float(np.std(x[:n], ddof=1))
    den = max(CI_ * SIG, s_n) if kind == 'V1' else CI_ * max(SIG, s_n)
    return S / (den * math.sqrt(n))

def judge(x, kind):
    """24 個有效月、無作廢的路徑。12 月：字面 G8 → 下修 → 升級（升級後待 24 月確認，不再下修）；18 月：下修；24 月：下修 → 升級。"""
    b = BV2 if kind == 'V1' else BA
    z = {n: zst(x, n, kind) for n in (12, 18, 24)}
    det = dict(S12=float(np.sum(x[:12])), S24=float(np.sum(x[:24])), z12=z[12], z18=z[18], z24=z[24])
    if np.sum(x[:12]) <= 0: return 'FORWARD_FAILED', 'literal', det
    if z[12] <= -CD: return 'FORWARD_FAILED', 'down12', det
    if z[12] >= b * math.sqrt(2): return 'S1_EXCEPTION_EVIDENCE', 'up12', det
    if z[18] <= -CD: return 'FORWARD_FAILED', 'down18', det
    if z[24] <= -CD: return 'FORWARD_FAILED', 'down24', det
    if z[24] >= b: return 'S1_EXCEPTION_EVIDENCE', 'up24', det
    return 'FORWARD_VALIDATING_NO_REJECTION_24', 'none', det

def apply(s):
    x = s.to_numpy(); n = len(x) - 24 + 1; starts = s.index[:n]
    rows = []
    for i in range(n):
        xi = x[i:i + 24]
        a, wa, da = judge(xi, 'A1'); v, wv, dv = judge(xi, 'V1')
        comb = ORDER[min(ORDER.index(a), ORDER.index(v))]
        rows.append(dict(start=str(starts[i]), A1=a, A1_why=wa, V1=v, V1_why=wv, comb=comb, lit=bool(np.sum(xi[:12]) > 0),
                         zA24=da['z24'], zV24=dv['z24'], zA12=da['z12'], zV12=dv['z12'], S24=da['S24']))
    D = pd.DataFrame(rows)
    def frac(d):
        o = dict(n=len(d))
        for col in ('comb', 'A1', 'V1'):
            o[col] = dict(S1=float((d[col] == ORDER[4]).mean()), FAILED=float((d[col] == ORDER[0]).mean()), NO_REJ=float((d[col] == ORDER[1]).mean()))
        o['literal_pass'] = float(d.lit.mean())
        return o
    yr = D.start.str[:4].astype(int)
    out = dict(n_months=len(x), first=str(s.index[0]), last=str(s.index[-1]), mean_pp=float(x.mean()), sd_pp=float(x.std(ddof=1)),
               all=frac(D), sel_starts_2006_2017=frac(D[D.start < '2017-05']), val_starts_2017_2022=frac(D[(D.start >= '2017-05') & (D.start < '2022-01')]),
               hold_starts_2022_on=frac(D[D.start >= '2022-01']), s2006_2012=frac(D[yr <= 2012]), s2013_2018=frac(D[(yr >= 2013) & (yr <= 2018)]), s2019_2024=frac(D[yr >= 2019]),
               n_comb_better_than_V1=int(sum(ORDER.index(a) > ORDER.index(b) for a, b in zip(D.comb, D.V1))),
               n_comb_worse_than_V1=int(sum(ORDER.index(a) < ORDER.index(b) for a, b in zip(D.comb, D.V1))),
               n_A1_fail_V1_not=int(((D.A1 == ORDER[0]) & (D.V1 != ORDER[0])).sum()), n_V1_fail_A1_not=int(((D.V1 == ORDER[0]) & (D.A1 != ORDER[0])).sum()),
               zA24=dict(max=float(D.zA24.max()), p90=float(D.zA24.quantile(0.9)), median=float(D.zA24.median()), n_ge_bA=int((D.zA24 >= BA).sum())),
               zV24=dict(max=float(D.zV24.max()), p90=float(D.zV24.quantile(0.9)), n_ge_bV2=int((D.zV24 >= BV2).sum())),
               max_zA24_start=str(D.start[D.zA24.idxmax()]),
               why_A1={k: int(v) for k, v in D.A1_why.value_counts().items()}, why_V1={k: int(v) for k, v in D.V1_why.value_counts().items()},
               last_start=rows[-1], last12_sum_pp=float(x[-12:].sum()), last24_sum_pp=float(x[-24:].sum()),
               s1_starts_comb=list(D.start[D.comb == ORDER[4]]), by_start=dict(zip(D.start, D.comb)))
    return out
C = {ln: apply(s) for ln, s in LINES.items()}
# 升級所需 24 個月平均月超額（pp），以 s_n ≤ σ_d 時的分母計
C['required_mean_pp'] = dict(A1_24=BA * CI_ * SIG / math.sqrt(24), V1_24=BV2 * CI_ * SIG / math.sqrt(24),
                             A1_12=BA * math.sqrt(2) * CI_ * SIG / math.sqrt(12), V1_12=BV2 * math.sqrt(2) * CI_ * SIG / math.sqrt(12),
                             hist_worse_100_mean=C['V7X_100萬_worse']['mean_pp'], hist_normal_100_mean=C['V7X_100萬_normal']['mean_pp'])
d1 = C['V7X_100萬_worse']
CHK = dict(comb_vs_HDA_all={f: (d1['all']['comb'][f], HDA['v2a'][f]) for f in ('S1', 'FAILED', 'NO_REJ')},
           V1_vs_HDA_v2={f: (d1['all']['V1'][f], HDA['v2'][f]) for f in ('S1', 'FAILED', 'NO_REJ')},
           V1_vs_HD2={f: (d1['all']['V1'][f], HD2['V1_worse']['all'][f]) for f in ('S1', 'FAILED', 'NO_REJ')},
           literal=(d1['all']['literal_pass'], HDA['literal_pass']),
           by_period={p: {f: (d1[p]['comb'][f], HDA['by_period'][p]['v2a'][f]) for f in ('S1', 'FAILED', 'NO_REJ')} for p in ('sel_starts_2006_2017', 'val_starts_2017_2022', 'hold_starts_2022_on')},
           n_starts=(d1['all']['n'], HDA['n_starts']), last=(d1['last_start']['start'], d1['last_start']['comb'], HDA['last_start'], HDA['last_v2a']),
           n_better=(d1['n_comb_better_than_V1'], HDA['n_v2a_better_than_v2']), V1_s1_starts_equal_HD2=(d1['s1_starts_comb'] == [] and HD2['V1_worse']['s1_starts'] == []))
def _eq(p): return abs(p[0] - p[1]) < 1e-12
CHK['consistent'] = bool(all(_eq(v) for v in CHK['comb_vs_HDA_all'].values()) and all(_eq(v) for v in CHK['V1_vs_HDA_v2'].values())
                         and all(_eq(v) for v in CHK['V1_vs_HD2'].values()) and _eq(CHK['literal']) and all(_eq(v) for p in CHK['by_period'].values() for v in p.values())
                         and CHK['n_starts'][0] == CHK['n_starts'][1] and CHK['last'][0] == CHK['last'][2] and CHK['last'][1] == CHK['last'][3]
                         and CHK['n_better'][0] == CHK['n_better'][1] == 0)
C['check'] = CHK
POWMIN = GAR['OC_FINAL']['summary']['qualification']['A1']['power_min']
C['oc_power_min_alpha1'] = POWMIN
RES['C_protocol'] = C
log('C 完成', CHK['consistent'])

# =============================================================== 判定
DEC = {}
PAIRS = [('C0_F1', 'C0'), ('C0_F2', 'C0'), ('U_opt_F1', 'U_opt'), ('U_opt_F2', 'U_opt')]
dA = {}
for k, r in PAIRS:
    o = {}
    for c, eg, _ in ENGS:
        x, y = A[f'{k}|{c}|{eg}|vs_{r}'], A[f'{r}|{c}|{eg}|vs_C0']
        o[f'{c}|{eg}'] = dict(d_ann=x['ALL']['ann'] - y['ALL']['ann'], d_t=x['ALL']['t_vs_0050'] - y['ALL']['t_vs_0050'],
                              seg1_gex=x['SEG1']['gex'], seg1_gex_ref=y['SEG1']['gex'], seg1_t=x['SEG1']['t_vs_0050'], seg1_t_ref=y['SEG1']['t_vs_0050'],
                              seg1_sign_same=bool(np.sign(x['SEG1']['gex']) == np.sign(y['SEG1']['gex'])), all_t=x['ALL']['t_vs_0050'],
                              t_diff_vs_ref=x['ALL']['t_diff_vs_ref'], n_win_diff=x['ALL']['n_win_diff'])
    dA[f'{k}_vs_{r}'] = o
negl = all(abs(v['normal|freeze']['d_ann']) <= 0.3 and abs(v['normal|freeze']['d_t']) <= 0.05 and v['normal|freeze']['seg1_sign_same'] for v in dA.values())
same = all(e['seg1_sign_same'] for v in dA.values() for e in v.values())
tier = 'R1 可忽略' if negl else ('R2 方向不變' if same else 'R3 可能翻號')
DEC['A'] = dict(diffs=dA, tier=tier,
                text={'R1 可忽略': '金融遮罩 PIT 化對主策略報酬的影響可忽略：維持 FM「只加註」，加註文字附上年化差與 t 差',
                      'R2 方向不變': '影響不可忽略但第一段超額方向不變：維持 FM「只加註」（FM 的事前計數規則 M ≤ 5% 不被整合推翻），加註年化差與 t 差區間，並建議研究中心評估是否改正式數字',
                      'R3 可能翻號': '影響大到第一段超額可能翻號：FM 計數判定不被推翻，但整合建議研究中心下一輪立題以 PIT 遮罩重算主策略正式數字'}[tier],
                G5_seg1='不新增項目（依 FM 事前規則 M＝1.73% ≤ 5%；整合結果不得推翻）', G4_seg1='未證實（依 ER2，不變）', sop='不改 SOP（現行金融排除以 2026 產業別為準，對當下名單正確；問題只在 2006～2013 歷史）')
c1 = C['V7X_100萬_worse']['all']
DEC['C'] = dict(consistent_with_hist_demo=CHK['consistent'], comb_S1=c1['comb']['S1'], comb_FAILED=c1['comb']['FAILED'], oc_power_min=POWMIN,
                statement=('不一致，待查（一票否決，不下結論）' if not CHK['consistent'] else
                           ('以歷史實際報酬看，G8P-2a 下 24 個月 S1 翻案路徑走不通' if c1['comb']['S1'] < POWMIN else '歷史起點升級比例不低於 OC 檢定力下限，描述但不改協定')),
                protocol_change='不改協定（描述性）')
DEC['B'] = '不做（PREREG.B_mix）：本輪兩題都沒有報酬序列；0050 固定比例混合在「不再開」清單，integrate_o14 已全部不可用'
DEC['overall'] = '主策略與所有變體都不可投資；SOP 不改；真錢維持 100% 0050'
RO1 = ro_snapshot()
d0 = {r[0]: r for r in RO0['rows']}; d1_ = {r[0]: r for r in RO1['rows']}
chg = sorted(p for p in set(d0) | set(d1_) if (p not in d0) or (p not in d1_) or d0[p][1] != d1_[p][1])
RES['READONLY_CHECK'] = dict(start=dict(utc=RO0['utc'], n=RO0['n'], sha=RO0['sha']), end=dict(utc=RO1['utc'], n=RO1['n'], sha=RO1['sha']),
                             identical=(RO0['sha'] == RO1['sha']), changed=chg,
                             changed_mtime_utc={p: pd.Timestamp(d1_[p][3], unit='s', tz='UTC').isoformat() for p in chg if p in d1_},
                             note='每日排程 06:49 台北（22:49Z）寫入的檔案以 mtime 時段區分；本程式對這兩個資料夾只讀')
RJ.write_text(json.dumps(dict(R, RESULTS=RES, DECISION=DEC, RUN_UTC=now()), ensure_ascii=False, indent=1, default=str), encoding='utf-8')
log('寫入 results.json')
print(json.dumps(DEC, ensure_ascii=False, indent=1, default=str))
