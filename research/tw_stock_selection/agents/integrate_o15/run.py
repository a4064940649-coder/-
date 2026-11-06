#!/usr/bin/env python3
"""integrate_o15 正式執行（只跑一次）：驗 PREREG 雜湊後，跑
  A 加分項／敏感度：把 o_rev_trunc_2009_erratum（ER2）的 U 口徑（缺營收的存活者偏誤型股票給營收百分位 1.0／0.9／沿用最後已知 y3）
    換進主策略（排除金融 n10，main_v2 gapfix 官方名單），量化 2006～2013 名單改變對報酬的影響（ER2 本身只計數，本塊是整合階段的報酬敏感度）；
  C 協定歷史套用：獨立重寫 G8P-2（G8_PROTOCOL_v2.json 選定版本 V1）的決策線與規則，套到 V7X 可執行歷史線，對照 G8P-1 與 G8P-2 的 HIST_DEMO。
  B（與 0050 或其他腿固定比例混合）依 PREREG 不做，理由見 PREREG.B_mix。
輸出寫入 results.json 的 RESULTS／DECISION（只寫一次）。
python3 -I -B run.py                  正式
INTEG_SMOKE=1 python3 -I -B run.py    登記前管線核對（C0 重現、U 槽位數與 ER2 一致、序列長度、決策線均值對 HIST_DEMO；不算任何組合績效，不寫 results.json）"""
import os, sys, json, math, hashlib, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
ER = AG / 'o_rev_trunc_2009_erratum'
G2 = AG / 'o_g8p2_protocol_refreeze'
G1 = AG / 'o_g8_protocol_fin_erratum'
RJ = HERE / 'results.json'
SMOKE = bool(os.environ.get('INTEG_SMOKE'))
T0 = time.time()
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)

if not SMOKE:
    R = json.loads(RJ.read_text(encoding='utf-8'))
    assert 'RESULTS' not in R, 'RESULTS 已存在：只跑一次'
    P = R['PREREG']
    assert hashlib.sha256(json.dumps(P, ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256'], 'PREREG 被改'
    for f, h in P['sha256'].items():
        assert sha(f) == h, f'{f} 雜湊不符'

# =============================================================== A：ER2 U 口徑 × 主策略
ER2 = json.loads((ER / 'ERRATUM2.json').read_text(encoding='utf-8'))
ER2R = json.loads((ER / 'results.json').read_text(encoding='utf-8'))
assert sha(ER / 'ERRATUM2_SPEC.json') == ER2R['PREREG']['sha256'] == '722f7d56eec3674980445d179996c8d3f04226d91fe502b7a0f14260c846126e'
sys.path.insert(0, str(ER))
import erlib as EL          # 只讀 import：營收缺值原因分類、VU（U 口徑排序）；不寫任何檔
sys.path.pop(0)
V, K = EL.V, EL.K
COLS = EL.COLS
LO, HI = EL.LO, EL.HI
VARS = {'C0': ('base', 'SURV'), 'U_opt': ('opt', 'SURV'), 'U_p90': ('p90', 'SURV'), 'U_carry': ('carry', 'SURV'), 'U_opt_ALL': ('opt', 'ALL')}
COST = {'normal': K.rates(0.001), 'stress': K.rates('stress')}
ENG = {'freeze': dict(), 'dzero': dict(delist_zero=True)}   # freeze＝正式引擎（凍結持股停在最後價值）；dzero＝下市最壞情境（之後再無成交的凍結持股歸零）
PL, NAV, AFF, SDS = {}, {}, {}, {}
for k, (mode, which) in VARS.items():
    S = EL.VU(mode, which)
    pl = S.plan(10, 'official'); PL[k] = pl
    t0, t1, sd = S.bounds(); SDS[k] = list(sd)
    AFF[k] = {d: set(v) for d, v in S.aff.items()}
    for c, (bb, ss) in COST.items():
        for eg, kw in ENG.items():
            if eg == 'dzero' and c == 'stress': continue
            NAV[(k, c, eg)], _ = K.simulate([(p[0], p[1]) for p in pl], bb, ss, t0, t1, **kw)
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
L0 = json.loads((G1 / 'ERRATUM.json').read_text(encoding='utf-8'))['lists_C0']
days13 = [d for d in PK['C0'] if LO <= d <= HI]
assert len(days13) == 93
for d in days13: assert PK['C0'][d] == L0[str(d.date())], d
CNT = {}
for k in VARS:
    if k == 'C0': continue
    ins = sorted((str(d.date()), c) for d in days13 for c in PK[k][d] if c in AFF[k].get(d, set()))
    u12 = sum(1 for s, _ in ins if s < '2013-01-01'); u13 = len(ins)
    after = int(sum(len(set(PK['C0'][d]) ^ set(PK[k][d])) // 2 for d in PK['C0'] if d > HI))
    CNT[k] = dict(U_2006_2012=u12, U_2006_2013=u13, changed_after_2013=after, slot_sha=hashlib.sha256(json.dumps([list(x) for x in ins]).encode()).hexdigest())
    ref = ER2['counts'][k]
    assert (u12, u13) == (ref['U_2006_2012'], ref['U_2006_2013']), (k, u12, u13, ref)
assert CNT['U_opt']['slot_sha'] == ER2R['ERRATUM2']['slot_list_sha256'], 'U_opt 槽位清單雜湊與 ER2 不同'
assert sorted(tuple(x) for x in ER2['U_opt_slot_list']) == sorted((str(d.date()), c) for d in days13 for c in PK['U_opt'][d] if c in AFF['U_opt'].get(d, set())), 'U_opt 槽位清單與 ER2 不同'
log('A 核對通過', CNT)

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
HD = json.loads((G2 / 'results.json').read_text(encoding='utf-8'))['HIST_DEMO']

if SMOKE:
    w = win_v2(NAV[('C0', 'normal', 'freeze')])
    out = dict(utc=pd.Timestamp.utcnow().isoformat(), C0_maxabs_vs_main_v2=C0DIFF, v2_windows=len(w), v2_first=str(w.index[0]), v2_last=str(w.index[-1]),
               counts=CNT, lines={k: dict(n=len(s), first=str(s.index[0]), last=str(s.index[-1]), mean=float(s.mean()), sd=float(s.std(ddof=1))) for k, s in LINES.items()},
               hist_demo_ref=dict(worse_mean=HD['V_sel_worse']['mean_pp'], worse_sd=HD['V_sel_worse']['sd_pp'], normal_mean=HD['G8P1_normal']['mean_pp']),
               note='只做管線核對；沒有計算任何變體或組合的報酬、t 值或協定結果。')
    print(json.dumps(out, ensure_ascii=False, indent=1))
    sys.exit(0)

# =============================================================== 統計工具
PERS = {'ALL': (None, None), 'SEG1': ('2006-01', '2012-12'), 'SEG2': ('2013-01', '2018-12'), 'SEG3': ('2019-01', None),
        'SEL': (None, '2019-04'), 'VAL': ('2019-05', None), 'HOLD': ('2024-01', None),
        'Y0609': ('2006-01', '2009-12'), 'Y1013': ('2010-01', '2013-12'), 'Y0613': ('2006-01', '2013-12'), 'Y14ON': ('2014-01', None)}
def pmask(idx, k):
    a, b = PERS[k]; m = np.ones(len(idx), bool)
    if a: m &= idx >= pd.Period(a, 'M')
    if b: m &= idx <= pd.Period(b, 'M')
    return m
def tt(x):
    x = np.asarray(x, float); s = x.std(ddof=1)
    return float(x.mean() / s * math.sqrt(len(x))) if len(x) > 2 and s > 0 else float('nan')
def ann(r): r = np.asarray(r, float); return float(np.prod(1 + r) ** (12 / len(r)) - 1) * 100
def mdd(r):
    nav = np.cumprod(1 + np.asarray(r, float)); pk_ = np.maximum.accumulate(np.concatenate([[1.0], nav]))[1:]
    return float((nav / pk_ - 1).min()) * 100
def corr(a, b): return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float('nan')

def stats_one(rv, r50, rM, idx, yrs):
    o = {}
    for k in PERS:
        m = pmask(idx, k)
        if m.sum() < 6: continue
        a, b, c = rv[m], r50[m], rM[m]
        o[k] = dict(n=int(m.sum()), ann=ann(a), ann_0050=ann(b), gex=ann(a) - ann(b), mdd=mdd(a), t_vs_0050=tt(a - b),
                    mean_ex_pp=float(np.mean(a - b) * 100), diff_vs_C0_pp=float(np.mean(a - c) * 100), t_diff_vs_C0=tt(a - c),
                    ann_diff_vs_C0=ann(a) - ann(c), corr_C0=corr(a, c), corr_0050=corr(a, b), corr_ex=corr(a - b, c - b),
                    n_win_diff=int(np.sum(np.abs(a - c) > 1e-12)))
    s = pd.Series(rv - r50).groupby(np.asarray(yrs)).sum()
    b3 = list(s.sort_values(ascending=False).index[:3])
    m = ~np.isin(yrs, b3)
    o['ALL']['t_drop_best3'] = tt(rv[m] - r50[m]); o['ALL']['best3'] = [int(x) for x in b3]
    yr = {}
    for y in range(2006, 2014):
        mm = yrs == y
        if mm.any(): yr[str(y)] = dict(ann=float((np.prod(1 + rv[mm]) - 1) * 100), ex_vs_0050=float((np.prod(1 + rv[mm]) - np.prod(1 + r50[mm])) * 100),
                                      ex_vs_C0=float((np.prod(1 + rv[mm]) - np.prod(1 + rM[mm])) * 100))
    o['yearly_2006_2013'] = yr
    return o

RES = {}
# =============================================================== A 結果
A = dict(C0_maxabs_vs_main_v2=C0DIFF, counts=CNT)
for eg in ENG:
    for c in COST:
        if eg == 'dzero' and c == 'stress': continue
        W0 = win_v2(NAV[('C0', c, eg)])
        for k in VARS:
            Wk = win_v2(NAV[(k, c, eg)]); W = W0.join(Wk[['r']], rsuffix='_k', how='inner')
            st = stats_one(W['r_k'].to_numpy(), W['r50'].to_numpy(), W['r'].to_numpy(), W.index, W['yr'].to_numpy())
            nav = NAV[(k, c, eg)]; st['ALL']['mdd_daily'] = float((nav / nav.cummax() - 1).min() * 100)
            A[f'{k}|{c}|{eg}'] = st
# 描述：進榜受影響股與被擠出原名單股的「訊號日收盤→下一訊號日收盤」紙上報酬（含息還原、下市後持平）
ADJ = K.ADJ.ffill()
def fwd(d, codes):
    i = SD.get_loc(d); b = SD[i + 1] if i + 1 < len(SD) else pd.Timestamp(V.END1012)
    a0 = ADJ.loc[:d].iloc[-1]; b0 = ADJ.loc[:b].iloc[-1]
    return [float(b0[c] / a0[c] - 1) for c in codes if np.isfinite(a0[c]) and np.isfinite(b0[c])]
DESC = {}
for k in ('U_opt', 'U_p90', 'U_carry'):
    ins_r, out_r, e50r = [], [], []
    for d in days13:
        ins = [c for c in PK[k][d] if c in AFF[k].get(d, set())]
        if not ins: continue
        outs = [c for c in PK['C0'][d] if c not in PK[k][d]]
        ins_r += fwd(d, ins); out_r += fwd(d, outs)
    DESC[k] = dict(n_ins=len(ins_r), mean_ins_pct=float(np.mean(ins_r) * 100) if ins_r else None, median_ins_pct=float(np.median(ins_r) * 100) if ins_r else None,
                   n_out=len(out_r), mean_out_pct=float(np.mean(out_r) * 100) if out_r else None, median_out_pct=float(np.median(out_r) * 100) if out_r else None,
                   t_ins_minus_out_welch=float((np.mean(ins_r) - np.mean(out_r)) / math.sqrt(np.var(ins_r, ddof=1) / len(ins_r) + np.var(out_r, ddof=1) / len(out_r))) if len(ins_r) > 2 and len(out_r) > 2 else None)
A['desc_inserted_vs_displaced'] = DESC
RES['A_bonus'] = A
log('A 完成')

# =============================================================== C 協定歷史套用（獨立重寫）
PV2 = json.loads((G2 / 'G8_PROTOCOL_v2.json').read_text(encoding='utf-8'))
assert sha(G2 / 'G8_PROTOCOL_v2.json') == 'dcab4dbd255fb11e0ad8f4508397ae109cae9a47090eee17ab2ec7952981b993'
PV1 = json.loads((G1 / 'G8_PROTOCOL.json').read_text(encoding='utf-8'))
assert sha(G1 / 'G8_PROTOCOL.json') == '799f8b845b69f4df6f18a6e9cc901aa804fcd0d5dee393ef2bd97f7449b113fa'
p2 = PV2['params']
SPEC = {'V1': dict(kind='MAXSD', sig=p2['sigma_design'], c=p2['c'], b=p2['b'], c_d=p2['c_d']),
        'G8P1': dict(kind='FIX_REFIT', sig=PV1['params']['sigma_design'], c=PV1['params']['c'], b=PV1['params']['b'], c_d=PV1['params']['c_d'])}
assert p2['stat'] == 'MAXSD' and p2['delta'] == 0.0 and p2['upgrade'] is True

def judge(x, sp):
    """一條 24 個月路徑 → (狀態, 細節)。順序：12 月 字面→下修→升級；18 月 下修；24 月 下修→升級；先觸發者吸收。"""
    x = np.asarray(x, float); S = np.cumsum(x); cs = sp['c'] * sp['sig']
    z = {}; sdn = {}; sig_fix = cs; trig = False
    for n in (12, 18, 24):
        s_n = float(np.std(x[:n], ddof=1)); sdn[n] = s_n
        if sp['kind'] == 'MAXSD':
            z[n] = S[n - 1] / (max(cs, s_n) * math.sqrt(n))
        else:
            if (not trig) and s_n > 1.5 * sp['sig']:
                sig_fix = max(cs, s_n); trig = True
            z[n] = S[n - 1] / (sig_fix * math.sqrt(n))
    det = dict(S12=float(S[11]), S24=float(S[23]), z12=float(z[12]), z18=float(z[18]), z24=float(z[24]), sd12=sdn[12], sd24=sdn[24])
    if S[11] <= 0: return 'FORWARD_FAILED', 'literal', det
    if z[12] <= -sp['c_d']: return 'FORWARD_FAILED', 'down12', det
    if z[12] >= sp['b'] * math.sqrt(2): return 'S1_EXCEPTION_EVIDENCE', 'up12', det
    if z[18] <= -sp['c_d']: return 'FORWARD_FAILED', 'down18', det
    if z[24] <= -sp['c_d']: return 'FORWARD_FAILED', 'down24', det
    if z[24] >= sp['b']: return 'S1_EXCEPTION_EVIDENCE', 'up24', det
    return 'FORWARD_VALIDATING_NO_REJECTION_24', 'none', det

def apply(s, sp):
    x = s.to_numpy(); n = len(x) - 24 + 1
    rows = [judge(x[i:i + 24], sp) for i in range(n)]
    st = np.array([r[0] for r in rows]); why = np.array([r[1] for r in rows])
    starts = s.index[:n]; yrs = np.array([p.year for p in starts])
    def frac(m):
        return dict(n=int(m.sum()), S1=float(np.mean(st[m] == 'S1_EXCEPTION_EVIDENCE')), FAILED=float(np.mean(st[m] == 'FORWARD_FAILED')),
                    NO_REJ=float(np.mean(st[m] == 'FORWARD_VALIDATING_NO_REJECTION_24')), literal=float(np.mean(why[m] != 'literal')),
                    down=float(np.mean(np.isin(why[m], ['down12', 'down18', 'down24']))))
    return dict(n_months=len(x), first=str(s.index[0]), last=str(s.index[-1]), mean_pp=float(x.mean()), sd_pp=float(x.std(ddof=1)),
                all=frac(np.ones(n, bool)), s2006_2012=frac(yrs <= 2012), s2013_2018=frac((yrs >= 2013) & (yrs <= 2018)), s2019_2024=frac(yrs >= 2019),
                sel_starts_2006_2017=frac(starts <= pd.Period('2017-04', 'M')), val_starts_2017_2022=frac((starts > pd.Period('2017-04', 'M')) & (starts <= pd.Period('2021-12', 'M'))),
                hold_starts_2022_on=frac(starts > pd.Period('2021-12', 'M')),
                s1_starts=[str(p) for p, q in zip(starts, st) if q == 'S1_EXCEPTION_EVIDENCE'],
                last_start=dict(start=str(starts[-1]), status=str(st[-1]), why=str(why[-1]), **rows[-1][2]),
                last12_sum_pp=float(x[-12:].sum()), last24_sum_pp=float(x[-24:].sum()),
                why_counts={k: int(v) for k, v in zip(*np.unique(why, return_counts=True))},
                by_start={str(p): str(q) for p, q in zip(starts, st)})
C = {}
for nm, sp in SPEC.items():
    for ln, s in LINES.items():
        C[f'{nm}|{ln}'] = apply(s, sp)
# 對照 HIST_DEMO（G8P-2 自報）與 integrate_o14（G8P-1）
O14 = json.loads((AG / 'integrate_o14' / 'results.json').read_text(encoding='utf-8'))['RESULTS']['C_protocol']['M_x|normal|100萬']['w=1.0']
CHK = dict(V1_worse_vs_HIST_DEMO={f: (C['V1|V7X_100萬_worse']['all'][f], HD['V1_worse']['all'][f]) for f in ('S1', 'FAILED', 'NO_REJ', 'literal')},
           V1_worse_bystart_equal_HIST_DEMO_s1=C['V1|V7X_100萬_worse']['s1_starts'] == HD['V1_worse']['s1_starts'],
           G8P1_normal_vs_HIST_DEMO={f: (C['G8P1|V7X_100萬_normal']['all'][f], HD['G8P1_normal']['all'][f]) for f in ('S1', 'FAILED', 'NO_REJ', 'literal')},
           G8P1_normal_vs_o14=dict(S1=(C['G8P1|V7X_100萬_normal']['all']['S1'], O14['all']['S1_EXCEPTION_EVIDENCE']),
                                   FAILED=(C['G8P1|V7X_100萬_normal']['all']['FAILED'], O14['all']['FORWARD_FAILED']),
                                   bystart_equal=all(C['G8P1|V7X_100萬_normal']['by_start'].get(k) == (v if v != 'FORWARD_VALIDATING_NO_REJECTION_24' else v) for k, v in O14['by_start'].items())))
CHK['consistent'] = bool(all(abs(a - b) < 1e-12 for a, b in CHK['V1_worse_vs_HIST_DEMO'].values()) and all(abs(a - b) < 1e-12 for a, b in CHK['G8P1_normal_vs_HIST_DEMO'].values())
                         and CHK['V1_worse_bystart_equal_HIST_DEMO_s1'] and CHK['G8P1_normal_vs_o14']['bystart_equal'])
C['check'] = CHK
C['shift_worse_minus_normal_pp'] = {t: float((LINES[f'V7X_{t}_worse'] - LINES[f'V7X_{t}_normal']).mean()) for t in TIERS}
RES['C_protocol'] = C
log('C 完成', CHK['consistent'])

# =============================================================== 判定
DEC = {}
def g(k, c='normal', eg='freeze'): return A[f'{k}|{c}|{eg}']
c0 = g('C0')
dA = {}
for k in ('U_opt', 'U_p90', 'U_carry'):
    o = {}
    for c, eg in (('normal', 'freeze'), ('stress', 'freeze'), ('normal', 'dzero')):
        x, y = g(k, c, eg), g('C0', c, eg)
        o[f'{c}|{eg}'] = dict(d_ann=x['ALL']['ann'] - y['ALL']['ann'], d_t=x['ALL']['t_vs_0050'] - y['ALL']['t_vs_0050'],
                              seg1_gex=x['SEG1']['gex'], seg1_gex_C0=y['SEG1']['gex'], seg1_t=x['SEG1']['t_vs_0050'],
                              seg1_sign_same=bool(np.sign(x['SEG1']['gex']) == np.sign(y['SEG1']['gex'])),
                              all_t=x['ALL']['t_vs_0050'])
    dA[k] = o
negl = all(abs(dA[k]['normal|freeze']['d_ann']) <= 0.3 and abs(dA[k]['normal|freeze']['d_t']) <= 0.05 and dA[k]['normal|freeze']['seg1_sign_same'] for k in ('U_opt', 'U_p90'))
same = all(dA[k][e]['seg1_sign_same'] for k in ('U_opt', 'U_p90', 'U_carry') for e in dA[k])
tier = 'R1 可忽略' if negl else ('R2 方向不變' if same else 'R3 可能翻號')
DEC['A'] = dict(diffs=dA, tier=tier,
                text={'R1 可忽略': '報酬敏感度可忽略：G4 第一段仍依 ER2 規則標「未證實」（真實營收未知、槽位上界 > 5%），但加註「U 口徑報酬差可忽略」',
                      'R2 方向不變': '報酬敏感度不可忽略但第一段超額方向不變：G4 第一段仍標「未證實」，加註 U 口徑的年化差與 t 差區間',
                      'R3 可能翻號': '報酬敏感度大到可能翻號：G4 第一段標「未證實」並加註「存活者缺口可能使 2006～2012 超額翻號」'}[tier],
                G4_seg1='未證實（依 ER2 事前規則，整合結果不得推翻）', G5_seg1='存活者偏誤未排除', sop='不改 SOP（現行月營收資料對當下名單是完整的；截斷只影響 2009-12～2013 歷史）')
c1 = C['V1|V7X_100萬_worse']['all']
DEC['C'] = dict(consistent_with_hist_demo=CHK['consistent'], V1_worse_S1=c1['S1'], V1_worse_FAILED=c1['FAILED'],
                G8P1_normal_S1=C['G8P1|V7X_100萬_normal']['all']['S1'],
                statement=('以歷史實際報酬看，G8P-2 的 24 個月 S1 翻案路徑走不通' if c1['S1'] < 0.046 else '歷史起點升級比例不低於 OC 檢定力下限，描述但不改協定'),
                protocol_change='不改協定（描述性）')
DEC['B'] = '不做（PREREG.B_mix）：本輪兩題都沒有報酬序列；0050 固定比例混合在「不再開」清單，integrate_o14 已全部不可用'
DEC['overall'] = '主策略與所有變體都不可投資；SOP 不改；真錢維持 100% 0050'
RJ.write_text(json.dumps(dict(R, RESULTS=RES, DECISION=DEC, RUN_UTC=pd.Timestamp.utcnow().isoformat()), ensure_ascii=False, indent=1, default=str), encoding='utf-8')
log('寫入 results.json')
print(json.dumps(DEC, ensure_ascii=False, indent=1, default=str))
