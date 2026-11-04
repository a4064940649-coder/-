#!/usr/bin/env python3
"""integrate_o14 正式執行（只跑一次）：驗 PREREG 雜湊後，跑 A 濾網（勘誤 PIT 金融分類換進主策略）、B 主策略×0050 固定比例混合、
C 凍結協定 G8P-1 的歷史套用。輸出寫入 results.json 的 RESULTS／DECISION（只寫一次）。
python3 -I -B run.py          正式
INTEG_SMOKE=1 python3 -I -B run.py   登記前管線核對（只做 C0 重現、名單異動數、序列長度；不算任何組合績效、不寫檔）"""
import os, sys, json, csv, re, math, hashlib, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
G8E = AG / 'o_g8_protocol_fin_erratum'
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

PROTO = json.loads((G8E / 'G8_PROTOCOL.json').read_text(encoding='utf-8'))
assert sha(G8E / 'G8_PROTOCOL.json') == '799f8b845b69f4df6f18a6e9cc901aa804fcd0d5dee393ef2bd97f7449b113fa'
assert sha(G8E / 'g8lib.py') == PROTO['code_sha256']['g8lib.py']
G8R = json.loads((G8E / 'results.json').read_text(encoding='utf-8'))
SPEC = G8R['ERRATUM_SPEC']
ERR = G8R['ERRATUM']

# =============================================================== A：濾網（勘誤 PIT 金融分類）
sys.path.insert(0, str(AG / 'main_v2'))
import v2_core as V
sys.path.pop(0)
K = V.K
COLS = list(K.COLS); CI = {c: j for j, c in enumerate(COLS)}
O13 = AG / 'o_pit_shares_industry_2006_2013'
TSE_FIN = 17
for rel, m in SPEC['snapshots'].items():
    assert sha(O13 / 'raw' / rel) == m['sha256'], rel

def load_snap(rel):
    p = O13 / 'raw' / rel; mp, nm = {}, {}
    rows = list(csv.reader(open(p, encoding='utf-8', errors='replace')))
    for r in rows[1:]:
        r = [x.strip().strip('"') for x in r]
        if len(r) >= 3 and r[0].isdigit() and len(r[0]) == 4:
            if 'twse_list' in rel:
                if len(r) >= 4 and r[2].isdigit(): mp[r[0]] = int(r[2]) == TSE_FIN; nm[r[0]] = r[1]
            else:
                try: mp[r[0]] = int(r[2]) == TSE_FIN; nm[r[0]] = r[1]
                except ValueError: pass
    return mp, nm

SN = []
for rel, m in SPEC['snapshots'].items():
    mp, nm = load_snap(rel)
    SN.append(dict(date=pd.Timestamp(m['effective_date']), fin=mp, name=nm))
SN.sort(key=lambda s: s['date'])
NAMES = {}
for s in SN:
    for c, n in s['name'].items(): NAMES.setdefault(c, set()).add(n)
for r in csv.DictReader(open('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', encoding='utf-8')):
    if r['code'].isdigit() and len(r['code']) == 4: NAMES.setdefault(r['code'], set()).add(r['name'])
KW = re.compile('金控|金融控股|銀行|商銀|保險|產險|人壽|壽險|證券|票券|期貨|(銀|金|證|保|壽)$')
EXCL = {'2049', '2947', '3049', '4576', '7819', '9917', '9925'}
C2SET = {c for c in COLS if c[:2] in ('28', '58', '60')} | {c for c in COLS if c in NAMES and c not in EXCL and any(KW.search(n) for n in NAMES[c])}
C2MASK = np.array([c in C2SET for c in COLS])
C0MASK = V.FINMASK.copy()
HI = pd.Timestamp('2013-12-31')

def c1_mask(d, back=False):
    pre = [s for s in SN if s['date'] <= d]
    s = pre[-1] if pre else (SN[0] if back else None)
    m = C0MASK.copy()
    if s is None: return m
    for c, f in s['fin'].items():
        j = CI.get(c)
        if j is not None: m[j] = f
    return m

MASKS = {'C0': lambda d: C0MASK,
         'C1': lambda d: c1_mask(d) if d <= HI else C0MASK,
         'C2': lambda d: C2MASK if d <= HI else C0MASK,
         'C1B': lambda d: c1_mask(d, back=True) if d <= HI else C0MASK}

YL, Y5, _ = V.load_revenue()
class VM(V.V2):
    def __init__(self, maskfn, **kw):
        self.maskfn = maskfn
        super().__init__('gapfix', YL, Y5, exfin=True, **kw)
    def y3(self, d, back=0):
        Y = self.Y5 if d >= V.SWITCH else self.YL
        y = K.yoy3_for_month(Y, pd.Period(d, 'M') - back)
        return np.where(self.maskfn(d), np.nan, y)

COST = {'normal': K.rates(0.001), 'stress': K.rates('stress')}
PL, NAV = {}, {}
for k, fn in MASKS.items():
    S = VM(fn)
    pl = S.plan(10, 'official'); PL[k] = pl
    t0, t1, sd = S.bounds()
    for c, (bb, ss) in COST.items():
        NAV[(k, c)], _ = K.simulate([(p[0], p[1]) for p in pl], bb, ss, t0, t1)
    log('A', k)
SD = pd.DatetimeIndex(sd)
OFF = pd.read_parquet(AG / 'main_v2' / 'navs_gapfix.parquet')
for c in COST:
    ref = OFF[f'EXFIN|n10|{c}'].dropna(); mine = NAV[('C0', c)].reindex(ref.index)
    dmax = float(np.nanmax(np.abs(mine.to_numpy() - ref.to_numpy())))
    assert dmax < 1e-9, f'C0 未重現 main_v2 {c}：{dmax}'
pk = {k: {d: set(COLS[j] for j in p[1]) for (_, _, _, d), p in zip(PL[k], PL[k])} for k in PL}
days13 = [d for d in pk['C0'] if pd.Timestamp('2006-01-01') <= d <= HI]
SLOTS = {k: int(sum(len(pk['C0'][d] - pk[k][d]) for d in days13)) for k in ('C1', 'C2', 'C1B')}
assert len(days13) == ERR['n_signal_days'], (len(days13), ERR['n_signal_days'])
for k in SLOTS: assert SLOTS[k] == ERR['slots'][k]['n_changed'], (k, SLOTS[k], ERR['slots'][k]['n_changed'])
SLOTS_AFTER = {k: int(sum(len(pk['C0'][d] - pk[k][d]) for d in pk['C0'] if d > HI)) for k in ('C1', 'C2', 'C1B')}
log('A 核對通過', SLOTS, '2014 起異動', SLOTS_AFTER)

# =============================================================== 共用：fslib（窗、錨定月、0050）
sys.path.insert(0, str(AG / 'o_forward_sequential_v2'))
import fslib as F
sys.path.pop(0)
E = F.e50(); Dm = F.deadlines()
JS = list(range(13))

def win_v2(nav):
    """訊號日窗：r、r50、月份（訊號日所屬月）。"""
    pts = list(SD[SD < V.END1012]) + [pd.Timestamp(V.END1012)]
    rows = []
    for a, b in zip(pts[:-1], pts[1:]):
        rows.append((pd.Period(a, 'M'), F._at(nav, b) / F._at(nav, a) - 1, F._at(E, b) / F._at(E, a) - 1, a.year))
    D = pd.DataFrame(rows, columns=['per', 'r', 'r50', 'yr']).set_index('per')
    return D[~D.index.duplicated(keep='last')].dropna()

def win_j(nav, strat, j):
    W = F.window_excess(nav, F.j_signals(j, strat), F.j_end(j, strat), E, Dm)
    W = W[~W.index.duplicated(keep='last')].dropna(subset=['r', 'r50'])
    W = W[[x is not None for x in W.index]]
    W.index = pd.PeriodIndex(W.index, freq='M'); W['yr'] = W.index.year
    return W[['r', 'r50', 'yr']]

if SMOKE:
    w = win_v2(NAV[('C0', 'normal')])
    X = pd.read_parquet(AG / 'o_exec_engine_rewrite_v7x' / 'out' / 'navs_oos_R4.parquet', columns=[f'B10|J{j}|K5T|normal|100萬|GROW' for j in JS])
    lens = {j: len(win_j(X[f'B10|J{j}|K5T|normal|100萬|GROW'], 'AB_V7_B10', j)) for j in JS}
    print(json.dumps(dict(v2_windows=len(w), v2_first=str(w.index[0]), v2_last=str(w.index[-1]), x_windows=lens, slots=SLOTS, slots_after=SLOTS_AFTER), ensure_ascii=False))
    sys.exit(0)

# =============================================================== 統計工具
PERS = {'ALL': (None, None), 'SEG1': ('2006-01', '2012-12'), 'SEG2': ('2013-01', '2018-12'), 'SEG3': ('2019-01', None),
        'SEL': (None, '2019-04'), 'VAL': ('2019-05', None), 'HOLD': ('2024-01', None)}
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
def corr(a, b): return float(np.corrcoef(a, b)[0, 1])

def mix(r, r50, w, tc):
    if w == 1.0: return np.asarray(r, float).copy()
    out = []; wd = None
    for a, b in zip(r, r50):
        cost = abs(wd - w) * tc if wd is not None else 0.0
        out.append(w * a + (1 - w) * b - cost)
        wd = w * (1 + a) / (w * (1 + a) + (1 - w) * (1 + b))
    return np.array(out)

def stats_one(rm, r50, rM, idx, yrs, best3=None):
    o = {}
    for k in PERS:
        m = pmask(idx, k)
        if m.sum() < 6: continue
        a, b, c = rm[m], r50[m], rM[m]
        o[k] = dict(n=int(m.sum()), ann=ann(a), ann_0050=ann(b), gex=ann(a) - ann(b), mdd=mdd(a), mdd_0050=mdd(b),
                    t_vs_0050=tt(a - b), mean_ex_pp=float(np.mean(a - b) * 100),
                    diff_vs_M_pp=float(np.mean(a - c) * 100), t_diff_vs_M=tt(a - c),
                    corr_M=corr(a, c), corr_0050=corr(a, b), corr_ex=corr(a - b, c - b) if np.std(a - b) > 0 else float('nan'))
    if best3 is not None:
        m = ~np.isin(yrs, best3)
        o['ALL']['t_drop_best3'] = tt(rm[m] - r50[m])
    return o

def best3_years(rM, r50, yrs):
    s = pd.Series(rM - r50).groupby(np.asarray(yrs)).sum()
    return list(s.sort_values(ascending=False).index[:3])

def agg_j(per_j):
    """per_j：{j: stats dict} → 每個 期間×欄位 的 j 中位數與 p10。"""
    out = {}
    js = list(per_j)
    for k in per_j[js[0]]:
        out[k] = {}
        for f in per_j[js[0]][k]:
            v = np.array([per_j[j][k][f] for j in js if k in per_j[j]], float)
            out[k][f] = dict(med=float(np.nanmedian(v)), p10=float(np.nanpercentile(v, 10)))
    return out

RES = {}
# =============================================================== A 結果
A = {'slots_2006_2013': SLOTS, 'slots_2014_after': SLOTS_AFTER, 'C0_reproduces_main_v2': True}
W0 = {c: win_v2(NAV[('C0', c)]) for c in COST}
for k in MASKS:
    A[k] = {}
    for c in COST:
        Wk = win_v2(NAV[(k, c)]); W = W0[c].join(Wk[['r']], rsuffix='_k', how='inner')
        st = stats_one(W['r_k'].to_numpy(), W['r50'].to_numpy(), W['r'].to_numpy(), W.index, W['yr'].to_numpy())
        nav = NAV[(k, c)]; st['ALL']['mdd_daily'] = float((nav / nav.cummax() - 1).min() * 100)
        A[k][c] = st
RES['A_filter'] = A
log('A 完成')

# =============================================================== B 混合
W_LIST = [1.0, 0.8, 0.7, 0.5, 0.3]
TC = {'normal': 0.005, 'stress': 0.010}
B = {}
legs = {}
for c in COST:
    W = W0[c]; legs[f'M_v2|{c}'] = {0: W}
JV7 = pd.read_parquet(AG / 'o_forward_sequential_v2' / 'raw/v7/navs_jscan_v7.parquet')
for c in COST:
    legs[f'M_v7|{c}'] = {j: win_j(JV7[f'AB_V7|{c}|J{j}'], 'AB_V7', j) for j in JS}
del JV7
TIERS = ['100萬', '300萬', '1000萬']
XC = [f'B10|J{j}|K5T|{c}|{t}|GROW' for j in JS for c in COST for t in TIERS]
X = pd.read_parquet(AG / 'o_exec_engine_rewrite_v7x' / 'out' / 'navs_oos_R4.parquet', columns=XC)
for c in COST:
    for t in TIERS:
        legs[f'M_x|{c}|{t}'] = {j: win_j(X[f'B10|J{j}|K5T|{c}|{t}|GROW'], 'AB_V7_B10', j) for j in JS}
del X
log('B 腿載入', {k: len(v) for k, v in legs.items()})
MIXR = {}
for lg, perj in legs.items():
    c = lg.split('|')[1]
    B[lg] = {}
    for w in W_LIST:
        sj = {}
        for j, W in perj.items():
            r, r50, yr = W['r'].to_numpy(), W['r50'].to_numpy(), W['yr'].to_numpy()
            rm = mix(r, r50, w, TC[c])
            MIXR[(lg, w, j)] = pd.Series(rm, index=W.index)
            sj[j] = stats_one(rm, r50, r, W.index, yr, best3=best3_years(r, r50, yr))
        B[lg][f'w={w}'] = agg_j(sj) if len(sj) > 1 else {k: {f: dict(med=v, p10=v) for f, v in d.items()} for k, d in sj[0].items()}
    # 0050 單獨
    W = perj[0]; B[lg]['E50'] = {k: dict(ann=ann(W['r50'].to_numpy()[pmask(W.index, k)]), mdd=mdd(W['r50'].to_numpy()[pmask(W.index, k)])) for k in PERS}
RES['B_mix'] = B
log('B 完成')

# =============================================================== C 協定歷史套用
sys.path.insert(0, str(G8E))
import g8lib as G
sys.path.pop(0)
par0 = dict(sigma=PROTO['params']['sigma_design'], c=PROTO['params']['c'], c_d=PROTO['params']['c_d'], b=PROTO['params']['b'],
            up_looks=PROTO['params']['up_looks'], dn_every=6)
assert PROTO['layer2_sequential']['rule']['rule'] == 'R-FIX'
def jmed_series(lg, w):
    M = pd.DataFrame({j: (MIXR[(lg, w, j)] - legs[lg][j]['r50']) * 100 for j in JS})
    return F.jmed(M, min_j=PROTO['rule']['jmed_min_valid_j'])
def apply_protocol(x, w):
    x = np.asarray(x, float); n = len(x) - 24 + 1
    Xp = np.stack([x[s:s + 24] for s in range(n)])
    par = dict(par0, sigma=par0['sigma'] * w)
    u, d = G.run_rule('R-FIX', Xp, par, Hdec=24, refit=True)
    g8 = Xp[:, :12].sum(axis=1) > 0
    uu = np.where(g8, u, 0); dd = np.where(g8, d, 12)
    fail = (dd > 0) & (dd <= 24); up = (uu > 0) & (uu <= 24) & ~fail
    st = np.where(fail, 'FORWARD_FAILED', np.where(up, 'S1_EXCEPTION_EVIDENCE', 'FORWARD_VALIDATING_NO_REJECTION_24'))
    return st, g8, u, d
C = {}
for lg in ('M_x|normal|100萬', 'M_v7|normal'):
    C[lg] = {}
    for w in W_LIST:
        s = jmed_series(lg, w)
        st, g8, u, d = apply_protocol(s.to_numpy(), w)
        starts = s.index[:len(st)]
        def frac(mask):
            m = np.asarray(mask)
            return dict(n=int(m.sum()), FORWARD_FAILED=float(np.mean(st[m] == 'FORWARD_FAILED')) if m.any() else None,
                        S1_EXCEPTION_EVIDENCE=float(np.mean(st[m] == 'S1_EXCEPTION_EVIDENCE')) if m.any() else None,
                        NO_REJECTION_24=float(np.mean(st[m] == 'FORWARD_VALIDATING_NO_REJECTION_24')) if m.any() else None,
                        literal_g8_pass=float(np.mean(g8[m])) if m.any() else None,
                        down_trigger=float(np.mean(d[m] > 0)) if m.any() else None)
        yrs = np.array([p.year for p in starts])
        C[lg][f'w={w}'] = dict(n_months=len(s), first=str(s.index[0]), last=str(s.index[-1]),
                               mean_pp=float(s.mean()), sd_pp=float(s.std(ddof=1)),
                               all=frac(np.ones(len(st), bool)), s2006_2012=frac(yrs <= 2012), s2013_2018=frac((yrs >= 2013) & (yrs <= 2018)),
                               s2019_2024=frac(yrs >= 2019),
                               last_start=dict(start=str(starts[-1]), status=str(st[-1]), z12=None),
                               last12_sum_pp=float(s.iloc[-12:].sum()), last12_from=str(s.index[-12]),
                               last24_sum_pp=float(s.iloc[-24:].sum()),
                               by_start=dict(zip([str(p) for p in starts], [str(x) for x in st])))
RES['C_protocol'] = C
log('C 完成')
try:
    G.SCR.rmdir()
except OSError:
    pass

# =============================================================== 判定
DEC = {}
a = RES['A_filter']
def adiff(k):
    x, y = a[k]['normal']['ALL'], a['C0']['normal']['ALL']
    return dict(d_ann=x['ann'] - y['ann'], d_t=x['t_vs_0050'] - y['t_vs_0050'],
                seg1_sign_same=bool(np.sign(a[k]['normal']['SEG1']['gex']) == np.sign(a['C0']['normal']['SEG1']['gex'])))
dA = {k: adiff(k) for k in ('C1', 'C2', 'C1B')}
negl = all(abs(dA[k]['d_ann']) <= 0.3 and abs(dA[k]['d_t']) <= 0.05 and dA[k]['seg1_sign_same'] for k in ('C1', 'C2'))
DEC['A'] = dict(diffs=dA, negligible=negl, verdict='影響可忽略：維持 G8E 結論（不加註、不改標 G4）' if negl else '影響不可忽略：建議主策略正式數字旁加註量化差',
                sop='不改 SOP')
b = RES['B_mix']
m1_t = b['M_v7|normal']['w=1.0']['ALL']['t_vs_0050']['med']
for w in W_LIST[1:]:
    k = f'w={w}'
    c1 = b['M_v7|normal'][k]['ALL']['t_vs_0050']['med'] >= 2.0
    c2 = b['M_x|normal|100萬'][k]['SEG3']['gex']['med'] > 0
    c3 = all(b[f'M_x|normal|{t}'][k]['ALL']['gex']['med'] > 0 and b[f'M_x|normal|{t}'][k]['ALL']['gex']['p10'] > 0 for t in TIERS)
    c4 = b['M_x|stress|100萬'][k]['ALL']['gex']['med'] > 0
    c5 = b['M_v7|normal'][k]['ALL']['t_vs_0050']['med'] > m1_t
    v = '可用' if all([c1, c2, c3, c4, c5]) else ('待觀察' if c1 else '不可用')
    DEC[k] = dict(c1=c1, c2=c2, c3=c3, c4=c4, c5=c5, M_alone_v7_t=m1_t, verdict=v)
RJ.write_text(json.dumps(dict(R, RESULTS=RES, DECISION=DEC, RUN_UTC=pd.Timestamp.utcnow().isoformat()), ensure_ascii=False, indent=1, default=str), encoding='utf-8')
log('寫入 results.json')
print(json.dumps(DEC, ensure_ascii=False, indent=1, default=str))
