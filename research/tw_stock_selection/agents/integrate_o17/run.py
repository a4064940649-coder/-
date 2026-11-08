#!/usr/bin/env python3
"""integrate_o17 正式執行（只跑一次）：驗 PREREG 雜湊與輸入 sha 後，把 o17 通過審查的兩題（都不產生選股訊號、沒有自己的報酬序列）
套到主策略 C0（main_v2 gapfix 官方名單、排除金融 n10），做「勘誤當濾網」的報酬敏感度（不取代主策略正式數字，也不回寫兩題）：
  A 營收可得時點濾網（o_rev_avail_ds_pit_count R1）：D=S／D>S 訊號日，凡 V6 分類器判定不是 AVAIL（上界口徑）／判定 UNAVAIL（下界口徑）的股票，
    營收改用前一個月（y3 back=1，和 main_v2 AB 相同的回退）；WORST＝D=S／D>S 日全部回退（v2_core.make_unavail('worst')，描述）。
  B 存活者缺口放回 × 下市對價引擎（R2 × o_engine_delist_consideration）：pclib.VU('opt', GOV) 放回期間 2013～2026（U13）與 2006～2026（UFULL），
    引擎 freeze／nofreeze／dzero／cons_main／cons_lo／cons_hi／cons_c50。
  C 官方處置期濾網（R3）：2006-01-01～2021-09-23 訊號日，官方全歷史處置清單在訊號日或成交日涵蓋者加入排除（與 ia_core 原遮罩取聯集）。
  D 合併（ALLPIT）：UFULL 放回 ＋ A 上界回退 ＋ C 官方處置，cons_main／lo／hi 與 freeze。
  E（第二條腿固定比例混合）依 PREREG 不做。
python3 -I -B run.py                  正式
INTEG_SMOKE=1 python3 -I -B run.py    登記前管線核對（C0 重現、名單計數與兩題公開計數逐筆相同；不算任何變體報酬，不寫 results.json）"""
import os, sys, json, math, hashlib, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
RA = AG / 'o_rev_avail_ds_pit_count'
CZ = AG / 'o_engine_delist_consideration'
ER = AG / 'o_rev_trunc_2009_erratum'
RJ = HERE / 'results.json'
SMOKE = bool(os.environ.get('INTEG_SMOKE'))
T0 = time.time()
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)
def now(): return pd.Timestamp.now('UTC').isoformat()

RO_DIRS = [AG / 'live_tracking' / 'main_forward', AG / 'o_forward_infra_live', AG / 'o_g8p2a_protocol_addendum']
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
    P_ = R['PREREG']
    assert hashlib.sha256(json.dumps(P_, ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256'], 'PREREG 被改'
    for f, h in P_['sha256'].items():
        assert sha(f) == h, f'{f} 雜湊不符'
    RO0 = ro_snapshot()

# =============================================================== 載入（只讀 import）
sys.path.insert(0, str(RA))
import pclib as P          # → main_v2/v2_core → independent_audit/ia_core
import avail as AV
sys.path.pop(0)
sys.path.insert(0, str(CZ))
import cz_engine as CE     # → 同一個 ia_core 模組
sys.path.pop(0)
V, K = P.V, P.K
assert CE.K is K
COLS, N, IDX = P.COLS, P.N, P.IDX
SPEC1 = json.loads((RA / 'SPEC.json').read_text(encoding='utf-8'))
assert sha(RA / 'SPEC.json') == '691695b65c26092860889a75292289f376bf9da753da20cc775ce08da9dc0f0b'
assert P.ENGINE_SHA == SPEC1['engine_sha256'], 'pclib 引擎 sha 與題 1 凍結不同'
R1J = json.loads((RA / 'results.json').read_text(encoding='utf-8'))
R2J = json.loads((CZ / 'results.json').read_text(encoding='utf-8'))
assert R2J['PREREG_sha256'] == '046d2dcb23fdd4022cba37262a4105076012c274d399d3b7cf2d51db60d2de1b'
assert sha(AG / 'independent_audit' / 'ia_core.py') == R2J['PREREG']['file_sha256']['ia_core.py']
EV = pd.read_csv(CZ / 'events_v2.csv', dtype={'code': str, 'successor': str})
assert sha(CZ / 'events_v2.csv') == '66e80725a4e48387690ded34fc1ebb5b4b7a71ca2b501448c81b30f583b0afbe'
DD_ = R2J['PREREG']['events_v2']['summary']['d']['d_used']
ST = {'cons_main': CE.build_settlement(EV, 'main', d=DD_), 'cons_lo': CE.build_settlement(EV, 'lo', d=DD_),
      'cons_hi': CE.build_settlement(EV, 'hi', d=DD_), 'cons_c50': CE.build_settlement(EV, 'main', d=DD_, cash_0050=True)}
EVCODES = set(EV.code)
log('載入完成')

# =============================================================== A：營收可得時點（V6 分類器，題 1 凍結版）
E6 = AV.Evidence('sig')
_ucache = {}
def case_of(d): return V.deadline_case(d)
def u_evidence(kind):
    """kind：'UB'（非 AVAIL 都回退）｜'LB'（只有 UNAVAIL 回退）。只在 D=S／D>S 訊號日作用。"""
    def U(d):
        key = (kind, d)
        if key in _ucache: return _ucache[key]
        u = np.zeros(N, bool)
        if case_of(d) in ('D=S', 'D>S'):
            ym = str(pd.Period(d, 'M') - 1)
            for j, c in enumerate(COLS):
                s = E6.classify(pd.Timestamp(d), c, ym, 'V6')[0]
                u[j] = (s != 'AVAIL') if kind == 'UB' else (s == 'UNAVAIL')
        _ucache[key] = u
        return u
    return U
UW = V.make_unavail('worst')
UNAV = {'UB': u_evidence('UB'), 'LB': u_evidence('LB'), 'WORST': UW}

# =============================================================== C：官方處置期（題 1 raw/disposal_official_universe.csv）
DU = pd.read_csv(RA / 'raw' / 'disposal_official_universe.csv', dtype={'code': str}, parse_dates=['start', 'end'])
assert sha(RA / 'raw' / 'disposal_official_universe.csv') == SPEC1['files_sha256'].get('raw/disposal_official_universe.csv', sha(RA / 'raw' / 'disposal_official_universe.csv'))
DSP_HI = pd.Timestamp('2021-09-23')
_orig_dm = K.disposal_mask
_dcache = {}
def dm_official(d):
    """ia_core 原遮罩 ∪（d ≤ 2021-09-23 時：官方處置期涵蓋訊號日 d 或成交日 e＝d 的下一個面板交易日）。"""
    d = pd.Timestamp(d)
    if d in _dcache: return _dcache[d]
    m = _orig_dm(d).copy()
    if d <= DSP_HI:
        t = IDX.get_loc(d); e = IDX[t + 1] if t + 1 < len(IDX) else d
        x = DU[((DU.start <= d) & (DU.end >= d)) | ((DU.start <= e) & (DU.end >= e))]
        for c in x.code:
            if c in P.CI: m[P.CI[c]] = True
    _dcache[d] = m
    return m

# =============================================================== 變體與名單
def build(kind):
    """回傳 (plan, t0, t1, sd, aff)。aff：放回股（VU）。"""
    rev, ins, dsp = kind
    K.disposal_mask = dm_official if dsp else _orig_dm
    try:
        if ins is None:
            S = V.V2('gapfix', P.YL, P.Y5, exfin=True, unavail=(UNAV[rev] if rev else None))
            aff = {}
        else:
            S = P.VU('opt', lo=ins, hi='2026-12-31', which='GOV')
            if rev: S.unavail = UNAV[rev]
            aff = S.aff
        pl = S.plan(10, 'official')
        t0, t1, sd = S.bounds()
        nold = {str(d.date()): int(S._cs[d]['nold']) for d in S._cs} if rev else {}
    finally:
        K.disposal_mask = _orig_dm
    return dict(plan=pl, t0=t0, t1=t1, sd=list(sd), aff={k: list(v) for k, v in (aff.items() if ins else [])}, nold=nold)

VARS = {'C0': (None, None, False),
        'A_UB': ('UB', None, False), 'A_LB': ('LB', None, False), 'A_WORST': ('WORST', None, False),
        'U13': (None, '2013-01-01', False), 'UFULL': (None, '2006-01-01', False),
        'C_DSP': (None, None, True),
        'ALLPIT': ('UB', '2006-01-01', True)}
B = {}
for k, kind in VARS.items():
    B[k] = build(kind); log('名單', k, len(B[k]['plan']))
assert all(B[k]['sd'] == B['C0']['sd'] for k in VARS), '訊號日不一致'
SD = pd.DatetimeIndex(B['C0']['sd'])
PK = {k: {p[3]: [COLS[j] for j in p[1]] for p in B[k]['plan']} for k in VARS}
DAYS = list(PK['C0'])
def ins_slots(k, a='2006-01-01', b='2026-12-31'):
    out = []
    for p in B[k]['plan']:
        d = p[3]
        if pd.Timestamp(a) <= d <= pd.Timestamp(b):
            A_ = set(B[k]['aff'].get(d, []))
            out += [(str(d.date()), COLS[j]) for j in p[1] if COLS[j] in A_]
    return out
def changed(k, a='2006-01-01', b='2026-12-31'):
    return int(sum(len(set(PK['C0'][d]) - set(PK[k][d])) for d in DAYS if pd.Timestamp(a) <= d <= pd.Timestamp(b)))

# ---- 核對（任何一條不符 → 停）
CHK = {}
L0 = json.loads((AG / 'o_g8_protocol_fin_erratum' / 'ERRATUM.json').read_text(encoding='utf-8'))['lists_C0']
days13 = [d for d in DAYS if d <= pd.Timestamp('2013-12-31')]
assert len(days13) == 93 and all(PK['C0'][d] == L0[str(d.date())] for d in days13), 'C0 與 o14 lists_C0 不同'
SL1 = pd.read_csv(RA / 'data' / 'c0_slots.csv', dtype={'code': str, 'sig': str})
SL1 = SL1[SL1.formal.astype(str) == 'True']
c0rows = sorted((s, c) for s, c in zip(SL1.sig, SL1.code))
assert c0rows == sorted((str(d.date()), c) for d in DAYS for c in PK['C0'][d]), 'C0 槽位與題 1 c0_slots 不同'
r2a = R1J['ERRATUM']['R2']['panelA_C0']
for seg, (a, b) in {'2013_2026': ('2013-01-01', '2026-12-31'), '2013_2018': ('2013-01-01', '2018-12-31'), '2019_2026': ('2019-01-01', '2026-12-31')}.items():
    n = len(ins_slots('U13', a, b)); assert n == r2a[seg]['U_opt'], ('U13', seg, n, r2a[seg]['U_opt'])
    CHK[f'U13_{seg}'] = n
ER2 = json.loads((ER / 'ERRATUM2.json').read_text(encoding='utf-8'))
uf13 = sorted(map(list, ins_slots('UFULL', '2006-01-01', '2013-12-31')))
CHK['UFULL_2006_2013_vs_ER2'] = dict(n=len(uf13), n_er2=len(ER2['U_opt_slot_list']), equal=(uf13 == sorted(map(list, ER2['U_opt_slot_list']))))
assert CHK['UFULL_2006_2013_vs_ER2']['equal'], 'UFULL 2006～2013 放回槽位與 ER2 不同'
CHK['UFULL_2014_2026'] = len(ins_slots('UFULL', '2014-01-01', '2026-12-31'))
CHK['ALLPIT_ins'] = len(ins_slots('ALLPIT'))
# A：D=S／D>S 日以外，名單只會因 held 集合（開盤漲停遞補）不同
dsd = [d for d in DAYS if case_of(d) in ('D=S', 'D>S')]
CHK['n_DS_DGS_days'] = len(dsd)
assert len(dsd) == 91, len(dsd)
for k in ('A_UB', 'A_LB', 'A_WORST'):
    CHK[f'{k}_changed_slots'] = dict(all=changed(k), on_DS=int(sum(len(set(PK['C0'][d]) - set(PK[k][d])) for d in dsd)),
                                     y2019=changed(k, '2019-01-01'), y0613=changed(k, '2006-01-01', '2013-12-31'),
                                     mean_lagged_top300_on_DS=float(np.mean([B[k]['nold'][str(d.date())] for d in dsd])))
# 題 1 C0 槽位層級的上界／下界和 A_UB／A_LB 的回退集合一致（只核對 C0 被選槽：上界＝非 AVAIL）
SA = pd.read_csv(RA / 'data' / 'slot_avail.csv', dtype={'code': str, 'sig': str})
SA = SA[SA.formal.astype(str) == 'True']
ub_c = {(s, c) for s, c, u in zip(SA.sig, SA.code, SA.upper) if str(u) == 'True'}
lb_c = {(s, c) for s, c, u in zip(SA.sig, SA.code, SA.lower) if str(u) == 'True'}
mine_ub = {(str(d.date()), c) for d in dsd for c in PK['C0'][d] if UNAV['UB'](d)[P.CI[c]]}
mine_lb = {(str(d.date()), c) for d in dsd for c in PK['C0'][d] if UNAV['LB'](d)[P.CI[c]]}
CHK['slot_UB_equal_topic1'] = dict(n_mine=len(mine_ub), n_topic1=len(ub_c), equal=(mine_ub == ub_c))
CHK['slot_LB_equal_topic1'] = dict(n_mine=len(mine_lb), n_topic1=len(lb_c), equal=(mine_lb == lb_c))
assert mine_ub == ub_c and mine_lb == lb_c, '槽位上／下界與題 1 不同'
# C：C0 槽位落在官方處置期的數目與題 1 R3 相同（訊號日或成交日）
r3 = R1J['ERRATUM']['R3']['counts']
def in_off(d, c):
    t = IDX.get_loc(d); e = IDX[t + 1]
    g = DU[DU.code == c]
    return bool((((g.start <= d) & (g.end >= d)) | ((g.start <= e) & (g.end >= e))).any())
for seg, (a, b) in {'2006_2013': ('2006-01-01', '2013-12-31'), '2014_2021-09-23': ('2014-01-01', '2021-09-23')}.items():
    hit = sorted(f'{d.date()} {c}' for d in DAYS if pd.Timestamp(a) <= d <= pd.Timestamp(b) for c in PK['C0'][d] if in_off(d, c))
    ref = sorted(set(r3[seg]['d_sig']['slots']) | set(r3[seg]['d_trd']['slots']))
    CHK[f'C0_in_official_disposal_{seg}'] = dict(n=len(hit), n_topic1_union=len(ref), equal=(hit == ref))
    assert hit == ref, ('處置槽位與題 1 不同', seg, hit, ref)
CHK['C_DSP_changed_slots'] = dict(all=changed('C_DSP'), y0613=changed('C_DSP', '2006-01-01', '2013-12-31'), y1421=changed('C_DSP', '2014-01-01', '2021-09-23'))
CHK['changed_slots'] = {k: changed(k) for k in VARS}
log('核對通過', CHK)

# =============================================================== 模擬
COST = {'normal': K.rates(0.001), 'stress': K.rates('stress')}
OLD = ('freeze', 'nofreeze', 'dzero')
ENG = {'C0': ('freeze', 'nofreeze', 'dzero', 'cons_main', 'cons_lo', 'cons_hi', 'cons_c50'),
       'A_UB': ('freeze', 'cons_main', 'cons_lo', 'cons_hi'), 'A_LB': ('freeze', 'cons_main', 'cons_lo', 'cons_hi'), 'A_WORST': ('freeze', 'cons_main', 'cons_lo', 'cons_hi'),
       'U13': ('freeze', 'nofreeze', 'dzero', 'cons_main', 'cons_lo', 'cons_hi', 'cons_c50'),
       'UFULL': ('freeze', 'nofreeze', 'dzero', 'cons_main', 'cons_lo', 'cons_hi', 'cons_c50'),
       'C_DSP': ('freeze', 'cons_main', 'cons_lo', 'cons_hi'),
       'ALLPIT': ('freeze', 'nofreeze', 'dzero', 'cons_main', 'cons_lo', 'cons_hi', 'cons_c50')}
NAV, LOGS = {}, {}
def sim(k, c, eg):
    pl = [(p[0], p[1]) for p in B[k]['plan']]; bb, ss = COST[c]
    if eg in OLD: nav, lg = CE.simulate(pl, bb, ss, B[k]['t0'], B[k]['t1'], mode=eg)
    else: nav, lg = CE.simulate(pl, bb, ss, B[k]['t0'], B[k]['t1'], mode='consideration', settle=ST[eg])
    return nav, lg
OFF = pd.read_parquet(AG / 'main_v2' / 'navs_gapfix.parquet')
C0DIFF = {}
for c in COST:
    nav, lg = sim('C0', c, 'freeze'); navk, _ = K.simulate([(p[0], p[1]) for p in B['C0']['plan']], *COST[c], B['C0']['t0'], B['C0']['t1'])
    ref = OFF[f'EXFIN|n10|{c}'].dropna()
    C0DIFF[c] = dict(vs_main_v2=float(np.nanmax(np.abs(nav.reindex(ref.index).to_numpy() - ref.to_numpy()))),
                     cz_vs_ia=float(np.nanmax(np.abs(nav.to_numpy() - navk.reindex(nav.index).to_numpy()))))
    assert C0DIFF[c]['vs_main_v2'] < 1e-9 and C0DIFF[c]['cz_vs_ia'] < 1e-12, C0DIFF
    NAV[('C0', c, 'freeze')] = nav
log('C0 重現', C0DIFF)

sys.path.insert(0, str(AG / 'o_forward_sequential_v2'))
import fslib as F
sys.path.pop(0)
E50 = F.e50()
END = pd.Timestamp(V.END1012)
def win(nav):
    pts = list(SD[SD < END]) + [END]
    rows = [(pd.Period(a, 'M'), F._at(nav, b) / F._at(nav, a) - 1, F._at(E50, b) / F._at(E50, a) - 1, a.year) for a, b in zip(pts[:-1], pts[1:])]
    W = pd.DataFrame(rows, columns=['per', 'r', 'r50', 'yr']).set_index('per')
    return W[~W.index.duplicated(keep='last')].dropna()

if SMOKE:
    w = win(NAV[('C0', 'normal', 'freeze')])
    ndis = {k: sum(1 for p in B[k]['plan'] for j in p[1] if COLS[j] in EVCODES) for k in VARS}
    out = dict(utc=now(), C0_reproduce=C0DIFF, n_windows=len(w), first=str(w.index[0]), last=str(w.index[-1]), checks=CHK,
               n_slots_in_event_table=ndis, n_settle_rules={k: sum(len(v) for v in s.values()) for k, s in ST.items()},
               lists_sha={k: hashlib.sha256(json.dumps({str(d.date()): v for d, v in PK[k].items()}, sort_keys=True).encode()).hexdigest() for k in VARS},
               note='只做管線核對；沒有計算任何變體的報酬、t 值（C0 只算 freeze 一般／壓力以核對 main_v2，C0 報酬早已公開）。')
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    sys.exit(0)

for k in VARS:
    for eg in ENG[k]:
        for c in (('normal', 'stress') if eg in ('freeze', 'cons_main', 'cons_lo') else ('normal',)):
            if (k, c, eg) in NAV: continue
            nav, lg = sim(k, c, eg); NAV[(k, c, eg)] = nav
            if c == 'normal':
                LOGS[f'{k}|{eg}'] = dict(n_frozen_events=len(lg['frozen']), frozen_codes=sorted({x for _, cs, _ in lg['frozen'] for x in cs}),
                                         frozen_max_share=max([x[2] for x in lg['frozen']] or [0.0]),
                                         n_settled=len(lg.get('settled', [])), settled_codes=sorted({s['code'] for s in lg.get('settled', [])}),
                                         settled_types=pd.Series([s['type'] for s in lg.get('settled', [])]).value_counts().to_dict() if lg.get('settled') else {})
    log('模擬', k)

# =============================================================== 統計
PERS = {'ALL': (None, None), 'SEG1': ('2006-01', '2012-12'), 'SEG2': ('2013-01', '2018-12'), 'SEG3': ('2019-01', None),
        'SEL': (None, '2019-04'), 'VAL': ('2019-05', None), 'HOLD': ('2024-01', None), 'Y0613': ('2006-01', '2013-12'), 'Y14ON': ('2014-01', None)}
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
        a, b, c = rv[m], r50[m], rM[m]
        o[k] = dict(n=int(m.sum()), ann=ann(a), ann_0050=ann(b), gex=ann(a) - ann(b), mdd=mdd(a), t_vs_0050=tt(a - b),
                    mean_ex_pp=float(np.mean(a - b) * 100), diff_vs_ref_pp=float(np.mean(a - c) * 100), t_diff_vs_ref=tt(a - c),
                    ann_diff_vs_ref=ann(a) - ann(c), corr_ref=corr(a, c), corr_0050=corr(a, b), corr_ex=corr(a - b, c - b),
                    n_win_diff=int(np.sum(np.abs(a - c) > 1e-12)))
    s = pd.Series(rv - r50).groupby(np.asarray(yrs)).sum()
    b3 = list(s.sort_values(ascending=False).index[:3]); m = ~np.isin(yrs, b3)
    o['ALL']['t_drop_best3'] = tt(rv[m] - r50[m]); o['ALL']['best3'] = [int(x) for x in b3]
    return o
WIN = {key: win(nav) for key, nav in NAV.items()}
ST_ = {}
for (k, c, eg), W in WIN.items():
    refs = [('C0', c, 'freeze')]
    if k in ('U13', 'UFULL', 'ALLPIT') and eg != 'freeze': refs.append((k, c, 'freeze'))
    for rk in refs:
        J = WIN[rk].join(W[['r']], rsuffix='_k', how='inner')
        st = stats_one(J['r_k'].to_numpy(), J['r50'].to_numpy(), J['r'].to_numpy(), J.index, J['yr'].to_numpy())
        nav = NAV[(k, c, eg)]; st['ALL']['mdd_daily'] = float((nav / nav.cummax() - 1).min() * 100)
        ST_[f'{k}|{c}|{eg}|vs_{rk[0]}{"" if rk[2] == "freeze" and rk[0] == "C0" else "_" + rk[2]}'] = st
RES = dict(checks=CHK, C0_reproduce=C0DIFF, logs=LOGS, stats=ST_)
# 描述：A 的回退股數、B 的放回槽位（依年、依發行人）
RES['desc'] = dict(A_nold_on_DS={k: {str(d.date()): B[k]['nold'][str(d.date())] for d in dsd} for k in ('A_UB', 'A_LB', 'A_WORST')},
                   U13_ins_by_year=pd.Series([s[:4] for s, _ in ins_slots('U13')]).value_counts().sort_index().to_dict(),
                   UFULL_ins_by_year=pd.Series([s[:4] for s, _ in ins_slots('UFULL')]).value_counts().sort_index().to_dict(),
                   U13_ins_by_issuer=pd.Series([c for _, c in ins_slots('U13')]).value_counts().to_dict(),
                   U13_issuers_not_in_event_table=sorted({c for _, c in ins_slots('U13')} - EVCODES))
log('統計完成')

# =============================================================== 判定（PREREG.decision）
PAIRS = {'A_UB': ['freeze'], 'A_LB': ['freeze'], 'C_DSP': ['freeze'],
         'U13': ['cons_main', 'cons_lo', 'cons_hi', 'cons_c50'], 'UFULL': ['cons_main', 'cons_lo', 'cons_hi', 'cons_c50'],
         'ALLPIT': ['cons_main', 'cons_lo', 'cons_hi', 'cons_c50']}
def ref_stats(c): return ST_[f'C0|{c}|freeze|vs_C0']
DEC = {}
for k, egs in PAIRS.items():
    rows = []
    for eg in egs:
        for c in ('normal', 'stress'):
            key = f'{k}|{c}|{eg}|vs_C0'
            if key not in ST_: continue
            x, y = ST_[key], ref_stats(c)
            r = dict(engine=eg, cost=c, d_ann=x['ALL']['ann'] - y['ALL']['ann'], d_t=x['ALL']['t_vs_0050'] - y['ALL']['t_vs_0050'],
                     t_all=x['ALL']['t_vs_0050'], t_seg3=x['SEG3']['t_vs_0050'], t_drop_best3=x['ALL']['t_drop_best3'],
                     t_diff_vs_C0=x['ALL']['t_diff_vs_ref'], corr_C0=x['ALL']['corr_ref'])
            for s in ('SEG1', 'SEG2', 'SEG3'):
                r[f'{s}_gex'] = x[s]['gex']; r[f'{s}_gex_C0'] = y[s]['gex']; r[f'{s}_sign_same'] = bool(np.sign(x[s]['gex']) == np.sign(y[s]['gex']))
            rows.append(r)
    nrm = [r for r in rows if r['cost'] == 'normal']
    worst = max(nrm, key=lambda r: max(abs(r['d_ann']) / 0.3, abs(r['d_t']) / 0.05))
    negl = all(abs(r['d_ann']) <= 0.3 and abs(r['d_t']) <= 0.05 and r['SEG1_sign_same'] and r['SEG2_sign_same'] and r['SEG3_sign_same'] for r in nrm)
    same = all(r['SEG1_sign_same'] and r['SEG2_sign_same'] and r['SEG3_sign_same'] for r in rows)
    tier = 'R1 可忽略' if negl else ('R2 方向不變' if same else 'R3 可能翻號')
    DEC[k] = dict(rows=rows, worst_normal=worst, tier=tier)
TXT = {'R1 可忽略': '對主策略報酬的影響可忽略；題目的事前計數判定（加註）不變，加註旁附上年化差與 t 差',
       'R2 方向不變': '影響不可忽略但三段超額方向不變；題目的事前計數判定不變，加註旁附上年化差與 t 差區間；建議研究中心評估是否改正式數字',
       'R3 可能翻號': '某段超額可能翻號；題目的事前計數判定不變，整合建議下一輪立題以該口徑重算主策略正式數字'}
for k in DEC: DEC[k]['text'] = TXT[DEC[k]['tier']]
x = ST_['ALLPIT|normal|cons_lo|vs_C0']
DEC['G2_under_ALLPIT_cons_lo'] = dict(t_all=x['ALL']['t_vs_0050'], t_seg3=x['SEG3']['t_vs_0050'], G2=bool(x['ALL']['t_vs_0050'] >= 3.0 and x['SEG3']['t_vs_0050'] >= 2.0))
DEC['E_mix'] = '不做（PREREG.E_mix）：兩題都沒有報酬序列；勘誤版主策略與 C0 相關 ≈ 1，當第二條腿等於自己混自己；0050 固定比例混合在「不再開」清單（integrate_o14 已全部不可用）'
DEC['overall'] = '主策略與所有變體都不可投資；SOP 不改；真錢維持 100% 0050'
RO1 = ro_snapshot()
d0 = {r[0]: r for r in RO0['rows']}; d1_ = {r[0]: r for r in RO1['rows']}
chg = sorted(p for p in set(d0) | set(d1_) if (p not in d0) or (p not in d1_) or d0[p][1] != d1_[p][1])
RES['READONLY_CHECK'] = dict(start=dict(utc=RO0['utc'], n=RO0['n'], sha=RO0['sha']), end=dict(utc=RO1['utc'], n=RO1['n'], sha=RO1['sha']),
                             identical=(RO0['sha'] == RO1['sha']), changed=chg)
RJ.write_text(json.dumps(dict(R, RESULTS=RES, DECISION=DEC, RUN_UTC=now(), runtime_s=time.time() - T0), ensure_ascii=False, indent=1, default=str), encoding='utf-8')
log('寫入 results.json')
print(json.dumps({k: (dict(tier=v['tier'], worst=v['worst_normal']) if isinstance(v, dict) and 'tier' in v else v) for k, v in DEC.items()}, ensure_ascii=False, indent=1, default=str))
