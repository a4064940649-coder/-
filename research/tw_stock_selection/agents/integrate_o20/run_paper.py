#!/usr/bin/env python3
"""integrate_o20 紙上整合（PREREG 之後只跑一次）：o18 正式管線（只讀 import o18lib），處置遮罩 M0（o18 原）／M1（審查員修正＝現行正式紙上上緣）／M2（全期統一，J1）；
C0（只 M0）、ALLPIT_REAL_{opt,p90,carry} × {一般, 壓力}；J4 固定比例混合 80/20、50/50（主策略 M1、第二條腿 M2），每窗再平衡。
遮罩定義與題目 k_target/k_paper.py 相同（本檔自寫一份，不 import 該檔）；名單 sha 與窗報酬須等於題目輸出。
python3 -I -B run_paper.py"""
import sys, json, hashlib, time, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
RJ = HERE / 'results.json'; R = json.loads(RJ.read_text(encoding='utf-8'))
assert hashlib.sha256(json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256']
assert 'RESULTS_PAPER' not in R, '只跑一次'
for f, h in R['PREREG']['file_sha256'].items(): assert hashlib.sha256((AG / f).read_bytes()).hexdigest() == h, f
T0 = time.time()
T1 = AG / 'o_delist_rev_backfill_allpit'; S2 = AG / 'o_g6_allpit_list_trace_stage2'
sys.path.insert(0, str(T1)); import o18lib as O; sys.path.pop(0)
X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/')
D5 = pd.read_csv(X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
U = pd.read_csv(AG / 'o_disposal_list_pit_audit_pkl18/data/universe_announce.csv', dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
base = O.dm_official
def nxt(d):
    t = O.IDX.get_loc(d); return O.IDX[t + 1] if t + 1 < len(O.IDX) else d
c1, c2 = {}, {}
def m1(d):
    d = pd.Timestamp(d)
    if d in c1: return c1[d]
    m = base(d).copy()
    if d > O.DSP_HI:
        e = nxt(d)
        for x in (D5[(D5.date <= d) & (D5.start <= e) & (D5.end >= e)].code, U[(U.announce <= d) & (U.start <= e) & (U.end >= e)].code):
            for c in x:
                if c in O.CI: m[O.CI[c]] = True
    c1[d] = m; return m
def m2(d):
    d = pd.Timestamp(d)
    if d in c2: return c2[d]
    e = nxt(d); m = np.zeros(len(O.COLS), bool)
    x = U[(U.announce <= d) & (U.start <= e) & (U.end >= e)]
    assert all((a <= d) and (s <= e <= en) for a, s, en in zip(x.announce, x.start, x.end))
    for c in x.code:
        if c in O.CI: m[O.CI[c]] = True
    c2[d] = m; return m
VARS = {'C0': ('base', None, None, False, None), 'ALLPIT_REAL_opt': ('opt', '2006-01-01', 'UB', True, 'B1'),
        'ALLPIT_REAL_p90': ('p90', '2006-01-01', 'UB', True, 'B1'), 'ALLPIT_REAL_carry': ('carry', '2006-01-01', 'UB', True, 'B1')}
FILLS = ['ALLPIT_REAL_opt', 'ALLPIT_REAL_p90', 'ALLPIT_REAL_carry']
KS = json.loads((S2 / 'out/k_paper_stats.json').read_text(encoding='utf-8'))
KW = np.load(S2 / 'out/k_paper_windows.npz')
B, PK, LSH = {}, {}, {}
for tag, fn in (('M0', base), ('M1', m1), ('M2', m2)):
    O.dm_official = fn
    for k, kind in VARS.items():
        if k == 'C0' and tag != 'M0': continue
        b = O.build(kind); B[(tag, k)] = b
        P = {p[3]: [O.COLS[j] for j in p[1]] for p in b['plan']}; PK[(tag, k)] = P
        LSH[(tag, k)] = hashlib.sha256(json.dumps({str(d.date()): v for d, v in P.items()}, sort_keys=True).encode()).hexdigest()
        assert LSH[(tag, k)] == KS[f'{tag}|{k}|lists_sha'], ('名單 sha 與題目不同', tag, k)
O.dm_official = base
SD0 = B[('M0', 'C0')]['sd']
NAV, W = {}, {}
for (tag, k), b in B.items():
    for c in ('normal', 'stress'):
        NAV[(tag, k, c)] = O.sim(b, c, 'freeze' if k == 'C0' else 'cons_main')[0]
        W[(tag, k, c)] = O.win(NAV[(tag, k, c)], SD0)
rep_win = {}
for key, w in W.items():
    kk = '|'.join(key); a = KW[kk + '|r']; assert len(a) == len(w), kk
    rep_win[kk] = float(np.max(np.abs(a - w['r'].to_numpy(float))))
assert max(rep_win.values()) < 1e-10, rep_win
# 窗端點（日曆年化用）
SDx = pd.DatetimeIndex(SD0); pts = list(SDx[SDx < O.END]) + [O.END]
WD = pd.DataFrame([(pd.Period(a, 'M'), a, b) for a, b in zip(pts[:-1], pts[1:])], columns=['per', 'a', 'b']).set_index('per')
WD = WD[~WD.index.duplicated(keep='last')]
def cal_ann(w, k):
    m = O.pmask(w.index, k); ww = w[m]
    if not len(ww): return None
    a, b = WD.loc[ww.index[0], 'a'], WD.loc[ww.index[-1], 'b']; days = (b - a).days
    return float(np.prod(1 + ww['r'].to_numpy()) ** (365.25 / days) - 1) * 100, float(np.prod(1 + ww['r50'].to_numpy()) ** (365.25 / days) - 1) * 100
S2R = json.loads((S2 / 'results.json').read_text(encoding='utf-8'))['RESULTS']['paper']
ST = {}
for (tag, k, c), w in W.items():
    ref = W[('M1', k, c)] if k != 'C0' else W[('M0', 'C0', c)]
    s = O.stats_pair(w, ref)
    nav = NAV[(tag, k, c)]; s['ALL']['mdd_daily'] = float((nav / nav.cummax() - 1).min() * 100)
    for p in O.PERS:
        ca = cal_ann(w, p)
        if ca: s[p]['ann_cal'], s[p]['ann_cal_0050'] = ca; s[p]['gex_cal'] = ca[0] - ca[1]
    if k != 'C0':
        s['vs_M0'] = {p: dict(diff_pp=v['diff_vs_ref_pp'], t=v['t_diff_vs_ref'], corr=v['corr_ref'], ann_diff=v['ann_diff_vs_ref']) for p, v in O.stats_pair(w, W[('M0', k, c)]).items() if p in O.PERS}
        x = S2R.get(f'{tag}|{k}|{c}', {}); s['breakeven_extra_oneway_pct_from_topic'] = x.get('breakeven_extra_oneway_pct'); s['turnover_oneway_month'] = x.get('turnover_oneway_month')
    s['vs_C0'] = {p: dict(diff_pp=v['diff_vs_ref_pp'], t=v['t_diff_vs_ref'], corr=v['corr_ref']) for p, v in O.stats_pair(w, W[('M0', 'C0', c)]).items() if p in O.PERS}
    ST[f'{tag}|{k}|{c}'] = s
# 名單差（M1→M2、M0→M1）
def ldiff(a, b):
    A, Bq = PK[a], PK[b]
    return [dict(signal=str(d.date()), out=sorted(set(A[d]) - set(Bq.get(d, []))), into=sorted(set(Bq.get(d, [])) - set(A[d]))) for d in A if set(A[d]) != set(Bq.get(d, []))]
LD = {f'{k}|M1->M2': ldiff(('M1', k), ('M2', k)) for k in FILLS}; LD.update({f'{k}|M0->M1': ldiff(('M0', k), ('M1', k)) for k in FILLS})
# 逐槽歸因：M1→M2 換股的窗報酬貢獻（一般成本，等權 1/10 近似）
# J4 混合
BL = {}
for k in FILLS:
    for c in ('normal', 'stress'):
        M = W[('M1', k, c)]; L = W[('M2', k, c)]
        J = M.join(L[['r']], rsuffix='_l', how='inner')
        for wt in (0.8, 0.5):
            Wb = J[['r50', 'yr']].copy(); Wb['r'] = wt * J['r'] + (1 - wt) * J['r_l']
            s = O.stats_pair(Wb[['r', 'r50', 'yr']], M)
            for p in O.PERS:
                ca = cal_ann(Wb, p)
                if ca: s[p]['ann_cal'] = ca[0]
            BL[f'{k}|{c}|leg=M2|w={wt}'] = s
smp = {}
for tag in ('M0', 'M1', 'M2'):
    for k in FILLS + ['C0']:
        key = f'{tag}|{k}|normal'
        if key not in ST: continue
        a = ST[key]['ALL']; se = abs(a['gex'] / a['t_vs_0050'])
        smp[f'{tag}|{k}'] = dict(gex=a['gex'], lo=a['gex'] - 1.96 * se, hi=a['gex'] + 1.96 * se)
# 中間檔（壓縮）存 data/
pd.DataFrame({'|'.join(k): v for k, v in NAV.items()}).astype('float32').to_csv(HERE / 'data/paper_daily_navs.csv.gz')
Z = {}
for k, w in W.items(): Z['|'.join(k) + '|r'] = w['r'].to_numpy(float); Z['|'.join(k) + '|r50'] = w['r50'].to_numpy(float)
Z['per'] = np.array([str(x) for x in W[('M0', 'C0', 'normal')].index]); np.savez_compressed(HERE / 'data/paper_windows.npz', **Z)
res = dict(utc=pd.Timestamp.now('UTC').isoformat(), runtime_s=round(time.time() - T0, 1), lists_sha={f'{a}|{b}': v for (a, b), v in LSH.items()},
           reproduce_topic_windows_maxabs=max(rep_win.values()), list_diffs=LD, n_list_diffs={k: len(v) for k, v in LD.items()},
           stats=ST, blends=BL, sampling_interval=smp)
R = json.loads(RJ.read_text(encoding='utf-8')); assert 'RESULTS_PAPER' not in R; R['RESULTS_PAPER'] = res
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
for key, s in ST.items():
    a = s['ALL']; print(key, 'ann %.3f cal %.3f t %.3f gex %.3f d %.4f(t %s) corr %s seg %.2f/%.2f/%.2f mdd %.1f' % (a['ann'], a.get('ann_cal', float('nan')), a['t_vs_0050'], a['gex'], a['diff_vs_ref_pp'], a['t_diff_vs_ref'], a['corr_ref'], s['SEG1']['gex'], s['SEG2']['gex'], s['SEG3']['gex'], a['mdd_daily']))
print('diffs', res['n_list_diffs'], 'rep', res['reproduce_topic_windows_maxabs'], res['runtime_s'])
