#!/usr/bin/env python3
"""R72：獨立重現 S72「SURGE 安慰劑檢定」＋額外稽核（公平版安慰劑）。
只依 S72 的 SPEC.json／PREREG.json／FINDINGS.md 與共用引擎；沒有讀 S72 的 run.py／posthoc.py。
用法：python3 -I -B replicate.py [--quick] [--n 300]
訊號產生器 build() 是本檔自己寫的（照 SPEC 的文字與 s10lib.make_signal_s10 的進出場邏輯），亂數種子與 S72 不同。
暫存 scratchpad/r72/（結束刪除）。不改任何既有檔案。"""
import sys, json, time, resource, gc, csv
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/r72')
CACHE = TMP / 'cache'; CACHE.mkdir(parents=True, exist_ok=True)
QUICK = '--quick' in sys.argv
NDRAW = int(sys.argv[sys.argv.index('--n') + 1]) if '--n' in sys.argv else 300
SEED = 7272031                                             # 本稽核自選（S72 用 [72, 1, i]）
T0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402
for mod, attr in ((s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                  (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(mod, attr, CACHE)
assert cslib.EXRIGHT_FIX is False and cslib.DEADLINE_FIX is False and cslib.INS15_MASK is False
cslib.EXRIGHT_FIX = True
P = s20lib.load()
assert P.exright_fix
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
F = s17lib.rev_family(P, defs=['STD'])                     # 兩個期限開關關閉下建 SURGE
s17lib.e50_fix_suspension(P)
S0 = F['S']['STD']; MONTHS = list(S0.index)
Rv = F['Rv']
YOY = (Rv / Rv.shift(12) - 1).replace([np.inf, -np.inf], np.nan)
YOY[YOY.index.year == 2013] = np.nan
YOY = YOY.reindex(S0.index).to_numpy(float)
SV = S0.to_numpy(float)
del F, Rv
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
cslib.DEADLINE_FIX = True; cslib.INS15_MASK = True
AV = cslib.rev_avail_days(P, MONTHS, True)
RT = dict(S=S0, av=AV)
print('載入', f'{time.time()-T0:.0f}s', rss(), 'MB', (P.T, P.N, str(P.IDX[-1].date())), flush=True)

lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
DE = P.IDX[-1]; D0 = P.IDX[P.t0]; D16 = lastb('2016-01-01'); D19 = lastb('2019-01-01'); D24 = lastb('2024-01-01')
E50 = pd.Series(P.E50, index=P.IDX)
SEGS = dict(since2019=(D19, DE), full=(D0, DE), seg16_18=(D16, D19), seg19_23=(D19, D24), seg24=(D24, DE))
E50C = {k: stlib.cagr(E50, a, b) for k, (a, b) in SEGS.items()}
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)


def month_rows(shift):
    """每個有訊號的月份：(月序 mi, 月份 m, 訊號列 k, INS15 遮罩後的 SURGE s)。是否有訊號照舊用 SURGE（未遮罩、位移前的列）判斷。"""
    out = []
    for mi, m in enumerate(MONTHS):
        k0 = AV[m]
        if k0 >= P.T or k0 < 61:
            continue
        s = SV[mi]
        if int((U[k0] & np.isfinite(s)).sum()) < 50:
            continue
        k = k0 + shift
        if k >= P.T:
            continue
        s = cslib.ins15_mask_row(P, m, k, s)
        out.append((mi, m, k, s))
    return out


MR = {sh: month_rows(sh) for sh in range(4)}


def build(shift, qfun, depth=10, keep=20, fair=None, mom_lo=0.10, nsel=10):
    """qfun(mi, k, s) → 排序分數 q（長度 N；缺值＝不合格）。
    合格＝股池內 且 SURGE 有值 且 q 有值；L_MA＝合格 且 還原收盤 > 60 日均線，依 q 由大到小（stable）。
    fair＝None：進場＝L_MA 前 depth 名中 RET20 > 10%（不遞補，最多 nsel）；出場＝不在 L_MA 前 keep 名。
    fair＝{月份字串: 檔數 n}（公平版）：沿 q 的名次往下走，直到湊滿 n 檔 RET20 > 10% 的股票（＝從「L_MA 且 RET20>10%」均勻隨機抽 n 檔）；
          設湊滿時走到第 d 名（n＝0 時 d＝10），出場＝不在 L_MA 前 2d 名（比例同現行的 10：20）。"""
    sig = np.zeros((P.T, P.N), bool); rank = np.full((P.T, P.N), np.nan); X = np.zeros((P.T, P.N), bool)
    info = []
    for mi, m, k, s in MR[shift]:
        q = qfun(mi, k, s)
        ok = U[k] & np.isfinite(s) & np.isfinite(q) & P.F_MA60[k]
        idx = np.flatnonzero(ok)
        o = idx[np.argsort(-q[idx], kind='stable')]
        r20 = P.RET20[k]
        if fair is None:
            pool = o[:depth]
            tgt = pool[r20[pool] > mom_lo][:nsel]; kd = keep
        else:
            n = fair.get(str(m), 0)
            if n > 0:
                pas = np.flatnonzero(r20[o] > mom_lo)
                tgt = o[pas[:n]]
                d = int(pas[min(n, len(pas)) - 1]) + 1 if len(pas) else 10
            else:
                tgt = o[:0]; d = 10
            kd = 2 * d
        sig[k, tgt] = True; rank[k, tgt] = q[tgt]
        X[k, :] = True; X[k, o[:kd]] = False
        info.append(dict(month=str(m), k=int(k), n_ma=int(len(o)), n_tgt=int(len(tgt)), n_pass=int((r20[o] > mom_lo).sum()), kd=int(kd),
                         targets=[str(P.COLS[j]) for j in tgt]))
    sigrows = np.zeros(P.T, bool); sigrows[[x['k'] for x in info]] = True
    return dict(sig=sig, rank=rank, X=X, SD=None, max_hold=None, info=info, XV={'E0': X}, RQ={}, sigrows=sigrows)


def sim(Sg, cap=1e6):
    return s10lib.simulate_s10(P, Sg, U, sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'], capital=cap)


def metrics(Sg, nav, lg, st):
    x = lg[lg['why'] != 'open_end'] if len(lg) else lg
    o = {k: stlib.cagr(nav, a, b) for k, (a, b) in SEGS.items()}
    o.update(maxdd_full=stlib.maxdd(nav, D0, DE), n_closed=int(len(x)), hold_days=float(x['hold_days'].mean()) if len(x) else float('nan'),
             avg_npos=float(st['avg_npos']), names_pm=float(np.mean([i['n_tgt'] for i in Sg['info']])),
             trade_ret=float(x['ret'].mean()) if len(x) else float('nan'), idle=float(st['avg_idle_frac']))
    return o


def run_std(qfun, **kw):
    Sg = build(0, qfun, **kw)
    nav, lg, st = sim(Sg)
    return metrics(Sg, nav, lg, st), Sg


def run_paths(qfun_of_shift, **kw):
    out = {}
    for sh in range(4):
        Sg = build(sh, qfun_of_shift(sh), **kw)
        for cap in CAPS:
            nav, lg, st = sim(Sg, cap)
            out[f'{int(cap/1e4)}萬|+{sh}'] = stlib.cagr(nav, D19, DE)
        del Sg
    return out


def dist(v, cur=None):
    v = np.asarray(v, float)
    o = dict(n=int(len(v)), median=float(np.median(v)), mean=float(v.mean()), p5=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)),
             min=float(v.min()), max=float(v.max()))
    if cur is not None:
        o.update(current=float(cur), pctile=float((v < cur).mean() * 100), p_one=float((1 + (v >= cur).sum()) / (len(v) + 1)),
                 gap_pp=float((cur - np.median(v)) * 100))
    return o


res = dict(seed=SEED, n_draws=NDRAW, e50=E50C, panel=dict(T=int(P.T), N=int(P.N), end=str(DE.date())))
# ------------------------------------------------------------ 1. 現行（兩種做法：引擎原函式、本檔 build）
q_surge = lambda mi, k, s: s
S_ref = s10lib.make_signal_s10(P, U, RT, shift=0)
nav, lg, st = sim(S_ref)
ref = metrics(S_ref, nav, lg, st)
cur, S_cur = run_std(q_surge)
same = bool(np.array_equal(S_ref['sig'], S_cur['sig']) and np.array_equal(S_ref['X'], S_cur['X']) and np.array_equal(S_ref['rank'], S_cur['rank'], equal_nan=True))
NCUR = {x['month']: x['n_tgt'] for x in S_cur['info']}
res['current'] = dict(std=cur, via_make_signal_s10=ref, build_equals_engine_signal=same,
                      n_ma_mean=float(np.mean([x['n_ma'] for x in S_cur['info']])), n_pass_mean=float(np.mean([x['n_pass'] for x in S_cur['info']])),
                      n_signal_months=len(S_cur['info']))
print('現行', repr(cur['since2019']), cur['n_closed'], 'names', round(cur['names_pm'], 3), 'same', same,
      'n_ma', res['current']['n_ma_mean'], 'n_pass', res['current']['n_pass_mean'], f'{time.time()-T0:.0f}s', flush=True)
assert same and cur['since2019'] == ref['since2019'] == 0.4875617247721724 and cur['n_closed'] == 381, '現行重現失敗'
del S_ref, nav, lg, st
if QUICK:
    t = time.time(); run_std(lambda mi, k, s: np.random.default_rng(1).random(P.N)); print('單次', time.time() - t, 's'); sys.exit()
cp = run_paths(lambda sh: q_surge)
res['current']['paths20'] = cp; res['current']['paths20_median'] = float(np.median(list(cp.values())))
print('現行 20 條中位數', res['current']['paths20_median'], flush=True)

# ------------------------------------------------------------ 2. 確定性安慰劑
G = P.G
def q_ret60(mi, k, s):
    with np.errstate(invalid='ignore', divide='ignore'):
        return G[k] / G[k - 60] - 1
DET = dict(P3_RET20=lambda mi, k, s: P.RET20[k], P3_RET60=q_ret60, P4_REVERSE=lambda mi, k, s: -s, P5_YOY=lambda mi, k, s: YOY[mi])
res['det'] = {}
for name, qf in DET.items():
    m, _ = run_std(qf)
    pp = run_paths(lambda sh: qf)
    res['det'][name] = dict(std=m, paths20=pp, paths20_median=float(np.median(list(pp.values()))))
    print(name, round(m['since2019'] * 100, 2), round(res['det'][name]['paths20_median'] * 100, 2), f'{time.time()-T0:.0f}s', flush=True)


# ------------------------------------------------------------ 3. 隨機安慰劑
def q_rand(tag, i):
    rng = np.random.default_rng([SEED, tag, i])
    Z = rng.random((len(MONTHS), P.N))
    return lambda mi, k, s: Z[mi]


def q_rand3(tag, i):
    """每檔股票的亂數持續 3 個月（相位各股錯開）。"""
    rng = np.random.default_rng([SEED, tag, i])
    Z = rng.random((len(MONTHS), P.N)); ph = rng.integers(0, 3, P.N)
    for mi in range(1, len(MONTHS)):
        hold = (mi + ph) % 3 != 0
        Z[mi, hold] = Z[mi - 1, hold]
    return lambda mi, k, s: Z[mi]


def top50(qf):
    """只在「L_MA 內 SURGE 前 50 名」裡給分數（其餘缺值＝不合格）。"""
    def f(mi, k, s):
        q = qf(mi, k, s)
        ok = U[k] & np.isfinite(s) & P.F_MA60[k]
        idx = np.flatnonzero(ok)
        top = idx[np.argsort(-s[idx], kind='stable')][:50]
        out = np.full(P.N, np.nan); out[top] = q[top]
        return out
    return f


FAM = dict(
    P1=dict(tag=1, q=q_rand, kw={}, desc='隨機排名（S72 的 P1）：前 10 進場、前 20 續抱'),
    P2=dict(tag=2, q=q_rand3, kw={}, desc='隨機分數延續 3 個月（S72 的 P2）'),
    FAIR=dict(tag=3, q=q_rand, kw=dict(fair=NCUR), desc='公平版：每月從「L_MA 且 RET20>10%」隨機抽現行當月檔數；續抱＝隨機名次前 2d'),
    FAIR3=dict(tag=4, q=q_rand3, kw=dict(fair=NCUR), desc='公平版＋分數延續 3 個月（持有天數向現行靠攏）'),
    B50=dict(tag=5, q=lambda t, i: top50(q_rand(t, i)), kw={}, desc='只在 SURGE 前 50 名（L_MA 內）隨機：前 10 進場、前 20 續抱'),
    B50FAIR=dict(tag=6, q=lambda t, i: top50(q_rand(t, i)), kw=dict(fair=NCUR), desc='SURGE 前 50 名內隨機＋檔數逐月對齊現行'),
)
KEYS = ('since2019', 'full', 'seg16_18', 'seg19_23', 'seg24', 'maxdd_full', 'n_closed', 'hold_days', 'avg_npos', 'names_pm', 'trade_ret', 'idle')
res['rand'] = {}
for name, f in FAM.items():
    rows = []
    for i in range(NDRAW):
        m, _ = run_std(f['q'](f['tag'], i), **f['kw'])
        rows.append(m)
    R = dict(desc=f['desc'], draws={k: [r[k] for r in rows] for k in KEYS})
    R['stats'] = {k: dist(R['draws'][k], cur[k]) for k in KEYS}
    res['rand'][name] = R
    s = R['stats']['since2019']
    print(name, 'med', round(s['median'] * 100, 2), 'p5', round(s['p5'] * 100, 2), 'p95', round(s['p95'] * 100, 2), 'max', round(s['max'] * 100, 2),
          'pct', s['pctile'], 'p', round(s['p_one'], 4), 'names', round(R['stats']['names_pm']['median'], 2), 'npos', round(R['stats']['avg_npos']['median'], 2),
          'hold', round(R['stats']['hold_days']['median'], 1), 'full', round(R['stats']['full']['median'] * 100, 2), f'{time.time()-T0:.0f}s', rss(), 'MB', flush=True)
    json.dump(res, open(TMP / 'partial.json', 'w'), ensure_ascii=False, default=str)

# 公平版 20 條路徑（前 20 次）：每次取 20 條中位數。檔數對齊用「各位移自己的現行檔數」
NC_SH = {}
for sh in range(4):
    Sg = build(sh, q_surge); NC_SH[sh] = {x['month']: x['n_tgt'] for x in Sg['info']}; del Sg
meds = []; allp = []
for i in range(20):
    out = {}
    for sh in range(4):
        Sg = build(sh, q_rand(13, i), fair=NC_SH[sh])
        for cap in CAPS:
            nav, lg, st = sim(Sg, cap); out[f'{int(cap/1e4)}萬|+{sh}'] = stlib.cagr(nav, D19, DE)
        del Sg
    meds.append(float(np.median(list(out.values())))); allp += list(out.values())
res['rand']['FAIR']['paths20_medians'] = dist(meds, res['current']['paths20_median'])
res['rand']['FAIR']['paths20_all400'] = dist(allp)
print('FAIR 20 條', res['rand']['FAIR']['paths20_medians'], flush=True)


# ------------------------------------------------------------ 4. 加抽：公平版與 SURGE 前 50 名公平版各 1000 次（另一組種子），把 p 的下限壓到 0.001
for name, tag in (('FAIR', 23), ('B50FAIR', 26)):
    f = FAM[name]; rows = []
    for i in range(1000):
        m, _ = run_std(f['q'](tag, i), **f['kw']); rows.append(m)
    res['rand'][name]['extra1000'] = {k: dist([r[k] for r in rows], cur[k]) for k in ('since2019', 'full', 'seg16_18', 'seg19_23', 'seg24')}
    res['rand'][name]['extra1000_top5_since2019'] = sorted(r['since2019'] for r in rows)[-5:]
    print(name, '1000 次', res['rand'][name]['extra1000']['since2019'], flush=True)

# ------------------------------------------------------------ 5. 現行拿掉最賺 5 檔（同 official_nav 的 block5）對各安慰劑分布（安慰劑不拿掉）
nav, lg, st = sim(S_cur)
blk, _ = s2lib.block_from(lg, 5)
nb = s10lib.simulate_s10(P, S_cur, U, sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'], capital=1e6, block=blk)[0]
b5 = stlib.cagr(nb, D19, DE)
res['current']['block5_since2019'] = b5
res['current']['block5_vs'] = {n: dist(res['rand'][n]['draws']['since2019'], b5) for n in FAM}
print('block5', b5, {n: (v['pctile'], round(v['p_one'], 4)) for n, v in res['current']['block5_vs'].items()}, flush=True)

# ------------------------------------------------------------ 6. 對照表
S72 = {'現行 2019 起 標準': 48.76, '現行 已出場筆數': 381, '現行 每月名單檔數': 4.05, '現行 20 條中位數': 46.40,
       'P1 2019 起 中位數': 26.92, 'P1 2019 起 P5': 22.34, 'P1 2019 起 P95': 31.35, 'P1 2019 起 最高': 33.47, 'P1 全期 中位數': 19.17,
       'P1 現行百分位': 100, 'P1 單尾 p': 0.0033, 'P1 每月名單檔數 中位數': 2.64,
       'P3 RET20 標準': 24.74, 'P3 RET60 標準': 33.38, 'P3 RET20 20 條中位數': 33.01, 'P3 RET60 20 條中位數': 39.42,
       'P4 反向 標準': 23.45, 'P5 年增率 標準': 26.88,
       'P1 2016～2018 現行百分位': 98.0, 'P1 2019～2023 現行百分位': 100, 'P1 2024 起 現行百分位': 100}
p1 = res['rand']['P1']['stats']; d = res['det']
R72 = {'現行 2019 起 標準': cur['since2019'] * 100, '現行 已出場筆數': cur['n_closed'], '現行 每月名單檔數': cur['names_pm'],
       '現行 20 條中位數': res['current']['paths20_median'] * 100,
       'P1 2019 起 中位數': p1['since2019']['median'] * 100, 'P1 2019 起 P5': p1['since2019']['p5'] * 100, 'P1 2019 起 P95': p1['since2019']['p95'] * 100,
       'P1 2019 起 最高': p1['since2019']['max'] * 100, 'P1 全期 中位數': p1['full']['median'] * 100,
       'P1 現行百分位': p1['since2019']['pctile'], 'P1 單尾 p': p1['since2019']['p_one'], 'P1 每月名單檔數 中位數': p1['names_pm']['median'],
       'P3 RET20 標準': d['P3_RET20']['std']['since2019'] * 100, 'P3 RET60 標準': d['P3_RET60']['std']['since2019'] * 100,
       'P3 RET20 20 條中位數': d['P3_RET20']['paths20_median'] * 100, 'P3 RET60 20 條中位數': d['P3_RET60']['paths20_median'] * 100,
       'P4 反向 標準': d['P4_REVERSE']['std']['since2019'] * 100, 'P5 年增率 標準': d['P5_YOY']['std']['since2019'] * 100,
       'P1 2016～2018 現行百分位': p1['seg16_18']['pctile'], 'P1 2019～2023 現行百分位': p1['seg19_23']['pctile'], 'P1 2024 起 現行百分位': p1['seg24']['pctile']}
rows = []
for k, v in S72.items():
    kind = '分布' if k.startswith('P1') else '確定性'
    tol = (1.0 if kind == '分布' else 0.3)
    if k in ('P1 現行百分位', 'P1 單尾 p', 'P1 每月名單檔數 中位數') or '百分位' in k:
        tol = {'P1 單尾 p': 0.001, 'P1 每月名單檔數 中位數': 0.1}.get(k, 1.0)
    if k in ('現行 已出場筆數',):
        tol = 0
    diff = R72[k] - v
    rows.append((k, v, round(R72[k], 4), round(diff, 4), tol, '符合' if abs(diff) <= tol + 0.005 else '不符'))
with open(HERE / 'compare.csv', 'w', newline='', encoding='utf-8-sig') as fh:
    w = csv.writer(fh); w.writerow(['項目', 'S72 宣稱', 'R72 重現', '差', '容許', '判定'])
    w.writerows(rows)
    w.writerow([]); w.writerow(['額外稽核（R72 自做，2019 起標準路徑；現行 48.76%）', '中位數%', 'P5%', 'P95%', '最高%', '現行百分位', '單尾 p', '每月名單檔數', '平均持股', '持有天數',
                                '2016～2018 百分位', '2019～2023 百分位', '2024 起 百分位'])
    for n in FAM:
        s = res['rand'][n]['stats']
        w.writerow([n + '｜' + FAM[n]['desc']] + [round(s['since2019'][x] * 100, 2) for x in ('median', 'p5', 'p95', 'max')] +
                   [s['since2019']['pctile'], round(s['since2019']['p_one'], 4), round(s['names_pm']['median'], 2), round(s['avg_npos']['median'], 2),
                    round(s['hold_days']['median'], 1), s['seg16_18']['pctile'], s['seg19_23']['pctile'], s['seg24']['pctile']])
res['compare'] = [dict(item=r[0], s72=r[1], r72=r[2], diff=r[3], tol=r[4], verdict=r[5]) for r in rows]
print(*rows, sep='\n')

res['resources'] = dict(peak_mb=rss(), secs=round(time.time() - T0))
json.dump(res, open(HERE / 'results.json', 'w'), ensure_ascii=False, indent=1, default=str)
print('done', res['resources'])
