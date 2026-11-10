# 反方審查（成本與可成交角度）：o19 review
# 目的：量化名單層「開盤漲停跳過遞補」用到成交日開盤價，對 P4 的 S0（訊號日收盤）與 P5 的影響。
# 只讀研究員模組（xdata、xlists），cache 改寫到 scratchpad，不改任何研究員檔案。
# 用法：python3 -I -B review_exec_o19.py
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import xdata
SCR = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rev_o19/'
os.makedirs(SCR, exist_ok=True)
xdata.SCR = SCR; xdata.CACHE = SCR + 'xdata_v1.npz'
import xlists

L = xlists.Lists(); X = L.X; dates = L.dates
G, ADJ, O, PC, TR, C = X['G'], X['ADJ'], X['O'], X['PC'], X['TR'], X['C']
A = np.cumprod(G, axis=0)
with np.errstate(invalid='ignore', divide='ignore'):
    Aop = np.vstack([A[:1], A[:-1]]) * ADJ * O / PC
Aop = np.where(np.isfinite(Aop) & TR, Aop, A)

def ranked_and_skips(si, held):
    """複製 pick() 的排序；回傳 (skip版 picks, 無跳過版 picks, 被跳過者)"""
    UPs = X['UP'][si + 1].copy()
    X['UP'][si + 1] = np.inf  # 暫時關閉跳過以取得純排序
    full = L.pick(si, None, set(), n=40)
    X['UP'][si + 1] = UPs
    if full is None: return None
    ei = si + 1; skip = []; picks = []
    for j in full:
        if len(picks) >= 10: break
        if (j not in held) and O[ei, j] >= UPs[j] - 1e-6: skip.append(j); continue
        picks.append(j)
    return picks, full[:10], skip

days = L.official_days()
rows = []; held = set(); held_ns = set()
recs = []
for rm, si in days:
    r = ranked_and_skips(si, held)
    if r is None: continue
    pk, pk_ns, sk = r
    recs.append((si, pk, pk_ns, sk)); held = set(pk)
sis = [r[0] for r in recs]
def win_ret(kind, which):
    out = []
    for k in range(len(recs) - 1):
        si, pk, pk_ns, sk = recs[k]; sn = recs[k + 1][0]
        names = pk if which == 'skip' else pk_ns
        if kind == 'close':
            rr = [A[sn, j] / A[si, j] for j in names]
        else:
            rr = [Aop[sn + 1, j] / Aop[si + 1, j] for j in names]
        out.append(np.mean(rr))
    return np.array(out)
E = X['E']
def ann(rets, kind, start=None):
    idx0 = [recs[k][0] for k in range(len(recs) - 1)]; idx1 = [recs[k + 1][0] for k in range(len(recs) - 1)]
    m = np.ones(len(rets), bool)
    if start is not None: m = np.array([dates[i] >= pd.Timestamp(start) for i in idx0])
    g = np.prod(rets[m]); i0 = np.array(idx0)[m][0]; i1 = np.array(idx1)[m][-1]
    yrs = (dates[i1] - dates[i0]).days / 365.25
    e = E[i1] / E[i0]
    return 100 * (g ** (1 / yrs) - 1), 100 * (e ** (1 / yrs) - 1)
out = {}
for nm, kind, which in [('S0_close_skip(研究員S0口徑)', 'close', 'skip'), ('S0_close_noskip(真收盤成交)', 'close', 'noskip'),
                        ('S1_open_skip(研究員S1口徑)', 'open', 'skip'), ('S1_open_noskip(買不到不計)', 'open', 'noskip')]:
    r = win_ret(kind, which)
    out[nm] = dict(full=ann(r, kind), y19=ann(r, kind, '2019-01-01'))
# 被跳過的開盤漲停股：隔夜跳空與後續報酬
gap = []; post = []; nsk = 0; ndays_sk = 0
for k in range(len(recs) - 1):
    si, pk, pk_ns, sk = recs[k]; sn = recs[k + 1][0]
    sk_top = [j for j in sk if j in pk_ns]
    nsk += len(sk_top); ndays_sk += bool(sk_top)
    for j in sk_top:
        gap.append(Aop[si + 1, j] / A[si, j] - 1); post.append(Aop[sn + 1, j] / Aop[si + 1, j] - 1)
out['skipped_in_top10'] = dict(n=nsk, n_days_with_skip=ndays_sk, n_days=len(recs) - 1,
                              gap_close_to_open_mean=float(np.mean(gap)) if gap else None,
                              open_to_next_open_mean=float(np.mean(post)) if post else None)
for k in ['S0_close_skip(研究員S0口徑)', 'S0_close_noskip(真收盤成交)', 'S1_open_skip(研究員S1口徑)']:
    pass
out['timing_gap_y19_researcher'] = out['S0_close_skip(研究員S0口徑)']['y19'][0] - out['S1_open_skip(研究員S1口徑)']['y19'][0]
out['timing_gap_y19_true_close'] = out['S0_close_noskip(真收盤成交)']['y19'][0] - out['S1_open_skip(研究員S1口徑)']['y19'][0]
out['timing_gap_full_researcher'] = out['S0_close_skip(研究員S0口徑)']['full'][0] - out['S1_open_skip(研究員S1口徑)']['full'][0]
out['timing_gap_full_true_close'] = out['S0_close_noskip(真收盤成交)']['full'][0] - out['S1_open_skip(研究員S1口徑)']['full'][0]
# 研究員自己的 O30（最小帳戶）與 N100 的 2019 起差：檢驗「衝擊 10pp 是 GROW 造成」
R = json.load(open(os.path.join(HERE, 'results.json')))
cs = R['RESULTS_FIX5']['cells']
out['own_cells_y19'] = {c: round(100 * cs[c]['sum']['exc_y19_med'], 2) for c in cs if c.startswith('E|REAL')}
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
json.dump(out, open(SCR + 'review_exec_o19_out.json', 'w'), ensure_ascii=False, indent=1, default=float)
