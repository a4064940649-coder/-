# -*- coding: utf-8 -*-
"""A：只依規格文字（a_register.json 凍結版）自建 2006～2013 C0 名單與 ER2 U_opt／U_p90／U_carry 放回名單。
不 import 任何研究員／複製員程式；只讀面板、tw_revenue、stock_list 與 lists_C0（基底閘門）。
python3 -I -B a_build.py VERSION [--targets]   （--targets 只能在登記表與版本凍結後使用）"""
import sys, json, hashlib, os, math
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); AG = os.path.dirname(HERE)
PAN = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/'
REG = json.load(open(os.path.join(HERE, 'a_register.json')))
assert REG['sha256'] == '0e6da71423d89a64ba3afbe6c42494fbe80f45757c079a6833fa05120e4d109d'
VER = sys.argv[1] if len(sys.argv) > 1 else 'v1'
OPT = dict(v1=dict(minp60=60, ref='close', held=True, need3=True),
           v2=dict(minp60=60, ref='close', held=False, need3=True),
           v3=dict(minp60=60, ref='prevadj', held=True, need3=True),
           v4=dict(minp60=60, ref='close', held=True, need3=True, allnames=False),
           v5=dict(minp60=60, ref='close', held=True, need3=True, vrank_after_filters=True),
           v6=dict(minp60=60, ref='close', held=True, need3=True, sigday_rule='lists'))[VER]
adj = pd.read_parquet(PAN + 'adj.parquet'); cl = pd.read_parquet(PAN + 'close_raw.parquet')
op = pd.read_parquet(PAN + 'open.parquet'); val = pd.read_parquet(PAN + 'value.parquet')
cols = list(adj.columns)
good = cl.notna().sum(1) >= 300
G = good[good].index
adjG, clG, valG = adj.loc[G], cl.loc[G], val.loc[G]
V60 = valG.rolling(60, min_periods=OPT['minp60']).mean()
MX = adjG.rolling(252, min_periods=252).max()
S52 = adjG / MX
# 營收
rv = pd.read_csv('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv', dtype={'code': str})
rv['m'] = pd.to_datetime(rv['month']).dt.to_period('M')
RV = rv.pivot_table(index='m', columns='code', values='revenue_k_twd', aggfunc='last')
RV = RV.reindex(pd.period_range(RV.index.min(), RV.index.max(), freq='M'))
base = RV.shift(12)
YOY = (RV / base - 1).where(base > 0)
# 名稱、金融
sl = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype={'code': str})
names = {}
for c, n in zip(sl.code, sl.name): names.setdefault(c, set()).add(str(n))
if OPT.get('allnames', True):
    for c, n in zip(rv.code, rv.name): names.setdefault(c, set()).add(str(n))
fin_ind = set(sl[sl.industry.astype(str).str.contains('金融')].code)
def is_fin(c): return c in fin_ind or c.startswith('28') or c in ('5876', '5880')
def is_foreign(c):
    if c.startswith('91'): return True
    return any(('KY' in n.upper()) or ('DR' in n.upper()) or n.startswith('F-') for n in names.get(c, ()))
FIN = np.array([is_fin(c) for c in cols]); FOR = np.array([is_foreign(c) for c in cols])
def tick(p):
    return 0.01 if p < 10 else 0.05 if p < 50 else 0.1 if p < 100 else 0.5 if p < 500 else 1.0 if p < 1000 else 5.0
def limup(ref, L):
    up = ref * (1 + L); t = tick(up); return math.floor(up / t + 1e-9) * t
# 訊號日
L0 = json.load(open(AG + '/o_g8_protocol_fin_erratum/ERRATUM.json'))['lists_C0']
if OPT.get('sigday_rule') == 'lists':
    SIG = [pd.Timestamp(d) for d in sorted(L0)]
else:
    SIG = []
    for per in pd.period_range('2006-04', '2013-12', freq='M'):
        cand = G[(G >= pd.Timestamp(per.start_time) + pd.Timedelta(days=10)) & (G <= per.end_time)]
        if len(cand): SIG.append(cand[0])
TD = cl.index  # 所有交易日（成交日＝下一個面板日）
def nxt(d): return TD[TD.get_loc(d) + 1]
def build(kind=None, A_fill=None):
    """kind None＝C0；'opt'/'p90'/'carry'＝ER2 放回。回傳 {d: [codes]}，{d: set(A)}"""
    out, Aset, held = {}, {}, set()
    for d in SIG:
        e = nxt(d)
        v = V60.loc[d]; c = clG.loc[d]
        ok = (v >= 1e7) & (c >= 10)
        top = v[ok].sort_values(ascending=False, kind='mergesort').index[:300]
        m = d.to_period('M')
        y3 = YOY.loc[[m - 1, m - 2, m - 3]].mean(skipna=False).reindex(cols) if m - 1 in YOY.index else pd.Series(np.nan, index=cols)
        pool = [x for x in top if np.isfinite(S52.loc[d, x]) and np.isfinite(op.loc[e, x]) and op.loc[e, x] > 0
                and not FIN[cols.index(x)] and not FOR[cols.index(x)]]
        el = [x for x in pool if np.isfinite(y3[x])]
        A = []
        if kind is not None:
            A = [x for x in pool if not np.isfinite(y3[x]) and A_fill(x, d) is not None]
            if kind == 'carry':
                y3 = y3.copy()
                for x in A: y3[x] = A_fill(x, d)
                el2 = el + A; A_rank = []
            else:
                el2 = el + A
        else:
            el2 = el
        df = pd.DataFrame(index=el2)
        df['s52'] = [S52.loc[d, x] for x in el2]
        df['p52'] = df['s52'].rank(pct=True, method='average')
        if kind in ('opt', 'p90'):
            non = [x for x in el2 if x not in A]
            r = pd.Series([y3[x] for x in non], index=non).rank(method='average') / len(el2)
            df['prev'] = [1.0 if (x in A and kind == 'opt') else 0.9 if x in A else r[x] for x in el2]
            p90 = np.nanpercentile([y3[x] for x in non], 90) if non else np.nan
            df['y3'] = [np.inf if (x in A and kind == 'opt') else p90 if x in A else y3[x] for x in el2]
        else:
            df['y3'] = [y3[x] for x in el2]
            df['prev'] = df['y3'].rank(pct=True, method='average')
        df['score'] = 0.5 * df['p52'] + 0.5 * df['prev']
        df['ci'] = [cols.index(x) for x in el2]
        df = df.sort_values(['score', 'y3', 'ci'], ascending=[False, False, True], kind='mergesort')
        pick = []
        for x in df.index:
            if len(pick) == 10: break
            ref = clG.loc[d, x]
            L = 0.07 if e < pd.Timestamp('2015-06-01') else 0.10
            hit = op.loc[e, x] >= limup(ref, L) - 1e-9
            if hit and not (OPT['held'] and x in held): continue
            pick.append(x)
        out[str(d.date())] = pick; Aset[str(d.date())] = set(A); held = set(pick)
    return out, Aset
def jacc(a, b):
    a, b = set(a), set(b); return len(a & b) / len(a | b) if (a | b) else 1.0
if __name__ == '__main__':
    C0, _ = build(None)
    common = sorted(set(C0) & set(L0))
    js = [jacc(C0[d], L0[d]) for d in common]
    diff = [(d, sorted(set(C0[d]) - set(L0[d])), sorted(set(L0[d]) - set(C0[d]))) for d in common if set(C0[d]) != set(L0[d])]
    res = dict(version=VER, opt=OPT, n_sig=len(SIG), n_common=len(common), sig_only_mine=sorted(set(C0) - set(L0)), sig_only_c0=sorted(set(L0) - set(C0)),
               jaccard_mean=float(np.mean(js)), slots_diff=int(sum(len(x[1]) for x in diff)), diff_days=diff[:40])
    print(json.dumps({k: res[k] for k in ['version', 'n_sig', 'n_common', 'sig_only_mine', 'sig_only_c0', 'jaccard_mean', 'slots_diff']}, ensure_ascii=False))
    for x in diff[:8]: print(x)
    p = os.path.join(HERE, 'a_iterations.json'); IT = json.load(open(p)) if os.path.exists(p) else []
    IT.append(res); json.dump(IT, open(p, 'w'), ensure_ascii=False, indent=1)
    json.dump(C0, open(os.path.join(HERE, f'out/a_C0_{VER}.json'), 'w'))
