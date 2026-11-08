"""R1 獨立複製：C0 D=S／D>S 槽的營收可得分類（V6）。"""
import re, pickle, json, sys
import numpy as np, pandas as pd
RAW = '/home/user/-/research/tw_stock_selection/agents/o_rev_avail_ds_pit_count/raw/'
UPFIRST_AS_NONE = '--upavail' not in sys.argv
UNIT_FROM_SNIP = '--snipunit' in sys.argv
o = pickle.load(open('c0.pkl', 'rb'))
form = o[False].copy(); form['formal'] = True
app = o[True]; app = app[app.sig == '2026-10-12'].copy(); app['formal'] = False
C = pd.concat([form, app], ignore_index=True)
C['rm'] = (pd.PeriodIndex(C.sig, freq='M') - 1).astype(str)
LEGAL_DGTS = {'2020-02-11'}
C['legal'] = np.where(C.sig.isin(LEGAL_DGTS), 'D>S', C.case)

# ---- 官方營收（千元）
tw = pd.read_csv('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv', dtype={'code': str})
tw['ym'] = tw.month.str[:7]
TW = tw.groupby(['code', 'ym']).revenue_k_twd.apply(list).to_dict()
r5 = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/rev_5y.csv.gz', dtype={'code': str})
R5 = r5.groupby(['code', 'ym']).revenue.apply(list).to_dict()

# ---- 新聞：末位 ±0.5 閘門
MUL = {'億': 100000.0, '萬': 10.0, '千': 1.0}
NUMRE = re.compile(r'([0-9][0-9,]*(?:\.[0-9]+)?)\s*(億|萬|千)')
def unit_of(rev_k, snip):
    """末位：預設用 rev_k 去掉尾端 0 後的末位（有效位數）；--snipunit 改由內文數字的小數位推得。"""
    best = None
    if UNIT_FROM_SNIP:
        for m in NUMRE.finditer(str(snip)):
            s = m.group(1).replace(',', ''); mul = MUL[m.group(2)]
            try: v = float(s) * mul
            except ValueError: continue
            if abs(v - rev_k) < 1e-6 * max(1, rev_k):
                dec = len(s.split('.')[1]) if '.' in s else 0
                u = mul * 10 ** (-dec)
                best = u if best is None else min(best, u)
    if best is None:
        x = int(round(rev_k)); u = 1
        while x and x % 10 == 0: x //= 10; u *= 10
        best = float(u)
    return best
nw = pd.read_csv(RAW + 'ts_news_mentions.csv.gz', dtype={'code': str})
nw = nw[nw.rev_k.notna()].copy()
nw['unit'] = [unit_of(r, s) for r, s in zip(nw.rev_k, nw.snip)]
def gate(row, mode):
    offs = TW.get((row.code, row.ym), []) + R5.get((row.code, row.ym), [])
    offs = [x for x in offs if np.isfinite(x)]
    if mode == 'strict':
        return any(abs(row.rev_k - x) <= 0.5 * row.unit + 1e-9 for x in offs)
    return any(abs(row.rev_k - x) <= 0.15 * abs(x) for x in offs)
nw['ok6'] = [gate(r, 'strict') for r in nw.itertuples()]
nw['ok5'] = [gate(r, 'loose') for r in nw.itertuples()]
nw['ts'] = pd.to_datetime(nw.ts); nw['has_time'] = nw.has_time.astype(str) == 'True'
NW = {k: g for k, g in nw.groupby(['code', 'ym'])}

ma = pd.read_csv(RAW + 'ts_mops_ann_date.csv', dtype=str)
MA = {(r.code, r.ym): pd.Timestamp(r.ann_date) for r in ma.itertuples()}
fs = pd.read_csv(RAW + 'ts_tasi_upstream_firstseen.csv.gz', dtype=str)
fs['T'] = pd.to_datetime(fs['T'], utc=True).dt.tz_convert('Asia/Taipei').dt.tz_localize(None)
fs['Tp'] = pd.to_datetime(fs['T_prev'], utc=True).dt.tz_convert('Asia/Taipei').dt.tz_localize(None)
FS = {k: g for k, g in fs.groupby(['code', 'ym'])}

def classify(code, rm, S, newsmode='ok6'):
    S = pd.Timestamp(S); cut = S + pd.Timedelta(hours=13, minutes=30); end24 = S + pd.Timedelta(days=1)
    av = un = ud = False; src = []; seen24 = None
    a = MA.get((code, rm))
    if a is not None:
        if a < S: av = True; src.append('mops<')
        elif a == S: ud = True; src.append('mops=')
        else: un = True; src.append('mops>')
    g = FS.get((code, rm))
    if g is not None:
        for r in g.itertuples():
            upfirst = r.source == 'upstream' and r.reason in ('file_add_no_prev', 'batch_backfill')
            if upfirst and UPFIRST_AS_NONE: continue
            if r.T <= cut: av = True; src.append(r.source + '<=cut')
            else:
                if pd.isna(r.Tp) or r.Tp.normalize() <= S: ud = True; src.append(r.source + '>cut')
                else: un = True; src.append(r.source + 'absent_after')
            if r.T < end24: seen24 = True
    g = NW.get((code, rm))
    if g is not None:
        g = g[g[newsmode]]
        for r in g.itertuples():
            d = r.ts.normalize()
            if r.has_time:
                if r.ts <= cut: av = True; src.append('news_t')
            else:
                if d < S: av = True; src.append('news<')
                elif d == S: ud = True; src.append('news=')
    if av and un: st = 'UNDET'
    elif av: st = 'AVAIL'
    elif un: st = 'UNAVAIL'
    elif ud: st = 'UNDET'
    else: st = 'NONE'
    return st, src, seen24

if __name__ == '__main__':
    D = C[C.case != 'D<S'].copy()
    res = [classify(r.code, r.rm, r.sig) for r in D.itertuples()]
    D['state'] = [x[0] for x in res]; D['src'] = [';'.join(x[1]) for x in res]; D['seen24'] = [x[2] for x in res]
    D['state_V5'] = [classify(r.code, r.rm, r.sig, 'ok5')[0] for r in D.itertuples()]
    D.to_csv('rep_slot_avail.csv', index=False)
    print('news gate pass strict', nw.ok6.sum(), 'loose', nw.ok5.sum(), 'of', len(nw))
