"""R2 獨立複製（面板 A）：2013 年後營收存活者缺口 U_opt（ER2 定義延伸）。
用法：python3 -I -B r2.py [--names=rev,sl|rev,sl,lm] [--start=2013-01-01]"""
import sys, json, pickle
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/main_v2')
import numpy as np, pandas as pd
import v2_core as V
K = V.K; IDX, COLS, N = V.IDX, V.COLS, V.N
ARG = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--') and '=' in a)
NAMES = ARG.get('names', 'rev,sl,snap').split(',')
START = pd.Timestamp(ARG.get('start', '2013-01-01'))
END = pd.Timestamp(ARG.get('end', '2026-09-11'))
SWITCH = V.SWITCH

YL, Y5, Y5f = V.load_revenue()
M = V.V2('gapfix', YL, Y5, exfin=True, latest=False)

# ---- 營收水準（長期：sii 優先去重）
tw = pd.read_csv(K.LONG, dtype={'code': str, 'market': str})
tw['p'] = pd.PeriodIndex(pd.to_datetime(tw['month']), freq='M')
tw['pri'] = (tw['market'] != 'sii').astype(int)
tw = tw.sort_values(['p', 'code', 'pri']).drop_duplicates(['p', 'code'])
LV = tw.pivot(index='p', columns='code', values='revenue_k_twd')
LV = LV.reindex(pd.period_range(LV.index.min(), LV.index.max(), freq='M'))
twv = tw[tw.revenue_k_twd.notna()]
TW_FIRST = twv.groupby('code').p.min().to_dict(); TW_LAST = twv.groupby('code').p.max().to_dict()
TW_ANY = set(tw.code)
r5 = pd.read_csv(K.X / 'rev_5y.csv.gz', dtype={'code': str}); r5['p'] = pd.PeriodIndex(r5.ym, freq='M')
r5 = r5.drop_duplicates(['p', 'code'], keep='last')
R5_ANY = set(r5.code); R5_FIRST = r5.groupby('code').p.min().to_dict(); R5_LAST = r5.groupby('code').p.max().to_dict()
R5_ROWS = set(zip(r5.code, r5.p))

# ---- 名稱（FOREIGN）
names = {}
def addn(c, n):
    if isinstance(n, str): names.setdefault(c, set()).add(n)
if 'rev' in NAMES:
    for c, n in zip(tw.code, tw.name): addn(c, n)
if 'sl' in NAMES:
    for c, n in zip(V.SL.index, V.SL['name']): addn(c, n)
if 'lm' in NAMES:
    lm = pd.read_csv('/home/user/-/research/tw_stock_selection/agents/o_v7_listing_exec_engine/listing_master.csv', dtype=str)
    for col in [c for c in lm.columns if 'name' in c]:
        for c, n in zip(lm.code, lm[col]): addn(c, n)
if 'snap' in NAMES:
    import csv
    SD = '/home/user/-/research/tw_stock_selection/agents/o_pit_shares_industry_2006_2013/raw/industry_snap/'
    for f in ('grs_568e6685_tools_twse_list.csv', 'grs_893c8772_grs_stock_no.csv', 'goristock_c3c718ab_grs_stock_no.csv'):
        for r in csv.reader(open(SD + f, encoding='utf-8', errors='replace')):
            r = [x.strip().strip('"') for x in r]
            if len(r) >= 2 and r[0].isdigit() and len(r[0]) == 4: addn(r[0], r[1])
EXTRAF = set(ARG.get('extraforeign', '').split(',')) - {''}
def foreign(c):
    if c.startswith('91') or c in EXTRAF: return True
    return any(('KY' in n.upper()) or n.endswith('DR') or n.startswith('F-') for n in names.get(c, ()))
LASTTRADE = {c: pd.Period(IDX[np.where(K.TRADED[:, j])[0][-1]], 'M') if K.TRADED[:, j].any() else None for j, c in enumerate(COLS)}

PRI = ['TRUNC', 'IFRS13', 'POST13', 'POST13_R5', 'PRE12', 'HOLE', 'NEWLIST', 'BASE_LE0']
def cause(c, d):
    m = pd.Period(d, 'M')
    if foreign(c): return 'FOREIGN'
    if d >= SWITCH:
        if c not in R5_ANY: return 'POST13_R5' if c in TW_ANY else 'NEVER_DOM'
        cs = []
        for x in (m - 1, m - 2, m - 3):
            if (c, x) in R5_ROWS:
                if not np.isfinite(Y5.at[x, c]) if (x in Y5.index and c in Y5.columns) else True: cs.append('BASE_LE0')
                continue
            if x > R5_LAST[c]: cs.append('POST13_R5')
            elif x < R5_FIRST[c]: cs.append('NEWLIST')
            else: cs.append('HOLE')
        return min(cs, key=PRI.index) if cs else 'OTHER'
    if c not in TW_ANY: return 'NEVER_DOM'
    first, last = TW_FIRST[c], TW_LAST[c]
    cs = []
    for k in (1, 2, 3):
        x = m - k; b = x - 12
        for z in (x, b):
            v = LV.at[z, c] if z in LV.index else np.nan
            if np.isfinite(v): continue
            lt = LASTTRADE.get(c)
            if z > last:
                if lt is not None and pd.Period('2009-12', 'M') <= lt <= pd.Period('2012-12', 'M'): cs.append('TRUNC')
                elif z >= pd.Period('2013-01', 'M') and last >= pd.Period('2012-12', 'M'):
                    cs.append('IFRS13' if d <= pd.Timestamp('2013-12-31') else 'POST13')
                else: cs.append('PRE12')
            elif z < first: cs.append('NEWLIST')
            else: cs.append('HOLE')
        vb = LV.at[b, c] if b in LV.index else np.nan
        vx = LV.at[x, c] if x in LV.index else np.nan
        if np.isfinite(vx) and np.isfinite(vb) and vb <= 0: cs.append('BASE_LE0')
    return min(cs, key=PRI.index) if cs else 'OTHER'

GOV = {'TRUNC', 'IFRS13', 'POST13', 'POST13_R5', 'PRE12', 'HOLE', 'NEVER_DOM'}

def pool_parts(d):
    cs = M.cross_section(d); t, e = cs['t'], cs['e']
    v60 = M.V60[t]; px = K.Cn[t]
    base = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10)
    vv = np.where(base, v60, -np.inf); order = np.argsort(-vv, kind='stable')
    top = np.zeros(N, bool); top[order[:min(300, base.sum())]] = True; top &= base
    pool = top & np.isfinite(cs['s52']) & ~K.disposal_mask(d) & np.isfinite(K.On[e]) & K.TRADED[e] & ~V.FINMASK
    miss = pool & ~np.isfinite(cs['y3'])
    assert ((pool & np.isfinite(cs['y3'])) == cs['el']).all()
    return cs, miss

def pr(x, m):
    return pd.Series(np.where(m, x, np.nan)).rank(pct=True).to_numpy()

def cs_put(cs, Amask, mode, Y3alt=None):
    el = cs['el']; s52 = cs['s52']; y3 = cs['y3'].copy()
    elp = el | Amask; n = elp.sum()
    p52 = pr(s52, elp)
    if mode in ('opt', 'p90'):
        nonA = elp & ~Amask
        rr = pd.Series(np.where(nonA, y3, np.nan)).rank().to_numpy() / n
        prv = np.where(Amask, 1.0 if mode == 'opt' else 0.9, rr)
        if mode == 'opt': y3t = np.where(Amask, np.inf, y3)
        else: y3t = np.where(Amask, np.nanpercentile(y3[el], 90), y3)
    else:   # carry：與其他股一起照正式規則
        y3t = np.where(Amask, Y3alt, y3); prv = pr(y3t, elp)
    sc = 0.5 * p52 + 0.5 * prv; sc[~elp] = np.nan
    y3o = np.where(np.isfinite(y3t), y3t, np.where(np.isposinf(y3t), 1e300, np.nan))
    return dict(t=cs['t'], e=cs['e'], el=elp, sc=sc, y3=y3o)

def run(mode='opt', which='gov', start=START, end=END):
    out = []; held = set(); recs = []
    for d in M.sig_days:
        cs = M.cross_section(d); el = cs['el']
        if el.sum() < 50: continue
        Amask = np.zeros(N, bool); causes = {}
        if start <= d <= end:
            cs0, miss = pool_parts(d)
            for j in np.where(miss)[0]:
                c = cause(COLS[j], d); causes[j] = c
                if (which == 'gov' and c in GOV) or (which == 'all'):
                    Amask[j] = True
            Y3alt = None
            if mode == 'carry':
                Y3alt = np.full(N, np.nan)
                for j in np.where(Amask)[0]:
                    for b in range(1, 25):
                        v = M.y3(d, b)[j]
                        if np.isfinite(v): Y3alt[j] = v; break
                Amask &= np.isfinite(Y3alt)
            csx = cs_put(cs, Amask, mode, Y3alt) if Amask.any() else cs
        else:
            csx = cs
        elx = csx['el']; e = cs['e']; gu = K.gap_up_open(e)
        cand = V.V2.order(csx, 'official')[:int(elx.sum())]
        pick = []
        for j in cand:
            if gu[j] and j not in held: continue
            pick.append(j)
            if len(pick) == 10: break
        for r, j in enumerate(pick):
            recs.append(dict(sig=d.date().isoformat(), rank=r + 1, code=COLS[j], put=bool(Amask[j]), cause=causes.get(j, '')))
        held = set(pick)
    return pd.DataFrame(recs)

if __name__ == '__main__':
    tag = ARG.get('tag', 'A')
    res = {}
    for mode, which in [(x.split(':')[0], x.split(':')[1]) for x in ARG.get('modes', 'opt:gov').split(',')]:
        R = run(mode, which)
        R.to_csv(f'rep_r2_{tag}_{mode}_{which}.csv', index=False)
        P = R[R.put]
        for nm, a, b, den in (('2013_2026', '2013', '2026-12', 1640), ('2013_2018', '2013', '2018-12-31', 720), ('2019_2026', '2019', '2026-12', 920), ('2006_2013', '2006', '2013-12-31', 930)):
            s = P[(P.sig >= a) & (P.sig <= b)]
            if len(s) == 0: continue
            top = s.code.value_counts()
            print(mode, which, nm, f'{len(s)}/{den}={len(s)/den:.4%} days={s.sig.nunique()} iss={s.code.nunique()} top={top.index[0]}:{top.iloc[0]} drop={(len(s)-top.iloc[0])/den:.4%}')
            if nm == '2013_2026' or nm == '2006_2013': print('   causes', s.cause.value_counts().to_dict()); print('   top', top.head(10).to_dict())
