# Independent replication (o13) - data and event list. Written from frozen_spec.txt / EXEC_SPEC.json only.
import json, numpy as np, pandas as pd
AG = '/home/user/-/research/tw_stock_selection/agents'
SCR = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel'
D = np.load(f'{AG}/o_exec_engine_rewrite_v7x/data/x_cache.npz', allow_pickle=True)
IDX = pd.DatetimeIndex(D['IDX']); COLS = list(D['COLS']); CI = {c: i for i, c in enumerate(COLS)}
C, O, V, G, PC, REF, V60p, SIGp = [D[k] for k in ['C', 'O', 'V', 'G', 'PC', 'REF', 'V60p', 'SIGp']]
E50 = D['E50']; GOOD = D['GOOD']
NT = len(IDX)
LA = np.cumsum(np.log(np.where(np.isfinite(G), G, 1.0)), axis=0)       # close level (log)
with np.errstate(all='ignore'):
    LAO = LA - np.log(C / O)                                             # open level
SRC = pd.read_parquet(f'{SCR}/src.parquet').reindex(index=IDX, columns=COLS).values
CPI = json.load(open(f'{AG}/o_v7_listing_exec_engine/raw/exec_inputs/cpi_tw.json'))['index_2006_100']
def cpi_ratio(y): return CPI[str(max(2006, min(2026, y)))] / CPI['2026']

# EW benchmark: prior-day liquid (V60p>=1e7, PC>=10), priced today
with np.errstate(all='ignore'):
    elig = (V60p >= 1e7) & (PC >= 10) & np.isfinite(C) & np.isfinite(G)
    rr = np.where(elig, G - 1, np.nan)
    EWr = np.nanmean(rr, axis=1); EWr = np.where(np.isfinite(EWr), EWr, 0.0)
    lco = np.where(elig & np.isfinite(O) & (O > 0), np.log(C / O), np.nan)
    ewlco = np.nanmean(lco, axis=1); ewlco = np.where(np.isfinite(ewlco), ewlco, 0.0)
EWL = np.cumsum(np.log1p(EWr)); EWLO = EWL - ewlco
E50L = np.log(E50); E50LO = E50L  # 0050 open level unknown -> use prior close level for opens
E50LO = np.r_[E50L[0], E50L[:-1]]

def tick(p):
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)
def limpct(d): return 0.07 if d < pd.Timestamp('2015-06-01') else 0.10
def limup(ref, d):
    x = ref * (1 + limpct(d)); t = tick(x); v = np.floor(x / t + 1e-9) * t
    if tick(v) != t: t = tick(v); v = np.floor(x / t + 1e-9) * t
    return v
def limdn(ref, d):
    x = ref * (1 - limpct(d)); t = tick(x); v = np.ceil(x / t - 1e-9) * t
    if tick(v) != t: t = tick(v); v = np.ceil(x / t - 1e-9) * t
    return v

# TAIEX bear switch
tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date').close
bear_s = tx < tx.rolling(200).mean()

GD = IDX[GOOD]; GPOS = np.where(GOOD)[0]

def load_events():
    d = pd.read_csv(f'{AG}/o_disposal_d0_exec_close/raw/disposal_hist_rejudge.csv', dtype={'code': str})
    d['fo'] = np.arange(len(d))
    d = d[d.code.str.fullmatch(r'\d{4}') & ~d.code.str.startswith('0') & ~d.name.str.contains('DR')].copy()
    for c in ['announce', 'start', 'end']: d[c] = pd.to_datetime(d[c])
    d = d[d.code.isin(CI)].copy()
    si = GD.searchsorted(d.start.values, 'left'); ei = GD.searchsorted(d.end.values, 'right') - 1
    ok = (si < len(GD)) & (ei >= 0)
    d = d[ok].copy(); si = si[ok]; ei = ei[ok]
    d['si'] = GPOS[si]; d['ei'] = GPOS[ei]; d['s'] = IDX[d.si]; d['e'] = IDX[d.ei]
    d = d[d.ei > d.si].copy()
    # R6 second flag (30 trading days)
    gidx = {p: k for k, p in enumerate(GPOS)}
    d['sg'] = d.si.map(gidx); d['eg'] = d.ei.map(gidx)
    d = d.sort_values(['start', 'announce', 'fo'])
    sec = []
    for _, r in d.iterrows(): sec.append(False)
    d['second'] = False
    for code, g in d.groupby('code', sort=False):
        sg = g.sg.values; eg = g.eg.values; f = []
        for k in range(len(g)):
            f.append(bool(np.any((eg[:k] >= sg[k] - 30) & (eg[:k] < sg[k]))))
        d.loc[g.index, 'second'] = f
    # R1 chain
    d['adopt'] = True
    for code, g in d.groupby('code', sort=False):
        last_end = None; a = []
        for _, r in g.iterrows():
            if last_end is not None and r.start <= last_end + pd.Timedelta(days=3): a.append(False)
            else: a.append(True); last_end = r.end
        d.loc[g.index, 'adopt'] = a
    d = d.sort_values(['s', 'announce', 'fo']).reset_index(drop=True)
    # R2 bear: last index day before s
    tpos = bear_s.index.searchsorted(d.s.values, 'left') - 1
    d['bear'] = bear_s.values[tpos].astype(bool)
    d['ci'] = d.code.map(CI)
    d['r4'] = [(V60p[r.si, r.ci] >= 1e7) and (PC[r.si, r.ci] >= 10) for r in d.itertuples()]
    d['ann_ok'] = [IDX[r.si - 1] >= r.announce for r in d.itertuples()]
    return d

if __name__ == '__main__':
    d = load_events()
    print(len(d), d.adopt.sum(), (d.adopt & ~d.bear & d.r4).sum(), d.ann_ok.mean())
    x = d[d.adopt & ~d.bear & d.r4 & (d.s >= '2006-04-01')]
    print(len(x), (x.s >= '2024-01-02').sum(), ((x.s >= '2021-10-01') & (x.s <= '2023-12-31')).sum())
    # announce lead
    lead = [GOOD[:r.si].sum() - GOOD[:IDX.searchsorted(r.announce)].sum() for r in x.itertuples()]
    print(pd.Series(lead).value_counts())
