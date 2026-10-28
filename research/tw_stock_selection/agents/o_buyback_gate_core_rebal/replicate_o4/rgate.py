#!/usr/bin/env python3
"""獨立複製：步驟 0 閘門（庫藏股決議公告後 20 日報酬）。只依凍結規格與 gate_prereg.json 的詞表。"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent.parent
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
G = json.loads((HERE.parent / 'gate_prereg.json').read_text(encoding='utf-8'))['PREREG']
RR, CR = G['resolution_rule'], G['completion_rule']
T = pd.read_parquet(AG / 'o_cb_board_runup' / 'mops_titles.parquet')
T = T[(T.date >= '2011-01-01') & (T.date <= '2018-12-31')].copy()
T['title'] = T['title'].astype(str); T['code'] = T['code'].astype(str).str.strip()
out = {'n_titles': len(T)}
has = lambda s, ws: any(w in s for w in ws)
def is_res(s, excl=True):
    ok = all(w in s for w in RR['must_contain_all']) and has(s, RR['must_contain_any_A']) and has(s, RR['must_contain_any_B'])
    return ok and (not excl or not has(s, RR['exclude_any']))
def is_comp(s):
    return all(w in s for w in CR['must_contain_all']) and has(s, CR['must_contain_any']) and not has(s, CR['exclude_any'])
T['res'] = T.title.map(is_res); T['comp'] = T.title.map(is_comp)
T['wide'] = T.title.map(lambda s: '買回' in s and has(s, RR['must_contain_any_A']))
out['n_res_titles'] = int(T.res.sum()); out['n_comp_titles'] = int(T.comp.sum()); out['n_wide_titles'] = int(T.wide.sum())
# 放回
ALL = pd.read_parquet(AG / 'o_cb_board_runup' / 'mops_titles.parquet'); ALL['code'] = ALL['code'].astype(str).str.strip(); ALL['title'] = ALL['title'].astype(str)
ALL['buy_noncomp'] = ALL.title.map(lambda s: '買回' in s and not has(s, CR['must_contain_any']) and '更正' not in s)
res = T[T.res][['code', 'date']]
resd = res.groupby('code')['date'].apply(list).to_dict()
pb = []
for _, r in T[T.comp].iterrows():
    d = r.date; c = r.code
    if any(d - pd.Timedelta(days=75) <= x <= d for x in resd.get(c, [])): continue
    cand = ALL[(ALL.code == c) & ALL.buy_noncomp & (ALL.date >= d - pd.Timedelta(days=75)) & (ALL.date <= d)]
    if len(cand): pb.append((c, cand.date.min(), False))
    else: pb.append((c, d - pd.Timedelta(days=61), True))
PB = pd.DataFrame(pb, columns=['code', 'date', 'imputed']).drop_duplicates(['code', 'date'])
out['n_putback_raw'] = len(pb); out['n_putback_dedup'] = len(PB); out['n_putback_imputed'] = int(PB.imputed.sum())
def merge(df):
    ev = []
    for c, g in df.sort_values('date').groupby('code'):
        last = None
        for d in g.date:
            if last is None or (d - last).days > 30: ev.append((c, d)); last = d
    return pd.DataFrame(ev, columns=['code', 'date'])
raw = pd.concat([res.assign(imputed=False), PB])
EV = merge(raw); out['n_events_merged'] = len(EV)
EVW = merge(T[T.wide][['code', 'date']]); out['n_wide_merged'] = len(EVW)
EVW2 = merge(pd.concat([T[T.wide][['code', 'date']], PB[['code', 'date']]])); out['n_wide_plus_putback_merged'] = len(EVW2)
# 面板
ADJ = pd.read_parquet(P / 'adj.parquet'); C = pd.read_parquet(P / 'close_raw.parquet'); O = pd.read_parquet(P / 'open.parquet'); VAL = pd.read_parquet(P / 'value.parquet')
good = C.notna().sum(axis=1) >= 300
Vg = VAL[good]; A60 = Vg.rolling(60, min_periods=50).mean()
IDX = C.index; IDXG = IDX[good.to_numpy()]
SL = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype=str)
FIN = set(SL[SL.industry.fillna('').str.contains('金融')].code)
def isfin(c): return c in FIN or c.startswith('28') or c.startswith('58')
def evrets(E, h=20, use_good_idx=False):
    rows = []
    ix = IDXG if use_good_idx else IDX
    for c, d in zip(E.code, E.date):
        if c not in C.columns: continue
        k = ix.searchsorted(d, side='right') - 1
        if k < 0: continue
        sd = ix[k]
        liq = A60[c].get(sd, np.nan) if sd in A60.index else A60[c][:sd].iloc[-1] if len(A60[c][:sd]) else np.nan
        ti = IDX.get_loc(sd)
        e = ti + 1; x = e + h
        r = np.nan
        if x < len(IDX):
            op, cl, ad = O[c].iloc[e], C[c].iloc[e], ADJ[c].iloc[e]
            if np.isfinite(op) and np.isfinite(cl) and cl > 0:
                ent = op * ad / cl; ex = ADJ[c].iloc[x]
                r = ex / ent - 1 - 0.0087
        rows.append((c, d, sd, liq, r, isfin(c)))
    return pd.DataFrame(rows, columns=['code', 'date', 'sd', 'liq', 'r', 'fin'])
for h in (20, 19):
    R = evrets(EV, h)
    s2 = R[R.liq >= 2e8]
    out[f'h{h}'] = dict(n_in_panel=len(R), n_1e8=int((R.liq >= 1e8).sum()), n_2e8=len(s2), n_2e8_exfin=int((~s2.fin).sum()),
                        by_year=s2.date.dt.year.value_counts().sort_index().to_dict(), median=float(s2.r.median() * 100), mean=float(s2.r.mean() * 100),
                        median_all=float(R.r.median() * 100))
RW = evrets(EVW); out['wide_n_in_panel'] = len(RW); out['wide_2e8'] = int((RW.liq >= 2e8).sum())
RW2 = evrets(EVW2); out['wide_pb_n_in_panel'] = len(RW2); out['wide_pb_2e8'] = int((RW2.liq >= 2e8).sum())
out['gate_pass'] = bool(out['h20']['n_2e8'] >= 200 and out['h20']['median'] > 0)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
(HERE / 'rgate.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
