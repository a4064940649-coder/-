#!/usr/bin/env python3
"""第 3 步（資料整理，不算統計量）：成交日誌 → 逐筆成本表、換股表（exit 為單位，B／A1／A2 三種配對）、新進表、持股槽表、換股日彙總表。
全部期間一起建表；統計量由 explore.py（只挑選期）與 analyze.py（凍結後）計算。
輸出 scratchpad/tfc/T_<cell>.pkl（dict of DataFrames）。
python3 -I -B build_tables.py"""
import sys, pickle, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import tfc_lib as TL; sys.path.pop(0)
SCR = TL.SCR
CELLS = [('normal', '100萬'), ('stress', '100萬'), ('normal', '30萬'), ('stress', '30萬'), ('normal', '300萬'), ('normal', '1000萬')]
E = TL.engine(); D = E['D']; S = E['S']; XE = E['XE']
IDX = D.IDX; COLS = D.COLS
with np.errstate(divide='ignore', invalid='ignore'):
    LA = np.cumsum(np.log(np.where(np.isfinite(D.ADJ) & (D.ADJ > 0), D.ADJ, 1.0)), axis=0)
E50 = D.E50
P0 = TL.load_plans(); P0 = P0[P0.seed == 0]
SIG = {(v, j): dict(zip(g.exec_idx.astype(int), g.signal)) for (v, j), g in P0.groupby(['variant', 'j'])}
RK = pd.read_csv(HERE / 'data/ranks_seed0.csv.gz', dtype={'code': str})
RKD = {k: g.set_index('code') for k, g in RK.groupby(['variant', 'j', 'signal'])}
SIGLIST = {(v, j): sorted(g.signal.unique()) for (v, j), g in RK.groupby(['variant', 'j'])}

def px_at(t, j):
    o = D.O[t, j]
    return o if np.isfinite(o) else D.CF[t - 1, j]
def wret(j, e, e2):
    a = px_at(e, j); b = D.O[e2, j]
    if not np.isfinite(b): b = D.CF[e2, j]
    return float(b * np.exp(LA[e2, j] - LA[e, j]) / a - 1)
def pbucket(p):
    return '<20' if p < 20 else '20~50' if p < 50 else '50~100' if p < 100 else '100~500' if p < 500 else '>=500'
def rbucket(rank, why):
    if why != 'el' or not np.isfinite(rank): return 'F_' + str(why)
    r = int(rank)
    return '<=10' if r <= 10 else '11~15' if r <= 15 else '16~30' if r <= 30 else '31~50' if r <= 50 else '>50'

def build(cell):
    F = pd.read_parquet(SCR / f'fills_{cell[0]}_{cell[1]}.parquet')
    F = F[F.shares > 0].copy()
    RB = pickle.load(open(SCR / f'rebs_{cell[0]}_{cell[1]}.pkl', 'rb'))
    stress = cell[0] == 'stress'; extra = 0.004 if stress else 0.0
    # 逐筆成本元件
    v_exec = F.shares * F.px
    rate = np.where(stress | (F.t < XE.I_FEE), 0.001425 * 0.6, 0.001425 * 0.28)
    F['fee_min_excess'] = F.fee - np.maximum(0, rate * v_exec)
    im_ex = (F.im - np.where(F.kind == 'W', extra, 0.0)).clip(lower=0)
    F['tick_amt'] = np.where(F.kind == 'W', F.notional * np.minimum(im_ex, F.tk), np.where(F.kind == 'D', F.notional * F.im, 0.0))
    F['imp_amt'] = np.where(F.kind == 'W', F.notional * np.maximum(0.0, im_ex - F.tk), 0.0)
    F['stress_amt'] = np.where(F.kind == 'W', F.notional * extra, 0.0)
    F['odd_amt'] = np.where(F.kind == 'O', F.notional * F.oddc, 0.0)
    oddpx = np.where(F.kind == 'O', F.notional / F.shares.where(F.shares > 0, 1), np.nan)
    F['pxdev_amt'] = np.where(F.kind == 'O', F.side * (oddpx - F.open) * F.shares, 0.0)
    F['pxdev_amt'] = np.where(F.kind == 'D', (F.open - F.notional / F.shares) * F.shares, F['pxdev_amt'])   # 處置賣在收盤：相對開盤的價差（正＝較差）
    F['cost'] = F.fee + F.tax + F.tick_amt + F.imp_amt + F.stress_amt + F.odd_amt
    sqx = np.where(F.kind == 'W', F.sq.fillna(0.0), 0.0)
    for h, nm in ((0.5, 'h05'), (1.0, 'h10')):
        F['cost_' + nm] = F.fee + F.tax + F.notional * (h * F.tk.fillna(0.0) + sqx + np.where(F.kind == 'W', extra, 0.0))
    F['price'] = F.open.where(np.isfinite(F.open), np.nan)
    F['pb'] = [pbucket(p) for p in F.price.fillna(F.notional / F.shares)]
    F['date'] = IDX[F.t.values]
    out_f, out_x, out_n, out_s, out_r = [], [], [], [], []
    for (v, jj), g in F.groupby(['variant', 'j_']):
        rebs = RB[(v, jj)]; rt = np.array([r[0] for r in rebs]); endi = D.TPOS[S.end_day('AB_V7_B10', int(jj))]
        k = np.searchsorted(rt, g.t.values, side='right') - 1
        assert (k >= 0).all()
        g = g.assign(k=k)
        pk = [set(r[1]) for r in rebs]; hb = [set(r[2]) for r in rebs]; ha = [set(r[3]) for r in rebs]; av = np.array([r[4] for r in rebs])
        cls = np.where(g.side < 0, np.where([c not in pk[kk] for c, kk in zip(g.j, g.k)], 'exit', 'rw_sell'),
                       np.where([c in hb[kk] for c, kk in zip(g.j, g.k)], 'rw_buy', 'entry'))
        g = g.assign(cls=cls, avail=av[g.k.values])
        out_f.append(g)
        sig = SIG[(v, jj)]; sigs = SIGLIST[(v, jj)]
        agg = g.groupby(['k', 'j', 'cls']).agg(notional=('notional', 'sum'), cost=('cost', 'sum'), cost_h05=('cost_h05', 'sum'), cost_h10=('cost_h10', 'sum'),
                                               tax=('tax', 'sum'), fee=('fee', 'sum'), tick=('tick_amt', 'sum'), imp=('imp_amt', 'sum'), odd=('odd_amt', 'sum'),
                                               nP=('ctx', lambda s: int((s == 'P').sum()))).reset_index()
        A = {(r.k, r.j, r.cls): r for r in agg.itertuples()}
        for kk, r in enumerate(rebs):
            e = r[0]; e2 = rt[kk + 1] if kk + 1 < len(rt) else endi
            if e2 <= e: continue
            sg = sig[e]; R0 = RKD.get((v, jj, sg)); si = sigs.index(sg); Rp = RKD.get((v, jj, sigs[si - 1])) if si > 0 else None
            sidx = D.TPOS[pd.Timestamp(sg)]
            exits = sorted({c for (k_, c, cl) in A if k_ == kk and cl == 'exit'})
            ents = sorted({c for (k_, c, cl) in A if k_ == kk and cl == 'entry'})
            def rinfo(c, RR):
                if RR is None or COLS[c] not in RR.index: return np.nan, 'na', np.nan
                q = RR.loc[COLS[c]]; return float(q['rank']), q['why'], float(q['score'])
            ent_rows = []
            for c in ents:
                a = A[(kk, c, 'entry')]; rk, why, sc = rinfo(c, R0); prk, pwhy, _ = rinfo(c, Rp)
                src = '>50/NA' if (not np.isfinite(prk)) or prk > 50 else ('11~15' if 11 <= prk <= 15 else ('<=10' if prk <= 10 else ('16~30' if prk <= 30 else '31~50')))
                ent_rows.append(dict(variant=v, j=jj, k=kk, signal=sg, e=e, code=COLS[c], rank=rk, score=sc, prev_rank=prk, src=src, price=px_at(e, c),
                                     buy_notional=a.notional, buy_cost=a.cost, buy_cost_h05=a.cost_h05, buy_cost_h10=a.cost_h10, r=wret(c, e, e2), deferred=a.nP))
            for q in ent_rows: q['buy_rate'] = q['buy_cost'] / q['buy_notional']; q['buy_rate_h05'] = q['buy_cost_h05'] / q['buy_notional']; q['buy_rate_h10'] = q['buy_cost_h10'] / q['buy_notional']
            out_n.extend(ent_rows)
            ex_rows = []
            for c in exits:
                a = A[(kk, c, 'exit')]; rk, why, sc = rinfo(c, R0)
                ex_rows.append(dict(variant=v, j=jj, k=kk, signal=sg, e=e, code=COLS[c], rank=rk, why=why, score=sc, rb=rbucket(rk, why), price=px_at(e, c),
                                    sell_notional=a.notional, sell_cost=a.cost, sell_cost_h05=a.cost_h05, sell_cost_h10=a.cost_h10, r=wret(c, e, e2), deferred=a.nP))
            for q in ex_rows: q['sell_rate'] = q['sell_cost'] / q['sell_notional']; q['sell_rate_h05'] = q['sell_cost_h05'] / q['sell_notional']; q['sell_rate_h10'] = q['sell_cost_h10'] / q['sell_notional']
            nX, nN = len(ex_rows), len(ent_rows)
            if nN:
                mr = np.mean([q['r'] for q in ent_rows]); mb = np.mean([q['buy_rate'] for q in ent_rows]); mb5 = np.mean([q['buy_rate_h05'] for q in ent_rows]); mb10 = np.mean([q['buy_rate_h10'] for q in ent_rows])
                msc = np.nanmean([q['score'] for q in ent_rows]) if any(np.isfinite(q['score']) for q in ent_rows) else np.nan
                mp = np.mean([q['price'] for q in ent_rows])
            # 配對：A2 邊際（exit 排名由好到壞；強制出場最後）× entry 排名由差到好；A1 反向（exit 由壞到好 × entry 由好到壞）
            kx = lambda q: (0 if q['why'] == 'el' and np.isfinite(q['rank']) else 1, q['rank'] if np.isfinite(q['rank']) else 1e9)
            xs = sorted(ex_rows, key=kx); ns_bad = sorted(ent_rows, key=lambda q: -(q['rank'] if np.isfinite(q['rank']) else 1e9))
            xs1 = list(reversed(xs)); ns_good = list(reversed(ns_bad))
            # A0：依 V7X 實際成交順序（該換股日日誌中第一次出現的先後）配對
            gk0 = g[g.k == kk]; ox = list(dict.fromkeys(gk0.j[(gk0.cls == 'exit')].tolist())); on = list(dict.fromkeys(gk0.j[(gk0.cls == 'entry')].tolist()))
            exd = {COLS[c]: q for c, q in zip(exits, ex_rows)} if False else {q['code']: q for q in ex_rows}; end_ = {q['code']: q for q in ent_rows}
            xs0 = [exd[COLS[c]] for c in ox if COLS[c] in exd]; ns0 = [end_[COLS[c]] for c in on if COLS[c] in end_]
            for q in ex_rows:
                q.update(n_exits=nX, n_entries=nN)
                if nN:
                    q.update(edge_B=mr - q['r'], rtc_B=q['sell_rate'] + mb, rtc_B_h05=q['sell_rate_h05'] + mb5, rtc_B_h10=q['sell_rate_h10'] + mb10,
                             dscore_B=msc - q['score'] if np.isfinite(q['score']) else np.nan, entry_price_mean=mp)
            for lst, nl, tag in ((xs, ns_bad, 'A2'), (xs1, ns_good, 'A1'), (xs0, ns0, 'A0')):
                for q, p in zip(lst, nl):
                    q.update({f'edge_{tag}': p['r'] - q['r'], f'rtc_{tag}': q['sell_rate'] + p['buy_rate'], f'rtc_{tag}_h05': q['sell_rate_h05'] + p['buy_rate_h05'],
                              f'rtc_{tag}_h10': q['sell_rate_h10'] + p['buy_rate_h10'], f'partner_{tag}': p['code'], f'partner_rank_{tag}': p['rank'], f'partner_price_{tag}': p['price'],
                              f'dscore_{tag}': (p['score'] - q['score']) if np.isfinite(q['score']) and np.isfinite(p['score']) else np.nan})
            out_x.extend(ex_rows)
            # 持股槽
            npk = len(r[1]); r50 = float(E50[e2 - 1] / E50[e - 1] - 1)
            for c in sorted(ha[kk] & pk[kk]):
                cst = sum(A[(kk, c, cl)].cost for cl in ('entry', 'rw_buy', 'rw_sell') if (kk, c, cl) in A)
                if (kk + 1, c, 'exit') in A: cst += A[(kk + 1, c, 'exit')].cost
                rs = wret(c, e, e2)
                out_s.append(dict(variant=v, j=jj, k=kk, signal=sg, e=e, code=COLS[c], price=px_at(e, c), r=rs, r50=r50, contrib=(rs - r50) / npk, contrib_raw=rs / npk,
                                  cost_pct=cst / av[kk], is_entry=int((kk, c, 'entry') in A)))
            # 換股日彙總
            gk = g[g.k == kk]
            row = dict(variant=v, j=jj, k=kk, signal=sg, e=e, e2=e2, avail=av[kk], n_exits=nX, n_entries=nN, n_picks=npk, r50_win=r50,
                       m21=float(E50[sidx] / E50[sidx - 21] - 1), m21_126=float(E50[sidx - 21] / E50[sidx - 126] - 1), m63=float(E50[sidx] / E50[sidx - 63] - 1))
            for cl in ('exit', 'rw_sell', 'entry', 'rw_buy'):
                q = gk[gk.cls == cl]; row[f'not_{cl}'] = float(q.notional.sum()); row[f'cost_{cl}'] = float(q.cost.sum())
            for comp in ('tax', 'fee', 'tick_amt', 'imp_amt', 'odd_amt', 'stress_amt', 'fee_min_excess', 'pxdev_amt', 'cost', 'cost_h05', 'cost_h10'):
                row[comp] = float(gk[comp].sum())
            out_r.append(row)
    T = dict(fills=pd.concat(out_f, ignore_index=True), exits=pd.DataFrame(out_x), entries=pd.DataFrame(out_n), slots=pd.DataFrame(out_s), rebs=pd.DataFrame(out_r))
    for k_, df in T.items():
        if 'e' in df.columns: df['date'] = IDX[df.e.values]
    pickle.dump(T, open(SCR / f'T_{cell[0]}_{cell[1]}.pkl', 'wb'))
    return {k_: len(df) for k_, df in T.items()}
if __name__ == '__main__':
    t0 = time.time()
    for c in CELLS:
        print(c, build(c), round(time.time() - t0), flush=True)
