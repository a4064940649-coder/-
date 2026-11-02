# P3：事件時鐘歷史模擬（只算周轉與成本，不看報酬）
import pandas as pd, numpy as np, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
P = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/'
SCR = '/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/'
cr = pd.read_parquet(P + 'close_raw.parquet'); adj = pd.read_parquet(P + 'adj.parquet')
op = pd.read_parquet(P + 'open.parquet'); val = pd.read_parquet(P + 'value.parquet'); ret = pd.read_parquet(P + 'ret.parquet')
good = cr.index[cr.notna().sum(axis=1) >= 300]
cr, adj, op, val, ret = [x.loc[good] for x in (cr, adj, op, val, ret)]
days = cr.index
W52 = adj / adj.rolling(252, min_periods=200).max()
V60 = val.rolling(60, min_periods=40).mean()
S60 = ret.rolling(60, min_periods=40).std()
sl = pd.read_csv(SCR + 'stock_list.csv', dtype=str)
fin = set(sl.loc[sl.industry == '金融保險業', 'code'])
kydr = set(sl.loc[sl.name.str.contains('KY|DR', na=False), 'code']) | {c for c in cr.columns if c.startswith('91')}
dsp = pd.concat([pd.read_csv(SCR + 'extras/disposal_5y.csv.gz', dtype=str), pd.read_csv(SCR + 'extras/warnings_hist.csv.gz', dtype=str)])
dsp = dsp[dsp.flag.str.contains('處置', na=False)].drop_duplicates(['code', 'start', 'end'])
r5 = pd.read_csv(SCR + 'extras/rev_5y.csv.gz', dtype={'code': str}); rh = pd.read_csv(SCR + 'extras/rev_hist.csv.gz', dtype={'code': str})
rev = pd.concat([r5[['code', 'ym', 'yoy']], rh[['code', 'ym', 'yoy']]]).drop_duplicates(['code', 'ym'])
YO = rev.pivot(index='ym', columns='code', values='yoy')
def rmean(m, lags):
    p = pd.Period(m, 'M'); return YO.reindex([str(p - k) for k in lags]).mean(skipna=False)
tasi = pd.read_csv(os.path.join(HERE, 'fs_tasi.csv.gz'), dtype={'code': str})
tasi['T'] = pd.to_datetime(tasi['T'], utc=True).dt.tz_convert('Asia/Taipei').dt.tz_localize(None)
def first_open_after(T):
    d0 = pd.Timestamp(T.date())
    if T.hour < 9 and d0 in days: return d0
    i = days.searchsorted(d0, side='right'); return days[i] if i < len(days) else pd.NaT
tasi['e'] = tasi['T'].map(first_open_after)
def sig_day(m):
    p = pd.Period(m, 'M'); return days[days >= pd.Timestamp(p.year, p.month, 11)][0]
def nxt(d): return days[days.searchsorted(d, side='right')]
def prv(d): return days[days.searchsorted(d) - 1]
def pool_scores(d, e, revser):
    # d＝資料日（收盤），e＝成交日
    v = V60.loc[d].dropna().sort_values(ascending=False)
    top = v.index[:300]
    ds = set(dsp[(dsp.start <= str(e.date())) & (dsp.end >= str(e.date()))].code)
    c = cr.loc[d, top]; w = W52.loc[d, top]
    ok = [x for x in top if c[x] >= 10 and pd.notna(w[x]) and x not in ds and x not in fin and x not in kydr]
    w = W52.loc[d, ok]; r = revser.reindex(ok)
    ok2 = r.dropna().index
    s = 0.5 * w[ok2].rank(pct=True) + 0.5 * r[ok2].rank(pct=True)
    return s.sort_values(ascending=False)
anchors = [str(p) for p in pd.period_range('2024-06', '2026-10', freq='M')]
SIG = {m: sig_day(m) for m in anchors}; EX = {m: nxt(SIG[m]) for m in anchors}
OFFL = {}
for m in anchors:
    s = pool_scores(SIG[m], EX[m], rmean(m, (1, 2, 3))); OFFL[m] = list(s.index[:10])
tick = lambda p: np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)
def sim(K_out, delta, correct, tag):
    hold = set(OFFL[anchors[0]]); off = set(OFFL[anchors[0]])
    rows = []; trades = []
    for k in range(1, len(anchors)):
        m = anchors[k]; e0, e1 = EX[anchors[k - 1]], EX[m]
        prevm = str(pd.Period(m, 'M') - 1)
        ev = tasi[(tasi.ym == prevm)]
        win = [d for d in days if e0 < d < e1]
        bought, sold = set(), set(); nb = ns = 0; early = []
        r_old = rmean(m, (2, 3, 4)); r_new = rmean(m, (1, 2, 3))
        for e in win:
            d = prv(e)
            known = set(ev[ev.e <= e].code)
            evs = set(ev[ev.e == e].code)
            if not evs: continue
            rs = r_old.copy(); kk = [c for c in known if c in rs.index]; rs[kk] = r_new[kk]
            s = pool_scores(d, e, rs); rank = {c: i + 1 for i, c in enumerate(s.index)}
            pc = cr.loc[d]; o = op.loc[e]
            can_buy = lambda c: pd.notna(o.get(c)) and o[c] < pc[c] * 1.095
            can_sell = lambda c: pd.notna(o.get(c)) and o[c] > pc[c] * 0.905
            def sell(c):
                nonlocal ns
                hold.discard(c); sold.add(c); ns += 1; trades.append((m, e, c, 'S'))
            def buy(c):
                nonlocal nb
                hold.add(c); bought.add(c); nb += 1; early.append(c); trades.append((m, e, c, 'B'))
            for c in sorted(evs & hold):
                if c in sold: continue
                if (rank.get(c, 10 ** 9) > K_out) and can_sell(c): sell(c)
            cand = sorted([c for c in evs - hold if rank.get(c, 99) <= 10 and c not in bought], key=lambda c: rank[c])
            for c in cand:
                if not can_buy(c): continue
                if len(hold) < 10: buy(c); continue
                elig = [h for h in hold if h not in bought and h not in sold and can_sell(h)]
                if not elig: continue
                lo = min(elig, key=lambda h: s.get(h, -1))
                if s[c] - s.get(lo, -1) >= delta: sell(lo); buy(c)
            if len(hold) < 10:
                for c in s.index[:10]:
                    if len(hold) >= 10: break
                    if c not in hold and c not in bought and can_buy(c): buy(c)
        L = set(OFFL[m])
        corr = (len(hold - L) + len(L - hold)) if correct else 0
        if correct:
            for c in hold - L: trades.append((m, e1, c, 'S'))
            for c in L - hold: trades.append((m, e1, c, 'B'))
            hold = set(L)
        offn = len(off - L) + len(L - off); off = set(L)
        rows.append(dict(anchor=m, ec=nb + ns + corr, off=offn, early_buys=nb, early_in_off=sum(c in L for c in early), n_early=len(early)))
    R = pd.DataFrame(rows)
    return R, pd.DataFrame(trades, columns=['anchor', 'e', 'code', 'side'])
out = {}
for tag, K, dl, cor in [('V1', 15, 0.0, True), ('V2', 20, 0.05, True), ('V3', 30, 0.10, True), ('V4', 20, 0.05, False)]:
    R, Tr = sim(K, dl, cor, tag)
    out[tag] = dict(n_anchors=len(R), ec=R.ec.mean(), off=R.off.mean(), extra=(R.ec - R.off).mean(), early_buys=R.early_buys.mean(),
                    early_in_off=R.early_in_off.sum() / max(R.n_early.sum(), 1))
    if tag == 'V2': Tr.to_csv(os.path.join(HERE, 'p3_trades_V2.csv'), index=False); R.to_csv(os.path.join(HERE, 'p3_monthly_V2.csv'), index=False)
    print(tag, out[tag], flush=True)
json.dump(out, open(os.path.join(HERE, 'p3_rep.json'), 'w'), indent=1, default=float)
