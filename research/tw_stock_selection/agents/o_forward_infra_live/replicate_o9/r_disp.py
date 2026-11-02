"""處置衛星三線（依凍結規格自寫）：事件帳本、時戳、進出場、成本、閒置資金放 0050。"""
import sys, json, math
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_forward_infra_live/replicate_o9')
import numpy as np, pandas as pd
import rlib as L

ASOF = pd.Timestamp('2026-11-02')
REPLAY0 = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp('2026-09-10')
NEW_REG = pd.Timestamp('2026-08-10')
import os
INT = os.environ.get('INT') == '1'

# ---------- 價格（上市＋上櫃，history ≤ asof）----------
h = L.H[(L.H['date'] <= ASOF) & (L.H['volume'] > 0) & (L.H['close'] > 0)].drop_duplicates(['date', 'code'], keep='last')
h = h[h['code'].map(L.MP.is4)]
h['val'] = h['volume'] * h['close']
C = h.pivot(index='date', columns='code', values='close').sort_index().astype(float)
O = h.pivot(index='date', columns='code', values='open').reindex_like(C).astype(float)
V = h.pivot(index='date', columns='code', values='val').reindex_like(C).astype(float)
ex = L.EX[(L.EX['date'] <= ASOF) & L.EX['code'].map(L.MP.is4) & (L.EX['ref_price'] > 0) & (L.EX['prev_close'] > 0)].copy()
ex['f'] = ex['prev_close'] / ex['ref_price']; ex = ex[(ex['f'] >= 1.0) & (ex['f'] < 3.0)]
R = L.MP.hist_returns(C, O, ex)
cal = C.index
V60 = V.rolling(60, min_periods=60).mean()
SIG = R.rolling(60, min_periods=40).std()
PCL = C.ffill().shift(1)
tx = L.TX[L.TX.index <= ASOF]
tx200 = tx.rolling(200).mean()
e50 = L.E50[L.E50.index <= ASOF].reindex(cal).ffill()
mk = L.LST.set_index('code')['market']


def tick(p): return float(L.MP.tick(np.array([p]))[0])


# ---------- 事件帳本 ----------
runs = [d for d in cal if REPLAY0 <= d <= ASOF]          # 每個交易日 asof＝d，時鐘＝d 隔天 06:49
clock = {d: pd.Timestamp(d.date()) + pd.Timedelta(days=1, hours=6, minutes=49) for d in runs}
ev = L.DSPU[L.DSPU['date'] <= ASOF].copy()
ev['market'] = ev['code'].map(mk)
fs = []
for d0 in ev['date']:
    k = [d for d in runs if d >= d0]
    fs.append(clock[k[0]] if k else pd.NaT)
ev['first_seen_ts'] = fs
ev['ts_ok'] = (ev['first_seen_ts'] >= ev['date'] + pd.Timedelta(hours=14, minutes=30)) & \
              (ev['first_seen_ts'] <= ev['start'] + pd.Timedelta(hours=13, minutes=30))
# 鏈規則與 second
ev = ev.sort_values(['code', 'start']).reset_index(drop=True)
ev['take'] = True; ev['second'] = False
for code, g in ev.groupby('code'):
    last_end = None
    for i in g.index:
        if last_end is not None and ev.at[i, 'start'] <= last_end + pd.Timedelta(days=3): ev.at[i, 'take'] = False
        else: last_end = ev.at[i, 'end']
        s = ev.at[i, 'start']; k = cal.searchsorted(s)
        lb = cal[max(k - 30, 0)] if k < len(cal) else s - pd.Timedelta(days=45)
        prior = g[(g['start'] < s) & (g['end'] >= lb)]
        ev.at[i, 'second'] = len(prior) > 0
ev['mm'] = np.where(ev['start'] >= NEW_REG, 2, np.where(ev['second'], 20, 5))
ev['sdate'] = [cal[cal >= s_][0] if (cal >= s_).any() else pd.NaT for s_ in ev['start']]   # 處置起日非交易日 → 第一個交易日
rep = ev[(ev['date'] >= REPLAY0) & (ev['date'] <= ASOF)]
print('replay events', len(rep), rep['market'].value_counts().to_dict(), 'ts_ok', int(rep['ts_ok'].sum()), 'take', int(rep['take'].sum()))


def fee(v, stress): return max(20.0, (0.001425 * 0.6 if stress else 0.001425 * 0.28) * v)


def sim(market, first_only, stress, trades):
    E = ev[(ev['market'] == market) & ev['take'] & ev['first_seen_ts'].notna()]
    if first_only: E = E[~E['second']]
    extra = 0.004 if stress else 0.0
    units = 1e6 / e50.loc[REPLAY0]        # 0050 單位（起始全放 0050）
    pos = {}                               # code -> dict(sh, e_end, mm)
    nav = {}
    nbuy = 0; nlate = 0
    for t in [d for d in cal if d >= REPLAY0]:
        it = cal.get_loc(t); p50 = e50.loc[t]
        # 持股依含息報酬滾動（除權息時股數×F 近似：以 R 推算價值）
        for c, p in pos.items():
            if it > p['last_i']:
                r = R[c].iloc[it]
                if np.isfinite(C[c].iloc[it]):
                    p['val'] *= (1 + (r if np.isfinite(r) else C[c].iloc[it] / C[c].iloc[p['last_i']] - 1)); p['last_i'] = it
        # 出場
        for c in list(pos):
            p = pos[c]
            if t < p['end']: continue
            cl = C[c].iloc[it]
            if not np.isfinite(cl): continue
            ref = PCL[c].iloc[it]; ldn = math.ceil(ref * 0.9 / tick(ref * 0.9) - 1e-9) * tick(ref * 0.9)
            if cl <= ldn + 1e-9: continue
            sh = p['val'] / cl
            Vd = V[c].iloc[it]
            Qref = Vd * p['mm'] / 270 if t <= p['end'] else 0.25 * Vd
            q = min(sh * cl, 0.2 * Qref)
            sg = SIG[c].iloc[it - 1]
            spr = max(tick(cl) / cl, 0.001) + 0.5 * (sg if np.isfinite(sg) else 0) * math.sqrt(q / Qref) + extra
            v = q * (1 - spr); proceeds = v - fee(v, stress) - 0.003 * v
            buy50 = proceeds - fee(proceeds, stress); units += buy50 / p50
            trades.append(dict(kind='paper_disp_close', line=market, code=c, date=str(t.date()), side='sell', value=q, px=cl, open=O[c].iloc[it]))
            p['val'] -= q
            if p['val'] <= 1e-6: del pos[c]
        # 進場
        bear = None
        prv = tx[tx.index < t]
        ok_mkt = len(prv) and np.isfinite(tx200.loc[prv.index[-1]]) and prv.iloc[-1] >= tx200.loc[prv.index[-1]]
        todays = E[E['sdate'] == t].sort_values(['date'])
        for _, e in todays.iterrows():
            c = e['code']
            if not e['ts_ok']: nlate += 1; DEC.append((market, first_only, 'LATE')); continue
            if not ok_mkt: DEC.append((market, first_only, 'BEAR')); continue
            if c in pos or len(pos) >= 10: DEC.append((market, first_only, 'FULL/HELD')); continue
            if c not in C.columns: DEC.append((market, first_only, 'NOPRICE')); continue
            cl = C[c].iloc[it]; r = R[c].iloc[it]
            if not np.isfinite(cl) or (np.isfinite(r) and r >= 0.09): DEC.append((market, first_only, 'UP9')); continue
            v60 = V60[c].iloc[it - 1]; pc = C[c].iloc[:it].dropna()
            if not (np.isfinite(v60) and v60 >= 1e7 and len(pc) and pc.iloc[-1] >= 10): DEC.append((market, first_only, 'LIQ', c, str(t.date()), v60, pc.iloc[-1] if len(pc) else None)); continue
            DEC.append((market, first_only, 'BUY'))
            navt = units * p50 + sum(p['val'] for p in pos.values())
            Vs = V[c].iloc[it]; mm = e['mm']
            amt = min(navt / 10, 0.2 * v60 * mm / 270, 0.2 * Vs * mm / 270)
            if INT: amt = math.floor(amt / cl) * cl     # 變體：整數股
            Qref = Vs * mm / 270; sg = SIG[c].iloc[it - 1]
            spr = max(tick(cl) / cl, 0.001) + 0.5 * (sg if np.isfinite(sg) else 0) * math.sqrt(amt / Qref) + extra
            cost = amt * spr + fee(amt, stress)
            need = amt + cost
            sell50 = need + fee(need, stress) + 0.001 * need
            units -= sell50 / p50
            pos[c] = dict(val=amt, end=e['end'], mm=mm, last_i=it)
            nbuy += 1
            trades.append(dict(kind='paper_disp', line=market, code=c, date=str(t.date()), side='buy', value=amt, px=cl, open=O[c].iloc[it]))
        nav[t] = units * p50 + sum(p['val'] for p in pos.values())
    return pd.Series(nav), nbuy, nlate


DEC = []
res = {}
od0, od1 = L.official_day(2026, 9), L.official_day(2026, 10)
for name, mkt, first in [('DISP_TWSE', 'TWSE', False), ('DISP_TWSE_FIRST', 'TWSE', True), ('DISP_TPEX', 'TPEX', False)]:
    out = {}
    for st in (False, True):
        tr = []
        nav, nb, nl = sim(mkt, first, st, tr)
        x = 100 * (nav.loc[od1] / nav.loc[od0] - (e50.loc[od1] / e50.loc[od0]))
        out['stress' if st else 'normal'] = dict(excess=x, n_buy=nb, n_late=nl, n_trades=len(tr))
    res[name] = out
    print(name, {k: (round(v['excess'], 2), v['n_buy']) for k, v in out.items()})
res['MIX_D80'] = 0.2 * res['DISP_TWSE']['normal']['excess']
res['n_events_replay'] = int(len(rep)); res['n_ts_ok'] = int(rep['ts_ok'].sum())
res['n_buys_total_normal'] = sum(res[k]['normal']['n_buy'] for k in ['DISP_TWSE', 'DISP_TWSE_FIRST', 'DISP_TPEX'])
import collections
print(collections.Counter((d[0], d[1], d[2]) for d in DEC))
print([d for d in DEC if d[2]=='LIQ' and d[0]=='TPEX'])
print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}))
(L.HERE / f'out_disp_{REPLAY0:%m%d}{"_int" if INT else ""}.json').write_text(json.dumps(res, indent=1, default=str))
