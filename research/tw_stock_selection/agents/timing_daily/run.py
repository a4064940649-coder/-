#!/usr/bin/env python3
"""timing_daily：擇時開關日頻長期驗證。規則見 results.json 的 PREREG（prereg.py 寫入，本程式只讀不改）。
執行：python3 -I -B prereg.py && python3 -I -B run.py"""
import json, hashlib, math
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; A = H.parent
RES = json.loads((H / 'results.json').read_text(encoding='utf-8'))
PR = RES['PREREG']
assert hashlib.sha256(json.dumps(PR, ensure_ascii=False, sort_keys=True).encode()).hexdigest() == RES['PREREG_sha256']

nav = pd.read_parquet(A / 'revenue_longhistory' / 'oos_nav.parquet')
IDX = nav.index
tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index()
e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
E50 = e50.reindex(e50.index.union(IDX)).ffill().reindex(IDX)
TGT = {'n30': nav['M2_n30_slip0.001'], 'n10': nav['M2_n10_slip0.001'], '0050': E50}
COST = {'n30': (0.00486, 0.00186), 'n10': (0.00486, 0.00186), '0050': (0.001855, 0.000855)}
RET = {k: v.pct_change().fillna(0.0) for k, v in TGT.items()}
R50 = RET['0050']

# ---------- 訊號（TAIEX 日曆上計算，t 日收盤後已知） ----------
ma200 = tx.rolling(200, min_periods=200).mean(); ma120 = tx.rolling(120, min_periods=120).mean()
def asof(s): return s.reindex(s.index.union(IDX)).ffill().reindex(IDX)
sig = {}
sig['S2'] = asof((tx >= ma200).astype(float))
sig['S5'] = asof((tx >= ma120).astype(float))
st = []; inn = None
for c, m in zip(tx.to_numpy(), ma200.to_numpy()):
    if not np.isfinite(m): st.append(1.0); continue
    if inn is None: inn = c >= m
    elif inn and c < m * 0.98: inn = False
    elif (not inn) and c > m * 1.02: inn = True
    st.append(float(inn))
sig['S3'] = asof(pd.Series(st, index=tx.index))
s_idx = pd.Series(IDX, index=IDX)
REB = pd.DatetimeIndex(s_idx.groupby([IDX.year, IDX.month]).apply(lambda s: s[s.dt.day >= 11].iloc[0] if (s.dt.day >= 11).any() else pd.NaT).dropna())
s2 = sig['S2']; s1 = pd.Series(np.nan, index=IDX); s1[REB] = s2[REB]; sig['S1'] = s1.ffill().fillna(1.0)
me = tx.groupby([tx.index.year, tx.index.month]).tail(1)
me_sig = (me >= me.rolling(10, min_periods=10).mean()).astype(float)
sig['S4'] = asof(me_sig)
SW = ['S1', 'S2', 'S3', 'S4', 'S5']
for k in SW: assert sig[k].notna().all()

def overlay(tg, sw, lag=2, cash_rate=0.0, cost=True):
    r = RET[tg].to_numpy().copy()
    if sw == 'none':
        pos = np.ones(len(IDX))
    else:
        pos = sig[sw].shift(lag).to_numpy().copy(); pos[:lag] = 1.0
    days = np.r_[0, np.diff(IDX.values).astype('timedelta64[D]').astype(float)]
    cr = (1 + cash_rate) ** (days / 365.0) - 1
    f = np.where(pos > 0.5, 1 + r, 1 + cr)
    nxt = np.r_[pos[1:], pos[-1]]
    ex = (pos > 0.5) & (nxt < 0.5); en = (pos < 0.5) & (nxt > 0.5)
    if cost:
        ce, cb = COST[tg]; f = f * np.where(ex, 1 - ce, 1.0) * np.where(en, 1 - cb, 1.0)
    return pd.DataFrame({'f': f, 'pos': pos, 'exit': ex, 'entry': en, 'r_under': r}, index=IDX)

def wmask(a, b): return (IDX >= pd.Timestamp(a)) & (IDX <= pd.Timestamp(b))
def tstat(x):
    x = pd.Series(x).dropna(); return round(float(x.mean() / x.std(ddof=1) * math.sqrt(len(x))), 2) if len(x) > 2 and x.std() > 0 else None

def cagr_dd(f, ix):
    eq = np.cumprod(f); first = np.argmax(ix); last = len(ix) - 1 - np.argmax(ix[::-1])
    yrs = (IDX[last] - IDX[first - 1 if first > 0 else first]).days / 365.25
    v = eq[ix] / (eq[first - 1] if first > 0 else 1.0)
    vv = np.r_[1.0, v]; dd = float((vv / np.maximum.accumulate(vv) - 1).min())
    return float(v[-1] ** (1 / yrs) - 1), dd, float(v[-1] - 1)

def episodes(df):
    p = df['pos'].to_numpy(); out = []; i = 0
    while i < len(p):
        if p[i] < 0.5:
            j = i
            while j + 1 < len(p) and p[j + 1] < 0.5: j += 1
            out.append((IDX[i], IDX[j], float(np.prod(1 + df['r_under'].to_numpy()[i:j + 1]) - 1), j - i + 1)); i = j + 1
        else: i += 1
    return out

def wstats(tg, sw, a, b, lag=2, cash_rate=0.0):
    df = overlay(tg, sw, lag, cash_rate); ix = wmask(a, b)
    c, dd, tot = cagr_dd(df['f'].to_numpy(), ix)
    c0, _, _ = cagr_dd(overlay(tg, sw, lag, cash_rate, cost=False)['f'].to_numpy(), ix)
    d = df[ix]; rr = d['f'] - 1
    m = (1 + rr).groupby([rr.index.year, rr.index.month]).prod() - 1
    mb = (1 + R50[ix]).groupby([rr.index.year, rr.index.month]).prod() - 1
    o = {'CAGR': round(c, 4), 'maxDD': round(dd, 4), 'Calmar': round(c / abs(dd), 3) if dd < 0 else None, 'total': round(tot, 4),
         't_vs_0050_m': tstat(m - mb), 'switches': int(d['exit'].sum() + d['entry'].sum()), 'exits': int(d['exit'].sum()),
         'pct_out': round(float((d['pos'] < 0.5).mean()), 3), 'cost_drag_pp': round((c0 - c) * 100, 2)}
    if sw != 'none':
        out = d['pos'] < 0.5; ru = d['r_under'][out]
        o['missed_sum'] = round(float(ru.sum()), 4); o['missed_compound'] = round(float((1 + ru).prod() - 1), 4)
        eps = [e for e in episodes(df) if pd.Timestamp(a) <= e[0] <= pd.Timestamp(b)]
        o['episodes'] = len(eps); o['ep_pos_sum'] = round(sum(e[2] for e in eps if e[2] > 0), 4)
        o['ep_neg_sum'] = round(sum(e[2] for e in eps if e[2] <= 0), 4); o['ep_pos_n'] = sum(1 for e in eps if e[2] > 0)
    return o

P = PR['periods']; WIN = {'selection': P['selection'], 'validation': P['validation'], 'holdout': P['holdout'], 'full': P['full_context_only']}
BEAR = {'2008': ('2008-01-01', '2008-12-31'), '2011': ('2011-01-01', '2011-12-31'), '2015': ('2015-01-01', '2015-12-31'),
        '2018': ('2018-01-01', '2018-12-31'), '2020-03': ('2020-02-01', '2020-04-30'), '2022': ('2022-01-01', '2022-12-31')}
ALL = ['none'] + SW
out = {'main_cash0': {}, 'cash1pct': {}, 'lag1_sensitivity': {}, 'yearly_pct': {}, 'bear': {}, 'episodes_full': {}}
for tg in TGT:
    for sw in ALL:
        k = f'{tg}|{sw}'
        out['main_cash0'][k] = {w: wstats(tg, sw, *ab) for w, ab in WIN.items()}
        out['cash1pct'][k] = {w: wstats(tg, sw, *ab, cash_rate=0.01) for w, ab in WIN.items()}
        if sw != 'none': out['lag1_sensitivity'][k] = {w: wstats(tg, sw, *ab, lag=1) for w, ab in WIN.items()}
        df = overlay(tg, sw); rr = df['f'] - 1; rr = rr[IDX >= pd.Timestamp('2006-04-12')]
        out['yearly_pct'][k] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in rr.groupby(rr.index.year)}
        out['bear'][k] = {}
        for bk, ab in BEAR.items():
            c, dd, tot = cagr_dd(df['f'].to_numpy(), wmask(*ab)); out['bear'][k][bk] = {'ret': round(tot, 4), 'maxDD': round(dd, 4)}
        if sw != 'none' and tg == 'n30':
            out['episodes_full'][k] = [[str(e[0].date()), str(e[1].date()), round(e[2], 4), e[3]] for e in episodes(df)]

# ---------- 挑選與判定 ----------
sel = {}
for tg in TGT:
    cal = {sw: out['main_cash0'][f'{tg}|{sw}']['selection']['Calmar'] for sw in SW}
    best = max(cal, key=lambda s: cal[s])
    base = out['main_cash0'][f'{tg}|none']; pick = out['main_cash0'][f'{tg}|{best}']
    ok = {}
    for w in ['validation', 'holdout']:
        ok[w] = bool(pick[w]['Calmar'] > base[w]['Calmar'] and pick[w]['CAGR'] >= base[w]['CAGR'] - 0.03)
    n = sum(ok.values()); verdict = '可用' if n == 2 else ('待觀察' if n == 1 else '不可用')
    # 每個開關的同規則結果（資訊用，不作決策）
    info = {}
    for sw in SW:
        p = out['main_cash0'][f'{tg}|{sw}']
        info[sw] = {w: bool(p[w]['Calmar'] > base[w]['Calmar'] and p[w]['CAGR'] >= base[w]['CAGR'] - 0.03) for w in ['validation', 'holdout']}
    sel[tg] = {'selection_calmar': cal, 'none_selection_calmar': base['selection']['Calmar'], 'picked': best, 'pass': ok, 'verdict': verdict,
               'all_switches_same_rule_info_only': info}
out['selection'] = sel
RES.update({'results': out})
(H / 'results.json').write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding='utf-8')

# ---------- 印出 ----------
for tg in TGT:
    print(f'\n=== {tg}  picked={sel[tg]["picked"]}  verdict={sel[tg]["verdict"]} {sel[tg]["pass"]}')
    for sw in ALL:
        r = out['main_cash0'][f'{tg}|{sw}']
        print(sw.ljust(5), ' | '.join(f"{w[:4]} {r[w]['CAGR']*100:5.1f}% {r[w]['maxDD']*100:6.1f}% C{r[w]['Calmar']:.2f} t{r[w]['t_vs_0050_m']} sw{r[w]['switches']} out{r[w]['pct_out']:.2f} drag{r[w]['cost_drag_pp']}" for w in WIN))
