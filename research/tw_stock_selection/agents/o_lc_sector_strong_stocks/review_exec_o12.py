#!/usr/bin/env python3
"""反方審查（成本與可成交）o12：唯讀引用研究員 lc_lib／lc_exec，不改任何研究員檔案。python3 -I -B review_exec_o12.py
1. 由 window_returns_key.parquet 重算主欄 t（13 j 中位數）。
2. 用 out/plans_SEL_CTRL_J0_J6_J12.csv.gz 重跑 run_exec（基準複製），再跑變體：
   LIM ：e 開盤價已在漲停（買）／跌停（賣）時該筆不成交（留現金／續抱）；
   ODD ：2020-10-26 前零股以 e 收盤價成交時，持股自 e 收盤起算（原引擎以 e 開盤價計值→多算 e 日盤中報酬；fcl 已算但沒用）。
3. PURE 版由 SAT 計畫反推（sleeve＝(tot−0.7·wB)/0.3）。"""
import sys, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import lc_lib as X
import lc_exec as E
L, TC = X.L, X.TC

W = pd.read_parquet(HERE / 'out' / 'window_returns_key.parquet')
PER = {'VAL': ('2017-01-01', '2026-12-31'), 'Y19': ('2019-01-01', '2026-12-31'), 'PIT': ('2014-01-01', '2026-12-31'), 'FULL': ('2006-01-01', '2026-12-31')}

def tst(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 2 else np.nan

def bench(days): return X.bench_win(days)

print('== 1. 主欄重算（window_returns_key，13 j 中位數 t）==')
for spec in ('SEL', 'SEL_PURE', 'CTRL_SEL', 'CTRL_SEL_PURE', 'B'):
    for col in ('100萬|GROW|normal', '100萬|GROW|stress'):
        row = {}
        for pk, (a, b) in PER.items():
            ts = []
            for j in range(13):
                c = f'exec|{spec}|J{j}|{col}'; s = W[c].dropna()
                days = list(s.index) ; dd = X.days_for(j, 'FULL'); assert [str(d.date()) for d in dd[:-1]] == [str(d.date()) for d in days]
                ex = s.to_numpy() - bench(dd)
                m = (s.index >= a) & (s.index <= b); ts.append(tst(ex[m]))
            row[pk] = round(float(np.median(ts)), 2)
        print(spec, col, row)

P = pd.read_csv(HERE / 'out' / 'plans_SEL_CTRL_J0_J6_J12.csv.gz', dtype={'code': str}, low_memory=False)
RATE, TAX, MINFEE = E.RATE, E.TAX, E.MINFEE
tickv = E.tickv
LIMDATE = pd.Timestamp('2015-06-01')
FLAG = L.Fn

def lim_prices(e, j):
    pc = L.PREVC[e, j]
    if not np.isfinite(pc) or pc <= 0: return np.inf, -np.inf
    pct = 0.07 if X.IDX[e] < LIMDATE else 0.10
    up = pc * (1 + pct); dn = pc * (1 - pct)
    tu = float(tickv(up)); td = float(tickv(dn))
    up = math.floor(up / tu + 1e-9) * tu; dn = math.ceil(dn / td - 1e-9) * td
    return up, dn

def run_exec2(days, plans, tier, mode, extra=0.0, lim=False, oddfix=False, sleeve_codes=None, log=None):
    """E.run_exec 的逐行複本，加 lim／oddfix 兩旗標（兩者皆 False 時應與原引擎逐值相同）。"""
    ti = [X.IDX.get_loc(d) for d in days]
    v = np.zeros(X.N); cash = tier * E.cpi_scale(days[0]); out = []
    for k in range(len(ti) - 1):
        t, t2 = ti[k], ti[k + 1]; e = t + 1
        V0 = v.sum() + cash
        fo = TC.factor_close_to_open(t, e); tr = np.isfinite(fo)
        dead = (v > 0) & (L.LAST_TRADE < e)
        cash += (v[dead] * (1 - RATE - TAX - 0.001 - extra)).sum(); v[dead] = 0.0
        v = np.where(tr, v * np.nan_to_num(fo, nan=1.0), v)
        frz = (v > 0) & ~tr
        Eq = v.sum() + cash; flow_cost = 0.0; r_on = Eq / V0
        idx, w = plans[k]
        wv = np.zeros(X.N); np.add.at(wv, idx, w); wv[frz] = 0.0; wv[~tr] = 0.0; sw_ = wv.sum(); wv = wv / sw_ if sw_ > 0 else wv
        Etr = Eq - v[frz].sum()
        O = L.On[e]; cur_sh = np.where(tr, v / np.where(tr, O, 1.0), 0.0)
        tgt_sh = np.where(tr, np.round(wv * Etr * 0.995 / np.where(tr, O, 1.0)), 0.0)
        dsh = tgt_sh - cur_sh
        small = (np.abs(dsh * np.nan_to_num(O)) < E.BAND * Eq) & (tgt_sh > 0)
        dsh[small] = 0.0
        dsh = np.where(tr, np.round(dsh), 0.0)
        cash_delta = 0.0; cash_close = 0.0; odd_buy_pre = np.zeros(X.N)
        for j in np.where(dsh != 0)[0]:
            q = dsh[j]; side = 1 if q > 0 else -1; a = abs(q); lots, odd = int(a // 1000), a % 1000
            o = O[j]; pc = L.PREVC[e, j] if np.isfinite(L.PREVC[e, j]) else o
            if lim and not (np.isfinite(FLAG[e, j]) and FLAG[e, j] == 1):
                up, dn = lim_prices(e, j)
                blocked = (side > 0 and o >= up - 1e-9) or (side < 0 and o <= dn + 1e-9)
                if log is not None and blocked:
                    log.append(dict(date=X.IDX[e], code=X.COLS[j], side=side, val=a * o, sleeve=(sleeve_codes is not None and X.COLS[j] in sleeve_codes.get(k, ()))))
                if blocked:
                    continue
            if side < 0 and E.DISP[e, j] and np.isfinite(L.Cn[e, j]):
                px = L.Cn[e, j] - float(tickv(L.Cn[e, j])); n = min(a, max(cur_sh[j], 0.0))
                for m in (lots * 1000, odd):
                    if m > 0:
                        val = m * px * (1 - extra); fee = max(MINFEE, RATE * val); tax = TAX * val
                        cash_close += val - fee - tax
                cur_sh[j] -= n
                continue
            vday = L.Vn[e, j] if np.isfinite(L.Vn[e, j]) and L.Vn[e, j] > 0 else E.V60[t, j]
            dec = int(min(max(math.ceil(E.VR[t, j] / 30), 1), 10)) if np.isfinite(E.VR[t, j]) else 10
            oa = E.SHARE.get(dec, E.SHARE[10]) * vday if np.isfinite(vday) else np.nan
            done = 0.0
            if lots:
                if np.isfinite(oa):
                    mx = max(1, int(math.floor(0.2 * oa / (1000 * pc))))
                    if lots > mx: lots = mx
                Q = lots * 1000 * pc
                sg = E.SIG[t, j]
                k5 = 0.001 + (0.5 * sg * math.sqrt(Q / oa) if (np.isfinite(sg) and np.isfinite(oa) and oa > 0) else 0.0)
                im = max(k5, float(tickv(o)) / o) + extra
                px = o * (1 + side * im); val = lots * 1000 * px
                fee = max(MINFEE, RATE * val); tax = TAX * val if side < 0 else 0.0
                cash_delta += -side * val - fee - tax
                done += side * lots * 1000
            if odd:
                pre = X.IDX[e] < E.ODD_INTRADAY
                if pre:
                    base = L.Cn[e, j]
                else:
                    g = o / pc - 1 if pc > 0 else 0.0
                    base = o * (1 + float(E.DR_V[int(np.searchsorted(E.DR_E, g, side='right'))]))
                pi = int(np.searchsorted(E.OPE, pc, side='right')); ri = int(np.searchsorted(E.ORE, E.VR[t, j] if np.isfinite(E.VR[t, j]) else 1e9, side='right'))
                c = float((E.OB if side > 0 else E.OS)[pi, ri]) + extra
                px = base * (1 + side * c); val = odd * px
                fee = max(MINFEE, RATE * val); tax = TAX * val if side < 0 else 0.0
                cash_delta += -side * val - fee - tax
                done += side * odd
                if pre and side > 0: odd_buy_pre[j] += odd
            cur_sh[j] += done
        cash = cash + cash_delta
        f2 = np.nan_to_num(TC.factor_open_to_close(e, t2), nan=1.0); fcc = TC.factor_close_to_close(t, t2)
        fcl = np.nan_to_num(TC.factor_close_to_close(e, t2), nan=1.0)
        v = np.where(tr, cur_sh * np.where(tr, O, 0.0) * f2, np.where(frz, v * fcc, 0.0))
        if oddfix:
            Cn = np.nan_to_num(L.Cn[e], nan=0.0)
            adj = odd_buy_pre * (np.nan_to_num(O) * f2 - Cn * fcl)
            v = v - np.where(tr, adj, 0.0)
        cash = cash + cash_close
        V1 = v.sum() + cash
        out.append(V1 / V0 - 1)
    return np.array(out)

def plans_for(spec, j, carrier):
    days = X.days_for(j, 'FULL')
    x = P[(P.spec == spec) & (P.j == j)]
    g = {s: d for s, d in x.groupby('signal')}
    pl = []; sl = {}
    for k, d in enumerate(days[:-1]):
        z = g[str(d.date())]
        idx = np.array([X.CI[c] for c in z.code]); w = z.weight.to_numpy(float)
        t = X.IDX.get_loc(d); mc = X.mcap_row(t)[idx]; wB = mc / mc.sum()
        ws = (w - 0.7 * wB) / 0.3; ws[ws < 1e-6] = 0.0
        sl[k] = set(z.code[ws > 0])
        if carrier == 'PURE':
            m = ws > 0; pl.append((idx[m], ws[m] / ws[m].sum()))
        else:
            pl.append((idx, w))
    return days, pl, sl

def summ(days, r):
    b = bench(days); ex = r - b; s = pd.Series(ex, index=pd.DatetimeIndex(days[:-1]))
    return {pk: round(tst(s[(s.index >= a) & (s.index <= bb)].to_numpy()), 2) for pk, (a, bb) in PER.items()}, s

print('\n== 2. 引擎變體（100 萬 GROW 一般成本；J0/J6/J12）==')
allog = []
for spec, carrier, key in (('SEL', 'SAT', 'SEL'), ('SEL', 'PURE', 'SEL_PURE'), ('CTRL_SEL', 'SAT', 'CTRL_SEL'), ('CTRL_SEL', 'PURE', 'CTRL_SEL_PURE')):
    for j in (0, 6, 12):
        days, pl, sl = plans_for(spec, j, carrier)
        r0 = run_exec2(days, pl, 1e6, 'GROW')
        ref = W[f'exec|{key}|J{j}|100萬|GROW|normal'].dropna().to_numpy()
        rep = float(np.nanmax(np.abs(r0 - ref)))
        orig = E.run_exec(days, pl, 1e6, 'GROW')[0]
        rep2 = float(np.nanmax(np.abs(orig - ref)))
        lg = []
        rl = run_exec2(days, pl, 1e6, 'GROW', lim=True, sleeve_codes=sl, log=lg)
        ro = run_exec2(days, pl, 1e6, 'GROW', oddfix=True)
        rb = run_exec2(days, pl, 1e6, 'GROW', lim=True, oddfix=True)
        rs = run_exec2(days, pl, 1e6, 'GROW', extra=0.005, lim=True, oddfix=True)
        a0, s0 = summ(days, r0); al, sl_ = summ(days, rl); ao, so = summ(days, ro); ab, sb = summ(days, rb); as_, ss = summ(days, rs)
        pre = s0.index < E.ODD_INTRADAY
        print(f'{key} J{j}: 複製誤差 copy={rep:.2e} orig={rep2:.2e}')
        print('   base', a0, '\n   LIM ', al, '\n   ODD ', ao, '\n   BOTH', ab, '\n   BOTH+stress', as_)
        print(f'   ODD 修正月均差（2020-10 前）{100*(so[pre]-s0[pre]).mean():+.3f}pp；LIM 月均差 {100*(sl_-s0).mean():+.3f}pp；漲跌停擋單 {len(lg)} 筆，其中 sleeve {sum(x["sleeve"] for x in lg)}')
        for x in lg: x.update(spec=key, j=j)
        allog += lg
lg = pd.DataFrame(allog)
if len(lg):
    print('\n漲跌停擋單明細（sleeve 名單，去重）：')
    q = lg[lg.sleeve].drop_duplicates(['date', 'code', 'side'])
    print(q.sort_values('date').to_string(max_rows=80))

print('\n== 3. sleeve 新進名單在 e 日處於處置期（預收款券、分盤撮合；原引擎買進照開盤成交）==')
for spec in ('SEL', 'CTRL_SEL'):
    for j in (0, 6, 12):
        days, pl, sl = plans_for(spec, j, 'SAT')
        n_new = n_disp = 0; ex = []
        prev = set()
        for k, d in enumerate(days[:-1]):
            e = X.IDX.get_loc(d) + 1; new = sl[k] - prev
            for c in new:
                n_new += 1
                if E.DISP[e, X.CI[c]]: n_disp += 1; ex.append((str(X.IDX[e].date()), c))
            prev = sl[k]
        print(spec, f'J{j}', f'新進 {n_new} 檔次，其中 e 日處置中 {n_disp}', ex[:12])
