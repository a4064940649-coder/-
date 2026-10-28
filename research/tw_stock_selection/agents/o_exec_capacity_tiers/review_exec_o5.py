#!/usr/bin/env python3
"""反方審查（成本與可成交）：o_exec_capacity_tiers。只 import 研究員的 ecore.py（不改檔），結果印到 stdout。
檢查研究員凍結規格沒有建模、但實盤一定遇到的摩擦，對「可行但打折」的格子（30 萬 GROW/FIX、100 萬 FIX）與 300 萬 FIX：
  LIM  ：開盤價在漲停（買）／跌停（賣）時，該筆整股與零股都不成交，順延到下一個交易日（每日重試）。
         漲跌幅 2015-06-01 前 7%、之後 10%；參考價＝C_t/G_t（含除權息調整），依升降單位取整。
  LOCK ：較寬鬆版，只在開盤與收盤都在漲／跌停（鎖死整天）才不成交。
  ODD0 ：零股賣成本下限 0（研究員表內賣成本為負＝賣零股比整股還多拿錢）。
  CASH2：每次換股保留 2% 現金（實盤不知道開盤價，需緩衝避免違約交割）。
  FEEMIX：2015 年以前用手續費 B（0.6 折），之後 A（研究員限制 5 自己承認 2.8 折早年不普遍）。
  ALL  ：LIM＋ODD0＋CASH2＋FEEMIX。
用法：python3 -I -B review_exec_o5.py
"""
import sys, json, math, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import ecore as E

MODEL = json.loads((E.DATA / 'model_spec.json').read_text(encoding='utf-8'))
SEEDS = list(range(40))
CELLS = [('30萬', 'GROW'), ('30萬', 'FIX'), ('100萬', 'FIX'), ('300萬', 'FIX')]
VARS = {'base': {}, 'LIM': dict(lim='open'), 'LOCK': dict(lim='lock'), 'ODD0': dict(odd0=True),
        'CASH2': dict(cash=0.02), 'FEEMIX': dict(feemix=True),
        'ALL': dict(lim='open', odd0=True, cash=0.02, feemix=True)}
CUR = {'t': None, 'opt': {}, 'blk_buy': 0, 'blk_sell': 0, 'blk_buy_val': 0.0, 'blk_sell_val': 0.0}

def tick(p):
    return 0.01 if p < 10 else 0.05 if p < 50 else 0.1 if p < 100 else 0.5 if p < 500 else 1.0 if p < 1000 else 5.0

def limits(D):
    ref = np.where(D.TR & np.isfinite(D.G) & (D.G > 0), D.C / D.G, D.PC)
    lim = np.where((D.IDX >= pd.Timestamp('2015-06-01'))[:, None], 0.10, 0.07)
    up_raw = ref * (1 + lim); dn_raw = ref * (1 - lim)
    vt = np.vectorize(tick)
    with np.errstate(invalid='ignore'):
        tu = vt(np.nan_to_num(up_raw, nan=1.0)); td = vt(np.nan_to_num(dn_raw, nan=1.0))
        up = np.floor(up_raw / tu + 1e-6) * tu; dn = np.ceil(dn_raw / td - 1e-6) * td
    return up, dn

def blocked(D, t, j, side):
    m = CUR['opt'].get('lim')
    if not m: return False
    o, c = D.O[t, j], D.C[t, j]
    if side > 0:
        b = o >= D.LU[t, j] - 1e-6 and (m == 'open' or c >= D.LU[t, j] - 1e-6)
    else:
        b = o <= D.LD[t, j] + 1e-6 and (m == 'open' or c <= D.LD[t, j] + 1e-6)
    return bool(b)

_orig_fee = E._fee
def _fee(value, fr, odd, fl):
    if CUR['opt'].get('feemix') and CUR['t'] is not None and CUR['D'].IDX[CUR['t']] < pd.Timestamp('2015-01-01') and fl['fees']:
        fr = E.FEES['B']
    return _orig_fee(value, fr, odd, fl)

class M0(E.ExecModel):
    def odd_cost(self, price, rank, side):
        c = super().odd_cost(price, rank, side)
        return max(c, 0.0) if (side < 0 and CUR['opt'].get('odd0')) else c

def _rebalance(D, t, pk, S, tr, frz, cash, fl, fr, model, acc, keep, rows, sg):
    CUR['t'] = t
    pka = list(pk)
    trade_ok = tr & ~frz
    cur = {j: int(round(S[j])) for j in np.where(trade_ok & (S > 0))[0]}
    Vtr = sum(S[j] * D.O[t, j] for j in cur) + cash
    k = len(pka)
    def deltas(T):
        out = {}
        for j in set(cur) | set(pka):
            tgt = E._plan_shares(T, D.PC[t, j]) if j in pka else 0
            d = tgt - cur.get(j, 0)
            if d != 0: out[j] = d
        return out
    ctx = E._Ctx(D, t, fl, fr, model, pka)
    def need(T): return sum(ctx.cost(j, d) for j, d in deltas(T).items())
    hi = Vtr / k * (1 - CUR['opt'].get('cash', 0.0)); lo = 0.0
    if need(hi) <= cash: lo = hi
    else:
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if need(mid) <= cash: lo = mid
            else: hi = mid
    dl = deltas(lo); pend = {}
    for j, d in sorted(dl.items(), key=lambda x: x[1]):
        side = 1 if d > 0 else -1; a = abs(d)
        if blocked(D, t, j, side):
            pend[j] = pend.get(j, 0) + d
            CUR['blk_buy' if side > 0 else 'blk_sell'] += 1
            CUR['blk_buy_val' if side > 0 else 'blk_sell_val'] += a * D.O[t, j] / max(acc['_eq'], 1.0)
            continue
        lots, odd = a // 1000, a % 1000
        if side > 0 and ctx.cost(j, d) > cash + 1e-9:
            pend[j] = pend.get(j, 0) + d; acc['deferred_days'] += 1; continue
        if odd:
            cash += E._odd_exec(D, t, j, odd, side, fl, fr, model, acc, rows, 'reb'); S[j] += side * odd
        if lots:
            n, dc = E._lot_exec(D, t, j, lots, side, fl, fr, model, acc, t - 1, rows, 'reb', 'pick' if j in pka else 'drop')
            cash += dc; S[j] += side * n * 1000
            if n < lots: pend[j] = pend.get(j, 0) + side * (lots - n) * 1000
    return S, cash, pend

def _work_pending(D, t, S, cash, pend, fl, fr, model, acc, rows):
    CUR['t'] = t
    newp = {}
    for j, d in sorted(pend.items(), key=lambda x: x[1]):
        if not (D.TR[t, j] and np.isfinite(D.O[t, j])):
            newp[j] = d; continue
        side = 1 if d > 0 else -1
        if blocked(D, t, j, side):
            newp[j] = d; continue
        a = abs(d)
        if side < 0: a = min(a, int(round(S[j])))
        lots, odd = a // 1000, a % 1000
        acc['deferred_days'] += 1
        if side > 0:
            unit = D.O[t, j] * 1000.0 * 1.05
            lots = min(lots, int(cash // unit)) if unit > 0 else 0
            if odd and E._est_cost(D, t, j, odd, fl, fr, model) > cash - lots * unit: odd = 0
        if odd:
            cash += E._odd_exec(D, t, j, odd, side, fl, fr, model, acc, rows, 'pend'); S[j] += side * odd
        n = 0
        if lots:
            n, dc = E._lot_exec(D, t, j, lots, side, fl, fr, model, acc, t - 1, rows, 'pend')
            cash += dc; S[j] += side * n * 1000
        rem = a - odd - n * 1000
        if side > 0: rem = abs(d) - odd - n * 1000
        if rem > 0: newp[j] = side * rem
    return S, cash, newp

_W = {}
def _init():
    D = E.Data(); D.LU, D.LD = limits(D); _W['D'] = D; CUR['D'] = D
    E._fee = _fee; E._rebalance = _rebalance; E._work_pending = _work_pending

def job(a):
    var, tier, ver, seed = a
    D = _W['D']
    CUR.update(opt=VARS[var], t=None, blk_buy=0, blk_sell=0, blk_buy_val=0.0, blk_sell_val=0.0)
    nav, info = E.simulate(D, seed, tier=E.TIERS[tier], version=ver, fee='A', model=M0(MODEL))
    return a, nav.to_numpy(), dict(bb=CUR['blk_buy'], bs=CUR['blk_sell'], bbv=CUR['blk_buy_val'], bsv=CUR['blk_sell_val'],
                                   cash=info['acc']['cash_frac_sum'])

if __name__ == '__main__':
    T0 = time.time()
    D = E.Data(); D.LU, D.LD = limits(D)
    IDXN = D.IDX[D.t0:D.t1 + 1]
    # ---- 1) 名單層級：換股日新進股開盤漲停、剔除股開盤跌停的頻率（seed 0～39）----
    P = pd.read_csv(E.DATA / 'plans_ab.csv', dtype={'picks': str})
    nb = nlu = nlk = ns = nld = 0; ret_lu = []; ret_all = []
    for s, g in P.groupby('seed'):
        prev = set()
        ex = list(g.exec_idx)
        for i, (e, pk) in enumerate(zip(g.exec_idx, g.picks)):
            cur = set(D.CI[c] for c in pk.split(','))
            e2 = ex[i + 1] if i + 1 < len(ex) else D.t1
            for j in cur - prev:
                if not (D.TR[e, j] and np.isfinite(D.O[e, j])): continue
                nb += 1
                r = D.C[e, j] / D.O[e, j] * np.prod(np.where(np.isfinite(D.G[e + 1:e2, j]), D.G[e + 1:e2, j], 1.0)) - 1
                ret_all.append(r)
                if D.O[e, j] >= D.LU[e, j] - 1e-6:
                    nlu += 1; ret_lu.append(r)
                    if D.C[e, j] >= D.LU[e, j] - 1e-6: nlk += 1
            for j in prev - cur:
                if not (D.TR[e, j] and np.isfinite(D.O[e, j])): continue
                ns += 1
                if D.O[e, j] <= D.LD[e, j] + 1e-6: nld += 1
            prev = cur
    print('== 名單層級（40 seeds）==')
    print(f'新進買單 {nb}，開盤漲停 {nlu}（{nlu / nb * 100:.2f}%），開收都漲停 {nlk}；剔除賣單 {ns}，開盤跌停 {nld}（{nld / ns * 100:.2f}%）')
    print(f'開盤漲停新進股 持有到下次換股前收盤 平均報酬 {np.mean(ret_lu) * 100:.2f}%（中位 {np.median(ret_lu) * 100:.2f}%），全部新進股平均 {np.mean(ret_all) * 100:.2f}%')
    sys.stdout.flush()
    # ---- 2) 模擬 ----
    jobs = [(v, ti, ve, s) for v in VARS for ti, ve in CELLS for s in SEEDS]
    OUT = {}
    with ProcessPoolExecutor(max_workers=4, initializer=_init) as ex:
        for i, (a, nav, st) in enumerate(ex.map(job, jobs, chunksize=4)):
            OUT[a] = (nav, st)
            if i % 200 == 0: print(f'[{time.time() - T0:5.0f}s] {i}/{len(jobs)}', flush=True)
    E50 = D.E50.reindex(IDXN); sd = D.sig_days
    res = {}
    print('== 模擬（手續費 A 為底，40 seed 平均日報酬淨值；t＝對 0050 換股窗超額）==')
    print('cell | var | CAGR | t | t_FROM2008 | t_HOLD | blk_buy/seed | blk_sell/seed | blk_buy 金額占權益(累計/seed)')
    for ti, ve in CELLS:
        for v in VARS:
            R = pd.DataFrame({s: pd.Series(OUT[(v, ti, ve, s)][0], index=IDXN).pct_change() for s in SEEDS})
            nav = (1 + R.mean(axis=1).fillna(0)).cumprod()
            cg = E.cagr_mdd(nav, E.A_, E.B_)[0]
            t = E.tstat(E.excess(nav, E50, sd)); t8 = E.tstat(E.excess(nav, E50, sd, *E.PER['FROM2008']))
            th = E.tstat(E.excess(nav, E50, sd, *E.PER['HOLD']))
            st = [OUT[(v, ti, ve, s)][1] for s in SEEDS]
            bb = np.mean([x['bb'] for x in st]); bs = np.mean([x['bs'] for x in st]); bbv = np.mean([x['bbv'] for x in st])
            res[f'{ti}|{ve}|{v}'] = dict(CAGR=cg, t=t, t2008=t8, tHOLD=th, blk_buy=bb, blk_sell=bs)
            print(f'{ti} {ve} | {v:6s} | {cg:6.2f} | {t:5.2f} | {t8:5.2f} | {th:5.2f} | {bb:6.1f} | {bs:6.1f} | {bbv:5.2f}')
    print(json.dumps(res, ensure_ascii=False))
    print(f'done {time.time() - T0:.0f}s')
