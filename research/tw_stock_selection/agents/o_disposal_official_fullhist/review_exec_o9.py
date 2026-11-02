#!/usr/bin/env python3
"""反方審查（成本與可成交，o9）：只讀研究員的程式與產出，不改任何檔、不寫任何檔。python3 -I -B review_exec_o9.py
檢查：
 A. 整張（1,000 股）可成交性：小帳戶每槽金額買不到 1 張的比例（2020-10-26 前沒有盤中零股）。
 B. 嚴格預收款券＋T+2 交割：需預收的事件（prepay_only 期全部、20 分鐘撮合的第二次處置）只能用已交割現金買；
    現金池固定 B 槽（0 報酬），不足就縮單或跳過；出場款 T+2 回到現金池，超出部分轉 0050。
 C. 事件群聚：3 個交易日內新進場筆數（現金池需要幾槽）。
 D. 周轉與手動操作次數。
 E. 只看 OOS 段的衛星 t（不含 disposal_v2 已看過的 2021-10 以後）。"""
import sys, math, json
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import core as K
from core import IDX, T, LA, E50, C, V, R, TR, LOGCO, LASTV, ETF_BUY, ETF_SELL, TAX, qref, side_cost, cpi_factor, limit_rate

OUT = {}
tr = pd.read_csv(HERE / 'trades_all.csv.gz', parse_dates=['date', 'exit_date'], dtype={'code': str})
O = tr[tr.seg == 'OOS'].copy()

# ---------- A. 整張可成交性 ----------
ODD = pd.Timestamp('2020-10-26')
A = {}
for acct in (3e5, 1e6, 3e6):
    for nm, d in (('OOS', O), ('FULL06+', tr[tr.date >= '2006-04-01'])):
        slot = acct / 10 * np.array([cpi_factor(int(i)) for i in d.i0])
        lot = d.px.to_numpy() * 1000
        nlots = np.floor(slot / lot)
        pre = (d.date < ODD).to_numpy()
        alloc = np.where(pre, nlots * lot / slot, 1.0)          # 2020-10-26 後可盤中零股（假設可全額）
        zero = pre & (nlots == 0)
        sub = d[~zero]
        A[f'{int(acct/1e4)}萬|{nm}'] = {
            'n': int(len(d)), 'share_slot_lt_1lot': round(float((slot < lot).mean()), 3),
            'share_unbuyable_wholelot_pre2020-10': round(float(zero.mean()), 3),
            'mean_alloc_frac_pre2020-10': round(float(alloc[pre].mean()), 3) if pre.any() else None,
            'net_x_all': K.S(d.assign(date=d.date), 'net_x'), 'net_x_buyable': K.S(sub, 'net_x'),
            'net_x_unbuyable_mean_pct': round(float(d[zero].net_x.mean() * 100), 2) if zero.any() else None}
OUT['A_lot'] = A

# ---------- C. 事件群聚 ----------
U = K.universe()
B0, _ = K.build({}, U)
B0 = B0[B0.date >= '2006-04-01']
need = (B0.regime == 'prepay_only') | (B0.match_min.fillna(0) >= 20)
cnt = {}
for nm, d in (('all', B0), ('need_prepay', B0[need])):
    s = pd.Series(1, index=d.i0).groupby(level=0).sum().reindex(range(T), fill_value=0)
    w3 = s.rolling(3).sum()
    cnt[nm] = {'n': int(len(d)), 'max_same_day': int(s.max()), 'max_in_3d': int(w3.max()),
               'days_with_ge3_in_3d': int((w3 >= 3).sum()), 'per_year': round(len(d) / 20.55, 1)}
cnt['need_prepay_share_OOS'] = round(float(need[(B0.date <= '2021-09-30')].mean()), 3)
OUT['C_cluster'] = cnt


# ---------- B. 嚴格預收款券＋T+2 ----------
def sleeve_strict(trades, tier=1e6, K_=10, cap=0.2, stress=False, B=2, need_all=False, t_start='2006-04-01'):
    """以 core_fix.sleeve_fix（FIX、每月縮放）為底；需預收的事件只能用已交割現金。"""
    t0 = int(IDX.searchsorted(pd.Timestamp(t_start))); t1 = T - 1
    ev = {}
    for r in trades.itertuples():
        if t0 + 1 <= r.i0 <= t1: ev.setdefault(r.i0 - 1, []).append(r)
    cap0 = tier * cpi_factor(t0); slot0 = cap0 / K_
    cash = min(B * slot0, cap0); u50 = (cap0 - cash) * (1 - ETF_BUY) / E50[t0 - 1]
    pend = []   # (settle_t, amt)：T+2 才能用的現金
    pos = {}; resv = []; nav = np.empty(t1 - t0 + 1); inv = np.empty(t1 - t0 + 1)
    navi = 1.0; prevE = cap0
    st = dict(entered=0, skipped_nocash=0, partial_cash=0, no_slot=0)

    def needs(r):
        return need_all or r.regime == 'prepay_only' or (np.isfinite(r.match_min) and r.match_min >= 20)

    def equity(t):
        pv = sum(p['units'] * math.exp(LA[t, p['j']] - LA[p['i0'], p['j']]) for p in pos.values())
        return cash + sum(a for _, a in pend) + u50 * E50[t] + pv, pv

    for t in range(t0, t1 + 1):
        # 交割入帳
        keep = []
        for st_, a in pend:
            if st_ <= t: cash += a
            else: keep.append((st_, a))
        pend = keep
        newresv = []
        for r, amt, fromcash in resv:
            if r.i0 != t: newresv.append((r, amt, fromcash)); continue
            j = r.j; Q0 = qref(t, j, r.n_auct, r.s_i <= t <= r.e_i)
            q = min(amt, cap * Q0) if Q0 > 0 else 0.0
            if q <= 0:
                if fromcash: cash += amt
                else: u50 += amt * (1 - ETF_BUY) / E50[t]
                continue
            cb = side_cost(t, j, q, Q0, +1, stress)
            pos[(r.code, r.ordr)] = dict(j=j, i0=t, units=q * (1 - cb), r=r)
            if amt - q > 0:
                if fromcash: cash += amt - q
                else: u50 += (amt - q) * (1 - ETF_BUY) / E50[t]
            st['entered'] += 1
        resv = newresv
        for key in list(pos.keys()):
            p = pos[key]; r = p['r']; j = p['j']
            if t < r.i1: continue
            lim = min(T - 1, LASTV[j])
            sellable = t >= lim or t == r.i1 or (TR[t, j] and np.isfinite(R[t, j]) and R[t, j] > -(limit_rate(t) - 0.01))
            if not sellable: continue
            val = p['units'] * math.exp(LA[t, j] - LA[p['i0'], j])
            Q1 = qref(t, j, r.n_auct, r.s_i <= t <= r.e_i)
            sell = val if t >= lim else (min(val, cap * Q1) if Q1 > 0 else 0.0)
            if sell <= 0: continue
            c1 = side_cost(t, j, sell, Q1 if Q1 > 0 else sell, -1, stress) + TAX
            pr = sell * (1 - c1)   # 賣出與買 0050 同為 T+2，可同日換回 0050；只有現金池缺口部分留成現金（T+2 入帳）
            gap = max(0.0, B * tier * cpi_factor(t) / K_ - cash - sum(a for _, a in pend))
            if gap > 0: pend.append((t + 2, min(gap, pr)))
            if pr - gap > 0: u50 += (pr - gap) * (1 - ETF_BUY) / E50[t]
            if sell / val >= 1 - 1e-9: pos.pop(key)
            else: p['units'] *= (1 - sell / val)
        # 現金池調整：超過 B 槽的已交割現金轉 0050；不足 B 槽 → 今日賣 0050（T+2 入帳）
        E, pv = equity(t)
        slot = tier * cpi_factor(t) / K_
        tgt = B * slot
        pend_tot = sum(a for _, a in pend)
        if cash > tgt:
            u50 += (cash - tgt) * (1 - ETF_BUY) / E50[t]; cash = tgt
        elif cash + pend_tot < tgt and u50 > 0:
            need_amt = min(tgt - cash - pend_tot, u50 * E50[t] * (1 - ETF_SELL))
            u50 -= need_amt / (1 - ETF_SELL) / E50[t]; pend.append((t + 2, need_amt))
        # 今晚看到公告 → 明天（i0）的委託
        for r in ev.get(t, []):
            if any(pp['r'].code == r.code for pp in pos.values()) or any(rr.code == r.code for rr, _, _ in resv): continue
            if len(pos) + len(resv) >= K_: st['no_slot'] += 1; continue
            if needs(r):
                usable = cash + sum(a for s_, a in pend if s_ <= t + 1)   # i0 早上前已交割
                amt = min(slot, usable)
                if amt <= 1: st['skipped_nocash'] += 1; continue
                if amt < slot: st['partial_cash'] += 1
                take = min(cash, amt); cash -= take; rest = amt - take
                if rest > 0:   # 動用 t+1 交割的款
                    nk = []
                    for s_, a in sorted(pend):
                        if rest > 0 and s_ <= t + 1:
                            u = min(a, rest); rest -= u
                            if a - u > 0: nk.append((s_, a - u))
                        else: nk.append((s_, a))
                    pend = nk
                resv.append((r, amt, True))
            else:
                avail = u50 * E50[t] * (1 - ETF_SELL); amt = min(slot, avail)
                if amt <= 0: continue
                u50 -= amt / (1 - ETF_SELL) / E50[t]; resv.append((r, amt, False))
        E, pv = equity(t); E += sum(a for _, a, _ in resv)
        navi *= E / prevE; prevE = E
        nav[t - t0] = navi; inv[t - t0] = pv / E if E > 0 else 0
        if t < t1 and IDX[t + 1].month != IDX[t].month and E > 0:
            sc = tier * cpi_factor(t + 1) / E
            u50 *= sc; cash *= sc; pend = [(s_, a * sc) for s_, a in pend]
            for p in pos.values(): p['units'] *= sc
            resv = [(rr, aa * sc, ff) for rr, aa, ff in resv]; prevE = E * sc
    s = pd.Series(nav, index=IDX[t0:t1 + 1]); st['mean_invested'] = round(float(inv.mean()), 3)
    return s, st


def rep(nav):
    return {'FULL': K.vs0050(nav), 'OOS': K.vs0050(nav, '2006-04-01', '2021-09-30'), '2019+': K.vs0050(nav, '2019-01-01'),
            'SEG_2006_2012': K.vs0050(nav, '2006-04-01', '2012-12-31')}


Bres = {}
B_all, _ = K.build({}, U)
for Bn in (0, 1, 2, 3):
    for na in (False, True):
        for stress in (False, True):
            if Bn == 0 and na: continue
            nav, st = sleeve_strict(B_all, B=Bn, need_all=na, stress=stress)
            r = rep(nav)
            Bres[f'B{Bn}|{"all_need" if na else "prepay+20min_need"}|{"stress" if stress else "normal"}'] = {
                'stat': st, **{k: {kk: v.get(kk) for kk in ('ex_month_mean_pp', 't_vs_0050', 'ann', 'ann_0050')} for k, v in r.items()}}
OUT['B_strict_prepay'] = Bres

# ---------- D. 周轉與操作次數 ----------
F06 = tr[tr.date >= '2006-04-01']
OUT['D_ops'] = {'events_per_year_full': round(len(F06) / 20.55, 1),
                'events_per_year_2019+': round(len(F06[F06.date >= '2019-01-01']) / 7.8, 1),
                'close_auction_actions_per_year_C1xCL': round(2 * len(F06) / 20.55, 1),
                'mean_hold_days': round(float(F06.hold.mean()), 2),
                'cost_x_mean_pct_OOS': round(float(O.cost_x.mean() * 100), 2),
                'gross_mean_pct_OOS': round(float(O.gross.mean() * 100), 2),
                'cost_over_gross_OOS': round(float(O.cost_x.mean() / O.gross.mean()), 2)}

# ---------- E. 只看 OOS 段 ----------
res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
SLV = res['POSTHOC_FIX']['sleeves']
OUT['E_oos_only_sleeve_t'] = {k: SLV[k]['OOS'].get('t_vs_0050') for k in SLV if '100萬|FIX' in k and ('C1xCL|' in k or 'O2xOA' in k)}
e = O.copy()
OUT['E_oos_exec_by_era'] = {'2006-04_2013': K.S(e[e.date < '2014-01-01'], 'net_x'), '2014_2018': K.S(e[(e.date >= '2014-01-01') & (e.date < '2019-01-01')], 'net_x'),
                            '2019_2021-09': K.S(e[e.date >= '2019-01-01'], 'net_x'),
                            '2006-04_2019-05-15_stress': K.S(e[e.date < '2019-05-16'], 'net_x_stress'),
                            '2006-04_2019-05-15_exec': K.S(e[e.date < '2019-05-16'], 'net_x')}
print(json.dumps(OUT, ensure_ascii=False, indent=1, default=str))

# ---------- F. 「總帳戶」級距：衛星＝總資金 20% → 每槽金額小很多（30 萬總帳戶＝每槽 6,000 元） ----------
Fres = {}
BO = B_all[(B_all.date >= '2006-04-01') & (B_all.date <= '2021-09-30')]
for tot in (3e5, 1e6, 5e6):
    slot = tot * 0.2 / 10
    x = K.exec_cost_event(BO, slot_2026=slot)
    xs = K.exec_cost_event(BO, slot_2026=slot, stress=True)
    sl = slot * np.array([cpi_factor(int(i)) for i in BO.i0]); lot = BO.px.to_numpy() * 1000
    Fres[f'總帳戶{int(tot/1e4)}萬_每槽{int(slot)}'] = {
        'OOS_net_x': K.S(x, 'net_x'), 'OOS_net_x_stress': K.S(xs.assign(net_x=xs.net_x), 'net_x'),
        'mean_cost_x_pct': round(float(x.cost_x.mean() * 100), 2),
        'share_slot_lt_1lot': round(float((sl < lot).mean()), 3)}
OUT['F_total_account'] = Fres
print(json.dumps(OUT['F_total_account'], ensure_ascii=False, default=str))
