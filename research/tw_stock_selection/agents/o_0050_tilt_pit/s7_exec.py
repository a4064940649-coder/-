#!/usr/bin/env python3
"""可執行版（PREREG 凍結後才可執行）：K5T 衝擊（V7 model_spec7：每邊 max(0.1%＋0.5·σ60·√(Q/OA), 一跳/開盤價)，OA＝成交值十分位開盤占比×當日成交值，
參與上限 20%×OA，超過的張數本月不成交）；零股（整股 1,000 股以下的部分）用 spec 的零股價差表＋開盤跳空漂移；2020-10-26 前沒有盤中零股 → 以收盤價（盤後零股）成交並標示「不可執行」。
手續費 0.1425%×0.28（V7 費率 A），每筆最低 20 元；證交稅 0.3%；FIX 每月把帳戶縮放回固定金額（CPI 調整），縮放的買賣也付費、稅與 0.1% 衝擊（修正 V7 的偏樂觀設定）。
不交易帶：|目標−現有| < 0.25% 權益的調整不下單（出場與宇宙外持股一律賣出）。級距：30 萬、100 萬、300 萬、1,000 萬 FIX，1,000 萬 GROW。
期間：POST2019（訊號窗 2019-01～2026-10），另列 2014～2018。寫 results.json['exec']。"""
import sys, json, math, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import tcore as T, tlib as L
PR = L.DATA / 'PREREG.json'
assert PR.exists(), 'PREREG 不存在'
PRE = json.loads(PR.read_text(encoding='utf-8'))
for f, h in PRE['code_sha256'].items(): assert L.sha(L.HERE / f) == h, f
SPEC = json.loads((L.AG / 'o_v7_listing_exec_engine' / 'data' / 'model_spec7.json').read_text(encoding='utf-8'))
CPI = {int(k): v for k, v in json.loads((L.AG / 'o_v7_listing_exec_engine' / 'raw' / 'exec_inputs' / 'cpi_tw.json').read_text(encoding='utf-8'))['index_2006_100'].items()}
def cpi_scale(ts): return CPI[min(max(ts.year, 2006), 2026)] / CPI[2026]
RATE = 0.001425 * 0.28; MINFEE = 20.0; TAX = 0.003
ODD_INTRADAY = pd.Timestamp('2020-10-26')
SHARE = {int(k): float(v) for k, v in SPEC['oa_share_by_dec'].items()}
DR_E, DR_V = np.array(SPEC['drift']['edges']), np.array(SPEC['drift']['vals'])
OPE, ORE = np.array(SPEC['odd']['price_edges']), np.array(SPEC['odd']['rank_edges'])
OB, OS = np.array(SPEC['odd']['buy']), np.array(SPEC['odd']['sell'])
BAND = 0.0025
V60 = T.V60; SIG = T.SIG
with np.errstate(invalid='ignore'):
    VR = pd.DataFrame(V60).rank(axis=1, ascending=False).to_numpy()
def tickv(p): return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)
def run_exec(days, plans, tier, mode):
    ti = [L.IDX.get_loc(d) for d in days]
    v = np.zeros(L.N); cash = tier * cpi_scale(days[0]); out = []; acc = dict(fee=0., tax=0., impact=0., odd=0., odd_val=0., lot_val=0., cap_hit=0, orders=0, flow_cost=0., pre_odd_val=0.)
    for k in range(len(ti) - 1):
        t, t2 = ti[k], ti[k + 1]; e = t + 1; d = days[k]
        V0 = v.sum() + cash
        fo = T.factor_close_to_open(t, e); tr = np.isfinite(fo)
        dead = (v > 0) & (L.LAST_TRADE < e)
        cash += (v[dead] * (1 - RATE - TAX - 0.001)).sum(); v[dead] = 0.0
        v = np.where(tr, v * np.nan_to_num(fo, nan=1.0), v)
        frz = (v > 0) & ~tr
        Eq = v.sum() + cash; flow_cost = 0.0; r_on = Eq / V0
        if mode == 'FIX':
            tgtE = tier * cpi_scale(L.IDX[e]); fl = tgtE - Eq
            flow_cost = abs(fl) * ((RATE + TAX + 0.001) if fl < 0 else (RATE + 0.001))
            sc = tgtE / Eq; v = v * sc; cash = cash * sc; Eq = tgtE; acc['flow_cost'] += flow_cost
        idx, w = plans[d]
        wv = np.zeros(L.N); np.add.at(wv, idx, w); wv[frz] = 0.0; wv[~tr] = 0.0; wv = wv / wv.sum()
        Etr = Eq - v[frz].sum()
        O = L.On[e]; cur_sh = np.where(tr, v / np.where(tr, O, 1.0), 0.0)
        tgt_sh = np.where(tr, np.round(wv * Etr * 0.995 / np.where(tr, O, 1.0)), 0.0)   # 0.5% 現金緩衝付成本
        dsh = tgt_sh - cur_sh
        small = (np.abs(dsh * np.nan_to_num(O)) < BAND * Eq) & (tgt_sh > 0)
        dsh[small] = 0.0
        dsh = np.where(tr, np.round(dsh), 0.0)
        cash_delta = 0.0
        for j in np.where(dsh != 0)[0]:
            q = dsh[j]; side = 1 if q > 0 else -1; a = abs(q); lots, odd = int(a // 1000), a % 1000
            o = O[j]; pc = L.PREVC[e, j] if np.isfinite(L.PREVC[e, j]) else o
            vday = L.Vn[e, j] if np.isfinite(L.Vn[e, j]) and L.Vn[e, j] > 0 else V60[t, j]
            dec = int(min(max(math.ceil(VR[t, j] / 30), 1), 10)) if np.isfinite(VR[t, j]) else 10
            oa = SHARE.get(dec, SHARE[10]) * vday if np.isfinite(vday) else np.nan
            done = 0.0
            if lots:
                if np.isfinite(oa):
                    mx = max(1, int(math.floor(0.2 * oa / (1000 * pc))))
                    if lots > mx: lots = mx; acc['cap_hit'] += 1
                Q = lots * 1000 * pc
                sg = SIG[t, j]
                k5 = 0.001 + (0.5 * sg * math.sqrt(Q / oa) if (np.isfinite(sg) and np.isfinite(oa) and oa > 0) else 0.0)
                im = max(k5, float(tickv(o)) / o)
                px = o * (1 + side * im); val = lots * 1000 * px
                fee = max(MINFEE, RATE * val); tax = TAX * val if side < 0 else 0.0
                cash_delta += -side * val - fee - tax; acc['fee'] += fee; acc['tax'] += tax; acc['impact'] += lots * 1000 * o * im; acc['lot_val'] += lots * 1000 * o; acc['orders'] += 1
                done += side * lots * 1000
            if odd:
                if L.IDX[e] < ODD_INTRADAY:
                    base = L.Cn[e, j]; acc['pre_odd_val'] += odd * base
                else:
                    g = o / pc - 1 if pc > 0 else 0.0
                    base = o * (1 + float(DR_V[int(np.searchsorted(DR_E, g, side='right'))]))
                pi = int(np.searchsorted(OPE, pc, side='right')); ri = int(np.searchsorted(ORE, VR[t, j] if np.isfinite(VR[t, j]) else 1e9, side='right'))
                c = float((OB if side > 0 else OS)[pi, ri])
                px = base * (1 + side * c); val = odd * px
                fee = max(MINFEE, RATE * val); tax = TAX * val if side < 0 else 0.0
                cash_delta += -side * val - fee - tax; acc['fee'] += fee; acc['tax'] += tax; acc['odd'] += odd * base * c; acc['odd_val'] += odd * base; acc['orders'] += 1
                done += side * odd
            cur_sh[j] += done
        cash = cash + cash_delta
        # 成交後：可交易持股以開盤價計值，之後 × 開盤→收盤因子；凍結持股 × 收盤→收盤
        f2 = np.nan_to_num(T.factor_open_to_close(e, t2), nan=1.0); fcc = T.factor_close_to_close(t, t2)
        v = np.where(tr, cur_sh * np.where(tr, O, 0.0) * f2, np.where(frz, v * fcc, 0.0))
        V1 = v.sum() + cash
        out.append(V1 / V0 - 1 if mode == 'GROW' else r_on * (V1 - flow_cost) / Eq - 1)
    return pd.Series(out, index=pd.DatetimeIndex(days[:-1])), acc
TIERS = [('30萬', 3e5, 'FIX'), ('100萬', 1e6, 'FIX'), ('300萬', 3e6, 'FIX'), ('1000萬', 1e7, 'FIX'), ('1000萬', 1e7, 'GROW')]
OUT = {}
for vname, (kind, lam, cap) in {'T1': ('T1', 1.5, False), 'T2': ('T2', 1.5, False)}.items():
    for pname, (a, z) in {'POST2019': ('2019-01-01', '2026-10-31'), 'SEL': ('2014-01-01', '2018-12-31')}.items():
        for tn, tier, mode in TIERS:
            per = []
            for j in range(13):
                ds = T.sig_days(j, a, z)
                pre = T.precompute(ds, kind); PT = {}; PB = {}
                for d, U, wb, tz in pre:
                    wt = wb.copy()
                    for nm, (idx, zz) in tz.items(): wt[idx] = T.tilt(wb[idx], zz, lam, cap)
                    PT[d] = (U, wt); PB[d] = (U, wb)
                r, acc = run_exec(ds, PT, tier, mode); rb, accb = run_exec(ds, PB, tier, mode); b = T.bench_win(ds)
                x = r - b; xb = r - rb
                pre20 = x.index < ODD_INTRADAY
                per.append({'vs0050': T.stats(x), 'vsB_exec': T.stats(xb), 'CAGR': T.cagr(r), 'CAGR_E50': T.cagr(b), 'vs0050_post2020_10': T.stats(x[~pre20]) if (~pre20).sum() > 3 else None,
                            'odd_share_of_traded': acc['odd_val'] / max(acc['odd_val'] + acc['lot_val'], 1), 'pre2020_odd_share': acc['pre_odd_val'] / max(acc['odd_val'] + acc['lot_val'], 1),
                            'cost_pp_per_month': 100 * (acc['fee'] + acc['tax'] + acc['impact'] + acc['odd'] + acc['flow_cost']) / (tier * len(x)), 'cap_hits': acc['cap_hit']})
            jm = lambda f: float(np.median([f(p) for p in per if f(p) is not None and np.isfinite(f(p))]))
            key = f'{vname}|{pname}|{tn}|{mode}'
            OUT[key] = {'vs0050_mean_pp_jmed': jm(lambda p: p['vs0050']['mean_pp']), 'vs0050_t_jmed': jm(lambda p: p['vs0050']['t']),
                        'vs0050_mean_pp_p10': float(np.percentile([p['vs0050']['mean_pp'] for p in per], 10)),
                        'vsB_exec_mean_pp_jmed': jm(lambda p: p['vsB_exec']['mean_pp']), 'vsB_exec_t_jmed': jm(lambda p: p['vsB_exec']['t']),
                        'CAGR_jmed': jm(lambda p: p['CAGR']), 'CAGR_E50_jmed': jm(lambda p: p['CAGR_E50']),
                        'vs0050_post2020_10_mean_pp_jmed': jm(lambda p: p['vs0050_post2020_10']['mean_pp'] if p['vs0050_post2020_10'] else None),
                        'odd_share_jmed': jm(lambda p: p['odd_share_of_traded']), 'pre2020_odd_share_jmed': jm(lambda p: p['pre2020_odd_share']),
                        'cost_pp_per_month_jmed': jm(lambda p: p['cost_pp_per_month']), 'cap_hits_jmed': jm(lambda p: p['cap_hits'])}
            print(key, {k: round(v, 3) for k, v in OUT[key].items()}, flush=True)
R = json.loads((L.HERE / 'results.json').read_text(encoding='utf-8'))
f_pass = {v: all(OUT[f'{v}|POST2019|{tn}|{mode}']['vs0050_mean_pp_jmed'] > 0 for tn, _, mode in TIERS) for v in ('T1', 'T2')}
R['exec'] = {'spec': 'K5T（V7 model_spec7）＋零股價差＋最低手續費 20 元＋FIX 縮放成本＋0.25% 不交易帶', 'cells': OUT, 'f_pass': f_pass}
(L.HERE / 'results.json').write_text(L.dumps(R), encoding='utf-8')
print('f_pass', f_pass)
