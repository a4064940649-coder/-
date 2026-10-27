#!/usr/bin/env python3
"""反方審查（成本與可成交）：只讀研究員檔案，不寫入任何研究員檔案。輸出到 stdout。
python3 -I -B review_exec_o2.py
"""
import sys, json, math
from pathlib import Path
HERE = Path('/home/user/-/research/tw_stock_selection/agents/o_sbl_foreign_level')
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import sbl_lib as L

K, V, IDX, COLS, N = L.K, L.V, L.IDX, L.COLS, L.N
S = L.strategy(True)
END = S.end_day
t_end = IDX.get_loc(END)
OUT = {}

def summ(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return {'n': int(len(x))}
    return {'n': int(len(x)), 'mean%': round(float(x.mean()) * 100, 3), 't': round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2)}

PER = {'SEL': L.SEL, 'VAL': L.VAL, 'HOLD': L.HOLD}
plans = {}
for pn, (a, b) in PER.items():
    plans[pn] = {'main': L.main_plan(S, a, b), 'B1': L.sleeve_plan(S, 'lvl_size', a, b),
                 'C2': L.main_plan(S, a, b, third='lvl', w3=0.2)}

# ---------- 1. 周轉與成本拖累 ----------
rt = sum(L.COST['normal']); rts = sum(L.COST['stress'])
tov = {}
for pn in PER:
    for k, pl in plans[pn].items():
        _, lg = L.run_plan(pl, 'normal', t_end)
        ch = [len(set(p[1]) - set(q[1])) for q, p in zip(pl[:-1], pl[1:])]
        tov[f'{pn}_{k}'] = {'sold_frac_per_reb': round(lg['turnover'], 3), 'names_changed_per_month': round(float(np.mean(ch)), 2),
                            'annual_drag_normal%': round(lg['turnover'] * 12 * rt * 100, 2),
                            'annual_drag_stress%': round(lg['turnover'] * 12 * rts * 100, 2)}
OUT['1_turnover'] = tov
OUT['1_roundtrip_cost%'] = {'normal': round(rt * 100, 3), 'stress': round(rts * 100, 3)}

# C2 和主策略名單差幾檔
for pn in PER:
    m, c = plans[pn]['main'], plans[pn]['C2']
    OUT.setdefault('1b_C2_vs_main_names_differ', {})[pn] = round(float(np.mean([len(set(p[1]) - set(q[1])) for p, q in zip(c, m)])), 2)

# ---------- 2. 漲停買不到（被遞補）、跌停賣不掉 ----------
def ld_price(ref, lim):
    raw = ref * (1 - lim); t = K.tick(raw)
    return np.ceil(raw / t - 1e-9) * t

def limit_stats(pl):
    held = set(); blk = 0; nbuy = 0; ldopen = 0; ldlock = 0; nsell = 0; gu_new_ret = []
    for i, p in enumerate(pl):
        e = p[0]; pk = set(p[1].tolist()); d = p[3]
        cs = S.cross_section(d)
        gu = K.gap_up_open(e)
        # 被跳過的：候選中排在 pick 之前、開盤漲停且非續抱
        newb = pk - held; nbuy += len(newb)
        lim = 0.10
        ref = K.ref_price(e)
        ldp = ld_price(ref, lim)
        sells = held - pk; nsell += len(sells)
        for j in sells:
            if np.isfinite(K.On[e, j]) and K.On[e, j] <= ldp[j] + 1e-6:
                ldopen += 1
                if np.isfinite(K.Hn[e, j]) and K.Hn[e, j] <= ldp[j] + 1e-6: ldlock += 1
        held = pk
    return {'new_buys': nbuy, 'sells': nsell, 'sell_at_limitdown_open': ldopen, 'sell_limitdown_locked_allday': ldlock}

lim_out = {}
for pn in ('SEL', 'VAL'):
    for k in ('main', 'B1', 'C2'):
        lim_out[f'{pn}_{k}'] = limit_stats(plans[pn][k])
# 遞補次數：重跑 _pick 時被擋的新買進
def blocked(pn, k):
    a, b = PER[pn]; cnt = 0; tot = 0
    pl = plans[pn][k]; held = set()
    for p in pl:
        e = p[0]; gu = K.gap_up_open(e)
        cs = S.cross_section(p[3]); el = cs['el']
        # 只算：如果不擋漲停，原本會選但因開盤漲停被跳過的
        cnt += int(sum(1 for j in np.where(el & gu)[0] if j not in held))
        held = set(p[1].tolist())
    return cnt
for pn in ('SEL', 'VAL'):
    for k in ('main', 'C2'):
        # 用主策略引擎計數
        pass
OUT['2_limit'] = lim_out

# ---------- 3. 整張／零股與容量 ----------
V60 = S.V60
def lots(pl, capital, n):
    alloc = capital / n
    need_odd = []; cash_drag = []; part = []
    for p in pl:
        e = p[0]; t = e - 1
        px = K.On[e, p[1]]
        lot = px * 1000
        need_odd.append(float(np.mean(lot > alloc)))
        # 只買整張時：沒買到的現金比例
        nl = np.floor(alloc / lot)
        cash_drag.append(float(1 - np.nansum(nl * lot) / capital))
        part.append(alloc / V60[t, p[1]])
    part = np.concatenate(part)
    return {'per_name': int(alloc), 'share_needing_odd_lot': round(float(np.mean(need_odd)), 3),
            'cash_left_if_round_lots_only': round(float(np.mean(cash_drag)), 3),
            'alloc/ADV60_median%': round(float(np.nanmedian(part)) * 100, 3), 'alloc/ADV60_p95%': round(float(np.nanpercentile(part, 95)) * 100, 3),
            'alloc/ADV60_max%': round(float(np.nanmax(part)) * 100, 3)}
cap = {}
for k, n in (('main', 10), ('C2', 10), ('B1', 30)):
    for capital in (3e5, 1e6, 1e7):
        cap[f'{k}_{int(capital/1e4)}萬'] = lots(plans['VAL'][k], capital, n)
OUT['3_lots_capacity_VAL'] = cap
# 價格分布
px = {}
for k in ('main', 'C2', 'B1'):
    v = np.concatenate([K.On[p[0], p[1]] for p in plans['VAL'][k]])
    px[k] = {'median_open': round(float(np.nanmedian(v)), 1), 'p90': round(float(np.nanpercentile(v, 90)), 1), 'share_>1000': round(float(np.mean(v > 1000)), 3)}
OUT['3b_price_VAL'] = px

# ---------- 4. 資料時點：保留段用「回推值」重跑 C2，看名單與績效對資料誤差的敏感度 ----------
Rq, Sq, Fq = L.load_qfiis(); W = L.load_flows()
anc2 = Rq.index.max()
bc = L.backcast(Rq, Sq, W, anc2, L.ANCHOR - pd.Timedelta(days=1))
alt = bc.reindex(columns=COLS)
alt = alt.reindex(IDX.union(alt.index)).sort_index().ffill(limit=5).reindex(IDX)
D0 = L.fo_data(); P_off = D0['P'].copy()
P_alt = P_off.copy()
m = (IDX >= L.ANCHOR) & (IDX <= anc2)
P_alt[m] = np.clip(alt.to_numpy(dtype=float)[m], 0, 1)
a, b = L.HOLD
c2_off = plans['HOLD']['C2']; main_h = plans['HOLD']['main']
L._FO['P'] = P_alt
c2_alt = L.main_plan(S, a, b, third='lvl', w3=0.2)
b1_alt = L.sleeve_plan(S, 'lvl_size', a, b)
L._FO['P'] = P_off
b1_off = plans['HOLD']['B1']
SD = [p[3] for p in main_h] + [END]
nm = L.run_plan(main_h, 'normal', t_end)[0]
no = L.run_plan(c2_off, 'normal', t_end)[0]; na = L.run_plan(c2_alt, 'normal', t_end)[0]
bo = L.run_plan(b1_off, 'normal', t_end)[0]; ba = L.run_plan(b1_alt, 'normal', t_end)[0]
ew = L.ew_nav(main_h, t_end)
OUT['4_backcast_sensitivity_HOLD'] = {
    'C2_names_differ_official_vs_backcast': round(float(np.mean([len(set(p[1]) - set(q[1])) for p, q in zip(c2_off, c2_alt)])), 2),
    'B1_names_differ_official_vs_backcast(of30)': round(float(np.mean([len(set(p[1]) - set(q[1])) for p, q in zip(b1_off, b1_alt)])), 2),
    'C2_official': {**L.cagr_dd(no, a, END), 'vs_main': summ(L.monthly_t(no, nm, SD, a, b))},
    'C2_backcast': {**L.cagr_dd(na, a, END), 'vs_main': summ(L.monthly_t(na, nm, SD, a, b))},
    'C2_backcast_minus_official_monthly': summ(L.monthly_t(na, no, SD, a, b)),
    'B1_official_vs_EW': summ(L.monthly_t(bo, ew, SD, a, b)), 'B1_backcast_vs_EW': summ(L.monthly_t(ba, ew, SD, a, b)),
    'main': L.cagr_dd(nm, a, END)}

# ---------- 5. C2 對主策略：壓力成本下逐段配對差＋「替代條件」在壓力成本下是否還成立 ----------
nav = pd.read_parquet(HERE / 'oos_navs.parquet')
SDv = [p[3] for p in plans['VAL']['main']] + [END]
alt5 = {}
for nmx, (lo, hi) in {'VAL_A': L.VAL_A, 'HOLD': L.HOLD, 'VAL': L.VAL}.items():
    sub = [d for d in SDv if lo <= d <= hi]; nd = [d for d in SDv if d > hi]; end = nd[0] if nd else END
    r = {}
    for col in ('main_stress', 'C2_stress', 'main_n10', 'C2'):
        r[col] = L.cagr_dd(nav[col].dropna(), sub[0] + pd.Timedelta(days=1), end)
    r['C2s_vs_mains'] = summ(L.monthly_t(nav['C2_stress'], nav['main_stress'], SDv, lo, hi))
    alt5[nmx] = r
OUT['5_C2_stress'] = alt5

# ---------- 6. 去掉單月：C2 對主策略的驗證期配對差是否靠一兩個月 ----------
x = L.monthly_t(nav['C2'], nav['main_n10'], SDv, *L.VAL)
xs = x.sort_values()
OUT['6_C2_vs_main_months'] = {'top3': {str(k.date()): round(v * 100, 2) for k, v in xs[-3:].items()},
                              'bottom3': {str(k.date()): round(v * 100, 2) for k, v in xs[:3].items()},
                              'ex_best1': summ(xs[:-1]), 'ex_best2': summ(xs[:-2]), 'share_months_positive': round(float((x > 0).mean()), 3)}

print(json.dumps(OUT, ensure_ascii=False, indent=1, default=str))

# ---------- 7. 去掉 C2 相對主策略最好的一個月後，驗證期年化替代條件是否還成立 ----------
o7 = {}
for cm, cc in (('main_n10', 'C2'), ('main_stress', 'C2_stress')):
    D = pd.DatetimeIndex([d for d in SDv if d in nav.index])
    pm = (nav[cm].reindex(D).shift(-1) / nav[cm].reindex(D) - 1).iloc[:-1]
    pc = (nav[cc].reindex(D).shift(-1) / nav[cc].reindex(D) - 1).iloc[:-1]
    w = (pm.index >= L.VAL[0]) & (pm.index <= L.VAL[1]); pm, pc = pm[w], pc[w]
    yrs = (D[-1] - pm.index[0]).days / 365.25
    g = lambda s: float(np.prod(1 + s) ** (1 / yrs) - 1) * 100
    best = (pc - pm).idxmax()
    o7[cc] = {'main_CAGR%': round(g(pm), 2), 'C2_CAGR%': round(g(pc), 2), 'best_rel_month': str(best.date()),
              'C2_CAGR_if_best_month_equal_main%': round(g(pc.where(pc.index != best, pm)), 2)}
print(json.dumps({'7_C2_best_month_removed': o7}, ensure_ascii=False, indent=1))
