#!/usr/bin/env python3
"""重測 r1_1：跳空下跌當沖的「收盤跌停能否賣出」假設、完整版本表、真鎖死比例、盤中 −9% 停損。

事前固定（寫程式前決定，不依結果調整）：
A. 完整表：gap_down_3、gap_down_5、gap_down_3to5 × 出場假設 × 窗口（S0/S1/V_official/V_all）。
   出場假設（收盤跌停以 stlib 容差判定：close/prev−1 ≤ −(lim−0.6%)，flag 0）：
     cons   ：收盤跌停一律隔日開盤賣、成本 0.5%（原 r2 主結果）
     half   ：收盤跌停者一半收盤賣、一半隔日開盤（原 r2）
     naive  ：收盤跌停也在收盤賣、成本 0.23%（原 r2）
     exact  ：用台股 tick 規則算精確跌停價；只有「收盤＝跌停價」者隔日開盤賣，容差內但不在跌停價者收盤賣
              （收盤＝跌停價必然 close＝low＝跌停價，所以這等同「close==low==跌停價」定義）
     exact_half：exact 中收盤在跌停價者一半收盤成交
   加總：每日等權 cohort（同 stlib.run_rule，t＝事件日序列），另報每筆等權＋日期 cluster t。
   通過門檻（同 r2）：淨 > 0 且 t ≥ 2。
B. 官方 history.csv.gz（2025-09-24～2026-10-22，有 low／volume）期間（窗口 H＝V_official ∩ 該期間）：
   1) 檢查面板與 history 的 open/close 一致性；檢查 tick 規則算出的跌停價 ≤ low（跌停價正確性）。
   2) 真鎖死定義：close == low == 跌停價（精確 tick）。「收盤前 5 分鐘仍在跌停價」需分 K，
      官方期間沒有分 K；以 src 3（Yahoo 5 分 K，收盤＝13:20）期間「13:20 價格＝跌停價」的比例作為近似對照。
      報告：容差跌停事件中真鎖死比例 q（H 期間、全樣本各窗口）。
   3) H 期間重算 cons／half／naive／exact／exact_half。
   4) 停損版本（只有 H 期間有 low）：停損價 sp＝floor_tick(prev×0.91)；開盤 > sp 且 low ≤ sp → 以 sp 賣出（當沖 0.23%）；
      沒觸及 → 收盤賣（不可能收在跌停）。開盤 ≤ sp 的事件停損不適用，改用 exact 出場（另列筆數）。
      滑價敏感度：sp 再低 1 tick、sp×(1−0.5%)（不低於跌停價）、最壞以跌停價成交。
"""
import sys, time, math, json
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/short_term')
import numpy as np, pandas as pd
import stlib as L

t0 = time.time()
D = L.load()
CAL = D['CAL']
on, idr, elig, F = D['on'], D['id'], D['elig'], D['flag']
C, O, pc = D['close_raw'], D['open'], D['pc']
lim = D['lim']
gdt, cdt = L.daytrade_exit(D)                 # cons
cld = D['close_limit_down'].fillna(False).astype(bool)


def tick_of(p):
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)


def limit_down_price(prev, limv):
    raw = prev * (1 - limv)
    t = tick_of(raw)
    ld = np.ceil(raw / t - 1e-7) * t
    # 進位後若跨到較大 tick 區間，仍須是新區間的 tick 倍數
    t2 = tick_of(ld)
    ld = np.where(t2 > t, np.ceil(ld / t2 - 1e-7) * t2, ld)
    return np.round(ld, 2)


def floor_tick(p):
    t = tick_of(p)
    return np.round(np.floor(p / t + 1e-7) * t, 2)


limv = np.broadcast_to(lim.values[:, None], C.shape)
LD = pd.DataFrame(limit_down_price(pc.values, limv), index=C.index, columns=C.columns)
exact = (np.abs(C - LD) < 1e-6) & (F == 0)           # 收盤＝精確跌停價
exact_ld = (cld & exact).fillna(False).astype(bool)

nxt_open = O.shift(-1)
with np.errstate(divide='ignore', invalid='ignore'):
    alt = nxt_open / O - 1
alt = alt.where(alt.notna(), idr)
COST_DT, COST_RT = L.COST_DT, L.COST_RT
VAR = {}
VAR['cons'] = (gdt, cdt)
VAR['half'] = (gdt.where(~cld, 0.5 * gdt + 0.5 * idr), cdt.where(~cld, 0.5 * COST_RT + 0.5 * COST_DT))
VAR['naive'] = (idr, pd.DataFrame(COST_DT, index=idr.index, columns=idr.columns))
g_ex = idr.where(~exact_ld, alt)
c_ex = pd.DataFrame(COST_DT, index=idr.index, columns=idr.columns).where(~exact_ld, COST_RT)
VAR['exact'] = (g_ex, c_ex)
VAR['exact_half'] = (g_ex.where(~exact_ld, 0.5 * alt + 0.5 * idr), c_ex.where(~exact_ld, 0.5 * COST_RT + 0.5 * COST_DT))

rules = {'gap_down_3': on <= -0.03, 'gap_down_5': on <= -0.05, 'gap_down_3to5': (on <= -0.03) & (on > -0.05)}
masks = {k: (elig & v).fillna(False).astype(bool) for k, v in rules.items()}


def events(mask, g, c, days, extra=None):
    m = (mask & g.notna()).reindex(days).fillna(False).astype(bool)
    st = m.stack(); st = st[st]; ix = st.index
    df = pd.DataFrame({'date': ix.get_level_values(0), 'code': ix.get_level_values(1),
                       'g': g.reindex(days).stack().reindex(ix).values, 'c': c.reindex(days).stack().reindex(ix).values})
    if extra:
        for k, v in extra.items():
            df[k] = v.reindex(days).stack().reindex(ix).values
    df['net'] = df['g'] - df['c']
    return df


def stats(df, col='net'):
    o = {'trades': int(len(df)), 'days': int(df['date'].nunique()) if len(df) else 0}
    if len(df) < 5: return o
    d = df.groupby('date')[col].mean()
    se = d.std(ddof=1) / math.sqrt(len(d))
    o['day_mean_pct'] = round(float(d.mean()) * 100, 3); o['day_t'] = round(float(d.mean() / se), 2) if se > 0 else None
    x = df[col].values; m = x.mean(); N = len(x)
    S = pd.Series(x - m).groupby(df['date'].values).sum().values; G = len(S)
    v = (G / (G - 1)) * (S ** 2).sum() / N ** 2 if G > 1 else float('nan')
    o['trade_mean_pct'] = round(m * 100, 3); o['trade_t_cluster'] = round(m / math.sqrt(v), 2) if v > 0 else None
    o['trade_median_pct'] = round(float(np.median(x)) * 100, 3)
    o['pass'] = bool(o['day_mean_pct'] > 0 and (o['day_t'] or 0) >= 2)
    return o


res = {'description': __doc__, 'A_full_table': {}, 'A_compare_r2_json': {}, 'A_limit_down_diag': {}}
r2 = json.loads((L.OUT / 'results_r2.json').read_text())
r2key = {'cons': 'daytrade', 'half': 'daytrade_half_fill_at_limit_down', 'naive': 'daytrade_naive_close_exit'}
rows = []
for rn, mask in masks.items():
    res['A_full_table'][rn] = {}
    for vn, (g, c) in VAR.items():
        rr, _, _, _ = L.run_rule(D, g, c, mask)
        res['A_full_table'][rn][vn] = {}
        for w in L.WINDOWS:
            dm = L.day_set(D, w); days = dm[dm].index
            st = stats(events(mask, g, c, days))
            st['stlib_mean_net_pct'] = rr[w].get('mean_net_pct'); st['stlib_t_net'] = rr[w].get('t_net')
            res['A_full_table'][rn][vn][w] = st
            if vn in r2key:
                j = r2[f'{rn}__{r2key[vn]}'][w]
                res['A_compare_r2_json'][f'{rn}|{vn}|{w}'] = {'r2_json': [j.get('mean_net_pct'), j.get('t_net'), j.get('trades')],
                                                              'rerun': [rr[w].get('mean_net_pct'), rr[w].get('t_net'), rr[w].get('trades')]}
            rows.append([rn, vn, w, st['trades'], st.get('day_mean_pct'), st.get('day_t'), st.get('trade_mean_pct'), st.get('trade_t_cluster'), st.get('pass')])
    # 跌停診斷
    for w in L.WINDOWS:
        dm = L.day_set(D, w); days = dm[dm].index
        df = events(mask, gdt, cdt, days, extra={'cld': cld, 'exact': exact_ld, 'idr': idr, 'alt': alt})
        df['cld'] = df['cld'].astype(bool); df['exact'] = df['exact'].astype(bool)
        sub = df[df['cld']]
        res['A_limit_down_diag'][f'{rn}|{w}'] = {
            'n': int(len(df)), 'share_tol_limit_down': round(float(df['cld'].mean()), 3),
            'share_exact_among_tol': round(float(sub['exact'].mean()), 3) if len(sub) else None,
            'tol_ld_mean_open_to_close_pct': round(float(sub['idr'].mean() * 100), 2) if len(sub) else None,
            'tol_ld_mean_open_to_next_open_pct': round(float(sub['alt'].mean() * 100), 2) if len(sub) else None,
            'next_open_minus_close_pct': round(float((sub['alt'] - sub['idr']).mean() * 100), 2) if len(sub) else None,
            'nonexact_tol_ld_next_open_minus_close_pct': round(float((sub.loc[~sub['exact'], 'alt'] - sub.loc[~sub['exact'], 'idr']).mean() * 100), 2) if (~sub['exact']).sum() else None,
            'exact_ld_next_open_minus_close_pct': round(float((sub.loc[sub['exact'], 'alt'] - sub.loc[sub['exact'], 'idr']).mean() * 100), 2) if sub['exact'].sum() else None}

# ---------- B：官方 history.csv.gz ----------
H = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/history.csv.gz', dtype={'code': str}, parse_dates=['date'])
H = H[H['code'].isin(set(C.columns))]
Hw = {k: H.pivot_table(index='date', columns='code', values=k, aggfunc='last').reindex(index=CAL, columns=C.columns) for k in ['open', 'low', 'close', 'volume', 'high']}
h0 = H['date'].min(); h1 = H['date'].max()
dmv = L.day_set(D, 'V_official')
hdays = dmv[dmv & (CAL >= h0) & (CAL <= h1)].index
# 一致性
both = Hw['close'].reindex(hdays).notna() & C.reindex(hdays).notna()
cons_chk = {
    'period': f'{h0.date()}~{h1.date()}', 'H_days': int(len(hdays)),
    'close_match_share': round(float(((Hw['close'].reindex(hdays) - C.reindex(hdays)).abs() < 1e-6)[both].sum().sum() / both.sum().sum()), 4),
    'open_match_share': round(float(((Hw['open'].reindex(hdays) - O.reindex(hdays)).abs() < 1e-6)[both].sum().sum() / both.sum().sum()), 4),
}
LOW = Hw['low']
fz = (F == 0).reindex(hdays) & LOW.reindex(hdays).notna() & pc.reindex(hdays).notna()
below = (LOW.reindex(hdays) < LD.reindex(hdays) - 1e-6) & fz
cons_chk['low_below_computed_limit_share'] = round(float(below.sum().sum() / fz.sum().sum()), 5)
cons_chk['low_below_examples'] = [[str(a.date()), b] for a, b in below.stack()[below.stack()].index[:10]]
exact_low = (exact & (np.abs(LOW - LD) < 1e-6))
res['B_consistency'] = cons_chk

# 停損
SP = pd.DataFrame(floor_tick(pc.values * 0.91), index=C.index, columns=C.columns)
tk = pd.DataFrame(tick_of(SP.values), index=C.index, columns=C.columns)
open_above = O > SP + 1e-9
hit = open_above & (LOW <= SP + 1e-9)
fills = {'stop_at_sp': SP, 'stop_sp_minus_1tick': np.maximum(SP - tk, LD), 'stop_sp_minus_0.5pct': np.maximum(SP * 0.995, LD), 'stop_at_limit_price': LD}
VARH = dict(VAR)
_el = (cld & exact_low).fillna(False).astype(bool)
VARH['exact_low'] = (idr.where(~_el, alt), pd.DataFrame(COST_DT, index=idr.index, columns=idr.columns).where(~_el, COST_RT))
for fn, fp in fills.items():
    with np.errstate(divide='ignore', invalid='ignore'):
        gs = (fp / O - 1)
    g = g_ex.where(~open_above, idr).where(~hit, gs)          # 開盤 > sp：沒觸及收盤賣、觸及以停損價；開盤 ≤ sp：exact 出場
    c = c_ex.where(~open_above, COST_DT).where(~hit, COST_DT)
    VARH[fn] = (g, c)

res['B_H_period'] = {}
srows = []
for rn, mask in masks.items():
    o = {}
    base = events(mask, gdt, cdt, hdays, extra={'cld': cld, 'exact': exact_ld, 'exact_low': exact_low, 'hit': hit, 'open_above': open_above,
                                                 'idr': idr, 'alt': alt, 'vol': Hw['volume'], 'low': LOW, 'LD': LD})
    for k in ['cld', 'exact', 'exact_low', 'hit', 'open_above']:
        base[k] = base[k].fillna(False).astype(bool)
    sub = base[base['cld']]
    o['n_events'] = int(len(base))
    o['share_tol_limit_down'] = round(float(base['cld'].mean()), 3)
    o['true_lock_close_eq_low_eq_limit_among_tol'] = round(float((sub['exact'] & sub['exact_low']).mean()), 3) if len(sub) else None
    o['n_tol_limit_down'] = int(len(sub)); o['n_true_lock'] = int((sub['exact'] & sub['exact_low']).sum())
    o['share_stop_hit'] = round(float(base['hit'].mean()), 3)
    o['n_open_at_or_below_stop'] = int((~base['open_above']).sum())
    o['stop_hit_mean_open_to_close_pct'] = round(float(base.loc[base['hit'], 'idr'].mean() * 100), 2) if base['hit'].any() else None
    o['stop_hit_share_close_limit_down'] = round(float(base.loc[base['hit'], 'cld'].mean()), 3) if base['hit'].any() else None
    o['stop_not_hit_mean_open_to_close_pct'] = round(float(base.loc[base['open_above'] & ~base['hit'], 'idr'].mean() * 100), 2)
    o['variants'] = {}
    for vn, (g, c) in VARH.items():
        df = events(mask, g, c, hdays)
        o['variants'][vn] = stats(df)
        o['variants'][vn]['gross_trade_mean_pct'] = round(float(df['g'].mean() * 100), 3)
        # 拿掉事件最多的 5 天
        cnt = df.groupby('date').size().sort_values(ascending=False, kind='mergesort')
        o['variants'][vn]['drop_top5_days'] = stats(df[~df['date'].isin(set(cnt.index[:5]))])
        srows.append([rn, vn, o['variants'][vn]['trades'], o['variants'][vn].get('day_mean_pct'), o['variants'][vn].get('day_t'),
                      o['variants'][vn].get('trade_mean_pct'), o['variants'][vn].get('trade_t_cluster'),
                      o['variants'][vn]['drop_top5_days'].get('day_mean_pct'), o['variants'][vn]['drop_top5_days'].get('trade_mean_pct')])
    top = base.groupby('date').size().sort_values(ascending=False).head(8)
    o['top_event_days'] = {str(k.date()): int(v) for k, v in top.items()}
    res['B_H_period'][rn] = o

# src 3（13:20 收盤近似）：容差跌停事件中 13:20 價格＝精確跌停價比例（「收盤前 5～10 分鐘仍在跌停」的近似）
dma = L.day_set(D, 'V_all')
s3 = dma[dma & (CAL >= pd.Timestamp('2024-02-15')) & (CAL <= pd.Timestamp('2025-09-23'))].index
res['B_src3_1320_proxy'] = {}
for rn, mask in masks.items():
    df = events(mask, gdt, cdt, s3, extra={'cld': cld, 'exact': exact_ld})
    df['cld'] = df['cld'].fillna(False).astype(bool); df['exact'] = df['exact'].fillna(False).astype(bool)
    sub = df[df['cld']]
    res['B_src3_1320_proxy'][rn] = {'n': int(len(df)), 'share_tol_ld_at_1320': round(float(df['cld'].mean()), 3),
                                    'share_exact_limit_at_1320_among_tol': round(float(sub['exact'].mean()), 3) if len(sub) else None}

# 以 q 重算全期（q＝真鎖死比例；把 q 套到容差跌停事件：q 隔日開盤、1−q 收盤）
res['C_q_reweighted'] = {}
for rn, mask in masks.items():
    qH = res['B_H_period'][rn]['true_lock_close_eq_low_eq_limit_among_tol']
    out = {'q_from_H': qH}
    for qlab, q in [('q_H', qH), ('q_0.5', 0.5), ('q_0.25', 0.25)]:
        g = gdt.where(~cld, q * gdt + (1 - q) * idr); c = cdt.where(~cld, q * COST_RT + (1 - q) * COST_DT)
        rr, _, _, _ = L.run_rule(D, g, c, mask)
        out[qlab] = {w: [rr[w].get('mean_net_pct'), rr[w].get('t_net')] for w in L.WINDOWS}
    res['C_q_reweighted'][rn] = out

# 損益兩平的「容差跌停事件賣不掉比例」q*：日 cohort 平均對 q 線性，q* = naive / (naive − cons)
res['D_breakeven_q'] = {}
for rn in masks:
    o = {}
    for w in L.WINDOWS:
        a = res['A_full_table'][rn]['naive'][w]['day_mean_pct']; b = res['A_full_table'][rn]['cons'][w]['day_mean_pct']
        o[w] = round(a / (a - b), 3) if a > 0 and a != b else None
    a = res['B_H_period'][rn]['variants']['naive']['day_mean_pct']; b = res['B_H_period'][rn]['variants']['cons']['day_mean_pct']
    o['H_2025-09_2026-10'] = round(a / (a - b), 3) if a > 0 and a != b else None
    res['D_breakeven_q'][rn] = o
print('breakeven q*', res['D_breakeven_q'])
res['runtime_sec'] = round(time.time() - t0, 1)
L.save('results_recheck_r1_1.json', res)
pd.set_option('display.width', 250)
print(json.dumps(res['B_consistency'], ensure_ascii=False))
print(pd.DataFrame(rows, columns=['rule', 'var', 'win', 'trades', 'D_mean', 'D_t', 'T_mean', 'T_tcl', 'pass']).to_string())
mism = {k: v for k, v in res['A_compare_r2_json'].items() if v['r2_json'] != v['rerun']}
print('r2 json mismatches:', mism)
for k, v in res['A_limit_down_diag'].items(): print(k, v)
print(pd.DataFrame(srows, columns=['rule', 'var', 'trades', 'D_mean', 'D_t', 'T_mean', 'T_tcl', 'D_drop5', 'T_drop5']).to_string())
for rn, o in res['B_H_period'].items():
    print(rn, {k: v for k, v in o.items() if k != 'variants'})
print(res['B_src3_1320_proxy'])
for k, v in res['C_q_reweighted'].items(): print(k, v)
print('runtime', res['runtime_sec'])
