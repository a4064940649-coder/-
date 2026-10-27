#!/usr/bin/env python3
"""交叉檢驗（2021 起資料，營收用 rev_5y＝exec_rebuild 原設定）：
(a) 主策略前 30 → 籌碼第二層取 10（A0～A4）；(b1) 持股進入處置續抱 vs 賣出；(b2) 處置事件依主策略排名分組。
規則見 results.json PREREG。結果寫入 results.json 的 CROSS。"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.dont_write_bytecode = True
import clib as K
L = K.L
OUT = K.HERE
IDX, COLS, N = L.IDX, L.COLS, L.N
SELa = ('2021-11-01', '2023-12-31'); SELb = ('2021-09-01', '2023-12-31'); VAL = ('2024-01-01', '2026-10-31')
KEEP = ('CAGR', 'maxDD', 't_vs_0050_m', 't_vs_EW_m', 'excess_vs_EW_pct_m', 'excess_vs_0050_pct_m', 'yearly_pct', '0050_CAGR', 'EW_CAGR', 'n_months')

def chips():
    x = pd.read_csv(L.X / 'inst_5y.csv.gz', dtype={'code': str}, usecols=['date', 'code', 'foreign', 'trust'], parse_dates=['date'])
    x = x[x['code'].isin(COLS)]; x['ft'] = x['foreign'].fillna(0) + x['trust'].fillna(0)
    ii = IDX[(IDX >= x['date'].min()) & (IDX <= x['date'].max() + pd.Timedelta(days=40))]
    FT = x.pivot_table(index='date', columns='code', values='ft', aggfunc='sum').reindex(ii).reindex(columns=COLS) * 1000
    VOL = (L.V / L.C).reindex(ii)
    FT = FT.where(VOL.notna()).fillna(0).where(VOL.notna())
    F = FT.rolling(20, min_periods=15).sum() / VOL.rolling(20, min_periods=15).sum()
    F = F.where(pd.Series(ii <= x['date'].max(), index=ii), np.nan)
    dt = pd.read_csv('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/scr_extra/daytrade_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
    dt = dt[dt['code'].isin(COLS)]
    DT = dt.pivot_table(index='date', columns='code', values='dt_ratio', aggfunc='last').reindex(columns=COLS)
    DT = DT.reindex(IDX[(IDX >= DT.index.min()) & (IDX <= DT.index.max())]).rolling(20, min_periods=10).mean()
    return F.reindex(IDX), DT.reindex(IDX), str(x['date'].min().date()), str(x['date'].max().date())

def pr(v):
    s = pd.Series(v); r = s.rank(pct=True); return r.fillna(0.5).to_numpy()

def select_layer(S, sc, el, rule, F, DT, start):
    GU = L.GAP_UP_OPEN_ERA; SC = sc.to_numpy(); EL = el.to_numpy()
    events, pools = {}, {}; held = set(); info = []
    for k, d in enumerate(S.dates[:-1]):
        if d < pd.Timestamp(start): continue
        i = L.pos(d); ex = i + 1
        trad = np.isfinite(L.OPEN_RAW[ex]) & np.isfinite(L.ADJ_RAW[ex])
        ok = EL[k] & trad & np.isfinite(SC[k])
        if ok.sum() < 50: continue
        pools[ex] = ('open', np.where(ok)[0])
        order = SC[k].copy(); order[~ok] = np.nan
        cand = np.argsort(-np.nan_to_num(order, nan=-9))[:int(ok.sum())]
        c30 = []
        nmax = 10 if rule == 'A0' else 30
        for j in cand:
            if GU[ex, j] and j not in held: continue
            c30.append(j)
            if len(c30) == nmax: break
        c30 = np.array(c30)
        if rule in ('A0', 'N30'): pick = c30[:10] if rule == 'A0' else c30
        else:
            f = F.iloc[i].to_numpy()[c30]; dtr = DT.iloc[i].to_numpy()[c30]; m = SC[k][c30]
            info.append((float(np.isfinite(f).mean()), float(np.isfinite(dtr).mean())))
            if rule == 'A1': z = pr(f)
            elif rule == 'A2': z = pr(-dtr)
            elif rule == 'A3': z = (pr(f) + pr(-dtr)) / 2
            elif rule == 'A4': z = (pr(m) + pr(f) + pr(-dtr)) / 3
            o = np.argsort(-z, kind='stable')     # 同分依主分數順序
            pick = c30[o[:10]]
        events[ex] = ('open', np.array(pick, dtype=int)); held = set(pick.tolist())
    cov = (round(float(np.mean([a for a, _ in info])), 3), round(float(np.mean([b for _, b in info])), 3)) if info else None
    return events, pools, cov

def ws(nav, segs, D, ew):
    o = L.daily_stats(nav, segs, D, L.E50, ew)
    return {k: v for k, v in o.items() if k in KEEP} if o else None

def part_a(S, sc, el, b, s, F, DT, res):
    out = {}
    navs = {}; mons = {}
    t0 = None
    for rule in ('N30', 'A0', 'A1', 'A2', 'A3', 'A4'):
        ev, pools, cov = select_layer(S, sc, el, rule, F, DT, '2021-11-01')
        if t0 is None:
            t0 = min(ev) - 1; t1 = L.pos(S.dates[-1])
            ew = K.sim(pools, 0, 0, t0, t1); mons['EW'] = K.monthly(ew, S.DATES)
        nav = K.sim(ev, b, s, t0, t1); navs[rule] = nav; mons[rule] = K.monthly(nav, S.DATES)
        out[rule] = {'chip_coverage_in_top30(F,DT)': cov,
                     'select': ws(nav, [SELa], S.DATES, ew), 'valid': ws(nav, [VAL], S.DATES, ew), 'full': ws(nav, [(SELa[0], VAL[1])], S.DATES, ew)}
    M = pd.DataFrame(mons)
    for rule in ('A1', 'A2', 'A3', 'A4', 'N30'):
        for tag, w in (('select', SELa), ('valid', VAL)):
            d = K.win(M[rule] - M['A0'], *w).dropna()
            out[rule][f'{tag}_diff_vs_A0_pct_m'] = round(float(d.mean() * 100), 3); out[rule][f'{tag}_t_diff_vs_A0'] = K.tstat(d)
    best = max(('A1', 'A2', 'A3', 'A4'), key=lambda r: out[r]['select_t_diff_vs_A0'])
    v = out[best]
    verdict = '可用' if v['select_t_diff_vs_A0'] >= 2 and v['valid_t_diff_vs_A0'] >= 2 else ('待觀察' if v['valid_diff_vs_A0_pct_m'] > 0 and v['valid_t_diff_vs_A0'] >= 1 else '不可用')
    out['SELECTED'] = best; out['verdict'] = verdict
    # 確認 A0 與 lhlib 主策略 n10 一致
    ev10, _ = S.selection(10)
    ev0, _, _ = select_layer(S, sc, el, 'A0', F, DT, '2021-11-01')
    out['A0_equals_lhlib_n10_picks'] = all(np.array_equal(np.sort(ev10[k][1]), np.sort(ev0[k][1])) for k in ev0)
    res['a_chips_second_layer'] = out

def part_b1(S, sc, el, b, s, res):
    dsp = L.dsp.dropna(subset=['start'])
    out = {}
    for n in (30, 10):
        ev, pools = K.select(S, sc, el, n)
        ks = sorted(ev); t0 = ks[0] - 1; t1 = L.pos(S.dates[-1])
        ew = K.sim(pools, 0, 0, t0, t1)
        sells = {}; evrows = []
        for a, bnext in zip(ks, ks[1:] + [t1 + 1]):
            pk = set(ev[a][1].tolist())
            for _, r in dsp.iterrows():
                c = r['code']
                if c not in L.CI or L.CI[c] not in pk: continue
                ts = IDX.searchsorted(r['start'])
                if ts >= len(IDX): continue
                if a <= ts < bnext:
                    j = L.CI[c]
                    if j in sells.get(ts, []): continue
                    if any(j in sells.get(tt, []) for tt in range(a, ts)): continue   # 同期已賣
                    sells.setdefault(ts, []).append(j)
                    end = min(bnext, t1)
                    rs = L.AO[end, j] / L.AO[ts, j] - 1 if bnext <= t1 else L.AC[t1, j] / L.AO[ts, j] - 1
                    re = ew.iloc[end - 1 - t0] / ew.iloc[ts - 1 - t0] - 1
                    evrows.append({'code': c, 'sell_day': str(IDX[ts].date()), 'same_day_as_rebalance': bool(ts == a), 'ret_hold_to_next_rebalance': rs, 'ret_minus_EW': rs - re})
        nav0 = K.sim(ev, b, s, t0, t1); nav1 = K.sim(ev, b, s, t0, t1, sells=sells)
        m0 = K.monthly(nav0, S.DATES); m1 = K.monthly(nav1, S.DATES)
        E = pd.DataFrame(evrows); E['d'] = pd.to_datetime(E['sell_day'])
        o = {}
        for tag, w in (('select', SELb), ('valid', VAL)):
            d = K.win(m1 - m0, *w).dropna()
            e = E[(E['d'] >= w[0]) & (E['d'] <= w[1])]
            o[tag] = {'B0_hold': ws(nav0, [w], S.DATES, ew), 'B1_sell': ws(nav1, [w], S.DATES, ew),
                      'diff_B1_minus_B0_pct_m': round(float(d.mean() * 100), 3), 't_diff': K.tstat(d),
                      'events': int(len(e)), 'events_same_day_as_rebalance': int(e['same_day_as_rebalance'].sum()),
                      'event_mean_ret_if_held_pct': round(float(e['ret_hold_to_next_rebalance'].mean() * 100), 2) if len(e) else None,
                      'event_mean_ret_minus_EW_pct': round(float(e['ret_minus_EW'].mean() * 100), 2) if len(e) else None,
                      'event_t_minus_EW': K.tstat(e['ret_minus_EW']) if len(e) > 2 else None,
                      'event_win_rate_vs_EW': round(float((e['ret_minus_EW'] > 0).mean()), 3) if len(e) else None}
            for kk in ('B0_hold', 'B1_sell'): o[tag][kk].pop('yearly_pct', None)
        o['verdict_B1'] = '採用 B1' if (o['select']['t_diff'] or -9) >= 2 and o['valid']['diff_B1_minus_B0_pct_m'] > 0 else '不採用 B1（續抱）'
        out[f'n{n}'] = o
    res['b1_hold_vs_sell'] = out

def part_b2(S, res):
    T = pd.read_csv(OUT.parent / 'disposal_reconcile' / 'official_trades_twse.csv', dtype={'code': str}, parse_dates=['date', 'ann_date'])
    assert (pd.DatetimeIndex(IDX[T['i0'].to_numpy()]) == pd.DatetimeIndex(T['date'])).all(), 'index mismatch'
    Y3 = L.yoy_rev5y().rolling(3, min_periods=3).mean()
    grp, rk = [], []
    for _, r in T.iterrows():
        i = int(r['ann_i']); d = IDX[i]
        m = pd.Period(d, 'M') - (1 if d.day >= 11 else 2)
        v60 = L.v60_fix.iloc[i]; px = L.C.iloc[i]; base = (v60 >= 1e7) & (px >= 10)
        t300 = base & (v60.where(base).rank(ascending=False) <= 300)
        y = Y3.loc[m].reindex(COLS) if m in Y3.index else pd.Series(np.nan, index=COLS)
        s52 = L.s52_fix.iloc[i]
        el = t300 & y.notna() & s52.notna()
        scr = 0.5 * s52.where(el).rank(pct=True) + 0.5 * y.where(el).rank(pct=True)
        c = r['code']
        if not el.get(c, False): grp.append('not_in_pool'); rk.append(np.nan); continue
        rnk = int(scr.rank(ascending=False)[c]); rk.append(rnk)
        grp.append('top100' if rnk <= 100 else 'pool_rest')
    T['grp'] = grp; T['main_rank'] = rk; T['m'] = T['date'].dt.to_period('M')
    def st(e):
        if len(e) == 0: return None
        mm = e.groupby('m')['net'].mean()
        return {'n': int(len(e)), 'mean_net_pct': round(float(e['net'].mean() * 100), 2), 'median_net_pct': round(float(e['net'].median() * 100), 2),
                'win': round(float((e['net'] > 0).mean()), 3), 't_month': K.tstat(mm), 'months': int(len(mm)),
                'mean_net_ex_top5pct': round(float(e['net'][e['net'] <= e['net'].quantile(0.95)].mean() * 100), 2)}
    out = {}
    for tag, w in (('select', ('2021-10-01', '2023-12-31')), ('valid', VAL)):
        e = T[(T['date'] >= w[0]) & (T['date'] <= w[1])]
        o = {g: st(e[e['grp'] == g]) for g in ('top100', 'pool_rest', 'not_in_pool')}
        o['all'] = st(e)
        a = e[e['grp'] == 'top100']['net']; bb = e[e['grp'] != 'top100']['net']
        if len(a) > 2:
            se = np.sqrt(a.var(ddof=1) / len(a) + bb.var(ddof=1) / len(bb))
            o['diff_top100_minus_rest_pct'] = round(float((a.mean() - bb.mean()) * 100), 2); o['welch_t'] = round(float((a.mean() - bb.mean()) / se), 2)
            ma = e[e['grp'] == 'top100'].groupby('m')['net'].mean(); mb = e[e['grp'] != 'top100'].groupby('m')['net'].mean()
            dd = (ma - mb).dropna(); o['month_clustered_diff_t'] = K.tstat(dd); o['months_both'] = int(len(dd))
        out[tag] = o
    s_, v_ = out['select'], out['valid']
    ok_s = s_.get('diff_top100_minus_rest_pct', -1) > 0 and s_.get('welch_t', 0) >= 2
    ok_v = v_.get('diff_top100_minus_rest_pct', -1) > 0 and v_.get('welch_t', 0) >= 2
    out['verdict'] = '可用' if ok_s and ok_v else ('待觀察' if ok_v else '不可用')
    res['b2_disposal_by_main_rank'] = out

def main():
    res = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
    S = L.Setup(L.yoy_rev5y(), start='2021-01-01')
    el = S.t300 & S.s52.notna() & ~S.inD & S.yoy3.notna()
    sc = 0.5 * S.s52.where(el).rank(axis=1, pct=True) + 0.5 * S.yoy3.where(el).rank(axis=1, pct=True)
    b, s = L.cost_rates(0.001)
    F, DT, f0, f1 = chips()
    cr = {'inst_range': [f0, f1], 'note': '營收 rev_5y；M2；滑價 0.1%；t＝月窗'}
    part_a(S, sc, el, b, s, F, DT, cr)
    part_b1(S, sc, el, b, s, cr)
    part_b2(S, cr)
    res['CROSS'] = cr
    (OUT / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    a = cr['a_chips_second_layer']
    for r in ('N30', 'A0', 'A1', 'A2', 'A3', 'A4'):
        print(r, a[r]['chip_coverage_in_top30(F,DT)'], {t: (a[r][t]['CAGR'], a[r][t]['maxDD'], a[r][t]['t_vs_0050_m'], a[r][t]['t_vs_EW_m']) for t in ('select', 'valid', 'full')},
              a[r].get('select_diff_vs_A0_pct_m'), a[r].get('select_t_diff_vs_A0'), a[r].get('valid_diff_vs_A0_pct_m'), a[r].get('valid_t_diff_vs_A0'))
    print('A selected', a['SELECTED'], a['verdict'], a['A0_equals_lhlib_n10_picks'])
    print(json.dumps(cr['b1_hold_vs_sell'], ensure_ascii=False))
    print(json.dumps(cr['b2_disposal_by_main_rank'], ensure_ascii=False))

if __name__ == '__main__':
    main()
