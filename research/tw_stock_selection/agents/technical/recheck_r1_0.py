#!/usr/bin/env python3
"""重測 r1_0：技術組「探索用合成 X1（帶量突破持 20／60 日）」的三項質疑。

A. 重現：照 t2_composite.py 的做法重跑 X1，確認是否等同 T07b，並統計 T07c 在合成裡實際進場幾次。
B. 修正合成（事前固定，全部報告）：
   B1 split15：T07b、T07c 各占一半資金、各自上限 15 檔分開模擬，每日再平衡 50/50。
   B2 split30：同上但各自上限 30 檔（各 sleeve 用原規則設定），每日再平衡 50/50。
   B3 split30_nr：同 B2，但兩個 sleeve 不再平衡（期初各半，之後各自滾動）。
   B4 dual30：同一池、上限 30 檔，但以（規則, 股票）為鍵，同一檔可在兩條規則下各持一份。
   B5 dual60：同 B4，上限 60 檔。
C. 存活者偏誤：
   C1 列出缺漏：面板各年下市檔數、在 2019-05-16（Asoul 來源截止日）被截斷的已下市股與其流動性。
   C2 補回：本環境 twse.com.tw 被代理擋（403），無法下載 2019-05～2023 已下市股的日 K → 無法真補回。
   C3 代替做法：在挑選期（含已下市股）做「假存活者」實驗：取長度與 2019-05-17～2023-12-29 相同（4.62 年）的視窗，
      把視窗內下市的股票從視窗起點起整段刪掉（與面板在驗證期的缺漏方式相同），比較刪除前後上市等權、T07b、修正合成的
      年化與月超額，得到偏誤的經驗估計；再用它校正驗證期的 t。
D. 檢定力：驗證期月超額的標準差 → 80% 檢定力、雙尾 5% 下可偵測的最小年化超額；以及 0050 下跌月份的表現。
輸出 recheck_r1_0.json（同資料夾）。中間計算不寫入其他檔案。
"""
import sys, json, math, pickle, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tlib as L

OUTF = 'recheck_r1_0.json'
WINS = ['S', 'V_all', 'V_off']
t0 = time.time()


def log(*a):
    print(f'[{time.time()-t0:6.0f}s]', *a, flush=True)


# ---------------- 模擬（可選鍵） ----------------
def simulate_keyed(D, ev, cap, dual):
    """與 tlib.simulate 相同的帳務；dual=True 時以 (rule, j) 為鍵（同一檔可在不同規則下各持一份）。
    另回傳各規則的進場次數。"""
    T = len(D['idx'])
    R = D['R_oo'].to_numpy()
    ev = ev.sort_values(['e', 'strength'], ascending=[True, False], kind='mergesort')
    by_e = {}
    for e, g in ev.groupby('e', sort=False):
        by_e[e] = list(zip(g['j'].to_numpy(), g['x'].to_numpy(), g['src_rule'].to_numpy()))
    held = {}
    gross = np.zeros(T); cost = np.zeros(T); sg = np.zeros(T); sc = np.zeros(T); nh = np.zeros(T, int)
    n_prev = 0; half = L.RT / 2; ent_by = {}
    for d in range(T):
        exits = [k for k, x in held.items() if x == d]
        for k in exits: del held[k]
        entries = 0
        if d in by_e and len(held) < cap:
            for j, x, r in by_e[d]:
                if len(held) >= cap: break
                k = (r, j) if dual else j
                if k in held: continue
                held[k] = x; entries += 1; ent_by[r] = ent_by.get(r, 0) + 1
        n = len(held)
        cost[d] = (half * len(exits) / n_prev if n_prev else 0.0) + (half * entries / n if n else 0.0)
        sc[d] = half * (len(exits) + entries) / cap
        if n:
            cols = [k[1] if dual else k for k in held.keys()]
            r_ = np.nan_to_num(R[d, cols], nan=0.0)
            gross[d] = r_.mean(); sg[d] = r_.sum() / cap
        nh[d] = n; n_prev = n
    pf = pd.DataFrame({'gross': gross, 'cost': cost, 'slot_gross': sg, 'slot_cost': sc, 'n_held': nh}, index=D['idx'])
    return pf, ent_by


def sim_single(D, ev, cap, name):
    e = ev.copy(); e['src_rule'] = name
    return simulate_keyed(D, e, cap, dual=False)


def blend(pa, pb, rebal=True):
    """兩個各半資金的 sleeve。rebal=True：每日再平衡 50/50。False：期初各半，之後不再平衡（用淨值權重）。"""
    if rebal:
        out = 0.5 * pa[['gross', 'cost', 'slot_gross', 'slot_cost']] + 0.5 * pb[['gross', 'cost', 'slot_gross', 'slot_cost']]
    else:
        na, nb = pa['gross'] - pa['cost'], pb['gross'] - pb['cost']
        Ea, Eb = (1 + na).cumprod(), (1 + nb).cumprod()
        wa = (Ea.shift(1, fill_value=1.0) / (Ea.shift(1, fill_value=1.0) + Eb.shift(1, fill_value=1.0)))
        out = pd.DataFrame({c: wa * pa[c] + (1 - wa) * pb[c] for c in ['gross', 'cost', 'slot_gross', 'slot_cost']})
    out['n_held'] = pa['n_held'] + pb['n_held']
    return out


def comp_events(evs, names):
    parts = []
    for k in names:
        e = evs[k].copy()
        e['strength'] = e.groupby('e')['strength'].rank(pct=True)
        e['src_rule'] = k
        parts.append(e)
    return pd.concat(parts, ignore_index=True)


def ps(D, pf, w):
    p = L.port_stats(D, pf, w)
    return {'CAGR': p['net'].get('CAGR'), 'MDD': p['net'].get('maxDD'), 'vol': p['net'].get('vol'), 'CAGR_cost1pct': p['net_cost1pct'],
            't_vs_EW': p['t_vs_EW'], 't_vs_0050': p['t_vs_0050'], 'xs_EW_pct_m': p['excess_vs_EW_pct_m'], 'xs_0050_pct_m': p['excess_vs_0050_pct_m'],
            'EW_CAGR': p['EW'].get('CAGR'), '0050_CAGR': p['0050'].get('CAGR'), 'months': p['months'], 'avg_held': p['avg_held'],
            'years_beat_EW': p['years_beat_EW'], 'yearly_net_pct': p['yearly_net_pct'], 'yearly_EW_pct': p['yearly_EW_pct']}


def trade_n(D, evs_list, w):
    ev = pd.concat([L.dedup(e) for e in evs_list], ignore_index=True)
    return L.trade_stats(D, ev, w).get('n')


def build_volbreak(D):
    C, V = D['AC'], D['value']
    hi60 = C.shift(1).rolling(60, min_periods=60).max()
    vr = V / V.shift(1).rolling(20, min_periods=20).mean()
    vb = (C > hi60) & (vr > 2)
    return {'T07b_volbreak_hold20': L.make_events(D, vb, vr, None, 20), 'T07c_volbreak_hold60': L.make_events(D, vb, vr, None, 60)}


def all_variants(D, evs):
    b, c = evs['T07b_volbreak_hold20'], evs['T07c_volbreak_hold60']
    pb30, _ = sim_single(D, b, 30, 'b'); pc30, _ = sim_single(D, c, 30, 'c')
    pb15, _ = sim_single(D, b, 15, 'b'); pc15, _ = sim_single(D, c, 15, 'c')
    ce = comp_events(evs, ['T07b_volbreak_hold20', 'T07c_volbreak_hold60'])
    pX1, entX1 = simulate_keyed(D, ce, 30, dual=False)
    pd30, ent30 = simulate_keyed(D, ce, 30, dual=True)
    pd60, ent60 = simulate_keyed(D, ce, 60, dual=True)
    pfs = {'T07b_alone': pb30, 'T07c_alone': pc30, 'X1_original': pX1,
           'B1_split15': blend(pb15, pc15), 'B2_split30': blend(pb30, pc30), 'B3_split30_norebal': blend(pb30, pc30, rebal=False),
           'B4_dual30': pd30, 'B5_dual60': pd60}
    ents = {'X1_original': entX1, 'B4_dual30': ent30, 'B5_dual60': ent60}
    return pfs, ents


# ---------------- 假存活者實驗用：遮蔽讀檔 ----------------
_orig_read = pd.read_parquet


def masked_loader(cols, a):
    def rd(path, *args, **kw):
        df = _orig_read(path, *args, **kw)
        cc = [c for c in cols if c in df.columns]
        if cc:
            df = df.copy()
            df.loc[df.index >= a, cc] = np.nan
        return df
    return rd


def window_stats(D, pf, a, b):
    m = (D['idx'] >= a) & (D['idx'] <= b)
    days = D['idx'][m]
    p = pf.reindex(days)
    net = p['gross'] - p['cost']
    ew = D['r_ew'].reindex(days).fillna(0.0)
    mn, me = L.monthly(net), L.monthly(ew)
    return {'CAGR': L.perf(net).get('CAGR'), 'EW_CAGR': L.perf(ew).get('CAGR'), 'xs_EW_pct_m': float((mn - me).mean() * 100), 't_vs_EW': L.tstat(mn - me),
            'EW_m_mean_pct': float(me.mean() * 100)}


def main():
    out = {}
    D = L.load()
    log('loaded')
    with open(L.PKL, 'rb') as f:
        pk = pickle.load(f)
    evs_pk = pk['events']
    b, c = evs_pk['T07b_volbreak_hold20'], evs_pk['T07c_volbreak_hold60']
    # ---------- A. 重現 ----------
    kb = b[['j', 't', 'e']].sort_values(['j', 't']).reset_index(drop=True)
    kc = c[['j', 't', 'e']].sort_values(['j', 't']).reset_index(drop=True)
    same_events = bool(len(kb) == len(kc) and (kb.to_numpy() == kc.to_numpy()).all())
    ev_new = build_volbreak(D)
    same_as_pickle = bool(len(ev_new['T07b_volbreak_hold20']) == len(b) and len(ev_new['T07c_volbreak_hold60']) == len(c))
    ce = comp_events(evs_pk, ['T07b_volbreak_hold20', 'T07c_volbreak_hold60'])
    oX1, pfX1, _ = L.evaluate(D, ce, 'X1_repro')           # 與 t2_composite.py 完全相同的呼叫
    pfb = pk['daily']['T07b_volbreak_hold20']
    diff = float(((pfX1['gross'] - pfX1['cost']) - (pfb['gross'] - pfb['cost'])).abs().max())
    pfs, ents = all_variants(D, evs_pk)
    A = {'T07b_T07c_same_signal_events(j,t,e)': same_events, 'n_events_T07b': int(len(b)), 'n_events_T07c': int(len(c)),
         'events_rebuilt_match_pickle': same_as_pickle,
         'X1_repro': {w: {'n': oX1[w]['trades_all_signals'].get('n'), 'CAGR': oX1[w]['portfolio']['net'].get('CAGR'), 'MDD': oX1[w]['portfolio']['net'].get('maxDD'),
                          't_vs_EW': oX1[w]['portfolio']['t_vs_EW'], 't_vs_0050': oX1[w]['portfolio']['t_vs_0050']} for w in WINS},
         'max_abs_daily_net_diff_X1_vs_T07b': diff,
         'X1_entries_by_rule_full_sample': ents['X1_original']}
    out['A_reproduce'] = A
    log('A', json.dumps(A, ensure_ascii=False))
    # ---------- B. 修正合成 ----------
    nmap = {'T07b_alone': [b], 'T07c_alone': [c], 'X1_original': [L.dedup(ce)], 'B1_split15': [b, c], 'B2_split30': [b, c],
            'B3_split30_norebal': [b, c], 'B4_dual30': [b, c], 'B5_dual60': [b, c]}
    B = {}
    for k, pf in pfs.items():
        B[k] = {w: {**ps(D, pf, w), 'n_trades_signal_level': trade_n(D, nmap[k], w)} for w in WINS}
    B['entries_by_rule'] = {k: {kk: int(vv) for kk, vv in v.items()} for k, v in ents.items()}
    out['B_composites'] = B
    rows = []
    for k in pfs:
        for w in WINS:
            r = B[k][w]
            rows.append({'variant': k, 'win': w, 'n': r['n_trades_signal_level'], 'CAGR': r['CAGR'], 'MDD': r['MDD'], 'CAGR1%': r['CAGR_cost1pct'],
                         'tEW': r['t_vs_EW'], 't0050': r['t_vs_0050'], 'EW': r['EW_CAGR'], '0050': r['0050_CAGR'], 'held': r['avg_held'], 'yrsEW': r['years_beat_EW']})
    print(pd.DataFrame(rows).to_string(index=False))
    print('entries', B['entries_by_rule'])
    # ---------- C1. 缺漏清單 ----------
    C_ = D['close_raw']
    last = C_.apply(lambda s: s.last_valid_index())
    du = set(json.loads((L.P / 'delisted_unknown.json').read_text()))
    cut = pd.Timestamp('2019-05-16')
    trunc = sorted(last[(last == cut) & last.index.isin(du)].index)
    i_cut = D['idx'].get_loc(cut)
    elig_cut = ((D['v60'] >= 1e7) & (D['close_raw'] >= 10)).iloc[i_cut]
    alive_cut = int(D['AC'].iloc[i_cut].notna().sum())
    ew_n = ((D['v60'] >= 1e7) & (D['close_raw'] >= 10)).shift(1, fill_value=False) & D['flag'].isin([0, 1, 2])
    ewn_ser = ew_n.sum(axis=1)
    dl_year = last[last < D['idx'][-1] - pd.Timedelta(days=30)].dt.year.value_counts().sort_index()
    sel_rate = float(dl_year.reindex(range(2009, 2019)).fillna(0).mean())
    C1 = {'panel_delist_count_by_year(last data date)': {int(k): int(v) for k, v in dl_year.items()},
          'panel_delist_2020_2023': int(dl_year.reindex(range(2020, 2024)).fillna(0).sum()),
          'truncated_at_2019-05-16_delisted_stocks': trunc, 'n_truncated': len(trunc),
          'n_truncated_in_EW_pool_at_cut(v60>=1e7&px>=10)': int(elig_cut.reindex(trunc).fillna(False).sum()),
          'n_stocks_with_price_at_cut': alive_cut, 'EW_pool_size_2019-05-16': int(ewn_ser.iloc[i_cut]),
          'EW_pool_size_mean_2019-05_2023-12': float(ewn_ser['2019-05-17':'2023-12-29'].mean()),
          'selection_period_delist_rate_per_yr_2009_2018': sel_rate,
          'expected_delists_in_4.62yr_at_that_rate': round(sel_rate * 4.62, 1),
          'unknown_extra': '2019-05-17 之後才上市、且在 2024 年前就下市的股票，面板完全沒有，檔數無法從現有資料得知（預期很少）',
          'add_back_possible': False,
          'add_back_reason': 'twse.com.tw / mops / tpex / Yahoo / FinMind 在本環境皆被代理拒絕（CONNECT 403）；本機資料（Asoul 截至 2019-05-16、k66 只含抓取時仍上市者）沒有這些股票 2019-05-17 之後的價格'}
    out['C1_missing'] = C1
    log('C1', json.dumps({k: v for k, v in C1.items() if k != 'truncated_at_2019-05-16_delisted_stocks'}, ensure_ascii=False))
    # ---------- C3. 假存活者實驗 ----------
    span = pd.Timestamp('2023-12-29') - pd.Timestamp('2019-05-17')
    starts = [pd.Timestamp(f'{y}-01-01') for y in range(2005, 2015)]
    base = {}
    exp = []
    for a0 in starts:
        a = D['idx'][D['idx'] >= a0][0]
        bnd = a + span
        if bnd > pd.Timestamp('2019-04-30'):
            bnd = pd.Timestamp('2019-04-30')
        drop = list(last[(last > a) & (last <= bnd)].index)
        pd.read_parquet = masked_loader(drop, a)
        try:
            Dm = L.load()
        finally:
            pd.read_parquet = _orig_read
        evm = build_volbreak(Dm)
        pfm, _ = all_variants(Dm, evm)
        rec = {'start': str(a.date()), 'end': str(bnd.date()), 'n_dropped': len(drop),
               'n_dropped_in_EW_pool_at_start': int(((D['v60'] >= 1e7) & (D['close_raw'] >= 10)).loc[a].reindex(drop).fillna(False).sum())}
        for k in ['T07b_alone', 'T07c_alone', 'B1_split15', 'B2_split30', 'B4_dual30']:
            full = window_stats(D, pfs[k], a, bnd)
            surv = window_stats(Dm, pfm[k], a, bnd)
            rec[k] = {'full': full, 'surv': surv,
                      'bias_CAGR': surv['CAGR'] - full['CAGR'], 'bias_EW_CAGR': surv['EW_CAGR'] - full['EW_CAGR'],
                      'bias_xs_EW_pct_m': surv['xs_EW_pct_m'] - full['xs_EW_pct_m'], 'bias_EW_m_pct': surv['EW_m_mean_pct'] - full['EW_m_mean_pct']}
        exp.append(rec)
        log('C3', rec['start'], rec['end'], 'drop', len(drop), 'EWbias', round(rec['B1_split15']['bias_EW_CAGR'], 4),
            'B1 bias', round(rec['B1_split15']['bias_CAGR'], 4), 'xs bias %/m', round(rec['B1_split15']['bias_xs_EW_pct_m'], 3),
            'T07b bias', round(rec['T07b_alone']['bias_CAGR'], 4))
        del Dm
    summ = {}
    for k in ['T07b_alone', 'T07c_alone', 'B1_split15', 'B2_split30', 'B4_dual30']:
        bx = np.array([r[k]['bias_xs_EW_pct_m'] for r in exp]); be = np.array([r[k]['bias_EW_CAGR'] for r in exp]); bc = np.array([r[k]['bias_CAGR'] for r in exp])
        summ[k] = {'bias_EW_CAGR_mean': float(be.mean()), 'bias_EW_CAGR_min_max': [float(be.min()), float(be.max())],
                   'bias_rule_CAGR_mean': float(bc.mean()), 'bias_rule_CAGR_min_max': [float(bc.min()), float(bc.max())],
                   'bias_xs_EW_pct_m_mean': float(bx.mean()), 'bias_xs_EW_pct_m_min_max': [float(bx.min()), float(bx.max())]}
    out['C3_pseudo_survivor'] = {'windows': exp, 'summary': summ, 'n_dropped_mean': float(np.mean([r['n_dropped'] for r in exp]))}
    # ---------- C4. 校正驗證期 t ----------
    # 偏誤只在 2019-05-17～2023-12-29；把該段月超額減去（存活-全樣本）偏誤的「最不利」與「平均」估計，再算 t。
    C4 = {}
    vm = L.win_mask(D, 'V_all'); days = vm[vm].index
    ew = D['r_ew'].reindex(days).fillna(0.0); b50 = D['r0050'].reindex(days).fillna(0.0)
    me, mb = L.monthly(ew), L.monthly(b50)
    affected = np.array([(pd.Timestamp(y, m, 1) >= pd.Timestamp('2019-05-01')) and (pd.Timestamp(y, m, 1) <= pd.Timestamp('2023-12-01')) for y, m in me.index])
    for k in ['T07b_alone', 'B1_split15', 'B2_split30', 'B4_dual30']:
        p = pfs[k].reindex(days); mn = L.monthly(p['gross'] - p['cost'])
        x = (mn - me).to_numpy()
        s = summ[k]
        r = {'t_vs_EW_raw': L.tstat(x), 'xs_EW_pct_m_raw': float(x.mean() * 100), 'n_months': int(len(x)), 'n_affected_months': int(affected.sum())}
        for lab, adj in [('mean', s['bias_xs_EW_pct_m_mean']), ('min', s['bias_xs_EW_pct_m_min_max'][0]), ('max', s['bias_xs_EW_pct_m_min_max'][1])]:
            xa = x - affected * adj / 100
            r[f't_vs_EW_adj_{lab}'] = L.tstat(xa); r[f'xs_EW_pct_m_adj_{lab}'] = float(xa.mean() * 100)
        C4[k] = r
    # EW 本身的校正（年化）：把受影響月份的 EW 月報酬減去平均偏誤
    bew = summ['B1_split15']['bias_EW_CAGR_mean']
    ewm_adj = me.to_numpy() - affected * np.mean([r['B1_split15']['bias_EW_m_pct'] for r in exp]) / 100
    yrs = float(L.SPAN.reindex(days).sum()) / 365.25
    C4['EW_V_all_CAGR_raw'] = L.perf(ew).get('CAGR')
    C4['EW_V_all_CAGR_adj_mean_bias'] = float(np.prod(1 + ewm_adj) ** (1 / yrs) - 1)
    C4['EW_bias_CAGR_mean_in_window'] = bew
    out['C4_adjusted_validation'] = C4
    log('C4', json.dumps(C4, ensure_ascii=False))
    # ---------- D. 檢定力 ----------
    Dd = {}
    for k in ['T07b_alone', 'B1_split15', 'B4_dual30']:
        p = pfs[k].reindex(days); mn = L.monthly(p['gross'] - p['cost'])
        for lab, bm in [('EW', me), ('0050', mb)]:
            x = (mn - bm)
            sd = float(x.std(ddof=1)); n = len(x)
            mde_m = (1.96 + 0.8416) * sd / math.sqrt(n)
            Dd[f'{k}_vs_{lab}'] = {'sd_m_pct': round(sd * 100, 3), 'n': n, 'MDE_monthly_pct_80pow': round(mde_m * 100, 3),
                                   'MDE_annual_pct_approx': round(((1 + mde_m) ** 12 - 1) * 100, 1),
                                   'observed_xs_annual_pct': round(((1 + x.mean()) ** 12 - 1) * 100, 1)}
        down = (mb < 0).to_numpy()
        Dd[f'{k}_0050_down_months'] = {'n_down': int(down.sum()), 'mean_net_pct': float(mn[down].mean() * 100), 'mean_0050_pct': float(mb[down].mean() * 100),
                                       'mean_EW_pct': float(me[down].mean() * 100), 't_vs_EW_down': L.tstat((mn - me)[down]),
                                       'mean_net_up_pct': float(mn[~down].mean() * 100), 'mean_EW_up_pct': float(me[~down].mean() * 100), 't_vs_EW_up': L.tstat((mn - me)[~down])}
    out['D_power'] = Dd
    log('D', json.dumps(Dd, ensure_ascii=False))
    L.save(OUTF, out)
    log('saved', L.OUT / OUTF)


if __name__ == '__main__':
    main()
