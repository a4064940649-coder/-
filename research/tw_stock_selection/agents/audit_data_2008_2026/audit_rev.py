#!/usr/bin/env python3
"""項目 4（月營收兩來源、SURGE 重算）與項目 5（前視檢查）。python3 -I -B audit_rev.py"""
import sys, json, glob, re; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import alib as A
SEED = 20261120
rng = np.random.default_rng(SEED)
out = {'seed': SEED}
# ---------------------------------------------------------------- 來源 1：引擎用 tw_revenue.csv（讀法照規格：同代碼優先上市、≤0 缺值、千元）
r = pd.read_csv('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv', dtype={'code': str}, parse_dates=['month'])
r['pri'] = (r['market'] != 'sii').astype(int)
r1 = r.sort_values(['month', 'code', 'pri']).drop_duplicates(['month', 'code'], keep='first')
Rv = r1.pivot(index='month', columns='code', values='revenue_k_twd').sort_index(); Rv.index = pd.PeriodIndex(Rv.index, freq='M')
Rv = Rv.reindex(pd.period_range(Rv.index.min(), Rv.index.max(), freq='M')); Rv = Rv.where(Rv > 0)
out['src1'] = dict(rows=int(len(r)), first=str(Rv.index[0]), last=str(Rv.index[-1]), dup_month_code=int(r.duplicated(['month', 'code', 'market']).sum()))
# ---------------------------------------------------------------- 來源 2：官方倉庫 mops/revenue_hist（2015-01 起；去年當月營收 → 2014）
fr = []
for f in sorted(glob.glob(str(A.RS / 'raw/jb7k6295r7-bot_tw-stock-data/data/mops/revenue_hist/*_*.csv'))):
    mkt = f.rsplit('_', 1)[1][:-4]
    if mkt not in ('twse', 'tpex'): continue
    x = pd.read_csv(f, dtype={'stock_id': str}, usecols=['stock_id', 'period', '當月營收', '去年當月營收']); x['mkt'] = mkt; fr.append(x)
o = pd.concat(fr, ignore_index=True); o['ym'] = pd.PeriodIndex(o['period'], freq='M')
o = o.sort_values(['ym', 'stock_id', 'mkt']).drop_duplicates(['ym', 'stock_id'], keep='last' )
O2 = o.pivot(index='ym', columns='stock_id', values='當月營收'); OL = o.pivot(index='ym', columns='stock_id', values='去年當月營收'); OL.index = OL.index - 12
out['src2'] = dict(rows=int(len(o)), first=str(O2.index[0]), last=str(O2.index[-1]))
# 來源 3：FinMind（hawan0228，113 檔，2015 起，單位元）；來源 4：stevenhsieh rev_5y（千元）；來源 5：taiwanan（2015-01～2019-08）
fr = []
for f in glob.glob('/home/user/hawan0228/fintech_ai/data/external/raw/revenue/*_revenue.csv'):
    fr.append(pd.read_csv(f, dtype={'stock_id': str}, encoding='utf-8-sig', usecols=['stock_id', 'revenue', 'revenue_month', 'revenue_year']))
h = pd.concat(fr); h['ym'] = pd.PeriodIndex(h['revenue_year'].astype(str) + '-' + h['revenue_month'].astype(str).str.zfill(2), freq='M')
H3 = h.drop_duplicates(['ym', 'stock_id'], keep='last').pivot(index='ym', columns='stock_id', values='revenue') / 1000.0
s5 = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/rev_5y.csv.gz', dtype={'code': str}); s5['ym'] = pd.PeriodIndex(s5['ym'], freq='M')
S4 = s5.drop_duplicates(['ym', 'code'], keep='last').pivot(index='ym', columns='code', values='revenue')
tn = pd.read_csv(A.RS / 'agents/o_rev_survivorship/src_taiwanan.csv', dtype={'code': str}); tn['ym'] = pd.PeriodIndex(tn['ym'], freq='M')
T5 = tn.drop_duplicates(['ym', 'code'], keep='last').pivot(index='ym', columns='code', values='rev')
ALT = {'官方倉庫revenue_hist': O2, '官方倉庫revenue_hist（次年檔的去年當月營收）': OL, 'FinMind(hawan)': H3, 'stevenhsieh_rev_5y': S4, 'taiwanan': T5}
def lookup(code, m):
    res = {}
    for k, X in ALT.items():
        if code in X.columns and m in X.index and pd.notna(X.at[m, code]):
            res[k] = float(X.at[m, code])
    return res
# ---------------------------------------------------------------- 4a 分層隨機抽 60（上市、營收 > 0；每年 3 個＋全期另抽 3 個）
sii = r1[(r1['market'] == 'sii') & (r1['revenue_k_twd'] > 0) & (r1['month'] >= '2008-01-01')].copy(); sii['year'] = sii['month'].dt.year
pick = []
for y in range(2008, 2027):
    g = sii[sii['year'] == y]; pick += list(g.index[rng.choice(len(g), 3, replace=False)])
rest = sii.drop(index=pick); pick += list(rest.index[rng.choice(len(rest), 3, replace=False)])
samp = []
for i in pick:
    x = sii.loc[i]; m = pd.Period(x['month'], 'M'); alt = lookup(x['code'], m); v = float(x['revenue_k_twd'])
    prim = {k: a for k, a in alt.items() if '去年當月' not in k}        # 當期公布值的來源；「次年檔的去年當月營收」是事後重編值，只在沒有別的來源時才拿來比
    use = prim or alt
    rel = {k: (a / v - 1) for k, a in use.items()}
    rs = {k: (a / v - 1) for k, a in alt.items() if '去年當月' in k}
    samp.append(dict(code=x['code'], name=x['name'], month=str(m), src1=v, alt=alt, n_alt=len(alt), n_primary_alt=len(prim), max_abs_rel_diff=max([abs(z) for z in rel.values()], default=None),
                     restated_rel_diff=(list(rs.values())[0] if rs else None), only_restated=bool(alt and not prim),
                     verdict=('無法驗證（本地沒有第二來源）' if not alt else ('通過' if max(abs(z) for z in rel.values()) < 1e-6 else '不一致'))))
S = pd.DataFrame(samp)
out['sample60'] = dict(n=len(S), verdict=S['verdict'].value_counts().to_dict(), by_year={int(y): g['verdict'].value_counts().to_dict() for y, g in S.groupby(S['month'].str[:4].astype(int))}, rows=samp)
print(S.drop(columns=['alt']).to_string())
# ---------------------------------------------------------------- 4b 普查：上市 2014-01～2026-09，來源 1 對官方倉庫
cen = {}
for nm, X in (('官方倉庫revenue_hist', O2), ('官方倉庫revenue_hist（次年檔的去年當月營收）', OL), ('FinMind(hawan)', H3), ('stevenhsieh_rev_5y', S4), ('taiwanan', T5)):
    cc = [c for c in Rv.columns if c in X.columns]; ii = Rv.index.intersection(X.index)
    a = Rv.loc[ii, cc]; b = X.loc[ii, cc]; both = a.notna() & b.notna() & (b > 0)
    rel = (b / a - 1).where(both); ad = rel.abs()
    st = ad.stack()
    cen[nm] = dict(first=str(ii.min()), last=str(ii.max()), n_both=int(both.sum().sum()), exact=int((st < 1e-6).sum()), n_diff_gt_1e6=int((st >= 1e-6).sum()), n_diff_gt_1pct=int((st > 0.01).sum()), n_diff_gt_10pct=int((st > 0.10).sum()),
                   match_rate=float((st < 1e-6).sum() / max(int(both.sum().sum()), 1)), src1_only=int((a.notna() & ~b.notna()).sum().sum()), alt_only=int((b.notna() & (b > 0) & ~a.notna()).sum().sum()))
    cen[nm]['by_year_diff_gt_1e6'] = {int(y): int(v) for y, v in (ad >= 1e-6).sum(axis=1).groupby(ad.index.year).sum().items()}
out['census'] = cen
print(json.dumps(cen, ensure_ascii=False, indent=1))
# ---------------------------------------------------------------- 4c SURGE 重算（全部交易；V_BRIDGE 的 2013～2015 用 B2 規則）
def jump_n(y, n):
    P_ = lambda t: pd.Period(t, 'M')
    a = Rv.loc[P_(f'{y}-01'):P_(f'{y}-{n:02d}')]; b = Rv.loc[P_(f'{y - 1}-{13 - n:02d}'):P_(f'{y - 1}-12')]
    return np.log(a.sum(min_count=n) / b.sum(min_count=n))
KN = {}
for n in (1, 2, 3):
    pri = pd.concat([jump_n(y, n) for y in (2008, 2009, 2010, 2011, 2012)], axis=1)
    a = jump_n(2013, n) - pri.median(axis=1, skipna=True); a[pri.notna().sum(axis=1) < 3] = np.nan
    KN[n] = np.exp(a.clip(lower=0))
BREAK = pd.Period('2013-01', 'M')
def surge(code, m, rule):
    """SURGE＝(D[m] − 前 24 個月 D 平均) ÷ 前 24 個月 D 樣本標準差；D＝營收 − 12 個月前營收。只用到 m（含）以前的營收。回傳 (z, 用到的最晚月份, D, μ, σ)。"""
    s = Rv[code].loc[:m].copy()                                # 截到 m：之後的資料根本不給
    if rule == 'B2' and m.year >= 2013:
        n = 1 if m == pd.Period('2013-01', 'M') else 2 if m == pd.Period('2013-02', 'M') else 3
        k = KN[n].get(code, np.nan)
        s2 = s.copy(); s2[s2.index < BREAK] = s2[s2.index < BREAK] * (k if np.isfinite(k) else 1.0)
        D = s2 - s2.shift(12)
        if not np.isfinite(k): D[D.index.year == 2013] = np.nan
    else:
        D = s - s.shift(12); D[D.index.year == 2013] = np.nan
    w = D.iloc[-25:-1].to_numpy(float); d = D.iloc[-1]
    if len(w) < 24 or not np.isfinite(w).all() or not np.isfinite(d): return np.nan, None, d, np.nan, np.nan
    mu = w.mean(); sd = np.sqrt(((w - mu) ** 2).sum() / 23)
    if not sd > 1e-9 * (np.abs(w).mean() + 1e-12): return np.nan, None, d, mu, sd
    return float((d - mu) / sd), str(s.dropna().index.max()), float(d), float(mu), float(sd)
sig = pd.read_csv(A.S75 / 'signals.csv', dtype=str); sig['sig_date'] = pd.to_datetime(sig['sig_date'])
s2m = {(v, d): pd.Period(m, 'M') for v, d, m in zip(sig['version'], sig['sig_date'], sig['month'])}
T = A.trades()
rows = []
for t in T.itertuples():
    m = s2m[(t.version, t.sig_date)]
    z, last, d, mu, sd = surge(t.code, m, 'B2' if t.version == 'V_BRIDGE' else 'MASK')
    rows.append(dict(version=t.version, code=t.code, sig_date=t.sig_date, month=str(m), score=t.score, z=z, diff=(z - t.score) if z == z else np.nan, rev_m=float(Rv.at[m, t.code]), rev_m12=float(Rv.at[m - 12, t.code]),
                     alt_m=lookup(t.code, m), alt_m12=lookup(t.code, m - 12)))
Z = pd.DataFrame(rows)
Z.drop(columns=['alt_m', 'alt_m12']).to_csv(A.HERE / 'surge_recalc.csv', index=False)
out['surge_recalc'] = dict(n=int(len(Z)), n_computed=int(Z['z'].notna().sum()), max_abs_diff=float(Z['diff'].abs().max()), n_diff_gt_1e6=int((Z['diff'].abs() > 1e-6).sum()), n_nan=int(Z['z'].isna().sum()),
                           by_version={v: dict(n=int(len(g)), max_abs_diff=float(g['diff'].abs().max()), n_bad=int((g['diff'].abs() > 1e-6).sum()) + int(g['z'].isna().sum())) for v, g in Z.groupby('version')},
                           b2_rows=int(((Z['version'] == 'V_BRIDGE') & (Z['month'] >= '2013-01') & (Z['month'] <= '2015-12')).sum()))
print(out['surge_recalc'])
# 交易用到的營收格（當月與 12 個月前）對第二來源
def cell_ok(v, alt):
    prim = {k: a for k, a in alt.items() if '去年當月' not in k and 'FinMind' not in k and 'taiwanan' not in k} or {k: a for k, a in alt.items() if '去年當月' not in k} or alt
    return None if not alt else bool(max(abs(a / v - 1) for a in prim.values()) < 1e-6)
u = Z.drop_duplicates(['code', 'month'])
cm = [cell_ok(a, b) for a, b in zip(u['rev_m'], u['alt_m'])]; c12 = [cell_ok(a, b) for a, b in zip(u['rev_m12'], u['alt_m12'])]
out['trade_rev_cells'] = dict(unique_code_months=int(len(u)), month_m=dict(checked=int(sum(x is not None for x in cm)), ok=int(sum(x is True for x in cm)), no_second_source=int(sum(x is None for x in cm))),
                              month_m12=dict(checked=int(sum(x is not None for x in c12)), ok=int(sum(x is True for x in c12)), no_second_source=int(sum(x is None for x in c12))),
                              mismatches=[dict(code=a.code, month=a.month, rev_m=a.rev_m, alt_m=a.alt_m, rev_m12=a.rev_m12, alt_m12=a.alt_m12) for a, x, y in zip(u.itertuples(), cm, c12) if x is False or y is False][:40])
print(json.dumps(out['trade_rev_cells'], ensure_ascii=False)[:3000])
win = []
for c, e in [('2408', '2025-09-12'), ('3653', '2026-08-12'), ('3017', '2025-05-14'), ('2344', '2026-05-13'), ('2059', '2026-06-12'), ('1325', '2020-03-12'), ('2344', '2025-11-12')]:
    t = T[(T['code'] == c) & (T['ent_date'] == e) & (T['version'] == 'V_BRIDGE')].iloc[0]; m = s2m[('V_BRIDGE', t.sig_date)]
    z, last, d, mu, sd = surge(c, m, 'B2')
    win.append(dict(code=c, ent=e, sig_date=str(t.sig_date.date()), rev_month=str(m), rev_m_k=float(Rv.at[m, c]), rev_m12_k=float(Rv.at[m - 12, c]), yoy_pct=float(Rv.at[m, c] / Rv.at[m - 12, c] - 1) * 100, D=d, mu24=mu, sd24=sd, surge_recalc=z, score_in_trades=float(t.score),
                    abs_diff=abs(z - t.score), latest_month_used=last, rev_m_other_sources=lookup(c, m), rev_m12_other_sources=lookup(c, m - 12),
                    window_cells_checked=int(sum(1 for q in range(0, 37) if lookup(c, m - q))), window_cells_match=int(sum(1 for q in range(0, 37) if lookup(c, m - q) and max(abs(a / Rv.at[m - q, c] - 1) for a in lookup(c, m - q).values()) < 1e-6))))
out['winners_surge'] = win
for w in win: print(w)
# ---------------------------------------------------------------- 項目 5：前視檢查（隨機 40 個訊號月，V_BRIDGE）
D_ = A.Data(set(T['code']) | set(' '.join(sig['targets'].fillna('')).split()))
IDX = D_.IDX; pos = {d: i for i, d in enumerate(IDX)}
ext = pd.read_csv(A.RS / 'raw/rev_deadline_exceptions/deadline_exceptions.csv', dtype=str); ext = {pd.Period(a, 'M'): pd.Timestamp(b) for a, b, c in zip(ext['rev_month'], ext['actual_deadline'], ext['apply']) if c.strip() == '1'}
ins = pd.read_csv(A.RS / 'raw/rev_deadline_exceptions/ins15_list.csv', dtype=str).fillna('')
sb = sig[(sig['version'] == 'V_BRIDGE')].reset_index(drop=True)
# 全部訊號月先做日期檢查
dates_all = []
for x in sb.itertuples():
    m = pd.Period(x.month, 'M'); d10 = (m + 1).to_timestamp() + pd.Timedelta(days=9)
    dl = max(d10, ext[m]) if m in ext else d10
    dl_td = IDX[IDX >= dl][0]                                   # 期限日遇非交易日順延
    nxt = IDX[IDX > dl_td][0]
    dates_all.append(dict(month=x.month, sig_date=x.sig_date, deadline=dl, deadline_rolled=dl_td, weekday_10th=int(d10.weekday()), sig_after_deadline=bool(x.sig_date > dl_td), sig_is_next_trading_day=bool(x.sig_date == nxt),
                          days_after=int((x.sig_date - dl_td).days), exception=bool(m in ext)))
DA = pd.DataFrame(dates_all)
out['item5_dates_all'] = dict(n=int(len(DA)), sig_after_deadline=int(DA['sig_after_deadline'].sum()), sig_is_next_trading_day=int(DA['sig_is_next_trading_day'].sum()), exceptions_hit=DA[DA['exception']][['month', 'sig_date', 'deadline']].astype(str).to_dict('records'),
                              bad=DA[~DA['sig_after_deadline']].astype(str).to_dict('records'), not_next=DA[~DA['sig_is_next_trading_day']].astype(str).to_dict('records')[:10])
nz = sb[sb['n_tgt'].astype(int) > 0]
ch = nz.iloc[np.sort(rng.choice(len(nz), 40, replace=False))]
TB = T[T['version'] == 'V_BRIDGE']
res = []; stock_rows = []
for x in ch.itertuples():
    m = pd.Period(x.month, 'M'); k = pos[x.sig_date]; da = DA[DA['month'] == x.month].iloc[0]
    tg = x.targets.split()
    ok_all = True
    for c in tg:
        f = D_.frame(c)
        z, last, d, mu, sd = surge(c, m, 'B2')
        tr = TB[(TB['code'] == c) & (TB['sig_date'] == x.sig_date)]
        G = np.cumprod(1 + f['R'].to_numpy()[:k + 1])              # 只用到訊號日（含）以前
        ma60 = G[k - 59:k + 1].mean(); ret20 = G[k] / G[k - 20] - 1
        amt = np.where(f['mk'].to_numpy()[:k + 1] > 0, np.nan_to_num(f['AMT'].to_numpy()[:k + 1]), np.nan); v60 = amt[k - 59:k + 1].mean()
        c_ok = bool(f['C'].iat[k] >= 10 and f['has'].iat[k]); 
        # 對照：若偷用隔天收盤，條件是否仍成立（只做參考）
        ins_hit = ins[(ins['code'] == c)]
        ins_block = False
        if len(ins_hit) and m >= pd.Period(ins_hit['start_month'].iloc[0], 'M'):
            d15 = IDX[IDX >= (m + 1).to_timestamp() + pd.Timedelta(days=int(ins_hit['deadline_day'].iloc[0]) - 1)][0]
            ins_block = bool(x.sig_date < d15)
        row = dict(month=x.month, sig_date=str(x.sig_date.date()), code=c, surge=z, latest_rev_month_used=last, rev_month_le_m=bool(last is not None and pd.Period(last, 'M') <= m), surge_finite=bool(z == z),
                   score_in_trades=float(tr['score'].iloc[0]) if len(tr) else None, score_match=(abs(z - float(tr['score'].iloc[0])) < 1e-6) if len(tr) else None,
                   above_ma60_at_sig=bool(G[k] > ma60), ret20_at_sig=float(ret20), ret20_gt_10pct=bool(ret20 > 0.10), v60=float(v60), liq_ok=bool(v60 >= 1e7), close_ok=c_ok, twse=bool(f['mk'].iat[k] == 1),
                   bought=bool(len(tr)), ent_is_next_day=bool(len(tr) and pos[tr['ent_date'].iloc[0]] == k + 1) if len(tr) else None, ins15_violation=ins_block)
        row['all_ok'] = bool(row['rev_month_le_m'] and row['surge_finite'] and row['above_ma60_at_sig'] and row['ret20_gt_10pct'] and row['liq_ok'] and row['close_ok'] and row['twse'] and not ins_block and (row['score_match'] in (None, True)) and (row['ent_is_next_day'] in (None, True)))
        ok_all &= row['all_ok']; stock_rows.append(row)
    res.append(dict(month=x.month, sig_date=str(x.sig_date.date()), deadline=str(da['deadline_rolled'].date()), sig_after_deadline=bool(da['sig_after_deadline']), n_targets=len(tg), all_ok=bool(ok_all and da['sig_after_deadline'])))
SR = pd.DataFrame(stock_rows); SR.to_csv(A.HERE / 'lookahead_stocks.csv', index=False)
out['item5'] = dict(n_months=len(res), months_all_ok=int(sum(r_['all_ok'] for r_ in res)), n_stock_rows=int(len(SR)), stock_rows_ok=int(SR['all_ok'].sum()),
                    fails={c: int((~SR[c]).sum()) for c in ('rev_month_le_m', 'surge_finite', 'above_ma60_at_sig', 'ret20_gt_10pct', 'liq_ok', 'close_ok', 'twse')},
                    ins15_violation=int(SR['ins15_violation'].sum()), bought=int(SR['bought'].sum()), score_match=int((SR['score_match'] == True).sum()), ent_is_next_day=int((SR['ent_is_next_day'] == True).sum()),
                    months=res, bad_rows=SR[~SR['all_ok']].to_dict('records'))
print(json.dumps({k2: v for k2, v in out['item5'].items() if k2 != 'months'}, ensure_ascii=False, indent=1, default=str)[:3000]); print(json.dumps(out['item5_dates_all'], ensure_ascii=False, default=str)[:1500])
(A.HERE / 'results_rev.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
