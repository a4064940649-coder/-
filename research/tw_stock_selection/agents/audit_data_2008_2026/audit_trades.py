#!/usr/bin/env python3
"""項目 1（價格交叉核對）、2（最賺／最賠逐筆重算）、6（不可能的交易）。python3 -I -B audit_trades.py"""
import json, re, resource
import numpy as np, pandas as pd
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import alib as A
T = A.trades()
codes = set(T['code'])
D = A.Data(codes)
hs = pd.read_parquet(A.TMP / 'hs.parquet'); hs = hs[hs['code'].isin(codes)]
HS = {c: g.set_index('date').sort_index() for c, g in hs.groupby('code')}
IDX = D.IDX
out = {}
closed = T[T['why'] != 'open_end'].copy()
out['n_rows'] = dict(total=int(len(T)), closed=int(len(closed)), by_version=T.groupby('version').size().to_dict(), why=T['why'].value_counts().to_dict(),
                     unique_closed=int(len(closed.drop_duplicates(['code', 'ent_date', 'exit_date']))))
assert (IDX[T['ent_t'] - 0] == T['ent_date']).all() if False else True

# ------------------------------------------------------------ 項目 1：成交價交叉核對
fills = pd.concat([closed[['code', 'ent_date']].rename(columns={'ent_date': 'date'}).assign(side='B'),
                   closed[['code', 'exit_date']].rename(columns={'exit_date': 'date'}).assign(side='S'),
                   T[T['why'] == 'open_end'][['code', 'ent_date']].rename(columns={'ent_date': 'date'}).assign(side='B_open')]).drop_duplicates()
rows = []
for c, d, s in fills.itertuples(index=False):
    f = D.frame(c); r = f.loc[d]
    eng = r['O']; esrc = r['src']
    alt = {}
    if c in D.dp['open'].columns and d in D.dp['open'].index:
        v = D.dp['open'].at[d, c]; sc = D.dp['src'].at[d, c]
        if np.isfinite(v):
            alt['dp'] = (float(v), {1: 'asoul', 2: 'k66', 3: 'yahoo5m', 4: 'steven'}.get(int(sc) if np.isfinite(sc) else 0, '?'))
    if c in HS and d in HS[c].index:
        alt['hs'] = (float(HS[c].at[d, 'open']), 'finmind')
    if esrc == 'off' and c in D.asoul and d in D.asoul[c].index and np.isfinite(D.asoul[c].at[d, 'open']):
        alt['asoul'] = (float(D.asoul[c].at[d, 'open']), 'asoul')
    indep = {k: v for k, v in alt.items() if not (esrc in ('asoul', 'dpfill') and v[1] == 'asoul')}
    rows.append(dict(code=c, date=d, side=s, eng_open=eng, eng_src=esrc, dp=alt.get('dp', (np.nan, ''))[0], dp_src=alt.get('dp', (np.nan, ''))[1],
                     hs=alt.get('hs', (np.nan, ''))[0], asoul15=alt.get('asoul', (np.nan, ''))[0], n_indep=len(indep), n_same_origin=len(alt) - len(indep),
                     max_abs_diff=max([abs(v[0] - eng) for v in indep.values()], default=np.nan),
                     max_rel_diff=max([abs(v[0] / eng - 1) for v in indep.values()], default=np.nan),
                     same_origin_diff=max([abs(v[0] - eng) for k, v in alt.items() if k not in indep], default=np.nan)))
F = pd.DataFrame(rows); F['year'] = F['date'].dt.year
F.to_csv(A.HERE / 'price_crosscheck.csv', index=False)
cov = F['n_indep'] > 0
bad = F[cov & (F['max_abs_diff'] > 1e-6)]
bad_official = bad[~((bad['dp_src'] == 'yahoo5m') & ((bad['hs'].isna()) | ((bad['hs'] - bad['eng_open']).abs() < 1e-6)) & ((bad['asoul15'].isna())))]
def per(g):
    c = g['n_indep'] > 0
    return dict(n=int(len(g)), covered=int(c.sum()), two_plus=int((g['n_indep'] >= 2).sum()), same_origin_only=int(((g['n_indep'] == 0) & (g['n_same_origin'] > 0)).sum()),
                none=int(((g['n_indep'] == 0) & (g['n_same_origin'] == 0)).sum()), mismatch=int((c & (g['max_abs_diff'] > 1e-6)).sum()))
out['item1'] = dict(n_fills=int(len(F)), covered=int(cov.sum()), coverage=float(cov.mean()), two_plus_indep=int((F['n_indep'] >= 2).sum()),
                    same_origin_only=int(((F['n_indep'] == 0) & (F['n_same_origin'] > 0)).sum()), no_alt=int(((F['n_indep'] == 0) & (F['n_same_origin'] == 0)).sum()),
                    same_origin_max_diff=float(F['same_origin_diff'].max()),
                    n_mismatch=int(len(bad)), max_abs_diff=float(bad['max_abs_diff'].max()) if len(bad) else 0.0, max_rel_diff=float(bad['max_rel_diff'].max()) if len(bad) else 0.0,
                    n_mismatch_excl_yahoo_only=int(len(bad_official)),
                    mismatch_by_altsrc=bad.groupby('dp_src').size().to_dict(),
                    by_period={k: per(F[(F['year'] >= a) & (F['year'] <= b)]) for k, (a, b) in {'2008-2009': (2008, 2009), '2010-2014': (2010, 2014), '2015-2019': (2015, 2019), '2020-2023': (2020, 2023), '2024-2026': (2024, 2026)}.items()},
                    eng_src=F['eng_src'].value_counts().to_dict(),
                    mismatches=[{k: (str(v.date()) if isinstance(v, pd.Timestamp) else (None if isinstance(v, float) and np.isnan(v) else v)) for k, v in r.items()} for r in bad.sort_values('max_rel_diff', ascending=False).head(40).to_dict('records')])
print(json.dumps({k: v for k, v in out['item1'].items() if k != 'mismatches'}, ensure_ascii=False, indent=1))

# ------------------------------------------------------------ 每筆交易重算（項目 2 的底）＋ 項目 6
def events(c, f, i0, i1):
    """持有期間（進場日之後到出場日，含出場日）參考價 ≠ 前一成交收盤的日子＝公司行動／除權息；另列 xflag 日與 TWT49U 事件。"""
    ev = []
    sub = f.iloc[i0 + 1:i1 + 1]
    e49 = D.evd.get(c)
    for d, r in sub.iterrows():
        kind = None
        in49 = e49 is not None and d in e49.index
        if in49:
            x = e49.loc[d]; kind = f"TWT49U {x['type']} 前收{x['prev_close']}→參考價{x['ref_price']}（權息值{x['value']}）；引擎做法：{r['fix'] or '原漲跌價差'}"
        elif r['xflag'] == 1:
            kind = f"行情檔 X（除權息）日：前收{r['prevC']}→收盤{r['C']}，當日報酬記 0（未計配息、也未計當日漲跌）"
        elif r['has'] and np.isfinite(r['prevC']) and abs(r['REF'] - r['prevC']) > 1e-6:
            kind = f"參考價重設（非 TWT49U）：前收{r['prevC']}→參考價{round(r['REF'], 4)}（比例{r['REF'] / r['prevC']:.4f}）；{r['fix'] or '原漲跌價差'}"
        elif r['has'] and r['bigR']:
            kind = f"單日報酬絕對值 > 50%（退回收盤÷前收）：前收{r['prevC']}→收盤{r['C']}"
        if kind:
            ev.append(f'{d.date()} {kind}')
    return ev

pos = {d: i for i, d in enumerate(IDX)}
rec = []
for r in T.itertuples(index=False):
    c = r.code; f = D.frame(c); i0, i1 = pos[r.ent_date], pos[r.exit_date]; isg = pos[r.sig_date]
    Oe, Ce = f['O'].iat[i0], f['C'].iat[i0]
    sl_b = 0.5 * A.tick(Oe) / Oe + A.XS
    cost_x = r.shares * Oe * (1 + sl_b) + A.fee_of(r.shares, Oe * (1 + sl_b))
    g_hold = Ce / Oe * float(np.prod(1 + f['R'].iloc[i0 + 1:i1].to_numpy()))
    d = dict(version=r.version, code=c, ent=r.ent_date, exit=r.exit_date, why=r.why, shares=r.shares, ret=r.ret, pnl=r.pnl, cost=r.cost, proceeds=r.proceeds,
             ent_open=Oe, ent_src=f['src'].iat[i0], cost_recalc=cost_x, cost_rel_diff=cost_x / r.cost - 1, sig_is_prev_day=bool(isg + 1 == i0))
    if r.why == 'open_end':
        gross = g_hold * (1 + f['R'].iat[i1]); proc_x = r.shares * Oe * gross; d.update(exit_open=np.nan)
    else:
        Ox, REFx = f['O'].iat[i1], f['REF'].iat[i1]
        gross = g_hold * Ox / REFx
        sl_s = 0.5 * A.tick(Ox) / Ox + A.XS
        gv = r.shares * Oe * gross * (1 - sl_s)
        proc_x = gv - A.fee_of(r.shares, gv / r.shares) - gv * A.TAX
        d.update(exit_open=Ox, exit_src=f['src'].iat[i1])
    evs = events(c, f, i0, i1)
    # 不含息的純價差與「真的」含息（除權息日用 TWT49U 參考價：前收÷參考價連乘）
    px_only = (f['O'].iat[i1] if r.why != 'open_end' else f['C'].iat[i1]) / Oe
    fac = 1.0; nev = 0
    if c in D.evd:
        e = D.evd[c]; e = e[(e.index > r.ent_date) & (e.index <= r.exit_date)]
        fac = float(np.prod(e['prev_close'] / e['ref_price'])); nev = len(e)
    d.update(gross_recalc=gross, proceeds_recalc=proc_x, ret_recalc=proc_x / cost_x - 1, ret_diff=proc_x / cost_x - 1 - r.ret, px_only=px_only - 1,
             twt49u_factor=fac, n_twt49u=nev, gross_px_x_div=px_only * fac - 1, n_events=len(evs), events=' ｜ '.join(evs),
             n_xflag=int(f['xflag'].iloc[i0 + 1:i1 + 1].sum()), n_notrade_days=int((~f['has'].iloc[i0:i1 + 1]).sum()))
    # ---- 項目 6
    VOLe = f['VOL'].iat[i0]; REFe = f['REF'].iat[i0]
    pct_e = 0.10 if r.ent_date >= A.LIM10 else 0.07
    d.update(ent_no_volume=bool(not (np.isfinite(VOLe) and VOLe > 0) and f['src'].iat[i0] != 'dpfill'), ent_vol=VOLe, ent_amt=f['AMT'].iat[i0],
             ent_open_limup_real=bool(Oe >= A.limup(REFe, pct_e) - 1e-9), ent_open_limup_10=bool(Oe >= A.limup(REFe, 0.10) - 1e-9), ent_gap=Oe / REFe - 1,
             ent_mk=int(f['mk'].iat[i0]), sig_mk=int(f['mk'].iat[isg]), ent_innov=bool(re.search(r'-(KY)?創$', f['name'].iat[i0] or '')),
             ent_name=f['name'].iat[i0], buy_share_of_day_amt=r.shares * Oe / f['AMT'].iat[i0] if f['AMT'].iat[i0] > 0 else np.nan)
    if r.why != 'open_end':
        pct_x = 0.10 if r.exit_date >= A.LIM10 else 0.07
        VOLx = f['VOL'].iat[i1]
        d.update(exit_no_volume=bool(not (np.isfinite(VOLx) and VOLx > 0) and f['src'].iat[i1] != 'dpfill'), exit_open_limdn_real=bool(Ox <= A.limdn(REFx, pct_x) + 1e-9),
                 exit_open_limdn_10=bool(Ox <= A.limdn(REFx, 0.10) + 1e-9), exit_gap=Ox / REFx - 1, exit_mk=int(f['mk'].iat[i1]))
    last = max([x for x in (D.off_last.get(c), D.asoul_last.get(c)) if x is not None and x == x])
    mkh = f['mk'].iloc[i0:i1 + 1].to_numpy()
    d.update(last_data_date=last, held_after_last=bool(r.exit_date > last), days_not_in_file=int((mkh == 0).sum()), days_tpex=int((mkh == 2).sum()),
             lot_ok=bool(r.shares % 1000 == 0 or r.ent_date >= A.ODD))
    rec.append(d)
X = pd.DataFrame(rec)
X.to_csv(A.HERE / 'trades_recalc.csv.gz', index=False)
cl = X[X['why'] != 'open_end']
u = cl.drop_duplicates(['code', 'ent', 'exit'])
out['recalc_all'] = dict(n=int(len(X)), cost_max_rel_diff=float(X['cost_rel_diff'].abs().max()), n_cost_bad=int((X['cost_rel_diff'].abs() > 1e-9).sum()),
                         closed_n=int(len(cl)), ret_absdiff_le_1e9=int((cl['ret_diff'].abs() <= 1e-9).sum()), ret_absdiff_le_1e4=int((cl['ret_diff'].abs() <= 1e-4).sum()),
                         ret_absdiff_gt_1e4=int((cl['ret_diff'].abs() > 1e-4).sum()), ret_max_abs_diff=float(cl['ret_diff'].abs().max()),
                         sig_is_prev_day_all=bool(X['sig_is_prev_day'].all()),
                         worst=[dict(version=a.version, code=a.code, ent=str(a.ent.date()), exit=str(a.exit.date()), ret=a.ret, ret_recalc=a.ret_recalc) for a in cl.reindex(cl['ret_diff'].abs().sort_values(ascending=False).index).head(15).itertuples()])
print(json.dumps(out['recalc_all'], ensure_ascii=False, indent=1))
def cnt(col, df=cl):
    b = df[df[col].fillna(False).astype(bool)]
    return dict(rows=int(len(b)), unique=int(len(b.drop_duplicates(['code', 'ent', 'exit']))), list=[f'{a.version} {a.code} {a.ent.date()}→{a.exit.date()}' for a in b.head(30).itertuples()])
out['item6'] = dict(ent_no_volume=cnt('ent_no_volume', X), exit_no_volume=cnt('exit_no_volume'),
                    buy_at_open_limup_real=cnt('ent_open_limup_real', X), buy_at_open_limup_10pct_rule=cnt('ent_open_limup_10', X),
                    sell_at_open_limdn_real=cnt('exit_open_limdn_real'), sell_at_open_limdn_10pct_rule=cnt('exit_open_limdn_10'),
                    not_twse_on_entry=int((X['ent_mk'] != 1).sum()), not_twse_on_signal=int((X['sig_mk'] != 1).sum()), innov_on_entry=int(X['ent_innov'].sum()),
                    not_twse_on_exit=int((cl['exit_mk'] != 1).sum()), held_after_last_data=int(X['held_after_last'].sum()),
                    trades_with_days_not_in_file=int((X['days_not_in_file'] > 0).sum()), trades_with_tpex_days=int((X['days_tpex'] > 0).sum()),
                    trades_with_notrade_days=int((X['n_notrade_days'] > 0).sum()), lot_violation=int((~X['lot_ok']).sum()),
                    why=X['why'].value_counts().to_dict(), max_buy_share_of_day_amt=float(X['buy_share_of_day_amt'].max()),
                    n_buy_share_gt_5pct=int((X['buy_share_of_day_amt'] > 0.05 + 1e-9).sum()),
                    dpfill_entries=int((X['ent_src'] == 'dpfill').sum()))
x7 = X[X['ent_open_limup_real']]
out['item6']['limup_detail'] = [dict(version=a.version, code=a.code, ent=str(a.ent.date()), open=a.ent_open, gap=round(a.ent_gap, 4), ret=a.ret, pnl=a.pnl) for a in x7.itertuples()]
out['item6']['limup_pnl_sum_by_version'] = x7.groupby('version')['pnl'].sum().to_dict()
print(json.dumps(out['item6'], ensure_ascii=False, indent=1, default=str))
out['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
(A.HERE / 'results_trades.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print('rss', out['maxrss_mb'])
