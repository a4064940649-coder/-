#!/usr/bin/env python3
"""第二輪新增資料集的交叉檢查。用法：python3 -I -B crosscheck_round2.py <新資料集所在根目錄> <既有 raw/tsd_extra> <twt49u 封存資料夾> <輸出 json> [要跑的段落,逗號分隔]
只讀，不改任何資料。"""
import sys, os, glob, json, gzip, io
import numpy as np, pandas as pd

NEW, OLD, TWT, OUT = sys.argv[1:5]
ONLY = set(sys.argv[5].split(',')) if len(sys.argv) > 5 else None
res = json.load(open(OUT, encoding='utf-8')) if os.path.exists(OUT) else {}


def ld(root, name, start=None, end=None, columns=None):
    fs = sorted(glob.glob(os.path.join(root, name, '[0-9]' * 4, '*.parquet')))
    pick = []
    for f in fs:
        st = os.path.basename(f)[:-8]
        if start and st < start[:len(st)]: continue
        if end and st > end[:len(st)]: continue
        pick.append(f)
    df = pd.concat([pd.read_parquet(f, columns=columns) for f in pick], ignore_index=True)
    if 'date' in df.columns:
        df['date'] = df['date'].astype(str)
        if start: df = df[df.date >= start]
        if end: df = df[df.date <= end]
    return df.reset_index(drop=True)


def sec(name):
    return ONLY is None or name in ONLY


def r(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


if sec('tdcc'):
    o = {}
    weeks = []; bad17 = 0; rows = 0; dup = 0; pctdiff = []; nopct = []; neg = 0
    for f in sorted(glob.glob(os.path.join(NEW, 'tdcc', '*', '*.parquet'))):
        d = pd.read_parquet(f); d['date'] = d['date'].astype(str); rows += len(d)
        weeks += sorted(d.date.unique())
        g = d.groupby(['date', 'code'])["level"].agg(['count', 'nunique', 'min', 'max'])
        bad17 += int(((g['count'] != 17) | (g['nunique'] != 17) | (g['min'] != 1) | (g['max'] != 17)).sum())
        dup += int(d.duplicated(['date', 'code', 'level']).sum())
        neg += int(((d.people < 0) | (d.shares < 0)).sum())
        tot = d[d.level == 17].set_index(['date', 'code']).shares
        x = d[d.level <= 15].join(tot.rename('tot'), on=['date', 'code'])
        x = x[x.tot > 0]
        nopct += sorted(x[x.pct.isna()].date.unique())
        y = x[x.pct.notna()]
        pctdiff.append((y.shares / y.tot * 100 - y.pct).abs().max())
        # 1～16 級股數加總＝合計列
        s15 = d[d.level <= 15].groupby(['date', 'code']).shares.sum(); l16 = d[d.level == 16].set_index(['date', 'code']).shares
        o.setdefault('levels_1_15_minus_level16_ne_total', 0); o['levels_1_15_minus_level16_ne_total'] += int(((s15 - l16).reindex(tot.index) != tot).sum())
        o.setdefault('stock_weeks_with_nonzero_level16', 0); o['stock_weeks_with_nonzero_level16'] += int((l16 != 0).sum())
        p16 = d[d.level <= 15].groupby(['date', 'code']).people.sum(); pt = d[d.level == 17].set_index(['date', 'code']).people
        o.setdefault('sum_people_ne_total', 0); o['sum_people_ne_total'] += int((p16.reindex(pt.index) != pt).sum())
    wd = pd.to_datetime(pd.Series(weeks))
    gaps = [(weeks[i - 1], weeks[i], int((wd[i] - wd[i - 1]).days)) for i in range(1, len(weeks)) if (wd[i] - wd[i - 1]).days > 9]
    o.update(rows=rows, weeks=len(weeks), first_week=weeks[0], last_week=weeks[-1], gaps_over_9_days=gaps,
             stock_weeks_without_exactly_levels_1_to_17=bad17, duplicate_rows=dup, negative_values=neg,
             weeks_without_pct=sorted(set(nopct)), max_abs_diff_pct_vs_shares_ratio=r(np.nanmax(pctdiff)),
             weekday_counts=wd.dt.dayofweek.value_counts().sort_index().to_dict())
    # 彙總表 vs 長表
    s = ld(NEW, 'tdcc_summary'); o['summary_rows'] = len(s)
    o['summary_dup'] = int(s.duplicated(['date', 'code']).sum())
    lastf = sorted(glob.glob(os.path.join(NEW, 'tdcc', '*', '*.parquet')))[-1]
    d = pd.read_parquet(lastf); d['date'] = d['date'].astype(str)
    chk = d[d.level.between(12, 15)].groupby(['date', 'code']).pct.sum().rename('p').reset_index().merge(s, on=['date', 'code'])
    o['summary_pct_ge400_vs_upstream_pct_sum_max_abs_diff_last_month'] = r((chk.p - chk.pct_ge400).abs().max())
    # 集保庫存總股數 vs 日 K 發行股數（上市櫃、同日）
    out = []
    for y in ('2020', '2023', '2026'):
        sy = s[s.date.str[:4] == y]
        dk = ld(OLD, 'daily', y + '-01-01', y + '-12-31', ['date', 'code', 'market', 'shares', 'close'])
        m = sy.merge(dk[dk.market.isin(['twse', 'tpex']) & dk.code.str.match(r'^[1-9]\d{3}$') & (dk.shares > 0)], on=['date', 'code'])
        ratio = m.shares_total / m.shares
        out.append(dict(year=y, matched_stock_weeks=len(m), median_ratio=r(ratio.median()), share_within_2pct=r(((ratio - 1).abs() <= 0.02).mean()),
                        share_within_10pct=r(((ratio - 1).abs() <= 0.10).mean()), holders_le_0=int((m.holders <= 0).sum()),
                        median_holders=r(m.holders.median(), 0), pct_out_of_0_100=int(((m.pct_ge400 < 0) | (m.pct_ge400 > 100.001) | (m.pct_lt10 < 0) | (m.pct_lt10 > 100.001)).sum()),
                        tdcc_weeks_not_trading_day=sorted(set(sy.date) - set(dk.date))))
    o['tdcc_total_shares_vs_daily_issued_shares'] = out
    x = s[(s.code == '2330')].sort_values('date').tail(1).iloc[0]
    o['example_2330_latest'] = {k: (x[k] if isinstance(x[k], str) else r(x[k], 3)) for k in s.columns}
    res['tdcc'] = o

if sec('breadth'):
    b = ld(NEW, 'breadth_twse')
    dk = ld(OLD, 'daily', columns=['date', 'code', 'market', 'close', 'change', 'limit', 'volume', 'price_basis'])
    dk = dk[(dk.market == 'twse')]
    o = dict(days=len(b), first=b.date.min(), last=b.date.max(), dup_days=int(b.date.duplicated().sum()))
    best = None
    for nm, mask in (('四碼普通股', dk.code.str.match(r'^[1-9]\d{3}$')), ('全部上市證券', dk.code.notna()),
                     ('四碼＋特別股等非 ETF', ~dk.code.str.match(r'^0'))):
        t = dk[mask & (dk.volume > 0)]
        c = pd.DataFrame({'up': (t.change > 0).groupby(t.date).sum(), 'down': (t.change < 0).groupby(t.date).sum(), 'flat': (t.change == 0).groupby(t.date).sum(),
                          'limit_up': (t.limit == 'up').groupby(t.date).sum(), 'limit_down': (t.limit == 'down').groupby(t.date).sum()})
        m = b.set_index('date').join(c, rsuffix='_calc', how='inner')
        st = {k: dict(exact_days_share=r((m[k] == m[k + '_calc']).mean()), within_3_share=r(((m[k] - m[k + '_calc']).abs() <= 3).mean()),
                      median_abs_diff=r((m[k] - m[k + '_calc']).abs().median(), 1), corr=r(np.corrcoef(m[k], m[k + '_calc'])[0, 1])) for k in ('up', 'down', 'flat', 'limit_up', 'limit_down')}
        o['universe_' + nm] = st
    # 漲跌停：用前收×(1±10%) 依升降單位取整的漲跌停價，自己數收盤在漲跌停價的四碼普通股
    t = dk[dk.code.str.match(r'^[1-9]\d{3}$') & (dk.volume > 0)].copy(); pc = t.close - t.change
    def tick(p):
        return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)
    up = pc * 1.1; dn = pc * 0.9
    lu = np.floor(np.round(up / tick(up), 6)) * tick(up); ld_ = np.ceil(np.round(dn / tick(dn), 6)) * tick(dn)
    c2 = pd.DataFrame({'limit_up': ((t.close - lu).abs() < 0.0051).groupby(t.date).sum(), 'limit_down': ((t.close - ld_).abs() < 0.0051).groupby(t.date).sum()})
    m = b.set_index('date').join(c2, rsuffix='_calc', how='inner'); m = m[m.index >= '2015-06-01']
    o['limit_counts_vs_close_at_limit_price_from_daily_since_2015_06'] = {k: dict(exact_days_share=r((m[k] == m[k + '_calc']).mean()), within_3_share=r(((m[k] - m[k + '_calc']).abs() <= 3).mean()),
        corr=r(np.corrcoef(m[k], m[k + '_calc'])[0, 1])) for k in ('limit_up', 'limit_down')}
    o['sum_up_down_flat_vs_traded_common_stocks_median_diff'] = r(((b.set_index('date')[['up', 'down', 'flat']].sum(axis=1)) - t.groupby('date').size()).median(), 1)
    last = b.sort_values('date').tail(1).iloc[0].to_dict(); o['last_row'] = {k: (v if isinstance(v, str) else int(v)) for k, v in last.items()}
    res['breadth'] = o

if sec('lists'):
    o = {}
    dk = ld(OLD, 'daily', columns=['date', 'code', 'market'])
    for nm in ('chtm_tpex', 'fulldelivery_twse', 'tib_twse', 'stophalt'):
        d = ld(NEW, nm)
        tw = dk[dk.market == ('tpex' if nm == 'chtm_tpex' else 'twse')]; key = set(zip(tw.date, tw.code))
        inn = np.array([(a, b) in key for a, b in zip(d.date, d.code)])
        o[nm] = dict(rows=len(d), first=d.date.min(), last=d.date.max(), dup=int(d.duplicated(['date', 'code']).sum()), codes=int(d.code.nunique()),
                     share_rows_found_in_daily_same_market_same_day=r(inn.mean()), median_rows_per_day=r(d.groupby('date').size().median(), 0))
    c = ld(NEW, 'chtm_tpex'); tp = dk[dk.market == 'tpex']; key = set(zip(tp.date, tp.code))
    inn = np.array([(a, b) in key for a, b in zip(c.date, c.code)])
    o['chtm_tpex']['not_halted_rows_found_in_daily_tpex_share'] = r(inn[(c.halted == 0).to_numpy()].mean())
    o['chtm_tpex']['halted_rows_share'] = r((c.halted == 1).mean())
    res['lists'] = o


def exdiv(kind):
    if kind == 'twse':
        fs = sorted(glob.glob(os.path.join(TWT, 'twse-exright', '*', '*.csv.gz')))
        e = pd.concat([pd.read_csv(f, dtype={'code': str}) for f in fs], ignore_index=True)
    else:
        e = ld(OLD, 'otc_exright', columns=['date', 'code', 'value', 'type'])
    e['date'] = e['date'].astype(str)
    return e


if sec('per'):
    o = {}
    for nm, mk in (('per_twse', 'twse'), ('per_tpex', 'tpex')):
        p = ld(NEW, nm); e = exdiv(mk)
        q = dict(rows=len(p), first=p.date.min(), last=p.date.max(), dup=int(p.duplicated(['date', 'code']).sum()), days=int(p.date.nunique()),
                 null_share={c: r(p[c].isna().mean()) for c in ('close', 'yield_pct', 'dividend_year', 'per', 'pbr')},
                 negative=int(((p.yield_pct < 0) | (p.per < 0) | (p.pbr < 0)).sum()))
        # 有 close 的列：與日 K 收盤比
        dk = ld(OLD, 'daily', columns=['date', 'code', 'market', 'close']); dk = dk[dk.market == mk].drop(columns='market').rename(columns={'close': 'c'})
        m = p.merge(dk, on=['date', 'code'], how='left')
        q['share_rows_found_in_daily'] = r(m.c.notna().mean())
        h = m[m.close.notna() & m.c.notna()]
        q['close_vs_daily_close'] = dict(n=len(h), share_equal=r(((h.close - h.c).abs() < 0.006).mean()))
        # 殖利率抽查：各年最後一個交易日，殖利率×日 K 收盤 ≈ 當年只配現金（type=息）的除息金額合計
        chk = []
        for y in range(2016, 2026):
            dy = m[m.date.str[:4] == str(y)]
            last = dy.date.max(); dy = dy[dy.date == last]
            ey = e[e.date.str[:4] == str(y)]
            only_cash = ey.groupby('code')['type'].agg(lambda s: set(s) == {'息'})
            cash = ey[ey.code.isin(only_cash[only_cash].index)].groupby('code').value.sum().rename('div')
            j = dy.merge(cash, on='code'); j = j[j.c.notna() & j.yield_pct.notna()]
            imp = j.yield_pct * j.c / 100
            ok = (imp - j['div']).abs() <= np.maximum(0.03, 0.01 * j['div'])
            chk.append(dict(date=last, stocks=len(j), share_match=r(ok.mean()), dividend_year_mode=r(j.dividend_year.mode().iloc[0], 0) if j.dividend_year.notna().any() else None))
        q['yield_x_close_vs_cash_dividend_events'] = chk
        o[nm] = q
    res['per'] = o

if sec('early'):
    o = {}
    cal = [ln.strip() for ln in open(os.path.join(os.path.dirname(OUT), 'early_daily_calendar.txt'))] if os.path.exists(os.path.join(os.path.dirname(OUT), 'early_daily_calendar.txt')) else []
    for nm, later in (('inst_twse_early', 'inst_twse'), ('instamt_twse_market_early', 'instamt_twse_market'), ('marginmkt_twse_market_early', 'marginmkt_twse_market'),
                      ('margin_tpex_early', 'margin_tpex'), ('per_twse_early', 'per_twse'), ('per_tpex_early', 'per_tpex')):
        e = ld(NEW, nm); root2 = OLD if os.path.isdir(os.path.join(OLD, later)) else NEW
        l = ld(root2, later, '2015-01-01', '2015-01-31')
        days = sorted(e.date.unique()); ds = set(days)
        q = dict(rows=len(e), first=days[0], last=days[-1], days=len(days), dup=int(e.duplicated([c for c in ('date', 'code', 'investor', 'item') if c in e.columns]).sum()),
                 same_column_names_and_order_as_later=bool(list(e.columns) == list(l.columns)))
        if cal:
            q['missing_days_vs_early_daily_calendar'] = [d for d in cal if days[0] <= d <= days[-1] and d not in ds]
            q['days_not_in_early_daily_calendar'] = [d for d in days if d not in set(cal) and d >= cal[0]]
        a = e[e.date == days[-1]]; b = l[l.date == l.date.min()]
        q['boundary'] = dict(early_last=days[-1], later_first=l.date.min())
        if 'code' in e.columns:
            q['boundary']['codes_early_last_day'] = int(a.code.nunique()); q['boundary']['codes_later_first_day'] = int(b.code.nunique())
            q['boundary']['codes_in_both'] = len(set(a.code) & set(b.code))
        if nm == 'margin_tpex_early':
            j = a.merge(b, on='code', suffixes=('_a', '_b'))
            q['boundary']['m_balance_eq_next_m_prev_share'] = r((j.m_balance_a == j.m_prev_b).mean()); q['boundary']['s_balance_eq_next_s_prev_share'] = r((j.s_balance_a == j.s_prev_b).mean())
            # 內部連續性：全期間 今日餘額＝隔日前日餘額
            e2 = e.sort_values(['code', 'date']); nx = e2.groupby('code').m_prev.shift(-1); nd = e2.groupby('code').date.shift(-1)
            dmap = dict(zip(days[:-1], days[1:])); okd = nd == e2.date.map(dmap)
            q['internal_m_balance_eq_next_day_m_prev_share'] = r((e2.m_balance[okd] == nx[okd]).mean())
        if nm == 'marginmkt_twse_market_early':
            j = a.merge(b, on='item', suffixes=('_a', '_b'))
            q['boundary']['items'] = sorted(a.item); q['boundary']['balance_vs_next_prev_rel_diff'] = {x.item: r(x.prev_b / x.balance_a - 1, 6) for x in j.itertuples()}
            ee = e.sort_values(['item', 'date']); nxx = ee.groupby('item').prev.shift(-1)
            q['typical_abs_rel_diff_balance_vs_next_day_prev_p50_p99'] = [r((nxx / ee.balance - 1).abs().quantile(x), 6) for x in (0.5, 0.99)]
            q['note'] = '官方的「前日餘額」常與前一日「今日餘額」差一點（事後調整），2015 後的資料也是這樣，不是銜接問題'
            e2 = e.sort_values(['item', 'date']); nx = e2.groupby('item').prev.shift(-1)
            q['internal_balance_eq_next_day_prev_share'] = r((e2.balance == nx)[nx.notna()].mean())
        if nm == 'instamt_twse_market_early':
            q['boundary']['investors_early_last'] = sorted(a.investor); q['boundary']['investors_later_first'] = sorted(b.investor)
            q['investor_names_by_first_date'] = {k: v for k, v in e.groupby('investor').date.min().sort_values().items()}
            t = e[e.investor == '合計'].set_index('date'); s = e[e.investor != '合計'].groupby('date')[['buy', 'sell', 'net']].sum()
            q['total_row_eq_sum_of_others_share'] = r((t[['buy', 'sell', 'net']] == s.reindex(t.index)).all(axis=1).mean())
        if nm == 'inst_twse_early':
            q['total_eq_foreign_trust_dealer_share'] = r((e.total == e.foreign + e.trust + e.dealer).mean())
            q['first_day_with_dealer_split'] = e[e.dealer_self.notna()].date.min()
            # 與法人金額（早期）方向比對：外資逐股買賣超股數×?? 無價格，改比「外資淨買超為正的日子」與金額表外資 net 的符號一致率
            am = ld(NEW, 'instamt_twse_market_early', '2012-05-02', '2014-12-31')
            fa = am[am.investor.str.startswith('外資')].groupby('date').net.sum()
            px = pd.concat([pd.read_parquet(f, columns=['date', 'code', 'close']) for f in sorted(glob.glob(os.path.join(OLD, 'daily', '2015', '2015-01.parquet')))])
            q['note_sign_check'] = '逐股表沒有價格，無法換算金額；僅以 2014-12 各日「外資買賣超股數合計」與金額表外資淨額的符號一致率粗查'
            fs_ = e[e.date >= '2014-01-01'].groupby('date').foreign.sum()
            jj = pd.concat([fs_, fa], axis=1, keys=['sh', 'amt']).dropna()
            q['foreign_net_shares_vs_net_amount_sign_agree_2014'] = r((np.sign(jj.sh) == np.sign(jj.amt)).mean()); q['corr_2014'] = r(np.corrcoef(jj.sh, jj.amt)[0, 1])
        if nm.startswith('per_'):
            j = a.merge(b, on='code', suffixes=('_a', '_b'))
            q['boundary']['median_abs_pbr_change'] = r((j.pbr_a - j.pbr_b).abs().median()); q['boundary']['pbr_corr'] = r(j[['pbr_a', 'pbr_b']].dropna().corr().iloc[0, 1])
            q['boundary']['yield_corr'] = r(j[['yield_pct_a', 'yield_pct_b']].dropna().corr().iloc[0, 1])
        o[nm] = q
    res['early'] = o

json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
for k in (ONLY or res):
    print(json.dumps({k: res[k]}, ensure_ascii=False, default=str)[:6000])
