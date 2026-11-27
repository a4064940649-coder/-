#!/usr/bin/env python3
"""S106 題目 A：公開申購抽籤（新股／現增）每個身分證的年期望淨收益、撥券日實際可賣價格、資金凍結、破發風險。

資料（全部本機既有、公開）：
  L：/home/user/mjib007/company-law-cases/data/ipo-lottery-history-2020-2026-with-price.csv
     （證交所公開申購公告彙整；「申購結束日市價」為 FinMind 收盤價，初上市櫃者是興櫃價）
  D：/home/user/-/research/tw_stock_selection/raw/tsd_extra/daily/<年>/*.parquet
     （上市＋上櫃＋部分興櫃 日 K，未還原；2020-01～2026-11）
用法：python3 -I -B s106_ipo.py   → 寫 results.json、yearly_table.csv、rows.csv.gz 到本資料夾

假設（全部寫進 results.json['assumptions']）：
  - 每件申購處理費 20 元（中不中都收）；中籤通知郵寄工本費 50 元（扣款時一併預扣，未中籤退還，中籤才實收）。
  - 未中籤退款：不另收費（銀行退款入帳，無郵資）。原程式 ipo_lottery_ev.py 只扣 20 元、沒扣 50 元、沒扣賣出成本。
  - 賣出：撥券日（上市櫃日）開盤價賣；手續費 0.1425%（不打折，保守）＋證交稅 0.3%。
  - 資金凍結：申購結束日收盤前要有款（實務上扣款日＝申購截止次一營業日，即抽籤日前一營業日），
    未中籤者退款日＝抽籤日次一營業日。保守用 [申購結束日, 抽籤日+1 營業日] 都算凍結。
    中籤者：價款變成股票，到撥券日賣出、T+2 交割才回到現金 → [抽籤日+1, 撥券賣出日+2 營業日] 以「中籤率×價款」計期望占用。
"""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
LCSV = Path('/home/user/mjib007/company-law-cases/data/ipo-lottery-history-2020-2026-with-price.csv')
DDIR = Path('/home/user/-/research/tw_stock_selection/raw/tsd_extra/daily')
FEE_APP, FEE_MAIL = 20.0, 50.0
SELL_COST = 0.001425 + 0.003
E50_RT = 2 * 0.001425 + 0.001          # 賣 0050 換現金再買回：兩邊手續費（不打折）＋ETF 證交稅 0.1%
E50_RT_DISC = 2 * 0.000399 + 0.001     # 同上，手續費 2.8 折（0.0399%，主策略說明採用的費率）
E50_DAYS_OUT = 7                        # 申購結束日前 2 營業日賣 0050（T+2 款才到）→ 退款日買回，約 7 個日曆日不在市場
E50_EXCESS = 0.10                       # 0050 相對活存的預期超額（年；0050 長期約 14%、活存約 1.5～2% → 取 10% 保守整數）
IPO_MKTS = {'初上市', '初上櫃', '第一上市公司初上市', '第一上櫃公司初上櫃', '創新板初上市', '創新板轉列上市', '創新板轉列上櫃'}


def roc(s):
    s = str(s).strip()
    if not s or s == 'nan':
        return pd.NaT
    y, m, d = s.split('/')
    return pd.Timestamp(int(y) + 1911, int(m), int(d))


def num(s):
    return pd.to_numeric(s.astype(str).str.replace(',', ''), errors='coerce')


def load_lottery():
    df = pd.read_csv(LCSV, encoding='utf-8-sig', dtype={'證券代號': str})
    df['code'] = df['證券代號'].str.strip()
    df['L'] = df['抽籤日期'].map(roc)
    df['S'] = df['申購開始日'].map(roc)
    df['E'] = df['申購結束日'].map(roc)
    df['C'] = df['撥券日期(上市、上櫃日期)'].map(roc)
    for c, n in [('承銷價_num', 'px'), ('申購股數_num', 'sh'), ('中籤率(%)', 'wr'), ('申購結束日市價', 'p_end'), ('總合格件', 'napps')]:
        df[n] = num(df[c])
    df['ipo'] = df['發行市場'].isin(IPO_MKTS)
    df['year'] = df['L'].dt.year
    return df


def load_daily(codes):
    fs = sorted(glob.glob(str(DDIR / '20[12]*' / '*.parquet')))
    fs = [f for f in fs if Path(f).parent.name >= '2019']
    parts = []
    for f in fs:
        x = pd.read_parquet(f, columns=['date', 'code', 'market', 'open', 'close', 'volume', 'amount'])
        parts.append(x[x['code'].isin(codes)])
    d = pd.concat(parts, ignore_index=True)
    d['date'] = pd.to_datetime(d['date'])
    return d


def main():
    df = load_lottery()
    raw_n = len(df)
    d = load_daily(set(df['code']))
    allcal = pd.DatetimeIndex(sorted(d['date'].unique()))
    lst = d[d['market'].isin(['twse', 'tpex'])].copy()
    lst = lst[(lst['volume'] > 0) & lst['open'].notna() & (lst['open'] > 0)]
    lst['vwap'] = np.where(lst['volume'] > 0, lst['amount'] / lst['volume'], np.nan)
    g = {c: x.set_index('date').sort_index() for c, x in lst.groupby('code')}

    def nxt_bd(t, k=1):
        i = allcal.searchsorted(t, side='right') + (k - 1)
        return allcal[min(i, len(allcal) - 1)]

    rec = []
    for i, r in df.iterrows():
        o = dict(idx=i)
        x = g.get(r['code'])
        if x is not None and pd.notna(r['C']):
            after = x[x.index >= r['C']]
            if len(after):
                t = after.index[0]
                o.update(sell_date=t, c_open=after['open'].iat[0], c_close=after['close'].iat[0], c_vwap=after['vwap'].iat[0],
                         sell_lag_days=(t - r['C']).days)
            before_E = x[x.index < r['E']] if pd.notna(r['E']) else x.iloc[:0]
            if len(before_E) and (r['E'] - before_E.index[-1]).days <= 7:
                o['p_dec'] = before_E['close'].iat[-1]            # 申購結束日前一交易日收盤（決策時已知）
            atE = x[x.index <= r['E']] if pd.notna(r['E']) else x.iloc[:0]
            if len(atE) and (r['E'] - atE.index[-1]).days <= 7:
                o['p_E_panel'] = atE['close'].iat[-1]
            atL = x[x.index <= r['L']]
            if len(atL) and (r['L'] - atL.index[-1]).days <= 7:
                o['p_L'] = atL['close'].iat[-1]
        rec.append(o)
    R = df.join(pd.DataFrame(rec).set_index('idx'))
    # IPO：決策價只能用 CSV 的興櫃價（申購結束日收盤，輕微前視：申購截止當天盤中即可見）
    R['p_dec'] = np.where(R['ipo'] | R['p_dec'].isna(), R['p_end'], R['p_dec'])
    drawn = R['napps'].fillna(0) > 0
    base = R[R['px'].notna() & (R['sh'] > 0) & drawn].copy()
    base['p'] = base['wr'] / 100
    base['cost'] = base['px'] * base['sh']
    base['ret_end'] = base['p_end'] / base['px'] - 1                      # 原程式口徑
    base['ret_dec'] = base['p_dec'] / base['px'] - 1
    base['ret_open'] = base['c_open'] / base['px'] - 1                     # 撥券日開盤
    base['ret_close'] = base['c_close'] / base['px'] - 1
    base['ret_vwap'] = base['c_vwap'] / base['px'] - 1
    base['drift_E_to_open'] = base['c_open'] / base['p_end'] - 1           # 申購結束日市價 → 撥券日開盤
    base['drift_L_to_open'] = base['c_open'] / base['p_L'] - 1             # 抽籤日市價 → 撥券日開盤（增資才有）
    base['gain_net'] = base['c_open'] * base['sh'] * (1 - SELL_COST) - base['cost'] - FEE_MAIL   # 中籤一次的淨損益
    base['gain_end_gross'] = (base['p_end'] - base['px']) * base['sh']
    base['ev_app'] = base['p'] * base['gain_net'] - FEE_APP
    base['ev_app_orig'] = base['p'] * base['gain_end_gross'] - FEE_APP
    base['ok'] = base['c_open'].notna()
    coverage = dict(rows_csv=raw_n, rows_drawn_with_px=int(len(base)), rows_with_sell_price=int(base['ok'].sum()),
                    missing_sell=base.loc[~base['ok'], ['抽籤日期', '證券名稱', 'code', '發行市場']].astype(str).values.tolist(),
                    undrawn_excluded=int((~drawn & R['px'].notna()).sum()),
                    sell_lag_gt3d=int((base['sell_lag_days'] > 3).sum()))
    B = base[base['ok']].copy()

    # ---------- 原程式重跑核對（同口徑：申購結束日市價、只扣 20 元、含未抽出列）----------
    orig = R[R['px'].notna() & R['p_end'].notna() & (R['sh'] > 0)].copy()
    orig['g'] = (orig['p_end'] - orig['px']) * orig['sh']
    orig['ev'] = orig['wr'] / 100 * orig['g'] - 20
    recheck = dict(rows_valid=int(len(orig)), avg_ret_pct=round(float((orig['p_end'] / orig['px'] - 1).mean() * 100), 1),
                   share_profitable_pct=round(float((orig['g'] > 0).mean() * 100), 1),
                   avg_ev_per_app=round(float(orig['ev'].mean()), 0),
                   sum_ev_by_year=orig.groupby('year')['ev'].sum().round(0).astype(int).to_dict(),
                   note='原 ipo_lottery_ev.py 的取消判定字串（Y/是）對不到 CSV 的「取消申購」，但這 3 列承銷價為空已被排除，不影響結果；14 列 2026-09 未抽籤（中籤率 0）被當成 −20 元計入。')

    # ---------- 政策 ----------
    pol = {
        'P0_全部申購': B.index,
        'P1_決策價差≥5%': B.index[B['ret_dec'] >= 0.05],
        'P2_決策價差≥10%': B.index[B['ret_dec'] >= 0.10],
    }

    def mc(sub, n=20000, seed=0):
        rng = np.random.default_rng(seed)
        p = sub['p'].clip(0, 1).values; gain = sub['gain_net'].values
        out = np.zeros(n)
        for k in range(0, n, 2000):
            dr = rng.random((2000, len(p))) < p
            out[k:k + 2000] = (dr * gain).sum(1) - FEE_APP * len(p)
        return out

    yearly = []
    for pn, ix in pol.items():
        S = B.loc[ix]
        for y, s in S.groupby('year'):
            sim = mc(s, seed=int(y))
            yearly.append(dict(policy=pn, year=int(y), n_apps=len(s), fees_20=FEE_APP * len(s),
                               exp_wins=round(float(s['p'].sum()), 2),
                               gross_EV_sum_p_gain=round(float((s['p'] * s['gain_net']).sum()), 0),
                               net_EV=round(float(s['ev_app'].sum()), 0),
                               net_EV_orig_basis=round(float(s['ev_app_orig'].sum()), 0),
                               top3_share_of_gross=round(float((s['p'] * s['gain_net']).nlargest(3).sum() / max((s['p'] * s['gain_net']).sum(), 1)), 2),
                               sim_p10=round(float(np.percentile(sim, 10)), 0), sim_median=round(float(np.median(sim)), 0),
                               sim_p90=round(float(np.percentile(sim, 90)), 0), sim_prob_neg=round(float((sim < 0).mean()), 3),
                               breaks_n=int((s['ret_open'] < 0).sum()),
                               exp_break_wins=round(float(s.loc[s['gain_net'] < 0, 'p'].sum()), 2)))
    Y = pd.DataFrame(yearly)

    # ---------- 資金凍結（逐營業日）----------
    cal = allcal[(allcal >= '2020-01-01') & (allcal <= '2026-10-31')]

    def freeze(S):
        tot = pd.Series(0.0, index=cal); exp_win = pd.Series(0.0, index=cal)
        for _, r in S.iterrows():
            a, b = r['E'], nxt_bd(r['L'], 1)
            tot[(cal >= a) & (cal <= b)] += r['cost'] + FEE_APP + FEE_MAIL
            c2 = nxt_bd(r['sell_date'], 2)
            exp_win[(cal > b) & (cal <= c2)] += r['p'] * r['cost']
        return tot, exp_win

    frz = []
    worst_win_tie = {}
    for pn, ix in pol.items():
        S = B.loc[ix]
        tot, ew = freeze(S)
        allf = tot + ew
        for y in sorted(S['year'].unique()):
            m = allf[allf.index.year == y]; t0 = tot[tot.index.year == y]
            frz.append(dict(policy=pn, year=int(y), max_frozen=round(float(m.max()), 0), max_date=str(m.idxmax().date()),
                            p95_frozen=round(float(m.quantile(0.95)), 0), mean_frozen=round(float(m.mean()), 0),
                            days_with_freeze_pct=round(float((t0 > 0).mean() * 100), 1),
                            max_single_lot=round(float(S.loc[S['year'] == y, 'cost'].max()), 0),
                            median_lot=round(float(S.loc[S['year'] == y, 'cost'].median()), 0)))
    F = pd.DataFrame(frz)

    # ---------- 資金上限模擬：K 元現金池，同一申購結束日按「決策價差金額」排序，放得下才申購 ----------
    B['dec_amt'] = (B['p_dec'] - B['px']) * B['sh']

    def capped(K, thr=0.05, key='ret_dec'):
        S = B[B['ret_dec'] >= thr].sort_values(['E', key], ascending=[True, False])
        busy = []                 # (release_date, amount)
        chosen = []
        for i, r in S.iterrows():
            busy = [(t, a) for t, a in busy if t >= r['E']]
            used = sum(a for _, a in busy)
            need = r['cost'] + FEE_APP + FEE_MAIL
            if used + need <= K:
                busy.append((nxt_bd(r['L'], 1), need))
                # 中籤時價款再占用到賣出 T+2：以期望值計（p×cost）
                busy.append((nxt_bd(r['sell_date'], 2), r['p'] * r['cost']))
                chosen.append(i)
        C = B.loc[chosen]
        return C

    cap_rows = []
    for K, key in [(k, kk) for k in [100_000, 200_000, 300_000, 500_000, 1_000_000] for kk in ['ret_dec', 'dec_amt']]:
        C = capped(K, key=key)
        for y, s in C.groupby('year'):
            ev = float(s['ev_app'].sum())
            opp = K * E50_EXCESS * (9 / 12 if y == 2026 else 1)
            cap_rows.append(dict(cash_pool=K, priority=key, year=int(y), n_apps=len(s), net_EV=round(ev, 0),
                                 opp_cost_if_pool_else_in_0050=round(opp, 0), EV_minus_opp=round(ev - opp, 0)))
    CAP = pd.DataFrame(cap_rows)

    # 每筆賣 0050 換現金的成本（不留現金池）：每次申購的價款 × 來回成本，與該筆期望淨收益比
    B['e50_cost'] = B['cost'] * E50_RT
    B['e50_cost_disc'] = B['cost'] * (E50_RT_DISC + E50_EXCESS * E50_DAYS_OUT / 365)
    sell0050 = []
    for pn, ix in pol.items():
        S = B.loc[ix]
        for y, s in S.groupby('year'):
            sell0050.append(dict(policy=pn, year=int(y), net_EV=round(float(s['ev_app'].sum()), 0),
                                 e50_roundtrip_cost_if_each_app_funded=round(float(s['e50_cost'].sum()), 0),
                                 e50_cost_disc_plus_out_of_market=round(float(s['e50_cost_disc'].sum()), 0),
                                 apps_with_EV_below_e50_cost_pct=round(float((s['ev_app'] < s['e50_cost']).mean() * 100), 1)))
    SELL = pd.DataFrame(sell0050)

    # ---------- 價格風險：申購結束日市價／抽籤日 → 撥券日開盤 ----------
    def q(s):
        s = s.dropna()
        return dict(n=int(len(s)), mean=round(float(s.mean()), 4), median=round(float(s.median()), 4),
                    p05=round(float(s.quantile(.05)), 4), p25=round(float(s.quantile(.25)), 4), p75=round(float(s.quantile(.75)), 4),
                    p95=round(float(s.quantile(.95)), 4), share_neg=round(float((s < 0).mean()), 3))
    risk = {}
    for lab, m in [('IPO', B['ipo']), ('現增', ~B['ipo'])]:
        s = B[m]
        risk[lab] = dict(
            drift_end_to_open=q(s['drift_E_to_open']), drift_lottery_to_open=q(s['drift_L_to_open']),
            ret_end_basis=q(s['ret_end']), ret_open=q(s['ret_open']), ret_close=q(s['ret_close']), ret_vwap=q(s['ret_vwap']),
            break_open_pct=round(float((s['ret_open'] < 0).mean() * 100), 2),
            break_close_pct=round(float((s['ret_close'] < 0).mean() * 100), 2),
            net_loss_pct=round(float((s['gain_net'] < 0).mean() * 100), 2),
            avg_loss_when_break_ret=round(float(s.loc[s['ret_open'] < 0, 'ret_open'].mean()), 4) if (s['ret_open'] < 0).any() else None,
            avg_loss_when_break_NT=round(float(s.loc[s['gain_net'] < 0, 'gain_net'].mean()), 0) if (s['gain_net'] < 0).any() else None,
            worst5=s.nsmallest(5, 'gain_net')[['抽籤日期', '證券名稱', 'code', 'px', 'p_end', 'c_open', 'wr', 'gain_net']].astype(str).values.tolist())
    # 決策價差 ≥5% 的篩選後，破發情形
    s5 = B[B['ret_dec'] >= 0.05]
    risk['P1_篩選後'] = dict(n=int(len(s5)), break_open_pct=round(float((s5['ret_open'] < 0).mean() * 100), 2),
                           net_loss_pct=round(float((s5['gain_net'] < 0).mean() * 100), 2),
                           exp_loss_wins_per_year=round(float(s5.loc[s5['gain_net'] < 0, 'p'].sum() / 6.75), 3))
    # 被篩掉的（決策價差 <5%）：中籤率高、常虧
    s_lo = B[B['ret_dec'] < 0.05]
    risk['決策價差<5%的列'] = dict(n=int(len(s_lo)), avg_win_rate_pct=round(float(s_lo['wr'].mean()), 2),
                              net_loss_pct=round(float((s_lo['gain_net'] < 0).mean() * 100), 1),
                              sum_ev=round(float(s_lo['ev_app'].sum()), 0))
    # 多檔同時中籤的總風險：年度最壞情境＝全部 gain_net<0 的列都中（上界）、以及模擬 p01
    multi = {}
    for y, s in B.loc[pol['P1_決策價差≥5%']].groupby('year'):
        sim = mc(s, seed=100 + int(y))
        multi[int(y)] = dict(sim_p01=round(float(np.percentile(sim, 1)), 0), sim_min=round(float(sim.min()), 0),
                             sum_neg_if_all_neg_win=round(float(s.loc[s['gain_net'] < 0, 'gain_net'].sum() - FEE_APP * len(s)), 0))

    # 同一申購結束日同時開放的檔數
    same_day = B.groupby('E').size()
    B[['抽籤日期', '證券名稱', 'code', '發行市場', 'ipo', 'year', 'px', 'sh', 'cost', 'wr', 'napps', 'p_end', 'p_dec', 'p_L',
       'sell_date', 'c_open', 'c_close', 'c_vwap', 'ret_end', 'ret_dec', 'ret_open', 'drift_E_to_open', 'drift_L_to_open',
       'gain_net', 'ev_app']].to_csv(HERE / 'rows.csv.gz', index=False, encoding='utf-8-sig')

    # ---------- 主表：每年期望收益 vs 資金凍結 ----------
    main = Y[Y.policy == 'P1_決策價差≥5%'][['year', 'n_apps', 'exp_wins', 'gross_EV_sum_p_gain', 'fees_20', 'net_EV', 'sim_p10', 'sim_median', 'sim_p90', 'sim_prob_neg']] \
        .merge(F[F.policy == 'P1_決策價差≥5%'][['year', 'max_frozen', 'p95_frozen', 'mean_frozen', 'max_single_lot']], on='year')
    p0 = Y[Y.policy == 'P0_全部申購'][['year', 'net_EV', 'net_EV_orig_basis']].rename(columns={'net_EV': 'P0_net_EV', 'net_EV_orig_basis': 'P0_orig_basis'})
    main = main.merge(p0, on='year')
    main.to_csv(HERE / 'yearly_table.csv', index=False, encoding='utf-8-sig')
    full = main[main.year <= 2025]
    res = dict(
        assumptions=dict(fee_app=FEE_APP, fee_mail_if_win=FEE_MAIL, refund_fee=0, sell_cost=SELL_COST, sell_price='撥券日（上市櫃日）開盤價；另報收盤、均價',
                         freeze_window='[申購結束日, 抽籤日+1 營業日] 全額（價款+20+50）；中籤者另以 p×價款 占用到賣出日+2 營業日',
                         e50_roundtrip=E50_RT, e50_roundtrip_disc=E50_RT_DISC, e50_days_out=E50_DAYS_OUT, e50_excess_over_deposit=E50_EXCESS,
                         decision_price='現增：申購結束日前一交易日收盤（日 K）；IPO：CSV 興櫃價（申購結束日收盤，輕微前視）',
                         win_prob='官方實際中籤率（事後值，但只當期望值的機率，不用來篩選）'),
        coverage=coverage, recheck_original=recheck,
        yearly_by_policy=Y.to_dict(orient='records'), freeze_by_policy=F.to_dict(orient='records'),
        cash_pool_sim=CAP.to_dict(orient='records'), fund_by_selling_0050=SELL.to_dict(orient='records'),
        price_risk=risk, multi_win_tail=multi,
        same_E_date_count=dict(max=int(same_day.max()), mean=round(float(same_day.mean()), 2)),
        main_table=main.to_dict(orient='records'),
        summary_2020_2025_P1=dict(median_net_EV=float(full['net_EV'].median()), worst_year=int(full.loc[full['net_EV'].idxmin(), 'year']),
                                  worst_net_EV=float(full['net_EV'].min()), mean_net_EV=round(float(full['net_EV'].mean()), 0)),
    )
    (HERE / 'results_ipo.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    pd.set_option('display.width', 250)
    print(json.dumps(coverage, ensure_ascii=False)[:800]); print(json.dumps(recheck, ensure_ascii=False))
    print(Y.to_string()); print(F.to_string()); print(CAP.to_string()); print(SELL.to_string())
    print(json.dumps(risk, ensure_ascii=False, indent=1)); print(json.dumps(multi, ensure_ascii=False)); print(main.to_string())
    print(res['summary_2020_2025_P1'], res['same_E_date_count'])


if __name__ == '__main__':
    main()
