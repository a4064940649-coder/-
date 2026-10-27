#!/usr/bin/env python3
"""第 0 步：資料可得性調查與重疊期一致率（不看任何報酬）。輸出 results_data_audit.json。"""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import sbl_lib as L

IDX, COLS = L.IDX, L.COLS
LISTED = set(COLS)
out = {}


def tw_codes(s):
    return s[s.isin(LISTED)]


def cov(df, dcol, ccol, name, fields, note):
    d = df.copy(); d[dcol] = pd.to_datetime(d[dcol])
    days = pd.DatetimeIndex(sorted(d[dcol].unique()))
    pdays = IDX[(IDX >= days.min()) & (IDX <= days.max())]
    miss = [str(x.date()) for x in pdays if x not in set(days)]
    lc = d[d[ccol].isin(LISTED)]
    # 覆蓋率：每日「面板有收盤的上市代碼」中有出現在來源的比例（中位數）
    per = []
    for x in days[:: max(1, len(days) // 40)]:
        if x not in IDX: continue
        t = IDX.get_loc(x)
        traded = {COLS[j] for j in np.where(L.K.TRADED[t])[0]}
        have = set(lc.loc[lc[dcol] == x, ccol])
        if traded: per.append(len(traded & have) / len(traded))
    return {'source': name, 'fields': fields, 'start': str(days.min().date()), 'end': str(days.max().date()), 'n_days': int(len(days)),
            'n_listed_codes_in_panel': int(lc[ccol].nunique()), 'panel_days_missing_in_source': len(miss), 'missing_examples': miss[:8],
            'coverage_of_panel_traded_median': round(float(np.median(per)), 4) if per else None, 'note': note}


inst5 = L.load_inst5y()
bb_i = pd.read_parquet(L.RAW / 'bb_inst.parquet')
bb_m = pd.read_parquet(L.RAW / 'bb_margin.parquet')
sbl = pd.read_parquet(L.RAW / 'bb_sbl_twt93u.parquet')
vff = pd.read_parquet(L.RAW / 'vf_twse_flows.parquet')
vfq = pd.read_parquet(L.RAW / 'vf_twse_foreign.parquet')
mh = pd.read_csv(L.X / 'margin_hist.csv.gz', dtype={'code': str})
shh = pd.read_csv(L.X / 'shares_hist.csv', dtype={'code': str})

T = []
T.append(cov(sbl, 'Date', 'Ticker', 'BlackBoxOP/tw_stock data/margin/sbl（TWT93U 信用額度總量管制餘額表）',
             '融券今日餘額、借券賣出前日餘額／當日賣出／還券／調整／當日餘額、次日限額（股）',
             'H3a 的核心欄位（借券賣出餘額）。唯一找到的 GitHub 鏡像，2026-07-30 才開始存檔'))
T.append(cov(bb_m, 'Date', 'Ticker', 'BlackBoxOP/tw_stock data/margin（MI_MARGN 融資融券）', '融資餘額、融券賣出／買進／現券償還／前日餘額／今日餘額（張）',
             'P2 對照（融券）。2024-01 起'))
T.append(cov(bb_i, 'Date', 'Ticker', 'BlackBoxOP/tw_stock data/institutional（T86 三大法人）', '外資（不含自營）、外資自營、投信、自營淨買超（股）', '2024-01 起'))
T.append(cov(inst5.rename(columns={'date': 'Date', 'code': 'Ticker'}), 'Date', 'Ticker', '本機 extras/inst_5y（T86）', '外資淨買超（張）', '2021-10-08 起；回推外資持股用'))
T.append(cov(vff, 'date', 'code', 'voidful/tw-institutional-stocker data/twse_flows.csv（T86）', '外資／投信／自營淨買超（股）', '2025-09-11 起'))
T.append(cov(vfq, 'date', 'code', 'voidful/tw-institutional-stocker data/twse_foreign.csv（MI_QFIIS）', '發行股數、外資持股股數、外資持股比率',
             'H3b 的核心欄位（外資持股比率）。2025-10-07 才開始'))
out['coverage'] = T

# ---------------- 重疊期一致率 ----------------
A = {}
m = inst5.merge(bb_i.assign(date=pd.to_datetime(bb_i.Date)), left_on=['date', 'code'], right_on=['date', 'Ticker'])
m = m[m.code.isin(LISTED)]
A['inst_5y 外資 vs BlackBoxOP Foreign_Net（2024-01～2026-10-07，上市）'] = {
    'n': int(len(m)), 'exact_match_rate': round(float((np.abs(m.foreign - m.Foreign_Net) <= 1).mean()), 5),
    'foreign_dealer_net_nonzero_rows': int((m.Foreign_Dealer_Net != 0).sum())}
m = vff.merge(bb_i, left_on=['date', 'code'], right_on=['Date', 'Ticker'])
m = m[m.code.isin(LISTED)]
A['voidful foreign_net vs BlackBoxOP Foreign_Net（2025-09-11～）'] = {'n': int(len(m)), 'exact_match_rate': round(float((np.abs(m.foreign_net - m.Foreign_Net) <= 1).mean()), 5)}
mh2 = mh.rename(columns={'code': 'Ticker'}); mh2['Date'] = mh2['date']
m = bb_m.merge(mh2, on=['Date', 'Ticker'])
m = m[m.Ticker.isin(LISTED)]
A['BlackBoxOP 融券餘額 vs 本機 margin_hist short_bal（2025-09-24～，張）'] = {'n': int(len(m)), 'exact_match_rate': round(float((m.Short_Balance == m.short_bal).mean()), 5),
                                                                       'margin_bal_match_rate': round(float((m.Margin_Balance == m.margin_bal).mean()), 5)}
m = sbl.merge(mh2, on=['Date', 'Ticker'])
m = m[m.Ticker.isin(LISTED)]
A['TWT93U 融券今日餘額（股）/1000 vs 本機 margin_hist short_bal（2026-07-30～）'] = {
    'n': int(len(m)), 'exact_match_rate': round(float((np.abs(m.Margin_Short_Balance / 1000 - m.short_bal) < 0.5).mean()), 5)}
s = sbl.sort_values(['Ticker', 'Date']).copy()
ident = (s.SBL_Prev_Balance + s.SBL_Daily_Sell - s.SBL_Daily_Return + s.SBL_Daily_Adjust - s.SBL_Balance).abs() < 1
s['prev_obs'] = s.groupby('Ticker').SBL_Balance.shift(1)
chain = (s.prev_obs.notna()) & ((s.prev_obs - s.SBL_Prev_Balance).abs() < 1)
A['TWT93U 內部恆等式（前日＋賣出−還券＋調整＝當日）'] = {'n': int(len(s)), 'rate': round(float(ident.mean()), 5)}
A['TWT93U 前日餘額＝上一筆當日餘額'] = {'n': int(s.prev_obs.notna().sum()), 'rate': round(float(chain[s.prev_obs.notna()].mean()), 5)}
q = vfq.copy(); q['date'] = pd.to_datetime(q.date); q = q[q.total_shares > 0]
q['ym'] = q.date.dt.strftime('%Y-%m')
qm = q.sort_values('date').groupby(['code', 'ym']).last().reset_index()
m = qm.merge(shh, on=['code', 'ym'])
m = m[m.code.isin(LISTED)]
A['MI_QFIIS 發行股數 vs 本機 shares_hist（月）'] = {'n': int(len(m)), 'match_rate_0.1%': round(float((np.abs(m.total_shares / m.shares - 1) < 0.001).mean()), 5)}
# 外資持股股數的日變動 vs 外資淨買超
F = q.pivot_table(index='date', columns='code', values='foreign_shares', aggfunc='last')
W = L.load_flows()
dF = F.diff()
common = dF.index.intersection(W.index)[1:]
cc = [c for c in F.columns if c in LISTED]
a = dF.loc[common, cc].to_numpy().ravel(); b = W.reindex(index=common, columns=cc).to_numpy().ravel()
ok = np.isfinite(a) & np.isfinite(b)
a, b = a[ok], b[ok]
S0 = q.pivot_table(index='date', columns='code', values='total_shares', aggfunc='last').reindex(index=common, columns=cc).to_numpy().ravel()[ok]
tol = np.maximum(1000, 0.0002 * S0)
A['MI_QFIIS 外資持股日變動 vs 外資淨買超（2025-10～2026-10，上市）'] = {
    'n': int(len(a)), 'exact_rate(±1000股)': round(float((np.abs(a - b) <= 1000).mean()), 4),
    'within_0.02%_of_shares_rate': round(float((np.abs(a - b) <= tol).mean()), 4),
    'corr': round(float(np.corrcoef(a, b)[0, 1]), 4),
    'note': '兩者差異來自外資自營商以外的帳戶別、ADR 轉換、員工認股、現增／可轉債轉換、借券還券等非交易移轉'}
out['agreement'] = A

# ---------------- 回推驗證：從 2026-10-21 回推到 2025-10-07，和官方值比 ----------------
R, S, Fs = L.load_qfiis()
last = R.index.max()
bc = L.backcast(R, S, W, last, L.ANCHOR - pd.Timedelta(days=1))
val = {}
for dstr in ['2025-10-07', '2026-01-12', '2026-04-13', '2026-07-13']:
    d = pd.Timestamp(dstr)
    if d not in bc.index or d not in R.index: continue
    x = bc.loc[d]; y = R.loc[d].reindex(x.index)
    k = x.notna() & y.notna() & x.index.isin(LISTED)
    e = (x[k] - y[k]) * 100
    val[dstr] = {'n': int(k.sum()), 'years_back': round((last - d).days / 365.25, 2), 'MAE_pp': round(float(e.abs().mean()), 3),
                 'p90_abs_err_pp': round(float(e.abs().quantile(0.9)), 3), 'share_within_1pp': round(float((e.abs() <= 1).mean()), 4),
                 'spearman': round(float(x[k].rank().corr(y[k].rank())), 4)}
# 6 個月變化的回推誤差：2025-10-07 → 2026-04-13
d0, d1 = pd.Timestamp('2025-10-07'), pd.Timestamp('2026-04-13')
if d0 in bc.index and d1 in bc.index:
    x = bc.loc[d1] - bc.loc[d0]; y = (R.loc[d1] - R.loc[d0]).reindex(x.index)
    k = x.notna() & y.notna() & x.index.isin(LISTED)
    val['delta6m_2025-10-07_to_2026-04-13'] = {'n': int(k.sum()), 'MAE_pp': round(float(((x[k] - y[k]) * 100).abs().mean()), 3),
                                               'spearman': round(float(x[k].rank().corr(y[k].rank())), 4),
                                               'sd_true_delta_pp': round(float(y[k].std() * 100), 3)}
out['backcast_validation'] = val
out['search_log'] = {
    'reachable': 'GitHub（git clone 匿名讀取、MCP code／repo search）。twse.com.tw、openapi.twse.com.tw、FinMind API、Hugging Face、Kaggle、TPEx、MOPS 全部連不上（2026-10-27 實測 curl 回 000／proxy 403）',
    'candidates_checked': {
        'ryanovovo/market-data-twse': '只有抓取程式與 schema，沒有存檔資料',
        'tonystark20-hy/etf_rebalance': 'download_margin_history.py 是程式；repo 內只有 ETF 成分股 universe 快照，沒有融券／借券歷史',
        'Pregaine/stock': '借還券擷取程式（25 個檔），沒有資料',
        'foxstock/garden': 'LENDX／CMTSL 程式與筆記本，沒有資料',
        'wenchiehlee/python-actions.finmind（本機）': '約 140 檔個股、2021 起，type13 融資融券日檔多數下載失敗；沒有外資持股、沒有借券',
        'voidful/tw-institutional-stocker（本機）': 'MI_QFIIS 外資持股 2025-10-07 起（每日全上市）、T86 2025-09-11 起 → 採用',
        'BlackBoxOP/tw_stock': 'TWT93U 借券賣出餘額 2026-07-30 起；融資融券、三大法人 2024-01 起 → 採用',
        'twcs-analysis/tw-stock-collector': 'lending／margin 只有 2026 年',
        'Edricklam123/TWSE_ShortQuotaScrape': '盤中借券可用額度即時快照（2022 年幾天），不是歷史餘額',
        'acronhuang/TW_STOCK、kaigiii/StockHunter_Finmind、AChao0212/clarifindata、anhtuan18602/TWSE、Justin179/twse-stock-fetcher': '抓取程式，沒有存檔',
        'Yuan-Hsi/SALL_Agency': 'TEJ 匯出 2022～2023 個股日資料（含融資融券、無借券、無外資持股比率），僅部分個股',
        'MingKaiLin666/final-project': '單一標的 2016 起外資持股比率（研究用），不足以做橫斷面',
        'JokerRW/ETF_python': 'FinMind 借券 2019～2022 只有 0050、2618 兩檔',
        'flagtech/stock-db（本機）': '融券餘額 2013 起，但資料庫放在 Hugging Face（被擋），repo 只有程式；沒有借券、沒有外資持股',
        'code search': 'TWT93U、MI_QFIIS、SBLShortSalesCurrentDayBalance、ForeignInvestmentSharesRatio、借券賣出餘額、外資持股比率等關鍵字共約 1,000 筆結果，全部是程式、文件或單一個股小檔',
    },
    'conclusion': '借券賣出餘額（TWT93U）與外資持股比率（MI_QFIIS）都找不到 2015 年以前、甚至 2021 年以前的全市場存檔。'
                  'H3b 可用「2025-10-07 MI_QFIIS 快照 − 之後的外資淨買超 ÷ 股數」回推到 2021-10-08（inst_5y 起點）；H3a 的借券只有 2026-07-30 起 59 個交易日，無法測試。',
}
(L.HERE / 'results_data_audit.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps({k: v for k, v in out.items() if k != 'search_log'}, ensure_ascii=False, indent=1, default=str))
