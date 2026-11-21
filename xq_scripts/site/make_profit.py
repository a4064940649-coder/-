# 產生網站用的回測淨值與逐年報酬 profit.json（build_site_data.py 的 <profit.json> 參數）。
# 用法：python3 -I -B xq_scripts/site/make_profit.py [<navs.csv.gz> <0050 還原價 csv> <輸出 profit.json>]
# 預設：xq_scripts/site/navs_official.csv.gz（2026-11-21 起正式數字＝除權息＋期限修正＋減資復牌修正＋0050 官方含息序列：
#   EXRIGHT_FIX＋DEADLINE_FIX＋INS15_MASK＋REDUCE_FIX 全開、E50_SOURCE＝官方自建檔，research/tw_stock_selection/agents/official_nav/run_official.py 產生；
#   欄位同 navs_exright.csv.gz）、research/tw_stock_selection/raw/e50_tr_official/etf_0050_tr.csv（0050 官方含息序列，正式管線 P.E50 的來源；
#   2026-11-20 以前用的是 /home/user/jonatw/finlab-txd/data/raw/etf_0050.csv，要重現舊圖可當第二個參數給）、xq_scripts/site/profit.json。
# - 現行策略＝navs 的 E0|W10_70 欄、XQ版＝XT|W10_70 欄（照抄，不改）。
#   navs_exright.csv.gz 的 XT|W10_70＝XQ 門檻版、換股日用 A10（10 日（含）後第一個交易日的下一個交易日，＝統一後的 20／21／22 號腳本）；
#   XT_B11|W10_70＝舊腳本換股日 B11（第一個日期 ≥ 11 的交易日），只供對照，本程式不讀。
#   navs_official.csv.gz：E0 欄兩項期限修正都生效；XT 欄（A10）只有期限例外表生效（INS15 遮蔽不適用：XQ 門檻版訊號函式不讀這個開關）；XT_B11 欄不受期限開關影響。
#   舊版（保留不刪，可當第一個參數給）：xq_scripts/site/navs_exright.csv.gz＝只有除權息修正（2026-11-16～11-17 的網站）；
#   research/tw_stock_selection/agents/s19_xq_threshold/navs.csv.gz＝修正前（XT 欄是 B11）。
# - 0050：navs 的 0050 欄存成整數（97～854，四捨五入誤差最多約 0.5%），所以改用精確的 0050 含息還原價（adj），
#   照引擎做法對到 navs 的交易日（reindex＋前值補），0050 淨值＝100 萬 × adj ÷ 起始日 adj。
# - 週資料：每週取最後一個交易日，日期標籤就是那一天（不會超過回測最後一天，不倒退、不重複）。
# - 逐年：當年最後一個交易日 ÷ 前一年最後一個交易日 − 1（第一年從回測起始日起算），% 取 1 位。
import sys, json
import pandas as pd

HERE = '/home/user/-'
navs_p = sys.argv[1] if len(sys.argv) > 1 else f'{HERE}/xq_scripts/site/navs_official.csv.gz'
e50_p = sys.argv[2] if len(sys.argv) > 2 else f'{HERE}/research/tw_stock_selection/raw/e50_tr_official/etf_0050_tr.csv'
out_p = sys.argv[3] if len(sys.argv) > 3 else f'{HERE}/xq_scripts/site/profit.json'

n = pd.read_csv(navs_p, parse_dates=['date']).set_index('date').sort_index()
e = pd.read_csv(e50_p, parse_dates=['date']).set_index('date')['adj'].sort_index()
E = e.reindex(n.index.union(e.index)).ffill().reindex(n.index)
assert E.notna().all(), '0050 還原價沒有涵蓋整段回測'
S = {'cur': n['E0|W10_70'], 'xq': n['XT|W10_70'], 'etf': E / E.iloc[0] * 1e6}

# 週：以週五結束的週分組，取該週最後一個交易日（標籤＝那一天）
wk = n.index.to_period('W-FRI')
last_days = pd.Series(n.index, index=n.index).groupby(wk).max().tolist()
dates = [d.strftime('%Y-%m-%d') for d in last_days]
assert dates == sorted(set(dates)) and dates[-1] == n.index[-1].strftime('%Y-%m-%d')


def yearly(s):
    ye = s.groupby(s.index.year).last(); out = {}; prev = s.iloc[0]
    for y in ye.index:
        out[str(y)] = round(float(ye.loc[y] / prev - 1) * 100, 1); prev = ye.loc[y]
    return out


def cagr(s, d0):
    s = s.loc[d0:]; yrs = (s.index[-1] - s.index[0]).days / 365.25
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1)


P = {'dates': dates}
for k, s in S.items():
    P[k] = [int(round(float(s.loc[d]))) for d in last_days]
P['stats'] = {k: {'final': float(s.iloc[-1]), 'cagr': cagr(s, s.index[0]), 'dd': float((s / s.cummax() - 1).min()), 'c19': cagr(s, '2019-01-01')} for k, s in S.items()}
P['years'] = {k: yearly(s) for k, s in S.items()}
P['source'] = {'navs': navs_p, '0050': e50_p + '（adj 欄＝含息還原價，正式管線 P.E50 的來源）', 'program': 'xq_scripts/site/make_profit.py'}
json.dump(P, open(out_p, 'w', encoding='utf-8'), ensure_ascii=False)
print('ok', out_p, len(dates), dates[0], dates[-1], P['years']['etf'])
