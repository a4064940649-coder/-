# -*- coding: utf-8 -*-
"""反方審查（成本與可成交）：o_cb_board_runup。只讀研究員檔案，不計算任何事件報酬（主題閘門 (b) 未開）。
python3 -I -B review_exec_o3.py"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent
sys.path.insert(0, str(AG / 'independent_audit'))
import ia_core as K
sys.path.pop(0)
out = {}

# ---------- A. 主題：157 筆事件的可成交性 ----------
E = pd.read_csv(HERE / 'events_board.csv', dtype={'code': str}, parse_dates=['board', 'list_date', 'fsc_recv', 'entry_date'])
cb = E[(E.kind == 'CB') & (E.board >= '2010-01-01') & (E.board <= '2018-12-31')]
base = cb[cb.j.notna() & cb.traded_entry.fillna(False).astype(bool)]
liq = base[(base.adv60 >= 3e7) & ~base.fin.astype(bool)].copy()
out['n_reproduced'] = int(len(liq))
tm = pd.to_datetime(liq['tm'], format='%H:%M:%S', errors='coerce')
h = tm.dt.hour + tm.dt.minute / 60
out['ann_time'] = dict(before_0900=int((h < 9).sum()), intraday_0900_1330=int(((h >= 9) & (h < 13.5)).sum()), after_1330=int((h >= 13.5).sum()), na=int(h.isna().sum()))
# 公告日是不是交易日（非交易日公告 → 隔一交易日開盤，可行）
isTD = liq.board.isin(K.IDX)
out['ann_on_nontrading_day'] = int((~isTD).sum())
# 進場日開盤漲停（買不到）
lu = []
for _, r in liq.iterrows():
    t = int(K.IDX.get_loc(r['entry_date'])); c = K.CI.get(r['code'])
    lu.append(bool(K.gap_up_open(t)[c]) if c is not None else None)
liq['lu_open'] = lu
out['entry_open_limit_up'] = int(np.nansum([x for x in lu if x is not None]))
# 容量：單筆 ≤ 60 日均成交值 5%
q = liq.adv60.quantile([.1, .25, .5, .75, .9]).div(1e6).round(1).to_dict()
out['adv60_mn_quantiles'] = q
out['capacity'] = {f'adv60>={k}': int((liq.adv60 >= v).sum()) for k, v in
                   [('0.6億(30萬@0.5%)', 6e7), ('1億', 1e8), ('2億(1000萬@5%)', 2e8), ('3.33億(1000萬@3%)', 3.333e8)]}
out['capacity_by_year_ge2e8'] = liq[liq.adv60 >= 2e8].groupby(liq.board.dt.year).size().to_dict()
out['pos_share_of_adv_at_3e7'] = dict(cap_30w=3e5 / 3e7, cap_1000w=1e7 / 3e7)
# 持有期（董事會→基準日）與資金使用率
hb = liq.h_board_base.dropna()
out['hold_td'] = dict(median=float(hb.median()), q25=float(hb.quantile(.25)), q75=float(hb.quantile(.75)), n=int(len(hb)), n_missing_base=int(liq.h_board_base.isna().sum()))
out['events_per_year'] = float(len(liq) / 8)
out['avg_concurrent_positions'] = float(len(liq) / 8 * hb.median() / 245)
out['cost_rt_normal_pct'] = round((K.rates(0.001)[0] + K.rates(0.001)[1]) * 100, 4)

# ---------- B. 上界 199 的另一種算法：用 TPEx CB 掛牌（不受 MOPS 存活者清單與 FSC 2016 缺年限制） ----------
TP = pd.read_csv(AG / 'o_issuance_avoid' / 'tpex_cb_listings.csv', dtype={'cb_code': str, 'issuer': str}, parse_dates=['list_date'])
TP = TP[TP.issuer_twse.astype(str) == 'True']
SL = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype=str).drop_duplicates('code').set_index('code')
V60 = K.V.rolling(60, min_periods=60).mean().to_numpy()
FINC = set(c for c in K.COLS if str(SL['industry'].get(c, '')).find('金融') >= 0 or c[:2] in ('28', '58'))
rows = []
for _, r in TP.iterrows():
    c = r['issuer']; j = K.CI.get(c)
    if j is None: continue
    tl = int(K.IDX.searchsorted(r['list_date']))
    tb = tl - 51   # 董事會→掛牌中位數 51 交易日
    if tb < 60 or tb >= len(K.IDX): continue
    bd = K.IDX[tb]
    if not (pd.Timestamp('2010-01-01') <= bd <= pd.Timestamp('2018-12-31')): continue
    if '-' in str(r['cb_name']) or 'E' in str(r['cb_code']): pass
    rows.append(dict(code=c, cb=r['cb_code'], board_est=bd, adv60=V60[tb - 1, j], fin=c in FINC))
TB = pd.DataFrame(rows)
tb_ok = TB[(TB.adv60 >= 3e7) & ~TB.fin]
out['ub_tpex_listed_liquid_exfin'] = int(len(tb_ok))
out['ub_tpex_by_year'] = tb_ok.groupby(tb_ok.board_est.dt.year).size().to_dict()
# 加上 FSC 撤回／退件（非生效）＋ MOPS 未送件
F = pd.read_csv(AG / 'o_issuance_avoid' / 'fsc_cases_all.csv', dtype={'code': str}, parse_dates=['recv'])
F = F[(F.ctype == '上市') & F['item'].astype(str).str.contains('轉換') & ~F['item'].astype(str).str.contains('海外|私募|金融債')]
Fw = F[(F.result != '生效') & (F.recv >= '2010-01-01') & (F.recv <= '2018-12-31')]
nw = 0
for _, f in Fw.iterrows():
    j = K.CI.get(f['code']); t0 = int(K.IDX.searchsorted(f['recv'])) - 1
    if j is None or t0 < 60: continue
    if V60[t0, j] >= 3e7 and f['code'] not in FINC: nw += 1
out['ub_fsc_withdrawn_liquid'] = nw
nf = int((liq.outcome.isin(['no_filing', 'unknown_fsc_gap'])).sum())
out['ub_alt_total'] = int(len(tb_ok) + nw + nf)

# ---------- C. 替代題：轉上市標記來源 ----------
LS = pd.read_csv(HERE / 'alt_ipo_lockup' / 'listings.csv', dtype={'code': str}, parse_dates=['first', 'otc_rev_first', 'fsc_otc_first'])
tr = LS[LS.kind == 'transfer']
rev_only = tr[tr.fsc_otc_first.isna() | (tr.fsc_otc_first >= tr['first'])]
out['transfer_flag_revenue_only'] = int(len(rev_only))
out['transfer_flag_total'] = int(len(tr))
ipo = LS[LS.kind == 'ipo']
out['ipo_2016_listed'] = int((ipo['first'].dt.year == 2016).sum())
out['ipo_with_otc_rev_any'] = int(ipo.otc_rev_first.notna().sum())

# ---------- D. 替代題：A5 受影響股的成交面 ----------
for sg in ('SEL', 'VAL'):
    T = pd.read_csv(HERE / 'alt_ipo_lockup' / f'oos_v2_affected_{sg}.csv', dtype={'code': str}, parse_dates=['d'])
    out[f'A5_{sg}'] = dict(n_skipped=int((T.side == 'skipped').sum()), n_added=int((T.side == 'added').sum()),
                          months=int(T.d.nunique()))
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
