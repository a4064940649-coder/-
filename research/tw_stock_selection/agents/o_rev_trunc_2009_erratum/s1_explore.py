#!/usr/bin/env python3
"""第 1 步：探索（只看覆蓋，不看任何名次、分數或報酬）。
(1) 截斷形狀：每檔「營收最後月份」對「面板最後交易日」；每月自然結束檔數；缺口月數分布；19 檔（listing_master 2010～2011 下市）。
(2) 93 個訊號日（2006-04～2013-12），主策略 C0 池（前 300 流動性、價格與成交值合格、52 週有值、非處置、隔日有開盤、排除金融）內
    營收 3 月年增率缺值的股月，依「原因」分類：
      TRUNC   ：營收最後月份落在截斷帶（2009-09～2009-11）且之後仍有交易 → (a) 2009-11 截斷
      NOLEVEL ：其他「需要的月份完全沒有營收水準」的國內股 → (b) 其他 NOLEVEL
      FOREIGN ：KY／DR（代號 91xx、名稱含 KY 或 DR）或從未出現在 _0 國內表 → (c)
      NOBASE  ：有水準但缺 12 個月前基期（新上市）或基期 ≤ 0（設計上本來就缺，非存活者偏誤）
    與 o_rev_survivorship 的 382 個 NOLEVEL 股月（全市場、含金融、2006～2012）逐筆對帳。
本程式不計算任何名次、分數、報酬。用法：python3 -I -B s1_explore.py"""
import sys, json, csv, time, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
sys.path.insert(0, str(AG / 'main_v2'))
import numpy as np, pandas as pd
import v2_core as V
sys.path.pop(0)
K = V.K
T0 = time.time()
COLS = list(K.COLS); CI = {c: j for j, c in enumerate(COLS)}; N = len(COLS)
TRUNC_BAND = (pd.Period('2009-09', 'M'), pd.Period('2009-11', 'M'))

# ---------- 營收 ----------
d = pd.read_csv(K.LONG, dtype={'code': str, 'market': str})
d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
d['pri'] = (d['market'] != 'sii').astype(int)
d = d.sort_values(['p', 'code', 'pri']).drop_duplicates(['p', 'code'])
LV = d.pivot(index='p', columns='code', values='revenue_k_twd')
LV = LV.reindex(pd.period_range(LV.index.min(), LV.index.max(), freq='M'))
lastrev = {c: LV[c].dropna().index.max() for c in LV.columns}
firstrev = {c: LV[c].dropna().index.min() for c in LV.columns}
NAME = {}
for c, nm in zip(d['code'], d['name']): NAME.setdefault(c, nm)
for r in csv.DictReader(open(K.X.parent / 'stock_list.csv', encoding='utf-8')): NAME.setdefault(r['code'], r['name'])
LM = {r['code']: r for r in csv.DictReader(open(AG / 'o_v7_listing_exec_engine/listing_master.csv', encoding='utf-8'))}
# 歷史名稱（含 F- 前綴的外國公司）：o13 收的舊代碼表（只讀）
SNAPN = {}
for f in ('grs_568e6685_tools_twse_list.csv', 'grs_893c8772_grs_stock_no.csv', 'goristock_c3c718ab_grs_stock_no.csv'):
    for r in csv.reader(open(AG / 'o_pit_shares_industry_2006_2013/raw/industry_snap' / f, encoding='utf-8', errors='replace')):
        r = [x.strip().strip('"') for x in r]
        if len(r) >= 2 and r[0].isdigit() and len(r[0]) == 4: SNAPN.setdefault(r[0], set()).add(r[1])

def is_foreign(c):
    nms = {NAME.get(c, '') or ''} | SNAPN.get(c, set())
    return c[:2] == '91' or any(('KY' in n.upper()) or n.endswith('DR') or n.startswith('F-') for n in nms)

# ---------- (1) 截斷形狀 ----------
lt = {}
for j, c in enumerate(COLS):
    w = np.where(K.TRADED[:, j])[0]
    lt[c] = K.IDX[w[-1]] if len(w) else pd.NaT
END = K.IDX[-1]
rows = []
for c in COLS:
    if pd.isna(lt[c]): continue
    ltm = pd.Period(lt[c], 'M')
    lr = lastrev.get(c)
    gap = (ltm - lr).n if lr is not None and not pd.isna(lr) else None
    rows.append(dict(code=c, name=NAME.get(c, ''), last_trade=str(lt[c].date()), last_trade_m=str(ltm), last_rev=str(lr) if lr is not None else '',
                     gap_months=gap, delist_date=LM.get(c, {}).get('delist_date', ''), foreign=is_foreign(c),
                     alive=bool(lt[c] >= END - pd.Timedelta(days=30))))
T = pd.DataFrame(rows)
dead = T[~T.alive]
# 每月「營收自然結束」檔數：已下市股（面板內）營收最後月份的分布
nat = dead[dead.last_rev != ''].groupby('last_rev').size()
nat_0812 = {str(p): int(nat.get(str(p), 0)) for p in pd.period_range('2008-01', '2013-12', freq='M')}
# P1：最後交易日在 2009-12～2012-12 的國內股
dom = T[~T.foreign]
w = dom[(dom.last_trade_m >= '2009-12') & (dom.last_trade_m <= '2012-12')]
p1_n = len(w); p1_le = int(((w.last_rev != '') & (w.last_rev <= '2009-11')).sum()); p1_norev = int((w.last_rev == '').sum())
w13 = dom[(dom.last_trade_m >= '2013-01') & (dom.last_trade_m <= '2014-12')]
p1_13_le = int(((w13.last_rev != '') & (w13.last_rev <= '2009-11')).sum())
# 19 檔
L19 = [c for c, r in LM.items() if r['delist_date'][:4] in ('2010', '2011')]
t19 = []
for c in sorted(L19):
    t19.append(dict(code=c, name=NAME.get(c, ''), delist=LM[c]['delist_date'], last_listed=LM[c]['last_listed_day'],
                    last_trade_panel=str(lt[c].date()) if c in lt and pd.notna(lt[c]) else '', last_rev=str(lastrev.get(c, '')),
                    foreign=is_foreign(c), fin=bool(c in V.FIN)))
gt2 = dom[(~dom.alive) & dom.gap_months.notna() & (dom.gap_months > 2)].sort_values('last_trade')
gaps_hist = dom[(~dom.alive) & dom.gap_months.notna()].gap_months.clip(upper=48).value_counts().sort_index()

# ---------- (2) 93 個訊號日 C0 池缺值分類 ----------
YL, Y5, _ = V.load_revenue()
S = V.V2('gapfix', YL, Y5, exfin=True)
days = [x for x in S.sig_days if pd.Timestamp('2006-01-01') <= x <= pd.Timestamp('2013-12-31')]
assert len(days) == 93, len(days)

def month_cause(c, x):
    """營收面板在月份 x 沒有 c 的水準（或水準 ≤ 0 當基期）的原因。"""
    lr, fr = lastrev[c], firstrev[c]
    ltm = pd.Period(lt[c], 'M')
    if x > lr:
        if pd.Period('2009-12', 'M') <= ltm <= pd.Period('2012-12', 'M'): return 'TRUNC'     # (a) 2009-11 斷崖：2010～2012 下市
        if x >= pd.Period('2013-01', 'M') and lr >= pd.Period('2012-12', 'M'): return 'IFRS13'  # (b) 2013 IFRS 年度頁只剩存活者
        return 'PRE12'                                                                     # (b) 舊頁：下市前約 12 個月營收消失
    if x < fr: return 'NEWLIST'
    return 'HOLE'                                                                          # (b) 序列中間的洞（2011-01／02、2012-01 等）

PRIO = ['TRUNC', 'IFRS13', 'PRE12', 'HOLE', 'NEWLIST', 'BASE_LE0']
def cause(c, m):
    if is_foreign(c): return 'FOREIGN'
    if c not in LV.columns: return 'NEVER_DOM'
    ks = set()
    for x in [m - k for k in (1, 2, 3)] + [m - k for k in (13, 14, 15)]:
        v = LV[c].get(x, np.nan)
        if not np.isfinite(v): ks.add(month_cause(c, x))
        elif v <= 0 and x <= m - 13: ks.add('BASE_LE0')
    for k in PRIO:
        if k in ks: return k
    return 'UNKNOWN'

SURV = ('TRUNC', 'IFRS13', 'PRE12', 'HOLE', 'NEVER_DOM')   # 存活者偏誤型（判定口徑）

G = []; perday = []
for dd in days:
    t = K.IDX.get_loc(dd); e = t + 1
    v60 = S.V60[t]; px = K.Cn[t]
    base = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10)
    vv = np.where(base, v60, -np.inf)
    order = np.argsort(-vv, kind='stable')
    top = np.zeros(N, bool); top[order[:min(300, base.sum())]] = True; top &= base
    el0 = top & np.isfinite(S.S52[t]) & ~K.disposal_mask(dd) & np.isfinite(K.On[e]) & K.TRADED[e]
    y3 = K.yoy3_for_month(YL, pd.Period(dd, 'M'))
    miss_all = el0 & ~np.isfinite(y3)
    miss = miss_all & ~V.FINMASK
    m = pd.Period(dd, 'M')
    cnt = {}
    for j in np.where(miss)[0]:
        c = COLS[j]; k = cause(c, m)
        G.append(dict(sig=str(dd.date()), code=c, name=NAME.get(c, ''), cause=k, last_rev=str(lastrev.get(c, '')),
                      last_trade=str(lt[c].date()) if pd.notna(lt[c]) else ''))
        cnt[k] = cnt.get(k, 0) + 1
    perday.append(dict(sig=str(dd.date()), pool_ex_rev_C0=int((el0 & ~V.FINMASK).sum()), eligible_C0=int((el0 & ~V.FINMASK & np.isfinite(y3)).sum()),
                       miss=int(miss.sum()), miss_fin=int((miss_all & V.FINMASK).sum()), **cnt))
G = pd.DataFrame(G); PD = pd.DataFrame(perday).fillna(0)
G.to_csv(HERE / 'miss_stockmonths_C0.csv', index=False)

def seg(df, a, b):
    x = df[(df.sig >= a) & (df.sig <= b)]
    return {k: int(v) for k, v in x.groupby('cause').size().items()}
segs = {'2006-04~2009-11': seg(G, '2006-04-01', '2009-11-30'), '2009-12~2013-12': seg(G, '2009-12-01', '2013-12-31'),
        '2006~2012': seg(G, '2006-01-01', '2012-12-31'), '2010~2012': seg(G, '2010-01-01', '2012-12-31'),
        '2006~2013': seg(G, '2006-01-01', '2013-12-31')}
PD['surv'] = sum(PD[k] if k in PD else 0 for k in SURV)
pd_0913 = PD[(PD.sig >= '2009-12-01') & (PD.sig <= '2013-12-31')]

# 對帳：o_rev_survivorship 382（全市場含金融、2006～2012、gap.py 定義）
O = pd.read_csv(AG / 'o_rev_survivorship/gap_stockmonths.csv', dtype={'code': str})
O = O[(O.type == 'NOLEVEL') & (O.sig < '2013-01-01')]
assert len(O) == 382, len(O)
Gk = set(zip(G.sig, G.code)); Ok = set(zip(O.sig, O.code))
only_o = sorted(Ok - Gk); only_g = sorted(k for k in Gk - Ok if k[0] < '2013-01-01')
def why_only_o(sig, c):
    j = CI[c]
    if V.FINMASK[j]: return 'C0 金融（本題排除）'
    return 'other'
oo = pd.Series([why_only_o(s, c) for s, c in only_o]).value_counts().to_dict()
oo_other = [(s, c, NAME.get(c, '')) for s, c in only_o if why_only_o(s, c) == 'other']
Gc = G.set_index(['sig', 'code'])
og = pd.Series([Gc.loc[k, 'cause'] for k in only_g]).value_counts().to_dict() if only_g else {}
# 382 本身的原因分類
O['cause'] = [cause(c, pd.Period(s, 'M')) for s, c in zip(O.sig, O.code)]
O['fin'] = [bool(V.FINMASK[CI[c]]) for c in O.code]
c382 = O.groupby(['cause', 'fin']).size().reset_index().rename(columns={0: 'n'}).to_dict(orient='records')
o382_by_year = O.groupby(O.sig.str[:4]).size().to_dict()
# 19 檔在 C0 池中的缺值股月
g19 = G[G.code.isin(L19)].groupby('code').size().to_dict()

out = dict(
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), no_ranks='本程式未計算任何名次、分數或報酬',
    P1=dict(n_domestic_last_trade_2009_12_2012_12=p1_n, n_last_rev_le_2009_11=p1_le, n_no_rev_ever=p1_norev,
            share_le_2009_11=p1_le / max(p1_n, 1), share_le_2009_11_excl_norev=p1_le / max(p1_n - p1_norev, 1),
            n_last_trade_2013_2014=len(w13), n_2013_2014_last_rev_le_2009_11=p1_13_le,
            stocks=w[['code', 'name', 'last_trade', 'last_rev', 'delist_date']].to_dict(orient='records')),
    natural_end_by_month_delisted=nat_0812,
    gap_months_hist_delisted_domestic={str(int(k)): int(v) for k, v in gaps_hist.items()},
    gap_gt2_domestic_delisted=gt2[['code', 'name', 'last_trade', 'last_rev', 'gap_months']].to_dict(orient='records'),
    L19=t19,
    C0_missing=dict(n_days=len(days), segs=segs, per_day_mean_SURV_2009_12_2013_12=float(pd_0913.surv.mean()), per_day_mean_SURV_2006_04_2009_11=float(PD[PD.sig < '2009-12-01'].surv.mean()),
                    per_day_range_0913=[int(pd_0913.surv.min()), int(pd_0913.surv.max())], SURV=list(SURV),
                    per_day=PD.to_dict(orient='records'), L19_stockmonths_in_C0_pool=g19,
                    codes_TRUNC=sorted(G[G.cause == 'TRUNC'].code.unique().tolist()),
                    codes_by_cause={k: sorted(G[G.cause == k].code.unique().tolist()) for k in G.cause.unique()}),
    reconcile_382=dict(n_382=382, by_year=o382_by_year, cause_fin=c382, n_only_in_382=len(only_o), only_in_382_reason=oo, only_in_382_other=oo_other,
                       n_only_in_mine_pre2013=len(only_g), only_in_mine_cause=og,
                       note='382＝gap.py：全市場（exfin=False）、只算 NOLEVEL（三個月都無水準）；本題 C0 排除金融，且依原因分類（截斷後頭兩個月 gap.py 判 NOBASE）'),
    runtime_s=time.time() - T0)
(HERE / 'explore.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps({k: v for k, v in out.items() if k not in ('L19',)}, ensure_ascii=False, default=str)[:6000])
print(pd.DataFrame(t19).to_string())
# 逐檔摘要（依原因）
Gs = G.groupby(['cause', 'code']).agg(n=('sig', 'size'), first=('sig', 'min'), last=('sig', 'max'), name=('name', 'first'),
                                      last_rev=('last_rev', 'first'), last_trade=('last_trade', 'first')).reset_index()
Gs['delist'] = Gs.code.map(lambda c: LM.get(c, {}).get('delist_date', ''))
ex = json.loads((HERE / 'explore.json').read_text(encoding='utf-8'))
ex['C0_missing']['by_stock'] = Gs.to_dict(orient='records')
dom09 = G[(G.sig >= '2009-12-01') & (G.cause != 'FOREIGN')]
ex['P2'] = dict(domestic_missing_2009_12_2013_12=int(len(dom09)), domestic_SURV_2009_12_2013_12=int(dom09.cause.isin(SURV).sum()),
                domestic_SURV_2006_04_2009_11=int(((G.sig < '2009-12-01') & G.cause.isin(SURV)).sum()),
                SURV_2006_2012=int(((G.sig < '2013-01-01') & G.cause.isin(SURV)).sum()), SURV_2006_2013=int(G.cause.isin(SURV).sum()))
(HERE / 'explore.json').write_text(json.dumps(ex, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(Gs.to_string())
print(json.dumps(ex['P2'], ensure_ascii=False))
