# -*- coding: utf-8 -*-
"""步驟 0：庫藏股公告閘門。規則全部取自 gate_prereg.json（事前登記，先斷言存在）。輸出 gate.json、gate_events.csv。"""
import json, hashlib, datetime
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent
G = json.loads((HERE / 'gate_prereg.json').read_text(encoding='utf-8'))
assert G['sha256'] == hashlib.sha256(json.dumps(G['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
P = G['PREREG']
PANEL = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
T = pd.read_parquet(AG / 'o_cb_board_runup' / 'mops_titles.parquet')
T = T[(T.date >= '2011-01-01') & (T.date <= '2018-12-31')].copy()
t = T.title.fillna('')
def has_any(words): return t.apply(lambda s: any(w in s for w in words))
R = P['resolution_rule']; C = P['completion_rule']
is_res = t.str.contains('買回') & has_any(R['must_contain_any_A']) & has_any(R['must_contain_any_B']) & ~has_any(R['exclude_any'])
is_cmp = t.str.contains('買回') & has_any(C['must_contain_any']) & ~has_any(C['exclude_any'])
is_any_bb = t.str.contains('買回') & ~has_any(['更正', '補充', '修正', '子公司', '附買回', '公司債', '債券', '可轉換']) & ~is_cmp
wide = t.str.contains('買回') & has_any(['董事會', '決議', '通過'])
res = T[is_res][['code', 'name', 'date', 'tm', 'title']].assign(kind='resolution')
cmp_ = T[is_cmp][['code', 'date', 'title']]
anyb = T[is_any_bb][['code', 'date', 'tm', 'title']]
# 放回：完成類標題前 75 日內沒有決議事件
put = []
rs = {c: g.date.values for c, g in res.groupby('code')}
for c, d, ti in zip(cmp_.code, cmp_.date, cmp_.title):
    prev = rs.get(c, np.array([], dtype='datetime64[ns]'))
    if len(prev) and ((prev <= np.datetime64(d)) & (prev >= np.datetime64(d - pd.Timedelta(days=75)))).any():
        continue
    cand = anyb[(anyb.code == c) & (anyb.date < d) & (anyb.date >= d - pd.Timedelta(days=75))]
    if len(cand):
        r0 = cand.sort_values('date').iloc[0]
        put.append(dict(code=c, name=None, date=r0.date, tm=r0.tm, title=r0.title, kind='putback_found'))
    else:
        put.append(dict(code=c, name=None, date=d - pd.Timedelta(days=61), tm='99:99:99', title='[imputed from] ' + ti, kind='putback_imputed'))
E = pd.concat([res, pd.DataFrame(put)], ignore_index=True).sort_values(['code', 'date', 'tm']).reset_index(drop=True)
# 同公司 30 日內合併
keep = []; last = {}
for i, r in E.iterrows():
    lp = last.get(r.code)
    if lp is not None and (r.date - lp).days <= 30: continue
    keep.append(i); last[r.code] = r.date
E = E.loc[keep].reset_index(drop=True)
E = E[E.date <= '2018-12-31']
# 面板
val = pd.read_parquet(PANEL / 'value.parquet'); adj = pd.read_parquet(PANEL / 'adj.parquet')
opn = pd.read_parquet(PANEL / 'open.parquet'); clr = pd.read_parquet(PANEL / 'close_raw.parquet')
good = val.notna().sum(axis=1) >= 300
v60 = val[good].rolling(60, min_periods=50).mean().reindex(val.index).ffill()
IDX = val.index
E = E[E.code.isin(val.columns)].copy()
sig = []
for d, tm in zip(E.date, E.tm):
    i = IDX.searchsorted(d, 'right') - 1      # 公告日之前（含）最後一個交易日
    if IDX[i] == d and str(tm) < '13:30:00':  # 盤中公告：訊號日仍取當天收盤後可知（隔日開盤進場）——當天收盤後才算訊號，同樣隔日開盤
        pass
    sig.append(i)
E['ti'] = sig
E['sig'] = IDX[E.ti.values]
E['v60'] = [v60.at[IDX[i], c] if c in v60.columns else np.nan for i, c in zip(E.ti, E.code)]
SL = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype=str)
fin = set(SL[SL.industry == '金融保險業'].code) | {c for c in val.columns if c.startswith('28')} | {'5876', '5880'}
E['fin'] = E.code.isin(fin)
e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj']
e50 = e50.reindex(IDX.union(e50.index)).ffill().reindex(IDX)
r20 = []; x20 = []
for i, c in zip(E.ti, E.code):
    e = i + 1; x = e + 19
    if x >= len(IDX): r20.append(np.nan); x20.append(np.nan); continue
    o = opn.iat[e, opn.columns.get_loc(c)]; a = adj.iat[e, adj.columns.get_loc(c)]; cr = clr.iat[e, clr.columns.get_loc(c)]
    a20 = adj.iloc[e:x + 1][c].dropna()
    if not (o > 0 and cr > 0 and a > 0) or len(a20) == 0: r20.append(np.nan); x20.append(np.nan); continue
    ent = o * a / cr
    rr = a20.iloc[-1] / ent - 1 - 0.0087
    r20.append(rr)
    eo = e50.iat[e - 1]  # 0050 只有收盤：以前一日收盤近似進場
    x20.append(rr - (e50.iat[x] / eo - 1))
E['r20_net'] = r20; E['x20_net_vs0050'] = x20
E.to_csv(HERE / 'gate_events.csv', index=False)
def summ(m):
    s = E[m]
    return dict(n=int(len(s)), n_imputed=int((s.kind == 'putback_imputed').sum()), n_putback_found=int((s.kind == 'putback_found').sum()),
                by_year={int(k): int(v) for k, v in s.groupby(s.date.dt.year).size().items()},
                median_r20_net=None if s.r20_net.notna().sum() == 0 else float(s.r20_net.median()),
                mean_r20_net=None if s.r20_net.notna().sum() == 0 else float(s.r20_net.mean()),
                median_x20_net_vs0050=None if s.x20_net_vs0050.notna().sum() == 0 else float(s.x20_net_vs0050.median()),
                n_days=int(s.sig.nunique()))
out = dict(prereg_sha256=G['sha256'], time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
           n_titles_2011_2018=int(len(T)), n_resolution_titles=int(is_res.sum()), n_completion_titles=int(is_cmp.sum()),
           n_wide_titles_upper=int(wide.sum()), n_putback=len(put),
           all=summ(E.v60.notna() | True), ge1e8=summ(E.v60 >= 1e8), ge2e8=summ(E.v60 >= 2e8),
           ge2e8_exfin=summ((E.v60 >= 2e8) & ~E.fin), ge2e8_no_imputed=summ((E.v60 >= 2e8) & (E.kind != 'putback_imputed')),
           codes_in_source_2011_2018=int(T.code.nunique()))
# 寬上界（不作判定）：任何含「買回」＋「董事會／決議／通過」，同公司 30 日合併
W = T[wide][['code', 'date']].sort_values(['code', 'date'])
kw = []; lw = {}
for c, d in zip(W.code, W.date):
    if c in lw and (d - lw[c]).days <= 30: continue
    kw.append((c, d)); lw[c] = d
W = pd.DataFrame(kw, columns=['code', 'date']); W = W[W.code.isin(val.columns)]
wi = IDX.searchsorted(W.date.values, 'right') - 1
W['v60'] = [v60.at[IDX[i], c] for i, c in zip(wi, W.code)]
out['wide_upper_bound'] = dict(n=int(len(W)), ge2e8=int((W.v60 >= 2e8).sum()), ge1e8=int((W.v60 >= 1e8).sum()))
a_ok = out['ge2e8']['n'] >= 200
m = out['ge2e8']['median_r20_net']
b_ok = (m is not None) and m > 0
out['gate'] = dict(a_count_ge200=a_ok, b_median_r20_net_gt0=b_ok, passed=bool(a_ok and b_ok),
                   decision='主題關閉，改做替代題' if not (a_ok and b_ok) else '主題開啟')
(HERE / 'gate.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
