#!/usr/bin/env python3
"""獨立複製 o2：從原始來源（voidful csv、BlackBoxOP parquet、extras/inst_5y、margin_hist）自建外資持股比率面板與資料稽核數字。
不讀研究員的程式或其整理後檔案。輸出 my_fo.parquet（比率，0~1）、my_short.parquet、my_sbl.parquet、my_anchor.csv、my_data_audit.json"""
import json, glob, os
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
SP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad')
BB = SP / 'sbl_probe/BlackBoxOP_tw_stock/data'
VF = SP / 'sbl_probe/vf/data'
X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
PCOLS = pd.read_parquet(SP / 'panel/close_raw.parquet').columns
PC = set(PCOLS)
A = pd.Timestamp('2025-10-07')
out = {}

def rd_bb(sub):
    fs = sorted(glob.glob(str(BB / sub / '*.parquet')))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)

# ---------- MI_QFIIS ----------
q = pd.read_csv(VF / 'twse_foreign.csv', dtype={'code': str}, parse_dates=['date'])
q = q[q['code'].isin(PC) & (q['total_shares'] > 0)]
out['qfiis_days'] = int(q['date'].nunique()); out['qfiis_range'] = [str(q['date'].min().date()), str(q['date'].max().date())]
QS = q.pivot(index='date', columns='code', values='foreign_shares').astype(float)
QT = q.pivot(index='date', columns='code', values='total_shares').astype(float)
QR = QS / QT
anchor_sh = QT.loc[A].dropna()
anchor_sh.rename('shares').to_csv(HERE / 'my_anchor.csv')

# ---------- 外資淨買超（股） ----------
i5 = pd.read_csv(X / 'inst_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
i5 = i5[i5['code'].isin(PC)]
F5 = i5.pivot_table(index='date', columns='code', values='foreign', aggfunc='last') * 1000.0
bi = rd_bb('institutional'); bi['date'] = pd.to_datetime(bi['Date'])
bi = bi[bi['Ticker'].isin(PC)]
FB = bi.pivot_table(index='date', columns='Ticker', values='Foreign_Net', aggfunc='last').astype(float)
# 重疊期一致率：inst_5y vs BB
ov = F5.index.intersection(FB.index)
a = F5.loc[ov].stack(); b = FB.loc[ov].stack()
j = pd.concat([a.rename('a'), b.rename('b')], axis=1).dropna()
out['inst5y_vs_bb'] = dict(n=len(j), match=float((np.abs(j.a.round() - j.b) <= 1).mean()), days=len(ov))
vf = pd.read_csv(VF / 'twse_flows.csv', dtype={'code': str}, parse_dates=['date'])
vf = vf[vf['code'].isin(PC)]
FV = vf.pivot_table(index='date', columns='code', values='foreign_net', aggfunc='last').astype(float)
ov2 = FV.index.intersection(FB.index)
j2 = pd.concat([FV.loc[ov2].stack().rename('a'), FB.loc[ov2].stack().rename('b')], axis=1).dropna()
out['vf_vs_bb'] = dict(n=len(j2), match=float((j2.a == j2.b).mean()), days=len(ov2))
# 也試 voidful = 外資＋外資自營
FBd = bi.pivot_table(index='date', columns='Ticker', values='Foreign_Dealer_Net', aggfunc='last').astype(float)
j3 = pd.concat([FV.loc[ov2].stack().rename('a'), (FB + FBd.reindex_like(FB).fillna(0)).loc[ov2].stack().rename('b')], axis=1).dropna()
out['vf_vs_bb_incl_dealer'] = float((j3.a == j3.b).mean())
NET = pd.concat([F5[F5.index <= pd.Timestamp('2026-10-07')], FB[FB.index >= pd.Timestamp('2026-10-08')]]).sort_index()
NET = NET.fillna(0.0)

# ---------- MI_QFIIS 日變動 vs 淨買超（同日、遞延） ----------
dQ = QS.diff()
for lag in (0, 1, 2):
    n = NET.shift(lag).reindex(dQ.index)
    jj = pd.concat([dQ.stack().rename('d'), n.stack().rename('n'), QT.stack().rename('T')], axis=1).dropna()
    jj = jj.iloc[1:]
    ok = np.abs(jj.d - jj.n) <= 0.0002 * jj['T']
    nz = jj[(jj.d != 0) | (jj.n != 0)]
    out[f'qfiis_vs_net_lag{lag}'] = dict(n=len(jj), within_002pct=float(ok.mean()), corr=float(jj.d.corr(jj.n)),
                                         within_002pct_nonzero=float((np.abs(nz.d - nz.n) <= 0.0002 * nz['T']).mean()),
                                         corr_nonzero=float(nz.d.corr(nz.n)))

# ---------- 回推 ----------
def backcast(anchor_day, ratio_anchor, shares, net, days):
    """r(t) = r(anchor) − Σ_{s∈(t, anchor]} net(s)/shares，days ≤ anchor_day"""
    nn = net.reindex(columns=ratio_anchor.index).fillna(0.0)
    nn = nn[(nn.index <= anchor_day)]
    # 反向累加：cum(t) = Σ_{s>t, s≤A} net(s)
    rev = nn.iloc[::-1].cumsum().iloc[::-1]          # Σ_{s≥t}
    after = rev.shift(-1).fillna(0.0)                # Σ_{s>t}
    r = ratio_anchor - after / shares
    return r.reindex(days)

# 回推品質：在 MI_QFIIS 期內，從期末錨點往回推
end = QR.index[-1]
for hz, lab in ((252, '1y'), (126, '0.5y')):
    qd = QR.index
    t = qd[-1 - hz] if len(qd) > hz else qd[0]
    ra = QR.loc[end].dropna(); sh = QT.loc[end].reindex(ra.index)
    bc = backcast(end, ra, sh, NET, [t]).iloc[0].clip(0, 1)
    act = QR.loc[t].reindex(ra.index)
    e = (bc - act).dropna() * 100
    from scipy.stats import spearmanr
    out[f'backcast_{lab}'] = dict(t=str(t.date()), anchor=str(end.date()), n=len(e), MAE_pp=float(e.abs().mean()), p90_pp=float(e.abs().quantile(0.9)),
                                  spearman=float(spearmanr(bc.reindex(e.index), act.reindex(e.index)).correlation))
# 6 個月增量：用錨點 end 回推 t 與 t−126 的值，比較回推增量與實際增量
qd = QR.index; t2 = qd[-1 - 0]; t1 = qd[-1 - 126]
ra = QR.loc[end].dropna(); sh = QT.loc[end].reindex(ra.index)
# 用 1 年回推窗內的增量：t_a = end−252, t_b = end−126
ta, tb = qd[-1 - 252] if len(qd) > 252 else qd[0], qd[-1 - 126]
bcab = backcast(end, ra, sh, NET, [ta, tb]).clip(0, 1)
d_bc = (bcab.loc[tb] - bcab.loc[ta]); d_ac = (QR.loc[tb] - QR.loc[ta]).reindex(ra.index)
jj = pd.concat([d_bc.rename('b'), d_ac.rename('a')], axis=1).dropna()
from scipy.stats import spearmanr
out['incr6m_spearman'] = dict(ta=str(ta.date()), tb=str(tb.date()), n=len(jj), sp=float(spearmanr(jj.b, jj.a).correlation))
# 回推誤差 vs 同期報酬
ADJ = pd.read_parquet(SP / 'panel/adj.parquet')
t = qd[0]
bc = backcast(end, ra, sh, NET, [t]).iloc[0].clip(0, 1)
err = (bc - QR.loc[t].reindex(ra.index))
adj = ADJ.ffill()
rr = adj.loc[:end].iloc[-1] / adj.loc[:t].iloc[-1] - 1
jj = pd.concat([err.rename('e'), rr.rename('r')], axis=1).dropna()
out['err_vs_ret_spearman'] = dict(t=str(t.date()), n=len(jj), sp=float(spearmanr(jj.e, jj.r).correlation),
                                  sp_abs=float(spearmanr(jj.e.abs(), jj.r).correlation))

# ---------- 正式面板：2025-10-07 錨點回推 + 之後實值 ----------
ra = QR.loc[A].dropna(); sh = anchor_sh.reindex(ra.index)
days_b = NET.index[NET.index < A]
BC = backcast(A, ra, sh, NET, days_b)
out['backcast_lt_minus05pp_share'] = float(((BC < -0.005).sum().sum()) / BC.notna().sum().sum())
out['backcast_lt0_share'] = float(((BC < 0).sum().sum()) / BC.notna().sum().sum())
BC = BC.clip(0, 1)
FO = pd.concat([BC, QR.reindex(columns=ra.index.union(QR.columns))[QR.index >= A]]).sort_index()
FO.to_parquet(HERE / 'my_fo.parquet')
out['fo_first'] = str(FO.index[0].date()); out['fo_days'] = int(len(FO))

# ---------- 融券 ----------
bm = rd_bb('margin'); bm['date'] = pd.to_datetime(bm['Date'])
bm = bm[bm['Ticker'].isin(PC)]
SH = bm.pivot_table(index='date', columns='Ticker', values='Short_Balance', aggfunc='last').astype(float)
SH.to_parquet(HERE / 'my_short.parquet')
mh = pd.read_csv(X / 'margin_hist.csv.gz', dtype={'code': str}, parse_dates=['date'])
MH = mh.pivot_table(index='date', columns='code', values='short_bal', aggfunc='last')
ov = SH.index.intersection(MH.index)
j = pd.concat([SH.loc[ov].stack().rename('a'), MH.loc[ov].reindex(columns=SH.columns).stack().rename('b')], axis=1).dropna()
out['short_vs_margin_hist'] = dict(n=len(j), match=float((j.a == j.b).mean()), first=str(SH.index[0].date()))
# 融資也比一次
MB = bm.pivot_table(index='date', columns='Ticker', values='Margin_Balance', aggfunc='last').astype(float)
MM = mh.pivot_table(index='date', columns='code', values='margin_bal', aggfunc='last')
j = pd.concat([MB.loc[ov].stack().rename('a'), MM.loc[ov].reindex(columns=MB.columns).stack().rename('b')], axis=1).dropna()
out['margin_vs_margin_hist'] = float((j.a == j.b).mean())

# ---------- TWT93U ----------
sb = rd_bb('margin/sbl'); sb['date'] = pd.to_datetime(sb['Date'])
out['sbl_days'] = int(sb['date'].nunique()); out['sbl_range'] = [str(sb['date'].min().date()), str(sb['date'].max().date())]
idn = sb['SBL_Prev_Balance'] + sb['SBL_Daily_Sell'] - sb['SBL_Daily_Return'] + sb['SBL_Daily_Adjust'] == sb['SBL_Balance']
out['sbl_identity'] = float(idn.mean())
sbp = sb[sb['Ticker'].isin(PC)]
out['sbl_codes_in_panel'] = int(sbp['Ticker'].nunique()); out['sbl_codes_all'] = int(sb['Ticker'].nunique())
out['sbl_codes_4digit'] = int(sb['Ticker'].str.fullmatch(r'\d{4}').sum() and sb.loc[sb['Ticker'].str.fullmatch(r'\d{4}'), 'Ticker'].nunique())
SB = sbp.pivot_table(index='date', columns='Ticker', values='SBL_Balance', aggfunc='last').astype(float)
SB.to_parquet(HERE / 'my_sbl.parquet')
NET.to_parquet(HERE / 'my_net.parquet')
(HERE / 'my_data_audit.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))
