#!/usr/bin/env python3
"""第 3 步：建構 MARGN 核心欄位（融資今日餘額）三個版本＋QFIIS 日頻／月頻表＋一致性副閘門。
大型中間檔放 scratchpad/opm2/build/。只做資料層，不算任何報酬。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, glib
OUT = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/opm2/build'); OUT.mkdir(parents=True, exist_ok=True)
L, V, FD, LD = glib.denominator()
days, codes = L.index, L.columns
def wide(df, val, idx=days):
    w = df.pivot_table(index='date', columns='code', values=val, aggfunc='last')
    return w.reindex(index=idx, columns=codes)
yh = glib.load_yh(); bb = glib.load_bb(); mh = glib.load_mh(); jm = glib.load_joeyc('margin-twse')
fb = glib.load_fb('margin_buy')
B = wide(yh, 'mbal'); C = wide(bb, 'Margin_Balance'); E = wide(mh, 'margin_bal'); D = wide(jm, 'margin_lots')
# 原始餘額（V1）：優先序 官方月快照 D > BlackBoxOP C > margin_hist E > TEJ B
V1 = D.combine_first(C).combine_first(E).combine_first(B)
raw_src = pd.DataFrame(np.select([D.notna(), C.notna(), E.notna(), B.notna()], [1, 2, 3, 4], 0), index=days, columns=codes)
A = fb.reindex(columns=codes)  # 融資日增減，index＝FinBrain 自身交易日
A = A.reindex(A.index.intersection(days))
fbdays = A.index
def derive(reanchor, carry):
    """在 FinBrain 期間以向前累加補 V1 缺值；錨點只用資料日 ≤ t。"""
    out = V1.copy()
    lo, hi = fbdays[0], fbdays[-1]
    V1f = V1.loc[:hi]
    for c in codes:
        a = A[c] if c in A.columns else None
        if a is None or a.notna().sum() == 0: continue
        raw = V1f[c]
        f = FD.get(c)
        zero_start = (not pd.isna(f)) and f >= lo
        lvl = np.nan; started = False; nan_run = 0; first_anchor_used = False
        res = {}
        prior = raw.loc[:lo].iloc[:-1].dropna()  # 期初前最近原始餘額
        if len(prior): lvl = prior.iloc[-1]; started = True
        for t in fbdays:
            r = raw.get(t, np.nan)
            d = a.get(t, np.nan)
            if started and not np.isnan(lvl):
                if np.isnan(d):
                    nan_run += 1
                    if not (carry and nan_run <= 2): lvl = np.nan
                else:
                    nan_run = 0; lvl = lvl + d
            elif not started and zero_start and not np.isnan(d):
                lvl = d; started = True  # 新上市股自 0 起算（首筆增減＝首日餘額）
            if not np.isnan(r):
                if reanchor or np.isnan(lvl): lvl = r; started = True
            res[t] = lvl
        s = pd.Series(res)
        fill = out.loc[fbdays, c].isna() & s.notna()
        out.loc[fbdays[fill.values], c] = s[fill].values
    return out
V2 = derive(reanchor=False, carry=False)
V3 = derive(reanchor=True, carry=True)
for nm, w in [('V1', V1), ('V2', V2), ('V3', V3)]: w.to_parquet(OUT / f'margn_{nm}.parquet')
raw_src.to_parquet(OUT / 'margn_rawsrc.parquet')
# ---- 一致性副閘門 ----
cons = {}
# (b) TEJ：Δ餘額 ≤ 買進−賣出
cal = pd.Series(np.arange(len(days)), index=days)  # 交易日序號：只比「前一交易日」
y = yh[yh.date.isin(days)].sort_values(['code', 'date']).copy(); y['dbal'] = y.groupby('code').mbal.diff()
y['consec'] = y.groupby('code').date.transform(lambda s: cal.reindex(s).diff().values == 1)
ok = y[['dbal', 'mbuy', 'msell']].notna().all(axis=1) & y.consec
cons['TEJ_ineq_dbal_le_buy_minus_sell'] = {'n': int(ok.sum()), 'rate': float((y.loc[ok, 'dbal'] <= y.loc[ok, 'mbuy'] - y.loc[ok, 'msell']).mean()),
    'rate_eq': float((y.loc[ok, 'dbal'] == y.loc[ok, 'mbuy'] - y.loc[ok, 'msell']).mean())}
# (a) BlackBoxOP：前日餘額(t)＝今日餘額(t−1)
b = bb[bb.date.isin(days)].sort_values(['code', 'date']).copy(); b['prevbal'] = b.groupby('code').Margin_Balance.shift(1)
b['consec'] = b.groupby('code').date.transform(lambda s: cal.reindex(s).diff().values == 1)
ok = b.prevbal.notna() & b.consec; cons['BB_prev_continuity'] = {'n': int(ok.sum()), 'rate': float((b.loc[ok, 'Margin_Prev_Balance'] == b.loc[ok, 'prevbal']).mean())}
# (c) FinBrain Δ 與 TEJ Δ（同交易日、TEJ 前一列為前一交易日）
Bd = B.loc[fbdays].diff()
Bprev_ok = B.loc[fbdays].shift(1).notna() & B.loc[fbdays].notna()
m = Bprev_ok & A.notna()
cons['FinBrain_vs_TEJ_delta'] = {'n': int(m.values.sum()), 'rate': float((A.values[m.values] == Bd.values[m.values]).mean())}
Nb = wide(yh.assign(net=yh.mbuy - yh.msell), 'net').loc[fbdays]; m2 = Nb.notna() & A.notna()
cons['FinBrain_vs_TEJ_buy_minus_sell'] = {'n': int(m2.values.sum()), 'rate': float((A.values[m2.values] == Nb.values[m2.values]).mean()),
    'note': '欄位語意判定：FinBrain margin_buy＝融資買進−融資賣出（不含現金償還），不是餘額增減；累加無法還原餘額'}
def cmp(X, Y, nm):
    m = X.notna() & Y.notna(); n = int(m.values.sum())
    cons[nm] = {'n': n, 'rate': float((X.values[m.values] == Y.values[m.values]).mean()) if n else None,
                'dates': [str(X.index[m.any(axis=1)].min().date()), str(X.index[m.any(axis=1)].max().date())] if n else None}
cmp(B, C, 'TEJ_vs_BlackBoxOP_level'); cmp(D, B, 'joeyc_vs_TEJ_level'); cmp(D, C, 'joeyc_vs_BlackBoxOP_level'); cmp(D, E, 'joeyc_vs_margin_hist_level'); cmp(C, E, 'BlackBoxOP_vs_margin_hist_level')
for nm, w in [('V1', V1), ('V2', V2), ('V3', V3)]:
    cons[f'range_nonneg_{nm}'] = float((w.values[~np.isnan(w.values)] >= 0).mean())
# 衍生值準確度（描述）：V2/V3 衍生格 vs 當日 TEJ／官方月快照（只在留一比對：衍生格本來就不覆蓋原始格，故改比對「下一個原始餘額日」的累加結果）
# QFIIS
jq = glib.load_joeyc('qfii-twse'); vf = glib.load_vf()
Qd = wide(vf, 'foreign_ratio'); Qm = wide(jq, 'foreign_pct')
Q = Qm.combine_first(Qd); Q.to_parquet(OUT / 'qfiis_daily.parquet')
jq2 = jq.merge(vf[['date', 'code', 'foreign_ratio', 'total_shares', 'foreign_shares']], on=['date', 'code'])
cons['joeyc_vs_voidful_qfii'] = {'n': len(jq2), 'ratio_eq_rate': float((jq2.foreign_pct.round(2) == jq2.foreign_ratio.round(2)).mean()) if len(jq2) else None,
    'issued_eq_rate': float((jq2.issued == jq2.total_shares).mean()) if len(jq2) else None, 'held_eq_rate': float((jq2.foreign_held == jq2.foreign_shares).mean()) if len(jq2) else None}
cons['qfii_range'] = {'pct_in_0_100': float(jq.foreign_pct.between(0, 100).mean()), 'held_le_issued': float((jq.foreign_held <= jq.issued).mean()), 'n': len(jq)}
json.dump(cons, open(Path(__file__).resolve().parent / 'data' / 'consistency.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(cons, ensure_ascii=False, indent=1))
