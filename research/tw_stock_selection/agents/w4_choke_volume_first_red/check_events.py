#!/usr/bin/env python3
"""事後診斷（不影響判定）：條件命中率、抽樣手算核對、低點是否早於高點、嚴格「第一根」紅 K、固定持有期（不含出場規則）的毛異常。"""
import sys, math, json
from pathlib import Path
HERE = Path('/home/user/-/research/tw_stock_selection/agents/w4_choke_volume_first_red')
sys.path.insert(0, str(HERE))
import run as R, numpy as np, pandas as pd

out = {}
U = R.UNIV.fillna(False)
nu = int(U.sum().sum())
ch35 = R.choke(0.35)
out['univ_stock_days'] = nu
out['cond_share_in_univ'] = {k: round(float((U & c.fillna(False)).sum().sum()) / nu, 3) for k, c in
                             [('C1_prior_run', R.C1), ('C2_choke035', ch35), ('C2_choke025', R.choke(0.25)), ('C3_no_break', R.C3), ('C4_red_k', R.C4), ('NH', R.CNH)]}
# 0.35×max60 是否幾乎從不綁定（0.5×VMA20 已經更嚴）
VOL, MX, V20 = R.VOL, R.MX60, R.VMA20
a = ((VOL <= 0.5 * V20) & VOL.notna()); b = a & (VOL <= 0.35 * MX)
out['choke_days_vma20_only'] = int(a.sum().sum()); out['choke_days_with_035max60'] = int(b.sum().sum())
out['median_max60_over_vma20'] = round(float(np.nanmedian((MX / V20).where(U).to_numpy())), 2)
print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)

df = pd.read_csv(R.L.P.parent / 'w4_choke_CHOKE35_events.csv.gz', parse_dates=['sig', 'entry'], dtype={'code': str})
WIN = [('select', '2005-01-01', '2019-04-30'), ('valid', '2019-05-01', '2026-10-31')]
def st(g, col='ab'):
    gm = g.groupby(g['entry'].dt.to_period('M'))[col].mean()
    return {'n': int(len(g)), 'mean_pct': round(float(g[col].mean() * 100), 3), 't_month': R.tstat(gm)}

# 低點早於高點？
ADJ = R.ADJ.to_numpy()
lowfirst = []
for _, r in df.iterrows():
    j = R.cols.index(r['code']); t = R.IDX.get_loc(r['sig'])
    a0 = max(t - 120, 0); w60 = ADJ[t - 60:t, j]; w120 = ADJ[a0:t, j]
    ih = t - 60 + int(np.nanargmax(w60)); il = a0 + int(np.nanargmin(w120))
    lowfirst.append(il < ih)
df['lowfirst'] = lowfirst
out['lowfirst_share'] = round(float(df['lowfirst'].mean()), 3)
for nm, a_, b_ in WIN:
    s = df[(df.sig >= a_) & (df.sig <= b_)]
    out[f'lowfirst_{nm}'] = {'low_before_high': st(s[s.lowfirst]), 'low_after_high': st(s[~s.lowfirst])}

# 嚴格「第一根」：最近一個窒息日之後、t 之前沒有其他 C4 紅 K
pos = pd.DataFrame(np.arange(R.T)[:, None].repeat(R.N, 1), index=R.IDX, columns=R.cols).astype(float)
chd = ((VOL <= 0.35 * MX) & (VOL <= 0.5 * V20) & VOL.notna())
last_choke = pos.where(chd).ffill().shift(1)              # t 之前最近窒息日
last_c4 = pos.where(R.C4.fillna(False)).ffill().shift(1)  # t 之前最近 C4 紅 K
first = (last_c4.isna() | (last_c4 < last_choke)).to_numpy()
df['first'] = [bool(first[R.IDX.get_loc(r.sig), R.cols.index(r.code)]) for r in df.itertuples()]
out['strict_first_share'] = round(float(df['first'].mean()), 3)
for nm, a_, b_ in WIN:
    s = df[(df.sig >= a_) & (df.sig <= b_)]
    out[f'strict_first_{nm}'] = {'first': st(s[s['first']]), 'not_first': st(s[~s['first']])}

# 固定持有期（不含出場規則）毛異常：t+1 開盤 → t+1+h 開盤，對同池等權
AO = R.AOn; PX = R.PX
for h in (5, 10, 20, 40):
    rows = []
    for r in df.itertuples():
        j = R.cols.index(r.code); e = R.IDX.get_loc(r.entry); x = e + h
        if x >= R.T: continue
        raw = PX[x, j] / AO[e, j] - 1; ew = math.expm1(R.LEW[x] - R.LEW[e])
        rows.append((r.entry, r.sig, raw - ew, (1 + raw) * (1 - R.SELL) / (1 + R.BUY) - 1 - ew))
    f = pd.DataFrame(rows, columns=['entry', 'sig', 'abg', 'abn'])
    for nm, a_, b_ in WIN:
        s = f[(f.sig >= a_) & (f.sig <= b_)]
        out[f'fixed_h{h}_{nm}'] = {'gross': st(s, 'abg'), 'net': st(s, 'abn')}

# 抽樣手算核對
row = df.sample(1, random_state=11).iloc[0]
c = row['code']; j = R.cols.index(c); t = R.IDX.get_loc(row['sig'])
a = R.ADJ[c]; v = R.VOL[c]
HH = a.iloc[t - 60:t].max(); LL = a.iloc[max(t - 120, 0):t].min()
chk = {'code': c, 'sig': str(row['sig'].date()), 'HH/LL': round(HH / LL, 3), 'min10': round(a.iloc[t - 10:t].min(), 4), 'LL+0.382': round(LL + 0.382 * (HH - LL), 4),
       'open_raw': float(R.OP[c].iloc[t]), 'close_raw': float(R.CR[c].iloc[t]), 'ret': round(a.iloc[t] / a.iloc[t - 1] - 1, 4),
       'vol_t': float(v.iloc[t]), 'vma5_prev': float(v.iloc[t - 5:t].mean()), 'v60_value': float(R.V60[c].iloc[t])}
cd = []
for d in range(t - 10, t):
    mx = v.iloc[d - 60:d].max(); m20 = v.iloc[d - 19:d + 1].mean()
    if v.iloc[d] <= 0.35 * mx and v.iloc[d] <= 0.5 * m20: cd.append((str(R.IDX[d].date()), float(v.iloc[d]), float(mx), float(m20)))
chk['choke_days'] = cd
e = t + 1; x = e + int(row['hold'])
chk['entry_open_adj'] = float(AO[e, j]); chk['stop'] = float(AO[t, j])
chk['path'] = [(str(R.IDX[s].date()), round(float(R.ADJn[s, j]), 4), round(float(R.MA10n[s, j]), 4)) for s in range(e, x)]
chk['exit'] = (str(R.IDX[x].date()), float(PX[x, j])); chk['raw_manual'] = round(float(PX[x, j] / AO[e, j] - 1), 5); chk['raw_csv'] = round(float(row['raw']), 5)
chk['why'] = row['why']
out['spot_check'] = chk
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
f = HERE / 'results.json'
d = json.loads(f.read_text(encoding='utf-8')); d['diagnostics_posthoc'] = out
f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
