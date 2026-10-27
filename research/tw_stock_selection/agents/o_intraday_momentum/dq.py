#!/usr/bin/env python3
"""研究計畫 (1) 資料品質：盤中／隔夜拆分的分布、極端值、open 缺值（逐年），以及 src 3（13:20 近似收盤）的拆分誤差。
驗證期年份只報分布形狀（標準差、分位數、缺值與極端值筆數），不報平均（平均屬於描述 (2)，凍結後才補）。
src 3 誤差：tw_stocker 5 分 K 日彙總（src 3 的原始來源）在 2025-09-24～2026-04-01 與官方 src 4 同股同日對照。
輸出 dq.json。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import olib as L
K = L.K; IDX, COLS, N = L.IDX, L.COLS, L.N

ID, ON, info = L.daily_split()
F = K.Fn; S = K.SRC.reindex(index=IDX, columns=COLS).to_numpy()
C, O = K.Cn, K.On
out = {'split_info': info, 'by_year': {}}
yrs = IDX.year
for y in sorted(set(yrs)):
    m = yrs == y
    tr = np.isfinite(C[m])
    o = {'stock_days_traded': int(tr.sum()),
         'open_missing_when_close': int((tr & ~np.isfinite(O[m])).sum()),
         'open_eq_close_value0': int((tr & np.isfinite(O[m]) & (O[m] == C[m]) & (np.nan_to_num(K.V.to_numpy()[m]) == 0)).sum()),
         'flag345': int(np.isin(F[m], [3, 4, 5]).sum()), 'flag1': int((F[m] == 1).sum()), 'flag2': int((F[m] == 2).sum())}
    for nm, A in (('ID', ID[m]), ('ON', ON[m])):
        a = A[np.isfinite(A)]
        o[nm] = {'n': int(len(a)), 'sd_pct': float(a.std() * 100), 'p01_pct': float(np.percentile(a, 1) * 100), 'p99_pct': float(np.percentile(a, 99) * 100),
                 'share_abs_gt_5pct': float((np.abs(a) > 0.05).mean()), 'share_zero': float((a == 0).mean())}
        if y <= 2019 and pd.Timestamp(f'{y}-12-31') <= pd.Timestamp('2019-04-30') or y < 2019:
            o[nm]['mean_bp'] = float(a.mean() * 1e4)
    o['src'] = {str(int(k)): int(v) for k, v in pd.Series(S[m][np.isfinite(S[m])]).value_counts().items()}
    out['by_year'][int(y)] = o

# 7% 與 10% 漲跌幅：極端值剔除的份額
out['limit_regime_note'] = '7%：2015-05-29 以前；10%：2015-06-01 起。極端值門檻隨之改變。'

# ---------------- src 3 誤差 ----------------
TS = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/tw_stocker_daily')
AGG = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/vf_agg')
a0, a1 = pd.Timestamp('2025-09-24'), pd.Timestamp('2026-04-01')
s0, s1 = pd.Timestamp('2025-03-01'), pd.Timestamp('2025-09-23')
rows = []; same = []
for f in sorted(TS.glob('*.csv')):
    c = f.stem
    if c not in K.CI: continue
    d = pd.read_csv(f, parse_dates=['date']).set_index('date')
    j = K.CI[c]
    pc = K.C[c]; po = K.O[c]
    # (a) 官方期間：近似 vs 官方
    dd = d[(d.index >= a0) & (d.index <= a1)]
    dd = dd[dd.index.isin(IDX)]
    if len(dd):
        offc = pc.reindex(dd.index); offo = po.reindex(dd.index)
        rows.append(pd.DataFrame({'code': c, 'date': dd.index, 'c_apx': dd['close'].to_numpy(), 'o_apx': dd['open'].to_numpy(),
                                  'c_off': offc.to_numpy(), 'o_off': offo.to_numpy()}))
    # (b) src 3 期間：確認面板就是這個來源
    ds = d[(d.index >= s0) & (d.index <= s1)]
    ds = ds[ds.index.isin(IDX)]
    if len(ds):
        same.append(np.nanmean(np.isclose(ds['close'].to_numpy(), pc.reindex(ds.index).to_numpy(), rtol=1e-6)))
E = pd.concat(rows)
E = E[(E.c_apx > 0) & (E.c_off > 0) & (E.o_apx > 0) & (E.o_off > 0)]
E['e_close'] = np.log(E.c_apx / E.c_off)
E['e_open'] = np.log(E.o_apx / E.o_off)
E['e_id'] = np.log(E.c_apx / E.o_apx) - np.log(E.c_off / E.o_off)
# 隔夜誤差＝ e_open(t) − e_close(t−1)；用同股前一日
E = E.sort_values(['code', 'date'])
E['e_on'] = E['e_open'] - E.groupby('code')['e_close'].shift(1)
def dist(x):
    x = x[np.isfinite(x)]
    return {'n': int(len(x)), 'mean_bp': float(x.mean() * 1e4), 'sd_bp': float(x.std() * 1e4), 'mad_bp': float(np.median(np.abs(x)) * 1e4),
            'p05_bp': float(np.percentile(x, 5) * 1e4), 'p95_bp': float(np.percentile(x, 95) * 1e4), 'share_exact0': float((np.abs(x) < 1e-9).mean())}
src3 = {'compare_window': '2025-09-24～2026-04-01（tw_stocker 5 分 K 日彙總 vs 官方 src 4）', 'n_stocks': int(E.code.nunique()),
        'close_err': dist(E.e_close.to_numpy()), 'open_err': dist(E.e_open.to_numpy()), 'intraday_err': dist(E.e_id.to_numpy()), 'overnight_err': dist(E.e_on.to_numpy()),
        'panel_src3_equals_tw_stocker_close_share': float(np.nanmean(same)) if same else None}
# 252 日累積盤中的誤差：每股 252 日 e_close−e_open 的總和，估算 = 252 × 平均 ± sqrt(252) × sd（假設獨立）
m_ = np.nanmean(E.e_id); s_ = np.nanstd(E.e_id)
# 每股平均誤差（系統性偏差）的橫斷面分布
pm = E.groupby('code')['e_id'].mean()
src3['per_stock_mean_intraday_err_bp'] = {'p10': float(pm.quantile(.1) * 1e4), 'p50': float(pm.median() * 1e4), 'p90': float(pm.quantile(.9) * 1e4)}
src3['cum252_intraday_err_approx'] = {'bias_logpct': float(252 * m_ * 100), 'noise_sd_logpct_iid': float(np.sqrt(252) * s_ * 100),
                                      'per_stock_bias_p10_p90_logpct': [float(pm.quantile(.1) * 252 * 100), float(pm.quantile(.9) * 252 * 100)]}
# 對照：挑選期 252 日累積盤中的橫斷面標準差（前 300 池）
sp = L.Split()
S52, V60 = L.V.rolling_signals('gapfix')
cs_sd = []
for d in [x for x in L.REB_ALL if pd.Timestamp('2006-01-01') <= x <= pd.Timestamp('2019-04-30')]:
    t = IDX.get_loc(d)
    el, _ = L.base_pool(t, S52, V60)
    CI, CO, ok, _ = sp.window(t, 252, 0)
    x = CI[el & np.isfinite(CI)]
    if len(x) > 50: cs_sd.append(x.std())
src3['selection_cs_sd_CI252_logpct'] = float(np.median(cs_sd) * 100)
# 2025-01-21 中午價
d0121 = pd.Timestamp('2025-01-21')
if d0121 in IDX:
    t = IDX.get_loc(d0121)
    a = ID[t]; b = ON[t + 1]
    src3['2025-01-21'] = {'ID_sd_pct': float(np.nanstd(a) * 100), 'ID_mean_pct': float(np.nanmean(a) * 100), 'next_ON_mean_pct': float(np.nanmean(b) * 100),
                          'note': '中午價當收盤：當日盤中與次日隔夜互相抵銷，兩日合計不受影響'}
out['src3_error'] = src3
(L.HERE / 'dq.json').write_text(L.jdump(out), encoding='utf-8')
print(json.dumps(L.r4(src3), ensure_ascii=False, indent=1))
for y in (2005, 2010, 2014, 2016, 2019, 2022, 2024, 2025, 2026):
    o = out['by_year'][y]; print(y, o['open_missing_when_close'], o['flag345'], L.r4(o['ID']), L.r4(o['ON']))
print(info)
