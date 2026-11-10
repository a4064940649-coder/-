#!/usr/bin/env python3
"""反方審查（資料與統計）o19：o_disposal_list_pit_audit_pkl18。只讀研究員檔案，不寫任何研究員檔案；輸出只印到 stdout。
python3 -I -B review_stat_o19.py
檢查：
 R1 TWSE 資料庫是否「下市即刪」（存活者機制 H_inc 的一種）：各年處置列中「後來停止交易」股票的比例 vs 當年面板母體中後來停止交易的比例
 R2 2019 在逐年殘差的經驗位置（不靠主區間的常態/NB 假設）
 R3 s7b「強觸發未對應」對 H_inc 的檢定力：若 2019 隨機缺 m 比例，16/25 對應的似然
 R4 主策略紙上名單 2019 年持股與「面板近似觸發但不在清單」事件的重疊（H_inc 下可能被高估的槽數上界）
 R5 盤後公告公開證據（s13）逐筆內容核對
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from math import comb
HERE = Path(__file__).resolve().parent; AG = HERE.parent
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
out = {}
# ---------------- R1
cr = pd.read_parquet(P / 'close_raw.parquet')
keep = cr.notna().sum(axis=1) >= 300; cr = cr[keep]
cols = [c for c in cr.columns if len(c) == 4 and not c.startswith('0')]; cr = cr[cols]
last = cr.apply(lambda s: s.last_valid_index())
first = cr.apply(lambda s: s.first_valid_index())
END = cr.index.max()
dead = last < (END - pd.Timedelta(days=120))
U = pd.read_csv(HERE / 'data/universe_announce.csv', dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
U['year'] = U.start.dt.year
r1 = {}
for y in range(2006, 2026):
    g = U[U.year == y]
    g = g[g.code.isin(cols)]
    yd = cr.loc[str(y)]
    act = yd.columns[yd.notna().any()]
    pop_dead = float(dead[act].mean())
    rows_dead = float(dead[g.code].mean()) if len(g) else np.nan
    r1[y] = dict(n_rows=int(len(g)), share_rows_later_dead=round(rows_dead, 3), n_rows_later_dead=int(dead[g.code].sum()),
                 pop_active=int(len(act)), pop_share_later_dead=round(pop_dead, 3))
out['R1_delisted_in_list'] = r1
out['R1_panel_dead_by_lastyear'] = {int(k): int(v) for k, v in last[dead].dt.year.value_counts().sort_index().items()}
# ---------------- R2
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
Y = R['RESULTS_MODEL']['yearly']
zs = {int(y): v['z'] for y, v in Y.items() if int(y) != 2026}
srt = sorted(zs.items(), key=lambda t: t[1])
out['R2_rank_2019_lowest_first'] = [k for k, _ in srt].index(2019) + 1
out['R2_n_years'] = len(zs)
out['R2_sorted'] = srt[:6]
fit = {y: z for y, z in zs.items() if Y[str(y)]['in_fit']}
oos = {y: z for y, z in zs.items() if not Y[str(y)]['in_fit']}
out['R2_fit_years_outside_PI'] = sum(Y[str(y)]['below_PI'] or Y[str(y)]['above_PI'] for y in fit)
out['R2_oos_years_outside_PI'] = {y: (Y[str(y)]['below_PI'], Y[str(y)]['above_PI']) for y in oos}
# P(>=4 of 17 outside | 5%)
p = 0.05; n = len(fit); k = out['R2_fit_years_outside_PI']
out['R2_binom_p_ge_k_of_fit'] = round(sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)), 4)
# empirical SD of fit-year z; 2019 z standardized by empirical SD
sd_emp = float(np.std(list(fit.values()), ddof=1))
out['R2_fit_z_empirical_sd'] = round(sd_emp, 3)
out['R2_2019_z_over_empirical_sd'] = round(zs[2019] / sd_emp, 3)
# ---------------- R3
S7 = json.loads((HERE / 's7b_strong_unmatched.json').read_text(encoding='utf-8'))['by_year']
base_years = [str(y) for y in (2017, 2018, 2020, 2021, 2022, 2023, 2024, 2025)]
ns = sum(S7[y]['n_strong'] for y in base_years); nu = sum(S7[y]['n_unmatched'] for y in base_years)
q = 1 - nu / ns  # baseline matched rate
n19 = S7['2019']['n_strong']; h19 = n19 - S7['2019']['n_unmatched']
def binom_cdf(k, n, pp): return sum(comb(n, i) * pp ** i * (1 - pp) ** (n - i) for i in range(0, k + 1))
r3 = dict(baseline_matched=round(q, 3), n2019=n19, matched2019=h19)
for m in (0.0, 0.2, 0.3, 0.4, 0.5):
    r3[f'P(matched<={h19} | missing={m})'] = round(binom_cdf(h19, n19, q * (1 - m)), 4)
# m implied by observed count: E=62.9 -> missing share 1-31/62.9
r3['m_implied_by_model'] = round(1 - 31 / 62.88, 3)
out['R3_s7b_power'] = r3
# ---------------- R5
T = json.loads((HERE / 's13_timing.json').read_text(encoding='utf-8'))
out['R5_timing_rows'] = [dict(code=r['code'], post=r['post_time'], announce=r['announce'], start=r['start'], title=r['title'][:30], snip=r['snip'][:80]) for r in T['rows']]
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
# ---------------- R4（重；最後跑；加 --skip-r4 略過）
if '--skip-r4' not in sys.argv:
  exec(compile(r'''
sys.path.insert(0, str(AG / 'o_delist_rev_backfill_allpit')); import o18lib as O; sys.path.pop(0)
b = O.build(('opt', '2006-01-01', 'UB', True, 'B1'))
IDX = O.IDX
hold = []
for e, pk, el, d in b['plan']:
    if pd.Timestamp('2019-01-01') <= d <= pd.Timestamp('2019-12-31'):
        hold.append((d, e, [O.COLS[j] for j in pk]))
PC = pd.read_csv(HERE / 'data/proxy_unmatched_candidates.csv', dtype={'code': str}, parse_dates=['trig'])
PC = PC[PC.year == 2019]
UM = json.loads((HERE / 's7b_strong_unmatched.json').read_text(encoding='utf-8'))['unmatched_2019']
def ix(t): return IDX.searchsorted(pd.Timestamp(t))
hits_proxy, hits_strong = [], []
for d, e, codes in hold:
    ie = ix(e) if not isinstance(e, (int, np.integer)) else int(e)
    for c in codes:
        for t in PC[PC.code == c].trig:
            it = ix(t)
            # 假想處置期：觸發日下一交易日起 10 個交易日；成交日 e 落在其中，且觸發日 <= 訊號日
            if it + 1 <= ie <= it + 10:
                hits_proxy.append(dict(signal=str(pd.Timestamp(d).date()), code=c, trig=str(t.date())))
        for u in UM:
            if u['code'] != c: continue
            it = ix(u['trig'])
            if it + 1 <= ie <= it + 12:
                hits_strong.append(dict(signal=str(pd.Timestamp(d).date()), code=c, trig=u['trig']))
nslots = sum(len(c) for _, _, c in hold)
print(json.dumps(dict(R4=dict(n_signal_2019=len(hold), n_slots_2019=nslots, entry_type=str(type(hold[0][1])) if hold else None,
                              proxy_candidates_2019=int(len(PC)), slots_hit_proxy=hits_proxy, slots_hit_strong_unmatched=hits_strong,
                              codes_2019_holdings=sorted({c for _, _, cs in hold for c in cs}))), ensure_ascii=False, indent=1))
''', 'R4', 'exec'))

# ---------------- R6（追加）面板 2019-05 斷點：之後只剩存活者。量化對 K_reg／強觸發與 Y 的影響
if __name__ == '__main__':
    adj = pd.read_parquet(P / 'adj.parquet'); adj = adj[keep][cols]; IX = adj.index
    TX = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].reindex(IX).ffill()
    r6 = adj / adj.shift(6) - 1; m6 = TX / TX.shift(6) - 1
    th = np.where(IX >= pd.Timestamp('2015-06-01'), 0.32, 0.25)[:, None]
    up = (r6.to_numpy() > th) & (r6.sub(m6, axis=0).to_numpy() > 0.2)
    F = pd.DataFrame(up, index=IX, columns=cols).astype(int)
    trig = F.rolling(3).sum() == 3
    prev = trig.astype(int).shift(1).rolling(15, min_periods=1).sum().fillna(0) > 0
    new = trig & ~prev
    drop = set(last[(last >= '2019-05-01') & (last < '2019-06-01')].index)
    E = new.stack(); E = E[E]
    E = E.reset_index(); E.columns = ['date', 'code', 'v']
    E['drop'] = E.code.isin(drop); E['y'] = E.date.dt.year
    g = E[(E.y >= 2015) & (E.y <= 2019)].groupby('y').agg(n=('v', 'size'), n_from_dropped=('drop', 'sum'))
    print('R6 強觸發中來自 2019-05 斷點股的件數'); print(g)
    U2 = U[(U.year >= 2015) & (U.year <= 2019)].copy(); U2['drop'] = U2.code.isin(drop)
    print('R6 處置列中來自斷點股'); print(U2.groupby('year').drop.agg(['size', 'sum']))
    U19 = U[U.year == 2019].copy(); U19['last'] = U19.code.map(last)
    print('R6 2019 處置起日晚於該股面板最後一日的列：', int((U19.start > U19['last']).sum()))
    MF = pd.read_csv(HERE / 'data/monthly_features.csv')
    fitm = MF[(MF.month >= '2006-01') & (MF.month <= '2023-12') & ~MF.month.str.startswith('2019')]
    print('R6 var/mean 擬合樣本 Y_start：', round(fitm.Y_start.var() / fitm.Y_start.mean(), 2))
    print('R6 2019 H1/H2 K_reg：', MF[MF.month.str.startswith('2019')].assign(h=lambda d: d.month.str[5:7].astype(int) > 6).groupby('h')[['K_reg', 'Y_start']].sum().to_dict())
