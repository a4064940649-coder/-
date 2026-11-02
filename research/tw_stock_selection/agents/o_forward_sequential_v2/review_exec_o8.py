#!/usr/bin/env python3
"""反方審查（成本與可成交，o8）：只讀研究員檔案，不寫 results.json。用法：python3 -I -B review_exec_o8.py
A 周轉；B K5T-lite（make_forward）與 V7 K5T 的成本差＝前瞻決策線量測偏誤；C 零股比例；D 容量（開盤參與率）；
E 處置在成交日；F 開盤跳空分布；G 乾跑窗對帳的有號差。"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import fslib as F

PANEL = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
RAW = HERE / 'raw'
res = json.loads((HERE / 'results.json').read_text())

# ---------- A 周轉（官方日 B10 計畫，2019-05 起）
pv = pd.read_csv(RAW / 'v7/plan_V7_B10.csv', parse_dates=['signal', 'exec'])
pv['set'] = pv.picks.str.split(',').map(set)
pv = pv.sort_values('signal').reset_index(drop=True)
to = [np.nan] + [1 - len(pv.set[i] & pv.set[i - 1]) / 10 for i in range(1, len(pv))]
pv['turn'] = to
post = pv[pv.signal >= '2019-05-01']
print('A 官方日 B10 單邊月周轉 2019-05 起 n=%d 平均 %.3f 中位 %.2f；年化單邊 %.1f 倍' % (len(post), post.turn.mean(), post.turn.median(), post.turn.mean() * 12))
pj = pd.read_csv(RAW / 'v7/plans_j7.csv', usecols=['strat', 'j', 'seed', 'signal', 'picks'], parse_dates=['signal'])
pj = pj[(pj.strat == 'AB_V7_B10') & (pj.seed == 0)].sort_values(['j', 'signal'])
tj = []
for j, g in pj.groupby('j'):
    s = g.picks.str.split(',').map(set).tolist(); d = g.signal.tolist()
    for i in range(1, len(s)):
        if d[i] >= pd.Timestamp('2019-05-01'): tj.append(1 - len(s[i] & s[i - 1]) / 10)
print('A J 線（seed0, 13 j）單邊月周轉平均 %.3f' % np.mean(tj))
T = post.turn.mean()
lite = T * (2 * 0.001425 * 0.28 + 2 * 0.001 + 0.003)
print('A K5T-lite 以 0.1%%/邊估計的月成本 %.2fpp（年 %.1fpp）；只算證交稅 %.2fpp/月' % (lite * 100, lite * 1200, T * 0.003 * 100))

# ---------- B V7 K5T 100萬 FIX 與零衝擊的月成本（J 窗），對照 K5T-lite
X = F.exec_navs(); js = pd.read_parquet(RAW / 'v7/navs_jscan_v7.parquet')
rows = []
for j in range(13):
    k = X[F.cell_col(j, '100萬', 'FIX')]; z = js[f'AB_V7_B10|normal|J{j}'] if f'AB_V7_B10|normal|J{j}' in js else js[f'V7_B10|normal|J{j}']
    for tier in ('30萬', '100萬', '300萬', '1000萬'):
        kk = X[F.cell_col(j, tier, 'FIX')]
        a = kk.dropna(); a = a[a.index >= '2019-05-01']; b = z.reindex(a.index)
        gap = (np.log(b).diff() - np.log(a).diff()).dropna()
        rows.append(dict(j=j, tier=tier, cost_pp_month=float(gap.sum() / len(gap) * 21 * 100)))
B = pd.DataFrame(rows).groupby('tier').cost_pp_month.mean()
print('B V7 K5T 對零衝擊（零衝擊已含手續費與證交稅）的額外衝擊月拖累（pp，13 j 平均，2019-05 起）：', B.round(2).to_dict())
imp_lite = T * 2 * 0.001
print('B K5T-lite 的衝擊部分（只算 0.1%%/邊下限）≈ %.2fpp/月，V7 K5T 100萬 %.2fpp/月：平均水準相近，但 K5T-lite 沒有零股、現金 2%%、參與上限、處置賣出規則' % (imp_lite * 100, B['100萬']))
st = res['PSEUDO_FORWARD']['VAL_2019-05']
print('B 偽前瞻 j 中位數月超額 一般 %.2f 壓力 %.2f（差 %.2fpp/月）' % (st['ex_mean'], st['ex_mean_stress'], st['ex_mean'] - st['ex_mean_stress']))

# ---------- C 零股比例（成交日開盤價 × 1000 > 每檔配置）
O = pd.read_parquet(PANEL / 'open.parquet'); V = pd.read_parquet(PANEL / 'value.parquet')
recs = []
for _, r in post.iterrows():
    if r['exec'] not in O.index: continue
    o = O.loc[r['exec']]
    for c in r['set']:
        p = o.get(c, np.nan)
        if not np.isfinite(p): continue
        i = V.index.get_loc(r['signal']) if r['signal'] in V.index else None
        v60 = V[c].iloc[max(i - 59, 0):i + 1].mean() if (i is not None and c in V) else np.nan
        recs.append(dict(signal=r['signal'], code=c, px=p, lot=p * 1000, v60=v60))
P = pd.DataFrame(recs)
for tier in (3e5, 1e6, 3e6, 1e7):
    per = tier / 10
    print('C 每檔 %s 元：一張 > 每檔配置（必須零股或大幅偏離等權）占 %.1f%%；零股部分≥一半的占 %.1f%%' %
          (f'{per:,.0f}', 100 * (P.lot > per).mean(), 100 * (((per % P.lot) / per) >= 0.5).where(P.lot <= per, True).mean()))
P['yr'] = P.signal.dt.year
print('C 年度：每檔 10 萬元時一張超過配置的比例', P.groupby('yr').apply(lambda g: round(100 * (g.lot > 1e5).mean(), 1)).to_dict())

# ---------- D 容量：每檔下單 ÷（開盤集合競價成交值 ≈ oa_share×V60）
spec = json.loads((RAW / 'v7/model_spec7.json').read_text()); oa = np.median(list(spec['oa_share_by_dec'].values()))
for tier in (3e5, 1e6, 3e6, 1e7):
    part = (tier / 10) / (oa * P.v60)
    print('D %s：開盤參與率 中位 %.1f%%，> 20%%（V7 上限）占 %.1f%%，> 50%% 占 %.1f%%' % (f'{tier:,.0f}', 100 * part.median(), 100 * (part > 0.2).mean(), 100 * (part > 0.5).mean()))

# ---------- E 處置：成交日在處置期間（訊號日未處置）
dsp = pd.read_csv(RAW / 'screener/data/extras/disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
p5 = post[post.signal >= dsp.date.min()]
n = hit = hold_hit = 0
for _, r in p5.iterrows():
    for c in r['set']:
        n += 1
        d = dsp[(dsp.code == c)]
        if ((d.start <= r['exec']) & (d.end >= r['exec'])).any(): hit += 1
print('E 2021-09 起官方日 B10 買進／持有 %d 檔次中，成交日處於處置（需預收款券、分盤）%d 檔次（%.1f%%），每月平均 %.2f 檔' % (n, hit, 100 * hit / max(n, 1), hit / max(len(p5), 1)))
# 持有期間遇到處置（賣出或續抱時需預收券）
hh = 0
for i in range(len(p5) - 1):
    r = p5.iloc[i]; nxt = p5.iloc[i + 1]
    for c in r['set']:
        d = dsp[dsp.code == c]
        if ((d.start <= nxt['exec']) & (d.end >= r['exec'])).any(): hh += 1
print('E 持有期間（成交日～下一成交日）任一天處置的檔次占 %.1f%%' % (100 * hh / max(n, 1)))

# ---------- F 開盤跳空（成交日開盤／訊號日收盤）
C = pd.read_parquet(PANEL / 'close_raw.parquet')
g = []
for _, r in post.iterrows():
    if r['exec'] not in O.index or r['signal'] not in C.index: continue
    for c in r['set']:
        a, b = C.loc[r['signal']].get(c, np.nan), O.loc[r['exec']].get(c, np.nan)
        if np.isfinite(a) and np.isfinite(b): g.append(b / a - 1)
g = np.array(g)
print('F 買進股的開盤跳空：平均 %+.2f%%，中位 %+.2f%%，≥ +3%% 占 %.1f%%，≥ +7%% 占 %.1f%%' % (100 * g.mean(), 100 * np.median(g), 100 * (g >= .03).mean(), 100 * (g >= .07).mean()))

# ---------- G 乾跑窗對帳有號差
R = pd.DataFrame(res['DRYRUN']['window_recon_2026-09']).dropna()
print('G 2026-09 窗 ours−V7 K5T 有號平均 %+.2fpp（n=%d），ours−V7 零衝擊 %+.2fpp' % (R.d_exec.mean(), len(R), R.d_zero.mean()))
ex = R[R.j.isin([1, 3, 4, 6, 7, 8])]
print('G 名單完全相同的 j（1,3,4,6,7,8）：ours−K5T %+.2fpp，ours−零衝擊 %+.2fpp' % (ex.d_exec.mean(), ex.d_zero.mean()))

# ---------- I 真實下修力：OC 表 R2 在 alpha=-0.5 與 0
oc = pd.read_csv(HERE / 'data/oc_final.csv'); r2 = oc[(oc.rule == 'R2') & (oc.h == 36)]
print('I R2 36 個月下修率（D1/D2/D3）：', r2[r2.alpha.isin([-1.0, -0.5, 0.0])].pivot(index='alpha', columns='dgp', values='down').round(3).to_dict('index'))
pf = st['policies']['P100']['normal']; e = st['comparators']['E50']
print('I 2019-05 起 算術月超額 %.2fpp，但年化（幾何）落後 %.2fpp／年＝log %.2fpp／月：決策線的 alpha=0 不等於財富打平' % (st['ex_mean'], pf['ann'] - e['ann'], 100 * math.log((1 + pf['ann'] / 100) / (1 + e['ann'] / 100)) / 12))

# ---------- H 前瞻乾跑：開盤漲停遞補與成交日處置
pl = sorted((HERE / 'forward_dry/paper_lists').glob('j*_2026*.csv')) + sorted((HERE / 'forward_dry/paper_lists').glob('official_*.csv'))
pl = [f for f in pl if 'jall' not in f.name]
L = pd.concat([pd.read_csv(f, dtype={'code': str}) for f in pl])
nl = L.groupby(['line', 'anchor']).ngroups
print('H 乾跑 %d 份名單：開盤漲停被跳過 %d 檔次（%.2f／份），買進股成交日處置 %d 檔次，訊號日已公告處置 %d' %
      (nl, int((L.skipped.fillna('').str.contains('漲停')).sum()), (L.skipped.fillna('').str.contains('漲停')).sum() / nl,
       int(L[L.bought == 1].disp_on_exec.sum()), int(L[L.bought == 1].disp_announced_by_signal.sum())))
