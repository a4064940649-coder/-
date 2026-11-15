#!/usr/bin/env python3
"""S12 第 0 步：市值資料檢查（PREREG 前做，不算任何報酬）→ data_check.json。
檢查：股數檔涵蓋（股池內有 PIT 市值的比例、逐年）、落後 63 日 vs 當日股數的分層差異、市值分層 vs 60 日均成交值分層的重疊、
每層股票數、每層每月 L_MA 檔數與 F1 名單檔數（只數檔數）、下市股涵蓋、現行 0050 成分（2026-10）在最後一個訊號日的層別（只當合理性檢查）。
python3 -I -B s0_datacheck.py"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s12lib
from s12lib import s4lib, s2lib, stlib

H = Path(__file__).resolve().parent
t0 = time.time()
P = s12lib.load()
U = stlib.universe(P, 'twse') & ~s4lib.innov_panel(P)[0]
RT = s2lib.rev_tables(P)
SH = s12lib.load_shares(P)
MC, src = s12lib.mcap_panel(P, SH, s12lib.LAG)
MC0, src0 = s12lib.mcap_panel(P, SH, 0)
del SH
ADV = P.V60
S = s12lib.make_signal_tier(P, U, RT, None, None, 0)
rows = []
for x in S['info']:
    k = x['k']; u = U[k]
    t_mc = s12lib.tiers_at(u, MC[k]); t_mc0 = s12lib.tiers_at(u, MC0[k]); t_adv = s12lib.tiers_at(u, ADV[k])
    s = RT['S'].loc[pd.Period(x['month'], 'M')].to_numpy()
    r = dict(month=x['month'], sig_date=x['sig_date'], n_pool=int(u.sum()),
             cov_lag=float((src[k][u] == 0).mean()), cov_any=float((src[k][u] <= 1).mean()), n_nomc=int((src[k][u] == 2).sum()),
             n_L=int((t_mc == 0).sum()), n_M=int((t_mc == 1).sum()), n_S=int((t_mc == 2).sum()),
             mc_cut100_e8=float(np.sort(np.nan_to_num(MC[k][u], nan=-1))[::-1][99] / 1e8), mc_cut300_e8=float(np.sort(np.nan_to_num(MC[k][u], nan=-1))[::-1][299] / 1e8))
    for nm, tt in (('lag0', t_mc0), ('adv', t_adv)):
        r[f'L_overlap_{nm}'] = int(((t_mc == 0) & (tt == 0)).sum())
        r[f'M_overlap_{nm}'] = int(((t_mc == 1) & (tt == 1)).sum())
        r[f'same_tier_{nm}'] = float((t_mc[u] == tt[u]).mean())
    for ti, nm in enumerate(('L', 'M', 'S')):
        ok = u & np.isfinite(s) & P.F_MA60[k] & (t_mc == ti)
        o = np.flatnonzero(ok)[np.argsort(-s[ok], kind='stable')]
        r[f'n_ma_{nm}'] = int(ok.sum())
        r[f'n_f1_{nm}'] = int((P.RET20[k][o[:10]] > 0.10).sum())
        r[f'n_surge_{nm}'] = int((u & np.isfinite(s) & (t_mc == ti)).sum())
    r['n_f1_cur'] = int(x['n_tgt'])
    rows.append(r)
df = pd.DataFrame(rows)
df['year'] = df['month'].str[:4]
# 下市股涵蓋：曾在股池、面板最後一天已不在官方日 K 的股票
ever = U.any(0); gone = ever & (P.MK[-1] == 0)
gone_has = [str(c) for c, j in zip(P.COLS[gone], np.flatnonzero(gone)) if np.isfinite(MC[:, j][U[:, j]]).any()]
# 現行 0050 成分（2026-10，非 PIT，只當合理性檢查）
tw50 = json.load(open(s12lib.AG / 'o_largecap_leg' / 'tw50_members.json'))['current_2026-10-09']
kl = S['info'][-1]['k']; tl = s12lib.tiers_at(U[kl], MC[kl])
tw50_tier = {c: (int(tl[P.ci[c]]) if c in P.ci else None) for c in tw50}
agg = {c: (float(df[c].mean()), float(df[c].min()), float(df[c].max())) for c in df.columns if c not in ('month', 'sig_date', 'year')}
byyear = df.groupby('year')[['n_pool', 'cov_lag', 'cov_any', 'n_L', 'n_M', 'n_S', 'n_ma_L', 'n_ma_M', 'n_ma_S', 'n_f1_L', 'n_f1_M', 'n_f1_S', 'n_f1_cur',
                             'same_tier_adv', 'L_overlap_adv', 'L_overlap_lag0', 'mc_cut100_e8', 'mc_cut300_e8']].mean().round(3)
out = dict(source=dict(shares=str(s12lib.SHARES.relative_to(s12lib.AG)), note='o_0050_tilt_pit（o8）s1_shares.py 重建的 PIT 股數（2013-01 起，季末錨點＝普通股股本 ÷ 面額、MI_QFIIS、TDCC、weihautin 快照；兩錨點間除權事件跳階、無事件對數內插），'
                   '加 o9 複製員發現的 4763、2317 修正（o_lc_sector_rotation_pit lclib.FIX4763／FIX2317）；2006～2012 回推段本題用不到。'),
           pit_status='閘門（o8 gate.json）：2015-05-22、2017-07-06、2021-07-02、2021-09-24 四個獨立真值點，前 100 名股數誤差中位數 ≤ 0.05%、市值名次 Spearman ≥ 0.998；'
                      '但季內內插用到下一季股本（o8 統計審查指出的前視）。本題把股數落後 63 個交易日，再乘上之後的含息漲跌（G 比值），當作 PIT 市值代理。',
           n_signal_months=len(df), first=df['sig_date'].iloc[0], last=df['sig_date'].iloc[-1],
           summary_mean_min_max=agg, by_year=byyear.reset_index().to_dict('records'),
           delisted_in_pool=int(gone.sum()), delisted_with_mc=len(gone_has),
           tw50_2026_tier_at_last_signal=dict(n_L=sum(1 for v in tw50_tier.values() if v == 0), n_M=sum(1 for v in tw50_tier.values() if v == 1),
                                              n_S=sum(1 for v in tw50_tier.values() if v == 2), n_out=sum(1 for v in tw50_tier.values() if v in (None, -1)),
                                              out=[c for c, v in tw50_tier.items() if v in (None, -1)]),
           monthly=df.drop(columns='year').round(4).to_dict('records'),
           runtime_s=round(time.time() - t0, 1), maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
(H / 'data_check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != 'monthly'}, ensure_ascii=False, indent=0)[:6000])
