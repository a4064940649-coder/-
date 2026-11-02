#!/usr/bin/env python3
"""步驟 5：正式檢定（PREREG 後只跑一次）。
建構＝v1（fa_build.DEFAULT），全期錨定月 2006-04～2026-09（2019-01～04 不輸出），A／M50／FO／COMBO 四種分組。
檢定量與判定見 results.json PREREG；這裡逐條計算並寫入 RESULTS_OOS。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import fa_common as C
R = C.check_prereg()
import fa_build as B, fa_stats as ST

months = set(pd.period_range(C.PRE[0], C.PRE[1], freq='M')) | set(pd.period_range(C.POST[0], C.POST[1], freq='M'))
A, CR = B.build(months)
A.to_parquet(C.DATA / 'alpha_all_v1.parquet')
CF = B.contrib_frame(CR); CF.to_parquet(C.DATA / 'contrib_all_v1.parquet')
O = {}

def full(sch, leg, hi, lo, per, col='alpha'):
    s, I, bh, bl = ST.interaction(A, sch, leg, hi, lo, col=col, per=per)
    return s, I, bh, bl

# ---------- 主檢定（A、main、一般成本） ----------
s_post, I_post, bh_post, bl_post = full('A', 'main', 2, 0, C.POST)
s_pre, I_pre, bh_pre, bl_pre = full('A', 'main', 2, 0, C.PRE)
O['primary'] = dict(post=s_post, pre=s_pre, DDD=ST.ddd(I_post, I_pre),
                    val=full('A', 'main', 2, 0, C.VAL)[0], hold=full('A', 'main', 2, 0, C.HOLD)[0],
                    gross_post=full('A', 'main', 2, 0, C.POST, 'alpha_gross')[0], stress_post=full('A', 'main', 2, 0, C.POST, 'alpha_stress')[0],
                    mid_post=ST.summ(ST.month_slopes(A, 'A', 'main', 1, per=C.POST)), mid_pre=ST.summ(ST.month_slopes(A, 'A', 'main', 1, per=C.PRE)))
O['alpha_by_j'] = {per: {g: ST.alpha_by_j(A, 'A', 'main', g, per=pp) for g in (0, 1, 2)} for per, pp in (('PRE', C.PRE), ('POST', C.POST), ('VAL', C.VAL), ('HOLD', C.HOLD))}
O['alpha_level_pp'] = {per: {g: float(A[(A.scheme == 'A') & (A.leg == 'main') & (A.g == g) & (pd.PeriodIndex(A.m, freq='M') >= pp[0]) & (pd.PeriodIndex(A.m, freq='M') <= pp[1])].alpha.mean() * 100)
                             for g in (0, 1, 2)} for per, pp in (('PRE', C.PRE), ('POST', C.POST))}
# ---------- 副指標 ----------
sub = {}
s, I_m, _, _ = full('M50', 'main', 1, 0, C.POST); s2, I_m0, _, _ = full('M50', 'main', 1, 0, C.PRE)
sub['M50'] = dict(post=s, pre=s2, DDD=ST.ddd(I_m, I_m0))
fo_per = (C.P('2021-10'), C.POST[1])
s, I_f, _, _ = full('FO', 'main', 2, 0, fo_per)
sub['FO'] = dict(post_2021_10=s, note='外資持股 2021-10 起（o_sbl 回推值），無前期、無 DDD')
sub['MCAP'] = dict(available=False, note='PIT 市值需第 2 題 PIT 股數過閘門；o_0050_tilt_pit 尚未產出 → 不用（事前登記：未過閘門就不用）')
# 同期比較：A 指標在 2021-10 起
sub['A_same_window_as_FO'] = full('A', 'main', 2, 0, fo_per)[0]
O['sub_indicators'] = sub
avail = {k: sub[k]['post']['I']['mean_pp'] if k == 'M50' else sub[k]['post_2021_10']['I']['mean_pp'] for k in ('M50', 'FO')}
O['sub_direction'] = dict(I_post_pp=avail, n_negative=int(sum(v < 0 for v in avail.values())), n_available=2,
                          rule='3 個副指標中至少 2 個 I_後 < 0；市值無法使用 → 需 M50 與 FO 兩個都 < 0')
# ---------- 子樣本 b（後期） ----------
O['subsample_b_post'] = {'M50_member': ST.summ(ST.month_slopes(A, 'M50', 'main', 1, per=C.POST)),
                         'FO_high': ST.summ(ST.month_slopes(A, 'FO', 'main', 2, per=fo_per)),
                         'COMBO_lowA_non50': ST.summ(ST.month_slopes(A, 'COMBO', 'main', 0, per=C.POST)),
                         'COMBO_pre': ST.summ(ST.month_slopes(A, 'COMBO', 'main', 0, per=C.PRE))}
# ---------- 腿分解 ----------
O['legs'] = {leg: dict(post=full('A', leg, 2, 0, C.POST)[0], pre=full('A', leg, 2, 0, C.PRE)[0]) for leg in ('rev', 'px')}
for leg in ('rev', 'px'):
    Ip = full('A', leg, 2, 0, C.POST)[1]; Iq = full('A', leg, 2, 0, C.PRE)[1]
    O['legs'][leg]['DDD'] = ST.ddd(Ip, Iq)
# ---------- 穩健性 ----------
O['loyo_I_post'] = ST.loyo(I_post)
ys, ymax = ST.year_share(I_post); O['year_share_I_post'] = dict(by_year=ys, max_share=ymax)
c, nm = ST.stock_contrib(CF, C.POST); O['stock_I_post'] = ST.stock_bootstrap(c); O['stock_I_post']['months'] = nm
O['stock_I_post']['top5_codes'] = [(B.COLS[int(k)], v) for k, v in O['stock_I_post']['top5']]
cp, _ = ST.stock_contrib(CF, C.PRE); O['stock_I_pre'] = ST.stock_bootstrap(cp)
# 拿掉最大貢獻股（對 I 同號貢獻最大者）後的 I（未扣成本）
top = c.idxmax() if c.sum() > 0 else c.idxmin()
O['stock_I_post']['drop_top_stock_I_pp'] = float((c.sum() - c[top]) * 100); O['stock_I_post']['top_stock'] = B.COLS[int(top)]
# 分產業：高、低組入選股產業占比（後期）
ind = pd.read_csv(C.SCR / 'stock_list.csv', dtype=str).set_index('code')['industry']
CFp = CF[pd.PeriodIndex(CF.m, freq='M') >= C.POST[0]]
pk = CFp[CFp.w > 0]
pk = pk.assign(code=[B.COLS[int(x)] for x in pk.col]); pk['ind'] = pk.code.map(ind).fillna('NA')
O['industry_share_picks_post'] = {int(g): gg.ind.value_counts(normalize=True).head(6).round(3).to_dict() for g, gg in pk.groupby('g')}
# 產業貢獻 to I（後期）
cc = c.rename(index=lambda k: B.COLS[int(k)]); ci = cc.groupby(cc.index.map(lambda x: ind.get(x, 'NA'))).sum()
O['industry_contrib_I_post_pp'] = (ci.sort_values() * 100).head(6).round(4).to_dict()

# ---------- 否證與判定（事前登記） ----------
P = O['primary']; bh, bl, I = P['post']['b_hi'], P['post']['b_lo'], P['post']['I']
falsA = bool(bl['t'] <= -2 and bh['t'] <= -2 and P['post']['t_diff'] > -1)
falsB = bool(P['pre']['I']['t'] <= -2)
cond = {
    'I_post_t<=-2.0': bool(I['t'] <= -2.0),
    'DDD_t<=-1.65': bool(P['DDD']['t'] <= -1.65),
    '|I_pre_t|<1.5': bool(abs(P['pre']['I']['t']) < 1.5),
    'b_lo_post_CI含0': bool(bl['ci95_pp'][0] <= 0 <= bl['ci95_pp'][1]),
    '|b_lo|<0.5|b_hi|': bool(abs(bl['mean_pp']) < 0.5 * abs(bh['mean_pp'])),
    '副指標≥2同向(市值缺→M50與FO皆<0)': bool(O['sub_direction']['n_negative'] >= 2),
    'LOYO符號不變': bool(all((v['mean_pp'] < 0) == (I['mean_pp'] < 0) for v in O['loyo_I_post'].values())),
    '單一年份占比≤40%': bool(ymax <= 0.40),
    '單一股票占比≤30%': bool(O['stock_I_post']['max_share'] <= 0.30),
}
O['conditions'] = cond; O['falsification'] = {'A_全面加速定價': falsA, 'B_前期已有I<0': falsB}
if falsA or falsB: mech = '不支持'
elif all(cond.values()): mech = '支持'
elif I['t'] <= -1.65 and P['DDD']['t'] <= -1.65: mech = '部分支持'
else: mech = '不支持'
O['mechanism'] = mech
C.save_key('RESULTS_OOS', O)
print(json.dumps(C.conv({'primary_post': {k: (round(P['post'][k]['mean_pp'], 4), round(P['post'][k]['t'], 2)) for k in ('b_hi', 'b_lo', 'I')},
                         'primary_pre': {k: (round(P['pre'][k]['mean_pp'], 4), round(P['pre'][k]['t'], 2)) for k in ('b_hi', 'b_lo', 'I')},
                         'DDD': P['DDD'], 'cond': cond, 'fals': O['falsification'], 'mech': mech}), ensure_ascii=False, indent=1))
