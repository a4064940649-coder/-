"""步驟 8：部位模擬（10 槽，閒置放 0050，日頻 NAV）＋與主策略、處置部位的月相關與配置。"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import rlib as L

AG = L.TOP.parent
x = L.event_frame(); d = L.panel(); idx = d['idx']
END = pd.Timestamp('2026-10-12')
F50, TAX50 = 0.000855, 0.001

def run(buy=0.001, sell=0.004, slots=10):
    ev = x[x.tradable & x.is_del & x.high].sort_values(['eff', 'pressure'], ascending=[True, False])
    ev = ev.drop_duplicates(['code', 'eff'], keep='first')
    byday = {int(t): g for t, g in ev.groupby('t')}
    t0 = int(ev.t.min()) - 1
    A = d['adjf']; e50 = d['e50'].to_numpy()
    u50 = 1.0 / e50[t0]          # 0050 單位
    pos = []                     # [code, t_exit, value]
    nav = {}; used = []; skipped = 0; trades = []
    for t in range(t0, len(idx)):
        if t > t0:
            for p in pos:
                p[2] *= A[p[0]].iloc[t] / A[p[0]].iloc[t - 1]
        # 出場（E+20 收盤）
        keep = []
        for p in pos:
            if p[1] == t:
                u50 += p[2] * (1 - sell) * (1 - F50) / e50[t]
                trades[p[3]]['exit_val'] = p[2]
            else: keep.append(p)
        pos = keep
        V = u50 * e50[t] + sum(p[2] for p in pos)
        if t in byday:
            for r in byday[t].itertuples():
                if len(pos) >= slots: skipped += 1; continue
                amt = min(V / slots, 0.05 * r.adv20)
                need = amt * (1 + buy) / ((1 - F50 - TAX50))      # 賣 0050 籌錢
                if need > u50 * e50[t]: amt *= u50 * e50[t] / need; need = u50 * e50[t]
                u50 -= need / e50[t]
                trades.append({'code': r.code, 'eff': r.eff, 'amt': amt, 'w': amt / V})
                pos.append([r.code, t + L.H, amt, len(trades) - 1])
        nav[idx[t]] = u50 * e50[t] + sum(p[2] for p in pos)
        used.append(len(pos))
    nav = pd.Series(nav)
    return nav, pd.Series(used, index=nav.index), skipped, pd.DataFrame(trades)

nav, used, skipped, trades = run()
navs, _, _, _ = run(buy=0.005, sell=0.005)
nav.to_frame('normal').assign(stress=navs).to_parquet(L.HERE / 'sleeve_nav_r.parquet')

main = pd.read_parquet(AG / 'main_v2/navs_gapfix.parquet')
MAIN = main['EXFIN|n10|normal'].dropna(); EW = main['EXFIN|EW'].dropna()
DISP = pd.read_parquet(AG / 'strategy_ensemble/strategy_nav.parquet')['DISP_REAL'].dropna()      # 處置部位（正式事件、閒置 0050）
DISP_ALT = pd.read_parquet(AG / 'disposal_v2/navs.parquet')['TWSE_K10_0050'].dropna()
E50 = d['e50']

def seg(s, a, b):
    z = s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))].dropna()
    yrs = (z.index[-1] - z.index[0]).days / 365.25
    return z, (z.iloc[-1] / z.iloc[0]) ** (1 / yrs) - 1, float((z / z.cummax() - 1).min())

def mret(s):
    return s.groupby(s.index.to_period('M')).last().pct_change()

out = {'skipped_full': skipped, 'n_trades': int(len(trades)), 'avg_slots_used': round(float(used[used.index <= END].mean()), 3),
       'share_days_holding': round(float((used[used.index <= END] > 0).mean()), 3),
       'avg_weight_per_trade': round(float(trades.w.mean()), 4), 'cap_binding_share': round(float((trades.w < 0.0999).mean()), 3)}
segs = {'2010-11_2018': ('2010-11-30', '2018-12-31'), '2019_2026-10-12': ('2019-01-02', '2026-10-12')}
for k, (a, b) in segs.items():
    z, c, dd = seg(nav, a, b)
    o = {'sleeve_CAGR': round(c * 100, 2), 'sleeve_maxDD_daily': round(dd * 100, 1)}
    for nm, s in (('0050', E50), ('main', MAIN), ('EW', EW)):
        o[nm + '_CAGR'] = round(seg(s, a, b)[1] * 100, 2)
    o['stress_CAGR'] = round(seg(navs, a, b)[1] * 100, 2)
    pa, pb = pd.Period(a, 'M'), pd.Period(b, 'M')
    sel = lambda m: m[(m.index >= pa) & (m.index <= pb)].dropna()
    ms = mret(nav[nav.index <= pd.Timestamp(b)]); m5 = mret(E50[E50.index <= pd.Timestamp(b)]); mm = mret(MAIN[MAIN.index <= pd.Timestamp(b)])
    if pd.Timestamp(a).day > 3: pa = pa + 1    # 段起點在月底：從下個月算
    ex = sel(ms - m5); exm = sel(ms - mm)
    o['t_vs_0050'] = round(L.tstat(ex), 2); o['mean_monthly_ex_0050'] = round(ex.mean() * 100, 3); o['t_vs_main'] = round(L.tstat(exm), 2)
    o['n_months'] = int(len(ex))
    out[k] = o

# 月相關
ms, mm, md, m5 = mret(nav[nav.index <= END]), mret(MAIN), mret(DISP[DISP.index <= END]), mret(E50[E50.index <= END])
J = pd.concat({'s': ms, 'm': mm, 'd': md, 'e': m5}, axis=1)
J = J[J.index >= pd.Period('2010-12', 'M')]
J.loc[J.index < pd.Period('2021-10', 'M'), 'd'] = np.nan      # 處置部位只用 2021-10 起（與 disposal_v2 同窗）
Jm = J[['s', 'm', 'e']].dropna(); Jd = J.dropna()
_a = pd.concat([ms, mret(DISP_ALT[DISP_ALT.index <= END])], axis=1).dropna()
out['corr_alt_disposal_v2_TWSE_K10_0050'] = round(_a.corr().iloc[0, 1], 3)
out['corr'] = {'sleeve_main': round(Jm.s.corr(Jm.m), 3), 'sleeve_disposal': round(Jd.s.corr(Jd.d), 3),
               'sleeve_main_overlap_disposal': round(Jd.s.corr(Jd.m), 3),
               'excess_sleeve_vs_main_excess': round((Jm.s - Jm.e).corr(Jm.m - Jm.e), 4),
               'excess_sleeve_vs_main_excess_2021-10+': round((Jd.s - Jd.e).corr(Jd.m - Jd.e), 4),
               'n_months_main': int(len(Jm)), 'n_months_disp': int(len(Jd))}

# 配置（月再平衡、月頻回撤、t 對 0050）
def blend(w, a, b):
    z = J[(J.index >= pd.Period(a, 'M')) & (J.index <= pd.Period(b, 'M'))]
    r = sum(z[k] * v for k, v in w.items())
    r = r.dropna(); e = z['e'].reindex(r.index)
    eq = (1 + r).cumprod()
    yrs = len(r) / 12
    # 年化用實際天數：起點＝第一個月前一月底，終點＝2026-10-12
    return {'CAGR': round((eq.iloc[-1] ** (1 / yrs) - 1) * 100, 2), 't_vs_0050': round(L.tstat(r - e), 2),
            'maxDD_monthly': round(float((eq / eq.cummax() - 1).min()) * 100, 1), 'n_months': int(len(r))}
out['alloc'] = {
    '2021-10_2026-10|70/20/10': blend({'m': .7, 'd': .2, 's': .1}, '2021-10', '2026-10'),
    '2021-10_2026-10|80/20': blend({'m': .8, 'd': .2}, '2021-10', '2026-10'),
    '2010-12_2026-10|90/10': blend({'m': .9, 's': .1}, '2010-12', '2026-10'),
    '2010-12_2026-10|100': blend({'m': 1.0}, '2010-12', '2026-10'),
}
# 用實際天數的年化（最後一個月只到 10-12）
def blend_days(w, a, b):
    z = J[(J.index >= pd.Period(a, 'M')) & (J.index <= pd.Period(b, 'M'))]
    r = sum(z[k] * v for k, v in w.items()).dropna()
    eq = (1 + r).cumprod()
    d0 = (pd.Period(a, 'M') - 1).to_timestamp(how='end'); yrs = (END - d0).days / 365.25
    return round((eq.iloc[-1] ** (1 / yrs) - 1) * 100, 2)
for k, w in (('2021-10_2026-10|70/20/10', {'m': .7, 'd': .2, 's': .1}), ('2021-10_2026-10|80/20', {'m': .8, 'd': .2}),
             ('2010-12_2026-10|90/10', {'m': .9, 's': .1}), ('2010-12_2026-10|100', {'m': 1.0})):
    a = k.split('|')[0].split('_')[0]
    out['alloc'][k]['CAGR_days'] = blend_days(w, a, '2026-10')
L.dump(out, 'sleeve.json')
import pprint; pprint.pprint(out, width=160)
