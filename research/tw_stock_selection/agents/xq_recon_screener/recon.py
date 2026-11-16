#!/usr/bin/env python3
"""XQ「選股回測」對帳：用研究中心資料重現 20 號腳本（MinSurge 2.5、RET20 > 10%、季線上、B11 訊號日）＋「最大持有 21 期、下期開盤進出、同檔不重複」。
每個訊號一筆獨立交易（不受 10 檔上限、資金限制）。只用 s19lib／s16lib 的資料與訊號（不改既有檔案）。
情境：範圍 {只上市 TW、上市＋上櫃 ALL（上櫃 2018-10 營收用 s16 補洞檔、SURGE σ>0）} × 持有定義 {H21、H22} × 同檔判定 {strict、loose} × 價格 {還原 adj、原始 raw}；費用 {0、0.2%、0.6%} 事後套用。
python3 -I -B recon.py → results.json、trades.csv.gz（同目錄）。中間檔放 scratchpad/xqrecon/。單一行程。"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd

H = Path(__file__).resolve().parent
AG = H.parent
XT = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/xqrecon')
sys.path.insert(0, str(AG / 's19_xq_threshold'))
import s19lib                                                       # noqa: E402
from s19lib import s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib   # noqa: E402
for mod, a in [(s19lib, 'S19'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
               (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')]:
    setattr(mod, a, XT)
PATCH = AG / 's16_surge_otc' / 'rev_patch_otc_201810.csv'

t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s19lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗：面板和 S6～S19 不同'
fix = s19lib.e50_fix_suspension(P)
INNOV, innov_names = s4lib.innov_panel(P)
print('載入', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

# ---------------------------------------------------------------- 股池
U_mon = {'TW': stlib.universe(P, 'twse') & ~INNOV, 'ALL': stlib.universe(P, 'all') & ~INNOV}   # 只決定「哪些月有訊號」（同 s19：≥ 50 檔 SURGE）
UX_tw, Cadj, liqX = s19lib.script_universe(P, INNOV)
with np.errstate(invalid='ignore'):
    UX_all = (P.MK > 0) & ~INNOV & P.not91[None, :] & P.has & (Cadj >= 10.0) & (liqX >= 1e7)   # 同 script_universe，只是市場別放寬到上櫃
UX = {'TW': UX_tw, 'ALL': UX_all}
del Cadj, liqX
# 原始值版股池（XQ「原始日線」）：收盤 ≥ 10、60 日均（收盤 × 股數）≥ 1,000 萬；季線、RET20 也用原始收盤
val = np.where(P.listed, P.CF * P.VOLF, np.nan)
liqR = pd.DataFrame(val).rolling(60, min_periods=60).mean().to_numpy(); del val
with np.errstate(invalid='ignore'):
    baseR = ~INNOV & P.not91[None, :] & P.has & (P.CF >= 10.0) & (liqR >= 1e7)
UXR = {'TW': baseR & (P.MK == 1), 'ALL': baseR & (P.MK > 0)}
del baseR, liqR
with np.errstate(invalid='ignore', divide='ignore'):
    RET20_raw = P.CF / cslib.shift(P.CF, 20) - 1
    F_MA60_raw = P.CF > stlib.rmean(P.CF, 60)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'NEARHI', 'V60', 'liq', 'LIMUP', 'LIMDN', 'REF', 'F_MA20', 'AMT', 'VOLF'):
    if hasattr(P, a):
        delattr(P, a)
del INNOV
gc.collect()
print('股池', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

# ---------------------------------------------------------------- SURGE（σ>0）：TW＝原營收檔（同 s19）；ALL＝加上 s16 上櫃 2018-10 補洞
S_tw, n0_tw = s19lib.rev_std(P)
r = pd.read_csv(stlib.REV_CSV, dtype={'code': str})
pt = pd.read_csv(PATCH, dtype={'code': str})
have = set(r.loc[r['month'] == '2018-10-01', 'code'])
pt = pt[~pt['code'].isin(have)]
patched = XT / 'rev_patched.csv'
pd.concat([r, pt[['month', 'market', 'code', 'name', 'revenue_k_twd']]], ignore_index=True).to_csv(patched, index=False)
n_patch = int(len(pt)); del r, pt
orig = stlib.REV_CSV; stlib.REV_CSV = str(patched)
S_all, n0_all = s19lib.rev_std(P)
stlib.REV_CSV = orig
SS = {'TW': S_tw, 'ALL': S_all}
months = list(S_tw.index)
avB, _ = s19lib.b11_rows(P, months, 11)
print('SURGE', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)


def signals(scope, price):
    """門檻版訊號（s19lib.make_signal_thr，XT：腳本股池、MinSurge 2.5、RET20 > 10%、季線上）→ [(k, j, surge)]。"""
    if price == 'raw':
        keep = (P.RET20, P.F_MA60); P.RET20, P.F_MA60 = RET20_raw, F_MA60_raw
        ub = UXR[scope]
    else:
        ub = UX[scope]
    Sg = s19lib.make_signal_thr(P, U_mon[scope], ub, None, SS[scope], avB, 2.5, 2.0)
    if price == 'raw':
        P.RET20, P.F_MA60 = keep
    out = []
    for x in Sg['info']:
        k = x['k']
        for c in x['targets']:
            j = P.ci[c]
            out.append((k, j, float(SS[scope].loc[pd.Period(x['month'], 'M')].iloc[j])))
    months_used = [x['month'] for x in Sg['info']]
    del Sg; gc.collect()
    return out, months_used


BARS = {}


def bars(j):
    if j not in BARS:
        BARS[j] = np.flatnonzero(P.has[:, j] & np.isfinite(P.O[:, j]) & (P.O[:, j] > 0))
    return BARS[j]


def trades(sig, hold, block, price):
    """每個訊號一筆：訊號日 k 的下一根 K 棒開盤買（該股自己的成交日）；第 hold 根之後的開盤賣：
    H21＝買進那根算第 1 根、第 21 根收盤觸發 → 第 22 根開盤賣（open-to-open 共 21 根）；H22＝買進那根不算 → 第 23 根開盤賣（22 根）。
    block：strict＝訊號日收盤仍持有（前一筆出場在訊號日之後）就不進；loose＝前一筆在本筆買進的同一個開盤出場時允許再進。
    下市：最後一根收盤出；面板結束仍持有：以 2026-10-22 收盤估值（open_end）。"""
    nb = hold
    last_exit = {}
    rows = []
    for k, j, s in sorted(sig, key=lambda z: (z[0], -z[2])):
        b = bars(j)
        p = int(np.searchsorted(b, k, side='right'))
        if p >= len(b):
            continue
        e = int(b[p])
        le = last_exit.get(j, -1)
        if (block == 'strict' and le > k) or (block == 'loose' and le > e):
            continue
        if p + nb < len(b):
            x = int(b[p + nb]); why = 'hold'
        else:
            lastb = int(b[-1])
            if not P.MK[lastb + 1:, j].any() and lastb < P.T - 1:
                x = lastb; why = 'delist'
            else:
                x = P.T - 1; why = 'open_end'
        if price == 'adj':
            pe = P.AO[e, j]
            px = P.AO[x, j] if why == 'hold' else P.G[x, j]
        else:
            pe = P.O[e, j]
            px = P.O[x, j] if why == 'hold' else P.CF[x, j]
        rows.append((k, e, x, j, s, why, float(px / pe - 1)))
        last_exit[j] = x
    df = pd.DataFrame(rows, columns=['k', 'ent_t', 'exit_t', 'j', 'surge', 'why', 'ret'])
    df['code'] = P.COLS[df['j'].to_numpy()]
    df['mk'] = P.MK[df['ent_t'].to_numpy(), df['j'].to_numpy()]
    df['sig_date'] = P.IDX[df['k'].to_numpy()].strftime('%Y-%m-%d')
    df['ent_date'] = P.IDX[df['ent_t'].to_numpy()].strftime('%Y-%m-%d')
    df['exit_date'] = P.IDX[df['exit_t'].to_numpy()].strftime('%Y-%m-%d')
    return df


def net(ret, f):
    return (1 + ret) * (1 - f) / (1 + f) - 1


def stats(df, f):
    d = df.copy(); d['r'] = net(d['ret'], f)
    c = d[d['why'] != 'open_end']
    o = dict(n=int(len(c)), n_open_end=int((d['why'] == 'open_end').sum()), n_delist=int((d['why'] == 'delist').sum()),
             win=float((c['r'] > 0).mean()), n_win=int((c['r'] > 0).sum()), n_loss=int((c['r'] <= 0).sum()),
             mean=float(c['r'].mean()), median=float(c['r'].median()), max=float(c['r'].max()), min=float(c['r'].min()),
             sum=float(c['r'].sum()), max_code=str(c.loc[c['r'].idxmax(), 'code']), max_date=str(c.loc[c['r'].idxmax(), 'ent_date']),
             min_code=str(c.loc[c['r'].idxmin(), 'code']), min_date=str(c.loc[c['r'].idxmin(), 'ent_date']),
             incl_open=dict(n=int(len(d)), win=float((d['r'] > 0).mean()), mean=float(d['r'].mean()), sum=float(d['r'].sum())))
    # 最大連續獲利／虧損（依進場日、代號排序；連續同號交易的報酬加總與連乘）
    c2 = c.sort_values(['ent_t', 'code'])
    best = dict(win_sum=0.0, win_prod=0.0, loss_sum=0.0, loss_prod=0.0, win_len=0, loss_len=0)
    def flush(run, sign):
        ssum, sprod = float(np.sum(run)), float(np.prod(1 + np.array(run)) - 1)
        if sign > 0:
            best['win_sum'] = max(best['win_sum'], ssum); best['win_prod'] = max(best['win_prod'], sprod); best['win_len'] = max(best['win_len'], len(run))
        else:
            best['loss_sum'] = min(best['loss_sum'], ssum); best['loss_prod'] = min(best['loss_prod'], sprod); best['loss_len'] = max(best['loss_len'], len(run))
    run, sign = [], 0
    for v in c2['r']:
        sg = 1 if v > 0 else -1
        if sg != sign and run:
            flush(run, sign); run = []
        sign = sg; run.append(v)
    if run:
        flush(run, sign)
    o['streaks_by_entry'] = best
    y = c.assign(year=c['ent_date'].str[:4]).groupby('year')['r']
    o['by_year'] = {k: dict(n=int(len(v)), win=float((v > 0).mean()), mean=float(v.mean()), sum=float(v.sum())) for k, v in y}
    return o


IDX = pd.DatetimeIndex(P.IDX)
D16 = int(np.searchsorted(P.IDX.values, np.datetime64('2016-01-04')))
D18e = IDX[IDX < pd.Timestamp('2019-01-01')][-1]


def curve(df, f, price):
    """(a) 每天把所有持有中交易的「當日報酬」等權平均，沒持股＝0，串接。進場日＝收盤÷開盤，中間＝含息日報酬，出場日＝開盤÷前一日收盤；買賣各扣 f。"""
    sm = np.zeros(P.T); ct = np.zeros(P.T)
    for e, x, j, why in zip(df['ent_t'], df['exit_t'], df['j'], df['why']):
        if price == 'adj':
            pe, cl = P.AO[e, j], P.G[:, j]
            px_open = P.AO[x, j]
        else:
            pe, cl = P.O[e, j], P.CF[:, j]
            px_open = P.O[x, j]
        if x == e:
            continue
        rr = np.empty(x - e + 1)
        rr[0] = (cl[e] / pe) / (1 + f) - 1
        if price == 'adj':
            rr[1:x - e] = P.R[e + 1:x, j]
        else:
            rr[1:x - e] = cl[e + 1:x] / cl[e:x - 1] - 1
        if why == 'hold':
            rr[-1] = (px_open / cl[x - 1]) * (1 - f) - 1
        else:                                                       # 下市／面板結束：最後一天收盤
            rr[-1] = (cl[x] / cl[x - 1]) * (1 - f) - 1
        sm[e:x + 1] += rr; ct[e:x + 1] += 1
    dr = np.where(ct > 0, sm / np.maximum(ct, 1), 0.0)
    dr[:D16] = 0.0
    nav = pd.Series(np.cumprod(1 + dr), index=IDX).iloc[D16 - 1:]
    nav = nav / nav.iloc[0]
    runup = float((nav / nav.cummin() - 1).max())
    yr = nav.groupby(nav.index.year).last()
    yr = (yr / yr.shift(1).fillna(1.0) - 1)
    return dict(total=float(nav.iloc[-1] - 1), maxdd=float((nav / nav.cummax() - 1).min()), max_runup=runup,
                cagr_since2019=stlib.cagr(nav, D18e, IDX[-1]), cagr_full=stlib.cagr(nav, None, None),
                yearly={int(k): float(v) for k, v in yr.items()}, avg_open_trades=float(ct[D16:].mean()), days_flat=float((ct[D16:] == 0).mean())), nav


def chain(df, f):
    """(b) 依出場順序連乘每筆報酬（log10 表示）；另列 (c) 報酬加總＝筆數 × 平均。"""
    c = df[df['why'] != 'open_end'].sort_values(['exit_t', 'code'])
    r = net(c['ret'], f).to_numpy()
    cs = np.concatenate([[0.0], np.cumsum(r)])
    return dict(b_chain_log10=float(np.sum(np.log10(1 + r))), c_sum=float(r.sum()),
                c_sum_curve_maxdd_pp=float((cs - np.maximum.accumulate(cs)).min()), c_sum_curve_max_runup_pp=float((cs - np.minimum.accumulate(cs)).max()))


FEES = {'f0': 0.0, 'f02': 0.002, 'f06': 0.006}
res = dict(meta=dict(panel=dict(T=P.T, N=P.N, last=str(IDX[-1].date())), e50_fix=fix, n_sigma0=dict(TW=int(n0_tw), ALL=int(n0_all)), n_patch_rows=n_patch,
                     note='每個訊號一筆；B11 訊號日；XT 門檻（MinSurge 2.5、RET20>10%、還原收盤>60MA、腳本股池 Cadj≥10、60 日均 Cadj×股數≥1,000 萬、排除創新板與 91 開頭）'),
           scen={})
alltr = []
NAV = {}
for price in ('adj', 'raw'):
    for scope in ('TW', 'ALL'):
        sig, mu = signals(scope, price)
        sd = pd.DataFrame(sig, columns=['k', 'j', 's'])
        sig_info = dict(n_signals=len(sig), n_months=len(mu), first_month=mu[0], last_month=mu[-1],
                        by_year=sd.assign(y=IDX[sd['k'].to_numpy()].year).groupby('y').size().to_dict())
        for hold in (21, 22):
            for block in ('strict', 'loose'):
                if price == 'raw' and block == 'loose':
                    continue
                name = f'{scope}_{price}_H{hold}_{block}'
                df = trades(sig, hold, block, price)
                o = dict(signals=sig_info, fees={})
                for fk, f in FEES.items():
                    st = stats(df, f)
                    cv, nav = curve(df, f, price)
                    st['curve_a'] = cv; st['chain'] = chain(df, f)
                    o['fees'][fk] = st
                    if price == 'adj':
                        NAV[f'{name}_{fk}'] = nav
                res['scen'][name] = o
                df['scen'] = name
                alltr.append(df)
                s = o['fees']['f06']
                print(name, 'n', s['n'], 'win %.3f mean %.4f max %.3f min %.3f sum %.1f | curve tot %.1f mdd %.3f runup %.1f' %
                      (s['win'], s['mean'], s['max'], s['min'], s['sum'], s['curve_a']['total'], s['curve_a']['maxdd'], s['curve_a']['max_runup']),
                      f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

# 只上市版與上市＋上櫃版的訊號差異（主情境 adj H21 strict）
a = alltr[[d['scen'].iloc[0] for d in alltr].index('TW_adj_H21_strict')]
b = alltr[[d['scen'].iloc[0] for d in alltr].index('ALL_adj_H21_strict')]
res['diff_TW_vs_ALL'] = dict(all_otc_share=float((b['mk'] == 2).mean()), n_tw_in_all=int((b['mk'] == 1).sum()))
res['innov_names'] = innov_names
res['runtime_s'] = round(time.time() - t0, 1); res['maxrss_mb'] = rss()
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
tr = pd.concat(alltr, ignore_index=True)
tr = tr[tr['scen'].isin(['TW_adj_H21_strict', 'TW_adj_H22_strict', 'ALL_adj_H21_strict', 'ALL_adj_H22_strict',
                         'TW_adj_H21_loose', 'ALL_adj_H21_loose', 'TW_raw_H21_strict', 'ALL_raw_H21_strict'])]
tr[['scen', 'code', 'mk', 'sig_date', 'ent_date', 'exit_date', 'why', 'surge', 'ret']].round(6).to_csv(H / 'trades.csv.gz', index=False)
pd.DataFrame({k: v for k, v in NAV.items() if k.endswith('_f06') or k.endswith('_f02')}).round(6).to_csv(XT / 'navs.csv.gz')
print('完成', res['runtime_s'], 's', res['maxrss_mb'], 'MB')
