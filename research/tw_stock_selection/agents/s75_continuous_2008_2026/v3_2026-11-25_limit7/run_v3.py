#!/usr/bin/env python3
"""S75 連續帳戶（2008-01-02～2026-10-22）v3：漲跌幅限制照當時制度（工程 ENG-LIM7，2026-11-25）。不覆蓋 v2，只寫本資料夾；不改任何既有檔。
用法：python3 -I -B run_v3.py        （約 1 分鐘；需先有連續面板快取 scratchpad/s75/cache_cont，見 ../build_panel.py）

與 v2（../v2_2026-11-21/run_v2.py v2 模式）唯一的差別：載入面板後，只在記憶體覆寫 P.LIMUP／P.LIMDN 的 2015-06-01 以前各列。
  - 2015-06-01 以前：漲跌幅 7%（引擎原本全期 10%，cslib.py 第 208～209 行）。
  - 2008-10-13～2008-10-24（含）：跌幅 3.5%、漲幅仍 7%。
  - 2015-06-01 起：引擎原值不動（10%）。
  - 取整與跳動點級距沿用引擎寫法：漲停＝floor(REF×(1＋p)÷tick＋1e-9)×tick、跌停＝ceil(REF×(1−p)÷tick−1e-9)×tick，tick＝cslib.tick(未取整的漲跌停價)。
  引擎讀漲跌停的地方只有 s10lib.simulate_s10 三處（買進 o ≥ LIMUP 不買不遞補；賣出 o ≤ LIMDN 順延；rebalance 減碼跳過 o ≤ LIMDN），都是執行當下讀 P.LIMUP／P.LIMDN。
其餘（0050 銜接檔、REDUCE_FIX、面板、訊號、成本、20 條路徑＝5 種本金 × 位移 0～3）與 v2 完全相同。

同一支程式依序跑三個情境（每個情境 2 版本 × 20 條路徑，一次一格、跑完釋放）：
  v2   ＝不覆寫（全期 10%）。必須與 ../v2_2026-11-21/ 既有的淨值（逐日）、交易明細、results_v2.json 逐項相同，否則中止、不寫任何 v3 檔。
  lim7 ＝只改 7%（不含 3.5%）。只用來量 3.5% 跌幅的影響，以及對 R97（../../r97_replicate_limit7/）。
  v3   ＝7% ＋ 2008-10 的 3.5% 跌幅（正式）。
驗收：V_BRIDGE 標準路徑（100 萬、位移 0）的 lim7／v3 要與 R97 results.json、nav_lim7.csv.gz、trades_lim7.csv.gz 相符，否則中止。
輸出：nav_2008_2026_v3.csv.gz、trades_v3.csv.gz、yearly_v3.csv、results_v3.json、compare_v2_v3.json、external_numbers_v3.json。"""
import sys, json, time, resource, gc, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
S75 = HERE.parent
V2 = S75 / 'v2_2026-11-21'
R97 = S75.parent / 'r97_replicate_limit7'
sys.path.insert(0, str(S75))
import numpy as np, pandas as pd
import s75lib as K
from s75lib import R, cslib

CUT = '2015-06-01'                      # 這一天起漲跌幅 10%（之前 7%）
HALF = ('2008-10-13', '2008-10-24')     # 跌幅減半（3.5%）期間，含頭尾
SCENS = ('v2', 'lim7', 'v3')
STD = '100萬|+0'
t0 = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
assert cslib.REDUCE_FIX is False and cslib.E50_SOURCE is None

# ---------------------------------------------------------------- v2 口徑：0050 銜接檔（記憶體重算後與 v2 既有檔比對，不重寫）＋ REDUCE_FIX
old = pd.read_csv(cslib.E50_CSV, parse_dates=['date']).set_index('date')['adj'].sort_index()
new = pd.read_csv(cslib.E50_OFFICIAL_CSV, parse_dates=['date']).set_index('date')['adj'].sort_index()
seam = new.index[0]; assert str(seam.date()) == '2015-01-05' and seam in old.index
sp = pd.concat([old.loc[:seam].iloc[:-1] * float(new.loc[seam] / old.loc[seam]), new])
ex = pd.read_csv(V2 / 'e50_spliced.csv', parse_dates=['date']).set_index('date')['adj']
assert sp.index.equals(ex.index) and float((sp - ex).abs().max()) < 1e-9, '0050 銜接檔與 v2 不同（上游 0050 檔已變）'
cslib.E50_SOURCE = V2 / 'e50_spliced.csv'; cslib.REDUCE_FIX = True
P, U = K.load_cont()
assert P.reduce_fix and P.exright_fix
info = dict(reduce_fix=True, reduce_info=P.reduce_info, e50_source=str(P.e50_source), e50_missing=int(P.e50_missing), panel=dict(T=int(P.T), N=int(P.N)))
D0, DE = P.IDX[P.t0], P.IDX[-1]
assert (str(D0.date()), str(DE.date())) == ('2008-01-02', '2026-10-22')
D19 = K.last_before(P.IDX, '2019-01-01'); D157 = pd.Timestamp('2015-07-01'); DH1 = K.last_before(P.IDX, '2015-07-01')
bench = pd.Series(P.E50, index=P.IDX).loc[D0:]; bench = bench / bench.iloc[0] * 1e6


# ---------------------------------------------------------------- 漲跌停價（三套，只差 2015-06-01 以前的列）
def limits(ref, up_pct, dn_pct):
    """與 cslib.Panel.__init__ 第 208～209 行同一寫法，只是百分比可換。"""
    u = ref * (1 + up_pct); t = cslib.tick(u); u = np.floor(u / t + 1e-9) * t
    d = ref * (1 - dn_pct); t = cslib.tick(d); d = np.ceil(d / t - 1e-9) * t
    return u, d


iv = P.IDX.values
cut = int(np.searchsorted(iv, np.datetime64(CUT), side='left')); assert str(P.IDX[cut].date()) == CUT
h0 = int(np.searchsorted(iv, np.datetime64(HALF[0]), side='left')); h1 = int(np.searchsorted(iv, np.datetime64(HALF[1]), side='right'))
assert (str(P.IDX[h0].date()), str(P.IDX[h1 - 1].date())) == HALF and P.t0 < h0 < h1 < cut
LU10, LD10 = P.LIMUP, P.LIMDN                                   # 引擎原值（不改它，只換 P 上的指標）
u, d = limits(P.REF, 0.10, 0.10)                                # 自己的寫法先對引擎：全面板逐格相同才往下
assert np.array_equal(u, LU10, equal_nan=True) and np.array_equal(d, LD10, equal_nan=True), 'limits() 與引擎 10% 寫法不一致'
del u, d
u7, d7 = limits(P.REF[:cut], 0.07, 0.07)
LU7 = LU10.copy(); LU7[:cut] = u7; LD7 = LD10.copy(); LD7[:cut] = d7
_, d35 = limits(P.REF[h0:h1], 0.07, 0.035)
LD35 = LD7.copy(); LD35[h0:h1] = d35
for a, b in ((LU7, LU10), (LD7, LD10), (LD35, LD10)):          # 2015-06-01 起逐位不動
    assert np.array_equal(a[cut:], b[cut:], equal_nan=True)
assert np.array_equal(np.delete(LD35, np.s_[h0:h1], 0), np.delete(LD7, np.s_[h0:h1], 0), equal_nan=True)
LIM = dict(v2=(LU10, LD10), lim7=(LU7, LD7), v3=(LU7, LD35))
fin = np.isfinite(LU10[:cut])
info['limit_override'] = dict(
    engine_10pct_formula_recheck='limits(REF,10%) 與引擎 LIMUP／LIMDN 全面板逐格相同',
    cut_row=cut, cut_date=CUT, rows_7pct=[0, cut - 1], dates_7pct=[str(P.IDX[0].date()), str(P.IDX[cut - 1].date())],
    sim_first_row=int(P.t0), sim_rows_7pct=[int(P.t0), cut - 1], n_sim_rows_7pct=int(cut - P.t0),
    rows_dn_3_5pct=[h0, h1 - 1], dates_dn_3_5pct=[str(P.IDX[h0].date()), str(P.IDX[h1 - 1].date())], n_rows_dn_3_5pct=int(h1 - h0),
    cells_changed_7pct=dict(limup=int((fin & (LU7[:cut] != LU10[:cut])).sum()), limdn=int((fin & (LD7[:cut] != LD10[:cut])).sum()), finite_cells=int(fin.sum())),
    cells_changed_3_5pct_vs_7pct=int((np.isfinite(LD7[h0:h1]) & (LD35[h0:h1] != LD7[h0:h1])).sum()),
    rows_from_cut_unchanged=True)
del u7, d7, d35, fin


def data_check():
    """用面板自己的收盤價核對制度日期（不上網）：每天「收盤超出該級漲跌停價」的格數。制度對的話，期間內 ≈ 0（只剩新上市無漲跌幅日、參考價缺漏）。"""
    C = P.C; has = P.has

    def cnt(rows, up_pct, dn_pct):
        uu, dd = limits(P.REF[rows], up_pct, dn_pct); c = C[rows]; m = has[rows]
        return dict(n_traded=int(m.sum()), close_above_limup=int((m & (c > uu + 1e-6)).sum()), close_at_limup=int((m & (np.abs(c - uu) <= 1e-6)).sum()),
                    close_below_limdn=int((m & (c < dd - 1e-6)).sum()), close_at_limdn=int((m & (np.abs(c - dd) <= 1e-6)).sum()))
    o = {}
    o['250 個交易日（切換日前）對 7%'] = cnt(slice(cut - 250, cut), 0.07, 0.07)
    o['250 個交易日（切換日起）對 7%'] = cnt(slice(cut, cut + 250), 0.07, 0.07)
    o['250 個交易日（切換日起）對 10%'] = cnt(slice(cut, cut + 250), 0.10, 0.10)
    o['切換日前後各 3 天對 7%'] = {str(P.IDX[t].date()): cnt(slice(t, t + 1), 0.07, 0.07) for t in range(cut - 3, cut + 3)}
    o['2008-10 跌幅減半前後逐日（對 3.5% 跌停／7% 跌停）'] = {
        str(P.IDX[t].date()): dict(in_window=bool(h0 <= t < h1), vs_3_5pct={k: v for k, v in cnt(slice(t, t + 1), 0.07, 0.035).items() if 'limdn' in k or k == 'n_traded'},
                                   vs_7pct={k: v for k, v in cnt(slice(t, t + 1), 0.07, 0.07).items() if 'limdn' in k}) for t in range(h0 - 3, h1 + 3)}
    return o


info['data_check'] = data_check()


# ---------------------------------------------------------------- 執行
def lock_list(sg):
    """2015-06-01 以前每月名單（換股日隔天開盤）對三套漲停價的狀態；不管有沒有空位都列（引擎實際有沒有因此不買，看 st 的 blk_limup）。"""
    rows = []
    for i in sg['info']:
        t = i['k'] + 1
        if t >= cut or t >= P.T:
            continue
        for c in i['targets']:
            j = P.ci[c]; o = P.O[t, j]
            if np.isfinite(o) and o >= LU7[t, j] - 1e-9:
                rows.append(dict(month=i['month'], date=str(P.IDX[t].date()), code=str(c), ref=float(P.REF[t, j]), open=float(o), close=float(P.C[t, j]),
                                 limup_7=float(LU7[t, j]), limup_10=float(LU10[t, j]), lock_10=bool(o >= LU10[t, j] - 1e-9)))
    return rows


def top_share(lg, final_nav):
    """S82／R82 口徑：全部交易（含期末未平倉）依報酬率排序，前 5% 的損益 ÷ 全部選股交易損益。筆數取 ceil(n×5%)（R82 寫法）；另附 round（S82 寫法）。"""
    r = lg['ret'].to_numpy(float); pnl = lg['pnl'].to_numpy(float); n = len(r); order = np.argsort(-r, kind='stable')
    o = dict(n_all=int(n), win_rate_all=float((r > 0).mean()), mean_ret=float(r.mean()), median_ret=float(np.median(r)), pnl_sum_all=float(pnl.sum()), nav_gain=float(final_nav - 1e6))
    for nm, m in (('ceil', int(math.ceil(n * 0.05))), ('round', int(round(n * 0.05)))):
        top = order[:m]
        o[nm] = dict(n_top=m, threshold_ret=float(r[top[-1]]), pnl_top=float(pnl[top].sum()), share_of_trade_pnl=float(pnl[top].sum() / pnl.sum()),
                     share_of_nav_gain=float(pnl[top].sum() / (final_nav - 1e6)))
    return o


def run_scen(sc):
    P.LIMUP, P.LIMDN = LIM[sc]
    res = {}; NAV = {}; TR = []; PN = {}
    for v, rule in K.VERSIONS.items():
        F, RT = K.surge(P, rule)
        paths = {}; pn = {}
        for s in K.SHIFTS:
            sg = K.signal(P, U, RT, s)
            for cap in K.CAPS:
                nav, lg, st = K.sim(P, sg, U, cap)              # 一次一格
                key = f'{int(cap / 1e4)}萬|+{s}'
                paths[key] = dict(cagr_full=K.cagr(nav, D0, DE), maxdd=R.maxdd(nav), cagr_since2019=K.cagr(nav, D19, DE), final_nav=float(nav.iloc[-1]), n_all=int(len(lg)),
                                  blk_limup=int(st['blk_limup']), sell_defer_limdn=int(st['sell_defer_limdn']))
                pn[key] = nav.to_numpy(float).copy()
                if s == 0 and cap == 1e6:
                    std = (nav, lg, [dict(month=i['month'], sig_date=i['sig_date'], targets=' '.join(i['targets'])) for i in sg['info']], lock_list(sg),
                           {k: int(st[k]) for k in ('blk_limup', 'sell_defer_limdn', 'sell_defer_noopen', 'n_buy')})
                del nav, lg, st
            del sg; gc.collect()
        nav, lg, sig, locks, stc = std; NAV[v] = nav; PN[v] = pn; b = bench.reindex(nav.index)
        x = lg[lg['why'] != 'open_end']; yr = K.yearly_table(nav, b)
        cf = [p['cagr_full'] for p in paths.values()]; md = [p['maxdd'] for p in paths.values()]
        res[v] = dict(final_nav=float(nav.iloc[-1]), cagr=K.cagr(nav, D0, DE), maxdd=K.dd_info(nav), cagr_since2019=K.cagr(nav, D19, DE), cagr_since2015_07=K.cagr(nav, D157, DE),
                      cagr_2008_2015H1=K.cagr(nav, D0, DH1), n_closed=int(len(x)), n_all=int(len(lg)), win_rate=float((x['ret'] > 0).mean()),
                      final_nav_0050=float(b.iloc[-1]), cagr_0050=K.cagr(b, D0, DE), maxdd_0050=K.dd_info(b), cagr_0050_since2019=K.cagr(b, D19, DE),
                      cagr_0050_since2015_07=K.cagr(b, D157, DE), cagr_0050_2008_2015H1=K.cagr(b, D0, DH1),
                      yearly=yr, years_win=int(sum(y['strat'] > y['e0050'] for y in yr)), years_lose=int(sum(y['strat'] <= y['e0050'] for y in yr)),
                      years_lost=[y['year'] for y in yr if y['strat'] <= y['e0050']], signals=sig,
                      paths20=dict(cagr_full_median=float(np.median(cf)), cagr_full_min=float(min(cf)), cagr_full_max=float(max(cf)), maxdd_median=float(np.median(md)), maxdd_worst=float(min(md))),
                      paths=paths, limit_counts_std=stc, list_open_at_7pct_limup_pre_cut=locks, top5pct=top_share(lg, float(nav.iloc[-1])))
        TR.append(lg.drop(columns=[c for c in ('j',) if c in lg.columns]).assign(version=v))
        print(sc, v, {k: (round(z, 6) if isinstance(z, float) else z) for k, z in res[v].items() if k in ('final_nav', 'cagr', 'maxdd', 'n_closed', 'n_all', 'cagr_2008_2015H1', 'cagr_since2019', 'paths20', 'limit_counts_std')},
              f'{time.time() - t0:.0f}s', rss(), 'MB', flush=True)
        del std, F, RT; gc.collect()
    df = pd.DataFrame({'V_MASK': NAV['V_MASK'], 'V_BRIDGE': NAV['V_BRIDGE'], '0050': bench.reindex(NAV['V_MASK'].index)}).round(2); df.index.name = 'date'
    return res, df, pd.concat(TR), PN, NAV


def trades_equal(a, b):
    """a＝本次（DataFrame）、b＝既有檔讀回：筆數、每筆的訊號／進場／出場列、代號、股數、原因相同，成本與賣得金額差 < 1e-6。"""
    if len(a) != len(b):
        return False
    ok = all((a[c].to_numpy() == b[c].to_numpy()).all() for c in ('sig_t', 'ent_t', 'exit_t', 'shares', 'why')) and (a['code'].astype(str).to_numpy() == b['code'].astype(str).to_numpy()).all()
    return bool(ok and np.allclose(a['cost'].to_numpy(float), b['cost'].to_numpy(float), rtol=0, atol=1e-6) and np.allclose(a['proceeds'].to_numpy(float), b['proceeds'].to_numpy(float), rtol=0, atol=1e-6))


OUT = {}
# ---- 情境 v2：必須逐項重現既有 v2
res, df, tr, pn, navf = run_scen('v2'); OUT['v2'] = dict(res=res, df=df, tr=tr, pn=pn, navf=navf)
ref_nav = pd.read_csv(V2 / 'nav_2008_2026_v2.csv.gz', index_col=0, parse_dates=True)
ref_tr = pd.read_csv(V2 / 'trades_v2.csv.gz', dtype={'code': str})
ref_res = json.loads((V2 / 'results_v2.json').read_text())['versions']
KEYS = ('final_nav', 'cagr', 'cagr_since2019', 'cagr_since2015_07', 'cagr_2008_2015H1', 'n_closed', 'n_all', 'win_rate', 'final_nav_0050', 'cagr_0050', 'cagr_0050_since2019', 'cagr_0050_2008_2015H1')
base_chk = dict(nav_same_shape=bool(df.shape == ref_nav.shape and list(df.columns) == list(ref_nav.columns) and df.index.equals(ref_nav.index)))
base_chk['nav_max_abs_diff'] = float((df - ref_nav).abs().max().max()) if base_chk['nav_same_shape'] else None
base_chk['trades_same'] = {v: trades_equal(tr[tr['version'] == v], ref_tr[ref_tr['version'] == v]) for v in K.VERSIONS}
base_chk['results_same'] = {v: bool(all(res[v][k] == ref_res[v][k] for k in KEYS) and res[v]['maxdd'] == ref_res[v]['maxdd'] and res[v]['maxdd_0050'] == ref_res[v]['maxdd_0050']
                                    and res[v]['paths20'] == ref_res[v]['paths20'] and [(y['year'], y['strat'], y['e0050']) for y in res[v]['yearly']] == [(y['year'], y['strat'], y['e0050']) for y in ref_res[v]['yearly']])
                            for v in K.VERSIONS}
base_chk['pass'] = bool(base_chk['nav_same_shape'] and base_chk['nav_max_abs_diff'] == 0 and all(base_chk['trades_same'].values()) and all(base_chk['results_same'].values()))
print('v2 基準重現', base_chk, flush=True)
if not base_chk['pass']:
    print('v2 基準重現失敗：中止，不寫 v3 檔'); sys.exit(2)
del ref_tr; gc.collect()
for sc in ('lim7', 'v3'):
    res, df, tr, pn, navf = run_scen(sc); OUT[sc] = dict(res=res, df=df, tr=tr, pn=pn, navf=navf)
P.LIMUP, P.LIMDN = LU10, LD10

# ---------------------------------------------------------------- 驗收：對 R97（V_BRIDGE 標準路徑）
r97 = json.loads((R97 / 'results.json').read_text())['修正後']
r97_nav = pd.read_csv(R97 / 'nav_lim7.csv.gz', index_col=0, parse_dates=True)['nav']
r97_tr = pd.read_csv(R97 / 'trades_lim7.csv.gz', dtype={'code': str})
r97_h = json.loads((R97 / 'out_lim7h.json').read_text())
r97_p = json.loads((R97 / 'paths20_lim7.json').read_text())
acc = {}
for sc in ('lim7', 'v3'):
    a = OUT[sc]['res']['V_BRIDGE']; n = OUT[sc]['navf']['V_BRIDGE']; t = OUT[sc]['tr']; t = t[t['version'] == 'V_BRIDGE']
    items = {k: dict(mine=a[k], r97=r97[k], same=bool(a[k] == r97[k] if isinstance(a[k], int) else abs(a[k] - r97[k]) < 1e-9))
             for k in ('cagr', 'cagr_2008_2015H1', 'cagr_since2019', 'final_nav', 'n_closed', 'n_all', 'win_rate', 'years_win', 'years_lose')}
    items['cagr_since2015_07'] = dict(mine=a['cagr_since2015_07'], r97=r97['cagr_since2015_07_base0701'], same=bool(abs(a['cagr_since2015_07'] - r97['cagr_since2015_07_base0701']) < 1e-9))
    items['maxdd'] = dict(mine=a['maxdd'], r97=r97['maxdd'], same=bool(a['maxdd'] == r97['maxdd']))
    same_idx = bool(n.index.equals(r97_nav.index))
    acc[sc] = dict(items=items, nav_same_index=same_idx, nav_max_abs_diff_vs_r97_lim7=float((n.round(2) - r97_nav).abs().max()) if same_idx else None, trades_same_as_r97_lim7=trades_equal(t, r97_tr))
    acc[sc]['pass'] = bool(all(x['same'] for x in items.values()) and same_idx and acc[sc]['nav_max_abs_diff_vs_r97_lim7'] == 0 and acc[sc]['trades_same_as_r97_lim7'])
acc['v3_vs_r97_lim7h'] = dict(final_nav=[OUT['v3']['res']['V_BRIDGE']['final_nav'], r97_h['final_nav']], cagr=[OUT['v3']['res']['V_BRIDGE']['cagr'], r97_h['cagr']],
                              same=bool(abs(OUT['v3']['res']['V_BRIDGE']['final_nav'] - r97_h['final_nav']) < 1e-6))
mp = OUT['lim7']['res']['V_BRIDGE']['paths']                    # R97 的 20 條路徑（只有 V_BRIDGE、只有 7%）逐條比
acc['paths20_lim7_vs_r97'] = dict(n_paths=len(mp), same_keys=bool(set(mp) == set(r97_p['paths'])),
                                  max_abs_cagr_diff=float(max(abs(mp[k]['cagr_full'] - r97_p['paths'][k]['cagr_full']) for k in mp)),
                                  max_abs_maxdd_diff=float(max(abs(mp[k]['maxdd'] - r97_p['paths'][k]['maxdd']) for k in mp)),
                                  counts_same=bool(all(mp[k]['blk_limup'] == r97_p['paths'][k]['blk_limup'] and mp[k]['sell_defer_limdn'] == r97_p['paths'][k]['sell_defer_limdn'] for k in mp)),
                                  median_mine=OUT['lim7']['res']['V_BRIDGE']['paths20']['cagr_full_median'], median_r97=r97_p['median'])
acc['paths20_lim7_vs_r97']['pass'] = bool(acc['paths20_lim7_vs_r97']['same_keys'] and acc['paths20_lim7_vs_r97']['max_abs_cagr_diff'] < 1e-12 and acc['paths20_lim7_vs_r97']['counts_same'])
acc['target'] = dict(cagr_pct=23.5017, maxdd_pct=-49.9106, final_nav=52939143, n_closed=619)
a = OUT['v3']['res']['V_BRIDGE']
acc['target_hit'] = bool(round(a['cagr'] * 100, 4) == 23.5017 and round(a['maxdd']['maxdd'] * 100, 4) == -49.9106 and round(a['final_nav']) == 52939143 and a['n_closed'] == 619)
acc['pass'] = bool(acc['lim7']['pass'] and acc['v3']['pass'] and acc['v3_vs_r97_lim7h']['same'] and acc['target_hit'] and acc['paths20_lim7_vs_r97']['pass'])
print('R97 驗收', {k: (v['pass'] if isinstance(v, dict) and 'pass' in v else v) for k, v in acc.items() if k in ('lim7', 'v3', 'paths20_lim7_vs_r97', 'target_hit', 'pass')}, flush=True)
if not acc['pass']:
    (HERE / 'acceptance_failed.json').write_text(json.dumps(acc, ensure_ascii=False, indent=1, default=str))
    print('與 R97 不符：中止，不寫 v3 檔（明細見 acceptance_failed.json）'); sys.exit(3)

# ---------------------------------------------------------------- 3.5% 跌幅的影響（lim7 對 v3；兩版本 × 20 條路徑逐日比）
half = {}
idx = OUT['v3']['navf']['V_BRIDGE'].index
for v in K.VERSIONS:
    hp = {}
    for key in OUT['v3']['pn'][v]:
        a7, a3 = OUT['lim7']['pn'][v][key], OUT['v3']['pn'][v][key]
        dif = np.flatnonzero(a7 != a3)
        hp[key] = dict(n_days_differ=int(len(dif)), first_day_differs=str(idx[dif[0]].date()) if len(dif) else None, max_abs_diff=float(np.abs(a7 - a3).max()),
                       cagr_lim7=OUT['lim7']['res'][v]['paths'][key]['cagr_full'], cagr_v3=OUT['v3']['res'][v]['paths'][key]['cagr_full'],
                       sell_defer_limdn_lim7=OUT['lim7']['res'][v]['paths'][key]['sell_defer_limdn'], sell_defer_limdn_v3=OUT['v3']['res'][v]['paths'][key]['sell_defer_limdn'])
    half[v] = dict(std_path_identical_daily=bool(hp[STD]['n_days_differ'] == 0), n_paths_changed=int(sum(x['n_days_differ'] > 0 for x in hp.values())),
                   paths_changed=[k for k, x in hp.items() if x['n_days_differ'] > 0],
                   paths20_lim7=OUT['lim7']['res'][v]['paths20'], paths20_v3=OUT['v3']['res'][v]['paths20'], by_path=hp)


def held_in_half_window(sc, v):
    """標準路徑在 2008-10-13～10-24 期間有沒有賣出判定：列出這段期間出場的交易，以及出場日開盤對 3.5%／7% 跌停的位置。"""
    t = OUT[sc]['tr']; t = t[(t['version'] == v) & (t['exit_t'] >= h0) & (t['exit_t'] < h1) & (t['why'] != 'open_end')]
    rows = []
    for c, xt in zip(t['code'].astype(str), t['exit_t'].astype(int)):
        j = P.ci[c]
        rows.append(dict(code=c, exit_date=str(P.IDX[xt].date()), open=float(P.O[xt, j]), limdn_3_5=float(LD35[xt, j]), limdn_7=float(LD7[xt, j]), open_at_3_5_limdn=bool(P.O[xt, j] <= LD35[xt, j] + 1e-9)))
    return rows


for v in K.VERSIONS:
    half[v]['std_exits_in_window_lim7'] = held_in_half_window('lim7', v)
    half[v]['std_exits_in_window_v3'] = held_in_half_window('v3', v)

# ---------------------------------------------------------------- 寫檔
A, B = OUT['v2']['res'], OUT['v3']['res']
old_sig = pd.read_csv(S75 / 'signals.csv', dtype=str).fillna('')
out = dict(mode='v3', date='2026-11-25', 說明='v2 口徑 ＋ 漲跌幅照當時制度：2015-06-01 以前 7%；2008-10-13～10-24 跌幅 3.5%；2015-06-01 起 10%（引擎原值）。只在記憶體覆寫 P.LIMUP／P.LIMDN。',
           info=info, v2_base_check=base_chk, r97_acceptance=acc, half_limit_effect=half, versions={})
for v in K.VERSIONS:
    n = dict(B[v]); sig = pd.DataFrame(n.pop('signals')); o = old_sig[old_sig['version'] == v]; same_n = len(o) == len(sig)
    n['signals_vs_S75'] = dict(n_months=int(len(sig)), n_months_old=int(len(o)), sig_dates_same=bool(same_n and list(o['sig_date']) == list(sig['sig_date'])),
                               months_targets_differ=[m for m, a_, b_ in zip(sig['month'], o['targets'], sig['targets']) if a_.split() != b_.split()] if same_n else None)
    n['signals_same_as_v2_run'] = bool(A[v]['signals'] == B[v]['signals'])
    out['versions'][v] = n
df3 = OUT['v3']['df']; df2 = OUT['v2']['df']
df3.to_csv(HERE / 'nav_2008_2026_v3.csv.gz')
OUT['v3']['tr'].to_csv(HERE / 'trades_v3.csv.gz', index=False)
ym, yb = B['V_MASK']['yearly'], B['V_BRIDGE']['yearly']
pd.DataFrame([dict(year=m['year'], V_MASK=m['strat'], V_BRIDGE=q['strat'], e0050=m['e0050']) for m, q in zip(ym, yb)]).to_csv(HERE / 'yearly_v3.csv', index=False)

pct = lambda x: round(x * 100, 4)
cmp = dict(說明='S75 連續帳戶 v3（漲跌幅照當時制度）對 v2（全期 10%）。舊＝本程式 v2 情境（已核對與 ../v2_2026-11-21/ 逐日、逐筆、逐項相同）；新＝v3。0050 與訊號名單兩版相同。',
           limit_override=info['limit_override'], v2_base_check=base_chk, versions={})
for v in K.VERSIONS:
    a, n = A[v], B[v]
    c = {k: dict(舊=a[k], 新=n[k], 差=(n[k] - a[k])) for k in KEYS + ('cagr_0050_since2015_07', 'years_win', 'years_lose')}
    c['maxdd'] = dict(舊=a['maxdd'], 新=n['maxdd'], 差=n['maxdd']['maxdd'] - a['maxdd']['maxdd']); c['maxdd_0050'] = dict(舊=a['maxdd_0050'], 新=n['maxdd_0050'])
    c['years_lost'] = dict(舊=a['years_lost'], 新=n['years_lost'])
    c['paths20'] = dict(舊=a['paths20'], 新=n['paths20'], 差={k: n['paths20'][k] - a['paths20'][k] for k in a['paths20']})
    c['paths'] = {k: dict(舊_cagr=a['paths'][k]['cagr_full'], 新_cagr=n['paths'][k]['cagr_full'], 差_pp=(n['paths'][k]['cagr_full'] - a['paths'][k]['cagr_full']) * 100,
                          舊_maxdd=a['paths'][k]['maxdd'], 新_maxdd=n['paths'][k]['maxdd'], 舊_期末=a['paths'][k]['final_nav'], 新_期末=n['paths'][k]['final_nav'],
                          舊_鎖漲停不買=a['paths'][k]['blk_limup'], 新_鎖漲停不買=n['paths'][k]['blk_limup'], 舊_鎖跌停順延=a['paths'][k]['sell_defer_limdn'], 新_鎖跌停順延=n['paths'][k]['sell_defer_limdn'])
                  for k in a['paths']}
    c['n_paths_cagr_lower'] = int(sum(n['paths'][k]['cagr_full'] < a['paths'][k]['cagr_full'] for k in a['paths']))
    c['yearly'] = [dict(year=x['year'], 舊=round(x['strat'] * 100, 2), 新=round(y['strat'] * 100, 2), 差_pp=round((y['strat'] - x['strat']) * 100, 2), e0050=round(x['e0050'] * 100, 2),
                        舊_贏0050=bool(x['strat'] > x['e0050']), 新_贏0050=bool(y['strat'] > y['e0050'])) for x, y in zip(a['yearly'], n['yearly'])]
    c['limit_counts_std'] = dict(舊=a['limit_counts_std'], 新=n['limit_counts_std'])
    c['list_open_at_7pct_limup_pre_cut'] = n['list_open_at_7pct_limup_pre_cut']
    c['top5pct'] = dict(舊=a['top5pct'], 新=n['top5pct'])
    dd = (df3[v] != df2[v]).to_numpy()
    c['first_day_nav_differs'] = str(df3.index[dd.argmax()].date()) if dd.any() else None
    c['n_days_nav_differs'] = int(dd.sum())
    ta = OUT['v2']['tr']; ta = ta[ta['version'] == v]; tb = OUT['v3']['tr']; tb = tb[tb['version'] == v]
    ka = set(zip(ta['code'].astype(str), ta['ent_t'])); kb = set(zip(tb['code'].astype(str), tb['ent_t']))
    f = lambda t, ks: [dict(code=str(c_), ent_date=str(pd.Timestamp(e).date()), exit_date=str(pd.Timestamp(x).date()), ret=round(float(r), 4), pnl=round(float(p)))
                       for c_, et, e, x, r, p in zip(t['code'], t['ent_t'], t['ent_date'], t['exit_date'], t['ret'], t['pnl']) if (str(c_), et) in ks]
    c['trades_only_in_v2'] = f(ta, ka - kb); c['trades_only_in_v3'] = f(tb, kb - ka)
    c['signals_same'] = out['versions'][v]['signals_same_as_v2_run']
    cmp['versions'][v] = c
cmp['0050_same'] = bool((df3['0050'] == df2['0050']).all())
cmp['half_limit_effect'] = {v: {k: half[v][k] for k in ('std_path_identical_daily', 'n_paths_changed', 'paths_changed', 'paths20_lim7', 'paths20_v3')} for v in half}

# ---- 對外數字對照表（對外用 V_BRIDGE；V_MASK 並列備查）
def ext(v):
    a, n = A[v], B[v]
    row = lambda name, x, y, unit: dict(項目=name, v2舊值=x, v3新值=y, 單位=unit)
    ta, tn = a['top5pct'], n['top5pct']
    return [
        row('全期年化（2008-01-02～2026-10-22，100 萬、位移 0）', pct(a['cagr']), pct(n['cagr']), '%'),
        row('0050 全期年化', pct(a['cagr_0050']), pct(n['cagr_0050']), '%'),
        row('最大回撤（策略）', pct(a['maxdd']['maxdd']), pct(n['maxdd']['maxdd']), '%'),
        row('最大回撤期間（策略：高點→低點→收復）', f"{a['maxdd']['peak']}→{a['maxdd']['trough']}→{a['maxdd']['recovered']}", f"{n['maxdd']['peak']}→{n['maxdd']['trough']}→{n['maxdd']['recovered']}", '日期'),
        row('最大回撤（0050）', pct(a['maxdd_0050']['maxdd']), pct(n['maxdd_0050']['maxdd']), '%'),
        row('100 萬期末金額（策略）', round(a['final_nav']), round(n['final_nav']), '元'),
        row('100 萬期末金額（0050）', round(a['final_nav_0050']), round(n['final_nav_0050']), '元'),
        row('19 個年度贏 0050／輸 0050', f"{a['years_win']}／{a['years_lose']}", f"{n['years_win']}／{n['years_lose']}", '年'),
        row('輸 0050 的年份', a['years_lost'], n['years_lost'], '年份'),
        row('交易筆數（全部，含期末未平倉）', a['n_all'], n['n_all'], '筆'),
        row('交易筆數（已出場）', a['n_closed'], n['n_closed'], '筆'),
        row('勝率（已出場）', pct(a['win_rate']), pct(n['win_rate']), '%'),
        row('賺錢比例（全部交易，文件「47.8%」的口徑）', pct(ta['win_rate_all']), pct(tn['win_rate_all']), '%'),
        row('每筆報酬中位數（全部交易）', pct(ta['median_ret']), pct(tn['median_ret']), '%'),
        row('2008～2015 年中年化（策略，2008-01-02→2015-06-30）', pct(a['cagr_2008_2015H1']), pct(n['cagr_2008_2015H1']), '%'),
        row('2008～2015 年中年化（0050）', pct(a['cagr_0050_2008_2015H1']), pct(n['cagr_0050_2008_2015H1']), '%'),
        row('連續帳戶 2019 起年化（基底 2018-12-28）', pct(a['cagr_since2019']), pct(n['cagr_since2019']), '%'),
        row('連續帳戶 2015-07 起年化（基底 2015-07-01 收盤）', pct(a['cagr_since2015_07']), pct(n['cagr_since2015_07']), '%'),
        row('20 條路徑全期年化：最少', pct(a['paths20']['cagr_full_min']), pct(n['paths20']['cagr_full_min']), '%'),
        row('20 條路徑全期年化：中位數', pct(a['paths20']['cagr_full_median']), pct(n['paths20']['cagr_full_median']), '%'),
        row('20 條路徑全期年化：最多', pct(a['paths20']['cagr_full_max']), pct(n['paths20']['cagr_full_max']), '%'),
        row('20 條路徑最大回撤：中位數', pct(a['paths20']['maxdd_median']), pct(n['paths20']['maxdd_median']), '%'),
        row('20 條路徑最大回撤：最深', pct(a['paths20']['maxdd_worst']), pct(n['paths20']['maxdd_worst']), '%'),
        row('前 5% 交易：筆數（ceil(n×5%)，R82 寫法）', ta['ceil']['n_top'], tn['ceil']['n_top'], '筆'),
        row('前 5% 交易：門檻（每筆報酬 ≥）', pct(ta['ceil']['threshold_ret']), pct(tn['ceil']['threshold_ret']), '%'),
        row('前 5% 交易：占選股交易損益', pct(ta['ceil']['share_of_trade_pnl']), pct(tn['ceil']['share_of_trade_pnl']), '%'),
        row('前 5% 交易：占帳戶總獲利（期末−100 萬）', pct(ta['ceil']['share_of_nav_gain']), pct(tn['ceil']['share_of_nav_gain']), '%'),
        row('前 5% 交易（四捨五入 round(n×5%)，S82 寫法）：筆數／門檻／占選股損益', f"{ta['round']['n_top']}／{pct(ta['round']['threshold_ret'])}／{pct(ta['round']['share_of_trade_pnl'])}",
            f"{tn['round']['n_top']}／{pct(tn['round']['threshold_ret'])}／{pct(tn['round']['share_of_trade_pnl'])}", '筆／%／%'),
        row('選股交易損益合計', round(ta['pnl_sum_all']), round(tn['pnl_sum_all']), '元'),
    ]


extj = dict(說明='對外文件用的新舊對照。對外數字用 V_BRIDGE（標準路徑：100 萬、位移 0）；V_MASK 並列備查。v2 舊值＝本程式 v2 情境重算（與 ../v2_2026-11-21/ 逐項相同）；v3 新值＝漲跌幅照當時制度。'
                   '注意：文件現有的「632 筆、32 筆、≥ +57.6%、七成、47.5%」是 S82／R82 用 S75 原版（v2 之前）算的，與本表 v2 欄可能差在小數。',
            來源=dict(v2='../v2_2026-11-21/results_v2.json、trades_v2.csv.gz', v3='results_v3.json、trades_v3.csv.gz', 程式='run_v3.py'),
            V_BRIDGE=ext('V_BRIDGE'), V_MASK=ext('V_MASK'),
            逐年_V_BRIDGE=cmp['versions']['V_BRIDGE']['yearly'])
out['runtime_s'] = round(time.time() - t0, 1); out['maxrss_mb'] = rss()
cmp['runtime_s'] = out['runtime_s']; cmp['maxrss_mb'] = out['maxrss_mb']
(HERE / 'results_v3.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
(HERE / 'compare_v2_v3.json').write_text(json.dumps(cmp, ensure_ascii=False, indent=1, default=str))
(HERE / 'external_numbers_v3.json').write_text(json.dumps(extj, ensure_ascii=False, indent=1, default=str))
print('done', out['runtime_s'], 's', out['maxrss_mb'], 'MB')
