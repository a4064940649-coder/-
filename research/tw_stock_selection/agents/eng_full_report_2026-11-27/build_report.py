#!/usr/bin/env python3
"""SURGE 月營收驚奇 — 完整報告產生（工程 ENG-RPT，2026-11-27）

執行: python3 -I -B build_report.py        （約 2～3 分鐘、峰值記憶體約 1.3GB）
輸出: SURGE月營收驚奇_完整報告.html（同資料夾；另複製到 xq_scripts/）、results.json

依第三方「策略完整報告 — 框架規格書」產出；樣本＝2008-01-02～2026-10-22 連續帳戶 v3（V_BRIDGE、100 萬、位移 0）。
只算、只寫報告：不改任何選股／進出場／配置規則，不改任何既有檔，不上網抓資料。
前提：連續面板快取已存在（沒有就先跑 ../s75_continuous_2008_2026/v3_2026-11-25_limit7/rebuild_cache.py；
      再沒有就先 bash research/tw_stock_selection/restore_env.sh）。需要 plotly（pip install plotly）。

本檔結構：
  A. 引擎包裝（本檔自寫）：照 run_v3.py 的設定載入面板、覆寫漲跌停、跑標準路徑與 16 組敏感度網格。
  B. 規格書元件（第三、四、五、六、七、七點五節）：CSS、5 個檢定函數、圖表函數、蒙地卡羅、滾動報酬、perf_summary
     ——由規格書原文逐字複製（標記「規格書原文」的區塊請勿修改）。
  C. 主流程（本檔自寫）：欄位對應（改編）、14 項檢定、Phase 1 擴充表、HTML 組裝。
"""
import sys, math, json, time, resource, gc, datetime, shutil, html as _html
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S75 = AG / 's75_continuous_2008_2026'
V2 = S75 / 'v2_2026-11-21'
V3 = S75 / 'v3_2026-11-25_limit7'
REPO = HERE.parents[3]
OUT_NAME = 'SURGE月營收驚奇_完整報告.html'
OUT_HTML = HERE / OUT_NAME
OUT_COPY = REPO / 'xq_scripts' / OUT_NAME
OUT_JSON = HERE / 'results.json'
PLOTLY_CDN = 'https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js'   # 改編：規格為 cdn.plot.ly/plotly-2.35.2.min.js（同版本）
CAPITAL = 1_000_000          # 固定本金口徑
N_TRIALS_OURS = 4040         # 我們實際試驗次數（BACKLOG.md 2026-11-27 S100：累計約 4,040）
GRID_N = (5, 10, 15, 20)     # 買進名次 N（基準 10）
GRID_MOM = (0.0, 0.05, 0.10, 0.15)   # 前 20 日漲幅門檻（基準 10%）
BASE = dict(cagr_pct=23.50, maxdd_pct=-49.91, trough='2008-11-20', n_trades=627, cagr_0050_pct=14.54, final_nav=52939143)
T0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


# ============================================================
# A. 引擎包裝（設定逐行同 run_v3.py；只在記憶體覆寫 P.LIMUP／P.LIMDN）
# ============================================================
def run_engine():
    sys.path.insert(0, str(S75))
    import s75lib as K
    from s75lib import cslib, s10lib
    cache = K.TMP / 'cache_cont'
    if not cache.exists():
        print(f'缺連續面板快取 {cache}：先跑 {V3 / "rebuild_cache.py"}（必要時先 restore_env.sh）'); sys.exit(4)
    assert cslib.REDUCE_FIX is False and cslib.E50_SOURCE is None
    old = pd.read_csv(cslib.E50_CSV, parse_dates=['date']).set_index('date')['adj'].sort_index()
    new = pd.read_csv(cslib.E50_OFFICIAL_CSV, parse_dates=['date']).set_index('date')['adj'].sort_index()
    seam = new.index[0]; assert str(seam.date()) == '2015-01-05' and seam in old.index
    sp = pd.concat([old.loc[:seam].iloc[:-1] * float(new.loc[seam] / old.loc[seam]), new])
    ex = pd.read_csv(V2 / 'e50_spliced.csv', parse_dates=['date']).set_index('date')['adj']
    assert sp.index.equals(ex.index) and float((sp - ex).abs().max()) < 1e-9, '0050 銜接檔與 v2 不同（上游 0050 檔已變）'
    cslib.E50_SOURCE = V2 / 'e50_spliced.csv'; cslib.REDUCE_FIX = True
    twii = pd.read_csv(cslib.TWII_CSV, parse_dates=['date']).set_index('date')['close'].sort_index()   # repo 既有的加權指數檔（引擎本來就讀）
    P, U = K.load_cont()
    assert P.reduce_fix and P.exright_fix
    D0, DE = P.IDX[P.t0], P.IDX[-1]
    assert (str(D0.date()), str(DE.date())) == ('2008-01-02', '2026-10-22')

    def limits(ref, up_pct, dn_pct):
        u = ref * (1 + up_pct); t = cslib.tick(u); u = np.floor(u / t + 1e-9) * t
        d = ref * (1 - dn_pct); t = cslib.tick(d); d = np.ceil(d / t - 1e-9) * t
        return u, d
    iv = P.IDX.values
    cut = int(np.searchsorted(iv, np.datetime64('2015-06-01'), side='left')); assert str(P.IDX[cut].date()) == '2015-06-01'
    h0 = int(np.searchsorted(iv, np.datetime64('2008-10-13'), side='left')); h1 = int(np.searchsorted(iv, np.datetime64('2008-10-24'), side='right'))
    LU10, LD10 = P.LIMUP, P.LIMDN
    u7, d7 = limits(P.REF[:cut], 0.07, 0.07)
    LU7 = LU10.copy(); LU7[:cut] = u7; LD7 = LD10.copy(); LD7[:cut] = d7
    _, d35 = limits(P.REF[h0:h1], 0.07, 0.035)
    LD35 = LD7.copy(); LD35[h0:h1] = d35
    P.LIMUP, P.LIMDN = LU7, LD35                                   # v3：2015-06-01 前 7%；2008-10-13～10-24 跌幅 3.5%
    del u7, d7, d35

    F, RT = K.surge(P, 'B2')                                       # V_BRIDGE
    sg = K.signal(P, U, RT, 0)
    nav, lg, st = K.sim(P, sg, U, 1e6)
    bench = pd.Series(P.E50, index=P.IDX).loc[D0:]; bench = bench / bench.iloc[0] * 1e6

    # ---- 基準核對（對不上就停）
    ref_nav = pd.read_csv(V3 / 'nav_2008_2026_v3.csv.gz', index_col=0, parse_dates=True)
    ddi = K.dd_info(nav)
    chk = dict(cagr_pct=round(K.cagr(nav, D0, DE) * 100, 4), maxdd_pct=round(ddi['maxdd'] * 100, 4), trough=ddi['trough'], peak=ddi['peak'], recovered=ddi['recovered'],
               n_trades=int(len(lg)), n_closed=int((lg['why'] != 'open_end').sum()), cagr_0050_pct=round(K.cagr(bench, D0, DE) * 100, 4),
               final_nav=float(nav.iloc[-1]), final_nav_0050=float(bench.iloc[-1]),
               nav_same_index=bool(nav.index.equals(ref_nav.index)))
    chk['nav_max_abs_diff_vs_v3_file'] = float((nav.round(2) - ref_nav['V_BRIDGE']).abs().max()) if chk['nav_same_index'] else None
    chk['bench_max_abs_diff_vs_v3_file'] = float((bench.round(2) - ref_nav['0050']).abs().max()) if chk['nav_same_index'] else None
    chk['pass'] = bool(round(chk['cagr_pct'], 2) == BASE['cagr_pct'] and round(chk['maxdd_pct'], 2) == BASE['maxdd_pct'] and chk['trough'] == BASE['trough']
                       and chk['n_trades'] == BASE['n_trades'] and round(chk['cagr_0050_pct'], 2) == BASE['cagr_0050_pct'] and round(chk['final_nav']) == BASE['final_nav']
                       and chk['nav_same_index'] and chk['nav_max_abs_diff_vs_v3_file'] == 0)
    print('基準核對', chk, flush=True)
    if not chk['pass']:
        print('基準對不上：中止，不產報告'); sys.exit(2)

    # ---- 進場名次（照 s10lib.make_signal_s10 的同一段排序重算；不改引擎）
    S, av = RT['S'], RT['av']; rankpos = {}
    for m in S.index:
        k0 = av[m]
        if k0 >= P.T or k0 < 61:
            continue
        s = S.loc[m].to_numpy()
        if int((U[k0] & np.isfinite(s)).sum()) < 50:
            continue
        k = k0
        s = cslib.ins15_mask_row(P, m, k, s)
        fin = U[k] & np.isfinite(s); ok_ma = fin & P.F_MA60[k]
        o_ma = np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]
        rankpos[int(k)] = {int(j): r + 1 for r, j in enumerate(o_ma[:30])}
    lg = lg.copy()
    lg['entry_rank'] = [rankpos[int(k)][int(j)] for k, j in zip(lg['sig_t'], lg['j'])]
    lg['ret20_sig'] = P.RET20[lg['sig_t'].to_numpy(), lg['j'].to_numpy()]
    assert lg['entry_rank'].between(1, 10).all() and (lg['ret20_sig'] > 0.10).all(), '進場名次重算與引擎不一致'

    # ---- 兩條腿（改編）：0050 底倉（引擎的 B 序列）對 衛星（＝帳戶 − 底倉；含個股與停泊在 0050 的衛星閒置資金）
    t0 = P.t0; E50 = P.E50[t0:]; E50O = P.E50O[t0:]; B = st['B'].to_numpy(float); nv = nav.to_numpy(float)
    base_d = np.full(len(nv), np.nan)
    base_d[1:] = B[:-1] * (E50O[1:] / E50[:-1] - 1) + B[1:] * (1 - E50O[1:] / E50[1:])      # 隔夜（前收→開盤）用前一日底倉；盤中（開盤→收盤）用當日收盤底倉回推
    exec_days = sorted(int(x['k']) + 1 for x in sg['info'])
    eng = dict(nav=nav, bench=bench, lg=lg.drop(columns=['j']), B=st['B'], stockfrac=st['stockfrac'], base_d=pd.Series(base_d, index=nav.index),
               exec_dates=[P.IDX[t] for t in exec_days if t < P.T], sig_dates=[P.IDX[int(x['k'])] for x in sg['info']],
               twii=twii, chk=chk, st={k: (int(v) if isinstance(v, (int, np.integer)) else float(v)) for k, v in st.items() if isinstance(v, (int, float, np.integer, np.floating))},
               D0=D0, DE=DE, yearly=K.yearly_table(nav, bench.reindex(nav.index)))

    # ---- 檢定 10：4×4 敏感度網格（買進名次 N × 前 20 日漲幅門檻；其餘規則、持股上限 10 檔、每檔＝衛星 ÷ 10 都不動）
    grid = []
    for n in GRID_N:
        for mo in GRID_MOM:
            sg2 = s10lib.make_signal_s10(P, U, RT, shift=0, depth=n, mom_lo=mo, nsel=n)
            nav2, lg2, _ = K.sim(P, sg2, U, 1e6)
            j = nav2.pct_change().iloc[1:] * CAPITAL
            row = dict(N=n, mom=mo, sharpe=float(j.mean() / j.std(ddof=1) * math.sqrt(252)), pnl=float(j.sum()), cagr=float(K.cagr(nav2, D0, DE)),
                       maxdd=float(K.dd_info(nav2)['maxdd']), n_trades=int(len(lg2)), final_nav=float(nav2.iloc[-1]))
            if n == 10 and mo == 0.10:
                row['same_as_base'] = bool(np.array_equal(nav2.to_numpy(), nav.to_numpy()))
                assert row['same_as_base'], '網格基準格 (10, 10%) 與標準路徑不同'
            grid.append(row); print('grid', row, f'{time.time() - T0:.0f}s', rss(), 'MB', flush=True)
            del sg2, nav2, lg2; gc.collect()
    eng['grid'] = grid
    P.LIMUP, P.LIMDN = LU10, LD10
    return eng


# ============================================================
# B. 規格書元件（規格書原文，逐字複製；請勿修改）
# ============================================================
# ---- 規格書原文：第三節 UNIFIED_CSS
UNIFIED_CSS = """
* { box-sizing: border-box; }
body { font-family: 'Microsoft JhengHei', 'PingFang TC', sans-serif;
       max-width: 1400px; margin: 30px auto; padding: 0 20px;
       color: #333; line-height: 1.6; background: #f9f9fa; }
.title { font-size: 36px; color: #1f77b4; border-bottom: 4px solid #1f77b4;
         padding-bottom: 14px; margin-bottom: 8px; }
.subtitle { color: #888; font-size: 14px; margin-bottom: 30px; }
.master-toc { background: white; padding: 22px 30px; border-radius: 10px;
              margin: 30px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
.master-toc h2 { margin-top: 0; padding-left: 0; border: 0; color: #1f77b4; font-size: 20px; }
.master-toc ol { columns: 2; column-gap: 30px; padding-left: 20px; font-size: 15px; line-height: 1.9; }
.master-toc a { color: #1f77b4; text-decoration: none; }
.master-toc a:hover { text-decoration: underline; }
.chapter { margin-top: 60px; padding-top: 30px; border-top: 3px dashed #ccd; }
h1.chapter-title { font-size: 30px; color: #1f77b4; border-bottom: 3px solid #1f77b4;
                   padding-bottom: 12px; margin-top: 0; margin-bottom: 25px; }
h1.chapter-title .num { display: inline-block; background: #1f77b4; color: white;
                         padding: 6px 14px; border-radius: 8px; margin-right: 12px;
                         font-size: 22px; vertical-align: middle; }
h2 { font-size: 22px; color: #2c3e50; margin-top: 35px; padding-left: 12px; border-left: 5px solid #1f77b4; }
h3 { font-size: 17px; color: #1f77b4; margin-top: 20px; }
h4 { font-size: 16px; color: #2c3e50; margin-top: 18px; }
.box { background: white; padding: 18px 22px; border-radius: 8px; margin: 15px 0;
       box-shadow: 0 1px 3px rgba(0,0,0,0.06); border-left: 4px solid #1f77b4; }
.warning { background: #fef5f5; border-left: 4px solid #e74c3c; padding: 14px 18px;
           border-radius: 4px; margin: 15px 0; }
.note { background: #f0f7ff; border-left: 4px solid #3498db; padding: 12px 16px;
        border-radius: 4px; margin: 12px 0; font-size: 14px; }
.tip { background: #f0fdf4; border-left: 4px solid #27ae60; padding: 14px 18px;
       border-radius: 4px; margin: 15px 0; font-size: 14px; }
.danger { background: #fef5f5; border: 2px solid #e74c3c; padding: 16px 20px;
          border-radius: 6px; margin: 15px 0; }
.danger h4 { color: #e74c3c; margin-top: 0; }
.caveat { background: #fef5f5; padding: 14px 18px; border-left: 4px solid #E74C3C;
          border-radius: 4px; margin: 15px 0; font-size: 13.5px; }
.executive { background: linear-gradient(135deg, #fff8e7, #fff); padding: 18px 25px;
             border-radius: 8px; margin: 20px 0; border-left: 5px solid #ff9500; }
.executive h2 { margin-top: 0; border-left: none; padding-left: 0; }
.verdict-final { background: linear-gradient(135deg, #1f77b4, #2c3e50); color: white;
                 padding: 25px; border-radius: 10px; margin: 30px 0; }
.verdict-final h2, .verdict-final h3 { color: white; border-left-color: white; }
.verdict-final .label { color: rgba(255,255,255,0.85); }
.verdict-final .metric { background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.25); }
.verdict-final .metric .value { color: white !important; font-size: 32px; }
.verdict-final .metric .label { color: rgba(255,255,255,0.85) !important; }
.callout-success { background: #f0fdf4; border-left: 5px solid #27ae60; padding: 16px 22px;
                   border-radius: 6px; margin: 18px 0; }
.summary-box { display: flex; gap: 15px; margin: 25px 0; }
.summary-box > div { flex: 1; padding: 16px; border-radius: 8px; }
.summary-pass { background: #fadbd8; border-left: 5px solid #E74C3C; }
.summary-warn { background: #fef5e7; border-left: 5px solid #f39c12; }
.summary-fail { background: #d5f5e3; border-left: 5px solid #27ae60; }
.summary-box h4 { margin: 0 0 8px 0; }
.summary-box small { font-size: 12.5px; color: #555; line-height: 1.55; display: block; }
.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 20px 0; }
.metric { background: white; padding: 14px; border-radius: 8px; text-align: center;
          box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.metric .label { font-size: 12px; color: #888; }
.metric .value { font-size: 22px; font-weight: bold; color: #1f77b4; margin-top: 4px; }
.metric .label.sub { font-size: 11px; color: #999; margin-top: 2px; }
.metric.pass .value { color: #E74C3C; }
.metric.warn .value { color: #e67e22; }
.metric.fail .value { color: #27ae60; }
table { width: 100%; border-collapse: collapse; margin: 15px 0; background: white;
        border-radius: 6px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
th { background: #34495e; color: white; padding: 11px 14px; text-align: left; font-size: 14px; }
td { padding: 10px 14px; border-bottom: 1px solid #eee; font-size: 14px; vertical-align: top; }
.win-cell { background: #fadbd8; font-weight: bold; color: #E74C3C; }
.fail-cell { background: #d5f5e3; color: #27ae60; }
.warn-cell { background: #fdebd0; color: #e67e22; font-weight: bold; }
tr.pass td:first-child { border-left: 4px solid #E74C3C; }
tr.warn td:first-child { border-left: 4px solid #f39c12; background: #fffbf0; }
tr.fail td:first-child { border-left: 4px solid #27ae60; background: #f0fdf4; }
.formula { background: #2c3e50; color: #ecf0f1; padding: 16px 22px; border-radius: 8px;
           font-family: Consolas, Menlo, monospace; font-size: 14px; margin: 15px 0;
           line-height: 1.7; overflow-x: auto; }
.formula b { color: #f1c40f; }
.chart-container { background: white; padding: 12px; border-radius: 6px; margin: 18px 0;
                   box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.step { display: flex; gap: 16px; margin: 15px 0; align-items: flex-start; }
.step-num { background: #1f77b4; color: white; width: 32px; height: 32px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center; font-weight: bold;
            flex-shrink: 0; }
.step-content { flex: 1; padding-top: 4px; }
.timeline { display: flex; gap: 14px; margin: 20px 0; flex-wrap: wrap; }
.timeline-step { flex: 1; min-width: 130px; padding: 14px; background: white; border-radius: 8px;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.06); border-top: 4px solid #1f77b4; }
.timeline-step .time { font-size: 17px; font-weight: bold; color: #1f77b4; }
.timeline-step .action { font-size: 13px; color: #555; margin-top: 6px; }
.checklist { background: white; padding: 18px 22px; border-radius: 6px; margin: 15px 0;
             box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.checklist li { margin: 8px 0; line-height: 1.6; }
ul, ol { margin: 8px 0; padding-left: 24px; }
li { margin: 6px 0; }
.back-top { position: fixed; bottom: 30px; right: 30px; background: #1f77b4; color: white;
            padding: 10px 14px; border-radius: 50%; text-decoration: none;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2); font-size: 18px; z-index: 1000; }
.back-top:hover { background: #2c3e50; }
.footer { text-align: center; color: #999; font-size: 12px; margin-top: 50px; padding: 20px;
          border-top: 1px solid #ddd; }
"""


# ---- 規格書原文：第四節 5 個檢定函數
def hac_tstat(x):
    """1. PnL t-test with Newey-West HAC SE."""
    x = pd.Series(x).dropna().astype(float).reset_index(drop=True)
    if len(x) < 2:
        return float('nan'), float('nan')
    X = np.ones(len(x))
    model = sm.OLS(x.values, X).fit(cov_type='HAC',
                                     cov_kwds={'maxlags': int(len(x) ** 0.25)})
    return float(model.tvalues[0]), float(model.pvalues[0])


def sharpe_ci_lobato(j, alpha=0.05):
    """2. Sharpe 95% CI (Lobato adjusted for skew & kurt)."""
    j = pd.Series(j).dropna().astype(float)
    n = len(j)
    if n < 30:
        return float('nan'), float('nan'), float('nan')
    mu = j.mean(); sd = j.std(ddof=1)
    if sd == 0: return float('nan'), float('nan'), float('nan')
    sr_d = mu / sd
    skew = stats.skew(j, bias=False); kurt = stats.kurtosis(j, bias=False)
    var_sr = (1 - skew * sr_d + (kurt) / 4 * sr_d ** 2) / (n - 1)
    se = math.sqrt(max(var_sr, 0))
    z = stats.norm.ppf(1 - alpha / 2)
    sr_ann = sr_d * math.sqrt(252); se_ann = se * math.sqrt(252)
    return sr_ann, sr_ann - z * se_ann, sr_ann + z * se_ann


def psr(j, sr_target=0.0):
    """3. Probabilistic Sharpe Ratio (Bailey-DePrado)."""
    j = pd.Series(j).dropna().astype(float)
    n = len(j)
    if n < 30: return float('nan')
    mu = j.mean(); sd = j.std(ddof=1)
    if sd == 0: return float('nan')
    sr = mu / sd
    skew = stats.skew(j, bias=False); kurt = stats.kurtosis(j, bias=False)
    z = (sr - sr_target) * math.sqrt(n - 1) / math.sqrt(1 - skew * sr + (kurt) / 4 * sr ** 2)
    return float(stats.norm.cdf(z))


def bootstrap_sharpe(j, n_boot=5000, seed=42):
    """4. Bootstrap 5000 次抽樣 Sharpe 分布."""
    j = pd.Series(j).dropna().astype(float).values
    rng = np.random.default_rng(seed)
    n = len(j)
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        x = j[idx]
        sd = x.std(ddof=1)
        out[i] = (x.mean() / sd * math.sqrt(252)) if sd > 0 else 0
    return out


def deflated_sharpe(j, n_trials):
    """14. Deflated Sharpe Ratio (Lopez de Prado)."""
    j = pd.Series(j).dropna().astype(float)
    n = len(j)
    if n < 30: return float('nan'), float('nan'), float('nan')
    mu = j.mean(); sd = j.std(ddof=1)
    sr = mu / sd
    skew = stats.skew(j, bias=False); kurt = stats.kurtosis(j, bias=False)
    if n_trials > 1:
        e_max = math.sqrt(2 * math.log(n_trials)) - (
            (math.log(math.log(max(n_trials, 3))) + math.log(4 * math.pi))
            / (2 * math.sqrt(2 * math.log(n_trials))))
    else:
        e_max = 0
    sr_target_daily = e_max / math.sqrt(n)
    z = (sr - sr_target_daily) * math.sqrt(n - 1) / math.sqrt(
        max(1 - skew * sr + (kurt) / 4 * sr ** 2, 1e-12))
    return float(stats.norm.cdf(z)), e_max, z


# ---- 規格書原文：第五節 嵌入函數與圖表函數
def div(fig):
    return fig.to_html(include_plotlyjs=False, full_html=False,
                       config={'displayModeBar': False, 'responsive': True})


# ① 權益曲線
def chart_equity(res):
    df = res.copy()
    df['cum_long']  = df['combo_long'].fillna(0).cumsum()
    df['cum_short'] = df['combo_short'].fillna(0).cumsum()
    df['cum_combo'] = df['combo_pnl'].fillna(0).cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['cum_combo'], name='合計',
                             line=dict(color='#1f77b4', width=2.5)))
    fig.add_trace(go.Scatter(x=df['date'], y=df['cum_long'], name='多單',
                             line=dict(color='#e74c3c', width=1.5)))
    fig.add_trace(go.Scatter(x=df['date'], y=df['cum_short'], name='空單',
                             line=dict(color='#27ae60', width=1.5)))
    fig.update_layout(title='① 權益曲線 (多 / 空 / 合計)',
                      xaxis_title='', yaxis_title='累積 P&L (元)',
                      hovermode='x unified', height=420,
                      margin=dict(l=70, r=30, t=80, b=40),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02,
                                   xanchor='center', x=0.5))
    return fig


# ② 回撤曲線
def chart_drawdown(res):
    df = res.copy()
    cum = df['combo_pnl'].fillna(0).cumsum()
    dd = cum - cum.cummax()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=dd, fill='tozeroy', name='回撤',
                             line=dict(color='#27ae60', width=1.5)))
    fig.update_layout(title=f'② 回撤曲線 (合計, MDD = {dd.min():,.0f})',
                      xaxis_title='', yaxis_title='Drawdown (元)',
                      height=380, margin=dict(l=70, r=30, t=80, b=40),
                      showlegend=False)
    return fig


# ③ 月度 PnL 熱力 (紅漲綠跌)
def chart_monthly_heatmap(res):
    df = res.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    monthly = df.groupby(['year', 'month'])['combo_pnl'].sum().unstack(fill_value=np.nan)
    monthly = monthly.reindex(columns=range(1, 13))
    z = monthly.values
    text = [[f'{v:,.0f}' if pd.notna(v) else '' for v in row] for row in z]
    custom_scale = [
        [0.0, '#27ae60'],   # 最深綠 (大跌)
        [0.5, '#ffffff'],   # 白 (中性)
        [1.0, '#e74c3c'],   # 最深紅 (大漲)
    ]
    fig = go.Figure(data=go.Heatmap(
        z=z, x=[f'{m}月' for m in range(1, 13)],
        y=[str(y) for y in monthly.index],
        text=text, texttemplate='%{text}', textfont={'size': 10},
        colorscale=custom_scale, zmid=0,
        hovertemplate='%{y} %{x}<br>PnL: %{z:,.0f}<extra></extra>'
    ))
    fig.update_layout(title='③ 月度 PnL 熱力 (合計, 紅漲綠跌)', height=520,
                      margin=dict(l=60, r=20, t=50, b=40))
    return fig


# ④ 年度 PnL Bar
def chart_yearly_bar(res):
    df = res.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    yearly = df.groupby('year')['combo_pnl'].sum()
    colors = ['#27ae60' if v < 0 else '#e74c3c' for v in yearly.values]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=yearly.index, y=yearly.values, marker_color=colors,
                         text=[f'{v:,.0f}' for v in yearly.values], textposition='outside'))
    fig.update_layout(title='④ 年度 PnL', xaxis_title='年', yaxis_title='PnL (元)',
                      height=400, margin=dict(l=60, r=20, t=50, b=40))
    return fig


# ⑤ 月度 PnL Bar
def chart_monthly_bar(res):
    df = res.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['ym'] = df['date'].dt.to_period('M').astype(str)
    monthly = df.groupby('ym')['combo_pnl'].sum()
    colors = ['#27ae60' if v < 0 else '#e74c3c' for v in monthly.values]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly.index, y=monthly.values, marker_color=colors,
                         hovertemplate='%{x}<br>PnL: %{y:,.0f}<extra></extra>'))
    fig.update_layout(title='⑤ 月度 PnL', xaxis_title='', yaxis_title='PnL (元)',
                      height=380, margin=dict(l=60, r=20, t=50, b=70),
                      xaxis=dict(tickangle=-45, tickfont=dict(size=9)))
    return fig


# ⑥ 滾動 252 日 (1 年) Sharpe
def chart_rolling_sharpe(res):
    df = res.copy()
    pnl = df['combo_pnl'].fillna(0)
    rsr = (pnl.rolling(252).mean() / pnl.rolling(252).std(ddof=1)) * math.sqrt(252)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=rsr, name='Rolling 252d (1Y) Sharpe',
                             line=dict(color='#1f77b4', width=1.5)))
    fig.add_hline(y=0, line=dict(color='gray', dash='dash'))
    fig.add_hline(y=1, line=dict(color='#e74c3c', dash='dash'))
    fig.update_layout(title='⑥ 滾動 252 日 (1 年) Sharpe', height=380,
                      yaxis_title='Sharpe', margin=dict(l=60, r=20, t=50, b=40))
    return fig


# ⑦ Sharpe Bootstrap 5000 次分布
def chart_stat_bootstrap(boot, lo, hi):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=boot, nbinsx=60, marker_color='#9b59b6', opacity=0.8))
    fig.add_vline(x=lo, line=dict(color='red', dash='dash'),
                  annotation_text=f'2.5%={lo:.2f}')
    fig.add_vline(x=hi, line=dict(color='red', dash='dash'),
                  annotation_text=f'97.5%={hi:.2f}')
    fig.update_layout(title='⑦ Sharpe Bootstrap 5000 次分布',
                      height=300, margin=dict(l=50, r=20, t=40, b=30))
    return fig


# ⑧ 週日效應
def chart_weekday_effect(res):
    df = res.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['weekday'] = df['date'].dt.weekday
    weekday_names = ['一', '二', '三', '四', '五']
    grp = df.groupby('weekday').agg(
        n=('combo_pnl', lambda s: s.notna().sum()),
        sum=('combo_pnl', 'sum'),
        mean=('combo_pnl', 'mean'),
    )
    grp = grp.loc[range(5)]
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('累計 PnL (按星期)', '平均每筆 PnL (按星期)'))
    fig.add_trace(go.Bar(x=weekday_names, y=grp['sum'].values,
                         marker_color=['#e74c3c' if v >= 0 else '#27ae60' for v in grp['sum']],
                         text=[f'{v:,.0f}' for v in grp['sum']], textposition='outside'),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=weekday_names, y=grp['mean'].values,
                         marker_color=['#e74c3c' if v >= 0 else '#27ae60' for v in grp['mean']],
                         text=[f'{v:,.0f}' for v in grp['mean']], textposition='outside'),
                  row=1, col=2)
    fig.update_layout(title='⑧ 週日效應',
                      height=380, margin=dict(l=60, r=20, t=70, b=40), showlegend=False)
    return fig


# ⑨ 月份效應
def chart_month_effect(res):
    df = res.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    month_names = [f'{m}月' for m in range(1, 13)]
    grp = df.groupby('month').agg(
        n=('combo_pnl', lambda s: s.notna().sum()),
        sum=('combo_pnl', 'sum'),
        mean=('combo_pnl', 'mean'),
    )
    grp = grp.reindex(range(1, 13))
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('累計 PnL (按月份)', '平均每筆 PnL (按月份)'))
    fig.add_trace(go.Bar(x=month_names, y=grp['sum'].values,
                         marker_color=['#e74c3c' if v >= 0 else '#27ae60' for v in grp['sum']],
                         text=[f'{v:,.0f}' for v in grp['sum']], textposition='outside'),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=month_names, y=grp['mean'].values,
                         marker_color=['#e74c3c' if v >= 0 else '#27ae60' for v in grp['mean']],
                         text=[f'{v:,.0f}' for v in grp['mean']], textposition='outside'),
                  row=1, col=2)
    fig.update_layout(title='⑨ 月份效應',
                      height=400, margin=dict(l=60, r=20, t=70, b=40), showlegend=False)
    return fig


# 額外：子期表現 (3.2 章)
def chart_subperiod_performance(sub_results):
    """子期累計 PnL (萬) + 子期 Sharpe 雙 bar subplot.

    Args:
        sub_results: dict, key 為子期名 (例 '2008-2013'), value 為 {'sr': float, 'n': int, 'pnl': float}
    """
    sub_pnl_万 = [v['pnl'] / 10000 for v in sub_results.values()]
    sub_sr = [v['sr'] for v in sub_results.values()]
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('子期累計 PnL (萬)', '子期 Sharpe'))
    fig.add_trace(go.Bar(x=list(sub_results.keys()), y=sub_pnl_万,
                         marker_color=['#e74c3c' if v > 0 else '#27ae60' for v in sub_pnl_万],
                         text=[f'{v:.1f}' for v in sub_pnl_万], textposition='outside'),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=list(sub_results.keys()), y=sub_sr,
                         marker_color=['#e74c3c' if v > 1 else '#e67e22' if v > 0 else '#27ae60'
                                       for v in sub_sr],
                         text=[f'{v:.2f}' for v in sub_sr], textposition='outside'),
                  row=1, col=2)
    fig.update_layout(title=dict(text='統計圖 — 子期表現', y=0.97),
                      height=480, margin=dict(l=50, r=30, t=130, b=40),
                      showlegend=False)
    # subplot 標題往下挪, 數值 label 留空間
    for ann in fig['layout']['annotations']:
        ann['y'] = 1.02
        ann['font'] = dict(size=13)
    fig.update_yaxes(automargin=True)
    # y 軸往上 / 下擴展 20-25% 給文字留空間
    max_pnl, min_pnl = max(sub_pnl_万, default=0), min(sub_pnl_万, default=0)
    fig.update_yaxes(range=[min(0, min_pnl) * 1.2, max(0, max_pnl) * 1.25], row=1, col=1)
    max_sr, min_sr = max(sub_sr, default=0), min(sub_sr, default=0)
    fig.update_yaxes(range=[min(0, min_sr) * 1.2, max(0, max_sr) * 1.25], row=1, col=2)
    return fig


# ---- 規格書原文：第六節 蒙地卡羅
def monte_carlo_bootstrap(pnl_series, n_sims=5000, seed=42, n_path_samples=80):
    """Bootstrap with replacement Monte Carlo simulation."""
    pnl_arr = pnl_series.fillna(0).values
    n_days = len(pnl_arr)
    rng = np.random.default_rng(seed)
    final_pnls = np.empty(n_sims)
    mdds = np.empty(n_sims)
    tuws = np.empty(n_sims)
    sample_paths = []
    for i in range(n_sims):
        idx = rng.integers(0, n_days, size=n_days)
        sampled = pnl_arr[idx]
        cum = np.cumsum(sampled)
        final_pnls[i] = cum[-1]
        dd = cum - np.maximum.accumulate(cum)
        mdds[i] = dd.min()
        tuws[i] = (dd < 0).mean()
        if i < n_path_samples:
            sample_paths.append(cum)
    return {
        'final_pnls': final_pnls,
        'mdds': mdds,
        'tuws': tuws,
        'sample_paths': sample_paths,
    }


# ⑩ 模擬路徑視覺化 (只畫 40 條, 每 10 天抽 1 點 — 控制 HTML 檔大小)
def chart_mc_paths(mc_results, actual_cum, dates):
    fig = go.Figure()
    n_days = len(actual_cum)
    step = 10
    x_sub = list(range(0, n_days, step))
    for path in mc_results['sample_paths'][:40]:
        fig.add_trace(go.Scatter(x=x_sub, y=path[::step],
                                  line=dict(color='rgba(155,89,182,0.12)', width=0.6),
                                  showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x_sub, y=actual_cum[::step],
                              name='實際回測 path',
                              line=dict(color='#e74c3c', width=2.8)))
    fig.update_layout(title='⑩ 40 條蒙地卡羅模擬路徑 vs 實際回測',
                      xaxis_title='交易日 index (每 10 日抽樣)',
                      yaxis_title='累積 PnL (元)',
                      height=450, margin=dict(l=70, r=20, t=70, b=40),
                      hovermode='x unified')
    return fig


# ⑪ MDD 分布
def chart_mc_mdd_dist(mdds, actual_mdd):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=mdds, nbinsx=60, marker_color='#9b59b6',
                                opacity=0.85, name='MC 模擬'))
    fig.add_vline(x=actual_mdd, line=dict(color='#e74c3c', dash='dash', width=2.5),
                   annotation_text=f'實際 MDD = {actual_mdd:,.0f}',
                   annotation_position='top right')
    p5 = float(np.percentile(mdds, 5))
    fig.add_vline(x=p5, line=dict(color='gray', dash='dot'),
                   annotation_text=f'5% 分位 = {p5:,.0f}',
                   annotation_position='top left')
    fig.update_layout(title='⑪ MDD 模擬分布 (5000 次)',
                      xaxis_title='MDD (元)', yaxis_title='出現次數',
                      height=380, margin=dict(l=70, r=20, t=70, b=40))
    return fig


# ⑫ 全期 PnL 分布
def chart_mc_final_dist(final_pnls, actual_final):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=final_pnls, nbinsx=60, marker_color='#3498db',
                                opacity=0.85, name='MC 模擬'))
    fig.add_vline(x=actual_final, line=dict(color='#e74c3c', dash='dash', width=2.5),
                   annotation_text=f'實際 PnL = {actual_final:,.0f}',
                   annotation_position='top right')
    p5 = float(np.percentile(final_pnls, 5))
    p95 = float(np.percentile(final_pnls, 95))
    fig.add_vline(x=p5, line=dict(color='gray', dash='dot'),
                   annotation_text=f'5% = {p5:,.0f}', annotation_position='top left')
    fig.add_vline(x=p95, line=dict(color='gray', dash='dot'),
                   annotation_text=f'95% = {p95:,.0f}', annotation_position='top left')
    fig.update_layout(title='⑫ 全期 PnL 模擬分布 (5000 次)',
                      xaxis_title='全期 PnL (元)', yaxis_title='出現次數',
                      height=380, margin=dict(l=70, r=20, t=70, b=40))
    return fig


# ---- 規格書原文：第七節 滾動報酬
def rolling_returns_analysis(pnl_series, windows=(63, 252, 504)):
    """對每個 window N, 計算所有起始日的 PnL / MDD."""
    pnl = pnl_series.fillna(0).values
    n_days = len(pnl)
    results = {}
    for w in windows:
        if w >= n_days:
            continue
        n_starts = n_days - w
        pnls = np.empty(n_starts)
        mdds = np.empty(n_starts)
        for s in range(n_starts):
            window = pnl[s:s + w]
            cum = np.cumsum(window)
            pnls[s] = cum[-1]
            dd = cum - np.maximum.accumulate(cum)
            mdds[s] = dd.min()
        results[w] = {
            'pnls': pnls, 'mdds': mdds,
            'win_rate': float((pnls > 0).mean()),
            'mean_pnl': float(pnls.mean()),
            'median_pnl': float(np.median(pnls)),
            'p5_pnl': float(np.percentile(pnls, 5)),
            'p25_pnl': float(np.percentile(pnls, 25)),
            'p75_pnl': float(np.percentile(pnls, 75)),
            'p95_pnl': float(np.percentile(pnls, 95)),
            'mean_mdd': float(mdds.mean()),
            'median_mdd': float(np.median(mdds)),
            'p5_mdd': float(np.percentile(mdds, 5)),
            'p95_mdd': float(np.percentile(mdds, 95)),
        }
    return results


# ⑬ PnL 分布 (3 窗口)
def chart_rolling_pnl_dist(rolling_results):
    windows = sorted(rolling_results.keys())
    fig = make_subplots(rows=1, cols=len(windows),
                        subplot_titles=[f'{w} 日 ({"3 個月" if w == 63 else "1 年" if w == 252 else "2 年"}) 持有 PnL' for w in windows])
    color_map = {63: '#3498db', 252: '#9b59b6', 504: '#e74c3c'}
    for i, w in enumerate(windows):
        data = rolling_results[w]['pnls']
        col = color_map.get(w, '#888888')
        fig.add_trace(go.Histogram(x=data, nbinsx=60, marker_color=col,
                                    opacity=0.85, showlegend=False),
                      row=1, col=i + 1)
        fig.add_vline(x=0, line=dict(color='black', dash='dash'), row=1, col=i + 1)
        fig.add_vline(x=rolling_results[w]['median_pnl'],
                      line=dict(color='#e74c3c', dash='dot'),
                      row=1, col=i + 1)
    fig.update_layout(title='⑬ 滾動報酬: 持有 N 日的 PnL 分布',
                      height=380, margin=dict(l=60, r=20, t=80, b=40))
    return fig


# ⑭ MDD 分布 (3 窗口)
def chart_rolling_mdd_dist(rolling_results):
    windows = sorted(rolling_results.keys())
    fig = make_subplots(rows=1, cols=len(windows),
                        subplot_titles=[f'{w} 日 ({"3 個月" if w == 63 else "1 年" if w == 252 else "2 年"}) 持有 MDD' for w in windows])
    color_map = {63: '#3498db', 252: '#9b59b6', 504: '#e74c3c'}
    for i, w in enumerate(windows):
        data = rolling_results[w]['mdds']
        col = color_map.get(w, '#888888')
        fig.add_trace(go.Histogram(x=data, nbinsx=60, marker_color=col,
                                    opacity=0.85, showlegend=False),
                      row=1, col=i + 1)
        fig.add_vline(x=rolling_results[w]['median_mdd'],
                      line=dict(color='#e74c3c', dash='dot'),
                      row=1, col=i + 1)
    fig.update_layout(title='⑭ 滾動報酬: 持有 N 日的 MDD 分布',
                      height=380, margin=dict(l=60, r=20, t=80, b=40))
    return fig


# ⑮ 依起始日 PnL 變化
def chart_rolling_over_time(rolling_results, dates):
    fig = go.Figure()
    color_map = {63: '#3498db', 252: '#9b59b6', 504: '#e74c3c'}
    for w in sorted(rolling_results.keys()):
        n = len(rolling_results[w]['pnls'])
        start_dates = dates[:n]
        fig.add_trace(go.Scatter(x=start_dates, y=rolling_results[w]['pnls'],
                                  mode='lines', name=f'{w} 日 ({"3 個月" if w == 63 else "1 年" if w == 252 else "2 年"}) 持有',
                                  line=dict(color=color_map.get(w, '#888888'), width=1.2),
                                  opacity=0.8))
    fig.add_hline(y=0, line=dict(color='black', dash='dash'))
    fig.update_layout(title='⑮ 依起始日: 持有 N 天 PnL 變化',
                      xaxis_title='起始日期', yaxis_title='持有期結束 PnL (元)',
                      height=420, margin=dict(l=70, r=30, t=80, b=40),
                      hovermode='x unified',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02,
                                   xanchor='center', x=0.5))
    return fig


# ---- 規格書原文：第七點五節 perf_summary
def perf_summary(pnl_series, label='', rf_annual=0.01):
    """核心績效摘要 (給 1.2 章 9 metric 用).
    
    Args:
        pnl_series: pandas.Series, 含每日 PnL (NaN 代表空倉日).
        label: 標籤
        rf_annual: 年化無風險利率 (預設 1%)
    
    Returns:
        dict with keys: label, trades, total_pnl, win_rate, avg_win, avg_loss,
                        pl_ratio, kelly, sharpe, sortino, mdd
    """
    j = pnl_series.dropna().astype(float)
    n = len(j)
    if n == 0:
        return {'label': label, 'trades': 0}
    total = j.sum()
    win_rate = (j > 0).mean()
    avg_win = j[j > 0].mean() if (j > 0).any() else 0
    avg_loss = j[j < 0].mean() if (j < 0).any() else 0
    pl_ratio = -avg_win / avg_loss if avg_loss != 0 else np.nan
    kelly = win_rate - (1 - win_rate) / pl_ratio if pl_ratio and pl_ratio > 0 else np.nan
    sd = j.std(ddof=1)
    sharpe = (j.mean() / sd) * math.sqrt(252) if sd > 0 else np.nan  # 注意: 252 trading days
    downside = j[j < 0].std(ddof=1)
    sortino = (j.mean() / downside) * math.sqrt(252) if downside and downside > 0 else np.nan
    cum = j.cumsum()
    mdd = (cum - cum.cummax()).min()
    return {
        'label': label, 'trades': int(n), 'total_pnl': float(total),
        'win_rate': float(win_rate), 'avg_win': float(avg_win), 'avg_loss': float(avg_loss),
        'pl_ratio': float(pl_ratio) if pl_ratio == pl_ratio else None,
        'kelly': float(kelly) if kelly == kelly else None,
        'sharpe': float(sharpe), 'sortino': float(sortino), 'mdd': float(mdd),
    }


# ============================================================
# C. 主流程（本檔自寫）
# ============================================================
WLAB = {63: '3 個月', 252: '1 年', 504: '2 年'}
NA_RULE = '本策略目前未設此規則，待使用者決定'


def f0(v):
    return f'{v:,.0f}'


def build_res(eng):
    """欄位對應（改編）：固定本金 100 萬、不複利。combo_pnl＝帳戶日報酬 × 1,000,000。
    combo_long → 衛星選股腿、combo_short → 0050 底倉腿（本策略只做多，沒有空單）。"""
    nav = eng['nav']; nv = nav.to_numpy(float)
    ret = nv[1:] / nv[:-1] - 1
    base = eng['base_d'].to_numpy(float)[1:] / nv[:-1] * CAPITAL
    res = pd.DataFrame(dict(date=nav.index[1:], Data=nav.index[1:], combo_pnl=ret * CAPITAL))
    res['combo_short'] = base                       # 改編：0050 底倉 30%
    res['combo_long'] = res['combo_pnl'] - base     # 改編：衛星選股 70%（含個股與停泊在 0050 的衛星閒置資金）
    res['trend_pnl'] = res['combo_pnl']; res['rev_pnl'] = 0.0
    res['nav'] = nv[1:]; res['nav_prev'] = nv[:-1]
    res['bench_ret'] = eng['bench'].reindex(nav.index).pct_change().to_numpy()[1:]
    assert res['combo_pnl'].notna().all() and res['combo_long'].notna().all() and res['combo_short'].notna().all()
    return res


# ---------------------------------------------------------------- Phase 1 擴充（規格書只給行為描述，沒有程式；依描述與範例報告版型自寫）
def year_stats(res):
    df = res.copy(); df['year'] = pd.to_datetime(df['date']).dt.year
    rows = []
    for y, g in df.groupby('year'):
        j = g['combo_pnl']; cum = j.cumsum()
        rows.append(dict(year=int(y), pnl=float(j.sum()), sharpe=float(j.mean() / j.std(ddof=1) * math.sqrt(252)),
                         mdd=float((cum - cum.cummax()).min()), n=int(len(j)), win=float((j > 0).mean())))
    return rows


def build_yearly_html(res, yearly_comp):
    rows = year_stats(res); end = pd.Timestamp(res['date'].iloc[-1])
    comp = {y['year']: y for y in yearly_comp}
    full = [r for r in rows if pd.Timestamp(r['year'], 12, 31) <= end]
    part = [r for r in rows if r not in full]
    best = max(full, key=lambda r: r['pnl']); worst = min(full, key=lambda r: r['pnl'])
    bs = max(full, key=lambda r: r['sharpe']); ws = min(full, key=lambda r: r['sharpe'])
    note = (f'<div class="note"><b>最佳完整年度</b>: {best["year"]} (PnL {best["pnl"]:+,.0f}, Sharpe {best["sharpe"]:.2f})<br>'
            f'<b>最差完整年度</b>: {worst["year"]} (PnL {worst["pnl"]:+,.0f}, Sharpe {worst["sharpe"]:.2f})<br>'
            f'<b>Sharpe 最高完整年</b>: {bs["year"]} (Sharpe {bs["sharpe"]:.2f})<br><b>Sharpe 最低完整年</b>: {ws["year"]} (Sharpe {ws["sharpe"]:.2f})<br>'
            f'<small>* 標記為部分年 ({", ".join(str(r["year"]) + " (" + str(r["n"]) + " 個交易日)" for r in part)}), 不列入最佳/最差比較。'
            f'「複利年報酬」「0050」兩欄為改編加列（連續帳戶年底 ÷ 前一年底 − 1；第一年以 2008-01-02 為基底）。</small></div>')
    tr = ''
    for r in rows:
        sty = ' style="background:#fadbd8;"' if r is best else (' style="background:#d5f5e3;"' if r is worst else '')
        c = comp[r['year']]
        tr += (f'<tr{sty}><td>{r["year"]}{"*" if r in part else ""}</td><td>{r["pnl"]:,.0f}</td><td>{r["sharpe"]:.2f}</td><td>{r["mdd"]:,.0f}</td>'
               f'<td>{r["n"]}</td><td>{r["win"] * 100:.1f}%</td><td>{c["strat"] * 100:+.2f}%</td><td>{c["e0050"] * 100:+.2f}%</td></tr>')
    return ('<p>每年的 PnL / Sharpe / MDD 詳細紀錄 (綠色 = 最差完整年, 紅色 = 最佳完整年)。PnL、MDD 為固定本金 100 萬口徑（元）；勝率＝賺錢交易日比例:</p>' + note +
            '<table><tr><th>年度</th><th>PnL</th><th>Sharpe</th><th>MDD</th><th>交易日</th><th>勝率</th><th>複利年報酬（改編加列）</th><th>0050 含息（改編加列）</th></tr>' + tr + '</table>'), rows


def adv_risk(res):
    j = res['combo_pnl'].astype(float); a = j.to_numpy()
    q = {p: float(np.percentile(a, p)) for p in (1, 5, 25, 50, 75, 95, 99)}
    cum = j.cumsum(); eq = CAPITAL + cum; ddp = eq / eq.cummax() - 1; dd = cum - cum.cummax()
    pos = float(a[a > 0].sum()); neg = float(a[a < 0].sum())
    return dict(var95=q[5], var99=q[1], cvar95=float(a[a <= q[5]].mean()), cvar99=float(a[a <= q[1]].mean()), tail_ratio=float(q[95] / abs(q[5])),
                ulcer=float(math.sqrt((ddp ** 2).mean())), pain=float((-dd).mean()), gpr=float(pos / abs(neg)), pct=q,
                skew=float(stats.skew(a, bias=False)), kurt=float(stats.kurtosis(a, bias=False)))


def build_advanced_risk_html(r):
    gp = r['gpr']; lab = 'outstanding' if gp > 3 else 'excellent' if gp > 2 else 'very good' if gp > 1.5 else 'decent' if gp > 1 else '賠多賺少'
    return f"""<p>除了 Sharpe / MDD, 還有其他風險指標可進一步刻畫策略的風險側寫（固定本金 100 萬口徑, 元 = 日報酬 × 100 萬）:</p>
<h3>4.3.1 VaR / CVaR (Value at Risk / Expected Shortfall)</h3>
<table><tr><th>指標</th><th>數值</th><th>解讀</th></tr>
<tr><td><b>95% VaR (日)</b></td><td>{r['var95']:,.0f}</td><td>每 20 天最壞 1 天的閾值</td></tr>
<tr><td><b>99% VaR (日)</b></td><td>{r['var99']:,.0f}</td><td>每 100 天最壞 1 天的閾值</td></tr>
<tr><td><b>95% CVaR (日)</b></td><td>{r['cvar95']:,.0f}</td><td>落在 5% 最差日子的<b>平均</b>損失</td></tr>
<tr><td><b>99% CVaR (日)</b></td><td>{r['cvar99']:,.0f}</td><td>落在 1% 最差日子的平均損失</td></tr></table>
<div class="note">CVaR (Expected Shortfall) 比 VaR 更嚴格 — 它告訴你「真的進到尾部時平均虧多少」, 而不只是閾值。1 萬元 = 帳戶的 1%。</div>
<h3>4.3.2 其他風險量化指標</h3>
<table><tr><th>指標</th><th>數值</th><th>意涵</th></tr>
<tr><td><b>Tail Ratio</b> (右尾 95% / |左尾 5%|)</td><td>{r['tail_ratio']:.3f}</td><td>&gt; 1 表示獲利尾比虧損尾長 (本策略{'獲利尾較長' if r['tail_ratio'] > 1 else '虧損尾較長'})</td></tr>
<tr><td><b>Ulcer Index</b> (RMS of % DD)</td><td>{r['ulcer'] * 100:.2f}%</td><td>衡量「平均回撤痛苦度」, 越低越好（以 100 萬＋累積 PnL 的權益計）</td></tr>
<tr><td><b>Pain Index</b> (avg DD 深度)</td><td>{r['pain']:,.0f}</td><td>任何時點平均水下金額</td></tr>
<tr><td><b>Gain-to-Pain Ratio</b></td><td>{gp:.2f}</td><td>Schwager 原版: 正日總和 / |負日總和|; &gt;1.0 decent, &gt;1.5 very good, &gt;2.0 excellent, &gt;3.0 outstanding (本策略: {lab})</td></tr></table>
<h3>4.3.3 日 PnL 分布百分位</h3>
<table><tr><th>百分位</th><th>日 PnL</th></tr>{''.join(f'<tr><td>{p}%</td><td>{v:,.0f}</td></tr>' for p, v in r['pct'].items())}</table>
<div class="note">偏度 (Skew) = {r['skew']:.2f} ({'右偏' if r['skew'] > 0 else '左偏 (大賠日比大賺日極端)'}); 超額峰度 (Kurt) = {r['kurt']:.2f} ({'厚尾' if r['kurt'] > 3 else '接近常態'})</div>"""


def streaks(vals, labels):
    """連續為負的區段：回傳 [(長度, 起, 止, 合計)]，由長到短、同長取虧損較大者。"""
    out = []; i = 0; n = len(vals)
    while i < n:
        if vals[i] < 0:
            k = i
            while k + 1 < n and vals[k + 1] < 0:
                k += 1
            out.append((k - i + 1, labels[i], labels[k], float(np.sum(vals[i:k + 1])))); i = k + 1
        else:
            i += 1
    return sorted(out, key=lambda x: (-x[0], x[3]))


def dd_episodes(cum, dates):
    """回撤區段（前高 → 底部 → 回到前高）。cum＝累積值（加法口徑）或淨值（mult=True 時用比例）。"""
    c = np.asarray(cum, float); pk = np.maximum.accumulate(c); uw = c < pk - 1e-9
    out = []; i = 0; n = len(c)
    while i < n:
        if uw[i]:
            k = i
            while k + 1 < n and uw[k + 1]:
                k += 1
            seg = c[i:k + 1]; b = i + int(seg.argmin()); rec = k + 1 if k + 1 < n else None
            out.append(dict(peak=str(pd.Timestamp(dates[i - 1]).date()) if i > 0 else str(pd.Timestamp(dates[0]).date()), trough=str(pd.Timestamp(dates[b]).date()),
                            rec=str(pd.Timestamp(dates[rec]).date()) if rec is not None else None, depth=float(c[b] - pk[b]), depth_pct=float(c[b] / pk[b] - 1) if pk[b] > 0 else None,
                            d_down=int(b - (i - 1)), d_up=int(rec - b) if rec is not None else None))
            i = k + 1
        else:
            i += 1
    return out


def extremes(res, eng):
    j = res['combo_pnl'].to_numpy(float); d = [str(pd.Timestamp(x).date()) for x in res['date']]
    order = np.argsort(j)[:5]
    worst_days = [dict(date=d[i], pnl=float(j[i]), pct=float(j[i] / CAPITAL)) for i in order]
    sd = [dict(n=a, start=b, end=c, pnl=e) for a, b, c, e in streaks(j, d)[:3]]
    m = res.assign(ym=pd.to_datetime(res['date']).dt.to_period('M').astype(str)).groupby('ym')['combo_pnl'].sum()
    wm = [dict(ym=k, pnl=float(v)) for k, v in m.sort_values().head(5).items()]
    sm_ = [dict(n=a, start=b, end=c, pnl=e) for a, b, c, e in streaks(m.to_numpy(), list(m.index))[:3]]
    cum = np.cumsum(j); ep = dd_episodes(np.concatenate([[0.0], cum]), [eng['nav'].index[0]] + list(res['date']))
    top = sorted(ep, key=lambda x: x['depth'])[:5]
    big = [e for e in ep if e['depth'] <= -100000 and e['d_up'] is not None]
    epc = sorted(dd_episodes(eng['nav'].to_numpy(float), list(eng['nav'].index)), key=lambda x: x['depth_pct'])[:5]
    return dict(worst_days=worst_days, loss_day_streaks=sd, worst_months=wm, loss_month_streaks=sm_, dd_top5_fixed=top,
                dd_big_avg=dict(n=len(big), down=float(np.mean([e['d_down'] for e in big])) if big else None, up=float(np.mean([e['d_up'] for e in big])) if big else None),
                dd_top5_compound=epc)


def build_extremes_html(x):
    h = '<p>本表列出樣本內所有極值（固定本金 100 萬口徑，元；1 萬元 = 帳戶 1%）。規格書原意是當作 8.4 三段保護機制觸發條件的歷史依據；<b>本策略目前沒有以這些極值為門檻的減碼或停手規則</b>（見 8.4），這裡只當「歷史上發生過什麼」的紀錄。</p>'
    h += '<h3>7.2.1 單日最大虧損 Top 5</h3><table><tr><th>排名</th><th>日期</th><th>當日 PnL</th><th>占帳戶</th></tr>' + ''.join(
        f'<tr><td>#{i + 1}</td><td>{r["date"]}</td><td>{r["pnl"]:,.0f}</td><td>{r["pct"] * 100:.2f}%</td></tr>' for i, r in enumerate(x['worst_days'])) + '</table>'
    h += '<h3>7.2.2 最長連續虧損次數 Top 3</h3><p style="font-size:13px;color:#666;">改編：本策略每天都有部位，「次數」＝連續虧損的交易日數。</p><table><tr><th>排名</th><th>連虧次數</th><th>期間</th><th>累計虧損</th></tr>' + ''.join(
        f'<tr><td>#{i + 1}</td><td>{r["n"]} 個交易日</td><td>{r["start"]} → {r["end"]}</td><td>{r["pnl"]:,.0f}</td></tr>' for i, r in enumerate(x['loss_day_streaks'])) + '</table>'
    h += '<h3>7.2.3 單月最大虧損 Top 5</h3><table><tr><th>排名</th><th>年-月</th><th>當月 PnL</th></tr>' + ''.join(
        f'<tr><td>#{i + 1}</td><td>{r["ym"]}</td><td>{r["pnl"]:,.0f}</td></tr>' for i, r in enumerate(x['worst_months'])) + '</table>'
    h += '<h3>7.2.4 最長連續虧損月份 Top 3</h3><table><tr><th>排名</th><th>連虧月數</th><th>期間</th><th>累計虧損</th></tr>' + ''.join(
        f'<tr><td>#{i + 1}</td><td>{r["n"]} 月</td><td>{r["start"]} → {r["end"]}</td><td>{r["pnl"]:,.0f}</td></tr>' for i, r in enumerate(x['loss_month_streaks'])) + '</table>'
    h += '<h3>7.2.5 最深 5 個 Drawdown Episodes</h3><table><tr><th>#</th><th>前高日</th><th>底部日</th><th>回復日</th><th>DD 深（固定本金，元）</th><th>高→底 (交易日)</th><th>底→新高 (交易日)</th></tr>' + ''.join(
        f'<tr><td>#{i + 1}</td><td>{r["peak"]}</td><td>{r["trough"]}</td><td>{r["rec"] or "(未回復)"}</td><td>{r["depth"]:,.0f}</td><td>{r["d_down"]}</td><td>{r["d_up"] if r["d_up"] is not None else "N/A"}</td></tr>'
        for i, r in enumerate(x['dd_top5_fixed'])) + '</table>'
    a = x['dd_big_avg']
    if a['n']:
        h += f'<div class="note">有意義 DD (≥ −10 萬，已回復的 {a["n"]} 段) 平均: <b>高→底 {a["down"]:.0f} 個交易日</b>, <b>底→新高 {a["up"]:.0f} 個交易日</b>。</div>'
    h += '<p><b>複利口徑（連續帳戶淨值，改編加列）最深 5 段</b>:</p><table><tr><th>#</th><th>前高日</th><th>底部日</th><th>回復日</th><th>回撤 %</th><th>高→底 (交易日)</th><th>底→新高 (交易日)</th></tr>' + ''.join(
        f'<tr><td>#{i + 1}</td><td>{r["peak"]}</td><td>{r["trough"]}</td><td>{r["rec"] or "(未回復)"}</td><td>{r["depth_pct"] * 100:.2f}%</td><td>{r["d_down"]}</td><td>{r["d_up"] if r["d_up"] is not None else "N/A"}</td></tr>'
        for i, r in enumerate(x['dd_top5_compound'])) + '</table>'
    return h


# ---------------------------------------------------------------- 14 項檢定（依規格書七點五節；改編處逐項註明）
def run_tests(res, eng):
    T = {}
    pnl_combo = res['combo_pnl'].fillna(0)
    pnl_combo_trades = res['combo_pnl'].dropna().astype(float)
    lg = eng['lg']
    # 1～4
    t1, p1 = hac_tstat(pnl_combo_trades)
    sr2, lo2, hi2 = sharpe_ci_lobato(pnl_combo_trades)
    p3 = psr(pnl_combo_trades)
    boot = bootstrap_sharpe(pnl_combo_trades.values, n_boot=5000)
    lo4 = float(np.percentile(boot, 2.5)); hi4 = float(np.percentile(boot, 97.5))
    # 5（改編）：逐筆——進場時 SURGE（score）、進場名次 對 該筆交易報酬 的 Spearman（627 筆，含期末未平倉）
    ic_s = stats.spearmanr(lg['score'], lg['ret']); ic_r = stats.spearmanr(lg['entry_rank'], lg['ret'])
    ic_long = (float(ic_s[0]), float(ic_s[1])); ic_rank = (float(ic_r[0]), float(ic_r[1]))
    # 6
    res['year'] = pd.to_datetime(res['date']).dt.year
    y_min, y_max = res['year'].min(), res['year'].max()
    seg = (y_max - y_min) // 3
    cuts = [(y_min, y_min + seg - 1), (y_min + seg, y_min + 2 * seg - 1), (y_min + 2 * seg, y_max)]
    sub_results = {}
    for lo, hi in cuts:
        mask = (res['year'] >= lo) & (res['year'] <= hi)
        sub = res.loc[mask, 'combo_pnl'].dropna()
        sub_results[f'{lo}-{hi}'] = {'sr': float(sub.mean() / sub.std(ddof=1) * math.sqrt(252)) if len(sub) > 30 else float('nan'), 'n': len(sub), 'pnl': float(sub.sum())}
    # 7
    rsr = (pnl_combo.rolling(252).mean() / pnl_combo.rolling(252).std(ddof=1)) * math.sqrt(252)
    rsr_clean = rsr.dropna(); pct_below_0 = float((rsr_clean < 0).mean()); rsr_mean = float(rsr_clean.mean())
    # 8
    cum = pnl_combo.cumsum(); dd = cum - cum.cummax()
    mdd_v = float(dd.min()); trough_i = int(dd.idxmin()); uw = float((dd < 0).mean())
    mdd_date = str(pd.Timestamp(res['date'].iloc[trough_i]).date())
    # 9（改編）：衛星選股腿 對 0050 底倉腿
    s9_long = perf_summary(res['combo_long'], '衛星選股 70%'); s9_short = perf_summary(res['combo_short'], '0050 底倉 30%')
    # 10（改編）：v3 引擎 2008～2026 重跑 16 組
    rob_df = pd.DataFrame(eng['grid'])
    high_sr_pct = float((rob_df['sharpe'] > 1.0).mean()); rob_best = float(rob_df['sharpe'].max())
    # 11：日 PnL 版（規格）＋逐筆版（改編加列）
    j_arr = pnl_combo_trades.values; j_sorted = np.sort(j_arr)[::-1]
    top5_pct = max(1, int(len(j_arr) * 0.05)); top1_pct = max(1, int(len(j_arr) * 0.01))
    contrib = float(j_sorted[:top5_pct].sum() / j_arr.sum()); contrib_1 = float(j_sorted[:top1_pct].sum() / j_arr.sum())
    skew_v = float(stats.skew(j_arr, bias=False)); kurt_v = float(stats.kurtosis(j_arr, bias=False))
    r_ = lg['ret'].to_numpy(float); pn = lg['pnl'].to_numpy(float); m5 = int(math.ceil(len(r_) * 0.05)); od = np.argsort(-r_, kind='stable')[:m5]   # 同 run_v3.py top_share 的 ceil 寫法
    trade_top = dict(n_all=int(len(r_)), n_top=m5, threshold_ret=float(r_[od[-1]]), share_of_trade_pnl=float(pn[od].sum() / pn.sum()),
                     share_of_nav_gain=float(pn[od].sum() / (eng['chk']['final_nav'] - 1e6)), pnl_sum_all=float(pn.sum()))
    # 12：結算日（每月第三個週三）＋ 換股執行日（改編加列）
    dts = pd.to_datetime(res['date']); wd = dts.dt.weekday; day = dts.dt.day
    is_3rd_wed = (wd == 2) & (day >= 15) & (day <= 21)
    pnl_total = float(res['combo_pnl'].sum()); pnl_settle = float(res.loc[is_3rd_wed, 'combo_pnl'].sum())
    next_set = {d.date() + datetime.timedelta(days=1) for d in dts[is_3rd_wed]}
    is_post = dts.dt.date.apply(lambda d: d in next_set)
    pnl_post = float(res.loc[is_post, 'combo_pnl'].sum())
    settle_pct = pnl_settle / pnl_total; post_pct = pnl_post / pnl_total
    is_exec = dts.isin(eng['exec_dates']); is_sig = dts.isin(eng['sig_dates'])
    pnl_exec = float(res.loc[is_exec, 'combo_pnl'].sum()); pnl_sig = float(res.loc[is_sig, 'combo_pnl'].sum())
    # 13：CAPM α（vs TAIEX；repo 既有的加權指數檔＝價格指數，不含息）；另加列 vs 0050 含息
    equity = CAPITAL + pnl_combo.cumsum()
    strat_ret = equity.pct_change().fillna(0).values

    def capm(bm_ret):
        X = sm.add_constant(bm_ret); c = sm.OLS(strat_ret, X).fit()
        return dict(alpha_a=float((1 + c.params[0]) ** 252 - 1), t_alpha=float(c.tvalues[0]), beta=float(c.params[1]), r2=float(c.rsquared))
    tw = eng['twii']; twr = tw.pct_change().reindex(pd.DatetimeIndex(res['Data']))
    capm_tw = capm(twr.fillna(0).values); capm_tw['bm_missing_days'] = int(twr.isna().sum())
    capm_50 = capm(pd.Series(res['bench_ret']).fillna(0).values)
    # 14：DSR
    dsr_rows = []
    for n_t in (10, 50, 200, 1000, 4500):
        prob, e_max, z = deflated_sharpe(pnl_combo_trades, n_t)
        dsr_rows.append({'n_trials': n_t, 'prob': prob, 'z': z, 'e_max': e_max})
    dsr_n200_prob = dsr_rows[2]['prob']; dsr_n4500_prob = dsr_rows[4]['prob']
    prob, e_max, z = deflated_sharpe(pnl_combo_trades, N_TRIALS_OURS)
    dsr_ours = {'n_trials': N_TRIALS_OURS, 'prob': prob, 'z': z, 'e_max': e_max}

    summary_rows = [
        ('1. PnL t-test (HAC)', f't={t1:.3f}, p={p1:.6f}', '✓ 顯著' if p1 < 0.05 else '✗ 未顯著'),
        ('2. Sharpe 95% CI (Lobato)', f'[{lo2:.3f}, {hi2:.3f}]', '✓ 全為正' if lo2 > 0 else '✗'),
        ('3. PSR (Bailey-DeP)', f'P(SR>0) = {p3:.2%}', '✓' if p3 > 0.95 else '⚠'),
        ('4. Bootstrap 5000', f'95% CI [{lo4:.3f}, {hi4:.3f}]', '✓ 穩定為正' if lo4 > 0 else '⚠'),
        ('5. 訊號 IC（改編：逐筆）', f"SURGE ρ={ic_long[0]:.3f} (p={ic_long[1]:.3f}); 進場名次 ρ={ic_rank[0]:.3f} (p={ic_rank[1]:.3f})；{len(lg)} 筆",
         '△' if (ic_long[1] < 0.05 or ic_rank[1] < 0.05) else '⚠'),
        ('6. 子期穩定性', ', '.join(f"{k} SR={v['sr']:.2f}" for k, v in sub_results.items()),
         '✓ 三段皆顯著正' if all(v['sr'] > 0 for v in sub_results.values()) else '⚠'),
        ('7. Rolling 252d SR', f"平均 {rsr_mean:.2f}, 期間 < 0 比例 {pct_below_0:.1%}", '✓' if pct_below_0 < 0.15 else '⚠'),
        ('8. Drawdown', f"MDD = {mdd_v:,.0f} ({mdd_date}), 在水下 {uw:.1%}", '✓' if uw < 0.85 else '⚠'),
        ('9. 多 / 空 拆解（改編：衛星 / 底倉）', f"衛星 SR={s9_long['sharpe']:.2f}, 底倉 SR={s9_short['sharpe']:.2f}",
         '△' if min(s9_long['sharpe'], s9_short['sharpe']) > 0.5 else '⚠'),
        ('10. 參數 robustness', f"SR>1 比例 {high_sr_pct:.1%}, best={rob_best:.2f}", '✓' if high_sr_pct > 0.5 else '⚠'),
        ('11. 交易集中度', f"Top 5% 貢獻 {contrib:.1%}, skew={skew_v:.2f}, kurt={kurt_v:.2f}；逐筆版（改編加列）前 5%＝{trade_top['n_top']} 筆占選股損益 {trade_top['share_of_trade_pnl']:.2%}",
         '△' if abs(contrib) > 0.5 else '✓'),
        ('12. 結算日污染', f"結算日 {settle_pct:.2%}, 結算後首日 {post_pct:.2%}；換股執行日（改編加列）{pnl_exec / pnl_total:.2%}", '✓' if abs(settle_pct) < 0.20 else '⚠'),
        ('13. CAPM α (vs TAIEX)', f"α={capm_tw['alpha_a']:.1%}/年 (t={capm_tw['t_alpha']:.2f}), β={capm_tw['beta']:.3f}, R²={capm_tw['r2']:.1%}", '✓' if (capm_tw['t_alpha'] > 1.96) else '⚠'),
        ('14. DSR (過擬合)', f"N=10 {dsr_rows[0]['prob']:.0%}, N=200 {dsr_rows[2]['prob']:.0%}, N=1000 {dsr_rows[3]['prob']:.0%}, N=4500 {dsr_rows[4]['prob']:.0%}；我們實際 N={N_TRIALS_OURS:,}：{dsr_ours['prob']:.0%}",
         '✓' if dsr_n4500_prob > 0.95 else '⚠'),
    ]
    passes = sum(1 for r in summary_rows if r[2].startswith('✓'))
    warns = sum(1 for r in summary_rows if r[2].startswith('⚠'))
    triangs = sum(1 for r in summary_rows if r[2].startswith('△'))
    crosses = sum(1 for r in summary_rows if r[2].startswith('✗'))

    # verdict-final（規格書七點五節 3-tier，邏輯不動）
    pnl_wan = float(pnl_combo.sum() / 10000)
    economic_verdict = '✓ 強' if pnl_wan > 100 else ('△ 中' if pnl_wan > 50 else '⚠ 弱')
    overfit_verdict = '✓ 低' if dsr_n200_prob > 0.9 else ('△ 中' if dsr_n200_prob > 0.7 else '⚠ 高')
    mid_period_sr = list(sub_results.values())[1]['sr']
    stability_verdict = '✓ 良好' if mid_period_sr > 0.5 else '△ 中等'
    if passes >= 12 and warns <= 2:
        verdict_strength = '✓ 強'; tier = 1
        verdict_para = '<b>結論</b>: 本策略 <span style="color:#E74C3C;font-weight:bold;">具備統計上顯著的 alpha</span>, 通過嚴謹檢定。建議補上 OOS 紙上交易後可分批上線。'
    elif passes >= 8:
        verdict_strength = '✓ 顯著'; tier = 2
        verdict_para = f'<b>結論</b>: 本策略 <span style="color:#E74C3C;font-weight:bold;">具備統計上顯著的 alpha</span>, 但有 {warns + triangs} 項警訊 (見 3.1 表)。建議延長 OOS 觀察期, 並針對警訊調整出場機制後再分批上線。'
    else:
        verdict_strength = '⚠ 弱'; tier = 3
        verdict_para = '<b>結論</b>: 本策略統計顯著性不足。不建議實單, 需重新檢視訊號設計或延長樣本後再評估。'

    T.update(t1=t1, p1=p1, sr2=sr2, lo2=lo2, hi2=hi2, p3=p3, boot=boot, lo4=lo4, hi4=hi4, ic_long=ic_long, ic_rank=ic_rank, sub_results=sub_results,
             rsr_mean=rsr_mean, pct_below_0=pct_below_0, mdd_v=mdd_v, mdd_date=mdd_date, uw=uw, s9_long=s9_long, s9_short=s9_short, rob_df=rob_df,
             high_sr_pct=high_sr_pct, rob_best=rob_best, contrib=contrib, contrib_1=contrib_1, skew_v=skew_v, kurt_v=kurt_v, trade_top=trade_top,
             pnl_total=pnl_total, pnl_settle=pnl_settle, pnl_post=pnl_post, settle_pct=settle_pct, post_pct=post_pct, n_settle=int(is_3rd_wed.sum()), n_post=int(is_post.sum()),
             pnl_exec=pnl_exec, exec_pct=pnl_exec / pnl_total, n_exec=int(is_exec.sum()), pnl_sig=pnl_sig, sig_pct=pnl_sig / pnl_total, n_sig=int(is_sig.sum()),
             capm_tw=capm_tw, capm_50=capm_50, dsr_rows=dsr_rows, dsr_ours=dsr_ours, dsr_n200_prob=dsr_n200_prob, dsr_n4500_prob=dsr_n4500_prob,
             summary_rows=summary_rows, passes=passes, warns=warns, triangs=triangs, crosses=crosses, pnl_wan=pnl_wan,
             verdict_strength=verdict_strength, economic_verdict=economic_verdict, overfit_verdict=overfit_verdict, stability_verdict=stability_verdict,
             mid_period_sr=mid_period_sr, verdict_para=verdict_para, tier=tier, cum=cum, pnl_combo=pnl_combo)
    return T


def jclean(o):
    if isinstance(o, dict):
        return {str(k): jclean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jclean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime.date)):
        return str(pd.Timestamp(o).date())
    return o


def cell_class(judge):
    if judge.startswith('✓'):
        return 'pass'
    if judge.startswith('⚠'):
        return 'warn'
    return 'warn'  # △ 也用 warn 樣式（規格書原文）



# ---- 修正（2026-11-27，使用者回報部分圖空白）：
# 1) plotly.js 2.35.2 直接內嵌（不依賴 CDN，離線／擋外部腳本的檢視器也能畫）
# 2) 每張圖進入畫面才繪製，並 try/catch；失敗時在圖框內顯示錯誤訊息，不再一片空白
# 3) 日期字串去掉 T00:00:00.000000000，減少檔案大小
PLOTLY_LOCAL = Path(__file__).resolve().parent / 'plotly-2.35.2.min.js'
_LAZY_JS = """<script>
(function(){
  var q=[];
  function draw(a){
    try{ var r=Plotly.newPlot.apply(Plotly,a); if(r&&r.catch){r.catch(function(e){fail(a[0],e);});} }
    catch(e){ fail(a[0],e); }
  }
  function fail(id,e){ var d=document.getElementById(id); if(d){ d.innerHTML='<div style="padding:20px;color:#e74c3c">圖表載入失敗：'+(e&&e.message?e.message:e)+'（請改用電腦版 Chrome／Edge 開啟）</div>'; } }
  window.__plot=function(){
    var a=arguments, el=document.getElementById(a[0]);
    if(!window.Plotly){ fail(a[0],'Plotly 未載入'); return; }
    if(!el||!('IntersectionObserver' in window)){ draw(a); return; }
    var io=new IntersectionObserver(function(es){ es.forEach(function(x){ if(x.isIntersecting){ io.disconnect(); draw(a); } }); },{rootMargin:'600px 0px'});
    io.observe(el);
  };
})();
</script>"""

def harden_html(html):
    import re
    js = PLOTLY_LOCAL.read_text(encoding='utf-8').replace('\ufffd', '\\ufffd')   # 4 處皆在 JS 字串常值內，跳脫後等價
    tag = f'<script src="{PLOTLY_CDN}"></script>'
    assert html.count(tag) == 1
    n = html.count('Plotly.newPlot(')
    assert n == 16, n
    html = html.replace('Plotly.newPlot(', 'window.__plot(')
    html = html.replace('T00:00:00.000000000', '')
    html = html.replace(tag, '<script>' + js.replace('</script', '<\\/script') + '</script>' + _LAZY_JS)
    return html

def main(eng=None):
    eng = eng or run_engine()
    res = build_res(eng); lg = eng['lg']; chk = eng['chk']
    n_days = len(res)
    start_date = str(pd.Timestamp(eng['nav'].index[0]).date()); end_date = str(pd.Timestamp(res['date'].iloc[-1]).date())
    n_years = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.25
    print('Running 14 statistical tests...', flush=True)
    T = run_tests(res, eng)
    pnl_combo = T['pnl_combo']; cum = T['cum']
    ps = perf_summary(res['combo_pnl'], '合計')
    total_pnl = float(pnl_combo.sum()); pnl_wan = total_pnl / 10000; ann_wan = total_pnl / n_years / 10000
    mdd_v = T['mdd_v']; mdd_wan = abs(mdd_v) / 10000; uw = T['uw']; calmar = ann_wan / mdd_wan
    closed = lg[lg['why'] != 'open_end']; wr_closed = float((closed['ret'] > 0).mean()); wr_all = float((lg['ret'] > 0).mean())
    tw_ = lg[lg['ret'] > 0]['ret']; tl_ = lg[lg['ret'] <= 0]['ret']
    trade = dict(n_all=int(len(lg)), n_closed=int(len(closed)), n_open_end=int(len(lg) - len(closed)), win_rate_closed=wr_closed, win_rate_all=wr_all,
                 mean_ret=float(lg['ret'].mean()), median_ret=float(lg['ret'].median()), avg_win_ret=float(tw_.mean()), avg_loss_ret=float(tl_.mean()),
                 pl_ratio_trade=float(-tw_.mean() / tl_.mean()), median_hold_days=float(lg['hold_days'].median()), mean_hold_days=float(lg['hold_days'].mean()))
    core9 = dict(total_pnl=total_pnl, sharpe=ps['sharpe'], sortino=ps['sortino'], ann_pnl=total_pnl / n_years, mdd=mdd_v, mdd_date=T['mdd_date'], calmar=calmar,
                 win_rate_daily=ps['win_rate'], pl_ratio_daily=ps['pl_ratio'], n_trades=trade['n_all'], n_days=n_days, n_years=n_years, underwater=uw, kelly_daily=ps['kelly'])
    comp = dict(cagr_pct=chk['cagr_pct'], maxdd_pct=chk['maxdd_pct'], peak=chk['peak'], trough=chk['trough'], recovered=chk['recovered'], final_nav=chk['final_nav'],
                cagr_0050_pct=chk['cagr_0050_pct'], final_nav_0050=chk['final_nav_0050'])

    print('Running Monte Carlo simulation (5000 paths)...', flush=True)
    mc = monte_carlo_bootstrap(pnl_combo, n_sims=5000)
    actual_final = float(cum.iloc[-1])
    pc = lambda a, p: float(np.percentile(a, p))
    mcq = dict(final={p: pc(mc['final_pnls'], p) for p in (5, 50, 95)}, mdd={p: pc(mc['mdds'], p) for p in (5, 50, 95)}, tuw={p: pc(mc['tuws'], p) for p in (5, 50, 95)},
               final_2_5=pc(mc['final_pnls'], 2.5), final_97_5=pc(mc['final_pnls'], 97.5),
               actual_final_pctile=float((mc['final_pnls'] < actual_final).mean()), actual_mdd_pctile=float((mc['mdds'] < mdd_v).mean()), actual_tuw_pctile=float((mc['tuws'] < uw).mean()))
    print('Running Rolling Returns analysis (63/252/504)...', flush=True)
    rolling_results = rolling_returns_analysis(pnl_combo)

    yearly_html, yrows = build_yearly_html(res, eng['yearly'])
    ar = adv_risk(res); ex = extremes(res, eng)

    print('Building 16 charts...', flush=True)
    fig_eq = chart_equity(res)
    # 改編：只改圖例與標題文字（多單→衛星選股、空單→0050 底倉），函數本體與配色不動
    fig_eq.data[1].name = '衛星選股 70%（改編：原「多單」）'; fig_eq.data[2].name = '0050 底倉 30%（改編：原「空單」）'
    fig_eq.update_layout(title='① 權益曲線 (衛星 / 底倉 / 合計)；固定本金 100 萬、不複利', yaxis_title='累積 P&L (元)')
    fig_dd = chart_drawdown(res); fig_hm = chart_monthly_heatmap(res); fig_yr = chart_yearly_bar(res); fig_mb = chart_monthly_bar(res)
    fig_rs = chart_rolling_sharpe(res); fig_boot = chart_stat_bootstrap(T['boot'], T['lo4'], T['hi4'])
    fig_wd = chart_weekday_effect(res); fig_me = chart_month_effect(res)
    fig_sub = chart_subperiod_performance(T['sub_results'])
    fig_mc_paths = chart_mc_paths(mc, np.asarray(cum.values, float), res['date'])
    fig_mc_mdd = chart_mc_mdd_dist(mc['mdds'], mdd_v); fig_mc_final = chart_mc_final_dist(mc['final_pnls'], actual_final)
    fig_roll_pnl = chart_rolling_pnl_dist(rolling_results); fig_roll_mdd = chart_rolling_mdd_dist(rolling_results)
    fig_roll_time = chart_rolling_over_time(rolling_results, res['date'])
    n_charts = 16

    # ---------------- results.json
    grid = eng['grid']
    out = dict(
        meta=dict(report='SURGE 月營收驚奇 完整報告', built='2026-11-27', sample=[start_date, end_date], n_days=n_days, n_years=n_years, version='S75 連續帳戶 v3、V_BRIDGE、100 萬、位移 0',
                  capital_fixed=CAPITAL, plotly_cdn=PLOTLY_CDN, runtime_s=round(time.time() - T0, 1), maxrss_mb=rss()),
        baseline_check=chk,
        core9_fixed_capital=core9, compound=comp, trade_level=trade,
        legs=dict(satellite=T['s9_long'], base0050=T['s9_short'], base_share_of_nav_mean=float((eng['B'] / eng['nav']).mean()), stock_share_of_nav_mean=float(eng['stockfrac'].mean()),
                  avg_npos=eng['st']['avg_npos'], b_clamped_days=eng['st']['b_clamped']),
        tests=dict(
            t01_hac=dict(t=T['t1'], p=T['p1']), t02_lobato=dict(sr=T['sr2'], lo=T['lo2'], hi=T['hi2']), t03_psr=T['p3'], t04_bootstrap=dict(lo=T['lo4'], hi=T['hi4'], n_boot=5000, seed=42),
            t05_ic_trade_level=dict(surge=dict(rho=T['ic_long'][0], p=T['ic_long'][1]), entry_rank=dict(rho=T['ic_rank'][0], p=T['ic_rank'][1]), n=trade['n_all']),
            t06_subperiods=T['sub_results'], t07_rolling252=dict(mean=T['rsr_mean'], pct_below_0=T['pct_below_0']),
            t08_drawdown=dict(mdd=mdd_v, date=T['mdd_date'], underwater=uw), t09_legs=dict(satellite_sr=T['s9_long']['sharpe'], base_sr=T['s9_short']['sharpe']),
            t10_robustness=dict(high_sr_pct=T['high_sr_pct'], best=T['rob_best'], grid=grid),
            t11_concentration=dict(daily_top5=T['contrib'], daily_top1=T['contrib_1'], skew=T['skew_v'], kurt=T['kurt_v'], trade_level=T['trade_top']),
            t12_settlement=dict(settle_pct=T['settle_pct'], post_pct=T['post_pct'], pnl_settle=T['pnl_settle'], pnl_post=T['pnl_post'], n_settle=T['n_settle'], n_post=T['n_post'],
                                exec_day_pct=T['exec_pct'], pnl_exec=T['pnl_exec'], n_exec=T['n_exec'], sig_day_pct=T['sig_pct'], n_sig=T['n_sig']),
            t13_capm=dict(vs_taiex_spec_formula=T['capm_tw'], vs_0050_spec_formula=T['capm_50']),
            t14_dsr=dict(rows=T['dsr_rows'], ours=T['dsr_ours'])),
        summary_rows=[dict(test=r[0], detail=r[1], verdict=r[2]) for r in T['summary_rows']],
        counts=dict(passes=T['passes'], warns=T['warns'], triangs=T['triangs'], crosses=T['crosses']),
        verdicts=dict(statistical=T['verdict_strength'], economic=T['economic_verdict'], overfit=T['overfit_verdict'], stability=T['stability_verdict'], tier=T['tier']),
        monte_carlo=dict(n_sims=5000, seed=42, actual_final=actual_final, actual_mdd=mdd_v, actual_tuw=uw, **mcq),
        rolling={str(w): {k: v for k, v in r.items() if k not in ('pnls', 'mdds')} | dict(n_starts=int(len(r['pnls']))) for w, r in rolling_results.items()},
        yearly=[dict(r, compound=next(y['strat'] for y in eng['yearly'] if y['year'] == r['year']), e0050=next(y['e0050'] for y in eng['yearly'] if y['year'] == r['year'])) for r in yrows],
        advanced_risk=ar, extremes=ex, engine_counters=eng['st'])

    # ---- 13 項加列：以帳戶實際日報酬（＝combo_pnl ÷ 100 萬）重算 CAPM（不計入判定）
    r_act = (res['combo_pnl'] / CAPITAL).values

    def capm_act(b):
        c = sm.OLS(r_act, sm.add_constant(b)).fit()
        return dict(alpha_a=float((1 + c.params[0]) ** 252 - 1), t_alpha=float(c.tvalues[0]), beta=float(c.params[1]), r2=float(c.rsquared))
    twr = eng['twii'].pct_change().reindex(pd.DatetimeIndex(res['Data'])).fillna(0).values
    ca_tw = capm_act(twr); ca_50 = capm_act(res['bench_ret'].fillna(0).values)
    out['tests']['t13_capm']['vs_taiex_actual_daily_return'] = ca_tw; out['tests']['t13_capm']['vs_0050_actual_daily_return'] = ca_50
    alt_pass = T['passes'] + (1 if (T['capm_tw']['t_alpha'] <= 1.96 and ca_tw['t_alpha'] > 1.96) else 0)
    alt_tier = 1 if (alt_pass >= 12 and T['warns'] - (alt_pass - T['passes']) <= 2) else (2 if alt_pass >= 8 else 3)
    out['sensitivity_capm_basis'] = dict(note='若第 13 項改用帳戶實際日報酬判定（不是規格公式）', passes=alt_pass, tier=alt_tier)

    # 對照組（總監複核加列，不計入判定）：同一套檢定套在 0050 本身、以及「策略減 0050」的日超額報酬
    def _ctrl(x):
        x = pd.Series(x).dropna().astype(float)
        return dict(sr=float(x.mean() / x.std(ddof=1) * math.sqrt(252)), hac_t=hac_tstat(x)[0],
                    dsr200=deflated_sharpe(x, 200)[0], dsr_ours=deflated_sharpe(x, N_TRIALS_OURS)[0])
    b_ret = res['bench_ret'].fillna(0).values
    ctrl = {'strategy': _ctrl(r_act), 'b0050': _ctrl(b_ret), 'excess': _ctrl(r_act - b_ret)}
    out['control_benchmark'] = ctrl
    OUT_JSON.write_text(json.dumps(jclean(out), ensure_ascii=False, indent=1), encoding='utf-8')

    html = build_html(locals())  # 含 ctrl
    html = harden_html(html)
    OUT_HTML.write_text(html, encoding='utf-8')
    shutil.copyfile(OUT_HTML, OUT_COPY)
    print(f'  → {OUT_HTML.name}  ({OUT_HTML.stat().st_size / 1024:.0f} KB)；複製到 {OUT_COPY}', flush=True)
    for r in T['summary_rows']:
        print(r)
    print('passes/warns/triangs/✗', T['passes'], T['warns'], T['triangs'], T['crosses'], '｜', T['verdict_strength'], T['economic_verdict'], T['overfit_verdict'], T['stability_verdict'])
    print('done', round(time.time() - T0, 1), 's', rss(), 'MB')
    return out


def build_html(c):
    """組裝完整 HTML — 依規格書第八／九節模板；股票月頻長多的改編處在文中標「改編」。"""
    T = c['T']; res = c['res']; eng = c['eng']; chk = c['chk']; comp = c['comp']; trade = c['trade']; ps = c['ps']; mcq = c['mcq']; rr = c['rolling_results']
    n_days = c['n_days']; n_years = c['n_years']; start_date = c['start_date']; end_date = c['end_date']
    pnl_wan = c['pnl_wan']; ann_wan = c['ann_wan']; mdd_v = c['mdd_v']; mdd_wan = c['mdd_wan']; uw = c['uw']; actual_final = c['actual_final']
    ca_tw = c['ca_tw']; ca_50 = c['ca_50']; ar = c['ar']
    passes, warns, triangs, crosses = T['passes'], T['warns'], T['triangs'], T['crosses']
    sl, ss = T['s9_long'], T['s9_short']; tt = T['trade_top']; dso = T['dsr_ours']
    strategy_name = 'SURGE 月營收驚奇'
    one_liner = '每月營收公布後，買「營收驚奇分數」排名前 10、站上 60 日均線且前 20 日漲幅 > 10% 的上市股票；0050 底倉 30%＋選股 70%，每月換股一次（只做多）。'

    rows_summary_html = ''.join(f'<tr class="{cell_class(r[2])}"><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td></tr>' for r in T['summary_rows'])
    sub_rows_html = ''.join(f'<tr><td>{k}</td><td>{v["sr"]:.3f}</td><td>{v["n"]:,}</td><td>{v["pnl"]:,.0f}</td></tr>' for k, v in T['sub_results'].items())
    dj = lambda r: "排除過擬合" if r["prob"] > 0.95 else ("可信邊緣" if r["prob"] > 0.85 else "可疑")
    drows = list(T['dsr_rows'][:4]) + [dict(dso, ours=True)] + [T['dsr_rows'][4]]
    dsr_rows_html = ''.join(
        f'<tr{" style=&quot;background:#fffbf0;&quot;".replace("&quot;", chr(34)) if r.get("ours") else ""}><td>N = {int(r["n_trials"]):,}{"（改編加列：我們實際試驗次數，約）" if r.get("ours") else ""}</td>'
        f'<td>{r["e_max"]:.3f}</td><td>{r["z"]:.3f}</td><td>{r["prob"]:.2%}</td><td>{dj(r)}</td></tr>' for r in drows)
    rolling_rows_html = ''.join(
        f'<tr><td><b>{w} 日 ({WLAB[w]})</b></td><td>{rr[w]["win_rate"] * 100:.1f}%</td><td>{rr[w]["mean_pnl"]:,.0f}</td><td>{rr[w]["median_pnl"]:,.0f}</td>'
        f'<td>{rr[w]["p5_pnl"]:,.0f}</td><td>{rr[w]["p95_pnl"]:,.0f}</td><td>{rr[w]["median_mdd"]:,.0f}</td><td>{rr[w]["p5_mdd"]:,.0f}</td></tr>' for w in sorted(rr))
    # 檢定 10 網格表（列＝買進名次 N；欄＝前 20 日漲幅門檻）
    gd = {(g['N'], g['mom']): g for g in eng['grid']}

    def gcell(n, mo):
        g = gd[(n, mo)]; base = (n == 10 and mo == 0.10)
        cls = 'win-cell' if g['sharpe'] > 1.0 else 'warn-cell'
        return (f'<td class="{cls}">{"★ " if base else ""}SR {g["sharpe"]:.3f}<br><span style="font-weight:normal;color:#555;font-size:12.5px;">年化 {g["cagr"] * 100:.2f}%｜回撤 {g["maxdd"] * 100:.1f}%｜{g["n_trades"]:,} 筆</span></td>')
    grid_html = ('<table><thead><tr><th>買進名次 N ＼ 前 20 日漲幅門檻</th>' + ''.join(f'<th>&gt; {mo * 100:.0f}%</th>' for mo in GRID_MOM) + '</tr></thead><tbody>' +
                 ''.join(f'<tr><td><b>前 {n} 名</b></td>' + ''.join(gcell(n, mo) for mo in GRID_MOM) + '</tr>' for n in GRID_N) + '</tbody></table>')

    names = lambda ch: '、'.join(r[0].split('. ', 1)[1].split('（')[0] for r in T['summary_rows'] if r[2].startswith(ch)) or '無'
    if T['tier'] == 1:
        cond = '<span style="color:#E74C3C;font-weight:bold;">具備統計上顯著的 alpha</span>（依規格 3-tier：通過 ≥ 12 項且警訊 ≤ 2 項）。'
    elif T['tier'] == 2:
        cond = f'<span style="color:#E74C3C;font-weight:bold;">具備統計上顯著的 alpha</span>，但有 {warns + triangs} 項警訊／中性（依規格 3-tier：通過 ≥ 8 項）。'
    else:
        cond = f'<span style="color:#e67e22;font-weight:bold;">統計顯著性不足</span>（依規格 3-tier：通過 {passes} 項，未達 8 項門檻，判定「⚠ 弱」）。'
    alt = c['out']['sensitivity_capm_basis']
    tier_name = {1: '✓ 強', 2: '✓ 顯著', 3: '⚠ 弱'}
    in_final = mcq['final'][5] <= actual_final <= mcq['final'][95]; in_mdd = mcq['mdd'][5] <= mdd_v <= mcq['mdd'][95]; in_tuw = mcq['tuw'][5] <= uw <= mcq['tuw'][95]
    io = lambda b: '落在' if b else '<b>落在區間外</b>：不在'
    yrs_lost = [y['year'] for y in eng['yearly'] if y['strat'] <= y['e0050']]

    return f"""<!DOCTYPE html><html lang="zh-TW"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{strategy_name} 完整報告</title>
<script src="{PLOTLY_CDN}"></script>
<style>{UNIFIED_CSS}</style></head><body>

<h1 class="title">📘 {strategy_name} — 完整報告</h1>
<p class="subtitle">{one_liner}<br>
標的: <b>台股上市普通股（排除創新板）＋元大台灣 50（0050）</b> · <b>0050 底倉 30%＋衛星選股 70%（最多 10 檔，每檔＝衛星 ÷ 10）</b> · 成本＝嚴成本（每邊 0.5 個跳動點＋0.3% 滑價，另計手續費與證交稅）·
樣本: {start_date} ~ {end_date} ({n_years:.1f} 年, {n_days:,} 個交易日)<br>
本報告依第三方「策略完整報告 — 框架規格書」產出；該規格是為期貨當沖／波段寫的，套到股票月頻長多策略時凡有改動都標「改編」並寫理由，不適用的標 N/A。產出日 2026-11-27。</p>

<div class="master-toc">
<h2>📑 目錄</h2>
<ol>
<li><a href="#ch1">第一章 · 總覽</a></li>
<li><a href="#ch2">第二章 · 策略邏輯</a></li>
<li><a href="#ch3">第三章 · 績效統計</a></li>
<li><a href="#ch4">第四章 · 視覺化分析 (4 互動圖)</a></li>
<li><a href="#ch5">第五章 · 蒙地卡羅模擬</a></li>
<li><a href="#ch6">第六章 · 滾動報酬分析 (Rolling Returns)</a></li>
<li><a href="#ch7">第七章 · 風險管理</a></li>
<li><a href="#ch8">第八章 · 實單 SOP</a></li>
</ol>
</div>

<!-- ==================== 第一章 ==================== -->
<section class="chapter" id="ch1">
<h1 class="chapter-title"><span class="num">1</span>總覽</h1>

<h2>1.1 策略總覽 (5 分鐘速讀)</h2>
<div class="box">
<p><b>策略名稱</b>: {strategy_name}（現行策略；XQ 腳本 20／21／22）</p>
<p><b>標的</b>: 台股上市普通股（排除創新板）＋0050；0050 底倉 30%＋衛星選股 70%，衛星最多 10 檔、每檔＝衛星資金 ÷ 10，沒用到的錢買 0050</p>
<p><b>樣本期間</b>: {start_date} ~ {end_date} ({n_years:.1f} 年, {n_days:,} 個交易日)；2008～2026 連續帳戶 v3（漲跌幅照當時制度）、V_BRIDGE、起始 100 萬、位移 0</p>
<p><b>核心邏輯</b>: 每月營收公布後，算每家公司的營收驚奇分數 SURGE＝（本月營收年增金額 D − 過去 24 個月平均 μ24）÷ 過去 24 個月標準差 σ24。只看收盤站上 60 日均線的股票，依分數排名；買前 10 名且前 20 個交易日漲幅 &gt; 10% 的；換股日（訊號日）掉出前 20 名或收盤 ≤ 60 日均線才賣，平常不賣。每月換股一次，訊號日隔天開盤執行。</p>
<p><b>目前狀態</b>: 模型帳戶自 2026-11-12 起逐日追蹤；使用者目前空手，預計 12/14 進場。本報告全部是歷史回測，不是模型帳戶的成績。</p>
</div>
<table>
<tr><th style="width:26%">口徑</th><th>數字</th><th style="width:34%">說明</th></tr>
<tr><td><b>主要樣本：2008～2026 連續帳戶 v3</b></td><td>年化 {comp['cagr_pct']:.2f}%、最大回撤 {comp['maxdd_pct']:.2f}%（低點 {comp['trough']}）、{trade['n_all']} 筆交易、100 萬 → {comp['final_nav']:,.0f}；同期 0050 年化 {comp['cagr_0050_pct']:.2f}%</td><td>本報告所有檢定、圖表都用這個樣本。產報告前先用引擎重跑，與存檔淨值逐日相同（差 {chk['nav_max_abs_diff_vs_v3_file']:.0f}）才往下做。</td></tr>
<tr><td>註記：2015-07 起官方口徑（正式引擎）</td><td>全期年化 34.46%、2019 起 48.86%、381 筆</td><td>引自 xq_scripts/README.md 第 12 節的正式數字；本報告沒有重算、也不另做一整套。</td></tr>
</table>
<div class="note"><b>怎麼讀這份報告的金額（改編）</b>：規格書的「每日 PnL」是期貨固定口數的金額。連續帳戶是複利的，直接把每天賺賠的金額相加會被後期（帳戶變大以後）主導，所以本報告的主口徑是<b>「固定本金 100 萬、不複利」</b>：每日 PnL＝帳戶當日報酬 × 1,000,000 元。也就是說 <b>1 萬元＝帳戶的 1%</b>。Sharpe、t 值、PSR、DSR 這類比值不受這個換算影響（和直接用日報酬算完全一樣）；會不同的是「總 PnL」與「最大回撤」，見 1.2。</div>

<h2>1.2 核心績效指標</h2>
<p><b>固定本金口徑（主口徑；每日 PnL＝日報酬 × 100 萬，不複利）</b></p>
<div class="metric-grid" style="grid-template-columns: repeat(3, 1fr);">
<div class="metric pass"><div class="label">總 PnL</div>
  <div class="value">{pnl_wan:.0f} 萬</div>
  <div class="label sub">嚴成本, {n_years:.1f} 年；＝每日報酬的加總 × 100 萬</div></div>
<div class="metric pass"><div class="label">年化 Sharpe</div>
  <div class="value">{ps['sharpe']:.2f}</div>
  <div class="label sub">未扣無風險利率（規格的 perf_summary 實際沒有扣）</div></div>
<div class="metric pass"><div class="label">Sortino</div>
  <div class="value">{ps['sortino']:.2f}</div></div>
<div class="metric pass"><div class="label">年化 PnL</div>
  <div class="value">{ann_wan:.1f} 萬</div>
  <div class="label sub">＝每年平均的單利報酬 {ann_wan:.1f}%</div></div>
<div class="metric pass"><div class="label">最大回撤</div>
  <div class="value">{mdd_wan:.0f} 萬</div>
  <div class="label sub">{T['mdd_date']}（日報酬累加口徑）</div></div>
<div class="metric pass"><div class="label">Calmar</div>
  <div class="value">{ann_wan / mdd_wan:.2f}</div>
  <div class="label sub">年化PnL / |MDD|</div></div>
<div class="metric pass"><div class="label">勝率</div>
  <div class="value">{ps['win_rate'] * 100:.1f}%</div>
  <div class="label sub">賺錢交易日比例；逐筆（已出場 {trade['n_closed']} 筆）{trade['win_rate_closed'] * 100:.2f}%</div></div>
<div class="metric"><div class="label">盈虧比</div>
  <div class="value">{ps['pl_ratio']:.2f}</div>
  <div class="label sub">平均獲利日 / 平均虧損日；逐筆 {trade['pl_ratio_trade']:.2f}</div></div>
<div class="metric"><div class="label">交易筆數</div>
  <div class="value">{trade['n_all']:,}</div>
  <div class="label sub">逐筆（含期末未平倉 {trade['n_open_end']} 筆）；交易日 {n_days:,} 天</div></div>
</div>
<p><b>複利口徑（連續帳戶實際淨值；對外數字）</b></p>
<div class="metric-grid">
<div class="metric pass"><div class="label">年化報酬</div><div class="value">{comp['cagr_pct']:.2f}%</div><div class="label sub">0050 同期 {comp['cagr_0050_pct']:.2f}%</div></div>
<div class="metric fail"><div class="label">最大回撤</div><div class="value">{comp['maxdd_pct']:.2f}%</div><div class="label sub">{comp['peak']} → {comp['trough']}，{comp['recovered']} 收復</div></div>
<div class="metric"><div class="label">100 萬期末</div><div class="value">{comp['final_nav'] / 1e4:,.0f} 萬</div><div class="label sub">{comp['final_nav']:,.0f} 元；0050 {comp['final_nav_0050']:,.0f} 元</div></div>
<div class="metric"><div class="label">贏 0050 年數</div><div class="value">{len(eng['yearly']) - len(yrs_lost)} / {len(eng['yearly'])}</div><div class="label sub">輸的年份：{'、'.join(str(y) for y in yrs_lost)}</div></div>
</div>
<div class="note"><b>兩種口徑差在哪</b>：固定本金口徑把每天的報酬率直接相加（單利），所以總 PnL {pnl_wan:.0f} 萬＝{n_years:.1f} 年的日報酬加總 {pnl_wan:.0f}%，不是帳戶實際賺的錢；帳戶實際是複利，100 萬變成 {comp['final_nav'] / 1e4:,.0f} 萬。回撤也不同：日報酬累加的最大回撤是 {mdd_wan:.1f} 萬（相當於累加 −{mdd_wan:.1f}%，低點 {T['mdd_date']}），帳戶實際淨值從高點到低點是 {comp['maxdd_pct']:.2f}%（低點 {comp['trough']}）；連續下跌時，把跌幅相加會比實際淨值跌幅大，兩者低點日期也因此不同。<b>要知道「帳戶會跌多深」請看複利口徑的 {comp['maxdd_pct']:.2f}%。</b></div>

<h2>1.3 權益與回撤</h2>
<div class="chart-container">{div(c['fig_eq'])}</div>
<div class="note"><b>改編</b>：規格的圖 ① 是「多單／空單／合計」。本策略只做多、沒有空單，改成兩條腿：<b>衛星選股 70%</b>（紅線，原多單位置）對 <b>0050 底倉 30%</b>（綠線，原空單位置）。底倉＝引擎逐日記錄的底倉市值（平均占帳戶 {c['out']['legs']['base_share_of_nav_mean'] * 100:.1f}%，在換股日偏離超過 10 個百分點才拉回 30%）；衛星＝帳戶 − 底倉，包含個股與暫時停泊在 0050 的衛星閒置資金（個股平均只占帳戶 {c['out']['legs']['stock_share_of_nav_mean'] * 100:.1f}%、平均持有 {c['out']['legs']['avg_npos']:.1f} 檔）。全部交易成本都算在衛星這條腿。綠色在這裡只是沿用規格配色，不代表虧損。</div>
<div class="chart-container">{div(c['fig_dd'])}</div>

<h2>1.4 月度與年度績效</h2>
<div class="chart-container">{div(c['fig_hm'])}</div>
<div class="chart-container">{div(c['fig_yr'])}</div>
<div class="chart-container">{div(c['fig_mb'])}</div>

<h2>1.5 年度績效對照表</h2>
{c['yearly_html']}
</section>

<!-- ==================== 第二章 ==================== -->
<section class="chapter" id="ch2">
<h1 class="chapter-title"><span class="num">2</span>策略邏輯</h1>
<p>本章只是把既有規則照寫一遍（出處：根目錄 README.md、xq_scripts/README.md、引擎 s10lib／cslib）。本報告沒有更動任何選股、進出場或配置規則。</p>

<h2>2.1 標的與資料來源</h2>
<table>
<tr><th>項目</th><th>內容</th></tr>
<tr><td>交易標的</td><td><b>台股上市普通股（排除創新板）</b>＋<b>元大台灣 50（0050）</b>。改編：規格的「點值、口數」不適用（N/A），改寫成資金配置。</td></tr>
<tr><td>價格資料</td><td>官方日 K（開高低收、成交值、漲跌價差）；含息日報酬＝收盤 ÷ 參考價 − 1（參考價＝收盤 − 漲跌價差，除權息日為除權息參考價）。0050 用含息序列（2015-01-05 起官方日 K 自建，之前用舊檔按比例銜接）。</td></tr>
<tr><td>訊號來源</td><td>上市公司月營收（依公布時點；15 日期限的公司在訊號日看不到當月營收時視為缺值）。對外數字用 V_BRIDGE 版（2013 年 IFRS 轉換前後營收口徑的水準校正，S71 事前登記）。</td></tr>
<tr><td>樣本期間</td><td>{start_date} ~ {end_date}（{n_days:,} 個有報酬的交易日；第一天 {start_date} 是起始日）</td></tr>
<tr><td>大盤指數</td><td>發行量加權股價指數（TAIEX，價格指數、不含息；repo 既有檔，只用於檢定 13）</td></tr>
</table>

<h2>2.2 訊號計算邏輯</h2>
<p>每月營收公布後，對每家公司算「營收驚奇分數」：這個月營收比去年同月多出來的金額，和過去兩年的平常水準比，超出幾個標準差。</p>
<div class="formula">
<b># 營收驚奇分數（只有一個方向：做多）</b>
D     = 本月營收 − 去年同月營收        （年增金額）
μ24   = 過去 24 個月 D 的平均
σ24   = 過去 24 個月 D 的標準差
<b>SURGE = (D − μ24) / σ24</b>

<b># 空單訊號</b>
N/A（本策略只做多，沒有空單）
</div>

<h2>2.3 過濾規則</h2>
<div class="formula">
<b># 排名前的過濾</b>
股池     = 上市普通股，排除創新板
季線     = 還原收盤 &gt; 60 日均線才列入排名
<b># 進場前的過濾</b>
已在漲   = 訊號日前 20 個交易日漲幅 &gt; 10%
</div>

<h2>2.4 進場規則</h2>
<div class="formula">
<b>多單條件</b>: 訊號日 SURGE 排名前 10 名（站上 60 日均線者之中）且 前 20 日漲幅 &gt; 10%，手上有空位（最多 10 檔）才買
<b>空單條件</b>: N/A
<b>訊號日</b>: 每月 10 日（含）之後第一個交易日的下一個交易日，收盤後出名單
<b>進場價</b>: 訊號日的下一個交易日開盤；開盤就漲停的不買、也不遞補；買進金額不到目標一半的不買
</div>

<h2>2.5 出場規則</h2>
<div class="formula">
<b>停損</b>: 無（本策略沒有停損、沒有停利；平常跌破均線也不賣）
<b>正常出場</b>: 只在每月訊號日檢查——掉出前 20 名，或收盤 ≤ 60 日均線 → 下一個交易日開盤賣出；開盤跌停或無成交則順延
</div>

<h2>2.6 資金管理</h2>
<div class="metric-grid">
<div class="metric"><div class="label">0050 底倉</div>
  <div class="value">30%</div>
  <div class="label sub">換股日偏離超過 10 個百分點才調回</div></div>
<div class="metric"><div class="label">衛星選股</div>
  <div class="value">70%</div>
  <div class="label sub">最多 10 檔</div></div>
<div class="metric"><div class="label">每檔金額</div>
  <div class="value">衛星 ÷ 10</div>
  <div class="label sub">不因名單檔數少而加碼</div></div>
<div class="metric"><div class="label">閒置資金</div>
  <div class="value">買 0050</div>
  <div class="label sub">回測平均個股只占帳戶 {c['out']['legs']['stock_share_of_nav_mean'] * 100:.0f}%</div></div>
</div>
<div class="note">改編：規格這一格是「口數、原始保證金、建議本金＝2× 保證金＋1× MDD」。股票現股沒有保證金與口數（N/A），改列既有的配置規則。回測起始本金 100 萬；2020-10-26（盤中零股上路）以前只能買整張，之後可買零股。</div>

<h2>2.7 成本假設</h2>
<table>
<tr><th>項目</th><th>金額</th><th>說明</th></tr>
<tr><td>手續費</td><td>0.1425% × 2.8 折，每筆最低 20 元</td><td>買賣各收一次（嚴成本口徑）</td></tr>
<tr><td>證交稅</td><td>股票賣出 0.3%；0050 賣出 0.1%</td><td></td></tr>
<tr><td>滑價</td><td>每邊 0.5 個跳動點 ＋ 0.3%</td><td>以開盤價為基準往不利方向加（買貴、賣便宜）</td></tr>
<tr><td>成交量限制</td><td>單筆不超過該股當日成交值的 5%</td><td>超過的部分買不到／分次賣</td></tr>
<tr><td>漲跌停</td><td>開盤 ≥ 漲停價不買；開盤 ≤ 跌停價不賣（順延）</td><td>漲跌幅照當時制度：2015-06-01 以前 7%；2008-10-13～10-24 跌幅 3.5%；之後 10%</td></tr>
<tr><td>除權息</td><td>以官方除權息參考價計含息報酬</td><td>2014 年以前的除權息參考價為近似（已知限制）</td></tr>
</table>
<div class="note">全期實際計入的成本（100 萬起始的複利帳戶，元）：手續費 {eng['st']['fee']:,.0f}、證交稅 {eng['st']['tax']:,.0f}、滑價 {eng['st']['slip']:,.0f}、0050 買賣成本 {eng['st']['e50_cost']:,.0f}。因開盤漲停沒買到 {eng['st']['blk_limup']} 次、買不起一張而放棄 {eng['st']['blk_lot']} 次、賣出因無成交順延 {eng['st']['sell_defer_noopen']} 次。</div>
</section>

<!-- ==================== 第三章 ==================== -->
<section class="chapter" id="ch3">
<h1 class="chapter-title"><span class="num">3</span>績效統計</h1>

<div class="executive">
<h2>📋 執行摘要</h2>
<p style="font-size: 15px;"><b>結論</b>: 本策略 {cond}
{passes} 項通過嚴謹檢定 · {warns} 項警訊 · {triangs} 項中性{f' · {crosses} 項未顯著' if crosses else ''}。</p>
<div class="summary-box">
  <div class="summary-pass"><h4>✓ 通過 ({passes}/14)</h4>
    <small>{names('✓')}</small></div>
  <div class="summary-warn"><h4>△ 警訊 ({warns + triangs}/14)</h4>
    <small>⚠：{names('⚠')}<br>△：{names('△')}</small></div>
  <div class="summary-fail"><h4>✗ 嚴重警訊 ({crosses}/14)</h4><small>{names('✗')}</small></div>
</div>
<p style="font-size: 13.5px;">讀法：日報酬本身明顯為正（第 1～4 項都過），但規格要求的穩定度（滾動一年、在水下時間）、對大盤的超額（第 13 項，規格公式）與多重測試校正（第 14 項）沒有過；第 5、9、11 項在規格裡最好也只會是 △。第 13 項對算法很敏感，見 3.1 表下說明。</p>
</div>

<h2>3.1 14 項統計檢定總表</h2>
<table>
<thead><tr><th style="width:22%">檢定</th><th>細節</th><th style="width:30%">判定</th></tr></thead>
<tbody>{rows_summary_html}</tbody>
</table>
<div class="note"><b>各項改編與讀法</b>（門檻與判定邏輯全部照規格，沒有調整）：
<ul>
<li><b>第 1～4、6～8、11（日 PnL 版）、14 項</b>：照規格，輸入＝固定本金口徑的每日 PnL（{n_days:,} 天，沒有空倉日）。</li>
<li><b>第 5 項（改編）</b>：規格是「昨日訊號值對當日 PnL」。本策略每月才出一次訊號，改成逐筆：進場時的 SURGE、進場名次，對該筆交易報酬的 Spearman 相關（{trade['n_all']} 筆）。規格的判定是「p &lt; 0.05 → △，否則 ⚠」，這一項沒有 ✓。</li>
<li><b>第 8 項</b>：在水下時間用「日報酬累加」的權益算，{uw * 100:.1f}% ≥ 85% 門檻，判 ⚠。</li>
<li><b>第 9 項（改編）</b>：多／空改成衛星選股／0050 底倉兩條腿（定義見 1.3）。規格的判定是「兩腿 SR 較小者 &gt; 0.5 → △，否則 ⚠」，這一項沒有 ✓。</li>
<li><b>第 10 項（改編）</b>：兩個參數＝買進名次 N、前 20 日漲幅門檻，用同一個 v3 引擎在 2008～2026 重跑 16 組，見下表。只是敏感度描述，不是用來選參數。</li>
<li><b>第 11 項</b>：規格用「日 PnL 最大的 5% 天數合計 ÷ 總 PnL」，因為每日有賺有賠互相抵銷，這個比值會超過 100%（{T['contrib'] * 100:.0f}%），規格判定是「&gt; 50% → △」。另加一列逐筆版：報酬率最高的 {tt['n_top']} 筆（前 5%，每筆 ≥ {tt['threshold_ret'] * 100:+.2f}%）占選股交易損益 {tt['share_of_trade_pnl'] * 100:.2f}%。</li>
<li><b>第 12 項</b>：結算日照規格算（每月第三個週三，{T['n_settle']} 天）。本策略不交易期貨，這一項只是檢查獲利有沒有集中在特定日子；另加「換股執行日」（{T['n_exec']} 天）占比 {T['exec_pct'] * 100:.2f}%。</li>
<li><b>第 13 項</b>：照規格公式——權益＝100 萬＋累積 PnL，日報酬＝權益變動率，對 TAIEX（價格指數、不含息）日報酬做回歸：α＝{T['capm_tw']['alpha_a'] * 100:.1f}%／年、t＝{T['capm_tw']['t_alpha']:.2f}，未達 1.96，判 ⚠。<b>這個結果對算法很敏感</b>：規格公式把固定 100 萬賺的錢除以越來越大的權益（期末約 {(CAPITAL + actual_final) / 1e4:,.0f} 萬），後期的報酬率被稀釋，β 只有 {T['capm_tw']['beta']:.2f}。若改用帳戶實際日報酬（＝每日 PnL ÷ 100 萬）：對 TAIEX α＝{ca_tw['alpha_a'] * 100:.1f}%／年（t＝{ca_tw['t_alpha']:.2f}，β＝{ca_tw['beta']:.2f}；TAIEX 不含股利，α 會被高估約一個殖利率）；對 0050 含息 α＝{ca_50['alpha_a'] * 100:.1f}%／年（t＝{ca_50['t_alpha']:.2f}，β＝{ca_50['beta']:.2f}）。<b>本報告的判定仍照規格公式記 ⚠</b>；若改用實際日報酬判定，通過數會是 {alt['passes']} 項、統計顯著改列「{tier_name[alt['tier']]}」。這一點請使用者決定採哪個口徑，本報告不代為選擇。</li>
<li><b>第 14 項</b>：照規格 N＝10／50／200／1000／4500，判定看 N＝4500；另加一列我們實際的試驗次數（約 {N_TRIALS_OURS:,} 次，研究紀錄總表累計），見 3.3。</li>
</ul></div>

<h3>檢定 10 明細：16 組敏感度網格（v3 引擎，{start_date} ~ {end_date}）</h3>
{grid_html}
<div class="note">★＝現行參數（前 10 名、&gt; 10%）。每格的 SR 用該組的固定本金每日 PnL 計算；紅底＝SR &gt; 1，橘底＝SR ≤ 1。16 組中 SR &gt; 1 的有 {int(round(T['high_sr_pct'] * 16))} 組（{T['high_sr_pct'] * 100:.1f}%），最高 {T['rob_best']:.3f}，最低 {min(g['sharpe'] for g in eng['grid']):.3f}。改 N 時只改「看前幾名」，持股上限仍是 10 檔、每檔仍是衛星 ÷ 10（N 大於 10 時依名次買到滿 10 檔為止）。這張表只描述敏感度：16 組的 SR 都擠在 {min(g['sharpe'] for g in eng['grid']):.2f}～{T['rob_best']:.2f} 之間，「過不過 1」只差在小數第二位，不要當成參數很穩或某一格比較好的證據。</div>

<h2>二大主要警訊 (請務必了解)</h2>
<div class="caveat"><b>1. 獲利集中於少數大賺交易</b><br>
{trade['n_all']} 筆交易裡，賺錢的只有 {trade['win_rate_all'] * 100:.1f}%，每筆報酬中位數 {trade['median_ret'] * 100:.2f}%。報酬率最高的 {tt['n_top']} 筆（前 5%，每筆 ≥ {tt['threshold_ret'] * 100:+.2f}%）占選股交易損益的 <b>{tt['share_of_trade_pnl'] * 100:.2f}%</b>（占帳戶總獲利的 {tt['share_of_nav_gain'] * 100:.2f}%，其餘來自 0050 部位）。
日 PnL 版：Top 5% 貢獻 {T['contrib'] * 100:.1f}%, <b>Top 1% 貢獻 {T['contrib_1'] * 100:.1f}%</b>。偏度 {T['skew_v']:.2f}, 超額峰度 {T['kurt_v']:.2f}。
這代表漏掉少數幾筆大贏家，成績就會差很多；進場當下分不出哪一筆會是大贏家（xq_scripts/README.md 第 32.1 節）。</div>
<div class="caveat"><b>2. 多重測試校正後不顯著（前後試過約 {N_TRIALS_OURS:,} 次）</b><br>
這套規則是從很多次嘗試裡留下來的。DSR 把「試了 N 次、挑最好的」算進去之後：N＝{N_TRIALS_OURS:,} 時機率 {dso['prob'] * 100:.2f}%（z＝{dso['z']:.3f}）、N＝4500 時 {T['dsr_n4500_prob'] * 100:.2f}%，都沒有達到 95% 的門檻。也就是說，單看這條回測曲線很漂亮，但扣掉「挑過」的成分後，統計上不能排除只是運氣。既有研究的結論相同（S52：扣掉資料窺探後對 0050 的超額不顯著；S46：進場參數有被挑過的跡象）。</div>
<div class="note"><b>對照組（複核加列，不計入判定）：同一套檢定也套在 0050 本身。</b>規格第 1～4、14 項檢定的是「報酬 &gt; 0」，對只做多的股票策略，其中一大部分是大盤本身的報酬。
<table><thead><tr><th>序列（日報酬）</th><th>年化 Sharpe</th><th>HAC t</th><th>DSR N=200</th><th>DSR N={N_TRIALS_OURS:,}</th></tr></thead><tbody>
<tr><td>本策略</td><td>{c['ctrl']['strategy']['sr']:.3f}</td><td>{c['ctrl']['strategy']['hac_t']:.2f}</td><td>{c['ctrl']['strategy']['dsr200'] * 100:.1f}%</td><td>{c['ctrl']['strategy']['dsr_ours'] * 100:.1f}%</td></tr>
<tr><td>0050 含息（什麼都不做）</td><td>{c['ctrl']['b0050']['sr']:.3f}</td><td>{c['ctrl']['b0050']['hac_t']:.2f}</td><td>{c['ctrl']['b0050']['dsr200'] * 100:.1f}%</td><td>{c['ctrl']['b0050']['dsr_ours'] * 100:.1f}%</td></tr>
<tr><td><b>策略 − 0050（超額）</b></td><td><b>{c['ctrl']['excess']['sr']:.3f}</b></td><td><b>{c['ctrl']['excess']['hac_t']:.2f}</b></td><td><b>{c['ctrl']['excess']['dsr200'] * 100:.1f}%</b></td><td><b>{c['ctrl']['excess']['dsr_ours'] * 100:.1f}%</b></td></tr>
</tbody></table>
0050 本身的 t 也遠大於 1.96，所以第 1～4 項「通過」不能證明選股有本事；真正要問的是最後一列：超額報酬 t 約 {c['ctrl']['excess']['hac_t']:.1f}（單次檢定算顯著），但扣掉約 {N_TRIALS_OURS:,} 次試驗後 DSR 只有 {c['ctrl']['excess']['dsr_ours'] * 100:.0f}%，與 S98 的結論一致（S98 用月資料與實測試驗變異數，校正後更低，約 0～12%）。</div>

<h2>3.2 子期穩定性</h2>
<table><thead><tr><th>子期</th><th>Sharpe</th><th>交易筆數</th><th>累計 PnL (元)</th></tr></thead>
<tbody>{sub_rows_html}</tbody></table>
<div class="note">「交易筆數」欄在本報告是交易日數（每天都有部位）。三段 SR 都大於 0，規格判 ✓；但差距很大：最早一段只有 {list(T['sub_results'].values())[0]['sr']:.2f}，最近一段 {list(T['sub_results'].values())[2]['sr']:.2f}，成績主要來自後段。</div>
<div class="chart-container">{div(c['fig_sub'])}</div>

<h2>3.3 過擬合風險分析 (Deflated Sharpe Ratio)</h2>
<p>DSR: 假設你嘗試過 N 組策略參數, 觀察到的 SR 是否仍顯著超過「隨機抽中最佳組合」的期望值?</p>
<table>
<thead><tr><th>N (假設嘗試次數)</th><th>E[max SR | null]</th>
<th>DSR z 分數</th><th>DSR 機率</th><th>判定</th></tr></thead>
<tbody>{dsr_rows_html}</tbody></table>

<h2>3.4 結算日污染分析</h2>
<table>
<tr><th>日期類型</th><th>累計 PnL</th><th>佔總 PnL %</th></tr>
<tr><td>結算日 (每月第三週三)（{T['n_settle']} 天，占全部交易日 {T['n_settle'] / n_days * 100:.1f}%）</td>
    <td>{T['pnl_settle']:,.0f}</td><td>{T['settle_pct'] * 100:.2f}%</td></tr>
<tr><td>結算後首日（{T['n_post']} 天）</td>
    <td>{T['pnl_post']:,.0f}</td><td>{T['post_pct'] * 100:.2f}%</td></tr>
<tr><td>換股執行日（改編加列；{T['n_exec']} 天，占全部交易日 {T['n_exec'] / n_days * 100:.1f}%）</td>
    <td>{T['pnl_exec']:,.0f}</td><td>{T['exec_pct'] * 100:.2f}%</td></tr>
<tr><td>訊號日（改編加列；{T['n_sig']} 天）</td>
    <td>{T['pnl_sig']:,.0f}</td><td>{T['sig_pct'] * 100:.2f}%</td></tr>
</table>

<div class="verdict-final">
<h2>🎯 3.5 最終 alpha 判定</h2>
<div class="metric-grid">
<div class="metric"><div class="label">統計顯著</div>
  <div class="value">{T['verdict_strength']}</div></div>
<div class="metric"><div class="label">經濟顯著</div>
  <div class="value">{T['economic_verdict']}</div></div>
<div class="metric"><div class="label">過擬合風險</div>
  <div class="value">{T['overfit_verdict']}</div></div>
<div class="metric"><div class="label">穩定性</div>
  <div class="value">{T['stability_verdict']}</div></div>
</div>
<p>{T['verdict_para'].replace('color:#E74C3C', 'color:#ffd6d0')}<br>
<span style="font-size:13.5px;">（以上一句是規格書 3-tier 的固定文案，照規則產生、沒有修飾。判定依據：通過 {passes} 項、警訊 {warns} 項、中性 {triangs} 項；經濟顯著看固定本金總 PnL {pnl_wan:.0f} 萬 &gt; 100 萬；過擬合風險依規格看 DSR N＝200 的機率 {T['dsr_n200_prob'] * 100:.1f}%——但我們實際試了約 {N_TRIALS_OURS:,} 次，那一列是 {dso['prob'] * 100:.1f}%，不到 95%，所以「過擬合風險低」這一格偏樂觀；穩定性看中間子期 SR {T['mid_period_sr']:.2f}。第 13 項若改用帳戶實際日報酬，通過數為 {alt['passes']} 項、統計顯著為「{tier_name[alt['tier']]}」。）</span><br><br>
<b>建議實單前</b>（改編：只列既有的凍結規則，沒有新增任何門檻）:<br>
① <b>先看模型帳戶</b>：模型帳戶自 2026-11-12 起逐日追蹤，照十年驗證計畫每年 10 月底檢視一次。規格的「紙上交易 60-90 天」門檻：{NA_RULE}。<br>
② <b>固定配置</b>：0050 底倉 30%＋衛星 70%，每檔＝衛星 ÷ 10，名單全買、每月都做（規格的「固定口數」改編）。<br>
③ <b>風控紀律</b>：既有的是「失效警戒線」（每月底檢查，亮燈＝檢討）與十年驗證計畫的停用規則（見 8.4）。規格的「累計虧損超過保證金 50% 強制暫停」：{NA_RULE}。</p>
</div>
</section>

<!-- ==================== 第四章 ==================== -->
<section class="chapter" id="ch4">
<h1 class="chapter-title"><span class="num">4</span>視覺化分析</h1>
<p>以下 4 個 Plotly 互動圖, 涵蓋滾動 Sharpe、Bootstrap、週日效應、月份效應。
(權益曲線 / 回撤 / 月度與年度績效 已併入第一章 1.3 / 1.4 節)</p>

<h2>4.1 滾動穩定性與信賴區間</h2>
<div class="chart-container">{div(c['fig_rs'])}</div>
<div class="note">滾動 252 日 Sharpe 平均 {T['rsr_mean']:.2f}；低於 0 的期間占 {T['pct_below_0'] * 100:.1f}%（規格門檻 15%）。</div>
<div class="chart-container">{div(c['fig_boot'])}</div>

<h2>4.2 週日與月份效應</h2>
<div class="chart-container">{div(c['fig_wd'])}</div>
<div class="chart-container">{div(c['fig_me'])}</div>
<div class="note">圖中「平均每筆 PnL」在本報告是「平均每個交易日 PnL」（每天都有部位）。這兩張圖只是描述，本策略沒有、也不建議依星期或月份進出。</div>

<h2>4.3 進階風險指標</h2>
{build_advanced_risk_html(ar)}
</section>

<!-- ==================== 第五章 ==================== -->
<section class="chapter" id="ch5">
<h1 class="chapter-title"><span class="num">5</span>蒙地卡羅模擬</h1>

<h2>5.1 方法說明</h2>
<div class="box">
<p>蒙地卡羅模擬用 <b>Bootstrap with replacement</b> 重現策略的不確定性。</p>
</div>
<div class="formula">
<b># 步驟</b>
1. 取全期 {n_days} 個日 PnL 序列（固定本金 100 萬口徑）
2. 隨機抽 {n_days} 個 (有放回) 形成新樣本
3. 計算該樣本的累計 PnL / MDD / 在水下時間
4. 重複 5000 次 (seed=42), 得到 5000 個結果分布
</div>
<div class="note">限制：這個方法把每一天當成互相獨立、隨機重排，所以「連續好幾天一起大跌」（例如 2008 年）在模擬裡會被打散，模擬出來的回撤通常比真實的淺；也不會產生歷史上沒發生過的單日極端。</div>

<h2>5.2 模擬結果 95% 信賴區間</h2>
<table>
<thead><tr><th>指標</th><th>實際回測</th><th>5% 悲觀</th>
<th>50% 中位</th><th>95% 樂觀</th><th>解讀</th></tr></thead>
<tbody>
<tr><td><b>全期 PnL (元)</b></td>
    <td><b>{actual_final:,.0f}</b></td>
    <td>{mcq['final'][5]:,.0f}</td>
    <td>{mcq['final'][50]:,.0f}</td>
    <td>{mcq['final'][95]:,.0f}</td>
    <td>5%～95% 分位（涵蓋 90% 的模擬）；2.5%～97.5% 為 {mcq['final_2_5']:,.0f}～{mcq['final_97_5']:,.0f}</td></tr>
<tr><td><b>MDD (元)</b></td>
    <td><b>{mdd_v:,.0f}</b></td>
    <td>{mcq['mdd'][5]:,.0f}</td>
    <td>{mcq['mdd'][50]:,.0f}</td>
    <td>{mcq['mdd'][95]:,.0f}</td>
    <td>5% 悲觀 = 最差情境；實際值比 {(1 - mcq['actual_mdd_pctile']) * 100:.1f}% 的模擬更深</td></tr>
<tr><td><b>在水下時間 (%)</b></td>
    <td><b>{uw * 100:.1f}%</b></td>
    <td>{mcq['tuw'][5] * 100:.1f}%</td>
    <td>{mcq['tuw'][50] * 100:.1f}%</td>
    <td>{mcq['tuw'][95] * 100:.1f}%</td>
    <td>策略 drawdown 期間比例</td></tr>
</tbody>
</table>

<h2>5.3 模擬路徑視覺化</h2>
<div class="chart-container">{div(c['fig_mc_paths'])}</div>

<h2>5.4 MDD 分布</h2>
<div class="chart-container">{div(c['fig_mc_mdd'])}</div>

<h2>5.5 全期 PnL 分布</h2>
<div class="chart-container">{div(c['fig_mc_final'])}</div>

<h2>5.6 結論</h2>
<div class="{'callout-success' if (in_final and in_mdd) else 'warning'}">
<p>蒙地卡羅模擬（5000 次）的結果：</p>
<ul>
<li>實際全期 PnL ({actual_final:,.0f}) {io(in_final)} 5000 次模擬的 [{mcq['final'][5]:,.0f}, {mcq['final'][95]:,.0f}] 區間內（這一項幾乎必然成立：重抽樣的平均本來就是實際值）</li>
<li>實際 MDD ({mdd_v:,.0f}) {io(in_mdd)}模擬 [{mcq['mdd'][5]:,.0f}, {mcq['mdd'][95]:,.0f}] 區間內{'' if in_mdd else '——實際回撤比 95% 以上的模擬都深，代表真實的下跌是連續發生的（2008 年），隨機重排會低估回撤'}</li>
<li>實際在水下時間 ({uw * 100:.1f}%) {io(in_tuw)}模擬 [{mcq['tuw'][5] * 100:.1f}%, {mcq['tuw'][95] * 100:.1f}%] 區間內</li>
<li>5% 悲觀情境：全期 PnL {mcq['final'][5]:,.0f}（仍為正）、MDD {mcq['mdd'][5]:,.0f}</li>
</ul>
</div>
</section>

<!-- ==================== 第六章 ==================== -->
<section class="chapter" id="ch6">
<h1 class="chapter-title"><span class="num">6</span>滾動報酬分析 (Rolling Returns)</h1>

<h2>6.1 方法說明</h2>
<div class="box">
<p>滾動報酬分析回答一個實單最關心的問題:
<b>「如果我在任何一個時間點上線跑這策略, 持有 N 天, 我會賺多少 / 虧多少?」</b></p>
</div>
<div class="formula">
<b># 三個持有期</b>
63 日   ≈ 3 個月  (短期看持有壓力)
252 日  ≈ 1 年    (典型實單觀察期)
504 日  ≈ 2 年    (長期策略穩定性)
</div>
<div class="note">金額是固定本金 100 萬口徑：1 萬元＝1%（單利加總）。這裡是「連續帳戶中途任一天起算」，不是「那一天才從空手開始建倉」；新開始的人前幾個月持股檔數較少，結果會不同。</div>

<h2>6.2 三窗口統計摘要</h2>
<table>
<thead><tr>
<th>持有期</th><th>勝率</th><th>平均 PnL</th><th>中位 PnL</th>
<th>5% 悲觀</th><th>95% 樂觀</th><th>中位 MDD</th><th>5% 悲觀 MDD</th>
</tr></thead>
<tbody>{rolling_rows_html}</tbody>
</table>

<h2>6.3 持有期 PnL 分布</h2>
<div class="chart-container">{div(c['fig_roll_pnl'])}</div>

<h2>6.4 持有期 MDD 分布</h2>
<div class="chart-container">{div(c['fig_roll_mdd'])}</div>

<h2>6.5 依起始日: 持有 N 天 PnL 變化</h2>
<div class="chart-container">{div(c['fig_roll_time'])}</div>

<h2>6.6 結論</h2>
<div class="callout-success">
<p>滾動報酬分析告訴你「任一天開始、持有一段時間」歷史上的最差 / 最好情境:</p>
<ul>
<li><b>持有 63 日 (3 個月)</b>: 勝率 {rr[63]['win_rate'] * 100:.0f}%, 中位 PnL = {rr[63]['median_pnl']:,.0f} 元；最差的 5% 低於 {rr[63]['p5_pnl']:,.0f} 元</li>
<li><b>持有 252 日 (1 年)</b>: 勝率 {rr[252]['win_rate'] * 100:.0f}%, 中位 PnL = {rr[252]['median_pnl']:,.0f} 元；最差的 5% 低於 {rr[252]['p5_pnl']:,.0f} 元</li>
<li><b>持有 504 日 (2 年)</b>: 勝率 {rr[504]['win_rate'] * 100:.0f}%, 中位 PnL = {rr[504]['median_pnl']:,.0f} 元；最差的 5% 低於 {rr[504]['p5_pnl']:,.0f} 元</li>
</ul>
<p>也就是說，持有一年仍有約 {100 - rr[252]['win_rate'] * 100:.0f}% 的起始日是賠錢的；相鄰起始日的結果高度重疊，這些比例不是獨立樣本。</p>
</div>
</section>

<!-- ==================== 第七章 ==================== -->
<section class="chapter" id="ch7">
<h1 class="chapter-title"><span class="num">7</span>風險管理</h1>

<h2>7.1 主要風險</h2>
<div class="warning"><b>1. 回撤深、而且長時間在水下</b><br>
帳戶實際最大回撤 {comp['maxdd_pct']:.2f}%（{comp['peak']} 高點 → {comp['trough']} 低點，{comp['recovered']} 才收復）。固定本金口徑最大回撤 {mdd_wan:.0f} 萬（{T['mdd_date']}），樣本期 {uw * 100:.0f}% 時間在水下。
策略和大盤一起跌（對 0050 的 β 約 {ca_50['beta']:.2f}）；開始後可能很快就遇到深回撤，需要有心理與資金準備。既有文件的建議是回撤準備 −35%～−40%（2019 年起樣本），2008 年出現過更深的。</div>

<div class="warning"><b>2. 高度依賴少數大賺交易</b><br>
前 5% 的 {tt['n_top']} 筆交易占選股損益 {tt['share_of_trade_pnl'] * 100:.2f}%；賺錢的交易只有 {trade['win_rate_all'] * 100:.1f}%。日 PnL 的 Top 5% 交易日貢獻 {T['contrib'] * 100:.0f}% 總獲利，
日 PnL 偏度 {T['skew_v']:.2f}、超額峰度 {T['kurt_v']:.2f}。漏買、提早賣、挑著買都可能剛好錯過那幾筆。</div>

<div class="warning"><b>3. 訊號 IC 弱</b><br>
進場時的 SURGE 分數與該筆交易報酬的相關很弱：ρ = {T['ic_long'][0]:.3f}（p = {T['ic_long'][1]:.3f}）；進場名次 ρ = {T['ic_rank'][0]:.3f}（p = {T['ic_rank'][1]:.3f}）。
也就是在已經入選的前 10 名裡面，分數更高或名次更前並沒有明顯賺更多；不能靠分數大小再挑股或加碼。</div>

<div class="warning"><b>4. 兩條腿不對稱（改編：原「多空不對稱」）</b><br>
衛星選股 SR ≈ {sl['sharpe']:.2f}（累計 {sl['total_pnl'] / 1e4:,.0f} 萬）, 0050 底倉 SR ≈ {ss['sharpe']:.2f}（累計 {ss['total_pnl'] / 1e4:,.0f} 萬）。
本策略只做多、沒有空單對沖；底倉與衛星同向，大盤大跌時兩條腿會一起跌，底倉不是避險。</div>

<div class="warning"><b>5. 過擬合與樣本外偏弱</b><br>
前後試過約 {N_TRIALS_OURS:,} 種組合，DSR 在 N＝{N_TRIALS_OURS:,} 時 {dso['prob'] * 100:.1f}%，未達 95%。成績集中在後段：三段子期 SR 為 {' / '.join(f"{v['sr']:.2f}" for v in T['sub_results'].values())}。
既有研究：沒用來調參數的 2008～2013 每年只贏 0050 約 1.3 個百分點、不顯著（S55）；2008～2015 年中連續帳戶年化 6.74%（0050 5.20%）；2019 年後大贏家變多主要是市場上飆股變多，飆股比例若回到 2019 年前，大贏家筆數預期減半以上（S82）。
另有已知的回測近似：2014 年以前除權息參考價為近似；新上市無漲跌幅日、處置股、全額交割股未處理。</div>

<h2>7.2 樣本內極值總表 (對應 8.4 三段保護 SOP)</h2>
{build_extremes_html(c['ex'])}
</section>

<!-- ==================== 第八章 ==================== -->
<section class="chapter" id="ch8">
<h1 class="chapter-title"><span class="num">8</span>實單 SOP</h1>
<div class="note"><b>本章改編原則</b>：章節名稱與編號照規格；內容只取自 repo 既有的凍結文件（README「散戶照做須知」、xq_scripts/README.md 第 30～33 節、十年驗證計畫 PLAN_FROZEN、研究紀錄總表）。規格要求、但我們沒有既定規則的項目，一律寫「{NA_RULE}」，本報告不自行訂任何門檻。期貨的保證金／點值／口數改寫成資金、零股／整股、0050 底倉。目前只有<b>模型帳戶</b>（2026-11-12 起）；使用者目前空手，預計 12/14 進場。</div>

<h2>8.1 上線前準備</h2>

<h3>8.1.1 資金準備</h3>
<table>
<tr><th>項目</th><th>金額</th><th>說明</th></tr>
<tr><td><b>原始保證金／維持保證金</b></td><td>N/A</td><td>現股買賣，沒有保證金與追繳（改編）</td></tr>
<tr><td><b>0050 底倉</b></td><td>總資金的 30%</td><td>以 100 萬為例＝30 萬；底倉不因選股而動用，換股日偏離超過 10 個百分點才調回</td></tr>
<tr><td><b>衛星選股</b></td><td>總資金的 70%，每檔＝衛星 ÷ 10</td><td>以 100 萬為例＝70 萬、每檔 7 萬；名單不到 10 檔時，剩下的錢買 0050</td></tr>
<tr><td><b>零股／整股</b></td><td>依本金</td><td>5 萬以下：沒有測，先買 0050 累積；5～30 萬：可以做，一定要用零股；30～50 萬：買得起整張的買整張、不足的用零股補；堅持只買整張者 50 萬以下不建議（S62）</td></tr>
<tr><td><b>回撤緩衝</b></td><td>歷史最大回撤 {comp['maxdd_pct']:.2f}%</td><td>規格的「建議總本金＝2× 保證金＋1× MDD」不適用（N/A）。請用「這筆錢跌掉一半還睡得著嗎」來決定投入多少；要放多少比例：{NA_RULE}</td></tr>
</table>

<h3>8.1.2 券商選擇</h3>
<ul>
<li><b>手續費</b>: 回測使用 0.1425% × 2.8 折、每筆最低 20 元，賣出另計證交稅 0.3%（0050 為 0.1%）。你的折數若較差，成本會比回測高</li>
<li><b>零股低消</b>: 小資金（30 萬以下）請找零股低消 1 元的券商；低消 20 元每年多付約 1～4 個百分點（S62）</li>
<li><b>下單方式</b>: 開盤集合競價；買單掛漲停價下一檔、賣單掛跌停價上一檔；開盤沒成交立刻刪單。規格的「API 速度 30 秒內」不適用（N/A，月頻策略前一晚就知道名單）</li>
<li><b>主流選擇</b>: 既有文件沒有指定券商（N/A）</li>
</ul>

<h3>8.1.3 程式工具</h3>
<ul>
<li><b>每月名單</b>: research/tw_stock_selection/agents/s18_live_list/live_surge/make_list.py；每月換股流程用 xq_scripts/site/monthly_rebalance.py（步驟見 xq_scripts/site/state/README.md）</li>
<li><b>XQ 腳本</b>: xq_scripts/20（選股）、21（出場）、22（交易合併版）</li>
<li><b>模型帳戶網站</b>: xq_scripts/site/（持股、名單、淨值、風險表，每日更新）</li>
<li><b>本報告重現</b>: python3 -I -B build_report.py（同資料夾）</li>
</ul>

<h3>8.1.4 紙上交易 60-90 天 (重要！)</h3>
<div class="tip">既有做法：模型帳戶自 2026-11-12 起逐日記錄（含息），並依十年驗證計畫每年 10 月底檢視一次。規格的「連續 60 天紙交易 PnL &gt; 0 才上實單」：{NA_RULE}。提醒：既有研究顯示分批進場、等回落再進場在歷史上都比一次進場差（S65），但那段期間是少見的多頭。</div>

<h2>8.2 每日操作 SOP</h2>
<p>改編：本策略每月只換股一次，平常不需要操作。以下把規格的「每日時間軸」改成「每月換股時間軸」。</p>
<div class="timeline">
<div class="timeline-step"><div class="time">每月 10 日</div>
  <div class="action">月營收<br>公布截止</div></div>
<div class="timeline-step"><div class="time">訊號日<br>收盤後</div>
  <div class="action">算分數、排名<br>出買賣名單</div></div>
<div class="timeline-step"><div class="time">執行日<br>開盤前</div>
  <div class="action">掛賣單<br>掛買單</div></div>
<div class="timeline-step"><div class="time">執行日<br>09:00</div>
  <div class="action">開盤成交<br>沒成交刪單</div></div>
<div class="timeline-step"><div class="time">執行日<br>收盤後</div>
  <div class="action">記帳<br>剩餘現金買 0050</div></div>
<div class="timeline-step"><div class="time">其餘<br>交易日</div>
  <div class="action">不動作<br>（跌破均線也不賣）</div></div>
</div>

<div class="step"><div class="step-num">1</div><div class="step-content">
<b>訊號日</b>＝每月 10 日（含）之後第一個交易日的<b>下一個交易日</b>。收盤後用當天收盤資料出名單。</div></div>
<div class="step"><div class="step-num">2</div><div class="step-content">
<b>賣出名單</b>: 手上持股在訊號日掉出前 20 名，或收盤 ≤ 60 日均線的，全部列入賣出。</div></div>
<div class="step"><div class="step-num">3</div><div class="step-content">
<b>買進名單</b>: 排名前 10 名且前 20 日漲幅 &gt; 10%、而且手上還沒有的；依名次買到滿 10 檔為止。每檔金額＝衛星資金 ÷ 10。</div></div>
<div class="step"><div class="step-num">4</div><div class="step-content">
<b>執行日</b>＝訊號日的下一個交易日開盤。賣單掛跌停價上一檔、買單掛漲停價下一檔；開盤就漲停的不買、也不改買別檔；開盤沒成交立刻刪單（賣不掉的隔天開盤再掛）。</div></div>
<div class="step"><div class="step-num">5</div><div class="step-content">
<b>收尾</b>: 沒用到的錢買 0050；底倉偏離 30% 超過 10 個百分點才調回。記下成交價、股數、剩餘現金。</div></div>
<div class="step"><div class="step-num">6</div><div class="step-content">
<b>2026-12 這一次</b>: 依上面規則推算，12/10（四）之後第一個交易日是 12/10，下一個交易日 12/11（五）收盤出名單，12/14（一）開盤執行；若遇休市以 monthly_rebalance.py 的輸出為準。</div></div>

<h2>8.3 績效監控指標</h2>

<h3>8.3.1 每日記錄</h3>
<div class="checklist"><b>改編：平常不必每天記；每次換股後、以及每月底記錄</b>:
<ul>
<li>訊號日、執行日；買賣名單與每檔的 SURGE 分數、名次</li>
<li>每檔成交價、當天開盤價、股數（整股／零股）、手續費與稅</li>
<li>沒買到／沒賣掉的原因（開盤漲停、無成交、金額不足）</li>
<li>換股後持股、0050 股數（其中底倉幾股）、剩餘現金</li>
<li>月底：帳戶淨值、同期 0050 含息、兩者的比值與各自離前高的距離</li>
</ul></div>

<h3>8.3.2 警戒指標 (週監控)</h3>
<p>改編：既有的是「失效警戒線」（xq_scripts/README.md 第 30 節，S27），<b>每月底檢查</b>（不是每週）。亮燈的意思是「要檢討」，不是減碼或停手的指令。</p>
<table>
<tr><th>代號</th><th>條件（亮燈）</th><th>策略有效時誤報（1／3／5 年）</th><th>歷史</th></tr>
<tr><td>A</td><td>滾動 36 個月對 0050 年化超額 &lt; −8 個百分點</td><td>2%／5%／8%</td><td>未觸發</td></tr>
<tr><td>E</td><td>策略 ÷ 0050 對前高 &lt; −24%</td><td>2%／6%／10%</td><td>未觸發</td></tr>
<tr><td>B</td><td>滾動 12 個月超額 &lt; −26 個百分點</td><td>2%／5%／9%</td><td>未觸發</td></tr>
<tr><td>C</td><td>月底回撤 &lt; −29.9%（約日內 −37%）</td><td>2%／6%／9%</td><td>未觸發</td></tr>
<tr><td>D</td><td>連續 ≥ 約 50 個月沒追回相對高點</td><td>0%／5%／10%</td><td>不單獨用</td></tr>
<tr><td>黃燈</td><td>任一條</td><td>6%／18%／27%</td><td>—</td></tr>
<tr><td>紅燈</td><td>同月 2 條以上（事後新增，只當描述）</td><td>1%／4%／7%</td><td>未觸發</td></tr>
</table>
<div class="note">上表的門檻、誤報率與「歷史」欄照抄既有文件，是用 2019 年起的正式引擎樣本訂的，不是本報告用 2008～2026 樣本算的（2008 年的回撤 {comp['maxdd_pct']:.2f}% 比 C 線深）。</div>
<table>
<tr><th>規格要求的指標</th><th>正常範圍</th><th>警戒值</th><th>嚴重值</th></tr>
<tr><td>Rolling 60 日 Sharpe</td><td colspan="3">{NA_RULE}</td></tr>
<tr><td>勝率 (Rolling 30 筆)</td><td colspan="3">{NA_RULE}（參考：歷史逐筆賺錢比例 {trade['win_rate_all'] * 100:.1f}%，一半以上的交易會賠是常態）</td></tr>
<tr><td>累積 drawdown（幾倍 MDD）</td><td colspan="3">{NA_RULE}（既有的是上表 C、E 兩條線）</td></tr>
<tr><td>連續水下時間 (TUW)</td><td colspan="3">{NA_RULE}（既有的是上表 D 線，看相對 0050）</td></tr>
<tr><td>實際滑價</td><td colspan="3">{NA_RULE}（參考：回測假設每邊 0.5 個跳動點＋0.3%；既有估計每邊多 0.1% 約少 0.85 個百分點／年）</td></tr>
</table>

<h2>8.4 三段保護機制 SOP (實單應對)</h2>
<p>改編：規格的三段保護是期貨「綠燈正常口數 → 橙燈減口數 → 紅燈全停」。本策略<b>沒有減碼規則</b>；既有的凍結規則只有兩種：失效警戒線（亮燈＝檢討）與十年驗證計畫（每年檢視一次、可能停用）。對應如下，沒有的部分不補。</p>
<table>
<tr><th>燈號</th><th>規格原意</th><th>本策略既有的規則</th><th>動作</th></tr>
<tr><td>🟢 綠燈</td><td>正常口數</td><td>五條警戒線都沒亮</td><td>照規則每月換股：名單全買、每月都做、不自行停手</td></tr>
<tr><td>🟠 橙燈</td><td>減口數</td><td>黃燈＝任一條警戒線亮（每月底檢查）</td><td>檢討（先查執行落差：漏單、滑價、買不到）。<b>減碼幅度與門檻：{NA_RULE}</b></td></tr>
<tr><td>🔴 紅燈</td><td>全停</td><td>(a) 同月 2 條以上警戒線亮（只當描述）；(b) 十年驗證計畫的停用條件（正式規則，只在每年 10 月最後一個交易日判定）</td><td>(a) 檢討；(b) 判定停用時改買 0050（紙上紀錄照做）</td></tr>
</table>

<h3>8.4.1 從 🟢 切換到 🟠 的觸發條件 (任一達成)</h3>
<div class="checklist"><ul>
<li>既有：8.3.2 表的 A、E、B、C、D 任一條亮燈（黃燈）。動作是檢討，不是減碼。</li>
<li>規格要求的「單日虧損／連虧次數／單月虧損達到 7.2 極值即減倉」：{NA_RULE}。</li>
</ul></div>

<h3>8.4.2 為何選「小台 2 口」減倉</h3>
<p>N/A（期貨口數）。改編成「要不要減碼、減多少」：{NA_RULE}。既有研究可供參考（不是規則）：帳戶回落 20% 就停手改放 0050、等 6 個月，歷史上每年少 4.8 個百分點，而且最大回撤沒有變淺（S64）；想少承擔選股風險的既有做法是一開始就一半資金買 0050 不動、一半照做（S63）。</p>

<h3>8.4.3 從 🟠 切回 🟢 的恢復條件</h3>
<p>既有：警戒線的數值回到門檻以內就不再亮燈，沒有另設恢復條件。規格要求的「連續幾個月獲利才恢復」：{NA_RULE}。</p>

<h3>8.4.4 從 🟠 升級到 🔴 全停的紅燈條件 (任一觸發)</h3>
<div class="checklist"><b>既有的正式停用規則＝十年驗證計畫（2026-11-19 採用；界線數值凍結在 PLAN_FROZEN.json，不得更改）</b>:
<ul>
<li>對象：模型帳戶（含息）對只買 0050（股利再投入），自 2026-11-12 起算；每年 10 月最後一個交易日收盤後檢視一次，共 10 次（2027-10～2036-10）。</li>
<li>第 1～4 年：幾乎任何結果都繼續做；只有極端落後（統計上顯著低於每年 +3 個百分點）才判定無效。</li>
<li>第 5～9 年：累積下來輸給 0050 → 停用，改買 0050。</li>
<li>第 10 年：累積年化超額 &lt; +3 個百分點 → 停用；+3～約 +9 → 由產品擁有者決定；約 +9 以上 → 判定有效。</li>
<li>已知代價：策略真有每年 +5 個百分點時，約 27% 的機率會被「滿 5 年輸 0050 就停用」誤停。</li>
<li>規格要求的「橙燈期間再虧多少就全停」：{NA_RULE}。</li>
</ul></div>

<h3>8.4.5 操作範例</h3>
<p>N/A。規格的範例 A／A2／B／C 是期貨減倉後「連續獲利月計數」的例子；本策略沒有減碼與恢復規則，沒有可以舉的例子，也不自行編造。</p>

<h3>8.4.6 預期橙燈持續時間</h3>
<p>既有文件只有「會不會亮」的機率，沒有估「亮多久」：策略照舊有效時，黃燈在 1／3／5 年內誤報的機率是 6%／18%／27%；超額歸零時 1／3／5 年內亮燈的機率是 58%／85%／94%。持續時間：N/A（未估）。</p>

<h3>8.4.7 每日追蹤表 (新增欄位)</h3>
<p>改編成每月底追蹤五個數字（既有文件 2026-10 月底現值）：A +15.0 個百分點、E −7.1%、B +33.5 個百分點、C −5.4%（日內 −10.6%）、D 9 個月，全部正常。</p>

<h3>8.4.8 預期績效 (依模式區分)</h3>
<p>規格的「綠燈／橙燈模式各自的預期績效」不適用（沒有減碼模式）。既有文件對模型帳戶的預期（2019 年起樣本，不是本報告的 2008～2026 樣本）：心裡放年化約 40%、回撤準備 −35%～−40%；參數挑選後的參考預期約 40%（S46）。本報告的 2008～2026 連續帳戶年化是 {comp['cagr_pct']:.2f}%、最大回撤 {comp['maxdd_pct']:.2f}%，明顯低於 2019 年起的數字。</p>

<h3>8.4.9 一句話 SOP</h3>
<div class="box"><p><b>三段保護機制</b>（本策略版）：平常照規則每月換股、不自行停手；每月底看五條警戒線，亮燈就檢討；每年 10 月底照十年驗證計畫判定，要停用時改買 0050。減碼門檻與恢復條件目前沒有規則，待使用者決定。</p></div>

<h2>🎯 最後叮嚀</h2>
<div class="warning">
<p style="font-size: 16px;">
<b>1. 紀律執行</b>: 規則很簡單、一個月只做一次；最大的敵人是自己想挑股、想提早賣、想跳過這個月。<br>
<b>2. 一半以上的交易會賠</b>: 歷史賺錢比例 {trade['win_rate_all'] * 100:.1f}%、每筆中位數 {trade['median_ret'] * 100:.2f}%，連續好幾筆不賺是正常的；成績靠少數大贏家。<br>
<b>3. 不要優化</b>: 規則已凍結；要改必須另題事前登記，不能看了這份報告就調參數。<br>
<b>4. 部位管理</b>: 0050 底倉 30%＋衛星 70%，每檔＝衛星 ÷ 10；名單全買、不挑股、開盤漲多也照買（開盤就漲停才不買）。<br>
<b>5. 心理優先</b>: 歷史最大回撤 {comp['maxdd_pct']:.2f}%。任何讓你失眠的金額都太大了，投入多少請自己決定。<br>
<b>6. 持續記錄</b>: 每次換股與每月底都記帳，才能分辨「策略失效」和「自己沒照做」。</p>
</div>
<p style="font-size:13px;color:#777;">這不是投資建議；實際下單與盈虧自負。本報告只用公開資料與 repo 既有資料。</p>
</section>

<a href="#" class="back-top" title="回到頂部">↑</a>
<p class="footer">
{strategy_name} 完整報告 | 樣本期間 {start_date} ~ {end_date}<br>
依「策略完整報告 — 框架規格書」產出（股票月頻長多改編版）；Plotly 2.35.2 直接內嵌於檔案（改編：規格為 CDN 載入，同版本；內嵌後離線也能顯示）；重現：python3 -I -B build_report.py
</p>
</body></html>"""


if __name__ == '__main__':
    main()
