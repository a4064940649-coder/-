#!/usr/bin/env python3
"""數據稽核部 2026-11-21：S75 連續帳戶新口徑的獨立核對（不讀 s75_continuous_2008_2026/v2_2026-11-21/ 的程式）。
用法：python3 -I -B s75_recheck.py <base|v2>
  base＝S75 原口徑（三個開關、舊 0050 序列）：要逐日重現 s75_continuous_2008_2026/nav_2008_2026.csv.gz，確認我的執行方式沒問題；
  v2  ＝新口徑：多開 cslib.REDUCE_FIX（減資事件檔只有 2015-08 起）＋0050 改用「2015-01-05 起官方含息序列、之前沿用舊來源並等比例銜接」的自建檔。
只 import S75 原引擎（s75lib → s71lib／r55lib／s47lib → s10lib …），不改任何原檔；連續面板快取複製到自己的暫存區。
輸出（本資料夾 s75/）：nav_<模式>.csv.gz、trades_<模式>.csv.gz、yearly_<模式>.csv、res_<模式>.json、e50_spliced_audit.csv（v2）。"""
import sys, json, shutil, time, resource
from pathlib import Path
import numpy as np, pandas as pd

MODE = sys.argv[1]
assert MODE in ('base', 'v2')
HERE = Path(__file__).resolve().parent
AG = HERE.parent
OUT = HERE / 's75'; OUT.mkdir(exist_ok=True)
SCR = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/v2recalc/s75')
SRC_CACHE = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s75/cache_cont')
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
(SCR / 'cache_cont').mkdir(parents=True, exist_ok=True)
for f in ('panel_cs.npz', 'e50_ohlc.csv', 'innov.npz'):
    if not (SCR / 'cache_cont' / f).exists():
        shutil.copy2(SRC_CACHE / f, SCR / 'cache_cont' / f)

sys.path.insert(0, str(AG / 's75_continuous_2008_2026'))
import s75lib as K                                           # noqa: E402
from s75lib import R, cslib, S71                             # noqa: E402
K.TMP = SCR; S71.TMP = SCR; R.TMP = SCR
assert cslib.EXRIGHT_FIX and cslib.DEADLINE_FIX and cslib.INS15_MASK and cslib.REDUCE_FIX is False and cslib.E50_SOURCE is None
info = dict(mode=MODE)
if MODE == 'v2':
    SEAM = pd.Timestamp('2015-01-05')
    old = pd.read_csv(cslib.E50_CSV, parse_dates=['date']).set_index('date')['adj'].sort_index()
    off = pd.read_csv(cslib.E50_OFFICIAL_CSV, parse_dates=['date']).set_index('date')['adj'].sort_index()
    assert off.index[0] == SEAM and SEAM in old.index
    scale = float(off.loc[SEAM] / old.loc[SEAM])
    sp = pd.concat([old.loc[old.index < SEAM] * scale, off])
    assert sp.index.is_monotonic_increasing and sp.index.is_unique
    sp.index.name = 'date'; sp.rename('adj').to_csv(OUT / 'e50_spliced_audit.csv')
    cslib.REDUCE_FIX = True
    cslib.E50_SOURCE = OUT / 'e50_spliced_audit.csv'
    info['splice'] = dict(seam=str(SEAM.date()), scale=scale, n_pre=int((old.index < SEAM).sum()), n_official=int(len(off)))

P, U = K.load_cont()
assert P.reduce_fix == (MODE == 'v2')
info.update(reduce_info=P.reduce_info, e50_source=P.e50_source, e50_missing=int(P.e50_missing), T=int(P.T), N=int(P.N))
D0, DE = P.IDX[P.t0], P.IDX[-1]
assert (str(D0.date()), str(DE.date())) == ('2008-01-02', '2026-10-22')
D19 = K.last_before(P.IDX, '2019-01-01')
bench = pd.Series(P.E50, index=P.IDX).loc[D0:]; bench = bench / bench.iloc[0] * 1e6
res = dict(info=info, versions={})
NAV = {}; TR = []
for v, rule in K.VERSIONS.items():
    F, RT = K.surge(P, rule)
    sg = K.signal(P, U, RT, 0)
    nav, lg, st = K.sim(P, sg, U, 1e6)
    b = bench.reindex(nav.index)
    x = lg[lg['why'] != 'open_end']
    res['versions'][v] = dict(final_nav=float(nav.iloc[-1]), cagr=K.cagr(nav, D0, DE), cagr_since2019=K.cagr(nav, D19, DE), maxdd=K.dd_info(nav),
                              n_closed=int(len(x)), n_all=int(len(lg)), win_rate=float((x['ret'] > 0).mean()),
                              final_nav_0050=float(b.iloc[-1]), cagr_0050=K.cagr(b, D0, DE), cagr_0050_since2019=K.cagr(b, D19, DE), maxdd_0050=K.dd_info(b),
                              yearly=K.yearly_table(nav, b), bears=K.bear_block(nav, b),
                              sig_dates=[i['sig_date'] for i in sg['info']], targets=[' '.join(i['targets']) for i in sg['info']])
    NAV[v] = nav; TR.append(lg.drop(columns=['j']).assign(version=v))
    print(v, round(float(nav.iloc[-1])), f"{K.cagr(nav, D0, DE):.4%}", res['versions'][v]['maxdd'], len(x), f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
df = pd.DataFrame({'V_MASK': NAV['V_MASK'], 'V_BRIDGE': NAV['V_BRIDGE'], '0050': bench.reindex(NAV['V_MASK'].index)}).round(2); df.index.name = 'date'
df.to_csv(OUT / f'nav_{MODE}.csv.gz')
pd.concat(TR).to_csv(OUT / f'trades_{MODE}.csv.gz', index=False)
ym, yb = res['versions']['V_MASK']['yearly'], res['versions']['V_BRIDGE']['yearly']
pd.DataFrame([dict(year=m['year'], V_MASK=m['strat'], V_BRIDGE=q['strat'], e0050=m['e0050']) for m, q in zip(ym, yb)]).to_csv(OUT / f'yearly_{MODE}.csv', index=False)
# ---- 與 S75 原檔（base）或工程新口徑輸出（v2，只讀輸出檔）逐日比對
ref_p = AG / 's75_continuous_2008_2026' / ('nav_2008_2026.csv.gz' if MODE == 'base' else 'v2_2026-11-21/nav_2008_2026_v2.csv.gz')
ref = pd.read_csv(ref_p, index_col=0, parse_dates=True)
same = ref.shape == df.shape and (ref.index == df.index).all()
res['vs_ref'] = dict(ref=str(ref_p), same_shape=bool(same), max_abs_diff={c: float((ref[c] - df[c]).abs().max()) for c in df.columns} if same else None)
tref_p = AG / 's75_continuous_2008_2026' / ('trades.csv.gz' if MODE == 'base' else 'v2_2026-11-21/trades_v2.csv.gz')
tref = pd.read_csv(tref_p, dtype={'code': str}); mine = pd.read_csv(OUT / f'trades_{MODE}.csv.gz', dtype={'code': str})
res['vs_ref']['trades_same_shape'] = bool(tref.shape == mine.shape)
if tref.shape == mine.shape:
    num = [c for c in mine.columns if pd.api.types.is_numeric_dtype(mine[c])]
    oth = [c for c in mine.columns if c not in num]
    res['vs_ref']['trades_text_cols_equal'] = bool((tref[oth].astype(str).to_numpy() == mine[oth].astype(str).to_numpy()).all())
    res['vs_ref']['trades_num_max_abs_diff'] = float(np.nanmax(np.abs(tref[num].to_numpy(float) - mine[num].to_numpy(float))))
res['peak_mb'] = rss(); res['secs'] = round(time.time() - t0, 1)
json.dump(res, open(OUT / f'res_{MODE}.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res['vs_ref'], ensure_ascii=False), res['peak_mb'], 'MB', res['secs'], 's')
