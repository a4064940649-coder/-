#!/usr/bin/env python3
"""R55 收尾：IFRS 斷點檢查＋彙整 results.json、compare.csv（S55 宣稱值對 R55 主做法與 S55 等價做法）。
先跑 step1_is.py、build_panel_r55.py（各變體）、step2_oos.py（各變體）、spotcheck_source.py。python3 -I -B make_results.py"""
import sys, json, inspect, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r55lib as R
from r55lib import s17lib, stlib, s2lib
import numpy as np, pandas as pd

H = Path(__file__).resolve().parent
J = lambda n: json.loads((R.TMP / n).read_text())
s1 = J('step1.json'); sp = J('spotcheck.json')
V = {c: J(f'oos_{c}.json') for c in ('cache_pre', 'cache_pre_blanknan_fillchgnan', 'cache_pre_fillchgnan', 'cache_pre_blanknan', 'cache_pre_nofill', 'cache_pre_gapfill')}
PN = {c: J(f'panel_{c}.json') for c in V}

# ---- IFRS 斷點：引擎原始碼與營收資料
src = inspect.getsource(s17lib.rev_family)
s17_mtime = (R.AG / 's17_surge_definition_family/s17lib.py').stat().st_mtime
s55_spec_mtime = (R.AG / 's55_pre2015_oos/SPEC.json').stat().st_mtime
r = pd.read_csv(stlib.REV_CSV, dtype={'code': str}, parse_dates=['month'])
tw = r[r['market'] == 'sii']
rev_codes_2013 = int(tw[tw['month'].dt.year == 2013]['code'].nunique())
R.point_cache(R.TMP / 'cache_pre')
stlib.START = pd.Timestamp('2008-01-02'); s17lib.FIRST_MONTH = pd.Period('2008-01', 'M')
P = R.s20lib.load()
F = s17lib.rev_family(P, defs=['STD'])
S = F['S']['STD']; Rv = F['Rv']
nfin = np.isfinite(S.to_numpy()).sum(1)
cnt = pd.Series(nfin, index=S.index.astype(str))
Draw = Rv - Rv.shift(12)
d2013 = int(np.isfinite(Draw[Draw.index.year == 2013].to_numpy()).sum())
ifrs = dict(
    engine_lines=[l.strip() for l in src.split('\n') if '2013' in l],
    also_in=dict(s2lib_rev_tables='D[(D.index.year == 2013)] = np.nan' in inspect.getsource(s2lib.rev_tables),
                 stlib_sig_revsurge='D[(D.index.year == 2013)] = np.nan' in inspect.getsource(stlib.sig_revsurge),
                 s20lib_growth_features='D[D.index.year == 2013] = np.nan' in inspect.getsource(R.s20lib.growth_features)),
    s17lib_mtime_before_s55_spec=bool(s17_mtime < s55_spec_mtime),
    in_sample_first_month_2016_01_reason='官方樣本內 s17lib.FIRST_MONTH＝2016-01 正是因為 2013 年 D 缺值後要 24 個月 σ 才算得出；首訊號 2016-02-16（step1）',
    in_sample_first_signal=s1['first_signal'],
    twse_codes_with_revenue_2013=rev_codes_2013,
    finite_D_2013_if_unmasked=d2013,
    finite_SURGE_by_month={k: int(v) for k, v in cnt.loc['2012-09':'2016-03'].items()},
    verdict='2013 年 D 設缺值寫在共用引擎 s17lib.rev_family（以及 s2lib／stlib／s20lib 的同一行），S55 之前就存在、樣本內也照用；R55 只改 START／FIRST_MONTH／快取，'
            '未碰這行，2013-01～2015-12 營收月份 SURGE 仍全缺 → 是凍結規則本來的行為，不是 S55 的處理。營收資料本身 2013 年有值（D 若不遮罩有 '
            f'{d2013} 格），缺值純粹來自引擎遮罩。')

# ---- 對照表
main, s55eq = V['cache_pre'], V['cache_pre_blanknan_fillchgnan']
claims = [  # (項目, S55 宣稱, 容許, 取值函式)
    ('官方面板 現行 2019 起年化', 0.4876, 0.0001, lambda o: s1['A0']['since2019'], 'ratio'),
    ('官方面板 已平倉筆數', 381, 0, lambda o: s1['A0']['n_closed'], 'n'),
    ('第一個訊號日', '2008-02-13', None, lambda o: o['signal']['first'], 's'),
    ('最後訊號日', '2013-01-11', None, lambda o: o['signal']['last'], 's'),
    ('訊號月數', 60, 0, lambda o: o['signal']['n_signal_months'], 'n'),
    ('交易筆數（買進）', 146, 0, lambda o: o['trades']['n_buy'], 'n'),
    ('主窗 月數', 61, 0, lambda o: o['windows']['primary']['months'], 'n'),
    ('主窗 策略年化 %', 2.99, 0.1, lambda o: o['windows']['primary']['cagr_strat'] * 100, 'pp'),
    ('主窗 0050 年化 %', 1.84, 0.1, lambda o: o['windows']['primary']['cagr_0050'] * 100, 'pp'),
    ('主窗 年化超額（百分點）', 1.27, 0.1, lambda o: o['windows']['primary']['ex_ann_pp'], 'pp'),
    ('主窗 資訊比率', 0.23, 0.02, lambda o: o['windows']['primary']['IR'], 'ratio'),
    ('主窗 NW6 t', 0.51, 0.05, lambda o: o['windows']['primary']['t_nw6'], 't'),
    ('主窗 最大回撤 %', -49.2, 0.1, lambda o: o['windows']['primary']['maxdd'] * 100, 'pp'),
    ('次窗 年化超額（百分點）', -1.10, 0.1, lambda o: o['windows']['secondary']['ex_ann_pp'], 'pp'),
    ('次窗 NW6 t', -0.52, 0.05, lambda o: o['windows']['secondary']['t_nw6'], 't'),
    ('逐年超額 2008', 5.44, 0.1, lambda o: o['yearly']['2008']['ex_pp'], 'pp'),
    ('逐年超額 2009', 10.12, 0.1, lambda o: o['yearly']['2009']['ex_pp'], 'pp'),
    ('逐年超額 2010', 4.94, 0.1, lambda o: o['yearly']['2010']['ex_pp'], 'pp'),
    ('逐年超額 2011', -2.13, 0.1, lambda o: o['yearly']['2011']['ex_pp'], 'pp'),
    ('逐年超額 2012', -11.35, 0.1, lambda o: o['yearly']['2012']['ex_pp'], 'pp'),
    ('合併 主窗＋樣本內 資訊比率', 0.61, 0.02, lambda o: o['windows']['merged_primary_is']['IR'], 'ratio'),
    ('合併 主窗＋樣本內 NW t', 2.80, 0.05, lambda o: o['windows']['merged_primary_is']['t_nw6'], 't'),
    ('（附）主窗判定', '方向一致但不顯著', None, lambda o: o['decision_primary'], 's'),
    ('（附）卡住期 年化超額', -6.09, 0.1, lambda o: o['windows']['stuck']['ex_ann_pp'], 'pp'),
    ('（附）逐年超額 2013', -2.93, 0.1, lambda o: o['yearly']['2013']['ex_pp'], 'pp'),
    ('（附）樣本內 年化超額', 11.68, 0.1, lambda o: o['windows']['in_sample']['ex_ann_pp'], 'pp'),
]
rows = []
for name, cl, tol, fn, kind in claims:
    a, b = fn(main), fn(s55eq)
    ok = lambda v: (v == cl) if tol is None else (abs(v - cl) <= tol + 1e-12)
    fmt = lambda v: v if isinstance(v, str) else (int(v) if kind == 'n' else round(float(v), 4))
    rows.append(dict(item=name, s55_claim=cl, tol=tol, r55_main=fmt(a), r55_main_ok=bool(ok(a)), r55_s55equiv=fmt(b), r55_s55equiv_ok=bool(ok(b))))
cmp = pd.DataFrame(rows)
cmp.to_csv(H / 'compare.csv', index=False, encoding='utf-8')

sens = {c: dict(n_buy=o['trades']['n_buy'], primary_ex=o['windows']['primary']['ex_ann_pp'], primary_IR=o['windows']['primary']['IR'], primary_t=o['windows']['primary']['t_nw6'],
                secondary_ex=o['windows']['secondary']['ex_ann_pp'], secondary_t=o['windows']['secondary']['t_nw6'],
                merged_IR=o['windows']['merged_primary_is']['IR'], merged_t=o['windows']['merged_primary_is']['t_nw6'], decision_primary=o['decision_primary'],
                yearly={y: round(v['ex_pp'], 3) for y, v in o['yearly'].items()}) for c, o in V.items()}
ta = pd.read_csv(R.TMP / 'oos_trades_cache_pre.csv', dtype={'code': str}); tb = pd.read_csv(R.TMP / 'oos_trades_cache_pre_blanknan_fillchgnan.csv', dtype={'code': str})
m = ta.merge(tb, on=['code', 'ent_date', 'exit_date'], suffixes=('_r55', '_s55eq'))
dtr = m[(m['ret_r55'] - m['ret_s55eq']).abs() > 1e-9][['code', 'ent_date', 'exit_date', 'ret_r55', 'ret_s55eq']].to_dict('records')
s55_panel = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s55/cache_pre/panel_cs.npz')
eq = None
if s55_panel.exists():
    a = np.load(R.TMP / 'cache_pre_blanknan_fillchgnan/panel_cs.npz'); b = np.load(s55_panel)
    eq = bool(all(np.array_equal(a[k], b[k], equal_nan=(a[k].dtype.kind == 'f')) for k in b.keys()))
res = dict(
    task='R55：獨立重現 S55（凍結現行規則 2008-01～2013-01 樣本外）；未讀 S55 的 s55lib.py／prep.py／run.py／posthoc.py／build_pre_panel.py／results*.json',
    setup=dict(engine='s47lib.sim（→ s10lib.simulate_s10，W10_70、E0、strict、100 萬）；股池 stlib.universe(twse) & ~創新板；SURGE s17lib.rev_family STD；F1 make_signal_s10(shift=0)；'
                      'cslib.EXRIGHT_FIX＝DEADLINE_FIX＝INS15_MASK＝True；s17lib.e50_fix_suspension',
               runtime_overrides=dict(cache='scratchpad/r55/cache_pre（自建 panel_cs.npz、e50_ohlc.csv（官方 0050 只 2015-01-05 起）、innov.npz 全 False）；不呼叫 git',
                                      stlib_START='2008-01-02', s17lib_FIRST_MONTH='2008-01', cslib_END='不變'),
               panel_main='長表（空白漲跌價差且有成交→0）＋ data_panel 補 72,224 格（45 檔上櫃轉上市等 Asoul 缺列，2010 起）：補格漲跌價差＝收−收÷(1＋data_panel ret)，'
                          '使除權息日 ret＝0 → 報酬 0（與 SPEC「2014 以前除權息日報酬記 0」一致）；MK＝有觀測的格',
               panel_s55equiv='同上，但空白漲跌價差留 NaN、補格漲跌價差一律 NaN（引擎退路＝收÷前收−1）；與 S55 的 cache_pre/panel_cs.npz 逐陣列相同：' + str(eq)),
    official_A0=s1['A0'], in_sample=s1['is_monthly'],
    panel_checks={c: PN[c] for c in ('cache_pre', 'cache_pre_blanknan_fillchgnan')}, panel_identical_to_s55=eq,
    spotcheck_source=dict(codes=sp['codes'], n_cells=sp['n_cells'], n_match=sp['n_match'], src_sha_all_ok=sp['all_src_sha_ok'], long_sha_ok=sp['long_sha_ok'],
                          full_file_all_match=all(v['all_match'] for v in sp['full_file'].values()),
                          src_rows_in_range_not_in_long=sum(v['src_rows_in_range_not_in_long'] for v in sp['full_file'].values())),
    oos_main=main, oos_s55equiv=s55eq, sensitivity=sens, trades_differing_main_vs_s55equiv=dtr,
    cause_of_difference='唯一差異是 data_panel 補格的漲跌價差：S55 留 NaN → 引擎用收÷前收−1，除權息日把除息跌價當損失（3209 全科 2010-07-22 −9.43%、3419 2013-09-10 −4.96%），'
                        '兩筆交易報酬各少約 9.9／3.5 個百分點；R55 主做法照 data_panel ret（除權息日 0），符合 SPEC／PREREG 宣告的「2014 以前除權息日報酬記 0」。'
                        '長表空白漲跌價差設 0 或留 NaN 結果完全相同。',
    ifrs_check=ifrs, compare=rows)
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print(cmp.to_string())
print(json.dumps(ifrs, ensure_ascii=False, indent=1))
