#!/usr/bin/env python3
"""減資恢復買賣日／面額變更日：盤點正式引擎的處理，並試算「改用事件參考價」對正式數字的影響。
做法同 audit_daily/daily_check.py：run_official.py 複製到暫存區、輸出改寫到暫存區，不覆蓋正式檔。
  第 1 次（盤點）：原樣跑，另外把事件格子的引擎數值（R、REF、收盤、前收）與 E0｜W10_70 標準路徑的交易明細倒出來。應與正式檔逐日差 0。
  第 2 次（試算）：在 cslib.exright_fix_chg 回傳之後多做一步——上市、當天有成交、當天在事件表 → 漲跌價差＝收盤 − 事件參考價（蓋掉原值，含規則 (b) 已改的）。
輸出：reduce/event_cells.csv、reduce/trades_E0.csv.gz、reduce/held_events.csv、reduce/whatif_reduce.json、reduce/navs_official_reducefix.csv.gz（候選，不是正式檔）
python3 -I -B run_official_reduce_whatif.py"""
import sys, json, subprocess, shutil, re
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
SITE = Path('/home/user/-/xq_scripts/site')
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/eng_reduce_whatif')
EV = HERE / 'tables/reduce_parvalue_events.csv'
PATCH = '''
# ==== 試算插入：減資恢復買賣日、面額變更日用事件參考價 ====
_ev = pd.read_csv(%r, dtype={'code': str}, parse_dates=['date']); _ev = _ev[_ev['kind'].isin(['reduce', 'parvalue'])]
_orig_fix = cslib.exright_fix_chg
def _fix(IDX, COLS, C, CHG, MK, ev):
    CHG2, info = _orig_fix(IDX, COLS, C, CHG, MK, ev)
    ci = {c: j for j, c in enumerate(COLS)}; di = {d: i for i, d in enumerate(pd.DatetimeIndex(IDX))}
    n = 0
    for r in _ev.itertuples():
        t, j = di.get(r.date), ci.get(r.code)
        if t is None or j is None or MK[t, j] != 1 or not (np.isfinite(C[t, j]) and C[t, j] > 0):
            continue
        CHG2[t, j] = C[t, j] - float(r.ref_price); n += 1
    info = dict(info, n_reduce_parvalue=n); print('事件參考價套用', n, '格', flush=True)
    return CHG2, info
cslib.exright_fix_chg = _fix
'''
DUMP_CELLS = '''
_ev0 = pd.read_csv(%r, dtype={'code': str}, parse_dates=['date'])
_di = {d: i for i, d in enumerate(pd.DatetimeIndex(P.IDX))}; _rows = []
for r in _ev0.itertuples():
    t, j = _di.get(r.date), P.ci.get(r.code)
    if t is None or j is None:
        _rows.append(dict(date=r.date, code=r.code, kind=r.kind, in_panel=False)); continue
    tp = t - 1
    while tp > 0 and not (np.isfinite(P.C[tp, j]) and P.C[tp, j] > 0):
        tp -= 1
    _rows.append(dict(date=r.date, code=r.code, kind=r.kind, reason=r.reason, in_panel=True, mk=int(P.MK[t, j]), close=float(P.C[t, j]), open=float(P.O[t, j]),
                      engine_R=float(P.R[t, j]), engine_REF=float(P.REF[t, j]), prev_close_panel=float(P.C[tp, j]), prev_trade_date=str(P.IDX[tp].date()),
                      prev_day_traded=bool(np.isfinite(P.C[t - 1, j]) and P.C[t - 1, j] > 0), evt_pre_close=float(r.pre_close), evt_ref=float(r.ref_price)))
pd.DataFrame(_rows).to_csv(%r, index=False)
'''
DUMP_LG = '''
            if sid == 'E0' and s == 0 and cap == 1e6:
                lg.drop(columns=['j']).to_csv(%r, index=False)
'''


def run(tag, patch):
    d = TMP / tag; (d / 'site').mkdir(parents=True, exist_ok=True); (d / 'out').mkdir(exist_ok=True)
    src = (AG / 'official_nav' / 'run_official.py').read_text(encoding='utf-8')
    rep = [("HERE = Path(__file__).resolve().parent", f"HERE = Path({str(AG / 'official_nav')!r})"),
           ("SITE = Path('/home/user/-/xq_scripts/site')", f"SITE = Path({str(d / 'site')!r})"),
           ("OUT = HERE / 'out'; OUT.mkdir(exist_ok=True)", f"OUT = Path({str(d / 'out')!r})"),
           ("P = s19lib.load()\n", (PATCH % str(EV) if patch else '') + "P = s19lib.load()\n" + DUMP_CELLS % (str(EV), str(d / 'event_cells.csv'))),
           ("            nav, lg, st = sim(sid, Sg, cap)\n", "            nav, lg, st = sim(sid, Sg, cap)\n" + DUMP_LG % str(d / 'trades_E0.csv'))]
    for a, b in rep:
        assert src.count(a) == 1, a
        src = src.replace(a, b)
    for f in SITE.glob('navs_*.csv.gz'):
        if f.name != 'navs_official.csv.gz':
            shutil.copy(f, d / 'site' / f.name)
    (d / 'run_copy.py').write_text(src, encoding='utf-8')
    r = subprocess.run([sys.executable, '-I', '-B', str(d / 'run_copy.py')], capture_output=True, text=True,
                       env={'GIT_NO_LAZY_FETCH': '1', 'PATH': '/usr/bin:/bin:/usr/local/bin', 'HOME': '/root'})
    print(tag, 'rc', r.returncode, r.stdout[-300:], r.stderr[-200:])
    return d, r.returncode


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not re.search(r'sec|rss|time|elapsed|clock|peak_mb', k, re.I)}
    return [strip(v) for v in o] if isinstance(o, list) else o


off = pd.read_csv(SITE / 'navs_official.csv.gz', parse_dates=['date']).set_index('date')
ref = json.loads((AG / 'official_nav/out/res_official.json').read_text())
R = {}
d0, rc0 = run('control', False)
c = pd.read_csv(d0 / 'site/navs_official.csv.gz', parse_dates=['date']).set_index('date')
R['control'] = dict(rc=rc0, nav_max_abs_diff_vs_official=float((c - off).abs().max().max()),
                    res_json_same=strip(json.loads((d0 / 'out/res_official.json').read_text())) == strip(ref))
# ---- 盤點
cells = pd.read_csv(d0 / 'event_cells.csv', dtype={'code': str}, parse_dates=['date'])
lg = pd.read_csv(d0 / 'trades_E0.csv', dtype={'code': str}, parse_dates=['sig_date', 'ent_date', 'exit_date'])
lg.to_csv(HERE / 'reduce/trades_E0.csv.gz', index=False)
D0, DE = off.index[0], off.index[-1]
x = cells[(cells['in_panel'] == True) & cells['kind'].isin(['reduce', 'parvalue']) & (cells['date'] >= D0) & (cells['date'] <= DE)].copy()
x = x[(x['mk'] == 1) & (x['close'] > 0)]
x['true_R'] = x['close'] / x['evt_ref'] - 1
x['engine_case'] = np.where(x['engine_R'].abs() < 1e-12, '報酬記 0', np.where((x['engine_REF'] - x['prev_close_panel']).abs() < 1e-6, '用前一成交收盤當參考價（規則 b 或官方漲跌價差）', '其他'))
x['close_factor_true_over_engine'] = (1 + x['true_R']) / (1 + x['engine_R'])                 # 持有到當天收盤：正確 ÷ 引擎
x['open_factor_true_over_engine'] = x['engine_REF'] / x['evt_ref']                            # 當天開盤賣出：正確 ÷ 引擎
x['pre_close_matches_panel'] = (x['evt_pre_close'] - x['prev_close_panel']).abs() < 0.011
x['true_R_beyond_limit'] = x['true_R'].abs() > 0.101
x.to_csv(HERE / 'reduce/event_cells.csv', index=False)
R['events_in_period'] = dict(n=int(len(x)), by_kind=x['kind'].value_counts().to_dict(), by_engine_case=x['engine_case'].value_counts().to_dict(),
                             n_pre_close_mismatch=int((~x['pre_close_matches_panel']).sum()), n_true_R_beyond_10pct=int(x['true_R_beyond_limit'].sum()),
                             mean_close_factor_pct=float((x['close_factor_true_over_engine'].mean() - 1) * 100),
                             mean_close_factor_pct_by_case={k: float((g.mean() - 1) * 100) for k, g in x.groupby('engine_case')['close_factor_true_over_engine']})
held = []
for e in x.itertuples():
    for p in lg[lg['code'] == e.code].itertuples():
        if p.ent_date < e.date <= p.exit_date and not (p.why == 'open_end' and e.date > DE):
            sold_at_open = (e.date == p.exit_date) and p.why != 'open_end'
            f = e.open_factor_true_over_engine if sold_at_open else e.close_factor_true_over_engine
            held.append(dict(date=str(e.date.date()), code=e.code, kind=e.kind, reason=e.reason, ent_date=str(p.ent_date.date()), exit_date=str(p.exit_date.date()), why=p.why,
                             timing='當天開盤賣出' if sold_at_open else '持有到當天收盤', evt_pre_close=e.evt_pre_close, evt_ref=e.evt_ref, close=e.close, engine_R_pct=round(e.engine_R * 100, 4),
                             true_R_pct=round(e.true_R * 100, 4), engine_case=e.engine_case, factor_pct=round((f - 1) * 100, 4), trade_cost=round(p.cost), trade_proceeds=round(p.proceeds),
                             est_impact_on_proceeds=round(p.proceeds * (f - 1)), nav_that_day=float(off['E0|W10_70'].loc[e.date])))
        elif p.ent_date == e.date:
            held.append(dict(date=str(e.date.date()), code=e.code, kind=e.kind, reason=e.reason, ent_date=str(p.ent_date.date()), exit_date=str(p.exit_date.date()), why=p.why,
                             timing='當天開盤才買進（買價已是事件後價格；只有漲停判斷用錯參考價）', evt_pre_close=e.evt_pre_close, evt_ref=e.evt_ref, close=e.close,
                             engine_R_pct=round(e.engine_R * 100, 4), true_R_pct=round(e.true_R * 100, 4), engine_case=e.engine_case, factor_pct=0.0, trade_cost=round(p.cost),
                             trade_proceeds=round(p.proceeds), est_impact_on_proceeds=0, nav_that_day=float(off['E0|W10_70'].loc[e.date])))
H = pd.DataFrame(held); H.to_csv(HERE / 'reduce/held_events.csv', index=False)
R['held_events_E0_standard'] = held
# ---- 試算
d1, rc1 = run('reducefix', True)
n = pd.read_csv(d1 / 'site/navs_official.csv.gz', parse_dates=['date']).set_index('date')
res1 = json.loads((d1 / 'out/res_official.json').read_text())
shutil.copy(d1 / 'site/navs_official.csv.gz', HERE / 'reduce/navs_official_reducefix.csv.gz')
(HERE / 'reduce/res_official_reducefix.json').write_text(json.dumps(res1, ensure_ascii=False, indent=1))
lg1 = pd.read_csv(d1 / 'trades_E0.csv', dtype={'code': str}, parse_dates=['sig_date', 'ent_date', 'exit_date'])
key = lambda L: set(zip(L['code'], L['ent_date'].astype(str), L['exit_date'].astype(str)))
R['whatif'] = dict(rc=rc1, n_cells_patched=res1.get('exright_info', {}).get('n_reduce_parvalue'), columns={}, trades_same_set=key(lg) == key(lg1),
                   trades_only_now=sorted(key(lg) - key(lg1)), trades_only_new=sorted(key(lg1) - key(lg)),
                   targets_changed_months=[m for m in ref['strats']['E0|W10_70']['targets'] if ref['strats']['E0|W10_70']['targets'][m] != res1['strats']['E0|W10_70']['targets'].get(m)])
for k in off.columns:
    rel = (n[k] / off[k] - 1) * 100
    R['whatif']['columns'][k] = dict(end_now=float(off[k].iloc[-1]), end_new=float(n[k].iloc[-1]), end_diff_pct=float(rel.iloc[-1]), max_abs_diff_pct=float(rel.abs().max()),
                                     first_day_diff=(str(rel[rel.abs() > 1e-6].index[0].date()) if (rel.abs() > 1e-6).any() else None))
for sid in ref['strats']:
    a, b = ref['strats'][sid], res1['strats'][sid]
    R['whatif'][sid] = dict(std={k: dict(now=a['std'][k], new=b['std'][k]) for k in ('cagr_since2019', 'cagr_full', 'cagr_since2024', 'maxdd', 'final_nav', 'n_trades_closed')},
                            yearly={y: dict(now=round(a['std']['yearly'][y] * 100, 3), new=round(b['std']['yearly'][y] * 100, 3)) for y in a['std']['yearly'] if abs(a['std']['yearly'][y] - b['std']['yearly'][y]) > 5e-6},
                            paths_summary=dict(now=a.get('paths_summary'), new=b.get('paths_summary')))
(HERE / 'reduce/whatif_reduce.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str))
print(json.dumps(R, ensure_ascii=False, indent=1, default=str)[:9000])
shutil.rmtree(TMP, ignore_errors=True)
