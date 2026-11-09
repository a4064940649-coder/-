#!/usr/bin/env python3
"""o18 題 2 獨立複製（步驟 2）：V7X 下市對價 v3 口徑（複製員自寫結算表），13 j × 20 seeds（plans_j7 原名單）。
只讀 import 共用引擎：o_engine_delist_consideration/cz_engine.py（只用來產生 o17 cons_main 對照與非 33 起事件的舊規則）、
cz_v7x.CSim、o_exec_engine_rewrite_v7x/x_*。結算規則依 SPEC PREREG 文字自行寫成 rule dict。"""
import sys, json, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
TOP = HERE.parent; AG = TOP.parent
sys.path.insert(0, str(AG / 'o_engine_delist_consideration'))
sys.path.insert(0, str(AG / 'o_exec_engine_rewrite_v7x'))
import cz_v7x as Z
import cz_engine as CZ
import x_stats as S
D, X = Z.D, Z.X
sys.path.pop(0); sys.path.pop(0)

R1 = json.loads((HERE / 'r1_classify.json').read_text())
E2 = pd.read_csv(AG / 'o_engine_delist_consideration' / 'events_v2.csv', dtype=str)
for c in ('cash', 'ratio', 'pref_ratio', 'pref_value'): E2[c] = pd.to_numeric(E2[c])
DELTA, DELTA12 = R1['delta'], R1['delta_pre2013']
O17V = set(E2[E2.verified == '已核實（新聞）'].code)
MINE = set(R1['text']) | O17V           # 33 起：v3 規則
WSDOWN = {'2432', '2394', '3009', '3012', '2473', '8078'}   # 事後敏感度：WebSearch 核實降回市場證據


def _idx(d): return int(np.searchsorted(D.DATES, np.datetime64(pd.Timestamp(d))))


def _sfirst(sj, d):
    s0 = _idx(d); w = np.flatnonzero(D.TR[:, sj] & np.isfinite(D.O[:, sj])); w = w[w >= s0]
    return int(w[0]) if len(w) else None


def v3_settle(bound, wsdown=False):
    out = {}
    for r in E2[E2.code.isin(MINE)].itertuples():
        j = D.CI.get(r.code)
        ltU = pd.Timestamp(r.last_trade)
        term = pd.Timestamp(r.terminate_date) if isinstance(r.terminate_date, str) and r.terminate_date else ltU + pd.Timedelta(days=14)
        w = np.flatnonzero(D.TR[:, j]); w = w[D.IDX[w] <= max(term, ltU)]; lt = int(w[-1])
        lc = float(D.C[lt, j]); ti = _idx(term); pay = min(ti + 20, len(D.IDX) - 1)
        de = DELTA12 if ltU < pd.Timestamp('2013-01-01') else DELTA
        db = {'main': 0.0, 'lo': -de, 'hi': de}[bound]
        ru = dict(code=r.code, lt=lt, trig=lt + 1, lc=lc, pay=pay, term=ti, cash_0050=False, parts=[])
        P = ru['parts']
        sj = D.CI.get(r.successor) if isinstance(r.successor, str) else None
        sf = _sfirst(sj, r.succ_first_trade) if (sj is not None and isinstance(r.succ_first_trade, str)) else None
        if r.code in O17V:    # o17 已核實條款（events_v2）
            ru['type'] = 'v_' + r.type
            if r.type == 'cash': P.append(dict(kind='cash', per_share=float(r.cash), rel=pay))
            else:
                if np.isfinite(r.ratio): P.append(dict(kind='stock', succ=sj, ratio=float(r.ratio), rel=sf, per_share_window=lc))
                if np.isfinite(r.cash): P.append(dict(kind='cash', per_share=float(r.cash), rel=pay))
                if np.isfinite(r.pref_ratio): P.append(dict(kind='fixed_sell', per_share=float(r.pref_ratio * r.pref_value), rel=sf))
            out.setdefault(j, []).append(ru); continue
        T = R1['text'][r.code]; st = T['status']; t = T['terms']; mk = R1['market'][r.code]
        if wsdown and r.code in WSDOWN: st = 'TYPE_TEXT'
        if st == 'VERIFIED_TEXT':
            ru['type'] = 'VERIFIED_' + t['kind']
            if t['kind'] == 'cash': P.append(dict(kind='cash', per_share=t['cash'], rel=pay))
            elif t['kind'] in ('swap', 'mixed', 'mixed_pref'):
                if sj is None:
                    sj = D.CI.get(t['succ']); sf = _sfirst(sj, term) if sj is not None else None
                assert sj == D.CI.get(t['succ']), r.code
                if sf is None:   # 存續股不在面板 → 價值型
                    ru['type'] += '->fixed'; P.append(dict(kind='cash', per_share=lc, rel=pay)); out.setdefault(j, []).append(ru); continue
                P.append(dict(kind='stock', succ=sj, ratio=t['ratio'], rel=sf, per_share_window=lc))
                if t['kind'] == 'mixed': P.append(dict(kind='cash', per_share=t['cash'], rel=pay))
                if t['kind'] == 'mixed_pref' and bound == 'hi': P.append(dict(kind='fixed_sell', per_share=t['pref'] * 10.0, rel=sf))
        elif t['kind'] == 'conv_newsucc':      # 1716 → 3705
            sj = D.CI['3705']; sf = _sfirst(sj, '2011-01-03')
            ru['type'] = 'CONV_NEWSUCC'
            P.append(dict(kind='stock', succ=sj, ratio=lc * (1 + db) / float(D.O[sf, sj]), rel=sf, per_share_window=lc))
        elif mk['market'] == 'MARKET_TRACK' and sj is not None and sf is not None:
            ru['type'] = st + '+TRACK'
            P.append(dict(kind='stock', succ=sj, ratio=mk['ratio'] * (1 + db), rel=sf, per_share_window=lc))
        elif st == 'SEARCHED_NOT_FOUND' and mk['market'] == 'MARKET_NONE':
            ru['type'] = 'SNF'
            if bound == 'lo': P.append(dict(kind='fixed_sell', per_share=lc, rel=lt + 1, zero=True))
            else: P.append(dict(kind='cash', per_share=lc, rel=pay))
        else:                                  # LOCK、TENDER、TYPE 無比例、NONE → 固定請求權
            ru['type'] = st + '+' + mk['market'] + '->fixed'
            P.append(dict(kind='cash', per_share=lc * (1 + db), rel=pay))
        out.setdefault(j, []).append(ru)
    # 其他（非 33 起）事件：沿用 o17 規則（main）；若被結算會在 log 中出現
    rest = CZ.build_settlement(E2[~E2.code.isin(MINE)], 'main', d=0.057, TR=D.TR, IDX=D.IDX, Cn=D.C, CI=D.CI, On=D.O)
    for j, L in rest.items():
        for ru in L: ru['type'] = 'REST_' + ru['type']
        out.setdefault(j, []).extend(L)
    return out


ENG = {}
CELLS = [('K5T', 'normal', 1e6, '100萬'), ('K5T', 'stress', 1e6, '100萬'), ('K5T', 'normal', 3e6, '300萬'), ('K5T', 'normal', 1e7, '1000萬')]
BASE = dict(stress_odd_extra=True, rank_right=True, f1=True, f2=True, f3=0.5, f4=True, mode='GROW')


def _job(a):
    key, pl, eng, cell = a
    m, co, tier, tl = cell
    cfg = dict(BASE, tier=tier, model=m, cost=co)
    st = ENG[eng]
    s = Z.CSim(pl, cfg, settle=st, cmode='orig' if st is None else 'consideration')
    nav = s.run()
    lg = [(e['code'], e['type']) for e in s.log]
    return key, eng, tl + '|' + co, nav, lg, s.acc['n_fallback'], s.acc['n_merger_delay']


if __name__ == '__main__':
    t0 = time.time()
    ENG['orig'] = None
    ENG['cons_main'] = CZ.build_settlement(E2, 'main', d=0.057, TR=D.TR, IDX=D.IDX, Cn=D.C, CI=D.CI, On=D.O)
    for b in ('main', 'lo', 'hi'): ENG['v3_' + b] = v3_settle(b)
    for b in ('main', 'lo', 'hi'): ENG['wsd_' + b] = v3_settle(b, wsdown=True)
    # 結算表摘要
    summ = {}
    for j, L in ENG['v3_main'].items():
        for ru in L:
            if ru['code'] in MINE: summ[ru['code']] = dict(type=ru['type'], parts=[{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in p.items()} for p in ru['parts']])
    P = D.plans('AB_V7_B10', js=list(range(13)), seeds=list(range(20)))
    jobs = []
    for eng in ENG:
        cl = CELLS if not eng.startswith('wsd') else CELLS[:1]
        if eng == 'cons_main': cl = CELLS[:1]
        for cell in cl:
            for key, pl in P.items(): jobs.append((key, pl, eng, cell))
    navs = {}; logs = {}; fb = {}; md = {}
    with Pool(4) as pool:
        for key, eng, cl, nav, lg, nf, nm in pool.imap_unordered(_job, jobs, chunksize=4):
            navs.setdefault((eng, cl, key[0]), []).append(nav)
            for x in lg: logs.setdefault((eng, cl), set()).add(x)
            fb[(eng, cl)] = fb.get((eng, cl), 0) + nf; md.setdefault((eng, cl), []).append(nm)
    print('runs done', round(time.time() - t0), flush=True)
    res = {}; yr = {}
    avg = {}
    for (eng, cl, j), L in navs.items():
        avg[(eng, cl, j)] = S.avg_nav(L)
    for eng in ENG:
        for cell in CELLS:
            cl = cell[3] + '|' + cell[1]
            if (eng, cl, 0) not in avg: continue
            per = [S.stats(avg[(eng, cl, j)], 'AB_V7_B10', j) for j in range(13)]
            sm = S.jsumm(per)
            res[eng + '|' + cl] = {k: sm[k] for k in ('exc_med', 't_0050_med', 'exc_p10', 'exc_Y19on_med', 't_Y19on_med', 'exc_Y19on_p10', 'CAGR_med', 'maxDD_med', 'exc_SEL_med')}
            res[eng + '|' + cl]['exc_by_j'] = [p['exc'] for p in per]
            res[eng + '|' + cl]['settled'] = sorted(logs.get((eng, cl), []))
            res[eng + '|' + cl]['n_fallback'] = fb[(eng, cl)]; res[eng + '|' + cl]['merger_delay_mean'] = float(np.mean(md[(eng, cl)]))
            yr[eng + '|' + cl] = pd.DataFrame({j: S.yearly_exc(avg[(eng, cl, j)], 'AB_V7_B10', j) for j in range(13)})
    # R4 重現
    R4 = pd.read_parquet(AG / 'o_exec_engine_rewrite_v7x' / 'out' / 'navs_oos_R4.parquet')
    mx = 0.0
    for cell in CELLS:
        for j in range(13):
            a = avg[('orig', cell[3] + '|' + cell[1], j)]; b = R4[f'B10|J{j}|K5T|{cell[1]}|{cell[3]}|GROW'].dropna()
            mx = max(mx, float(np.nanmax(np.abs(a.reindex(b.index).to_numpy() / b.to_numpy() - 1))))
    res['R4_maxreldiff'] = mx
    # 逐年差 v3_main 對 orig（100 萬一般）
    a, b = yr['v3_main|100萬|normal'], yr['orig|100萬|normal']
    dmed = (a.median(1) - b.median(1)); dpj = (a - b).median(1)
    res['yearly_v3_vs_orig'] = dict(diff_of_medians={int(k): float(v) for k, v in dmed.items()}, sum=float(dmed.sum()), abssum=float(dmed.abs().sum()),
                                    median_of_diffs={int(k): float(v) for k, v in dpj.items()}, sum_mdiff=float(dpj.sum()), abssum_mdiff=float(dpj.abs().sum()))
    res['settle_v3_main'] = summ
    (HERE / 'r2_v7x.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    for k, v in res.items():
        if isinstance(v, dict) and 'exc_med' in v:
            print(k, *(f'{x}={v[x]:.3f}' for x in ('exc_med', 't_0050_med', 'exc_p10', 'exc_Y19on_med', 't_Y19on_med', 'maxDD_med')))
    print('R4', mx); print(res['yearly_v3_vs_orig']['sum'], res['yearly_v3_vs_orig']['abssum'])
    print(sorted(res['yearly_v3_vs_orig']['diff_of_medians'].items(), key=lambda kv: -abs(kv[1]))[:5])
    print('time', round(time.time() - t0))
