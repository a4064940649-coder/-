#!/usr/bin/env python3
"""合併各版本結果、判定、診斷（P1、P3、P4、P5、價格修補、穩健性）。python3 -I -B diag.py → results.json（保留 PREREG）＋ 印出表格。"""
import sys, json, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import osurv_core as O
V, K = O.V, O.K
RJ = HERE / 'results.json'
RES = json.loads(RJ.read_text(encoding='utf-8'))
assert RES.get('PREREG') and hashlib.sha256(json.dumps(RES['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == RES['PREREG_sha256']
VERS = ['ORIG', 'R_DEL', 'B05_DEL', 'B10_DEL', 'R_ALL', 'B05_ALL', 'B10_ALL', 'PX_ORIG', 'PX_DEL']
R = {v: json.loads((HERE / f'run_{v}.json').read_text(encoding='utf-8')) for v in VERS}
PL = {v: json.loads((HERE / f'plans_{v}.json').read_text(encoding='utf-8')) for v in VERS}
SEGS = ['S1', 'S2', 'S3', 'S4', 'P1', 'ALL']
CONFS = ['ALL|n10', 'EXFIN|n10', 'EXFIN|n30', 'ALL|n30']
MET = ['CAGR', 'maxDD', 'Calmar', 't_0050', 't_EW', 't_0050_cal', 't_EW_cal', 'ex_0050', 'ex_EW', 'E50_CAGR', 'EW_CAGR']
SLn = V.SL['name']; SLi = V.SL['industry']
out = {}

# ---------------- 1. 主表 ----------------
TAB = {}
for v in VERS:
    for cf in CONFS:
        for mode in ('BASE', 'AB'):
            for c in ('normal', 'stress'):
                st = R[v][mode][cf][c]
                for s in SEGS:
                    for m in MET:
                        TAB[f'{v}|{cf}|{mode}|{c}|{s}|{m}'] = st[s][m]
out['TABLE'] = TAB
def g(v, cf, s, m, mode='BASE', c='normal'): return TAB[f'{v}|{cf}|{mode}|{c}|{s}|{m}']
DIFF = {f'{v}|{cf}|{mode}|{s}': {'dCAGR': g(v, cf, s, 'CAGR', mode) - g('ORIG', cf, s, 'CAGR', mode), 'dt_0050': g(v, cf, s, 't_0050', mode) - g('ORIG', cf, s, 't_0050', mode),
                                  'dt_EW': g(v, cf, s, 't_EW', mode) - g('ORIG', cf, s, 't_EW', mode), 'dmaxDD': g(v, cf, s, 'maxDD', mode) - g('ORIG', cf, s, 'maxDD', mode)}
        for v in VERS for cf in CONFS for mode in ('BASE', 'AB') for s in SEGS}
out['DIFF_vs_ORIG'] = DIFF
# 與 main_v2 正式數字核對（ORIG 必須重現）
MV = json.loads((HERE.parent / 'main_v2' / 'results.json').read_text(encoding='utf-8'))
chk = {}
for cf, key in (('ALL|n10', 'ALL|n10'), ('EXFIN|n10', 'EXFIN|n10'), ('EXFIN|n30', 'EXFIN|n30'), ('ALL|n30', 'ALL|n30')):
    for s in SEGS:
        for m in ('CAGR', 't_0050', 't_EW', 'maxDD'):
            a = g('ORIG', cf, s, m); b = MV['OFFICIAL'][f'{key.split("|")[0]}|{key.split("|")[1]}|normal|{s}|{m}']
            chk[f'{cf}|{s}|{m}'] = [a, b]
            assert abs(a - b) < 1e-3, (cf, s, m, a, b)
        a = g('ORIG', cf, s, 'CAGR', 'AB'); b = MV['AB_SUMMARY'][f'{key}|normal|{s}|CAGR']['gapfix']
        assert abs(a - b) < 1e-3, ('AB', cf, s, a, b)
out['ORIG_reproduces_main_v2'] = True

# ---------------- 2. 判定 ----------------
def level(drop, t):
    if drop > 3 or t < 2.0: return 3
    if drop > 1.5 or t < 2.3: return 2
    return 1
VERD = {}
for v in VERS[1:]:
    drop = g('ORIG', 'EXFIN|n10', 'ALL', 'CAGR') - g(v, 'EXFIN|n10', 'ALL', 'CAGR'); t = g(v, 'EXFIN|n10', 'ALL', 't_0050')
    VERD[v] = {'drop_pp': round(drop, 3), 't_0050': t, 'level': level(drop, t)}
lev = max(VERD['R_DEL']['level'], VERD['B05_DEL']['level'])
out['VERDICT'] = {'by_version': VERD, 'official_level': lev, 'worst_case_B10_DEL_level': VERD['B10_DEL']['level'],
                  'rule': RES['PREREG']['verdict_rule']['which_version'],
                  'text': {1: '(1) 正式數字維持，加註「已修補營收存活者偏誤（覆蓋率不足，附界限）」', 2: '(2) 正式數字改用修補版，實盤預期同步下修',
                           3: '(3) 跨循環顯著降為邊緣'}[lev] + ('；但最壞情況（缺營收者全設最高營收）會降為邊緣' if VERD['B10_DEL']['level'] == 3 and lev < 3 else '')}

# ---------------- 3. 實盤預期 ----------------
EXP = {}
for v in VERS:
    for n in (10, 30):
        for c in ('normal', 'stress'):
            for s in ('ALL', 'P1', 'S3', 'S4'):
                a = R[v]['AB'][f'EXFIN|n{n}'][c][s]; ex = a['CAGR'] - a['E50_CAGR']
                mc = R[v]['AB'][f'EXFIN|n{n}'][c + '_mc_p05_p50_p95'][s]
                EXP[f'{v}|n{n}|{c}|{s}'] = {'backtest': a['CAGR'], 'E50': a['E50_CAGR'], 'excess': ex, 'expected': a['E50_CAGR'] + 0.5 * ex,
                                            'expected_mc_p05_p95': [a['E50_CAGR'] + 0.5 * (mc[0] - a['E50_CAGR']), a['E50_CAGR'] + 0.5 * (mc[2] - a['E50_CAGR'])],
                                            'maxDD': a['maxDD'], 't_0050': a['t_0050']}
out['EXPECTATION'] = EXP

# ---------------- 4. P1：名單改變 ----------------
G = pd.read_csv(HERE / 'gap_stockmonths_filled.csv', dtype={'code': str})
lt = pd.Series({c: K.IDX[np.where(K.TRADED[:, j])[0][-1]] for j, c in enumerate(K.COLS) if K.TRADED[:, j].any()})
pend = {c: pd.Timestamp(x) for c, x in O.PP_META['patch_end_by_code'].items()}
REAL_END = {}
for c, d in lt.items():
    if d == pd.Timestamp('2019-05-16') and c in pend:
        REAL_END[c] = pend[c] if pend[c] < pd.Timestamp('2021-04-29') else pd.NaT     # 2021-04 之後仍存活：下市日不明（晚於 2021-04）
    else:
        REAL_END[c] = d if d < K.IDX[-1] - pd.Timedelta(days=30) else pd.NaT
DELISTED = {c for c, d in lt.items() if d < K.IDX[-1] - pd.Timedelta(days=30)}
G['delisted'] = G['code'].isin(DELISTED)
gd = G[~G['foreign'] & G['delisted'] & (G['sig'] >= '2013-01-01') & (G['sig'] < '2019-01-01')].groupby('sig').size()
sig13_18 = [s for s in sorted(set(G['sig'])) if '2013-04-01' <= s < '2019-01-01']
out['P1_gap_domestic_delisted_per_signal_2013_04_2018'] = {'mean': float(gd.reindex(sig13_18, fill_value=0).mean()), 'min': int(gd.reindex(sig13_18, fill_value=0).min()),
                                                           'max': int(gd.reindex(sig13_18, fill_value=0).max())}
gk = G[G['foreign'] & (G['sig'] >= '2013-04-01') & (G['sig'] < '2019-01-01')].groupby('sig').size()
out['P1_gap_foreign_per_signal_2013_04_2018'] = float(gk.reindex(sig13_18, fill_value=0).mean())
FILLED_DOM = set(G.loc[G['fill_del'], 'code']); FILLED_ALL = set(G.loc[G['fill_all'], 'code'])
def changes(va, vb, cf):
    A = {p['sig']: set(p['codes']) for p in PL[va][cf]}; B = {p['sig']: set(p['codes']) for p in PL[vb][cf]}
    rows = []
    for s in sorted(set(A) | set(B)):
        a, b = A.get(s, set()), B.get(s, set())
        rows.append({'sig': s, 'n_changed': len(b - a), 'in': sorted(b - a), 'out': sorted(a - b)})
    df = pd.DataFrame(rows); df['y'] = df['sig'].str[:4]
    by = df.groupby('y')['n_changed'].sum().to_dict()
    ins = pd.Series([c for x in df['in'] for c in x]).value_counts()
    return df, {'total_slots_changed': int(df['n_changed'].sum()), 'n_signals_changed': int((df['n_changed'] > 0).sum()), 'by_year': {k: int(v) for k, v in by.items()},
                'top_in': [(c, SLn.get(c, ''), int(n), c in FILLED_ALL) for c, n in ins.head(25).items()]}
CH = {}
for vb in ('R_DEL', 'R_ALL', 'B05_DEL', 'B10_DEL'):
    for cf in ('EXFIN|n10', 'ALL|n10'):
        df, s = changes('ORIG', vb, cf); CH[f'{vb}|{cf}'] = s
        if vb == 'R_DEL' and cf == 'EXFIN|n10': CHDF = df
out['P1_list_changes_vs_ORIG'] = CH

# ---------------- 5. P3：新進股的持有期報酬 ----------------
def entrant_returns(va, vb, cf, codeset=None):
    A = {p['sig']: set(p['codes']) for p in PL[va][cf]}
    rows = []
    for p in PL[vb][cf]:
        s = p['sig']; a = A.get(s, set())
        r = dict(zip(p['codes'], p['ret']))
        new = [c for c in p['codes'] if c not in a and (codeset is None or c in codeset)]
        oth = [c for c in p['codes'] if c in a]
        if not new or not oth: continue
        for c in new:
            rows.append({'sig': s, 'code': c, 'ret': r[c], 'others_mean': float(np.mean([r[x] for x in oth])), 'filled': c in FILLED_ALL,
                         'delisted': c in DELISTED, 'foreign': c in O.FOREIGN})
    df = pd.DataFrame(rows)
    if df.empty: return df, {}
    df['ex'] = df['ret'] - df['others_mean']
    cl = df.groupby('sig')['ex'].mean()
    return df, {'n_events': int(len(df)), 'n_months': int(len(cl)), 'mean_ex_pp_event': float(df['ex'].mean() * 100), 'mean_ex_pp_month': float(cl.mean() * 100),
                't_by_month': float(K.tstat(cl)), 'median_ex_pp': float(df['ex'].median() * 100), 'share_negative': float((df['ex'] < 0).mean())}
P3 = {}
for vb, cf in (('R_DEL', 'EXFIN|n10'), ('R_DEL', 'ALL|n10'), ('R_ALL', 'EXFIN|n10'), ('R_DEL', 'EXFIN|n30')):
    df, s = entrant_returns('ORIG', vb, cf); P3[f'{vb}|{cf}|all_entrants'] = s
    df2, s2 = entrant_returns('ORIG', vb, cf, FILLED_ALL); P3[f'{vb}|{cf}|filled_entrants'] = s2
    if vb == 'R_DEL' and cf == 'EXFIN|n10':
        P3['R_DEL|EXFIN|n10|filled_entrant_list'] = df2.assign(name=df2['code'].map(lambda c: SLn.get(c, ''))).round(4).to_dict(orient='records') if not df2.empty else []
    # 被擠掉的（ORIG 有、修補版沒有）在 ORIG 裡的報酬
    dfo, so = entrant_returns(vb, 'ORIG', cf); P3[f'{vb}|{cf}|displaced_in_ORIG'] = so
out['P3'] = P3
# 單股貢獻最差 20 名
def contrib(v, cf, a='2006-04-01', b='2026-10-12'):
    con = {}
    for p in PL[v][cf]:
        if not (a <= p['sig'] <= b): continue
        k = len(p['codes'])
        for c, r in zip(p['codes'], p['ret']): con[c] = con.get(c, 0.0) + r / k
    return pd.Series(con).sort_values()
W = {}
for v in ('R_DEL', 'R_ALL'):
    for cf in ('EXFIN|n10', 'ALL|n10'):
        cs = contrib(v, cf)
        w20 = cs.head(20)
        W[f'{v}|{cf}'] = {'worst20': [(c, SLn.get(c, ''), round(float(x) * 100, 2), c in FILLED_ALL, c in DELISTED) for c, x in w20.items()],
                          'n_filled_in_worst20': int(sum(c in FILLED_ALL for c in w20.index)),
                          'best10': [(c, SLn.get(c, ''), round(float(x) * 100, 2), c in FILLED_ALL) for c, x in cs.tail(10)[::-1].items()],
                          'filled_total_contrib_pp': round(float(cs[[c for c in cs.index if c in FILLED_ALL]].sum()) * 100, 2)}
out['P3_contrib'] = W

# ---------------- 6. P4：同池等權 ----------------
P4 = {}
for v in VERS:
    for u in ('ALL', 'EXFIN'):
        P4[f'{v}|{u}'] = {s: R[v]['BASE'][f'{u}|EW'][s]['CAGR'] for s in SEGS}
out['P4_EW_CAGR'] = P4
out['P4_t_EW_change'] = {f'{v}|{cf}': {s: DIFF[f'{v}|{cf}|BASE|{s}']['dt_EW'] for s in SEGS} for v in ('R_DEL', 'R_ALL', 'B05_DEL', 'B10_DEL') for cf in ('EXFIN|n10', 'ALL|n10')}

# ---------------- 7. P5：下市前營收暴增股在入選股與池中的比例 ----------------
P5 = {}
for vname in ('R_DEL', 'R_ALL'):
    Y, Y5 = O.load_yl(vname)
    for univ in ('EXFIN', 'ALL'):
        S = V.V2('gapfix', Y, Y5, exfin=(univ == 'EXFIN'))
        picks = {p['sig']: set(p['codes']) for p in PL[vname][f'{univ}|n10']}
        pool_n = pool_x = pick_n = pick_x = 0; pool_d = pick_d = 0; ex_list = []
        pool_hi = pick_hi = 0
        for d in S.sig_days:
            if not (pd.Timestamp('2015-04-01') <= d <= pd.Timestamp('2019-12-31')): continue
            cs = S.cross_section(d); el = np.where(cs['el'])[0]; s = str(d.date())
            pk = picks.get(s, set())
            for j in el:
                c = K.COLS[j]; L = REAL_END.get(c, pd.NaT)
                pre = pd.notna(L) and (d < L <= d + pd.DateOffset(months=12))
                hi = cs['y3'][j] > 100
                x = pre and hi
                pool_n += 1; pool_x += x; pool_d += pre; pool_hi += hi
                if c in pk:
                    pick_n += 1; pick_x += x; pick_d += pre; pick_hi += hi
                    if pre: ex_list.append((s, c, SLn.get(c, ''), round(float(cs['y3'][j]), 1), str(L.date())))
        P5[f'{vname}|{univ}|n10'] = {'window': '2015-04～2019-12 訊號（修補資料可用期）', 'pool_stockmonths': pool_n, 'pick_stockmonths': pick_n,
                                     'share_pool_pre_delist_12m': pool_d / pool_n, 'share_pick_pre_delist_12m': pick_d / max(pick_n, 1),
                                     'share_pool_pre_delist_and_y3gt100': pool_x / pool_n, 'share_pick_pre_delist_and_y3gt100': pick_x / max(pick_n, 1),
                                     'share_pool_y3gt100': pool_hi / pool_n, 'share_pick_y3gt100': pick_hi / max(pick_n, 1),
                                     'n_pick_pre_delist': pick_d, 'n_pick_pre_delist_y3gt100': pick_x, 'picked_pre_delist': ex_list}
out['P5'] = P5

# ---------------- 8. 價格修補 ----------------
out['PRICE_PATCH'] = {f'{a}->{b}|{cf}|{mode}': {s: {'dCAGR': g(b, cf, s, 'CAGR', mode) - g(a, cf, s, 'CAGR', mode), 'dt_0050': g(b, cf, s, 't_0050', mode) - g(a, cf, s, 't_0050', mode)} for s in SEGS}
                      for a, b in (('ORIG', 'PX_ORIG'), ('R_DEL', 'PX_DEL')) for cf in ('EXFIN|n10', 'ALL|n10', 'EXFIN|n30') for mode in ('BASE', 'AB')}
out['PRICE_PATCH_frozen'] = {v: R[v]['BASE']['EXFIN|n10']['frozen'][:10] for v in ('R_DEL', 'PX_DEL')}

# ---------------- 9. 穩健性 ----------------
ROB = {}
for v in ('ORIG', 'R_DEL', 'R_ALL'):
    for cf in ('EXFIN|n10', 'ALL|n10'):
        ROB[f'{v}|{cf}|yearly'] = R[v]['BASE'][cf]['normal']['ALL'].get('yearly')
        ROB[f'{v}|{cf}|contrib_ALL'] = R[v]['BASE'][cf]['contrib']['ALL']
        ROB[f'{v}|{cf}|contrib_S2'] = R[v]['BASE'][cf]['contrib']['S2']
yo, yd, ya = (R[v]['BASE']['EXFIN|n10']['normal']['ALL']['yearly'] for v in ('ORIG', 'R_DEL', 'R_ALL'))
ROB['yearly_diff_EXFIN_n10'] = {y: {'ORIG': yo[y], 'R_DEL': yd[y], 'R_ALL': ya[y], 'd_DEL': round(yd[y] - yo[y], 1), 'd_ALL': round(ya[y] - yo[y], 1)} for y in yo}
ind_in = pd.Series([SLi.get(c, '（下市／無）') if isinstance(SLi.get(c, None), str) else '（下市／無）' for x in CHDF['in'] for c in x]).value_counts().to_dict()
ind_out = pd.Series([SLi.get(c, '（下市／無）') if isinstance(SLi.get(c, None), str) else '（下市／無）' for x in CHDF['out'] for c in x]).value_counts().to_dict()
ROB['industry_changed_slots_R_DEL_EXFIN_n10'] = {'in': ind_in, 'out': ind_out}
out['ROBUST'] = ROB
out['BOUND_gap_used_mean'] = {v: {u: R[v]['BASE'].get(f'{u}|gap_used_mean') for u in ('ALL', 'EXFIN')} for v in ('B05_DEL', 'B10_DEL', 'B05_ALL', 'B10_ALL')}

RES['RESULTS'] = out
RES['p6'] = {v: json.loads((HERE / f'p6_{v}.json').read_text(encoding='utf-8')) for v in ('ORIG', 'R_DEL')}
RES['explore'] = json.loads((HERE / 'explore.json').read_text(encoding='utf-8'))
RES['data_freeze'] = {k: v for k, v in O.FREEZE.items() if k not in ('foreign_codes',)}
RES['price_patch_meta'] = {k: v for k, v in O.PP_META.items() if k != 'src_sha256'}
RJ.write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x)), encoding='utf-8')

# ---------------- 印表 ----------------
f1 = lambda x: f'{x:.1f}'; f2 = lambda x: f'{x:.2f}'
for cf in ('EXFIN|n10', 'ALL|n10', 'EXFIN|n30'):
    for mode in ('BASE', 'AB'):
        print(f'\n### {cf} {mode} 一般成本：年化%（對 ORIG 差）／t0050／tEW／回撤')
        print('| 版本 | ' + ' | '.join(SEGS) + ' |'); print('|---' * (len(SEGS) + 1) + '|')
        for v in VERS:
            cells = []
            for s in SEGS:
                d = DIFF[f'{v}|{cf}|{mode}|{s}']
                cells.append(f"{f1(g(v, cf, s, 'CAGR', mode))}（{d['dCAGR']:+.1f}）／{f2(g(v, cf, s, 't_0050', mode))}／{f2(g(v, cf, s, 't_EW', mode))}／{f1(g(v, cf, s, 'maxDD', mode))}")
            print(f'| {v} | ' + ' | '.join(cells) + ' |')
print('\n日曆月 t（BASE 一般；對 0050／對等權）')
for cf in ('EXFIN|n10', 'ALL|n10', 'EXFIN|n30'):
    for v in ('ORIG', 'R_DEL', 'B05_DEL', 'B10_DEL', 'R_ALL'):
        print(cf, v, {s: (g(v, cf, s, 't_0050_cal'), g(v, cf, s, 't_EW_cal')) for s in SEGS})
print('\n壓力成本 BASE（年化／t0050）')
for cf in ('EXFIN|n10', 'ALL|n10', 'EXFIN|n30'):
    for v in ('ORIG', 'R_DEL', 'B05_DEL', 'B10_DEL', 'R_ALL'):
        print(cf, v, {s: (round(g(v, cf, s, 'CAGR', 'BASE', 'stress'), 1), g(v, cf, s, 't_0050', 'BASE', 'stress')) for s in SEGS})
print('\nVERDICT', json.dumps(out['VERDICT'], ensure_ascii=False))
print('\nEXPECTATION n10 normal', {v: {s: round(EXP[f'{v}|n10|normal|{s}']['expected'], 1) for s in ('ALL', 'P1', 'S3', 'S4')} for v in VERS})
print('EXPECTATION n10 stress', {v: {s: round(EXP[f'{v}|n10|stress|{s}']['expected'], 1) for s in ('ALL', 'S3')} for v in ('ORIG', 'R_DEL', 'B05_DEL', 'B10_DEL', 'R_ALL')})
print('EXPECTATION n30 normal', {v: {s: round(EXP[f'{v}|n30|normal|{s}']['expected'], 1) for s in ('ALL', 'P1', 'S3', 'S4')} for v in ('ORIG', 'R_DEL', 'B05_DEL', 'B10_DEL', 'R_ALL')})
print('\nP1', json.dumps({k: out[k] for k in ('P1_gap_domestic_delisted_per_signal_2013_04_2018', 'P1_gap_foreign_per_signal_2013_04_2018')}, ensure_ascii=False))
for k, v in CH.items(): print('P1', k, json.dumps(v, ensure_ascii=False))
print('\nP3', json.dumps({k: v for k, v in P3.items() if 'list' not in k}, ensure_ascii=False, indent=0))
print('P3 list', json.dumps(P3.get('R_DEL|EXFIN|n10|filled_entrant_list'), ensure_ascii=False))
for k, v in W.items(): print('CONTRIB', k, json.dumps(v, ensure_ascii=False))
print('\nP4', json.dumps({k: v for k, v in P4.items() if k.split('|')[0] in ('ORIG', 'R_DEL', 'R_ALL', 'B05_DEL', 'B10_DEL')}, ensure_ascii=False))
print('P4 dtEW', json.dumps(out['P4_t_EW_change'], ensure_ascii=False))
print('\nP5', json.dumps(P5, ensure_ascii=False, indent=0))
print('\nPRICE', json.dumps(out['PRICE_PATCH'], ensure_ascii=False))
print('\nYEARLY', json.dumps(ROB['yearly_diff_EXFIN_n10'], ensure_ascii=False))
print('IND', json.dumps(ROB['industry_changed_slots_R_DEL_EXFIN_n10'], ensure_ascii=False))
for v in ('ORIG', 'R_DEL', 'R_ALL'):
    print('TOPC', v, json.dumps({k: {kk: ROB[f'{v}|EXFIN|n10|{k}'][kk] for kk in ('top_code', 'top_name', 'top_contrib_pp', 'gross_CAGR_approx', 'gross_CAGR_ex_top', 'ex_top_drop_pp')} for k in ('contrib_ALL', 'contrib_S2')}, ensure_ascii=False))
print('GAPUSED', out['BOUND_gap_used_mean'])
print('P6', json.dumps(RES['p6'], ensure_ascii=False)[:2500])
