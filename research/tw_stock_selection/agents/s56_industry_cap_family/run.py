#!/usr/bin/env python3
"""S56 同產業持股上限家族：同一 PIT 產業（證交所產業別）最多持有 {2,3,4} 檔 × {遞補, 不遞補}，已持有計入上限、不強制賣；E0 出場不動。
規格 SPEC.json、判定 PREREG.json（寫於任何非現行格報酬計算之前，sha256 啟動時核對）。
用法：python3 -I -B run.py [--quick]（--quick 只跑驗收 A2～A5）
載入與訊號逐行照 agents/s54_market_filter_family/run.py（＝s48）；引擎用 inspect 副本＋字串替換（原檔不動）。暫存 scratchpad/s56/（結束刪除）。"""
import sys, json, time, resource, gc, hashlib, inspect
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s56')
CACHE = TMP / 'cache'
CACHE.mkdir(parents=True, exist_ok=True)
QUICK = '--quick' in sys.argv
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for f in ('SPEC', 'PREREG'):
    assert sha(HERE / f'{f}.json') == (HERE / f'{f}.sha256').read_text().split()[0], f'{f}.json 與登記的 sha256 不符'

sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402
for mod, attr in ((s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                  (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(mod, attr, CACHE)
assert cslib.DEADLINE_FIX is False and cslib.INS15_MASK is False
cslib.EXRIGHT_FIX = True

# ---- 引擎副本：買進步驟加同產業上限（cap_n）與遞補（refill）；原檔不動
_SRC0 = inspect.getsource(s10lib.simulate_s10)
_REPL = [
    ("def simulate_s10(", "def simulate_s56("),
    ("exec_mode='open', trail=None, trail_daily=True, requal=None):",
     "exec_mode='open', trail=None, trail_daily=True, requal=None, cap_n=None, capind=None, reserve=None, refill=False):"),
    ("""        for j in order:
            target = u
""",
     """        def _capgen(order):
            if cap_n is None:
                yield from order
                return
            ind = capind[t - 1]

            def full(j):
                g = ind[j]
                if g < 0:
                    return False
                return sum(1 for kk in pos if ind[kk] == g) >= cap_n
            nb = 0
            for j in order:
                if full(j):
                    nb += 1; st['cap_blk'] = st.get('cap_blk', 0) + 1
                    st.setdefault('cap_ev', []).append((int(t), int(j), int(ind[j])))
                    continue
                yield j
            if refill and nb > 0:
                for j in reserve.get(t - 1, ()):
                    if nb <= 0:
                        break
                    if j in pos or j in sold_today or not univ[t - 1, j] or full(j):
                        continue
                    n0 = len(pos)
                    yield j
                    if len(pos) > n0:
                        nb -= 1; st['refill_buy'] = st.get('refill_buy', 0) + 1
                        st.setdefault('refill_ev', []).append((int(t), int(j), int(ind[j])))
        for j in _capgen(order):
            target = u
"""),
]
_SRC = _SRC0
for _a, _b in _REPL:
    assert _SRC.count(_a) == 1, f'替換點不唯一或不存在：{_a[:40]}'
    _SRC = _SRC.replace(_a, _b)
_G = dict(s10lib.__dict__)
exec(compile(_SRC, '<s56 simulate_s56>', 'exec'), _G)
simulate_s56 = _G['simulate_s56']
SRC_SHA = dict(simulate_s10=hashlib.sha256(_SRC0.encode()).hexdigest(), simulate_s56=hashlib.sha256(_SRC.encode()).hexdigest())

P = s20lib.load()
assert P.exright_fix
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), '面板不符'
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
F = s17lib.rev_family(P, defs=['STD'])
s17lib.e50_fix_suspension(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
S0 = F['S']['STD']; months = list(S0.index)
del F; gc.collect()
cslib.DEADLINE_FIX = True; cslib.INS15_MASK = True
AV = cslib.rev_avail_days(P, months, True)
RT = dict(S=S0, av=AV)
print('載入', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

# ---- PIT 產業（證交所產業別單層）：INDD[t, j]＝P.IDX[t] 當天的產業代碼；-1＝缺值
IND_CSV = AG / 'o_lc_sector_rotation_pit' / 'raw' / 'industry_pit.csv'
MISSING = {'未分類', '', 'nan'}
_df = pd.read_csv(IND_CSV, dtype=str, keep_default_na=False)
_df['industry'] = _df['industry'].str.strip()
IND_NAMES = sorted(set(_df['industry']) - MISSING)
_nid = {n: i for i, n in enumerate(IND_NAMES)}
_col = {str(c): j for j, c in enumerate(P.COLS)}
INDD = np.full((P.T, P.N), -1, np.int16)
_idx = P.IDX.values
for code, s_, e_, ind in _df[['code', 'start', 'end', 'industry']].itertuples(index=False):
    j = _col.get(str(code))
    if j is None or ind in MISSING:
        continue
    a = int(np.searchsorted(_idx, np.datetime64(pd.Timestamp(s_)), side='left'))
    b = P.T if e_ == '' else int(np.searchsorted(_idx, np.datetime64(pd.Timestamp(e_)), side='left'))
    INDD[a:b, j] = _nid[ind]
del _df

lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TB = P.IDX[-1]; D19 = lastb('2019-01-01'); FULL0 = P.IDX[P.t0]
iD19 = int(P.IDX.get_loc(D19))
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
CUR = 'NONE'
RUN = [(2, 'R'), (2, 'N'), (3, 'R'), (3, 'N'), (4, 'R'), (4, 'N')]
cells = [CUR] if QUICK else [CUR] + [f'C{c}_{r}' for c, r in RUN]
SEGK = list(s17lib.SEGMENTS)
NIND = len(IND_NAMES)


def build_reserve(sg, shift):
    """照 make_signal_s10 同一迴圈重建每個訊號列的 o_ma；回傳 reserve[k]＝L_MA 第 11～20 名中 RET20 > 10% 者（SURGE 序），並驗 F1（A5）。"""
    res = {}; ks = []; cov = []
    info = {int(x['k']): x for x in sg['info']}
    for m in RT['S'].index:
        k0 = RT['av'][m]
        if k0 >= P.T or k0 < 61:
            continue
        s = RT['S'].loc[m].to_numpy()
        if int((U[k0] & np.isfinite(s)).sum()) < 50:
            continue
        k = k0 + shift
        if k >= P.T:
            continue
        s = cslib.ins15_mask_row(P, m, k, s)
        fin = U[k] & np.isfinite(s)
        ok_ma = fin & P.F_MA60[k]
        o_ma = np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]
        r20 = P.RET20[k]
        pool = o_ma[:10]
        tgt = pool[r20[pool] > 0.10][:10]
        assert [str(P.COLS[j]) for j in tgt] == info[int(k)]['targets'], ('A5 重建 F1 失敗', str(m), k)
        res[int(k)] = [int(j) for j in o_ma[10:20] if r20[j] > 0.10]
        ks.append(int(k))
        cov.append(float((INDD[k, o_ma[:20]] >= 0).mean()) if len(o_ma) else 1.0)
    assert ks == [int(x['k']) for x in sg['info']], '訊號列不符'
    return res, float(np.mean(cov))


def conc(lg):
    """每日收盤持股依當天 PIT 產業分組的最大檔數：2019 起平均、全期平均、≥ 3 檔天數占比（2019 起）。"""
    cnt = np.zeros((P.T, NIND + 1), np.int16)
    for j, a, b, why in lg[['j', 'ent_t', 'exit_t', 'why']].itertuples(index=False):
        end = P.T if why == 'open_end' else int(b)
        d = np.arange(int(a), end)
        g = INDD[d, int(j)].astype(int)
        g = np.where(g >= 0, g, NIND)                      # 缺值放最後一欄，不計入最大值
        np.add.at(cnt, (d, g), 1)
    mx = cnt[:, :NIND].max(axis=1)
    m19 = mx[iD19 + 1:]; mf = mx[P.t0 + 1:]
    return dict(maxsame_2019=float(m19.mean()), maxsame_full=float(mf.mean()), frac_ge3_2019=float((m19 >= 3).mean()),
                frac_ge4_2019=float((m19 >= 4).mean()))


def metrics(nav, lg, st):
    cl = lg[lg['why'] != 'open_end']
    m = dict(since2019=stlib.cagr(nav, D19, TB), full=stlib.cagr(nav, FULL0, TB), maxdd_full=stlib.maxdd(nav, FULL0, TB),
             maxdd_2019=stlib.maxdd(nav, D19, TB), n_closed=int(len(cl)), n_buy=int(st['n_buy']), avg_npos=float(st['avg_npos']),
             cap_blk=int(st.get('cap_blk', 0)), refill_buy=int(st.get('refill_buy', 0)))
    m['calmar'] = m['since2019'] / abs(m['maxdd_full'])
    m['seg_dd'] = s17lib.seg_dd(nav)
    m.update(conc(lg))
    return m


def ev_summary(st):
    o = {}
    for key in ('cap_ev', 'refill_ev'):
        ev = st.get(key, [])
        o[key + '_n'] = len(ev)
        o[key + '_by_industry'] = pd.Series([IND_NAMES[g] for _, _, g in ev]).value_counts().head(8).to_dict() if ev else {}
        o[key + '_list'] = [(str(P.IDX[t].date()), str(P.COLS[j]), IND_NAMES[g]) for t, j, g in ev]
    return o


R = {c: dict(paths={}) for c in cells}
ACC = dict(A3={}, A4={}, A5_ind_coverage_top20={})
EV = {}
for sft in range(4):
    sg = s10lib.make_signal_s10(P, U, RT, shift=sft, depth=10, mom_lo=0.10, nsel=10)
    S2, ekw = s10lib.variant_S(sg, 'E0')
    RES, cov = build_reserve(sg, sft)
    ACC['A5_ind_coverage_top20'][sft] = cov
    for c in cells:
        for cap in CAPS:
            kw = dict(cost=COSTS['strict'], capital=cap, **CELL, **ekw)
            if c == CUR:
                nav, lg, st = s10lib.simulate_s10(P, S2, U, **kw)
            else:
                n, r = int(c[1]), c[3] == 'R'
                nav, lg, st = simulate_s56(P, S2, U, cap_n=n, capind=INDD, reserve=RES, refill=r, **kw)
            m = metrics(nav, lg, st)
            R[c]['paths'][f'{int(cap/1e4)}萬|+{sft}'] = m
            if c == CUR and cap == 1e6:                    # 驗收 A3／A4（每個位移都驗）
                nav2, lg2, _ = simulate_s56(P, S2, U, **kw)
                ACC['A3'][sft] = bool(np.array_equal(nav.to_numpy(), nav2.to_numpy(), equal_nan=True) and len(lg) == len(lg2))
                nav3, lg3, st3 = simulate_s56(P, S2, U, cap_n=99, capind=INDD, reserve=RES, refill=True, **kw)
                ACC['A4'][sft] = bool(np.array_equal(nav.to_numpy(), nav3.to_numpy(), equal_nan=True) and len(lg) == len(lg3)
                                      and st3.get('cap_blk', 0) == 0)
                assert ACC['A3'][sft] and ACC['A4'][sft], f'驗收 A3／A4 失敗（位移 {sft}）'
                del nav2, lg2, nav3, lg3, st3
            if sft == 0 and cap == 1e6:
                R[c]['main'] = m
                if c != CUR:
                    EV[c] = ev_summary(st)
            del nav, lg, st
    del sg, S2, RES; gc.collect()
    print('位移', sft, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

ref = json.loads((AG / 'deadline_engine_fix' / 'out' / 'results.json').read_text())
cur = R[CUR]
acc = dict(ACC, main_since2019=cur['main']['since2019'], ref_since2019=ref['standard']['ALL']['since2019'],
           main_n_closed=cur['main']['n_closed'], ref_n_closed=ref['standard']['ALL']['n_closed'])
acc['main_bitwise'] = bool(acc['main_since2019'] == acc['ref_since2019'] and acc['main_n_closed'] == acc['ref_n_closed']
                           and cur['main']['full'] == ref['standard']['ALL']['full'] and cur['main']['maxdd_full'] == ref['standard']['ALL']['maxdd_full'])
acc['paths_bitwise'] = bool(all(cur['paths'][p]['since2019'] == ref['paths']['ALL'][p]['since2019'] for p in ref['paths']['ALL']))
acc['paths_median'] = float(np.median([cur['paths'][p]['since2019'] for p in cur['paths']]))
acc['ok'] = acc['main_bitwise'] and acc['paths_bitwise'] and all(ACC['A3'].values()) and all(ACC['A4'].values())
print('驗收', json.dumps(acc, ensure_ascii=False), flush=True)
assert acc['ok'], '驗收失敗'
if QUICK:
    print('quick ok', f'{time.time()-t0:.0f}s', rss(), 'MB'); sys.exit(0)

keys = list(cur['paths'])
for c, o in R.items():
    g = lambda f: np.array([o['paths'][p][f] for p in keys])
    gs = lambda s: np.array([o['paths'][p]['seg_dd'][s] for p in keys])
    o['path'] = dict(median=float(np.median(g('since2019'))), min=float(g('since2019').min()), max=float(g('since2019').max()),
                     maxdd_full_median=float(np.median(g('maxdd_full'))), maxdd_2019_median=float(np.median(g('maxdd_2019'))),
                     calmar_median=float(np.median(g('calmar'))), n_closed_median=float(np.median(g('n_closed'))),
                     avg_npos_median=float(np.median(g('avg_npos'))), maxsame_2019_median=float(np.median(g('maxsame_2019'))),
                     maxsame_full_median=float(np.median(g('maxsame_full'))), frac_ge3_2019_median=float(np.median(g('frac_ge3_2019'))),
                     cap_blk_median=float(np.median(g('cap_blk'))), refill_buy_median=float(np.median(g('refill_buy'))),
                     seg_dd_median={s: float(np.median(gs(s))) for s in SEGK})
    if c == CUR:
        o['label'] = '現行'; continue
    gc_ = lambda f: np.array([cur['paths'][p][f] for p in keys])
    dC = g('calmar') - gc_('calmar')
    dR = (g('since2019') - gc_('since2019')) * 100
    dD = (g('maxdd_full') - gc_('maxdd_full')) * 100
    dS = {s: (gs(s) - np.array([cur['paths'][p]['seg_dd'][s] for p in keys])) * 100 for s in SEGK}
    nC, mC = int((dC > 1e-12).sum()), float(np.median(dC))
    lab = '較好' if (mC >= 0.10 and nC >= 15) else ('較差' if (mC <= -0.10 and nC <= 5) else '差不多')
    mR, nR = float(np.median(dR)), int((dR > 1e-9).sum())
    clear = bool(lab == '較好' and mR >= -3)
    labR = '比現行好' if (nR >= 15 and mR >= 2) else ('比現行差' if (nR <= 5 and mR <= -2) else '差不多')
    o['vs_current'] = dict(calmar_median_diff=mC, calmar_n_pos=nC, label=lab, clear_better=clear,
                           since2019_median_pp=mR, since2019_n_pos=nR, return_label_S17=labR,
                           maxdd_full_median_pp=float(np.median(dD)), maxdd_n_shallower=int((dD > 1e-9).sum()),
                           maxsame_2019_median_diff=float(np.median(g('maxsame_2019') - gc_('maxsame_2019'))),
                           seg_dd={s: dict(median_pp=float(np.median(v)), n_shallower=int((v > 1e-9).sum())) for s, v in dS.items()})
    o['events_main'] = EV[c]

fam = [CUR] + [f'C{c}_{r}' for c, r in RUN]
meds = {c: R[c]['path']['median'] for c in fam}
fm = float(np.median(list(meds.values())))
p3 = (meds[CUR] - fm) * 100
s17 = dict(family=fam, family_median=fm, current_median=meds[CUR], current_minus_median_pp=p3,
           current_rank_return=int(1 + sum(v > meds[CUR] for v in meds.values())),
           current_rank_calmar=int(1 + sum(R[c]['path']['calmar_median'] > R[CUR]['path']['calmar_median'] for c in fam)),
           verdict='現行偏樂觀（被挑過）' if p3 >= 3 else '沒被挑過')
rf_cmp = {}
for n in (2, 3, 4):
    a, b = R[f'C{n}_R'], R[f'C{n}_N']
    rf_cmp[f'C{n}'] = dict(calmar=float(np.median([a['paths'][p]['calmar'] - b['paths'][p]['calmar'] for p in keys])),
                           since2019_pp=float(np.median([(a['paths'][p]['since2019'] - b['paths'][p]['since2019']) * 100 for p in keys])),
                           maxdd_pp=float(np.median([(a['paths'][p]['maxdd_full'] - b['paths'][p]['maxdd_full']) * 100 for p in keys])))
res = dict(spec_sha256=sha(HERE / 'SPEC.json'), prereg_sha256=sha(HERE / 'PREREG.json'), run_sha256=sha(Path(__file__)), src_sha256=SRC_SHA,
           flags=dict(EXRIGHT_FIX=cslib.EXRIGHT_FIX, DEADLINE_FIX=cslib.DEADLINE_FIX, INS15_MASK=cslib.INS15_MASK),
           industry_layer='證交所上市產業別（單層）', industry_names=IND_NAMES,
           acceptance=acc, s17_return_rule=s17, refill_minus_norefill=rf_cmp,
           cells=R, path_keys=keys, runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
for c in fam:
    o = R[c]; pth = o['path']; v = o.get('vs_current', {})
    print(c, f"主跑 {o['main']['since2019']*100:.2f} 中位 {pth['median']*100:.2f} DD {pth['maxdd_full_median']*100:.2f} Cal {pth['calmar_median']:.3f}",
          f"maxsame {pth['maxsame_2019_median']:.2f} ge3 {pth['frac_ge3_2019_median']:.3f} blk {pth['cap_blk_median']} rf {pth['refill_buy_median']}",
          {s: round(x * 100, 2) for s, x in pth['seg_dd_median'].items()}, json.dumps({k: v[k] for k in v if k != 'seg_dd'}, ensure_ascii=False),
          {s: (round(x['median_pp'], 2), x['n_shallower']) for s, x in v.get('seg_dd', {}).items()})
print(json.dumps(s17, ensure_ascii=False)); print(json.dumps(rf_cmp, ensure_ascii=False))
print('done', res['runtime_s'], 's', res['maxrss_mb'], 'MB')
