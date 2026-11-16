#!/usr/bin/env python3
"""S25 獨立複製 B：因子 12 個月趨勢 ≤ 0 時衛星新買減半／不買。
驗收 A0、A1 → 訓練期 4 規則＋CUR × 20 路徑（全期連續切訓練窗）→ 自行依 PREREG 選規則凍結 → 驗證期（重新起跑）× 20 路徑、V1～V4 →
其餘 3 條驗證期事後描述 → 隨機對照（描述）。輸出 scratchpad/s25rep/b.json。
依 PREREG.json 文字自寫；未讀、未 import、未執行 s25lib.py／prep.py／describe.py／train.py／validate.py／posthoc.py。
python3 -I -B rep_s25_b.py"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rep_s25_lib as L
from rep_s25_lib import s10lib, s17lib, s2lib, stlib, COSTS

t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, INNOV, RT, fix = L.load_panel()
del INNOV
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'NEARHI', 'V60'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
TRD = pd.read_csv(L.REP / 'trend12.csv', index_col=0); TRD.index = pd.PeriodIndex(TRD.index, freq='M')
RULES = {'RMOM_HALF': ('RMOM', 'half'), 'RMOM_STOP': ('RMOM', 'stop'), 'MOM_HALF': ('MOM', 'half'), 'MOM_STOP': ('MOM', 'stop')}
CANDS = list(RULES); CFG = ['CUR'] + CANDS
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE = P.IDX[P.t0], P.IDX[-1]
TA, TB, T18 = lastb('2016-02-01'), lastb('2022-01-01'), lastb('2019-01-01')
D19 = T18
cg, mdd, med = stlib.cagr, stlib.maxdd, (lambda v: float(np.median(v)))
SEGV = ('2022', '2024-07~2025-04')
keyof = lambda cap, s: f'{int(cap/1e4)}萬|+{s}'


def sim(S, cap, fr=False, block=None):
    if fr:
        S = L.fresh(S)
    return s10lib.simulate_s10(P, S, U, sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'], capital=cap, block=block)


def trig_rows(S0, rule, thr=0.0):
    """觸發的訊號列：訊號日所在日曆月份的『前一月底已知 12 月趨勢』≤ thr（缺值不觸發）。回傳 (set(k), 營收月份清單)。"""
    var = RULES[rule][0]
    ks, ms = set(), []
    for x in S0['info']:
        sm = pd.Period(x['sig_date'][:7], 'M')
        v = TRD[var].get(sm, np.nan)
        if np.isfinite(v) and v <= thr:
            ks.add(int(x['k'])); ms.append(x['month'])
    return ks, ms


def build(S0, c, thr=0.0):
    if c == 'CUR':
        return S0, 0, []
    ks, ms = trig_rows(S0, c, thr)
    S2, dr = L.apply_rule(P, S0, ks, RULES[c][1])
    return S2, dr, ms


res = {'note': '獨立複製：未 import／執行／讀取 s25lib.py、prep.py、describe.py、train.py、validate.py、posthoc.py；引擎 s20lib→s17lib→s10lib 預設旗標；觸發、規則、選擇、判定自寫'}
CONT = {c: {} for c in CFG}; S0s = {}; eff = {c: {} for c in CANDS}
for s in range(4):
    S0 = s10lib.make_signal_s10(P, U, RT, shift=s); S0s[s] = S0
    if s == 0:
        navc, lgc, _ = sim(S0, 1e6)
        a0 = dict(since2019=cg(navc, D19, DE), full=cg(navc, D0, DE), maxdd=mdd(navc, D0, DE))
        a0['ok'] = bool(abs(a0['since2019'] - 0.4813) < 0.00025 and abs(a0['full'] - 0.3408) < 0.00025 and abs(a0['maxdd'] + 0.3340) < 0.00025)
        Sn, _, _ = build(S0, 'RMOM_STOP', thr=-np.inf)
        nava, lga, _ = sim(Sn, 1e6)
        cols = ['code', 'ent_t', 'exit_t', 'why']
        a1 = dict(trades_identical=bool(len(lga) == len(lgc) and (lga[cols].reset_index(drop=True) == lgc[cols].reset_index(drop=True)).all().all()),
                  nav_identical=bool(np.allclose(nava.to_numpy(), navc.to_numpy(), rtol=0, atol=1e-6)))
        res['acceptance'] = dict(A0=a0, A1_never_trigger=a1)
        print('驗收', res['acceptance'], flush=True)
        del navc, lgc, nava, lga, Sn
    for c in CFG:
        Sx, dr, ms = build(S0, c)
        if c != 'CUR':
            mtr = [m for m in ms if pd.Period(m, 'M') <= L.TRAIN_LAST]
            eff[c][f'+{s}'] = dict(months_triggered_train=len(mtr), f1_dropped_train=None, trig_months_train=mtr if s == 0 else None,
                                   months_triggered_valid=len(ms) - len(mtr), trig_months_valid=[m for m in ms if pd.Period(m, 'M') >= L.VALID_FIRST] if s == 0 else None)
            # 訓練期少買檔數
            ks, _ = trig_rows(S0, c)
            dtr = 0
            for x in S0['info']:
                if int(x['k']) in ks and pd.Period(x['month'], 'M') <= L.TRAIN_LAST:
                    n = len(x['targets']); dtr += (n - int(np.ceil(n / 2))) if RULES[c][1] == 'half' else n
            eff[c][f'+{s}']['f1_dropped_train'] = dtr
        for cap in CAPS:
            nav, lg, _ = sim(Sx, cap)
            sg = s17lib.seg_dd(nav)
            CONT[c][keyof(cap, s)] = dict(train=cg(nav, TA, TB), train_dd=mdd(nav, TA, TB), y1618=cg(nav, TA, T18), y1921=cg(nav, T18, TB),
                                          since2019=cg(nav, D19, DE), full=cg(nav, D0, DE), valid=cg(nav, TB, DE), maxdd=mdd(nav, D0, DE), seg=sg)
            del nav, lg
        del Sx
    gc.collect()
    print('連續 位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
keys = list(CONT['CUR'])


def pair(D, a, b, f, sub=None):
    g = (lambda d: d[f]) if sub is None else (lambda d: d[f][sub])
    d = np.array([g(D[a][k]) - g(D[b][k]) for k in keys]) * 100
    return dict(median_pp=med(d), npos=int((d > 0).sum()), n_ge025=int((d >= 0.25).sum()), min_pp=float(d.min()), max_pp=float(d.max()))


summ = {'CUR': dict(train_median=med([CONT['CUR'][k]['train'] for k in keys]), train_dd_median=med([CONT['CUR'][k]['train_dd'] for k in keys]))}
for c in CANDS:
    p = pair(CONT, c, 'CUR', 'train')
    summ[c] = dict(train_median=med([CONT[c][k]['train'] for k in keys]), score_median_pp=p['median_pp'], n_imp_ge_min=p['n_ge025'], npos=p['npos'],
                   diff_min_pp=p['min_pp'], diff_max_pp=p['max_pp'], dd_diff=pair(CONT, c, 'CUR', 'train_dd'),
                   y1618=pair(CONT, c, 'CUR', 'y1618'), y1921=pair(CONT, c, 'CUR', 'y1921'),
                   months_triggered_shift0=eff[c]['+0']['months_triggered_train'], f1_dropped_shift0=eff[c]['+0']['f1_dropped_train'])
order = sorted(CANDS, key=lambda c: (-summ[c]['score_median_pp'], -summ[c]['n_imp_ge_min'], summ[c]['months_triggered_shift0']))
FZ = order[0]
T0 = bool(summ[FZ]['score_median_pp'] >= 1.0 and summ[FZ]['n_imp_ge_min'] >= 14)
res['train'] = dict(window=[str(TA.date()), str(TB.date())], summary=summ, order=order, frozen_rep=FZ, T0_pass=T0, rule_effect=eff)
print('訓練', {c: (round(summ[c]['score_median_pp'], 3), summ[c]['n_imp_ge_min'], summ[c]['npos']) for c in CANDS}, '凍結', FZ, 'T0', T0, flush=True)

# ---------------------------------------------------------------- 驗證：重新起跑
S0 = S0s[0]
navx = sim(S0, 1e6, fr=True)[0]
ratio = float(navx.loc[TB] / 1e6)
FCAP = {cap: cap / ratio for cap in CAPS}
MAIN, mlg = {}, {}
for c in ('CUR', FZ):
    Sx, _, _ = build(S0, c)
    nav, lg, _ = sim(Sx, FCAP[1e6], fr=True)
    MAIN[c] = dict(valid=cg(nav, TB, DE), valid_dd=mdd(nav, TB, DE), seg={g: s17lib.seg_dd(nav)[g] for g in SEGV}, n_trades=int(len(lg)),
                   first_entry=str(lg['ent_date'].min().date()))
    mlg[c] = lg
res['acceptance']['A3_fresh_no_trade_before_2022'] = bool((mlg['CUR']['ent_date'] >= pd.Timestamp('2022-01-01')).all())
res['fresh_capital_ratio'] = ratio
res['main_fresh'] = MAIN
blk = {}
for c in ('CUR', FZ):
    b, _ = s2lib.block_from(mlg[c], 5)
    for j, v in b.items():
        blk.setdefault(j, []).extend(v)
res['V3_block_union'] = {str(P.COLS[j]): [[str(P.IDX[a].date()), str(P.IDX[b].date())] for a, b in v] for j, v in blk.items()}
del mlg
FR = {c: {} for c in CFG}; FRB = {'CUR': {}, FZ: {}}
for s in range(4):
    for c in CFG:
        Sx, _, _ = build(S0s[s], c)
        for cap in CAPS:
            k = keyof(cap, s)
            nav = sim(Sx, FCAP[cap], fr=True)[0]
            sg = s17lib.seg_dd(nav)
            FR[c][k] = dict(valid=cg(nav, TB, DE), valid_dd=mdd(nav, TB, DE), seg={g: sg[g] for g in SEGV})
            if c in FRB:
                FRB[c][k] = dict(valid=cg(sim(Sx, FCAP[cap], fr=True, block=blk)[0], TB, DE))
        del Sx
    gc.collect()
    print('重新起跑 位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
pv = dict(valid=pair(FR, FZ, 'CUR', 'valid'), valid_dd=pair(FR, FZ, 'CUR', 'valid_dd'), blk=pair(FRB, FZ, 'CUR', 'valid'),
          seg={g: pair(FR, FZ, 'CUR', 'seg', g) for g in SEGV})
V = dict(V1=pv['valid']['npos'] >= 15, V2=pv['valid']['median_pp'] >= 2.0, V3=pv['blk']['median_pp'] > 0, V4=all(pv['seg'][g]['median_pp'] >= -2.0 for g in SEGV))
V = {k: bool(v) for k, v in V.items()}
res['valid'] = dict(pairs_FZ_vs_CUR=pv, medians={c: med([FR[c][k]['valid'] for k in keys]) for c in CFG},
                    medians_blk={c: med([FRB[c][k]['valid'] for k in keys]) for c in FRB})
res['verdict'] = dict(T0=T0, **V, overall='通過' if (T0 and all(V.values())) else '不通過')
res['posthoc_valid_fresh'] = {c: dict(valid=pair(FR, c, 'CUR', 'valid'), dd=pair(FR, c, 'CUR', 'valid_dd'), seg={g: pair(FR, c, 'CUR', 'seg', g) for g in SEGV}) for c in CANDS}
res['cont_FZ_vs_CUR'] = {f: pair(CONT, FZ, 'CUR', f) for f in ('since2019', 'full', 'valid', 'maxdd')}
print('判定', res['verdict'], json.dumps({k: (v if k != 'seg' else v) for k, v in pv.items()}, ensure_ascii=False), flush=True)

# ---------------------------------------------------------------- 隨機對照（描述；位移 0、100 萬；驗證期同月數隨機觸發 stop）
info = S0['info']
vks = [int(x['k']) for x in info if pd.Period(x['month'], 'M') >= L.VALID_FIRST]
ks_fz, _ = trig_rows(S0, FZ)
n_v = len([k for k in ks_fz if k in set(vks)])
act = RULES[FZ][1]
base = MAIN['CUR']['valid']; fzd = (MAIN[FZ]['valid'] - base) * 100
R = []
for i in range(20):
    rng = np.random.default_rng(20261116 + i)
    pick = set(int(k) for k in rng.choice(vks, size=n_v, replace=False))
    Sr, _ = L.apply_rule(P, S0, pick, act)
    R.append((cg(sim(Sr, FCAP[1e6], fr=True)[0], TB, DE) - base) * 100)
R = np.array(R)
res['random_control'] = dict(n_months_valid=n_v, FZ_diff_pp=fzd, rand_diff_pp=R.tolist(), rand_q10_50_90=[float(np.quantile(R, q)) for q in (0.1, 0.5, 0.9)],
                             FZ_percentile=float((R <= fzd).mean()), seeds='20261116+i, i=0..19（複製員自訂；原研究種子未知）')
print('隨機', res['random_control']['FZ_percentile'], res['random_control']['rand_q10_50_90'], flush=True)
res['paths_cont'] = CONT; res['paths_fresh'] = FR; res['paths_fresh_blk'] = FRB
res['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss())
(L.REP / 'b.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print('完成', res['meta'])
