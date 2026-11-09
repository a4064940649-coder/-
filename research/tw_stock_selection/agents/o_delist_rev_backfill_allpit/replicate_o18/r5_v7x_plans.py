#!/usr/bin/env python3
"""G6 複製員 r5：V7X 名單（另開行程，v7env 把 ia_core 換成 V7 面板）。不讀研究員 *.py；
orig＝s4_jscan7 的 AB_V7_B10（JV7 類別照 s4_jscan7 公開程式重寫），REAL_f＝凍結規格三.2：
  營收：R_FIX 中「tw_revenue 年增缺值、R_FIX 有值」的格 → 本複製員補回表的值，沒有就設缺。
  unavail：MC(seed)（st_core.unavail_for）∪ A_UB（avail V6，v2_core.deadline_case 為 D=S／D>S 的日子，非 AVAIL → 回退一個月）；
           前一個月 y3 在資料裡算不出（有值 ≤ 100 檔）的資料邊界日不套 A_UB。
  處置：DMT ∪ 官方全歷史清單（2021-09-23 以前，訊號日或成交日涵蓋）。
  放回：GAP（v4lib.gap_dict，訊號日 ≤ 終止日）∪ GOV（pclib.cause ∈ GOV），2006 年起；opt＝百分位 1.0（V4 B10 同口徑）、p90＝0.9。
python3 -I -B r5_v7x_plans.py [variants...] [--ubday=anchor|v2] [--ubym=anchor|sig] [--table=mine|theirs] [--tag=]"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
TOP = HERE.parent; AG = TOP.parent
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o18')
ARGS = [a for a in sys.argv[1:] if not a.startswith('--')]
OPT = dict(a[2:].split('=') for a in sys.argv[1:] if a.startswith('--'))
UBYM = OPT.get('ubym', 'anchor'); UBDAY = OPT.get('ubday', 'anchor'); TABF = OPT.get('table', 'mine'); TAG = OPT.get('tag', '')
VARIANTS = ARGS or ['orig', 'REAL_opt']
sys.path.insert(0, str(AG / 'o_v7_listing_exec_engine'))
import v7env
L = v7env.load('v7'); K, V = L.K, L.V
sys.path.pop(0)
XR = L.setup_px()
YFX, Y5F = L.load_rfix(); DMT = L.disp_mask_fn('trade'); GAP = L.gap_dict()
sys.path.insert(0, str(AG / 'o_stagger_tranche')); import st_core as STC; sys.path.pop(0)
sys.path.insert(0, str(AG / 'o_rev_avail_ds_pit_count')); import pclib as P; import avail as AV; sys.path.pop(0)
assert P.K is K
COLS, N, IDX = K.COLS, K.N, K.IDX
OFF = L.V4(YFX, Y5F, DMT)
ANCH = [pd.Period(d, 'M') for d in OFF.sig_days]

# ---------------- REAL 營收
BF = pd.read_csv(HERE / 'my_backfill.csv' if TABF == 'mine' else TOP / 'data' / 'rev_backfill_B1.csv', dtype={'code': str, 'ym': str})
BF = BF[BF.yoy_pct.notna()]
BFD = {(c, pd.Period(y, 'M')): v for c, y, v in zip(BF.code, BF.ym, BF.yoy_pct)}
YL0 = K.yoy_from_long()
def real_table(YF, Y0):
    Y = YF.copy(); info = dict(gap=0, replaced=0, dropped=0, agree05=0, added=0)
    Y0r = Y0.reindex(index=Y.index, columns=Y.columns)
    gapm = Y.notna().to_numpy() & Y0r.isna().to_numpy()
    ii, jj = np.where(gapm); info['gap'] = int(len(ii))
    arr = Y.to_numpy().copy()
    for i, j in zip(ii, jj):
        k = (Y.columns[j], Y.index[i])
        if k in BFD:
            info['replaced'] += 1; info['agree05'] += int(abs(BFD[k] - arr[i, j]) <= 0.5); arr[i, j] = BFD[k]
        else:
            info['dropped'] += 1; arr[i, j] = np.nan
    Y = pd.DataFrame(arr, index=Y.index, columns=Y.columns)
    for (c, p), v in BFD.items():   # 補回表有、R_FIX 與 tw_revenue 都缺的格（研究員報 0）
        if p not in Y.index: continue
        yc = Y.at[p, c] if c in Y.columns else np.nan
        y0 = Y0r.at[p, c] if c in Y0r.columns else np.nan
        was_gap = c in Y.columns and gapm[Y.index.get_loc(p), Y.columns.get_loc(c)]
        if not was_gap and not np.isfinite(yc) and not np.isfinite(y0):
            if c not in Y.columns: Y[c] = np.nan
            Y.at[p, c] = v; info['added'] += 1
    return Y, info
YR, INFO_L = real_table(YFX, YL0)
Y5R, INFO_5 = real_table(Y5F, P.Y5)

# ---------------- A_UB、官方處置
E6 = AV.Evidence('sig'); _uc = {}
def ub(d, anchor):
    key = (d, anchor)
    if key in _uc: return _uc[key]
    u = np.zeros(N, bool)
    if UBDAY == 'v2':
        on = V.deadline_case(d) in ('D=S', 'D>S')
    else:   # 錨定月的法定期限 D（st_core.deadline）：d ≤ D 才是 D=S／D>S（j 掃描下只有 J0）
        on = d <= STC.deadline(anchor)
    if on:
        ym = str(pd.Period(d, 'M') - 1) if UBYM == 'sig' else str(anchor - 1)
        for j, c in enumerate(COLS): u[j] = E6.classify(pd.Timestamp(d), c, ym, 'V6')[0] != 'AVAIL'
    _uc[key] = u; return u
DU = pd.read_csv(AG / 'o_rev_avail_ds_pit_count' / 'raw' / 'disposal_official_universe.csv', dtype={'code': str}, parse_dates=['start', 'end'])
_dc = {}
def dm_real(d):
    if d in _dc: return _dc[d]
    m = DMT(d).copy()
    if d <= pd.Timestamp('2021-09-23'):
        t = IDX.get_loc(d); e = IDX[t + 1] if t + 1 < len(IDX) else d
        for c in DU[((DU.start <= d) & (DU.end >= d)) | ((DU.start <= e) & (DU.end >= e))].code:
            if c in K.CI: m[K.CI[c]] = True
    _dc[d] = m; return m

# ---------------- 策略
class JV7(L.V4):
    def set_days(self, days, end_day):
        self.anchor = {d: m for m, d in days}; self.sig_days = [d for m, d in days]; self.end_day = end_day; self._cs = {}
    def y3(self, d, back=0):
        m = self.anchor[d] - back
        Y = self.Y5 if d >= V.SWITCH else self.YL
        y = K.yoy3_for_month(Y, m)
        if self.exfin: y = np.where(V.FINMASK, np.nan, y)
        return y

class JV7R(JV7):
    """V4.cross_section（照公開 v4lib 寫法）＋ GOV 放回併入 gm；fill：opt／p90／carry。"""
    fill = 'opt'
    def cross_section(self, d):
        if d in self._cs: return self._cs[d]
        t = IDX.get_loc(d); e = t + 1
        v60 = self.V60[t]; px = K.Cn[t]
        base = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10)
        vv = np.where(base, v60, -np.inf)
        order = np.argsort(-vv, kind='stable')
        top = np.zeros(N, bool); top[order[:min(300, base.sum())]] = True; top &= base
        y3 = self.y3(d)
        if self.unavail is not None:
            U = self.unavail(d)
            if U is not None and U.any(): y3 = np.where(U, self.y3(d, 1), y3)
        s52 = self.S52[t]
        ok_other = top & np.isfinite(s52) & ~self.dmask(d) & np.isfinite(K.On[e]) & K.TRADED[e] & ~L.FIN2MASK & ~L.FORMASK
        miss = ok_other & ~np.isfinite(y3)
        gm = self.gm_codes & miss & (np.datetime64(d) <= self.gm_until)
        for j in np.where(miss & ~gm)[0]:
            if P.cause(COLS[j], d) in P.GOV: gm[j] = True
        if self.fill == 'carry':
            for j in np.where(gm)[0]:
                yy = np.nan
                for b in range(1, 25):
                    z = self.y3(d, b)[j]
                    if np.isfinite(z): yy = float(z); break
                y3[j] = yy
            gm &= np.isfinite(y3)
            el = ok_other & np.isfinite(y3)
            pr52 = pd.Series(np.where(el, s52, np.nan)).rank(pct=True).to_numpy()
            prr = pd.Series(np.where(el, y3, np.nan)).rank(pct=True).to_numpy()
            y3o = y3
        else:
            el_rev = ok_other & np.isfinite(y3)
            el = el_rev | gm
            pr52 = pd.Series(np.where(el, s52, np.nan)).rank(pct=True).to_numpy()
            prr = pd.Series(np.where(el_rev, y3, np.nan)).rank(pct=True).to_numpy()
            pf = 1.0 if self.fill == 'opt' else 0.9
            if gm.any(): prr = np.where(gm, pf, prr)
            y3o = np.where(gm, np.inf if pf >= 1 else np.nan, y3) if gm.any() else y3
        sc = 0.5 * pr52 + 0.5 * prr; sc[~el] = np.nan
        cs = dict(t=t, e=e, el=el, sc=sc, s52=s52, y3=y3o, gm=gm, top=top, ok_other=ok_other)
        self._cs[d] = cs
        return cs

def days_for(j):
    out = []
    for m in ANCH:
        d = STC.day_for(m, j)
        if d is None: continue
        t = K.IDX.get_loc(d)
        if not K.GOOD[t + 1]: continue
        out.append((m, d))
    de = STC.day_for(pd.Period('2026-10', 'M'), j)
    if de is None or de > K.IDX[-1]: de = K.IDX[-1]
    return [(m, d) for m, d in out if d < de], de

def job(a):
    var, j, seed = a
    days, de = days_for(j)
    if var == 'orig':
        S = JV7(YFX, Y5F, DMT, unavail=(lambda d, _s=seed: STC.unavail_for(S.anchor[d], d, _s)), gap=GAP, pfix=1.0)
        S.set_days(days, de)
    else:
        S = JV7R(YR, Y5R, dm_real, gap=GAP, pfix=1.0)
        S.fill = var.split('_')[1]
        S.set_days(days, de)
        def U(d, _s=seed):
            u = STC.unavail_for(S.anchor[d], d, _s)
            if d in EXEMPT: return u
            return u | ub(d, S.anchor[d])
        S.unavail = U
    pl = S.plan(10, 'official')
    rows = []; nins = 0
    for p in pl:
        gm = S._cs[p[3]]['gm']
        ins = [COLS[jj] for jj in p[1] if gm[jj]]; nins += len(ins)
        rows.append((var, j, seed, str(p[3].date()), str(K.IDX[p[0]].date()), int(p[0]), ','.join(COLS[jj] for jj in p[1]), ';'.join(ins)))
    return var, j, seed, rows, nins

# 資料邊界日：錨定月 −1 的 y3 有值 ≤ 100 檔
EXEMPT = set()
for j in range(13):
    for m, d in days_for(j)[0]:
        Y = Y5R if d >= V.SWITCH else YR
        if np.isfinite(np.where(V.FINMASK, np.nan, K.yoy3_for_month(Y, m - 1))).sum() <= 100 and (UBDAY == 'v2' or d <= STC.deadline(m)): EXEMPT.add(d)
if OPT.get('exempt') == 'fixed': EXEMPT = {pd.Timestamp('2006-04-10'), pd.Timestamp('2021-03-10')}

if __name__ == '__main__':
    T0 = time.time()
    print('real table', INFO_L, INFO_5, 'exempt', sorted(str(d.date()) for d in EXEMPT), flush=True)
    jobs = [(v, j, s) for v in VARIANTS for j in range(13) for s in range(20)]
    with Pool(4) as pool: out = pool.map(job, jobs, chunksize=4)
    rows = [r for o in out for r in o[3]]
    DF = pd.DataFrame(rows, columns=['strat', 'j', 'seed', 'signal', 'exec', 'exec_idx', 'picks', 'inserted'])
    f = SCR / f'plans_r5{TAG}.csv'; DF.to_csv(f, index=False)
    res = dict(real_table_L=INFO_L, real_table_5=INFO_5, exempt=sorted(str(d.date()) for d in EXEMPT), ubym=UBYM, ubday=UBDAY, table=TABF,
               counts={v: dict(n_rows=int((DF.strat == v).sum()), inserted_mean_per_plan=float(np.mean([o[4] for o in out if o[0] == v]))) for v in VARIANTS})
    if 'orig' in VARIANTS:
        ref = pd.read_csv(AG / 'o_v7_listing_exec_engine' / 'data' / 'plans_j7.csv', dtype={'picks': str})
        ref = ref[(ref.strat == 'AB_V7_B10') & (ref.seed < 20)]
        a = DF[DF.strat == 'orig'][['j', 'seed', 'signal', 'exec_idx', 'picks']].sort_values(['j', 'seed', 'signal']).reset_index(drop=True)
        b = ref[['j', 'seed', 'signal', 'exec_idx', 'picks']].sort_values(['j', 'seed', 'signal']).reset_index(drop=True)
        res['orig_equals_plans_j7_seed_lt20'] = bool(len(a) == len(b) and (a.astype(str).to_numpy() == b.astype(str).to_numpy()).all())
    res['runtime_s'] = time.time() - T0
    (HERE / f'r5_v7x_plans{TAG}.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=1))
