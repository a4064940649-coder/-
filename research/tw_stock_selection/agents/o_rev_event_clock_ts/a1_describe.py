#!/usr/bin/env python3
"""分支 A1：不看報酬的描述統計（只用有時間戳的股月）。
(1) 公告日分布、早公告比例；(2) P2：早公告（rel ≥ 3）減期限日公告（rel ≤ 0）的營收百分位差（t 依錨定月分群）；
(3) P3：事件時鐘（EC）的每月額外換股次數（只用分數與名次模擬，不計任何報酬）；(4) 每次額外換股的損益兩平漂移（V7X 成本與 K5T 公式）。
EC 版本在計算前登記於 results.json A1_PLAN（凍結選版規則：預設 V2；若 V2 每月額外換股 > 4 檔次，改選 reconcile＝是 且 ≤ 4 中額外換股最少者）。
執行：python3 -I -B a1_describe.py"""
import sys, json, math, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import ts_common as T

T.check_gate_prereg()
GATE = T.jload(T.HERE / 'gate.json'); assert GATE and GATE['branch'] == 'A', '本腳本只用於分支 A'

VERSIONS = {
    'V1': dict(K_out=15, delta=0.00, reconcile=True),
    'V2': dict(K_out=20, delta=0.05, reconcile=True),
    'V3': dict(K_out=30, delta=0.10, reconcile=True),
    'V4': dict(K_out=20, delta=0.05, reconcile=False),
}
PLAN = dict(versions=VERSIONS, default='V2', n=10, max_trades_per_stock_month=1,
            select_rule='預設 V2（K_out 20、δ 0.05、每檔每月最多進出各 1 次、官方日全名單校正）。若 V2 每月平均額外換股 > 4 檔次，'
                        '改選 reconcile＝是 且每月額外換股 ≤ 4 的版本中額外換股最少者；全部 > 4 則仍用 V2 並在結論寫明 P3 不成立。選版不看任何報酬。',
            tiers=[1_000_000, 3_000_000, 10_000_000], utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
r = T.jload(T.RES_F)
if 'A1_PLAN' not in r:
    T.save_key('A1_PLAN', dict(PLAN, sha256=T.canon_sha(PLAN)))

E = T.engine(); OFF, K, V, STC = E['OFF'], E['K'], E['V'], E['STC']
TCAL = STC.TCAL; IDX = K.IDX; COLS = list(K.COLS); CI = {c: j for j, c in enumerate(COLS)}
FS = pd.read_csv(T.DATA / 'firstseen_hist.csv', dtype={'code': str})

# ---------------- (1)(2) 公告日分布與 P2 ----------------
def p2(df, label):
    early = df[df.rel >= 3]; dead = df[df.rel <= 0]
    g = df.assign(grp=np.where(df.rel >= 3, 'early', np.where(df.rel <= 0, 'dead', 'mid')))
    mm = g[g.grp != 'mid'].groupby(['anchor', 'grp']).rev_pct.mean().unstack()
    mm = mm.dropna()
    d = (mm['early'] - mm['dead']) if len(mm) else pd.Series(dtype=float)
    t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 2 else None
    return dict(label=label, n_stock_months=len(df), n_early=len(early), n_deadline=len(dead),
                share_early=len(early) / len(df) if len(df) else None, share_deadline=len(dead) / len(df) if len(df) else None,
                pooled_early_minus_deadline=float(early.rev_pct.mean() - dead.rev_pct.mean()) if len(early) and len(dead) else None,
                monthly_mean_diff=float(d.mean()) if len(d) else None, t_by_anchor=t, n_anchors=len(d),
                rel_hist=df.rel.value_counts().sort_index().to_dict())
ts = FS[FS.valid_any & FS.rel.notna() & FS.rev_pct.notna()].copy(); ts['rel'] = ts.rel.astype(int)
P2 = dict(strict=p2(ts, '合格時間戳（閘門口徑）'))
# 敏感度：Tasi 所有首見（含檔案加入 commit、週末缺口 > 36h；時間戳偏晚 → rel 偏小，早公告會被低估）
def rel_lenient(row):
    if not isinstance(row.tasi_T, str) or not row.tasi_T: return None
    a = T.announce_day(datetime.datetime.fromisoformat(row.tasi_T), TCAL); D = STC.deadline(pd.Period(row.anchor, 'M'))
    return int(TCAL.get_loc(D) - TCAL.get_loc(a)) if a is not None and D is not None else None
ln = FS[FS.tasi_seen & FS.rev_pct.notna()].copy(); ln['rel'] = [rel_lenient(x) for x in ln.itertuples()]
ln = ln[ln.rel.notna()]; ln['rel'] = ln.rel.astype(int)
P2['lenient_tasi_all_seen'] = p2(ln, 'Tasi 所有首見（敏感度，非閘門口徑）')
# 逐年
P2['by_year_lenient'] = {y: p2(g, y) for y, g in ln.groupby(ln.anchor.str[:4])}
for v in P2['by_year_lenient'].values(): v.pop('rel_hist', None)
# 公告日（月內第幾個日曆日）分布
ln['aday'] = pd.to_datetime([T.announce_day(datetime.datetime.fromisoformat(x), TCAL) for x in ln.tasi_T])
DIST = dict(lenient_day_of_month=ln.aday.dt.day.value_counts().sort_index().to_dict(),
            strict_rel=P2['strict']['rel_hist'], lenient_rel=P2['lenient_tasi_all_seen']['rel_hist'])

# ---------------- (3) EC 換股模擬（名次，不計報酬） ----------------
def yfor(d, m):
    Y = OFF.Y5 if d >= V.SWITCH else OFF.YL
    return K.yoy3_for_month(Y, m)

def ranks_on(d, y3):
    """d 收盤：股池（ok_other）內 0.5×52 週百分位＋0.5×營收百分位；回傳 (score 陣列, 名次陣列 1..n；不在池＝inf)。"""
    cs = OFF.cross_section(d); ok = cs['ok_other'] & np.isfinite(y3) & np.isfinite(cs['s52'])
    p52 = pd.Series(np.where(ok, cs['s52'], np.nan)).rank(pct=True).to_numpy()
    pr = pd.Series(np.where(ok, y3, np.nan)).rank(pct=True).to_numpy()
    sc = np.where(ok, 0.5 * p52 + 0.5 * pr, np.nan)
    scr = np.round(np.nan_to_num(sc, nan=-9), 10); yy = np.nan_to_num(y3, nan=-1e18)
    order = np.lexsort((np.arange(len(sc)), -yy, -scr))
    order = [j for j in order if ok[j]]
    rk = np.full(len(sc), np.inf); rk[order] = np.arange(1, len(order) + 1)
    return sc, rk, order

def official(d):
    cs = OFF.cross_section(d); sc = cs['sc']; y3 = cs['y3']; el = cs['el']
    scr = np.round(np.nan_to_num(sc, nan=-9), 10); yy = np.nan_to_num(y3, nan=-1e18)
    order = [j for j in np.lexsort((np.arange(len(sc)), -yy, -scr)) if el[j]]
    return order[:10]

anchors = sorted(FS.anchor[FS.valid_any].unique())
ann = {}   # (anchor) → {j: a_day}
for a, g in FS[FS.valid_any].groupby('anchor'):
    ann[a] = {CI[c]: pd.Timestamp(x) for c, x in zip(g.code, g.a_day) if c in CI and isinstance(x, str) and x}
sig = {str(pd.Period(d, 'M')): d for d in OFF.sig_days}
sig['2026-10'] = pd.Timestamp(GATE['denominator']['sig_day_2026_10'])

def sim_month(a, ver):
    m = pd.Period(a, 'M'); d_m = sig.get(a); d_p = sig.get(str(m - 1))
    if d_m is None or d_p is None: return None
    O_prev = official(d_p); O_m = official(d_m)
    H = list(O_prev)
    win = [d for d in TCAL if d > d_p and d < d_m and d >= pd.Timestamp(m.year, m.month, 1)]
    A = ann.get(a, {})
    newm, oldm = yfor(d_m, m - 1), yfor(d_m, m - 2)
    traded_in, traded_out = set(), set(); trades = []   # (day, sell j, buy j, kind)
    for d in win:
        known = np.array([A.get(j) is not None and A[j] <= d for j in range(len(COLS))])
        y3 = np.where(known & np.isfinite(newm), newm, oldm)
        sc, rk, order = ranks_on(d, y3)
        Nd = {j for j, x in A.items() if x == d}
        if not Nd: continue
        # 出場：新公告持股名次 > K_out 或不在池
        for h in list(H):
            if h in Nd and h not in traded_in and h not in traded_out and rk[h] > ver['K_out']:
                H.remove(h); traded_out.add(h); trades.append((d, h, None, 'exit'))
        # 進場：新公告非持股、名次 ≤ 10
        cands = sorted([j for j in Nd if j not in H and rk[j] <= 10 and j not in traded_out], key=lambda j: rk[j])
        for c in cands:
            if len(H) < 10:
                H.append(c); traded_in.add(c); trades.append((d, None, c, 'fill_new'))
                continue
            elig = [h for h in H if h not in traded_in and h not in traded_out]
            if not elig: break
            w = min(elig, key=lambda h: (np.nan_to_num(sc[h], nan=-9), -h))
            if np.nan_to_num(sc[c], nan=-9) - np.nan_to_num(sc[w], nan=-9) >= ver['delta']:
                H.remove(w); traded_out.add(w); H.append(c); traded_in.add(c); trades.append((d, w, c, 'swap'))
        # 空位：用全池最佳未持有（名次 ≤ 10）補
        for j in order:
            if len(H) >= 10: break
            if j not in H and j not in traded_out and rk[j] <= 10:
                H.append(j); traded_in.add(j); trades.append((d, None, j, 'fill_any'))
    # 官方日
    if ver['reconcile']:
        sells = [h for h in H if h not in O_m]; buys = [j for j in O_m if j not in H]
    else:
        cs_sc, rk, order = ranks_on(d_m, OFF.cross_section(d_m)['y3'])
        sells = [h for h in H if rk[h] > ver['K_out']]
        keep = [h for h in H if h not in sells]
        buys = [j for j in order if j not in keep][:10 - len(keep)]
    n_win_buys = sum(1 for t in trades if t[2] is not None)
    n_ec = n_win_buys + len(buys)               # 檔次＝買進筆數（每筆對應一筆賣出）
    n_off = len([j for j in O_m if j not in O_prev])
    early_in_off = [t[2] for t in trades if t[2] is not None]
    return dict(anchor=a, n_ann_valid=len(A), n_window_buys=n_win_buys, n_reconcile_buys=len(buys), n_ec=n_ec, n_off=n_off,
                extra=n_ec - n_off, early_buys_in_official=sum(1 for j in early_in_off if j in O_m), n_early_buys=len(early_in_off),
                trades=[(str(t[0].date()), COLS[t[1]] if t[1] is not None else '', COLS[t[2]] if t[2] is not None else '', t[3]) for t in trades],
                reconcile_sells=[COLS[h] for h in sells], reconcile_buys=[COLS[j] for j in buys])

SIM = {}
for vk, ver in VERSIONS.items():
    rows = [x for x in (sim_month(a, ver) for a in anchors) if x is not None]
    SIM[vk] = dict(months=rows, mean_extra=float(np.mean([x['extra'] for x in rows])) if rows else None,
                   mean_ec=float(np.mean([x['n_ec'] for x in rows])) if rows else None,
                   mean_off=float(np.mean([x['n_off'] for x in rows])) if rows else None,
                   mean_window_buys=float(np.mean([x['n_window_buys'] for x in rows])) if rows else None,
                   early_buy_hit_official=(sum(x['early_buys_in_official'] for x in rows) / max(1, sum(x['n_early_buys'] for x in rows))),
                   n_months=len(rows))
sel = 'V2'
if SIM['V2']['mean_extra'] is not None and SIM['V2']['mean_extra'] > 4:
    ok = [k for k, v in SIM.items() if VERSIONS[k]['reconcile'] and v['mean_extra'] is not None and v['mean_extra'] <= 4]
    if ok: sel = min(ok, key=lambda k: SIM[k]['mean_extra'])

# ---------------- (4) 損益兩平漂移（V7X 成本公式；不計報酬） ----------------
Z = np.load(T.AG / 'o_exec_engine_rewrite_v7x' / 'data' / 'x_cache.npz')
ZIDX = pd.DatetimeIndex(Z['IDX']); ZC = {c: j for j, c in enumerate(Z['COLS'])}
SIGp, V60p, VRp, PC = Z['SIGp'], Z['V60p'], Z['VRp'], Z['PC']
SP = json.loads((T.AG / 'o_v7_listing_exec_engine' / 'data' / 'model_spec7.json').read_text())
OASH = np.array([SP['oa_share_by_dec'][str(i)] for i in range(1, 11)])
OD_PE = np.array(SP['odd']['price_edges']); OD_RE = SP['odd']['rank_edges'][0]
OD_B = np.array(SP['odd']['buy']); OD_S = np.array(SP['odd']['sell'])
def tick(p): return float(np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0))
CPI = json.loads((T.AG / 'o_v7_listing_exec_engine' / 'raw' / 'exec_inputs' / 'cpi_tw.json').read_text())['index_2006_100']

def side_cost(e, code, Q, side, stress):
    """成交日 e、金額 Q（元）的單邊成本率（V7X：K5T 整股＋零股價差＋手續費最低 20 元；賣方另加證交稅 0.3%）。"""
    t = ZIDX.get_loc(e); j = ZC[code]
    px = PC[t, j]; sg = SIGp[t, j]; v60 = V60p[t, j]; vr = VRp[t, j]
    if not (np.isfinite(px) and px > 0): return None
    dec = int(min(10, max(1, math.ceil(vr / 30)))) if np.isfinite(vr) and vr <= 300 else 10
    oa = OASH[dec - 1] * v60
    L = int(Q // (1000 * px)); q = int((Q - L * 1000 * px) // px)
    fr = 0.001425 * (0.6 if stress else 0.28)
    extra = SP['stress']['extra_per_side'] if stress else 0.0
    cost = 0.0
    if L > 0:
        qq = L * 1000 * px
        imp = 0.001 + 0.5 * (sg if np.isfinite(sg) else 0.03) * math.sqrt(qq / oa) if np.isfinite(oa) and oa > 0 else 0.01
        imp = max(imp, tick(px) / px) + extra
        cost += qq * imp + max(20.0, qq * fr)
    if q > 0:
        qv = q * px
        pb = int(np.searchsorted(OD_PE, px, side='right')); rb = int((vr if np.isfinite(vr) else 1e9) >= OD_RE)
        c = OD_B[pb, rb] if side > 0 else max(OD_S[pb, rb], 0.0)
        if stress and side > 0: c *= 2
        cost += qv * (c + extra) + max(20.0, qv * fr)
    tot = L * 1000 * px + q * px
    rate = cost / tot if tot > 0 else None
    if rate is not None and side < 0: rate += 0.003
    return rate

def be_rows(vk):
    out = []
    for x in SIM[vk]['months']:
        for (dd, s, b, kind) in x['trades']:
            d = pd.Timestamp(dd); k = TCAL.get_loc(d)
            if k + 1 >= len(TCAL): continue
            e = TCAL[k + 1]
            if e not in ZIDX: continue
            for tier in PLAN['tiers']:
                Q = tier * CPI[str(e.year)] / CPI['2026'] / 10 if str(e.year) in CPI else tier / 10
                for st in (False, True):
                    cb = side_cost(e, b, Q, +1, st) if b else None
                    cs_ = side_cost(e, s, Q, -1, st) if s else None
                    # 只有單邊（補空位或單純出場）時，另一邊用同一筆的對稱估計（出場的資金最後仍要換成一檔股票）
                    if cb is None and cs_ is not None: cb = side_cost(e, s, Q, +1, st)
                    if cs_ is None and cb is not None: cs_ = side_cost(e, b, Q, -1, st)
                    if cb is None or cs_ is None: continue
                    out.append(dict(anchor=x['anchor'], tier=tier, stress=st, rt=cb + cs_))
    return pd.DataFrame(out)
BE = be_rows(sel)
be_tab = {}
if len(BE):
    for (tier, st), g in BE.groupby(['tier', 'stress']):
        be_tab[f'{tier}_{"stress" if st else "normal"}'] = dict(n=len(g), median_rt=float(g.rt.median()), mean_rt=float(g.rt.mean()),
                                                                 p90_rt=float(g.rt.quantile(0.9)))
# 主策略每月超額（已發表的 V7X 數字，不重算報酬）：全期 100 萬 j 中位數 16.0～16.8% 對 0050 14.37%；2019 年起少 3.7～9.8pp／年
MAIN = dict(full_period_pp_per_month=[(16.0 - 14.37) / 12, (16.8 - 14.37) / 12], since2019_pp_per_month=[-9.8 / 12, -3.7 / 12],
            source='original_o8 DIRECTOR_REPORT／報告 13.35（V7X、K5T、j 中位數）')
cost_pp = {}
ex = SIM[sel]['mean_extra'] or 0.0
for k, v in be_tab.items():
    cost_pp[k] = ex * v['mean_rt'] * 100 / 10     # 每月組合層級 pp：額外檔次 × 來回成本 × 權重 1/10
P3 = dict(selected_version=sel, mean_extra_trades_per_month=SIM[sel]['mean_extra'], P3_turnover_ok=bool(ex <= 4),
          breakeven_drift_per_extra_trade=be_tab, extra_cost_pp_per_month=cost_pp, main_excess=MAIN,
          cost_vs_half_main_full_period={k: bool(v > 0.5 * MAIN['full_period_pp_per_month'][1]) for k, v in cost_pp.items()},
          note='「損益兩平漂移」＝每次額外換股的來回成本（新買股相對被賣股，在官方日前必須多賺的幅度）。2019 年起主策略月超額為負，任何正成本都大於其一半。')
OUT = dict(P2=P2, distribution=DIST, ec_sim={k: {kk: vv for kk, vv in v.items() if kk != 'months'} for k, v in SIM.items()},
           ec_sim_months={k: [{kk: vv for kk, vv in m.items() if kk not in ('trades',)} for m in v['months']] for k, v in SIM.items()},
           P3=P3, anchors_used=anchors)
T.jdump(OUT, T.HERE / 'explore.json')
T.jdump(dict(note='分支 A：沒有任何報酬檢定；以下 4 版只比較周轉與成本（不看報酬），選版規則見 results.json A1_PLAN。',
             versions=VERSIONS, results={k: {kk: vv for kk, vv in v.items() if kk != 'months'} for k, v in SIM.items()},
             selected=sel, n_versions=len(VERSIONS)), T.HERE / 'iterations.json')
pd.DataFrame([dict(version=sel, anchor=m['anchor'], day=t[0], sell=t[1], buy=t[2], kind=t[3]) for m in SIM[sel]['months'] for t in m['trades']]).to_csv(T.DATA / 'ec_sim_trades.csv', index=False)
print(json.dumps(T.conv(dict(P2={k: {kk: vv for kk, vv in v.items() if kk not in ('rel_hist',)} for k, v in P2.items() if k != 'by_year_lenient'},
                             ec={k: {kk: vv for kk, vv in v.items() if kk != 'months'} for k, v in SIM.items()}, P3=P3)), ensure_ascii=False, indent=1))
