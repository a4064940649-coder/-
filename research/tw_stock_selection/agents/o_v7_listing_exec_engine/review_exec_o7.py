#!/usr/bin/env python3
"""反方審查（成本與可成交角度）：只讀研究員檔案，不寫回研究員檔案；輸出到 stdout 與 scratchpad。
檢查：
 (1) j 分布的可執行超額拆成 SEL（2006-2018）與 2019 起（從 results.json）。
 (2) FIX 版本的「免費縮放」：研究員 FIX 每次換股把持股與現金等比例縮放到 級距×CPI，提領／注資不付稅費與衝擊。
     改成真實作法：提領／注資只經由現金（持股照常賣出，付稅費衝擊），量差異。
 (3) TICK 下界在大級距的合理性：記錄每筆整股單 Q/OA，推算 K5 衝擊。
用法：python3 -I -B review_exec_o7.py"""
import sys, os, json, math, inspect, textwrap
from pathlib import Path
HERE = Path('/home/user/-/research/tw_stock_selection/agents/o_v7_listing_exec_engine')
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import multiprocessing as mp

R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
EX = R['RESULTS_EXEC']

def part1():
    T = EX['table']; out = {}
    for model in ('K5T', 'TICK'):
        for tier in ('30萬', '100萬', '300萬', '1000萬'):
            for ver in ('FIX', 'GROW'):
                sel, y19, hold = [], [], []
                for j in range(13):
                    v = T.get(f'B10|J{j}|{model}|normal|{tier}|{ver}')
                    if v is None: continue
                    sel.append(v['CAGR_SEL'] - v['E50_CAGR_SEL']); y19.append(v['CAGR_Y19on'] - v['E50_CAGR_Y19on']); hold.append(v['CAGR_HOLD'] - v['E50_CAGR_HOLD'])
                out[f'{model}|{tier}|{ver}'] = dict(SEL_excess_med=float(np.median(sel)), Y19_excess_med=float(np.median(y19)), Y19_excess_max=float(np.max(y19)),
                                                   n_j_Y19_pos=int(sum(x > 0 for x in y19)), HOLD_excess_med=float(np.median(hold)))
    return out

# ---------------- 引擎載入（與 run_exec7 相同） ----------------
import v7env, v7stats as ST, ecore7 as E
L = v7env.load('v7'); K = L.K
XR = L.setup_px()
D = E.Data(L, XR)
SPEC = json.loads((HERE / 'data' / 'model_spec7.json').read_text(encoding='utf-8'))
MODELS = {k: E.Model(SPEC, k) for k in ('K5T', 'TICK', 'K5')}
END = pd.Timestamp('2026-10-12')
def to_plan(g): return [(int(e), [D.CI[c] for c in pk.split(',')], sg) for e, pk, sg in zip(g.exec_idx, g.picks, g.signal)]
PA = pd.read_csv(HERE / 'data' / 'plans_ab7.csv', dtype={'picks': str})
PJ = pd.read_csv(HERE / 'data' / 'plans_j7.csv', dtype={'picks': str})
JEND = json.loads((HERE / 'data' / 'jscan_end_days.json').read_text(encoding='utf-8'))
PLANS, BOUNDS, GRID = {}, {}, {}
for (st, s), g in PA.groupby(['strat', 'seed']):
    if int(s) > 4: continue
    key = ('AB' if st == 'AB_V7' else 'B10', 'OFF', int(s)); PLANS[key] = to_plan(g)
    BOUNDS[key[:2]] = (int(K.IDX.get_loc(pd.Timestamp(g.signal.iloc[0]))), int(K.IDX.get_loc(END)))
    GRID[key[:2]] = [pd.Timestamp(x) for x in g.signal] + [END]
for (st, j, s), g in PJ.groupby(['strat', 'j', 'seed']):
    if int(s) > 2 or st != 'AB_V7_B10': continue
    key = ('B10', f'J{j}', int(s)); PLANS[key] = to_plan(g)
    de = pd.Timestamp(JEND[f'{st}|J{j}'])
    BOUNDS[key[:2]] = (int(K.IDX.get_loc(pd.Timestamp(g.signal.iloc[0]))), int(K.IDX.get_loc(de)))
    GRID[key[:2]] = [pd.Timestamp(x) for x in g.signal] + [de]
NORMAL = dict(lim=True, odd0=True, cash=0.02, feemix=True, zpart=1.0, disp=True, mdelay=True)

# ---- (2) FIX 真實提領／注資：不縮放持股，flow 直接進出現金；負現金允許（記錄次數），由當日賣出補回 ----
OLD1 = "                    S = S * sc; cash = cash * sc; flow = target - eq_pre; self.acc['flow'] += flow\n                    for r in rec: r[0] *= sc\n"
NEW1 = "                    flow = target - eq_pre; cash = cash + flow; self.acc['flow'] += flow; self.acc['absflow'] = self.acc.get('absflow', 0.0) + abs(flow) / eq_pre\n"
OLD2 = "            assert cash > -1e-6 * max(1.0, eq), (t, cash)\n"
NEW2 = "            if cash < -1e-6 * max(1.0, eq): self.acc['neg_cash_days'] = self.acc.get('neg_cash_days', 0) + 1\n"
src = inspect.getsource(E.Sim.run)
assert src.count(OLD1) == 1 and src.count(OLD2) == 1
src_new = src.replace(OLD1, NEW1).replace(OLD2, NEW2)
ns = dict(E.__dict__)
exec(compile('class SimCash(Sim):\n' + src_new, 'simcash', 'exec'), ns)
SimCash = ns['SimCash']

# rebalance 用 Vtr＝持股＋現金 計目標；現金為負時目標自動變小 → 等同先賣後買。

# ---- (3) Q/OA 記錄 ----
class SimQ(E.Sim):
    def imp(self, t, j, Q):
        oa = self.oa(t, j)
        if np.isfinite(oa) and oa > 0 and Q > 0 and self.M.kind != 'ENG':
            self.qoa.append((Q / oa, D.SIG[t - 1, j]))
        return super().imp(t, j, Q)
    def run(self, *a, **k):
        self.qoa = []
        return super().run(*a, **k)

def stats(nav, grp):
    sd = GRID[grp]; t0, t1 = BOUNDS[grp]; a, b = K.IDX[t0], K.IDX[t1]
    E50 = K.E50
    cg = ST.cagr_mdd(nav, a, b)[0]; c50 = ST.cagr_mdd(E50.reindex(nav.index).ffill(), a, b)[0]
    o = {'CAGR': cg, 'ex': cg - c50, 't': ST.tstat(ST.xs(nav, E50, sd, a, b)),
         't_Y19': ST.tstat(ST.xs(nav, E50, sd, '2019-01-01', b))}
    return o

def job(a):
    key, model, tier, kind = a
    t0, t1 = BOUNDS[key[:2]]
    cls = {'orig': E.Sim, 'cash': SimCash}[kind]
    nav, acc = cls(D, MODELS[model], NORMAL).run(PLANS[key], t0, t1, E.TIERS[tier], 'FIX', 'A')
    return a, nav.pct_change().to_numpy(dtype=np.float64), {k: float(acc.get(k, 0.0)) for k in ('absflow', 'neg_cash_days', 'n_cap_hit')}

def jobq(a):
    key, model, tier, ver = a
    t0, t1 = BOUNDS[key[:2]]
    s = SimQ(D, MODELS[model], NORMAL)
    nav, acc = s.run(PLANS[key], t0, t1, E.TIERS[tier], ver, 'A')
    q = np.array([x[0] for x in s.qoa]); sg = np.array([x[1] for x in s.qoa])
    k5 = 0.001 + 0.5 * np.nan_to_num(sg) * np.sqrt(q)
    yrs = (K.IDX[t1] - K.IDX[t0]).days / 365.25
    return a, dict(n=int(len(q)), qoa_med=float(np.median(q)), qoa_p90=float(np.percentile(q, 90)), share_ge_0p1=float((q >= 0.1).mean()),
                   k5_med_bp=float(np.median(k5) * 1e4), k5_p90_bp=float(np.percentile(k5, 90) * 1e4),
                   end_equity_ntd=float(acc['end_equity']), start_equity_ntd=float(acc['start_equity']), years=yrs,
                   n_cap_hit=int(acc['n_cap_hit']), deferred_days=int(acc['deferred_days']))

if __name__ == '__main__':
    out = {'part1_excess_split_B10_jmedian': part1()}
    print(json.dumps(out['part1_excess_split_B10_jmedian'], ensure_ascii=False, indent=0))
    # (3)
    qj = [(('B10', 'OFF', 0), m, tier, ver) for m in ('TICK',) for tier in ('300萬', '1000萬') for ver in ('FIX', 'GROW')]
    with mp.get_context('fork').Pool(4) as P: rq = P.map(jobq, qj)
    out['part3_qoa'] = {'|'.join(map(str, a[0])) + f'|{a[1]}|{a[2]}|{a[3]}': v for a, v in rq}
    print(json.dumps(out['part3_qoa'], ensure_ascii=False, indent=0))
    # (2)
    keys = [k for k in PLANS if (k[1] == 'OFF' and k[0] == 'B10') or k[1] != 'OFF']
    jobs = [(k, m, tier, kind) for k in keys for m in ('K5T', 'TICK') for tier in ('30萬', '1000萬') for kind in ('orig', 'cash')]
    with mp.get_context('fork').Pool(4) as P: res = P.map(job, jobs, chunksize=2)
    agg, accs = {}, {}
    for (a, r, acc) in res:
        key, model, tier, kind = a
        g = (key[0], key[1], model, tier, kind); agg.setdefault(g, []).append(r); accs.setdefault(g, []).append(acc)
    tab = {}
    for g, rs in agg.items():
        t0, t1 = BOUNDS[g[:2]]
        nav = pd.Series((1 + np.nan_to_num(np.mean(rs, axis=0), nan=0.0)).cumprod(), index=K.IDX[t0:t1 + 1])
        o = stats(nav, g[:2]); o.update({k: float(np.mean([a[k] for a in accs[g]])) for k in accs[g][0]})
        tab['|'.join(g)] = o
    summ = {}
    for model in ('K5T', 'TICK'):
        for tier in ('30萬', '1000萬'):
            dd = {}
            for grp in sorted({k[1] for k in keys}):
                a = tab.get(f'B10|{grp}|{model}|{tier}|orig'); b = tab.get(f'B10|{grp}|{model}|{tier}|cash')
                if a and b: dd[grp] = dict(orig=a['CAGR'], cash=b['CAGR'], d_pp=b['CAGR'] - a['CAGR'], ex_orig=a['ex'], ex_cash=b['ex'], t_orig=a['t'], t_cash=b['t'],
                                           absflow_per_yr=b['absflow'] / 20.5, neg_cash_days=b['neg_cash_days'])
            js = [v for k, v in dd.items() if k.startswith('J')]
            summ[f'{model}|{tier}|FIX'] = dict(OFF=dd.get('OFF'), j_median_d_pp=float(np.median([v['d_pp'] for v in js])),
                                               j_median_ex_orig=float(np.median([v['ex_orig'] for v in js])), j_median_ex_cash=float(np.median([v['ex_cash'] for v in js])),
                                               j_median_t_orig=float(np.median([v['t_orig'] for v in js])), j_median_t_cash=float(np.median([v['t_cash'] for v in js])))
    out['part2_fix_flow'] = summ
    print(json.dumps(summ, ensure_ascii=False, indent=0))
    sp = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/review_exec_o7.json')
    sp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('saved', sp)
