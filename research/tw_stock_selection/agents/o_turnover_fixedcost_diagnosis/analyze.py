#!/usr/bin/env python3
"""凍結後全樣本分析（一次跑）：斷言 PREREG 與 PREREG_FREEZE 存在且 sha 相符。
→ results.json['RESULTS']、data/*.csv.gz（成交日誌、換股表、逐槽表、兩平表、Shapley 摘要）。
python3 -I -B analyze.py"""
import sys, json, pickle, hashlib, time, itertools, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import diag_lib as DL; import tfc_lib as TL; sys.path.pop(0)
RJ = HERE / 'results.json'; R = json.loads(RJ.read_text(encoding='utf-8'))
assert 'PREREG' in R and 'PREREG_FREEZE' in R, '凍結前不得計算樣本外'
assert hashlib.sha256(json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256']
assert hashlib.sha256(json.dumps(R['PREREG_FREEZE'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_FREEZE_sha256']
for f, h in R['PREREG_FREEZE']['files_sha256'].items(): assert hashlib.sha256((HERE / f).read_bytes()).hexdigest() == h, f
SCR = TL.SCR; t0 = time.time()
CFG = dict(pairing='B', price_def='exit', cost='ENG', slot='raw', reversal='obs')
VARS = ['REAL_opt', 'REAL_p90', 'REAL_carry']
CELLS = [('normal', '100萬'), ('stress', '100萬'), ('normal', '30萬'), ('stress', '30萬'), ('normal', '300萬'), ('normal', '1000萬')]
T = {c: pickle.load(open(SCR / f'T_{c[0]}_{c[1]}.pkl', 'rb')) for c in CELLS}
E = TL.engine(); S = E['S']; D = E['D']
OUT = {'utc_start': pd.Timestamp.now('UTC').isoformat(), 'cfg': CFG}
def log(*a): print(round(time.time() - t0), *a, flush=True)
# ---------- A. 正式數字重現（20 seeds）與 seed 0 差 ----------
AV = pd.read_parquet(SCR / 'navs_100n_avg.parquet'); N0 = pd.read_parquet(SCR / 'navs_seed0.parquet')
OFF = json.loads((HERE.parent / 'integrate_o19/results.json').read_text(encoding='utf-8'))['RESULTS_V7X']['table']
rep = {}
for v in VARS:
    st20 = [S.stats(AV[f'{v}|{j}'], 'AB_V7_B10', j) for j in range(13)]; st0 = [S.stats(N0[f'{v}|{j}|normal|100萬'], 'AB_V7_B10', j) for j in range(13)]
    a = S.jsumm(st20); b = S.jsumm(st0); o = OFF[f'strict|{v}|normal|100萬|std']
    rep[v] = {k: dict(mine20=a[k], official=o[k], diff=a[k] - o[k], seed0=b[k]) for k in ('CAGR_Y19on_med', 'exc_Y19on_med', 'exc_med', 't_0050_med', 'exc_p10', 'maxDD_med', 'CAGR_med')}
    rep[v]['per_j_diff_Y19on'] = dict(med=float(np.median([x['CAGR_Y19on'] - y['CAGR_Y19on'] for x, y in zip(st20, st0)])), maxabs=float(np.max([abs(x['CAGR_Y19on'] - y['CAGR_Y19on']) for x, y in zip(st20, st0)])))
OUT['A_reproduce_official'] = rep
log('A', {v: round(rep[v]['CAGR_Y19on_med']['diff'], 6) for v in VARS})
# 各格 seed 0 年化（對照用）
seed0 = {}
for c in CELLS:
    for v in VARS:
        st = [S.stats(N0[f'{v}|{j}|{c[0]}|{c[1]}'], 'AB_V7_B10', j) for j in range(13)]; a = S.jsumm(st)
        seed0[f'{v}|{c[0]}|{c[1]}'] = {k: a[k] for k in ('CAGR_Y19on_med', 'exc_Y19on_med', 'exc_med', 'exc_POSTODD_med', 'CAGR_POSTODD_med', 'maxDD_med')}
OUT['A_seed0_cells'] = seed0
# ---------- B. 會計成本拖累與周轉 ----------
def years(per, j):
    a, b = DL.PERIODS[per]; e = S.end_day('AB_V7_B10', j)
    a = max(pd.Timestamp(a), pd.Timestamp('2006-04-01')); b = min(pd.Timestamp(b), e)
    yrs = (b - a).days / 365.25
    if per.endswith('xsrc3'): yrs -= (DL.SRC3[1] - DL.SRC3[0]).days / 365.25
    return yrs
COMPS = ['cost', 'tax', 'fee', 'tick_amt', 'imp_amt', 'odd_amt', 'stress_amt', 'fee_min_excess', 'pxdev_amt', 'cost_h05', 'cost_h10']
def drag(c, per):
    rb = DL.cut(T[c]['rebs'], per, col='date'); o = {}
    for v in VARS:
        r = rb[rb.variant == v]; d = {}
        for comp in COMPS:
            d[comp] = float(np.median([(r[r.j == j][comp] / r[r.j == j].avail).sum() / years(per, j) * 100 for j in range(13)]))
        for cl in ('exit', 'rw_sell', 'entry', 'rw_buy'):
            d['turn_' + cl] = float(np.median([(r[r.j == j][f'not_{cl}'] / r[r.j == j].avail).sum() / years(per, j) for j in range(13)]))
        d['exits_per_reb'] = float(r.n_exits.mean())
        o[v] = d
    o['worst_cost_variant'] = max(VARS, key=lambda v: o[v]['cost'])
    return o
OUT['B_drag'] = {f'{c[0]}|{c[1]}|{p}': drag(c, p) for c in CELLS for p in ('Y19on', 'ALL', 'S0612', 'S1318', 'POSTODD', 'PREODD', 'Y19on_xsrc3')}
log('B')
# 周轉來源：exit 依原因（強制／排名級距）、entry 依來源
def turn_src(c, per):
    X = DL.cut(T[c]['exits'], per); N = DL.cut(T[c]['entries'], per); rb = DL.cut(T[c]['rebs'], per)
    o = {}
    for v in VARS:
        x = X[X.variant == v].merge(rb[rb.variant == v][['j', 'k', 'avail']], on=['j', 'k']); n = N[N.variant == v].merge(rb[rb.variant == v][['j', 'k', 'avail']], on=['j', 'k'])
        x['w'] = x.sell_notional / x.avail; n['w'] = n.buy_notional / n.avail
        o[v] = dict(exit_by_rank={b: dict(n=int(len(g)), share_turn=float(g.w.sum() / x.w.sum())) for b, g in x.groupby('rb')},
                    entry_by_src={b: dict(n=int(len(g)), share_turn=float(g.w.sum() / n.w.sum())) for b, g in n.groupby('src')},
                    forced_share=float(x.w[x.rb.str.startswith('F_')].sum() / x.w.sum()))
    return o
OUT['B_turnover_sources'] = {f'{c[0]}|{c[1]}|{p}': turn_src(c, p) for c in [('normal', '100萬'), ('normal', '30萬')] for p in ('Y19on', 'S0618', 'ALL')}
# 逐筆成本分布（依帳戶與年份）
def fdist(c):
    F = T[c]['fills']; F = F[F.variant == 'REAL_opt'].assign(rate=lambda d: d.cost / d.notional, y=lambda d: pd.to_datetime(d.date).dt.year)
    o = {}
    for y, g in F.groupby('y'):
        o[int(y)] = {k: dict(n=int(len(gg)), med=DL.med(gg.rate), p90=DL.q(gg.rate, 90), p99=DL.q(gg.rate, 99), tk_med=DL.med(gg.tk)) for k, gg in g.groupby('kind')}
    return o
OUT['B_fill_cost_dist_by_year_opt'] = {f'{c[0]}|{c[1]}': fdist(c) for c in [('normal', '100萬'), ('normal', '30萬'), ('normal', '1000萬')]}
log('B2')
# ---------- C. P3～P6 ----------
MET = {}
plan = [(c, p, 2000) for c in [('normal', '100萬'), ('stress', '100萬')] for p in ('Y19on', 'S0618')] + \
       [(c, 'POSTODD', 2000) for c in [('normal', '30萬'), ('stress', '30萬')]] + \
       [(c, p, 500) for c in [('normal', '30萬'), ('stress', '30萬')] for p in ('Y19on', 'S0618')] + \
       [(('normal', '100萬'), p, 500) for p in ('S0612', 'S1318', 'ALL', 'Y19on_xsrc3', 'POSTODD')] + \
       [(c, 'Y19on', 0) for c in [('normal', '300萬'), ('normal', '1000萬')]]
for c, p, B in plan:
    for v in VARS:
        MET[f'{c[0]}|{c[1]}|{p}|{v}'] = DL.metrics(T[c], v, p, CFG, B=max(B, 1), boot=B > 0)
    log('C', c, p)
# 並列：A2、A0 配對、H05／H10 成本、超額版槽貢獻（100 萬一般 Y19on、S0618，B=500）
PAR = {}
for name, cf in (('A2', dict(CFG, pairing='A2')), ('A0', dict(CFG, pairing='A0')), ('h05', dict(CFG, cost='h05')), ('h10', dict(CFG, cost='h10')), ('slot_excess', dict(CFG, slot='excess')), ('rev_post', dict(CFG, reversal='post'))):
    for p in ('Y19on', 'S0618'):
        for v in VARS:
            PAR[f'{name}|{p}|{v}'] = DL.metrics(T[('normal', '100萬')], v, p, cf, B=500, boot=True)
    log('PAR', name)
OUT['C_metrics'] = MET; OUT['C_parallel'] = PAR
# ---------- D. P2 ----------
def p2(per):
    o = {}
    for v in VARS:
        r = {}
        for tier in ('30萬', '100萬', '300萬', '1000萬'):
            F = DL.cut(T[('normal', tier)]['fills'], per, col='date'); F = F[F.variant == v]
            sp = F.tick_amt + F.imp_amt + F.odd_amt
            r[tier] = dict(agg=float(sp.sum() / F.notional.sum()), med=DL.med(sp / F.notional), tick_layer_share=float((F.tick_amt.sum() + F.odd_amt.sum()) / sp.sum()),
                           whole_tick_binding_share=float(((F.kind == 'W') & (F.sq <= F.tk)).sum() / max((F.kind == 'W').sum(), 1)), n=int(len(F)))
        r['ratio_30_100_agg'] = r['30萬']['agg'] / r['100萬']['agg']; r['ratio_30_100_med'] = r['30萬']['med'] / r['100萬']['med']
        o[v] = r
    o['worst_ratio'] = min(o[v]['ratio_30_100_agg'] for v in VARS)
    return o
OUT['D_P2'] = {p: p2(p) for p in ('POSTODD', 'Y19on', 'S0618', 'ALL')}
log('D')
# ---------- E. 兩平表 ----------
rows = []
for c in CELLS:
    for p in ('Y19on', 'POSTODD', 'S0618', 'ALL'):
        X = DL.cut(T[c]['exits'], p); X = X[np.isfinite(X.edge_B)]
        dr = OUT['B_drag'].get(f'{c[0]}|{c[1]}|{p}')
        for v in VARS:
            x = X[X.variant == v].assign(pc=lambda d: DL.pclass(d.price.to_numpy()))
            nrebyr = x.groupby('j').k.nunique().median() / np.median([years(p, j) for j in range(13)])
            for pcl, g in list(x.groupby('pc')) + [('ALL', x)]:
                rows.append(dict(cell=f'{c[0]}|{c[1]}', tier=c[1], cost=c[0], period=p, variant=v, price=pcl, n=len(g), rtc_med=DL.med(g.rtc_B), rtc_mean=float(g.rtc_B.mean()),
                                 rtc_h05_med=DL.med(g.rtc_B_h05), rtc_h10_med=DL.med(g.rtc_B_h10), edge_med=DL.med(g.edge_B), frac_below=float((g.edge_B < g.rtc_B).mean()),
                                 swaps_per_yr=float(len(g) / 13 / np.median([years(p, j) for j in range(13)])),
                                 gross_excess_needed_if_all_swaps_like_this_pp=float(x.groupby('j').size().median() / np.median([years(p, j) for j in range(13)]) * g.rtc_B.mean() / 10 * 100),
                                 annual_cost_drag_pp=dr[v]['cost'] if dr else None))
BE = pd.DataFrame(rows); BE.to_csv(HERE / 'data/breakeven_table.csv.gz', index=False, compression='gzip', float_format='%.6g')
def be(tier, p, pc):
    q = BE[(BE.tier == tier) & (BE.cost == 'normal') & (BE.period == p) & (BE.price == pc)]
    return {v: float(q[q.variant == v].rtc_med.iloc[0]) for v in VARS if (q.variant == v).any()}
lo_hi = lambda tier, p: dict(low={k: be(tier, p, k) for k in ('<20', '20~50')}, high={k: be(tier, p, k) for k in ('100~500', '>=500')})
# 代表格：<50 與 ≥100 合併中位數
def be2(tier, cost, p, low):
    X = DL.cut(T[(cost, tier)]['exits'], p); X = X[np.isfinite(X.edge_B)]
    o = {}
    for v in VARS:
        x = X[X.variant == v]; g = x[x.price < 50] if low else x[x.price >= 100]
        o[v] = dict(rtc_med=DL.med(g.rtc_B), rtc_mean=float(g.rtc_B.mean()), n=int(len(g)), edge_med=DL.med(g.edge_B), frac_below=float((g.edge_B < g.rtc_B).mean()))
    o['worst_rtc_med'] = max(o[v]['rtc_med'] for v in VARS)
    return o
OUT['E_breakeven_rep'] = {f'{t}|{c}|{p}|{"<50" if lo else ">=100"}': be2(t, c, p, lo) for t, p in (('100萬', 'Y19on'), ('30萬', 'POSTODD')) for c in ('normal', 'stress') for lo in (True, False)}
log('E')
# ---------- F. 6.22pp 會計對照 ----------
def acct(c, per):
    X = DL.cut(T[c]['exits'], per); rb = DL.cut(T[c]['rebs'], per); o = {}
    for v in VARS:
        x = X[(X.variant == v) & np.isfinite(X.edge_B)].merge(rb[rb.variant == v][['j', 'k', 'avail']], on=['j', 'k'])
        x['rt_nt'] = x.sell_cost + (x.rtc_B - x.sell_rate) * x.sell_notional
        bel = x.edge_B < x.rtc_B
        o[v] = dict(below_cost_pp=float(np.median([(x[(x.j == j) & bel].rt_nt / x[(x.j == j) & bel].avail).sum() / years(per, j) * 100 for j in range(13)])),
                    all_swaps_cost_pp=float(np.median([(x[x.j == j].rt_nt / x[x.j == j].avail).sum() / years(per, j) * 100 for j in range(13)])),
                    below_share_n=float(bel.mean()))
    o['worst'] = max(VARS, key=lambda v: o[v]['below_cost_pp'])
    return o
OUT['F_accounting'] = {f'{c[0]}|{c[1]}|{p}': acct(c, p) for c in [('normal', '100萬'), ('stress', '100萬'), ('normal', '30萬')] for p in ('Y19on', 'S0618', 'ALL', 'POSTODD')}
log('F')
# ---------- G. AUC ----------
def aucs(c, per):
    X = DL.cut(T[c]['exits'], per); o = {}
    for v in VARS:
        x = X[(X.variant == v) & np.isfinite(X.edge_B)]; y = (x.edge_B < x.rtc_B).to_numpy()
        feats = dict(exit_price=x.price, exit_rank=np.where(x.why == 'el', x['rank'], 999.0), dscore=x.dscore_B, n_exits=x.n_exits, rtc=x.rtc_B)
        o[v] = {k: dict(auc=DL.auc(np.asarray(f, float), y), auc_dir=max(DL.auc(np.asarray(f, float), y), 1 - DL.auc(np.asarray(f, float), y))) for k, f in feats.items()}
    return o
OUT['G_auc'] = {f'normal|100萬|{p}': aucs(('normal', '100萬'), p) for p in ('Y19on', 'S0618')}
log('G')
# ---------- H. 穩健性 ----------
SL = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype=str).set_index('code')['industry']
def robust(c):
    X = T[c]['exits']; X = X[np.isfinite(X.edge_B)].assign(below=lambda d: d.edge_B < d.rtc_B, gap=lambda d: d.edge_B - d.rtc_B, y=lambda d: pd.to_datetime(d.signal).dt.year)
    o = {}
    for v in VARS:
        x = X[X.variant == v]
        o[v] = dict(by_year={int(y): dict(n=int(len(g)), frac_below=float(g.below.mean()), gap_med=DL.med(g.gap), frac_below_1115=float(g.below[g.rb == '11~15'].mean()) if (g.rb == '11~15').any() else None,
                                          frac_below_gt50=float(g.below[g.rb == '>50'].mean()) if (g.rb == '>50').any() else None) for y, g in x.groupby('y')},
                    by_industry={k: dict(n=int(len(g)), frac_below=float(g.below.mean())) for k, g in x.assign(ind=x.code.map(SL).fillna('NA')).groupby('ind') if len(g) >= 300})
    # 去掉紙上貢獻最大 5 檔（全期槽貢獻合計）
    sl = T[c]['slots']; top5 = sl[sl.variant == 'REAL_opt'].groupby('code').contrib_raw.sum().nlargest(5).index.tolist()
    T2 = dict(T[c]); T2['exits'] = T[c]['exits'][~T[c]['exits'].code.isin(top5)]
    o['drop_top5'] = dict(codes=top5, Y19on={v: DL.metrics(T2, v, 'Y19on', CFG, B=500)['P4'] for v in VARS})
    return o
OUT['H_robust'] = robust(('normal', '100萬'))
log('H')
# ---------- I. Shapley ----------
SH = pd.read_csv(HERE / 'data/shapley_coalitions.csv.gz', dtype={'coalition': str})
FAC = ['T', 'X', 'F', 'K', 'I', 'C']; NAMES = dict(T='時點', X='證交稅', F='手續費', K='跳動點／價差', I='衝擊', C='參與上限')
def key(s): return ''.join(f for f in FAC if f in s) or 'none'
def shapley(vf):
    perms = list(itertools.permutations(FAC)); marg = {f: [] for f in FAC}; costcls = []; o19 = None
    for pm in perms:
        cur = ''; tot_cost = 0.0
        for f in pm:
            nx = key(cur + f); d = vf[nx] - vf[key(cur)]; marg[f].append(d)
            if f != 'T': tot_cost += d
            cur = nx
        costcls.append(-tot_cost)
    return marg, costcls
def ladder(vf, order):
    cur = ''; out = []
    for grp in order:
        nx = key(cur + grp); out.append((grp, -(vf[nx] - vf[key(cur)]))); cur = nx
    return out
SHO = {}
for tier in ('100萬', '30萬'):
    for v in VARS:
        for per in ('CAGR_Y19on', 'CAGR', 'S0612', 'S1318', 'CAGR_POSTODD'):
            sub = SH[(SH.tier == tier) & (SH.variant == v)]
            vf = sub.groupby('coalition')[per].median().to_dict(); vf = {('none' if k == 'none' else key(k)): x for k, x in vf.items()}
            marg, cc = shapley(vf)
            perj = []
            for j in range(13):
                vj = {('none' if k == 'none' else key(k)): x for k, x in sub[sub.j == j].set_index('coalition')[per].to_dict().items()}
                mj, ccj = shapley(vj); perj.append({f: -float(np.mean(mj[f])) for f in FAC} | {'cost_class': float(np.mean(ccj))})
            SHO[f'{tier}|{v}|{per}'] = dict(values=dict(none=vf['none'], timing_only=vf['T'], all_on=vf['TXFKIC']), total_gap=vf['T' if False else 'none'] - vf['TXFKIC'],
                                             shapley={NAMES[f]: dict(mean=-float(np.mean(marg[f])), min=-float(np.max(marg[f])), max=-float(np.min(marg[f]))) for f in FAC},
                                             cost_class=dict(mean=float(np.mean(cc)), min=float(np.min(cc)), max=float(np.max(cc)), any_lt_12=bool(np.min(cc) < 12)),
                                             o19_order=ladder(vf, ['T', 'X', 'F', 'KIC']), cost_first_order=ladder(vf, ['X', 'F', 'KIC', 'T']),
                                             perj_median={k: float(np.median([p[k] for p in perj])) for k in perj[0]})
OUT['I_shapley'] = SHO
o19 = dict(timing=2.38475831252547, tax=3.3501617024784807, fee=0.8784938747900668, imp_caps=10.137764060218398, cost_class=14.366419637486946, total=18.173108196635958)
mine = SHO['100萬|REAL_opt|CAGR_Y19on']
lad = dict(mine['o19_order'])
OUT['I_vs_o19'] = dict(o19=o19, mine_o19_order=dict(timing=lad['T'], tax=lad['X'], fee=lad['F'], imp_caps=lad['KIC'], cost_class=lad['X'] + lad['F'] + lad['KIC']),
                       mine_shapley_cost_class=mine['cost_class']['mean'],
                       engine_effect_cost_class=lad['X'] + lad['F'] + lad['KIC'] - o19['cost_class'],
                       order_effect_cost_class=mine['cost_class']['mean'] - (lad['X'] + lad['F'] + lad['KIC']),
                       note='引擎效果＝本題 o19 順序（正式 V7X：F1/F3/F4 修正後、整股零股結構與 2% 現金恆開、seed 0）− o19 凍結階梯；順序效果＝Shapley 平均 − 本題 o19 順序')
log('I')
# ---------- J. P7 β ----------
NK = pd.read_parquet(SCR / 'navs_shapley_keep.parquet'); E50 = S.E50
def winr(nav, j):
    e = S.end_day('AB_V7_B10', j); sd = S.SIGD[('AB_V7_B10', j)]; sd = sd[sd < e].append(pd.DatetimeIndex([e]))
    r = nav.reindex(sd); rb = E50.reindex(sd)
    return pd.DataFrame({'r': (r.shift(-1) / r - 1).iloc[:-1], 'b': (rb.shift(-1) / rb - 1).iloc[:-1]}).dropna()
P7 = {}
for v in VARS:
    rr = {'beta_sel': [], 'beta_all': [], 'exc_post_Y19_vs_0050': [], 'exc_post_Y19_vs_beta': [], 'exc_pre_Y19_vs_0050': [], 'exc_pre_Y19_vs_beta': [], 'exc_post_all_vs_beta': [], 'exc_pre_all_vs_beta': []}
    for j in range(13):
        post = NK[f'TXFKIC|{v}|{j}'].dropna(); pre = NK[f'T|{v}|{j}'].dropna(); e = S.end_day('AB_V7_B10', j); post = post.loc[:e]; pre = pre.loc[:e]
        w = winr(post, j); ws = w[w.index <= '2019-04-30']
        bs = float(np.polyfit(ws.b, ws.r, 1)[0]); ba = float(np.polyfit(w.b, w.r, 1)[0])
        b50 = E50.reindex(post.index).pct_change().fillna(0); bnav = (1 + bs * b50).cumprod()
        s19 = post.index[post.index >= '2019-01-01'][0]
        c = lambda n, a: S.cagr(n, a, e)
        rr['beta_sel'].append(bs); rr['beta_all'].append(ba)
        rr['exc_post_Y19_vs_0050'].append(c(post, s19) - c(E50, s19)); rr['exc_post_Y19_vs_beta'].append(c(post, s19) - c(bnav, s19))
        rr['exc_pre_Y19_vs_0050'].append(c(pre, s19) - c(E50, s19)); rr['exc_pre_Y19_vs_beta'].append(c(pre, s19) - c(bnav, s19))
        a0 = post.index[0]; rr['exc_post_all_vs_beta'].append(c(post, a0) - c(bnav, a0)); rr['exc_pre_all_vs_beta'].append(c(pre, a0) - c(bnav, a0))
    P7[v] = {k: dict(med=float(np.median(x)), min=float(np.min(x)), max=float(np.max(x))) for k, x in rr.items()}
OUT['J_P7'] = P7
log('J')
# ---------- K. 判定 ----------
def worst(key_fn, keys):
    vals = [key_fn(MET[k]) for k in keys if k in MET]; return vals
cellsK = [f'{c}|{p}|{v}' for c in ('normal|100萬', 'stress|100萬') for p in ('Y19on',) for v in VARS]
cellsD = [f'{c}|S0618|{v}' for c in ('normal|100萬', 'stress|100萬') for v in VARS]
def p3_strict(m): return bool(m['P3']['ratio'] >= 1.5 and m['P3']['diff_pp'] >= 10 and m['P3'].get('boot_p_diff_gt_0', 1) < 0.05)
def p4_strict(m): return bool(m['P4']['c1_1115_edge_lt_cost'] and m['P4']['c2_share_ge_40'] and m['P4']['c3_gt50_edge_gt_cost'] and m['P4'].get('boot_p', 1) < 0.05)
K = dict(P3_strict_Y19={k: p3_strict(MET[k]) for k in cellsK}, P4_strict_Y19={k: p4_strict(MET[k]) for k in cellsK},
         P3_bootp_Y19={k: MET[k]['P3'].get('boot_p_diff_gt_0') for k in cellsK}, P4_bootp_Y19={k: MET[k]['P4'].get('boot_p') for k in cellsK},
         P3_diff_dir_S0618={k: MET[k]['P3']['diff_pp'] > 0 for k in cellsD}, P4_diff_dir_S0618={k: MET[k]['P4'].get('diff_med_gap_gt50_minus_1115', np.nan) > 0 for k in cellsD})
K['P3_pass_all_cells'] = all(K['P3_strict_Y19'].values()) and all(K['P3_diff_dir_S0618'].values())
K['P4_pass_all_cells'] = all(K['P4_strict_Y19'].values()) and all(K['P4_diff_dir_S0618'].values())
K['P3_lenient_all'] = all((p or 1) < 0.05 for p in K['P3_bootp_Y19'].values()) and all(K['P3_diff_dir_S0618'].values())
K['P4_lenient_all'] = all((p or 1) < 0.05 for p in K['P4_bootp_Y19'].values()) and all(K['P4_diff_dir_S0618'].values())
fr = [MET[k]['P4']['by_rank'] for k in cellsK]
spread = [max(d[b]['frac_lt'] for b in ('11~15', '16~30', '31~50', '>50') if b in d) - min(d[b]['frac_lt'] for b in ('11~15', '16~30', '31~50', '>50') if b in d) for d in fr]
spread3 = [max(v_['frac_lt'] for k_, v_ in MET[k]['P3']['by_pc'].items() if v_['n'] >= 100) - min(v_['frac_lt'] for k_, v_ in MET[k]['P3']['by_pc'].items() if v_['n'] >= 100) for k in cellsK]
K['rank_frac_spread_pp_max'] = float(max(spread) * 100); K['price_frac_spread_pp_max'] = float(max(spread3) * 100)
if K['P3_pass_all_cells'] or K['P4_pass_all_cells']: verdict = '結構成立（暫定：待不同子代理複製）'
elif K['rank_frac_spread_pp_max'] < 10 and K['price_frac_spread_pp_max'] < 10 and not (K['P3_lenient_all'] or K['P4_lenient_all']): verdict = '結構不成立（成本均勻分布、屬訊號內生代價）'
else: verdict = '無法判定'
K['verdict'] = verdict
OUT['K_verdict'] = K
log('K', verdict)
# ---------- L. 匯出 ----------
def exp_fills(c, v, name):
    F = T[c]['fills']; F = F[F.variant == v].copy()
    F['code'] = [D.COLS[j] for j in F.j]; F['date'] = pd.to_datetime(F.date).dt.strftime('%Y-%m-%d')
    cols = ['variant', 'j_', 'date', 'k', 'code', 'side', 'kind', 'cls', 'ctx', 'shares', 'req', 'open', 'px', 'notional', 'tk', 'sq', 'im', 'oddc', 'fee', 'tax', 'avail']
    F[cols].rename(columns={'j_': 'j'}).to_csv(HERE / f'data/{name}', index=False, compression='gzip', float_format='%.8g')
exp_fills(('normal', '100萬'), 'REAL_opt', 'tradelog_100n_REAL_opt_seed0.csv.gz')
exp_fills(('normal', '30萬'), 'REAL_opt', 'tradelog_30n_REAL_opt_seed0.csv.gz')
X = T[('normal', '100萬')]['exits']; X[X.variant == 'REAL_opt'].drop(columns=['date']).to_csv(HERE / 'data/swaps_exitlevel_100n_REAL_opt_seed0.csv.gz', index=False, compression='gzip', float_format='%.8g')
SLt = T[('normal', '100萬')]['slots']; SLt[SLt.variant == 'REAL_opt'].drop(columns=['date']).to_csv(HERE / 'data/slots_100n_REAL_opt_seed0.csv.gz', index=False, compression='gzip', float_format='%.8g')
RBt = pd.concat([T[c]['rebs'].assign(cell=f'{c[0]}|{c[1]}') for c in CELLS]); RBt.drop(columns=['date']).to_csv(HERE / 'data/rebs_allcells_seed0.csv.gz', index=False, compression='gzip', float_format='%.8g')
AV.to_parquet(HERE / 'data/navs_100n_avg20.parquet')
shs = [dict(key=k, factor=f, **d) for k, s in SHO.items() for f, d in s['shapley'].items()]
pd.DataFrame(shs).to_csv(HERE / 'data/shapley_summary.csv.gz', index=False, compression='gzip', float_format='%.6g')
OUT['L_files'] = {p.name: dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted((HERE / 'data').iterdir())}
OUT['utc_end'] = pd.Timestamp.now('UTC').isoformat()
R = json.loads(RJ.read_text(encoding='utf-8')); assert 'RESULTS' not in R, '只跑一次'
R['RESULTS'] = OUT; RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
log('done')
