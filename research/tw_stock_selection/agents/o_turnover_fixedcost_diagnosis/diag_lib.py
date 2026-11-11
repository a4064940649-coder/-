#!/usr/bin/env python3
"""診斷統計共用函式（研究員）：區塊 bootstrap、AUC、分組摘要、期間切片。不含任何引擎。"""
import numpy as np, pandas as pd
SEL_END = pd.Timestamp('2019-04-30')
PERIODS = {'SEL': ('2006-01-01', '2019-04-30'), 'ALL': ('2006-01-01', '2026-12-31'), 'S0612': ('2006-01-01', '2012-12-31'), 'S1318': ('2013-01-01', '2018-12-31'),
           'S0618': ('2006-01-01', '2018-12-31'), 'Y19on': ('2019-01-01', '2026-12-31'), 'POSTODD': ('2020-10-26', '2026-12-31'), 'PREODD': ('2006-01-01', '2020-10-25'),
           'Y19on_xsrc3': ('2019-01-01', '2026-12-31')}
SRC3 = (pd.Timestamp('2024-02-15'), pd.Timestamp('2025-09-23'))
def cut(df, per, col='signal'):
    a, b = PERIODS[per]; d = pd.to_datetime(df[col])
    m = (d >= a) & (d <= b)
    if per.endswith('xsrc3'):
        de = pd.to_datetime(df['date']) if 'date' in df.columns else d
        m &= ~((de >= SRC3[0]) & (de <= SRC3[1]))
    return df[m]
def auc(score, y):
    s = np.asarray(score, float); y = np.asarray(y, bool); ok = np.isfinite(s); s, y = s[ok], y[ok]
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0: return float('nan')
    r = pd.Series(s).rank().to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def block_boot(df, stat, B=2000, seed=20261112, col='signal'):
    """月份區塊 bootstrap：以訊號日所在日曆月為區塊（13 j 同月合併）重抽。回傳 (觀察值, bootstrap 陣列)。"""
    df = df.reset_index(drop=True)
    m = pd.to_datetime(df[col]).dt.to_period('M').astype(str).to_numpy()
    ids = [np.flatnonzero(m == u) for u in np.unique(m)]
    obs = stat(df); rng = np.random.default_rng(seed); out = np.empty(B)
    n = len(ids)
    for b in range(B):
        idx = np.concatenate([ids[i] for i in rng.integers(0, n, n)])
        out[b] = stat(df.iloc[idx])
    return obs, out
def q(x, p): x = np.asarray(x, float); x = x[np.isfinite(x)]; return float(np.percentile(x, p)) if len(x) else float('nan')
def med(x): return q(x, 50)

def pclass(p):
    return np.select([p < 20, p < 50, p < 100, p < 500], ['<20', '20~50', '50~100', '100~500'], '>=500')

def metrics(T, v, per, cfg, B=2000, boot=True):
    """依凍結（或迭代）設定 cfg 計算 P3～P6 的量測。T＝build_tables 的 dict；v＝變體；per＝期間代號。
    cfg：pairing（B／A2／A1／A0）、price_def（exit／min_A2）、cost（ENG／h05／h10）、slot（excess／raw）、reversal（obs／post）。"""
    tag = cfg['pairing']; ce = '' if cfg.get('cost', 'ENG') == 'ENG' else '_' + cfg['cost']
    X = cut(T['exits'], per); X = X[(X.variant == v) & np.isfinite(X[f'edge_{tag}'])].copy()
    X['edge'] = X[f'edge_{tag}']; X['rtc'] = X[f'rtc_{tag}{ce}']; X['gap'] = X.edge - X.rtc; X['below'] = X.edge < X.rtc
    if cfg.get('price_def', 'exit') == 'exit': X['pc'] = pclass(X.price.to_numpy())
    else: X['pc'] = pclass(np.fmin(X.price, X.partner_price_A2).to_numpy())
    F = cut(T['fills'], per, col='date'); F = F[F.variant == v]
    o = {'n_swaps': int(len(X)), 'frac_lt': float(X.below.mean()), 'edge_med': med(X.edge), 'rtc_med': med(X.rtc)}
    # P3
    lowF = F.price.fillna(F.notional / F.shares) < 50
    costcol = 'cost' if ce == '' else 'cost' + ce
    x_ = float(lowF.mean()); cs = float(F.loc[lowF, costcol].sum() / F[costcol].sum())
    lo = X[X.price < 50]; hi = X[X.price >= 100]
    if cfg.get('price_def', 'exit') != 'exit':
        pm = np.fmin(X.price, X.partner_price_A2); lo = X[pm < 50]; hi = X[pm >= 100]
    d_obs = float(lo.below.mean() - hi.below.mean()) if len(lo) and len(hi) else float('nan')
    P3 = dict(fill_lowprice_share_x=x_, cost_share_low=cs, ratio=cs / x_ if x_ else float('nan'), frac_lt_low=float(lo.below.mean()) if len(lo) else None,
              frac_lt_hi=float(hi.below.mean()) if len(hi) else None, diff_pp=d_obs * 100, n_low=int(len(lo)), n_hi=int(len(hi)),
              by_pc={k: dict(n=int(len(g)), frac_lt=float(g.below.mean()), edge_med=med(g.edge), rtc_med=med(g.rtc)) for k, g in X.groupby('pc')})
    if boot and len(lo) and len(hi):
        def st3(d):
            a = d[d._low]; b = d[d._hi]
            return (a.below.mean() - b.below.mean()) if len(a) and len(b) else np.nan
        Y = X.assign(_low=X.index.isin(lo.index), _hi=X.index.isin(hi.index))
        ob, bb = block_boot(Y, st3, B=B); bb = bb[np.isfinite(bb)]
        P3['boot_p_diff_gt_0'] = float(np.mean(bb <= 0)); P3['boot_p_diff_gt_10pp'] = float(np.mean(bb <= 0.10)); P3['boot_ci90_pp'] = [q(bb * 100, 5), q(bb * 100, 95)]
    o['P3'] = P3
    # P4
    byrb = {b: dict(n=int(len(g)), frac_lt=float(g.below.mean()), edge_med=med(g.edge), rtc_med=med(g.rtc), gap_med=med(g.gap)) for b, g in X.groupby('rb')}
    nlt = int(X.below.sum()); m1115 = X.rb == '11~15'; m50 = X.rb == '>50'
    P4 = dict(by_rank=byrb, share_1115_in_lt=float((X.below & m1115).sum() / nlt) if nlt else None, share_1115_in_all=float(m1115.mean()),
              c1_1115_edge_lt_cost=bool(byrb.get('11~15', {}).get('edge_med', np.nan) < byrb.get('11~15', {}).get('rtc_med', np.nan)),
              c3_gt50_edge_gt_cost=bool(byrb.get('>50', {}).get('edge_med', np.nan) > byrb.get('>50', {}).get('rtc_med', np.nan)))
    P4['c2_share_ge_40'] = bool(P4['share_1115_in_lt'] is not None and P4['share_1115_in_lt'] >= 0.40)
    if boot and m1115.any() and m50.any():
        def st4(d):
            a = d.gap[d.rb == '>50']; b = d.gap[d.rb == '11~15']
            return (np.median(a) - np.median(b)) if len(a) and len(b) else np.nan
        ob, bb = block_boot(X, st4, B=B); bb = bb[np.isfinite(bb)]
        P4['diff_med_gap_gt50_minus_1115'] = float(ob); P4['boot_p'] = float(np.mean(bb <= 0)); P4['boot_ci90'] = [q(bb, 5), q(bb, 95)]
    o['P4'] = P4
    # P5
    RB = cut(T['rebs'], per); RB = RB[RB.variant == v].copy()
    RB['nb'] = pd.cut(RB.n_exits, [-1, 2, 4, 99], labels=['<=2', '3~4', '>=5'])
    if cfg.get('reversal', 'obs') == 'obs': RB['rev'] = np.sign(RB.m21) != np.sign(RB.m21_126)
    else: RB['rev'] = np.sign(RB.m63) != np.sign(RB.r50_win)
    X2 = X.merge(RB[['j', 'k', 'nb', 'rev']], on=['j', 'k'], how='left')
    P5 = {str(b): dict(n_reb=int((RB.nb == b).sum()), n_swaps=int((X2.nb == b).sum()), edge_med=med(X2.edge[X2.nb == b]), gap_med=med(X2.gap[X2.nb == b]),
                       frac_lt=float(X2.below[X2.nb == b].mean()) if (X2.nb == b).any() else None, rev_share=float(RB.rev[RB.nb == b].mean()) if (RB.nb == b).any() else None)
          for b in ['<=2', '3~4', '>=5']}
    P5['c1_hi_edge_below_lo'] = bool(P5['>=5']['edge_med'] < P5['<=2']['edge_med']) if P5['<=2']['n_swaps'] else None
    P5['c2_rev_concentrated'] = bool(P5['>=5']['rev_share'] > P5['<=2']['rev_share']) if P5['<=2']['n_reb'] else None
    if boot and (X2.nb == '<=2').any():
        def st5(d):
            a = d.edge[d.nb == '<=2']; b = d.edge[d.nb == '>=5']
            return (np.median(a) - np.median(b)) if len(a) and len(b) else np.nan
        ob, bb = block_boot(X2, st5, B=B); bb = bb[np.isfinite(bb)]
        P5['boot_p_lo_minus_hi_gt0'] = float(np.mean(bb <= 0))
    o['P5'] = P5
    # P6
    SL = cut(T['slots'], per); SL = SL[SL.variant == v]
    cc = SL.contrib if cfg.get('slot', 'raw') == 'excess' else SL.contrib_raw
    band = (cc >= -0.02) & (cc <= 0.02)
    o['P6'] = dict(n_slots=int(len(SL)), band_share=float(band.mean()), band_contrib_sum=float(cc[band].sum()), band_cost_sum=float(SL.cost_pct[band].sum()),
                   all_contrib_sum=float(cc.sum()), all_cost_sum=float(SL.cost_pct.sum()), holds=bool(SL.cost_pct[band].sum() > cc[band].sum()),
                   decile={str(i): dict(contrib_sum=float(cc[d].sum()), cost_sum=float(SL.cost_pct[d].sum()), n=int(d.sum())) for i, d in
                           ((i, pd.qcut(cc, 10, labels=False, duplicates='drop') == i) for i in range(10))})
    return o
