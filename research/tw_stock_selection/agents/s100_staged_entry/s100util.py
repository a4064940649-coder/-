#!/usr/bin/env python3
"""S100 純函式（只依賴 numpy／pandas；正式引擎與 2008 起連續引擎共用）：變體定義、R2 縮放對照、補買檔數、第一次檢查統計、集中度統計。"""
import math
import numpy as np, pandas as pd

# 變體（事前固定，見 PREREG.json）：名稱＝初次比例→補滿後大小@第幾次換股檢查
CELLS = {
    'A|1/2→1@1': dict(init=1 / 2, final=1.0, check=1),
    'A|1/2→1@2': dict(init=1 / 2, final=1.0, check=2),
    'B|1/3→1@1': dict(init=1 / 3, final=1.0, check=1),
    'B|1/3→1@2': dict(init=1 / 3, final=1.0, check=2),
    'C|1/2→1.5@1': dict(init=1 / 2, final=1.5, check=1),
    'C|1/2→1.5@2': dict(init=1 / 2, final=1.5, check=2),
    'D|1/2→2@1': dict(init=1 / 2, final=2.0, check=1),
    'D|1/2→2@2': dict(init=1 / 2, final=2.0, check=2),
}


def topup_K(st):
    """某次分批模擬「實際補買成功」的檔數：{執行日列 t: 檔數}（給隨機對照用）。"""
    tp = st['topups']
    return {int(t): int(n) for t, n in tp.groupby('t').size().items()} if len(tp) else {}


def scaled_current(nav_cur, bench, lam, a, b):
    """R2 對照：把現行策略線性縮放到相同曝險——窗內每日報酬＝lam × 現行日報酬 ＋ (1 − lam) × 0050 日報酬（收盤對收盤、每日再平衡、不另計成本）。回傳窗內淨值（起點 1）。"""
    rc = nav_cur.loc[a:b].pct_change().iloc[1:]; rb = bench.reindex(nav_cur.index).loc[a:b].pct_change().iloc[1:]
    r = lam * rc + (1.0 - lam) * rb
    out = pd.concat([pd.Series([1.0], index=[nav_cur.loc[a:b].index[0]]), (1.0 + r).cumprod()])
    return out


def first_check_stats(P, lg, S, bench, top_frac=0.05):
    """現行交易逐筆：第一次換股檢查（進場後第一個訊號列 c1）有沒有過。
    沒過＝c1 當天被出場規則標記（X[c1, j]）或在 c1 以前就出場（下市）；過＝續抱。尚未遇到檢查（資料尾端）另列。
    gain1＝原進場日開盤 → 第一次檢查次日（換股執行日）開盤的含息累積報酬（不扣成本）＝「補買那一份比第一份買貴多少」。
    after＝(1＋整筆淨報酬) ÷ (1＋gain1) − 1（檢查次日開盤買進、抱到出場的近似報酬；未另扣補買成本）；after_ex＝再減 0050 同期間（c1＋1 收盤 → 出場日收盤）。"""
    sigrows = np.flatnonzero(S['sigrows']); X = S['X']; C, O, R, REF = P.CF, P.O, P.R, P.REF
    b = bench.reindex(P.IDX).to_numpy()
    rows = []
    for r in lg.itertuples():
        j, e, x = int(r.j), int(r.ent_t), int(r.exit_t)
        nx = sigrows[sigrows > e]
        ex_all = r.ret - (b[x] / b[e] - 1)
        n_checks_passed = int(sum((not X[c, j]) for c in nx[nx < x])) if len(nx) else 0
        if not len(nx) or nx[0] + 1 >= P.T:
            rows.append(dict(j=j, ent_t=e, grp='no_check', ret=r.ret, ex=ex_all, pnl=r.pnl, cost=r.cost, hold=x - e, gain1=np.nan, after=np.nan, after_ex=np.nan, pass2=False)); continue
        c1 = int(nx[0])
        if x <= c1 or X[c1, j]:
            rows.append(dict(j=j, ent_t=e, grp='fail1', ret=r.ret, ex=ex_all, pnl=r.pnl, cost=r.cost, hold=x - e, gain1=np.nan, after=np.nan, after_ex=np.nan, pass2=False)); continue
        g = C[e, j] / O[e, j] * float(np.prod(1.0 + np.nan_to_num(R[e + 1:c1 + 1, j])))
        o1 = O[c1 + 1, j]
        g *= (o1 / REF[c1 + 1, j]) if np.isfinite(o1) and o1 > 0 else 1.0
        after = (1 + r.ret) / g - 1
        t1 = min(c1 + 1, x)
        pass2 = bool(len(nx) > 1 and x > nx[1] and not X[int(nx[1]), j])
        rows.append(dict(j=j, ent_t=e, grp='pass1', ret=r.ret, ex=ex_all, pnl=r.pnl, cost=r.cost, hold=x - e, gain1=g - 1, after=after, after_ex=after - (b[x] / b[t1] - 1), pass2=pass2))
    d = pd.DataFrame(rows)
    n = len(d); m = int(math.ceil(n * top_frac)); top = np.zeros(n, bool); top[np.argsort(-d['ret'].to_numpy(), kind='stable')[:m]] = True
    d['big'] = top

    def blk(y, cols=('ret', 'ex')):
        o = dict(n=int(len(y)), pnl_sum=float(y['pnl'].sum()), hold_mean=float(y['hold'].mean()) if len(y) else None)
        for c in cols:
            v = y[c].dropna().to_numpy()
            o[c] = dict(mean=float(v.mean()), median=float(np.median(v)), win=float((v > 0).mean()), q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)), min=float(v.min()), max=float(v.max())) if len(v) else None
        return o
    p1 = d[d['grp'] == 'pass1']
    out = dict(n_all=n, n_big=m, big_min_ret=float(d.loc[d['big'], 'ret'].min()),
               fail1=blk(d[d['grp'] == 'fail1']), pass1=blk(p1, ('ret', 'ex', 'gain1', 'after', 'after_ex')), no_check=blk(d[d['grp'] == 'no_check']),
               pass1_then_fail2=blk(p1[~p1['pass2']], ('ret', 'gain1', 'after', 'after_ex')), pass2=blk(p1[p1['pass2']], ('ret', 'gain1', 'after', 'after_ex')),
               big=dict(by_group={g_: int(((d['grp'] == g_) & d['big']).sum()) for g_ in ('fail1', 'pass1', 'no_check')}, pass1=blk(d[d['big'] & (d['grp'] == 'pass1')], ('ret', 'gain1', 'after', 'after_ex')),
                        n_pass2=int((d['big'] & d['pass2']).sum())),
               rest=dict(by_group={g_: int(((d['grp'] == g_) & ~d['big']).sum()) for g_ in ('fail1', 'pass1', 'no_check')}, fail1=blk(d[~d['big'] & (d['grp'] == 'fail1')]),
                         pass1=blk(d[~d['big'] & (d['grp'] == 'pass1')], ('ret', 'ex', 'gain1', 'after', 'after_ex'))))
    return out, d


def conc_stats(lg, lots, bench, top_frac=0.05):
    """集中度（R98 口徑＋分批的處理）。lg＝部位明細（一檔一列；分批時 cost／proceeds 是兩批合計，ret＝金額加權）；lots＝各批明細（現行＝每部位一批）。
    每批超額＝該批報酬 − 0050 同期間（該批進場日收盤 → 出場日收盤）。部位超額＝各批超額依成本加權。
    前 5%＝依部位報酬率由高到低取 ceil(5% × 部位數)。回傳：前 5% 占全部部位損益比例、其餘 95% 的等權平均超額（部位為單位）、金額加權平均超額（Σ 成本×超額 ÷ Σ 成本）等。"""
    b = bench
    lt = lots.copy()
    lt['bench'] = b.reindex(pd.DatetimeIndex(lt['exit_date'])).to_numpy() / b.reindex(pd.DatetimeIndex(lt['ent_date'])).to_numpy() - 1
    lt['ex'] = lt['ret'] - lt['bench']; lt['expnl'] = lt['cost'] * lt['ex']
    key = ['j', 'sig_t']
    assert not lg.duplicated(key).any()
    agg = lt.groupby(key).agg(expnl=('expnl', 'sum'), lcost=('cost', 'sum')).reset_index()
    x = lg.merge(agg, on=key, how='left'); assert x['expnl'].notna().all() and np.allclose(x['lcost'], x['cost'], rtol=1e-9)
    x['ex'] = x['expnl'] / x['cost']
    n = len(x); m = int(math.ceil(n * top_frac)); o = np.argsort(-x['ret'].to_numpy(), kind='stable')[:m]
    top = np.zeros(n, bool); top[o] = True
    rest = x[~top]; tp = x[top]
    ltm = lt.merge(x.assign(_top=top)[key + ['_top']], on=key)
    lrest = ltm[~ltm['_top']]
    return dict(n_positions=int(n), n_lots=int(len(lt)), n_top=m, top_min_ret=float(tp['ret'].min()), pnl_sum=float(x['pnl'].sum()), top_pnl_sum=float(tp['pnl'].sum()),
                top_share_of_pnl=float(tp['pnl'].sum() / x['pnl'].sum()), top_cost_share=float(tp['cost'].sum() / x['cost'].sum()),
                all_mean_ex=float(x['ex'].mean()), all_mean_ret=float(x['ret'].mean()), all_cw_ex=float(x['expnl'].sum() / x['cost'].sum()),
                rest_n=int(len(rest)), rest_mean_ret=float(rest['ret'].mean()), rest_mean_ex=float(rest['ex'].mean()), rest_median_ex=float(rest['ex'].median()), rest_frac_ex_pos=float((rest['ex'] > 0).mean()),
                rest_cw_ex=float(rest['expnl'].sum() / rest['cost'].sum()), rest_pnl_sum=float(rest['pnl'].sum()), rest_expnl_sum=float(rest['expnl'].sum()), rest_cost_sum=float(rest['cost'].sum()),
                rest_lots_n=int(len(lrest)), rest_lots_mean_ex=float(lrest['ex'].mean()), top_expnl_sum=float(tp['expnl'].sum()))
