#!/usr/bin/env python3
"""S3 共用程式庫：在 S2 的 D1（SURGE＋季線）上加兩個細節——寬出場（掉出 L_MA 前 N 名才賣）、進場前 20 日漲幅門檻。
引擎、成本、統計全部直接呼叫 s2lib／stlib（不改）；這裡只做訊號矩陣、交易特徵與兩個診斷。
一律 python3 -I -B。中間檔（面板快取）放 scratchpad/s3/。規格見 PREREG.json。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S3 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s3')
sys.path.insert(0, str(AG / 's2_surge_strict_details'))
import s2lib                                               # noqa: E402
from s2lib import stlib, cslib, COSTS                      # noqa: E402,F401
s2lib.S2 = S3; stlib.S1 = S3; cslib.TMP = S3              # 面板快取放本題的 scratchpad（不動 s1、s2 的）


def load():
    P = s2lib.load()
    with np.errstate(invalid='ignore', divide='ignore'):
        P.RET20 = P.G / cslib.shift(P.G, 20) - 1           # 還原收盤 ÷ 20 個交易日前的還原收盤 − 1
        P.AO = np.where(np.isfinite(P.O) & (P.O > 0), P.O * P.G / P.CF, cslib.shift(P.G, 1))   # 還原開盤（無開盤價用前一日還原收盤）
    return P


def make_signal(P, univ, RT, nsel=10, mom=None, hold_n=None, shift=0):
    """每月一次。L_MA＝訊號日股池內、SURGE 有值、還原收盤 > 60 日均線，依 SURGE 由大到小（stable）。
    買進名單＝L_MA（mom 有值時再加 RET20 > mom）的前 nsel 名。
    出場：hold_n=None → 不在買進名單就賣（D1／E2 的機制）；hold_n=N → 不在 L_MA 前 N 名才賣（E1／E3／E4）。
    有訊號的月份一律以未位移訊號日股池內 SURGE 有值 ≥ 50 檔決定（同 s2lib）。"""
    S, av = RT['S'], RT['av']
    sig = np.zeros((P.T, P.N), bool); X = np.zeros((P.T, P.N), bool); rank = np.full((P.T, P.N), np.nan)
    info = []; pos_ma = {}; alt20 = {}
    for m in S.index:
        k0 = av[m]
        if k0 >= P.T or k0 < 61:
            continue
        s = S.loc[m].to_numpy()
        if int((univ[k0] & np.isfinite(s)).sum()) < 50:
            continue
        k = k0 + shift
        if k >= P.T:
            continue
        ok_all = univ[k] & np.isfinite(s)
        ok_ma = ok_all & P.F_MA60[k]
        o_ma = np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]
        ok_ent = ok_ma if mom is None else (ok_ma & (P.RET20[k] > mom))
        o_ent = np.flatnonzero(ok_ent)[np.argsort(-s[ok_ent], kind='stable')]
        tgt = o_ent[:nsel]
        keep = tgt if hold_n is None else o_ma[:hold_n]
        sig[k, tgt] = True; rank[k, tgt] = s[tgt]
        X[k, :] = True; X[k, keep] = False
        pos_ma[k] = {int(j): i + 1 for i, j in enumerate(o_ma)}                 # L_MA 名次（1 起算）
        o_all = np.flatnonzero(ok_all)[np.argsort(-s[ok_all], kind='stable')]
        alt20[k] = set(int(j) for j in o_all[:20] if ok_ma[j])                   # 另一種讀法：全股池前 20 名且在季線上
        nth = lambda n: float(s[o_ma[n - 1]]) if len(o_ma) >= n else float('nan')
        info.append(dict(month=str(m), k=int(k), sig_date=str(P.IDX[k].date()), n_ma=int(ok_ma.sum()), n_ent=int(ok_ent.sum()),
                         targets=[str(P.COLS[j]) for j in tgt], scores=[round(float(s[j]), 4) for j in tgt],
                         surge_10th=nth(10), surge_20th=nth(20), surge_30th=nth(30)))
    return dict(sig=sig, rank=rank, X=X, SD=None, max_hold=None, info=info, pos_ma=pos_ma, alt20=alt20)


def npos_series(P, lg):
    """由交易紀錄重建每日收盤持股檔數（進場列號 ≤ t < 出場列號；期末未平倉算到最後一天）。"""
    d = np.zeros(P.T + 1)
    for r in lg.itertuples():
        d[int(r.ent_t)] += 1
        d[(P.T if r.why == 'open_end' else int(r.exit_t))] -= 1
    return np.cumsum(d)[:P.T]


def extra_stats(P, nav, lg, st, n_sig_days):
    """平均持股檔數、閒置比例（全期與第一筆進場日起）、年化單邊周轉、每個訊號日平均賣出筆數。"""
    t1 = int(lg['ent_t'].min()); nd = P.T - 1 - P.t0; nblank = t1 - P.t0 - 1
    npos = npos_series(P, lg)
    v = nav.to_numpy()[t1 - P.t0:]
    yrs = (P.IDX[-1] - P.IDX[t1]).days / 365.25
    closed = lg[lg['why'] != 'open_end']
    return dict(avg_npos_full=float(st['avg_npos']), avg_idle_full=float(st['avg_idle_frac']),
                avg_npos_from_first=float(npos[t1:].mean()), avg_idle_from_first=float((st['avg_idle_frac'] * nd - nblank) / (nd - nblank)),
                first_entry=str(P.IDX[t1].date()), turnover_oneway_ann=float((st['buy_val'] + st['sell_val']) / 2 / v.mean() / yrs),
                sells_per_signal_day=float(len(closed) / n_sig_days), buys=int(st['n_buy']), noslot=int(st['noslot']),
                blk_limup=int(st['blk_limup']), blk_lot=int(st['blk_lot']), blk_nocash=int(st['blk_nocash']), blk_cap=int(st['blk_cap']),
                roundtrip_cost_pct=float((st['fee'] + st['tax'] + st['slip']) / st['buy_val']) if st['buy_val'] > 0 else None)


def feats(P, S, lg, exec_days):
    """進場當下（訊號日）的特徵＋這筆交易跨了幾個換股月。"""
    lg = lg.copy()
    k = lg['sig_t'].to_numpy(); j = lg['j'].to_numpy()
    lg['surge'] = S['rank'][k, j]
    lg['pre20'] = P.RET20[k, j]
    lg['pos_ma'] = [S['pos_ma'].get(int(a), {}).get(int(b), np.nan) for a, b in zip(k, j)]
    lg['close_sig'] = P.C[k, j]; lg['v60'] = P.V60[k, j]
    lg['near_hi240'] = P.NEARHI[k, j]
    ex = np.asarray(exec_days)
    lg['n_months'] = [1 + int(((ex > a) & (ex < b)).sum()) for a, b in zip(lg['ent_t'], lg['exit_t'])]
    return lg


def _desc(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(n=0)
    return dict(n=int(len(x)), mean=float(x.mean()), median=float(np.median(x)), win=float((x > 0).mean()), sum=float(x.sum()))


def segments(P, S, lg):
    """每筆交易依換股成交日切段（不含成本、還原開盤對還原開盤）。回傳 DataFrame：trade 序號、第幾個月、報酬、段首決定日在 L_MA 的名次。"""
    ex = np.array(sorted(k + 1 for k in S['pos_ma'] if k + 1 < P.T))
    rows = []
    for i, r in enumerate(lg.itertuples()):
        j, e, x = int(r.j), int(r.ent_t), int(r.exit_t)
        b = [e] + [int(t) for t in ex[(ex > e) & (ex < x)]] + [x]
        for n in range(len(b) - 1):
            p0 = P.AO[b[n], j]
            last = (n == len(b) - 2)
            if last and r.why == 'open_end':
                p1 = P.G[x, j]
            elif last and r.why == 'delist':
                p1 = P.G[x - 1, j]
            else:
                p1 = P.AO[b[n + 1], j]
            rows.append(dict(trade=i, j=j, month_no=n + 1, t0=b[n], t1=b[n + 1], ret=float(p1 / p0 - 1),
                             pos_ma=S['pos_ma'].get(b[n] - 1, {}).get(j, np.nan), ent_year=int(P.IDX[e].year), last=last, why=r.why))
    return pd.DataFrame(rows)


def diag1(P, S, lg, hold_n=20):
    """E1 續抱的部位：各月別的當月報酬；續抱當下的 L_MA 名次；被擠掉的新名單；另一種讀法不同的次數。
    殘段＝第 2 段以後、段首決定日名次已不在前 hold_n 名（規則要賣，但開盤跌停／無開盤價順延，或受成交值 5% 上限分批賣）——不算續抱，另外列。"""
    sg = segments(P, S, lg)
    sg['resid'] = (sg['month_no'] >= 2) & ~(sg['pos_ma'] <= hold_n)
    out = {'by_month_no': {}, 'by_month_no_since2019': {}}
    ok = ~sg['resid']
    for lab, f in (('1', sg['month_no'] == 1), ('2', sg['month_no'] == 2), ('3', sg['month_no'] == 3), ('4+', sg['month_no'] >= 4), ('2+', sg['month_no'] >= 2)):
        out['by_month_no'][lab] = _desc(sg.loc[f & ok, 'ret'])
        out['by_month_no_since2019'][lab] = _desc(sg.loc[f & ok & (sg['ent_year'] >= 2019), 'ret'])
    out['residual_segments'] = _desc(sg.loc[sg['resid'], 'ret'])
    out['residual_segments']['avg_days'] = float((sg.loc[sg['resid'], 't1'] - sg.loc[sg['resid'], 't0']).mean()) if sg['resid'].any() else None
    c = sg[sg['month_no'] >= 2]
    out['cont_by_rank'] = {'1-10': _desc(c.loc[c['pos_ma'] <= 10, 'ret']), '11-20': _desc(c.loc[(c['pos_ma'] > 10) & (c['pos_ma'] <= 20), 'ret']),
                           'other': _desc(c.loc[~(c['pos_ma'] <= 20), 'ret'])}
    for nm, a, b in (('half1', 0, 2020), ('half2', 2021, 9999)):
        cc = c[(P.IDX[c['t0'].to_numpy()].year >= a) & (P.IDX[c['t0'].to_numpy()].year <= b)]
        out['cont_by_rank_' + nm] = {'1-10': _desc(cc.loc[cc['pos_ma'] <= 10, 'ret']), '11-20': _desc(cc.loc[(cc['pos_ma'] > 10) & (cc['pos_ma'] <= 20), 'ret'])}
    out['cont_rank_hist'] = {str(int(a)): int(b) for a, b in c['pos_ma'].value_counts().sort_index().items()}
    # 被擠掉的新名單：換股後持股滿 10 檔的訊號日，L_MA 前 10 名中沒持有的（下一段還原開盤對還原開盤）
    ks = sorted(S['pos_ma']); ex = [k + 1 for k in ks]
    ent, exi, jj = lg['ent_t'].to_numpy(), lg['exit_t'].to_numpy(), lg['j'].to_numpy()
    opn = (lg['why'] == 'open_end').to_numpy()
    disp, newb, alt_diff, alt_tot, n_full = [], [], 0, 0, 0
    for n, k in enumerate(ks):
        e = k + 1
        if e >= P.T:
            continue
        e2 = ex[n + 1] if n + 1 < len(ex) and ex[n + 1] < P.T else None
        held_after = set(int(a) for a in jj[(ent <= e) & ((exi > e) | opn)])
        held_before = [int(a) for a in jj[(ent <= k) & ((exi > k) | opn)]]
        for j in held_before:
            a = S['pos_ma'][k].get(j, 99999) <= 20; b = j in S['alt20'][k]
            alt_tot += 1; alt_diff += int(a != b)
        tg = [j for j in np.flatnonzero(S['sig'][k])]
        if len(held_after) >= 10:
            n_full += 1
            for j in tg:
                if int(j) not in held_after:
                    p0 = P.AO[e, j]; p1 = P.AO[e2, j] if e2 is not None else P.G[P.T - 1, j]
                    disp.append(float(p1 / p0 - 1))
        for j in jj[ent == e]:
            p0 = P.AO[e, j]; p1 = P.AO[e2, j] if e2 is not None else P.G[P.T - 1, j]
            newb.append(float(p1 / p0 - 1))
    out['displaced_top10_not_held_when_full'] = _desc(disp); out['n_signal_days_full'] = n_full
    out['newly_bought_same_window'] = _desc(newb)
    out['alt_reading'] = dict(decisions=alt_tot, differ=alt_diff)
    out['months_per_trade_hist'] = {str(int(a)): int(b) for a, b in sg[~sg['resid']].groupby('trade')['month_no'].max().value_counts().sort_index().items()}
    return out, sg


def diag2(lgD1, lgE2, S_E2, stE2x):
    """E2 擋掉的是什麼：D1 已平倉交易依進場訊號日前 20 日漲幅分組。"""
    out = {}
    for nm, lg in (('D1', lgD1), ('E2', lgE2)):
        x = lg[lg['why'] != 'open_end']
        g = {'le10': x['pre20'] <= 0.10, 'gt10': x['pre20'] > 0.10, 'gt10_le30': (x['pre20'] > 0.10) & (x['pre20'] <= 0.30), 'gt30': x['pre20'] > 0.30,
             'le5': x['pre20'] <= 0.05, 'gt5_le10': (x['pre20'] > 0.05) & (x['pre20'] <= 0.10), 'gt10_le15': (x['pre20'] > 0.10) & (x['pre20'] <= 0.15), 'gt15': x['pre20'] > 0.15}
        out[nm] = {a: _desc(x.loc[f, 'ret']) for a, f in g.items()}
        for a, f in g.items():
            out[nm][a]['avg_hold'] = float(x.loc[f, 'hold_days'].mean()) if f.any() else None
            out[nm][a]['loss_gt15'] = float((x.loc[f, 'ret'] < -0.15).mean()) if f.any() else None
        y = x[x['ent_date'] >= pd.Timestamp('2019-01-01')]
        out[nm + '_since2019'] = {'le10': _desc(y.loc[y['pre20'] <= 0.10, 'ret']), 'gt10': _desc(y.loc[y['pre20'] > 0.10, 'ret']), 'gt30': _desc(y.loc[y['pre20'] > 0.30, 'ret'])}
        h = pd.Timestamp('2021-02-19')
        out[nm + '_half1'] = {'le10': _desc(x.loc[(x['pre20'] <= 0.10) & (x['ent_date'] <= h), 'ret']), 'gt10': _desc(x.loc[(x['pre20'] > 0.10) & (x['ent_date'] <= h), 'ret']),
                              'gt30': _desc(x.loc[(x['pre20'] > 0.30) & (x['ent_date'] <= h), 'ret'])}
        out[nm + '_half2'] = {'le10': _desc(x.loc[(x['pre20'] <= 0.10) & (x['ent_date'] > h), 'ret']), 'gt10': _desc(x.loc[(x['pre20'] > 0.10) & (x['ent_date'] > h), 'ret']),
                              'gt30': _desc(x.loc[(x['pre20'] > 0.30) & (x['ent_date'] > h), 'ret'])}
    ne = np.array([i['n_ent'] for i in S_E2['info']])
    out['E2_list'] = dict(n_signal_days=int(len(ne)), months_lt10=int((ne < 10).sum()), months_lt5=int((ne < 5).sum()), median_n=float(np.median(ne)), min_n=int(ne.min()),
                          mean_targets=float(np.minimum(ne, 10).mean()))
    out['E2_hold'] = stE2x
    return out
