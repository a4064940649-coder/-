#!/usr/bin/env python3
"""S67 第 0 步（寫 PREREG 之前；只算現行，不算任何候選的報酬）：python3 -I -B prep.py → data_check.json
- P0 面板；正式開關全開；
- A0 現行 100 萬位移 0 逐位重現 2019 起 48.76%／全期 34.40%／回撤 −32.12%／已平倉 381 筆（與 deadline_engine_fix ALL 逐位相同）；
- A2 現行 20 條路徑與 deadline_engine_fix ALL 逐條相同；
- A3 引擎副本 simulate_s67（mult=None）在 20 條連續路徑＋重新起跑主跑＋禁買主跑與原引擎 simulate_s10 逐位相同；
- A4 名單重建（pool_of）在全部訊號列 × 位移 0～3 與 make_signal_s10 的名單逐檔相同（C3 用的前 10 名重建正確）；
- 第 0 步描述（現行口徑，主跑 100 萬位移 0）：
    每月名單檔數、新買進檔數分布（全樣本只算「檔數」；報酬比較在寫 PREREG 前只看訓練期＝訊號 2016-02～2021-12）；
    名單 ≤2 檔月份買進的交易 vs 3～4 檔 vs ≥5 檔：筆數、平均／中位數報酬、勝率、同持有期 0050 報酬、超額（逐筆與逐月平均）。
- 候選涉及的「檔數」資訊（不含報酬）：名單太少月份數、C3 可補進檔數。"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s67lib as L
from s67lib import s10lib, s2lib, stlib, cslib

H = Path(__file__).resolve().parent
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = L.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
U, RT = L.base(P)
fix = L.s17lib.e50_fix_suspension(P)
L.slim(P); gc.collect()
print('載入完成', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19, TB = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01'), lastb('2022-01-01')
dl = json.loads((L.AG / 'deadline_engine_fix' / 'out' / 'results.json').read_text())
cols = ['j', 'ent_t', 'exit_t', 'why', 'shares', 'cost', 'proceeds']
same = lambda a, b: bool(np.array_equal(a[0].to_numpy(), b[0].to_numpy(), equal_nan=True) and a[1][cols].equals(b[1][cols]))
out = dict(flags=dict(EXRIGHT_FIX=cslib.EXRIGHT_FIX, DEADLINE_FIX=cslib.DEADLINE_FIX, INS15_MASK=cslib.INS15_MASK),
           P0=dict(T=P.T, N=P.N, first=str(D0.date()), last=str(DE.date()), train_end=str(TB.date())), e50_fix=fix, src_sha256=L.SRC_SHA, paths={}, A3={}, A4={})
a3 = True; a4 = True
for s in L.SHIFTS:
    sg = s10lib.make_signal_s10(P, U, RT, shift=s)
    bad = 0
    for x in sg['info']:
        pool, sc, r20 = L.pool_of(P, U, RT, x)
        tgt = pool[r20[pool] > L.MOM_LO][:L.NSEL]
        bad += int(set(int(j) for j in tgt) != set(int(j) for j in np.flatnonzero(sg['sig'][x['k']])))
    out['A4'][f'+{s}'] = dict(months=len(sg['info']), mismatch=bad); a4 &= bad == 0
    for cap in L.CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        r0 = L.sim(P, sg, U, cap=cap, orig=True)
        r1 = L.sim(P, sg, U, cap=cap)
        e = same(r0, r1); out['A3'][key] = e; a3 &= e
        nav, lg, st = r0
        out['paths'][key] = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE))
        if s == 0 and cap == 1e6:
            main = dict(out['paths'][key], n_closed=int((lg['why'] != 'open_end').sum()))
            S0, LG0 = sg, lg
            f0 = L.sim(P, sg, U, cap=cap, fresh=True, orig=True); f1 = L.sim(P, sg, U, cap=cap, fresh=True)
            out['A3']['fresh|100萬|+0'] = same(f0, f1); a3 &= out['A3']['fresh|100萬|+0']
            out['fresh'] = dict(no_trade_before_2022=bool((f0[1]['ent_date'] >= pd.Timestamp('2022-01-01')).all()), first_entry=str(f0[1]['ent_date'].min().date()),
                                ratio_nav_TB=float(f0[0].loc[TB] / 1e6))
            blk = s2lib.block_from(f0[1], 5)[0]
            b0 = L.sim(P, sg, U, cap=cap, fresh=True, block=blk, orig=True); b1 = L.sim(P, sg, U, cap=cap, fresh=True, block=blk)
            out['A3']['fresh_block|100萬|+0'] = same(b0, b1); a3 &= out['A3']['fresh_block|100萬|+0']
            del f0, f1, b0, b1
        del r0, r1
    gc.collect()
    print('位移', s, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
out['A0'] = dict(main, tol_ok=bool(abs(main['since2019'] - L.ACC['since2019']) < 2.5e-4 and abs(main['full'] - L.ACC['full']) < 2.5e-4
                                   and abs(main['maxdd'] - L.ACC['maxdd']) < 2.5e-4 and main['n_closed'] == L.ACC['n_closed']),
                 bitwise_vs_deadline_ALL=bool(main['since2019'] == dl['standard']['ALL']['since2019'] and main['full'] == dl['standard']['ALL']['full']
                                              and main['maxdd'] == dl['standard']['ALL']['maxdd_full']))
d2 = [abs(out['paths'][k]['since2019'] - v['since2019']) for k, v in dl['paths']['ALL'].items()]
out['A2'] = dict(n=len(d2), max_abs_diff=float(max(d2)), ok=bool(len(d2) == 20 and max(d2) < 1e-9))
out['A3_all_ok'] = bool(a3); out['A4_all_ok'] = bool(a4)

# ---------------------------------------------------------------- 第 0 步描述（現行，主跑）
info = pd.DataFrame([dict(month=x['month'], k=x['k'], sig_date=x['sig_date'], n_tgt=x['n_tgt']) for x in S0['info']])
nb = LG0.groupby('sig_t').size()
info['n_new'] = info['k'].map(nb).fillna(0).astype(int)
info['train'] = pd.to_datetime(info['sig_date']) < pd.Timestamp('2022-01-01')
info['in_train_window'] = info['train'] & (pd.to_datetime(info['sig_date']) >= pd.Timestamp('2016-02-01'))


def dist(v):
    v = np.asarray(v)
    return dict(n_months=int(len(v)), mean=float(v.mean()), median=float(np.median(v)), p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)),
                counts={str(i): int((v == i).sum()) for i in range(int(v.max()) + 1)}, n_le2=int((v <= 2).sum()), frac_le2=float((v <= 2).mean()),
                n_zero=int((v == 0).sum()), n_ge5=int((v >= 5).sum()))


step0 = dict(note='檔數分布含全樣本（只是檔數，不是報酬）；報酬比較在寫 PREREG 前只算訓練期（訊號 2016-02～2021-12）。',
             list_size=dict(all=dist(info['n_tgt']), train_window=dist(info.loc[info['in_train_window'], 'n_tgt']), valid=dist(info.loc[~info['train'], 'n_tgt'])),
             new_buys=dict(all=dist(info['n_new']), train_window=dist(info.loc[info['in_train_window'], 'n_new']), valid=dist(info.loc[~info['train'], 'n_new'])),
             few_months_all=[dict(month=r.month, sig_date=r.sig_date, n_tgt=int(r.n_tgt), n_new=int(r.n_new)) for r in info[info['n_tgt'] <= L.FEW].itertuples()])
lg = LG0.copy()
lg['n_tgt'] = lg['sig_t'].map(dict(zip(info['k'], info['n_tgt'])))
assert lg['n_tgt'].notna().all()
e50 = P.E50; e50o = P.E50O
lg['r50'] = e50[lg['exit_t'].to_numpy()] / e50o[lg['ent_t'].to_numpy()] - 1       # 同持有期 0050（進場日開盤 → 出場日收盤；毛報酬）
lg['ex'] = lg['ret'] - lg['r50']
lg['grp'] = np.where(lg['n_tgt'] <= L.FEW, '≤2', np.where(lg['n_tgt'] >= 5, '≥5', '3～4'))


def grp_stats(x):
    o = {}
    for g in ('≤2', '3～4', '≥5'):
        y = x[x['grp'] == g]
        if not len(y):
            o[g] = dict(n=0); continue
        mm = y.groupby('sig_t')['ex'].mean()
        o[g] = dict(n=int(len(y)), n_months=int(y['sig_t'].nunique()), mean_ret=float(y['ret'].mean()), median_ret=float(y['ret'].median()), win=float((y['ret'] > 0).mean()),
                    mean_r50=float(y['r50'].mean()), mean_ex=float(y['ex'].mean()), median_ex=float(y['ex'].median()), win_vs_0050=float((y['ex'] > 0).mean()),
                    month_mean_ex=float(mm.mean()), month_t=(float(mm.mean() / (mm.std(ddof=1) / np.sqrt(len(mm)))) if len(mm) > 2 else None),
                    avg_hold_days=float(y['hold_days'].mean()), n_open_end=int((y['why'] == 'open_end').sum()),
                    pnl_sum=float(y['pnl'].sum()), pnl_share=float(y['pnl'].sum() / x['pnl'].sum()))
    return o


tr = lg[(lg['sig_date'] >= pd.Timestamp('2016-02-01')) & (lg['sig_date'] < pd.Timestamp('2022-01-01'))]
step0['trades_train_window'] = grp_stats(tr)
# 訓練期名單太少月份中 C3 可補進幾檔（只是檔數）
c3 = []
for x in S0['info']:
    if x['n_tgt'] <= L.FEW and x['sig_date'] < '2022-01-01':
        pool, sc, r20 = L.pool_of(P, U, RT, x)
        with np.errstate(invalid='ignore'):
            ex = pool[(r20[pool] > L.C3_LO) & ~(r20[pool] > L.MOM_LO)]
        c3.append(dict(month=x['month'], n_tgt=x['n_tgt'], n_5to10=int(len(ex)), n_add=int(min(len(ex), max(L.C3_FILL - x['n_tgt'], 0)))))
step0['c3_fill_counts_train'] = dict(months=len(c3), total_add=int(sum(c['n_add'] for c in c3)), months_reach3=int(sum(c['n_tgt'] + c['n_add'] >= 3 for c in c3)), detail=c3)
out['step0'] = step0
out['runtime_s'] = round(time.time() - t0, 1); out['maxrss_mb'] = rss()
(H / 'data_check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
print(json.dumps({k: out[k] for k in ('A0', 'A2', 'A3_all_ok', 'A4_all_ok', 'fresh', 'runtime_s', 'maxrss_mb')}, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in step0.items() if k not in ('few_months_all',)}, ensure_ascii=False, indent=1)[:6000])
assert out['A0']['tol_ok'] and out['A0']['bitwise_vs_deadline_ALL'] and out['A2']['ok'] and a3 and a4 and out['fresh']['no_trade_before_2022'], '驗收失敗'
