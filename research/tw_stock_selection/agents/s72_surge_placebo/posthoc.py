#!/usr/bin/env python3
"""S72 事後補充（不在 PREREG 內、不影響判定；看過主結果後才寫）：
主結果發現 P1 隨機排名每月只選到 2.6 檔（現行 4.0 檔）、平均持股 2.4 檔（現行 4.7 檔）→ 安慰劑有較多錢停在 0050。
H1 持股檔數對齊：隨機排名但把名單放寬成「前 15 名進場／前 30 名續抱」與「前 20／前 40」（其餘不動），各 100 次、標準路徑。
H2 每筆交易：現行與 P1（前 100 次）已出場交易的平均報酬、中位數、勝率、損益合計 ÷ 起始資金。
用法：python3 -I -B posthoc.py（重用 run.py 的載入段與函式；輸出 posthoc.json）。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
src = (HERE / 'run.py').read_text()
cut = src.index('# ---------------------------------------------------------------- 驗收（現行格）')
sys.argv = [str(HERE / 'run.py'), '--quick']
g = {'__file__': str(HERE / 'run.py'), '__name__': 's72run'}
exec(compile(src[:cut], str(HERE / 'run.py'), 'exec'), g)
np, P, U, S0, AV, SARR, cslib, s10lib, COSTS, CELL = (g[k] for k in ('np', 'P', 'U', 'S0', 'AV', 'SARR', 'cslib', 's10lib', 'COSTS', 'CELL'))
metrics, scorefn, build = g['metrics'], g['scorefn'], g['build']


def build2(fn, depth, nexit):
    sg = build(0, fn, depth=depth)
    if nexit != 20:                                        # 重建出場矩陣（同一個迴圈、同條件）
        X = np.zeros((P.T, P.N), bool)
        for mi, m in enumerate(S0.index):
            k0 = AV[m]
            if k0 >= P.T or k0 < 61:
                continue
            s = SARR[mi]
            if int((U[k0] & np.isfinite(s)).sum()) < 50:
                continue
            k = k0
            s = cslib.ins15_mask_row(P, m, k, s)
            q = fn(mi, k, s)
            ok = U[k] & np.isfinite(s) & np.isfinite(q) & P.F_MA60[k]
            o = np.flatnonzero(ok)[np.argsort(-q[ok], kind='stable')]
            X[k, :] = True; X[k, o[:nexit]] = False
        sg['X'] = X
    return sg


def one(fn, depth, nexit):
    sg = build2(fn, depth, nexit)
    nav, lg, st = s10lib.simulate_s10(P, sg, U, cost=COSTS['strict'], capital=1e6, **CELL)
    m = metrics(nav, lg, st, sg)
    cl = lg[lg['why'] != 'open_end']
    m.update(trade_ret_mean=float(cl['ret'].mean()), trade_ret_median=float(cl['ret'].median()), win_rate=float((cl['ret'] > 0).mean()),
             pnl_sum_over_capital=float(cl['pnl'].sum() / 1e6))
    return m


F = ['since2019', 'full', 'maxdd_full', 'ex_since2019', 'avg_npos', 'n_tgt_mean', 'n_closed', 'hold_mean', 'turnover', 'trade_ret_mean', 'trade_ret_median', 'win_rate',
     'seg16_18', 'seg19_23', 'seg24']
cur = one(scorefn('CUR', 0), 10, 20)
assert cur['since2019'] == 0.4875617247721724
out = dict(note='事後補充，不影響判定', current={f: cur[f] for f in F})
for name, depth, nexit in (('P1_base_10_20', 10, 20), ('H1_15_30', 15, 30), ('H1_20_40', 20, 40)):
    rs = [one(scorefn('P1', i), depth, nexit) for i in range(100)]
    d = {}
    for f in F:
        a = np.array([r[f] for r in rs])
        d[f] = dict(median=float(np.median(a)), p5=float(np.percentile(a, 5)), p95=float(np.percentile(a, 95)), max=float(a.max()),
                    current=cur[f], current_percentile=float((a < cur[f]).mean() * 100), p_one_sided=float((1 + (a >= cur[f]).sum()) / (len(a) + 1)))
    out[name] = d
    print(name, {f: (round(d[f]['median'], 4), round(d[f]['p5'], 4), round(d[f]['p95'], 4), round(d[f]['max'], 4)) for f in F}, flush=True)
print('CUR', {f: round(cur[f], 4) for f in F})
(HERE / 'posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
