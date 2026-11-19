#!/usr/bin/env python3
"""S74 事後補充（未登記，看過主結果後才加；不影響 PREREG 用詞判定）：
主分析的事件是『首次跌破日』，深度天生很淺（沒有 < −10% 的事件）。網站每天顯示的是『今天離季線多遠』，所以補一張『持股日』表：
全部持有期間內、非訊號列、還原收盤 ≤ 60 日均線的每一個持股日（同一檔會出現很多天），當日收盤 → 下一個換股執行日開盤對 0050 的超額，
依當日深度、首次跌破後已過交易日數、距換股執行日天數分組。群聚 t 以換股月份分群（同月同檔的重複日高度相關，n 不是獨立樣本數）。
用法：python3 -I -B posthoc.py（會先整個執行 run.py）。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as R                                            # noqa: E402
P, lg, BELOW, DEPTH, G, GO, E50, E50O, sr, sk, T = R.P, R.lg, R.BELOW, R.DEPTH, R.G, R.GO, R.E50, R.E50O, R.sr, R.sk, R.T
rows = []
for hid, h in enumerate(lg.itertuples(index=False)):
    j, a, b = int(h.j), int(h.ent_t), int(h.exit_t)
    last = b if h.why == 'open_end' else b - 1
    bl = np.flatnonzero(BELOW[a:last + 1, j])
    if not len(bl):
        continue
    t1 = a + int(bl[0])
    for t in a + bl:
        t = int(t)
        if sr[t]:
            continue
        nk = sk[sk > t]
        if not len(nk) or int(nk[0]) + 1 >= T:
            continue
        k = int(nk[0]); e = k + 1
        if not np.isfinite(GO[e, j]):
            continue
        run_len = 1
        while t - run_len >= a and BELOW[t - run_len, j]:
            run_len += 1
        rows.append(dict(hid=hid, j=j, t=t, k=k, depth=float(DEPTH[t, j]), since=int(t - t1), run=run_len, D=int(e - t),
                         ex=float(GO[e, j] / G[t, j] - 1 - (E50O[e] / E50[t] - 1)), ret_stock=float(GO[e, j] / G[t, j] - 1),
                         above_at_k=bool(not BELOW[k, j]), sold_at_k=bool(R.X[k, j])))
Dd = pd.DataFrame(rows)


def st(d):
    if not len(d):
        return dict(n=0)
    s = R.stats(d)
    s['n_holdings'] = int(d['hid'].nunique()); s['unreliable'] = bool(s['n_holdings'] < 20)
    s['above_at_k'] = float(d['above_at_k'].mean()); s['sold_at_k'] = float(d['sold_at_k'].mean()); s['stock_mean'] = float(d['ret_stock'].mean())
    return s


dp, si, D, rn = Dd['depth'].to_numpy(), Dd['since'].to_numpy(), Dd['D'].to_numpy(), Dd['run'].to_numpy()
GR = {
 '當日深度': {'0～−2%': dp > -0.02, '−2～−5%': (dp <= -0.02) & (dp > -0.05), '−5～−10%': (dp <= -0.05) & (dp > -0.10), '< −10%': dp <= -0.10},
 '首次跌破後交易日數': {'0（當天）': si == 0, '1～2': (si >= 1) & (si <= 2), '3～5': (si >= 3) & (si <= 5), '>5': si > 5},
 '連續在季線下日數': {'1': rn == 1, '2': rn == 2, '3～5': (rn >= 3) & (rn <= 5), '>5': rn > 5},
 '距換股執行日': {'≤5 日': D <= 5, '6～10 日': (D >= 6) & (D <= 10), '>10 日': D > 10},
}
out = dict(note='事後、未登記；持股日層級（同檔同月重複，n 不是獨立樣本），群聚 t 以換股月份分群', n_days=int(len(Dd)), n_holdings=int(Dd['hid'].nunique()),
           all=st(Dd), groups={g: {k: dict(st(Dd[m]), vs_rest=R.diff_t(Dd, m)) for k, m in ms.items()} for g, ms in GR.items()})
# 每檔每月只取最深的一天（去重複）
idx = Dd.groupby(['hid', 'k'])['depth'].idxmin(); Dm = Dd.loc[idx]; dm = Dm['depth'].to_numpy()
out['deepest_day_per_holding_month'] = {k: st(Dm[m]) for k, m in {'0～−2%': dm > -0.02, '−2～−5%': (dm <= -0.02) & (dm > -0.05), '−5～−10%': (dm <= -0.05) & (dm > -0.10), '< −10%': dm <= -0.10}.items()}
out['run_sha256'] = R.sha(Path(__file__))
(R.HERE / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
f = lambda s: 'n=0' if not s.get('n') else f"日 {s['n']} 檔次 {s['n_holdings']} 月 {s['n_months']} 平均 {s['mean']*100:+.2f}% 中位 {s['median']*100:+.2f}% 勝率 {s['win']*100:.0f}% P10 {s['p10']*100:+.1f}% P90 {s['p90']*100:+.1f}% t {s['t_cluster'] and round(s['t_cluster'],2)} 站回 {s['above_at_k']:.2f} 賣 {s['sold_at_k']:.2f}"
print('== 事後 持股日'); print('全部', f(out['all']))
for g, o in out['groups'].items():
    for k, v in o.items():
        print(g, k, f(v), v['vs_rest'] and {a: round(b, 4) for a, b in v['vs_rest'].items() if b is not None})
for k, v in out['deepest_day_per_holding_month'].items():
    print('每檔每月最深日', k, f(v))
