#!/usr/bin/env python3
"""S13 事後描述（驗證期判定之後才寫；只描述，不影響判定、不換特徵、不另計 N）。
讀 cand_train.csv.gz、cand_valid.csv.gz：(1) 六個特徵五組的大輸家比例（超額 ≤ 訓練期第 10 百分位）與超額標準差——檢查「AUC 高但平均沒變好」是不是因為同時也多了大輸家；
(2) 現行 F1 名單 vs 沒買的候選；(3) 驗證期大贏家（前 20）在凍結規則下被剔除的情形。python3 -I -B posthoc.py → results_posthoc.json"""
import json
from pathlib import Path
import numpy as np, pandas as pd

H = Path(__file__).resolve().parent
FEATURES = ['ACC', 'REVHI12', 'GMYOY', 'NEAR52', 'RET60', 'ADVCHG']
tr = pd.read_csv(H / 'cand_train.csv.gz', parse_dates=['sig_date'], dtype={'code': str})
va = pd.read_csv(H / 'cand_valid.csv.gz', parse_dates=['sig_date'], dtype={'code': str})
FZ = json.loads((H / 'FROZEN.json').read_text())
lo = float(np.quantile(tr['excess'], 0.1)); hi = FZ['big_winner_threshold_excess']
out = {'note': '事後描述，未登記；不影響判定。', 'big_loser_threshold_excess_train_p10': lo, 'big_winner_threshold_excess': hi, 'by_feature': {}}


def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, bool); ok = np.isfinite(s); s, y = s[ok], y[ok]
    n1, n0 = y.sum(), (~y).sum(); r = pd.Series(s).rank().to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


for f in FEATURES:
    d = {}
    for name, df in (('train', tr), ('valid', va)):
        g = df[np.isfinite(df[f])].copy()
        g['q'] = pd.qcut(g[f], 5, labels=False) + 1
        d[name] = dict(auc_big_winner=auc(g[f], g['excess'] >= hi), auc_big_loser=auc(g[f], g['excess'] <= lo),
                       quintiles=[dict(q=int(q), n=int(len(x)), bw=float((x['excess'] >= hi).mean()), bl=float((x['excess'] <= lo).mean()),
                                       mean=float(x['excess'].mean()), median=float(x['excess'].median()), sd=float(x['excess'].std()))
                                  for q, x in g.groupby('q')])
    out['by_feature'][f] = d
for name, df in (('train', tr), ('valid', va)):
    out[f'f1_vs_rest_{name}'] = {str(k): dict(n=int(len(x)), mean=float(x['excess'].mean()), median=float(x['excess'].median()),
                                              bw=float((x['excess'] >= hi).mean()), bl=float((x['excess'] <= lo).mean()))
                                 for k, x in df.groupby('f1')}
    x = df[df['f1']]
    out[f'f1_top5_share_{name}'] = float(x['excess'].nlargest(5).sum() / x['excess'].sum())
    out[f'f1_mean_ex_top5_{name}'] = float(x.drop(x['excess'].nlargest(5).index)['excess'].mean())
# 驗證期 F1 名單中被凍結規則剔除的大贏家
x = va[va['f1'] & (va['excess'] >= hi)]
out['valid_f1_big_winners'] = x.sort_values('excess', ascending=False)[['sig_date', 'code', 'rank', 'ADVCHG', 'keep', 'excess']].astype(str).values.tolist()
# RET60 前 1/5 的大贏家／大輸家，兩期（事後觀察）
(H / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
for f in FEATURES:
    for name in ('train', 'valid'):
        d = out['by_feature'][f][name]
        print(f, name, 'AUCbw', round(d['auc_big_winner'], 3), 'AUCbl', round(d['auc_big_loser'], 3),
              [(q['q'], round(q['bw'] * 100, 1), round(q['bl'] * 100, 1), round(q['mean'] * 100, 2), round(q['sd'] * 100, 1)) for q in d['quintiles']])
for k in out:
    if k.startswith('f1'):
        print(k, out[k])
print(len(out['valid_f1_big_winners']), out['valid_f1_big_winners'][:12])
