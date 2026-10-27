#!/usr/bin/env python3
"""事後診斷（不改判定）：訊號同日群聚（疑似 ETF 定期調整）vs 單獨事件；抽樣核對；開盤漲停次數。"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R  # noqa: E402

out = {}
for key in ['FB_MCAP', 'FB_PROXY', 'FB_HOLD60']:
    df = pd.read_csv(HERE / f'events_{key}.csv.gz', parse_dates=['sig', 'entry', 'exit'])
    cnt = df.groupby('sig')['code'].transform('size')
    df['cluster'] = cnt >= 5
    o = {'share_clustered': round(float(df['cluster'].mean()), 3),
         'top_signal_dates': {str(k.date()): int(v) for k, v in df.groupby('sig').size().sort_values(ascending=False).head(12).items()}}
    for w, (a, b) in R.WIN.items():
        sub = df[(df['sig'] >= a) & (df['sig'] <= b)]
        o[f'{w}_clustered'] = R.ev_stats(sub[sub['cluster']]); o[f'{w}_isolated'] = R.ev_stats(sub[~sub['cluster']])
    # 依出場原因
    o['by_exit_reason'] = {k: {'n': int(len(g)), 'ab_net_pct': round(g['ab'].mean() * 100, 2), 'raw_pct': round(g['raw'].mean() * 100, 2)} for k, g in df.groupby('why')}
    # 依進場後前 5 日（看是否一開始就被追高）
    out[key] = o
# 開盤漲停在股池中的總次數（確認 GAP_UP 有作用）
m = R.SMALL_MCAP.loc['2022':].to_numpy()
gp = R.GAP_UP[R.PIDX >= pd.Timestamp('2022-01-01')]
out['gap_up_open_count_in_small_pool_2022+'] = int(np.nansum(gp[1:] & m[:-1]))
# 抽樣：第一筆事件的投信序列
df = pd.read_csv(HERE / 'events_FB_MCAP.csv.gz', parse_dates=['sig'], dtype={'code': str})
samp = {}
for _, r in df.sample(3, random_state=1).iterrows():
    s = R.TR[r['code']]; i = s.index.get_loc(r['sig'])
    samp[f"{r['code']}@{r['sig'].date()}"] = {'buy_days_prev60': int((s.iloc[i - 62:i - 2] > 0).sum()), 'last5': s.iloc[i - 4:i + 1].round(1).tolist(),
                                            'shares_k': float(R.SH[r['code']] / 1000), 'mcap_e8': round(float(R.MCAP.loc[r['sig'], r['code']] / 1e8), 1)}
out['sample_check'] = samp
f = HERE / 'results.json'
d = json.loads(f.read_text(encoding='utf-8')); d['diag_posthoc'] = out
f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
for k in ['FB_MCAP']:
    o = out[k]
    print(k, o['share_clustered'], o['top_signal_dates'])
    for w in R.WIN:
        for c in ['clustered', 'isolated']:
            e = o[f'{w}_{c}']; print(w, c, e.get('n'), e.get('mean_raw_pct'), e.get('mean_ab_net_pct'), e.get('t_month_ab_net'), e.get('hit_ab_net'))
    print(o['by_exit_reason'])
print(out['gap_up_open_count_in_small_pool_2022+'], out['sample_check'])
for k in ['FB_PROXY']:
    for w in R.WIN:
        for c in ['clustered', 'isolated']:
            e = out[k][f'{w}_{c}']; print(k, w, c, e.get('n'), e.get('mean_ab_net_pct'), e.get('t_month_ab_net'))
