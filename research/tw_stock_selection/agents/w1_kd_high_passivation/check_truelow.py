#!/usr/bin/env python3
"""補充（非判定用）：2025-09-24 起以 history.csv.gz 真實 low 取代 L9 近似，重算 KD 與事件，與近似版同期比較。"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as Rn  # noqa: E402

h = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/history.csv.gz', dtype={'code': str}, parse_dates=['date'])
h = h[h['code'].isin(Rn.cols)]
lo = h.pivot_table(index='date', columns='code', values='low', aggfunc='last').reindex(index=Rn.IDX, columns=Rn.cols)
lo = lo.where(lo > 0) * Rn.fac
LOW2 = Rn.LOWP.copy()
m = (lo.index >= pd.Timestamp('2025-09-24'))[:, None] & lo.notna().to_numpy()
LOW2 = LOW2.where(~m, np.fmin(lo, LOW2))           # 真實 low（且不高於 min(open,close)）
K2, D2 = Rn.kd(Rn.AH.to_numpy(), LOW2.to_numpy(), Rn.ADJ.to_numpy())
STK2 = Rn.streak(K2)
a, b = pd.Timestamp('2025-11-01'), pd.Timestamp('2026-10-31')
out = {'window_signal': '2025-11-01~2026-10-22（真 low 版 KD 自 2025-09-24 起，留 5 週收斂）'}
for nm, S in (('approx', Rn.sig_v1(3)), ('true_low', (STK2 == 3) & Rn.ELIG)):
    for ex in ('A', 'C20'):
        if nm == 'true_low' and ex == 'A':
            pass
        ev, nb, nno = Rn.build_events(S, ex)
        df = Rn.event_table(ev); df = df[(df['sig'] >= a) & (df['sig'] <= b)]
        out[f'{nm}_N3_{ex}'] = Rn.ev_stats(df)
# 出場 A 用 MA5，與 K 無關；出場 B 需要 K → 真 low 版改用 K2
Kbak = Rn.Kn; Rn.Kn = K2
ev, _, _ = Rn.build_events((STK2 == 3) & Rn.ELIG, 'B'); df = Rn.event_table(ev); out['true_low_N3_B'] = Rn.ev_stats(df[(df['sig'] >= a) & (df['sig'] <= b)])
Rn.Kn = Kbak
ev, _, _ = Rn.build_events(Rn.sig_v1(3), 'B'); df = Rn.event_table(ev); out['approx_N3_B'] = Rn.ev_stats(df[(df['sig'] >= a) & (df['sig'] <= b)])
for k, v in out.items(): print(k, v)
f = HERE / 'results.json'; d = json.loads(f.read_text(encoding='utf-8')); d['truelow_check'] = out
f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
