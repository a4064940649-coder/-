#!/usr/bin/env python3
"""容器被重置後，重建 run_v3.py（與 run_v2.py）需要的連續面板快取 scratchpad/s75/cache_cont。不改任何既有檔、不上網、不呼叫 git。
用法：python3 -I -B rebuild_cache.py      （約 1 分鐘、峰值約 0.9GB）
前提：正式面板快取 scratchpad/offfix/cache/{panel_cs.npz,e50_ohlc.csv,innov.npz} 已存在（restore_env.sh 之後跑過每日例行的 daily_check／run_official 就會有）。
步驟：
 1. offfix/cache 是照「今天的」官方日 K 重建的（列到最新交易日、欄含之後才出現的代號）；../build_panel.py 當時用的是到 2026-10-30 的那一份（eng/cache，2,880 列 × 2,142 欄）。
    這裡把 offfix/cache 截回同樣範圍（列 ≤ 2026-10-30；丟掉截止日前完全沒有資料的欄，丟之前先確認真的全空）存到 scratchpad/eng/cache。
 2. 原樣執行 ../build_panel.py（只把它寫 panel_check.json 的位置改到暫存，不覆蓋倉庫內的檔）。
 3. 新的 panel_check.json 與倉庫內 ../panel_check.json 逐項比對（執行時間、記憶體除外）；不同就以非 0 結束。
判準仍然是 run_v3.py 開頭的 v2 基準重現（逐日、逐筆、逐項）；這支只是把快取放回去。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
S75 = HERE.parent
OLD = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad')
SRC = OLD / 'offfix' / 'cache'; DST = OLD / 'eng' / 'cache'
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s75')
CUT = pd.Timestamp('2026-10-30')
for f in ('panel_cs.npz', 'e50_ohlc.csv', 'innov.npz'):
    assert (SRC / f).exists(), f'缺 {SRC / f}：先跑每日例行的 daily_check（或 official_nav/run_official.py）讓正式面板快取重建'
DST.mkdir(parents=True, exist_ok=True); TMP.mkdir(parents=True, exist_ok=True)
z = np.load(SRC / 'panel_cs.npz', allow_pickle=False); idx = pd.DatetimeIndex(z['IDX']); k = int((idx <= CUT).sum())
assert str(idx[k - 1].date()) == '2026-10-30'
keep = (z['MK'][:k] > 0).any(0)
for a in ('C', 'O', 'H', 'L', 'VOL', 'AMT', 'CHG'):
    assert not np.isfinite(z[a][:k][:, ~keep]).any(), f'要丟的欄在 {CUT.date()} 以前有 {a} 資料'
np.savez_compressed(DST / 'panel_cs.npz', IDX=z['IDX'][:k], COLS=z['COLS'][keep], **{a: z[a][:k][:, keep] for a in ('C', 'MK', 'O', 'H', 'L', 'VOL', 'AMT', 'CHG')})
zi = np.load(SRC / 'innov.npz', allow_pickle=False)
assert np.array_equal(zi['COLS'], z['COLS']) and not zi['A'][:, ~keep].any()
np.savez_compressed(DST / 'innov.npz', A=zi['A'][:, keep], IDX=zi['IDX'], COLS=zi['COLS'][keep], codes=zi['codes'], names=zi['names'])
e = pd.read_csv(SRC / 'e50_ohlc.csv', dtype=str); e[pd.to_datetime(e['date']) <= CUT].to_csv(DST / 'e50_ohlc.csv', index=False)
print('正式面板（今天重建）', z['C'].shape, '最後一天', idx[-1].date(), '→ 截回', (k, int(keep.sum())), '丟掉的欄', [str(c) for c in z['COLS'][~keep]], flush=True)
del z, zi
for f in ('panel_cs.npz', 'e50_ohlc.csv', 'innov.npz'):          # build_panel.py 會把 eng/cache 複製到 s75/cache_is；舊的先清掉以免沿用
    (TMP / 'cache_is' / f).unlink(missing_ok=True)
src = (S75 / 'build_panel.py').read_text()
a = "HERE = Path(__file__).resolve().parent"; assert src.count(a) == 1
exec(compile(src.replace(a, f"HERE = Path({str(TMP)!r})"), 'build_panel.py', 'exec'), {'__name__': '__main__', '__file__': str(S75 / 'build_panel.py')})
x = json.loads((S75 / 'panel_check.json').read_text()); y = json.loads((TMP / 'panel_check.json').read_text())
for d in (x, y):
    d.pop('runtime_s'); d.pop('maxrss_mb')
dif = [key for key in x if x[key] != y.get(key)]
print('panel_check 與倉庫內相同：', not dif, dif)
sys.exit(0 if not dif else 2)
