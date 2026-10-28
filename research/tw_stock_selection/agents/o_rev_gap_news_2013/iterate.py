#!/usr/bin/env python3
"""步驟 2 迭代（不看任何策略報酬）：解析器版本 P1／P2／P3 × 代理版本 N3／N2／N1／Q／NQ。
判準只有：(1) 非缺口格（有真值）上，代理（只用新聞／季報重建）對真值的百分位 Spearman 與誤差分布；(2) 前 10 名入選一致率。
另報：缺口格（2013-01～2021-12，732 格，含非下市格）PIT 覆蓋率；B10 實際入選 133 股月的覆蓋率（只供描述）。
用法：python3 -I -B iterate.py P1   → 結果併入 iterations.json
"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import rglib as G
import proxy as PX
K, V, L = G.K, G.V, G.L
IT = sys.argv[1]
b = G.base()
S = G.mk()
M, LV, L5, Q = PX.load_inputs(K, HERE / 'data' / f'mentions_{IT}.parquet')
P = PX.Proxy(M, LV, L5, Q, S.YL, S.Y5, V.SWITCH)
X = pd.read_csv(HERE / 'gap_cells_2013_2021.csv', dtype={'code': str})

out = {'iter': IT, 'time_utc': pd.Timestamp.utcnow().isoformat(), 'n_mentions': int(len(M)), 'versions': {}}
# ---- 缺口格覆蓋 ----
cov = {}
for ver in PX.VERSIONS:
    v = np.array([np.isfinite(P.y3(c, pd.Timestamp(s), ver)[0]) for s, c in zip(X.sig, X.code)])
    cov[ver] = v
    out['versions'][ver] = {'gap_cov_all': float(v.mean()), 'gap_cov_B10picks': float(v[X.in_B10.to_numpy()].mean()),
                            'gap_cov_by_type': {t: float(v[(X.type == t).to_numpy()].mean()) for t in ('merger', 'deterior', 'nondelist')},
                            'gap_cov_by_year': {y: float(v[(X.sig.str[:4] == y).to_numpy()].mean()) for y in sorted(X.sig.str[:4].unique())},
                            'n_gap': int(len(X)), 'n_cov': int(v.sum())}
# ---- 非缺口格驗證 ----
rng = np.random.default_rng(0)
val = {ver: {'tp': [], 'pp': [], 'agree': [], 'n_cov': 0, 'n': 0, 'agree_cov': []} for ver in PX.VERSIONS}
for d in S.sig_days:
    if not (G.W0 <= d <= G.W1): continue
    cs = S.cross_section(d)
    ok = cs['ok_other']; y3 = S.y3(d)
    el = ok & np.isfinite(y3)
    J = np.where(el)[0]
    if len(J) < 50: continue
    tv = y3[J]
    s52 = cs['s52'][J]
    pr52 = pd.Series(s52).rank(pct=True).to_numpy()
    def top10(yv):
        sc = 0.5 * pr52 + 0.5 * pd.Series(yv).rank(pct=True).to_numpy()
        return set(np.argsort(-sc, kind='stable')[:10])
    T = top10(tv)
    srt = np.sort(tv)
    def pct(x, self_v):
        lo = np.searchsorted(srt, x, 'left'); hi = np.searchsorted(srt, x, 'right')
        # 扣掉自己
        n = len(srt) - 1
        r = lo + 0.5 * (hi - lo)
        if self_v is not None:
            if self_v < x: r -= 1
            elif self_v == x: r -= 0.5
        return r / n
    for ver in PX.VERSIONS:
        pv = np.array([P.y3(K.COLS[j], d, ver, only_news=True)[0] for j in J])
        f = np.isfinite(pv)
        V_ = val[ver]; V_['n'] += len(J); V_['n_cov'] += int(f.sum())
        for k in np.where(f)[0]:
            V_['tp'].append(pct(tv[k], tv[k])); V_['pp'].append(pct(pv[k], tv[k]))
        yv = np.where(f, pv, tv)
        Pp = top10(yv)
        V_['agree'].append(len(T & Pp) / 10)
        cov_top = {k for k in T | Pp if f[k]}
        if cov_top: V_['agree_cov'].append(len({k for k in cov_top if k in T and k in Pp}) / len(cov_top))
for ver in PX.VERSIONS:
    V_ = val[ver]
    tp, pp = np.array(V_['tp']), np.array(V_['pp'])
    e = np.abs(pp - tp)
    o = out['versions'][ver]
    o['val_n_cells'] = V_['n']; o['val_cov'] = V_['n_cov'] / max(V_['n'], 1)
    o['val_spearman_pct'] = float(spearmanr(tp, pp).correlation) if len(tp) > 10 else None
    o['val_abs_pct_err'] = {'median': float(np.median(e)) if len(e) else None, 'p75': float(np.percentile(e, 75)) if len(e) else None,
                            'p90': float(np.percentile(e, 90)) if len(e) else None}
    o['val_top10_agree_all'] = float(np.mean(V_['agree'])) if V_['agree'] else None
    o['val_top10_agree_covered'] = float(np.mean(V_['agree_cov'])) if V_['agree_cov'] else None
# 選定規則：Spearman ≥ 0.80 中，缺口覆蓋最高；同分取較嚴格（順序 N3 > N2 > Q > NQ > N1 的嚴格度：N3 最嚴）
STRICT = ['N3', 'N2', 'Q', 'NQ', 'N1']
ok = [v for v in STRICT if (out['versions'][v]['val_spearman_pct'] or 0) >= 0.80]
sel = max(ok, key=lambda v: (round(out['versions'][v]['gap_cov_all'], 6), -STRICT.index(v))) if ok else None
out['selected'] = sel
out['G1_cov_selected'] = out['versions'][sel]['gap_cov_all'] if sel else 0.0
F = HERE / 'iterations.json'
J_ = json.loads(F.read_text(encoding='utf-8')) if F.exists() else {'note': '步驟 2 迭代紀錄（解析器 P1～P3 × 代理 N3／N2／N1／Q／NQ）；判準只看資料品質，不看任何策略報酬。', 'rounds': []}
J_['rounds'] = [r for r in J_['rounds'] if r['iter'] != IT] + [out]
F.write_text(json.dumps(J_, ensure_ascii=False, indent=1), encoding='utf-8')
pd.DataFrame({'sig': X.sig, 'code': X.code, **{f'cov_{v}': cov[v] for v in PX.VERSIONS}}).to_csv(HERE / 'data' / f'gap_cov_{IT}.csv', index=False)
print(json.dumps({v: {k: out['versions'][v][k] for k in ('gap_cov_all', 'gap_cov_B10picks', 'gap_cov_by_type', 'val_cov', 'val_spearman_pct', 'val_abs_pct_err', 'val_top10_agree_all', 'val_top10_agree_covered')} for v in PX.VERSIONS}, ensure_ascii=False, indent=1))
print('selected', sel, out['G1_cov_selected'])
