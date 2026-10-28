#!/usr/bin/env python3
"""步驟 3：閘門（只看資料覆蓋與一致率）→ results.json 的 GATE；通過才寫 data_freeze_v6.json。
母體＝V6 每個換股訊號日的前 300 股池 ∪ 當期持股（V6 名單），乘上持有窗內面板交易日（只取替換窗 2024-01-02～2025-09-23），
只算當天在任一來源（官方或 src 3）有成交的股日。官方 OHLC＝src ∈ {5,6,7} 且開高低收齊全。"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import v6common as C

P = {k: pd.read_parquet(C.DATA / f'{k}.parquet') for k in ('close_raw', 'open', 'high', 'low', 'src', 'value')}
IDX = P['close_raw'].index
pv6 = json.loads((C.DATA / 'pools_v6.json').read_text()); pv4 = json.loads((C.DATA / 'pools_v4.json').read_text())
sig = sorted(k for k in pv6 if not k.startswith('_'))
rows = []
for i, d in enumerate(sig):
    a = pd.Timestamp(d); b = pd.Timestamp(sig[i + 1]) if i + 1 < len(sig) else C.W_B + pd.Timedelta(days=1)
    days = IDX[(IDX >= max(a, C.W_A)) & (IDX < b) & (IDX <= C.W_B)]
    names = set(pv6[d]['top300']) | set(pv6['_picks'].get(d, []))
    if i > 0: names |= set(pv6['_picks'].get(sig[i - 1], []))
    for t in days:
        for c in names: rows.append((t, c))
POP = pd.DataFrame(rows, columns=['date', 'code']).drop_duplicates()
cl = P['close_raw']; ii = IDX.get_indexer(POP.date); jj = cl.columns.get_indexer(POP.code)
cv = cl.to_numpy()[ii, jj]; sv = P['src'].to_numpy()[ii, jj]
ohlc_ok = np.isfinite(P['open'].to_numpy()[ii, jj]) & np.isfinite(P['high'].to_numpy()[ii, jj]) & np.isfinite(P['low'].to_numpy()[ii, jj]) & np.isfinite(cv)
traded = np.isfinite(cv)
off = traded & np.isin(sv, [5, 6, 7]) & ohlc_ok
POP['traded'] = traded; POP['official'] = off; POP['src'] = sv
def cov(m):
    x = POP[m & POP.traded]
    return {'n_stock_days_traded': int(len(x)), 'official_ohlc_share': float(x.official.mean()) if len(x) else None,
            'n_not_official': int((~x.official).sum()), 'by_src': {str(int(k)) if np.isfinite(k) else 'nan': int(v) for k, v in x.src.value_counts().items()}}
gate = {'population': '見檔頭說明'}
gate['coverage_ALL'] = cov(np.ones(len(POP), bool))
gate['coverage_2024_01_02_to_02_14'] = cov((POP.date >= C.GAP_A) & (POP.date <= C.GAP_B))
gate['coverage_2024_02_15_to_2025_05_28'] = cov((POP.date >= '2024-02-15') & (POP.date <= '2025-05-28'))
gate['coverage_2025_05_29_to_09_23'] = cov((POP.date >= '2025-05-29') & (POP.date <= C.W_B))
gate['n_untraded_population_stock_days'] = int((~POP.traded).sum())
gate['not_official_list'] = POP[POP.traded & ~POP.official][['date', 'code', 'src']].astype(str).values.tolist()[:60]
it = json.loads((C.HERE / 'iterations.json').read_text(encoding='utf-8'))
meta = json.loads((C.DATA / 'panel_v6_meta.json').read_text(encoding='utf-8'))
mt = it['data_versions'][meta['version']]['metrics']
ex = json.loads((C.HERE / 'explore.json').read_text(encoding='utf-8'))
o2o = {o['pair']: o['close']['exact'] for o in ex['official_vs_official'] if 'close' in o}
gate['official_vs_official_close_agree'] = {'chosen_cells_vs_other_official': mt['chosen_close_agree_all'], 'raw_pairs': o2o,
                                            'hschi_vs_tasi_overlap(預測 2 的對照)': o2o.get('hschi vs tasi')}
gate['exdiv_prevclose_match'] = mt['exdiv_prevclose_match_rate']
smoke = json.loads((C.HERE / 'smoke.json').read_text())
gate['smoke_2006_2023_identical'] = smoke['all_identical']
g1 = gate['coverage_ALL']['official_ohlc_share'] >= 0.99
g2 = gate['coverage_2024_01_02_to_02_14']['official_ohlc_share'] >= 1.0 - 1e-12
g3 = mt['chosen_close_agree_all'] >= 0.995
g4 = gate['exdiv_prevclose_match'] >= 0.99
gate['checks'] = {'pool_official_ohlc>=99%': g1, '2024-01 gap 100%': g2, 'official_vs_official_close>=99.5%': g3, 'exdiv_ref_match>=99%': g4,
                  'smoke_identical': gate['smoke_2006_2023_identical']}
gate['PASS'] = bool(g1 and g2 and g3 and g4 and gate['smoke_2006_2023_identical'])
# 93 檔缺值與 2024-04～06 股池漏檔（V6 對 V4 的前 300 股池）
pl = {}
for d in sig:
    if d in pv4 and d >= '2024-02-01' and d <= '2024-09-30':
        a, b = set(pv6[d]['top300']), set(pv4[d]['top300'])
        pl[d] = {'n_v6_only': len(a - b), 'n_v4_only': len(b - a), 'v6_only_sample': sorted(a - b)[:15]}
gate['pool_diff_v6_vs_v4_2024'] = pl
gate['v6_signal_days_not_in_v4'] = [d for d in sig if d not in pv4]
res_f = C.HERE / 'results.json'
res = json.loads(res_f.read_text(encoding='utf-8')) if res_f.exists() else {}
res['GATE'] = gate
C.jdump(res, res_f)
print(json.dumps({k: gate[k] for k in ('coverage_ALL', 'coverage_2024_01_02_to_02_14', 'coverage_2025_05_29_to_09_23', 'checks', 'PASS', 'v6_signal_days_not_in_v4')}, ensure_ascii=False, indent=0))
print({d: (v['n_v6_only'], v['n_v4_only']) for d, v in pl.items()})
