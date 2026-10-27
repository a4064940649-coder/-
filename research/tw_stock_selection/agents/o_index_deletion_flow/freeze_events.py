"""資料階段關卡：在算任何報酬之前，加上成交值查證欄、凍結 events_v2.csv（sha256）並判定關卡。只讀 value（成交值），不讀價格與報酬。"""
import os, json, hashlib, datetime as dt
import pandas as pd, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
P = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/'
V = pd.read_parquet(P + 'value.parquet'); S = pd.read_parquet(P + 'src.parquet')
ev = pd.read_csv(os.path.join(HERE, 'events_v2.csv'), dtype=str)
idx = V.index
ratios, srcs = [], []
for r in ev.itertuples():
    if r.code not in V.columns: ratios.append(np.nan); srcs.append(np.nan); continue
    E = pd.Timestamp(r.eff); i = idx.get_loc(E)
    base = V[r.code].iloc[max(0, i - 25):i - 5].median()
    ratios.append(float(V[r.code].iloc[i] / base) if base and base > 0 else np.nan)
    srcs.append(S[r.code].iloc[i])
ev['vol_ratio_E'] = np.round(ratios, 2); ev['src_E'] = srcs
ev['period'] = np.where(pd.to_datetime(ev.eff) <= '2018-12-31', 'select', 'validate')
ev.to_csv(os.path.join(HERE, 'events_v2.csv'), index=False)
sha = hashlib.sha256(open(os.path.join(HERE, 'events_v2.csv'), 'rb').read()).hexdigest()
ev['in_panel'] = ev.in_panel == 'True'; ev['src3_period'] = ev.src3_period == 'True'
dele = ev[ev.type.isin(['MSCI_STD_DEL', 'MSCI_SC_DEL']) & ev.in_panel & ~ev.src3_period & (pd.to_datetime(ev.eff) >= '2008-01-01')]
n = len(dele); n_ver = int((dele.verify == 'eff_stated_in_source').sum())
# 成交值查證：同一審核日的事件群組中位數倍數（個股可能因停牌等缺值）
grp = dele.groupby('eff').vol_ratio_E.median()
gate = {'n_msci_ftse_deletions_valid': n, 'n_eff_verified_in_source': n_ver, 'share_verified': round(n_ver / n, 3) if n else 0,
        'review_group_median_volume_ratio': {k: (None if pd.isna(v) else round(float(v), 2)) for k, v in grp.items()},
        'n_reviews': int(grp.size), 'n_reviews_spike_ge_1.5': int((grp >= 1.5).sum()),
        'ftse_all_world_events': 0, 'ftse_note': 'FTSE 全球指數台股增刪在台灣媒體與 GitHub 鏡像中幾乎沒有報導，未能蒐集',
        'pass': bool(n >= 60 and n_ver >= n / 2)}
res_path = os.path.join(HERE, 'results.json')
res = json.load(open(res_path)) if os.path.exists(res_path) else {}
res['DATA_FREEZE'] = {'file': 'events_v2.csv', 'sha256': sha, 'n_rows': int(len(ev)),
                      'timestamp_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
                      'note': '凍結時尚未計算任何報酬；只用名單、交易日曆與成交值', 'gate': gate}
json.dump(res, open(res_path, 'w'), ensure_ascii=False, indent=1)
print(json.dumps(res['DATA_FREEZE'], ensure_ascii=False, indent=1)[:3000])
