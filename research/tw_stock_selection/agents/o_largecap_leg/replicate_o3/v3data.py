"""V3 資料（P＋X＋D_hi＋R_DEL）：依 o_infra_v3_rebuild PREREG 文字自行實作，不 import 其程式。"""
import sys, pickle, copy, json
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_largecap_leg/replicate_o3')
from rlib import *
SC = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/'
V3 = AG + 'o_infra_v3_rebuild/'

D = pickle.load(open(SC + 'rep_o3_data.pkl', 'rb'))
D3 = copy.copy(D)
dp = pd.read_parquet(V3 + 'delist_prices.parquet')
dp['date'] = pd.to_datetime(dp['date']); dp['code'] = dp['code'].astype(str)
close, opn, val, ret, adj = D.close.copy(), D.open.copy(), D.value.copy(), D.ret.copy(), D.adj.copy()
for c, g in dp.groupby('code'):
    g = g.set_index('date').sort_index()
    g = g[g.index.isin(close.index)]
    close.loc[g.index, c] = g.close.values; opn.loc[g.index, c] = g.open.values
    val.loc[g.index, c] = g.value.values; ret.loc[g.index, c] = g.ret.values
    # 還原指數：補的日子用 ret（缺時用收盤比）接續
    a = adj[c].copy(); cl = close[c]
    first = g.index[0]
    prev_i = a.loc[:first - pd.Timedelta(days=1)].last_valid_index()
    pv, pc = a[prev_i], close.at[prev_i, c]
    for d in g.index:
        r = ret.at[d, c]
        if pd.isna(r):
            r = cl[d] / pc - 1
        pv = pv * (1 + r); pc = cl[d]
        a[d] = pv
    adj[c] = a
D3.close, D3.open, D3.value, D3.ret, D3.adj = close, opn, val, ret, adj
cnt = adj.notna().sum(1); vidx = adj.index[cnt >= 300]
D3.r52 = adj.loc[vidx] / gap_roll(adj.loc[vidx], 252, 'max')
D3.v60 = gap_roll(val.loc[vidx], 60, 'mean')
D3.adjopen = opn * (adj / close); D3.adjf = adj.ffill()
# 處置：2021-09-24 以前用近似名單
dh = pd.read_csv(V3 + 'disposal_hist.csv', dtype={'code': str})
dh['start'] = pd.to_datetime(dh.start); dh['end'] = pd.to_datetime(dh.end)
D3.dsp_hist = dh
prm = json.load(open(V3 + 'results.json'))['PREREG_params']
D3.dhi_extra = {pd.Timestamp(k): v for k, v in prm['D_hi_extra'].items()}
# X 規則
mc = pd.read_csv(V3 + 'merger_consideration.csv', dtype={'code': str, 'successor': str})
D3.mc = mc
pickle.dump(D3, open(SC + 'rep_o3_data_v3.pkl', 'wb'))
print('ok', len(D3.dhi_extra), sum(D3.dhi_extra.values()), [k for k in prm.keys()])
