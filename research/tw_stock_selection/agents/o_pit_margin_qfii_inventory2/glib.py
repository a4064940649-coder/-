"""共用：分母、流動性池、來源載入（只讀）。所有日期為交易日。"""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent
RAW = HERE / 'raw'
PANEL = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
SPEC = json.load(open(HERE / 'GATE_SPEC.json'))
SEG = {k: (pd.Timestamp(a), pd.Timestamp(b)) for k, (a, b) in SPEC['segments'].items()}
def common(cols): return [c for c in cols if len(c) == 4 and c.isdigit() and not c.startswith('0')]

def denominator():
    """回傳 (listed bool 寬表 2006-01-02～2026-10-31, value 寬表, first_day Series, last_day Series)。"""
    lt = pd.read_parquet(AG / 'o_v7_listing_exec_engine/data/listed_twse.parquet')
    lt = lt[common(lt.columns)]
    lt.index = pd.to_datetime(lt.index).astype('datetime64[ns]')
    lt = lt.loc['2006-01-02':'2013-12-31']
    mi = pd.read_parquet(RAW / 'joeyc_mi_index_4digit_2014_2026.parquet')
    mi = mi[~mi.code.str.startswith('0') & (mi.date <= '2026-10-31')]
    li = mi.assign(v=True).pivot_table(index='date', columns='code', values='v', aggfunc='first').notna()
    listed = pd.concat([lt, li], axis=0).fillna(False).astype(bool).sort_index()
    pv = pd.read_parquet(PANEL / 'value.parquet'); pv.index = pd.to_datetime(pv.index).astype('datetime64[ns]')
    pv = pv.loc['2005-06-01':'2013-12-31']
    vi = mi.pivot_table(index='date', columns='code', values='value', aggfunc='first')
    val = pd.concat([pv, vi], axis=0).sort_index()
    val = val[~val.index.duplicated()]
    # 上市首日：listing_master listed_date；沒有的用分母首次出現日（2006-01-02/2014-01-02 首日者視為舊股）
    lm = pd.read_csv(AG / 'o_v7_listing_exec_engine/listing_master.csv', dtype=str).set_index('code')
    first_seen = listed.idxmax()
    fd = pd.to_datetime(lm['listed_date'], errors='coerce').reindex(listed.columns)
    fd = fd.fillna(first_seen.where(first_seen > pd.Timestamp('2014-01-02'), pd.Timestamp('1990-01-01')))
    last_day = listed[::-1].idxmax()
    return listed, val, fd, last_day

def new_listing_mask(listed, fd, n=130):
    """上市首日起 n 個交易日內＝True（MARGN 分母扣除）。"""
    days = listed.index
    pos = pd.Series(np.arange(len(days)), index=days)
    out = pd.DataFrame(False, index=days, columns=listed.columns)
    for c in listed.columns:
        f = fd.get(c)
        if pd.isna(f) or f < days[0] - pd.Timedelta(days=400): continue
        i = days.searchsorted(f)
        out.iloc[i:i + n, out.columns.get_loc(c)] = True
    return out

def pool300(listed, val):
    v = val.reindex(index=listed.index, columns=listed.columns)
    m60 = val.reindex(columns=listed.columns).rolling(60, min_periods=40).mean().shift(1).reindex(listed.index)
    m60 = m60.where(listed)
    rk = m60.rank(axis=1, ascending=False, method='first')
    return (rk <= 300) & listed

def load_yh():
    y = pd.read_parquet(RAW / 'yh_tej_margin_2005_2024.parquet'); y['date'] = pd.to_datetime(y.date).astype('datetime64[ns]'); return y
def load_fb(nm='margin_buy'):
    d = pd.read_parquet(RAW / f'finbrain_{nm}.parquet'); d.index = pd.to_datetime(d.index).astype('datetime64[ns]'); return d[common(d.columns)]
def load_bb():
    b = pd.read_parquet(AG / 'o_sbl_foreign_level/raw/bb_margin.parquet')
    b = b[b.Market == 'TWSE'].copy(); b['date'] = pd.to_datetime(b.Date).astype('datetime64[ns]'); b['code'] = b.Ticker.astype(str); return b
def load_mh():
    m = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/margin_hist.csv.gz', dtype={'code': str}); m['date'] = pd.to_datetime(m.date); return m
def load_joeyc(kind):
    rows = []
    for f in sorted(glob.glob(str(RAW / 'joeyc' / kind / '*.csv.gz'))):
        d = pd.read_csv(f, dtype={'code': str}); d['date'] = pd.Timestamp(Path(f).name[:10]); rows.append(d)
    return pd.concat(rows, ignore_index=True)
def load_vf():
    q = pd.read_parquet(AG / 'o_sbl_foreign_level/raw/vf_twse_foreign.parquet'); q['date'] = pd.to_datetime(q.date).astype('datetime64[ns]'); q['code'] = q.code.astype(str); return q
