"""獨立複製 o9：資料截斷、日曆、名單（只依凍結規格；不讀研究員程式）。"""
import sys, json, hashlib, importlib.util, shutil
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE / 'work'; WORK.mkdir(exist_ok=True)
ROOT = Path('/home/user/-/research/tw_stock_selection')
RAW = ROOT / 'agents/o_forward_infra_live/raw/screener/data'
FIN = ROOT / 'agents/o_forward_infra_live/raw/finlab'
PANEL = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


MP = load_mod('mp_frozen', ROOT / 'main_picks_today.py')
assert hashlib.sha256((ROOT / 'main_picks_today.py').read_bytes()).hexdigest().startswith('0897787f617cec3b')

# ---------------- 原始資料（一次讀入） ----------------
H = pd.read_csv(RAW / 'history.csv.gz', dtype={'code': str}, parse_dates=['date'])
LST = pd.read_csv(RAW / 'stock_list.csv', dtype=str)
EX = pd.read_csv(RAW / 'extras/exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
R5 = pd.read_csv(RAW / 'extras/rev_5y.csv.gz', dtype={'code': str})
RH = pd.read_csv(RAW / 'extras/rev_hist.csv.gz', dtype={'code': str})
DSP5 = pd.read_csv(RAW / 'extras/disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
WH = pd.read_csv(RAW / 'extras/warnings_hist.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
ATT = pd.read_csv(RAW / 'extras/attention_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
HOL = pd.DatetimeIndex(json.loads((RAW / 'extras/tw_holidays.json').read_text())['years']['2026'])
E50 = pd.read_csv(FIN / 'etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
TX = pd.read_csv(FIN / 'taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index()
HCAL = pd.DatetimeIndex(sorted(H['date'].unique()))
HS = HCAL[0]
DATA_END = HCAL[-1]


def calendar(until='2027-12-31'):
    """已知交易日（history）＋之後以平日扣除 tw_holidays 預估。"""
    fut = pd.bdate_range(DATA_END + pd.Timedelta(days=1), until)
    fut = fut[~fut.isin(HOL)]
    return HCAL.append(fut)


CAL = calendar()


def d_m(y, m, cal=CAL):
    return cal[(cal >= pd.Timestamp(y, m, 10))][0]


def j_days(y, m, cal=CAL):
    i = cal.get_loc(d_m(y, m, cal))
    return list(cal[i:i + 13])


def official_day(y, m, cal=CAL):
    return cal[(cal >= pd.Timestamp(y, m, 11))][0]


def statutory(ym):
    p = pd.Period(ym, 'M') + 1
    return CAL[CAL >= pd.Timestamp(p.year, p.month, 10)][0]


# ---------------- 營收 ann_eff ----------------
_ann = RH.dropna(subset=['ann_date']).drop_duplicates(['code', 'ym'], keep='last').set_index(['code', 'ym'])['ann_date']
_stat = {ym: statutory(ym) for ym in sorted(set(R5['ym']) | set(RH['ym']))}


def _eff(df):
    a = pd.to_datetime(pd.Series([_ann.get((c, y)) for c, y in zip(df['code'], df['ym'])], index=df.index))
    st = pd.to_datetime(df['ym'].map(_stat))
    return a.fillna(st), a


R5['ann_eff'], R5['ann_raw'] = _eff(R5)
RH['ann_eff'], RH['ann_raw'] = _eff(RH)

# 處置：disposal_5y ∪ warnings_hist(處置)，依 (code,start,end) 去重保留最早公告日
DSPU = pd.concat([DSP5[DSP5['flag'] == '處置'], WH[WH['flag'] == '處置']]).dropna(subset=['start', 'end'])
DSPU = DSPU.sort_values('date').drop_duplicates(['code', 'start', 'end'], keep='first').reset_index(drop=True)

# 面板子集 [2025-01-01, HS)
PSUB = WORK / 'panel_sub'
if not (PSUB / 'adj.parquet').exists():
    PSUB.mkdir(exist_ok=True)
    for f in ['adj', 'close_raw', 'open', 'high', 'value', 'ret']:
        p = pd.read_parquet(PANEL / f'{f}.parquet')
        p.loc[(p.index >= '2025-01-01') & (p.index < HS)].to_parquet(PSUB / f'{f}.parquet')


def trunc_dir(asof):
    """建立截斷到 asof 的資料夾（main_picks_today 讀得懂的格式）。"""
    asof = pd.Timestamp(asof)
    d = WORK / f'trunc_{asof:%Y%m%d}'
    if (d / 'done').exists(): return d
    (d / 'extras').mkdir(parents=True, exist_ok=True)
    shutil.copy(RAW / 'stock_list.csv', d / 'stock_list.csv')
    H[H['date'] <= asof].to_csv(d / 'history.csv.gz', index=False, date_format='%Y-%m-%d', compression={'method': 'gzip', 'compresslevel': 1})
    EX[EX['date'] <= asof].to_csv(d / 'extras/exdiv_5y.csv.gz', index=False, date_format='%Y-%m-%d')
    R5[R5['ann_eff'] <= asof].drop(columns=['ann_eff', 'ann_raw']).to_csv(d / 'extras/rev_5y.csv.gz', index=False)
    RH[RH['ann_eff'] <= asof].drop(columns=['ann_eff', 'ann_raw']).to_csv(d / 'extras/rev_hist.csv.gz', index=False)
    DSPU[DSPU['date'] <= asof].to_csv(d / 'extras/disposal_5y.csv.gz', index=False, date_format='%Y-%m-%d')
    (d / 'done').write_text('1')
    return d


_cache = {}


def mp_data(asof):
    asof = pd.Timestamp(asof)
    if asof in _cache: return _cache[asof]
    D = trunc_dir(asof)
    data = MP.load_prices(D, asof, PSUB)
    YOY, YOY3, nf = MP.load_revenue(D)
    dsp = MP.load_disposal(D)
    # 斷言
    for k in ['C', 'O', 'H', 'V', 'R', 'ADJ']:
        assert data[k].index.max() <= asof, k
    assert dsp['date'].max() <= asof
    assert YOY.index[YOY.notna().sum(axis=1) > 0].max() <= pd.Period(asof, 'M') - 1
    out = dict(data=data, YOY=YOY, YOY3=YOY3, dsp=dsp, D=D)
    _cache[asof] = out
    return out


def official_list(sd):
    m = mp_data(sd)
    order, df, mm, nel = MP.score_on(m['data'], m['YOY3'], m['dsp'], sd, ex_fin=True)
    return order, df


def j_list(sd, j, missing_as_unannounced=True):
    """J 線：m−1 營收已公告才用，否則用 m−2 的 3 月平均。"""
    M = mp_data(sd); data = M['data']
    mprev = pd.Period(sd, 'M') - 1
    cols = data['C'].columns
    # 公告判定（以截斷後資料中的 m−1 列）
    sub5 = R5[(R5['ym'] == str(mprev)) & (R5['ann_eff'] <= sd)]
    subh = RH[(RH['ym'] == str(mprev)) & (RH['ann_eff'] <= sd)]
    present = set(sub5['code']) | set(subh['code'])
    annraw = {}
    for c, a in zip(RH['code'][RH['ym'] == str(mprev)], RH['ann_raw'][RH['ym'] == str(mprev)]):
        annraw[c] = a
    announced = {}
    miss = set()
    for c in cols:
        if c not in present: announced[c] = False; continue
        a = annraw.get(c, pd.NaT)
        if pd.isna(a):
            miss.add(c)
            announced[c] = (j >= 1) or (not missing_as_unannounced)
        else:
            announced[c] = a <= sd
    Y3 = M['YOY3']
    y_now = Y3.reindex([mprev]).iloc[0].reindex(cols) if mprev in Y3.index else pd.Series(np.nan, index=cols)
    y_old = Y3.reindex([mprev - 1]).iloc[0].reindex(cols)
    ann = pd.Series(announced).reindex(cols).fillna(False).astype(bool)
    ymix = y_now.where(ann, y_old)
    Y3J = Y3.copy()
    Y3J.loc[mprev] = ymix.reindex(Y3J.columns)
    order, df, mm, nel = MP.score_on(data, Y3J, M['dsp'], sd, ex_fin=True)
    nmiss_pool = len(set(order.index) & miss)
    return order, df, nmiss_pool, miss
