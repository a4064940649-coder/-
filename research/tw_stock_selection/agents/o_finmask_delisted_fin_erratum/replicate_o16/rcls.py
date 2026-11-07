"""o16 獨立複製：分類器 F1／F2（只依 ERRATUM_SPEC_FM.json 文字，自行從原始檔重建）。
不 import 研究員本題程式，也不 import erlib。"""
import hashlib
from pathlib import Path
import pandas as pd

AG = Path('/home/user/-/research/tw_stock_selection/agents')
RAW = AG / 'o_pit_shares_industry_2006_2013' / 'raw'
SNAP = RAW / 'industry_snap'
CEN = RAW / 'center'
SL26 = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv')
TWREV = Path('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv')

KW = ['銀行', '商銀', '金控', '金融', '證券', '證', '票券', '保險', '人壽', '產險', '期貨', '投信']
PFX = ('28', '58', '60')
FIN_LABELS = {'金融保險業', '金融保險', '金融業'}

SHA = {
    SNAP / 'goristock_14393dd2_grs_stock_no.csv': 'ffd9050122ec49ef057b6675059c4316d63f575ee97669a60355189fe118f2a8',
    SNAP / 'goristock_d9d8662c_grs_stock_no.csv': '92474e7c0153ba450b1cd991cee154da45672cfdd5c13125db15ae8f4fbba220',
    SNAP / 'goristock_235da71f_grs_old_stock_no.csv': '0004e426427221be3f4f4e66ec45d2d19768b71f2490a2471ff5aa74b71b8c9d',
    SNAP / 'goristock_c3c718ab_grs_stock_no.csv': 'df09d5a32b700dfc08d1903c348ac02f961d7cab140fdff96b88ee50e6db3965',
    SNAP / 'grs_893c8772_grs_stock_no.csv': '52c68861590dfaef1ec0deac7807e4ee90a93d82ab3f6439e454249c59946d09',
    SNAP / 'grs_568e6685_tools_twse_list.csv': '00fa81b5d85a17687c575f6825741f517a5931f441e249a05de087d676c04534',
    CEN / 'StockList_20150522.csv': '73dfca4f109ef083f0022c9cb02a08b4b6e26b9fa1b1fb3997ebf6a044dc8362',
    CEN / 'StockList_20160401.csv': '72a640a8e0fe1255ccc3220c352ac27bbbc6c6225f34fabcd8ce70dedff25292',
    CEN / 'StockList_20170706.csv': '26e485f8d0f33dac1baee32a962e7b33a496c01e5fea1cbac752afb6615e0349',
    CEN / 'isin_20230116_listed.csv': '37107c39666e272ee6f5af355e4248c866e8b0c85c40bf1827337d28a2558eae',
    SL26: '06111252bee0f002a85434a71623e93acc74a597f36412c4a3871d243cca695e',
    TWREV: '02657e7c415783020f16255d42f66b0d40d69823fa294113557be6f4584a3898',
    CEN / 'listing_master.csv': 'bdb7d6d478c095655c9a9707f550551a128df78839e7a563d81feab09f4988c8',
    CEN / 'delist_official.csv': '41f51cb7e1aadbf1d28244f26cb6452f3ede20f2fbb594350848ed8f55a1f2bd',
}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def check_sha():
    bad = {str(p): sha(p) for p, h in SHA.items() if sha(p) != h}
    assert not bad, bad

def _snap_goristock(path):
    """回傳 (生效日 or None, {code: (name, tse_int)})"""
    import csv, io
    rows = list(csv.reader(io.StringIO(Path(path).read_text(encoding='utf-8-sig'))))
    eff = None; out = {}
    for parts in rows[1:]:
        parts = [x.strip() for x in parts]
        if len(parts) < 2 or not parts[0]: continue
        if parts[0] == 'UPDATE': eff = parts[1]; continue
        k = parts[2] if len(parts) > 2 else ''
        out[parts[0]] = (parts[1], int(k) if k.isdigit() else None)
    return eff, out

def _snap_twse_list(path):
    import csv, io
    rows = list(csv.reader(io.StringIO(Path(path).read_text(encoding='utf-8-sig'))))
    out = {}
    for p in rows[2:]:
        p = [x.strip() for x in p]
        if len(p) < 4: continue
        try: k = int(p[2])
        except ValueError: k = None
        out[p[0]] = (p[1], k)
    return out

def load_sources():
    check_sha()
    snaps = []   # (date, {code: (name, is_fin)})
    for f, commit_date in [('goristock_14393dd2_grs_stock_no.csv', None), ('goristock_d9d8662c_grs_stock_no.csv', '2011-08-17'),
                           ('goristock_235da71f_grs_old_stock_no.csv', None), ('goristock_c3c718ab_grs_stock_no.csv', None),
                           ('grs_893c8772_grs_stock_no.csv', None)]:
        eff, m = _snap_goristock(SNAP / f)
        eff = eff or commit_date
        snaps.append((pd.Timestamp(eff), {c: (n, k == 17) for c, (n, k) in m.items()}, f))
    m = _snap_twse_list(SNAP / 'grs_568e6685_tools_twse_list.csv')
    snaps.append((pd.Timestamp('2013-12-18'), {c: (n, k == 17) for c, (n, k) in m.items()}, 'twse_list'))
    snaps.sort(key=lambda x: x[0])
    later = []
    for f, dt in [('StockList_20150522.csv', '2015-05-22'), ('StockList_20160401.csv', '2016-04-01'), ('StockList_20170706.csv', '2017-07-06')]:
        d = pd.read_csv(CEN / f, dtype=str, encoding='utf-8-sig')
        later.append((pd.Timestamp(dt), {str(c).strip(): (n, str(i).strip() in FIN_LABELS) for c, n, i in zip(d['代號'], d['名稱'], d['產業別'])}, f))
    d = pd.read_csv(CEN / 'isin_20230116_listed.csv', dtype=str)
    later.append((pd.Timestamp('2023-01-16'), {c: (n, str(i).strip() in FIN_LABELS) for c, n, i in zip(d['code'], d['name'], d['industry'])}, 'isin'))
    d = pd.read_csv(SL26, dtype=str)
    later.append((pd.Timestamp('2026-10-01'), {c: (n, str(i).strip() in FIN_LABELS) for c, n, i in zip(d['code'], d['name'], d['industry'])}, 'stock_list_2026'))
    # 名稱來源
    names = {}
    def addn(c, n):
        if isinstance(n, str) and n.strip(): names.setdefault(c, set()).add(n.strip())
    for _, mm, _ in snaps + later:
        for c, (n, _) in mm.items(): addn(c, n)
    tr = pd.read_csv(TWREV, dtype=str, usecols=['code', 'name']).drop_duplicates()
    for c, n in zip(tr['code'], tr['name']): addn(c, n)
    do = pd.read_csv(CEN / 'delist_official.csv', dtype=str)
    for c, n in zip(do['code'], do['name']): addn(c, n)
    sl26 = pd.read_csv(SL26, dtype=str).drop_duplicates('code').set_index('code')
    lm = pd.read_csv(CEN / 'listing_master.csv', dtype=str).set_index('code')
    return dict(snaps=snaps, later=later, names=names, sl26=sl26, lm=lm, twrev_codes=set(tr['code']))

def evidence(c, S):
    return c[:2] in PFX or any(k in n for n in S['names'].get(c, ()) for k in KW)

def f1_status(c, d, S):
    d = pd.Timestamp(d)
    pit = [m[c][1] for dt, m, _ in S['snaps'] if dt <= d and c in m]
    if pit:
        return 'FIN' if pit[-1] else 'NONFIN'
    src = sorted([(dt, m[c][1]) for dt, m, _ in S['snaps'] + S['later'] if dt > d and c in m], key=lambda x: x[0])
    ev = evidence(c, S)
    if not src:
        return 'UNKNOWN_EVID' if ev else 'UNKNOWN_NOEVID'
    fin = src[0][1]
    if fin and ev: return 'FIN_Q'
    if (not fin) and (not ev): return 'NONFIN_Q'
    return 'UNKNOWN_EVID' if ev else 'UNKNOWN_NOEVID'

def f2_fin(c, S):
    return evidence(c, S)   # 人工排除清單為空集合

def is_delisted(c, S):
    lm = S['lm']
    dd = lm['delist_date'].get(c) if c in lm.index else None
    return (c not in S['sl26'].index) or (isinstance(dd, str) and dd.strip() != '')
