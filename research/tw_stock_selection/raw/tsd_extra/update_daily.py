#!/usr/bin/env python3
"""官方資料倉庫（tw-stock-data）→ 自有快照（raw/tsd_extra/）的建置＋每日增量更新，同一支程式。

用法（一律 python3 -I -B；下載來的內容只當資料讀，不 checkout、不執行）：
  update_daily.py <官方倉庫 clone 路徑> [--out <輸出資料夾，預設＝本檔所在資料夾>]
                  [--only 名稱1,名稱2] [--refresh-bulk] [--allow-older] [--quiet]
  update_daily.py --verify [--out …]            # 只檢查每個面板檔的 sha256 與 SHA256SUMS 是否一致
  update_daily.py --truncate N --out <複本>     # 測試用：把「每日／每週」面板截掉最後 N 個日期（只准對複本做）

做法：
  1. 只讀 git 的目錄表（不需要檔案內容）拿到每個來源檔的 blob sha，和上次記下的（各資料集的 _sources.tsv）比。
     沒有新檔、也沒有內容變動的檔 → 該資料集印「無新資料」，什麼都不寫。
  2. 有新檔或內容變了的檔：一次批次把這些檔的內容抓回來（每天每類 1 個小檔），用 git cat-file 讀。
  3. 以「來源鍵」（每日類＝日期）為單位：先把面板裡同鍵的舊列拿掉，再放進新列，依鍵排序後重寫那一個分檔。
     所以同一天跑兩次結果相同；上游事後改了舊檔也會跟著更新（改了哪個檔記在 _revisions.tsv）。
  4. 月營收等「原檔快照」類：第一次看到的版本存 first/，之後內容變了另存 rev/（檔名帶新版 sha），first/ 永不覆蓋。
  5. 每個資料集各自 try：一類失敗不影響其他類；最後有任何一類失敗就回傳代碼 1 並印原因。

輸出結構：<資料集>/<年>/<分檔>.parquet（每日類按月分檔、集保按週、季資料與事件類按年）；
  每個年資料夾有 SHA256SUMS、_parts.tsv（各分檔列數與鍵範圍）、_sources.tsv（來源檔→blob sha→列數）；
  根目錄 manifest.json 是總表（涵蓋到哪一天、列數、缺日）。
回傳代碼：0＝成功（含無新資料）；1＝至少一類失敗；2＝用法錯誤。
"""
import sys, os, io, re, json, gzip, hashlib, subprocess, threading, argparse, datetime, traceback

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

URL = 'https://github.com/jb7k6295r7-bot/tw-stock-data'
ENV_NOFETCH = dict(os.environ, GIT_NO_LAZY_FETCH='1')


# ───────────────────────── git 小工具（只讀；抓內容時一次批次抓） ─────────────────────────
def git(repo, *a, inp=None, env=ENV_NOFETCH):
    r = subprocess.run(['git', '-C', repo, '-c', 'core.quotePath=false', *a], input=inp, capture_output=True, env=env)
    if r.returncode != 0:
        raise RuntimeError('git %s 失敗：%s' % (' '.join(a[:3]), r.stderr.decode(errors='replace')[-800:]))
    return r.stdout


def repo_head(repo):
    h = git(repo, 'rev-parse', 'HEAD').decode().strip()
    d = git(repo, 'log', '-1', '--format=%cI').decode().strip()
    return h, d


def ls_tree(repo, path):
    out = {}
    for ln in git(repo, 'ls-tree', '-r', 'HEAD', path).decode().splitlines():
        meta, p = ln.split('\t', 1)
        m = meta.split()
        if m[1] == 'blob':
            out[p] = m[2]
    return out


def fetch_blobs(repo, shas):
    shas = sorted(set(shas))
    if not shas:
        return
    local = set(git(repo, 'cat-file', '--batch-all-objects', '--unordered', '--batch-check=%(objectname)').decode().split())
    need = [x for x in shas if x not in local]
    if not need:
        return
    r = subprocess.run(['git', '-C', repo, '-c', 'fetch.negotiationAlgorithm=noop', 'fetch', 'origin', '--no-tags', '--no-write-fetch-head',
                        '--recurse-submodules=no', '--filter=blob:none', '--stdin', '-q'], input=('\n'.join(need) + '\n').encode(), capture_output=True)
    if r.returncode != 0:
        raise RuntimeError('抓檔案內容失敗（%d 個檔）：%s' % (len(need), r.stderr.decode(errors='replace')[-800:]))


def iter_blobs(repo, ents):
    """ents=[(路徑, sha)] → 依序 (路徑, bytes)。"""
    p = subprocess.Popen(['git', '-C', repo, 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=ENV_NOFETCH)

    def feed():
        p.stdin.write(('\n'.join(s for _, s in ents) + '\n').encode()); p.stdin.close()
    threading.Thread(target=feed, daemon=True).start()
    try:
        for path, sha in ents:
            hd = p.stdout.readline().decode().split()
            if len(hd) != 3 or hd[0] != sha or hd[1] != 'blob':
                raise RuntimeError('讀不到內容：%s %s' % (path, hd))
            body = p.stdout.read(int(hd[2])); p.stdout.read(1)
            yield path, body
    finally:
        p.stdout.close(); p.wait()


# ───────────────────────── 檔案小工具 ─────────────────────────
def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def write_if_changed(path, data):
    """內容相同就不動檔（重複執行不產生差異）。回傳是否有寫。先寫暫存檔再改名。"""
    if os.path.exists(path):
        with open(path, 'rb') as f:
            if f.read() == data:
                return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(data)
    os.replace(tmp, path)
    return True


def read_tsv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        return [ln.rstrip('\n').split('\t') for ln in f if ln.strip() and not ln.startswith('#')]


def tsv_bytes(header, rows):
    return ('#' + '\t'.join(header) + '\n' + ''.join('\t'.join(str(x) for x in r) + '\n' for r in rows)).encode('utf-8')


def gz_bytes(b):
    buf = io.BytesIO()
    with gzip.GzipFile(filename='', mode='wb', fileobj=buf, compresslevel=9, mtime=0) as g:
        g.write(b)
    return buf.getvalue()


# ───────────────────────── 表格：解析、型別、排序、寫檔 ─────────────────────────
def to_num(s, kind, col, path):
    """字串欄 → 數值（空白＝空值）。解析不了就整類失敗（寧可停，不要悄悄變空值）。"""
    s = s.astype(str).str.strip().str.replace(',', '', regex=False)
    empty = (s == '').to_numpy()
    v = pd.to_numeric(s.where(~empty, None), errors='coerce').to_numpy(dtype='float64', na_value=np.nan)
    bad = np.isnan(v) & ~empty
    if bad.any():
        raise ValueError('%s 欄 %s 有 %d 個不是數字的值，例：%r' % (path, col, int(bad.sum()), s[bad].iloc[0]))
    if kind == 'int':
        ok = v[~empty]
        if ok.size and (np.abs(ok) >= 2 ** 53).any():   # 超過浮點可精確表示的整數範圍 → 逐值用 Python 整數
            vals = [None if e else int(x) for x, e in zip(s, empty)]
            return pa.array(vals, type=pa.int64())
        if ok.size and (ok != np.round(ok)).any():
            raise ValueError('%s 欄 %s 應為整數但有小數' % (path, col))
        return pa.array(np.where(empty, 0, v).astype('int64'), mask=empty, type=pa.int64())
    if kind == 'f32':
        return pa.array(v.astype('float32'), mask=empty, type=pa.float32())
    return pa.array(v, mask=empty, type=pa.float64())


def csv_table(body, path, spec):
    """一個來源 csv → arrow 表（寬型別：整數一律 int64，寫檔前才縮）。欄位要與規格一致，否則報錯。"""
    df = pd.read_csv(io.BytesIO(body), dtype=str, keep_default_na=False, encoding='utf-8-sig')
    ren = spec.get('rename', {})
    cols = spec['cols']
    drop = set(spec.get('drop', []))
    want = set(ren.get(c, c) for c in df.columns if c not in drop)
    known = set(cols)
    if not want <= known:
        raise ValueError('%s 出現沒看過的欄位 %s（上游改了格式？）' % (path, sorted(want - known)))
    df = df.rename(columns=ren)
    if 'check' in spec:
        spec['check'](df, path)
    arrs = {}
    for c, kind in cols.items():
        if c not in df.columns:
            if kind == 'date':
                raise ValueError('%s 缺日期欄 %s' % (path, c))
            arrs[c] = pa.nulls(len(df), type={'int': pa.int64(), 'float': pa.float64(), 'f32': pa.float32(), 'str': pa.string(), 'ts': pa.timestamp('s')}[kind])
        elif kind == 'date':
            arrs[c] = pa.array(pd.to_datetime(df[c], format='%Y-%m-%d').dt.date, type=pa.date32())
        elif kind == 'ts':
            arrs[c] = pa.array(pd.to_datetime(df[c].where(df[c] != '', None), format='%Y-%m-%d %H:%M:%S').to_numpy().astype('datetime64[s]'), type=pa.timestamp('s'))
        elif kind == 'str':
            arrs[c] = pa.array(df[c].astype(str).to_numpy(dtype=object), type=pa.string())
        else:
            arrs[c] = to_num(df[c], kind, c, path)
    return pa.table(arrs)


def shrink_ints(t):
    """整數欄縮到放得下的最小實體型別（parquet 實體只有 32／64 位元；int8／int16 存起來一樣是 32 位元，所以只分這兩種）。"""
    out = []
    for f in t.schema:
        col = t.column(f.name)
        if pa.types.is_integer(f.type):
            mm = pc.min_max(col).as_py()
            if mm['min'] is None or (mm['min'] >= -2 ** 31 and mm['max'] < 2 ** 31):
                col = col.cast(pa.int32())
            else:
                col = col.cast(pa.int64())
        out.append(col.combine_chunks())
    return pa.table(out, names=t.schema.names)


def widen(t, cols):
    out = []
    for name, kind in cols.items():
        col = t.column(name)
        if kind == 'int':
            col = col.cast(pa.int64())
        out.append(col)
    return pa.table(out, names=list(cols))


def parquet_bytes(t):
    buf = io.BytesIO()
    pq.write_table(shrink_ints(t), buf, compression='zstd', compression_level=19, use_dictionary=True, write_statistics=True,
                   data_page_size=1 << 20, row_group_size=1 << 20)
    return buf.getvalue()


def key_strings(t, spec):
    return spec['rowkey'](t)


def sort_table(t, spec):
    """依排序欄做穩定排序（同鍵的列保持來源檔裡的順序）。"""
    if t.num_rows == 0:
        return t
    keys = []
    for c in reversed(spec['sort']):
        col = t.column(c)
        if pa.types.is_date(col.type) or pa.types.is_timestamp(col.type):
            col = col.cast(pa.int64() if pa.types.is_timestamp(col.type) else pa.int32())
        a = col.to_pandas()
        if a.dtype == object:
            a = a.fillna('')
            _, a = np.unique(a.to_numpy(dtype=str), return_inverse=True)
        else:
            a = a.to_numpy()
        keys.append(a)
    return t.take(pa.array(np.lexsort(keys)))


# ───────────────────────── 資料集規格 ─────────────────────────
def k_date(t):
    return [d.isoformat() for d in t.column('date').to_pylist()]


def part_month(k): return k[:4], k[:7]
def part_day(k): return k[:4], k[:10]
def part_year(k): return k[:4], k[:4]


def daily_spec(name, src, cols, desc, part=part_month, early=None, **kw):
    srcs = [dict(dir=x, pat=r'^\d{4}-\d{2}-\d{2}\.csv$', srckeys=lambda b: [b[:10]]) for x in ([early] if early else []) + [src]]
    d = dict(name=name, src=srcs, kind='table', cols=cols,
             rename={'stock_id': 'code'}, sort=['date', 'code'] if 'code' in cols else ['date'], rowkey=k_date, part=part, calendar=True, desc=desc)
    d.update(kw)
    return d


def chk_daily(df, path):
    if not (df['key'] == df['date'] + '_' + df['code']).all():
        raise ValueError('%s 的 key 欄不等於 日期_代號' % path)


I, F, S, D = 'int', 'float', 'str', 'date'
INST = {'date': D, 'code': S, 'foreign': I, 'trust': I, 'dealer': I, 'total': I, 'dealer_self': I, 'dealer_hedge': I}
MARGIN = {'date': D, 'code': S, 'm_buy': I, 'm_sell': I, 'm_balance': I, 'm_limit': I, 's_buy': I, 's_sell': I, 's_balance': I,
          'm_prev': I, 'm_ret': I, 's_prev': I, 's_ret': I, 'note': S}
SBL = {'date': D, 'code': S, 's_prev': I, 's_sell': I, 's_buy': I, 's_ret': I, 's_balance': I, 's_limit': I, 'sbl_prev': I, 'sbl_sell': I,
       'sbl_return': I, 'sbl_adj': I, 'sbl_balance': I, 'sbl_limit': I, 'note': S}
AMT = {'date': D, 'investor': S, 'buy': I, 'sell': I, 'net': I}
D_INST = '三大法人逐股買賣超，單位＝股。foreign 外資、trust 投信、dealer 自營商合計（＝dealer_self 自行買賣＋dealer_hedge 避險）、total 三者合計。'
D_MARGIN = ('融資融券逐股，單位＝張（交易單位）。m_＝融資：buy 買進、sell 賣出、ret 現金償還、prev 前日餘額、balance 今日餘額、limit 限額；'
            's_＝融券：sell 賣出、buy 買進（回補）、ret 現券償還、prev 前日餘額、balance 今日餘額。note＝官方註記（O 停止融資、X 停止融券等）。券資比＝s_balance÷m_balance，自己算。')
D_SBL = ('信用額度總量管制餘額表，單位＝股。s_＝融券（同融資融券表、乘 1000）；sbl_＝借券賣出：prev 前日餘額、sell 當日賣出、return 當日還券、adj 當日調整、'
         'balance 當日餘額（＝借券賣出餘額）、limit 次一營業日可限額。')

BS_STD = {'資產總計': 'total_assets', '資產總額': 'total_assets', '負債總計': 'total_liabilities', '負債總額': 'total_liabilities',
          '權益總計': 'equity_total', '權益總額': 'equity_total', '歸屬於母公司業主之權益合計': 'equity_parent', '歸屬於母公司業主權益合計': 'equity_parent',
          '股本': 'share_capital', '流動資產': 'current_assets', '非流動資產': 'noncurrent_assets', '流動負債': 'current_liabilities',
          '非流動負債': 'noncurrent_liabilities', '現金及約當現金': 'cash', '資本公積': 'capital_surplus', '保留盈餘': 'retained_earnings',
          '保留盈餘（或累積虧損）': 'retained_earnings', '其他權益': 'other_equity', '庫藏股票': 'treasury_stock', '非控制權益': 'minority_interest',
          '每股參考淨值': 'bvps_ref', '應付公司債': 'bonds_payable', '應付債券': 'bonds_payable', '應付金融債券': 'bank_debentures_payable',
          '短期債務': 'short_term_debt', '其他借款': 'other_borrowings', '應收款項－淨額': 'receivables_net', '應收款項': 'receivables_net'}


def parse_bs(body, path, spec):
    """資產負債表簡明表（寬表、各業別欄位不同）→ 長表：一格一列，空白格不產生列。金額單位＝千元。"""
    base = os.path.basename(path)[:-4]
    per, tpl, mk = base.split('_')
    df = pd.read_csv(io.BytesIO(body), dtype=str, keep_default_na=False, encoding='utf-8-sig')
    if list(df.columns[:4]) != ['stock_id', 'name', 'period', 'market']:
        raise ValueError('%s 前四欄不是 stock_id,name,period,market' % path)
    if len(df) and not ((df['period'] == per) & (df['market'] == mk)).all():
        raise ValueError('%s 內容的季別／市場與檔名不符' % path)
    acc = list(df.columns[4:])
    m = df.melt(id_vars=['stock_id'], value_vars=acc, var_name='account', value_name='v')
    m['_r'] = np.tile(np.arange(len(df)), len(acc)); m['_c'] = np.repeat(np.arange(len(acc)), len(df))
    m['v'] = m['v'].str.strip().str.replace(',', '', regex=False)
    m = m[m['v'] != ''].sort_values(['_r', '_c'], kind='stable')
    val = pd.to_numeric(m['v'], errors='coerce')
    if val.isna().any():
        raise ValueError('%s 有不是數字的值：%r' % (path, m['v'][val.isna()].iloc[0]))
    n = len(m)
    return pa.table({'period': pa.array([per] * n, pa.string()), 'code': pa.array(m['stock_id'].str.strip().to_numpy(dtype=object), pa.string()),
                     'market': pa.array([mk] * n, pa.string()), 'template': pa.array([tpl] * n, pa.string()),
                     'account': pa.array(m['account'].to_numpy(dtype=object), pa.string()),
                     'item': pa.array([BS_STD.get(a) for a in m['account']], pa.string()), 'value': pa.array(val.to_numpy(dtype='float64'), pa.float64())})


def parse_tdcc_hist(body, path, spec):
    t = pq.read_table(io.BytesIO(body))
    names = set(t.schema.names)
    if names != {'date', 'stock_id', 'level', 'people', 'shares', 'pct'}:
        raise ValueError('%s 欄位不是預期的集保格式：%s' % (path, sorted(names)))
    d = pd.to_datetime(t.column('date').cast(pa.string()).to_pandas(), format='%Y-%m-%d').dt.date
    return pa.table({'date': pa.array(d, pa.date32()), 'code': t.column('stock_id').cast(pa.string()), 'level': t.column('level').cast(pa.int64()),
                     'people': t.column('people').cast(pa.int64()), 'shares': t.column('shares').cast(pa.int64()), 'pct': t.column('pct').cast(pa.float32())})


def norm_kind(df, path):
    if 'type' in df.columns:
        df['kind_raw'] = df['type']
        df['type'] = df['type'].str.replace('除', '', regex=False)


EV_SORT = ['date', 'code']
DATASETS = [
    daily_spec('daily', 'data/universe/daily',
               {'date': D, 'code': S, 'name': S, 'market': S, 'open': F, 'high': F, 'low': F, 'close': F, 'volume': I, 'amount': I, 'change': F,
                'limit': S, 'shares': I, 'transactions': I, 'price_basis': S, 'last_price': F}, drop=['key'], check=chk_daily,
               desc='日 K（上市 twse、上櫃 tpex、興櫃 emerging 同一張表）。open／high／low／close 未還原價；volume 成交股數；amount 成交金額（元）；change 漲跌價差；'
                    'limit 漲跌停註記（up／down／flat／空白）；shares 發行股數；transactions 成交筆數；price_basis「無成交」等註記；last_price 興櫃最後成交價。'
                    '原始檔的 key 欄（＝日期_代號）沒存，可由 date、code 還原。'),
    daily_spec('inst_twse', 'data/universe/inst', INST, '上市' + D_INST,
               early=None),  # 待辦：early='data/early/inst'（2012-05 起）磁碟夠再補
    daily_spec('inst_tpex', 'data/universe/otcinst', INST, '上櫃' + D_INST),
    daily_spec('margin_twse', 'data/universe/margin', MARGIN, '上市' + D_MARGIN),
    daily_spec('margin_tpex', 'data/universe/otcmargin', MARGIN, '上櫃' + D_MARGIN),
    daily_spec('sbl_twse', 'data/universe/sbl', SBL, '上市' + D_SBL),
    daily_spec('sbl_tpex', 'data/universe/otcsbl', SBL, '上櫃' + D_SBL),
    daily_spec('instamt_twse_market', 'data/universe/instamt', AMT, '上市三大法人買賣金額（全市場合計，不是逐股），單位＝元。investor＝法人類別（各年代名稱不同）。',
               part=part_year, early=None),  # 待辦：early='data/early/instamt'
    daily_spec('instamt_tpex_market', 'data/universe/otcinstamt', AMT, '上櫃三大法人買賣金額（全市場合計），單位＝元；2017-01-03 起。', part=part_year, calendar_from='2017-01-03'),
    daily_spec('marginmkt_twse_market', 'data/universe/marginmkt', {'date': D, 'item': S, 'buy': I, 'sell': I, 'repay': I, 'prev': I, 'balance': I},
               '上市融資融券全市場統計：融資（交易單位）、融券（交易單位）、融資金額（仟元）。',
               part=part_year, early=None),  # 待辦：early='data/early/marginmkt'
    daily_spec('marginratio_notices_twse', 'data/universe/marginratio',
               {'date': D, 'code': S, 'name': S, 'reason': S, 'adjust_from': S, 'restore_date': S, 'margin_cut': I, 'margin_cum': I, 'short_raise': I, 'short_cum': I},
               '上市「調整融資比率／融券保證金成數」公告名單（不是券資比）。同一檔同一天有多列（各原因一列＋合計一列），列的順序同官方檔。', part=part_year, sort=['date']),
    dict(name='tdcc', enabled=False, kind='table', part=part_day, sort=['date', 'code', 'level'], rowkey=k_date, weekly=True,
         cols={'date': D, 'code': S, 'level': I, 'people': I, 'shares': I, 'pct': 'f32'}, rename={'stock_id': 'code'},
         src=[dict(dir='data/tdcc_hist', pat=r'^\d{4}\.parquet$', srckeys=None, parse=parse_tdcc_hist, bulk=True),
              dict(dir='data/tdcc', pat=r'^\d{4}-\d{2}-\d{2}\.csv$', srckeys=lambda b: [b[:10]])],
         desc='集保股權分散表（每週一份、每檔 17 列）。level 1～15＝持股分級（1：1～999 股、2：1,000～5,000、3：5,001～10,000、4：～15,000、5：～20,000、6：～30,000、'
              '7：～40,000、8：～50,000、9：～100,000、10：～200,000、11：～400,000、12：～600,000、13：～800,000、14：～1,000,000、15：1,000,001 股以上）、'
              '16＝差異數調整、17＝合計。people 人數、shares 股數、pct 占集保庫存比例（%；2017-07-21 那一週上游沒有此欄）。'
              '常用：400 張以上大戶＝level 12～15 的 pct 合計；散戶（<10 張）＝level 1～3；股東人數＝level 17 的 people。'),
    dict(name='bs_hist', kind='table', part=part_year, sort=['period', 'code', 'template'], weekly=False,
         rowkey=lambda t: [a + '_' + b + '_' + c for a, b, c in zip(t.column('period').to_pylist(), t.column('template').to_pylist(), t.column('market').to_pylist())],
         cols={'period': S, 'code': S, 'market': S, 'template': S, 'account': S, 'item': S, 'value': F},
         src=[dict(dir='data/mops/bs_hist', pat=r'^\d{4}Q[1-4]_[a-z]+_[a-z]+\.csv$', srckeys=lambda b: [b[:-4]], parse=parse_bs)],
         desc='資產負債表（MOPS 簡明表）長表：period 季別、code 代號、market twse／tpex、template 業別表（ci 一般業、basi 銀行、bd 證券、fh 金控、ins 保險、other 其他）、'
              'account 官方原始科目名、item 標準化代稱（對不到的是空值）、value 數值（金額＝新台幣千元；每股參考淨值＝元；股數欄＝股）。空白格不產生列。'
              '注意：一般業（ci）簡明表沒有現金、應收、存貨、借款細項；現金／應收／存貨請用 fs_xbrl；借款（有息負債）上游目前沒有。'),
    dict(name='fs_xbrl', kind='table', part=part_year, sort=['period', 'code'], rowkey=lambda t: [p[:4] for p in t.column('period').to_pylist()],
         cols={'period': S, 'code': S, 'industry_template': S, 'report': S, 'period_end': D, 'inventories': F, 'accounts_receivable': F,
               'accounts_receivable_related': F, 'notes_receivable': F, 'contract_liabilities_current': F, 'contract_liabilities_noncurrent': F, 'cash': F,
               'total_assets': F, 'total_equity': F, 'revenue_ytd': F, 'profit_before_tax_ytd': F, 'net_income_ytd': F, 'operating_cash_flow_ytd': F,
               'depreciation_ytd': F, 'capex_ytd': F, 'cost_of_sales_ytd': F, 'gross_profit_ytd': F, 'operating_income_ytd': F, 'source_file': S},
         rename={'stock_id': 'code'}, src=[dict(dir='data/meta/fs_xbrl', pat=r'^fs_xbrl_\d{4}\.csv$', srckeys=lambda b: [b[8:12]])],
         desc='財報細項（MOPS XBRL 季度整包），一家一季一列，金額單位＝新台幣元。存貨、應收帳款（含關係人、票據）、合約負債、現金、總資產、總權益是期末餘額；'
              '_ytd 欄是年初至今累計。空白＝該公司該季沒有這個科目，不是 0。含未上市櫃的公開發行公司。report：cr 合併、ir 個體。'),
    dict(name='rd_hist', kind='table', part=part_year, sort=['period', 'code'], rowkey=lambda t: t.column('period').to_pylist(),
         cols={'code': S, 'period': S, 'report': S, 'alt': S, 'taxonomy': S, 'rd_q': F, 'rd_ytd': F, 'rev_q': F, 'rev_ytd': F}, rename={'stock_id': 'code'},
         src=[dict(dir='data/mops/rd_hist', pat=r'^\d{4}Q[1-4]\.csv$', srckeys=lambda b: [b[:6]])],
         desc='研發費用（XBRL），單位＝新台幣元：rd_q 單季、rd_ytd 累計；rev_q／rev_ytd 營收（同一份報表，方便算研發占比）。report：cr 合併、ir 個體。'),
    dict(name='filing_dates', kind='table', part=part_year, sort=['code', 'year', 'season', 'uploaded_at', 'file'], whole=True,
         rowkey=lambda t: [str(y) for y in t.column('year').to_pylist()],
         cols={'code': S, 'year': I, 'season': I, 'doc_code': S, 'doc_type': S, 'file': S, 'bytes': I, 'uploaded_at': 'ts', 'corrected': S}, rename={'stock_id': 'code'},
         src=[dict(dir='data/meta/filing_dates.csv', pat=r'^filing_dates\.csv$', srckeys=None)],
         desc='財報電子檔實際上傳時間（公開資訊觀測站），一份文件一列：year／season 財報年季（西元）、doc_type 文件類別、uploaded_at 上傳時戳（台北時間、到秒）、'
              'corrected 更正註記。同一家同一季會有多列（合併／個體、中英文、重編）。要「財報最早可用日」：取該季各列 uploaded_at 的最小值。'),
    dict(name='otc_exright', kind='table', part=part_year, sort=EV_SORT, rowkey=k_date, check=norm_kind,
         cols={'date': D, 'code': S, 'prev_close': F, 'ref_price': F, 'value': F, 'type': S, 'open_base': F, 'source': S, 'kind_raw': S},
         rename={'stock_id': 'code', 'pre_close': 'prev_close', 'kind': 'type'},
         src=[dict(dir='data/universe/otcexright', pat=r'^\d{4}-\d{2}-\d{2}\.csv$', srckeys=lambda b: [b[:10]])],
         desc='上櫃除權息事件，欄位比照上市封存（date, code, prev_close 前收, ref_price 除權息參考價, value 權值＋息值, type 權／息／權息）。'
              'type 已去掉「除」字以對齊上市封存寫法，原字在 kind_raw；open_base 開盤競價基準；source 上游取得管道（exDailyQ＝櫃買中心；finmind＝第三方補的早年資料）。'),
    dict(name='otc_reduce', kind='table', part=part_year, sort=EV_SORT, rowkey=k_date,
         cols={'date': D, 'code': S, 'prev_close': F, 'ref_price': F, 'reason': S, 'open_base': F, 'ex_ref_price': F, 'official_factor': F, 'source': S},
         rename={'stock_id': 'code', 'pre_close': 'prev_close'},
         src=[dict(dir='data/universe/otcreduce', pat=r'^\d{4}-\d{2}-\d{2}\.csv$', srckeys=lambda b: [b[:10]])],
         desc='上櫃減資恢復買賣：prev_close 停止買賣前收盤、ref_price 恢復買賣參考價、reason 減資原因、official_factor 官方換算係數（ref_price÷prev_close 的依據）。'),
    dict(name='otc_parvalue', kind='table', part=part_year, sort=EV_SORT, rowkey=k_date,
         cols={'date': D, 'code': S, 'prev_close': F, 'ref_price': F, 'reason': S, 'open_base': F, 'ex_ref_price': F, 'halt_date': S, 'derived_from': S},
         rename={'stock_id': 'code', 'pre_close': 'prev_close'},
         src=[dict(dir='data/universe/otcparvalue', pat=r'^\d{4}-\d{2}-\d{2}\.csv$', srckeys=lambda b: [b[:10]])],
         desc='上櫃面額變更：prev_close、ref_price（變更後參考價）。注意 derived_from=shares_ratio 表示上游是用股數比例推導，不是官方公告的參考價。'),
    dict(name='revenue_raw', kind='raw',
         src=[dict(dir='data/mops/revenue_hist', pat=r'\.csv$'), dict(dir='data/mops/revenue', pat=r'\.csv$'),
              dict(dir='data/early/revenue', pat=r'\.csv$'), dict(dir='data/meta/revenue_seen', pat=r'\.csv$')],
         desc='月營收彙總表的原檔快照（逐位元保存、gzip）。first/＝本快照第一次看到的版本（永不覆蓋）；rev/＝之後每次內容變動的版本（檔名帶該版 blob sha 前 12 碼）；'
              '_index.tsv 記每個來源檔的首見版、最新版與看到時的上游 commit。revenue_hist＝2015 起各月各市場彙總（千元）；revenue＝最近幾個月含出表日期的原始表；'
              'early/revenue＝2003～2014；meta/revenue_seen＝上游自己記的「各公司營收首次出現時點」分批檔。'),
]
BY_NAME = {d['name']: d for d in DATASETS}


# ───────────────────────── 更新主流程 ─────────────────────────
class State:
    """一個資料集的來源檔狀態：_sources.tsv 依來源檔名裡的年份分放在各年資料夾（沒有年份的放資料集根目錄）。"""

    def __init__(self, root):
        self.root = root
        self.src = {}          # path -> [sha, rows]
        if os.path.isdir(root):
            for dp, dn, fn in os.walk(root):
                if '_sources.tsv' in fn:
                    for r in read_tsv(os.path.join(dp, '_sources.tsv')):
                        self.src[r[0]] = [r[1], int(r[2])]

    @staticmethod
    def year_of(path):
        m = re.search(r'(?<!\d)((?:19|20)\d\d)', os.path.basename(path))
        return m.group(1) if m else ''

    def save(self):
        by = {}
        for p, (sha, rows) in self.src.items():
            by.setdefault(self.year_of(p), []).append((p, sha, rows))
        for y, rows in by.items():
            write_if_changed(os.path.join(self.root, y, '_sources.tsv'), tsv_bytes(['source_path', 'blob_sha', 'rows'], sorted(rows)))


def list_sources(repo, spec):
    """回傳 [(來源規格, 路徑, sha)]，依來源順序（前面的先處理，後面的同鍵會蓋掉前面的）。"""
    out = []
    for s in spec['src']:
        tree = ls_tree(repo, s['dir'])
        if not tree:
            raise RuntimeError('上游找不到 %s' % s['dir'])
        for p in sorted(tree):
            if re.search(s['pat'], os.path.basename(p)):
                out.append((s, p, tree[p]))
    return out


def parts_index(root):
    idx = {}
    if os.path.isdir(root):
        for y in sorted(os.listdir(root)):
            for r in read_tsv(os.path.join(root, y, '_parts.tsv')):
                idx[os.path.join(y, r[0])] = r
    return idx


def save_year_index(root, y, entries):
    """entries: {檔名: (rows, min_key, max_key, sha256, bytes)}（整個年資料夾的全部分檔）"""
    rows = [(f, *entries[f]) for f in sorted(entries)]
    write_if_changed(os.path.join(root, y, '_parts.tsv'), tsv_bytes(['file', 'rows', 'min_key', 'max_key', 'sha256', 'bytes'], rows))
    write_if_changed(os.path.join(root, y, 'SHA256SUMS'), ''.join('%s  %s\n' % (entries[f][3], f) for f in sorted(entries)).encode())


def update_table(repo, out, spec, head, opts, log):
    root = os.path.join(out, spec['name'])
    st = State(root)
    srcs = list_sources(repo, spec)
    todo = []
    for s, p, sha in srcs:
        old = st.src.get(p)
        if old is not None and old[0] == sha:
            continue
        if s.get('bulk') and old is not None and not opts.refresh_bulk:
            log('  （%s 上游內容有變，屬於只在首次建置讀的歷史大檔，未重抓；要重抓加 --refresh-bulk）' % p)
            continue
        todo.append((s, p, sha))
    if not todo:
        return dict(status='無新資料')
    # 依年份分批抓與處理，控制暫存與記憶體
    batches = {}
    for i, (s, p, sha) in enumerate(todo):
        batches.setdefault((spec['src'].index(s), State.year_of(p)), []).append((s, p, sha))
    pidx = parts_index(root)
    revisions = read_tsv(os.path.join(root, '_revisions.tsv'))
    n_new = n_rev = 0
    touched_years = set()
    whole_seen_parts = set()
    for bk in sorted(batches):
        items = batches[bk]
        fetch_blobs(repo, [sha for _, _, sha in items])
        tabs = []; repl = set(); newstate = {}
        for (s, p, sha), (_, body) in zip(items, iter_blobs(repo, [(p, sha) for _, p, sha in items])):
            t = (s.get('parse') or csv_table)(body, p, spec)
            t = widen(t, spec['cols'])
            ks = key_strings(t, spec)
            fk = s['srckeys'](os.path.basename(p)) if s.get('srckeys') else None
            if fk is not None:
                bad = set(ks) - set(fk)
                if bad:
                    raise ValueError('%s 內容的鍵 %s 與檔名不符' % (p, sorted(bad)[:3]))
                repl.update(fk)
            repl.update(ks)
            tabs.append((t, ks))
            newstate[p] = [sha, t.num_rows]
        # 依分檔分組
        bucket = {}
        for t, ks in tabs:
            if t.num_rows == 0:
                continue
            ka = np.array(ks, dtype=object)
            pk = np.array(['/'.join(spec['part'](k)) for k in ks], dtype=object)
            for u in np.unique(pk):
                bucket.setdefault(u, []).append(t.filter(pa.array(pk == u)))
        affected = set(bucket) | {'/'.join(spec['part'](k)) for k in repl}
        if spec.get('whole'):
            affected |= {os.path.splitext(k)[0].replace(os.sep, '/') for k in pidx}
        for part in sorted(affected):
            y, nm = part.split('/')
            f = os.path.join(root, y, nm + '.parquet')
            pieces = []
            if os.path.exists(f) and not spec.get('whole'):
                old = widen(pq.read_table(f), spec['cols'])
                keep = np.array([k not in repl for k in key_strings(old, spec)], dtype=bool)
                pieces.append(old.filter(pa.array(keep)))
            pieces += bucket.get(part, [])
            t = sort_table(pa.concat_tables(pieces), spec) if pieces else None
            rel = os.path.join(y, nm + '.parquet')
            touched_years.add(y)
            if t is None or t.num_rows == 0:
                if os.path.exists(f):
                    os.remove(f)
                pidx.pop(rel, None)
                continue
            data = parquet_bytes(t)
            write_if_changed(f, data)
            ks = key_strings(t, spec)
            pidx[rel] = [nm + '.parquet', t.num_rows, min(ks), max(ks), sha256_bytes(data), len(data)]
        for p, v in newstate.items():
            if p in st.src:
                n_rev += 1
                revisions.append([p, st.src[p][0], v[0], st.src[p][1], v[1], head[0], head[1]])
            else:
                n_new += 1
            st.src[p] = v
    for y in sorted(touched_years):
        ent = {os.path.basename(k): tuple(v[1:]) for k, v in pidx.items() if k.split(os.sep)[0] == y}
        if ent:
            save_year_index(root, y, ent)
        else:
            for fn in ('_parts.tsv', 'SHA256SUMS'):
                fp = os.path.join(root, y, fn)
                if os.path.exists(fp):
                    os.remove(fp)
    if n_rev:
        write_if_changed(os.path.join(root, '_revisions.tsv'),
                         tsv_bytes(['source_path', 'old_blob_sha', 'new_blob_sha', 'old_rows', 'new_rows', 'upstream_commit', 'upstream_commit_time'], revisions))
    st.save()   # 最後才記來源狀態：中途失敗的話下次會重做同一批（重做結果相同）
    return dict(status='已更新', new_files=n_new, revised_files=n_rev, last_change_commit=head[0], last_change_commit_time=head[1])


def update_raw(repo, out, spec, head, opts, log):
    """原檔快照：first/ 永不覆蓋；內容變了另存 rev/<檔名>.<sha12>.gz。_index.tsv：path, first_sha, first_commit, first_time, latest_sha, latest_commit, latest_time, n_versions"""
    root = os.path.join(out, spec['name'])
    idxf = os.path.join(root, '_index.tsv')
    idx = {r[0]: r for r in read_tsv(idxf)}
    srcs = list_sources(repo, spec)
    todo = [(s, p, sha) for s, p, sha in srcs if p not in idx or idx[p][4] != sha]
    if not todo:
        return dict(status='無新資料')
    fetch_blobs(repo, [sha for _, _, sha in todo])
    n_new = n_rev = 0
    for (s, p, sha), (_, body) in zip(todo, iter_blobs(repo, [(p, sha) for _, p, sha in todo])):
        rel = p[len('data/'):] if p.startswith('data/') else p
        if p not in idx:
            f = os.path.join(root, 'first', rel + '.gz')
            if os.path.exists(f):
                with gzip.open(f, 'rb') as g:
                    if g.read() != body:
                        raise RuntimeError('%s 已存在但內容不同，且索引沒有它的紀錄；為了不覆蓋首見版，停止' % f)
            else:
                write_if_changed(f, gz_bytes(body))
            idx[p] = [p, sha, head[0], head[1], sha, head[0], head[1], '1']
            n_new += 1
        else:
            write_if_changed(os.path.join(root, 'rev', rel + '.' + sha[:12] + '.gz'), gz_bytes(body))
            r = idx[p]
            idx[p] = [p, r[1], r[2], r[3], sha, head[0], head[1], str(int(r[7]) + 1)]
            n_rev += 1
    write_if_changed(idxf, tsv_bytes(['source_path', 'first_blob_sha', 'first_seen_upstream_commit', 'first_seen_commit_time', 'latest_blob_sha',
                                      'latest_upstream_commit', 'latest_commit_time', 'n_versions'], [idx[k] for k in sorted(idx)]))
    return dict(status='已更新', new_files=n_new, revised_files=n_rev, last_change_commit=head[0], last_change_commit_time=head[1])


def dataset_summary(out, spec, cal):
    root = os.path.join(out, spec['name'])
    s = dict(description=spec['desc'], source_dirs=[x['dir'] for x in spec['src']])
    if spec['kind'] == 'raw':
        idx = read_tsv(os.path.join(root, '_index.tsv'))
        nb = nf = 0
        for dp, dn, fn in os.walk(root):
            for f in fn:
                if f.endswith('.gz'):
                    nf += 1; nb += os.path.getsize(os.path.join(dp, f))
        s.update(source_files=len(idx), files_with_revisions=sum(1 for r in idx if r[1] != r[4]), stored_files=nf, bytes=nb,
                 layout='first/<來源路徑去掉 data/>.gz；rev/<同上>.<blob sha 前 12 碼>.gz；_index.tsv')
        return s
    pidx = parts_index(root)
    st = State(root)
    s.update(columns=spec['cols'], rows=sum(int(v[1]) for v in pidx.values()), files=len(pidx), bytes=sum(int(v[5]) for v in pidx.values()),
             max_file_bytes=max([int(v[5]) for v in pidx.values()] or [0]),
             first_key=min([v[2] for v in pidx.values()] or ['']), last_key=max([v[3] for v in pidx.values()] or ['']), source_files=len(st.src),
             layout='<年>/<分檔>.parquet；每年資料夾有 SHA256SUMS、_parts.tsv、_sources.tsv', revised_source_files=len(read_tsv(os.path.join(root, '_revisions.tsv'))))
    if spec.get('calendar') or spec.get('weekly'):
        days = sorted(os.path.basename(p)[:10] for p, v in st.src.items() if re.match(r'^\d{4}-\d{2}-\d{2}\.csv$', os.path.basename(p)))
        s['empty_source_days'] = sorted(os.path.basename(p)[:10] for p, v in st.src.items() if v[1] == 0 and re.match(r'^\d{4}-\d{2}-\d{2}\.csv$', os.path.basename(p)))
        if spec.get('calendar') and cal and days:
            lo = max(days[0], spec.get('calendar_from', '0'))
            ds = set(days)
            miss = [d for d in cal if lo <= d <= cal[-1] and d not in ds]
            s['missing_days_vs_daily_calendar'] = miss
            s['days'] = len(days)
    return s


def write_manifest(out, results):
    cal = sorted(os.path.basename(p)[:10] for p in State(os.path.join(out, 'daily')).src)
    mf = os.path.join(out, 'manifest.json')
    old = json.load(open(mf, encoding='utf-8')) if os.path.exists(mf) else {}
    man = dict(what='官方資料倉庫 tw-stock-data 的自有快照；由 update_daily.py 建置與每日增量更新', upstream_repo=URL,
               update_command='python3 -I -B research/tw_stock_selection/raw/tsd_extra/update_daily.py <官方倉庫 clone 路徑>',
               writer='pyarrow %s；parquet zstd 第 19 級；整數欄 int32（放不下才 int64）' % pa.__version__, datasets={})
    for spec in DATASETS:
        if not os.path.isdir(os.path.join(out, spec['name'])):
            continue
        s = dataset_summary(out, spec, cal)
        prev = old.get('datasets', {}).get(spec['name'], {})
        r = results.get(spec['name'], {})
        for k in ('last_change_commit', 'last_change_commit_time'):
            s[k] = r.get(k) or prev.get(k)
        man['datasets'][spec['name']] = s
    man['total_bytes'] = sum(d.get('bytes', 0) for d in man['datasets'].values())
    write_if_changed(mf, (json.dumps(man, ensure_ascii=False, indent=1, sort_keys=True) + '\n').encode('utf-8'))


def verify(out):
    bad = 0; n = 0
    for spec in DATASETS:
        root = os.path.join(out, spec['name'])
        if spec['kind'] != 'table' or not os.path.isdir(root):
            continue
        for y in sorted(os.listdir(root)):
            sf = os.path.join(root, y, 'SHA256SUMS')
            if not os.path.exists(sf):
                continue
            listed = set()
            for ln in open(sf, encoding='utf-8'):
                h, f = ln.rstrip('\n').split('  ', 1); listed.add(f); n += 1
                fp = os.path.join(root, y, f)
                if not os.path.exists(fp) or sha256_bytes(open(fp, 'rb').read()) != h:
                    bad += 1; print('不符：', fp)
            for f in os.listdir(os.path.join(root, y)):
                if f.endswith('.parquet') and f not in listed:
                    bad += 1; print('沒列在 SHA256SUMS：', os.path.join(root, y, f))
    for spec in DATASETS:      # 原檔快照：解壓後算 git blob sha，要等於索引記的首見版 sha／檔名裡的 sha
        root = os.path.join(out, spec['name'])
        if spec['kind'] != 'raw' or not os.path.isdir(root):
            continue
        for r in read_tsv(os.path.join(root, '_index.tsv')):
            rel = r[0][len('data/'):] if r[0].startswith('data/') else r[0]
            chk = [(os.path.join(root, 'first', rel + '.gz'), r[1])]
            if r[4] != r[1]:
                chk.append((os.path.join(root, 'rev', rel + '.' + r[4][:12] + '.gz'), r[4]))
            for fp, sha in chk:
                n += 1
                try:
                    body = gzip.open(fp, 'rb').read()
                    ok = hashlib.sha1(b'blob %d\0' % len(body) + body).hexdigest() == sha
                except OSError:
                    ok = False
                if not ok:
                    bad += 1; print('不符：', fp)
    print('檢查 %d 個檔，%d 個不符' % (n, bad))
    return 1 if bad else 0


def truncate(out, n):
    """測試用：把每日／每週面板截掉最後 n 個日期（面板列、來源狀態、索引一起退回），之後跑更新應補回一模一樣的檔。"""
    for spec in DATASETS:
        root = os.path.join(out, spec['name'])
        if spec['kind'] != 'table' or not os.path.isdir(root) or spec['rowkey'] is not k_date:
            continue
        st = State(root)
        dated = sorted({os.path.basename(p)[:10] for p in st.src if re.match(r'^\d{4}-\d{2}-\d{2}\.csv$', os.path.basename(p))})
        cut = set(dated[-n:])
        if not cut:
            continue
        pidx = parts_index(root)
        years = set()
        for rel in sorted(pidx):
            v = pidx[rel]
            if v[3] < min(cut):
                continue
            f = os.path.join(root, rel); y = rel.split(os.sep)[0]; years.add(y)
            t = widen(pq.read_table(f), spec['cols'])
            keep = np.array([k not in cut for k in key_strings(t, spec)], dtype=bool)
            t = t.filter(pa.array(keep))
            if t.num_rows == 0:
                os.remove(f); pidx.pop(rel); continue
            data = parquet_bytes(sort_table(t, spec)); write_if_changed(f, data)
            ks = key_strings(t, spec)
            pidx[rel] = [v[0], t.num_rows, min(ks), max(ks), sha256_bytes(data), len(data)]
        for p in [p for p in st.src if os.path.basename(p)[:10] in cut and re.match(r'^\d{4}-\d{2}-\d{2}\.csv$', os.path.basename(p))]:
            years.add(State.year_of(p)); del st.src[p]
        for y in years:
            ent = {os.path.basename(k): tuple(v[1:]) for k, v in pidx.items() if k.split(os.sep)[0] == y}
            if ent:
                save_year_index(root, y, ent)
            sf = os.path.join(root, y, '_sources.tsv')
            if os.path.exists(sf):
                os.remove(sf)
        st.save()
        print('%s：截掉 %s' % (spec['name'], sorted(cut)))


def main():
    ap = argparse.ArgumentParser(description='官方資料倉庫 → 自有快照：建置與每日增量更新')
    ap.add_argument('repo', nargs='?')
    ap.add_argument('--out', default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--only', default='')
    ap.add_argument('--refresh-bulk', action='store_true')
    ap.add_argument('--allow-older', action='store_true', help='clone 比面板舊也照做（只在刻意回溯時用）')
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--truncate', type=int, default=0)
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    if a.verify:
        return verify(out)
    if a.truncate:
        if out == os.path.dirname(os.path.abspath(__file__)):
            print('--truncate 只准對複本做（請用 --out 指到複本）'); return 2
        truncate(out, a.truncate); write_manifest(out, {}); return 0
    if not a.repo:
        ap.print_usage(); return 2
    only = [x for x in a.only.split(',') if x]
    unknown = [x for x in only if x not in BY_NAME]
    if unknown:
        print('沒有這個資料集：%s；可用：%s' % (unknown, ', '.join(BY_NAME))); return 2
    log = (lambda *x: None) if a.quiet else print
    try:
        head = repo_head(a.repo)
    except Exception as e:
        print('讀不到官方倉庫 %s：%s' % (a.repo, e)); return 1
    # 防呆：clone 的 HEAD 比面板上次更新時的上游版本還舊 → 不做（否則會把面板「更新」回舊內容）
    mf = os.path.join(out, 'manifest.json')
    if os.path.exists(mf) and not a.allow_older:
        try:
            seen = [d.get('last_change_commit_time') or '' for d in json.load(open(mf, encoding='utf-8')).get('datasets', {}).values()]
            newest = max([pd.Timestamp(x) for x in seen if x] or [pd.Timestamp('1970-01-01T00:00:00+00:00')])
            if pd.Timestamp(head[1]) < newest:
                print('結果：無新資料——官方倉庫 clone 的版本（%s）比面板上次更新用的版本（%s）舊，未更新；請先更新 clone' % (head[1], newest.isoformat()))
                return 0
        except Exception as e:
            print('manifest.json 讀不了（%s），略過版本新舊檢查' % e)
    results = {}; failed = {}
    for spec in DATASETS:
        if only and spec['name'] not in only:
            continue
        if not only and spec.get('enabled') is False:      # 還沒建置的資料集：不指名就不做（免得每日排程突然去抓歷史大檔）
            continue
        try:
            r = (update_raw if spec['kind'] == 'raw' else update_table)(a.repo, out, spec, head, a, log)
            results[spec['name']] = r
            if r['status'] == '無新資料':
                log('%s：無新資料' % spec['name'])
            else:
                log('%s：已更新（新檔 %d、內容變動的舊檔 %d）' % (spec['name'], r['new_files'], r['revised_files']))
        except Exception as e:
            failed[spec['name']] = '%s: %s' % (type(e).__name__, e)
            print('%s：失敗——%s' % (spec['name'], failed[spec['name']]))
            if not a.quiet:
                traceback.print_exc(limit=3)
    try:
        write_manifest(out, results)
    except Exception as e:
        failed['manifest'] = '%s: %s' % (type(e).__name__, e); print('manifest：失敗——%s' % failed['manifest'])
    changed = [k for k, v in results.items() if v['status'] != '無新資料']
    if failed:
        print('結果：%d 類失敗（%s）；%d 類已更新；上游 commit %s' % (len(failed), '、'.join(failed), len(changed), head[0][:12]))
        return 1
    print('結果：%s；上游 commit %s（%s）' % ('已更新 ' + '、'.join(changed) if changed else '無新資料', head[0][:12], head[1]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
