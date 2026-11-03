#!/usr/bin/env python3
"""前瞻追蹤資料層（o_forward_infra_live）：把來源資料截斷到 asof，寫成 main_picks_today.py 可直接讀的資料夾，並以斷言檢查。

截斷規則（A1，事前寫定）
- history.csv.gz、exdiv_5y、attention_5y、oddlot_hist、warnings_hist：date ≤ asof。
- 月營收（rev_5y、rev_hist）：保留 ann_eff ≤ rev_cutoff 的列（rev_cutoff 預設＝asof；
  bugfix 2026-11-03：官方日線改用 rev_cutoff＝成交日，即「成交日開盤前已抓到」的營收，見 bugfix_log.md BF-1）。ann_eff＝rev_hist 的 ann_date；缺值時＝法定期限日
  （ym 次月 10 日起第一個交易日，交易日依 history 日曆，超出日曆時用平日扣 tw_holidays.json）。
  J0 線另外把「m−1 月 ann_date 缺值」視為未公告（make_forward.score_j），不在這裡處理。
- 處置名單：disposal_5y ∪ warnings_hist（flag＝處置）去重（code、start、end，保留最早公告日），公告日 date ≤ asof；
  寫成 extras/disposal_5y.csv.gz（disposal_5y 在來源停在 2026-10-05，warnings_hist 每日更新，2026-06-22 起）。
- 研究面板（history 不足 252 日時拼接用）：panel_pre＝[2025-01-01, history 起日) 子集，所有 asof 共用（最大日期 < history 起日 ≤ asof）。
- oddlot_hist 不需要，不寫（省磁碟）。
- 0050 含息、加權指數：date ≤ asof。
- stock_list.csv 原樣複製（名稱、市場、產業為現況，不是 PIT；已知限制）。
- 官方月營收補值（選用；bugfix 2026-11-03 BF-5，見 bugfix_log.md）：Source(official=<raw/official_rev>) 時，
  讀 rev_official_fetch.py 的抓取目錄（驗 sha256），在「上游截斷後仍缺」的 (code, ym) 用官方值補進 rev_hist：
  · 可得時點＝抓取完成時間 fetched_ts（不得往回套）。使用期限 t_off：
      rev_cutoff > asof（官方日線，截到成交日）→ t_off＝成交日 09:00（台北），抓取完成須早於開盤；
      否則 → t_off＝rev_cutoff 隔天 00:00（抓取日期 ≤ rev_cutoff，與 rev_hist ann_date 的「日」粒度一致）；
      另外不晚於 official_clock（make_forward 設成執行時鐘／重播 sim-clock）。
  · 補進的列 ann_date＝最早一次抓到的日期（台北），數值＝t_off 前最後一次抓取的值。
  · 只補上游有營收歷史的股票（ym 前 3 個月內上游至少有一筆），不把上游從來沒有的股票（例如 KY 公司）帶進股池。
  · 官方值只補缺，不覆蓋上游已有的列；沒有補到任何列時，截斷資料夾與 MANIFEST 和不開這個選項時完全相同（A4）。
gzip 輸出 mtime＝0，同一來源、同一 asof 重建的檔案 sha256 相同（A4）。
"""
import gzip, hashlib, io, json, shutil
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / 'raw/screener/data'
E50_SRC = HERE / 'raw/finlab/etf_0050.csv'
TAIEX_SRC = HERE / 'raw/finlab/taiex_twii.csv'
PANEL_SRC = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
PANEL_FILES = ['close_raw', 'open', 'high', 'value', 'ret', 'adj']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def write_csv(df, path, gz=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    b = df.to_csv(index=False, float_format='%.10g', lineterminator='\n').encode('utf-8')
    gz = str(path).endswith('.gz') if gz is None else gz
    if gz:
        bio = io.BytesIO()
        with gzip.GzipFile(fileobj=bio, mode='wb', mtime=0, compresslevel=6, filename='') as fh:
            fh.write(b)
        b = bio.getvalue()
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_bytes(b); tmp.replace(path)
    return sha_bytes(b)


class Source:
    """來源資料（只讀一次）。"""
    def __init__(self, src=SRC, e50=E50_SRC, taiex=TAIEX_SRC, panel=PANEL_SRC, official=None):
        self.src, self.panel = Path(src), Path(panel)
        D = self.src
        self.hist = pd.read_csv(D / 'history.csv.gz', dtype={'code': str}, parse_dates=['date'])
        self.stock_list_path = D / 'stock_list.csv'
        self.lst = pd.read_csv(self.stock_list_path, dtype=str)
        self.r5 = pd.read_csv(D / 'extras/rev_5y.csv.gz', dtype={'code': str})
        self.rh = pd.read_csv(D / 'extras/rev_hist.csv.gz', dtype={'code': str})
        self.d5 = pd.read_csv(D / 'extras/disposal_5y.csv.gz', dtype={'code': str})
        wp = D / 'extras/warnings_hist.csv.gz'
        self.wh = pd.read_csv(wp, dtype={'code': str}) if wp.exists() else pd.DataFrame(columns=['date', 'code', 'flag', 'start', 'end'])
        self.ex = pd.read_csv(D / 'extras/exdiv_5y.csv.gz', dtype={'code': str})
        self.att = pd.read_csv(D / 'extras/attention_5y.csv.gz', dtype={'code': str})
        op = D / 'extras/oddlot_hist.csv.gz'
        self.odd = pd.read_csv(op, dtype={'code': str}) if op.exists() else None
        self.e50 = pd.read_csv(e50, parse_dates=['date'])
        self.taiex = pd.read_csv(taiex, parse_dates=['date'])
        hp = D / 'extras/tw_holidays.json'
        self.holidays = set()
        if hp.exists():
            for y, v in json.loads(hp.read_text()).get('years', {}).items():
                self.holidays |= set(pd.to_datetime(v))
        self.hcal = pd.DatetimeIndex(sorted(self.hist['date'].unique()))
        self.files = {k: sha(D / k) for k in ['history.csv.gz', 'stock_list.csv', 'extras/rev_5y.csv.gz', 'extras/rev_hist.csv.gz',
                                              'extras/disposal_5y.csv.gz', 'extras/exdiv_5y.csv.gz', 'extras/attention_5y.csv.gz']
                      if (D / k).exists()}
        if wp.exists(): self.files['extras/warnings_hist.csv.gz'] = sha(wp)
        self.files['etf_0050.csv'] = sha(e50); self.files['taiex_twii.csv'] = sha(taiex)
        self._ann = None
        self._disp = None
        self.official_root = Path(official) if official else None
        self.official_clock = None          # make_forward 設定：官方補值不得晚於這個時鐘（重播用 sim-clock）
        self.off, self.off_info = load_official(self.official_root) if self.official_root else (None, None)

    # ---- 日曆 ----
    def is_trading(self, d):
        d = pd.Timestamp(d)
        if d <= self.hcal[-1]: return d in self.hcal
        return d.weekday() < 5 and d not in self.holidays

    def next_trading(self, d):
        d = pd.Timestamp(d) + pd.Timedelta(days=1)
        while not self.is_trading(d): d += pd.Timedelta(days=1)
        return d

    def deadline(self, ym):
        """ym（'YYYY-MM'）營收的法定期限日：次月 10 日起第一個交易日。"""
        p = pd.Period(ym, 'M') + 1
        d = pd.Timestamp(year=p.year, month=p.month, day=10)
        while not self.is_trading(d): d += pd.Timedelta(days=1)
        return d

    # ---- 營收公告日 ----
    def ann_eff(self):
        if self._ann is None:
            yms = sorted(set(self.r5['ym']) | set(self.rh['ym']))
            dl = {y: self.deadline(y) for y in yms}
            a = self.rh[['code', 'ym', 'ann_date']].copy()
            a['ann_date'] = pd.to_datetime(a['ann_date'], errors='coerce')
            a = a.dropna(subset=['ann_date']).drop_duplicates(['code', 'ym'], keep='first')
            self._ann = (dict(zip(zip(a.code, a.ym), a.ann_date)), dl)
        return self._ann

    def _rev_trunc(self, df, asof):
        ann, dl = self.ann_eff()
        eff = [ann.get((c, y), dl[y]) for c, y in zip(df['code'], df['ym'])]
        return df[np.array([e <= asof for e in eff], bool)]

    # ---- 官方月營收補值（BF-5） ----
    def official_t_off(self, asof, rc):
        asof, rc = pd.Timestamp(asof), pd.Timestamp(rc)
        t = (rc + pd.Timedelta(hours=9)) if rc > asof else (rc + pd.Timedelta(days=1))
        t = t.tz_localize(TZ)
        if self.official_clock is not None: t = min(t, pd.Timestamp(self.official_clock).tz_convert(TZ))
        return t

    def official_fill(self, r5t, rht, asof, rc):
        """回傳 (要補進 rev_hist 的列, 摘要)。r5t、rht＝已截斷到 rc 的上游營收。沒有官方資料時回傳 (空表, None)。"""
        if self.off is None or not len(self.off): return pd.DataFrame(columns=OFF_COLS), None
        return official_fill_rows(self.off, r5t, rht, self.official_t_off(asof, rc), str(pd.Period(rc, 'M') - 1))

    def rev_present(self, ym, cut_ts):
        """cut_ts（台北）時點已存在 ym 月營收列的股票：上游 ann_eff（日）≤ cut_ts 的日期，加官方抓取完成 < cut_ts。"""
        cut_ts = pd.Timestamp(cut_ts).tz_convert(TZ)
        d = pd.Timestamp(cut_ts.date())
        ann, dl = self.ann_eff()
        have = set()
        for df in (self.r5, self.rh):
            x = df[df['ym'] == ym]
            have |= {c for c in x['code'] if ann.get((c, ym), dl.get(ym, self.deadline(ym))) <= d}
        off = set()
        if self.off is not None and len(self.off):
            o = self.off[(self.off['ym'] == ym) & (self.off['fetched_ts'] < cut_ts)]
            off = set(o['code'])
        return have, off

    # ---- 處置 ----
    def disposal(self):
        if self._disp is None:
            w = self.wh[self.wh['flag'] == '處置'][['date', 'code', 'flag', 'start', 'end']]
            d = pd.concat([self.d5[['date', 'code', 'flag', 'start', 'end']].assign(_src=0), w.assign(_src=1)], ignore_index=True)
            d = d.dropna(subset=['start', 'end'])
            d = d.sort_values(['code', 'start', 'end', 'date', '_src']).drop_duplicates(['code', 'start', 'end'], keep='first')
            self._disp = d.sort_values(['date', 'code', 'start']).reset_index(drop=True)
        return self._disp

    # ---- 面板（只取 history 起日之前；所有 asof 共用） ----
    def panel_pre(self, work, first='2025-01-01'):
        """研究面板 [first, history 起日) 的子集：拼接時 52 週窗（252 日）與 60 日窗所需，且最大日期 < history 起日 ≤ 任何 asof。"""
        out = Path(work) / 'panel_pre'
        hs = self.hcal[0]
        key = dict(hist_start=str(hs.date()), first=first, src={f: (self.panel / f'{f}.parquet').stat().st_size for f in PANEL_FILES
                                                                if (self.panel / f'{f}.parquet').exists()})
        mp = out / 'MANIFEST.json'
        if mp.exists() and json.loads(mp.read_text()).get('key') == key: return out
        if not (self.panel / 'close_raw.parquet').exists(): return out
        if out.exists(): shutil.rmtree(out)
        out.mkdir(parents=True)
        mx = {}
        for f in PANEL_FILES:
            p = pd.read_parquet(self.panel / f'{f}.parquet'); p = p.loc[(p.index >= first) & (p.index < hs)]
            p.to_parquet(out / f'{f}.parquet'); mx[f] = str(p.index.max().date())
        mp.write_text(json.dumps(dict(key=key, max_dates=mx), ensure_ascii=False, indent=1))
        return out

    # ---- 截斷資料夾 ----
    def build_asof(self, asof, out_dir, rev_cutoff=None):
        """寫出截斷到 asof 的資料夾；回傳 manifest。已存在而且來源 sha 相同時直接沿用。
        rev_cutoff：月營收的截斷日（預設＝asof）；價格、處置等其他資料一律截斷到 asof。"""
        asof = pd.Timestamp(asof); out = Path(out_dir)
        rc = asof if rev_cutoff is None else pd.Timestamp(rev_cutoff)
        man_p = out / 'MANIFEST.json'
        key = dict(asof=str(asof.date()), source=self.files)
        if rc != asof: key['rev_cutoff'] = str(rc.date())
        r5t, rh = self._rev_trunc(self.r5, rc), self._rev_trunc(self.rh, rc)
        add, off_info = self.official_fill(r5t, rh, asof, rc)
        if len(add):                                     # BF-5：有補到列才改 key（沒補到＝與不開選項完全相同）
            key['official_fill'] = dict(sha256=off_info['filled_sha256'], t_off=off_info['t_off'])
            rh = pd.concat([rh, add.astype({'ym': str})], ignore_index=True)
        if man_p.exists():
            m = json.loads(man_p.read_text())
            if m.get('key') == key: return m
        if out.exists(): shutil.rmtree(out)
        (out / 'extras').mkdir(parents=True)
        ds = lambda s: pd.to_datetime(s).dt.date.astype(str)
        F = {}
        h = self.hist[self.hist['date'] <= asof].copy(); h['date'] = h['date'].dt.strftime('%Y-%m-%d')
        F['history.csv.gz'] = write_csv(h, out / 'history.csv.gz')
        shutil.copyfile(self.stock_list_path, out / 'stock_list.csv'); F['stock_list.csv'] = sha(out / 'stock_list.csv')
        F['extras/rev_5y.csv.gz'] = write_csv(r5t, out / 'extras/rev_5y.csv.gz')
        F['extras/rev_hist.csv.gz'] = write_csv(rh, out / 'extras/rev_hist.csv.gz')
        dp = self.disposal(); dp = dp[pd.to_datetime(dp['date']) <= asof].drop(columns='_src')
        F['extras/disposal_5y.csv.gz'] = write_csv(dp, out / 'extras/disposal_5y.csv.gz')
        for name, df in [('exdiv_5y', self.ex), ('attention_5y', self.att), ('warnings_hist', self.wh)]:
            F[f'extras/{name}.csv.gz'] = write_csv(df[pd.to_datetime(df['date']) <= asof], out / f'extras/{name}.csv.gz')
        e = self.e50[self.e50['date'] <= asof].copy(); e['date'] = ds(e['date'])
        F['etf_0050.csv'] = write_csv(e, out / 'etf_0050.csv')
        t = self.taiex[self.taiex['date'] <= asof].copy(); t['date'] = ds(t['date'])
        F['taiex_twii.csv'] = write_csv(t, out / 'taiex_twii.csv')
        n_hist = int((self.hcal <= asof).sum())
        with_panel = False
        m = dict(key=key, files=F, n_hist_days=n_hist, panel=bool(with_panel))
        if off_info is not None and len(add): m['official_fill'] = off_info
        man_p.write_text(json.dumps(m, ensure_ascii=False, indent=1, sort_keys=True))
        return m


# ---------------- 官方月營收抓取目錄（BF-5） ----------------
TZ = 'Asia/Taipei'


OFF_COLS = ['code', 'ym', 'revenue', 'yoy', 'mom', 'ann_date']


def official_fill_rows(off, r5t, rht, t_off, max_ym):
    """官方補值（BF-5）：off＝load_official 的表；r5t、rht＝上游營收（已截斷）；只用抓取完成 < t_off、ym ≤ max_ym 的官方列，
    補上游缺的 (code, ym)，而且只補上游 ym 前 3 個月內有營收的股票。回傳 (補進 rev_hist 的列, 摘要)。"""
    cols = OFF_COLS
    t_off = pd.Timestamp(t_off)
    o = off[(off['fetched_ts'] < t_off) & (off['ym'] <= max_ym)]
    info = dict(t_off=t_off.isoformat(), n_official_rows=int(len(o)), fetch_dirs=sorted(set(o['fetch_dir'])), n_filled=0)
    if not len(o): return pd.DataFrame(columns=cols), info
    have = set(zip(r5t['code'], r5t['ym'])) | set(zip(rht['code'], rht['ym']))
    up = pd.concat([r5t[['code', 'ym']], rht[['code', 'ym']]]).drop_duplicates()
    up_p = {}
    for c, y in zip(up['code'], up['ym']): up_p.setdefault(c, set()).add(pd.Period(y, 'M'))
    o = o.sort_values(['code', 'ym', 'fetched_ts'])
    g = o.groupby(['code', 'ym'], sort=True)
    first, last = g['fetched_ts'].min(), g.tail(1).set_index(['code', 'ym'])
    rows, no_hist = [], 0
    for (c, y), r in last.iterrows():
        if (c, y) in have: continue
        p = pd.Period(y, 'M')
        if not (up_p.get(c, set()) & {p - 1, p - 2, p - 3}):
            no_hist += 1; continue                        # 上游沒有營收歷史：不擴大股池
        rows.append(dict(code=c, ym=y, revenue=r['revenue'], yoy=r['yoy'], mom=r['mom'],
                         ann_date=first[(c, y)].tz_convert(TZ).strftime('%Y-%m-%d')))
    add = pd.DataFrame(rows, columns=cols)
    if len(add):
        mx = max(first[(c, y)] for c, y in zip(add['code'], add['ym']))
        assert mx < t_off, f'A1 失敗：官方補值抓取時間 {mx} 不早於使用期限 {t_off}'
        info.update(max_fetched_ts=mx.isoformat())
    info.update(n_filled=int(len(add)), n_skipped_no_upstream_history=int(no_hist), filled_by_ym={k: int(v) for k, v in add.groupby('ym').size().items()} if len(add) else {},
                filled_sample=sorted(add['code'].tolist())[:40], filled_sha256=sha_bytes(add.to_csv(index=False).encode()) if len(add) else None)
    return add, info


def load_official(root):
    """讀 root 底下所有 rev_official_fetch.py 抓取目錄（sha256 不符的整個目錄不用）。
    回傳 (DataFrame[code, ym, revenue, yoy, mom, fetched_ts（台北 tz）, source, fetch_dir], 各目錄摘要)。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location('rev_official_fetch', HERE / 'rev_official_fetch.py')
    ROF = importlib.util.module_from_spec(spec); spec.loader.exec_module(ROF)
    cols = ['code', 'ym', 'revenue', 'yoy', 'mom', 'fetched_ts', 'source', 'fetch_dir']
    rows, info = [], []
    root = Path(root)
    for d in sorted(p for p in root.glob('*') if p.is_dir() and (p / 'FETCH.json').is_file()) if root.exists() else []:
        try:
            rr, meta = ROF.parse_dir(d)
        except Exception as e:
            info.append(dict(dir=d.name, error=f'{type(e).__name__}: {e}')); continue
        info.append(dict(dir=d.name, n_rows=len(rr), rejected=bool(meta.get('rejected')), fetch_json_sha256=sha(d / 'FETCH.json'),
                         errors=meta.get('parse_errors') or None))
        rows += rr
    df = pd.DataFrame(rows, columns=cols)
    if len(df):
        df['fetched_ts'] = pd.to_datetime(df['fetched_ts'], utc=True).dt.tz_convert(TZ)
        for c in ('revenue', 'yoy', 'mom'): df[c] = pd.to_numeric(df[c], errors='coerce')
    return df, info


# ---------------- 斷言（A1） ----------------
def max_date(obj):
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        if isinstance(obj.index, pd.DatetimeIndex): return obj.index.max()
        for c in ('date', 'ann_date'):
            if c in getattr(obj, 'columns', []): return pd.to_datetime(obj[c], errors='coerce').max()
    return None


def assert_asof(asof, **frames):
    """每個資料框的最大日期都要 ≤ asof；回傳 {名稱: 最大日期}。"""
    asof = pd.Timestamp(asof); rep = {}
    for k, v in frames.items():
        m = max_date(v)
        rep[k] = None if m is None or pd.isna(m) else str(pd.Timestamp(m).date())
        assert m is None or pd.isna(m) or pd.Timestamp(m) <= asof, f'A1 失敗：{k} 最大日期 {m} > asof {asof.date()}'
    return rep


def assert_dir_asof(d, asof, rev_cutoff=None):
    """讀回截斷資料夾的每個檔案，斷言最大日期 ≤ asof（營收另斷言 ym ≤ rev_cutoff 前一月、ann_date ≤ rev_cutoff；rev_cutoff 預設＝asof）。"""
    d = Path(d); asof = pd.Timestamp(asof); rep = {}
    rc = asof if rev_cutoff is None else pd.Timestamp(rev_cutoff)
    for f in ['history.csv.gz', 'extras/disposal_5y.csv.gz', 'extras/exdiv_5y.csv.gz', 'extras/attention_5y.csv.gz',
              'extras/warnings_hist.csv.gz', 'extras/oddlot_hist.csv.gz', 'etf_0050.csv', 'taiex_twii.csv']:
        if (d / f).exists():
            x = pd.read_csv(d / f, usecols=['date'], parse_dates=['date'])
            rep[f] = assert_asof(asof, **{f: x})[f]
    for f in ['extras/rev_5y.csv.gz', 'extras/rev_hist.csv.gz']:
        x = pd.read_csv(d / f, dtype={'code': str})
        mx = pd.Period(x['ym'].max(), 'M')
        assert mx <= pd.Period(rc, 'M') - 1, f'A1 失敗：{f} 最新營收月份 {mx} 不早於營收截斷日 {rc.date()} 的月份'
        if 'ann_date' in x.columns: assert_asof(rc, **{f: x[['ann_date']]})
        rep[f] = f'ym≤{mx}'
    return rep


def assert_panel_asof(pdir, asof):
    rep = {}
    for f in PANEL_FILES:
        if (Path(pdir) / f'{f}.parquet').exists():
            p = pd.read_parquet(Path(pdir) / f'{f}.parquet', columns=[])
            rep[f'panel/{f}'] = assert_asof(asof, **{f: p})[f]
    return rep


# ---------------- 實際時鐘推算「應有的最後交易日」（bugfix 2026-11-03 BF-2） ----------------
READY_HOUR = 18     # 上游「每日資料」約 15:25、17:40 提交；台北 18:00 以後才要求當天收盤資料已在來源中


def expected_last_day(now, is_trading, ready_hour=READY_HOUR):
    """依台北時鐘 now 推算來源資料「應該已經有」的最後交易日：
    now 當天是交易日而且已過 ready_hour → 當天；否則 → now 之前最近的交易日。is_trading(d) 為交易日判斷（history 日曆＋tw_holidays.json）。"""
    now = pd.Timestamp(now)
    if now.tzinfo is not None: now = now.tz_convert('Asia/Taipei').tz_localize(None)
    d = now.normalize()
    if not (is_trading(d) and now.hour >= ready_hour):
        d -= pd.Timedelta(days=1)
        while not is_trading(d): d -= pd.Timedelta(days=1)
    return d


def fill_dir_official(D, out_rev_hist, root, t_cut):
    """fallback／check_official 用：資料夾 D（未截斷的上游快照）的 rev_hist 加上官方補值（抓取完成 < t_cut），寫到 out_rev_hist。
    回傳摘要（沒有可用官方資料時 n_filled＝0，仍照原樣寫出 rev_hist）。"""
    D = Path(D)
    r5 = pd.read_csv(D / 'extras/rev_5y.csv.gz', dtype={'code': str})
    rh = pd.read_csv(D / 'extras/rev_hist.csv.gz', dtype={'code': str})
    off, dirs = load_official(root)
    t_cut = pd.Timestamp(t_cut)
    add, info = official_fill_rows(off, r5, rh, t_cut, str(pd.Period(t_cut.tz_convert(TZ).tz_localize(None), 'M') - 1)) if len(off) \
        else (pd.DataFrame(columns=OFF_COLS), dict(t_off=t_cut.isoformat(), n_official_rows=0, fetch_dirs=[], n_filled=0))
    out = pd.concat([rh, add], ignore_index=True) if len(add) else rh
    info['rev_hist_sha256'] = write_csv(out, out_rev_hist)
    info['official_dirs'] = dirs
    return info
