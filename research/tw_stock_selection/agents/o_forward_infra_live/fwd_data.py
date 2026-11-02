#!/usr/bin/env python3
"""前瞻追蹤資料層（o_forward_infra_live）：把來源資料截斷到 asof，寫成 main_picks_today.py 可直接讀的資料夾，並以斷言檢查。

截斷規則（A1，事前寫定）
- history.csv.gz、exdiv_5y、attention_5y、oddlot_hist、warnings_hist：date ≤ asof。
- 月營收（rev_5y、rev_hist）：保留 ann_eff ≤ asof 的列。ann_eff＝rev_hist 的 ann_date；缺值時＝法定期限日
  （ym 次月 10 日起第一個交易日，交易日依 history 日曆，超出日曆時用平日扣 tw_holidays.json）。
  J0 線另外把「m−1 月 ann_date 缺值」視為未公告（make_forward.score_j），不在這裡處理。
- 處置名單：disposal_5y ∪ warnings_hist（flag＝處置）去重（code、start、end，保留最早公告日），公告日 date ≤ asof；
  寫成 extras/disposal_5y.csv.gz（disposal_5y 在來源停在 2026-10-05，warnings_hist 每日更新，2026-06-22 起）。
- 研究面板（history 不足 252 日時拼接用）：panel_pre＝[2025-01-01, history 起日) 子集，所有 asof 共用（最大日期 < history 起日 ≤ asof）。
- oddlot_hist 不需要，不寫（省磁碟）。
- 0050 含息、加權指數：date ≤ asof。
- stock_list.csv 原樣複製（名稱、市場、產業為現況，不是 PIT；已知限制）。
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
    def __init__(self, src=SRC, e50=E50_SRC, taiex=TAIEX_SRC, panel=PANEL_SRC):
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
    def build_asof(self, asof, out_dir):
        """寫出截斷到 asof 的資料夾；回傳 manifest。已存在而且來源 sha 相同時直接沿用。"""
        asof = pd.Timestamp(asof); out = Path(out_dir)
        man_p = out / 'MANIFEST.json'
        key = dict(asof=str(asof.date()), source=self.files)
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
        F['extras/rev_5y.csv.gz'] = write_csv(self._rev_trunc(self.r5, asof), out / 'extras/rev_5y.csv.gz')
        rh = self._rev_trunc(self.rh, asof)
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
        man_p.write_text(json.dumps(m, ensure_ascii=False, indent=1, sort_keys=True))
        return m


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


def assert_dir_asof(d, asof):
    """讀回截斷資料夾的每個檔案，斷言最大日期 ≤ asof（營收另斷言 ym ≤ asof 前一月、ann_date ≤ asof）。"""
    d = Path(d); asof = pd.Timestamp(asof); rep = {}
    for f in ['history.csv.gz', 'extras/disposal_5y.csv.gz', 'extras/exdiv_5y.csv.gz', 'extras/attention_5y.csv.gz',
              'extras/warnings_hist.csv.gz', 'extras/oddlot_hist.csv.gz', 'etf_0050.csv', 'taiex_twii.csv']:
        if (d / f).exists():
            x = pd.read_csv(d / f, usecols=['date'], parse_dates=['date'])
            rep[f] = assert_asof(asof, **{f: x})[f]
    for f in ['extras/rev_5y.csv.gz', 'extras/rev_hist.csv.gz']:
        x = pd.read_csv(d / f, dtype={'code': str})
        mx = pd.Period(x['ym'].max(), 'M')
        assert mx <= pd.Period(asof, 'M') - 1, f'A1 失敗：{f} 最新營收月份 {mx} 不早於 asof 月份'
        if 'ann_date' in x.columns: assert_asof(asof, **{f: x[['ann_date']]})
        rep[f] = f'ym≤{mx}'
    return rep


def assert_panel_asof(pdir, asof):
    rep = {}
    for f in PANEL_FILES:
        if (Path(pdir) / f'{f}.parquet').exists():
            p = pd.read_parquet(Path(pdir) / f'{f}.parquet', columns=[])
            rep[f'panel/{f}'] = assert_asof(asof, **{f: p})[f]
    return rep
