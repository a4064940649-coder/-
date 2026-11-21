#!/usr/bin/env python3
"""S18 現行策略（S10 E0＝XQ 20＋21 號預設、W10_70）每月名單產生器＋紙上追蹤紀錄。

規則（和 s4lib.make_signal(depth=10, mom_lo=0.10, exit='top20')、s10lib.make_signal_s10 的 E0、s9lib／s10lib 引擎逐條相同）
- 股池（訊號日當天）：上市（官方日 K market＝twse）、四碼普通股（代號 [1-9]ddd，不含 91 開頭）、當天有成交、收盤 ≥ 10、
  近 60 個交易日都在日 K 內且 60 日平均成交值 ≥ 1,000 萬、股名不以「-創」或「-KY創」結尾（創新板）。
- SURGE（營收月份 m＝訊號月份 − 1）：D＝本月營收 − 去年同月營收；SURGE＝（D(m) − 過去 24 個月 D 的平均）÷ 過去 24 個月 D 的標準差（ddof 1），
  標準差 ≤ 0 或任一個月缺值 → 沒有 SURGE。
- L_MA：股池內、有 SURGE、還原收盤 > 60 日均線（還原＝含息日報酬連乘），依 SURGE 由大到小（同分依代號）。
- F1 買進名單：L_MA 前 10 名中 RET20（還原收盤 ÷ 20 個交易日前 − 1）> 10%，依 SURGE 排序。
- E0 出場：持股不在 L_MA 前 20 名就賣（跌破季線、名次掉出 20、營收缺值、離開股池都算）。只在訊號日判斷，次一交易日開盤賣。
- 訊號日（A10）：訊號月份 10 日（含）之後的第一個交易日的「下一個交易日」。成交：訊號日次一交易日開盤。
- 有訊號的月份：訊號日股池內有 SURGE 的股票 ≥ 50 檔。
- 金額（W10_70）：底倉 0050 記帳 30%；衛星 Sv＝總資產 − 底倉市值；每檔目標＝Sv ÷ 10；最多 10 檔；
  買得到的金額 < 目標一半就不買；2020-10-26 起可零股，之前只能整張；衛星占總資產超出 70% ± 10 個百分點就把底倉重設成總資產 30%；
  剩下的現金 ≥ 2 萬買 0050；錢不夠先用現金再賣 0050（不鎖底倉，同 W10_70）。

資料（只讀，公開資料）
- 官方日 K：git 倉庫的 data/universe/daily/YYYY-MM-DD.csv（--repo 可給多個，後面的覆蓋前面的同日檔；預設研究中心的 raw 封存）。
- 月營收：tw_revenue.csv（--rev-csv）；營收檔沒有的新月份，用 --repo 倉庫的 data/mops/revenue_hist/YYYY-MM_twse.csv（公開資訊觀測站彙總表）補。
  --rev-cutoff（UTC ISO 時間）時，補進來的月份只用 data/meta/revenue_seen/<月份>/ 在截止前已看到的公司（時點檢查）。

用法
  python3 -I -B make_list.py list --month 2026-11 [--total 1000000] [--holdings h.csv] [--cash N] [--base-shares N]
                                  [--repo DIR ...] [--rev-cutoff 2026-11-11T14:00:00Z] [--out-prefix lists/2026-11] [--paper-log log.csv]
  python3 -I -B make_list.py fill --paper-log log.csv [--repo DIR ...]      # 事後把實際開盤價、漲跌停、與回測差異填進紀錄
  python3 -I -B make_list.py holdings --paper-log log.csv --month 2026-11 --out hold.csv   # 算出成交後的紙上持股，給下個月用
holdings.csv：欄位「代號,股數」（也接受 code,shares）；0050 也寫一列（含底倉）。
這是紙上追蹤工具：只產生名單與假想下單，不連任何券商、不下單。

除權息參考價修正（2026-11-16 加；list 預設開啟，--no-exright-fix 關閉；輸出 meta.exright 標明）
- 官方日 K 的漲跌價差在除權息日與「前一日無成交」日一律是 0（agents/data_exright_fix/AUDIT.md），還原價（季線、RET20）會漏股利。
- 開啟時規則同 cslib.exright_fix_chg（＝exr_common.corrected_chg(mode='main')）：上市除權息日用證交所 TWT49U 參考價
  （封存 raw/twt49u/，--twt49u-dir 可改；現增除權收盤落在參考價 ±10.1% 外改用前收）；上市前一日無成交且漲跌 ≤ 10.5% 用前一成交收盤。
- meta.exright.unverified：前 20 名、F1、持股在近 60 個交易日裡「漲跌價差＝0、收盤有變、又不在 TWT49U」的格子（含 TWT49U 封存還沒涵蓋的日子），請人工確認。
- compute() 的 exright 參數預設 False（validate.py 等舊呼叫逐位不變）；CLI 預設開啟。

營收申報期限例外（2026-11-17 加；list 預設開啟，--no-deadline-fix 關閉；輸出 meta.deadline 標明）
- 開啟時換股日用共用引擎的例外表（cslib.deadline_exceptions()＝raw/rev_deadline_exceptions/deadline_exceptions.csv 中 apply＝1 的月份，
  目前 2019-01 延至 2019-02-15、2023-03 延至 2023-04-12）：該營收月份的 10 日改成延後後的期限，訊號日＝期限（含）以後第一個交易日的下一個交易日
  （＝cslib.rev_avail_days 開 DEADLINE_FIX，正式回測 2026-11-17 起同口徑）。不在例外表的月份與關閉時完全相同。
- 實盤本來就看不到未公布的營收；這個開關主要是讓歷史重算與正式回測一致（例外表每年依金管會公告更新，見該資料夾 README）。
- 15 日期限的保險／金控股（S41）：換股日當月營收尚未公布＝沒有 SURGE，依規則賣出（見持股說明），與正式回測的 INS15_MASK 一致，不需另外開關。
- compute() 的 deadline 參數預設 False（validate.py 等舊呼叫逐位不變）；CLI 預設開啟。

減資／面額變更恢復買賣日修正（2026-11-21 加，**候選，預設關閉**；--reduce-fix 才開；輸出 meta.reduce 標明）
- 這份檔（make_list_v2.py）＝make_list.py 原樣複製＋這個開關。開關關閉時輸出與 make_list.py 逐欄相同（meta 不多任何欄位）。
- 開啟時規則同 cslib.reduce_fix_chg（正式回測候選版 official_nav/run_official_v2.py 的 REDUCE_FIX）：上市、當天有成交、當天在
  減資／面額變更事件封存（raw/twse_reduce/twse_reduce_events.csv，--reduce-csv 可改）→ 漲跌價差＝收盤 − 事件參考價（接在除權息修正之後）。
  影響的是還原價（季線、RET20、距季線）與次一交易日漲跌停價；SURGE 與股池不受影響。
- compute() 的 reduce 參數預設 False；CLI 也預設關閉（產品擁有者同意採用後才改預設）。"""
import argparse, csv, io, json, math, os, re, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd

RAW_DEFAULT = '/home/user/-/research/tw_stock_selection/raw/jb7k6295r7-bot_tw-stock-data'
REV_DEFAULT = '/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv'
WINDOW = 320                      # 讀訊號日前幾個交易日（季線 60、均成交值 60、RET20 20；多讀以免邊界）
FEE, TAX, MINFEE, TAX_ETF = 0.001425 * 0.28, 0.003, 20.0, 0.001
XSLIP = 0.003                     # 研究中心嚴成本：每邊再加 0.3% 滑價
ODD_DATE = pd.Timestamp('2020-10-26')
REV_MIN_COVER = 0.95              # 補充營收至少要有上月上市家數的 95%
SAT_W, BAND, SLOTS, DEPTH, MOM_LO, KEEP_N, MIN_FRAC, SWEEP_MIN = 0.70, 0.10, 10, 10, 0.10, 20, 0.5, 20000.0
INNOV_PAT = re.compile(r'-(KY)?創$')
CODE_PAT = re.compile(r'^[1-9]\d{3}$')
COLS_USE = ['date', 'stock_id', 'name', 'market', 'open', 'high', 'low', 'close', 'volume', 'amount', 'change']
TWT49U_DEFAULT = '/home/user/-/research/tw_stock_selection/raw/twt49u/twse-exright'
CSLIB_DIR = '/home/user/-/research/tw_stock_selection/agents/o_ext_canslim_xs'   # 期限例外表讀 cslib.deadline_exceptions()
REDUCE_DEFAULT = '/home/user/-/research/tw_stock_selection/raw/twse_reduce/twse_reduce_events.csv'   # 減資／面額變更事件封存（候選）


def deadline_exceptions():
    """{營收月份 Period: 延後後的期限日}（cslib.deadline_exceptions()，apply＝1 的列）。"""
    if CSLIB_DIR not in sys.path:
        sys.path.insert(0, CSLIB_DIR)
    import cslib
    return cslib.deadline_exceptions()


def tick(p):
    p = np.asarray(p, float)
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)


def etf_tick(p):
    return 0.01 if p < 50 else 0.05


# ---------------------------------------------------------------- 資料層
class Repo:
    """一個 git 倉庫（blobless 可）：列出日 K 檔、讀檔。lazy＝允許缺的 blob 從遠端補抓（新 clone 用）。"""

    def __init__(self, path):
        self.path = str(path)
        self.env = dict(os.environ)
        self.head = self._git('rev-parse', 'HEAD').strip()

    def _git(self, *a):
        return subprocess.run(['git', '-C', self.path, *a], capture_output=True, text=True, check=True, env=self.env).stdout

    def ls(self, sub):
        out = self._git('ls-tree', 'HEAD', sub.rstrip('/') + '/')
        res = {}
        for line in out.strip().split('\n'):
            if not line:
                continue
            meta, path = line.split('\t', 1)
            res[path] = meta.split()[2]
        return res

    def ls_r(self, sub):
        out = self._git('ls-tree', '-r', 'HEAD', sub.rstrip('/') + '/')
        res = {}
        for line in out.strip().split('\n'):
            if line:
                meta, path = line.split('\t', 1)
                res[path] = meta.split()[2]
        return res

    def read(self, shas):
        """讀多個 blob（sha → bytes）。"""
        out = {}
        if not shas:
            return out
        p = subprocess.Popen(['git', '-C', self.path, 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=self.env)
        for s in shas:
            p.stdin.write((s + '\n').encode()); p.stdin.flush()
            hdr = p.stdout.readline().decode().split()
            if len(hdr) < 3 or hdr[1] != 'blob':
                continue
            n = int(hdr[2]); out[s] = p.stdout.read(n); p.stdout.read(1)
        p.stdin.close(); p.wait()
        return out


class Data:
    """日 K 與月營收。repos：由舊到新；同一天的檔，後面的倉庫覆蓋前面的（並記錄 blob 是否相同）。"""

    def __init__(self, repos, rev_csv=REV_DEFAULT, rev_cutoff=None):
        self.repos = [Repo(r) for r in repos]
        self.daily = {}                      # 日期 → (repo, blob)
        self.overlap_diff = []
        for rp in self.repos:
            for path, sha in rp.ls('data/universe/daily').items():
                if not path.endswith('.csv'):
                    continue
                d = path.split('/')[-1][:10]
                if d in self.daily:
                    if self.daily[d][1] == sha:
                        continue                 # 同一個 blob：沿用前面（本機已有）的倉庫，不從遠端補抓
                    self.overlap_diff.append(d)
                self.daily[d] = (rp, sha)
        self.dates = sorted(self.daily)
        self.IDX = pd.DatetimeIndex(self.dates)
        self.cache = {}
        self.rev_csv = rev_csv
        self.rev_cutoff = pd.Timestamp(rev_cutoff) if rev_cutoff else None
        self._S = None
        self.rev_info = {}

    # ---- 日 K
    def _parse_day(self, b):
        d = pd.read_csv(io.BytesIO(b), dtype={'stock_id': str}, usecols=COLS_USE)
        e = d[d['stock_id'] == '0050']
        e50 = {c: (float(pd.to_numeric(e[c], errors='coerce').iloc[0]) if len(e) else np.nan) for c in ('open', 'close', 'change')}
        d = d[d['stock_id'].str.fullmatch(r'[1-9]\d{3}') & d['market'].isin(['twse', 'tpex'])].drop_duplicates('stock_id', keep='last')
        vals = d[['open', 'close', 'amount', 'change']].apply(pd.to_numeric, errors='coerce').to_numpy(float)
        return dict(codes=d['stock_id'].to_numpy(), names=d['name'].astype(str).str.strip().to_numpy(),
                    mk=np.where(d['market'].to_numpy() == 'twse', 1, 2).astype(np.int8), vals=vals, e50=e50)

    def days(self, dates):
        need = [d for d in dates if d not in self.cache]
        by = {}
        for d in need:
            rp, sha = self.daily[d]
            by.setdefault(id(rp), (rp, []))[1].append((d, sha))
        for rp, lst in by.values():
            got = rp.read([s for _, s in lst])
            for d, s in lst:
                self.cache[d] = self._parse_day(got[s])
        keep = set(dates)
        for d in list(self.cache):              # 只留這次用到的日子（逐月跑時記憶體不累積）
            if d not in keep:
                del self.cache[d]
        return [self.cache[d] for d in dates]

    # ---- 月營收
    def surge_table(self):
        if self._S is not None:
            return self._S
        r = pd.read_csv(self.rev_csv, dtype={'code': str}, parse_dates=['month'])
        base_max = r['month'].max()
        self.rev_info['rev_csv'] = self.rev_csv
        self.rev_info['rev_csv_last_month'] = str(base_max.date())[:7]
        sup = {}
        for rp in self.repos:
            try:
                files = rp.ls('data/mops/revenue_hist')
            except subprocess.CalledProcessError:
                continue
            for path, sha in files.items():
                mname = path.split('/')[-1]
                mm = re.match(r'^(\d{4}-\d{2})_twse\.csv$', mname)
                if not mm or pd.Timestamp(mm.group(1) + '-01') <= base_max:
                    continue
                sup[mm.group(1)] = (mm.group(1), rp, sha, path)      # 同月份：後面（較新）的倉庫覆蓋前面的
        prev_codes = set(r.loc[(r['month'] == base_max) & (r['market'] == 'sii'), 'code'])
        for mon, rp, sha, path in sorted(sup.values(), key=lambda x: x[0]):
            b = rp.read([sha])[sha]
            o = pd.read_csv(io.BytesIO(b), dtype={'stock_id': str})
            o = o[o['stock_id'].str.fullmatch(r'[1-9]\d{3}')]
            v = pd.to_numeric(o['當月營收'], errors='coerce')
            add = pd.DataFrame(dict(month=pd.Timestamp(mon + '-01'), market='sii', code=o['stock_id'].to_numpy(), name=o['name'].to_numpy(), revenue_k_twd=v.to_numpy()))
            n_prev = int(((r['month'] == base_max) & (r['market'] == 'sii')).sum())
            info = dict(source=f'{rp.path}@{rp.head[:12]}:{path}', blob=sha, n_rows=int(len(add)), n_prev_month_sii=n_prev)
            if self.rev_cutoff is not None:
                seen = self._seen_codes(rp, mon)
                if seen is None:
                    info['cutoff'] = '倉庫沒有 revenue_seen 紀錄，未做時點檢查'
                else:
                    ok_codes, late, unk = seen
                    keep = add['code'].isin(ok_codes)
                    info['cutoff'] = str(self.rev_cutoff)
                    info['excluded_after_cutoff'] = sorted(set(add.loc[~keep & add['code'].isin(late), 'code']))
                    info['excluded_never_seen'] = sorted(set(add.loc[~keep & ~add['code'].isin(late), 'code']))
                    add = add[keep]
            info['n_used'] = int(len(add)); info['coverage'] = round(len(set(add['code']) & prev_codes) / max(len(prev_codes), 1), 4)   # 上月營收檔上市公司中，本月已有的比例
            self.rev_info.setdefault('supplement', {})[mon] = info
            sup_df = add
            r = pd.concat([r, sup_df], ignore_index=True)
        # 以下逐行同 s2lib.rev_tables（σ > 0 條件）
        r['pri'] = (r['market'] != 'sii').astype(int)
        r = r.sort_values(['month', 'code', 'pri']).drop_duplicates(['month', 'code'], keep='first')
        Rv = r.pivot(index='month', columns='code', values='revenue_k_twd').sort_index()
        Rv.index = pd.PeriodIndex(Rv.index, freq='M')
        Rv = Rv.reindex(pd.period_range(Rv.index.min(), Rv.index.max(), freq='M'))
        Rv = Rv.where(Rv > 0)
        D = Rv - Rv.shift(12)
        D[(D.index.year == 2013)] = np.nan
        mu = D.shift(1).rolling(24, min_periods=24).mean()
        sd = D.shift(1).rolling(24, min_periods=24).std(ddof=1)
        S = ((D - mu) / sd.where(sd > 0)).replace([np.inf, -np.inf], np.nan)
        self._S = S; self._Rv = Rv
        return S

    def _seen_codes(self, rp, mon):
        try:
            files = rp.ls_r(f'data/meta/revenue_seen/{mon}')
        except subprocess.CalledProcessError:
            return None
        if not files:
            return None
        ok, late = set(), set()
        cut = self.rev_cutoff.tz_convert('UTC') if self.rev_cutoff.tzinfo else self.rev_cutoff.tz_localize('UTC')
        got = rp.read(list(files.values()))
        for path, sha in files.items():
            m = re.match(r'^(\d{4}-\d{2}-\d{2})T(\d{2})(\d{2})(\d{2})', path.split('/')[-1])   # 檔名時間＝UTC
            if not m:
                continue
            ts = pd.Timestamp(f'{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}', tz='UTC')
            codes = set(pd.read_csv(io.BytesIO(got[sha]), dtype={'stock_id': str})['stock_id'])
            (ok if ts <= cut else late).update(codes)
        late -= ok
        return ok, late, None


# ---------------------------------------------------------------- 除權息參考價修正
def load_twt49u(d=TWT49U_DEFAULT):
    """TWT49U 封存（四碼普通股；同日同代號留最後一筆）。同 cslib.load_twt49u。"""
    fr = [pd.read_csv(f, dtype={'code': str}) for f in sorted(Path(d).glob('*/*.csv.gz'))]
    if not fr:
        raise SystemExit(f'找不到 TWT49U 封存：{d}（或加 --no-exright-fix）')
    e = pd.concat(fr, ignore_index=True)
    e['date'] = pd.to_datetime(e['date'])
    e = e[e['code'].str.fullmatch(r'[1-9]\d{3}')]
    return e.drop_duplicates(['date', 'code'], keep='last')


def exright_fix_chg(IDX, COLS, C, CHG, MK, ev):
    """逐行同 cslib.exright_fix_chg（說明見該處）。回傳（修正後 CHG, info, 布林矩陣：漲跌價差＝0 且收盤有變、不在 TWT49U）。"""
    IDX = pd.DatetimeIndex(IDX); CHG = np.array(CHG, dtype=float, copy=True)
    T, N = C.shape
    has = np.isfinite(C) & (C > 0)
    CF = pd.DataFrame(np.where(has, C, np.nan)).ffill().to_numpy()
    prevC = np.vstack([np.full((1, N), np.nan), CF[:-1]])
    zero = np.nan_to_num(CHG) == 0
    with np.errstate(invalid='ignore'):
        z = has & np.isfinite(prevC) & zero & (np.abs(C - prevC) > 1e-9)
    REFX = np.full((T, N), np.nan)
    ci = {c: j for j, c in enumerate(COLS)}; di = {d: i for i, d in enumerate(IDX)}
    e = ev[ev['code'].isin(ci) & ev['date'].isin(di)]
    REFX[e['date'].map(di).to_numpy(), e['code'].map(ci).to_numpy()] = e['ref_price'].to_numpy()
    inx = np.isfinite(REFX) & (MK == 1)
    prev_tr = np.vstack([np.zeros((1, N), bool), has[:-1]])
    m = has & inx & zero
    with np.errstate(invalid='ignore', divide='ignore'):
        rr = C / REFX
        oob = m & ~((rr >= 0.899) & (rr <= 1.101))
        REFX[oob] = prevC[oob]
        CHG[m] = C[m] - REFX[m]
        m3 = z & ~inx & ~prev_tr & (MK == 1) & (np.abs(C / prevC - 1) <= 0.105)
    CHG[m3] = C[m3] - prevC[m3]
    info = dict(n_twt49u=int(m.sum()), n_rights_issue_fallback=int(oob.sum()), n_prev_untraded=int(m3.sum()))
    return CHG, info, (z & ~inx), m3


def load_reduce(f=REDUCE_DEFAULT):
    """減資／面額變更恢復買賣事件封存（同日同代號留最後一筆、參考價 > 0）。同 cslib.load_reduce。"""
    if not Path(f).exists():
        raise SystemExit(f'找不到減資事件封存：{f}（或不要加 --reduce-fix）')
    e = pd.read_csv(f, dtype={'code': str})
    e['date'] = pd.to_datetime(e['date']); e['ref_price'] = pd.to_numeric(e['ref_price'], errors='coerce')
    e = e.drop_duplicates(['date', 'code'], keep='last')
    return e[e['ref_price'] > 0]


def reduce_fix_chg(IDX, COLS, C, CHG, MK, ev):
    """逐行同 cslib.reduce_fix_chg：上市、當天有成交、當天在事件表 → CHG＝收盤 − 事件參考價。回傳（修正後 CHG, info, 套用的格子布林矩陣）。"""
    CHG = np.array(CHG, dtype=float, copy=True)
    di = {d: i for i, d in enumerate(pd.DatetimeIndex(IDX))}; ci = {c: j for j, c in enumerate(COLS)}
    n = 0; hit = np.zeros(C.shape, bool); cells = []
    for d, c, ref in zip(ev['date'], ev['code'], ev['ref_price']):
        if d in di and c in ci:
            t, j = di[d], ci[c]
            if MK[t, j] == 1 and np.isfinite(C[t, j]) and C[t, j] > 0:
                CHG[t, j] = C[t, j] - ref; n += 1; hit[t, j] = True
                cells.append(dict(date=str(d.date()), code=str(c), close=float(C[t, j]), ref_price=float(ref)))
    return CHG, dict(n_reduce_parvalue=n, reduce_last=str(ev['date'].max().date()) if len(ev) else None, cells=cells), hit


# ---------------------------------------------------------------- 訊號日
def signal_day(D, month, deadline=False):
    """訊號月份 → (訊號日索引, 次一交易日索引或 None, 說明)。資料不夠新時索引為 None。
    deadline：True＝營收月份在期限例外表時，10 日改用延後後的期限（同 cslib.rev_avail_days 開 DEADLINE_FIX）。"""
    m = pd.Period(month, freq='M')
    d10 = m.to_timestamp() + pd.Timedelta(days=9)
    if deadline:
        d10 = max(d10, deadline_exceptions().get(m - 1, d10))
    k = int(np.searchsorted(D.IDX.values, np.datetime64(d10), side='left'))
    if k >= len(D.dates):
        return None, None, f'日 K 最新到 {D.dates[-1]}，還沒有 {d10.date()}（含）之後的交易日'
    if k + 1 >= len(D.dates):
        return None, None, f'日 K 最新到 {D.dates[-1]}；10 日後第一個交易日是 {D.dates[k]}，訊號日是它的下一個交易日，資料還沒到'
    ex = k + 2 if k + 2 < len(D.dates) else None
    return k + 1, ex, ''


def est_signal_day(month, deadline=False):
    """沒有日 K 時的預估（只扣週末，不扣國定假日）。"""
    m = pd.Period(month, freq='M')
    d = m.to_timestamp() + pd.Timedelta(days=9)
    if deadline:
        d = max(d, deadline_exceptions().get(m - 1, d))
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    d += pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return str(d.date())


# ---------------------------------------------------------------- 計算
def compute(D, month, window=WINDOW, allow_partial=False, exright=False, twt49u_dir=TWT49U_DEFAULT, extra_codes=(), deadline=False,
            reduce=False, reduce_csv=REDUCE_DEFAULT):
    """回傳當月的 L_MA 排名、F1、全部股票在訊號日的欄位（dict）。
    exright：True＝除權息參考價修正（見檔頭）；False＝舊口徑（逐位同 2026-11-16 之前）。extra_codes：另外要做可疑格子檢查的代號（持股）。
    deadline：True＝換股日套用營收期限例外表（見檔頭）；False＝舊口徑（逐位同 2026-11-17 之前）。
    reduce：True＝減資／面額變更恢復買賣日用事件參考價（候選，見檔頭）；False＝現行口徑（輸出不多任何欄位）。"""
    ks, kx, why = signal_day(D, month, deadline)
    m_sig = pd.Period(month, freq='M'); m_rev = m_sig - 1
    out = dict(signal_month=str(m_sig), rev_month=str(m_rev), data_last_day=D.dates[-1], est_signal_day=est_signal_day(month, deadline))
    if deadline:
        ext = deadline_exceptions().get(m_rev)
        out['deadline'] = dict(on=True, rule='cslib.deadline_exceptions()（raw/rev_deadline_exceptions/deadline_exceptions.csv，apply＝1）',
                               exception=None if ext is None else str(ext.date()),
                               note=(f'{m_rev} 營收期限延至 {ext.date()}，訊號日＝期限（含）以後第一個交易日的下一個交易日' if ext is not None
                                     else f'{m_rev} 不在例外表，換股日照一般規則'))
    if ks is None:
        out.update(status='資料不足', reason=why); return out
    S = D.surge_table()
    if m_rev not in S.index:
        out.update(status='資料不足', reason=f'月營收最新到 {S.index.max()}，沒有 {m_rev}（{month} 的訊號要用 {m_rev} 營收）', sig_date=D.dates[ks]); return out
    supi = D.rev_info.get('supplement', {}).get(str(m_rev))
    if supi is not None and supi['coverage'] < REV_MIN_COVER and not allow_partial:
        out.update(status='資料不足', sig_date=D.dates[ks],
                   reason=f'{m_rev} 營收只補到 {supi["n_used"]} 家上市公司（上月 {supi["n_prev_month_sii"]} 家，{supi["coverage"]:.0%}）；回測假設訊號日營收已全部公布，請等資料更新（或加 --allow-partial-rev 看暫定名單）')
        return out
    lo = max(0, ks - window + 1)
    hi = ks + 1 if kx is None else kx + 1
    dates = D.dates[lo:hi]
    days = D.days(dates)
    codes = np.array(sorted(set(np.concatenate([x['codes'] for x in days]))))
    ci = {c: j for j, c in enumerate(codes)}
    T, N = len(days), len(codes)
    O = np.full((T, N), np.nan); C = O.copy(); AMT = O.copy(); CHG = O.copy(); MK = np.zeros((T, N), np.int8)
    for t, x in enumerate(days):
        j = np.array([ci[c] for c in x['codes']])
        O[t, j], C[t, j], AMT[t, j], CHG[t, j] = x['vals'][:, 0], x['vals'][:, 1], x['vals'][:, 2], x['vals'][:, 3]
        MK[t, j] = x['mk']
    k = ks - lo                                   # 訊號日在視窗內的列
    exr = None
    if exright:
        ev = load_twt49u(twt49u_dir)
        CHG0 = CHG
        CHG, xinfo, zx, m3 = exright_fix_chg(pd.DatetimeIndex(dates), codes, C, CHG, MK, ev)
        tw_last = ev['date'].max()
        exr = dict(on=True, rule="cslib.exright_fix_chg（＝data_exright_fix/exr_common.corrected_chg(mode='main')）", twt49u_dir=str(twt49u_dir),
                   twt49u_last=str(tw_last.date()), window=f'{dates[0]}～{dates[-1]}', **xinfo,
                   days_after_twt49u_in_window=[d for d in dates[:k + 1] if pd.Timestamp(d) > tw_last])
    rdc = None; rhit = None
    if reduce:
        rev_ = load_reduce(reduce_csv)
        CHG, rinfo, rhit = reduce_fix_chg(pd.DatetimeIndex(dates), codes, C, CHG, MK, rev_)
        rdc = dict(on=True, rule='cslib.reduce_fix_chg（候選；上市、當天有成交、在事件表 → 漲跌價差＝收盤 − 事件參考價）', reduce_csv=str(reduce_csv),
                   window=f'{dates[0]}～{dates[-1]}', **rinfo)
    names = {}
    for x in (days[k],):
        names = dict(zip(x['codes'], x['names']))
    # 同 cslib.Panel／stlib.load
    has = np.isfinite(C) & (C > 0)
    listed = MK > 0
    ref = C - CHG
    ref[~(ref > 0)] = np.nan
    with np.errstate(invalid='ignore', divide='ignore'):
        R = C / ref - 1
    R[np.abs(R) > 0.5] = np.nan
    CF = pd.DataFrame(C).ffill().to_numpy()
    prevC = np.vstack([np.full((1, N), np.nan), CF[:-1]])
    with np.errstate(invalid='ignore', divide='ignore'):
        R = np.where(np.isfinite(R), R, np.where(has, C / prevC - 1, 0.0))
    R = np.where(np.isfinite(R), R, 0.0)
    REF = np.where(np.isfinite(ref), ref, prevC)
    G = np.cumprod(1 + R, axis=0)
    amt = np.where(listed, np.nan_to_num(AMT), np.nan)
    V60 = pd.DataFrame(amt[:k + 1]).rolling(60, min_periods=60).mean().to_numpy()[k]
    MA60 = pd.DataFrame(G[:k + 1]).rolling(60, min_periods=60).mean().to_numpy()[k]
    with np.errstate(invalid='ignore', divide='ignore'):
        RET20 = G[k] / G[k - 20] - 1 if k >= 20 else np.full(N, np.nan)
    not91 = np.array([not c.startswith('91') for c in codes])
    innov = np.array([bool(INNOV_PAT.search(names.get(c, ''))) for c in codes])
    with np.errstate(invalid='ignore'):
        liq = (V60 >= 1e7) & (C[k] >= 10) & has[k] & not91
    univ = liq & (MK[k] == 1) & ~innov
    s = S.loc[m_rev].reindex(codes).to_numpy(float)
    fin = univ & np.isfinite(s)
    out['n_univ'] = int(univ.sum()); out['n_univ_surge'] = int(fin.sum())
    with np.errstate(invalid='ignore'):
        fma = G[k] > MA60
    ok_ma = fin & fma
    o_ma = np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]
    out['n_ma'] = int(ok_ma.sum())
    if out['n_univ_surge'] < 50:
        out.update(status='無訊號', reason=f'股池內有 SURGE 的股票只有 {out["n_univ_surge"]} 檔（< 50）', sig_date=D.dates[ks]); return out
    pool = o_ma[:DEPTH]
    tgt = pool[RET20[pool] > MOM_LO][:SLOTS]
    rank_ma = {int(j): i + 1 for i, j in enumerate(o_ma)}
    # 漲跌停（次一交易日，事後檢查用）
    ex = None
    if kx is not None:
        kxx = kx - lo
        rx = REF[kxx]
        lu = np.floor(rx * 1.10 / tick(rx * 1.10) + 1e-9) * tick(rx * 1.10)
        ld = np.ceil(rx * 0.90 / tick(rx * 0.90) - 1e-9) * tick(rx * 0.90)
        ex = dict(date=D.dates[kx], open=O[kxx], limup=lu, limdn=ld, e50_open=days[kxx]['e50']['open'])

    def row(j):
        return dict(code=str(codes[j]), name=names.get(codes[j], ''), surge=None if not np.isfinite(s[j]) else float(s[j]),
                    rank_ma=rank_ma.get(int(j)), ret20=float(RET20[j]) if np.isfinite(RET20[j]) else None,
                    above_ma60=bool(fma[j]), dist_ma60=float(G[k, j] / MA60[j] - 1) if np.isfinite(MA60[j]) and MA60[j] > 0 else None,
                    v60=float(V60[j]) if np.isfinite(V60[j]) else None, close=float(CF[k, j]) if np.isfinite(CF[k, j]) else None,
                    in_univ=bool(univ[j]), listed_twse=bool(MK[k, j] == 1), innov=bool(innov[j]), traded=bool(has[k, j]))
    out.update(status='有訊號', sig_date=D.dates[ks], exec_date=D.dates[kx] if kx is not None else None,
               top20=[row(j) for j in o_ma[:KEEP_N]], f1=[row(j) for j in tgt],
               e50_close=days[k]['e50']['close'])
    if exright:
        # 可疑格子：前 20 名＋F1＋持股，近 60 個交易日（到訊號日）漲跌價差＝0、收盤有變、不在 TWT49U（rule (b) 已補的也列，標出來）
        chk = {r['code'] for r in out['top20']} | {r['code'] for r in out['f1']} | {str(c) for c in extra_codes}
        rows = []
        for c in sorted(chk):
            j = ci.get(c)
            if j is None:
                continue
            for t in range(max(0, k - 59), k + 1):
                if zx[t, j]:
                    rows.append(dict(code=c, date=dates[t], close=float(C[t, j]), chg_raw=float(np.nan_to_num(CHG0[t, j])),
                                     fixed='減資／面額變更恢復買賣日，已改用事件參考價' if (rhit is not None and rhit[t, j]) else
                                     '前一日無成交，已改用前一成交收盤' if m3[t, j] else
                                     ('TWT49U 封存還沒涵蓋這天，未修正' if pd.Timestamp(dates[t]) > tw_last else '未修正（原因不明）')))
        exr['unverified'] = rows
    else:
        exr = dict(on=False, note='舊口徑：漲跌價差照官方日 K（除權息日 0，還原價漏股利）')
    out['exright'] = exr
    if rdc is not None:
        chk = {r['code'] for r in out['top20']} | {r['code'] for r in out['f1']} | {str(c) for c in extra_codes}
        rdc['cells_in_top20_f1_holdings'] = [c for c in rdc['cells'] if c['code'] in chk]
        out['reduce'] = rdc
    out['_ci'] = ci; out['_row'] = row; out['_ex'] = ex; out['_codes'] = codes
    return out


def held_status(res, code):
    """持股在訊號日的狀態與 E0 判定。"""
    ci = res['_ci']
    keep = {r['code'] for r in res['top20']}
    if code not in ci:
        return dict(code=code, name='', action='賣出', reason='訊號日不在官方日 K（下市或停止交易）'), None
    r = res['_row'](ci[code])
    if code in keep:
        return dict(r, action='續抱', reason=f'L_MA 第 {r["rank_ma"]} 名'), r
    if not r['listed_twse']:
        why = '不是上市股（股池只看上市）'
    elif r['innov']:
        why = '創新板'
    elif not r['in_univ']:
        why = '不在股池（60 日均成交值 < 1,000 萬、收盤 < 10 或當天無成交）'
    elif r['surge'] is None:
        why = '沒有 SURGE（營收缺值或歷史不足；若是 15 日營收期限的保險／金控股，換股日當月營收尚未公布，依規則賣出，屬正常，見 S41）'
    elif not r['above_ma60']:
        why = '跌破季線（還原收盤 ≤ 60 日均線）'
    else:
        why = f'掉出 L_MA 前 20 名（第 {r["rank_ma"]} 名）'
    return dict(r, action='賣出', reason=why), r


def plan(res, holdings=None, total=None, cash=None, base_shares=None):
    """依 W10_70 規則算出當月要做的事（價格用訊號日收盤估；實際在次一交易日開盤成交）。"""
    ex_ts = pd.Timestamp(res['exec_date']) if res.get('exec_date') else pd.Timestamp(res['sig_date']) + pd.offsets.BDay(1)
    odd_ok = ex_ts >= ODD_DATE                                   # 成交日（訊號日次一交易日）≥ 2020-10-26 才能零股
    p50 = float(res['e50_close'])
    holdings = dict(holdings or {})
    e50_sh = float(holdings.pop('0050', 0))
    hold_rows, sells, keeps = [], [], []
    mv = 0.0
    for code, sh in holdings.items():
        st, r = held_status(res, code)
        px = (r or {}).get('close') or 0.0
        st['shares'] = sh; st['value'] = sh * px
        mv += sh * px
        hold_rows.append(st)
        (sells if st['action'] == '賣出' else keeps).append(st)
    if total is None:
        total = mv + e50_sh * p50 + (cash or 0.0)
    if cash is None:
        cash = total - mv - e50_sh * p50
    notes = []
    if cash < -1:
        notes.append(f'輸入的總資產比持股市值還少 {-cash:,.0f} 元，請檢查。')
    # 賣出（估：訊號日收盤 × (1 − 半個跳動點 − 0.3% 滑價)，扣手續費與證交稅）
    proceeds = 0.0
    for st in sells:
        px = st.get('close') or 0.0
        if px > 0 and st['shares'] > 0:
            sl = 0.5 * float(tick(px)) / px + XSLIP
            g = st['shares'] * px * (1 - sl)
            st['est_proceeds'] = g - max(MINFEE, g * FEE) - g * TAX
            proceeds += st['est_proceeds']
    cash_after = cash + proceeds
    keep_mv = sum(st['value'] for st in keeps)
    eq = cash_after + keep_mv + e50_sh * p50
    # 底倉
    if base_shares is None:
        base_shares = (1 - SAT_W) * eq / p50
        notes.append('沒有提供底倉股數：視為新開始，底倉＝總資產 30% 的 0050。')
    B = base_shares * p50
    sv = eq - B
    dev = sv / eq - SAT_W if eq > 0 else 0.0
    band = dict(total_after_sells=eq, base_shares=base_shares, base_value=B, satellite=sv, satellite_frac=sv / eq if eq > 0 else None,
                dev_pp=dev * 100, ok=bool(abs(dev) <= BAND + 1e-12), action='不動')
    if abs(dev) > BAND + 1e-12:
        Bt = (1 - SAT_W) * eq
        if dev < 0:
            band['action'] = f'衛星占比 {sv/eq:.1%} < 60%：底倉股數改記成 {Bt/p50:,.0f} 股（只改帳，不交易）'
        else:
            idle = cash_after + e50_sh * p50
            trim = max(Bt - idle, 0.0)
            band['action'] = f'衛星占比 {sv/eq:.1%} > 80%：底倉股數改記成 {Bt/p50:,.0f} 股' + (f'；現金＋0050 不足，需依市值等比例減碼續抱股共約 {trim:,.0f} 元' if trim > 0 else '（現金＋0050 足夠，不用賣股）')
            band['trim_value'] = trim
        base_shares = Bt / p50; B = Bt; sv = eq - B
    band['base_shares_new'] = base_shares
    u = sv / SLOTS
    # 買進
    held_now = {st['code'] for st in keeps}
    free = SLOTS - len(keeps)
    e50_liq = max(e50_sh * p50 * (1 - 0.5 * etf_tick(p50) / p50) * (1 - FEE - TAX_ETF) - (MINFEE if e50_sh > 0 else 0.0), 0.0)
    avail_cash = cash_after
    buys = []
    for r in res['f1']:
        b = dict(r)
        if r['code'] in held_now:
            b.update(action='續抱（已持有）'); buys.append(b); continue
        if free <= 0:
            b.update(action='不買：已滿 10 檔'); buys.append(b); continue
        o = r['close']
        sl = 0.5 * float(tick(o)) / o + XSLIP
        px = o * (1 + sl)
        avail = max(avail_cash, 0.0) + e50_liq
        budget = min(u, avail)
        sh = int(budget / (px * (1 + FEE)))
        if not odd_ok:
            sh = (sh // 1000) * 1000
        fee = max(MINFEE, sh * px * FEE) if sh > 0 else 0.0
        while sh > 0 and sh * px + fee > avail + 1e-6:
            sh -= 1 if odd_ok else 1000
            fee = max(MINFEE, sh * px * FEE) if sh > 0 else 0.0
        if sh <= 0 or sh * o < MIN_FRAC * u:
            b.update(action='不買：買得到的金額不到目標一半', target=u, shares=sh); buys.append(b); continue
        cst = sh * px + fee
        if avail_cash < cst:
            e50_liq -= (cst - max(avail_cash, 0.0))
        avail_cash -= cst
        warn = []
        if r['v60'] and u > 0.05 * r['v60']:
            warn.append('目標金額 > 60 日均成交值 5%（回測單日最多買當天成交值 5%）')
        b.update(action='買進', target=u, shares=sh, est_value=sh * o, est_cost=cst, lots=f'{sh // 1000} 張 {sh % 1000} 股', warn='；'.join(warn))
        buys.append(b); free -= 1
    # 0050
    e50_target_sh = e50_sh
    if avail_cash < -1e-6:
        need = -avail_cash
        sell_sh = math.ceil(need / (p50 * (1 - 0.5 * etf_tick(p50) / p50) * (1 - FEE - TAX_ETF)))
        e50_target_sh = e50_sh - sell_sh
        cash_left = avail_cash + sell_sh * p50 * (1 - 0.5 * etf_tick(p50) / p50) * (1 - FEE - TAX_ETF)
        e50_action = f'賣 0050 {sell_sh:,} 股補買股的錢'
    elif avail_cash >= SWEEP_MIN:
        add = int((avail_cash - max(MINFEE, avail_cash * FEE)) / (p50 * (1 + 0.5 * etf_tick(p50) / p50) * (1 + FEE)))
        if not odd_ok:
            add = (add // 1000) * 1000
        e50_target_sh = e50_sh + add
        cash_left = avail_cash - add * p50 * (1 + 0.5 * etf_tick(p50) / p50) - max(MINFEE, add * p50 * FEE)
        e50_action = f'剩下現金約 {avail_cash:,.0f} 元，買 0050 {add:,} 股'
    else:
        cash_left = avail_cash
        e50_action = f'剩下現金約 {avail_cash:,.0f} 元（< 2 萬），不動 0050'
    if e50_target_sh < base_shares:
        notes.append(f'0050 股數 {e50_target_sh:,.0f} 少於底倉股數，底倉改記成 {e50_target_sh:,.0f} 股（賣底倉去買股）。')
        band['base_shares_new'] = e50_target_sh
    return dict(total_in=total, cash_in=cash, sells=sells, keeps=keeps, buys=buys, band=band, unit=u, e50=dict(price=p50, shares_before=e50_sh,
                shares_after=e50_target_sh, action=e50_action), cash_left=cash_left, notes=notes, odd_lot=odd_ok)


# ---------------------------------------------------------------- 輸出
def f2(x, n=2):
    return '' if x is None else f'{x:.{n}f}'


def write_outputs(res, pl, prefix, D, meta):
    prefix = Path(prefix); prefix.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in res['top20']:
        rows.append(dict(區塊='L_MA前20', 名次=r['rank_ma'], 代號=r['code'], 名稱=r['name'], SURGE=f2(r['surge'], 3), RET20pct=f2(r['ret20'] * 100, 1),
                         季線上='是' if r['above_ma60'] else '否', 距季線pct=f2((r['dist_ma60'] or 0) * 100, 1), 均成交值60日_萬=f2((r['v60'] or 0) / 1e4, 0),
                         收盤=r['close'], F1='是' if r['code'] in {x['code'] for x in res['f1']} else '', 動作='', 目標金額='', 建議股數='', 說明=''))
    if pl:
        for st in pl['sells']:
            rows.append(dict(區塊='持股_賣出', 名次=st.get('rank_ma') or '', 代號=st['code'], 名稱=st.get('name', ''), SURGE=f2(st.get('surge'), 3),
                             RET20pct=f2((st.get('ret20') or 0) * 100, 1), 季線上='是' if st.get('above_ma60') else '否', 距季線pct=f2((st.get('dist_ma60') or 0) * 100, 1),
                             均成交值60日_萬=f2((st.get('v60') or 0) / 1e4, 0), 收盤=st.get('close', ''), F1='', 動作='賣出（次一交易日開盤）', 目標金額='',
                             建議股數=st['shares'], 說明=st['reason']))
        for st in pl['keeps']:
            rows.append(dict(區塊='持股_續抱', 名次=st.get('rank_ma') or '', 代號=st['code'], 名稱=st.get('name', ''), SURGE=f2(st.get('surge'), 3),
                             RET20pct=f2((st.get('ret20') or 0) * 100, 1), 季線上='是', 距季線pct=f2((st.get('dist_ma60') or 0) * 100, 1),
                             均成交值60日_萬=f2((st.get('v60') or 0) / 1e4, 0), 收盤=st.get('close', ''), F1='', 動作='續抱', 目標金額='', 建議股數=st['shares'], 說明=st['reason']))
        for b in pl['buys']:
            rows.append(dict(區塊='F1_買進名單', 名次=b['rank_ma'], 代號=b['code'], 名稱=b['name'], SURGE=f2(b['surge'], 3), RET20pct=f2(b['ret20'] * 100, 1),
                             季線上='是', 距季線pct=f2((b['dist_ma60'] or 0) * 100, 1), 均成交值60日_萬=f2((b['v60'] or 0) / 1e4, 0), 收盤=b['close'], F1='是',
                             動作=b['action'], 目標金額=f2(b.get('target'), 0), 建議股數=b.get('shares', '') if b['action'] == '買進' else '', 說明=b.get('warn', '')))
        rows.append(dict(區塊='0050', 名次='', 代號='0050', 名稱='元大台灣50', SURGE='', RET20pct='', 季線上='', 距季線pct='', 均成交值60日_萬='', 收盤=pl['e50']['price'], F1='',
                         動作=pl['e50']['action'], 目標金額='', 建議股數=f"{pl['e50']['shares_after']:.0f}", 說明=f"底倉 {pl['band']['base_shares_new']:,.0f} 股；{pl['band']['action']}"))
    else:
        for r in res['f1']:
            rows.append(dict(區塊='F1_買進名單', 名次=r['rank_ma'], 代號=r['code'], 名稱=r['name'], SURGE=f2(r['surge'], 3), RET20pct=f2(r['ret20'] * 100, 1), 季線上='是',
                             距季線pct=f2((r['dist_ma60'] or 0) * 100, 1), 均成交值60日_萬=f2((r['v60'] or 0) / 1e4, 0), 收盤=r['close'], F1='是', 動作='', 目標金額='', 建議股數='', 說明=''))
    pd.DataFrame(rows).to_csv(str(prefix) + '.csv', index=False, encoding='utf-8-sig')
    j = {k: v for k, v in res.items() if not k.startswith('_')}
    j['plan'] = pl; j['meta'] = meta
    Path(str(prefix) + '.json').write_text(json.dumps(j, ensure_ascii=False, indent=1, default=str))


def paper_rows(res, pl, logged_on):
    out = []
    ex = res.get('exec_date') or f'{res["sig_date"]} 的次一交易日'
    base = dict(紀錄日=logged_on, 訊號月份=res['signal_month'], 訊號日=res['sig_date'], 預期成交=f'{ex} 開盤')
    for st in pl['sells']:
        out.append(dict(base, 動作='賣出', 代號=st['code'], 名稱=st.get('name', ''), 股數=st['shares'], 參考價_訊號日收盤=st.get('close', ''), 說明=st['reason']))
    for st in pl['keeps']:
        out.append(dict(base, 動作='續抱', 代號=st['code'], 名稱=st.get('name', ''), 股數=st['shares'], 參考價_訊號日收盤=st.get('close', ''), 說明=st['reason']))
    for b in pl['buys']:
        if b['action'] == '買進':
            out.append(dict(base, 動作='買進', 代號=b['code'], 名稱=b['name'], 股數=b['shares'], 參考價_訊號日收盤=b['close'], 說明=f'L_MA 第 {b["rank_ma"]} 名，目標 {b["target"]:,.0f} 元'))
    d50 = pl['e50']['shares_after'] - pl['e50']['shares_before']
    if abs(d50) >= 1:
        out.append(dict(base, 動作='買進0050' if d50 > 0 else '賣出0050', 代號='0050', 名稱='元大台灣50', 股數=int(abs(d50)), 參考價_訊號日收盤=pl['e50']['price'], 說明=pl['e50']['action']))
    out.append(dict(base, 動作='0050合計', 代號='0050', 名稱='元大台灣50', 股數=int(round(pl['e50']['shares_after'])), 參考價_訊號日收盤=pl['e50']['price'], 說明='換股後 0050 總股數（含底倉）'))
    out.append(dict(base, 動作='現金餘額', 代號='', 名稱='', 股數=int(round(pl['cash_left'])), 參考價_訊號日收盤='', 說明='估計值（元）'))
    out.append(dict(base, 動作='底倉記帳', 代號='0050', 名稱='元大台灣50', 股數=int(round(pl['band']['base_shares_new'])), 參考價_訊號日收盤=pl['e50']['price'],
                    說明=f"衛星占比 {pl['band']['satellite_frac']:.1%}（偏離 {pl['band']['dev_pp']:+.1f} 個百分點）；{pl['band']['action']}"))
    return out


LOG_COLS = ['紀錄日', '訊號月份', '訊號日', '動作', '代號', '名稱', '股數', '參考價_訊號日收盤', '預期成交', '實際開盤價', '漲跌停檢查', '開盤對參考價pct',
            '回測會不會成交', '實際成交價', '實際成交股數', '與回測差異', '說明']


def append_log(path, rows):
    path = Path(path)
    new = not path.exists()
    if not new and rows:
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        if (old['訊號月份'] == rows[0]['訊號月份']).any():
            print(f'紀錄裡已經有 {rows[0]["訊號月份"]} 的列，不重複寫入。')
            return 0
    with open(path, 'a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_COLS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in LOG_COLS})


def holdings_after(path, month, out):
    """從紀錄算出某訊號月份成交後的紙上持股（下個月 --holdings、--base-shares、--cash 用）。
    續抱照列；買進以『實際成交股數』為準，沒填時『回測會不會成交』＝會才算買到；賣出若回測不會成交（開盤跌停、沒開盤價）視為仍持有（需人工追蹤順延）。"""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df[df['訊號月份'] == month]
    hold, base, cash, pending = {}, None, 0.0, []
    for r in df.itertuples():
        a, c = r.動作, r.代號
        sh = float(r.股數) if r.股數 else 0.0
        if a == '續抱':
            hold[c] = sh
        elif a == '買進':
            if r.實際成交股數:
                hold[c] = float(r.實際成交股數)
            elif r.回測會不會成交 == '會':
                hold[c] = sh
            elif not r.回測會不會成交:
                pending.append(c)
        elif a == '賣出' and r.回測會不會成交 and r.回測會不會成交 != '會':
            hold[c] = sh; pending.append(c + '（賣出順延）')
        elif a == '0050合計':
            hold['0050'] = sh
        elif a == '底倉記帳':
            base = sh
        elif a == '現金餘額':
            cash = sh
    pd.DataFrame({'代號': list(hold), '股數': [int(v) if float(v).is_integer() else v for v in hold.values()]}).to_csv(out, index=False)
    return dict(holdings=hold, base_shares=base, cash=cash, pending=pending)


def fill_log(path, D):
    """把『預期成交』那天的實際開盤價填進去，並依回測規則判斷（開盤 ≥ 漲停價不買；開盤 ≤ 跌停價順延賣）。"""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    n = 0
    for i, r in df.iterrows():
        if r['實際開盤價'] and r['實際開盤價'] != 'NA' and r['實際成交價'] and not r['與回測差異']:
            try:                                  # 使用者填了實際成交 → 和回測（次日開盤、建議股數）比
                fp, op = float(r['實際成交價']), float(r['實際開盤價'])
                msg = f'成交價對開盤 {(fp / op - 1) * 100:+.2f}%'
                if r['實際成交股數'] and r['股數']:
                    msg += f'；股數差 {float(r["實際成交股數"]) - float(r["股數"]):+,.0f}'
                df.at[i, '與回測差異'] = msg; n += 1
            except ValueError:
                pass
            continue
        if r['實際開盤價'] or r['動作'] in ('續抱', '底倉記帳', '0050合計', '現金餘額'):
            continue
        d = r['預期成交'][:10]
        if d not in D.daily:
            continue
        day = D.days([d])[0]
        if r['代號'] == '0050':
            o = day['e50']['open']; refp = day['e50']['close'] - day['e50']['change']; lu = ld = None
        else:
            idx = np.flatnonzero(day['codes'] == r['代號'])
            if not len(idx):
                df.at[i, '實際開盤價'] = 'NA'; df.at[i, '回測會不會成交'] = '當天不在日 K（停牌或下市）'; n += 1; continue
            o, c, chg = day['vals'][idx[0], 0], day['vals'][idx[0], 1], day['vals'][idx[0], 3]
            refp = c - chg
            lu = float(np.floor(refp * 1.10 / tick(refp * 1.10) + 1e-9) * tick(refp * 1.10))
            ld = float(np.ceil(refp * 0.90 / tick(refp * 0.90) - 1e-9) * tick(refp * 0.90))
        if not np.isfinite(o) or o <= 0:
            df.at[i, '實際開盤價'] = 'NA'; df.at[i, '回測會不會成交'] = '沒有開盤價（回測：買不進；賣出順延）'; n += 1; continue
        df.at[i, '實際開盤價'] = f'{o:g}'
        ref0 = float(r['參考價_訊號日收盤']) if r['參考價_訊號日收盤'] else np.nan
        if np.isfinite(ref0) and ref0 > 0:
            df.at[i, '開盤對參考價pct'] = f'{(o / ref0 - 1) * 100:.2f}'
        if r['動作'] == '買進' and lu is not None:
            hit = o >= lu - 1e-9
            df.at[i, '漲跌停檢查'] = f'漲停價 {lu:g}' + ('（開盤漲停）' if hit else '')
            df.at[i, '回測會不會成交'] = '不會（開盤漲停不買、不遞補）' if hit else '會'
        elif r['動作'] == '賣出' and ld is not None:
            hit = o <= ld + 1e-9
            df.at[i, '漲跌停檢查'] = f'跌停價 {ld:g}' + ('（開盤跌停）' if hit else '')
            df.at[i, '回測會不會成交'] = '不會（開盤跌停順延到可賣的第一天）' if hit else '會'
        else:
            df.at[i, '回測會不會成交'] = '會'
        n += 1
    df.to_csv(path, index=False, encoding='utf-8')
    return n


def md_report(res, pl, meta):
    L = [f"# {res['signal_month']} 現行策略名單（S10 E0、W10_70；紙上追蹤）", '']
    L.append(f"- 訊號日：**{res['sig_date']}**（營收月份 {res['rev_month']}）；預期成交：{res.get('exec_date') or '訊號日次一交易日'} 開盤")
    L.append(f"- 股池 {res['n_univ']} 檔，其中有 SURGE {res['n_univ_surge']} 檔，站上季線（L_MA）{res['n_ma']} 檔")
    L.append(f"- 資料：日 K 最新 {meta['data_last_day']}；營收 {meta.get('rev_note', '')}")
    x = meta.get('exright') or {}
    if x.get('on'):
        L.append(f"- 除權息參考價修正：開啟（TWT49U 封存到 {x['twt49u_last']}；本次視窗修正 {x['n_twt49u']} 個除權息格、{x['n_prev_untraded']} 個前日無成交格）")
        if x.get('unverified'):
            L.append(f"- 請人工確認（近 60 日漲跌價差＝0、收盤有變、不在 TWT49U）：" + '、'.join(f"{u['code']} {u['date']}（{u['fixed']}）" for u in x['unverified']))
    elif x:
        L.append('- 除權息參考價修正：關閉（舊口徑）')
    dl = meta.get('deadline') or {}
    if dl:
        L.append(f"- 營收期限例外表：{'開啟' if dl.get('on') else '關閉'}（{dl.get('note', '')}）")
    rd = meta.get('reduce') or {}
    if rd.get('on'):                                      # 只有開 --reduce-fix 才會多這一行
        L.append(f"- 減資／面額變更恢復買賣日修正（候選）：開啟（事件封存到 {rd['reduce_last']}；本次視窗修正 {rd['n_reduce_parvalue']} 格，其中前 20 名／F1／持股 {len(rd.get('cells_in_top20_f1_holdings', []))} 格）")
    L.append('- 這是紙上追蹤，不下真錢單。'); L.append('')
    L.append('## L_MA 前 20 名'); L.append('')
    L.append('| 名次 | 代號 | 名稱 | SURGE | RET20 | 季線上 | 距季線 | 60 日均成交值（萬） | 收盤 | F1 |')
    L.append('|---|---|---|---|---|---|---|---|---|---|')
    f1c = {x['code'] for x in res['f1']}
    for r in res['top20']:
        L.append(f"| {r['rank_ma']} | {r['code']} | {r['name']} | {r['surge']:.3f} | {r['ret20']*100:.1f}% | {'是' if r['above_ma60'] else '否'} | {r['dist_ma60']*100:.1f}% | {r['v60']/1e4:,.0f} | {r['close']:g} | {'是' if r['code'] in f1c else ''} |")
    L.append(''); L.append(f"## F1 買進名單（前 10 名且 RET20 > 10%）：{len(res['f1'])} 檔"); L.append('')
    if pl:
        L.append(f"每檔目標＝衛星 {pl['band']['satellite']:,.0f} ÷ 10＝**{pl['unit']:,.0f} 元**（價格用訊號日收盤估，實際以次一交易日開盤為準）"); L.append('')
        L.append('| 名次 | 代號 | 名稱 | SURGE | RET20 | 收盤 | 動作 | 建議股數 | 預估金額 |'); L.append('|---|---|---|---|---|---|---|---|---|')
        for b in pl['buys']:
            L.append(f"| {b['rank_ma']} | {b['code']} | {b['name']} | {b['surge']:.3f} | {b['ret20']*100:.1f}% | {b['close']:g} | {b['action']} | {b.get('shares', '') if b['action']=='買進' else ''} | {b.get('est_value', 0):,.0f} |")
        L.append('')
        if pl['sells'] or pl['keeps']:
            L.append('## 持股：賣出／續抱'); L.append('')
            L.append('| 代號 | 名稱 | 股數 | 收盤 | 動作 | 原因 |'); L.append('|---|---|---|---|---|---|')
            for st in pl['sells'] + pl['keeps']:
                L.append(f"| {st['code']} | {st.get('name','')} | {st['shares']:,.0f} | {st.get('close') or ''} | {st['action']} | {st['reason']} |")
            L.append('')
        b = pl['band']
        L.append('## 底倉 0050 ±10 個百分點檢查'); L.append('')
        L.append(f"- 賣出後總資產約 {b['total_after_sells']:,.0f} 元；底倉 {b['base_shares']:,.0f} 股（{b['base_value']:,.0f} 元）；衛星 {b['satellite']:,.0f} 元，占 {b['satellite_frac']:.1%}（偏離 70% {b['dev_pp']:+.1f} 個百分點）")
        L.append(f"- 判定：{'在 60%～80% 內，' if b['ok'] else ''}{b['action']}")
        L.append(f"- 0050：{pl['e50']['action']}；換股後 0050 共 {pl['e50']['shares_after']:,.0f} 股，其中底倉 {b['base_shares_new']:,.0f} 股")
        for n in pl['notes']:
            L.append(f'- 注意：{n}')
    else:
        L.append('| 名次 | 代號 | 名稱 | SURGE | RET20 | 收盤 |'); L.append('|---|---|---|---|---|---|')
        for r in res['f1']:
            L.append(f"| {r['rank_ma']} | {r['code']} | {r['name']} | {r['surge']:.3f} | {r['ret20']*100:.1f}% | {r['close']:g} |")
    return '\n'.join(L) + '\n'


def read_holdings(p):
    df = pd.read_csv(p, dtype=str)
    cc = '代號' if '代號' in df.columns else 'code'
    sc = '股數' if '股數' in df.columns else 'shares'
    out = {}
    for r in df.itertuples():
        c = str(getattr(r, cc)).strip()
        if not c or c == 'nan':
            continue
        out[c] = out.get(c, 0.0) + float(str(getattr(r, sc)).replace(',', ''))
    return {c: (int(v) if float(v).is_integer() else v) for c, v in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description='現行策略（S10 E0、W10_70）每月名單產生器（紙上追蹤）')
    sub = ap.add_subparsers(dest='cmd')
    a = sub.add_parser('list')
    a.add_argument('--month', required=True, help='訊號月份，例如 2026-11（用 2026-10 營收）')
    a.add_argument('--total', type=float, help='總資產（元）；不給且有持股時＝持股市值＋--cash')
    a.add_argument('--holdings', help='目前持股 CSV（代號,股數；0050 也列）')
    a.add_argument('--cash', type=float)
    a.add_argument('--base-shares', type=float, help='底倉 0050 股數（帳上）；不給＝新開始，總資產 30%')
    a.add_argument('--repo', action='append', default=None, help='日 K／營收 git 倉庫，可多次；後面的覆蓋前面的同日檔')
    a.add_argument('--rev-csv', default=REV_DEFAULT)
    a.add_argument('--rev-cutoff', help='補充營收的時點截止（UTC ISO，例如 2026-11-11T14:00:00Z＝台北 22:00）')
    a.add_argument('--allow-partial-rev', action='store_true', help='補充營收不到上月家數 95% 時仍產生（暫定）名單')
    a.add_argument('--out-prefix')
    a.add_argument('--paper-log')
    a.add_argument('--quiet', action='store_true')
    a.add_argument('--exright-fix', dest='exright_fix', action='store_true', default=True,
                   help='除權息參考價修正（預設開啟；TWT49U 封存，見檔頭）')
    a.add_argument('--no-exright-fix', dest='exright_fix', action='store_false', help='關閉修正（舊口徑，和 2026-11-16 之前的名單逐位相同）')
    a.add_argument('--twt49u-dir', default=TWT49U_DEFAULT, help='TWT49U 封存資料夾（YYYY/YYYY-MM.csv.gz）')
    a.add_argument('--deadline-fix', dest='deadline_fix', action='store_true', default=True,
                   help='換股日套用營收期限例外表（預設開啟；cslib.deadline_exceptions()，見檔頭）')
    a.add_argument('--reduce-fix', dest='reduce_fix', action='store_true', default=False,
                   help='減資／面額變更恢復買賣日用事件參考價（候選，預設關閉；同 cslib.REDUCE_FIX）')
    a.add_argument('--reduce-csv', default=REDUCE_DEFAULT, help='減資／面額變更事件封存（date,code,prev_close,ref_price,reason）')
    a.add_argument('--no-deadline-fix', dest='deadline_fix', action='store_false', help='關閉（換股日一律 10 日規則，和 2026-11-17 之前逐位相同）')
    f = sub.add_parser('fill')
    f.add_argument('--paper-log', required=True)
    f.add_argument('--repo', action='append', default=None)
    h = sub.add_parser('holdings')
    h.add_argument('--paper-log', required=True)
    h.add_argument('--month', required=True)
    h.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    if args.cmd == 'holdings':
        r = holdings_after(args.paper_log, args.month, args.out)
        print(json.dumps(r, ensure_ascii=False))
        print(f"下個月：--holdings {args.out} --base-shares {r['base_shares']} --cash {r['cash']:.0f}" + (f"（待確認：{'、'.join(r['pending'])}）" if r['pending'] else ''))
        return 0
    repos = args.repo or [RAW_DEFAULT]
    if args.cmd == 'fill':
        D = Data(repos)
        n = fill_log(args.paper_log, D)
        print(f'已填 {n} 列（日 K 最新 {D.dates[-1]}）')
        return 0
    if args.cmd != 'list':
        ap.print_help(); return 2
    D = Data(repos, rev_csv=args.rev_csv, rev_cutoff=args.rev_cutoff)
    hold0 = read_holdings(args.holdings) if args.holdings else None
    res = compute(D, args.month, allow_partial=args.allow_partial_rev, exright=args.exright_fix, twt49u_dir=args.twt49u_dir,
                  extra_codes=[c for c in (hold0 or {}) if c != '0050'], deadline=args.deadline_fix,
                  reduce=args.reduce_fix, reduce_csv=args.reduce_csv)
    meta = dict(data_last_day=D.dates[-1], repos=[f'{r.path}@{r.head}' for r in D.repos], overlap_blob_diff=D.overlap_diff, rev=D.rev_info,
                exright=res.pop('exright', None),
                deadline=res.pop('deadline', None) or dict(on=False, note='舊口徑：換股日一律用 10 日規則，不看期限例外表'))
    if 'reduce' in res:                                   # 只有開 --reduce-fix 才有；關閉時 meta 與 make_list.py 完全相同
        meta['reduce'] = res.pop('reduce')
    sup = D.rev_info.get('supplement', {})
    meta['rev_note'] = f"tw_revenue.csv 到 {D.rev_info.get('rev_csv_last_month')}" + (('；補充：' + '、'.join(f'{k}（公開資訊觀測站彙總表，{v["n_rows"]} 筆）' for k, v in sup.items())) if sup else '')
    if res['status'] != '有訊號':
        print(json.dumps({k: v for k, v in res.items() if not k.startswith('_')}, ensure_ascii=False, indent=1))
        if args.out_prefix:
            Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out_prefix + '.json').write_text(json.dumps(dict({k: v for k, v in res.items() if not k.startswith('_')}, meta=meta), ensure_ascii=False, indent=1))
        return 1
    hold = hold0
    total = args.total
    if hold is None and total is None:
        total = 1e6
    pl = plan(res, hold, total=total, cash=args.cash, base_shares=args.base_shares)
    if args.out_prefix:
        write_outputs(res, pl, args.out_prefix, D, meta)
        Path(args.out_prefix + '.md').write_text(md_report(res, pl, meta))
    if args.paper_log:
        append_log(args.paper_log, paper_rows(res, pl, pd.Timestamp.now(tz='Asia/Taipei').strftime('%Y-%m-%d')))
    if not args.quiet:
        print(md_report(res, pl, meta))
    return 0


if __name__ == '__main__':
    sys.exit(main())
