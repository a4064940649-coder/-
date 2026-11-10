# 名單層（clean-room）：V4 規格套 V7 面板＋P；官方日／j 日；AB（自訂亂數）；B10 放回
import numpy as np, pandas as pd, json, math, os
import xdata
AG = xdata.AG; D7 = xdata.D7
SL = '/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv'
RA = AG + 'rev_announce/'

class Lists:
    def __init__(self):
        X = xdata.load(); self.X = X
        self.dates = X['dates']; self.codes = X['codes']; N = len(self.codes); T = len(self.dates)
        self.ci = {c: j for j, c in enumerate(self.codes)}
        # v3：52 週比用自算還原價（G 連乘；o4 FINDINGS 366 行「自算還原價」）
        G = X['G']; TR = X['TR']
        A2 = np.where(TR, np.cumprod(G, axis=0), np.nan)
        gi = np.where(X['GOOD'])[0]
        Ag = A2[gi]
        exg = np.isin(self.dates[gi], pd.DatetimeIndex(xdata.EXEMPT))  # 凍結後修正 F6：gapfix 豁免日
        okA = pd.DataFrame((np.isnan(Ag) & ~exg[:, None]).astype(np.int16)).rolling(252, min_periods=252).sum().values == 0
        mx = np.where(okA, pd.DataFrame(Ag).rolling(252, min_periods=1).max().values, np.nan)
        r52g = Ag / mx
        R52 = np.full((T, N), np.nan); R52[gi] = r52g
        self.R52 = R52
        # 名稱與產業
        sl = pd.read_csv(open(SL), dtype={'code': str})
        self.name = dict(zip(sl.code, sl.name)); self.ind = dict(zip(sl.code, sl.industry))
        tr = pd.read_csv(open('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv'), dtype=str, usecols=lambda c: c in ('code', 'stock_id', 'name', 'company_name', 'industry'))
        self.tr_cols = list(tr.columns)
        idc = 'code' if 'code' in tr.columns else ('stock_id' if 'stock_id' in tr.columns else None)
        nmc = 'name' if 'name' in tr.columns else ('company_name' if 'company_name' in tr.columns else None)
        self.names_all = {c: set() for c in self.codes}
        for c, n in self.name.items():
            if c in self.names_all: self.names_all[c].add(str(n))
        if idc and nmc:
            for c, n in tr[[idc, nmc]].drop_duplicates().itertuples(index=False):
                if c in self.names_all: self.names_all[c].add(str(n))
        dl = pd.read_csv(open(AG + 'o_infra_v3_rebuild/delist_official.csv'), dtype={'code': str})
        for c, n in zip(dl.code, dl.name):
            if c in self.names_all: self.names_all[c].add(str(n))
        mc = pd.read_csv(open(AG + 'o_infra_v3_rebuild/merger_consideration.csv'), dtype={'code': str})
        for c, n in zip(mc.code, mc.name):
            if c in self.names_all: self.names_all[c].add(str(n))
        self.dl = dl
        fin_codes = set(sl[sl.industry.astype(str).str.contains('金融')].code)
        self.FIN = np.array([(c in fin_codes) or c.startswith('28') or c in ('5876', '5880', '5854', '6004', '6012') for c in self.codes])
        def foreign(c):
            ns = self.names_all.get(c, set())
            return c.startswith('91') or any(('KY' in n.upper()) or ('DR' in n.upper()) or n.startswith('F-') for n in ns)
        self.FOR = np.array([foreign(c) for c in self.codes])
        # 營收
        self.RF = xdata.rpq(AG + 'o_v4_revenue_patch_official/yoy_long_R_FIX.parquet')
        self.Y5 = xdata.rpq(AG + 'o_v4_revenue_patch_official/yoy_rev5y_FIX.parquet')
        # 處置（V4 官方）
        self.disp = xdata.disposal_v4()
        # 終止上市 gap 名單（4 碼、非外國、非金融）
        self.term = {}
        for c, d in zip(dl.code, dl.date):
            if c in self.ci and len(c) == 4:
                j = self.ci[c]
                if not self.FOR[j] and not self.FIN[j]:
                    self.term[j] = max(self.term.get(j, pd.Timestamp('1900-01-01')), pd.Timestamp(d))
        self.cal = pd.read_csv(open(RA + 'deadline_calendar.csv'))
        if '2023-12' not in set(self.cal.rev_month):  # 凍結後修正 F2：補 2023-12 營收月（L01 規則）
            tx = pd.read_csv(open('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv'), parse_dates=[0]).iloc[:, 0]
            txi = pd.DatetimeIndex(sorted(set(tx) | set(self.dates[self.dates >= pd.Timestamp('2024-01-01')])))
            Dd = txi[txi >= pd.Timestamp('2024-01-10')][0]; Sd = self.dates[(self.dates >= pd.Timestamp('2024-01-11'))][0]
            D15 = txi[txi >= pd.Timestamp('2024-01-15')][0]
            row = dict(rev_month='2023-12', signal_day=str(Sd.date()), tenth_weekday=pd.Timestamp('2024-01-10').day_name(), tenth_is_trading=bool(pd.Timestamp('2024-01-10') in set(txi)),
                       deadline_D=str(Dd.date()), case='D<S' if Dd < Sd else 'D=S', D15=str(D15.date()), ins15_case=np.nan, special=np.nan)
            self.cal = pd.concat([self.cal, pd.DataFrame([row])], ignore_index=True).sort_values('rev_month').reset_index(drop=True)
        pdj = json.load(open(RA + 'p_delay.json'))
        self.pind = pdj['p_by_industry']; self.pall = pdj['p_overall']; self.ins15 = set(pdj['INS15'])
        fs = pd.read_csv(open(RA + 'firstseen_2026-09.csv'), dtype=str)
        fs.columns = ['code'] + list(fs.columns[1:])
        self.fs_before = set(fs[fs['cat'] == 'A_before'].code)

    def y3(self, d, months_back):
        """d：訊號日；months_back：(1,2,3) 或 (2,3,4)；回傳每檔 y3（缺任一月→NaN）"""
        m = pd.Period(d, 'M')
        tab = self.RF if d < pd.Timestamp('2021-03-01') else self.Y5
        vals = []
        for k in months_back:
            key = str(m - k)
            if key in tab.index: vals.append(tab.loc[key].reindex(self.codes).values.astype(float))
            else: vals.append(np.full(len(self.codes), np.nan))
        v = np.vstack(vals)
        return np.where(np.isnan(v).any(0), np.nan, v.mean(0))

    def disp_mask(self, d, e):
        u = self.disp
        mk = (u.announce <= d) & (u.start <= e) & (u.end >= e)
        out = np.zeros(len(self.codes), bool)
        for c in u.code[mk]:
            j = self.ci.get(c)
            if j is not None: out[j] = True
        return out

    def pick(self, si, unavail, held, b10=True, n=10):
        X = self.X; d = self.dates[si]; ei = si + 1; e = self.dates[ei]
        V60 = X['V60'][si]; C = X['C'][si]
        elig = (V60 >= 1e7) & (C >= 10)
        order = np.argsort(-np.where(elig, V60, -np.inf), kind='stable')
        top = np.zeros(len(self.codes), bool); top[order[:300]] = True; top &= elig
        r52 = self.R52[si]
        base = top & ~np.isnan(r52) & np.isfinite(X['O'][ei]) & X['TR'][ei] & ~self.FIN & ~self.FOR & ~self.disp_mask(d, e)
        y = self.y3(d, (1, 2, 3))
        if unavail is not None and unavail.any():
            y2 = self.y3(d, (2, 3, 4)); y = np.where(unavail, y2, y)
        pool = base & ~np.isnan(y)
        A = np.zeros(len(self.codes), bool)
        if b10:
            for j, td in self.term.items():
                if base[j] and np.isnan(y[j]) and d <= td: A[j] = True
        el = pool | A
        idx = np.where(el)[0]; Nn = len(idx)
        if Nn < 50: return None  # 合格股 < 50 檔的換股日跳過（o4 FINDINGS 375 行）
        r = pd.Series(r52[idx]).rank(method='average').values / Nn
        isA = A[idx]
        yv = y[idx]
        ry = np.ones(Nn)
        if (~isA).any(): ry[~isA] = pd.Series(yv[~isA]).rank(method='average').values / (int((~isA).sum()) if os.environ.get('G6_Y3PCT') == 'rev' else Nn)
        sc = np.round(0.5 * r + 0.5 * ry, 10)
        ytie = np.where(isA, np.inf, yv)
        o = np.lexsort((idx, -ytie, -sc))
        ranked = idx[o]
        picks = []
        UP = X['UP'][ei]; O = X['O'][ei]
        for j in ranked:
            if len(picks) >= n: break
            if (j not in held) and O[j] >= UP[j] - 1e-6: continue
            picks.append(int(j))
        return picks

    def jdays(self, j):
        """v2：J＝deadline_D（10 日起第一個加權指數交易日）之後第 j 個「加權指數交易日」，再對到面板日；
        次一面板日不是 GOOD 日的月份跳過（o_rev_freshness_attention FINDINGS 29 行）；超出面板不交易"""
        if not hasattr(self, 'tx'):
            tx = pd.read_csv(open('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv'), parse_dates=[0]).iloc[:, 0]
            self.tx = pd.DatetimeIndex(sorted(set(tx) | set(self.dates[self.dates >= pd.Timestamp('2024-01-01')])))
        out = []
        dv = self.dates.values; tv = self.tx.values; G = self.X['GOOD']
        for r in self.cal.itertuples():
            D = pd.Timestamp(r.deadline_D)
            k = int(np.searchsorted(tv, np.datetime64(D), 'left')) + j
            if k >= len(tv): continue
            i = int(np.searchsorted(dv, tv[k], 'left'))
            if i + 1 >= len(self.dates): continue
            if not G[i + 1] or not G[i]: continue
            out.append((r.rev_month, i))
        return out

    def unavail(self, rev_month, si, rng, mode='mc'):
        d = self.dates[si]
        r = self.cal[self.cal.rev_month == rev_month].iloc[0]
        D = pd.Timestamp(r.deadline_D)
        if isinstance(r.special, str) and '2019-02-15' in r.special: D = pd.Timestamp('2019-02-15')
        if isinstance(r.special, str) and '2023-04-12' in r.special: D = pd.Timestamp('2023-04-12')
        N = len(self.codes)
        if rev_month == '2026-09':
            return np.array([c not in self.fs_before for c in self.codes])
        u = np.zeros(N, bool)
        if d < D: u[:] = True
        elif d == D or (rev_month == '2020-01' and d <= pd.Timestamp('2020-02-14')):
            if mode == 'worst': u[:] = True
            else:
                p = np.array([1.0 if self.FIN[k] else self.pind.get(self.ind.get(c, ''), self.pall) for k, c in enumerate(self.codes)])
                u = rng.random(N) < p
        if rev_month >= '2026-01' and rev_month <= '2026-08' and d < pd.Timestamp(r.D15):
            for c in self.ins15:
                if c in self.ci: u[self.ci[c]] = True
        return u

    def plan(self, days, seed=None, b10=True, mode='mc'):
        rng = np.random.default_rng(10_000 + (seed if seed is not None else 0))
        held = set(); out = []
        for rm, si in days:
            un = self.unavail(rm, si, rng, mode) if seed is not None else None
            pk = self.pick(si, un, held, b10=b10)
            if pk is None: continue
            out.append((si, si + 1, pk)); held = set(pk)
        return out

    def official_days(self):
        dv = self.dates.values; out = []
        for r in self.cal.itertuples():
            m = pd.Period(r.rev_month, 'M') + 1
            cand = np.where((self.dates.to_period('M') == m) & (self.dates.day >= 11))[0]
            if len(cand) == 0: continue
            si = int(cand[0])
            if self.dates[si] >= pd.Timestamp('2026-10-12') or si + 1 >= len(self.dates): continue
            out.append((r.rev_month, si))
        return out

if __name__ == '__main__':
    L = Lists()
    od = L.official_days(); print('official days', len(od), L.dates[od[0][1]].date(), L.dates[od[-1][1]].date(), 'tr_cols', L.tr_cols)
    p = L.plan(od, seed=None)
    print('slots', sum(len(x[2]) for x in p), 'short days', sum(len(x[2]) < 10 for x in p))
