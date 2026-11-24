#!/usr/bin/env python3
"""盤後選股表（網站資料集 screener）：每天收盤後跑一次，一檔股票一列。盤後資料、不是買賣建議、不涉及下單。

用法：
  python3 -I -B xq_scripts/site/make_screener.py <輸出 json> --repo <倉庫> [--repo ...]
          [--month 2026-11] [--extra <tsd_extra 資料夾>] [--min-amt 5000000] [--quiet]
  <輸出 json>   通常是 xq_scripts/site/data/screener.json（格式 {"rows":[...], "meta":[{"label","value"}...]}）
  --repo        官方日 K git 倉庫（可多個，由舊到新；同 preview_today.py／make_list.py）
  --month       策略欄位用的「訊號月份」（營收月份＝它的前一個月）。不給＝自動挑最新一期已到齊的月營收
                （從營收表最新月份往回試，第一個 compute 回「有訊號」的）。要和 preview_today.py 同一天的結果一致，請給同一個月份。
  --extra       自有資料快照 raw/tsd_extra（法人、融資融券、本益比等；預設倉庫內的路徑）。這些資料集的最後一天不是基準日時，該組欄位整組給 null 並在 meta 警告。
  --min-amt     20 日均成交值門檻（元），預設 500 萬。

做法（沿用既有程式，不另寫一套）：
- 策略欄位（SURGE、策略名次、入選）：直接呼叫名單產生器 make_list.py 的 Data／compute（除權息、期限例外、減資修正都開，同 preview_today.py），
  只把「訊號日」換成資料最後一天；不改 make_list.py。
- 還原價：同 make_list.compute 的口徑（還原＝含息日報酬連乘；漲跌價差先經 make_list.exright_fix_chg、reduce_fix_chg 修正），本檔只是把那幾行
  照抄一次以便多算幾個指標，並逐檔和 compute 的 RET20、距季線比對（差異寫進 meta）。
  上櫃：make_list 不修上櫃（上櫃除權息日的官方漲跌價差是空值，會被當成一般跌幅）。本檔多做一個近似：上櫃除權息／減資／面額變更日、
  漲跌價差是空值或 0 時，改用櫃買公告的參考價（raw/tsd_extra 的 otc_exright／otc_reduce／otc_parvalue；收盤落在參考價 ±10.1% 外就不修）。
  策略只看上市，這個近似不影響策略欄位。
- 股票範圍：上市＋上櫃四碼普通股（代號 [1-9]ddd，不含 91 開頭的存託憑證）；排除創新板（股名以「-創」「-KY創」結尾）、基準日無成交、
  近 20 個交易日未全程掛牌、20 日均成交值低於門檻。ETF、權證、特別股、興櫃因代號／市場不符，本來就不在內。"""
import argparse, importlib.util, io, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ML = ROOT / 'research/tw_stock_selection/agents/s18_live_list/live_surge/make_list.py'
EXTRA_DEFAULT = ROOT / 'research/tw_stock_selection/raw/tsd_extra'
W52, STREAK_MAX = 250, 60                    # 52 週＝近 250 個交易日；連買天數最多往回數 60 個交易日
COLS = ['stock_id', 'name', 'market', 'open', 'high', 'low', 'close', 'volume', 'amount', 'change', 'shares']

FIELDS = [   # （欄名, 小數位數或 None, 定義）
    ('基準日', None, '資料最後一個交易日（收盤後）'), ('代號', None, '股票代號'), ('名稱', None, '基準日的官方簡稱'), ('市場', None, '上市／上櫃'),
    ('產業', None, '官方產業別（官方日 K 倉庫 data/meta/industry.csv）'),
    ('收盤', 2, '基準日收盤價（未還原）'), ('漲幅%', 2, '基準日收盤 ÷ 當日參考價 − 1（除權息日用除權息參考價）'), ('昨漲%', 2, '前一個交易日的漲幅'),
    ('5日漲幅%', 1, '還原收盤 ÷ 5 個交易日前 − 1'), ('20日漲幅%', 1, '還原收盤 ÷ 20 個交易日前 − 1（＝策略的 RET20）'), ('成交值億', 3, '基準日成交金額（億元）'),
    ('距52週高%', 1, '收盤 ÷ 近 250 個交易日還原最高價 − 1（0＝創新高，負值＝離高點多遠）'), ('距52週低%', 1, '收盤 ÷ 近 250 個交易日還原最低價 − 1'),
    ('位階', 0, '（收盤 − 52 週低）÷（52 週高 − 52 週低）× 100，0＝最低、100＝最高；視窗內成交未滿 60 天不算'),
    ('乖離月線%', 1, '還原收盤 ÷ 20 日均線 − 1'), ('乖離季線%', 1, '還原收盤 ÷ 60 日均線 − 1（＝策略的距季線）'), ('乖離年線%', 1, '還原收盤 ÷ 240 日均線 − 1'),
    ('月線斜率%', 2, '今天的 20 日均線 ÷ 5 個交易日前的 20 日均線 − 1'), ('布林帶寬%', 1, '20 日、2 倍標準差（母體）：（上軌 − 下軌）÷ 中軌'),
    ('均量比', 2, '基準日成交股數 ÷ 近 20 個交易日（含當日）平均成交股數'), ('週轉率%', 2, '基準日成交股數 ÷ 發行股數'),
    ('本益比', 2, '官方每日本益比（虧損或無法計算＝空）'), ('殖利率%', 2, '官方每日殖利率'), ('股價淨值比', 2, '官方每日股價淨值比'),
    ('外資連買', 0, '外資連續買超天數（正＝連買、負＝連賣、0＝當日無買賣超；最多往回數 60 個交易日）'), ('投信連買', 0, '投信連續買超天數（同上）'),
    ('外資5日張', 0, '外資近 5 個交易日買賣超合計（張）'), ('投信5日張', 0, '投信近 5 個交易日買賣超合計（張）'),
    ('外資持股%', 2, '外資及陸資持股比率（官方月底表，不是每日；資料日見 meta）'),
    ('融資5日增減%', 1, '融資餘額 ÷ 5 個交易日前融資餘額 − 1'), ('券資比%', 1, '融券餘額 ÷ 融資餘額'),
    ('SURGE', 2, '月營收驚奇分數（最新一期已到齊的月營收；算法同策略。上櫃在營收主檔還沒有該月份時是空）'),
    ('站上季線', None, '還原收盤 > 60 日均線＝是'), ('策略名次', 0, '策略股池內、季線上，依 SURGE 由大到小的名次；只給前 50 名'),
    ('入選', None, '是＝策略名次前 10 名且 20 日漲幅 > 10%（策略的買進條件；只是試算，真正只看每月訊號日）'),
]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def read_panel(D, dates):
    """讀視窗內每天的日 K（四碼普通股、上市＋上櫃；篩法同 make_list.Data._parse_day），回傳 codes 與各欄 T×N 矩陣。"""
    by = {}
    for d in dates:
        rp, sha = D.daily[d]
        by.setdefault(id(rp), (rp, []))[1].append((d, sha))
    days = {}
    for rp, lst in by.values():
        got = rp.read([s for _, s in lst])
        for d, s in lst:
            x = pd.read_csv(io.BytesIO(got[s]), dtype={'stock_id': str}, usecols=COLS)
            x = x[x['stock_id'].str.fullmatch(r'[1-9]\d{3}') & x['market'].isin(['twse', 'tpex'])].drop_duplicates('stock_id', keep='last')
            days[d] = x
        del got
    codes = np.array(sorted(set(np.concatenate([days[d]['stock_id'].to_numpy() for d in dates]))))
    ci = {c: j for j, c in enumerate(codes)}
    T, N = len(dates), len(codes)
    P = {k: np.full((T, N), np.nan) for k in ('open', 'high', 'low', 'close', 'volume', 'amount', 'change', 'shares')}
    MK = np.zeros((T, N), np.int8)
    for t, d in enumerate(dates):
        x = days[d]; j = np.array([ci[c] for c in x['stock_id']])
        for k in P:
            P[k][t, j] = pd.to_numeric(x[k], errors='coerce').to_numpy(float)
        MK[t, j] = np.where(x['market'].to_numpy() == 'twse', 1, 2)
    last = days[dates[-1]]
    names = dict(zip(last['stock_id'], last['name'].astype(str).str.strip()))
    return codes, ci, P, MK, names


def otc_fix_chg(IDX, ci, C, CHG, MK, ev):
    """上櫃近似（見檔頭）：事件日、上櫃、有成交、漲跌價差是空值或 0 → 漲跌價差＝收盤 − 參考價（收盤÷參考價在 0.899～1.101 內才修）。"""
    CHG = np.array(CHG, float, copy=True); di = {d: i for i, d in enumerate(IDX)}; n = 0
    for d, c, ref in zip(ev['date'], ev['code'], ev['ref_price']):
        t, j = di.get(d), ci.get(c)
        if t is None or j is None or MK[t, j] != 2 or not (np.isfinite(C[t, j]) and C[t, j] > 0 and ref > 0):
            continue
        if (not np.isfinite(CHG[t, j]) or CHG[t, j] == 0) and 0.899 <= C[t, j] / ref <= 1.101:
            CHG[t, j] = C[t, j] - ref; n += 1
    return CHG, n


def adj_returns(C, CHG):
    """日報酬 R 與還原指數 G：逐行同 make_list.compute（「同 cslib.Panel／stlib.load」那一段）。"""
    N = C.shape[1]
    has = np.isfinite(C) & (C > 0)
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
    return R, np.cumprod(1 + R, axis=0), has


def streak(X):
    """每欄最後一天往回數，連續同號（>0 或 <0）的天數，帶正負號；最後一天是 0 → 0。"""
    s = np.sign(X); last = s[-1]; n = np.zeros(X.shape[1]); alive = last != 0
    for t in range(X.shape[0] - 1, -1, -1):
        alive = alive & (s[t] == last)
        if not alive.any():
            break
        n += alive
    return n * last


def extra_panel(L, name_pair, dates, ci, cols, warn, label):
    """讀 tsd_extra 的上市＋上櫃資料集，回傳 {欄: T×N 矩陣}；最後一天不是基準日 → None（該組欄位給 null）。"""
    try:
        df = pd.concat([L.load(n, dates[0], dates[-1], columns=['date', 'code'] + cols) for n in name_pair], ignore_index=True)
    except Exception as e:                                   # 快照缺檔等：這組欄位不做，其他照常
        warn.append(f'{label}：讀不到（{type(e).__name__}: {e}），整組欄位給空值'); return None
    df['date'] = df['date'].astype(str)
    have = set(df['date'])
    if dates[-1] not in have:
        warn.append(f'{label}：自有資料快照最新到 {max(have) if have else "無"}，不是基準日 {dates[-1]}，整組欄位給空值（先跑 raw/tsd_extra/update_daily.py）'); return None
    miss = [d for d in dates if d not in have]
    if miss:
        warn.append(f'{label}：視窗內缺 {len(miss)} 天（例 {miss[-1]}），當作 0')
    di = {d: i for i, d in enumerate(dates)}
    df = df[df['code'].isin(ci) & df['date'].isin(di)].drop_duplicates(['date', 'code'], keep='last')
    t = df['date'].map(di).to_numpy(); j = df['code'].map(ci).to_numpy(); out = {}
    for c in cols:
        M = np.full((len(dates), len(ci)), np.nan); M[t, j] = pd.to_numeric(df[c], errors='coerce').to_numpy(float); out[c] = M
    return out


def repo_file(D, path):
    """從最新的倉庫往回找一個檔，回傳 bytes（沒有＝None）。本機沒有的 blob 會由 git 向遠端補抓（已核准的來源）。"""
    for rp in reversed(D.repos):
        try:
            sha = rp.ls(str(Path(path).parent)).get(path)
            if sha:
                b = rp.read([sha]).get(sha)
                if b:
                    return b
        except Exception:
            continue
    return None


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--repo', action='append', required=True); ap.add_argument('--month')
    ap.add_argument('--extra', default=str(EXTRA_DEFAULT)); ap.add_argument('--min-amt', type=float, default=5e6); ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    sys.path.insert(0, str(ML.parent))
    m = load_module(ML, 'make_list')
    L = load_module(Path(a.extra) / 'tsd_extra_load.py', 'tsd_extra_load')
    warn = []
    D = m.Data(a.repo)
    last = len(D.dates) - 1; asof = D.dates[-1]
    m.signal_day = lambda D_, month, deadline=False: (last, None, '')      # 訊號日＝資料最後一天（同 preview_today.py）

    # ---- 策略欄位：make_list.compute
    if a.month:
        cands = [a.month]
    else:
        mx = D.surge_table().index.max()
        cands = [str(mx + 1 - i) for i in range(3)]
    res = None
    for mon in cands:
        res = m.compute(D, mon, exright=True, deadline=True, reduce=True)
        if res.get('status') == '有訊號':
            break
    strat_ok = res.get('status') == '有訊號'
    if not strat_ok:
        warn.append(f'策略欄位：{res.get("signal_month")} {res.get("status")}（{res.get("reason")}），SURGE／策略名次／入選給空值')

    # ---- 日 K 面板與還原價（視窗同 compute）
    dates = D.dates[max(0, last - m.WINDOW + 1):]
    IDX = pd.DatetimeIndex(dates)
    codes, ci, P, MK, names = read_panel(D, dates)
    C, H, Lo, VOL, AMT = P['close'], P['high'], P['low'], P['volume'], P['amount']
    T, N = C.shape; k = T - 1
    CHG, xinfo, _, _ = m.exright_fix_chg(IDX, codes, C, P['change'], MK, m.load_twt49u(m.TWT49U_DEFAULT))
    CHG, rinfo, _ = m.reduce_fix_chg(IDX, codes, C, CHG, MK, m.load_reduce(m.REDUCE_DEFAULT))
    n_otc = None
    try:
        ev = pd.concat([L.load(n, dates[0], dates[-1], columns=['date', 'code', 'ref_price']) for n in ('otc_exright', 'otc_reduce', 'otc_parvalue')], ignore_index=True)
        ev['date'] = pd.to_datetime(ev['date'].astype(str)); ev['ref_price'] = pd.to_numeric(ev['ref_price'], errors='coerce')
        CHG, n_otc = otc_fix_chg(IDX, ci, C, CHG, MK, ev.dropna(subset=['ref_price']))
    except Exception as e:
        warn.append(f'上櫃除權息參考價讀不到（{type(e).__name__}: {e}）：上櫃還原價未含股利')
    R, G, has = adj_returns(C, CHG)
    listed = MK > 0
    nlist = np.cumsum(listed[::-1], axis=0)[::-1]            # nlist[t]＝t 到最後一天掛牌的天數

    first = np.where(listed.any(axis=0), listed.argmax(axis=0), T)      # 視窗內第一次出現在日 K 的列

    def full(n):                                             # 最近 n 個交易日全程掛牌（同策略股池：視窗內都在日 K 裡）
        return (nlist[T - n] == n) if T >= n else np.zeros(N, bool)

    def seas(n):                                             # 掛牌已滿 n 個交易日（中間停牌、不在日 K 的日子還原價視為持平，同 compute 的 G）
        return (first <= T - n) if T >= n else np.zeros(N, bool)

    def ma(n):
        return np.where(seas(n), G[T - n:].mean(axis=0), np.nan) if T >= n else np.full(N, np.nan)

    with np.errstate(invalid='ignore', divide='ignore'):
        ret = lambda n: np.where(seas(n + 1), G[k] / G[k - n] - 1, np.nan) if k >= n else np.full(N, np.nan)
        ma20, ma60, ma240 = ma(20), ma(60), ma(240)
        ma20_5 = np.where(seas(25), G[T - 25:T - 5].mean(axis=0), np.nan) if T >= 25 else np.full(N, np.nan)
        sd20 = np.where(seas(20), G[T - 20:].std(axis=0, ddof=0), np.nan)
        f = G / C                                            # 還原係數（沒成交的日子是空值）
        w = slice(max(0, T - W52), T)
        hiw = np.where(listed[w] & has[w], H[w] * f[w], np.nan); low = np.where(listed[w] & has[w], Lo[w] * f[w], np.nan)
        nobs = np.isfinite(hiw).sum(axis=0)
        hh = np.where(nobs >= 60, np.nanmax(np.where(np.isfinite(hiw), hiw, -np.inf), axis=0), np.nan) / f[k]     # 換回今天的價格尺度
        ll = np.where(nobs >= 60, np.nanmin(np.where(np.isfinite(low), low, np.inf), axis=0), np.nan) / f[k]
        vol = np.where(listed, np.nan_to_num(VOL), np.nan); amt = np.where(listed, np.nan_to_num(AMT), np.nan)
        v20 = np.where(full(20), vol[T - 20:].mean(axis=0), np.nan)
        a20 = np.where(full(20), amt[T - 20:].mean(axis=0), np.nan)
        col = {
            '收盤': C[k], '漲幅%': R[k] * 100, '昨漲%': np.where(seas(2), R[k - 1] * 100, np.nan), '5日漲幅%': ret(5) * 100, '20日漲幅%': ret(20) * 100, '成交值億': AMT[k] / 1e8,
            '距52週高%': (C[k] / hh - 1) * 100, '距52週低%': (C[k] / ll - 1) * 100, '位階': np.where(hh > ll, (C[k] - ll) / (hh - ll) * 100, np.nan),
            '乖離月線%': (G[k] / ma20 - 1) * 100, '乖離季線%': (G[k] / ma60 - 1) * 100, '乖離年線%': (G[k] / ma240 - 1) * 100,
            '月線斜率%': (ma20 / ma20_5 - 1) * 100, '布林帶寬%': 4 * sd20 / ma20 * 100,
            '均量比': np.where(v20 > 0, VOL[k] / v20, np.nan), '週轉率%': np.where(P['shares'][k] > 0, VOL[k] / P['shares'][k] * 100, np.nan),
        }
        above = np.where(np.isfinite(ma60), G[k] > ma60, False)
    col['位階'] = np.clip(col['位階'], 0, 100)

    # ---- 範圍
    not91 = np.array([not c.startswith('91') for c in codes])
    innov = np.array([bool(m.INNOV_PAT.search(names.get(c, ''))) for c in codes])
    with np.errstate(invalid='ignore'):
        base = listed[k] & not91
        keep = base & ~innov & has[k] & full(20) & (a20 >= a.min_amt)
    excl = dict(創新板=int((base & innov).sum()), 基準日無成交=int((base & ~innov & ~has[k]).sum()),
                掛牌未滿20日=int((base & ~innov & has[k] & ~full(20)).sum()), 均成交值不足=int((base & ~innov & has[k] & full(20) & ~(a20 >= a.min_amt)).sum()))

    # ---- 估值、法人、信用交易（自有資料快照）
    d60 = dates[-min(STREAK_MAX, T):]
    per = extra_panel(L, ('per_twse', 'per_tpex'), dates[-1:], ci, ['per', 'yield_pct', 'pbr'], warn, '估值')
    if per:
        col.update({'本益比': per['per'][0], '殖利率%': per['yield_pct'][0], '股價淨值比': per['pbr'][0]})
    inst = extra_panel(L, ('inst_twse', 'inst_tpex'), d60, ci, ['foreign', 'trust'], warn, '法人')
    if inst:
        fo, tr = np.nan_to_num(inst['foreign']), np.nan_to_num(inst['trust'])
        col.update({'外資連買': streak(fo), '投信連買': streak(tr), '外資5日張': fo[-5:].sum(axis=0) / 1000, '投信5日張': tr[-5:].sum(axis=0) / 1000})
    mg = extra_panel(L, ('margin_twse', 'margin_tpex'), dates[-6:], ci, ['m_balance', 's_balance'], warn, '信用交易') if T >= 6 else None
    if mg:
        mb, sb = mg['m_balance'], mg['s_balance']
        with np.errstate(invalid='ignore', divide='ignore'):
            col.update({'融資5日增減%': np.where(mb[0] > 0, (mb[-1] / mb[0] - 1) * 100, np.nan), '券資比%': np.where(mb[-1] > 0, sb[-1] / mb[-1] * 100, np.nan)})

    # ---- 產業別、外資持股（官方倉庫的 meta 檔）
    ind = {}; fh = {}; fh_date = None
    b = repo_file(D, 'data/meta/industry.csv')
    if b:
        x = pd.read_csv(io.BytesIO(b), dtype=str); ind = dict(zip(x['stock_id'], x['industry_name'].fillna('')))
    else:
        warn.append('產業別：倉庫沒有 data/meta/industry.csv，產業欄給空值')
    try:
        files = {}
        for rp in D.repos:
            try:
                files.update({p: (rp, s) for p, s in rp.ls('data/meta/foreign_holding').items() if p.endswith('.csv')})
            except Exception:
                pass
        for p in sorted(files, reverse=True)[:2]:              # 最新年檔；年初檔裡還沒有資料時退一年
            rp, s = files[p]
            x = pd.read_csv(io.BytesIO(rp.read([s])[s]), dtype={'stock_id': str}, usecols=['date', 'stock_id', 'foreign_pct'])
            x = x[x['date'].astype(str) <= asof]
            if len(x):
                fh_date = str(x['date'].max()); x = x[x['date'].astype(str) == fh_date]
                fh = dict(zip(x['stock_id'], pd.to_numeric(x['foreign_pct'], errors='coerce'))); break
    except Exception as e:
        warn.append(f'外資持股：讀不到（{type(e).__name__}: {e}）')
    if not fh:
        warn.append('外資持股：沒有資料，外資持股% 給空值')

    # ---- 策略欄位對回每一檔，並和 compute 逐檔比對
    surge = np.full(N, np.nan); rank = np.full(N, np.nan); pick = np.zeros(N, bool); chk = {}
    if strat_ok:
        rci, row = res['_ci'], res['_row']
        d_ret = d_ma = 0.0; n_cmp = n_above_diff = 0
        for c, j in ci.items():
            if c not in rci:
                continue
            r = row(rci[c])
            if r['surge'] is not None:
                surge[j] = r['surge']
            if r['rank_ma'] is not None and r['rank_ma'] <= 50:
                rank[j] = r['rank_ma']
            if MK[k, j] == 1 and r['in_univ']:                # 策略股池內（上市）：本檔自己算的還原指標要和 compute 一樣
                n_cmp += 1
                if r['ret20'] is not None and np.isfinite(col['20日漲幅%'][j]):
                    d_ret = max(d_ret, abs(r['ret20'] * 100 - col['20日漲幅%'][j]))
                if r['dist_ma60'] is not None and np.isfinite(col['乖離季線%'][j]):
                    d_ma = max(d_ma, abs(r['dist_ma60'] * 100 - col['乖離季線%'][j]))
                n_above_diff += int(bool(r['above_ma60']) != bool(above[j]))
        for r in res['f1']:
            pick[ci[r['code']]] = True
        # 前 20 名：代號與名次逐一比對（compute 的 top20 對 本檔輸出列）
        bad = [f"{r['rank_ma']}.{r['code']}" for r in res['top20'] if not (keep[ci[r['code']]] and rank[ci[r['code']]] == r['rank_ma'])]
        chk = dict(top20=len(res['top20']), top20_bad=bad, n_cmp=n_cmp, d_ret=d_ret, d_ma=d_ma, n_above_diff=n_above_diff)

    # ---- 輸出
    dec = {f: d for f, d, _ in FIELDS}

    def num(f, j):
        v = col[f][j] if f in col else np.nan
        if v is None or not np.isfinite(v):
            return None
        v = round(float(v), dec[f])
        return int(v) if dec[f] == 0 else v

    rows = []
    for j in np.flatnonzero(keep):
        c = str(codes[j])
        r = {'基準日': asof, '代號': c, '名稱': names.get(c, ''), '市場': '上市' if MK[k, j] == 1 else '上櫃', '產業': ind.get(c) or None}
        for f, d, _ in FIELDS:
            if d is not None and f not in ('SURGE', '策略名次', '外資持股%'):
                r[f] = num(f, j)
        v = fh.get(c)
        r['外資持股%'] = None if v is None or not np.isfinite(v) else round(float(v), 2)
        r['SURGE'] = None if not np.isfinite(surge[j]) else round(float(surge[j]), 2)
        r['站上季線'] = None if not np.isfinite(ma60[j]) else ('是' if above[j] else '否')
        r['策略名次'] = None if not np.isfinite(rank[j]) else int(rank[j])
        r['入選'] = '是' if pick[j] else ''
        rows.append({f: r[f] for f, _, _ in FIELDS})
    rows.sort(key=lambda r: (-(r['成交值億'] or 0), r['代號']))

    mk_n = pd.Series([r['市場'] for r in rows]).value_counts().to_dict()
    meta = [{'label': '產生程式', 'value': 'xq_scripts/site/make_screener.py'}, {'label': '基準日', 'value': asof}, {'label': '檔數', 'value': len(rows)},
            {'label': '上市檔數', 'value': int(mk_n.get('上市', 0))}, {'label': '上櫃檔數', 'value': int(mk_n.get('上櫃', 0))},
            {'label': '營收月份', 'value': str(res.get('rev_month')) if strat_ok else None}, {'label': '策略狀態', 'value': str(res.get('status'))},
            {'label': '外資持股資料日', 'value': fh_date},
            {'label': '說明', 'value': '盤後資料，每個交易日收盤後更新一次，不是即時報價，也不是買賣建議。策略只在每月訊號日收盤後決定買賣；表內的策略名次、入選是把基準日當成訊號日的試算。'},
            {'label': '股票範圍', 'value': '上市＋上櫃四碼普通股（代號 1000～9999，不含 91 開頭的存託憑證）。ETF、權證、特別股、興櫃不在內。'},
            {'label': '排除條件', 'value': f'創新板（{excl["創新板"]} 檔）、基準日無成交（{excl["基準日無成交"]} 檔）、近 20 個交易日未全程掛牌（{excl["掛牌未滿20日"]} 檔）、20 日均成交值低於 {a.min_amt / 1e4:,.0f} 萬元（{excl["均成交值不足"]} 檔）。'},
            {'label': '成交值門檻（元）', 'value': a.min_amt},
            {'label': '還原價', 'value': '含息日報酬連乘（同 make_list.compute）；上市除權息、減資用 make_list.exright_fix_chg／reduce_fix_chg'
                                       f'（視窗內 TWT49U {xinfo["n_twt49u"]} 格、減資 {rinfo["n_reduce_parvalue"]} 格）；'
                                       f'上櫃用櫃買參考價近似（{n_otc if n_otc is not None else "未做"} 格，make_list 本身不修上櫃）。均線、漲幅、52 週高低都用還原價，收盤是未還原價。'},
            {'label': '資料來源', 'value': '官方日 K 倉庫（data/universe/daily、data/meta/industry.csv、data/meta/foreign_holding）；月營收 tw_revenue.csv＋倉庫 data/mops/revenue_hist；'
                                         '自有資料快照 raw/tsd_extra（per_*、inst_*、margin_*、otc_*）；證交所 TWT49U 與減資事件封存。'},
            {'label': '缺值規則', 'value': '掛牌未滿所需天數（均線 20／60／240 日、漲幅 N+1 日）的欄位是空值；中間停牌、不在日 K 的日子還原價視為持平（同策略）。52 週高低只看有成交的日子，未滿 60 天不算。'},
            {'label': '沒有的欄位', 'value': '當沖比（沒有逐股當沖資料）。外資持股是月底表，不是每日。'}]
    if strat_ok:
        meta.append({'label': '策略欄位檢查', 'value': f'前 20 名代號與名次對 make_list.compute：{chk["top20"] - len(chk["top20_bad"])}/{chk["top20"]} 相符'
                                                      + (f'（不符：{"、".join(chk["top20_bad"])}）' if chk['top20_bad'] else '')
                                                      + f'；策略股池 {chk["n_cmp"]} 檔的 20 日漲幅最大差 {chk["d_ret"]:.2e}、距季線最大差 {chk["d_ma"]:.2e}（百分點）、站上季線不同 {chk["n_above_diff"]} 檔'})
        meta.append({'label': '策略前20名', 'value': '、'.join(f"{t['rank_ma']}.{t['code']}{t['name']}" for t in res['top20'])})
    if warn:
        meta.append({'label': '警告', 'value': '；'.join(warn)})
    meta += [{'label': '欄位：' + f, 'value': d} for f, _, d in FIELDS]
    body = '{"rows":[\n' + ',\n'.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) for r in rows) + '\n],\n"meta":' + json.dumps(meta, ensure_ascii=False, indent=1, default=str) + '}\n'
    json.loads(body)                                         # 確認寫出去的是合法 JSON（NaN 等會在這裡出錯）
    Path(a.out).write_text(body, encoding='utf-8')
    if not a.quiet:
        for e in [e for e in meta if not e['label'].startswith('欄位：')]:
            print(f"{e['label']}：{e['value']}")
        print(f'輸出 {a.out}：{len(body.encode()) / 1024:.0f} KB，{time.time() - t0:.1f} 秒')
    return 1 if (strat_ok and (chk['top20_bad'] or chk['n_above_diff'])) else 0


if __name__ == '__main__':
    sys.exit(main())
