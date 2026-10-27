#!/usr/bin/env python3
"""主策略（52 週高點 × 月營收 3 個月年增率，可成交版 M2）每月下單清單。

規則（與 agents/exec_rebuild/exlib.py 的 selection('M2', n) 相同）：
  - 換股日＝每月第一個「日期 ≥ 11 日」的交易日。當天收盤後算訊號，隔一個交易日開盤買。
  - 股池：只做上市股；60 日平均成交值 ≥ 1,000 萬，且排在前 300 名；收盤 ≥ 10 元；換股日當天不在處置期間。
  - 分數＝0.5 ×「收盤／52 週（252 日）最高還原收盤」的百分位＋0.5 ×「近 3 個月營收年增率平均」的百分位。
    營收只用公告期限（次月 10 日）已過的月份，也就是換股日的前一個月往回 3 個月。
  - 等權買前 N 名。隔日開盤漲停（還原開盤 ÷ 前收 − 1 ≥ 9.4%）而且還沒持有的不買，由下一名遞補；
    已持有的照常續抱。
資料：history.csv.gz（官方日 K，2025-09-24 起）只要在訊號日前已有 252 個交易日就單獨使用；
      不夠時與研究面板（BRIEF 的 scratchpad/panel）拼接。面板不存在時，先執行 research/tw_stock_selection/restore_env.sh。
      月營收用 extras/rev_5y.csv.gz，缺的月份由 extras/rev_hist.csv.gz 補；處置用 extras/disposal_5y.csv.gz；名稱與市場用 stock_list.csv。
用法：python3 -I main_picks_today.py --data <tw-stock-screener>/data [--date YYYY-MM-DD] [--n 10] [--equity 1000000] [--held 2330,2317]
資料為第三方整理，下單前請以公開資訊觀測站（月營收）與證交所公告（處置）核對。
"""
import argparse, math
from pathlib import Path
import numpy as np, pandas as pd

PANEL = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
W52, W60, TOPN, MIN_VAL, MIN_PX = 252, 60, 300, 1e7, 10.0
LIMIT, LTOL = 0.10, 0.006


def is4(c):
    return isinstance(c, str) and len(c) == 4 and c.isdigit() and c[0] != '0'


# ---------------- 資料 ----------------
def hist_returns(C, O, ex):
    """history.csv 的日報酬，規則同 build_twse_panel.py 的非官方來源（src 4）：
    R = 收盤 / 前收 × F − 1（F＝兩筆觀測間除權息的 前收／參考價 連乘）；開盤偏離前收 > 10.5% 且無除權息 → 公司行動，R = 收盤 / 開盤 − 1；
    超過漲跌幅 ^ 間隔天數 + 1% 視為資料錯誤（缺值，還原指數視為 0）。"""
    cal = C.index; R = np.full(C.shape, np.nan)
    exg = {c: g for c, g in ex.groupby('code')}
    Cn, On = C.to_numpy(), O.to_numpy()
    for j, code in enumerate(C.columns):
        m = np.isfinite(Cn[:, j])
        if m.sum() < 2: continue
        pos = np.flatnonzero(m); idx = cal[pos]; c = Cn[pos, j]; o = On[pos, j]; N = len(c)
        prev = np.r_[np.nan, c[:-1]]; nskip = np.r_[0, np.diff(pos) - 1]
        F = np.ones(N)
        g = exg.get(code)
        if g is not None:
            for ed, f in zip(g['date'], g['f']):
                k = idx.searchsorted(ed)
                if 0 < k < N: F[k] *= f
        obs = np.arange(N) + (5 if pos[0] == 0 else 0)     # 資料第一天就有價者視為已上市多時（非新股）
        with np.errstate(divide='ignore', invalid='ignore'):
            r = c / prev * F - 1
            dev = np.abs(o / prev - 1)
            ca = (F == 1) & (obs >= 5) & np.isfinite(dev) & (dev > LIMIT + 0.005) & (o > 0)
            r[ca] = (c / o - 1)[ca]
            n = nskip + 1; ipo = obs < 5
            bad = (~ipo & ((r > (1 + LIMIT) ** n - 1 + 0.01) | (r < -(1 - (1 - LIMIT) ** n) - 0.01))) | (ipo & ((r > 1.5) | (r < -0.7)))
        r[bad] = np.nan; r[0] = np.nan
        R[pos, j] = r
    return pd.DataFrame(R, index=cal, columns=C.columns)


def load_prices(D, target, panel_dir=PANEL):
    """回傳 dict(C, O, H, V, R, ADJ, cal, mode, hist_start, data_end)。"""
    lst = pd.read_csv(D / 'stock_list.csv', dtype=str)
    twse = {c for c in lst.loc[lst['market'] == 'TWSE', 'code'] if is4(c)}
    panel_ok = (panel_dir / 'adj.parquet').exists()
    pcols = set(pd.read_parquet(panel_dir / 'close_raw.parquet').columns) if panel_ok else set()
    codes = twse | pcols
    h = pd.read_csv(D / 'history.csv.gz', dtype={'code': str}, parse_dates=['date'])
    h = h[h['code'].isin(codes) & (h['volume'] > 0) & (h['close'] > 0)].drop_duplicates(['date', 'code'], keep='last')
    h['val'] = h['volume'] * h['close']
    piv = {k: h.pivot(index='date', columns='code', values=v).sort_index().astype(float)
           for k, v in [('C', 'close'), ('O', 'open'), ('H', 'high'), ('V', 'val')]}
    cols = sorted(piv['C'].columns)
    piv = {k: v.reindex(columns=cols) for k, v in piv.items()}
    hs, he = piv['C'].index[0], piv['C'].index[-1]
    ex = pd.read_csv(D / 'extras/exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
    ex = ex[ex['code'].map(is4) & (ex['ref_price'] > 0) & (ex['prev_close'] > 0)].copy()
    ex['f'] = ex['prev_close'] / ex['ref_price']; ex = ex[(ex['f'] >= 1.0) & (ex['f'] < 3.0)]
    Rh = hist_returns(piv['C'], piv['O'], ex)
    hcal = piv['C'].index
    # 訊號日（含）之前需要 252 個交易日（52 週高點）；history 不夠就和面板拼接
    t_h = hcal[hcal <= target]
    sd_h = signal_day(hcal, t_h[-1]) if len(t_h) else None
    need_panel = sd_h is None or hcal.get_loc(sd_h) < W52 - 1
    out = dict(hist_start=hs, data_end=he)
    if not need_panel:
        out.update(C=piv['C'], O=piv['O'], H=piv['H'], V=piv['V'], R=Rh, mode='history.csv.gz 單獨使用')
    else:
        if not panel_ok:
            raise SystemExit(f'history.csv.gz 從 {hs.date()} 開始，到 {target.date()} 不足 52 週（252 個交易日），需要與研究面板拼接，\n'
                             f'但找不到面板 {panel_dir}。請先執行：bash research/tw_stock_selection/restore_env.sh')
        P = {k: pd.read_parquet(panel_dir / f'{f}.parquet') for k, f in
             [('C', 'close_raw'), ('O', 'open'), ('H', 'high'), ('V', 'value'), ('R', 'ret')]}
        pe = P['C'].index[-1]
        allc = sorted(set(cols) | set(P['C'].columns))
        M = {}
        for k in ['C', 'O', 'H', 'V']:
            M[k] = pd.concat([P[k].loc[P[k].index < hs].reindex(columns=allc), piv[k].reindex(columns=allc)])
        M['R'] = pd.concat([P['R'].reindex(columns=allc), Rh.loc[Rh.index > pe].reindex(columns=allc)])
        M['R'] = M['R'].reindex(M['C'].index)
        out.update(M, mode=f'研究面板（≤ {pe.date()} 的報酬）＋ history.csv.gz（{hs.date()} 起的價量）拼接')
    out['ADJ'] = (1 + out['R'].fillna(0)).cumprod().where(out['C'].notna())
    out['cal'] = out['C'].index
    out['names'] = lst.set_index('code')['name']
    return out


def load_revenue(D):
    r5 = pd.read_csv(D / 'extras/rev_5y.csv.gz', dtype={'code': str})
    rh = pd.read_csv(D / 'extras/rev_hist.csv.gz', dtype={'code': str}) if (D / 'extras/rev_hist.csv.gz').exists() else r5.iloc[:0]
    k5 = set(zip(r5['code'], r5['ym']))
    add = rh[[(c, y) not in k5 for c, y in zip(rh['code'], rh['ym'])]]
    rev = pd.concat([r5[['code', 'ym', 'yoy']], add[['code', 'ym', 'yoy']]], ignore_index=True)
    rev['p'] = pd.PeriodIndex(rev['ym'], freq='M')
    Y = rev.pivot_table(index='p', columns='code', values='yoy', aggfunc='last').sort_index()
    Y = Y.reindex(pd.period_range(Y.index.min(), Y.index.max(), freq='M'))
    return Y, Y.rolling(3, min_periods=3).mean(), len(add)


def load_disposal(D):
    return pd.read_csv(D / 'extras/disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])


# ---------------- 日曆 ----------------
def rebalance_day(cal, y, m):
    d = cal[(cal.year == y) & (cal.month == m) & (cal.day >= 11)]
    return d[0] if len(d) else None


def est_rebalance_day(y, m):
    d = pd.Timestamp(year=y, month=m, day=11)
    while d.weekday() >= 5: d += pd.Timedelta(days=1)
    return d


def signal_day(cal, t):
    """t 當天或之前最近的換股日；資料中找不到時回傳 None。"""
    rd = rebalance_day(cal, t.year, t.month)
    if rd is not None and rd <= t: return rd
    p = pd.Period(t, 'M') - 1
    return rebalance_day(cal, p.year, p.month)


def next_rebalance(cal, t):
    """t 之後的下一個換股日；資料日曆還沒涵蓋時用「第一個 ≥ 11 日的平日」預估（第二個值為 True）。"""
    for k in (0, 1, 2):
        p = pd.Period(t, 'M') + k
        rd = rebalance_day(cal, p.year, p.month)
        if rd is not None:
            if rd > t: return rd, False
            continue
        e = est_rebalance_day(p.year, p.month)
        if e > t: return e, True
    return None, True


# ---------------- 選股 ----------------
def score_on(data, YOY3, dsp, sd):
    """換股日 sd 收盤的股池與分數（pd.DataFrame，依分數由高到低排序）。"""
    cal = data['cal']; i = cal.get_loc(sd); cols = data['C'].columns
    adj = data['ADJ'].iloc[i - W52 + 1:i + 1]
    s52 = (adj.iloc[-1] / adj.max()).where(adj.notna().all()) if i >= W52 - 1 else pd.Series(np.nan, index=cols)
    v = data['V'].iloc[i - W60 + 1:i + 1]
    v60 = v.mean().where(v.notna().all()) if i >= W60 - 1 else pd.Series(np.nan, index=cols)
    px = data['C'].iloc[i]
    base = (v60 >= MIN_VAL) & (px >= MIN_PX)
    t300 = base & (v60.where(base).rank(ascending=False) <= TOPN)
    m = pd.Period(sd, 'M') - 1
    y3 = YOY3.reindex([m]).iloc[0].reindex(cols) if m in YOY3.index else pd.Series(np.nan, index=cols)
    inD = pd.Series(False, index=cols)
    for _, r in dsp[(dsp['start'] <= sd) & (dsp['end'] >= sd)].iterrows():
        if r['code'] in inD.index: inD[r['code']] = True
    el = t300 & y3.notna() & s52.notna() & ~inD
    sc = 0.5 * s52.where(el).rank(pct=True) + 0.5 * y3.where(el).rank(pct=True)
    df = pd.DataFrame({'s52': s52, 'yoy3': y3, 'score': sc, 'close': px, 'v60': v60, 'el': el, 't300': t300, 'inD': inD})
    df.index.name = 'code'
    order = df[df['el']].reset_index().sort_values(['score', 'code'], ascending=[False, True])
    return order.set_index('code'), df, m, int(el.sum())


def execute(data, order, sd, n, held):
    """若訊號日隔日的開盤資料已存在，套用 M2：隔日沒有開盤價（停牌）不買；開盤漲停且未持有不買、由下一名遞補。"""
    cal = data['cal']; i = cal.get_loc(sd)
    if i + 1 >= len(cal): return None
    nd = cal[i + 1]
    o = data['O'].loc[nd].reindex(order.index); c1 = data['C'].loc[nd].reindex(order.index)
    a1 = data['ADJ'].loc[nd].reindex(order.index); a0 = data['ADJ'].ffill().iloc[i].reindex(order.index)
    with np.errstate(divide='ignore', invalid='ignore'):
        gap = o * (a1 / c1) / a0 - 1
    trad = o.notna() & a1.notna()
    picks, skipped = [], []
    for code in order.index:
        if not trad[code]: skipped.append((code, '隔日無開盤價（停牌）')); continue
        if gap[code] >= LIMIT - LTOL and code not in held:
            skipped.append((code, f'隔日開盤漲停（{gap[code]:+.1%}）')); continue
        picks.append(code)
        if len(picks) == n: break
    return dict(next_day=nd, picks=picks, skipped=skipped, gap=gap, open=o)


def pick_list(data, YOY3, dsp, sd, n, held=()):
    """供驗證用：回傳 M2 實際買進名單（隔日資料存在時）或前 n 名。"""
    order, _, _, _ = score_on(data, YOY3, dsp, sd)
    ex = execute(data, order, sd, n, set(held))
    return ex['picks'] if ex else list(order.index[:n])


# ---------------- 輸出 ----------------
def fmt_table(order, codes, data, price, price_day, alloc, dsp, extra=None):
    rows = []
    for code in codes:
        r = order.loc[code]; p = price.get(code, np.nan)
        if not np.isfinite(p): p = r['close']
        note = []
        up = dsp[(dsp['code'] == code) & (dsp['start'] > price_day) & (dsp['start'] <= price_day + pd.Timedelta(days=7))]
        if len(up): note.append(f'{up["start"].min().date()} 起處置（需預收款券）')
        if extra and code in extra: note.append(extra[code])
        row = {'名次': int(order.index.get_loc(code)) + 1, '代號': code, '名稱': data['names'].get(code, ''),
               '52週高點比': f'{r["s52"]:.1%}', '營收3月年增': f'{r["yoy3"]:+.1f}%', '綜合分數': f'{r["score"]:.3f}', f'收盤{price_day:%m/%d}': f'{p:g}'}
        if alloc:
            row['整張(張)'] = int(alloc // (p * 1000)); row['零股(股)'] = int(alloc // p)
        row['備註'] = '；'.join(note)
        rows.append(row)
    return pd.DataFrame(rows).to_string(index=False) if rows else '（無）'


def tie_note(order, picks, nxt):
    if picks and nxt and abs(order.loc[picks[-1], 'score'] - order.loc[nxt[0], 'score']) < 1e-12:
        print(f'注意：{picks[-1]} 與 {nxt[0]} 綜合分數相同（{order.loc[picks[-1], "score"]:.4f}），依代號小者優先入選。'
              '（回測程式 exlib 的同分順序取決於 numpy 排序，可能選另一檔；兩者擇一皆符合規則）')


def main():
    ap = argparse.ArgumentParser(description='主策略 M2 每月下單清單')
    ap.add_argument('--data', required=True, help='tw-stock-screener/data 目錄')
    ap.add_argument('--date', help='查詢日期 YYYY-MM-DD（預設＝資料最後一天）')
    ap.add_argument('--n', type=int, default=10, help='持股檔數（預設 10）')
    ap.add_argument('--equity', type=float, help='主策略部位淨值（元），用來算建議股數')
    ap.add_argument('--held', default='', help='目前持股代號，逗號分隔，例如 2330,2317')
    ap.add_argument('--panel', default=str(PANEL), help='研究面板目錄（history 不足 52 週時拼接用）')
    a = ap.parse_args()
    D = Path(a.data); n = a.n
    held = [c.strip() for c in a.held.replace('，', ',').split(',') if c.strip()]
    hcal = pd.read_csv(D / 'history.csv.gz', usecols=['date'], parse_dates=['date'])['date']
    data_end = hcal.max()
    want = pd.Timestamp(a.date) if a.date else data_end
    data = load_prices(D, min(want, data_end), Path(a.panel))
    cal = data['cal']
    YOY, YOY3, n_fill = load_revenue(D)
    dsp = load_disposal(D)
    if want > data_end:
        print(f'注意：價格資料只到 {data_end.date()}，晚於此日的 {want.date()} 無法計算；以下以 {data_end.date()} 為準。請先更新 tw-stock-screener 資料。')
    t = cal[cal <= min(want, data_end)]
    if not len(t): raise SystemExit(f'{want.date()} 之前沒有價格資料')
    t = t[-1]
    sd = signal_day(cal, t)
    if sd is None: raise SystemExit(f'資料中找不到 {t.date()} 之前的換股日')
    nxt, est = next_rebalance(cal, t)
    print(f'查詢日 {t.date()}' + (f'（{want.date()} 非交易日，取前一交易日）' if a.date and t != want and want <= data_end else '')
          + f'｜價格資料最後日期 {data_end.date()}｜處置公告最後日期 {dsp["date"].max().date()}')
    print(f'資料來源：{data["mode"]}' + (f'；月營收另由 rev_hist 補 {n_fill} 筆' if n_fill else ''))
    if sd == t:
        print(f'今天是換股日：今天收盤後依下表名單，於下一個交易日開盤買進。')
    else:
        print(f'今天不是換股日。本月持股名單依最近一次換股日 {sd.date()} 的訊號（下表）；建議股數以 {t.date()} 收盤計算。')
    print(f'下一個換股日：{nxt.date()}' + ('（預估：該月第一個 ≥ 11 日的平日，未扣除國定假日）' if est else ''))

    order, full, m, n_el = score_on(data, YOY3, dsp, sd)
    cnt = YOY.notna().sum(axis=1); med = cnt.loc[:m].iloc[-13:-1].median()
    c_m = int(YOY.loc[m].notna().sum()) if m in YOY.index else 0
    print(f'營收月份：{m - 2}～{m}（{m} 營收公告期限 {(m + 1).to_timestamp().replace(day=10).date()} 已過）；{m} 有營收的公司 {c_m} 家'
          + ('' if c_m >= 0.8 * med else f'（只有前 12 個月中位數 {med:.0f} 家的 {c_m / med:.0%}，資料可能還沒更新完整，請先更新再下單）'))
    if dsp['date'].max() < sd - pd.Timedelta(days=3):
        print(f'注意：處置公告資料只到 {dsp["date"].max().date()}，比訊號日 {sd.date()} 早；請到證交所「處置有價證券」公告確認名單中沒有 {sd.date()} 當天處置中的股票：'
              'https://www.twse.com.tw/zh/announcement/punish.html')
    print(f'股池：60 日均成交值前 {TOPN} 名、≥ 1,000 萬、收盤 ≥ 10、非處置，且有營收與 52 週資料 → {n_el} 檔')
    if n_el < 50: print('警告：股池不足 50 檔，資料可能有問題')

    ex = execute(data, order, sd, n, set(held))
    price_day = t
    price = data['C'].loc[t] if t in data['C'].index else order['close']
    alloc = a.equity / n if a.equity else None
    if alloc: print(f'每檔金額 ≈ {a.equity:,.0f} ÷ {n} = {alloc:,.0f} 元（整張＝1,000 股；零股＝全部用盤中零股買的股數）')

    nback = max(30, n + 10)
    if ex is None:
        picks = list(order.index[:n])
        print(f'\n【前 {n} 名】（{sd.date()} 收盤訊號，下一個交易日開盤買；隔日開盤資料還沒有，尚未套用開盤漲停遞補）')
        print(fmt_table(order, picks, data, price, price_day, alloc, dsp))
        print(f'\n【備取 第 {n + 1}～{nback} 名】')
        print(fmt_table(order, list(order.index[n:nback]), data, price, price_day, alloc, dsp))
        tie_note(order, picks, list(order.index[n:n + 1]))
        print('\n提醒：隔日開盤若「開盤價 ÷ 今日收盤 − 1 ≥ 9.4%」（開盤漲停，除權息日請用參考價判斷）而且你還沒持有，就不買，'
              '改買備取中排名最前、尚未在名單內的股票；已持有的照常續抱。隔日停牌的也由備取遞補。')
    else:
        picks = ex['picks']
        sk = {c: r for c, r in ex['skipped']}
        last_rank = order.index.get_loc(picks[-1]) if picks else 0
        print(f'\n【M2 實際買進名單 {len(picks)} 檔】（{sd.date()} 收盤訊號，{ex["next_day"].date()} 開盤買；已套用開盤漲停遞補）')
        print(fmt_table(order, picks, data, price, price_day, alloc, dsp))
        if ex['skipped']:
            print(f'\n未買（由下一名遞補）：')
            print(fmt_table(order, [c for c, _ in ex['skipped']], data, price, price_day, None, dsp, sk))
        rest = [c for c in order.index[last_rank + 1:] if c not in sk][:nback - n]
        tie_note(order, picks, rest[:1])
        print(f'\n【備取】（名單之後的 {len(rest)} 名）')
        print(fmt_table(order, rest, data, price, price_day, alloc, dsp))

    if held or a.held:
        hs = set(held); ps = set(picks)
        print(f'\n【與目前持股比較】（持有 {len(hs)} 檔）' + ('' if sd == t else f'（今天不是換股日：照規則等 {nxt.date()} 換股；若是剛開始執行，可依本月名單建倉）'))
        def why(c):
            if c not in full.index: return '不是上市普通股或無資料'
            r = full.loc[c]
            if r['inD']: return '換股日處置中'
            if not r['t300']: return '不在 60 日均成交值前 300 名（或成交值 < 1,000 萬／股價 < 10）'
            if not r['el']: return '缺營收或 52 週資料'
            return f'排名第 {order.index.get_loc(c) + 1}'
        sell = sorted(hs - ps); buy = [c for c in picks if c not in hs]; keep = [c for c in picks if c in hs]
        print(f'要賣（{len(sell)} 檔，換股日隔日開盤賣）：' + ('、'.join(f'{c} {data["names"].get(c, "")}（{why(c)}）' for c in sell) or '無'))
        print(f'要買（{len(buy)} 檔）：' + ('、'.join(f'{c} {data["names"].get(c, "")}' for c in buy) or '無'))
        print(f'續抱（{len(keep)} 檔，調回等權即可）：' + ('、'.join(f'{c} {data["names"].get(c, "")}' for c in keep) or '無'))
    print(f'\n資料最後日期：價格 {data_end.date()}、處置公告 {dsp["date"].max().date()}、月營收最新月份 {YOY.index[YOY.notna().sum(axis=1) > 0].max()}。'
          '下單前請以公開資訊觀測站與證交所公告核對。')


if __name__ == '__main__':
    main()
