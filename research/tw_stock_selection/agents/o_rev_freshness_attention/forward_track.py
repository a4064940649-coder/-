#!/usr/bin/env python3
"""2026-11 起前瞻紙上追蹤（本題；規則在看任何 2019 年後分組報酬之前凍結，PREREG 雜湊涵蓋本檔）。
每個交易日盤後執行一次（資料先 git pull tw-stock-screener）：
  python3 -I -B forward_track.py --data /home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data [--date YYYY-MM-DD] [--out forward]
產出（--out 目錄，預設本資料夾 forward/）：
  firstseen.csv     逐檔營收首見日（code, ym, first_seen, ann_date）：PIT 時間戳＝min(rev_hist ann_date, 本腳本第一次看到該月營收的日子)。
  j_lists.csv       錨定月 m 的 J0／J1／J6 日（期限日 D_m 起第 0／1／6 個交易日）收盤：el 內 V60 三分位（低／中／高）各取主策略分數前 20%，
                    另記全 el 前 10（官方序）。這是 I（高關注 − 低關注衰減）的前瞻資料。
  r2_events.csv     R2 事件 A（D_m 收盤；營收用 D_m 盤後已公告值）：T_A 前 10、高關注組、建議的 X＝(H − 高關注) ∪ (T_A ∩ 高關注)。
  rroll_trades.csv、rroll_state.json：R_roll（逐檔公告即換股）。
規則（R_roll，凍結）：
  - 公告窗＝錨定月 m 的第一個交易日到期限日 D_m（含）。窗內每個交易日盤後：已公告 m−1 月營收（firstseen ≤ 當日）的股票用 m−1..m−3 年增率，
    其餘沿用 m−2..m−4（＝上月分數）；分數＝0.5×52 週比百分位＋0.5×營收百分位（el 內；el＝前 300 池、收盤 ≥ 10、非處置、排除金融）。
  - 持股 10 檔：先賣掉名次 > 15 或不在 el 的持股；再依名次由高到低，把前 10 名中未持有者補到 10 檔。全部在隔日開盤成交
    （開盤漲停且未持有者不買、由下一名遞補；停牌者不買）。窗外不交易（續抱）。
  - 不做緩衝帶選版；成本與成交紀錄由 live_tracking 的實際成交校準（本腳本只記名單與開盤價）。
本腳本只寫自己的 --out 目錄，不改其他資料夾。"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import main_picks_today as MP     # noqa: E402（只讀 import）
sys.path.pop(0)

def deadline(y, m, hol):
    d = pd.Timestamp(year=y, month=m, day=10)
    while d.weekday() >= 5 or str(d.date()) in hol: d += pd.Timedelta(days=1)
    return d

def tdays_from(d0, n, hol):
    out = []; d = d0
    while len(out) < n:
        if d.weekday() < 5 and str(d.date()) not in hol: out.append(d)
        d += pd.Timedelta(days=1)
    return out

def tercile(v, mask):
    q = v.where(mask).rank(pct=True)
    return pd.Series(np.select([q <= 1 / 3, q <= 2 / 3, q > 2 / 3], [0, 1, 2], -1), index=v.index).where(mask, -1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--date'); ap.add_argument('--out', default=str(HERE / 'forward'))
    a = ap.parse_args()
    D = Path(a.data); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
    hol = set(sum(json.loads((D / 'extras/tw_holidays.json').read_text(encoding='utf-8'))['years'].values(), []))
    target = pd.Timestamp(a.date) if a.date else None
    data = MP.load_prices(D, target if target is not None else pd.Timestamp('2100-01-01'))
    cal = data['cal']
    d = cal[cal <= target][-1] if target is not None else cal[-1]
    Y, Y3, _ = MP.load_revenue(D); dsp = MP.load_disposal(D)
    m = pd.Period(d, 'M'); Dm = deadline(m.year, m.month, hol)
    # ---- 首見日 ----
    rh = pd.read_csv(D / 'extras/rev_hist.csv.gz', dtype=str)
    r5 = pd.read_csv(D / 'extras/rev_5y.csv.gz', dtype={'code': str})
    fs_f = OUT / 'firstseen.csv'
    fs = pd.read_csv(fs_f, dtype=str) if fs_f.exists() else pd.DataFrame(columns=['code', 'ym', 'first_seen', 'ann_date'])
    seen = set(zip(fs.code, fs.ym))
    ann = dict(zip(zip(rh.code, rh.ym), rh.ann_date))
    cur = [(c, y) for c, y in set(zip(r5.code, r5.ym)) | set(zip(rh.code, rh.ym)) if y >= str(m - 2)]
    today = str(pd.Timestamp.today().date())   # 首見日＝本腳本實際執行的日曆日（不是 --date），回補歷史日時不會造成前視
    add = [(c, y, today, ann.get((c, y)) if isinstance(ann.get((c, y)), str) else '') for c, y in cur if (c, y) not in seen]
    if add: fs = pd.concat([fs, pd.DataFrame(add, columns=fs.columns)], ignore_index=True); fs.to_csv(fs_f, index=False)
    def announced(c, ym):
        r = fs[(fs.code == c) & (fs.ym == ym)]
        if not len(r): return False
        ts = [pd.Timestamp(x) for x in (r.first_seen.iloc[0], r.ann_date.iloc[0]) if isinstance(x, str) and x]
        return min(ts) <= d
    # ---- 分數（m−1 已公告者用新值） ----
    order0, df, _, _ = MP.score_on(data, Y3, dsp, d, ex_fin=True)
    cols = df.index
    pm1, pm2 = m - 1, m - 2
    new = Y3.loc[pm1].reindex(cols) if pm1 in Y3.index else pd.Series(np.nan, index=cols)
    old = Y3.loc[pm2].reindex(cols) if pm2 in Y3.index else pd.Series(np.nan, index=cols)
    ok = pd.Series([announced(c, str(pm1)) for c in cols], index=cols)
    y3 = new.where(ok & new.notna(), old)
    Ymix = Y3.copy(); Ymix.loc[pm1] = y3.reindex(Ymix.columns)
    order, df, _, nel = MP.score_on(data, Ymix, dsp, d, ex_fin=True)
    el = df['el']
    gA = tercile(df['v60'], el)
    rows = []
    jdays = tdays_from(Dm, 7, hol)
    jmap = {jdays[0]: 0, jdays[1]: 1, jdays[6]: 6}
    if d in jmap:
        j = jmap[d]
        for g in (0, 1, 2):
            sub = order[order.index.map(lambda c: gA.get(c, -1) == g)]
            n = max(1, int(round(0.2 * len(sub))))
            rows.append(dict(anchor=str(m), j=j, signal=str(d.date()), group=f'A{g}', n=n, picks=','.join(sub.index[:n])))
        rows.append(dict(anchor=str(m), j=j, signal=str(d.date()), group='top10', n=10, picks=','.join(order.index[:10])))
        f = OUT / 'j_lists.csv'; old_ = pd.read_csv(f, dtype=str) if f.exists() else None
        new_ = pd.DataFrame(rows)
        if old_ is not None: new_ = pd.concat([old_[~((old_.anchor == str(m)) & (old_.j == str(j)))], new_.astype(str)])
        new_.to_csv(f, index=False)
    # ---- R_roll ----
    st_f = OUT / 'rroll_state.json'
    st = json.loads(st_f.read_text(encoding='utf-8')) if st_f.exists() else {'holdings': [], 'last': None}
    in_window = (d >= cal[(cal.year == d.year) & (cal.month == d.month)][0]) and d <= Dm
    if in_window and st.get('last') != str(d.date()):
        rank = {c: i + 1 for i, c in enumerate(order.index)}
        H = [c for c in st['holdings'] if rank.get(c, 10 ** 9) <= 15]
        sells = [c for c in st['holdings'] if c not in H]
        ex = MP.execute(data, order, d, 10, set(st['holdings']))
        cand = [c for c in order.index[:10] if c not in H]
        buys = []
        for c in cand:
            if len(H) + len(buys) >= 10: break
            if ex is not None and c not in ex['picks'] and c in dict(ex['skipped']): continue
            buys.append(c)
        tr = OUT / 'rroll_trades.csv'
        rec = pd.DataFrame([dict(signal=str(d.date()), side=s, code=c, rank=rank.get(c)) for s, L_ in (('sell', sells), ('buy', buys)) for c in L_])
        if len(rec): rec.to_csv(tr, mode='a', header=not tr.exists(), index=False)
        st = {'holdings': H + buys, 'last': str(d.date()), 'n_announced_el': int((ok & el).sum()), 'n_el': int(el.sum())}
        st_f.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding='utf-8')
    # ---- R2 事件 A ----
    if d == Dm:
        hi = [c for c in order.index[:10] if gA.get(c, -1) == 2]
        e = dict(anchor=str(m), signal=str(d.date()), TA=','.join(order.index[:10]), TA_high=','.join(hi),
                 high_group=','.join(sorted(c for c in gA.index if gA[c] == 2)))
        f = OUT / 'r2_events.csv'; pd.DataFrame([e]).to_csv(f, mode='a', header=not f.exists(), index=False)
    print(json.dumps({'date': str(d.date()), 'anchor': str(m), 'deadline': str(Dm.date()), 'in_window': bool(in_window),
                      'n_el': int(el.sum()), 'n_announced_m-1_in_el': int((ok & el).sum()), 'j_logged': jmap.get(d), 'rroll_holdings': st.get('holdings')}, ensure_ascii=False))

if __name__ == '__main__':
    main()
