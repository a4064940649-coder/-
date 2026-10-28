#!/usr/bin/env python3
"""o7 市值匹配選股（o_capmatched_selection 凍結版）前瞻紙上追蹤，2026-11 換股日起。
每個換股日（每月第一個日 ≥ 11 的交易日）收盤後執行：
  python3 -I -B research/tw_stock_selection/agents/live_tracking/o7_capmatched/o7_paper.py --data /home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data
做兩件事：
  1) targets：對每個還沒記錄、且 ≥ 2026-11-01 的換股日，算凍結版目標權重（與 0050 自由流通複製權重），寫 targets.tsv
  2) paper：用已記錄的目標，從成交日開盤到下一成交日開盤（一般成本：買 0.0855%＋0.1%、賣 0.0855%＋0.3%＋0.1%）算紙上淨值，
     與 0050 含息（extras/etf_long.csv.gz）、0050 50%／70% 核心組合一起寫 paper_nav.tsv（每個成交日一列）
選股規格與回測相同（引用 o_capmatched_selection/cmstrat.py 的 weights_F1；面板外的價量用 main_picks_today.py 的 load_prices）：
  股數＝extras/shares_hist.csv 訊號月（沒有就最近月）；成分＝extras/etf0050_w.csv 訊號日前最近快照的 50 檔；
  自由流通係數＝o_capmatched_selection/data/float_phi.parquet（缺的用中位數）；營收＝rev_5y＋rev_hist，期限日（10 日起第一個交易日）≥ 訊號日時用 m−2..m−4；
  處置＝extras/disposal_5y（公告日 ≤ 訊號日、成交日在處置期間 → 排除）。
--targets-only：只算目標權重，不算報酬。判定依 SPEC.json 的序貫規則。"""
import sys, json, argparse, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
TOPIC = HERE.parent.parent / 'o_capmatched_selection'
TOOL = HERE.parents[2] / 'main_picks_today.py'
sys.path.insert(0, str(TOPIC))
import cmlib as L          # noqa: E402（只讀；載入 raw/ 面板，用到代碼表、金融／KY 遮罩、tick）
import cmstrat as T        # noqa: E402
spec = importlib.util.spec_from_file_location('mpt', TOOL); MPT = importlib.util.module_from_spec(spec); spec.loader.exec_module(MPT)
SPEC = json.loads((HERE / 'SPEC.json').read_text(encoding='utf-8'))
VERSION = SPEC['spec']['frozen_version']
START = pd.Timestamp(SPEC['spec']['start_signal_on_or_after'])

def build_cs(D, data, Y, dsp, sd, nd):
    cols = L.COLS; N = L.N; CI = L.CI
    order, df, m, nel = MPT.score_on(data, Y.rolling(3, min_periods=3).mean(), dsp, sd)
    df = df.reindex(cols)
    px = df['close'].to_numpy(float); s52 = df['s52'].to_numpy(float); v60 = df['v60'].to_numpy(float)
    sh = pd.read_csv(D / 'extras' / 'shares_hist.csv', dtype={'code': str})
    ym = str(pd.Period(sd, 'M')); sh = sh[sh.ym <= ym].sort_values('ym').drop_duplicates('code', keep='last').set_index('code')['shares']
    mc = sh.reindex(cols).to_numpy(float) * px
    live = np.isfinite(px) & ~L.DRMASK
    capuni = np.isfinite(mc) & (mc > 0) & live
    o = np.argsort(-np.where(capuni, mc, -np.inf), kind='stable'); rank = np.full(N, 10 ** 9); rank[o] = np.arange(1, N + 1); rank[~capuni] = 10 ** 9
    w = pd.read_csv(D / 'extras' / 'etf0050_w.csv', dtype={'code': str})
    asof = w['asof'][w['asof'] <= str(sd.date())]
    asof = asof.max() if len(asof) else w['asof'].min()
    mem = [c for c in w[w['asof'] == asof]['code'] if c in CI]
    js = np.array([CI[c] for c in mem], int)
    fa = mc[js] * T.PHI[js]; ok = np.isfinite(fa) & (fa > 0)
    R = np.zeros(N); R[js[ok]] = fa[ok] / fa[ok].sum()
    # 營收：期限日 ≥ 訊號日 → 改用 m−2..m−4
    cal = data['cal']
    dl = pd.Timestamp(sd.year, sd.month, 10); dlx = cal[cal >= dl]
    deadline = dlx[0] if len(dlx) else dl
    Y3 = Y.rolling(3, min_periods=3).mean()
    mm = pd.Period(sd, 'M') - (2 if deadline >= sd else 1)
    y3 = Y3.reindex([mm]).iloc[0].reindex(cols).to_numpy(float) if mm in Y3.index else np.full(N, np.nan)
    inD = np.zeros(N, bool)
    for _, r in dsp[(dsp['date'] <= sd) & (dsp['start'] <= nd) & (dsp['end'] >= nd)].iterrows():
        if r['code'] in CI: inD[CI[r['code']]] = True
    o_next = data['O'].loc[nd].reindex(cols).to_numpy(float) if nd in data['O'].index else np.full(N, np.nan)
    elig = (capuni & (rank <= 300) & ~L.FINMASK & ~L.FORMASK & np.isfinite(v60) & (v60 >= 1e7) & (px >= 10) & np.isfinite(s52) & np.isfinite(y3) & ~inD)
    if nd in data['O'].index: elig &= np.isfinite(o_next)
    def pr(x): return pd.Series(np.where(elig, x, np.nan)).rank(pct=True).to_numpy()
    sc = 0.5 * pr(s52) + 0.5 * pr(y3); sc[~elig] = np.nan
    blocked = np.zeros(N, bool)
    if nd in data['O'].index:
        prevc = data['C'].loc[:sd].ffill().iloc[-1].reindex(cols).to_numpy(float)
        r1 = data['R'].loc[nd].reindex(cols).to_numpy(float); c1 = data['C'].loc[nd].reindex(cols).to_numpy(float)
        with np.errstate(invalid='ignore', divide='ignore'):
            ref = np.where(np.isfinite(r1), c1 / (1 + r1), prevc)
        lu = MPT.limit_up_price(ref)
        blocked = np.isfinite(o_next) & np.isfinite(lu) & (o_next >= lu - 1e-6)
    cs = dict(R=R, mc=mc, rank=rank, members=js[ok], fin=L.FINMASK, elig=elig, sc=sc, y3=y3, capuni=capuni)
    return cs, blocked, asof, int(elig.sum())

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--targets-only', action='store_true')
    ap.add_argument('--start', default=None, help='（測試用）覆寫起始換股日')
    a = ap.parse_args(); D = Path(a.data)
    start = pd.Timestamp(a.start) if a.start else START
    data = MPT.load_prices(D, pd.Timestamp.today().normalize())
    Y, _, _ = MPT.load_revenue(D); dsp = pd.read_csv(D / 'extras' / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
    cal = data['cal']
    sdays = [MPT.rebalance_day(cal, p.year, p.month) for p in pd.period_range(start, cal[-1], freq='M')]
    sdays = [d for d in sdays if d is not None and d >= start and cal.get_loc(d) + 1 < len(cal)]
    tf = HERE / 'targets.tsv'
    old = pd.read_csv(tf, sep='\t', dtype={'code': str}) if tf.exists() else pd.DataFrame(columns=['signal', 'exec', 'code', 'w', 'R', 'members_asof'])
    held = set(old[old.signal == old.signal.max()].code) if len(old) else set()
    fam, kw = T.VERSIONS[VERSION]
    rows = []
    for sd in sdays:
        if str(sd.date()) in set(old.signal): continue
        nd = cal[cal.get_loc(sd) + 1]
        cs, blocked, asof, nel = build_cs(D, data, Y, dsp, sd, nd)
        w, nb, ns = T.weights_F1(cs, blocked=blocked, held=held, **kw)
        for j, x in w.items(): rows.append((str(sd.date()), str(nd.date()), L.COLS[j], float(x), float(cs['R'][j]), asof))
        held = set(L.COLS[j] for j in w)
        print(sd.date(), '成交', nd.date(), '持股', len(w), '衛星', ns, '開盤漲停跳過', nb, '合格', nel, '成分快照', asof)
    if rows:
        new = pd.DataFrame(rows, columns=['signal', 'exec', 'code', 'w', 'R', 'members_asof'])
        pd.concat([old, new]).to_csv(tf, sep='\t', index=False)
    if a.targets_only: return
    # 紙上淨值（成交日開盤 → 下一成交日開盤）
    tg = pd.read_csv(tf, sep='\t', dtype={'code': str})
    ex = sorted(tg['exec'].unique())
    b, s = L.rates('normal')
    e50 = pd.read_csv(D / 'extras' / 'etf_long.csv.gz', dtype={'code': str}, parse_dates=['date']); e50 = e50[e50.code == '0050'].set_index('date')['adj'].sort_index()
    C, O, R = data['C'], data['O'], data['R']
    out = []; navF = navR = 1.0; prevw = {}
    for i in range(len(ex) - 1):
        e0, e1 = pd.Timestamp(ex[i]), pd.Timestamp(ex[i + 1])
        g = tg[tg['exec'] == ex[i]]
        def wf(code):
            if code not in C.columns: return 1.0
            c0, o0 = C.at[e0, code], O.at[e0, code]
            seg = R[code].loc[(R.index > e0) & (R.index <= e1)].fillna(0)
            o1, c1 = O.at[e1, code], C.at[e1, code]
            f = (c0 / o0) * float((1 + seg).prod()) * (o1 / c1 if np.isfinite(o1) and np.isfinite(c1) and c1 > 0 else 1.0)
            return f if np.isfinite(f) else 1.0
        rF = sum(x * wf(c) for c, x in zip(g.code, g.w)) - 1
        rR = sum(x * wf(c) for c, x in zip(g.code, g.R) if x > 0) / max(g.R.sum(), 1e-9) - 1
        nw = dict(zip(g.code, g.w)); tov = 0.5 * sum(abs(nw.get(k, 0) - prevw.get(k, 0)) for k in set(nw) | set(prevw)); prevw = nw
        cost = tov * (b + s) if i > 0 else b
        r50 = float(e50.asof(e1) / e50.asof(e0) - 1)
        navF *= 1 + rF - cost; navR *= 1 + rR
        out.append((ex[i], ex[i + 1], rF - cost, rR, r50, 0.5 * r50 + 0.5 * (rF - cost), 0.7 * r50 + 0.3 * (rF - cost), navF, navR))
    pd.DataFrame(out, columns=['exec', 'next_exec', 'r_F', 'r_REP', 'r_0050', 'r_C50', 'r_C70', 'nav_F', 'nav_REP']).to_csv(HERE / 'paper_nav.tsv', sep='\t', index=False)
    print('paper_nav.tsv', len(out), '窗')

if __name__ == '__main__':
    main()
