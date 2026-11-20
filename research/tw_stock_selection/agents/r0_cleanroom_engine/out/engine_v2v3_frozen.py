# -*- coding: utf-8 -*-
"""R0 清潔室回測引擎（獨立重寫；不 import、不參考任何既有 *lib.py／run*.py／make_list.py）。
只根據白話規格（xq_scripts/README.md、底倉衛星_操作說明.md、每檔金額怎麼算_補充說明.md、22 號 XS、
raw/*/README.md、s18_live_list/FINDINGS.md）寫成。資料由 build_data.py 從原始檔整理成 data.npz。
用法：python3 -I -B engine.py <暫存目錄> <輸出前綴> [--opt key=value ...]
"""
import sys, json, math, csv
import numpy as np

EXC_CSV = '/home/user/-/research/tw_stock_selection/raw/rev_deadline_exceptions/deadline_exceptions.csv'
INS_CSV = '/home/user/-/research/tw_stock_selection/raw/rev_deadline_exceptions/ins15_list.csv'

CFG = dict(
    start='2015-07-01', end='2026-10-22', capital=1_000_000.0,
    sat=0.70, band=0.10, nslot=10, top_in=10, top_out=20, ret_min=0.10, ma=60, retlen=20,
    liq_days=60, liq_min=1e7, px_min=10.0,
    fee=0.001425 * 0.28, tax=0.003, tax_etf=0.001, slip=0.003, half_tick=True,
    odd_lot_from='2020-10-26', cap_frac=0.05, min_frac=0.5, sweep_min=20_000.0,
    deadline_fix=True, ins15=True, rev_pos=True,
    nomove_fix=False,   # 前一天沒成交的日子是否改用「收盤／上次收盤」算報酬（預設否：照漲跌價差欄）
    etf_slip=True,      # 0050 買賣是否也算半跳動點＋滑價
)

def tick_stock(p):
    if p < 10: return 0.01
    if p < 50: return 0.05
    if p < 100: return 0.1
    if p < 500: return 0.5
    if p < 1000: return 1.0
    return 5.0

def tick_etf(p):
    return 0.01 if p < 50 else 0.05

def limit_up(ref):
    raw = ref * 1.10; t = tick_stock(raw)
    return math.floor(raw / t + 1e-9) * t

def limit_dn(ref):
    raw = ref * 0.90; t = tick_stock(raw)
    return math.ceil(raw / t - 1e-9) * t

class Data:
    def __init__(self, tmp, cfg):
        z = np.load(tmp + '/data.npz', allow_pickle=True)
        self.dates = [str(x) for x in z['dates']]; self.codes = [str(x) for x in z['codes']]
        self.O, self.C, self.A, self.CH = z['O'], z['C'], z['A'], z['CH']
        self.present, self.twse, self.innov, self.dr = z['present'], z['twse'], z['innov'], z['dr']
        self.name = [str(x) for x in z['name']]
        self.ex_prev, self.ex_ref = z['ex_prev'], z['ex_ref']
        self.e_open, self.e_close, self.e_adj = z['e_open'], z['e_close'], z['e_adj']
        self.rev = z['rev']; self.rev_months = [str(x) for x in z['rev_months']]; self.rev_codes = [str(x) for x in z['rev_codes']]
        self.di = {d: i for i, d in enumerate(self.dates)}; self.ci = {c: i for i, c in enumerate(self.codes)}
        T, N = self.C.shape
        C, CH = self.C, self.CH
        traded = np.isfinite(C) & (C > 0)
        # 上次收盤（往前補）
        last = np.where(traded, C, np.nan)
        for t in range(1, T):
            m = ~np.isfinite(last[t]); last[t, m] = last[t - 1, m]
        self.last = last
        prev_last = np.vstack([np.full((1, N), np.nan), last[:-1]])
        prev_traded = np.vstack([np.zeros((1, N), bool), traded[:-1]])
        # 當日參考價：一般＝收盤−漲跌價差；除權息日＝TWT49U 參考價
        ref = np.where(traded, C - np.nan_to_num(CH), np.nan)
        if cfg['nomove_fix']:
            m = traded & ~prev_traded & np.isfinite(prev_last)
            ref[m] = prev_last[m]
        exm = np.isfinite(self.ex_ref) & traded
        ref[exm] = self.ex_ref[exm]
        self.ref = ref; self.exm = np.isfinite(self.ex_ref)
        r = np.where(traded & np.isfinite(ref) & (ref > 0), C / ref - 1.0, 0.0)
        self.ret = r
        adj = np.cumprod(1.0 + r, axis=0)
        adj[~np.isfinite(last)] = np.nan
        self.adj = adj; self.traded = traded
        # 0050：含息還原價（收盤）；開盤＝還原收盤×原始開盤／原始收盤
        # 0050 暫停交易日（2025-06-11～17 分割換發）：估值沿用上次還原收盤；當天不能買賣（e_o＝NaN）
        ec = self.e_adj.copy()
        for t in range(1, len(ec)):
            if not np.isfinite(ec[t]): ec[t] = ec[t - 1]
        self.e_c = ec
        self.e_o = self.e_adj * self.e_open / self.e_close
        # SURGE（月 × 代號）
        R = self.rev.copy()
        if cfg['rev_pos']: R[~(R > 0)] = np.nan
        M = len(self.rev_months)
        D = np.full_like(R, np.nan); D[12:] = R[12:] - R[:-12]
        for k, m in enumerate(self.rev_months):
            if m[:4] == '2013': D[k] = np.nan
        S = np.full_like(R, np.nan)
        for k in range(24, M):
            w = D[k - 24:k]
            ok = np.isfinite(w).all(axis=0) & np.isfinite(D[k])
            mu = np.nanmean(np.where(ok, w, 0.0), axis=0); sd = np.std(np.where(ok, w, 0.0), axis=0, ddof=1)
            ok &= sd > 0
            S[k, ok] = (D[k, ok] - mu[ok]) / sd[ok]
        self.surge = S; self.rci = {c: i for i, c in enumerate(self.rev_codes)}; self.rmi = {m: i for i, m in enumerate(self.rev_months)}

def signal_days(d, cfg):
    """每月：10 日（含）之後第一個交易日＝期限日；訊號日＝期限日的下一個交易日。
    例外表 apply=1 的營收月份：期限日改成 actual_deadline（含）之後第一個交易日。回傳 [(訊號月, 訊號日索引, 營收月)]"""
    exc = {}
    if cfg['deadline_fix']:
        for r in csv.DictReader(open(EXC_CSV, encoding='utf-8')):
            if r['apply'] == '1': exc[r['rev_month']] = r['actual_deadline']
    out = []
    months = sorted(set(x[:7] for x in d.dates))
    for m in months:
        y, mo = int(m[:4]), int(m[5:]); pm = '%04d-%02d' % ((y, mo - 1) if mo > 1 else (y - 1, 12))
        dl = exc.get(pm, m + '-10')
        cand = [i for i, x in enumerate(d.dates) if x >= dl and x[:7] <= dl[:7] or (x >= dl and x[:7] == m)]
        cand = [i for i, x in enumerate(d.dates) if x >= dl]
        if not cand or cand[0] + 1 >= len(d.dates): continue
        out.append((m, cand[0] + 1, pm))
    return out

def ins15_hidden(d, cfg, rev_month, sig_idx):
    """15 日期限公司：訊號日早於「次月 15 日（含）以後第一個交易日」時，當月 SURGE 當缺值。"""
    if not cfg['ins15']: return set()
    y, mo = int(rev_month[:4]), int(rev_month[5:]); nm = '%04d-%02d' % ((y, mo + 1) if mo < 12 else (y + 1, 1))
    hid = set()
    for r in csv.DictReader(open(INS_CSV, encoding='utf-8')):
        if rev_month < r['start_month'] or (r['end_month'] and rev_month > r['end_month']): continue
        dl = '%s-%02d' % (nm, int(r['deadline_day']))
        cand = [i for i, x in enumerate(d.dates) if x >= dl]
        if not cand or sig_idx < cand[0]: hid.add(r['code'])
    return hid

def make_signal(d, cfg, t, rev_month):
    """訊號日 t 收盤後：回傳 L_MA 排名表 [(code, surge, ret20, rank)] 與 F1 名單。"""
    ma, rl, ld = cfg['ma'], cfg['retlen'], cfg['liq_days']
    if rev_month not in d.rmi: return [], []
    k = d.rmi[rev_month]; hid = ins15_hidden(d, cfg, rev_month, t)
    rows = []
    for j, c in enumerate(d.codes):
        if not (d.twse[t, j] and d.traded[t, j]) or d.innov[t, j] or d.dr[t, j]: continue
        if d.C[t, j] < cfg['px_min']: continue
        if t - ld + 1 < 0 or not d.present[t - ld + 1:t + 1, j].all(): continue
        if np.nansum(d.A[t - ld + 1:t + 1, j]) / ld < cfg['liq_min']: continue
        if c not in d.rci or c in hid: continue
        s = d.surge[k, d.rci[c]]
        if not np.isfinite(s): continue
        w = d.adj[t - ma + 1:t + 1, j]
        if t - ma + 1 < 0 or not np.isfinite(w).all(): continue
        if not d.adj[t, j] > w.mean(): continue
        r20 = d.adj[t, j] / d.adj[t - rl, j] - 1.0 if t - rl >= 0 and np.isfinite(d.adj[t - rl, j]) else np.nan
        rows.append((c, float(s), float(r20)))
    rows.sort(key=lambda x: (-x[1], x[0]))
    lma = [(c, s, r, i + 1) for i, (c, s, r) in enumerate(rows)]
    f1 = [c for c, s, r, rk in lma if rk <= cfg['top_in'] and r > cfg['ret_min']]
    return lma, f1

def run(tmp, cfg):
    d = Data(tmp, cfg)
    t0, t1 = d.di[cfg['start']], d.di[cfg['end']]
    sigs = {si: (m, pm) for m, si, pm in signal_days(d, cfg) if t0 <= si < t1}
    fee, slip = cfg['fee'], cfg['slip']
    cash = cfg['capital']; eu = 0.0; base = 0.0   # eu＝0050 單位數（還原價口徑）；base＝帳上底倉單位數
    pos = {}       # code -> dict(sh, entry_date, entry_px, cost)
    pending_sell = set(); pending = None
    navs = []; trades = []; lists = []; log = []; rebal_log = []
    odd_from = cfg['odd_lot_from']; sweep_wait = [False]

    def etf_px(t, side):
        p = d.e_o[t]
        if not cfg['etf_slip']: return p
        ht = 0.5 * tick_etf(d.e_open[t]) / d.e_open[t]
        return p * (1 + slip + ht) if side > 0 else p * (1 - slip - ht)

    def buy_etf(t, amt):
        nonlocal cash, eu
        if amt <= 0: return
        if not np.isfinite(d.e_o[t]): sweep_wait[0] = True; return
        px = etf_px(t, 1); u = amt / (px * (1 + fee)); eu += u; cash -= u * px * (1 + fee)
        log.append((d.dates[t], '0050', 'BUY', u, px))

    def sell_etf(t, need):
        """賣 0050 換到 need 元現金（扣費稅後），回傳實際拿到的現金。"""
        nonlocal cash, eu, base
        if need <= 0 or eu <= 0 or not np.isfinite(d.e_o[t]): return 0.0
        px = etf_px(t, -1); net = px * (1 - fee - cfg['tax_etf'])
        u = min(eu, need / net); eu -= u; cash += u * net
        if eu < base: base = eu
        log.append((d.dates[t], '0050', 'SELL', u, px))
        return u * net

    def sell_stock(t, c, frac=1.0, reason=''):
        nonlocal cash
        j = d.ci[c]; o = d.O[t, j]; p = pos[c]
        ht = 0.5 * tick_stock(o) if cfg['half_tick'] else 0.0
        px = o * (1 - slip) - ht; sh = p['sh'] * frac
        got = sh * px * (1 - fee - cfg['tax']); cash += got
        log.append((d.dates[t], c, 'SELL', sh, px))
        if frac >= 1.0:
            trades.append(dict(code=c, name=d.name[j], entry_date=p['entry_date'], exit_date=d.dates[t], entry_px=p['entry_px'],
                               exit_px=px, shares_exit=sh, cost=p['cost'], proceeds=got + p.get('partial', 0.0),
                               ret=(got + p.get('partial', 0.0)) / p['cost'] - 1.0, reason=reason))
            del pos[c]
        else:
            p['sh'] -= sh; p['partial'] = p.get('partial', 0.0) + got

    def can_sell(t, c):
        j = d.ci[c]; o = d.O[t, j]
        if not (np.isfinite(o) and o > 0): return False
        ref = d.ref[t, j]
        if np.isfinite(ref) and o <= limit_dn(ref) + 1e-9: return False
        return True

    for t in range(t0, t1 + 1):
        day = d.dates[t]
        # 除權息：持股用「除權息前收盤／參考價」換算股數（等同當日報酬以參考價計、股利以參考價再投入）
        for c, p in pos.items():
            j = d.ci[c]
            if d.exm[t, j] and np.isfinite(d.ex_prev[t, j]):
                p['sh'] *= d.ex_prev[t, j] / d.ex_ref[t, j]
        traded_today = False
        if t == t0:
            buy_etf(t, cash); base = eu * (1 - cfg['sat']); traded_today = True
        # 順延賣出（先前開盤跌停或無成交沒賣掉的）
        for c in sorted(pending_sell):
            if c in pos and can_sell(t, c):
                sell_stock(t, c, reason=pos[c].get('exit_reason', 'deferred')); pending_sell.discard(c); traded_today = True
        if pending is not None and pending['exec'] == t:
            sg = pending; pending = None; traded_today = True
            # 1) 賣出
            for c in sg['exits']:
                if c not in pos: continue
                pos[c]['exit_reason'] = sg['why'][c]
                if can_sell(t, c): sell_stock(t, c, reason=sg['why'][c])
                else: pending_sell.add(c)
            # 2) 拉回檢查（賣完之後、買之前；用執行日開盤價估值）
            def stock_val():
                v = 0.0
                for c, p in pos.items():
                    j = d.ci[c]; o = d.O[t, j]
                    v += p['sh'] * (o if np.isfinite(o) and o > 0 else d.last[t - 1, j])
                return v
            p0 = d.e_o[t] if np.isfinite(d.e_o[t]) else d.e_c[t]
            total = cash + eu * p0 + stock_val(); sv = total - base * p0; ratio = sv / total
            reb = None
            if abs(ratio - cfg['sat']) > cfg['band'] + 1e-12:
                reb = 'low' if ratio < cfg['sat'] else 'high'
                base = total * (1 - cfg['sat']) / p0
                if eu < base:
                    # 現金先補 0050；仍不足才依市值等比例減碼個股
                    short = (base - eu) * p0
                    if cash < short:
                        sv_stock = stock_val()
                        frac = min(1.0, (short - cash) / sv_stock) if sv_stock > 0 else 0.0
                        for c in sorted(pos):
                            if c not in pending_sell and can_sell(t, c): sell_stock(t, c, frac=frac)
                        reb = 'high_trim'
                    buy_etf(t, min(cash, short))
                    if eu < base: base = eu
                total = cash + eu * p0 + stock_val(); sv = total - base * p0
            target = sv / cfg['nslot']
            rebal_log.append(dict(exec_date=day, total=total, sat_ratio=ratio, rebalance=reb, target=target))
            # 3) 買進：依 SURGE 由大到小；最多 nslot 檔；開盤漲停不買不遞補
            bought = []; skipped = {}
            for c in sg['f1']:
                if c in pos: skipped[c] = 'held'; continue
                if len(pos) >= cfg['nslot']: skipped[c] = 'full'; continue
                j = d.ci[c]; o = d.O[t, j]
                if not (np.isfinite(o) and o > 0): skipped[c] = 'no_open'; continue
                ref = d.ref[t, j]
                if np.isfinite(ref) and o >= limit_up(ref) - 1e-9: skipped[c] = 'limit_up'; continue
                ht = 0.5 * tick_stock(o) if cfg['half_tick'] else 0.0
                px = o * (1 + slip) + ht
                avail = cash + (eu * etf_px(t, -1) * (1 - fee - cfg['tax_etf']) if np.isfinite(d.e_o[t]) else 0.0)
                amt = min(target, avail / (1 + fee))
                if np.isfinite(d.A[t, j]): amt = min(amt, cfg['cap_frac'] * d.A[t, j])
                unit = 1 if day >= odd_from else 1000
                sh = math.floor(amt / px / unit + 1e-9) * unit
                if sh <= 0 or sh * px < cfg['min_frac'] * target: skipped[c] = 'below_half'; continue
                cost = sh * px * (1 + fee)
                if cash < cost: sell_etf(t, cost - cash)
                cash -= cost
                pos[c] = dict(sh=float(sh), entry_date=day, entry_px=px, cost=cost)
                log.append((day, c, 'BUY', sh, px)); bought.append(c)
            lists[-1].update(exec_date=day, bought=bought, skipped=skipped, sold=[c for c in sg['exits'] if c not in pos],
                             deferred=sorted(pending_sell))
        # 4) 閒置現金 ≥ 2 萬買 0050（只在有交易的日子檢查）
        if sweep_wait[0] and np.isfinite(d.e_o[t]): sweep_wait[0] = False; traded_today = True
        if traded_today and cash >= cfg['sweep_min']: buy_etf(t, cash)
        # 收盤淨值
        sv = 0.0
        for c, p in pos.items():
            sv += p['sh'] * d.last[t, d.ci[c]]
        nav = cash + eu * d.e_c[t] + sv
        navs.append((day, nav, sv, eu * d.e_c[t], cash, len(pos), base * d.e_c[t]))
        # 訊號日收盤後
        if t in sigs:
            m, pm = sigs[t]
            lma, f1 = make_signal(d, cfg, t, pm)
            rank = {c: rk for c, s, r, rk in lma}
            exits = []; why = {}
            for c in pos:
                rk = rank.get(c)
                if rk is None or rk > cfg['top_out']:
                    exits.append(c); why[c] = 'not_in_LMA' if rk is None else 'rank>%d' % cfg['top_out']
            pending = dict(exec=t + 1, f1=f1, exits=exits, why=why)
            lists.append(dict(month=m, rev_month=pm, sig_date=day, n_lma=len(lma), f1=f1, top20=[(c, round(s, 4), round(r, 4)) for c, s, r, rk in lma[:20]],
                              held=sorted(pos), exits=sorted(exits)))
    return dict(navs=navs, trades=trades, lists=lists, log=log, rebal=rebal_log, open_pos={c: p for c, p in pos.items()}, d=d)

def cagr(navs, start, base=None):
    """start 之前最後一個淨值當起點（沒有就用 base 與第一天日期）。"""
    import datetime as dt
    prev = [x for x in navs if x[0] < start]
    if prev: d0, v0 = prev[-1][0], prev[-1][1]
    else: d0, v0 = navs[0][0], base
    d1, v1 = navs[-1][0], navs[-1][1]
    yrs = (dt.date.fromisoformat(d1) - dt.date.fromisoformat(d0)).days / 365.25
    return (v1 / v0) ** (1 / yrs) - 1

def summarize(res, cfg):
    navs = res['navs']; v = np.array([x[1] for x in navs]); dates = [x[0] for x in navs]
    full = np.concatenate([[cfg['capital']], v])
    peak = np.maximum.accumulate(full); dd = full / peak - 1; k = int(dd.argmin())
    pk = int(np.argmax(full[:k + 1]))
    yearly = {}
    yrs = sorted(set(x[:4] for x in dates))
    for y in yrs:
        idx = [i for i, x in enumerate(dates) if x[:4] == y]
        b = cfg['capital'] if idx[0] == 0 else v[idx[0] - 1]
        yearly[y] = round((v[idx[-1]] / b - 1) * 100, 2)
    tr = res['trades']; rets = np.array([x['ret'] for x in tr]) if tr else np.array([])
    out = dict(cagr_2019=round(cagr(navs, '2019-01-01') * 100, 2), cagr_full=round(cagr(navs, '0000', cfg['capital']) * 100, 2),
               cagr_2024=round(cagr(navs, '2024-01-01') * 100, 2),
               maxdd=round(dd.min() * 100, 2), dd_peak=([cfg['start']] + dates)[pk], dd_trough=([cfg['start']] + dates)[k],
               final_nav=round(float(v[-1]), 0), n_closed=len(tr), n_open=len(res['open_pos']),
               win_rate=round(float((rets > 0).mean()) * 100, 1) if len(rets) else None,
               avg_win=round(float(rets[rets > 0].mean()) * 100, 2) if (rets > 0).any() else None,
               avg_loss=round(float(rets[rets <= 0].mean()) * 100, 2) if (rets <= 0).any() else None,
               yearly=yearly, n_signal_months=len(res['lists']), n_rebalance=sum(1 for x in res['rebal'] if x['rebalance']),
               avg_n_stocks=round(float(np.mean([x[5] for x in navs])), 2),
               avg_stock_weight=round(float(np.mean([x[2] / x[1] for x in navs])) * 100, 1))
    return out

def main():
    tmp, prefix = sys.argv[1], sys.argv[2]
    cfg = dict(CFG)
    for a in sys.argv[3:]:
        if '=' in a:
            k, val = a.lstrip('-').split('=', 1)
            cfg[k] = type(CFG[k])(json.loads(val)) if not isinstance(CFG[k], str) else val
    res = run(tmp, cfg)
    summ = summarize(res, cfg)
    json.dump(dict(cfg=cfg, summary=summ, rebalance=res['rebal']), open(prefix + '_summary.json', 'w'), ensure_ascii=False, indent=1)
    json.dump(res['lists'], open(prefix + '_lists.json', 'w'), ensure_ascii=False)
    with open(prefix + '_navs.csv', 'w') as f:
        f.write('date,nav,stock_value,etf_value,cash,n_stocks,base_value\n')
        for x in res['navs']: f.write('%s,%.2f,%.2f,%.2f,%.2f,%d,%.2f\n' % x)
    with open(prefix + '_trades.csv', 'w') as f:
        w = csv.writer(f); w.writerow(['code', 'name', 'entry_date', 'exit_date', 'entry_px', 'exit_px', 'cost', 'proceeds', 'ret', 'reason'])
        for x in res['trades']: w.writerow([x['code'], x['name'], x['entry_date'], x['exit_date'], round(x['entry_px'], 4), round(x['exit_px'], 4), round(x['cost'], 2), round(x['proceeds'], 2), round(x['ret'], 6), x['reason']])
    print(json.dumps(summ, ensure_ascii=False))

if __name__ == '__main__':
    main()
