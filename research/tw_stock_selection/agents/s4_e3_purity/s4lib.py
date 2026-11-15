#!/usr/bin/env python3
"""S4 共用程式庫：把 E3 的優勢留下、把雜訊去掉。
在 s3lib／s2lib（不改）之上包一層，加兩條固定規格：
  (1) 創新板（訊號日當天日 K 的 name 欄以「-創」或「-KY創」結尾）不進股池；
  (2) 最小買進金額：單筆實際可買金額（股數 × 開盤價）低於目標金額（當日淨值 ÷ 10）的一半就不買、不占名額。
引擎 simulate() 是 s2lib.simulate 的逐行複製，只多一個參數 min_frac（0＝與 s2lib 完全相同；0.5＝本題規格）。
一律 python3 -I -B。中間檔（面板快取、創新板旗標）放 scratchpad/s4/。規格見 PREREG.json。"""
import sys, io, csv, re, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S4 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s4')
sys.path.insert(0, str(AG / 's3_winners_momentum'))
import s3lib                                               # noqa: E402
from s3lib import s2lib, stlib, cslib, COSTS               # noqa: E402,F401
s3lib.S3 = S4; s2lib.S2 = S4; stlib.S1 = S4; cslib.TMP = S4   # 面板快取改放本題的 scratchpad（不動 s1～s3 的）
FEE, TAX, MINFEE, TAX_ETF = stlib.FEE, stlib.TAX, stlib.MINFEE, stlib.TAX_ETF
INNOV_PAT = re.compile(r'-(KY)?創$')
STOCK_LIST = '/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv'


# ---------------------------------------------------------------- 資料
def load():
    S4.mkdir(parents=True, exist_ok=True)
    P = s3lib.load()                                       # s2lib.load＋RET20＋還原開盤 AO
    return P


def innov_panel(P):
    """T×N 布林：該日日 K 的 name 欄符合 -(KY)?創$（創新板）。快取在 scratchpad/s4/innov.npz。"""
    f = S4 / 'innov.npz'
    if f.exists():
        z = np.load(f, allow_pickle=False)
        if z['IDX'].shape[0] == P.T and np.array_equal(z['COLS'], P.COLS.astype(str)):
            return z['A'], dict(zip(z['codes'].tolist(), z['names'].tolist()))
    di = {d: i for i, d in enumerate(pd.DatetimeIndex(P.IDX).strftime('%Y-%m-%d'))}
    A = np.zeros((P.T, P.N), bool); names = {}
    for path, fh in cslib._git_blobs('data/universe/daily'):
        d = path.split('/')[-1][:10]
        if d not in di:
            continue
        data = fh.read().decode('utf-8')
        if '創' not in data:
            continue
        lines = data.split('\n')
        for r in csv.DictReader(io.StringIO(lines[0] + '\n' + '\n'.join(x for x in lines[1:] if '創' in x))):
            sid = r['stock_id']; nm = r['name'].strip()
            if sid in P.ci and INNOV_PAT.search(nm):
                A[di[d], P.ci[sid]] = True; names.setdefault(sid, nm)
    np.savez_compressed(f, A=A, IDX=np.asarray(P.IDX).astype('datetime64[ns]'), COLS=P.COLS.astype(str),
                        codes=np.array(list(names.keys()), dtype=str), names=np.array(list(names.values()), dtype=str))
    return A, names


def industry_map(P):
    """產業：stock_list.csv 的 industry（現行分類）；另附代號前兩碼。"""
    try:
        sl = pd.read_csv(STOCK_LIST, dtype={'code': str}).drop_duplicates('code').set_index('code')['industry']
    except Exception:
        sl = pd.Series(dtype=str)
    return {c: (sl.get(c) if isinstance(sl.get(c), str) else '（不在現行清單）') for c in P.COLS}


# ---------------------------------------------------------------- 訊號
def make_signal(P, univ, RT, depth=None, mom_lo=0.10, mom_hi=None, nsel=10, exit='top20', shift=0, keep_info=False):
    """每月一次（訊號日同 S2／S3）。
    L_MA＝訊號日股池內、SURGE 有值、還原收盤 > 60 日均線，依 SURGE 由大到小（stable）。
    買進名單＝L_MA（depth 有值時只看前 depth 名）中 mom_lo < RET20（mom_hi 有值時再加 RET20 ≤ mom_hi），依 SURGE 取前 nsel 名。
    出場：exit='top20' → 不在 L_MA 前 20 名就賣（E3）；exit='f4' → 在 L_MA 前 10 名續抱，11～20 名且 RET20 > 0 續抱，其餘賣。
    照 E3 字面：換股日要賣的持股即使在買進名單內也賣、當天不買回。
    有訊號的月份以未位移訊號日股池內 SURGE 有值 ≥ 50 檔決定（同 s2lib）。"""
    S, av = RT['S'], RT['av']
    sig = np.zeros((P.T, P.N), bool); X = np.zeros((P.T, P.N), bool); rank = np.full((P.T, P.N), np.nan)
    info = []; pos_ma = {}; kmonth = {}
    for m in S.index:
        k0 = av[m]
        if k0 >= P.T or k0 < 61:
            continue
        s = S.loc[m].to_numpy()
        if int((univ[k0] & np.isfinite(s)).sum()) < 50:
            continue
        k = k0 + shift
        if k >= P.T:
            continue
        ok_ma = univ[k] & np.isfinite(s) & P.F_MA60[k]
        o_ma = np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]
        r20 = P.RET20[k]
        pool = o_ma if depth is None else o_ma[:depth]
        okr = r20[pool] > mom_lo
        if mom_hi is not None:
            okr = okr & (r20[pool] <= mom_hi)
        tgt = pool[okr][:nsel]
        if exit == 'top20':
            keep = o_ma[:20]
        elif exit == 'f4':
            mid = o_ma[10:20]
            keep = np.concatenate([o_ma[:10], mid[r20[mid] > 0]])
        else:
            raise ValueError(exit)
        sig[k, tgt] = True; rank[k, tgt] = s[tgt]
        X[k, :] = True; X[k, keep] = False
        kmonth[k] = m
        if keep_info:
            pos_ma[k] = {int(j): i + 1 for i, j in enumerate(o_ma)}
            nth = lambda n: float(s[o_ma[n - 1]]) if len(o_ma) >= n else float('nan')
            info.append(dict(month=str(m), k=int(k), sig_date=str(P.IDX[k].date()), n_ma=int(ok_ma.sum()), n_tgt=int(len(tgt)),
                             targets=[str(P.COLS[j]) for j in tgt], surge_10th=nth(10), surge_15th=nth(15), surge_20th=nth(20)))
        else:
            info.append(dict(k=int(k), n_tgt=int(len(tgt))))
    return dict(sig=sig, rank=rank, X=X, SD=None, max_hold=None, info=info, pos_ma=pos_ma, kmonth=kmonth)


# ---------------------------------------------------------------- 引擎（s2lib.simulate 逐行複製＋min_frac）
def simulate(P, S, univ, idle='cash', cost=None, block=None, capital=1e6, slots=10, cap_pct=0.05, sweep_min=20000.0, buy_at='open', e50_fix=True, min_frac=0.5):
    """同 s2lib.simulate（e50_fix 預設 True）。min_frac：最後決定的股數 × 開盤價 < min_frac × 目標金額（當日淨值 ÷ slots）就不買、不占名額
    （計入 st['blk_min']）。min_frac=0 時與 s2lib.simulate 逐筆相同。"""
    cost = cost or COSTS['light']; block = block or {}
    cm = cost['cmult']; xs = cost['xslip'] * cm; ox = cost['oddx'] * cm
    sig, rank, X, max_hold = S['sig'], S['rank'], S['X'], S['max_hold']
    T, t0 = P.T, P.t0
    O, C, R, REF, AMT, G, CR = P.O, P.CF, P.R, P.REF, P.AMT, P.G, P.C
    fee, tax, minfee = FEE * cm, TAX * cm, MINFEE * cm
    tick = cslib.tick; _fee = stlib._fee; etf_tick = stlib.etf_tick
    odd_t = int(np.searchsorted(P.IDX.values, np.datetime64(cslib.ODD_DATE), side='left'))
    cash = float(capital); v50 = 0.0; pos = {}
    nav = np.full(T, np.nan); nav[t0] = capital
    log = []
    st = dict(n_sig=0, n_buy=0, blk_limup=0, blk_noopen=0, blk_lot=0, blk_cap=0, blk_nocash=0, blk_min=0, noslot=0, blk_dropped=0, sell_defer_limdn=0, sell_defer_noopen=0,
              sell_partial_cap=0, n_delist=0, buy_val=0.0, sell_val=0.0, fee=0.0, tax=0.0, slip=0.0, e50_cost=0.0, e50_orders=0, npos_sum=0, idle_frac_sum=0.0,
              odd_buys=0, odd_buy_val=0.0)

    def banned(j, t):
        for a, b in block.get(j, ()):
            if a <= t <= b:
                return True
        return False

    def e50_net(po):
        nonlocal cash, v50
        sl50 = 0.5 * cm * etf_tick(po) / po
        if cash < -1e-6 and v50 > 0:
            need = -cash
            x = min(v50, (need + minfee) / ((1 - sl50) * (1 - fee - TAX_ETF * cm)))
            for _ in range(3):
                g = x * (1 - sl50); f = max(minfee, g * fee); net = g - f - g * TAX_ETF * cm
                if net >= need or x >= v50:
                    break
                x = min(v50, x + (need - net) * 1.002)
            g = x * (1 - sl50); f = max(minfee, g * fee); net = g - f - g * TAX_ETF * cm
            v50 -= x; cash += net; st['e50_cost'] += x - net; st['e50_orders'] += 1
        elif cash >= sweep_min:
            f = max(minfee, cash * fee)
            val = (cash - f) / (1 + sl50)
            st['e50_cost'] += cash - val; st['e50_orders'] += 1
            v50 += val; cash = 0.0

    def buys(t, cand, at_close, po):
        nonlocal cash
        free = slots - len(pos)
        if free <= 0 or not cand:
            return
        rk = rank[t - 1, cand]
        order = [cand[i] for i in np.lexsort((np.arange(len(cand)), -np.nan_to_num(rk, nan=-1e18)))]
        e50_liq = v50 * (1 - 0.5 * cm * etf_tick(po) / po) * (1 - fee - TAX_ETF * cm) - (minfee if v50 > 0 else 0.0)
        e50_liq = max(e50_liq, 0.0) if idle == 'e50' else 0.0
        if at_close:
            eq = cash + v50 + sum(p['v'] for p in pos.values())
        else:
            eq = cash + v50 + sum(p['v'] * (O[t, k] / REF[t, k] if np.isfinite(O[t, k]) and O[t, k] > 0 else 1.0) for k, p in pos.items())
        target = eq / slots
        for j in order:
            if free <= 0:
                st['noslot'] += 1; continue
            if banned(j, t):
                st['blk_dropped'] += 1; continue
            o = CR[t, j] if at_close else O[t, j]
            if not (np.isfinite(o) and o > 0):
                st['blk_noopen'] += 1; continue
            if o >= P.LIMUP[t, j] - 1e-9:
                st['blk_limup'] += 1; continue
            sl = 0.5 * cm * float(tick(o)) / o + xs
            px = o * (1 + sl)
            avail = (max(cash, 0.0) + e50_liq) if e50_fix else (cash + e50_liq)
            budget = min(target, avail)
            sh = int(budget / (px * (1 + fee) + (o * ox if t >= odd_t else 0.0)))
            if t < odd_t:
                sh = (sh // 1000) * 1000
            if sh <= 0:
                st['blk_lot' if budget >= 0.5 * target else 'blk_nocash'] += 1; continue
            if cap_pct and np.isfinite(AMT[t, j]):
                mx = int(cap_pct * AMT[t, j] / o)
                if t < odd_t:
                    mx = (mx // 1000) * 1000
                if sh > mx:
                    sh = mx; st['blk_cap'] += 1
                    if sh <= 0:
                        continue
            f = _fee(sh, px, fee, minfee); oc = (sh % 1000) * o * ox
            while sh > 0 and sh * px + f + oc > avail + 1e-6:
                sh -= 1000 if t < odd_t else 1
                f = _fee(sh, px, fee, minfee) if sh > 0 else 0.0; oc = (sh % 1000) * o * ox
            if sh <= 0:
                st['blk_nocash'] += 1; continue
            if min_frac > 0 and sh * o < min_frac * target:          # S4 固定規格 2：最小買進金額
                st['blk_min'] += 1; continue
            cst = sh * px + f + oc
            if cash < cst:
                e50_liq -= (cst - max(cash, 0.0))
            cash -= cst
            st['buy_val'] += sh * o; st['fee'] += f; st['slip'] += sh * o * sl + oc; st['n_buy'] += 1
            if sh % 1000:
                st['odd_buys'] += 1; st['odd_buy_val'] += (sh % 1000) * o
            pos[j] = dict(v=sh * o, sh=sh, sh0=sh, e=t, sig=t - 1, cost=cst, flag=False, why='', proc=0.0, new=not at_close)
            free -= 1

    for t in range(t0 + 1, T):
        if idle == 'e50' and v50 > 0:
            v50 *= P.E50O[t] / P.E50[t - 1]
        sold_today = set()
        for j in list(pos):
            p = pos[j]
            if P.MK[t, j] == 0 and not P.MK[t:, j].any():
                cash += p['v']; st['n_delist'] += 1
                log.append((j, p['sig'], p['e'], t, p['cost'], p['proc'] + p['v'], 'delist', p['sh0'])); del pos[j]; sold_today.add(j); continue
            if not p['flag']:
                continue
            o = O[t, j]
            if not (np.isfinite(o) and o > 0):
                st['sell_defer_noopen'] += 1; continue
            if o <= P.LIMDN[t, j] + 1e-9:
                st['sell_defer_limdn'] += 1; continue
            vopen = p['v'] * o / REF[t, j]
            frac = 1.0
            if cap_pct and np.isfinite(AMT[t, j]) and vopen > cap_pct * AMT[t, j]:
                frac = cap_pct * AMT[t, j] / vopen; st['sell_partial_cap'] += 1
            sh = int(round(p['sh'] * frac))
            sl = 0.5 * cm * float(tick(o)) / o + xs
            if ox > 0 and sh > 0 and t >= odd_t:
                sl += ox * (sh % 1000) / sh
            gross = vopen * frac * (1 - sl)
            f = _fee(max(sh, 1), gross / max(sh, 1), fee, minfee); tx = gross * tax
            cash += gross - f - tx; sold_today.add(j)
            st['sell_val'] += vopen * frac; st['fee'] += f; st['tax'] += tx; st['slip'] += vopen * frac * sl
            if frac >= 1.0:
                log.append((j, p['sig'], p['e'], t, p['cost'], p['proc'] + gross - f - tx, p['why'], p['sh0'])); del pos[j]
            else:
                p['proc'] += gross - f - tx; p['v'] *= (1 - frac); p['sh'] = max(p['sh'] - sh, 0)
        cand = np.flatnonzero(sig[t - 1] & univ[t - 1])
        if len(cand):
            held = set(pos) | sold_today
            cand = [j for j in cand if j not in held]
            st['n_sig'] += len(cand)
        else:
            cand = []
        po_o = P.E50O_raw[t] if np.isfinite(P.E50O_raw[t]) and P.E50O_raw[t] > 0 else 100.0
        close_day = (buy_at == 'close') and len(cand) > 0
        if buy_at == 'open' and cand:
            buys(t, cand, False, po_o)
        if idle == 'e50':
            if not close_day:
                e50_net(po_o)
            v50 *= P.E50[t] / P.E50O[t]
        for j, p in pos.items():
            if p['new']:
                p['v'] = p['v'] * (C[t, j] / O[t, j]); p['new'] = False
            else:
                p['v'] *= (1 + R[t, j])
        if close_day:
            po_c = P.E50C_raw[t] if np.isfinite(P.E50C_raw[t]) and P.E50C_raw[t] > 0 else 100.0
            buys(t, cand, True, po_c)
            if idle == 'e50':
                e50_net(po_c)
        tot = 0.0
        for j, p in pos.items():
            tot += p['v']
            if not p['flag']:
                if X[t, j]:
                    p['flag'] = True; p['why'] = 'rule'
                elif max_hold is not None and t - p['e'] >= max_hold - 1:
                    p['flag'] = True; p['why'] = 'time'
        nav[t] = cash + v50 + tot
        st['npos_sum'] += len(pos); st['idle_frac_sum'] += (cash + v50) / nav[t]
    for j, p in pos.items():
        log.append((j, p['sig'], p['e'], T - 1, p['cost'], p['proc'] + p['v'], 'open_end', p['sh0']))
    navs = pd.Series(nav[t0:T], index=P.IDX[t0:T])
    lg = pd.DataFrame(log, columns=['j', 'sig_t', 'ent_t', 'exit_t', 'cost', 'proceeds', 'why', 'shares'])
    if len(lg):
        lg['code'] = P.COLS[lg['j'].to_numpy()]; lg['ret'] = lg['proceeds'] / lg['cost'] - 1; lg['pnl'] = lg['proceeds'] - lg['cost']
        lg['hold_days'] = lg['exit_t'] - lg['ent_t']
        lg['sig_date'] = P.IDX[lg['sig_t'].to_numpy()]; lg['ent_date'] = P.IDX[lg['ent_t'].to_numpy()]; lg['exit_date'] = P.IDX[lg['exit_t'].to_numpy()]
        lg['market'] = np.where(P.MK[lg['ent_t'].to_numpy(), lg['j'].to_numpy()] == 1, 'twse', 'tpex')
        lg['score'] = rank[lg['sig_t'].to_numpy(), lg['j'].to_numpy()]
    nd = T - 1 - t0
    st['avg_npos'] = st.pop('npos_sum') / nd; st['avg_idle_frac'] = st.pop('idle_frac_sum') / nd
    return navs, lg, st


# ---------------------------------------------------------------- 統計小工具
def desc(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(n=0)
    return dict(n=int(len(x)), mean=float(x.mean()), median=float(np.median(x)), q25=float(np.percentile(x, 25)), q75=float(np.percentile(x, 75)),
                win=float((x > 0).mean()), loss_gt15=float((x < -0.15).mean()))


def auc(a, b):
    """P(a 的值 > b 的值)＋0.5×P(相等)：a、b 兩組特徵值的分得開程度（0.5＝分不開）。"""
    a = np.asarray(a, float); b = np.asarray(b, float); a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    g = (a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean()
    return float(g)
