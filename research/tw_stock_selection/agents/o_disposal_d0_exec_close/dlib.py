#!/usr/bin/env python3
"""o_disposal_d0_exec_close 共用核心。
資料與事件規則：只讀 import o_disposal_official_fullhist/core.py（disposal_v2 V_OFF 官方版逐字；V7X 快取 x_cache.npz），
名單改讀本題 raw/ 複本（sha256 與上游相同才放行）。
本檔新增的只有「可執行模型」：
  (1) 收盤成交 → 最後一盤（或第一盤）集合競價成交，參與上限 cap×(當日 V × 該盤占比)，超過的整張縮單（買）或順延（賣）；
  (2) 整張以外走零股（零股價差 odd_spread／邊，各自付最低手續費 20 元）；
  (3) 嚴格預收款券＋T+2 逐日現金帳（現金池 k 槽；出場款 T+2 才可用）；
  (4) R3 改成交規則：以漲停價（或鄰域限價）掛單，該盤價格 ≥ 限價（鎖漲停）視為未成交；R5 同理改成交規則（跌停鎖死順延）。
所有決策只用 s−1 收盤前已知的資訊（公告日 ≤ s−1、R4 用 V60p／PC、R2 用前一指數日）。
"""
import sys, math, json, hashlib
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
sys.path.insert(0, str(AG / 'o_disposal_official_fullhist'))
import core as K  # noqa: E402  只讀（o9 凍結核心）

SP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad')
LIST_RAW = HERE / 'raw' / 'disposal_hist_rejudge.csv'
assert K.sha(LIST_RAW) == K.sha(K.LIST_F), '本題 raw 名單與上游不同'

IDX, T, CI, COLS = K.IDX, K.T, K.CI, K.COLS
C, O, V, R, TR, PC = K.C, K.O, K.V, K.R, K.TR, K.PC
V60p, SIGp, REF, LIMUP, LIMDN = K.V60p, K.SIGp, K.REF, K.LIMUP, K.LIMDN
LA, LOGCO, LEW, LEW_OPEN, LE5, E50, LASTV = K.LA, K.LOGCO, K.LEW, K.LEW_OPEN, K.LE5, K.E50, K.LASTV
ETF_BUY, ETF_SELL, TAX = K.ETF_BUY, K.ETF_SELL, K.TAX
ODD_SWITCH = pd.Timestamp('2020-10-26'); I_ODD = int(IDX.searchsorted(ODD_SWITCH))
PERIODIC_START = pd.Timestamp('2010-12-01')
NEWREG = pd.Timestamp('2026-08-10')

PER = {'SEL': ('2021-10-01', '2023-12-31'), 'VAL': ('2024-01-02', '2026-10-22'),
       'FULL': ('2006-04-01', '2026-10-22'), 'NEWREG': ('2026-08-10', '2026-10-22'),
       'OOS_o9': ('2006-04-01', '2021-09-30'), 'P1_2006_2012': ('2006-04-01', '2012-12-31'),
       'P2_2013_2018': ('2013-01-01', '2018-12-31'), 'P3_2019_2026': ('2019-01-01', '2026-10-22'),
       'Y2006_2013': ('2006-04-01', '2013-12-31'), 'Y2014_': ('2014-01-01', '2026-10-22'), '2019+': ('2019-01-01', '2026-10-22')}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def canon_sha(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


# ---------- 共用面板 high 與來源（只用於鎖漲停描述與 src3 量修正） ----------
def _panel(name):
    p = pd.read_parquet(SP / 'panel' / f'{name}.parquet')
    return p.reindex(index=IDX, columns=COLS).to_numpy().astype(float)


HIGH = _panel('high')
SRC = _panel('src')


def load_list():
    d = pd.read_csv(LIST_RAW, dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
    d['ordr'] = np.arange(len(d))
    d['name'] = d['name'].fillna('')
    d['is_common'] = (d['code'].str.len() == 4) & (~d['code'].str.startswith('0')) & (~d['name'].str.contains('DR'))
    return d


U = K.universe(load_list())


# ---------- 執行參數（預設＝事前登記主欄；EXEC_SPEC.json 凍結） ----------
CALIB_F = HERE / 'calib5m.json'


def default_prm():
    cal = json.loads(CALIB_F.read_text()) if CALIB_F.exists() else {}
    sh = cal.get('frozen_shares', {})
    return dict(
        cap=0.20, kappa=0.5, odd_spread=0.004, stress=False, stress_extra=0.004, stress_floor_side=0.01,
        sh_last={'2': sh.get('last_2', 0.08), '5': sh.get('last_5', 0.06), '10': sh.get('last_10', 0.08),
                 '20': sh.get('last_20', 0.10), 'long': sh.get('last_long', 0.12)},
        sh_first={'2': sh.get('first_2', 0.05), '5': sh.get('first_5', 0.05), '10': sh.get('first_10', 0.06),
                  '20': sh.get('first_20', 0.08), 'long': sh.get('first_long', 0.10)},
        sh_close_norm=sh.get('close_norm', 0.05), sh_open_norm=sh.get('open_norm', 0.03),
        src3_close_fix=True,
        odd_min_fee=20.0, lot_min_fee=20.0)


def mm_key(mm):
    if mm is None or not np.isfinite(mm) or mm <= 0: return None
    if mm <= 3: return '2'
    if mm <= 7: return '5'
    if mm <= 12: return '10'
    if mm <= 22: return '20'
    return 'long'


def auction_Q(t, j, ev, kind, prm):
    """該盤可成交值參考 Qa（NT$）。處置期內且為分盤撮合 → 依撮合分鐘的最後一盤／第一盤占比；
    只預收期（連續撮合）與處置期外 → 一般收盤／開盤集合競價占比。src 3（Yahoo，缺 13:30 收盤盤）日的 V 先除以 (1−收盤占比) 還原。"""
    v = V[t, j]
    if not np.isfinite(v) or v <= 0: return 0.0
    ind = ev['s_i'] <= t <= ev['e_i']
    key = mm_key(ev['match_min']) if (ind and ev['regime'] == 'periodic_match') else None
    if key is None:
        sh = prm['sh_close_norm'] if kind == 'C' else prm['sh_open_norm']
    else:
        sh = prm['sh_last'][key] if kind == 'C' else prm['sh_first'][key]
    if prm.get('src3_close_fix') and SRC[t, j] == 3:
        v = v / (1 - sh if kind == 'C' else 1 - prm['sh_close_norm'])
    return v * sh


def fee_rate(t, stress):
    return K.fee_rate(t, stress)


def lot_imp(t, j, q, Qa, prm, px):
    tk = float(K.tick(px)) / px if px > 0 else 0.001
    sg = SIGp[t, j] if np.isfinite(SIGp[t, j]) else 0.03
    return max(tk, 0.001) + (prm['kappa'] * sg * math.sqrt(q / Qa) if Qa > 0 and q > 0 else 0.0) + (prm['stress_extra'] if prm['stress'] else 0.0)


def odd_imp(prm):
    return prm['odd_spread'] + (prm['stress_extra'] if prm['stress'] else 0.0)


def px_at(t, j, kind):
    return C[t, j] if kind == 'C' else O[t, j]


def lvl(t, j, kind):
    return LA[t, j] - (LOGCO[t, j] if kind == 'O' else 0.0)


def limit_price(t, j, off):
    """買進限價：off=0 → 漲停價；off>0 → REF×(1＋漲跌幅−off) 取跳動單位下緣。"""
    if off <= 0: return LIMUP[t, j]
    lim = K.limit_rate(t)
    p = REF[t, j] * (1 + lim - off)
    tk = float(K.tick(p))
    return math.floor(p / tk + 1e-9) * tk


def buy_fill_ok(t, j, kind, off):
    if t >= T or not TR[t, j]: return False
    p = px_at(t, j, kind)
    L = limit_price(t, j, off)
    if not (np.isfinite(p) and p > 0 and np.isfinite(L)): return False
    return bool(p < L - 1e-9)


def sell_ok_close(t, j, r5):
    if not TR[t, j]: return False
    if r5 == 'orig':
        return bool(np.isfinite(R[t, j]) and R[t, j] > -(K.limit_rate(t) - 0.01))
    return bool(np.isfinite(C[t, j]) and np.isfinite(LIMDN[t, j]) and C[t, j] > LIMDN[t, j] + 1e-9)


# ---------- 事件建構（事件規則逐字沿用 o9；只換 R3／R5 為成交規則） ----------
DEF_CFG = dict(entry='C1', exit='CL', bear=True, liq=1e7, px_min=10.0, off=0.0, r5='fill', first_only=False,
               kinds=('regular', 'changed_method', 'supervisory'), chain=True)


def build_x(cfg=None):
    cfg = dict(DEF_CFG, **(cfg or {}))
    st = dict(records=0, kind=0, chain_R1=0, first_only_drop=0, not_in_panel=0, invalid=0, busy=0, bear=0,
              unfilled_lock=0, illiquid=0, no_exit=0, kept=0, delayed=0, ann_late=0)
    rows = []; unf = []; busy = {}
    uu = U.sort_values(['s_i', 'announce', 'ordr'], kind='stable')
    for r in uu.itertuples():
        st['records'] += 1
        if r.kind not in cfg['kinds']: st['kind'] += 1; continue
        if cfg['chain'] and not r.take_R1: st['chain_R1'] += 1; continue
        if cfg['first_only'] and r.second30: st['first_only_drop'] += 1; continue
        if not r.in_D: st['not_in_panel'] += 1; continue
        if r.s_i >= T or r.e_i <= r.s_i: st['invalid'] += 1; continue
        if r.e_i >= T - 1 and cfg['exit'] == 'OA': st['invalid'] += 1; continue
        j = CI[r.code]
        i0 = r.s_i if cfg['entry'] in ('O1', 'C1') else r.s_i + 1
        ek = 'C' if cfg['entry'] == 'C1' else 'O'
        if i0 >= T: st['invalid'] += 1; continue
        if not (r.ann_i <= i0 - 1): st['ann_late'] += 1; continue      # 只用前一日資訊：公告日 ≤ 進場前一日
        if busy.get(r.code, -1) >= r.s_i: st['busy'] += 1; continue
        if cfg['bear'] and K.bear_before(IDX[r.s_i]): st['bear'] += 1; continue
        v60 = V60p[r.s_i, j]; pc = PC[r.s_i, j]
        if not (np.isfinite(v60) and v60 >= cfg['liq'] and np.isfinite(pc) and pc >= cfg['px_min']): st['illiquid'] += 1; continue
        base = dict(code=r.code, j=j, ordr=r.ordr, announce=r.announce, start=r.start, end=r.end, s_i=r.s_i, e_i=r.e_i, ann_i=r.ann_i,
                    i0=i0, ek=ek, date=IDX[i0], second=bool(r.second30), nth=int(r.nth), match_min=r.match_min, regime=r.regime,
                    kind=r.kind, v60=v60, pc=pc)
        if not buy_fill_ok(i0, j, ek, cfg['off']):
            st['unfilled_lock'] += 1; unf.append(dict(base, filled=False)); continue
        # 出場（成交規則）
        lim = min(T - 1, LASTV[j])
        if cfg['exit'] == 'OA':
            k = r.e_i + 1
            if k > lim: i1, xk = lim, 'C'
            elif TR[k, j] and np.isfinite(O[k, j]) and O[k, j] > 0 and not (np.isfinite(LIMDN[k, j]) and O[k, j] <= LIMDN[k, j] + 1e-9):
                i1, xk = k, 'O'
            else:
                k = min(k, lim)
                while k < lim and not sell_ok_close(k, j, cfg['r5']): k += 1
                i1, xk = k, 'C'
        else:
            k = min(r.e_i if cfg['exit'] == 'CL' else r.e_i - 1, lim)
            while k < lim and not sell_ok_close(k, j, cfg['r5']): k += 1
            i1, xk = k, 'C'
        if i1 < i0 or (i1 == i0 and not (ek == 'O' and xk == 'C')): st['no_exit'] += 1; continue
        busy[r.code] = i1; st['kept'] += 1
        planned = r.e_i if cfg['exit'] == 'CL' else (r.e_i - 1 if cfg['exit'] == 'CL1' else r.e_i + 1)
        st['delayed'] += int(i1 > planned)
        a = lvl(i0, j, ek); b = lvl(i1, j, xk)
        raw = math.expm1(b - a)
        ew0 = LEW_OPEN[i0] if ek == 'O' else LEW[i0]; ew1 = LEW_OPEN[i1] if xk == 'O' else LEW[i1]
        ew = math.expm1(ew1 - ew0)
        e5 = math.expm1(LE5[i1 - (1 if xk == 'O' else 0)] - LE5[i0 - (1 if ek == 'O' else 0)])
        run6 = (math.exp(LA[r.s_i - 1, j] - LA[r.s_i - 7, j]) - 1) if r.s_i >= 7 and TR[r.s_i - 1, j] else np.nan
        pre = V[max(r.s_i - 20, 0):r.s_i, j]; dur = V[r.s_i:r.e_i + 1, j]
        vdrop = (np.nanmean(dur) / np.nanmean(pre) - 1) if np.isfinite(pre).sum() >= 10 and np.isfinite(dur).sum() >= 2 else np.nan
        rows.append(dict(base, filled=True, i1=i1, xk=xk, exit_date=IDX[i1], hold=i1 - i0, plan_n=r.e_i - r.s_i,
                         raw=raw, ew=ew, e50=e5, gross=raw - ew, run6=run6, vdrop=vdrop, delayed=bool(i1 > planned),
                         px=px_at(i0, j, ek)))
    df = pd.DataFrame(rows)
    if len(df): df = df.sort_values(['date', 'announce', 'ordr']).reset_index(drop=True)
    return df, pd.DataFrame(unf), st


# ---------- 可執行成交（買／賣） ----------
def cpi_factor(t):
    return K.cpi_factor(t)


def do_buy(ev, t, kind, A, prm, off=0.0):
    """以限價（漲停或鄰域）買進預算 A（NT$）：股數＝floor(A／限價)；整張部分受參與上限縮單；零股另一張單。
    回傳 dict 或 None（0 股）。2020-10-26 前開盤進場的零股只能走當日盤後零股（以收盤價＋零股價差）。"""
    j = ev['j']; p = px_at(t, j, kind); L = limit_price(t, j, off)
    if not (np.isfinite(p) and p > 0 and np.isfinite(L) and L > 0): return None
    N = int(A // L)
    nl, no = N // 1000, N % 1000
    Qa = auction_Q(t, j, ev, kind, prm)
    nl_cap = int((prm['cap'] * Qa) // (p * 1000)) if Qa > 0 else 0
    capped = nl > nl_cap
    nl = min(nl, nl_cap)
    fr = fee_rate(t, prm['stress'])
    out = dict(nl=nl, no=no, capped=capped, Qa=Qa, N_want=N, L=L)
    cash = 0.0; legs = []
    if nl > 0:
        q = nl * 1000 * p
        imp = lot_imp(t, j, q, Qa, prm, p)
        c = q * imp + max(prm['lot_min_fee'], fr * q)
        if prm['stress']: c = max(c, prm['stress_floor_side'] * q)
        cash += q + c
        legs.append(('L', nl * 1000, p, lvl(t, j, kind)))
        out['lot_cost_rate'] = (q * imp + max(prm['lot_min_fee'], fr * q)) / q
    if no > 0:
        if kind == 'O' and t < I_ODD:
            po, lv = C[t, j], lvl(t, j, 'C')
        else:
            po, lv = p, lvl(t, j, kind)
        q = no * po
        c = q * odd_imp(prm) + max(prm['odd_min_fee'], fr * q)
        if prm['stress']: c = max(c, prm['stress_floor_side'] * q)
        cash += q + c
        legs.append(('O', no, po, lv))
        out['odd_cost_rate'] = (q * odd_imp(prm) + max(prm['odd_min_fee'], fr * q)) / q
    if not legs: return None
    out['cash'] = cash; out['legs'] = legs
    out['gross_val'] = sum(sh * px for _, sh, px, _ in legs)
    return out


def pos_value(pos, t, kind='C'):
    j = pos['j']; v = 0.0
    for _, sh, px, lv0 in pos['legs']:
        v += sh * px * math.exp(lvl(t, j, kind) - lv0)
    return v


def sell_step(pos, t, kind, prm, final=False):
    """在 t 的該盤賣出：零股全部（一張單）；整張受參與上限，剩餘順延。回傳 (淨入帳 NT$, 是否賣完)。"""
    ev = pos['ev']; j = ev['j']
    p = px_at(t, j, kind)
    if not (np.isfinite(p) and p > 0):
        kind = 'C'; p = C[t, j]
    fr = fee_rate(t, prm['stress'])
    got = 0.0; newlegs = []
    Qa = auction_Q(t, j, ev, kind, prm)
    for leg in pos['legs']:
        typ, sh, px0, lv0 = leg
        val_sh = px0 * math.exp(lvl(t, j, kind) - lv0)
        if typ == 'O':
            if kind == 'O' and t < I_ODD and not final:   # 2020-10-26 前開盤賣不到零股 → 當日盤後零股（收盤價）
                val_sh = px0 * math.exp(lvl(t, j, 'C') - lv0)
            q = sh * val_sh
            c = q * odd_imp(prm) + max(prm['odd_min_fee'], fr * q) + TAX * q
            if prm['stress']: c = max(c, prm['stress_floor_side'] * q)
            got += q - c
            continue
        lots = sh / 1000.0                      # FIX 縮放後可能非整數，視為連續
        cap_lots = int((prm['cap'] * Qa) // (p * 1000)) if Qa > 0 else 0
        sell_lots = lots if (final or cap_lots >= lots - 1e-9) else float(cap_lots)
        if sell_lots > 0:
            q = sell_lots * 1000 * val_sh
            imp = lot_imp(t, j, q, Qa, prm, p)
            c = q * imp + max(prm['lot_min_fee'], fr * q) + TAX * q
            if prm['stress']: c = max(c, prm['stress_floor_side'] * q)
            got += q - c
        if lots - sell_lots > 1e-9:
            newlegs.append((typ, (lots - sell_lots) * 1000, px0, lv0))
    pos['legs'] = newlegs
    return got, len(newlegs) == 0


def trade_exec(ev, A, prm, off=0.0):
    """逐筆可執行報酬（預算 A）：回傳 dict(ret_x, net_x, net50_x, fill, invest, sell_days, capped)。"""
    j = ev['j']
    b = do_buy(ev, ev['i0'], ev['ek'], A, prm, off)
    if b is None: return dict(ret_x=np.nan, net_x=np.nan, net50_x=np.nan, fill=0.0, invest=0.0, sell_days=0, capped=True)
    pos = dict(ev=ev, j=j, legs=list(b['legs']))
    t = ev['i1']; kind = ev['xk']; got = 0.0; nd = 0
    lim = min(T - 1, LASTV[j])
    while True:
        final = t >= lim
        g, done = sell_step(pos, t, kind, prm, final=final)
        got += g; nd += 1
        if done or final: break
        t += 1; kind = 'C'
        while t < lim and not sell_ok_close(t, j, 'fill'): t += 1
    ret = got / b['cash'] - 1
    return dict(ret_x=ret, net_x=ret - ev['ew'], net50_x=ret - ev['e50'], fill=b['gross_val'] / A, invest=b['cash'],
                sell_days=nd, capped=b['capped'], nl=b['nl'], no=b['no'])


def event_exec(df, slot_2026=1e5, prm=None, off=0.0):
    prm = prm or default_prm()
    out = []
    for ev in df.to_dict('records'):
        A = slot_2026 * cpi_factor(ev['i0'])
        out.append(trade_exec(ev, A, prm, off))
    x = pd.DataFrame(out, index=df.index)
    return pd.concat([df, x], axis=1)


# ---------- 衛星組合：逐日現金帳 ----------
def needs_prepay(ev, nl, rule):
    """預收判定。strict：全部預收。threshold：只預收期全收；第二次（含以上）全收；第一次單筆 ≥ 10 張全收。none：都不預收（o9 口徑）。"""
    if rule == 'strict': return True
    if rule == 'none': return False
    if ev['regime'] == 'prepay_only': return True
    if ev['second'] or ev['nth'] >= 2: return True
    return nl >= 10


def sleeve(trades, unfilled=None, tier_2026=1e6, mode='FIX', K_=10, prm=None, prepay='strict', pool_k=2,
           t_start='2006-04-01', t_end=None, off=0.0, min_frac=0.5, settle=2):
    """10 槽處置衛星。閒置資金：pool_k 槽現金（0 報酬），其餘 0050。
    strict：進場日需有已交割現金 ≥ 下單金額（股數×限價）；0050 賣出款與處置股出場款 T+2 才入帳。
    決策：進場前一日收盤後（公告日 ≤ 進場前一日）；同日依 (announce, 檔案順序)。
    0050 調整：每日收盤依「前一日已知」的待進場需求＋現金池目標，賣出 0050（T+2 入帳）或把多餘已交割現金買回 0050。
    回傳 (NAV Series, stat dict, 逐筆 DataFrame)。"""
    prm = prm or default_prm()
    t0 = int(IDX.searchsorted(pd.Timestamp(t_start))); t1 = T - 1 if t_end is None else int(IDX.searchsorted(pd.Timestamp(t_end), 'right') - 1)
    evs = [r for r in trades.to_dict('records') if t0 + 1 <= r['i0'] <= t1]
    if unfilled is not None and len(unfilled):
        evs += [r for r in unfilled.to_dict('records') if t0 + 1 <= r['i0'] <= t1]
    evs.sort(key=lambda r: (r['i0'], r['announce'], r['ordr']))
    by_dec = {}
    for r in evs: by_dec.setdefault(r['i0'] - 1, []).append(r)
    by_ann = {}
    for r in evs: by_ann.setdefault(max(r['ann_i'], t0), []).append(r)
    cap0 = tier_2026 * cpi_factor(t0)
    u50 = cap0 * (1 - ETF_BUY) / E50[t0 - 1]; S = 0.0
    rec = []            # (due_t, amt)
    pos = {}            # key -> dict(ev, j, legs, cash_in)
    resv = []           # [ev, A, key]
    known = {}          # 已知（公告日 ≤ t−1）且未決策的事件 → 預估需求
    nav = np.empty(t1 - t0 + 1); inv = np.empty(t1 - t0 + 1); cash_share = np.empty(t1 - t0 + 1)
    navi = 1.0; prevE = cap0
    st = dict(decided=0, entered=0, no_slot=0, held=0, cash_skip=0, cash_partial=0, unfilled_lock=0, zero_fill=0,
              capped=0, deferred_sell=0, n_eval=0, min_settled_cash=0.0)
    rows = []
    strictish = prepay in ('strict', 'threshold')

    def slot_amt(t, E):
        return (E / K_) if mode == 'GROW' else tier_2026 * cpi_factor(t) / K_

    def equity(t):
        pv = sum(pos_value(p, t, 'C') for p in pos.values())
        return u50 * E50[t] + S + sum(a for _, a in rec) + pv, pv

    for t in range(t0, t1 + 1):
        # 0) 交割入帳
        if rec:
            due = [a for d, a in rec if d <= t]
            if due:
                S += sum(due); rec = [(d, a) for d, a in rec if d > t]
        # 1) 進場
        keep = []
        for ev, A, key in resv:
            if ev['i0'] != t: keep.append((ev, A, key)); continue
            st['n_eval'] += 1
            if not ev.get('filled', True):
                st['unfilled_lock'] += 1; continue
            # 試算整張張數判定預收
            L = limit_price(t, ev['j'], off)
            nl_try = int((A // L) // 1000) if np.isfinite(L) and L > 0 else 0
            pp = needs_prepay(ev, nl_try, prepay) if strictish else False
            A_use = A
            if pp:
                if S < A * min_frac:
                    st['cash_skip'] += 1; rows.append(dict(code=ev['code'], date=IDX[t], status='cash_skip', A=A, S=S)); continue
                if S < A: st['cash_partial'] += 1; A_use = S
            b = do_buy(ev, t, ev['ek'], A_use, prm, off)
            if b is None:
                st['zero_fill'] += 1; continue
            if b['capped']: st['capped'] += 1
            if pp:
                S -= b['cash']
                st['min_settled_cash'] = min(st['min_settled_cash'], S)
            else:
                # 非預收：當日賣 0050 支應（T+2 對沖），以 0050 單位扣
                use = min(max(S, 0.0), b['cash']); S -= use
                need = b['cash'] - use
                if need > 0:
                    u50 -= need / (1 - ETF_SELL) / E50[t]
            pos[key] = dict(ev=ev, j=ev['j'], legs=list(b['legs']), cash_in=b['cash'], t_in=t)
            st['entered'] += 1
            rows.append(dict(code=ev['code'], date=IDX[t], status='entered', A=A, invest=b['cash'], fill=b['gross_val'] / A,
                             prepay=pp, nl=b['nl'], no=b['no'], capped=b['capped']))
        resv = keep
        # 2) 出場
        for key in list(pos.keys()):
            p = pos[key]; ev = p['ev']; j = p['j']
            if t < ev['i1']: continue
            lim = min(T - 1, LASTV[j])
            final = t >= lim
            if t == ev['i1']:
                kind = ev['xk']
            else:
                if not final and not sell_ok_close(t, j, 'fill'): st['deferred_sell'] += 1; continue
                kind = 'C'
            g, done = sell_step(p, t, kind, prm, final=final)
            if g:
                if strictish: rec.append((t + settle, g))
                else: u50 += g * (1 - ETF_BUY) / E50[t]
            if done or final:
                pos.pop(key)
            else:
                st['deferred_sell'] += 1
        # 3) 收盤後決策：明日進場的事件（公告日 ≤ t）
        E, pv = equity(t)
        for ev in by_dec.get(t, []):
            if any(pp['ev']['code'] == ev['code'] for pp in pos.values()) or any(rr['code'] == ev['code'] for rr, _, _ in resv):
                st['held'] += 1; continue
            if len(pos) + len(resv) >= K_:
                st['no_slot'] += 1; continue
            A = slot_amt(t + 1, E)
            if not strictish:
                avail = u50 * E50[t] * (1 - ETF_SELL) + S
                A = min(A, avail)
                if A <= 0: continue
            resv.append((ev, A, (ev['code'], ev['ordr'])))
            st['decided'] += 1
        # 4) 0050／現金池調整（strict／threshold）：今日收盤賣 0050 的依據只用 t−1 已知的事件
        if strictish:
            E, pv = equity(t)
            sl = slot_amt(t, E)
            # 已知未進場需求：已保留（明日以後進場）＋公告日 ≤ t−1、進場日 ≥ t+2 的事件
            need_known = sum(A for ev, A, _ in resv)
            n_open = len(pos) + len(resv)
            pend_ann = 0
            for d in range(max(t0, t - 3), t):
                for ev in by_ann.get(d, []):
                    if ev['i0'] >= t + 2 and n_open + pend_ann < K_: pend_ann += 1
            target = pool_k * sl + need_known + pend_ann * sl
            have = S + sum(a for _, a in rec)
            if have < target - 1e-6:
                sell = min(target - have, u50 * E50[t] * (1 - ETF_SELL))
                if sell > 0:
                    u50 -= sell / (1 - ETF_SELL) / E50[t]; rec.append((t + settle, sell))
            elif S > target + sl:
                buy = S - target
                S -= buy; u50 += buy * (1 - ETF_BUY) / E50[t]
        E, pv = equity(t)
        navi *= E / prevE; prevE = E
        nav[t - t0] = navi; inv[t - t0] = pv / E if E > 0 else 0; cash_share[t - t0] = (S + sum(a for _, a in rec)) / E if E > 0 else 0
        if mode == 'FIX' and t < t1 and IDX[t + 1].month != IDX[t].month and E > 0:
            sc = tier_2026 * cpi_factor(t + 1) / E
            u50 *= sc; S *= sc
            rec = [(d, a * sc) for d, a in rec]
            for p in pos.values():
                p['legs'] = [(ty, sh * sc, px, lv) for ty, sh, px, lv in p['legs']]
            resv = [(ev, A * sc, k) for ev, A, k in resv]
            prevE = E * sc
    s = pd.Series(nav, index=IDX[t0:t1 + 1])
    st['mean_invested'] = round(float(inv.mean()), 4)
    st['mean_cash_share'] = round(float(cash_share.mean()), 4)
    elig = st['entered'] + st['cash_skip']
    st['cash_skip_rate'] = round(st['cash_skip'] / elig, 4) if elig else None
    return s, st, pd.DataFrame(rows)


# ---------- 統計 ----------
tstat = K.tstat
S_ev = K.S
cluster_t = K.cluster_t


def sub(df, a, b):
    return df[(df['date'] >= pd.Timestamp(a)) & (df['date'] <= pd.Timestamp(b))]


def monthly(nav):
    return K.monthly(nav)


def ex_vs_0050(nav):
    nav = nav.dropna()
    e = pd.Series(E50, index=IDX).reindex(nav.index)
    return (monthly(nav) - monthly(e)).dropna()


def vs0050(nav, a=None, b=None):
    return K.vs0050(nav, a, b)


def mstats(x, a=None, b=None):
    x = x.dropna()
    if a: x = x[x.index >= pd.Timestamp(a)]
    if b: x = x[x.index <= pd.Timestamp(b) + pd.offsets.MonthEnd(0)]
    if len(x) < 3: return {'n_months': int(len(x))}
    return {'mean_pp': round(float(x.mean() * 100), 3), 't': tstat(x), 'n_months': int(len(x)),
            'sd_pp': round(float(x.std(ddof=1) * 100), 3)}


def per_stats(nav, a, b):
    """期間年化／最大回撤／對 0050 月超額 t（NAV 以期間起點切片）。"""
    n = nav[(nav.index >= pd.Timestamp(a)) & (nav.index <= pd.Timestamp(b))].dropna()
    if len(n) < 40: return {'n_days': int(len(n))}
    # 以期間前一日為基準：月報酬從期間第一個月起
    return K.vs0050(n)
