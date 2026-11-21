#!/usr/bin/env python3
"""R86 獨立複製用程式庫（不讀、不 import s86_retail_after_tax_frictions 底下任何 .py）。
做法：把共用引擎 s10lib.simulate_s10 的原始碼「讀成文字 → 在副本上插掛鉤 → 在記憶體裡執行」，不改任何既有檔案。
掛鉤（全部關閉時與原引擎逐日相同，run_r86.py 會驗收）：
  1. 每天開頭 _div_hook(t)：除息日（TWT49U）算出帳上個股與 0050 部位收到的現金股利（＝前一日收盤市值 × 息值 ÷ 除息前收盤），
     記錄；依設定從部位扣掉股利稅（tau）、補充保費（單筆 ≥ 門檻才扣 rate）；延後模式（delay>0）則整筆股利先離開部位、
     變成「應收股利」（算在淨值裡、不生息），delay 個交易日後以現金入帳（之後照引擎原規則：滿 2 萬才掃進 0050，或等下次買股）。
  2. 手續費折扣、最低手續費改成可調（執行時把副本的 FEE／MINFEE 換掉）。
  3. 每筆下單記錄「因 20 元低消多付的錢」（實付手續費 − 純費率手續費）。
基準口徑（S86 當時的正式數字）：EXRIGHT_FIX、DEADLINE_FIX、INS15_MASK 開；REDUCE_FIX 關；0050 用舊序列（E50_SOURCE＝None）。"""
import sys, gc, inspect, subprocess, io, os
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/r86/cache')
TMP.mkdir(parents=True, exist_ok=True)
_SRC_CACHE = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/offfix/cache')
for _f in ('e50_ohlc.csv', 'innov.npz', 'panel_cs.npz'):      # 原始日 K 面板快取（與開關無關）：正式快取有就複製一份到自己的暫存（不共用、不寫回）；沒有就由引擎從 git 重建
    if not (TMP / _f).exists() and (_SRC_CACHE / _f).exists():
        import shutil; shutil.copy(_SRC_CACHE / _f, TMP / _f)
sys.path.insert(0, str(AG / 's19_xq_threshold'))
import s19lib                                              # noqa: E402
from s19lib import s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402
for mod, attr in ((s19lib, 'S19'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                  (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(mod, attr, TMP)
assert cslib.EXRIGHT_FIX is False and cslib.DEADLINE_FIX is False and cslib.INS15_MASK is False
assert cslib.REDUCE_FIX is False and cslib.E50_SOURCE is None, '共用引擎預設必須是關（S86 口徑：REDUCE_FIX 關、0050 舊序列）'
cslib.EXRIGHT_FIX = True; cslib.DEADLINE_FIX = True; cslib.INS15_MASK = True
FEE_FULL = 0.001425


# ---------------------------------------------------------------- 引擎副本
def _patched_sim():
    src = inspect.getsource(s10lib.simulate_s10)
    def rep(a, b, n=1):
        nonlocal src
        assert src.count(a) == n, (a, src.count(a))
        src = src.replace(a, b)
    rep("def simulate_s10(", "def simulate_r86(")
    hook = '''
    FXd = FX; pend = []; FXd['cur_t'] = t0

    def _mfx(x):
        if FXd.get('mf') is not None and x > 1e-9:
            FXd['mf'].append((FXd['cur_t'], float(x)))

    def _div_hook(t):
        nonlocal cash, v50, B
        FXd['cur_t'] = t
        if pend:
            keep = []
            for due, amt, badd in pend:
                if due <= t:
                    cash += amt; B += badd
                else:
                    keep.append((due, amt, badd))
            pend[:] = keep
        tau, pr, thr, dly, rec = FXd.get('tau', 0.0), FXd.get('prem', 0.0), FXd.get('thr', 20000.0), FXd.get('delay', 0), FXd.get('rec')
        navp = nav[t - 1]
        y = FXd['y50'][t]
        if y > 0 and v50 > 0:
            D = v50 * y
            pm = pr * D if (pr > 0 and D >= thr) else 0.0
            tx = tau * D
            if rec is not None:
                rec.append((t, -1, D, navp, tx, pm))
            out = D if dly > 0 else tx + pm
            if out > 0:
                bpart = min(B, v50) * out / v50
                v50 -= out; B -= bpart
                if dly > 0:
                    pend.append((t + dly, D - tx - pm, bpart * (D - tx - pm) / D))
        yd = FXd['YD'].get(t)
        if yd:
            for j, p in pos.items():
                yy = yd.get(j)
                if yy and p['v'] > 0:
                    D = p['v'] * yy
                    pm = pr * D if (pr > 0 and D >= thr) else 0.0
                    tx = tau * D
                    if rec is not None:
                        rec.append((t, j, D, navp, tx, pm))
                    out = D if dly > 0 else tx + pm
                    if out > 0:
                        p['v'] -= out
                        if dly > 0:
                            pend.append((t + dly, D - tx - pm, 0.0))

    def px_of(t, k):'''
    rep("\n    def px_of(t, k):", hook)
    rep("    for t in range(t0 + 1, T):\n", "    for t in range(t0 + 1, T):\n        _div_hook(t)\n")
    rep("        nav[t] = cash + v50 + tot\n", "        nav[t] = cash + v50 + tot + sum(x[1] for x in pend)\n")
    rep("            st['buy_val'] += sh * o; st['fee'] += f;", "            _mfx(f - sh * px * fee); st['buy_val'] += sh * o; st['fee'] += f;")
    rep("st['sell_val'] += vopen * frac; st['fee'] += f;", "_mfx(f - gross * fee); st['sell_val'] += vopen * frac; st['fee'] += f;")
    rep("st['sell_val'] += vo * frac; st['fee'] += f;", "_mfx(f - gross * fee); st['sell_val'] += vo * frac; st['fee'] += f;")
    rep("            v50 -= x; cash += net; st['e50_cost'] += x - net;", "            _mfx(f - g * fee); v50 -= x; cash += net; st['e50_cost'] += x - net;")
    rep("            st['e50_cost'] += cash - val; st['e50_orders'] += 1", "            _mfx(f - cash * fee); st['e50_cost'] += cash - val; st['e50_orders'] += 1")
    ns = dict(s10lib.__dict__)
    exec(compile(src, 'r86_patched_simulate_s10', 'exec'), ns)
    return ns


NS = _patched_sim()


def sim(P, Sg, U, DV, cap=1e6, cost=None, disc=0.28, minfee=20.0, tau=0.0, prem=0.0, thr=20000.0, delay=0, rec=None, mf=None):
    """跑一次引擎副本。DV＝dividends() 的回傳。disc＝手續費折數（0.28＝引擎原設定）。"""
    NS['FEE'] = FEE_FULL * disc; NS['MINFEE'] = float(minfee)
    NS['FX'] = dict(y50=DV['y50'], YD=DV['YD'], tau=tau, prem=prem, thr=thr, delay=int(delay), rec=rec, mf=mf)
    return NS['simulate_r86'](P, Sg, U, sat_w=0.7, sizing='W10', lock=False, cost=cost or COSTS['strict'], capital=cap)


# ---------------------------------------------------------------- 資料
def load():
    P = s19lib.load()
    assert P.exright_fix and not P.reduce_fix and P.e50_source == cslib.E50_CSV
    s19lib.e50_fix_suspension(P)
    INNOV, _ = s4lib.innov_panel(P)
    U = stlib.universe(P, 'twse') & ~INNOV
    S, _ = s19lib.rev_std(P)
    months = list(S.index)
    avA = s19lib.a10_rows(P, months)
    for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'listed', 'NEARHI', 'V60', 'liq', 'has'):
        if hasattr(P, a):
            delattr(P, a)
    gc.collect()
    return P, U, S, avA


def raw_events():
    """TWT49U 封存全部列（含 0050 等非四碼代號）。"""
    fr = [pd.read_csv(f, dtype={'code': str}) for f in sorted(Path(cslib.TWT49U_DIR).glob('*/*.csv.gz'))]
    e = pd.concat(fr, ignore_index=True)
    e['date'] = pd.to_datetime(e['date'])
    return e.drop_duplicates(['date', 'code'], keep='last')


def per_yield(P, ev):
    """權息合併事件拆現金用：除權息前一個交易日的官方「殖利率」檔（data/universe/per）→ 現金股利估計＝殖利率 × 當日收盤 ÷ 100。回傳 Series（index 同 ev）。"""
    raw = str(cslib.RAW); env = dict(os.environ, GIT_NO_LAZY_FETCH='1')
    out = pd.Series(np.nan, index=ev.index)
    idx = pd.DatetimeIndex(P.IDX)
    for d, g in ev.groupby('date'):
        k = idx.searchsorted(d)
        if k == 0 or k >= len(idx) or idx[k] != d:
            continue
        pdv = idx[k - 1].strftime('%Y-%m-%d')
        r = subprocess.run(['git', '-C', raw, 'show', f'HEAD:data/universe/per/{pdv}.csv'], capture_output=True, env=env)
        if r.returncode != 0:
            continue
        q = pd.read_csv(io.BytesIO(r.stdout), dtype={'stock_id': str}).drop_duplicates('stock_id').set_index('stock_id')
        est = pd.to_numeric(q['yield_pct'], errors='coerce') * pd.to_numeric(q['close'], errors='coerce') / 100.0
        out.loc[g.index] = est.reindex(g['code']).to_numpy()
    return out


def dividends(P, mixed='est'):
    """現金股利「殖利率」表：YD＝{列號 t: {欄號 j: 息值 ÷ 除息前收盤}}（四碼普通股）、y50＝0050 每日息值 ÷ 前收（非除息日 0）。
    type＝息：息值＝value。type＝權：0。type＝權息（value＝權值＋息值，封存沒有分開）：
      mixed='est'（主口徑）：現金＝min(value, 前一交易日官方殖利率 × 收盤 ÷ 100)，查不到殖利率當 0；
      mixed='zero'：當 0（下限）；mixed='all'：整個 value 都當現金（上限）。"""
    e = raw_events()
    idx = pd.DatetimeIndex(P.IDX); di = {d: i for i, d in enumerate(idx)}
    e = e[e['date'].isin(di)].copy()
    e50 = e[e['code'] == '0050']
    assert (e50['type'] == '息').all()
    y50 = np.zeros(P.T)
    for d, v, pc in zip(e50['date'], e50['value'], e50['prev_close']):
        y50[di[d]] = v / pc
    s = e[e['code'].isin(P.ci)].copy()
    s['cash'] = np.where(s['type'] == '息', s['value'], 0.0)
    mx = s['type'] == '權息'
    info = dict(n_cash=int((s['type'] == '息').sum()), n_mixed=int(mx.sum()), n_stock_only=int((s['type'] == '權').sum()), n_e50=int(len(e50)), mixed=mixed)
    if mixed == 'all':
        s.loc[mx, 'cash'] = s.loc[mx, 'value']
    elif mixed == 'est':
        est = per_yield(P, s[mx])
        info['n_mixed_no_yield'] = int(est.isna().sum())
        s.loc[mx, 'cash'] = np.minimum(s.loc[mx, 'value'], est.fillna(0.0))
    YD = {}
    for d, c, cash, pc in zip(s['date'], s['code'], s['cash'], s['prev_close']):
        if cash > 0 and pc > 0:
            YD.setdefault(di[d], {})[P.ci[c]] = float(cash / pc)
    s['t'] = s['date'].map(di); s['j'] = s['code'].map(P.ci)
    return dict(y50=y50, YD=YD, info=info, events=s, e50=e50)


def bench(P, DV, cap=1e6, tau=0.0, prem=0.0, thr=20000.0, delay=0, rec=None):
    """只買 0050（期初全部買進、不計買進成本，淨值＝引擎用的 0050 含息序列）。除息日的現金股利＝前一日市值 × 息值 ÷ 前收。
    稅與補充保費當天從部位扣；delay>0：整筆股利先離開部位變應收（不生息、算在淨值裡），delay 個交易日後以當日收盤價位買回。"""
    E = P.E50; t0 = P.t0; T = P.T
    v = float(cap); pend = []; nav = np.full(T, np.nan); nav[t0] = v
    for t in range(t0 + 1, T):
        y = DV['y50'][t]
        if y > 0:
            D = v * y; pm = prem * D if (prem > 0 and D >= thr) else 0.0; tx = tau * D
            if rec is not None:
                rec.append((t, -1, D, nav[t - 1], tx, pm))
            out = D if delay > 0 else tx + pm
            v -= out
            if delay > 0:
                pend.append((t + delay, D - tx - pm))
        v *= E[t] / E[t - 1]
        for x in [x for x in pend if x[0] <= t]:
            v += x[1]; pend.remove(x)
        nav[t] = v + sum(x[1] for x in pend)
    return pd.Series(nav[t0:T], index=P.IDX[t0:T])
