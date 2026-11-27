#!/usr/bin/env python3
"""S103 疊加引擎：在 S75 v3 連續帳戶（引擎逐日 NAV＋stockfrac）上疊加台指期部位。
帳戶分四塊：股票 S（衛星選股，報酬 r_s）、0050 E、現金 C（年利率 rf）、期貨保證金帳戶 M；期貨名目 F（+多 −空）。
r_s 由引擎拆出：r_nav = w·r_s + (1−w)·r_0050（w＝前一日收盤 stockfrac；w<5% 時 r_s≡r_0050）。設計 'orig' 必須重建引擎 NAV。
每日流程（t−1 收盤 → t 收盤）：
  1. S×(1+r_s)、E×(1+r50)、C×(1+rf/245)、M ＋= F·r_fut（逐日結算）、F×(1+r_fut)。
  2. 追繳：F≠0 且 M < 維持率×|F| → 計 1 次，補到 k×原始保證金率×|F|；先用 C（即時），不足再賣 0050（計「需賣 0050」），再不足賣股票（計「需賣股」）。
     台股 T+2 交割，賣 0050／賣股補繳實務上趕不上期交所補繳期限，這兩類視為「斷頭風險事件」。
  3. 收盤調整：S＝w_t·A（照引擎的持股比例）；其餘 N＝A−S。
     不留 0050 的設計：E＝0，C＝N−M；C<0 時 M 讓出（M＝N）；若 M < 原始保證金×|F| → F 被保證金上限截斷（計「保證金不足日」）。
     保留 0050 的設計：C＋M＝期貨所需資金（換月日／追繳／口數變動時調到 k×原始×|F|，0050 進出每邊 0.1%），E＝N−C−M。
  4. 期貨交易：目標 F*（設計規則，用 t 收盤以前資料；期貨 13:45 收盤晚於現貨 13:30，可在 t 日期貨收盤成交）。
     換月日：舊口平倉＋新口開倉＝2 邊×|F|；其他日子 |F*−F| > band×|F*|、方向改變或歸零才交易。
     每邊成本＝名目×(期交稅 0.002% ＋ 手續費 fee÷(乘數×指數) ＋ 滑價 slip 點÷指數)；換月日同時把 M 調回目標。
連續名目（不取整）；口數取整另由 min_capital() 與 rounding 情境估計。"""
import numpy as np, pandas as pd

TAX = 2e-5
ETF_SIDE = 0.001          # 0050 進出（保證金撥補）每邊成本


def rolling_beta(y, x, n):
    y = pd.Series(y); x = pd.Series(x)
    b = y.rolling(n, min_periods=max(20, n // 3)).cov(x) / x.rolling(n, min_periods=max(20, n // 3)).var()
    return b.clip(0.0, 2.0).fillna(1.0).to_numpy()


def stock_returns(nav, sf, r50):
    """回傳 (r_s, r_nav, r_other)。w≥5%：r_s＝(r_nav−(1−w)·r50)/w、r_other＝r50；w<5%（多半是建倉初期現金尚未掃入 0050）：r_s＝0、r_other＝r_nav/(1−w)。"""
    rn = np.r_[0.0, nav[1:] / nav[:-1] - 1]
    w = np.r_[0.0, sf[:-1]]
    ok = w >= 0.05
    rs = np.where(ok, (rn - (1 - w) * r50) / np.where(ok, w, 1.0), 0.0)
    ro = np.where(ok, r50, rn / (1 - w))
    rs[0] = 0.0; ro[0] = 0.0
    return rs, rn, ro


def run(cell, nav, sf, r50, fut, cfg, capital=None, cmult=None, t_start=0):
    """cell：dict(design, sat, keep50, rule, ...)；nav/sf：引擎；r50：0050 日報酬；fut：futlib.build 的 DataFrame（同日期）。
    cfg：im（原始保證金率）、mm（維持÷原始）、k（撥補倍數）、rf、fee、mult、slip、band。
    capital＋cmult：口數取整（cmult＝每點價值，TX 200、小台 50、微台 10）。t_start：從第幾列開始（之前輸出 NaN；β 仍用完整歷史）。"""
    T = len(nav)
    sf = np.nan_to_num(np.asarray(sf, float), nan=0.0)
    rs, rn, ro = stock_returns(nav, sf, r50)
    rf_ = fut['r_fut'].to_numpy(); rtw = fut['r_twii'].to_numpy(); idx = fut['twii'].to_numpy(); roll = fut['roll'].to_numpy().astype(bool)
    im, mmr, k, band = cfg['im'], cfg['mm'] * cfg['im'], cfg['k'], cfg['band']
    rfd = cfg['rf'] / 245
    cs = TAX + cfg['fee'] / (cfg['mult'] * idx) + cfg['slip'] / idx            # 每邊成本（占名目）
    keep50 = cell['keep50']; rule = cell['rule']
    # 目標曝險函數所需序列（都只用到 t 以前資料）
    if rule in ('hedge_s', 'cond'):
        pass
    beta_s = rolling_beta(rs, rtw, cell.get('bwin', 60)) if rule == 'hedge_s' else None
    beta_n = rolling_beta(rn, rtw, cell.get('bwin', 60)) if rule in ('hedge_nav', 'cond') else None
    sig = None
    if rule == 'cond':
        tw = pd.Series(idx)
        s_ma = (tw < tw.rolling(200, min_periods=200).mean()).to_numpy()
        s_r20 = (tw / tw.shift(20) - 1 < -0.08).to_numpy()
        sig = {'ma200': s_ma, 'r20': s_r20, 'or': s_ma | s_r20}[cell['signal']]
    A = 1e6 if capital is None else float(capital)
    w0 = sf[t_start]
    S = w0 * A; F = 0.0; M = 0.0
    C = 0.0 if keep50 else A - S
    E = A - S - C
    out = np.full(T, np.nan); out[t_start] = A
    fexp = np.zeros(T)
    st = dict(calls=0, calls_need_0050=0, calls_need_stock=0, margin_capped_days=0, fut_cost=0.0, etf_cost=0.0, n_trades=0, n_rolls=0, max_loss_day_on_M=0.0)

    def target(t, A, S):
        if rule == 'none':
            return 0.0
        if rule == 'hedge_s':
            return -beta_s[t] * S
        if rule == 'hedge_nav':
            return -beta_n[t] * A
        if rule == 'lever':
            return cell['extra'] * A
        if rule == 'cond':
            return -cell['h'] * beta_n[t] * A if sig[t] else 0.0
        if rule == 'base_fut':                    # 0050 底倉＋空位改期貨多單；extra＝額外名目（占淨值）
            return (A - S) + cell['extra'] * A
        raise ValueError(rule)

    for t in range(t_start + 1, T):
        # 1. 損益
        S *= 1 + rs[t]; E *= 1 + ro[t]; C *= 1 + rfd
        pnl = F * rf_[t]; M += pnl; F *= 1 + rf_[t]
        if F != 0 and pnl < 0:
            st['max_loss_day_on_M'] = min(st['max_loss_day_on_M'], pnl / max(M - pnl, 1e-9))
        A = S + E + C + M
        if A <= 0:
            out[t:] = 0.0; st['ruin'] = str(fut.index[t].date()); break
        # 2. 追繳
        if F != 0 and M < mmr * abs(F):
            st['calls'] += 1
            need = k * im * abs(F) - M
            x = min(need, max(C, 0.0)); C -= x; M += x; need -= x
            if need > 1e-9:
                st['calls_need_0050'] += 1 if E > 1e-9 else 0
                x = min(need, max(E, 0.0)); E -= x; M += x * (1 - ETF_SIDE); st['etf_cost'] += x * ETF_SIDE; need -= x
            if need > 1e-9:
                st['calls_need_stock'] += 1
                S -= need; M += need * (1 - 0.004); need = 0.0
            A = S + E + C + M
        # 3. 收盤調整（股票比例照引擎）
        w = sf[t]; S = w * A; N = A - S
        Ft = target(t, A, S)
        if capital is not None:
            cv = cmult * idx[t]; Ft = np.round(Ft / cv) * cv
        trade = roll[t] or (Ft == 0 and F != 0) or (np.sign(Ft) != np.sign(F)) or (abs(Ft - F) > band * abs(Ft))
        if capital is not None and not roll[t]:
            trade = trade and abs(Ft - F) > 0.5 * cmult * idx[t]
        Fn = Ft if trade else F
        if keep50:
            D = C + M
            if trade and (roll[t] or Fn != F):
                Mt = k * im * abs(Fn)
                dD = Mt - D                               # 由 0050 撥入（>0）或撥回（<0）
                st['etf_cost'] += abs(dD) * ETF_SIDE
                C = 0.0; M = Mt; E = N - M - abs(dD) * ETF_SIDE
            else:
                E = N - C - M
            if E < 0:                                     # 股票吃掉 0050：先用 C，再減 M
                x = min(-E, C); C -= x; E += x
                if E < 0:
                    M += E; E = 0.0
        else:
            E = 0.0
            if trade and (roll[t] or Fn != F):
                M = min(k * im * abs(Fn), max(N, 0.0))
            C = N - M
            if C < 0:
                M = max(N, 0.0); C = N - M
        if Fn != 0 and M < im * abs(Fn) - 1e-6 and not keep50:
            capF = max(M, 0.0) / im
            if capital is not None:
                cv = cmult * idx[t]; capF = np.floor(capF / cv) * cv
            Fn = np.sign(Fn) * min(abs(Fn), capF)
            st['margin_capped_days'] += 1
        # 4. 期貨成本
        if roll[t] and F != 0:
            vol = abs(F) + abs(Fn) if np.sign(F) == np.sign(Fn) else abs(F) + abs(Fn)
            st['n_rolls'] += 1
        else:
            vol = abs(Fn - F)
        if vol > 0:
            c = vol * cs[t]; st['fut_cost'] += c; M -= c; st['n_trades'] += 1
        F = Fn
        out[t] = S + E + C + M
        fexp[t] = F / out[t] if out[t] > 0 else 0.0
    fx = fexp[t_start + 1:]
    st['avg_abs_fexp'] = float(np.mean(np.abs(fx)))
    st['max_abs_fexp'] = float(np.max(np.abs(fx)))
    st['frac_days_fut_on'] = float(np.mean(fx != 0))
    on = np.abs(fx[fx != 0])
    st['fexp_when_on_median'] = float(np.median(on)) if len(on) else 0.0
    st['fexp_when_on_p10'] = float(np.percentile(on, 10)) if len(on) else 0.0
    return out, fexp, st
