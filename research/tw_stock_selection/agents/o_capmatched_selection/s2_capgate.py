#!/usr/bin/env python3
"""步驟 2：市值資料閘門（只看資料，不看任何策略報酬）。
  A) 發行股數回推 → data/shares_est.parquet（面板日 × 代碼）
     錨點：MI_QFIIS 日資料（raw/agents/vf_twse_foreign.parquet，2025-10-07～2026-10-21，實值）；沒有的代碼用 screener shares_hist（2025-09 起）
     往回走事件（股數_前＝股數_後 ÷ g）：
       R1 2024-08-19 起：exdiv.csv 的 stock_ratio＋rights_ratio（實值）；corp_actions.csv 減資
          （彌補虧損 g＝前收/參考價；退還股款／現金減資 k＝(參考價−前收)/(參考價−10)、g＝1−k）
       R2 2021-10-07～2024-08-18：exdiv_5y（權：g＝前收/參考價；權息：現金股利用 pe_5y 殖利率×前收估計，g＝(前收−D)/參考價；息：g＝1）
       R3 2021-10-07 以前：面板 flag 1（除權息日）：f＝前收/收×(1＋當日非除權息股報酬中位數)；f < 0.8 → 減資型 g＝f；
          否則 s＝f×(1−y0)−1（y0＝該股 2021-09 殖利率），s > 2% 才算配股，g＝1＋s（上限 2）
       E5 任何期間：停止交易 ≥ 5 個面板日後恢復、價格比 > 1.25、當日沒有其他事件 → 減資型 g＝前收/恢復收盤
  B) 臺灣50 缺股數的成分（已下市：2888、2823、2311、2325、3474 等）：在其成分期間以「複製 0050 日報酬誤差最小」的固定股數校準
     （只用 0050 基準，不用策略報酬），只供骨架；不進市值分層。
  C) 閘門：
     (i)  重建臺灣50（審核基準日市值排名：非成分 ≤ 40 納入、成分 > 60 剔除、補足 50）與公開名單的一致率，2019 年後每次審核 ≥ 95%
     (ii) 2026-10-21 → 2025-10-07 一年回推（只用 R1 事件）對 MI_QFIIS 實值：前 50 名市值排名 Spearman ≥ 0.98；
          2021-10 錨點沒有實值（GitHub 找不到封存），改報 R3 啟發式在 2021-10～2024-08（有 exdiv_5y 實際種類）的分類準確率
     (iii) 臺灣50 實際成分 × 回推市值加權（每日依收盤市值）對 0050 含息：2019-01～2022-12 日曆月追蹤誤差 ≤ 3%／年
寫 data/shares_est.parquet、data/tw50_fill.json、data/membership.json、results_gate.json。"""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import cmlib as L

IDX, COLS, N, CI = L.IDX, L.COLS, L.N, L.CI
OUT = {}

# ---------------- 錨點 ----------------
vf = pd.read_parquet(L.RAW / 'agents' / 'vf_twse_foreign.parquet')
vf['date'] = pd.to_datetime(vf['date']); vf = vf[vf.code.isin(CI) & (vf.total_shares > 0)]
VFW = vf.pivot_table(index='date', columns='code', values='total_shares', aggfunc='last').sort_index()
sh = pd.read_csv(L.RAW / 'screener' / 'shares_hist.csv', dtype={'code': str})
sh = sh[sh.code.isin(CI) & (sh.shares > 0)]
SH0 = sh.sort_values('ym').drop_duplicates('code').set_index('code')['shares']   # 最早月（2025-09，backfill）

# ---------------- 事件 ----------------
pe = pd.read_csv(L.RAW / 'screener' / 'pe_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
pe = pe[pe.code.isin(CI)]
PEY = pe.pivot_table(index='date', columns='code', values='yield', aggfunc='last').sort_index()
Y0 = PEY.iloc[0].reindex(COLS).fillna(4.0).clip(0, 15).to_numpy() / 100.0      # 2021-09-01 殖利率（R3 用）

ev = {}   # (code, date) -> (g, src)
def add(c, d, g, src):
    if c not in CI or not np.isfinite(g) or g <= 0: return
    d = pd.Timestamp(d)
    k = (c, d)
    if k in ev:   # 同日多事件相乘
        g0, s0 = ev[k]; ev[k] = (g0 * g, s0 + '+' + src)
    else: ev[k] = (g, src)

ex = pd.read_csv(L.RAW / 'screener' / 'exdiv.csv', dtype={'code': str})
ex = ex[(ex.market == 'TWSE') & ex.code.isin(CI)]
for r in ex.itertuples():
    g = 1 + float(r.stock_ratio or 0) + float(r.rights_ratio or 0)
    if g > 1.0001: add(r.code, r.date, g, 'R1div')
ca = pd.read_csv(L.RAW / 'screener' / 'corp_actions.csv', dtype={'code': str})
ca = ca[(ca.market == 'TWSE') & ca.code.isin(CI)]
for r in ca.itertuples():
    if '彌補虧損' in r.kind: g = r.prev_close / r.ref_price
    elif '退還股款' in r.kind or '現金減資' in r.kind:
        k = (r.ref_price - r.prev_close) / (r.ref_price - 10.0) if r.ref_price > 10 else np.nan
        g = 1 - k if np.isfinite(k) and 0 < k < 1 else r.prev_close / r.ref_price
    else: continue
    add(r.code, r.date, g, 'R1cap')
R1_START = pd.Timestamp('2024-08-19')
e5 = pd.read_csv(L.RAW / 'screener' / 'exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
e5 = e5[(e5.market == 'TWSE') & e5.code.isin(CI)]
EXDIV5_START = e5.date.min()
def pey_at(c, d):
    if c not in PEY.columns: return np.nan
    s = PEY[c]; s = s[s.index <= d].dropna()
    return float(s.iloc[-1]) / 100 if len(s) else np.nan
n_r2 = {'權': 0, '權息': 0}
for r in e5[e5.date < R1_START].itertuples():
    if r.kind == '權':
        g = r.prev_close / r.ref_price; n_r2['權'] += 1
    elif r.kind == '權息':
        y = pey_at(r.code, r.date); D = (y if np.isfinite(y) else 0.04) * r.prev_close
        g = (r.prev_close - D) / r.ref_price; n_r2['權息'] += 1
    else: continue
    g = min(max(g, 1.0), 2.0)
    if g > 1.005: add(r.code, r.date, g, 'R2')

# R3：面板 flag 1（2021-10-07 以前）
mkt = np.array([np.nanmedian(np.where((np.nan_to_num(L.Fn[t], nan=0) == 0) & L.TRADED[t], L.Rn[t], np.nan)) if L.TRADED[t].sum() > 50 else 0.0 for t in range(len(IDX))])
# 配股門檻 R3_TH：用 2021-10～2024-08 有實際種類（息／權）的 2,290 筆事件比較 None／2%／5%／8%／12%／20%，
# 平均 |log 誤差| 最小的是「不偵測配股」（0.0028；2% 為 0.0131、12% 為 0.0039），故 R3 只保留減資型（f < 0.8）。
R3_TH = None
R3_CALIB = {'None': 0.0028, '0.02': 0.0131, '0.05': 0.0086, '0.08': 0.0058, '0.12': 0.0039, '0.2': 0.0031, 'n_events': 2290, 'criterion': 'mean |log(g_hat/g_true)|'}
def r3_g(t, j):
    with np.errstate(invalid='ignore', divide='ignore'):
        f = L.PREVC[t, j] / L.Cn[t, j] * (1 + mkt[t])
    if not np.isfinite(f): return np.nan, np.nan
    if f < 0.8: return f, f
    if R3_TH is None: return 1.0, f
    s = f * (1 - Y0[j]) - 1
    return (1 + min(s, 1.0)) if s > R3_TH else 1.0, f
r3_cnt = {'reduction': 0, 'stockdiv': 0, 'none': 0}
for t in range(1, len(IDX)):
    if IDX[t] >= EXDIV5_START: break
    js = np.where((L.Fn[t] == 1) & L.TRADED[t])[0]
    for j in js:
        g, f = r3_g(t, j)
        if not np.isfinite(g): continue
        if g < 1: r3_cnt['reduction'] += 1
        elif g > 1: r3_cnt['stockdiv'] += 1
        else: r3_cnt['none'] += 1
        if g != 1.0: add(COLS[j], IDX[t], g, 'R3')

# E5：停牌後恢復的跳升（減資型）
n_e5 = 0
TRc = L.TRADED
for j in range(N):
    tr = np.where(TRc[:, j])[0]
    if len(tr) < 2: continue
    gaps = np.diff(tr)
    for k in np.where(gaps > 5)[0]:
        a, b = tr[k], tr[k + 1]
        ratio = L.Cn[b, j] / L.Cn[a, j]
        if ratio > 1.25 and (COLS[j], IDX[b]) not in ev and L.Fn[b, j] != 1:
            add(COLS[j], IDX[b], 1 / ratio, 'E5'); n_e5 += 1

OUT['R3_threshold_calibration'] = R3_CALIB
OUT['events'] = {'n_total': len(ev), 'n_R1': sum('R1' in v[1] for v in ev.values()), 'n_R2': sum('R2' in v[1] for v in ev.values()),
                 'n_R3': sum('R3' in v[1] for v in ev.values()), 'n_E5': n_e5, 'R2_input_counts': n_r2, 'R3_flag1_classification': r3_cnt,
                 'exdiv5_start': str(EXDIV5_START.date())}

# ---------------- 回推 ----------------
EV_BY = {}
for (c, d), (g, s) in ev.items(): EV_BY.setdefault(c, []).append((d, g, s))
def backcast(c, anchor_date, anchor_val, use=lambda s: True):
    """回傳 Series（IDX 上，anchor_date 以前的股數）。事件日 d 的 g：d 當天起是新股數 → d 以前（< d）除以 g。"""
    s = pd.Series(np.nan, index=IDX)
    evs = sorted([e for e in EV_BY.get(c, []) if e[0] <= anchor_date and use(e[2])], key=lambda z: z[0], reverse=True)
    cur = anchor_val; prev_cut = anchor_date
    mask_all = IDX <= anchor_date
    # 從 anchor 往回：區間 [d, prev_cut] 用 cur；d 以前除以 g
    for d, g, src in evs:
        m = (IDX >= d) & (IDX <= prev_cut) & mask_all
        s[m & s.isna()] = cur
        cur = cur / g; prev_cut = d - pd.Timedelta(days=1)
    m = (IDX <= prev_cut)
    s[m & s.isna()] = cur
    return s

S = pd.DataFrame(np.nan, index=IDX, columns=COLS)
anchor_src = {}
vfi = VFW.reindex(IDX.union(VFW.index)).ffill().reindex(IDX)
for c in COLS:
    if c in VFW.columns and VFW[c].notna().any():
        first = VFW[c].first_valid_index(); val = float(VFW[c][first])
        a = IDX[IDX >= first][0] if (IDX >= first).any() else IDX[-1]
        col = backcast(c, a, val)
        live = vfi[c].where(IDX >= first)
        S[c] = live.where(live.notna(), col)
        anchor_src[c] = 'MI_QFIIS'
    elif c in SH0.index:
        a = pd.Timestamp('2025-09-30'); a = IDX[IDX <= a][-1]
        S[c] = backcast(c, a, float(SH0[c])).where(IDX <= a, float(SH0[c]))
        anchor_src[c] = 'shares_hist'
# 下市股：沒有錨點 → NaN（不進市值分層）
OUT['anchor_counts'] = pd.Series(anchor_src).value_counts().to_dict()
OUT['n_codes_no_anchor'] = int(N - len(anchor_src))

# ---------------- B) 臺灣50 缺股數成分的校準 ----------------
r50 = L.E50.pct_change()
all_mem = set(L._TW['current_2026-10-09'])
for _, mb in L._TW_STEPS: all_mem |= set(mb)
missing = sorted([c for c in all_mem if c in CI and not np.isfinite(S[c].to_numpy()).any()])
Sn = S.to_numpy()
def member_windows(c):
    days = [d for d in IDX if d > L.TW50_EARLIEST]
    inm = np.array([c in (L.tw50_at(d) or []) for d in days])
    return pd.DatetimeIndex(days)[inm]
_MEMCACHE = {}
def mem_at(d):
    if d not in _MEMCACHE: _MEMCACHE[d] = L.tw50_at(d)
    return _MEMCACHE[d]
fill = {}
for c in missing:
    win = pd.DatetimeIndex([d for d in IDX if d > L.TW50_EARLIEST and c in (mem_at(d) or [])])
    if len(win) < 20: continue
    j = CI[c]
    ti = np.array([IDX.get_loc(d) for d in win])
    # 其他成分的日報酬（以前一日市值加權）
    rows = []
    for t in ti:
        mem = [CI[x] for x in mem_at(IDX[t]) if x in CI and x != c]
        mc = Sn[t - 1, mem] * L.CFF[t - 1, mem]; ok = np.isfinite(mc)
        mem = np.array(mem)[ok]; mc = mc[ok]
        rr = L.GROW[t, mem] - 1
        rows.append((mc.sum(), float((mc * rr).sum() / mc.sum()), L.GROW[t, j] - 1, L.CFF[t - 1, j]))
    A = np.array(rows); y = r50.reindex(IDX[ti]).to_numpy()
    best = None
    for sh_ in np.exp(np.linspace(np.log(1e8), np.log(5e10), 400)):
        mcj = sh_ * A[:, 3]; w = mcj / (mcj + A[:, 0])
        rp = (1 - w) * A[:, 1] + w * A[:, 2]
        e = np.nanstd(rp - y)
        if best is None or e < best[0]: best = (e, sh_, float(np.nanmean(w)))
    lo_b, hi_b = 1e8 * 1.02, 5e10 / 1.02
    if best[1] <= lo_b or best[1] >= hi_b:
        # 校準落在格點邊界（資料無法辨識）→ 改用「該股市值＝同日成分中最小市值」的固定股數（取窗起點）
        t = ti[0]; mem = [CI[x] for x in mem_at(IDX[t]) if x in CI and x != c]
        mcs = Sn[t - 1, mem] * L.CFF[t - 1, mem]; mmin = np.nanmin(mcs)
        sh_fb = mmin / L.CFF[t - 1, j]
        fill[c] = {'shares': float(sh_fb), 'mode': 'fallback_min_member_cap', 'calib_boundary': best[1], 'days': int(len(ti)), 'window': [str(win[0].date()), str(win[-1].date())]}
        S.loc[win, c] = sh_fb
    else:
        fill[c] = {'shares': best[1], 'mode': 'calibrated', 'mean_weight': best[2], 'days': int(len(ti)), 'window': [str(win[0].date()), str(win[-1].date())]}
        S.loc[win, c] = best[1]
OUT['tw50_missing_filled'] = fill
S.astype('float64').to_parquet(L.DATA / 'shares_est.parquet')
(L.DATA / 'tw50_fill.json').write_text(L.dumps(fill), encoding='utf-8')
Sn = S.to_numpy()
MC = Sn * L.CFF

# ---------------- (ii) 一年回推檢查 ----------------
a_new = VFW.index.max(); a_old = VFW.index.min()
real_old = VFW.loc[a_old]; real_new = VFW.loc[a_new]
t_old = IDX.get_loc(IDX[IDX <= a_old][-1])
bc = {}
for c in real_new.dropna().index:
    if c not in CI: continue
    evs = [e for e in EV_BY.get(c, []) if a_old < e[0] <= a_new and e[2].startswith('R1')]
    v = float(real_new[c])
    for d, g, s in evs: v = v / g
    bc[c] = v
bc = pd.Series(bc)
both = pd.concat([bc.rename('bc'), real_old.rename('real')], axis=1).dropna()
px = pd.Series(L.CFF[t_old], index=COLS).reindex(both.index)
both['mc_bc'] = both.bc * px; both['mc_real'] = both.real * px
both = both.dropna()
top50 = both.sort_values('mc_real', ascending=False).head(50)
top300 = both.sort_values('mc_real', ascending=False).head(300)
from scipy.stats import spearmanr
def sp(df): return float(spearmanr(df.mc_bc, df.mc_real).correlation)
naive = both.copy(); naive['mc_bc'] = real_new.reindex(naive.index) * px    # 不回推（直接用一年後股數）
OUT['gate_ii'] = {'anchor_old': str(a_old.date()), 'anchor_new': str(a_new.date()),
                  'spearman_top50': sp(top50), 'spearman_top300': sp(top300),
                  'naive_constant_spearman_top50': float(spearmanr(naive.loc[top50.index].mc_bc, top50.mc_real).correlation),
                  'median_abs_share_err_top300_pct': float((top300.bc / top300.real - 1).abs().median() * 100),
                  'p90_abs_share_err_top300_pct': float((top300.bc / top300.real - 1).abs().quantile(0.9) * 100),
                  'naive_median_abs_err_top300_pct': float((real_new.reindex(top300.index) / top300.real - 1).abs().median() * 100),
                  'n_changed_top300': int(((real_new.reindex(top300.index) / top300.real - 1).abs() > 0.001).sum()),
                  'note_2021_10': '2021-10 沒有發行股數實值（GitHub 封存找不到；o_sbl 的 shares_anchor 是 2025-10-07 快照），無法做 2021-10 錨點的 Spearman；改報 R3 啟發式準確率與閘門 (i)(iii)'}
OUT['gate_ii']['pass'] = OUT['gate_ii']['spearman_top50'] >= 0.98

# R3 啟發式準確率：2021-10～2024-08 用 exdiv_5y 種類當真值
acc = {'n': 0, 'true_stock_detected': 0, 'true_stock_n': 0, 'cash_false_pos': 0, 'cash_n': 0, 'abs_err_g_stock': []}
e5v = e5[(e5.date < R1_START)]
for r in e5v.itertuples():
    if r.date not in IDX: continue
    t = IDX.get_loc(r.date); j = CI[r.code]
    if not L.TRADED[t, j]: continue
    g_hat, f = r3_g(t, j)
    if not np.isfinite(g_hat): continue
    acc['n'] += 1
    if r.kind == '息':
        acc['cash_n'] += 1; acc['cash_false_pos'] += int(g_hat != 1.0)
    elif r.kind == '權':
        acc['true_stock_n'] += 1; acc['true_stock_detected'] += int(g_hat > 1.0)
        acc['abs_err_g_stock'].append(abs(g_hat - r.prev_close / r.ref_price))
acc['abs_err_g_stock_median'] = float(np.median(acc['abs_err_g_stock'])) if acc['abs_err_g_stock'] else None
acc.pop('abs_err_g_stock')
acc['cash_false_pos_rate'] = acc['cash_false_pos'] / max(acc['cash_n'], 1); acc['stock_recall'] = acc['true_stock_detected'] / max(acc['true_stock_n'], 1)
OUT['R3_heuristic_accuracy_2021_10_to_2024_08'] = acc

# ---------------- (i) 臺灣50 重建 ----------------
def third_friday(y, m):
    d = pd.Timestamp(y, m, 15); return d + pd.Timedelta(days=(4 - d.weekday()) % 7)
evc = pd.read_csv(L.RAW / 'agents' / 'etf_rebalance_events.csv', dtype=str)
evc = evc[evc.etf == 'TW50']
CUT = {r.review: (pd.Timestamp(r.eff), pd.Timestamp(r.cutoff)) for r in evc.itertuples()}
ELIG = ~L.DRMASK
def rank_rule(prev, t):
    mc = np.where(ELIG & np.isfinite(MC[t]) & L.TRADED[max(t - 5, 0):t + 1].any(axis=0), MC[t], -np.inf)
    order = np.argsort(-mc); rank = np.empty(N, int); rank[order] = np.arange(1, N + 1)
    keep = [c for c in prev if c in CI and rank[CI[c]] <= 60]
    add_ = [COLS[j] for j in order[:40] if COLS[j] not in prev]
    S_ = list(dict.fromkeys(keep + add_))
    if len(S_) > 50:
        S_ = sorted(S_, key=lambda c: rank[CI[c]])[:50]
    k = 0
    while len(S_) < 50:
        c = COLS[order[k]]; k += 1
        if c not in S_: S_.append(c)
    return S_, rank
rev_rows = []
for y in range(2016, 2027):
    for m in (3, 6, 9, 12):
        key = f'{y}-{m:02d}'
        if key in CUT: eff, cut = CUT[key]
        else:
            eff = third_friday(y, m); cut = eff - pd.Timedelta(days=25)
        if eff > IDX[-1] or eff <= L.TW50_EARLIEST + pd.Timedelta(days=60): continue
        before = mem_at(IDX[IDX <= eff][-1]); after_day = IDX[IDX > eff]
        if len(after_day) == 0: continue
        after = mem_at(after_day[0])
        tc = IDX.get_loc(IDX[IDX <= cut][-1])
        pred, rank = rank_rule(before, tc)
        agree = len(set(pred) & set(after)) / 50
        true_ch = sorted(set(after) - set(before)); pred_ch = sorted(set(pred) - set(before))
        rev_rows.append({'review': key, 'agree': agree, 'true_adds': true_ch, 'pred_adds': pred_ch,
                         'true_add_ranks': [int(rank[CI[c]]) if c in CI else None for c in true_ch]})
rv = pd.DataFrame(rev_rows)
post = rv[rv.review >= '2019-03']
OUT['gate_i'] = {'reviews': rev_rows, 'post2019_min_agree': float(post.agree.min()), 'post2019_mean_agree': float(post.agree.mean()),
                 'n_post2019': int(len(post)), 'n_below_95': int((post.agree < 0.95).sum()), 'pass': bool((post.agree >= 0.95).all())}

# 2015-09 以前的成分：規則重建（P4 用），從 2005-06 起每季審核，種子＝2005-06 基準日市值前 50
memb = {}
t0 = IDX.get_loc(IDX[IDX <= pd.Timestamp('2005-05-20')][-1])
mc0 = np.where(ELIG & np.isfinite(MC[t0]), MC[t0], -np.inf); cur_m = [COLS[j] for j in np.argsort(-mc0)[:50]]
for y in range(2005, 2016):
    for m in (3, 6, 9, 12):
        eff = third_friday(y, m)
        if eff < pd.Timestamp('2005-06-01') or eff > L.TW50_EARLIEST: continue
        tc = IDX.get_loc(IDX[IDX <= eff - pd.Timedelta(days=25)][-1])
        cur_m, _ = rank_rule(cur_m, tc)
        memb[str(eff.date())] = cur_m
act = L.tw50_at(IDX[IDX > L.TW50_EARLIEST][0])
OUT['pre2015_rule_membership'] = {'n_reviews': len(memb), 'agree_with_actual_2015_09': len(set(cur_m) & set(act)) / 50}
(L.DATA / 'membership_pre2015.json').write_text(L.dumps(memb), encoding='utf-8')

# ---------------- 自由流通（投資權重）係數 ----------------
# FTSE 臺灣50 以投資權重（自由流通）調整市值；用 etf0050_w.csv 2026-10-09 實際權重 ÷ 回推市值權重，得每股固定係數 φ（台積電＝1）；
# 不在快照的股票用快照中位數。假設各股自由流通結構不隨時間改變（限制）。
w_snap = pd.read_csv(L.RAW / 'screener' / 'etf0050_w.csv', dtype={'code': str})
w_snap = w_snap[w_snap['asof'] == '2026-10-09'].set_index('code')['weight'] / 100
t_sn = IDX.get_loc(IDX[IDX <= pd.Timestamp('2026-10-08')][-1])
mc_sn = pd.Series(MC[t_sn], index=COLS)[w_snap.index]
phi = (w_snap / w_snap.sum()) / (mc_sn / mc_sn.sum()); phi = phi / phi['2330']
PHI = np.full(N, float(phi.drop('2330').median()))
for c, v in phi.items(): PHI[CI[c]] = v
pd.Series(PHI, index=COLS).to_frame('phi').to_parquet(L.DATA / 'float_phi.parquet')
OUT['float_phi'] = {'median_non_tsmc': float(phi.drop('2330').median()), 'min': [phi.idxmin(), float(phi.min())], 'n': int(len(phi))}

# ---------------- (iii) 0050 複製追蹤誤差 ----------------
GX = L.grow_exfix()
def replica_daily(a, b, G, fa=False):
    ts = [t for t in range(len(IDX)) if pd.Timestamp(a) <= IDX[t] <= pd.Timestamp(b)]
    out = []
    for t in ts:
        mem = mem_at(IDX[t])
        if mem is None:
            k = max(k for k in memb if pd.Timestamp(k) < IDX[t]) if any(pd.Timestamp(k) < IDX[t] for k in memb) else None
            mem = memb[k] if k else []
        js = np.array([CI[c] for c in mem if c in CI])
        mc = MC[t - 1, js] * (PHI[js] if fa else 1.0); ok = np.isfinite(mc) & (mc > 0)
        w = mc[ok] / mc[ok].sum()
        out.append((IDX[t], float((w * (G[t, js[ok]] - 1)).sum()), int(ok.sum()), float(w[js[ok] == L.J2330].sum())))
    return pd.DataFrame(out, columns=['d', 'r', 'n', 'w2330']).set_index('d')
gi = {}
for lab, G, fa in [('panel', L.GROW, False), ('exfix', GX, False), ('panel_FA', L.GROW, True), ('exfix_FA', GX, True)]:
    rep = replica_daily('2006-01-01', '2026-10-12', G, fa)
    rep.to_parquet(L.DATA / f'replica_daily_{lab}.parquet')
    x = pd.concat([rep.r, r50.reindex(rep.index)], axis=1).dropna(); x.columns = ['p', 'e']
    mm = (1 + x).groupby(x.index.to_period('M')).prod() - 1; dd = mm.p - mm.e
    for a, b in [('2006-01', '2015-09'), ('2015-10', '2018-12'), ('2019-01', '2022-12'), ('2023-01', '2026-10')]:
        z = dd[a:b]
        gi[f'{lab}|{a}~{b}'] = {'TE_month_ann_pct': float(z.std() * math.sqrt(12) * 100), 'mean_diff_pp_m': float(z.mean() * 100),
                                'TE_daily_ann_pct': float((x.p - x.e)[a:b].std() * math.sqrt(252) * 100), 'n_members_min': int(rep.n[a:b].min()) if len(rep.n[a:b]) else None}
    gi[f'{lab}|w2330_by_year'] = {int(y): round(float(v), 3) for y, v in rep.w2330.groupby(rep.index.year).mean().items()}
OUT['gate_iii'] = gi
# 事前規則（看結果前寫下）：骨架＝FA 與非 FA 中，2019-01～2022-12 exfix 月追蹤誤差較低者；閘門以該版判定，panel 口徑並列
use_fa = gi['exfix_FA|2019-01~2022-12']['TE_month_ann_pct'] < gi['exfix|2019-01~2022-12']['TE_month_ann_pct']
OUT['skeleton_float_adjusted'] = bool(use_fa)
lab = 'exfix_FA' if use_fa else 'exfix'
OUT['gate_iii']['pass'] = gi[f'{lab}|2019-01~2022-12']['TE_month_ann_pct'] <= 3.0
OUT['gate_iii']['pass_panel_convention'] = gi[f"{lab.replace('exfix', 'panel')}|2019-01~2022-12"]['TE_month_ann_pct'] <= 3.0
(L.DATA / 'skeleton.json').write_text(json.dumps({'float_adjusted': bool(use_fa)}), encoding='utf-8')
# 實際 0050 台積電權重（ex_tsmc.csv，2025-09 起）對照
xt = pd.read_csv(L.RAW / 'screener' / 'ex_tsmc.csv', parse_dates=['date']).set_index('date')
rep = pd.read_parquet(L.DATA / f'replica_daily_{lab}.parquet')
jj = rep.w2330.reindex(xt.index).dropna()
OUT['w2330_replica_vs_actual_2025_09_on'] = {'replica_mean': float(jj.mean()), 'actual_mean': float(xt.w2330.reindex(jj.index).mean() / 100), 'n': int(len(jj))}

# 市值覆蓋率：流動性前 300 名中有市值的比例
S52, V60, SIG = L.rolling_signals()
cov = {}
for d in L.official_days('2006-01-01', '2026-10-12'):
    t = IDX.get_loc(d)
    base = np.isfinite(V60[t]) & (V60[t] >= 1e7) & (L.Cn[t] >= 10)
    vv = np.where(base, V60[t], -np.inf); top = np.argsort(-vv)[:300]
    cov[str(d.date())] = float(np.isfinite(MC[t, top]).mean())
cs = pd.Series(cov); cs.index = pd.to_datetime(cs.index)
OUT['mcap_coverage_liq_top300'] = {y: round(float(v), 3) for y, v in cs.groupby(cs.index.year).mean().items()}
OUT['gate_pass_all'] = bool(OUT['gate_i']['pass'] and OUT['gate_ii']['pass'] and OUT['gate_iii']['pass'])
(L.HERE / 'results_gate.json').write_text(L.dumps(OUT), encoding='utf-8')
print(L.dumps({k: v for k, v in OUT.items() if k != 'gate_i'})[:6000])
print('gate_i', {k: v for k, v in OUT['gate_i'].items() if k != 'reviews'})
print(rv[['review', 'agree', 'true_adds', 'pred_adds']].to_string())
