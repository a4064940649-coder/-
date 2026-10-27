#!/usr/bin/env python3
"""集保戶股權分散表（TDCC 1-5）「大戶增、散戶減」週訊號：事件研究 + 與主策略 M2 交叉。

資料（唯讀，GitHub 公開備份；只讀 CSV、不執行其中程式）：
* /home/user/vicleetw/smartflowai-tdcc/data/2026MMDD.csv：2026-04-17～2026-10-22，28 週（與 wirelessr 版逐位元相同，抽查 10-22）
* /home/user/wirelessr/tdcc-opendata-archive/snapshots/2021/*.csv：2021-07-02～2021-09-24，13 週（另一段獨立樣本）

事前固定（寫程式前決定，全部報告）：
* 級距：lvl15＝持股 >1,000,000 股（千張大戶）；b400＝lvl12..15（>400 張）；small＝lvl1..8（≤50 張）；holders＝lvl17 總人數。
* 訊號（資料日 t 對 t−1 週或 t−4 週）：
  S1 = Δbig_1w（千張大戶比例一週變化，pp）——五分位
  S2 = Δbig_4w——五分位
  S3 = 網路常見：Δbig_1w > 0 且 Δholders_1w < 0（大戶增、總人數減）——事件
  S4 = Δbig_4w ≥ +1pp 且 Δsmall_4w ≤ −1pp（四週大戶增 1pp、散戶減 1pp）——事件
  S5（反向，供排除用）= Δbig_4w ≤ −1pp 且 Δsmall_4w ≥ +1pp（大戶減、散戶增）——事件
* 成交：資料日（週五或週最後營業日）之後第一個交易日開盤買（開盤漲停 ≥9.4% 視為買不到）；持有 5／20／60 個交易日，以第 h 日收盤賣。
* 異常報酬 AR = 個股報酬 − 0.5% − 同日進場的上市流動池等權報酬（池：60 日均成交值 ≥ 1,000 萬、股價 ≥ 10、4 碼普通股）。
* t：事件 AR 先按進場月份平均，再對月份序列做 t（月份分群）。另報週序列 t 供參考。
* 切分：2026 段前 14 週（資料日 ≤ 2026-07-17）＝挑選期，其後＝驗證期；2021 段另列（獨立樣本）。
* 主策略交叉：exlib M2（n=30、n=10），每次換股（訊號日 d）用 d 之前最新一週 TDCC；持股中標記 S5（或 S2 最低五分位），比較「標記 vs 未標記」持股的換股期報酬（開盤→下次換股開盤），以及剔除後組合（其餘等權、扣 0.5% 單趟近似）是否較好。
"""
import sys, json, math, glob, re
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'exec_rebuild'))
import exlib as L  # noqa: E402

TD_FILES = sorted(glob.glob('/home/user/vicleetw/smartflowai-tdcc/data/2026*.csv')) + \
    sorted(glob.glob('/home/user/wirelessr/tdcc-opendata-archive/snapshots/2021/*.csv'))

def load_tdcc():
    rows = []
    for f in TD_FILES:
        d = pd.read_csv(f, encoding='utf-8-sig', dtype=str)
        d.columns = [c.replace('﻿', '').strip() for c in d.columns]
        d.columns = ['date', 'code', 'lvl', 'n', 'sh', 'pct']
        d['code'] = d['code'].str.strip(); d['date'] = d['date'].str.replace('﻿', '').str.strip()
        d = d[d['code'].str.fullmatch(r'[1-9]\d{3}')]
        d['lvl'] = d['lvl'].astype(int); d['n'] = pd.to_numeric(d['n']); d['pct'] = pd.to_numeric(d['pct'])
        g = d.groupby('code')
        out = pd.DataFrame({
            'big': d[d.lvl == 15].set_index('code')['pct'],
            'b400': d[d.lvl.between(12, 15)].groupby('code')['pct'].sum(),
            'small': d[d.lvl.between(1, 8)].groupby('code')['pct'].sum(),
            'holders': d[d.lvl == 17].set_index('code')['n'],
        })
        out['date'] = pd.Timestamp(d['date'].iloc[0]); rows.append(out.reset_index())
    T = pd.concat(rows, ignore_index=True)
    return T

T = load_tdcc()
WEEKS = sorted(T['date'].unique())
def wide(col): return T.pivot_table(index='date', columns='code', values=col, aggfunc='last').sort_index()
BIG, SMALL, HOLD = wide('big'), wide('small'), wide('holders')

def seg_of(d): return '2021' if d.year == 2021 else '2026'
def diff(W, k):
    """同一段內 k 週差分（2021 與 2026 不相接）。"""
    out = []
    for s in ['2021', '2026']:
        w = W[[seg_of(d) == s for d in W.index]]
        out.append(w - w.shift(k))
    return pd.concat(out).sort_index()

dB1, dB4, dS4, dH1 = diff(BIG, 1), diff(BIG, 4), diff(SMALL, 4), diff(HOLD, 1)
SIG = {
    'S3_big_up_holders_down': (dB1 > 0) & (dH1 < 0),
    'S4_big4w_up1_small4w_dn1': (dB4 >= 1) & (dS4 <= -1),
    'S5_big4w_dn1_small4w_up1': (dB4 <= -1) & (dS4 >= 1),
}

IDX, COLS = L.IDX, L.COLS
AO, AC = L.AO, L.AC
v60 = L.v60_all; CR = L.C
pool_all = (v60 >= 1e7) & (CR >= 10)
common = np.array([bool(re.fullmatch(r'[1-9]\d{3}', c)) for c in COLS])

def entry_index(d):
    j = IDX.searchsorted(d, side='right')       # 資料日之後第一個交易日
    return j if j < len(IDX) else None

def fwd(i, h):
    """i 日開盤買、i+h-1 日收盤賣（無限期 ffill 價）。"""
    e = i + h - 1
    if e >= len(IDX): return None
    return AC[e] / AO[i] - 1

HS = [5, 20, 60]
def week_frame(d):
    """回傳該資料週：pool mask（以資料日 d 收盤判斷）、進場 i、各 h 的報酬陣列與池等權。"""
    i = entry_index(d)
    if i is None: return None
    dd = IDX[IDX <= d][-1]
    pm = pool_all.loc[dd].to_numpy() & common & np.isfinite(AO[i]) & ~L.GAP_UP_OPEN[i]
    out = {'i': i, 'pool': pm}
    for h in HS:
        r = fwd(i, h)
        if r is None: continue
        r = np.where(pm, r, np.nan); out[h] = (r, np.nanmean(r))
    return out

def label(d):
    if d.year == 2021: return 'y2021'
    return 'select_2026H1' if d <= pd.Timestamp('2026-07-17') else 'valid_2026H2'

def month_t(df):
    """df: 事件列（entry_month, ar）。按月平均 → t。"""
    if len(df) == 0: return {'n': 0}
    m = df.groupby('m')['ar'].mean()
    t = float(m.mean() / (m.std(ddof=1) / math.sqrt(len(m)))) if len(m) > 2 and m.std() > 0 else float('nan')
    return {'n': int(len(df)), 'months': int(len(m)), 'mean_ar_pct': round(float(df['ar'].mean() * 100), 2),
            'hit': round(float((df['ar'] > 0).mean()), 3), 't_month': round(t, 2)}

res = {'data': {'weeks_2026': [str(w.date()) for w in WEEKS if w.year == 2026],
                'weeks_2021': [str(w.date()) for w in WEEKS if w.year == 2021],
                'n_codes_last_week': int(BIG.iloc[-1].notna().sum())}}

# ---------- 事件型訊號 ----------
ev = {k: [] for k in SIG}
quint = {k: [] for k in ['S1_dBig1w', 'S2_dBig4w']}
for d in WEEKS:
    wf = week_frame(pd.Timestamp(d))
    if wf is None: continue
    i = wf['i']; pm = wf['pool']
    for k, S in SIG.items():
        if d not in S.index: continue
        s = S.loc[d].reindex(COLS).fillna(False).to_numpy().astype(bool) & pm
        for h in HS:
            if h not in wf: continue
            r, bm = wf[h]
            for j in np.where(s & np.isfinite(r))[0]:
                ev[k].append({'d': d, 'm': IDX[i].strftime('%Y-%m'), 'h': h, 'code': COLS[j], 'ar': r[j] - 0.005 - bm, 'raw': r[j]})
    for k, X in [('S1_dBig1w', dB1), ('S2_dBig4w', dB4)]:
        if d not in X.index: continue
        x = X.loc[d].reindex(COLS).to_numpy()
        for h in HS:
            if h not in wf: continue
            r, bm = wf[h]
            ok = pm & np.isfinite(x) & np.isfinite(r)
            if ok.sum() < 100: continue
            q = pd.qcut(pd.Series(x[ok]).rank(method='first'), 5, labels=False).to_numpy()
            rr = r[ok]
            row = {'d': d, 'm': IDX[i].strftime('%Y-%m'), 'h': h, 'n': int(ok.sum())}
            for qq in range(5): row[f'Q{qq+1}'] = float(np.mean(rr[q == qq]) - bm)
            quint[k].append(row)

out_ev = {}
for k, rows in ev.items():
    df = pd.DataFrame(rows)
    o = {}
    for h in HS:
        x = df[df.h == h] if len(df) else df
        o[f'h{h}'] = {'all': month_t(x)}
        for lab in ['select_2026H1', 'valid_2026H2', 'y2021']:
            o[f'h{h}'][lab] = month_t(x[[label(pd.Timestamp(v)) == lab for v in x['d']]] if len(x) else x)
    out_ev[k] = o
res['events'] = out_ev
res['events_per_week_h5'] = {k: round(len([e for e in v if e['h'] == 5]) / max(1, len({e['d'] for e in v if e['h'] == 5})), 1) for k, v in ev.items()}

out_q = {}
for k, rows in quint.items():
    df = pd.DataFrame(rows); o = {}
    for h in HS:
        x = df[df.h == h]
        if len(x) == 0: continue
        oh = {}
        for lab in ['all', 'select_2026H1', 'valid_2026H2', 'y2021']:
            y = x if lab == 'all' else x[[label(pd.Timestamp(v)) == lab for v in x['d']]]
            if len(y) == 0: continue
            sp = y['Q5'] - y['Q1']
            mm = y.assign(sp=sp).groupby('m')[['Q1', 'Q5', 'sp']].mean()
            def tt(s): return round(float(s.mean() / (s.std(ddof=1) / math.sqrt(len(s)))), 2) if len(s) > 2 and s.std() > 0 else None
            oh[lab] = {'weeks': int(len(y)), 'months': int(len(mm)), 'avg_pool_n': round(float(y['n'].mean()), 1),
                       **{f'Q{q}_ex_pct': round(float(y[f'Q{q}'].mean() * 100), 2) for q in range(1, 6)},
                       'Q5_net_ex_pct': round(float((y['Q5'] - 0.005).mean() * 100), 2),
                       'Q5minusQ1_pct': round(float(sp.mean() * 100), 2), 't_spread_month': tt(mm['sp']), 't_spread_week': tt(sp),
                       't_Q5net_month': tt(mm['Q5'] - 0.005)}
        o[f'h{h}'] = oh
    out_q[k] = o
res['quintiles'] = out_q

# ---------- 與主策略 M2 交叉 ----------
def latest_week_before(d):
    w = [x for x in WEEKS if x < d]
    if not w: return None
    w = w[-1]
    return w if (d - w).days <= 10 else None   # 只用 10 天內的最新一週

cross = {}
for n in (30, 10):
    events, _ = L.selection('M2', n)
    ex_days = sorted(events)
    rows = []
    for a, b in zip(ex_days[:-1], ex_days[1:]):
        dsig = IDX[a - 1]                       # 訊號日（M2：換股日收盤算訊號，隔日開盤成交）
        w = latest_week_before(dsig + pd.Timedelta(days=1))
        if w is None: continue
        pk = events[a][1]
        r = AO[b, pk] / AO[a, pk] - 1           # 開盤→下次換股開盤
        codes = COLS[pk]
        f5 = SIG['S5_big4w_dn1_small4w_up1'].loc[w].reindex(codes).fillna(False).to_numpy().astype(bool) if w in dB4.index else np.zeros(len(pk), bool)
        x = dB4.loc[w].reindex(COLS)
        pm = pool_all.loc[IDX[IDX <= w][-1]].to_numpy() & common & x.notna().to_numpy()
        thr = np.nanquantile(x.to_numpy()[pm], 0.2) if pm.sum() > 100 else np.nan
        fq1 = (x.reindex(codes).to_numpy() <= thr)
        for nm, fl in [('S5', f5), ('dBig4w_Q1', fq1)]:
            rows.append({'period': str(IDX[a].date()), 'flag': nm, 'n_pick': int(len(pk)), 'n_flag': int(fl.sum()),
                         'ret_all': float(np.nanmean(r)), 'ret_keep': float(np.nanmean(r[~fl])) if (~fl).sum() else float('nan'),
                         'ret_flagged': float(np.nanmean(r[fl])) if fl.sum() else float('nan'),
                         'week_used': str(w.date())})
    df = pd.DataFrame(rows)
    o = {}
    for nm in ['S5', 'dBig4w_Q1']:
        y = df[df.flag == nm]
        if len(y) == 0: continue
        diffk = (y['ret_keep'] - y['ret_all'])
        o[nm] = {'periods': int(len(y)), 'flagged_total': int(y['n_flag'].sum()), 'picks_total': int(y['n_pick'].sum()),
                 'mean_ret_all_pct': round(float(y['ret_all'].mean() * 100), 2), 'mean_ret_keep_pct': round(float(y['ret_keep'].mean() * 100), 2),
                 'mean_ret_flagged_pct': round(float(y['ret_flagged'].mean() * 100), 2) if y['ret_flagged'].notna().any() else None,
                 'keep_minus_all_pct': round(float(diffk.mean() * 100), 2),
                 't_keep_minus_all': L.tstat(diffk), 'detail': y.drop(columns='flag').round(4).to_dict('records')}
    cross[f'M2_n{n}'] = o
res['main_cross'] = cross

(HERE / 'results_tdcc.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k: v for k, v in res.items() if k != 'main_cross'}, ensure_ascii=False, indent=1, default=str)[:12000])
for k, v in cross.items():
    for nm, o in v.items():
        print(k, nm, {a: b for a, b in o.items() if a != 'detail'})
