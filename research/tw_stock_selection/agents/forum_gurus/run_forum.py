#!/usr/bin/env python3
"""PTT Stock [標的] 文績效驗證（forum_gurus）。

資料：posts_raw.parquet（build_posts.py 由 a607ernie/Ptt-Stock-Data 的 git 快照歷史合併，2023-01-09 ～ 2025-04）。
價格：研究中心上市日線面板（經 exec_rebuild/exlib.py 載入，重用其還原開盤、開盤漲停判定與主策略選股）。

事前固定的規則（寫在跑結果之前，全部報告）：
* 事件：TAG == '標的'（不含 Re:）；內文「分類」欄位判出唯一方向（多 或 空）；
  標題（找不到再看「標的：」那行）恰好對到一檔上市普通股（面板內代號）。同作者 5 個交易日內同檔同方向只留第一篇。
* 時間：PTT 時間即台灣時間。交易日 09:00 前發文 → 當日開盤；其餘（盤中、盤後、假日）→ 下一個交易日開盤（主規則）。
  敏感度：盤中（09:00–13:30）發文改用當日收盤進場。進場日距發文日 > 7 天（面板缺 2024-01-02～02-14）剔除。
* 報酬：還原開盤 → h 個交易日後的還原開盤（h = 5、20）。對手：同日流動性池（前一日 60 日均值 ≥ 1,000 萬、收盤 ≥ 10）等權。
  做多扣來回 0.5%（壓力 1%）。開盤漲停（跳空 ≥ 9.4%）視為買不到、剔除。股票本身也要過流動性門檻（主規則）。
* 統計：異常報酬依進場月份分群，月平均的 t（t_month）；另列事件數、命中率。
* 期間：前半（挑選）= 發文 < 2024-03-01；後半（驗證）= 發文 ≥ 2024-03-01。
* R1 全部標的文；做多文 = 跟單淨異常報酬；空方文只報「個股異常報酬」及「做多的反向」（−異常報酬，不扣成本，不實際放空）。
* R2 熱度：進場時點之前的「推」數（從內文留言時間戳逐則計算，避免偷看進場後的推文）；門檻 = 前半做多事件的第 90 百分位，固定後套用到後半。
* R3 大神：前半的事件中，20 日窗口在 2024-03-01 前結束的方向文（多＋空）≥ 10 篇的作者，命中 = 方向調整後 20 日異常報酬 > 0；
  主規則取命中率前 5 名（同分用平均方向調整報酬），另報「命中率 ≥ 60% 全部」。只在後半驗證他們的新文章。
* R4 反向指標：熱度前 5%（前半第 95 百分位門檻）的做多文，單尾檢驗異常報酬 < 0。
* 主策略重疊：exlib.selection('M2', 30)（52 週高點 × 月營收，可成交版）在進場日的持股名單。
"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild'))
import exlib as L  # noqa: E402

IDX, COLS, CI = L.IDX, L.COLS, L.CI
AO, AC = L.AO, L.AC
GAP = L.GAP_UP_OPEN; LOCKC = L.LOCK_UP_CLOSE
V60 = L.V.rolling(60).mean().to_numpy(); CR = L.C.to_numpy()
LIQ = (V60 >= 1e7) & (CR >= 10)                     # 第 t 列 = 第 t 日收盤時的流動性
NT = len(IDX)
SPLIT = pd.Timestamp('2024-03-01')
COST, COST_S = 0.005, 0.01
HS = (5, 20)

names = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype={'code': str})
name2code = {r.name_: r.code for r in names.rename(columns={'name': 'name_'}).itertuples() if isinstance(r.name_, str) and len(r.name_) >= 2}
code_mkt = dict(zip(names.code, names.market))

# ---------------- 解析 ----------------
raw = pd.read_parquet(HERE / 'posts_raw.parquet')
raw = raw[raw['TAG'] == '標的'].copy()
raw['t'] = pd.to_datetime(raw['TIME'], format='%a %b %d %H:%M:%S %Y', errors='coerce')
raw['author'] = raw['AUTHOR'].astype(str).str.strip().str.split(' ').str[0]

TEMPL = [r'多\s*/\s*空\s*/\s*(討論|請益)\s*/\s*心得', r'\(請選擇並刪除無關分類\)', r'（請選擇並刪除無關分類）']
def direction(s):
    m = re.search(r'分類\s*[:：]', s)
    if not m: return None, 'nocat'
    seg = s[m.end():]
    e = re.search(r'(分析\s*/\s*正文|進退場|\n\s*3\s*\.)', seg)
    seg = seg[:e.start()] if e else seg[:80]
    for p in TEMPL: seg = re.sub(p, ' ', seg)
    seg = re.sub(r'\(.*?\)|（.*?）', ' ', seg)
    hl, hs = ('多' in seg), ('空' in seg)
    if hl and not hs: return 'L', 'ok'
    if hs and not hl: return 'S', 'ok'
    if hl and hs: return None, 'both'
    return None, 'nondir'

def codes_in(text):
    cs = []
    for c in re.findall(r'(?<![0-9A-Za-z])([1-9][0-9]{3})(?![0-9])', text):
        if c not in cs: cs.append(c)
    return cs

def stock_of(title, content):
    t = re.sub(r'^\[標的\]', '', title)
    cs = [c for c in codes_in(t) if c in code_mkt or c in CI]
    cs = [c for c in cs if not (c.startswith('19') or c.startswith('20')) or any(n in t for n, cc in name2code.items() if cc == c)]
    if not cs:
        m = re.search(r'標的\s*[:：](.{0,40})', content, re.S)
        if m:
            line = m.group(1).strip().split('\n')[0]
            cs = [c for c in codes_in(line) if c in code_mkt or c in CI]
            if not cs:
                hit = sorted({cc for n, cc in name2code.items() if n in line}, key=lambda x: x)
                cs = hit
        if not cs:
            hit = {cc for n, cc in name2code.items() if len(n) >= 2 and n in t}
            cs = sorted(hit)
    if len(cs) != 1: return None, ('multi' if len(cs) > 1 else 'none')
    return cs[0], 'ok'

CMT = re.compile(r'^(推|噓|→)\s*(\S+?)\s*:.*?(\d{2})/(\d{2})\s+(\d{2}):(\d{2})\s*$', re.M)
def comments(content, t0):
    out = []
    for m in CMT.finditer(content):
        mo, dd, hh, mi = map(int, m.group(3, 4, 5, 6))
        y = t0.year + (1 if mo < t0.month - 6 else 0)
        try: ts = pd.Timestamp(y, mo, dd, hh, mi)
        except ValueError: continue
        out.append((m.group(1), ts))
    return out

rows = []
cnt = {'total': len(raw)}
for r in raw.itertuples():
    d, why = direction(r.CONTENT)
    cnt['dir_' + why] = cnt.get('dir_' + why, 0) + 1
    code, w2 = stock_of(r.TITLE, r.CONTENT)
    cnt['stk_' + w2] = cnt.get('stk_' + w2, 0) + 1
    rows.append(dict(id=r.ID, t=r.t, author=r.author, title=r.TITLE, dir=d, code=code,
                     mkt=code_mkt.get(code) if code else None, in_panel=(code in CI) if code else False,
                     push_final=r.PUSH, boo_final=r.BOO, last_snap=r.last_snap, cm=comments(r.CONTENT, r.t)))
P = pd.DataFrame(rows).sort_values('t').reset_index(drop=True)
cnt['directional_single_stock'] = int((P.dir.notna() & P.code.notna()).sum())
cnt['dir_single_listed_panel'] = int((P.dir.notna() & P.in_panel).sum())
cnt['dir_single_otc'] = int((P.dir.notna() & P.code.notna() & (P.mkt == 'TPEx')).sum())

# ---------------- 進場 ----------------
day_pos = {d: i for i, d in enumerate(IDX)}
idx_np = IDX.values
def entry(t, intraday_close=False):
    dt = t.normalize()
    if dt in day_pos:
        i = day_pos[dt]
        if t.hour < 9: return i, 'open'
        if intraday_close and (t.hour < 13 or (t.hour == 13 and t.minute < 30)): return i, 'close'
        j = i + 1
    else:
        j = int(np.searchsorted(idx_np, np.datetime64(dt), side='left'))
    if j >= NT or (IDX[j] - dt).days > 7: return None, None
    return j, 'open'

def px_at(i, mode, col=None):
    A = AO if mode == 'open' else AC
    return A[i] if col is None else A[i, col]

def bench(i, mode, h):
    """同日流動性池等權：進場價 → i+h 開盤。"""
    pool = LIQ[i - 1] & np.isfinite(L.OPEN_RAW[i]) & np.isfinite(L.ADJ_RAW[i])
    p0 = px_at(i, mode)[pool]; p1 = AO[i + h][pool]
    rr = p1 / p0 - 1
    return np.nanmean(rr)

# 主策略持股（M2：換股日隔日開盤成交，30 檔）
ev_main, _ = L.selection('M2', 30)
ex_days = np.array(sorted(ev_main))
def main_hold(i):
    k = np.searchsorted(ex_days, i, side='right') - 1
    if k < 0: return set()
    return set(ev_main[ex_days[k]][1].tolist())

def build(intraday_close=False):
    out = []
    for r in P[P.dir.notna() & P.in_panel].itertuples():
        i, mode = entry(r.t, intraday_close)
        if i is None or i + max(HS) >= NT: continue
        j = CI[r.code]
        if not (np.isfinite(L.ADJ_RAW[i, j]) and (mode == 'close' or np.isfinite(L.OPEN_RAW[i, j]))): continue
        blocked = bool(GAP[i, j]) if mode == 'open' else bool(LOCKC[i, j])
        etime = IDX[i] + (pd.Timedelta(hours=9) if mode == 'open' else pd.Timedelta(hours=13, minutes=30))
        push_pt = sum(1 for k, ts in r.cm if k == '推' and ts < etime)
        boo_pt = sum(1 for k, ts in r.cm if k == '噓' and ts < etime)
        rec = dict(id=r.id, t=r.t, author=r.author, code=r.code, dir=r.dir, ei=i, eday=IDX[i], mode=mode,
                   blocked=blocked, liq=bool(LIQ[i - 1, j]), push_pt=push_pt, boo_pt=boo_pt, push_final=r.push_final,
                   in_main=j in main_hold(i), src3=bool(L.SRC.iat[i, j] == 3) if np.isfinite(L.SRC.iat[i, j]) else False)
        p0 = px_at(i, mode, j)
        rec['pre20'] = AC[i - 1, j] / AC[i - 21, j] - 1 if i > 21 else np.nan
        for h in HS:
            rr = AO[i + h, j] / p0 - 1
            rec[f'r{h}'] = rr; rec[f'b{h}'] = bench(i, mode, h); rec[f'a{h}'] = rr - rec[f'b{h}']
        out.append(rec)
    E = pd.DataFrame(out)
    # 同作者 5 交易日內同檔同方向只留第一篇
    E = E.sort_values('t'); keep = []; last = {}
    for k, r in E.iterrows():
        key = (r.author, r.code, r.dir)
        if key in last and r.ei - last[key] < 5: keep.append(False); continue
        last[key] = r.ei; keep.append(True)
    E = E[keep].copy()
    E['month'] = E.eday.dt.to_period('M').astype(str)
    E['half'] = np.where(E.t < SPLIT, 'first', 'second')
    return E

def tmonth(x, m):
    g = pd.Series(np.asarray(x, float), index=np.asarray(m)).groupby(level=0).mean().dropna()
    if len(g) < 3: return np.nan, len(g)
    return float(g.mean() / g.std(ddof=1) * np.sqrt(len(g))), len(g)

def stat(E, col, sign=1.0, cost=0.0):
    if len(E) == 0: return dict(n=0)
    x = sign * E[col].to_numpy(float) - cost
    ok = np.isfinite(x); x = x[ok]; m = E.month.to_numpy()[ok]
    t, nm = tmonth(x, m)
    return dict(n=int(ok.sum()), mean=round(float(np.mean(x)) * 100, 3), median=round(float(np.median(x)) * 100, 3),
                hit=round(float((x > 0).mean()), 3), t_month=None if np.isnan(t) else round(t, 2), n_month=nm)

def block(E, label):
    """做多：跟單淨異常（0.5% 與 1%）；空方：個股異常與反向（不扣成本）。"""
    res = {}
    for half, sub in [('all', E), ('first', E[E.half == 'first']), ('second', E[E.half == 'second'])]:
        Lg = sub[sub.dir == 'L']; Sg = sub[sub.dir == 'S']
        d = {}
        for h in HS:
            d[f'long_net_abn{h}'] = stat(Lg, f'a{h}', 1, COST)
            d[f'long_net_abn{h}_cost1'] = stat(Lg, f'a{h}', 1, COST_S)
            d[f'long_gross_abn{h}'] = stat(Lg, f'a{h}', 1, 0)
            d[f'short_stock_abn{h}'] = stat(Sg, f'a{h}', 1, 0)
            d[f'short_reverse{h}'] = stat(Sg, f'a{h}', -1, 0)
        d['n_long'] = int(len(Lg)); d['n_short'] = int(len(Sg))
        d['long_pre20_mean'] = round(float(Lg.pre20.mean()) * 100, 2) if len(Lg) else None
        res[half] = d
    return res

E_all = build(False)
E_ic = build(True)
blocked_n = int(E_all[(E_all.dir == 'L')].blocked.sum())
liq_drop = int((~E_all.liq).sum())
base = E_all[E_all.liq & ~((E_all.dir == 'L') & E_all.blocked)].copy()
base_ic = E_ic[E_ic.liq & ~((E_ic.dir == 'L') & E_ic.blocked)].copy()

R = {'counts': cnt, 'n_events_built': int(len(E_all)), 'n_long_blocked_open_limitup': blocked_n,
     'n_illiquid_dropped': liq_drop, 'n_base': int(len(base)),
     'base_period': [str(base.eday.min().date()), str(base.eday.max().date())]}
R['R1'] = block(base, 'R1')
R['R1_intraday_close_entry'] = block(base_ic, 'R1ic')
R['R1_no_liq_filter'] = block(E_all[~((E_all.dir == 'L') & E_all.blocked)], 'R1nl')
R['R1_yearly_long_net20'] = {str(y): stat(g[g.dir == 'L'], 'a20', 1, COST) for y, g in base.groupby(base.eday.dt.year)}
R['R1_yearly_long_net5'] = {str(y): stat(g[g.dir == 'L'], 'a5', 1, COST) for y, g in base.groupby(base.eday.dt.year)}
R['R1_src3_share'] = round(float(base.src3.mean()), 3)

# ---------------- R2 熱度 ----------------
L1 = base[(base.dir == 'L') & (base.half == 'first')]
q90 = float(np.quantile(L1.push_pt, 0.90)); q95 = float(np.quantile(L1.push_pt, 0.95))
R['R2_threshold_push_pt_p90'] = q90; R['R4_threshold_push_pt_p95'] = q95
hot = base[base.push_pt >= q90]; cold = base[base.push_pt < q90]
R['R2_hot'] = block(hot, 'R2'); R['R2_rest'] = block(cold, 'R2rest')
# 敏感度：用最終推文數（含進場後推文，有前視，只作對照）
q90f = float(np.quantile(L1.push_final, 0.90))
R['R2_hot_finalpush_lookahead'] = block(base[base.push_final >= q90f], 'R2f')
# 熱度五分位（做多，全期，描述）
Lb = base[base.dir == 'L'].copy()
Lb['hq'] = pd.qcut(Lb.push_pt.rank(method='first'), 5, labels=False)
R['R2_long_quintiles_net20'] = {int(q): stat(g, 'a20', 1, COST) for q, g in Lb.groupby('hq')}
R['R2_long_quintiles_push_range'] = {int(q): [int(g.push_pt.min()), int(g.push_pt.max())] for q, g in Lb.groupby('hq')}

# ---------------- R4 反向指標 ----------------
ext = base[(base.dir == 'L') & (base.push_pt >= q95)]
R4 = {}
for half, sub in [('all', ext), ('first', ext[ext.half == 'first']), ('second', ext[ext.half == 'second'])]:
    R4[half] = {f'gross_abn{h}': stat(sub, f'a{h}', 1, 0) for h in HS}
R['R4_extreme_hot_long'] = R4

# ---------------- R3 大神 ----------------
F = base[(base.half == 'first')].copy()
F = F[F.ei + 20 < IDX.get_indexer([SPLIT], method='bfill')[0]]
F['sadj20'] = np.where(F.dir == 'L', F.a20, -F.a20)
au = F.groupby('author').agg(n=('sadj20', 'size'), hit=('sadj20', lambda x: (x > 0).mean()), m=('sadj20', 'mean'))
qual = au[au.n >= 10].sort_values(['hit', 'm'], ascending=False)
top5 = list(qual.index[:5]); hit60 = list(qual[qual.hit >= 0.6].index); bot5 = list(qual.sort_values(['hit', 'm']).index[:5])
S2 = base[base.half == 'second'].copy(); S2['sadj20'] = np.where(S2.dir == 'L', S2.a20, -S2.a20); S2['sadj5'] = np.where(S2.dir == 'L', S2.a5, -S2.a5)
def guru_eval(lst):
    g = S2[S2.author.isin(lst)]
    gl = g[g.dir == 'L']
    return dict(authors=lst, n_posts_second=int(len(g)), n_long=int(len(gl)), n_short=int((g.dir == 'S').sum()),
                long_net_abn5=stat(gl, 'a5', 1, COST), long_net_abn20=stat(gl, 'a20', 1, COST),
                dir_adj_gross5=stat(g, 'sadj5'), dir_adj_gross20=stat(g, 'sadj20'),
                per_author={a: dict(n=int((g.author == a).sum()), hit20=round(float((g[g.author == a].sadj20 > 0).mean()), 3) if (g.author == a).any() else None,
                                    mean20=round(float(g[g.author == a].sadj20.mean()) * 100, 2) if (g.author == a).any() else None) for a in lst})
R['R3'] = {'n_authors_ge10_first': int(len(qual)),
           'first_half_table': {a: dict(n=int(r.n), hit=round(float(r.hit), 3), mean_sadj20=round(float(r.m) * 100, 2)) for a, r in qual.iterrows()},
           'top5': guru_eval(top5), 'hit_ge_60': guru_eval(hit60), 'bottom5_contrast': guru_eval(bot5),
           'everyone_else_second_long_net20': stat(S2[(S2.dir == 'L') & ~S2.author.isin(top5)], 'a20', 1, COST)}
# 前半命中率與後半命中率的相關（全部 ≥10 篇且後半 ≥5 篇的作者）
au2 = S2.groupby('author').agg(n2=('sadj20', 'size'), hit2=('sadj20', lambda x: (x > 0).mean()))
pp = qual.join(au2, how='inner'); pp = pp[pp.n2 >= 5]
R['R3']['persistence'] = dict(n_authors=int(len(pp)), spearman_hit=round(float(pp.hit.rank().corr(pp.hit2.rank())), 3) if len(pp) > 2 else None)

# ---------------- 主策略重疊 ----------------
Lm = base[base.dir == 'L']
R['main_overlap'] = dict(share_long_posts_in_main30=round(float(Lm.in_main.mean()), 4), n_in=int(Lm.in_main.sum()),
                         in_main_net20=stat(Lm[Lm.in_main], 'a20', 1, COST), not_main_net20=stat(Lm[~Lm.in_main], 'a20', 1, COST),
                         in_main_net5=stat(Lm[Lm.in_main], 'a5', 1, COST), not_main_net5=stat(Lm[~Lm.in_main], 'a5', 1, COST),
                         share_short_posts_in_main30=round(float(base[base.dir == 'S'].in_main.mean()), 4))

# 描述：每月事件數、作者集中度
R['desc'] = dict(n_authors=int(base.author.nunique()), top10_author_share=round(float(base.author.value_counts().head(10).sum() / len(base)), 3),
                 long_share=round(float((base.dir == 'L').mean()), 3), events_per_month=base.groupby('month').size().to_dict(),
                 long_pre20_mean_pct=round(float(base[base.dir == 'L'].pre20.mean()) * 100, 2),
                 short_pre20_mean_pct=round(float(base[base.dir == 'S'].pre20.mean()) * 100, 2),
                 long_pre20_median_pct=round(float(base[base.dir == 'L'].pre20.median()) * 100, 2),
                 abs_abn20_gt50pct=int((base.a20.abs() > 0.5).sum()))
base.drop(columns=[]).to_parquet(HERE / 'events.parquet')
(HERE / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k: R[k] for k in ['counts', 'n_events_built', 'n_long_blocked_open_limitup', 'n_illiquid_dropped', 'n_base', 'base_period']}, ensure_ascii=False, default=str))
