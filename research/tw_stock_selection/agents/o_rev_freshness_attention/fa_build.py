#!/usr/bin/env python3
"""關注度 × 換股時點 j 的組合建構（本題核心；PREREG 雜湊涵蓋本檔）。

對每個錨定月 m、每個 j∈{0..12}、每個 AB 種子：
  1. 橫斷面＝V7 AB_V7 規格的合格池 el（前 300 流動性池、排除金融／KY／DR、成交日仍在處置者、次日無開盤者、缺營收或缺 52 週比者）。
  2. 分組：
     A    ＝el 內 60 日均成交值（訊號日 V60，gapfix 窗）的池內百分位：≤ 1/3 低（0）、(1/3, 2/3] 中（1）、> 2/3 高（2）。
     M50  ＝0050 成分（成交日開盤時的 PIT 名單；2015-09-18 前為 o_capmatched 規則名單）：1＝成分、0＝非成分。
     FO   ＝el ∩ 有外資持股比率者（o_sbl fo_panel，訊號日；2021-10-08 起）的三分位（0 低、1 中、2 高）。
     COMBO＝A 低三分位且非 0050 成分（只算一組，標 'lowA_non50'）；HI50＝0050 成分（同 M50 的 1）。
  3. 腿：main＝主策略分數（0.5×52 週比百分位＋0.5×近 3 月營收年增率百分位，el 內）；rev＝只用營收百分位；px＝只用 52 週比百分位。
  4. 組內依腿分數由高到低取前 n_g＝max(1, round(0.2×N_g)) 檔（同分依官方同分序：營收年增率高者先、再依欄位順序），開盤漲停（tick 漲停價）且上一期
     同組合未持有者跳過、由下一名遞補；等權。
  5. 持有：成交日 e（訊號日次一交易日）開盤 → 下一個有效錨定月同一 j 的成交日開盤（該 j 月份被跳過則續抱到下一個）；
     報酬＝還原開盤比（A×開/收；停牌或下市後持平於最後收盤，含 P 下市價補值；不做併購現金出清）。
     最後一窗的終點＝錨定 2026-10 同一 j 的成交日開盤；超出面板者該窗刪除。
  6. α_g(j)＝組合報酬（扣一般成本：買 0.1855%×新進比例＋賣 0.4855%×賣出比例；比例以檔數計）－同組 el 等權報酬（零成本）。
     壓力成本另存（每邊 0.5%）。種子平均：各種子 α 取平均（j≥1 多數月份各種子相同，直接重用）。
另存 A 方案 main 腿的高、低兩組逐檔權重（1[入選]/n_g − 1/N_g，種子平均）與窗報酬，供股票分群 bootstrap 與單一股票占比。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import fa_common as C

E = C.engine(); K, L, V, STC = E['K'], E['L'], E['V'], E['STC']
N = K.N; COLS = K.COLS; IDX = K.IDX; AO = E['AO']
V60 = None

DEFAULT = dict(block=True, tercile_on='el', seeds=C.SEEDS)

def _all_days():
    J = {}
    for j in C.JS:
        days, de = C.j_days(j)
        ms = sorted(days)
        ends = {}
        for k, m in enumerate(ms):
            if k + 1 < len(ms): ends[m] = days[ms[k + 1]][1]
            else:
                d2 = STC.day_for(pd.Period('2026-10', 'M'), j)
                if d2 is not None and d2 in IDX:
                    t2 = IDX.get_loc(d2)
                    ends[m] = t2 + 1 if (t2 + 1 < len(IDX) and K.GOOD[t2 + 1]) else None
                else: ends[m] = None
        J[j] = {m: (days[m][0], days[m][1], ends[m]) for m in ms}
    return J

_S = {}
def strat(seed):
    if seed not in _S:
        J = _all_days()
        amap = {v[0]: m for j in J for m, v in J[j].items()}
        S = E['JV7'](E['YFX'], E['Y5F'], E['DMT'], unavail=(lambda d, _s=seed: STC.unavail_for(S.anchor[d], d, _s)))
        S.set_anchor(amap)
        _S[seed] = (S, J, amap)
    return _S[seed]

def tercile(x, mask):
    q = pd.Series(np.where(mask, x, np.nan)).rank(pct=True).to_numpy()
    g = np.full(N, -1)
    g[mask & (q <= 1 / 3)] = 0; g[mask & (q > 1 / 3) & (q <= 2 / 3)] = 1; g[mask & (q > 2 / 3)] = 2
    return g

def pct(x, mask): return pd.Series(np.where(mask, x, np.nan)).rank(pct=True).to_numpy()

FO_IDX = None
def fo_row(d):
    global FO_IDX
    F = C.foreign_panel()
    if FO_IDX is None: FO_IDX = {c: i for i, c in enumerate(F.columns)}
    if d not in F.index: return np.full(N, np.nan)
    r = F.loc[d].reindex(COLS).to_numpy(dtype=float)
    return r

def groups_for(cs, d, e, opts):
    el = cs['el']; v60 = cs['v60x']
    gA = tercile(v60, cs['top'] if opts['tercile_on'] == 'top' else el)
    gA = np.where(el, gA, -1)
    mem, _ = C.tw50_at(IDX[e])
    mm = np.zeros(N, bool)
    for c in mem:
        if c in K.CI: mm[K.CI[c]] = True
    gM = np.where(el, mm.astype(int), -1)
    fo = fo_row(d)
    okf = el & np.isfinite(fo)
    gF = tercile(fo, okf) if okf.sum() >= 30 else np.full(N, -1)
    combo = np.where(el & (gA == 0) & ~mm, 0, -1)
    return {'A': gA, 'M50': gM, 'FO': gF, 'COMBO': combo}

LEGS_BY_SCHEME = {'A': ('main', 'rev', 'px'), 'M50': ('main',), 'FO': ('main',), 'COMBO': ('main',)}

def pick(keyv, members, n, gu, prev, y3=None):
    """官方同分序：分數（四捨五入到 1e-10）由高到低 → 近 3 月營收年增率由高到低 → 欄位順序（＝v2_core.order 'official'）。"""
    idx = np.where(members)[0]
    sc = np.round(np.nan_to_num(keyv[idx], nan=-9), 10)
    yy = -np.nan_to_num(y3[idx], nan=-1e18) if y3 is not None else np.zeros(len(idx))
    order = idx[np.lexsort((idx, yy, -sc))]
    out = []
    for jj in order:
        if gu[jj] and jj not in prev: continue
        out.append(int(jj))
        if len(out) == n: break
    return out

def build(months, opts=None, contrib=True):
    """months：允許輸出的錨定月集合（呼叫端負責期間紀律；超過 max(months) 的月份完全不計算）。
    回傳 (alpha DataFrame, contrib list[(m, j, g, idx ndarray, w ndarray, r ndarray)])。"""
    o = dict(DEFAULT); o.update(opts or {})
    rows = []; crow = []
    Ss = {s: strat(s) for s in o['seeds']}
    J = Ss[o['seeds'][0]][1]
    _, V60x = V.rolling_signals('gapfix')
    mmax = max(months)
    for j in C.JS:
        prev = {}
        for m in sorted(J[j]):
            if m > mmax: break
            d, e, e2 = J[j][m]
            if e2 is None: continue
            res_seed = []; cache = {}
            for s in o['seeds']:
                S = Ss[s][0]
                U = STC.unavail_for(m, d, s)
                key = U.tobytes()
                if key in cache: res_seed.append(cache[key]); continue
                cs = dict(S.cross_section(d)); cs['v60x'] = V60x[cs['t']]
                el = cs['el']
                r = AO[e2] / AO[e] - 1.0
                gu = K.gap_up_open(e) if o['block'] else np.zeros(N, bool)
                G = groups_for(cs, d, e, o)
                legs = {'main': cs['sc'], 'rev': pct(cs['y3'], el), 'px': pct(cs['s52'], el)}
                out = {}
                for sch, gv in G.items():
                    for g in sorted(set(gv[gv >= 0].tolist())):
                        mem = gv == g
                        Ng = int(mem.sum())
                        if Ng < 5: continue
                        n = max(1, int(round(C.TOP_FRAC * Ng)))
                        ew = float(np.nanmean(r[mem]))
                        for leg in LEGS_BY_SCHEME[sch]:
                            pk = pick(legs[leg], mem & np.isfinite(legs[leg]), n, gu, prev.get((sch, g, leg, s), set()), cs['y3'])
                            out[(sch, g, leg)] = (pk, ew, Ng, n, mem)
                cache[key] = (out, r)
                res_seed.append(cache[key])
            if m in months:
                keys = set().union(*[set(x[0]) for x in res_seed])
                for k in keys:
                    al, aln, als, gr, ewl, cn = [], [], [], [], [], []
                    for s, (out, r) in zip(o['seeds'], res_seed):
                        if k not in out: continue
                        pk, ew, Ng, n, mem = out[k]
                        gross = float(np.mean(r[pk]))
                        pp = prev.get((k[0], k[1], k[2], s))
                        if pp is None: newf, soldf = 1.0, 0.0
                        else: newf = len(set(pk) - pp) / len(pk); soldf = len(pp - set(pk)) / max(len(pp), 1)
                        cost_n = C.BUY_N * newf + C.SELL_N * soldf; cost_s = C.BUY_S * newf + C.SELL_S * soldf
                        al.append(gross - ew); aln.append(gross - cost_n - ew); als.append(gross - cost_s - ew); gr.append(gross); ewl.append(ew)
                        cn.append((Ng, n, newf, soldf))
                    if not al: continue
                    cn = np.array(cn, float).mean(axis=0)
                    rows.append(dict(m=str(m), j=j, scheme=k[0], g=int(k[1]), leg=k[2], alpha_gross=float(np.mean(al)), alpha=float(np.mean(aln)),
                                     alpha_stress=float(np.mean(als)), port=float(np.mean(gr)), ew=float(np.mean(ewl)), Ng=cn[0], n=cn[1], newf=cn[2], soldf=cn[3],
                                     d=str(d.date()), e=str(IDX[e].date()), e2=str(IDX[e2].date())))
                if contrib:
                    for g in (0, 2):
                        k = ('A', g, 'main'); W = np.zeros(N); cnt = 0; rr = None
                        for s, (out, r) in zip(o['seeds'], res_seed):
                            if k not in out: continue
                            pk, ew, Ng, n, mem = out[k]
                            w = np.where(mem, -1.0 / Ng, 0.0); w[pk] += 1.0 / len(pk)
                            W += w; cnt += 1; rr = r
                        if cnt == 0: continue
                        W /= cnt; ix = np.where(W != 0)[0]
                        crow.append((str(m), j, g, ix.astype(np.int32), W[ix], np.nan_to_num(rr[ix], nan=0.0)))
            for s, (out, r) in zip(o['seeds'], res_seed):
                for k, v in out.items(): prev[(k[0], k[1], k[2], s)] = set(v[0])
    return pd.DataFrame(rows), crow

def contrib_frame(crow):
    m, j, g, c, w, r = [], [], [], [], [], []
    for (mm, jj, gg, ix, W, R) in crow:
        k = len(ix); m += [mm] * k; j += [jj] * k; g += [gg] * k; c.append(ix); w.append(W); r.append(R)
    return pd.DataFrame({'m': m, 'j': np.array(j, np.int8), 'g': np.array(g, np.int8), 'col': np.concatenate(c) if c else [],
                         'w': np.concatenate(w) if w else [], 'r': np.concatenate(r) if r else []})
