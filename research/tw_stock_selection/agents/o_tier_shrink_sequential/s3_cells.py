#!/usr/bin/env python3
"""步驟 2a（輸入建置，不含任何 α 事後）：焦點格的平移分布與殘差序列、前瞻各線的歷史序列。
輸出：
  data/cells.json：每格的 base（AB 或 B10）、13 點平移 S_j（pp／窗）、官方日平移、原始 α̂（官方日）。
  data/cell_series.parquet：每格官方日窗超額（pp），配置模擬的殘差來源。
  data/lines_series.parquet、data/lines_meta.json：前瞻各線與新策略的窗超額（pp）、版本族、σ（HAC L6 長期 sd）。
用法：python3 -I -B s3_cells.py [--inputs v6]   （v7：待 V7 交件，只換輸入檔，見 INPUTS）
"""
import sys, json, math, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import tlib as T

TIERS = ['30萬', '100萬', '300萬', '1000萬']; MODES = ['FIX', 'GROW']
INPUTS = {
 'v6': {'v4': 'raw/navs/navs_v4.parquet', 'v6': 'raw/navs/navs_v6.parquet', 'js': 'raw/navs/navs_jscan_v6.parquet',
        'kappa': 'raw/navs/navs_abavg.parquet', 'tick': 'data/navs_tick_abavg.parquet', 'zi': 'raw/exec/bench.parquet'},
 # V7 交件後填入對應檔（只換輸入）：AB/B10 官方淨值、jscan、各 j 兩模型可執行淨值；欄名規則見 build() 的 v7 分支。
 'v7': None,
}
COST = {'bM': 0.001855, 'sM': 0.004855, 'bE': 0.001355, 'sE': 0.002355}      # o_concentration_regime 一般成本（單邊）
REGIME_VERSIONS = [dict(id='V1', sig='S1', L=3, xC=1.0, h=0.0), dict(id='V2', sig='S1', L=6, xC=1.0, h=0.0),
                   dict(id='V3', sig='S1', L=12, xC=1.0, h=0.0), dict(id='V4', sig='S1skip', L=12, xC=1.0, h=0.0),
                   dict(id='V5', sig='S2', L=6, xC=1.0, h=0.0), dict(id='V6', sig='S2', L=6, xC=0.7, h=0.0),
                   dict(id='V7', sig='S2', L=6, xC=1.0, h=0.02), dict(id='V8', sig='S2', L=6, xC=0.7, h=0.02)]
WBAR = 0.4791666666666667

def wr(s): return T.win_ret(s)

def regime_states(W, sp):
    d = W['d']; past = d.shift(1)
    if sp['sig'] == 'S1': S = past.rolling(sp['L'], min_periods=sp['L']).sum()
    elif sp['sig'] == 'S1skip': S = past.shift(1).rolling(sp['L'], min_periods=sp['L']).sum()
    else: S = W['top5'] - W['top5'].shift(sp['L'])
    out = []; prev = None
    for v in S.to_numpy():
        if not np.isfinite(v): out.append(None); continue
        st = 'C' if v > sp['h'] else ('D' if v < -sp['h'] else (prev if prev is not None else 'D'))
        out.append(st); prev = st
    return pd.Series(out, index=S.index)

def rotate_daily(mainnav, e50, exec_w):
    """o_concentration_regime cr_lib.rotate 的複製（日頻，exec 日收盤移轉，主腿與 0050 各付單邊成本）。"""
    bM, sM, bE, sE = COST['bM'], COST['sM'], COST['bE'], COST['sE']
    ix = mainnav.index
    rM = mainnav.pct_change().fillna(0).to_numpy()
    e = e50.reindex(ix.union(e50.index)).ffill().reindex(ix)
    rE = e.pct_change().fillna(0).to_numpy()
    ex = {ix.get_indexer([pd.Timestamp(d)])[0]: w for d, w in exec_w.items() if pd.Timestamp(d) in ix}
    first = min(ex)
    a = b = 0.0; nav = np.full(len(ix), np.nan)
    for i in range(first, len(ix)):
        if i == first:
            w = ex[i]; a, b = 1 - w, w
        else:
            a *= 1 + rM[i]; b *= 1 + rE[i]
            if i in ex:
                w = ex[i]; tot = a + b; tb = w * tot
                c = (tb - b) * (sM + bE) if b < tb else (b - tb) * (sE + bM)
                tot -= c; a, b = (1 - w) * tot, w * tot
        nav[i] = a + b
    return pd.Series(nav, index=ix).dropna()

def regime_series(W, mainnav, e50, r50, sp, const_w=None):
    st = regime_states(W, sp)
    ix = mainnav.index
    exec_w = {}
    for d, s_ in st.items():
        if s_ is None: continue
        k = ix.searchsorted(pd.Timestamp(d), side='right')
        if k < len(ix): exec_w[ix[k]] = float(const_w if const_w is not None else (sp['xC'] if s_ == 'C' else 0.0))
    nav = rotate_daily(mainnav, e50, exec_w)
    w = pd.Series([sp['xC'] if s_ == 'C' else 0.0 for s_ in st.reindex(T.WIN).values], index=T.WIN)
    return wr(nav), w, st

def lrsd(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(math.sqrt(T.hac_var_mean(x, 6) * len(x)))

def build(inputs='v6'):
    I = INPUTS[inputs]
    assert I is not None, 'V7 輸入尚未登記'
    P = {k: T.HERE / v for k, v in I.items()}
    r50 = wr(T.e50_nav())
    V4 = pd.read_parquet(P['v4']); V6 = pd.read_parquet(P['v6']); JS = pd.read_parquet(P['js'])
    KA = pd.read_parquet(P['kappa']); TK = pd.read_parquet(P['tick']); ZI = pd.read_parquet(P['zi'])
    base = {'AB': wr(V4['AB_V4|normal']), 'B10': wr(V4['AB_V4_B10|normal']),
            'AB_s': wr(V4['AB_V4|stress']), 'B10_s': wr(V4['AB_V4_B10|stress'])}
    zi = wr(ZI['AB_V4|normal'])
    assert float((zi - base['AB']).abs().max()) < 1e-12
    def djw(prefix, cost):
        off = wr(V6[f'{prefix}|{cost}'])
        return pd.DataFrame({j: (wr(JS[f'{prefix}|{cost}|J{j}']) - off).fillna(0.0) * 100 for j in range(13)})
    DJW = {('AB', 'normal'): djw('AB_V6', 'normal'), ('AB', 'stress'): djw('AB_V6', 'stress'),
           ('B10', 'normal'): djw('AB_V6_B10', 'normal'), ('B10', 'stress'): djw('AB_V6_B10', 'stress')}
    DJ = {k: [float(v[j].mean()) for j in range(13)] for k, v in DJW.items()}
    cells, ser = {}, {}
    for b in ('AB', 'B10'):
        ex_n = (base[b] - r50) * 100; ex_s = (base[b + '_s'] - r50) * 100
        st_shift = float((ex_s - ex_n).mean())
        for cost, sh0, exs in (('normal', 0.0, ex_n), ('stress', st_shift, ex_s)):
            nm = b if cost == 'normal' else b + '_stress'
            cells[nm] = {'base': b, 'S': [sh0], 'kind': 'official', 'raw_alpha_hat': float(exs.mean())}
            cells[nm + '_J'] = {'base': b, 'S': [sh0 + x for x in DJ[(b, cost)]], 'kind': 'jmix', 'raw_alpha_hat': float(exs.mean())}
            ser[nm] = exs; ser[nm + '_J'] = exs
    b10n = (base['B10'] - r50) * 100
    for t in TIERS:
        for m in MODES:
            for model, src, cn, cs in (('KAPPA', KA, f'main|{t}|{m}|A', f'stress|{t}|{m}|B'), ('TICK', TK, f'tick|{t}|{m}|A', f'tickstress|{t}|{m}|B')):
                for cost in ('normal', 'stress'):
                    col = cn if cost == 'normal' else cs
                    drag = (wr(src[col]) - zi) * 100
                    bser = (base['B10'] if cost == 'normal' else base['B10_s'] - 0)
                    off = (bser - r50) * 100 + drag
                    shift_off = float(off.mean() - b10n.mean())
                    nm = f'EXEC|{t}|{m}|{cost}|{model}'
                    cells[nm] = {'base': 'B10', 'S': [shift_off + x for x in DJ[('B10', cost)]], 'kind': 'jmix',
                                 'shift_official': shift_off, 'drag_mean': float(drag.mean()), 'raw_alpha_hat_official': float(off.mean()),
                                 'raw_alpha_hat_jmean': float(off.mean() + np.mean(DJ[('B10', cost)]))}
                    ser[nm] = off
    S = pd.DataFrame(ser)
    assert S.notna().all().all(), S.isna().sum()[S.isna().sum() > 0]
    return cells, S, DJ, r50, base, DJW

def cell_shifts(cells, S, DJW, base, r50, mask):
    """在窗子集 mask 上重算每格的 13 點平移（敏感度用：拿掉 2021、拿掉最好 10 窗）。"""
    ref = {'AB': ((base['AB'] - r50) * 100)[mask], 'B10': ((base['B10'] - r50) * 100)[mask]}
    out = {}
    for nm, c in cells.items():
        off = S[nm][mask]
        sh0 = float(off.mean() - ref[c['base']].mean())
        if c['kind'] == 'official':
            out[nm] = [sh0]
        else:
            cost = 'stress' if '|stress|' in nm or nm.endswith('stress_J') else 'normal'
            D = DJW[(c['base'], cost)][mask]
            out[nm] = [sh0 + float(D[j].mean()) for j in range(13)]
    return out

def build_lines(r50, base):
    out = {}; meta = {}
    # 0050 混合、2330 混合（integrate_o6，B10 基底，一般成本）
    IO = pd.read_parquet(T.RAW / 'navs' / 'integrate_o6_navs.parquet')
    out['M_E50_30'] = (wr(IO['M_E50_30|B10|normal']) - r50) * 100
    out['M_2330_30'] = (wr(IO['M_2330_30|B10|normal']) - r50) * 100
    TS = pd.read_parquet(T.RAW / 'spa' / 'tsmc_inputs.parquet')
    TS.index = pd.DatetimeIndex(TS.index)
    assert float((TS['r0050'] - r50).abs().max()) < 1e-9
    out['TSMC'] = (TS['r2330'] - r50) * 100
    # 市值匹配選股（11 版本、凍結版、C50／C70）
    CM = pd.read_parquet(T.RAW / 'newstrat' / 'capmatched_navs_oos.parquet')
    vers = [c for c in CM.columns if c.endswith('|normal')]
    for c in vers: out['CM:' + c] = (wr(CM[c]) - r50) * 100
    out['CM_C50'] = (wr(CM['C50']) - r50) * 100; out['CM_C70'] = (wr(CM['C70']) - r50) * 100
    meta['capmatched'] = {'versions': ['CM:' + c for c in vers], 'frozen': 'CM:F1_K5_N20_sqrt|normal', 'n_versions': len(vers)}
    # 權值集中輪動（窗層級重建 8 版）
    W = pd.read_parquet(T.RAW / 'newstrat' / 'regime_W_full.parquet')
    main_w = base['AB']
    mainnav = pd.read_parquet(T.HERE / INPUTS['v6']['v4'])['AB_V4|normal'].dropna()
    e50 = T.e50_nav(); e50 = e50[e50.index != pd.Timestamp('2026-09-20')]
    rv = {}
    for sp in REGIME_VERSIONS:
        rot, w, st = regime_series(W, mainnav, e50, r50, sp)
        out['RG:' + sp['id']] = (rot - r50) * 100; rv[sp['id']] = {'frac_C': float((w > 0).mean()), 'switches': int((np.diff(w) != 0).sum())}
        if sp['id'] == 'V5':
            stat, _, _ = regime_series(W, mainnav, e50, r50, sp, const_w=WBAR)
            out['RG_D'] = (rot - stat) * 100
            out['RG_CORE50'] = 0.5 * (rot - r50) * 100
            # 與原研究報告的期間年化核對（窗層級近似）
            def cg(s, a, b):
                m = T.wmask(s.index, a, b); g = np.prod(1 + s[m].values); yrs = m.sum() / 12.0
                return float(g ** (1 / yrs) - 1)
            meta['regime_check'] = {'SEL_2019_2022_CAGR_rot_approx': cg(rot, '2019-01-01', '2022-12-31'), 'SEL_reported': 0.22263,
                                    'SEL_CAGR_static_approx': cg(stat, '2019-01-01', '2022-12-31'), 'SEL_static_reported': 0.25222,
                                    'VAL_CAGR_rot_approx': cg(rot, '2023-01-01', '2026-10-12'), 'VAL_reported': 0.62937,
                                    'switches_2019_2022': int((np.diff(w[T.wmask(w.index, '2019-01-01', '2022-12-31')]) != 0).sum()), 'switches_reported_SEL': 11}
    meta['regime'] = {'versions': ['RG:' + sp['id'] for sp in REGIME_VERSIONS], 'frozen': 'RG:V5', 'stats': rv}
    L = pd.DataFrame(out)
    meta['sigma_lr_pp'] = {c: lrsd(L[c]) for c in L.columns}
    meta['mean_pp'] = {c: float(L[c].mean()) for c in L.columns}
    meta['t_iid'] = {c: T.tstat(L[c].dropna()) for c in L.columns}
    meta['n'] = {c: int(L[c].notna().sum()) for c in L.columns}
    return L, meta

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--inputs', default='v6'); a = ap.parse_args()
    cells, S, DJ, r50, base, DJW = build(a.inputs)
    chk = cell_shifts(cells, S, DJW, base, r50, np.ones(len(S), bool))
    assert max(abs(x - y) for k in cells for x, y in zip(chk[k], cells[k]['S'])) < 1e-9, '子集平移函式與全期不一致'
    pd.concat({f'{k[0]}|{k[1]}': v for k, v in DJW.items()}, axis=1).to_parquet(T.DATA / ('dJ_windows' + ('' if a.inputs == 'v6' else '_' + a.inputs) + '.parquet'))
    L, lmeta = build_lines(r50, base)
    sfx = '' if a.inputs == 'v6' else '_' + a.inputs
    S.to_parquet(T.DATA / f'cell_series{sfx}.parquet'); L.to_parquet(T.DATA / f'lines_series{sfx}.parquet')
    sig = {c: lrsd(S[c]) for c in S.columns}
    (T.DATA / f'cells{sfx}.json').write_text(json.dumps({'cells': cells, 'dJ_pp': {f'{k[0]}|{k[1]}': v for k, v in DJ.items()}, 'sigma_lr_pp': sig,
                                                        'inputs': a.inputs}, ensure_ascii=False, indent=1), encoding='utf-8')
    (T.DATA / f'lines_meta{sfx}.json').write_text(json.dumps(lmeta, ensure_ascii=False, indent=1, default=T._js), encoding='utf-8')
    print(json.dumps({k: (round(v['S'][0], 3), round(np.mean(v['S']), 3), round(np.std(v['S']), 3)) for k, v in cells.items()}, ensure_ascii=False, indent=0))
    print(json.dumps(lmeta.get('regime_check'), indent=1)); print({k: round(v, 2) for k, v in lmeta['sigma_lr_pp'].items()})

if __name__ == '__main__':
    main()
