#!/usr/bin/env python3
"""本題共用：族群讀取（只讀 import o_main_spa_breaks 的 fam_common／spa_lib）、IR／t、期間遮罩、NPMLE。"""
import json, hashlib, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
SPA_DIR = AG / 'o_main_spa_breaks'
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/g2fs')
SCR.mkdir(parents=True, exist_ok=True)
DATA = HERE / 'data'; DATA.mkdir(exist_ok=True)
RES = HERE / 'results.json'

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

FC = _load('fam_common_ro', SPA_DIR / 'fam_common.py')
SP = _load('spa_lib_ro', SPA_DIR / 'spa_lib.py')
WIN = FC.WIN
E50 = FC.e50_nav()
R50 = pd.Series(FC.win_ret(E50).to_numpy(), index=WIN)

PER = {'ALL': ('2006-04-01', '2026-10-12'), 'Y19': ('2019-01-01', '2026-10-12'),
       'SEG1': ('2006-04-01', '2012-12-31'), 'SEG2': ('2013-01-01', '2018-12-31'), 'SEG3': ('2019-01-01', '2026-10-12')}
def pmask(name, idx=WIN):
    a, b = PER[name]; return (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))

def ir(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 6: return np.nan
    s = x.std(ddof=1); return float(x.mean() / s * np.sqrt(12)) if s > 0 else np.nan

def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 6: return np.nan
    s = x.std(ddof=1); return float(x.mean() / s * np.sqrt(len(x))) if s > 0 else np.nan

def geo_ex(rs, rb):
    """幾何年化差（%/年）。rs、rb：同長窗報酬。"""
    rs = np.asarray(rs, float); rb = np.asarray(rb, float); ok = np.isfinite(rs) & np.isfinite(rb)
    n = ok.sum()
    if n < 3: return np.nan
    return float((np.prod(1 + rs[ok]) ** (12 / n) - np.prod(1 + rb[ok]) ** (12 / n)) * 100)

def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def canon(o): return hashlib.sha256(json.dumps(o, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def load_res(): return json.loads(RES.read_text(encoding='utf-8'))
def save_key(k, v):
    r = load_res(); assert 'PREREG' in r; r[k] = v
    RES.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
def check_prereg(need2=False):
    r = load_res()
    assert 'PREREG' in r and canon(r['PREREG']) == r['PREREG_sha256'], 'PREREG 缺或被改'
    if need2:
        assert 'PREREG2' in r and canon(r['PREREG2']) == r['PREREG2_sha256'], 'PREREG2 缺或被改'
        for rel, h in r['PREREG2']['hashes'].items():
            p = Path(rel) if rel.startswith('/') else HERE / rel
            assert sha(p) == h, f'{rel} 與 PREREG2 雜湊不符'
    return r

def family():
    """回傳 (X: 245×N 超額 DataFrame（未補值）, meta DataFrame[member_id, folder, src])。F_long 2,319＋o10 上櫃 15＝2,334。"""
    D = pd.read_parquet(SPA_DIR / 'data' / 'family_excess.parquet')
    MM = pd.read_csv(SPA_DIR / 'data' / 'family_members.csv')
    mm = MM[MM.F_long & MM.cov_W_long]
    X = D[mm.member_id.tolist()].copy()
    meta = mm[['member_id', 'folder', 'F_main']].assign(src='F_long').reset_index(drop=True)
    nav = pd.read_parquet(AG / 'o_tpex_pool_extension' / 'out' / 'navs_oos.parquet')
    names = ['M', 'R1', 'R2', 'BASE', 'IT_v1_ALL300_n10', 'IT_v2_ALL400_n10', 'IT_v5_ALL400_n10_floor2k', 'IT_v6_ALL400_n15', 'IT_v7_TP200_n10',
             'NB_K300', 'NB_K500', 'NB_floor0', 'NB_floor2k', 'NB_n15', 'NB_n5']
    add = {}
    for nm in names:
        r = pd.concat([nav[f'{nm}|J{j}|100萬|GROW|normal'].pct_change() for j in range(13)], axis=1).mean(axis=1)
        nv = (1 + r.fillna(0)).cumprod()
        add['OTPX:' + nm] = FC.win_ret(nv).to_numpy() - R50.to_numpy()
    A = pd.DataFrame(add, index=WIN)
    X = pd.concat([X, A], axis=1)
    meta = pd.concat([meta, pd.DataFrame({'member_id': A.columns, 'folder': 'o_tpex_pool_extension', 'F_main': False, 'src': 'o10_tpex'})], ignore_index=True)
    return X, meta

def v7x_paths(cell='100萬', cost='normal', run='R4'):
    """V7X 可執行 13 個 j 的日淨值（B10、K5T、GROW）。"""
    nav = pd.read_parquet(AG / 'o_exec_engine_rewrite_v7x' / 'out' / f'navs_oos_{run}.parquet')
    return {j: nav[f'B10|J{j}|K5T|{cost}|{cell}|GROW'] for j in range(13)}

def javg_nav(paths):
    r = pd.concat([p.pct_change() for p in paths.values()], axis=1).mean(axis=1)
    return (1 + r.fillna(0)).cumprod()

def win_ex(nav):
    return pd.Series(FC.win_ret(nav).to_numpy() - R50.to_numpy(), index=WIN)

def npmle(x, se, grid, iters=5000, tol=1e-10):
    x = np.asarray(x, float); se = np.asarray(se, float)
    L = np.exp(-0.5 * ((x[:, None] - grid[None, :]) / se[:, None]) ** 2) / se[:, None]
    w = np.full(len(grid), 1.0 / len(grid)); ll0 = -np.inf
    for it in range(iters):
        f = L @ w
        w = w * (L / f[:, None]).mean(axis=0)
        ll = np.log(L @ w).sum()
        if ll - ll0 < tol: break
        ll0 = ll
    return w, float(ll), it

def posterior(x0, se0, grid, w):
    p = w * np.exp(-0.5 * ((x0 - grid) / se0) ** 2); p = p / p.sum()
    cdf = np.cumsum(p)
    q = lambda a: float(grid[np.searchsorted(cdf, a)])
    return {'x_obs': float(x0), 'se': float(se0), 'post_mean': float((p * grid).sum()), 'post_median': q(0.5), 'ci90': [q(0.05), q(0.95)],
            'P_gt_0.3': float(p[grid > 0.3].sum()), 'P_ge_0.5': float(p[grid >= 0.5 - 1e-9].sum()), 'P_ge_0.66': float(p[grid >= 0.66 - 1e-9].sum())}
