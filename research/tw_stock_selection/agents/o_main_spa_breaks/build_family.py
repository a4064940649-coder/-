#!/usr/bin/env python3
"""步驟 1：族群蒐集。掃描 agents/ 下所有淨值 parquet（只讀），依事前寫死的規則分類欄位，
在主策略訊號窗格點上取窗報酬，寫出：
  data/members_all.csv      所有掃到的欄位與分類（納入／排除＋理由）
  data/family_ret.parquet   納入成員的窗報酬（T=245 × N），一般成本口徑（該研究自己的一般成本）
  data/bench.parquet        0050 含息窗報酬、主策略同池等權（AB_V4|EW）窗報酬
規則（PREREG 之前寫死；不依 p 值修改）：
  納入：曾被當作候選策略報告過的多頭規則淨值（網路規則、組合、配置、改版、穩健性版本）。
  排除：純基準（0050、同池等權、加權指數、單純持有 2330）、事後拆解（DIAG、placebo、shadow）、
        明說只當描述的序列（空方 SHORT）、扣成本前（gross／zero）與壓力成本版本（每條規則只留一般成本）、
        複製資料夾（replicate_*，除非原資料夾沒有淨值檔）、本輪（o5）同時進行尚未報告的資料夾、
        非淨值檔（個股面板、事件表）。
  完全相同的窗報酬序列（最大差 < 1e-10）只留一條。
"""
import os, re, json, glob, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fam_common as C

ROOT = C.AG
OUT = C.DATA; OUT.mkdir(exist_ok=True)
SKIP_FOLDERS = {'o_main_spa_breaks': '本題', 'o_v5_pit_revenue': '本輪 o5 同時進行，凍結時尚未報告',
                'o_rev_event_trigger': '本輪 o5 同時進行，凍結時尚未報告', 'value_fix': '資料修正紀錄，非淨值'}
BENCH_WHITELIST_EW = {'concentration_equity', 'w4_historical_price_band'}   # 這兩個資料夾的 EWk 是策略（前 k 名等權）

BENCH_RE = [r'^(EW|ew)(\d*$|[_|])', r'(\||_)EW$', r'^(0050|E0050|E50|A2330|EW300_ref|EW|EW4)$', r'^POOL_EW', r'^BM_',
            r'same_pool', r'EW_pool', r'^EW_', r'_EW$']
EXCL_RE = [
    (r'(?i)diag', '事後拆解（DIAG）'),
    (r'(?i)placebo', '安慰劑對照（描述用）'),
    (r'(?i)shadow', '影子等權（描述用）'),
    (r'SHORT|(^|[_|])short([_|]|$)', '空方序列'),
    (r'(?i)gross|(^|[_|])zero($|[_|])|nocost|slip0\.0000', '扣成本前'),
    (r'(?i)nodisc|fullfee', '非一般成本（手續費無折扣）'),
    (r'(?i)stress', '壓力成本'),
    (r'slip0\.00(0[5-9]|1[1-9]|[2-9])|slip0\.003|slipstress|(^|[|_])0\.003($|[|_])|(^|[|_])0\.002($|[|_])', '非一般滑價'),
    (r'(?i)flat|cap10m|cap\d+m$|plus03', '非一般成本（容量或平坦成本）'),
    (r'cost0\.0(0[89]|1\d|2)|cost1\.2|(^|[_|])1\.2$', '非一般成本（處置成本敏感度）'),
]

def is_stock_panel(df):
    cs = [str(c) for c in df.columns]
    return sum(bool(re.fullmatch(r'\d{4}[A-Z]?', c)) for c in cs) > 0.5 * len(cs)

def classify(folder, col):
    c = str(col)
    for pat, why in EXCL_RE:
        if re.search(pat, c): return False, why
    if folder not in BENCH_WHITELIST_EW:
        for pat in BENCH_RE:
            if re.search(pat, c): return False, '純基準'
    else:
        if re.search(r'^(EW_K10|EW_K5|EW_K10cap15)$', c) and folder == 'w4_historical_price_band': return False, '純基準'
    if c in ('0050', 'E0050', 'E50', 'A2330', 'EW_TAIEX', 'EW_0050'): return False, '純基準'
    return True, ''

MAIN_FOLDERS_ALL = {'main_refine', 'main_v2', 'o_infra_v3_rebuild', 'o_v4_revenue_patch_official', 'integrate_o1', 'integrate_o2',
                    'integrate_o4', 'independent_audit', 'revenue_longhistory', 'rev_announce', 'o_hold_buffer_fresh', 'o_stagger_tranche',
                    'o_disposal_rejudge', 'o_msci_inclusion_avoid', 'o_issuance_avoid', 'o_buyback_gate_core_rebal', 'breadth_derisk',
                    'manual_screen', 'tpex_main', 'o_rev_survivorship', 'o_capacity_tsmc_gap', 'concentration_equity', 'o_sbl_foreign_level'}
def is_main(folder, file, col):
    c = str(col)
    if folder in MAIN_FOLDERS_ALL: return True
    if folder == 'exec_rebuild': return 'main_nav' in file
    if folder == 'o_largecap_leg': return not c.startswith('L_ALONE')
    if folder == 'turnover_buffer': return c.startswith('B_MAIN')
    if folder == 'exit_rules': return c.startswith('main')
    if folder == 'combo_search': return c.startswith('C1+C2|')
    if folder == 'o_cb_board_runup': return 'oos_navs' in file
    if folder == 'disposal_v2': return c.startswith('combo_n10')
    if folder == 'strategy_ensemble': return c in ('N10', 'N30')
    if folder == 'w3_revenue_sue': return 'cross_nav' in file
    return False

def main():
    files = sorted(glob.glob(str(ROOT / '**' / '*.parquet'), recursive=True))
    # 去重時保留「正式來源」：主策略正式版資料夾優先掃描
    PRI = ['o_v4_revenue_patch_official', 'main_v2', 'o_infra_v3_rebuild', 'main_refine', 'revenue_longhistory', 'momentum_family']
    files.sort(key=lambda f: (PRI.index(os.path.relpath(f, ROOT).split('/')[0]) if os.path.relpath(f, ROOT).split('/')[0] in PRI else len(PRI), f))
    has_orig = {}
    for f in files:
        rel = os.path.relpath(f, ROOT); top = rel.split('/')[0]
        if '/replicate' not in rel and '/raw/' not in rel and 'yoy_' not in rel:
            has_orig.setdefault(top, []).append(rel)
    e50 = C.e50_nav()
    r50 = C.win_ret(e50)
    rows = []; series = {}
    for f in files:
        rel = os.path.relpath(f, ROOT); top = rel.split('/')[0]
        base = dict(folder=top, file=rel)
        if top in SKIP_FOLDERS:
            rows.append(dict(base, col='*', include=False, reason=SKIP_FOLDERS[top])); continue
        if '/raw/' in rel:
            rows.append(dict(base, col='*', include=False, reason='raw 原始資料')); continue
        if '/replicate' in rel:
            orig_nav = [x for x in has_orig.get(top, []) if x.endswith('.parquet')]
            # 原資料夾有自己的淨值檔 → 複製版重複，排除
            if orig_nav:
                rows.append(dict(base, col='*', include=False, reason='複製資料夾（原資料夾已有淨值）')); continue
        try:
            df = pd.read_parquet(f)
        except Exception as ex:
            rows.append(dict(base, col='*', include=False, reason=f'讀取失敗 {ex}'[:80])); continue
        if not isinstance(df.index, pd.DatetimeIndex):
            rows.append(dict(base, col='*', include=False, reason='非日頻淨值（事件表或面板）')); continue
        if is_stock_panel(df):
            rows.append(dict(base, col='*', include=False, reason='個股面板')); continue
        if 'search_monthly' in rel:
            rows.append(dict(base, col='*', include=False, reason='與 search_nav 同內容（月報酬版）')); continue
        for c in df.columns:
            ok, why = classify(top, c)
            mid = f'{rel}::{c}'
            r = dict(base, col=str(c), include=ok, reason=why, member_id=mid)
            if ok and str(c) in ('npos', 'r0050', 'n_pos', 'exposure'):
                ok = False; r.update(include=False, reason='非淨值欄（持股數、基準報酬）')
            if ok:
                s = pd.to_numeric(df[c], errors='coerce')
                v = s.dropna()
                if len(v) and (v < 0).any() and v.abs().max() < 1:
                    s = (1 + s.fillna(0)).cumprod(); r['converted'] = '日報酬→淨值'
                if s.notna().sum() < 40:
                    r.update(include=False, reason='有效值太少')
                else:
                    wr = C.win_ret(s)
                    if wr.notna().sum() < 12:
                        r.update(include=False, reason='有效窗 < 12')
                    elif (wr.abs() > 3).any():
                        r.update(include=False, reason='窗報酬絕對值 > 300%（資料異常）')
                    else:
                        series[mid] = wr
                        r['F_main'] = is_main(top, rel, c)
            rows.append(r)
    R = pd.DataFrame(series)
    # 完全相同的序列去重（依掃描順序保留第一條）
    keep = []; seen = {}
    X = R.to_numpy()
    for j, mid in enumerate(R.columns):
        x = X[:, j]; key = (int(np.isfinite(x).sum()), round(float(np.nansum(x)), 8))
        dup = None
        for k in seen.get(key, []):
            y = X[:, k]
            if np.array_equal(np.isfinite(x), np.isfinite(y)) and np.nanmax(np.abs(x - y)) < 1e-10:
                dup = R.columns[k]; break
        if dup is None:
            seen.setdefault(key, []).append(j); keep.append(mid)
        else:
            for r in rows:
                if r.get('member_id') == mid: r.update(include=False, reason=f'與 {dup} 完全相同')
    R = R[keep]
    M = pd.DataFrame(rows)
    M.to_csv(OUT / 'members_all.csv', index=False)
    R.to_parquet(OUT / 'family_ret.parquet')
    ew = pd.read_parquet(C.V4DIR / 'navs_v4.parquet')['AB_V4|EW']
    B = pd.DataFrame({'r0050': r50, 'rEW_ABV4': C.win_ret(ew)})
    B.to_parquet(OUT / 'bench.parquet')
    inc = M[M.include == True]
    print('scanned cols', len(M), 'included', len(inc), 'matrix', R.shape)
    print(M[M.include != True].reason.value_counts().head(30))
    print(inc.groupby('folder').size().sort_values(ascending=False).head(60).to_string())

if __name__ == '__main__':
    main()
