#!/usr/bin/env python3
"""步驟 0a：把本題全部輸入複製到 raw/（之後所有計算只讀 raw/ 與 data/，不讀其他資料夾、不讀 scratchpad）。
輸出：raw/<子目錄>/<檔名>、raw/manifest.json（來源路徑、sha256、大小、複製時間）。
用法：python3 -I -B s0_copy_raw.py
"""
import hashlib, json, shutil, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
AG = HERE.parent
RAW = HERE / 'raw'

SRC = {
    'spa': [AG / 'o_main_spa_breaks/data/family_excess.parquet', AG / 'o_main_spa_breaks/data/family_members.csv',
            AG / 'o_main_spa_breaks/data/bench.parquet', AG / 'o_main_spa_breaks/data/tsmc_inputs.parquet',
            AG / 'o_v4_revenue_patch_official/plan_V4.csv'],
    'o6shrink': [AG / 'o_shrink_core_satellite/sc_lib.py', AG / 'o_shrink_core_satellite/run_posterior.py',
                 AG / 'o_shrink_core_satellite/run_alloc.py', AG / 'o_shrink_core_satellite/build_families.py',
                 AG / 'o_shrink_core_satellite/review_exec_o6.py',
                 AG / 'o_shrink_core_satellite/data/clusters_k161.csv', AG / 'o_shrink_core_satellite/data/families.json',
                 AG / 'o_shrink_core_satellite/data/focus_windows.parquet', AG / 'o_shrink_core_satellite/data/focus_meta.json',
                 AG / 'o_shrink_core_satellite/results.json'],
    'navs': [AG / 'o_v4_revenue_patch_official/navs_v4.parquet', AG / 'o_v6_price_panel_official/navs_v6.parquet',
             AG / 'o_v6_price_panel_official/navs_jscan_v6.parquet', AG / 'o_exec_capacity_tiers/navs_abavg.parquet',
             AG / 'integrate_o6/navs.parquet', Path('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv')],
    'exec': [AG / 'o_exec_capacity_tiers/ecore.py', AG / 'o_exec_capacity_tiers/raw/cpi_tw.json'] +
            [AG / 'o_exec_capacity_tiers/data' / f for f in ('bench.parquet', 'feat_daily.parquet', 'model_spec.json', 'plan_v4.csv',
                                                            'plans_ab.csv', 'prevclose.parquet', 'px_close.parquet', 'px_grow.parquet',
                                                            'px_open.parquet', 'px_traded.parquet', 'xrules.json',
                                                            'zero_impact_seed_navs.parquet')],
    'newstrat': [AG / 'o_capmatched_selection/navs_oos.parquet', AG / 'o_capmatched_selection/iterations.json',
                 AG / 'o_capmatched_selection/results.json', AG / 'o_concentration_regime/data/W_full.parquet',
                 AG / 'o_concentration_regime/iterations.json', AG / 'o_concentration_regime/results.json'],
    'live': [AG / 'live_tracking/o7_capmatched/SPEC.json', AG / 'live_tracking/o7_regime/registration.json'],
}
RENAME = {('navs', 'navs.parquet'): 'integrate_o6_navs.parquet', ('o6shrink', 'results.json'): 'o6_results.json',
          ('newstrat', 'iterations.json'): None, ('newstrat', 'results.json'): None}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    man = {}
    for sub, files in SRC.items():
        d = RAW / sub; d.mkdir(parents=True, exist_ok=True)
        for f in files:
            name = f.name
            if sub == 'newstrat' and name in ('iterations.json', 'results.json', 'navs_oos.parquet', 'W_full.parquet'):
                pre = 'capmatched_' if 'capmatched' in str(f) else 'regime_'
                name = pre + name
            elif sub == 'live':
                name = f.parent.name + '_' + name
            elif (sub, name) in RENAME and RENAME[(sub, name)]:
                name = RENAME[(sub, name)]
            dst = d / name
            shutil.copy2(f, dst)
            assert sha(f) == sha(dst)
            man[f'{sub}/{name}'] = {'source': str(f), 'sha256': sha(dst), 'bytes': dst.stat().st_size}
    out = {'copied_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'files': man,
           'note': '原始檔唯讀複製；raw/ 視為不可信輸入，所有讀取以 python3 -I 進行。'}
    (RAW / 'manifest.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(len(man), 'files copied')

if __name__ == '__main__':
    main()
