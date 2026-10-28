#!/usr/bin/env python3
"""步驟 4：凍結。把收縮模型、先驗、模擬程式與所有輸入的 sha256 寫進 PREREG（results.json），之後才可以產出正式表。
PREREG＝PREREG_DESIGN（不改）＋ 迭代選定的版本 ＋ 雜湊。正式程式（run_posterior、run_alloc、run_oos、robust、crosscheck）開頭都斷言 check_prereg()。"""
import sys, json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sc_lib as S

def main():
    res = S.load_res()
    assert 'PREREG' not in res, 'PREREG 已存在'
    assert S.canon_sha(res['PREREG_DESIGN']) == res['PREREG_DESIGN_sha256'], 'PREREG_DESIGN 被改過'
    it = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
    sel = it['selected']
    files = sorted([p.name for p in HERE.glob('*.py')]) + ['explore.json', 'iterations.json']
    files += ['data/' + p.name for p in sorted((HERE / 'data').glob('*')) if p.suffix in ('.parquet', '.json', '.csv')]
    ext = [S.SPA / 'data' / 'family_excess.parquet', S.SPA / 'data' / 'family_members.csv', S.SPA / 'data' / 'bench.parquet',
           S.SPA / 'fam_common.py', S.SPA / 'spa_lib.py',
           S.AG / 'o_v4_revenue_patch_official' / 'navs_v4.parquet', S.AG / 'o_v4_revenue_patch_official' / 'plan_V4.csv',
           S.AG / 'o_v5_pit_revenue' / 'navs_v5.parquet', S.AG / 'o_v5_pit_revenue' / 'navs_jscan_v5.parquet',
           S.AG / 'o_exec_capacity_tiers' / 'navs_abavg.parquet', S.AG / 'o_exec_capacity_tiers' / 'data' / 'bench.parquet',
           Path('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv')]
    hashes = {f: S.sha(HERE / f) for f in files}
    hashes.update({str(p): S.sha(p) for p in ext})
    P = {'design': res['PREREG_DESIGN'], 'design_sha256': res['PREREG_DESIGN_sha256'],
         'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
         'selected_version': sel, 'iterations_n_versions': len(it['versions']),
         'frozen_choices': {'correlation_adjustment': it['selected_config'], 'G2': 'df 5、c0 0.1（迭代 v3）、自然樣條節點取格點等分位、θ∈[−8,4] 間距 0.02（原登記 [−6,6]，因族群 α̂ 最小 −6.04 於凍結前改，v3 以新格點重驗）', 'HAC_L': 6,
                            'paths': 20000, 'seeds': {'L6': 20261029, 'L12': 20261030, 'alpha': 7}},
         'hashes': hashes}
    res['PREREG'] = P
    res['PREREG_sha256'] = S.canon_sha(P)
    S.RES_F.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG', res['PREREG_sha256'], P['time_utc'], len(hashes), 'files')

if __name__ == '__main__':
    main()
