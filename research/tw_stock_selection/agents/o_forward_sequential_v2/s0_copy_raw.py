#!/usr/bin/env python3
"""步驟 0：把所有輸入複製進本題資料夾（原檔只讀、不改），並記錄 sha256。
- vendor_o7/：o_tier_shrink_sequential 的 tlib.py、hmethods.py、seqlib.py、s2c_cv.py（只當輸入；import 這份副本），
  以及 tlib 讀取的資料子集（vendor_o7/raw/...，維持 tlib 的相對路徑）。
- raw/v7/：o_v7_listing_exec_engine 的 navs_exec7、navs_jscan_v7、navs_v7、plan_V7*.csv、plans_j7.csv、jscan_end_days.json、model_spec7.json。
- raw/integrate_o7/navs.parquet（混合淨值）。
- raw/screener、raw/finlab：git clone（sparse，--depth 1）的 tw-stock-screener 與 finlab-txd（只當資料，python -I 讀）。
用法：python3 -I -B s0_copy_raw.py
"""
import json, hashlib, shutil, subprocess, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
AG = HERE.parent
RAW, VEN = HERE / 'raw', HERE / 'vendor_o7'
OT = AG / 'o_tier_shrink_sequential'
V7 = AG / 'o_v7_listing_exec_engine'

COPY = {
    # vendor：程式
    VEN / 'tlib.py': OT / 'tlib.py',
    VEN / 'hmethods.py': OT / 'hmethods.py',
    VEN / 'seqlib.py': OT / 'seqlib.py',
    VEN / 's2c_cv.py': OT / 's2c_cv.py',
    # vendor：tlib 需要的資料（相對路徑 raw/...）
    VEN / 'raw/spa/family_excess.parquet': OT / 'raw/spa/family_excess.parquet',
    VEN / 'raw/spa/family_members.csv': OT / 'raw/spa/family_members.csv',
    VEN / 'raw/spa/plan_V4.csv': OT / 'raw/spa/plan_V4.csv',
    VEN / 'raw/o6shrink/clusters_k161.csv': OT / 'raw/o6shrink/clusters_k161.csv',
    VEN / 'raw/navs/etf_0050.csv': OT / 'raw/navs/etf_0050.csv',
    VEN / 'o_tier_results.json': OT / 'results.json',
    # V7
    RAW / 'v7/navs_exec7.parquet': V7 / 'navs_exec7.parquet',
    RAW / 'v7/navs_jscan_v7.parquet': V7 / 'navs_jscan_v7.parquet',
    RAW / 'v7/navs_v7.parquet': V7 / 'navs_v7.parquet',
    RAW / 'v7/plan_V7.csv': V7 / 'plan_V7.csv',
    RAW / 'v7/plan_V7_B10.csv': V7 / 'plan_V7_B10.csv',
    RAW / 'v7/plans_j7.csv': V7 / 'data/plans_j7.csv',
    RAW / 'v7/plans_ab7.csv': V7 / 'data/plans_ab7.csv',
    RAW / 'v7/jscan_end_days.json': V7 / 'data/jscan_end_days.json',
    RAW / 'v7/model_spec7.json': V7 / 'data/model_spec7.json',
    RAW / 'v7/results_v7.json': V7 / 'results.json',
    # 整合
    RAW / 'integrate_o7/navs.parquet': AG / 'integrate_o7/navs.parquet',
    RAW / 'integrate_o7/results.json': AG / 'integrate_o7/results.json',
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    man = {'copied_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'files': {}, 'clones': {}}
    for dst, src in COPY.items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        man['files'][str(dst.relative_to(HERE))] = {'source': str(src), 'sha256': sha(dst), 'bytes': dst.stat().st_size}
    for name in ('screener', 'finlab'):
        d = RAW / name
        if d.exists():
            h = subprocess.run(['git', '-C', str(d), 'log', '-1', '--format=%H %cI'], capture_output=True, text=True).stdout.strip()
            url = subprocess.run(['git', '-C', str(d), 'remote', 'get-url', 'origin'], capture_output=True, text=True).stdout.strip()
            man['clones'][name] = {'url': url, 'head': h}
    for rel in ('screener/data/history.csv.gz', 'screener/data/stock_list.csv', 'screener/data/extras/rev_hist.csv.gz',
                'screener/data/extras/rev_5y.csv.gz', 'screener/data/extras/disposal_5y.csv.gz', 'finlab/data/raw/etf_0050.csv'):
        p = RAW / rel
        if p.exists(): man['files']['raw/' + rel] = {'source': 'git clone', 'sha256': sha(p), 'bytes': p.stat().st_size}
    (RAW / 'manifest.json').write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: v['sha256'][:12] for k, v in man['files'].items()}, ensure_ascii=False, indent=0))


if __name__ == '__main__':
    main()
