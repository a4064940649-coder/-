#!/usr/bin/env python3
"""讀 raw/tsd_extra 面板的小工具（研究程式 import 用；只讀）。

  from tsd_extra_load import load, load_revenue_file
  df = load('inst_twse', '2015-01-01', '2026-12-31', columns=['date', 'code', 'foreign', 'trust'])
  bs = load('bs_hist')                      # 季資料、事件類不給日期就是全部
  hold = load('tdcc_summary', '2020-01-01', '2026-12-31')   # 集保彙總（週資料；2025 以前一年一檔、2026 起一月一檔，一樣用日期挑）
  old = load('inst_twse_early')             # 2015 年以前的資料另開 *_early 資料集，欄位與 2015 後相同，可直接 pd.concat
  txt = load_revenue_file('data/mops/revenue_hist/2026-10_twse.csv')            # 首見版原檔文字
  txt = load_revenue_file('data/mops/revenue_hist/2026-10_twse.csv', 'latest')  # 最新版
資料集名稱與欄位說明看同資料夾的 manifest.json。
"""
import os, gzip, glob
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))


def load(name, start=None, end=None, columns=None, root=ROOT):
    """把一個資料集的分檔接起來。start／end（含）用 'YYYY-MM-DD'（每日／每週類）或 '2015Q1'、'2015'（季／年類），只用來挑檔與過濾。"""
    files = sorted(glob.glob(os.path.join(root, name, '[0-9][0-9][0-9][0-9]', '*.parquet')))
    if not files:
        raise FileNotFoundError('沒有資料集 %s' % name)
    pick = []
    for f in files:
        stem = os.path.basename(f)[:-8]            # 2015-01、2015-01-05 或 2015
        if start and stem < str(start)[:len(stem)]:
            continue
        if end and stem > str(end)[:len(stem)]:
            continue
        pick.append(f)
    if not pick:
        return pd.read_parquet(files[0], columns=columns).iloc[0:0]
    df = pd.concat([pd.read_parquet(f, columns=columns) for f in pick], ignore_index=True)
    key = 'date' if 'date' in df.columns else ('period' if 'period' in df.columns else None)
    if key and (start or end):
        k = df[key].astype(str)
        if start:
            df = df[k >= str(start)]
        if end:
            df = df[k.loc[df.index] <= str(end) + '￿']
    return df.reset_index(drop=True)


def load_revenue_file(source_path, version='first', root=ROOT):
    """回傳月營收來源檔的文字。version='first'＝本快照第一次看到的版本；'latest'＝最新看到的版本。"""
    idx = {}
    with open(os.path.join(root, 'revenue_raw', '_index.tsv'), encoding='utf-8') as f:
        for ln in f:
            if not ln.startswith('#'):
                r = ln.rstrip('\n').split('\t'); idx[r[0]] = r
    r = idx[source_path]
    rel = source_path[len('data/'):]
    if version == 'first' or r[1] == r[4]:
        p = os.path.join(root, 'revenue_raw', 'first', rel + '.gz')
    else:
        p = os.path.join(root, 'revenue_raw', 'rev', rel + '.' + r[4][:12] + '.gz')
    with gzip.open(p, 'rb') as g:
        return g.read().decode('utf-8-sig')
