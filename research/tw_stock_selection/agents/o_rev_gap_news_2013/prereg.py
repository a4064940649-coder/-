#!/usr/bin/env python3
"""PREREG：在 data_freeze_r.json 之後、gate.py 之前寫入 results.json（含 sha256 與時間戳）。"""
import json, hashlib
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
it = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
# 解析器輪次選定規則（事前寫死）：選定代理版本的缺口覆蓋最高者；同分取較早輪
best = max(it['rounds'], key=lambda r: (round(r['G1_cov_selected'], 6), -int(r['iter'][1])))
P = {
 'topic': 'o_rev_gap_news_2013：2013～2021 營收缺口 PIT 新聞代理與首次值擴充',
 'time_utc': pd.Timestamp.utcnow().isoformat(),
 'data_freeze_r_sha256': sha(HERE / 'data_freeze_r.json'),
 'strategy_spec': '主策略 V4 規格完全不改（52 週高點百分位 0.5＋近 3 月營收年增率百分位 0.5、前 300 流動性池、月換 10 檔、排除金融與 KY／DR、處置 trade 口徑、官方換股日、一般／壓力成本）；資料 V4 凍結版。',
 'gap_universe': '2013-01-01～2021-12-31 每個換股訊號日，前 300 池中營收以外都合格（ok_other）但 V4 非 AB 的 y3 缺值的股月；732 格（併購 425／惡化 235／非下市 72；2013～2020 為 719 格，其中非下市 65 格）。',
 'pit_rule': '來源時間戳 ≤ 訊號日 13:30；只有日期者視為當日 23:59（訊號日當天不合法）。季報以法定期限（Q1 5/15、Q2 8/14（2013 年 8/31）、Q3 11/14、年報 3/31）為時間戳。',
 'proxy_versions': ['N3', 'N2', 'N1', 'Q', 'NQ'], 'parser_rounds': ['P1', 'P2', 'P3'],
 'selection_rule': '非缺口格（只用新聞／季報重建）代理對真值的百分位 Spearman ≥ 0.80 的版本中，取缺口覆蓋最高者；同分取較嚴格（N3＞N2＞Q＞NQ＞N1）。解析器輪次取選定版本缺口覆蓋最高者，同分取較早。',
 'selected': {'parser': best['iter'], 'proxy_version': best['selected'], 'iter_G1_cov_seen': best['G1_cov_selected']},
 'G1': '選定版本在 732 個缺口格的覆蓋率 ≥ 60%（另報 2013～2020 的 719 格、B10 入選 133 股月，只供描述）',
 'G2': '2013～2021 年 AB_V4（40 seeds）前 10 名股月中，三個營收月全部有嚴格佐證首次值（≥ 2 個獨立來源一致，或一來源＋git 時間戳）的比例 ≥ 20%；只有最新月有佐證者並報。',
 'decision_tree': {'G1 不過': '維持 B10，主題結案；只交覆蓋報告與描述，不跑任何策略淨值。首次值寫「覆蓋 x%，無法估計」（< 5% 不得寫任何方向）。',
                   'G1 過、G2 不過': '只做缺口代理（AB_V4R、B10R、AB_V4R_N1；一般與壓力成本；附逐 seed、j 0～12、端點、成本兩平、S1～S4、2014 後 t、可執行版、SPA）；首次值寫「覆蓋 x%，無法估計」。',
                   'G1、G2 都過': '另加首次值欄位。'},
 'columns_if_G1_pass': ['AB_V4R', 'B10R', 'AB_V4R_N1'],
 'effective_tests': {'iteration_configs': 15, 'columns_x_cost': 6, 'j_scan': 13, 'main_strategy_revisions': '≥ 7', 'total_at_least': 41,
                     'note': 'G1 不過時第 2～4 項沒有執行，但仍照實列出；SPA 族群沿用 o_main_spa_breaks 的 2,319 條。'},
 'gate_script_sha256': sha(HERE / 'gate.py'), 'firstvalue_script_sha256': sha(HERE / 'firstvalue.py'),
}
canon = json.dumps(P, ensure_ascii=False, sort_keys=True).encode()
F = HERE / 'results.json'
res = json.loads(F.read_text(encoding='utf-8')) if F.exists() else {}
assert 'PREREG' not in res, 'PREREG 已存在，不得覆寫'
res['PREREG'] = P; res['PREREG_sha256'] = hashlib.sha256(canon).hexdigest()
F.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print(res['PREREG_sha256'], P['selected'])
