#!/usr/bin/env python3
"""凍結：PREREG（規格、選定格點、程式與資料 sha256、時戳、研究員點預測）寫入 results.json。之後才跑 s6～s8。"""
import json, datetime
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lclib as X
it = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
ex = json.loads((HERE / 'explore.json').read_text(encoding='utf-8'))
gate = json.loads((X.DATA / 'gate.json').read_text(encoding='utf-8'))
s2 = json.loads((X.DATA / 's2_power.json').read_text(encoding='utf-8'))
FROZEN = """【凍結規格 o_lc_sector_rotation_pit（2026-11-03）】
資料段：D1 閘門未通過（假裝不知道檢核：股數對數誤差中位數 2.14% ≤ 3%、前 100 重疊中位數 95.5，但最小 91 < 93）。
  → 正式段（計入 G2／G4）＝錨定月 2014-01～2026-10（o8 PIT 股數＋4763／2317 修正）；挑選期 SEL＝2014-01～2016-12（36 個月，太短）；驗證期 VAL＝2017-01～2026-10。
  → 2006-04～2013-12 只作描述（2013 以前宇宙改用 120 日均成交值前 100 代理池、權重用除權因子回推的非 PIT 市值），不計入 G2／G4。
宇宙：訊號日 d 收盤 PIT 市值前 100 名上市普通股（排除 ETF、91xx DR），近 6 日有成交、V7 上市身分；無股數的存活股以 ADV120×同期市值/ADV 中位數插補。
合成基準 B：同一宇宙 PIT 市值加權，同一引擎、同樣成本，每個訊號日換回市值權重。
中性：2330 固定 B 權重；金融保險業（證交所產業別或 tcore.FINMASK）固定 B 權重；成員（不含 2330、金融）< 2 家的產業與「未分類」維持 B 權重。
產業：raw/industry_pit.csv（證交所產業別，公開快照 2015-05-22／2016-04-01／2017-07-06／2023-01-16／2026-10 逐段生效；2015-05 以前回溯沿用最早快照；
  2007-07-02 以前電子 8 類合併為電子工業、化學／生技醫療合併、油電燃氣併入其他業；2023-07-03 新類別生效）。
訊號（d 收盤；產業指數＝每月第一個 ≥11 日交易日重設成分、期內市值買進持有、含息、不含 2330、用 d 當時分類套到歷史）：
  REV3＝產業前 100 成員近 3 月營收年增率（V4 已公告值 y3_ab）PIT 市值加權，個股截尾 ±100%；市值覆蓋 < 70% 或錨定月 2013-01～2014-02 → 中性（缺值）。
  RS＝產業指數 12-1、6-1、3-1（GOOD 日 252/126/63 → 21）三者排名百分位平均；H52＝產業指數 ÷ 其 252 日最高；
  NH＝產業內全部上市成員（60 日均成交值 ≥ 1,000 萬、股價 ≥ 10）近 20 個 GOOD 日曾創 52 週新高的占比；COMP＝四者排名百分位等權平均（缺值用其餘）。
建構：可輪動產業依訊號排序（同分比 B 權重）；緩衝帶：已持有且排名 ≤ k+2 續抱，其餘依序補到 k 個；
  前 k 產業各分 A/k 超配，單一產業超配上限 min(A/k, 3×B_s)，超過的再平分給其他前 k（水位法）；總超配 O；
  其餘可輪動產業依 B 權重等比例低配合計 O（O 不超過其 B 權重合計）；產業內依 PIT 市值加權。
選定格（V0、選定規則第一個通過者）：H52｜k=3｜A=15%｜每月。事前預設格（同時報告）：COMP｜k=5｜A=30%｜每月。全部 40 格熱圖照報。
時點：錨定月 m 的營收期限日 D_m（10 日；2019-02、2023-04 為 15 日；遇假日順延）之後第 j 個面板交易日 d，j＝0..12 全報，主欄＝13 個 j 的中位數；d 收盤算訊號、d+1 開盤成交。
紙上成本：tcore.rates（一般：買 0.0855%+0.1%、賣 0.0855%+0.3%+0.1%；壓力：每邊 0.5%）。
可執行：V7X x_exec.py（K5T、f1/f2/f3=0.5/f4）＋lcexec.WSim 權重子類＋0.25% 不交易帶（沿用 o_capmatched_pit_close s6_exec 的帶寬，未調參）；
  級距 30 萬 FIX、100 萬 FIX、100 萬 GROW、300 萬 GROW、1,000 萬 GROW；一般與壓力成本（V7X stress）。
基準：0050 含息（finlab-txd etf_0050.csv adj，同窗 d 收盤 → 下一個 d 收盤）與 B。
安慰劑：1,000 組 AR(1) 隨機產業分數，φ＝2·sin(π·ρ/6)，ρ＝挑選期實際訊號的月對月排名自相關（j 中位數）；同一錨定月所有 j 共用同一組分數；相同 k、A、上限、緩衝帶、頻率。
G3：研究中心族群窗（官方換股日）2014-01 起；族群 (i) 2,334＋40、(ii) F_main 410＋40；Hansen SPA_c（L=6、B=10,000）、DSR（N＝族群數與 Li-Ji 有效數）。"""
PRED = dict(
    P1_SEL='已在探索算出（看過）：COMP IC j 中位數 0.028、t 0.55 → P1 不成立（t<1：資訊假說在挑選期不成立）。',
    P2='驗證期選定格對 B 一般成本 j 中位數月超額 +0.02pp（區間 −0.10～+0.12），IR 0.15；對 0050 IR −0.25（B 本身在 2019 年後因 2330 權重低於 0050 而落後），t −0.7；G2 必不過。'
       '追蹤誤差（對 B）1.5%～2.5%（不是事前寫的 3%～7%：A=15% 的格本來就小）。',
    P3='REV3 的 j0～j4 減 j8～j12 約 0（挑選期 IC 差 0.0004）；高離散度月份超額 > 低離散度（弱）；空頭後 3 個月（2020-04～06、2022-11～2023-01）超額為負。',
    P4='驗證期扣成本前安慰劑百分位約 70（< p95，不成立）。',
    P5='拿掉 2021 航運與 2025 AI 產業月份後，對 B 的 t < 0.5。',
    P6='與主策略可執行版超額相關約 0.2（< 0.5，成立）；在 2019 年後 2330 領漲月份，對 0050 超額的 β 為負（約 −0.3，因為 2330 只拿 B 權重），對 B 超額的 β 約 0。',
    verdict_pred='不可用（G2 不過；待觀察條件「驗證期對 0050 t ≥ 2」不會成立）。')
pre = dict(topic='o_lc_sector_rotation_pit：大型股（PIT 市值前 100）產業輪動，產業營收與產業強度，有上限的產業超低配（台積電與金融中性）',
           time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), frozen_spec=FROZEN,
           D1=dict(pass_=gate['D1_pass'], D1a={k: v for k, v in gate['D1a'].items() if k != 'by_q'}, D1b={k: v for k, v in gate['D1b'].items() if k not in ('worst', 'by_year')},
                   label_2006_2012=gate['label_2006_2012']),
           periods=dict(SEL=('2014-01', '2016-12'), VAL=('2017-01', '2026-10'), FORMAL=('2014-01', '2026-10'), DESC=('2006-04', '2013-12')),
           chosen=dict(version=it['chosen']['version'], cfg=it['chosen']['cfg'], params=it['versions'][it['chosen']['version']]['params'], fallback=it['chosen']['fallback_to_default']),
           default=dict(cfg='COMP|k5|A30|M', params=it['versions']['V0']['params']),
           selection=dict(n_versions=it['n_versions'], n_cells_tried=it['n_cells_tried'], log_len=len(it['selection_log'])),
           SEL_numbers=dict(chosen=it['grid_SEL'][it['chosen']['version']][it['chosen']['cfg']], default=it['grid_SEL']['V0']['COMP|k5|A30|M'],
                            placebo_chosen=[e for e in it['selection_log'] if e.get('b_pass')]),
           placebo_persistence=ex['persistence_jmed'],
           exec=dict(engine='V7X x_exec.py + lcexec.WSim', band=0.0025, final_cfg=dict(f1=True, f2=True, f3=0.5, f4=True),
                     cells=['30萬|FIX', '100萬|FIX', '100萬|GROW', '300萬|GROW', '1000萬|GROW'], costs=['normal', 'stress']),
           code_sha256={f: X.sha(HERE / f) for f in ('lclib.py', 'lcrun.py', 'lcexec.py', 's6_oos.py', 's7_exec.py', 's8_g3.py')},
           data_sha256={'data/shares_full.parquet': X.sha(X.DATA / 'shares_full.parquet'), 'raw/industry_pit.csv': X.sha(X.RAW / 'industry_pit.csv'),
                        'data/gate.json': X.sha(X.DATA / 'gate.json')},
           predictions=PRED, S2=s2['S2_verdict'],
           amendment='PREREG 第 1 版（2026-11-03T03:58:24Z，sha 73f79257…）在 s6 第一個工作就因 lclib.jdays 的終點月 2026-11 不在交易日曆而 IndexError，'
                     '當時尚未計算任何驗證期報酬；只修 jdays 的終點處理（超出日曆 → 面板最後 GOOD 日），規格不變，重新凍結。',
           contamination='2017 年以後的大型股與產業資訊在 o7、o9（C 候選 Brinson：2021 航運、2025 AI）、industry 組都已部分看過，不是全新樣本外；2006～2013 的報酬本題凍結前沒有看過（非 PIT 描述段）。真正沒看過的只有前瞻。')
res = {'PREREG': pre, 'PREREG_sha256': X.canon(pre)}
(HERE / 'results.json').write_text(X.dumps(res), encoding='utf-8')
(X.DATA / 'PREREG.json').write_text(X.dumps(dict(PREREG=pre, sha256=res['PREREG_sha256'])), encoding='utf-8')
print('PREREG', res['PREREG_sha256'], pre['time_utc'], pre['chosen'])
