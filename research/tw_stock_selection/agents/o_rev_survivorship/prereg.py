#!/usr/bin/env python3
"""事前登記：python3 -I -B prereg.py（在任何策略回測之前執行；之後 run.py 斷言 PREREG 存在、sha256 相符、資料 sha256 相符）。"""
import json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
FZ = json.loads((HERE / 'data_freeze.json').read_text(encoding='utf-8'))
PM = json.loads((HERE / 'price_patch_meta.json').read_text(encoding='utf-8'))
RJ = HERE / 'results.json'
R = json.loads(RJ.read_text(encoding='utf-8')) if RJ.exists() else {}
assert 'PREREG' not in R, 'PREREG 已存在，不可覆寫'
cov = FZ['coverage']
PREREG = {
    'title': 'o_rev_survivorship：營收面板存活者偏誤修補後重算主策略 v2 正式數字',
    'strategy_spec': '完全沿用 main_v2 正式版（gapfix 窗、正式次排序＝營收 3 月年增率高者→代碼小者、跳動點開盤漲停遞補、凍結停牌、'
                     '一般成本＝手續費 0.0855%＋稅 0.3%＋滑價 0.1%、壓力＝買賣各 0.5%、評價到 2026-10-12、2021-03 起營收改用 rev_5y）。不改任何參數。',
    'data': {
        'revenue_versions': {
            'ORIG': '現行 tw_revenue.csv（用 main_v2 的 load_revenue），作為對照；必須重現 main_v2 的 gapfix 正式數字',
            'R_DEL': '主版本：只補國內公司（非 KY／DR）的缺口。現行回測與實盤工具因資料缺漏完全不含 KY／DR，R_DEL 不改變這個股池定義，只修存活者偏誤',
            'R_ALL': '次版本：另補 KY／DR。屬於「股池定義」問題（外國公司從未被納入），單獨報告，不作為正式數字判定依據'},
        'source_priority': FZ['source_priority'], 'fill_rule': '只補原本 NaN 的格子，不覆寫既有值；補進的年增率＝V4（來源內 12 個月前水準自算，缺時用 MOPS 當期公告的去年同月）',
        'freeze': {'timestamp': FZ['timestamp'], 'R_DEL_sha256': FZ['outputs']['R_DEL']['sha256'], 'R_ALL_sha256': FZ['outputs']['R_ALL']['sha256'],
                   'src_taiwanan_sha256': FZ['src_taiwanan_sha256'], 'price_patch_sha256': PM['sha256']},
        'gate_result': {'consistency_level': FZ['level_agree_0.5pct'], 'consistency_yoy_V4': [r for r in FZ['yoy_iterations'] if r['version'] == FZ['chosen']][0]['agree_rate_0.5pp'],
                        'consistency_pass': True,
                        'coverage_domestic_2013_2026': cov['domestic_all_2013_2026']['coverage'], 'coverage_domestic_2015_04_2019_05': cov['domestic_2015_04_2019_05']['coverage'],
                        'coverage_pass': cov['domestic_all_2013_2026']['coverage'] >= 0.8,
                        'note': '覆蓋率 < 80%：資料關卡未通過，判定改以界限分析為準（見 verdict_rule）'}},
    'versions_to_run': {
        'ORIG': 'ORIG', 'R_DEL': 'R_DEL', 'R_ALL': 'R_ALL',
        'B05_DEL': 'R_DEL ＋ 仍缺營收的國內池內股票營收百分位固定 0.5', 'B10_DEL': 'R_DEL ＋ 同上固定 1.0（最壞：假設都是最高營收）',
        'B05_ALL': 'R_ALL ＋ 仍缺營收的國內＋KY 池內股票固定 0.5', 'B10_ALL': 'R_ALL ＋ 固定 1.0',
        'PX_ORIG': 'ORIG ＋ 2019-05-17 起下市股價格修補', 'PX_DEL': 'R_DEL ＋ 價格修補（「營收＋價格都修補」版）'},
    'bound_rule': '缺營收的股票＝gap_stockmonths_filled.csv 中 2013-01 起、修補後仍未補到者；只要營收以外條件（前 300、52 週有值、非處置、隔日有開盤）都合格，就放回合格池，'
                  '52 週比百分位在擴大後的池內重算，營收百分位：有營收者在有營收者之間排，缺營收者固定為 p；p=0 視同排除＝R_DEL／R_ALL 本身。排除金融版另排除金融股。',
    'universes': ['全市場 n10', '全市場 n30', '排除金融 n10', '排除金融 n30'], 'modes': ['BASE', 'AB（rev_announce 期限日曆＋產業延後機率，Monte Carlo 40 次，seed 0～39）'],
    'periods': {'S1': '2006-04～2013-12', 'S2': '2014～2020', 'S3': '2021-03 訊號日～2026-10-12', 'S4': '2024-01～2026-10-12', 'P1': '2006-04～2020-12', 'ALL': '2006-04～2026-10-12'},
    'sample_split': {'selection': '訊號日 ≤ 2019-04-30（探索 explore.py 只用這段的描述統計，不跑淨值）', 'validation': '2019-05～2023-12', 'holdout': '2024-01～2026-10-12',
                     'note': '本題沒有參數可選，所以樣本外只是「同一份規格重算」；修補只影響 2015-04～2020 的訊號（來源 A 只有 2015-01～2019-08），驗證期與保留期的變化只來自持股路徑'},
    'stats': '年化、最大回撤（日頻）；t 對 0050 與對同池等權（正式＝訊號窗，附表＝日曆月窗）；Calmar；逐年；單股貢獻；與 ORIG 的差',
    'expectation_rule': '實盤預期＝排除金融＋AB、正式次排序、一般成本；預期年化＝同期 0050 年化＋0.5×（回測年化−0050 年化）；另報壓力成本',
    'verdict_metric': '排除金融 n10、BASE、一般成本、ALL 期間：年化下降幅度（ORIG − 版本，pp）與訊號窗對 0050 t',
    'verdict_rule': {
        'levels': {'(1)': '下降 ≤ 1.5pp 且 t ≥ 2.3：正式數字維持，加註已修補', '(2)': '下降 1.5～3pp 或 t 在 2.0～2.3：正式數字改用修補版，實盤預期同步下修',
                   '(3)': '下降 > 3pp 或 t < 2.0：跨循環顯著降為邊緣，SOP 預期段落改寫，下一輪優先研究 P5 避開規則'},
        'which_version': '資料關卡未通過（覆蓋率 < 80%），所以判定取 R_DEL 與 B05_DEL 兩者中較差的等級；B10_DEL 是極端最壞情況，另報但不單獨決定等級。'
                         '若 R_DEL 與 B05_DEL 都在 (1) 但 B10_DEL 落在 (3)，結論加註「最壞情況下會降為邊緣」。R_ALL／B*_ALL 只作為「是否該把 KY 納入」的獨立建議。',
        'mechanism': 'P3：補進的國內後來下市股，持有期報酬是否低於同月其他入選股（事件 t 按換股日分群）；支持＝方向一致且 R_DEL 的 S2 降幅最大'},
    'predictions': {
        'P1': '2013～2018 每個換股日池內缺營收、後來下市約 15～30 檔；補齊後 n10 入選名單每年約 1～4 個名額改變',
        'P2': '排除金融 n10 跨循環年化變化 ±1.5pp 以內，對 0050 t ≥ 2.3；下降 > 3pp 或 t < 2.0 即否定現行正式數字',
        'P3': '新進名單的下市公司持有期報酬低於同月其他入選股（左尾），S2 年化下降最多',
        'P4': '同池等權年化下降；對等權的 t 不降或略升',
        'P5': '下市前 6～12 個月營收年增率 > 100% 的公司，在入選股中的比例高於在池中的比例（成立也只能留到下一輪事前登記）',
        'P6': 'o_prepriced_growth、o_capacity_tsmc_gap 的結論不因修補改變'},
    'p6_spec': 'o_prepriced_growth：修補只加 2015-01 以後的格子，它的挑選期（2006～2013）與閘門在建構上完全不變（程式斷言）；'
               'o_capacity_tsmc_gap：用它自己的 ocore（只把 V.load_revenue 換成 R_DEL）重算 U300 w0、U300 w0.3、U51_300 w0 的 SEL／VAL／HOLD／ALL 關鍵數字',
    'multiple_testing': '策略規格 0 個自由參數；資料建構 4 版（只用重疊期一致率挑，不看報酬）；結果版本 9 個全部報告',
    'robustness': '分年、分產業（入選股產業分布差）、去掉最大貢獻股',
}
body = json.dumps(PREREG, ensure_ascii=False, sort_keys=True)
R['PREREG'] = PREREG
R['PREREG_sha256'] = hashlib.sha256(body.encode('utf-8')).hexdigest()
R['PREREG_time_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
R['iterations_json_sha256'] = hashlib.sha256((HERE / 'iterations.json').read_bytes()).hexdigest()
R['explore_json_sha256'] = hashlib.sha256((HERE / 'explore.json').read_bytes()).hexdigest()
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
print(R['PREREG_sha256'], R['PREREG_time_utc'])
