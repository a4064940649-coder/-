#!/usr/bin/env python3
"""凍結閘門協定：寫 data/GATE_SPEC.json（時間戳、建構程式 sha、選定版本、checkpoint 清單、門檻（照 o12 D1 原值）、ADV 插補規則），
並把 GATE_SPEC 的 sha256 寫進 results.json['PREREG']。之後 s4_gate.py 必須先斷言 GATE_SPEC 存在且 sha 相符才計算閘門統計。"""
import sys, json, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import plib as P
it = json.loads((P.HERE / 'iterations.json').read_text(encoding='utf-8'))
cps = [str(P.IDX[P.ti(q)].date()) for q in pd.date_range('2006-03-31', '2013-12-31', freq='QE')]
spec = dict(
    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    topic='2006～2013 PIT 股數與證交所產業別快照：資料重建與 D1／D2 閘門檢核（不回測任何報酬）',
    code_sha={f: P.sha(P.HERE / f) for f in ('plib.py', 'pbuild.py', 's1_inventory.py', 's2_iterate.py', 's3_freeze.py')},
    raw_manifest_sha256=P.sha(P.RAW / 'manifest.json'),
    construction=dict(version=it['selected'], versions_tried=list(it['versions']), truth_priority='同交易日：tejm＞tejq＞twq＞gq＞ga＞o8',
                      par='2013 年以前一律 10 元', events='pbuild.event_multipliers（開盤跳空估計因子；o12 同規則，不讀 ret）',
                      extrapolation='錨點外用事件乘數外推；沒有任何錨點 → ADV 插補',
                      adv_impute='市值＝ADV120×（同期有股數股 ADV 名次 30～200 的 市值/ADV 中位數）（o12 同規則）'),
    D1a_2006=dict(checkpoints=cps, rule='每個檢核日：拿掉所有股票、所有來源中日期距檢核日 ≤ 20 曆日的錨點，用 ±520 交易日內剩餘錨點重建；和檢核日錨點真值比。'
                  '真值前 100（存活上市普通股；沒有真值錨點者兩邊都用全錨點建構值，沒值則 ADV 插補）vs 留一前 100。',
                  thresholds=dict(pooled_med_abs_log_le=0.03, overlap_median_ge=95, overlap_min_ge=93)),
    D1c=dict(rule='2006-01～2013-12 每月最後交易日，全錨點建構的市值前 100 中，ADV 插補（沒有任何股數）的檔數', thresholds=dict(mean_le=1.0, max_from_2008_eq=0)),
    D2=dict(rule='主策略前 300 流動性池（每月第一個日 ≥ 11 的交易日；60 日均成交值 ≥ 1,000 萬、收盤 ≥ 10、前 300）2006-01～2013-12 股月；'
                 '有 dated 快照（evidence A/B，快照內有該股）距離 ≤ 24 個月（731 天）者為「dated」，否則「無法判定」。2007-07-02 改制只當描述（strict 口徑不採計）。',
            thresholds=dict(dated_share_ge=0.98, undeterminable_le=0.02),
            fin_flag='PIT 金融旗標＝最近（先取 ≤ d 的最晚快照，否則 24 個月內最早的後續快照）快照產業＝金融保險業；主策略旗標＝現行 stock_list 金融保險業 ∪ 28xx ∪ 5876、5880；報不一致的股票與股月數（P4：≤ 10 檔）'),
    D1_retro=dict(rule='o12 D1(a) 原協定：2021-03-31 錨點（o8 股數＋4763/2317 修正）只靠事件乘數回推到 2014-01，28 個季末和 o8 真值比；'
                       '路徑 B 版事件＝TWT49U 官方前收／參考價（權、權息；權息扣本股 ±3 年純息殖利率中位數）＋面板 flag 3 參考價重設（開盤跳空估計）；另報拆解：(i) 2021-03-31 沒有錨點的股票改用真值後的重疊（上限）。',
                  threshold_P1=dict(overlap_min_ge=93)),
    pass_rule='D1a_2006、D1c、D2 全過 → 2006～2013 標「近似 PIT（D1 通過）」並輸出 data/shares_pit_2006_2013.parquet、industry_pit_2006_2013.csv；否則只存 raw/ 與 data/gate.json，標「2006～2013 仍非 PIT」。',
    P5='臺灣50 2006～2013 官方審核名單找不到 → 不做',
    one_fix_rule='若發現純實作錯誤，只准修一次並揭露修正方向。')
txt = json.dumps(spec, ensure_ascii=False, indent=1, sort_keys=True)
(P.DATA).mkdir(exist_ok=True)
(P.DATA / 'GATE_SPEC.json').write_text(txt, encoding='utf-8')
h = P.sha(P.DATA / 'GATE_SPEC.json')
res_p = P.HERE / 'results.json'
res = json.loads(res_p.read_text(encoding='utf-8')) if res_p.exists() else {}
res['PREREG'] = dict(file='data/GATE_SPEC.json', sha256=h, created_utc=spec['created_utc'], selected_version=it['selected'])
res_p.write_text(P.dumps(res), encoding='utf-8')
print('GATE_SPEC sha256', h)
