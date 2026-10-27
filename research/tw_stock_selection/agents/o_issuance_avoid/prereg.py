# -*- coding: utf-8 -*-
"""Write the frozen specification (PREREG) into results.json BEFORE any validation/holdout number is computed."""
import json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
R = HERE / 'results.json'
res = json.loads(R.read_text(encoding='utf-8'))
assert 'EVENTS_FROZEN' in res
assert 'PREREG' not in res, 'PREREG already written'
it = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
assert it['chosen'] == 'V1_both_90d'
spec = {
    'version': 'V1_both_90d',
    'rule': ('主策略（排除金融、gapfix 滾動窗、正式次排序，n10 為主、n30 為輔）在換股日 d 收盤算出名單時，'
             '跳過「d 往前 90 個日曆日內（含 d）已公開的現金增資或國內可轉債事件」的股票，由下一名遞補；'
             '已持有的股票若仍在前 n 名則續抱（不強制賣出）。開盤漲停照原引擎跳過遞補。'),
    'events': {'SEO': 'FSC 申報案件彙總表：上市公司、案件類別以「現金」開頭（不含特別股）、結案＝生效/核減；已知日＝收文日',
               'CB': 'FSC 國內轉換公司債（上市公司）已知日＝收文日；FSC 鏡像缺口期間（2009-08-25~12、2010-07-30~12、2013-11-22~12、2016、2025-07-26 以後）用櫃買 CB 日行情首日（掛牌日）當已知日',
               'not_used': 'ECB、GDR、股票股利（SDIV 只當安慰劑）、交換債、私募',
               'events_sha256': res['EVENTS_FROZEN']['sha256']['events.csv']},
    'known_data_gap_in_validation': 'FSC 現增資料只到 2025-07-25 收文；2025-08 起避開名單只含櫃買 CB 掛牌（及之前收文的現增）。不再補資料、不改規格。',
    'execution': '訊號日收盤算分數，隔日開盤成交；開盤漲停（跳動點）買不到就遞補；處置股排除；停牌凍結（ia_core.simulate）',
    'costs': {'normal': '手續費 0.1425%×0.6、賣出稅 0.3%、每邊滑價 0.1%', 'stress': '一般成本再加每邊 0.3% 滑價'},
    'periods': {'selection': '訊號日 2006-04～2019-03（日頻年化 2006-04-01～2019-04-30）',
                'validation': '2019-05-01～2026-10-12（BRIEF 定義的驗證期；判定用）',
                'validation_sub': {'V_a': '2019-05-01～2023-12-31', 'V_b(保留段)': '2024-01-01～2026-10-12'},
                'cross_cycle': '2006-04-01～2026-10-12'},
    'comparisons': ['原版主策略（同引擎、同成本）配對差', '0050 含息', '同池等權（主策略合格池 EW）'],
    'gates': {'event_level': 'P1 或 P2 在挑選期成立（60 日淨漂移 ≤ −1.5%、分群 t ≤ −2），且 P4 安慰劑不成立',
              'strategy_level': '挑選期配對差 t ≥ 1.5；驗證期配對差 > 0 且 t ≥ 1.0；跨循環年化 ≥ 原版 +0.3pp；最大回撤不深於原版 2pp 以上；剔除最好的 1 年仍 > 0；受影響入選股月份 < 30 時最多待觀察',
              'verdict': '全部通過且事件表第 1／2 層才判可用；事件層級成立但主策略無改善→「機制成立、對主策略無增量」'},
    'selection_period_facts_known_at_freeze': {
        'n10_paired_t_sel': it['results']['V1_both_90d_n10']['paired']['paired_t'],
        'n30_paired_t_sel': it['results']['V1_both_90d_n30']['paired']['paired_t'],
        'event_level': 'SEO 收文 60 日 −0.93%（t_cl −0.79）、CB 掛牌 [−5,+20] −0.75%（t_cl −0.36）：事件層級門檻在挑選期未通過；安慰劑（股票股利）60 日 +2.1%，不是零',
        'winners_curse': '90 日是避開窗格點（30/60/90）的邊緣；n10 鄰近版 60 日 t 0.95、30 日 t 2.36（只影響 8 個入選股月）'},
    'versions_tried': it['n_versions'],
    'oos_reports': ['驗證期與保留段各跑一次；n10、n30；一般與壓力成本', '事件層級樣本外：SEO 收文 [0,2]、[1,60]；CB 掛牌 [−5,20]、[1,60]；SDIV [1,60]（同池等權與 52 週比配對兩種基準）',
                    '穩健性：逐年、受影響股的產業、剔除最大貢獻股、剔除最好的一年'],
}
blob = json.dumps(spec, ensure_ascii=False, sort_keys=True).encode('utf-8')
res['PREREG'] = spec
res['PREREG_sha256'] = hashlib.sha256(blob).hexdigest()
res['PREREG_time_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
res['PREREG_code_sha256'] = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in ['oi_lib.py', 'iterate.py', 'explore.py', 'explore2.py', 'build_events.py', 'build_fsc.py']}
R.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print(res['PREREG_sha256'], res['PREREG_time_utc'])
