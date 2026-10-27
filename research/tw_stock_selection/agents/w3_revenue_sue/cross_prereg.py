#!/usr/bin/env python3
"""交叉運用（SURGE × 主策略）事前設定。看任何交叉結果前執行一次：python3 -I cross_prereg.py → results_cross.json 的 PREREG。"""
import json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
    '寫於看結果前': True,
    '寫入時間': datetime.datetime.now().isoformat(timespec='seconds'),
    '主策略（對照組）': 'revenue_longhistory/lhlib.py 的 M2，參數不改：60 日均值前 300 名（≥1,000 萬、股價 ≥10、非處置）、0.5×52 週高點百分位 + 0.5×近 3 月營收年增率平均百分位、前 30／前 10 等權；每月第一個 ≥11 日交易日收盤訊號、隔日開盤成交、開盤漲停（2015-06 前 7%、後 10%）且未持有者跳過遞補；長期月營收檔。',
    'SURGE 訊號': '與 w3_revenue_sue/prereg.py 主版本 S_main 完全相同（D_t = Rev_t − Rev_{t−12}；μ、σ = D_{t−1..t−24}，24 個都要有；2013 IFRS 跨口徑 D 設缺值 → 營收月 2013-01～2015-12 無訊號）。訊號日 d 用 d 所在月前一月的 SURGE（同主策略營收腿）。',
    '缺 SURGE 處理（事前定好）': {
        '整月回退': '若某換股日主策略合格池（el & 可成交）中有 SURGE 的檔數 < 100，該月交叉策略 = 主策略原樣（同一份持股）。預期回退月：2006-04～2008-01、2013-02～2016-01 訊號日。',
        '(a) 個股缺值': '前 60 名內缺 SURGE 者排在有 SURGE 者之後（內部依主分數）。',
        '(b) 個股缺值': 'SURGE 百分位設 0.5（中性）。',
    },
    '交叉方式（全部報告）': {
        '(a) 第二層篩選': '主分數排序取前 60（合格且 d+1 可成交），依 SURGE 由大到小重排，取前 30（a30）／前 10（a10）；開盤漲停且未持有者跳過，往下遞補（先遞補前 60 內，前 60 用完再依主分數往下）。對照主策略 n30／n10。',
        '(b) 排名合成': '分數 = (52 週高點百分位 + 近 3 月營收年增率百分位 + SURGE 百分位) / 3，百分位都在主策略合格池內計算（rank pct）；取前 30（b30）／前 10（b10），M2 規則。對照主策略 n30／n10。',
        '(c) 狀態切換': 'SURGE 是個股層級（橫截面）訊號，規則本身沒有市場狀態定義 → 略過，不自行發明市場層級版本。',
    },
    '成本': 'lhlib.cost_rates：主 = 滑價 0.1%；另報壓力（買賣各 0.5%）。',
    '期間（資料可回到 2006）': {
        '挑選期': '2006-04～2013-12', '驗證期': '2014-01～2020-12', '保留期': '2021-03～2026-10',
        '另報': ['各期只算 SURGE 有效月（非回退月）', 'A 段 2008-02～2013-01 訊號（SURGE 個體口徑、無存活者偏誤）'],
    },
    '統計': '日頻 NAV 年化、最大回撤、逐年；配對 t = 換股日收盤→下一換股日收盤的月窗（交叉 − 主策略同 n），每月一個觀察值；另報對 0050、對同池等權月窗 t。',
    '判定（滑價 0.1%）': '對每個交叉組合（a30 vs n30、a10 vs n10、b30 vs n30、b10 vs n10）：驗證期與保留期都「年化 > 主策略且配對 t ≥ 2」→ 該組合可用。'
                      '整體：至少一個組合可用 → 可用（並註明 4 個組合的多重比較）；只有一段符合、或兩段都勝但 t < 2 → 待觀察；其餘 → 不可用。挑選期結果只報告。',
    '診斷（不參與判定）': ['SURGE 用 ifrs_skip 版（跳過 2013 缺值月）補滿 2014-01～2015-12 營收月', '與主策略持股重疊率', '我方 SURGE 與 w3 info_S_main 入選股一致性檢查'],
}
out = {'PREREG': PREREG, 'PREREG_sha256': hashlib.sha256(json.dumps(PREREG, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
f = HERE / 'results_cross.json'
assert not f.exists(), 'results_cross.json 已存在，不重寫 PREREG'
f.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(out['PREREG_sha256'])
