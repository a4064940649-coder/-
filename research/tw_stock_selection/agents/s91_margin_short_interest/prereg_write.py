#!/usr/bin/env python3
"""S91 事前登記：寫 PREREG.json＋PREREG.sha256（已存在就拒絕）。python3 -I -B prereg_write.py"""
import sys, json, hashlib, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
H = Path(__file__).resolve().parent
assert not (H / 'PREREG.json').exists(), 'PREREG.json 已存在'
import s91lib as L
assert len(L.CELLS) == 28
dc = json.loads((H / 'data_check.json').read_text())
G = dc['train_returns_by_group']['lag1']
sha = lambda n: hashlib.sha256((H / n).read_bytes()).hexdigest()
PR = dict(
    id='S91', title='散戶槓桿（融資）與放空（融券、借券賣出）資訊能不能幫 SURGE 避開壞交易', written_at=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
    question='訊號日當下的融資／融券／借券賣出餘額，能不能用來擋掉 SURGE 名單裡的壞交易（或擋掉平庸交易把錢放 0050／遞補），使年化在訓練期與驗證期都明顯提高？',
    hypotheses=dict(
        A='散戶槓桿過熱：融資餘額占股本很高或短期暴增＝散戶追高、籌碼凌亂，營收利多已被搶先反映 → 不買（R01～R04、R14）',
        B='沒有人放空＝沒有人氣：訓練期分組看到券資比／融券占比低的交易平庸（沒有大贏家也沒有大輸家）→ 不買（R05～R08）。這個方向是看了訓練期才定的，和「券資比高不買」相反（訓練期券資比高的組平均報酬最高，所以不登記「券資比高不買」）',
        C='借券賣出餘額高或增加＝法人看空；放空或融資餘額相對成交量大＝籌碼沉澱 → 不買（R09～R12）',
        D='組合：散戶 5 日內搶進又沒有人放空（R13）；過熱任一（R14＝R01 或 R02）'),
    engine='正式管線五個開關全開（EXRIGHT_FIX、DEADLINE_FIX、INS15_MASK、REDUCE_FIX、0050 官方含息；寫法同 agents/official_nav/run_official.py 的 E0）；s10lib.simulate_s10 原函式；W10_70、嚴成本、空位 0050',
    acceptance_done_in_prep=dc['acceptance'],
    data=dict(source='raw/tsd_extra：margin_twse／margin_tpex（張）、sbl_twse／sbl_tpex（股）、daily（發行股數、成交股數）', check=dc['data_check']['universe_cell_coverage_all_days'],
              timing='判定用格一律延遲 1 日（訊號日 d 只用到 d−1 的餘額；保守）。凍結格另跑延遲 0 日版（用到 d 當晚公布的餘額；<格>|L0）只作描述、不作判定。公布時點（當晚、次日開盤前）憑記憶、本機無法查證。',
              missing='缺資料列、基期不足 100 張、股本缺值 → 特徵 NaN → 規則一律不擋（保留）'),
    features=L.__doc__,
    rules_text={
        'R01_MUSE8': '融資餘額 > 發行股數 8% 不買。訓練期 20 筆、平均 +0.8%、大贏家 0、大輸家 4、損益占比 −3.2%',
        'R02_MD20_3': '近 20 日融資增加 > 發行股數 3% 不買。訓練期 9 筆、平均 −7.5%、大贏家 0、大輸家 3',
        'R03_MG5_P90': '近 5 日融資增幅在股池最高 10% 不買。訓練期 62 筆、平均 +4.8%（全體 +9.2%）、大贏家 2、大輸家 5',
        'R04_MG20_P90': '近 20 日融資增幅在股池最高 10% 不買（題目原例；訓練期 73 筆、平均 +11.5%、大贏家 4、大輸家 6，分組並不支持，照題目仍登記）',
        'R05_SM_P50': '券資比在股池後半不買。訓練期（含缺值）66 筆、平均 +3.6%、大贏家 1、大輸家 0',
        'R06_SM_ABS2': '券資比 < 2% 不買。訓練期 56 筆、平均 +1.6%、大贏家 0、大輸家 0、損益占比 2.8%',
        'R07_SM_P67': '券資比不在股池最高三分之一就不買。訓練期（含缺值）104 筆、平均 +3.4%、大贏家 2、大輸家 1',
        'R08_SS_P67': '融券餘額占股本不在股池最高三分之一就不買。訓練期（含缺值）83 筆、平均 +1.9%、大贏家 2、大輸家 1',
        'R09_SBLD20_P80': '借券賣出餘額近 20 日增加（占股本）在股池最高 20% 不買。訓練期 53 筆、平均 +9.1%（與全體相當；題目原例方向）',
        'R10_SBL_P80': '借券賣出餘額占股本在股池最高 20% 不買。訓練期 35 筆、平均 +13.3%、大贏家 3（分組不支持，照題目仍登記）',
        'R11_SHORTDV_P33': '（融券＋借券賣出餘額）÷ 20 日均量 不在股池最低三分之一就不買。訓練期 49 筆、平均 +3.2%、大贏家 1',
        'R12_MDV_P33': '融資餘額 ÷ 20 日均量 不在股池最低三分之一就不買。訓練期 93 筆、平均 +5.4%、大贏家 2、損益占比 23%',
        'R13_HOT_NOSHORT': '近 5 日融資增幅在股池最高 10% 且券資比不在最高三分之一 → 不買。訓練期 36 筆、平均 +2.6%、大贏家 1',
        'R14_OVERHEAT': 'R01 或 R02 成立就不買'},
    train_group_evidence=dict(ALL=G['ALL'], note='完整分組見 data_check.json 的 train_returns_by_group（只含訊號 2016-02～2021-12 的 186 筆）；絕對門檻（8%、3%、2%）是在訓練期多個門檻裡挑的，屬資料探勘，靠驗證期把關'),
    modes=dict(Z='被擋名額不遞補，錢照引擎放 0050', F='由 L_MA 第 11～20 名中 RET20 > 10% 且沒被同一規則擋掉者依 SURGE 遞補'),
    cells=list(L.CELLS), n_cells=len(L.CELLS),
    not_registered='「券資比高優先買」（只改買進順序）：名單通常不到 10 檔、資金夠買，順序幾乎不影響結果，不登記。',
    train=dict(window='訊號 2016-02～2021-12（淨值 2016-01-29 → 2021-12-30），全期連續模擬只量訓練窗；20 條路徑＝60／80／100／150／200 萬 × 位移 0～3',
               selection='28 格依訓練窗年化的 20 路徑中位數排序（同分看 ≥ +0.25 路徑數、再看登記順序），取第一名；需同時滿足：中位數 ≥ 現行 + 1.0 個百分點、最大回撤中位數不比現行深超過 2 個百分點、2016～2018 與 2019～2021 兩半的成對差中位數都 > 0，才算「選中」；否則 selected＝不改（仍以第一名跑驗證期，只作描述，總判定不通過）',
               T0='第一名的成對差中位數 ≥ +1.0 個百分點且 ≥ +0.25 的路徑 ≥ 14/20'),
    validate=dict(window='營收月份 ≥ 2021-12（訊號 2022-01 起）重新起跑，淨值 2021-12-30 → 2026-10-22；20 條路徑；只跑一次（results.json 存在就拒絕）',
                  V1='凍結格優於現行的路徑 ≥ 15/20', V2='成對差中位數 ≥ +2.0 個百分點', V3='禁買（現行與凍結格各自最賺 5 筆的聯集）後成對差中位數 > 0',
                  V4='2022、2024-07～2025-04 兩段空頭的段內最大跌幅成對差中位數都不深於 −2 個百分點',
                  overall='selected＝凍結格 且 T0 且 V1～V4 全過 才算通過（同 S25／S70／S88）'),
    descriptive_only=['全期／訓練／驗證的特徵分組表（凍結並驗證後才算）', '每格擋掉的現行交易（大贏家、大輸家、損益最大 10 筆）與凍結格在驗證段擋掉的逐筆明細',
                      '其餘 27 格驗證窗 20 路徑年化差', '凍結格的隨機剔除對照 20 次（seed 91；S88 的做法）', '凍結格延遲 0 日版'],
    trials=dict(new=28, note='另有凍結格延遲 0 日版 1 個（描述用）'),
    code_sha256_at_write={n: sha(n) for n in ('s91lib.py', 's91rules.py', 'train.py', 'validate.py', 'prep.py')},
)
txt = json.dumps(PR, ensure_ascii=False, indent=1, default=str)
(H / 'PREREG.json').write_text(txt)
h = hashlib.sha256(txt.encode()).hexdigest()
(H / 'PREREG.sha256').write_text(f'{h}  PREREG.json\n')
print('PREREG', h)
