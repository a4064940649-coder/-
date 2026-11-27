#!/usr/bin/env python3
"""S106：合併題目 A（results_ipo.json，由 s106_ipo.py 產生）與題目 B（處置股事件盤點，只整理既有結論、不新增回測）→ results.json。
用法：python3 -I -B s106_ipo.py && python3 -I -B s106_build_results.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
A = 'research/tw_stock_selection/agents/'

DISPOSAL_INVENTORY = {
    'scope': '只整理既有 FINDINGS／results 的結論，未新增任何試驗；數字逐字取自下列檔案。',
    'headline_claim_events': {
        'src': A + 'events/FINDINGS.md',
        'claim': '721 筆可成交，扣大盤 +3.62%/事件、t 6.6；挑選期 2021-10～2023 扣大盤 +3.11%（t 2.79）、驗證期 +3.76%（t 6.06）；逐年皆正',
        'caveat': 'disposal_reconcile 對帳：events 的 t 用毛額（未扣成本）；挑選期「2.79」是 t_date（同日分群，百餘群），不是 t_month（26 個月）。同一毛額樣本 t_month 1.58；扣 0.8% 成本後 1.01。所有定義下挑選期 t_month 毛額 1.2～1.8、淨額 0.7～1.2，沒有一個 ≥ 2；只有 2024 起的驗證期 3～6。',
        'src_caveat': A + 'disposal_reconcile/FINDINGS.md',
    },
    'timeline': [
        {'study': 'events（第 1 輪）', 'verdict': '主研究員整理為「獨立重現成立」', 'key': '+3.62% 扣大盤、t 6.6（毛額、事件層）'},
        {'study': 'disposal_reconcile', 'verdict': '待觀察', 'key': '正式版（實盤可做）挑選期 113 筆淨 +2.52%、t_month 1.14；驗證期 454 筆 +2.85%、t 3.92；去前 5% 後挑選期 +0.88%（t 0.58）、2026 年 +0.26%。10 槽閒置持現金年化 32.4%、t 0.46 不勝 0050；閒置持 0050 才 49.7%（t 1.83）——超額大半是 0050 帶來的。'},
        {'study': 'disposal_bear_longhistory', 'verdict': '不可用（長歷史否定）', 'key': '2005～2021-09 近似事件淨 +0.35%、中位數 −1.84%、t_month −0.89；前 10 名貢獻 81%；7% 漲跌幅期 −0.81%。'},
        {'study': 'disposal_x_main', 'verdict': '不可用', 'key': '處置×主策略前 100 名在 2006～2021 t_month 1.09，只在 2021 後成立。'},
        {'study': 'attention_disposal_mechanism', 'verdict': '機制不成立', 'key': '6 日漲幅×月份配對後，處置組 vs 未處置組差 +2.56%、t 1.16，驗證期 −0.51%：溢酬不是來自處置制度本身；B4 處置解除後續抱為負。'},
        {'study': 'disposal_v2', 'verdict': '待觀察、上限 20%', 'key': '挑選期 81 筆 +3.07%、t 1.74；驗證期 t 3.51；2026-08-10 制度大改（處置 10→5 日、撮合 2 分鐘），新制 3 個月 t 0.35 無法檢定；與主策略 80/20 對 0050 不顯著，貢獻是降回撤不是加報酬。'},
        {'study': 'o_disposal_official_fullhist（o9）', 'verdict': '待觀察→研究中心定「長歷史不成立，真錢上限 20%→0」', 'key': '官方名單 2006-04～2021-09 淨 +2.43%（t 2.37）但全部來自 2014 後；2006～2013 −0.79%；可執行 V7X 成本 +1.71%（t 1.50）、壓力 +0.84%（t 0.51）；G2/G3/G4/G7（容量）不通過；每槽 > 30 萬衝擊吃掉大半溢酬；與主策略月超額相關 0.40。'},
        {'study': 'o_disposal_d0_exec_close（o13）', 'verdict': '不可用，依 K1 結案，SOP 刪除處置配置、不再開變體', 'key': '逐筆仍 +1.70%，但組合層（最後一盤成交、零股、嚴格預收＋T+2 現金帳）驗證期對 0050 月 t −0.16；公告只領先 1 個交易日，賣 0050 的 T+2 款來不及 → 需約 26% 現金池，現金拖累吃掉溢酬；最寬鬆口徑 t 0.78 仍 < 1；與主策略可執行相關 0.52（2019 起 0.67）。'},
        {'study': 'o_disposal_rejudge（o3）', 'verdict': '主策略照舊排除處置股', 'key': '主策略選股層面「不排除−排除」挑選期 t 0.83。'},
        {'study': 'S14 s14_corporate_events（已獨立複製）', 'verdict': '不加進現行策略', 'key': '處置單獨 20/20 條贏 0050（2019 起中位數 36.87%）但 2024 起只 5/20；兩段空頭全比 0050 深；SURGE 一半＋處置一半 vs 一半 0050：2019～2023 +4.33、2024 起 −4.71（只靠一段）；對現行 W10_70 0/20。複製員：優勢靠「再處置就延長」撐住，不延長變不確定；2024 起未延長的第一次處置扣成本 −1.41%（t −2.03）。'},
        {'study': 'S11 s11_pead_second_signal', 'verdict': '不加（注意：S11 是季報 PEAD，不是處置股）', 'key': 'S99 inventory 把 S11／S14／S15／S85 歸成「第二套策略各半合併／閒置資金改放」一列，判「不通過」，理由：冷月與 SURGE 相關 0.58～0.70，SURGE 輸的月份候選也輸。'},
        {'study': 'S38 s38_risk_menu', 'verdict': '不涉處置股', 'key': 'S38 是衛星/0050/定存比例的風險選擇表，FINDINGS 中沒有處置股內容；降回撤最便宜是把 0050 換部分定存。'},
    ],
    'why_not_adopted_ranked': [
        '1. 驗證期以外不成立（最主要）：所有定義下挑選期 2021-10～2023 淨 t_month < 2；2006～2013 長歷史為負或 0；溢酬集中 2014 後、尤其 2020／2021／2026 少數年份與右尾事件（去前 5% 後幾近 0）。',
        '2. 可執行性／資金結構：公告只領先 1 個交易日，處置股要預收款券，賣 0050 的 T+2 款來不及 → 必須常備約 26% 現金池；最後一盤集合競價與零股成交。把這些放進去後組合層驗證期對 0050 t −0.16（o13 K1 結案）。',
        '3. 與主策略衝突：不是持股重疊（重疊僅 5%），而是報酬同漲同跌——可執行月超額相關 0.52～0.67、冷月相關 0.58～0.70，SURGE 輸的月份處置也輸，合併沒有分散效果；S14 各半合併對現行 0/20 條較好。',
        '4. 容量：o9 G7 不通過；每槽 > 30 萬衝擊吃掉大半溢酬，1,000 萬壓力成本下約 0。對一般帳戶不是第一限制。',
        '5. 制度變更：2026-08-10 起處置 5 日、2 分鐘撮合，舊樣本對新制的代表性未知；新制樣本 3 個月 t 0.35～0.40。',
    ],
    'independent_replication': '有。events→disposal_reconcile 四組逐筆對帳可完全重現；o9 有 replicate_o9、o13 有 replicate_o13；S14 由不同子代理獨立複製「事前判定全部一致」。重現的是「事件層近年為正」，不是「可投資」。',
    'biggest_current_limit': '缺乾淨樣本外：2015～2026 處置資料已被多題看過；舊制 2006～2013 為負；唯一乾淨的證據只能來自 2026-08-10 新制起的前瞻紙上帳本（BACKLOG：新制滿 12 個月 2027-08 重評）。現況：真錢上限 0、只做紙上。',
}


def main():
    ipo = json.loads((HERE / 'results_ipo.json').read_text(encoding='utf-8'))
    res = {'agent': 'S106', 'topic_A_ipo_lottery': ipo, 'topic_B_disposal_inventory': DISPOSAL_INVENTORY}
    (HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print('wrote results.json')


if __name__ == '__main__':
    main()
