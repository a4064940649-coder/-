"""凍結規格：寫入 results.json 的 PREREG（含 sha256 與時間戳）。只能在 iterate.py 之後、任何樣本外計算之前執行一次。"""
import os, json, hashlib, datetime as dt
HERE = os.path.dirname(os.path.abspath(__file__))
it = json.load(open(os.path.join(HERE, 'iterations.json')))
assert it['selected'] == 'V6'
sel = [v for v in it['versions'] if v['v'] == 'V6'][0]
SPEC = {
 'name': '指數刪除股收盤承接（o_index_deletion_flow）凍結版 V6',
 'events': '凍結事件表 events_v2.csv（sha256 b19b78f2d36d2a7de29e78d99aa8ced1fed24fc5eae7c309b72cf6a6aa705759）中的刪除事件：MSCI 標準指數刪除（含轉入小型）、MSCI 小型指數刪除（不含升級到標準）、臺灣50 刪除、高股息 ETF（0056、00878、00919、00713、00918、00929、00940）刪除。',
 'universe_filters': '上市股（面板內）；E−1 的 60 日均成交值 ≥ NT$1,000 萬且收盤 ≥ 10 元；E 日有成交；E 日收盤漲停（≥ 前收 ×(1+6.5%／9.5%)，2015-06-01 為界）不買；E 日落在 2024-02-15～2025-09-23（src 3 近似收盤）整筆排除。',
 'pressure': 'pressure = K_class ÷ E−1 的 20 日均成交值；K_class（NT$）：MSCI 標準刪除 5 億、臺灣50 刪除 3 億、高股息 ETF 刪除 3 億、MSCI 小型刪除 0.5 億。',
 'trade_group': f'只交易高賣壓組：pressure ≥ {sel["cuts"][0]:.4f}（挑選期刪除事件的中位數，固定不再更新）；低賣壓組＝pressure < 同一切點（只用來檢定劑量反應）。',
 'entry_exit': '生效日 E 收盤集合競價買進（價格＝官方收盤），持有 20 個交易日，E+20 收盤賣出；遇停牌以最後成交價計。',
 'benchmark': '上市等權：同一窗（E 收盤→E+20 收盤）全體上市股買進持有報酬的等權平均。事件報酬＝個股報酬 − 上市等權。',
 'costs': '一般來回 0.5%；壓力來回 1%。',
 'stats': '扣成本後淨平均、中位數、按事件月分群 t、剔除前 5% 後平均、勝率；高減低＝同事件月兩組平均差的 t。',
 'portfolio': '10 槽固定金額：每筆＝部位淨值 ÷ 10；同時最多 10 筆，滿槽跳過；同日多筆依 pressure 由高到低；單筆 ≤ E−1 的 20 日均成交值 × 5%；閒置資金放 0050（買賣 0050 手續費 0.0855%、賣出另加證交稅 0.1%）；事件買進成本 0.1%、賣出 0.4%（合計 0.5%）。',
 'periods': {'select': '2008-01～2018-12（已用）', 'validate': '2019-01-01～2023-12-31', 'holdout': '2025-09-24～2026-10-22（E+20 在資料內者）', 'final_judgement_window': '驗證＋保留合併 2019～2026（排除 src 3）'},
 'decision': {
   '可用': '(i) 高賣壓組淨平均 > 0、月分群 t ≥ 2、中位數 > 0、剔除前 5% 後 > 0；(ii) 高減低 > 0 且 t ≥ 1.5；(iii) 2019～2020 與 2021 以後兩段高賣壓組淨平均同號（皆 > 0）；(iv) 與主策略（EXFIN n10 gapfix）及處置部位（disposal_reconcile A_0050）的月報酬相關 < 0.3；(v) 主策略 70%＋處置 20%＋刪除承接 10% 對 0050 t 不低於主策略 80%＋處置 20%，最大回撤不加深（重疊期 2021-10～2026-10）；另報主策略 90%＋刪除 10% 對主策略 100%（2010-11～2026-10）。',
   '待觀察': '只過 (i) 沒過 (ii)：紙上追蹤 2027-05、2027-11 兩次 MSCI 調整。',
   '不可用': '(i) 不成立，或高賣壓組驗證期淨報酬 ≤ 0（P5）。'},
 'P4': {'rule': '主策略 EXFIN n10 gapfix 正式名單（main_v2 v2_core.V2(gapfix, exfin=True).plan(10)）；持有期 [進場日 e_k, 下個進場日 e_{k+1}) 內，持股若遇 MSCI 標準／小型或臺灣50 納入的生效日 E（e_k ≤ E < e_{k+1}−1），計算 E 收盤 → e_{k+1} 開盤的報酬，減去同窗其他持股平均（配對差）。E 在 src 3 期間者主檢定排除。',
        'modified_strategy': '遇納入就在 E 收盤賣出，資金放 0050 到下個換股日照常換股；額外成本＝該股權重 × 0.5%。',
        'test_period': '2006～2026 全期（實際有事件為 2010-11 起）', 'adopt_if': '配對差平均 < −1%、月分群 t ≤ −2、2006～2015 與 2016～2026 兩段皆 < 0；三者全部成立才加入 SOP。'},
 'n_versions_tried': it['n_versions'],
}
txt = json.dumps(SPEC, ensure_ascii=False, sort_keys=True)
res_path = os.path.join(HERE, 'results.json'); res = json.load(open(res_path))
assert 'PREREG' not in res, 'PREREG 已存在，不可重寫'
res['PREREG'] = {'spec': SPEC, 'sha256': hashlib.sha256(txt.encode()).hexdigest(), 'timestamp_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}
json.dump(res, open(res_path, 'w'), ensure_ascii=False, indent=1)
print(res['PREREG']['sha256'], res['PREREG']['timestamp_utc'])
