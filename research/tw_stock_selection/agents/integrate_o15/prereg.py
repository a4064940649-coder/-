#!/usr/bin/env python3
"""integrate_o15 事前登記（只寫一次）。o15 通過審查的兩題都不產生選股訊號、沒有自己的報酬序列：
  (1) o_g8p2_protocol_refreeze（G8P-2，待觀察）：主策略 G8 前瞻序貫協定 v2（G8_PROTOCOL_v2.json sha256 dcab4dbd…）。
  (2) o_rev_trunc_2009_erratum（ER2，可用）：營收面板 2009-11 截斷的存活者缺口槽位上界（U_opt 162／930＝17.4% > 5% → G4 第一段「未證實」）。
本整合把它們套到主策略（排除金融 n10）上。用法：python3 -I -B prereg.py"""
import json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
ER = AG / 'o_rev_trunc_2009_erratum'
G2 = AG / 'o_g8p2_protocol_refreeze'
G1 = AG / 'o_g8_protocol_fin_erratum'
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()

FILES = [HERE / 'run.py', HERE / 'smoke' / 'smoke.out',
         ER / 'erlib.py', ER / 'ERRATUM2.json', ER / 'ERRATUM2_SPEC.json', ER / 'results.json',
         G2 / 'G8_PROTOCOL_v2.json', G2 / 'FREEZE_MANIFEST_v2.json', G2 / 'results.json',
         G1 / 'G8_PROTOCOL.json', G1 / 'ERRATUM.json', AG / 'integrate_o14' / 'results.json',
         AG / 'main_v2' / 'v2_core.py', AG / 'main_v2' / 'navs_gapfix.parquet', AG / 'independent_audit' / 'ia_core.py',
         AG / 'o_forward_sequential_v2' / 'fslib.py', AG / 'o_forward_sequential_v2' / 'raw/v7/plans_j7.csv',
         AG / 'o_forward_sequential_v2' / 'raw/v7/jscan_end_days.json', AG / 'o_forward_sequential_v2' / 'raw/finlab/data/raw/etf_0050.csv',
         AG / 'o_exec_engine_rewrite_v7x' / 'out' / 'navs_oos_R4.parquet',
         Path('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv')]

PREREG = dict(
    written_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    topic='o15 整合：G8P-2 前瞻協定 v2 ＋ ER2 營收截斷勘誤 × 主策略（52 週高 0.5＋營收 3 月年增 0.5、前 300 池、月換 10 檔、排除金融；以下稱 M）',
    what_is_integrated=('兩題都沒有個股分數也沒有報酬序列，依 S1 本輪不產生新選股訊號。可整合的只有兩塊：'
                        'A 加分項（敏感度）＝把 ER2 的 U 口徑換進 M：2006～2013 訊號日中「存活者偏誤型缺營收」的池內股票給營收百分位 1.0（U_opt，判定口徑）／0.9（U_p90）／沿用最後已知 y3（U_carry），'
                        '其餘照 V2.plan（含開盤漲停遞補）；U_opt_ALL（所有缺值原因都放回）只描述。這就是「把新研究當成主策略的加分項」唯一可行的形式。'
                        'ER2 本身依規格只計數、不算報酬；本塊是整合階段的報酬敏感度，不取代主策略正式數字，也不回寫 ER2。'
                        'C 協定歷史套用＝獨立重寫 G8P-2 選定版本 V1（較差成本決策線＋σ_eff＝max(c·σ_d, s_n)）套到 V7X 可執行歷史線，對照 G8P-2 自報的 HIST_DEMO 與 integrate_o14 的 G8P-1 結果。'),
    B_mix=('不做。理由：(i) 兩題都沒有可當第二條腿的報酬序列；(ii) 主策略×0050 固定比例混合在本輪「不再開」清單；'
           '(iii) integrate_o14 已事前登記並跑過 w＝0.8／0.7／0.5／0.3，全部「不可用」（可執行 100 萬 p10 < 0、V7 紙上 t ≤ 1.67），重跑只會重複同一結果並增加測試數。FINDINGS 直接引用 o14 數字。'),
    legs=dict(
        M='main_v2 v2_core V2(gapfix, exfin=True) plan(10, official)，ia_core.simulate，一般 rates(0.001)、壓力 rates(stress)；C0 必須逐日重現 main_v2/navs_gapfix.parquet EXFIN|n10|{normal,stress}（|差| < 1e-9，否則停）。',
        U='erlib.VU(mode, which)（ER2 的凍結實作，只讀 import）。U_opt／U_p90／U_carry／U_opt_ALL 在 2006～2012、2006～2013 的進榜槽位數必須重現 ERRATUM2.json（119/162、76/105、16/18、151/228），U_opt 槽位清單 sha256 必須等於 a8e5a55c…，否則停。',
        engines='freeze＝正式引擎（換股日無成交的持股凍結、停在最後價值）；dzero＝下市最壞情境（凍結且之後再無成交的持股歸零，ia_core delist_zero=True），只跑一般成本。',
        X='V7X 可執行 navs_oos_R4.parquet B10|Jj|K5T|{normal,stress}|{100萬,300萬,1000萬}|GROW，窗＝plans_j7 AB_V7_B10 各 j 訊號日（fslib.window_excess）。決策線＝逐 J 窗 min(一般, 壓力) 超額（pp），有效 J ≥ 7 取中位數；歷史沒有 REAL_CALIB，adj＝0。',
        E='0050 含息 etf_0050.csv（fslib.e50）。'),
    periods='ALL＝全部訊號日窗（2006-04～2026-09 起點）；SEG1＝2006～2012；SEG2＝2013～2018；SEG3＝2019 起；SEL＝～2019-04；VAL＝2019-05 起；HOLD＝2024 起；另報 Y0609、Y1013、Y0613、Y14ON（描述）。窗依訊號日月份歸期。',
    A_bonus=dict(
        variants=['C0', 'U_opt', 'U_p90', 'U_carry', 'U_opt_ALL'],
        stats='各變體 × {一般 freeze、壓力 freeze、一般 dzero} × 期間：年化（窗幾何）、窗頻與日頻最大回撤、對 0050 窗超額 t、對 C0（同引擎同成本）窗差平均、t 與年化差、與 C0 相關、拿掉最好 3 個日曆年（依各自對 0050 超額）後 t、2006～2013 逐年。'
              '描述：進榜受影響股 vs 被擠出原名單股的訊號日收盤→下一訊號日收盤紙上報酬（含息還原、下市後持平）。',
        decision=('三級（一般成本 freeze 為主口徑）：R1 可忽略＝U_opt 與 U_p90 都 |ALL 年化差| ≤ 0.3pp、|ALL 對 0050 t 差| ≤ 0.05、SEG1 對 0050 幾何超額不變號（門檻沿用 integrate_o14）；'
                  'R2 方向不變＝不符 R1，但 U_opt／U_p90／U_carry 在三種成本×引擎下 SEG1 超額都不變號；R3 可能翻號＝其他。'
                  '任何一級都不改 ER2 的判定：G4 第一段維持「未證實」、G5 第一段「存活者偏誤未排除」（真實營收未知；U 口徑的報酬不是真實反事實報酬的上下界）。'
                  'R1／R2／R3 只決定主策略 FINDINGS 加註的文字。不改 SOP：現行月營收對當下名單是完整的，截斷只影響 2009-12～2013 歷史。')),
    C_protocol=dict(
        rule=('從 G8_PROTOCOL_v2.json params 獨立重寫（不 import g8lib2）：z_n＝S_n／(max(c·σ_d, s_n)·√n)，s_n 為前 n 月樣本 sd（ddof＝1）；'
              '第 12 月：字面 G8（S_12 > 0）→ 下修 z ≤ −c_d → 升級 z ≥ b·√2；第 18 月：下修；第 24 月：下修 → 升級 z ≥ b；先觸發者吸收。'
              'G8P-1 對照照 G8_PROTOCOL.json：σ_eff＝c·σ_d，檢視點 s_n > 1.5σ_d 時一次性改為 max(c·σ_d, s_n)，b 1.679、c_d 0.895，決策線＝一般成本。'),
        lines='V7X 100 萬 GROW 較差成本（決策線）與一般成本（G8P-1 線）；300 萬、1,000 萬只描述（協定證據只涵蓋 100 萬 GROW）。',
        starts='每個錨定月起點一條 24 個月路徑（223 條），高度重疊，只描述不做顯著性；依起點年 2006～12／2013～18／2019～24 與挑選／驗證／保留期起點分報。',
        consistency='一致性：V1 較差成本 100 萬的 S1／FAILED／NO_REJ／字面比例必須與 G8P-2 HIST_DEMO.V1_worse 完全相同；G8P-1 一般成本必須與 HIST_DEMO.G8P1_normal 及 integrate_o14 逐起點相同。不一致 → C 塊標「不一致，待查」，不下任何結論（一票否決）。',
        decision='描述性，不改協定。若 V1 決策線歷史升級比例 < 4.6%（G8P-2 OC 在 α＝+1 的檢定力下限），寫「以歷史實際報酬看，G8P-2 的 24 個月 S1 翻案路徑走不通」。'),
    predictions=('事前預測（寫在看任何變體報酬之前）：'
                 'A：進榜受影響股以併購／股份轉換中的大型股為主（思源、合庫、友尚、建興電、矽品、日月光、乾坤），宣告後價格多貼近收購價，預期報酬低於被擠出的高動能成長股；'
                 'U_opt 的 SEG1 對 0050 幾何超額比 C0（13.17pp）低 1～4pp 但仍 > 0，ALL 年化差 −0.3～−1.5pp、t 差 −0.02～−0.15 → R2；U_carry 差 ≤ 0.3pp；'
                 'dzero 會因下市當時還持有的股份轉換股歸零而再低 1～4pp。'
                 'C：獨立重寫與 HIST_DEMO、o14 完全一致；V1 決策線 S1 0%、FAILED 約 65%；300 萬、1,000 萬更差。'),
    honesty=('寫 PREREG 前我已讀過：ER2 的 results.json、ERRATUM2.json（槽位、受影響股、被擠出股清單）、erlib.py、s4_erratum2.py；G8P-2 的 results.json（含 HIST_DEMO：V1 較差成本線 S1 0%、FAILED 65.5%）、g8lib2.py、s7_hist_demo.py；'
             'integrate_o14 的 PREREG、run.py、FINDINGS 與結果（C0 SEG1 超額 13.17pp、t 1.88；0050 混合全部不可用）；main_v2 FINDINGS 正式數字。'
             '登記前的煙霧測試（smoke/smoke.out，sha256 記於下方）只核對 C0 重現（差 0）、U 槽位數與清單雜湊重現 ER2、窗數 245、各歷史線均值與 sd（後者已在 G8P-2 公開）；'
             '沒有算任何 U 變體的報酬、t 值或協定結果。2006～2026 資料在主策略各研究都已看過，沒有樣本外。'),
    tests='判定用：A 一條三級規則（2 個判定變體 × 3 口徑）；C 描述（一致性一票否決）。',
    sha256={str(f): sha(f) for f in FILES},
)
if __name__ == '__main__':
    RJ = HERE / 'results.json'
    assert not RJ.exists(), 'results.json 已存在：PREREG 只寫一次'
    h = hashlib.sha256(json.dumps(PREREG, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    RJ.write_text(json.dumps(dict(PREREG=PREREG, PREREG_sha256=h), ensure_ascii=False, indent=1), encoding='utf-8')
    print(h, PREREG['written_utc'])
