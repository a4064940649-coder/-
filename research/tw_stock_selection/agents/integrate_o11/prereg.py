#!/usr/bin/env python3
"""integrate_o11 事前登記：o11 唯一通過審查（待觀察）的是 o_g2_feasibility_stoprule（方法題：G2 門檻可達性與研究停損規則），
它不是選股訊號、沒有報酬序列。本整合把它的「產品」（S2 通過機率曲線、2,334 條族群的 G3 門檻、去卷積先驗）套到主策略與
主策略／0050 固定比例混合上。寫 results.json 的 PREREG 與 PREREG_sha256；之後 run.py 只跑一次。python3 -I -B prereg.py"""
import json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
G2 = AG / 'o_g2_feasibility_stoprule'
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()

FILES = [HERE / 'run.py', AG / 'o_v7_listing_exec_engine' / 'navs_jscan_v7.parquet', AG / 'o_v7_listing_exec_engine' / 'navs_v7.parquet',
         AG / 'o_exec_engine_rewrite_v7x' / 'out' / 'navs_oos_R4.parquet', Path('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv'),
         AG / 'o_main_spa_breaks' / 'fam_common.py', AG / 'o_main_spa_breaks' / 'spa_lib.py', AG / 'o_v4_revenue_patch_official' / 'plan_V4.csv',
         G2 / 'results.json', G2 / 'data' / 'family_ir.parquet', G2 / 'data' / 'sim_inputs.npz']

PREREG = dict(
    written_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    topic='o11 整合：o_g2_feasibility_stoprule（以下稱 G2S，方法題）× 主策略（排除金融 n10，AB_V7，以下稱 M）',
    what_is_integrated='G2S 沒有選股訊號也沒有報酬序列，所以「當濾網」「當加分項」「當第二條腿」三種方式全部不適用（照實寫，不另造變體）。'
                       'G2S 能整合進研究流程的是：(i) S2 立題規則（以去卷積收縮後的 IR 查 V1 主變體曲線，G1～G7 聯合通過機率 < 20% 不立題）；'
                       '(ii) 2,334 條族群的 G3 門檻（SPA_c 族群最大值分布 Tb_all、DSR 的族群 SR 變異）；(iii) S1 觸發後建議的模式「真錢 0050＋凍結候選前瞻追蹤」。'
                       '所以本整合測的組合＝M 與 0050 的固定比例混合（S1 觸發後的實際配置選項），並把 (i)(ii) 套到每個組合。',
    legs=dict(
        M_paper='o_v7_listing_exec_engine/navs_jscan_v7.parquet 的 AB_V7|{normal,stress}|J0..J12（理想引擎紙上，13 個 j）；鄰居 AB_V7_B10|normal|Jj。官方日 navs_v7.parquet AB_V7|normal 只作描述。',
        M_exec='o_exec_engine_rewrite_v7x/out/navs_oos_R4.parquet 的 B10|Jj|K5T|{normal,stress}|{100萬,300萬,1000萬}|GROW（V7X 正式可執行版，獲利滾入）。V7X 只有 B10，沒有 AB 可執行版。',
        E='0050 含息（etf_0050.csv，fam_common.e50_nav）。'),
    windows='一律用 G2S 的族群格點：o_main_spa_breaks/fam_common.GRID（AB_V4 訊號日＋2026-10-12），245 個窗。各腿窗報酬＝日淨值在格點上 asof 取值的比值；'
            '淨值尚未開始的窗（J2～J12 的第一窗 2006-04-11）當現金（報酬 0），與 G2S s3_inputs 的 fillna(0) 同慣例。'
            '期間：ALL 2006-04～2026-10、SEG1 2006-04～2012、SEG2 2013～2018、SEG3＝2019 起（＝G2b 的 Y19）、HOLD 2024-01 起（描述用）。',
    combos=dict(
        W_LIST=[1.0, 0.8, 0.5, 0.2],
        MIX='M 比例 w＝0.8（80/20）、0.5（50/50）、0.2（20/80，0050 核心）；w＝1 為 M 單獨對照，0050 單獨為基準。三個比例照任務指示與 integrate_o7～o9 慣例事前寫死，全部報告，不看保留期挑比例。'
            '袖子法（窗頻）：每窗起點為 w／1−w，窗末調回；移轉金額 m 的成本：一般成本 M→0050 為 m×(0.4855%＋0.1855%)，0050→M 為 m×(0.2855%＋0.1855%)（0050 賣方稅 0.1%）；壓力成本雙向 m×1.0%。'
            '可執行混合：M 腿用同一級距的 V7X 可執行淨值（級距＝M 腿資金，帳戶＝級距÷w；腿實際只有帳戶×w，用該級距淨值等於衝擊照級距算，屬中性到偏保守）。',
        FILTER='不適用：G2S 不產生個股分數。', BONUS='不適用：同上。', SECOND_LEG='不適用：G2S 沒有報酬序列；唯一可當第二條腿的是 S1 建議的 0050，即 MIX。'),
    stats='每個 w × 口徑（紙上一般／壓力、紙上 B10、官方日、可執行 一般／壓力 × 3 級距）× 期間：年化（幾何，12/n）、對 0050 幾何年化差、窗頻最大回撤、對 0050 月超額與 t、'
          '與 M 單獨（同口徑同 j）的月差與 t、相關（混合與 M、混合與 0050、混合超額與 M 超額）；ALL 另報拿掉最好 3 個日曆年後 t。全部取 13 個 j 的中位數與 p10（官方日只有 1 條）。',
    s2_and_gates=dict(
        javg='每個組合的 javg 超額＝13 個 j 窗報酬平均減 0050（窗平均；G2S 用日報酬平均。PREREG 前核對：w＝1、可執行 100 萬一般成本時本法 IR 0.051，G2S 0.089，差異來自定義，照實登記，不改）。',
        S2='IR＝javg 超額 IR（年化）；收縮後 IR＝用 G2S 族群（資料夾內去重 832 條、se_iid）重算 NPMLE 先驗的後驗平均（PREREG 前核對：重算先驗的真實 IR ≥ 0.5 比例與 G2S 報告值完全相同 7.5e−106）。'
           '以 G2S FINAL.table_V1 對 IR 線性內插 joint_G1_G7、joint_without_DSR、A、A_plus_G1_G7（IR < 0 取 0）。S2 立題條件＝收縮後可執行 IR 的 joint_G1_G7 ≥ 20%；另報量測紙上 IR（最樂觀）的值。'
           '近似註記：曲線以 AB_V4 的追蹤誤差（年化 20.8%）生成；混合把追蹤誤差縮成約 w 倍，G1、G2、G3、DSR 主要看 IR，影響小；G7 拖累不隨 w 等比縮，屬近似。',
        G1_G8='主欄＝可執行 V7X K5T 100 萬 GROW、13 個 j。G1：ALL 幾何超額 j 中位數與 p10 > 0，且壓力成本中位數 > 0。G2：ALL t 中位數 ≥ 3 且 SEG3 t 中位數 ≥ 2（另列紙上 t）。'
              'G3：javg 的 SPA_c p 下界（只用 G2S 的 Tb_all，PR 核 L=6 標準化）< 0.05 且 DSR（spa_lib.dsr，族群 SR 加候選，N=2,335）≥ 0.95；p 下界 ≥ 0.05 即確定不過。'
              'G4：三段幾何超額中位數都 > 0、拿掉最好 3 年 t 中位數 ≥ 2、鄰居（同 w 的紙上 AB_V7、紙上 B10、相鄰 w 的可執行 100 萬）ALL 幾何超額中位數都 > 0。'
              'G5：沿用 V7／V7X 資料判定，本整合不另驗（列「沿用」）。G6：本整合無獨立複製 → 不過。G7：300 萬、1,000 萬各自 G1（一般成本部分）與 G2 都成立。G8：未開始 → 不過。'),
    decision=dict(
        per_mix='對 w＝0.8、0.5、0.2 逐一判定：(c1) 紙上一般 ALL 對 0050 t 的 j 中位數 ≥ 2.0；(c2) 紙上一般 SEG3（2019 起）幾何超額 j 中位數 > 0；'
                '(c3) 可執行一般成本 3 級距 ALL 幾何超額的 j 中位數與 p10 都 > 0；(c4) 可執行壓力 100 萬 ALL 幾何超額 j 中位數 > 0；(c5) 紙上一般 ALL t 中位數高於 M 單獨。'
                '「可用（建議 SOP 改用此混合）」＝(c1)～(c5) 全過；「待觀察」＝(c1) 過但其他有不過；「不可用」＝(c1) 不過。',
        sop='只有「可用」才建議改 SOP。G1～G8 對每個組合逐條列；G6、G8 必不過，所以任何組合最多「待觀察」，不可能寫「可投資」或「歷史最高標準通過」。',
        g2s='G2S 本身是方法題，判定（可用／不可用）沿用其審查結果，本整合不改判；本整合只回答「把 G2S 的規則套到現有策略與混合後，結論是什麼」。'),
    predictions='研究員事前預測：混合的超額約為 w 倍 M 超額，IR 與 t 幾乎不變但因移轉成本略降：紙上 ALL t（M 約 2.5～2.6）80/20 約 2.5、50/50 約 2.4、20/80 約 2.2；'
                '可執行 100 萬 ALL t 全部 < 0.5；1,000 萬幾何超額 ≤ 0。最大回撤隨 0050 比例增加而縮小（20/80 約 −56～−58%，接近 0050 本身）。'
                '與 M 單獨的月差 t：ALL 為負或接近 0。S2：所有組合（含 M 單獨）收縮後 joint_G1_G7 ≈ 0%，即使用量測紙上 IR 也 < 20%，全部「不立題」。'
                '判定：三個比例都「待觀察」（c1 過，c3、c5 不過）；若 20/80 的 t 掉到 2 以下則為「不可用」。',
    honesty='寫 PREREG 前我已讀過：G2S 的 PREREG、FINAL（含 table_V1、stop_rule、main_strategy_table）、headline 與凍結規格；integrate_o8 的 FINDINGS（V7X 上 0050 混合全部不可用、2019 起全輸 0050）；'
            'integrate_o9 的 prereg；主策略 SOP 與 main_v2 FINDINGS 的正式數字。PREREG 前只做管線核對：各腿在格點上的缺值（J2～J12 第一窗）、'
            '以 w＝1（M 單獨，已知數字）跑 run.py 的煙霧測試（只看缺值數、可執行 javg IR 0.051 與先驗重現），沒有計算任何 w < 1 的混合績效。'
            '2006～2026 全部資料在主策略與 G2S 研究中都已看過，沒有樣本外；本整合的數字全是同一份歷史的重新組合。',
    tests='判定用：3 個比例 × 5 條件。S2 與 G1～G8 為規則套用（逐條列）；其餘描述性。',
    sha256={str(f): sha(f) for f in FILES},
)
if __name__ == '__main__':
    RJ = HERE / 'results.json'
    assert not RJ.exists(), 'results.json 已存在：PREREG 只寫一次'
    h = hashlib.sha256(json.dumps(PREREG, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    RJ.write_text(json.dumps(dict(PREREG=PREREG, PREREG_sha256=h), ensure_ascii=False, indent=1), encoding='utf-8')
    print(h, PREREG['written_utc'])
