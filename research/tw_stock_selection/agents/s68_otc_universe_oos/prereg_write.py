#!/usr/bin/env python3
"""S68 事前登記：寫 SPEC.json、PREREG.json 與 sha256（已存在就拒寫）。在 inventory.py、prep.py 之後、run.py 之前執行。
寫的當下只看過：資料盤點（不含報酬）、現行只上市 CUR 的驗收數字、OTC／ALL 的「檔數」。沒有看過任何上櫃或合併股池的報酬（S16 的舊口徑結果是已知背景，列在 prior_knowledge）。"""
import sys, json, hashlib, datetime
from pathlib import Path
H = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for f in ('SPEC.json', 'PREREG.json'):
    assert not (H / f).exists(), f'{f} 已存在，不可覆寫'
chk = json.loads((H / 'prep_check.json').read_text()); inv = json.loads((H / 'data_inventory.json').read_text())
assert chk['A0']['tol_ok'] and chk['A0']['bitwise_vs_deadline_ALL'] and chk['A2']['ok'] and chk['A3_all_ok']
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
SPEC = dict(
    id='S68', title='上櫃獨立股池：凍結的現行規則在沒用來調參數的股池上的樣本外檢驗', written_at=now,
    frozen_rules=dict(
        signal='STD SURGE：D＝月營收(m)−月營收(m−12)；(D−μ)÷σ，μ、σ 用 D(m−1…m−24)，σ>0 才有值（s17lib.rev_family STD）',
        entry='F1：股池內、SURGE 有值、還原收盤 > 60 日均線（L_MA）依 SURGE 取前 10 名，其中 20 日報酬 > 10% 者買進（s10lib.make_signal_s10 depth=10, mom_lo=0.10, nsel=10）',
        exit='E0：訊號日掉出當月 L_MA 前 20 名（或掉出股池）→ 次日開盤賣',
        timing='A10：法定申報期限（每月 10 日，例外依 DEADLINE_FIX）後第一個交易日的下一個交易日收盤算訊號，次日開盤成交；INS15_MASK 開',
        allocation='W10_70：0050 底倉 30%（±10 個百分點帶狀拉回）＋衛星 70%，每檔＝衛星 ÷ 10，最多 10 檔，空位買 0050',
        cost='嚴成本 COSTS["strict"]（手續費 0.1425%×0.28、最低 20 元、證交稅 0.3%、半個跳動點＋每邊 0.3% 滑價）；5% 當日成交值上限；開盤漲停不買、開盤跌停不賣',
        switches=dict(EXRIGHT_FIX=True, DEADLINE_FIX=True, INS15_MASK=True),
        engine='s10lib.simulate_s10 的副本 simulate_s68（只多一個計數器，驗收 A3 逐位相同）；sha256 見 src_sha256', src_sha256=chk['src_sha256'],
        note='所有參數（前 10 名、前 20 名、10%、60 日、÷10、70%）一個都不改。名次是絕對檔數：上櫃 L_MA 平均約 148 檔（上市約 279 檔），所以「前 10／前 20 名」在上櫃相對較寬鬆——這是把規則原封不動搬過去的後果，不做任何調整。'),
    universes=dict(
        OTC='主要：P.liq（60 日均成交值 ≥ 1,000 萬、收盤 ≥ 10 元、當日有成交、非 91xx）& 當日上櫃（P.MK＝2）& 非創新板。市場別看當日（轉上市前算上櫃，轉上市後掉出股池 → 下個訊號日依 E0 賣出）',
        ALL='描述：上市＋上櫃合併（P.liq & MK>0 & 非創新板）', CUR='驗收：現行只上市'),
    benchmark='0050 含息（底倉與比較對手都用 0050，同現行）',
    data=dict(panel_cache_sha256=inv['cache_sha256'], revenue='tw_revenue.csv＋S16 上櫃 2018-10 補洞檔（616 家；sha256 ' + inv['revenue']['patch']['sha256'] + '）',
              otc_exright=dict(
                  finding='上櫃官方日 K 的漲跌價差在除權息日是缺值（7,170 格；2021-10-07 起的 3,391 格與本地 exdiv_5y 的上櫃事件 100% 對上，無一例外）。共用引擎遇缺值以收盤÷前收計＝把股利當下跌（平均 −4.27%／次），EXRIGHT_FIX 只修上市。',
                  main='(a) 上櫃、有成交、漲跌價差缺值、前一日有成交：exdiv_5y（TPEX，2021-10-07～2026-10-07）有參考價且收盤在參考價 ±10.1% 內 → 漲跌價差＝收盤−參考價（同上市規則 a）；'
                       '否則（2021-10-06 以前全部、以及 2 格超出範圍者）→ 漲跌價差＝0，當日含息報酬記 0。(b) 上櫃、漲跌價差＝0、價變、前一日無成交、≤10.5% → 用前一成交收盤（同上市規則 b；1,743 格）。',
                  bias='2015-01～2021-10-06 的上櫃除權息日報酬記 0 而非「收盤÷參考價−1」：2021-10 起實測上櫃該值平均 +0.32%／次（上市稽核 +0.36%）→ 每檔每年約少算 0.3 個百分點 → 對上櫃策略略偏悲觀（保守）。'
                       '這段期間除權息日的漲跌停價以當日收盤為基準（同 EXRIGHT_FIX 之前的上市口徑），影響可忽略。',
                  sensitivities="'none'＝不處理（S16／共用引擎現況，偏悲觀的下界）；'zero'＝(a) 全部記 0（不用 exdiv_5y）。兩者只報告，不參與判定。"),
              other_biases=['下市櫃公司 2013 起沒有月營收 → 選不到（存活者），偏樂觀，S16 複製員量過約 0.15 個百分點／年', '上櫃成本模型同上市（實際價差可能較大）→ 偏樂觀',
                            'KY 公司不在營收檔 → 選不到', '上櫃沒有 INS15／期限例外的另行查核，照上市同一張表']),
    acceptance=dict(A0=chk['A0'], A2=chk['A2'], A3=chk['A3_all_ok'], A4=chk['A4']['twse_R_same'] and chk['A4']['twse_REF_same']),
    counts_seen_before_prereg=chk['counts_no_returns'])
PREREG = dict(
    id='S68', written_at=now, spec_sha256=None,
    run_once='run.py 只跑一次（results.json 存在就拒跑）；結果不論好壞照登記判定；不調任何參數、不換窗、不換除權息口徑。',
    primary=dict(cell='OTC', exright_mode='main', capital=1e6, shift=0,
                 stat='日曆月報酬，月超額＝策略−0050 含息；年化超額＝月平均×12；Newey-West t（Bartlett、落後 6 個月；同 S55）',
                 windows={'W2015': dict(frm='2015-07', to='2026-10', note='淨值起點 2015-07-01；第一個訊號 2016-02，之前全在 0050（超額≈0），照算'),
                          'W2019': dict(frm='2019-01', to='2026-10')}),
    decision=dict(t_threshold=1.65,
                  rule='兩個窗（W2015、W2019）都「年化超額 > 0 且 Newey-West t > 1.65」→「支持」（規則在沒看過的股池也有效）；'
                       '兩個窗年化超額都 > 0 但至少一個 t ≤ 1.65 →「方向一致但不顯著」；任一窗年化超額 ≤ 0 →「不支持」。',
                  auxiliary='20 條路徑（5 種資金 60／80／100／150／200 萬 × 換股日晚 0～3 天）：2019 起年化 − 0050 2019 起年化 的中位數，與為正的條數（≥15／20 為「輔助支持」）。輔助只並列報告，不改主要判定；主要與輔助不一致時明寫。',
                  interpretation='「支持」＝現行規則的超額不是上市樣本的巧合；「不支持」＝在上櫃看不到同樣的超額——可能是規則偏樂觀，也可能是上櫃本來就不同（流動性、SURGE 集中在上市 AI／記憶體股）；本題分不出兩者，只如實報告。不論結果，現行股池（只上市）不因本題更改。'),
    descriptive=['ALL（上市＋上櫃合併）同口徑的兩窗統計與 20 條路徑；對 CUR 的差', 'CUR（現行）同口徑數字作對照',
                 '上櫃容量：資金 1,000 萬（位移 0）時買進被 5% 成交值上限縮量的比例（縮量筆數 ÷ 進到上限檢查的筆數）、賣出被拆單次數、年化；另報 100 萬與 CUR 同資金對照',
                 '漲跌停買不到比例：開盤漲停而買不到的次數 ÷ 候選次數（n_sig），主跑與 20 條路徑合計；開盤跌停延後賣出次數',
                 "除權息口徑敏感度：OTC 在 'none'、'zero' 下的兩窗統計與 20 條路徑", '逐年報酬、最大回撤、交易筆數、勝率、每筆平均報酬、最賺 5 筆占比、持有中轉上市筆數、下市結清筆數'],
    prior_knowledge=['S16（2026-11-15，舊開關、上櫃除權息未處理）：OTC 主跑 2019 起 23.20%（0050 28.91%）、20 條中位數 24.26%、0／20 贏 0050、月超額 t −0.35。'
                     '本題差別：三個正式開關全開、上櫃除權息改 main 口徑（S16 的口徑把上櫃股利當下跌，偏悲觀）、判定改用 Newey-West 與兩窗。',
                     'S35：上櫃 F1 交易對同業平均為正但谷底翻揚特徵不顯著（另一題）。', '研究員事前預期：main 口徑會比 S16 高（每年多幾個百分點股利），但是否贏 0050 不知道。'],
    N_note='新格：OTC main（主要）＋ALL main（描述）＝2 格新組態；none／zero 是資料口徑敏感度，不計入。')
(H / 'SPEC.json').write_text(json.dumps(SPEC, ensure_ascii=False, indent=1))
PREREG['spec_sha256'] = sha(H / 'SPEC.json')
(H / 'PREREG.json').write_text(json.dumps(PREREG, ensure_ascii=False, indent=1))
for f in ('SPEC', 'PREREG'):
    (H / f'{f}.sha256').write_text(sha(H / f'{f}.json') + f'  {f}.json\n')
    print(f, sha(H / f'{f}.json'))
