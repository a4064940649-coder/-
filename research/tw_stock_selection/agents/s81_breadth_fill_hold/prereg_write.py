#!/usr/bin/env python3
"""S81 事前登記（寫完算 sha256，之後不可改）。python3 -I -B prereg_write.py → PREREG.json、PREREG.sha256、code.sha256
此時只跑過 accept.py（現行格重現），沒有看過任何新格子的成績。"""
import sys, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s81lib as Q
HERE = Path(__file__).resolve().parent
assert not (HERE / 'PREREG.json').exists(), 'PREREG.json 已存在，不可重寫'
acc = json.loads((HERE / 'accept.json').read_text()); assert acc['passed']
cells = Q.grid()
pre = dict(
    task='S81 照 S76 診斷方向（名單放寬／空名額遞補／拉長持有）設計規則並驗證',
    date='2026-11-20',
    data='S75 連續面板（2008-01-02～2026-10-22）、V_BRIDGE（B2 口徑 SURGE）、嚴格成本（strict）、本金 100 萬、底倉 W10_70 不動（70% 選股、30% 底倉 0050、空位買 0050）',
    acceptance=dict(current_cell=Q.CUR, cagr=acc['A2']['cagr'], final_nav=acc['A2']['final_nav'], note='現行格淨值與 S75 存檔逐日相同才往下做（accept.json）'),
    fixed_not_searched=['分數＝SURGE（24 個月、B2 口徑）', '60 日線過濾', '每月換股日與成交方式', '底倉 W10_70', '成本 strict', '本金 100 萬', '不加大盤過濾、不加其他資訊'],
    search_space=dict(
        A_breadth='(持股上限, 候選深度) 12 組：' + '、'.join(f'({a},{b})' for a, b in Q.PAIRS) + '；候選＝季線上依 SURGE 排名前「深度」名；每檔金額＝選股資金（淨值 − 底倉）÷ 持股上限；深度 < 上限的組合不測（永遠補不滿）',
        B_fill='動能門檻 4 種：M10（近 20 日漲幅 > 10%，現行）、M5（> 5%）、M0（> 0%）、NOMOM（不設門檻，依名次補滿）。「往下遞補到下一名有過門檻的」＝深度 > 上限（例：上限 10、深度 20／30／50＝在前 20／30／50 名內依名次往下找有過門檻的補到滿）；不遞補＝深度＝上限且 M10。'
               '進場名單＝深度內過門檻的全部，引擎依名次由前往後買到沒有空位為止（已持有的不重買；漲停／買不起一張的跳過換下一名）',
        C_hold='出場 5 種（一次只動一項）：H1＝每個訊號日檢查，不在季線上前 K0 名就賣（K0＝max(20, 深度)；深度 10／20 時＝現行「掉出前 20 名或收盤 ≤ 60 日線」）；'
               'H2／H3＝進場後第 2／3 個訊號日起才檢查名次（K0），但每個訊號日只要不在季線上合格名單（破季線、沒分數、出股池）仍照賣；K30／K50＝續抱名次放寬到前 30／50 名（只在大於 K0 時存在）',
        n_cells=len(cells), cells=[c['id'] for c in cells],
        versions='每一格同一條規則跑兩個帳戶版本：LOT＝歷史實際限制（2020-10-26 前只能整張）；ODD＝全期允許零股（cslib.ODD_DATE 改 2000-01-01，成本仍 strict）。兩版算同一個試驗（規則相同），不能各挑各的',
        n_trials=len(cells), n_account_sims=2 * len(cells), cap_trials=200),
    periods=dict(train=list(Q.TRAIN), valid=list(Q.VALID), subs=Q.SUBS,
                 blind='run_train.py 只算、只存 2008-01～2015-12 的統計（模擬是逐日往前、前段結果不受後段資料影響）；寫完 FROZEN.json＋sha256 後 run_valid.py 才算 2016-01～2026-10，只跑一次（valid_results.json 已存在就拒絕），不回頭改'),
    metrics=dict(excess='對 0050 的月超額平均 × 12（個百分點／年）', t='月超額的 Newey-West（6 期）t 值', IR='月超額平均 ÷ 月超額標準差 × √12', worst_year='逐年（日曆年、月複利）報酬減 0050 的最小值',
                 maxdd='日淨值最大跌幅', cagr='月複利年化'),
    selection=dict(
        step1='每格兩個版本各算訓練段 IR 與逐年超額最差值 W（2008～2015 共 8 年）；取兩版較差者：IRm＝min(IR_LOT, IR_ODD)、Wm＝min(W_LOT, W_ODD)',
        step2='192 格內各自轉成 0～1 百分位名次 p_IR、p_W；raw＝(p_IR＋p_W)／2（兩者並重）',
        step3='鄰近格＝只有一個維度差一級的格子（s81lib.neighbors）；nb＝鄰近格 raw 的平均；score＝0.5×raw＋0.5×nb',
        step4='合格條件：(i) raw − nb ≤ 0.20（不是孤立尖點）；(ii) 兩個版本的訓練段超額都高於同版本的現行格',
        step5='合格格中 score 最大者凍結；同分取格子序號小者（較接近現行）。沒有合格格 → 凍結「無」，結論＝維持現行，驗證段仍照跑全體分佈供參考',
        note='挑選只用訓練段；現行格 C10|D10|M10|H1 在格子裡，被挑到就代表「不用改」'),
    pass_criteria=dict(
        P1='驗證段（2016-01～2026-10）：凍結規則超額 ≥ 現行同段超額 − 3 個百分點',
        P2='訓練段（2008-01～2015-12）：凍結規則超額 ≥ 現行同段超額 ＋ 3 個百分點，而且凍結規則對 0050 月超額的 t（NW6）≥ 1.5',
        P3='全期（2008-01～2026-10）最大跌幅不比現行深 3 個百分點以上（凍結 maxdd ≥ 現行 maxdd − 0.03）',
        P4='LOT 與 ODD 兩個版本各自拿同版本的現行格來比，P1～P3 都要過，結論才算相同、才算通過',
        verdict='P1～P4 全過＝「有一條前段明顯好、後段不輸太多、跌幅不更深的規則」，交獨立複製；任何一項不過＝維持現行（可說明哪項沒過）'),
    report=['現行／凍結規則／0050 在前段、後段、全期、四小段的年化、超額、t、IR、最大跌幅、平均持股檔數、持股比例、交易數、換手、成本占毛超額比例（毛＝同規則 cost=zero 重跑）',
            '20 條路徑（訊號位移 0～3 天 × 本金 60／80／100／150／200 萬，同 S75）中位數：現行與凍結規則、兩個版本',
            '凍結後另算搜尋空間全體在驗證段的分佈（中位數、分位數、凍結規則排第幾、訓練段與驗證段名次相關）',
            '去掉報酬率前 5% 交易（各段各自的已平倉交易依報酬率排序，封鎖後重跑）後的超額',
            '補充（不影響判定）：凍結規則 ODD 版改用 oddlot 成本（零股每邊再加 0.3%）的結果；凍結規則減現行的月報酬差 t 值'],
    multiple_testing='本題 192 個試驗全部計入資料窺探族群；訓練段數字是 192 選 1（偏樂觀），判定只看驗證段一次性結果＋事前門檻；另報「訓練段名次對驗證段名次」的等級相關與驗證段全體分佈',
    limits='沿用 S75／S76：2014-01-16 前除權息日報酬記 0、營收為事後更正值、缺 KY 與下市股、2015-06 前漲跌停 7% 引擎用 10%；後段（2016～2026）已被前面多個研究看過，不是真正沒人看過的資料，只是本題的規則沒有用它來挑')
(HERE / 'PREREG.json').write_text(json.dumps(pre, ensure_ascii=False, indent=1))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
(HERE / 'PREREG.sha256').write_text(f"{sha(HERE / 'PREREG.json')}  PREREG.json\n")
(HERE / 'code.sha256').write_text(''.join(f"{sha(HERE / f)}  {f}\n" for f in ('s81lib.py', 'accept.py', 'prereg_write.py')))
print(open(HERE / 'PREREG.sha256').read(), open(HERE / 'code.sha256').read())
