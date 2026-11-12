#!/usr/bin/env python3
"""第 2 步迭代（只用挑選期 2006-04～2014-12，讀 explore.json，不碰 2015 年以後）。→ iterations.json
版本（全部報告；多重比較誠實揭露）：
  v1 退款＝cash，B1 分母＝成交日成本前權益 E_pre，主順序
  v2 退款＝prop（比例退款），其餘同 v1
  v3 v2＋B1 分母改用期間平均淨值（只並列）
  v4 v2＋實際成交價 edge 改用含成本的成交價（只並列）
  v0（實作修正紀錄）：run_engine 中心格第一版在 worker 回傳 Python tuple 日誌造成記憶體壓力、與另一題同機器時 OOM；改成 worker 端轉 DataFrame、工作程序 4→3。不影響任何數字（引擎決定性，逐位檢查照舊）。
選擇規則（事前寫在 PREREG.bridge_plan.refund_modes）：挑選期三變體平均 |U| 較小的退款模式為主欄；B1 與 edge 主欄維持事前定義（E_pre、參考價）。
python3 -I -B iterate.py"""
import json, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
X = json.loads((HERE / 'explore.json').read_text(encoding='utf-8'))
VARS = ['REAL_opt', 'REAL_p90', 'REAL_carry']
it = []
it.append(dict(version='v0', what='實作修正：日誌回傳格式與工作程序數（記憶體）', result='數字不變；check 階段逐位相同（7 組 0.0）、日誌與 o20 tradelog 一致'))
it.append(dict(version='v0b', what='實作修正：期間起點口徑統一（Y19on/POSTODD/V1520/H2126 用 x_stats 口徑：期間內第一個交易日收盤），使 Shapley 組合值與 o20 表逐位一致', result='只影響期間切點；中心格成交彙總為修正前產生（Y19on 多 1 個無換股交易日）'))
U = {}
for rk, ver in (('cash', 'v1'), ('prop', 'v2')):
    r = {v: dict(U=X['B_bridge'][f'{v}|{rk}']['residual_U'], bridges=X['B_bridge'][f'{v}|{rk}']['bridges_main'], start=X['B_bridge'][f'{v}|{rk}']['start'],
                 end=X['B_bridge'][f'{v}|{rk}']['end']) for v in VARS}
    U[rk] = float(np.mean([abs(r[v]['U']) for v in VARS]))
    it.append(dict(version=ver, refund=rk, SEL14=r, mean_abs_U=U[rk]))
it.append(dict(version='v3', what='B1 分母＝期間平均淨值', SEL14={v: dict(B1_Epre=X['A_acct_drag'][v]['b1']['B1'], B1_avgnav=X['A_acct_drag'][v]['b1']['B1_avgnav'], acct=X['A_acct_drag'][v]['acct']['cost']) for v in VARS},
               decision='只並列（平均淨值不是成本實際發生時的分母，無法做逐日乘法換算）'))
c = X['C_D_edges_quintiles']['normal|100萬|std']['REAL_opt']
it.append(dict(version='v4', what='實際成交價 edge 用含成本成交價', SEL14=dict(edge_open=c['edge_open_mean'], edge_fill_ref=c['edge_fill_mean'], edge_fill_allin=c['edge_allin_mean']),
               decision='只並列（含成本價會與 rtc 重複扣成本）'))
choice = min(U, key=U.get)
out = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), n_versions=len(it), versions=it, refund_main=choice, refund_rule='挑選期三變體平均 |U| 較小者', mean_abs_U=U)
(HERE / 'iterations.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(choice, U)
