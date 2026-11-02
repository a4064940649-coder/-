#!/usr/bin/env python3
"""步驟 10：登記 PREREG_FORWARD（2026-11-10 期限日前）：規則、σ_design、先驗檔、make_forward.py 與 seq2.py 的 sha256、線清單、時程。
之後初始化 live_tracking/main_forward/（只建立表頭與 fills_input.csv 範本；2026-11 第一個錨定月要等 11-10 收盤）。
用法：python3 -I -B s10_freeze_forward.py"""
import sys, json, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fslib as F


def main():
    PRE = F.check_prereg('PREREG')
    res = F.load_res(); oc = res['OC_FINAL']; dr = res['DRYRUN']
    assert dr['pass'], '乾跑未通過，不得登記'
    rule = oc['chosen']
    obj = dict(
        forward_start='2026-11',
        status=('FROZEN' if oc['chosen'] else 'NOT_FROZEN：最終 OC 無規則在三 DGP 都過（D2 誤升級超過 5%），依 PREREG 不得凍結判定規則；前瞻腳本照常產生資料，decision_log 只記描述統計'),
        rule=dict(name=rule, params=oc['params'], sigma_design=oc['sigma_design'], prior_file='data/forward_prior_main.npz',
                  forward_start='2026-11', decision_line='JMED（make_forward.py：13 個 J 紙上線窗超額的中位數，K5T-lite 一般成本）',
                  oc_summary={k: oc['summary'][rule][k] for k in ('worst_power_36', 'worst_false_up_36', 'worst_false_down_36', 'power_up_p1_36')} if rule else None,
                  shortest_feasible_H=oc.get('fallback', {}).get('shortest_feasible_H')),
        actions=dict(UPGRADE='主策略配置升到 100%（或新腿依同規則升級）；需總監確認後執行', DOWNGRADE='退回 0050 核心（主策略 0%，只紙上追蹤）',
                     CONTINUE='維持起始配置（SOP 現行）'),
        refit='前瞻 ≥ 12 個月且前瞻 sd > 1.5σ_design：σ ← 前瞻 sd 重算；升級 S_n 邊界只能變嚴。唯一允許的 refit。',
        schedule='每月換股日（第一個 ≥ 11 日的交易日）收盤後執行一次；J12 之後再執行一次讓 JMED 定稿；每 12 個月（2027-11、2028-11…）報告一次',
        lines=['MAIN_OFFICIAL', 'J0..J12（paper_lists 只規定 j0／j1／j6，另存 jall）', 'JMED', 'E50', 'M_E50_30', 'C_E50_70', 'M_2330_30',
               'CAPM_F1／C50／C70（讀 live_tracking/o7_capmatched/paper_nav.tsv）', 'REGIME_V5（讀 live_tracking/o7_regime/ledger.csv）',
               'R_ROLL（讀 o_rev_freshness_attention/forward/rroll_trades.csv）', 'R2、T1、T2（pending_external）'],
        scope='規則凍結後，主策略與各候選的任何改版只能依此規則與 2026-11 以後的資料判定',
        file_hashes={f: F.sha(F.HERE / f) for f in ('make_forward.py', 'seq2.py', 'data/forward_prior_main.npz', 'raw/v7/model_spec7.json')},
        main_picks_today_sha256=F.sha(F.AG.parent / 'main_picks_today.py'),
        dryrun_pass=dr['pass'], prereg_sha256=res['PREREG_sha256'], utc=F.now_utc())
    h = F.register('PREREG_FORWARD', obj)
    print('PREREG_FORWARD', h)
    r = subprocess.run([sys.executable, '-I', '-B', str(F.HERE / 'make_forward.py')], capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-1500:])


if __name__ == '__main__':
    main()
