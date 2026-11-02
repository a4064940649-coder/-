#!/usr/bin/env python3
"""步驟四：凍結。寫 PREREG（規格全文、選定版本、四項修正、判定規則、事前預測、所有程式與輸入的 sha256）進 results.json。
之後 s5_oos.py 會斷言 PREREG 存在、雜湊相符，才跑全期（含 2019-05 起驗證期與 2024 起保留期）。"""
import sys, json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import x_data as D

CHOSEN = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
CHOSEN_NAME = sys.argv[2] if len(sys.argv) > 2 else 'v1'

FROZEN = (HERE / 'frozen_spec.txt').read_text()

SCRIPTS = ['x_data.py', 'x_ideal.py', 'x_exec.py', 'x_stats.py', 'x_run.py', 's0_interp.py', 's1_gate_zero.py', 's2_explore.py',
           's3_iterate.py', 's4_prereg.py', 's5_oos.py', 's6_summary.py', 's7_apply.py', 's8_forward_calib.py', 'frozen_spec.txt',
           'interp_register.json', 'explore.json', 'iterations.json']
INPUTS = [D.PNL / f'{k}.parquet' for k in ('close_raw', 'open', 'value', 'ret', 'flag')] + [
    D.PNL / 'model_spec7.json', D.PNL / 'plans_j7.csv', D.PNL / 'plans_ab7.csv', D.PNL / 'jscan_end_days.json',
    D.O7 / 'raw' / 'exec_inputs' / 'cpi_tw.json', D.AG / 'o_infra_v3_rebuild' / 'delist_prices.parquet',
    D.AG / 'o_infra_v3_rebuild' / 'merger_consideration.csv', D.AG / 'o_infra_v3_rebuild' / 'delist_official.csv',
    D.AG / 'o_disposal_rejudge' / 'disposal_hist_rejudge.csv', D.EXT / 'disposal_5y.csv.gz',
    Path('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv'), D.O7 / 'navs_exec7.parquet', D.O7 / 'results.json']


def hashes():
    h = {}
    for s in SCRIPTS: h[s] = D.sha(HERE / s)
    for p in INPUTS: h[str(p)] = D.sha(p)
    return h


if __name__ == '__main__':
    p = HERE / 'results.json'
    R = json.loads(p.read_text()) if p.exists() else {}
    assert 'PREREG' not in R, 'PREREG 已存在，不可覆寫'
    final = dict(CHOSEN); final.update(f1=True, f2=True, f3=0.5, f4=True)
    pre = dict(topic='o_exec_engine_rewrite_v7x：V7 可執行引擎獨立重寫（V7X）', time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
               chosen_version=CHOSEN_NAME, orig_cfg=CHOSEN, final_cfg=final, frozen_spec=FROZEN,
               interp_register_sha256=json.loads((HERE / 'interp_register.json').read_text())['sha256'],
               n_versions_tried=len(json.loads((HERE / 'iterations.json').read_text())),
               hashes=hashes())
    s = json.dumps(pre, ensure_ascii=False, sort_keys=True)
    R['PREREG'] = pre; R['PREREG_sha256'] = hashlib.sha256(s.encode()).hexdigest()
    p.write_text(json.dumps(R, ensure_ascii=False, indent=1))
    print(R['PREREG_sha256'], pre['time_utc'])
