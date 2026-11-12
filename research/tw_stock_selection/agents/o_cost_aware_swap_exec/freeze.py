#!/usr/bin/env python3
"""凍結：讀 iterations.json 的選擇結果，寫 results.json['PREREG_FREEZE']（唯一候選規格、樣本外計畫、全部程式與關鍵資料 sha256）。
之後 oos.py 先斷言 PREREG／PREREG_POWER／PREREG_FREEZE 與全部雜湊才計算 2015 年後。
python3 -I -B freeze.py"""
import sys, json
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import cas_lib as CL; sys.path.pop(0)
RES = HERE / 'results.json'
R = json.loads(RES.read_text(encoding='utf-8'))
assert CL.canon(R['PREREG']) == R['PREREG_sha256'] and CL.canon(R['PREREG_POWER']) == R['PREREG_POWER_sha256']
assert 'PREREG_FREEZE' not in R, '已凍結'
IT = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
sel = IT['selection']; ch = sel['chosen']; opt = IT['versions'][ch]['opt']
nb = {}
if 'ca' in opt:
    for q in (0.75, 0.85, 0.95): nb[f'tau{int(q * 100)}'] = dict(opt, ca=q)
if opt.get('mt'):
    for m in (0.5, 2.0): nb[f'mt{m}'] = dict(opt, mt=m)
if opt.get('dc'):
    for dd in (3, 10): nb[f'dc{dd}'] = dict(opt, dc=dd)
code = {f.name: CL.sha(f) for f in sorted(HERE.glob('*.py'))}
data = {f: CL.sha(HERE / f) for f in ('data/cands_allseeds.csv.gz', 'explore.json', 'iterations.json', 'data/power_placebo.csv.gz', 'data/sel_navs.parquet', 'engine_sha.json')}
FZ = dict(
    written_utc=pd.Timestamp.now('UTC').isoformat(),
    chosen=ch, chosen_opt=opt, verdict_at_freeze=sel['verdict'], eligible=sel['eligible'],
    chosen_selection_metrics={c: IT['versions'][ch]['m'][c]['worst'] for c in ('N100', 'OFF100', 'S100', 'N30')},
    all_variants={k: dict(opt=v['opt'], N100=v['m']['N100']['worst']['med'], OFF100=v['m']['OFF100']['worst']['med'], S100=v['m']['S100']['worst']['med'],
                          N30=v['m']['N30']['worst']['med'], cond=v['m']['cond']) for k, v in IT['versions'].items() if 'm' in v},
    combo_choice=IT['combo_choice'],
    neighbors=nb,
    oos_plan=dict(
        main='全期（2006-04～各 j 終點）一次：BASE＋8 變體＋A20，100 萬一般，20 seeds × 13 j × 3 放回（G3 與描述表）；',
        chosen_cells='選中變體與 BASE：100 萬成本全關、100 萬壓力、30 萬一般、30 萬壓力、300 萬一般、1,000 萬一般（20 seeds）；',
        neighbors='鄰域（只報不選）100 萬一般 20 seeds；',
        aux='P3／P4 輔助（描述）：CA80、CA90、MT 與 BASE 在 30 萬／300 萬／1,000 萬一般，seeds 0～4；',
        periods=dict(validation=['2015-01-01', '2020-12-31'], holdout=['2021-01-01', 'j 終點'], full=['首訊號日', 'j 終點'], Y19on=['2019-01-01', 'j 終點'],
                     seg=[['2006-04-01', '2012-12-31'], ['2013-01-01', '2018-12-31'], ['2019-01-01', 'j 終點']]),
        stats='對 0050：x_stats.stats（訊號窗 t、日曆年化）於 20 seeds 平均淨值，13 j 中位數與 p10，放回版本取最差；配對差三層、月配對 t、月份 bootstrap（2,000 次，seed 20261112）；'
              'G3：o_g2_feasibility_stoprule/g2lib.family()（2,334 條可得序列）＋本題 8 變體（REAL_opt 13 j 平均淨值，官方換股窗超額），Hansen SPA_c（平穩 bootstrap L=6、B=10,000、seed 20261103）、DSR（N=2,581 名目與 Li-Ji）。'),
    code_sha256=code, data_sha256=data,
    note='凍結後不得修改規格；G6 須由工作流層分派不同子代理重寫。',
)
R['PREREG_FREEZE'] = FZ; R['PREREG_FREEZE_sha256'] = CL.canon(FZ)
RES.write_text(CL.dumps(R), encoding='utf-8')
print(ch, sel['verdict'], R['PREREG_FREEZE_sha256'], FZ['written_utc'])
