#!/usr/bin/env python3
"""接續 2026-11-12 暫停的一次性樣本外模擬（不改 oos.py／run_lib.py）：沿用 scratchpad 的批次檢查點（oos_*.pkl，凍結後所跑），只補跑缺的批次。
同機有其他研究員作業、共用 13.67GB 記憶體上限，所以：
  - 行程數由 OCAS_NPROC 指定（預設 1）；
  - 工作行程把自己的 oom_score_adj 設為 1000（記憶體不足時先犧牲本題的工作行程，不波及其他研究員）；
  - 開工作池前先等 cgroup 剩餘記憶體 ≥ 3.3GB；工作行程被砍時只補跑缺的工作（結果與順序無關）。
模擬內容、參數、seeds 與 oos.sim() 完全相同。python3 -I -B oos_resume.py [sim|post]"""
import os, sys, time, runpy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import run_lib as RL; sys.path.pop(0)
NP = int(os.environ.get('OCAS_NPROC', '1'))
CG = Path('/sys/fs/cgroup/memory/process_api')
def headroom_mb():
    try:
        d = next(CG.glob('*/claude-code-bash'))
        return (int((d / 'memory.limit_in_bytes').read_text()) - int((d / 'memory.usage_in_bytes').read_text())) / 1048576
    except Exception:
        return 1e9
def wait_mem(need=3300):
    n = 0
    while headroom_mb() < need:
        if n % 20 == 0: print('等記憶體：剩', round(headroom_mb()), 'MB', flush=True)
        n += 1; time.sleep(15)
_init0 = RL.init
def init2():
    try: Path('/proc/self/oom_score_adj').write_text('1000')
    except Exception: pass
    _init0()
def _iter(jobs, nproc, initf):
    todo = list(range(len(jobs)))
    while todo:
        done = []
        wait_mem()
        try:
            with ProcessPoolExecutor(nproc, initializer=init2) as ex:
                futs = {ex.submit(RL.job, jobs[i]): i for i in todo}
                for fu in as_completed(futs):
                    i = futs[fu]
                    yield fu.result(); done.append(i)
        except BrokenProcessPool:
            print('BrokenProcessPool：重開，剩', len(todo) - len(done), flush=True)
        dset = set(done); todo = [i for i in todo if i not in dset]
RL._iter_results = _iter
_orig = RL.run_batch
def _rb(jobs, nproc=2, log=None, ckpt=None): return _orig(jobs, nproc=NP, log=log, ckpt=ckpt)
RL.run_batch = _rb
if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'sim'
    if mode == 'sim':
        sys.path.insert(0, str(HERE)); import oos; sys.path.pop(0)
        oos.sim()
    elif mode == 'post':
        sys.path.insert(0, str(HERE))
        runpy.run_path(str(HERE / 'post.py'), run_name='__main__')
