#!/usr/bin/env python3
"""前瞻盲測序貫監測（2026-11 起，每月換股日之後執行）。規則見 prereg_forward.json（凍結、含雜湊）。
讀取（唯讀）：
  主策略：<live>/main_forward/forward_returns.csv（start,end,ret[,ret_0050][,ret_disp][,ret_2330]；ret＝實際帳戶主策略 100% 窗報酬）
          沒有時改讀 <live>/o7_regime/main_leg_returns.csv（start,end,ret）。
          成交帳本 <live>/main_forward/trades.csv（date,code,side,shares,fill_price,open_price,ticks_from_open,pct_open_volume）只做執行診斷。
  市值匹配：<live>/o7_capmatched/paper_nav.tsv（exec,next_exec,r_F,r_REP,r_0050,r_C50,r_C70）
  權值輪動：<live>/o7_regime/paper_nav.csv（start,rot,static,r0050,main,…）
  或用 --forward 合併檔（line,start,end,ret_line,ret_0050）覆蓋（測試、複製用）。
  0050 含息：--e50（預設 finlab etf_0050.csv），缺 ret_0050 時以 asof 價格計算。
輸出：JSON（--out；並印到標準輸出）。每條線：月數、平均月超額、更新後 P(α>0)、E_up、E_down、財富比、觸發狀態、配置建議。
  --state 讀入上次的配置狀態（每線目前配置、上次變更月），依「前瞻 ≥ 12 個月、±20pp 死區、每年最多一次」產生新狀態（--state_out）。
用法：python3 -I -B forward_monitor.py [--live <live_tracking 目錄>] [--forward fake.csv] [--tier 100萬 --mode FIX] [--out monitor.json]
決定性：不用任何亂數；同一輸入逐位相同。
"""
import sys, json, math, argparse, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import seqlib as Q

PREREG_F = HERE / 'prereg_forward.json'
LIVE = HERE.parent / 'live_tracking'
E50_DEFAULT = '/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_prereg():
    P = json.loads(PREREG_F.read_text(encoding='utf-8'))
    body = P['registration']
    h = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert h == P['registration_sha256'], 'prereg_forward.json 被改過'
    for rel, hv in body['hashes'].items():
        assert sha(HERE / rel) == hv, f'{rel} 與登記雜湊不符'
    return body

def e50_ret(e50, a, b):
    return float(e50.asof(pd.Timestamp(b)) / e50.asof(pd.Timestamp(a)) - 1)

def collect(live, fwd, e50_path, start):
    """回傳 {line_base: DataFrame(start, x_pp, r50)}，x＝線報酬 − 0050（pp）。"""
    out, notes = {}, []
    if fwd:
        F = pd.read_csv(fwd, parse_dates=['start', 'end'])
        F = F[F.start >= pd.Timestamp(start)]
        for ln, g in F.groupby('line'):
            g = g.sort_values('start')
            out[ln] = pd.DataFrame({'start': g.start.values, 'x_pp': (g.ret_line - g.ret_0050).values * 100, 'r50': g.ret_0050.values})
        return out, ['forward 合併檔：' + str(fwd)]
    e50 = None
    def get_e50():
        nonlocal e50
        if e50 is None:
            e50 = pd.read_csv(e50_path, parse_dates=['date']).set_index('date')['adj'].sort_index()
        return e50
    # 主策略
    mp = Path(live) / 'main_forward' / 'forward_returns.csv'
    if not mp.exists():
        mp2 = Path(live) / 'o7_regime' / 'main_leg_returns.csv'
        mp = mp2 if mp2.exists() else None
    if mp is not None:
        M = pd.read_csv(mp, parse_dates=['start', 'end']).sort_values('start')
        M = M[M.start >= pd.Timestamp(start)]
        if len(M):
            r50 = M['ret_0050'].values if 'ret_0050' in M else np.array([e50_ret(get_e50(), a, b) for a, b in zip(M.start, M.end)])
            rm = M['ret'].values
            out['MAIN'] = pd.DataFrame({'start': M.start.values, 'x_pp': (rm - r50) * 100, 'r50': r50})
            out['L3_MAIN70_E50_30'] = pd.DataFrame({'start': M.start.values, 'x_pp': 0.7 * (rm - r50) * 100, 'r50': r50})
            if 'ret_disp' in M:
                out['L2_MAIN80_DISP20'] = pd.DataFrame({'start': M.start.values, 'x_pp': (0.8 * rm + 0.2 * M.ret_disp.values - r50) * 100, 'r50': r50})
            if 'ret_2330' in M:
                out['L4_MAIN70_2330_30'] = pd.DataFrame({'start': M.start.values, 'x_pp': (0.7 * rm + 0.3 * M.ret_2330.values - r50) * 100, 'r50': r50})
        notes.append('主策略來源：' + str(mp))
    else:
        notes.append('主策略前瞻報酬檔不存在（main_forward/forward_returns.csv 或 o7_regime/main_leg_returns.csv）')
    cp = Path(live) / 'o7_capmatched' / 'paper_nav.tsv'
    if cp.exists():
        Cm = pd.read_csv(cp, sep='\t', parse_dates=['exec', 'next_exec']).sort_values('exec')
        Cm = Cm[Cm.exec >= pd.Timestamp(start)]
        for ln, col in (('L5_CAPM_F', 'r_F'), ('L6_CAPM_C50', 'r_C50'), ('L7_CAPM_C70', 'r_C70')):
            out[ln] = pd.DataFrame({'start': Cm.exec.values, 'x_pp': (Cm[col] - Cm.r_0050).values * 100, 'r50': Cm.r_0050.values})
    else:
        notes.append('o7_capmatched/paper_nav.tsv 不存在')
    rp = Path(live) / 'o7_regime' / 'paper_nav.csv'
    if rp.exists():
        Rg = pd.read_csv(rp, parse_dates=['start']).sort_values('start'); Rg = Rg[Rg.start >= pd.Timestamp(start)]
        out['L8_REGIME'] = pd.DataFrame({'start': Rg.start.values, 'x_pp': (Rg.rot - Rg.r0050).values * 100, 'r50': Rg.r0050.values})
        out['L9_REGIME_CORE50'] = pd.DataFrame({'start': Rg.start.values, 'x_pp': 0.5 * (Rg.rot - Rg.r0050).values * 100, 'r50': Rg.r0050.values})
        out['L10_REGIME_D'] = pd.DataFrame({'start': Rg.start.values, 'x_pp': (Rg.rot - Rg.static).values * 100, 'r50': Rg.static.values})
    else:
        notes.append('o7_regime/paper_nav.csv 不存在')
    return out, notes

def exec_diag(live):
    tp = Path(live) / 'main_forward' / 'trades.csv'
    if not tp.exists(): return {'status': 'no_trades_file'}
    D = pd.read_csv(tp, dtype={'code': str})
    o = {'n_trades': int(len(D))}
    for side in ('buy', 'sell'):
        g = D[D.side == side]
        if len(g):
            sgn = 1 if side == 'buy' else -1
            o[side] = {'n': int(len(g)), 'ticks_from_open_mean': float(g.ticks_from_open.mean()),
                       'slip_bps_mean': float((sgn * (g.fill_price / g.open_price - 1) * 1e4).mean()),
                       'pct_open_volume_median': float(g.pct_open_volume.median())}
    return o

def evaluate(body, data, tier, mode, state):
    Z = np.load(HERE / body['priors_file']); th = Z['theta']
    thr = body['thresholds']
    res = {}
    for line, spec in body['lines'].items():
        base = spec['data_key']
        if line.startswith('L1_'):
            key = f'L1_MAIN_B10_EXEC@{tier}|{mode}'; spec = body['lines_L1_variants'][key]
        else:
            key = line
        prior = Z[spec['prior_key']]
        if base not in data or len(data[base]) == 0:
            res[line] = {'status': 'no_data', 'prior_key': spec['prior_key']}; continue
        D = data[base]
        x = D.x_pp.values; r50 = D.r50.values
        st = Q.path_stats(x, prior, th, spec['sigma_pp'], spec['alpha_min'], r50=r50)
        tg = Q.triggers(st, thr)
        n = len(x)
        ft = {k: int(Q.first_time(v)[0]) for k, v in tg.items()}
        r = {'status': 'ok', 'n_months': n, 'first_start': str(pd.Timestamp(D.start.iloc[0]).date()), 'last_start': str(pd.Timestamp(D.start.iloc[-1]).date()),
             'mean_excess_pp': float(x.mean()), 'post_P_alpha_gt_0': float(st['post_P'][0, -1]),
             'E_up': float(np.exp(st['logE_up'][0, -1])), 'E_down': float(np.exp(st['logE_down'][0, -1])),
             'wealth_ratio_vs_0050': float(st['wealth_ratio'][0, -1]),
             'trigger_now': {k: bool(v[0, -1]) for k, v in tg.items()}, 'first_trigger_month': ft,
             'prior_key': spec['prior_key'], 'sigma_pp': spec['sigma_pp'], 'alpha_min': spec['alpha_min']}
        # 配置建議（只有主策略線與新策略的核心衛星線有配置意義）
        al = spec.get('allocation')
        if al:
            if line.startswith('L1_'):
                al = body['lines_L1_variants'][key]['allocation']
            cur = (state or {}).get(line, {}).get('alloc', al['favorable'])
            last = (state or {}).get(line, {}).get('last_change_month', 0)
            tgt = cur
            if r['trigger_now']['down']: tgt = al['unfavorable']
            elif r['trigger_now']['upgrade']: tgt = al['favorable']
            ok = (n >= thr['change_min_months']) and (abs(tgt - cur) >= thr['dead_zone_pp']) and (last == 0 or n - last >= 12)
            r['allocation'] = {'current_pct': cur, 'target_pct': tgt, 'change_allowed': bool(ok and tgt != cur),
                               'new_pct': tgt if (ok and tgt != cur) else cur, 'favorable_pct': al['favorable'], 'unfavorable_pct': al['unfavorable']}
        if r['trigger_now']['review']: r['note'] = '觸發總監審查（不自動停用）'
        res[line] = r
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--live', default=str(LIVE)); ap.add_argument('--forward', default=None); ap.add_argument('--e50', default=E50_DEFAULT)
    ap.add_argument('--tier', default='100萬'); ap.add_argument('--mode', default='FIX')
    ap.add_argument('--state', default=None); ap.add_argument('--state_out', default=None); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    body = load_prereg()
    data, notes = collect(a.live, a.forward, a.e50, body['forward_start'])
    if 'MAIN' in data: data['L1'] = data['MAIN']
    state = json.loads(Path(a.state).read_text(encoding='utf-8')) if a.state else None
    res = evaluate(body, data, a.tier, a.mode, state)
    out = {'prereg_sha256': json.loads(PREREG_F.read_text(encoding='utf-8'))['registration_sha256'], 'tier': a.tier, 'mode': a.mode,
           'notes': notes, 'lines': res, 'exec_diagnostics': exec_diag(a.live) if not a.forward else {'status': 'forward 合併檔模式'}}
    txt = json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True)
    if a.out: Path(a.out).write_text(txt, encoding='utf-8')
    if a.state_out:
        ns = {l: {'alloc': r['allocation']['new_pct'], 'last_change_month': (r['n_months'] if r['allocation']['change_allowed'] else (state or {}).get(l, {}).get('last_change_month', 0))}
              for l, r in res.items() if r.get('status') == 'ok' and 'allocation' in r}
        Path(a.state_out).write_text(json.dumps(ns, ensure_ascii=False, indent=1, sort_keys=True), encoding='utf-8')
    print(txt)

if __name__ == '__main__':
    main()
