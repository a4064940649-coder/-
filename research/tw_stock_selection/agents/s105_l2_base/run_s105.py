#!/usr/bin/env python3
"""S105 底倉資本效率：0050 底倉部分改用 00631L（台灣50正2）。三階段依序：
  python3 -I -B run_s105.py prereg     → PREREG.json／PREREG.sha256（格子、規則、窗、判準、假設、程式 sha256；不算任何格的績效）
  python3 -I -B run_s105.py train      → 核對 PREREG → 訓練窗（2008-01-02～2021-12-31）＋實際資料子窗（2015～2021，2014-12-31 重新開帳戶）→ 選格 → FROZEN.json／.sha256
  python3 -I -B run_s105.py validate   → 核對 FROZEN → 驗證窗（2021-12-30 收盤基底～2026-10-22）一次；results.json 已存在就拒跑
前置：s103_futures_overlay/sleeves.npz（S75 v3 口徑引擎 NAV＋stockfrac，sat 0.70／0.85 × 5 本金 × 位移 0～3）；build_rebal_days.py → rebal_days.json。"""
import sys, json, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import l2lib as X

CODE = ['build_rebal_days.py', 'l2lib.py', 'run_s105.py']
WIN = {'train': ('2008-01-02', '2021-12-31'), 'train_synth_2008_2014': ('2008-01-02', '2014-12-31'), 'train_actual_2015_2021': ('2015-01-01', '2021-12-31'),
       'valid': ('2022-01-01', '2026-10-22'), 'full': ('2008-01-02', '2026-10-22')}
BASE = dict(sat=70, Lw=0.0, Cw=0.0, mode='monthly')
CELLS = {
    'L1_M': dict(sat=85, Lw=0.15, Cw=0.0, mode='monthly', note='等曝險：00631L 15%（≈30% 0050），衛星 85%（每檔 8.5%，空位 0050）；每月換股日再平衡'),
    'L1_B': dict(sat=85, Lw=0.15, Cw=0.0, mode='band', note='同 L1，L 權重偏離 15%±5pp 才再平衡'),
    'L2_M': dict(sat=70, Lw=0.15, Cw=0.15, mode='monthly', note='等曝險、衛星不變：00631L 15%＋現金 15%（0 利率），衛星 70%；每月再平衡'),
    'L2_B': dict(sat=70, Lw=0.15, Cw=0.15, mode='band', note='同 L2，L 偏離 ±5pp 才再平衡（現金同時調回 15%）'),
    'L3_M': dict(sat=70, Lw=0.30, Cw=0.0, mode='monthly', note='加曝險：00631L 30%（≈60% 0050），衛星 70%；每月再平衡'),
    'L3_B': dict(sat=70, Lw=0.30, Cw=0.0, mode='band', note='同 L3，L 偏離 30%±5pp 才再平衡'),
}
SELECT_RULE = ('合格＝訓練窗（2008-01-02～2021-12-31，連續帳戶，2014-11 前 00631L 為合成）對基準三項同時通過：年化 > 基準、MDD ≥ 基準 MDD − 5pp、Sharpe ≥ 基準；'
               '且實際資料子窗（2015-01～2021-12，2014-12-31 收盤重新開帳戶，00631L 全為實際資料）年化 ≥ 同窗基準。'
               '合格格中取訓練窗 Sharpe 最高者，平手取年化較高者。無合格格時取訓練窗 Sharpe 最高者，標記「無合格格」，仍做驗證一次（只當資訊）。')
JUDGE = ('驗證窗（2021-12-30 收盤基底～2026-10-22，凍結格一次，連續帳戶）同時滿足：(1) 年化 > 基準；(2) MDD ≥ 基準 MDD − 5pp；(3) Sharpe ≥ 基準（日報酬、rf=0、×√252）。'
         '另報 20 條路徑（5 本金 × 位移 0～3，各與同路徑基準比）三項全過條數；建議採用＝標準路徑三項全過且 20 條中 ≥ 15 條全過，且訓練期合格。')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jsha(o):
    return hashlib.sha256(json.dumps(o, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


RD = json.loads((HERE / 'rebal_days.json').read_text())


def mask(d, shift):
    return np.isin(np.array([str(x.date()) for x in d]), RD[str(shift)]['trade'])


def run_cell(c, z, d, r50, rL, cap=100, shift=0, t_start=0, rf_d=None, slip=0.001):
    k = f"s{c['sat']}|{cap}|{shift}"
    return X.run(z['nav|' + k], z['sf|' + k], r50, rL, c['Lw'], c['Cw'], c['mode'], mask(d, shift), t_start=t_start, rf_d=rf_d, slip=slip)


def block(eq, d, e, wins):
    return {w: X.metrics(eq, d, *WIN[w], e50=e) for w in wins}


def stage_prereg():
    assert not (HERE / 'PREREG.json').exists(), 'PREREG.json 已存在'
    pre = dict(agent='S105', date='2026-11-27', title='底倉資本效率：0050 底倉部分改用元大台灣50正2（00631L），省下資金加碼選股或放現金',
               baseline='S75 v3 連續帳戶 V_BRIDGE（sat 0.70、W10、空位 0050、嚴成本），100 萬、位移 0；全期年化 23.50%、MDD −49.91%（以 s103 sleeves.npz 重建，相對誤差 < 1e-13）',
               cells=CELLS, base_cell=BASE, windows=WIN, select_rule=SELECT_RULE, judge=JUDGE,
               assumptions=dict(
                   l2_data='00631L 還原日序列 /home/user/jonatw/finlab-txd/data/raw/etf_00631L.csv（2014-10-31 起；w4_lev2_half_rebalance 已對 etf_tr_m 月報酬核對，TE 0.45%/年）。'
                           '對齊引擎交易日：檔中非交易日列（2026-07-10、2026-09-20）不用；缺 2026-03-25～30 → ffill，報酬併入 03-31。',
                   synthetic='2014-11-03 以前（含 2014-10-31）r_L＝2×r_0050（引擎同一還原 0050）− 1%/245。敏感度：再扣隔拆利率（rate）、以 2015～2021 實測落差校正（cal）。只影響訓練窗 2008～2014。',
                   rebalance='monthly：該位移的引擎換股交易日（訊號列 k+1）收盤；band：每日收盤檢查 |L/A − Lw| > 5pp。再平衡時 L→Lw、C→Cw，差額走 0050。'
                             '非再平衡日 E＝N−L−C，E<0 先用現金、再賣 L（被迫賣 L）。股票比例照引擎 stockfrac（與 S103 相同的近似：帳戶規模改變不重算引擎下單）。',
                   costs='00631L 與 0050 疊加交易每邊：買＝0.1425%×0.28＋滑價 0.1%；賣＝再加證交稅 0.1%。敏感度滑價 0.3%。股票、引擎內 0050 成本照引擎。',
                   cash='現金 0 利率（主）；敏感度用隔拆年均（同 w4 run.py）。',
                   sharpe='日報酬 mean/std×√252、rf=0；窗基底＝窗前最後交易日。',
                   paths20='sleeves.npz 的 5 本金（60/80/100/150/200 萬）× 位移 0～3；L1 用 sat 0.85 引擎、L2/L3 與基準用 sat 0.70；各路徑換股日用該位移的。'),
               disclosure='寫 PREREG 前已看過：基準重建數字、00631L 對 2×0050 的落差統計（含 2022～2026：實際 00631L 年化 42.1% 對每日2倍0050 57.3%，落差 −15.2pp/年；2015～2021 為 +2.2pp/年）。沒有算過任何格的績效。',
               code_sha256={f: sha(HERE / f) for f in CODE}, sleeves_sha256=sha(X.S103 / 'sleeves.npz'), rebal_days_sha256=sha(HERE / 'rebal_days.json'))
    (HERE / 'PREREG.json').write_text(json.dumps(pre, ensure_ascii=False, indent=1))
    (HERE / 'PREREG.sha256').write_text(sha(HERE / 'PREREG.json') + '  PREREG.json\n')
    print('PREREG', sha(HERE / 'PREREG.json'))


def check(name):
    h = (HERE / f'{name}.sha256').read_text().split()[0]
    assert sha(HERE / f'{name}.json') == h, f'{name} 雜湊不符'
    return json.loads((HERE / f'{name}.json').read_text())


def stage_train():
    pre = check('PREREG'); assert not (HERE / 'FROZEN.json').exists(), 'FROZEN.json 已存在'
    z, d, e, r50, rL, act, on = X.load()
    i14 = X.wslice(d, *WIN['train_actual_2015_2021'])[0]
    TR = ('train', 'train_synth_2008_2014')

    def evalc(c, rLx):
        eq, st, _ = run_cell(c, z, d, r50, rLx)
        o = block(eq, d, e, TR)
        eqa, _, _ = run_cell(c, z, d, r50, rLx, t_start=i14)
        o['train_actual_2015_2021'] = X.metrics(eqa, d, *WIN['train_actual_2015_2021'], e50=e)
        o['bears'] = {k: v for k, v in X.bears(eq, d).items() if k != '2022'}
        return o, st
    b, bst = evalc(BASE, rL)
    drag = X.lev_drag(d, e, rL, act, '2015-01-01', '2021-12-31')
    res = dict(base=b, cells={}, lev_drag_train_actual=drag)
    for nm, c in CELLS.items():
        o, st = evalc(c, rL)
        o['vs_base'] = {w: X.vs(o[w], b[w]) for w in ('train', 'train_synth_2008_2014', 'train_actual_2015_2021')}
        o['judge_train'] = X.judge(o['train'], b['train'])
        o['eligible'] = bool(o['judge_train']['pass_all'] and o['train_actual_2015_2021']['cagr'] >= b['train_actual_2015_2021']['cagr'])
        # 合成敏感度（只影響 2008～2014）
        sens = {}
        for sy in ('rate', 'cal'):
            _, _, _, _, rLs, _, _ = X.load(sy, gap_ann=drag['fee_fin_tracking_gap_pp'] / 100)
            eqs, _, _ = run_cell(c, z, d, r50, rLs)
            ms = X.metrics(eqs, d, *WIN['train'], e50=e)
            sens[sy] = dict(train_cagr=ms['cagr'], train_maxdd=ms['maxdd'], train_sharpe=ms['sharpe'], judge=X.judge(ms, b['train']))
        o['synth_sensitivity'] = sens
        res['cells'][nm] = o
        print(nm, 'train', round(o['train']['cagr'], 4), round(o['train']['maxdd'], 4), round(o['train']['sharpe'], 3), o['judge_train'], 'actual15-21', round(o['train_actual_2015_2021']['cagr'], 4), 'elig', o['eligible'], flush=True)
    print('BASE train', round(b['train']['cagr'], 4), round(b['train']['maxdd'], 4), round(b['train']['sharpe'], 3), 'actual', round(b['train_actual_2015_2021']['cagr'], 4))
    el = [k for k, v in res['cells'].items() if v['eligible']]
    pool = el if el else list(CELLS)
    pick = max(pool, key=lambda k: (res['cells'][k]['train']['sharpe'], res['cells'][k]['train']['cagr']))
    fr = dict(prereg_sha256=sha(HERE / 'PREREG.json'), frozen_cell=pick, cell=CELLS[pick], eligible_cells=el, no_eligible=not el,
              train_results=res, code_sha256={f: sha(HERE / f) for f in CODE})
    (HERE / 'FROZEN.json').write_text(json.dumps(fr, ensure_ascii=False, indent=1))
    (HERE / 'FROZEN.sha256').write_text(sha(HERE / 'FROZEN.json') + '  FROZEN.json\n')
    print('FROZEN', pick, 'eligible', el, sha(HERE / 'FROZEN.json'))


def stage_validate():
    fr = check('FROZEN'); assert not (HERE / 'results.json').exists(), 'results.json 已存在（驗證只做一次）'
    z, d, e, r50, rL, act, on = X.load()
    W = ('valid', 'full', 'train')
    eqb, bst, _ = run_cell(BASE, z, d, r50, rL)
    b = block(eqb, d, e, W); b['bears'] = X.bears(eqb, d); b['stats'] = bst
    res = dict(frozen_cell=fr['frozen_cell'], no_eligible=fr['no_eligible'], base=b, cells={},
               lev_drag=dict(valid=X.lev_drag(d, e, rL, act, *WIN['valid']), train_actual=X.lev_drag(d, e, rL, act, '2015-01-01', '2021-12-31'),
                             all_actual=X.lev_drag(d, e, rL, act, '2014-11-01', '2026-10-22'),
                             yearly={y: X.lev_drag(d, e, rL, act, f'{y}-01-01', f'{y}-12-31') for y in range(2015, 2027)}))
    rf_on = on / 245
    for nm, c in CELLS.items():                         # 全部格都算出來報告；判定只看凍結格
        eq, st, wL = run_cell(c, z, d, r50, rL)
        o = block(eq, d, e, W); o['bears'] = X.bears(eq, d); o['stats'] = st
        o['vs_base'] = {w: X.vs(o[w], b[w]) for w in W}
        o['judge_valid'] = X.judge(o['valid'], b['valid'])
        i0 = X.wslice(d, *WIN['valid'])[0]
        eqv, stv, _ = run_cell(c, z, d, r50, rL, t_start=i0); o['valid_window_stats'] = {k: stv[k] for k in ('n_rebal', 'forced_L_days', 'cash_used_days', 'turnover_L_ann', 'turnover_50_ann', 'cost_ann_pct', 'avg_wL', 'avg_wE', 'avg_wC', 'avg_sf', 'avg_index_exposure_ex_stocks')}
        # 敏感度（驗證窗）
        sens = {}
        for lab, kw in (('slip0.3%', dict(slip=0.003)), ('cash_rf_overnight', dict(rf_d=rf_on))):
            eqs, _, _ = run_cell(c, z, d, r50, rL, **kw); ms = X.metrics(eqs, d, *WIN['valid'])
            eqbs, _, _ = run_cell(BASE, z, d, r50, rL, **kw); mbs = X.metrics(eqbs, d, *WIN['valid'])
            sens[lab] = dict(cagr=ms['cagr'], maxdd=ms['maxdd'], sharpe=ms['sharpe'], judge=X.judge(ms, mbs))
        o['valid_sensitivity'] = sens
        # 20 條路徑
        P20 = {}
        for cap in (60, 80, 100, 150, 200):
            for s in (0, 1, 2, 3):
                eqp, _, _ = run_cell(c, z, d, r50, rL, cap=cap, shift=s); eqbp, _, _ = run_cell(BASE, z, d, r50, rL, cap=cap, shift=s)
                mv, mb = X.metrics(eqp, d, *WIN['valid']), X.metrics(eqbp, d, *WIN['valid'])
                mf, mbf = X.metrics(eqp, d, *WIN['full']), X.metrics(eqbp, d, *WIN['full'])
                P20[f'{cap}|{s}'] = dict(valid_cagr=mv['cagr'], base_valid_cagr=mb['cagr'], valid_maxdd=mv['maxdd'], base_valid_maxdd=mb['maxdd'],
                                         valid_sharpe=mv['sharpe'], base_valid_sharpe=mb['sharpe'], full_cagr=mf['cagr'], base_full_cagr=mbf['cagr'],
                                         full_maxdd=mf['maxdd'], base_full_maxdd=mbf['maxdd'], judge=X.judge(mv, mb))
        dv = [p['valid_cagr'] - p['base_valid_cagr'] for p in P20.values()]
        o['paths20'] = dict(n_pass_all=int(sum(p['judge']['pass_all'] for p in P20.values())), n_cagr_gt=int(sum(p['judge']['cagr_gt'] for p in P20.values())),
                            n_mdd_ok=int(sum(p['judge']['mdd_ok'] for p in P20.values())), n_sharpe_ge=int(sum(p['judge']['sharpe_ge'] for p in P20.values())),
                            valid_cagr_diff_pp=dict(min=min(dv) * 100, median=float(np.median(dv)) * 100, max=max(dv) * 100), paths=P20)
        res['cells'][nm] = o
        print(nm, 'valid', round(o['valid']['cagr'], 4), round(o['valid']['maxdd'], 4), round(o['valid']['sharpe'], 3), o['judge_valid'], 'paths', o['paths20']['n_pass_all'],
              'full', round(o['full']['cagr'], 4), round(o['full']['maxdd'], 4), flush=True)
    f = res['cells'][fr['frozen_cell']]
    res['verdict'] = dict(frozen=fr['frozen_cell'], judge_std=f['judge_valid'], paths20_pass=f['paths20']['n_pass_all'],
                          recommend=bool(f['judge_valid']['pass_all'] and f['paths20']['n_pass_all'] >= 15 and not fr['no_eligible']))
    print('BASE valid', round(b['valid']['cagr'], 4), round(b['valid']['maxdd'], 4), round(b['valid']['sharpe'], 3), 'VERDICT', res['verdict'])
    res['frozen_sha256'] = sha(HERE / 'FROZEN.json'); res['code_sha256'] = {x: sha(HERE / x) for x in CODE}
    (HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))


if __name__ == '__main__':
    {'prereg': stage_prereg, 'train': stage_train, 'validate': stage_validate}[sys.argv[1]]()
