#!/usr/bin/env python3
"""S103 衍生性商品疊加（台指期避險與槓桿）。三個階段，依序執行：
  python3 -I -B run_s103.py prereg     寫 PREREG.json／PREREG.sha256（格子、規則、窗、判準、假設、程式 sha256）。不算任何疊加績效。
  python3 -I -B run_s103.py train      核對 PREREG 雜湊 → 只算訓練窗（2008-01-02～2021-12-30）→ 依登記規則選格 → FROZEN.json／FROZEN.sha256。
  python3 -I -B run_s103.py validate   核對 FROZEN 雜湊 → 驗證窗（2021-12-30 收盤為基底～2026-10-22）一次 → 20 條路徑、敏感度、口數取整 → results.json。
                                       results.json 已存在就拒跑（驗證只做一次）。
前置：build_sleeves.py 產生 sleeves.npz（S75 v3 口徑引擎 NAV＋stockfrac；sat 0.70 標準路徑已逐日對過 v3）。"""
import sys, json, hashlib, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import futlib, overlay as O

CODE = ['build_sleeves.py', 'futlib.py', 'overlay.py', 'run_s103.py']
BASE_CFG = dict(im=0.15, mm=0.75, k=1.5, rf=0.0, fee=50.0, mult=200.0, slip=1.0, band=0.10)
WIN = {'train': ('2008-01-02', '2021-12-31'), 'train_2008_2015': ('2008-01-02', '2015-12-31'), 'train_2016_2021': ('2016-01-01', '2021-12-31'),
       'valid': ('2022-01-01', '2026-10-22'), 'full': ('2008-01-02', '2026-10-22')}
BASE = dict(sat=70, keep50=True, rule='none')
CELLS = {
    'F1a': {  # Beta 中性 alpha 帳戶
        'F1_N60': dict(sat=70, keep50=False, rule='hedge_s', bwin=60, note='衛星股票全留、0050 底倉改現金（不持有）；TX 空單＝β60(衛星對加權)×股票市值'),
        'F1_N120': dict(sat=70, keep50=False, rule='hedge_s', bwin=120, note='同上，β120'),
        'F1_H60': dict(sat=70, keep50=True, rule='hedge_nav', bwin=60, note='0050 底倉也留著一起對沖；TX 空單＝β60(帳戶淨值對加權)×淨值'),
        'F1_H120': dict(sat=70, keep50=True, rule='hedge_nav', bwin=120, note='同上，β120'),
    },
    'F1b': {  # 槓桿：用期貨增加多頭曝險
        'F1_X15': dict(sat=70, keep50=True, rule='lever', extra=0.5, note='原策略＋TX 多單 0.5×淨值（總曝險約 1.5 倍）；保證金由 0050 撥出'),
        'F1_X20': dict(sat=70, keep50=True, rule='lever', extra=1.0, note='原策略＋TX 多單 1.0×淨值（約 2 倍）'),
    },
    'F2': {  # 條件避險
        'F2_MA_50': dict(sat=70, keep50=True, rule='cond', signal='ma200', h=0.5, bwin=60, note='加權 < 200MA 時空 TX＝0.5×β60(淨值)×淨值'),
        'F2_MA_100': dict(sat=70, keep50=True, rule='cond', signal='ma200', h=1.0, bwin=60, note='加權 < 200MA 時 1.0×β60'),
        'F2_R20_50': dict(sat=70, keep50=True, rule='cond', signal='r20', h=0.5, bwin=60, note='加權 20 日報酬 < −8% 時 0.5×β60'),
        'F2_R20_100': dict(sat=70, keep50=True, rule='cond', signal='r20', h=1.0, bwin=60, note='20 日 < −8% 時 1.0×β60'),
        'F2_OR_50': dict(sat=70, keep50=True, rule='cond', signal='or', h=0.5, bwin=60, note='兩條件任一成立 0.5×β60'),
        'F2_OR_100': dict(sat=70, keep50=True, rule='cond', signal='or', h=1.0, bwin=60, note='任一成立 1.0×β60'),
    },
    'F3': {  # 資本效率：0050 底倉改 TX 多單
        'F3_S70': dict(sat=70, keep50=False, rule='base_fut', extra=0.0, note='衛星 70% 不變；0050（底倉＋空位）全改等值 TX 多單，釋出現金放貨幣（0 利率）'),
        'F3_S85_E100': dict(sat=85, keep50=False, rule='base_fut', extra=0.0, note='衛星 85%；其餘（15% 底倉＋空位）改 TX，總曝險 100%'),
        'F3_S85_E115': dict(sat=85, keep50=False, rule='base_fut', extra=0.15, note='衛星 85%＋TX 名目維持 30% 底倉（＋空位）→ 總曝險約 115%'),
        'F3_S100_E130': dict(sat=100, keep50=False, rule='base_fut', extra=0.30, note='衛星 100%＋TX 30%（＋空位）→ 約 130%；保證金只能來自空位現金，不足時口數被截斷'),
    },
}
SELECT_RULE = ('每個題組各自：合格＝訓練窗（2008-01-02～2021-12-31，連續帳戶）最大回撤 ≥ 基準訓練窗最大回撤 − 5pp；'
               '合格格中取訓練窗 Sharpe（日報酬、×√252、rf=0）最高者，平手取訓練窗年化較高者。無合格格時取 Sharpe 最高者並標記「無合格格」。'
               '兩個子窗（2008～2015、2016～2021）與基準的差只報告、不參與選擇。')
JUDGE = ('驗證窗（2021-12-30 收盤為基底～2026-10-22，凍結格一次）同時滿足：(1) 年化 > 基準年化；(2) 最大回撤 ≥ 基準最大回撤 − 5pp；(3) Sharpe > 基準 Sharpe。'
         '另報 20 條路徑（sat 對應的 5 種本金 × 位移 0～3，各路徑與同路徑基準比）三項全過的條數；建議採用需標準路徑通過且 20 條中 ≥ 15 條通過。')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load():
    z = np.load(HERE / 'sleeves.npz')
    d = pd.DatetimeIndex(z['dates']); fut, fst = futlib.build(d)
    e = z['e50']; r50 = np.r_[0.0, e[1:] / e[:-1] - 1]
    return z, d, fut, fst, r50


def wslice(d, a, b):
    """基底＝a 之前最後一個交易日（2008 窗用首日）；回傳 (i0, i1) 含頭尾。"""
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    i0 = int(np.searchsorted(d.values, a.to_datetime64(), side='left')); i0 = max(i0 - 1, 0)
    i1 = int(np.searchsorted(d.values, b.to_datetime64(), side='right')) - 1
    return i0, i1


def metrics(eq, d, a, b, ref=None, e50=None):
    i0, i1 = wslice(d, a, b)
    s = pd.Series(eq[i0:i1 + 1], index=d[i0:i1 + 1])
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    r = s.pct_change().dropna()
    dd = s / s.cummax() - 1
    yr = s.groupby(s.index.year).last(); prev = s.iloc[0]; yl = {}
    for y, v in yr.items():
        yl[int(y)] = float(v / prev - 1); prev = v
    first = d[i0 + 1].year if i0 > 0 else d[0].year
    yl = {y: v for y, v in yl.items() if y >= first}
    o = dict(cagr=float((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1), maxdd=float(dd.min()), dd_trough=str(dd.idxmin().date()),
             vol=float(r.std(ddof=1) * math.sqrt(252)), sharpe=float(r.mean() / r.std(ddof=1) * math.sqrt(252)),
             worst_year=min(yl.items(), key=lambda x: x[1]) if yl else None, yearly=yl)
    o['worst_year'] = dict(year=o['worst_year'][0], ret=o['worst_year'][1]) if o['worst_year'] else None
    if e50 is not None:
        q = pd.Series(e50[i0:i1 + 1], index=s.index); o['cagr_0050'] = float((q.iloc[-1] / q.iloc[0]) ** (1 / yrs) - 1); o['excess_vs_0050_pp'] = (o['cagr'] - o['cagr_0050']) * 100
    if ref is not None:
        rr = metrics(ref, d, a, b)
        o['vs_base'] = dict(cagr_pp=(o['cagr'] - rr['cagr']) * 100, maxdd_pp=(o['maxdd'] - rr['maxdd']) * 100, sharpe=o['sharpe'] - rr['sharpe'])
    return o


def run_cell(cell, z, fut, r50, key='100|0', cfg=None, **kw):
    k = f"s{cell['sat']}|{key}"
    nav = z['nav|' + k]; sf = z['sf|' + k]
    return O.run(cell, nav, sf, r50, fut, cfg or BASE_CFG, **kw)


def calls_in(st_full):
    return {x: st_full[x] for x in ('calls', 'calls_need_0050', 'calls_need_stock', 'margin_capped_days', 'n_trades', 'n_rolls', 'avg_abs_fexp', 'max_abs_fexp',
                                    'frac_days_fut_on', 'fexp_when_on_median', 'fexp_when_on_p10', 'max_loss_day_on_M') if x in st_full} | \
        {'fut_cost_frac_of_start': st_full['fut_cost'] / 1e6, 'etf_cost_frac_of_start': st_full['etf_cost'] / 1e6, 'ruin': st_full.get('ruin')}


def calls_window(cell, z, fut, r50, a, b, cfg=None):
    """窗內的追繳次數：從窗基底日起重新開帳戶跑（β 用完整歷史）。"""
    d = fut.index; i0, i1 = wslice(d, a, b)
    eq, fx, st = run_cell(cell, z, fut, r50, cfg=cfg, t_start=i0)
    return calls_in(st)


def stage_prereg():
    assert not (HERE / 'PREREG.json').exists(), 'PREREG.json 已存在'
    pre = dict(agent='S103', date='2026-11-27', title='衍生性商品疊加：台指期避險與槓桿（F1 Beta 中性／槓桿、F2 條件避險、F3 資本效率）',
               baseline='S75 v3 連續帳戶 V_BRIDGE（sat_w 0.70、W10、空位 0050、嚴成本），100 萬、位移 0；2008-01-02～2026-10-22，年化 23.50%、MDD −49.91%（build_sleeves.py 逐日核對）',
               cells=CELLS, base_cell=BASE, cfg=BASE_CFG, windows=WIN, select_rule=SELECT_RULE, judge=JUDGE,
               assumptions=dict(
                   futures='2016-10-06～2026-10-05 用 TX 實際近月（ChiJiun/tx-flow-trend-backtest 期交所每日行情），最後交易日前一交易日收盤換月；'
                           '之外（2008-01～2016-10-05、2026-10-06～10-22）r_fut＝r_加權(價格指數)＋g(曆月)，g＝2016-10～2026-09 實測每日「TX−加權」差的曆月平均（股息季節性＋基差）。'
                           '近似期間換月日＝每月第三個週三的前一交易日。',
                   costs='期交稅 0.002%/邊；手續費每口每邊 50 元（TX 乘數 200）；滑價 1 點/邊；換月 2 邊；0050 撥補保證金每邊 0.1%；股票被迫賣出補繳 0.4%。',
                   margin='原始保證金率 15%（repo 無期交所歷年保證金表，全期取保守 15%），維持保證金＝原始×0.75；保證金帳戶在換月日／口數變動時撥到 1.5×原始，'
                          '跌破維持即追繳（補回 1.5×原始）；先用現金，不足賣 0050、再不足賣股票（T+2 交割，視為斷頭風險事件）。',
                   cash='剩餘現金 0 利率（敏感度 1%）；Sharpe rf=0。',
                   engine='引擎股票部位與比例（stockfrac）照 S75 v3 原樣；疊加只換掉非股票部位（0050／現金）並加期貨。r_s 由 NAV 與 0050 報酬拆出。',
                   contracts='主計算用連續名目；口數取整另報（TX 200、小台 50、微台 10 元/點；微台 2024-07 才上市）。',
                   rebalance='期貨目標每日算，偏離 >10% 或換月、方向改變才交易；β＝滾動 OLS(日報酬對加權)，限 [0,2]，資料不足時 1。',
                   signal='F2 訊號用 t 日加權收盤（13:30），t 日期貨收盤（13:45）成交，承擔 t+1 起報酬。'),
               sensitivities=['fee×2（100 元）', 'rf 1%', '原始保證金 12%', '撥補倍數 k 1.2／2.0', '小台成本（乘數 50）', '全期用近似期貨（量測近似誤差）', '滑價 3 點'],
               code_sha256={f: sha(HERE / f) for f in CODE}, sleeves_sha256=sha(HERE / 'sleeves.npz'))
    txt = json.dumps(pre, ensure_ascii=False, indent=1)
    (HERE / 'PREREG.json').write_text(txt, encoding='utf-8')
    (HERE / 'PREREG.sha256').write_text(hashlib.sha256(txt.encode()).hexdigest() + '  PREREG.json\n')
    print('PREREG 寫入', hashlib.sha256(txt.encode()).hexdigest())


def check_hash(name):
    h = (HERE / f'{name}.sha256').read_text().split()[0]
    assert sha(HERE / f'{name}.json') == h, f'{name}.json 雜湊不符'
    return json.loads((HERE / f'{name}.json').read_text())


def stage_train():
    pre = check_hash('PREREG')
    assert not (HERE / 'FROZEN.json').exists(), 'FROZEN.json 已存在'
    for f in ('futlib.py', 'overlay.py', 'build_sleeves.py'):
        assert sha(HERE / f) == pre['code_sha256'][f], f'{f} 在登記後被改過'
    z, d, fut, fst, r50 = load()
    base_eq, _, _ = run_cell(BASE, z, fut, r50)
    assert np.allclose(base_eq / base_eq[0], z['nav|s70|100|0'] / z['nav|s70|100|0'][0], rtol=1e-10)
    TR = ('train', 'train_2008_2015', 'train_2016_2021')
    tab = {'BASE': {w: metrics(base_eq, d, *WIN[w], e50=z['e50']) for w in TR}}
    tab['BASE']['calls_train'] = None
    frozen = dict(stage='train', prereg_sha256=(HERE / 'PREREG.sha256').read_text().split()[0], select_rule=SELECT_RULE, families={})
    for fam, cells in CELLS.items():
        rows = {}
        for nm, c in cells.items():
            eq, fx, st = run_cell(c, z, fut, r50)
            rows[nm] = {w: metrics(eq, d, *WIN[w], ref=base_eq, e50=z['e50']) for w in TR}
            rows[nm]['calls_train'] = calls_window(c, z, fut, r50, *WIN['train'])
            for w in TR:
                rows[nm][w].pop('yearly')
        bmdd = tab['BASE']['train']['maxdd']
        ok = [n for n in rows if rows[n]['train']['maxdd'] >= bmdd - 0.05]
        pool = ok if ok else list(rows)
        best = max(pool, key=lambda n: (round(rows[n]['train']['sharpe'], 10), rows[n]['train']['cagr']))
        frozen['families'][fam] = dict(chosen=best, eligible=ok, no_eligible=not ok, cell=cells[best])
        tab.update(rows)
        print(fam, '→', best, {n: (round(r['train']['cagr'], 4), round(r['train']['maxdd'], 4), round(r['train']['sharpe'], 3)) for n, r in rows.items()}, flush=True)
    for w in TR:
        tab['BASE'][w].pop('yearly')
    print('BASE', {w: (round(tab['BASE'][w]['cagr'], 4), round(tab['BASE'][w]['maxdd'], 4), round(tab['BASE'][w]['sharpe'], 3)) for w in TR})
    frozen['train_table'] = tab
    frozen['futures_calibration'] = fst
    txt = json.dumps(frozen, ensure_ascii=False, indent=1)
    (HERE / 'FROZEN.json').write_text(txt, encoding='utf-8')
    (HERE / 'FROZEN.sha256').write_text(hashlib.sha256(txt.encode()).hexdigest() + '  FROZEN.json\n')
    print('FROZEN 寫入')


def judge(m, b):
    c1 = m['cagr'] > b['cagr']; c2 = m['maxdd'] >= b['maxdd'] - 0.05; c3 = m['sharpe'] > b['sharpe']
    return dict(cagr_gt=bool(c1), mdd_ok=bool(c2), sharpe_gt=bool(c3), pass_all=bool(c1 and c2 and c3))


CONTRACTS = {'TX': 200.0, 'MTX': 50.0, 'TMF': 10.0}


def stage_validate():
    pre = check_hash('PREREG'); fr = check_hash('FROZEN')
    assert not (HERE / 'results.json').exists(), 'results.json 已存在：驗證只做一次'
    for f in ('futlib.py', 'overlay.py', 'build_sleeves.py'):
        assert sha(HERE / f) == pre['code_sha256'][f], f'{f} 在登記後被改過'
    z, d, fut, fst, r50 = load()
    base_eq, _, _ = run_cell(BASE, z, fut, r50)
    res = dict(stage='validate', prereg_sha256=fr['prereg_sha256'], frozen_sha256=(HERE / 'FROZEN.sha256').read_text().split()[0],
               baseline=dict(full=metrics(base_eq, d, *WIN['full'], e50=z['e50']), valid=metrics(base_eq, d, *WIN['valid'], e50=z['e50'])),
               futures_calibration=fst, families={}, all_cells_report_only={})
    keys = sorted({k.split('|', 2)[2] for k in z.files if k.startswith('nav|s70|')})
    idx_end = float(fut['twii'].iloc[-1])
    for fam, f in fr['families'].items():
        nm = f['chosen']; c = f['cell']
        eq, fx, st = run_cell(c, z, fut, r50)
        o = dict(chosen=nm, cell=c, no_eligible=f['no_eligible'])
        o['valid'] = metrics(eq, d, *WIN['valid'], ref=base_eq, e50=z['e50'])
        o['full'] = metrics(eq, d, *WIN['full'], ref=base_eq, e50=z['e50'])
        o['judge_std'] = judge(o['valid'], res['baseline']['valid'])
        o['calls_full'] = calls_in(st)
        o['calls_valid'] = calls_window(c, z, fut, r50, *WIN['valid'])
        # 20 條路徑
        P20 = {}
        for k in keys:
            be, _, _ = O.run(BASE, z['nav|s70|' + k], z['sf|s70|' + k], r50, fut, BASE_CFG)
            ce, _, cst = run_cell(c, z, fut, r50, key=k)
            mv = metrics(ce, d, *WIN['valid'], ref=be); mb = metrics(be, d, *WIN['valid'])
            mf = metrics(ce, d, *WIN['full'], ref=be)
            P20[k] = dict(valid_cagr=mv['cagr'], base_valid_cagr=mb['cagr'], valid_maxdd=mv['maxdd'], base_valid_maxdd=mb['maxdd'], valid_sharpe=mv['sharpe'],
                          base_valid_sharpe=mb['sharpe'], full_cagr=mf['cagr'], full_vs_base_pp=mf['vs_base']['cagr_pp'], judge=judge(mv, mb), calls=cst['calls'],
                          calls_need_stock=cst['calls_need_stock'])
        vp = [p['valid_cagr'] - p['base_valid_cagr'] for p in P20.values()]
        o['paths20'] = dict(n=len(P20), n_pass_all=int(sum(p['judge']['pass_all'] for p in P20.values())),
                            n_cagr_gt=int(sum(p['judge']['cagr_gt'] for p in P20.values())), n_sharpe_gt=int(sum(p['judge']['sharpe_gt'] for p in P20.values())),
                            valid_cagr_diff_pp=dict(min=min(vp) * 100, median=float(np.median(vp)) * 100, max=max(vp) * 100),
                            full_cagr=dict(min=min(p['full_cagr'] for p in P20.values()), median=float(np.median([p['full_cagr'] for p in P20.values()])), max=max(p['full_cagr'] for p in P20.values())),
                            paths=P20)
        o['recommend_rule_pass'] = bool(o['judge_std']['pass_all'] and o['paths20']['n_pass_all'] >= 15)
        # 敏感度（標準路徑）
        sens = {}
        for lab, upd in [('fee_x2', dict(fee=100.0)), ('rf_1pct', dict(rf=0.01)), ('im_12pct', dict(im=0.12)), ('k_1.2', dict(k=1.2)), ('k_2.0', dict(k=2.0)),
                         ('MTX_cost', dict(mult=50.0)), ('slip_3pt', dict(slip=3.0))]:
            cfg = dict(BASE_CFG, **upd)
            e2, _, s2 = run_cell(c, z, fut, r50, cfg=cfg)
            sens[lab] = dict(valid=metrics(e2, d, *WIN['valid'], ref=base_eq), full_cagr=metrics(e2, d, *WIN['full'])['cagr'], calls=s2['calls'], calls_need_stock=s2['calls_need_stock'])
            sens[lab]['valid'].pop('yearly')
        fa = fut.copy(); fa['r_fut'] = fa['r_twii'] + pd.Series(fa.index.month, index=fa.index).map({int(k): v / 1e4 for k, v in fst['g_month_daily_bps'].items()}).to_numpy()
        fa.iloc[0, fa.columns.get_loc('r_fut')] = 0.0
        e3, _, _ = O.run(c, z[f"nav|s{c['sat']}|100|0"], z[f"sf|s{c['sat']}|100|0"], r50, fa, BASE_CFG)
        m3 = metrics(e3, d, *WIN['valid']); m3.pop('yearly')
        sens['approx_futures_everywhere'] = dict(valid=m3, full_cagr=metrics(e3, d, *WIN['full'])['cagr'],
                                                 note='驗證窗也改用近似期貨；與實測版差＝近似誤差的量級（2008～2016 用的是近似）')
        o['sensitivity'] = sens
        # 口數取整：驗證窗起點以本金 X 開帳戶
        i0, _ = wslice(d, *WIN['valid'])
        er, _, sr = run_cell(c, z, fut, r50, t_start=i0)
        mref = metrics(er, d, *WIN['valid'])
        rnd = {'continuous': dict(cagr=mref['cagr'], maxdd=mref['maxdd'])}
        for cn, mult in CONTRACTS.items():
            for cap in (1e6, 3e6, 1e7, 3e7):
                e4, _, s4 = run_cell(c, z, fut, r50, cfg=dict(BASE_CFG, mult=mult), t_start=i0, capital=cap, cmult=mult)
                m4 = metrics(e4, d, *WIN['valid'])
                rnd[f'{cn}|{int(cap / 1e4)}萬'] = dict(cagr=m4['cagr'], maxdd=m4['maxdd'], cagr_diff_vs_continuous_pp=(m4['cagr'] - mref['cagr']) * 100,
                                                     avg_abs_fexp=s4['avg_abs_fexp'], frac_days_fut_on=s4['frac_days_fut_on'], calls=s4['calls'])
        med = st['fexp_when_on_median'] if st['fexp_when_on_median'] > 0 else None
        mincap = {}
        for cn, mult in CONTRACTS.items():
            cv = mult * idx_end
            mincap[cn] = dict(contract_value_2026_10_22=cv, margin_15pct=cv * 0.15,
                              min_capital_1_contract=(cv / med) if med else None, min_capital_5_contracts_err_le_10pct=(5 * cv / med) if med else None)
        o['contracts'] = dict(rounding_valid_window=rnd, min_capital=mincap, fexp_when_on_median=med, twii_2026_10_22=idx_end,
                              note='微台（TMF）2024-07-29 才上市；2022～2024-07 的微台結果是假設可交易。最低資金＝契約值 ÷ 期貨占淨值比例（啟用時中位數）。')
        for k2 in ('valid', 'full'):
            o[k2 + '_yearly'] = o[k2].pop('yearly')
        res['families'][fam] = o
        print(fam, nm, 'valid', round(o['valid']['cagr'], 4), round(o['valid']['maxdd'], 4), round(o['valid']['sharpe'], 3), o['judge_std'], 'paths', o['paths20']['n_pass_all'], flush=True)
    # 其他格（只報告，不參與判定）
    for fam, cells in CELLS.items():
        for nm, c in cells.items():
            eq, _, st = run_cell(c, z, fut, r50)
            mv = metrics(eq, d, *WIN['valid'], ref=base_eq, e50=z['e50']); mf = metrics(eq, d, *WIN['full'], ref=base_eq, e50=z['e50'])
            res['all_cells_report_only'][nm] = dict(valid={k: mv[k] for k in mv if k != 'yearly'}, full={k: mf[k] for k in mf if k != 'yearly'}, full_yearly=mf['yearly'],
                                                    calls_full=calls_in(st), calls_valid=calls_window(c, z, fut, r50, *WIN['valid']))
    bv = res['baseline']['valid']; bv_y = bv.pop('yearly'); res['baseline']['valid_yearly'] = bv_y
    res['baseline']['full_yearly'] = res['baseline']['full'].pop('yearly')
    # 官方引擎（2015-07 起）對照：只報告基準
    try:
        off = pd.read_csv(HERE.parents[3] / 'xq_scripts' / 'site' / 'navs_official.csv.gz', index_col=0, parse_dates=True)
        res['official_engine_2015_07'] = {c: dict(cagr=float((off[c].iloc[-1] / off[c].iloc[0]) ** (365.25 / (off.index[-1] - off.index[0]).days) - 1),
                                                  maxdd=float((off[c] / off[c].cummax() - 1).min()), end=str(off.index[-1].date())) for c in off.columns}
    except Exception as ex:
        res['official_engine_2015_07'] = str(ex)
    (HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    print('results.json 寫入')


if __name__ == '__main__':
    {'prereg': stage_prereg, 'train': stage_train, 'validate': stage_validate}[sys.argv[1]]()
