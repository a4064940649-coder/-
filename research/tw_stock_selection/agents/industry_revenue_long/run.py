#!/usr/bin/env python3
"""產業營收循環長期驗證。分階段執行（每階段只計算／寫出允許看的期間）：
  python3 -I -B run.py prereg    # 只寫 PREREG
  python3 -I -B run.py select    # 挑選期 2006-07～2013-12（另報剔除 2013）→ 決定各檢定的指標，寫入 CHOSEN
  python3 -I -B run.py validate  # 驗證期 2014-01～2020-12（全部組態都報；判定只看 CHOSEN）
  python3 -I -B run.py holdout   # 保留期 2021-03～2026-10（最後才看）＋最終判定
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import irlib as I
L = I.L
OUT = I.HERE / 'results.json'

PREREG = {
    '寫於看任何報酬之前': True,
    '產業別': 'stock_list.csv 的 industry（現行分類往回套，含 2023 年新設的綠能環保／數位雲端／居家生活／運動休閒；分類偏誤）。'
             '面板中不在 stock_list 的已下市股：傳統產業依代碼前兩碼回推（同 industry 組 ilib.PREFIX_IND）；已下市電子股無法回推 → 無產業。'
             '金融保險業排除（營收定義不可比）。stock_list 的 industry 已是電子 8 細類（半導體、電腦週邊、光電、通信網路、電子零組件、電子通路、資訊服務、其他電子），'
             '此為主設定；穩健性：把 8 細類＋數位雲端合併成「電子工業」（粗分類）重跑被選中的組態。',
    '營收樣本': '長期月營收 tw_revenue.csv（上市，sii 優先去重），只用面板內代碼；營收 ≤ 0 視為缺值。',
    '指標（產業 g、月 m＝換股月的前一月）': {
        'YOY3': '產業內配對公司（m, m-1, m-2, m-12, m-13, m-14 皆有值）近 3 月營收加總 ÷ 去年同 3 月加總 − 1（加總後算）',
        'DIFF': '同一配對樣本中，個股近 3 月營收年增 > 0 的比例',
        'ACC': 'YOY3(m) − YOY3(m−3)',
        '最少公司數': '配對公司 ≥ 5，否則該產業該月無訊號',
        '方向': '三者都是越高越好（景氣向上）',
    },
    '換股與成交': 'lhlib M2：每月第一個 ≥ 11 日交易日收盤算訊號、隔日開盤成交、開盤漲停（2015-06 前 7%、之後 10%）且未持有者跳過遞補；'
              '成本 exlib：買 0.0855%+滑價、賣 0.0855%+0.3% 稅+滑價，主設定滑價 0.1%；穩健性 0.3% 與壓力（買賣各 0.5%）。',
    '基準': {'原主策略': 'lhlib M2 n30（0.5×52 週高百分位＋0.5×近 3 月營收年增率平均百分位，前 300 流動性池），滑價 0.1%',
           '0050': '含息', '同池等權': '原主策略前 300 流動性池等權（M2 成交、不計成本）'},
    '檢定': {
        'A_liq_{X}': '產業輪動：依指標 X 選前 3 產業（產業須有 ≥ 5 檔合格股：60 日均值 ≥ 1,000 萬、價 ≥ 10、非處置、隔日可交易），'
                     '產業內 60 日均值前 10 檔，全體等權（約 30 檔）',
        'A_main_{X}': '同上，但產業內取原主策略分數前 10（只在前 300 流動性池內；產業須有 ≥ 5 檔池內股票）',
        'B_{X}': '原主策略，但只在指標 X 前 1/2 的產業（名次 ≤ 有訊號產業數/2）中選股；無產業訊號的股票（金融、無法分類、公司數不足）中性保留，不排除；'
                 '百分位在遮罩後的池內重算；前 30 檔',
        'C_{X}': '原主策略兩腿＋產業腿三腿等權：(52 週高百分位＋營收腿百分位＋所屬產業指標 X 的百分位)/3，產業腿缺值填 0.5；前 30 檔',
        'X ∈': ['YOY3', 'DIFF', 'ACC'],
    },
    '期間': {
        'SEL': '挑選期：2006-07 換股（ACC 最早可算）之後的第一個完整持有日 ～ 2013-12-31',
        'SEL_ex2013': '挑選期剔除 2013（IFRS 斷點）：同起點 ～ 2012-12-31',
        'VAL': '驗證期 2014-01-01～2020-12-31',
        'VAL_clean': '驗證期去掉 IFRS 影響：2014-07-01～2020-12-31（ACC 用到的 m−5 月在 2014-01 之後）—診斷',
        'HOLD': '保留期 2021-03-01～2026-10（最後才看）',
        '空頭年': [2008, 2011, 2015, 2018, 2022],
    },
    '挑選規則': '只用 SEL：A 的 6 個組態（liq/main × 3 指標）選 1 個、B 的 3 個選 1 個、C 的 3 個選 1 個；'
              '準則＝SEL 期扣成本月報酬對原主策略的配對 t 最大者。SEL_ex2013 只報告不用於挑選。所有組態在每一期都報告。',
    '判定（A/B/C 各自，只看被選中的組態）': {
        '可用': 'VAL 與 HOLD 都：對原主策略配對 t ≥ 1.5 且月均差 > 0',
        '待觀察': '不符可用，但 VAL 與 HOLD 都：對同池等權 t ≥ 1.5 且月均差 > 0',
        '不可用': '其餘',
    },
    'D 上下游領先': {
        '配對（依現行 industry 名稱，事前指定）': {
            'D1a 半導體業 → 電子零組件業': '晶片／晶圓是被動元件、PCB、連接器模組的同期需求指標，半導體出貨先反映終端備貨',
            'D1b 半導體業 → 電腦及週邊設備業': 'CPU/GPU/記憶體等晶片是 PC、伺服器組裝的上游料件',
            'D2 電子通路業 → 其他電子業': '通路商（IC 代理）營收＝下游製造商的料件採購，是代工／組裝（其他電子含鴻海等）出貨的前置',
            'D3 塑膠工業 → 紡織纖維': '石化中間原料（PTA、EG、尼龍原料）為聚酯／尼龍纖維上游，台塑化纖體系上下游關係明確',
        },
        '訊號': '上游產業 ACC（換股日已知的 m 月）',
        '目標': '下游產業等權報酬（產業內合格股：60 日均值 ≥ 1,000 萬、價 ≥ 10、隔日可交易），從換股隔日開盤到 h 個換股後的隔日開盤，h=1,2,3；'
                '主目標＝減去全體合格股等權報酬（超額）；另報原始報酬',
        '統計': '各期間內時間序列 Spearman IC；t＝排名 OLS 的 Newey-West t（lag h−1）；另報控制下游自身 ACC 後上游係數的 NW t',
        '判定': 'SEL 期 IC > 0 且 t ≥ 2 才進入驗證；VAL t ≥ 2 且 HOLD t ≥ 1.5（同號）→ 可用；只過 VAL → 待觀察；其餘不可用。'
                '4 對 × 3 期 = 12 個檢定，5% 水準下約 0.6 個假陽性。',
    },
    '已知偏誤': ['現行產業分類往回套', '已下市電子股無產業（A 選不到它們→A 在電子產業有存活者偏誤，向上）',
              '2013 起營收只剩存活公司（產業營收也是存活者加總）', '2013 IFRS 合併口徑斷點（2013-02～2014-06 的指標失真）',
              '2019～2023 價格面板只有存活者', 'disposal 濾網 2021-09 前無資料'],
}

CONFIGS = [f'{f}_{x}' for f in ('A_liq', 'A_main', 'B', 'C') for x in ('YOY3', 'DIFF', 'ACC')]
FAMILY = {'A': [c for c in CONFIGS if c.startswith('A_')], 'B': [c for c in CONFIGS if c.startswith('B_')], 'C': [c for c in CONFIGS if c.startswith('C_')]}
KEYS = ('CAGR', 'maxDD', 'vol', 'n_months', 't_vs_0050_m', 'excess_vs_0050_pct_m', 't_vs_EW_m', 'excess_vs_EW_pct_m', 't_vs_main_m',
        'excess_vs_main_pct_m', 'hit_vs_main_m', 'corr_with_main_m', 'main_CAGR', 'main_maxDD', '0050_CAGR', '0050_maxDD', 'EW_CAGR')


def load_res():
    return json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else {}


def save_res(r):
    OUT.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str), encoding='utf-8')


def windows(S, stage):
    D = S.DATES
    d_jul = D[D >= pd.Timestamp('2006-07-01')][0]
    s0 = L.IDX[L.pos(d_jul) + 2]
    W = {}
    if stage in ('select', 'validate', 'holdout'):
        W['SEL'] = [(s0, '2013-12-31')]; W['SEL_ex2013'] = [(s0, '2012-12-31')]
        for y in (2008, 2011):
            W[f'bear_{y}'] = [(f'{y}-01-01', f'{y}-12-31')]
    if stage in ('validate', 'holdout'):
        W['VAL'] = [('2014-01-01', '2020-12-31')]; W['VAL_clean'] = [('2014-07-01', '2020-12-31')]
        for y in (2015, 2018):
            W[f'bear_{y}'] = [(f'{y}-01-01', f'{y}-12-31')]
    if stage == 'holdout':
        W['HOLD'] = [('2021-03-01', '2026-10-31')]; W['bear_2022'] = [('2022-01-01', '2022-12-31')]
    return W


class Ctx:
    def __init__(self, coarse=False):
        self.S = L.Setup(L.yoy_long(), start='2005-01-01')
        S = self.S
        self.IND, self.npref = I.industry_map(coarse=coarse)
        self.Lv = I.revenue_levels()
        self.T = I.industry_indicators(self.Lv, self.IND)
        self.indd = {x: I.ind_on_dates(self.T[x], S.dates) for x in ('YOY3', 'DIFF', 'ACC')}
        self.t0 = L.pos(S.dates[0]); self.t1 = L.pos(S.dates[-1])
        self.el = S.t300 & S.s52.notna() & ~S.inD & S.yoy3.notna()
        self.r52 = S.s52.where(self.el).rank(axis=1, pct=True); self.rrv = S.yoy3.where(self.el).rank(axis=1, pct=True)
        self.mainscore = 0.5 * self.r52 + 0.5 * self.rrv
        v60 = L.v60_fix.reindex(S.DATES); px = L.C.reindex(S.DATES)
        self.base = (v60 >= 1e7) & (px >= 10) & ~S.inD
        self.gcode = pd.Series(self.IND.reindex(L.COLS).to_numpy(), index=L.COLS)

    def stock_ind(self, x):
        """每個換股日、每檔股票所屬產業的指標值。"""
        tab = self.indd[x]
        arr = np.full((len(self.S.DATES), len(L.COLS)), np.nan)
        g = self.gcode.to_numpy()
        for j, name in enumerate(tab.columns):
            arr[:, g == name] = tab[name].to_numpy()[:, None]
        return pd.DataFrame(arr, index=self.S.DATES, columns=L.COLS)

    def events(self, cfg):
        S = self.S
        fam, x = cfg.rsplit('_', 1)
        if fam == 'MAIN':
            ev, _ = S.selection(30); return ev, {}
        if fam in ('A_liq', 'A_main'):
            within = 'liq' if fam == 'A_liq' else 'main'
            base = self.base if within == 'liq' else self.el
            ev, nb, ch = I.select_rotation(S, self.indd[x], self.IND, within, base=base, mainscore=self.mainscore)
            return ev, {'skipped_limit_up': nb, 'chosen_industries': ch}
        si = self.stock_ind(x)
        if fam == 'B':
            tab = self.indd[x]
            rk = tab.rank(axis=1, ascending=False); K = tab.notna().sum(axis=1)
            top = rk.le(K / 2, axis=0)
            g = self.gcode.to_numpy(); keep = np.ones((len(S.DATES), len(L.COLS)), bool)
            for name in tab.columns:
                has = tab[name].notna().to_numpy(); t = top[name].to_numpy()
                keep[:, g == name] = (~has | t)[:, None]
            el = self.el & pd.DataFrame(keep, index=S.DATES, columns=L.COLS)
            r52 = S.s52.where(el).rank(axis=1, pct=True); rrv = S.yoy3.where(el).rank(axis=1, pct=True)
            sc = 0.5 * r52 + 0.5 * rrv
            ev, nb = I.select_generic(S, sc, el, 30)
            frac = float((el.sum(axis=1) / self.el.sum(axis=1)).mean())
            return ev, {'skipped_limit_up': nb, 'mean_pool_frac_kept': round(frac, 3)}
        if fam == 'C':
            rind = si.where(self.el).rank(axis=1, pct=True)
            rind = rind.where(rind.notna() | ~self.el, 0.5)
            sc = (self.r52 + self.rrv + rind) / 3
            ev, nb = I.select_generic(S, sc, self.el, 30)
            return ev, {'skipped_limit_up': nb}
        raise ValueError(cfg)


def short(o):
    if o is None: return None
    d = {k: o[k] for k in KEYS if k in o}
    d['yearly_pct'] = o.get('yearly_pct'); d['0050_yearly_pct'] = o.get('0050_yearly_pct'); d['EW_yearly_pct'] = o.get('EW_yearly_pct')
    d['main_yearly_pct'] = o.get('main_yearly_pct')
    return d


def run_strats(ctx, stage, cfgs, slips=(0.001,), tag=''):
    S = ctx.S; W = windows(S, stage)
    evm, pools = S.selection(30)
    ew, _ = L.sim_portfolio(pools, 0.0, 0.0, ctx.t0, ctx.t1)
    mains = {}
    for slip in set(slips) | {0.001}:
        b, s = L.cost_rates(slip)
        mains[slip], _ = L.sim_portfolio(evm, b, s, ctx.t0, ctx.t1)
    main = mains[0.001]
    out = {}
    out['MAIN_n30'] = {'windows': {k: short(I.full_stats(main, w, S.DATES, ew)) for k, w in W.items()}}
    for cfg in cfgs:
        ev, info = ctx.events(cfg)
        for slip in slips:
            bb, ss = L.cost_rates(slip)
            nav, tov = L.sim_portfolio(ev, bb, ss, ctx.t0, ctx.t1)
            key = f'{cfg}{tag}' + ('' if slip == 0.001 else f'_slip{slip}')
            o = {'turnover_per_rebalance': round(tov, 3), 'rebalances': len(ev),
                 'mean_names': round(float(np.mean([len(v[1]) for v in ev.values()])), 1) if ev else 0,
                 'paired_vs_main_at_same_cost': True,
                 'windows': {k: short(I.full_stats(nav, w, S.DATES, ew, mains[slip])) for k, w in W.items()}}
            for k, v in info.items():
                if k != 'chosen_industries': o[k] = v
            if 'chosen_industries' in info:
                ch = info['chosen_industries']; o['industry_pick_freq'] = {}
                for wk in ('SEL', 'VAL', 'HOLD'):
                    if wk not in W: continue
                    a, z = W[wk][0]
                    lst = [g for dd, gs in ch.items() if pd.Timestamp(a) <= pd.Timestamp(dd) <= pd.Timestamp(z) for g in gs]
                    o['industry_pick_freq'][wk] = pd.Series(lst).value_counts().head(8).to_dict() if lst else {}
            out[key] = o
            w0 = o['windows']
            print(key, {k: (v['CAGR'], v['maxDD'], v.get('t_vs_0050_m'), v.get('t_vs_EW_m'), v.get('t_vs_main_m'))
                        for k, v in w0.items() if v and not k.startswith('bear')}, flush=True)
    return out


def leadlag(ctx, stage):
    S = ctx.S; D = S.DATES
    exs = [L.pos(d) + 1 for d in S.dates]
    g = ctx.gcode.to_numpy()
    # 每個換股日 k：合格股（base & 隔日可交易），持有到 k+h 的隔日開盤
    acc = ctx.indd['ACC']
    rows = []
    for k in range(len(S.dates) - 1):
        ex = exs[k]
        trad = np.isfinite(L.OPEN_RAW[ex]) & np.isfinite(L.ADJ_RAW[ex])
        ok = ctx.base.iloc[k].to_numpy() & trad & pd.notna(g)
        rec = {'date': D[k]}
        for h in (1, 2, 3):
            if k + h >= len(exs): continue
            r = L.AO[exs[k + h]] / L.AO[ex] - 1
            okh = ok & np.isfinite(r)
            rec[f'mkt_{h}'] = float(np.mean(r[okh])) if okh.sum() > 0 else np.nan
            for dn in ('電子零組件業', '電腦及週邊設備業', '其他電子業', '紡織纖維'):
                m = okh & (g == dn)
                rec[f'{dn}_{h}'] = float(np.mean(r[m])) if m.sum() >= 3 else np.nan
        rows.append(rec)
    R = pd.DataFrame(rows).set_index('date')
    pairs = {'D1a': ('半導體業', '電子零組件業'), 'D1b': ('半導體業', '電腦及週邊設備業'), 'D2': ('電子通路業', '其他電子業'), 'D3': ('塑膠工業', '紡織纖維')}
    W = windows(S, stage)
    out = {}
    for pk, (up, dn) in pairs.items():
        out[pk] = {'up': up, 'down': dn}
        for wk in ('SEL', 'SEL_ex2013', 'VAL', 'VAL_clean', 'HOLD'):
            if wk not in W: continue
            a, z = W[wk][0]
            sub = {}
            for h in (1, 2, 3):
                for tgt in ('excess', 'raw'):
                    y = R[f'{dn}_{h}'] - (R[f'mkt_{h}'] if tgt == 'excess' else 0)
                    df = pd.DataFrame({'x': acc[up].reindex(R.index), 'own': acc[dn].reindex(R.index), 'y': y})
                    df = df[(df.index >= pd.Timestamp(a)) & (df.index <= pd.Timestamp(z))].dropna()
                    if len(df) < 24: continue
                    rk = df.rank()
                    ic = float(rk['x'].corr(rk['y']))
                    t = I.nw_t(rk['y'], rk['x'], h - 1)
                    tp = I.nw_t(rk['y'], rk[['x', 'own']], h - 1, coef=1)
                    ic_own = float(rk['own'].corr(rk['y']))
                    sub[f'h{h}_{tgt}'] = {'n': len(df), 'IC': round(ic, 3), 't_NW': round(t, 2), 't_up_given_own': round(tp, 2), 'IC_own_ACC': round(ic_own, 3)}
            out[pk][wk] = sub
    return out


def main():
    stage = sys.argv[1]
    res = load_res()
    if stage == 'prereg':
        res = {'PREREG': PREREG}; save_res(res); print('PREREG written'); return
    assert 'PREREG' in res
    ctx = Ctx()
    res['meta'] = {'n_classified_cols': int(len(ctx.IND)), 'n_prefix_mapped_delisted': int(ctx.npref),
                   'n_unclassified_or_finance': int(len(L.COLS) - len(ctx.IND)),
                   'industries_with_signal_by_year': {int(y): round(float(v), 1) for y, v in ctx.T['YOY3'].notna().sum(axis=1).groupby(ctx.T['YOY3'].index.year).mean().items()},
                   'median_industry_YOY3_by_year': {int(y): round(float(v), 3) for y, v in ctx.T['YOY3'].median(axis=1).groupby(ctx.T['YOY3'].index.year).mean().items()}}
    if stage == 'select':
        out = run_strats(ctx, 'select', CONFIGS)
        res['SELECT'] = out
        ch = {}
        for fam, cs in FAMILY.items():
            tv = {c: out[c]['windows']['SEL']['t_vs_main_m'] for c in cs}
            ch[fam] = {'chosen': max(tv, key=tv.get), 'SEL_t_vs_main': tv}
        res['CHOSEN'] = ch
        res['D_SELECT'] = leadlag(ctx, 'select')
        adv = []
        for pk, v in res['D_SELECT'].items():
            for hk, st in v['SEL'].items():
                if hk.endswith('excess') and st['IC'] > 0 and st['t_NW'] >= 2: adv.append(f'{pk}_{hk}')
        res['D_ADVANCE'] = adv
        save_res(res); print(json.dumps(ch, ensure_ascii=False)); print('D advance', adv); return
    if stage == 'validate':
        out = run_strats(ctx, 'validate', CONFIGS)
        res['VALIDATE'] = out
        chosen = [v['chosen'] for v in res['CHOSEN'].values()]
        res['VALIDATE_robust'] = run_strats(ctx, 'validate', chosen, slips=(0.003, 'stress'))
        cc = Ctx(coarse=True)
        res['VALIDATE_coarse'] = run_strats(cc, 'validate', chosen, tag='_coarse')
        res['D_VALIDATE'] = leadlag(ctx, 'validate')
        save_res(res); return
    if stage == 'holdout':
        out = run_strats(ctx, 'holdout', CONFIGS)
        res['HOLDOUT'] = out
        chosen = [v['chosen'] for v in res['CHOSEN'].values()]
        res['HOLDOUT_robust'] = run_strats(ctx, 'holdout', chosen, slips=(0.003, 'stress'))
        cc = Ctx(coarse=True)
        res['HOLDOUT_coarse'] = run_strats(cc, 'holdout', chosen, tag='_coarse')
        res['D_HOLDOUT'] = leadlag(ctx, 'holdout')
        verdict = {}
        for fam, v in res['CHOSEN'].items():
            c = v['chosen']; wv = out[c]['windows']['VAL']; wh = out[c]['windows']['HOLD']
            ok_main = all(w['t_vs_main_m'] >= 1.5 and w['excess_vs_main_pct_m'] > 0 for w in (wv, wh))
            ok_ew = all(w['t_vs_EW_m'] >= 1.5 and w['excess_vs_EW_pct_m'] > 0 for w in (wv, wh))
            verdict[fam] = {'config': c, 'verdict': '可用' if ok_main else ('待觀察' if ok_ew else '不可用'),
                            'VAL_t_vs_main': wv['t_vs_main_m'], 'HOLD_t_vs_main': wh['t_vs_main_m'],
                            'VAL_t_vs_EW': wv['t_vs_EW_m'], 'HOLD_t_vs_EW': wh['t_vs_EW_m']}
        dv = {}
        for key in res.get('D_ADVANCE', []):
            pk, hk = key.split('_', 1)
            sv = res['D_HOLDOUT'][pk]['VAL'].get(hk); sh = res['D_HOLDOUT'][pk]['HOLD'].get(hk)
            okv = sv and sv['IC'] > 0 and sv['t_NW'] >= 2; okh = sh and sh['IC'] > 0 and sh['t_NW'] >= 1.5
            dv[key] = '可用' if (okv and okh) else ('待觀察' if okv else '不可用')
        verdict['D'] = dv if dv else '無配對在挑選期通過 → 不可用'
        res['VERDICT'] = verdict
        save_res(res); print(json.dumps(verdict, ensure_ascii=False, indent=1)); return


if __name__ == '__main__':
    main()
