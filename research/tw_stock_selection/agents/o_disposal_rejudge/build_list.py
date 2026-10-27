#!/usr/bin/env python3
"""第 0 步：建立 2003～2026 TWSE 歷史處置名單、交叉核對、凍結（sha256）。不算任何報酬。

來源 A（主）：GitHub tsungmin2001/tw-disposition 的 raw/twse_YYYY.json（TWSE rwd announcement/punish 全年查詢封存，
       欄位＝編號、公布日期、證券代號、證券名稱、累計、處置條件、處置起迄時間、處置措施、處置內容、備註）。視為不可信下載資料：
       只讀 JSON、逐欄解析，不執行任何其中的程式。
來源 B（核對）：tw-stock-screener extras/disposal_5y.csv.gz（2021-09-23 起，獨立蒐集的官方名單）。
來源 C（核對，若存在）：agents/o_infra_v3_rebuild/disposal_hist.csv（第 1 題的名單）。
來源 D（近似）：disposal_bear_longhistory 的價格近似規則（6 日還原漲幅 ≥32%、超額 ≥20pp、連 3 日）。

輸出：disposal_hist_rejudge.csv、data_freeze.json（sha256、逐年統計、一致率、近似規則精確率／召回率、閘門結果）。
"""
import json, re, hashlib, sys, glob, datetime
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw' / 'tsungmin2001_tw-disposition' / 'raw'
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
INFRA = HERE.parent / 'o_infra_v3_rebuild' / 'disposal_hist.csv'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

CN = {"一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
def cnum(s):
    s = str(s).strip().translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    if s.isdigit(): return int(s)
    if "十" not in s: return CN.get(s)
    a, _, b = s.partition("十")
    return (CN.get(a, 0) if a else 1) * 10 + (CN.get(b, 0) if b else 0)
RMIN = re.compile(r"約每\s*([0-9０-９一二兩三四五六七八九十]+)\s*分鐘")
def roc(s):
    s = s.strip().lstrip('*')
    y, m, d = s.split('/')
    return pd.Timestamp(int(y) + 1911, int(m), int(d))

def parse():
    rows = []; files = {}
    for f in sorted(glob.glob(str(RAW / 'twse_20??.json'))):
        y = int(Path(f).stem[-4:])
        files[Path(f).name] = sha(f)
        d = json.loads(Path(f).read_text(encoding='utf-8'))
        assert d.get('stat') == 'OK', f
        assert len(d['data']) == int(d['total']), (f, len(d['data']), d['total'])
        fl = d['fields']; ix = {k: fl.index(k) for k in ['公布日期', '證券代號', '證券名稱', '累計', '處置條件', '處置起迄時間', '處置措施', '處置內容']}
        for r in d['data']:
            per = r[ix['處置起迄時間']].replace(' ', '')
            a, b = per.split('～')
            txt = r[ix['處置內容']] or ''
            m = RMIN.findall(txt)
            rows.append(dict(file_year=y, announce=roc(r[ix['公布日期']]), code=str(r[ix['證券代號']]).strip(), name=str(r[ix['證券名稱']]).strip(),
                             cum=r[ix['累計']], condition=r[ix['處置條件']], start=roc(a), end=roc(b), measure=r[ix['處置措施']].strip(),
                             match_min=cnum(m[0]) if m else np.nan, content=txt))
    df = pd.DataFrame(rows)
    n0 = len(df)
    df = df.drop_duplicates(['announce', 'code', 'start', 'end', 'measure']).reset_index(drop=True)
    return df, files, n0

def classify(df):
    t = df['content'].fillna('')
    meas = df['measure']
    second = meas.isin(['第二次處置', '收足全部款券']) | t.str.contains('曾發布處置') | meas.str.contains('督導')
    df['nth'] = np.where(second, 2, 1)
    df['kind'] = np.select([meas.str.contains('督導'), meas == '人工管制撮合'], ['supervisory', 'changed_method'], 'regular')
    # 撮合間隔：有寫就用寫的；2010-12 前「收足五成／全部款券」為一般逐筆（0＝連續撮合，只預收）
    df['match_min'] = np.where(df['match_min'].notna(), df['match_min'], np.where(meas.isin(['收足五成款券', '收足全部款券']), 0, np.nan))
    df['regime'] = np.where(meas.isin(['收足五成款券', '收足全部款券']), 'prepay_only', 'periodic_match')
    return df

def main():
    df, files, n_raw = parse()
    df = classify(df)
    C = pd.read_parquet(P / 'close_raw.parquet')
    idx = C.index; cols = set(C.columns)
    df['in_panel'] = df['code'].isin(cols)
    # 處置期間交易日數（面板日曆；2024-01～02-14 面板缺口照實）
    def ntd(a, b):
        return int(((idx >= a) & (idx <= b)).sum())
    df['n_tdays'] = [ntd(a, b) for a, b in zip(df.start, df.end)]
    out = df.drop(columns=['content']).sort_values(['start', 'code']).reset_index(drop=True)
    out.to_csv(HERE / 'disposal_hist_rejudge.csv', index=False, date_format='%Y-%m-%d')

    stk = out[out.in_panel]
    # ---- 逐年統計（面板內上市股；以處置開始日的年份）----
    by_year = {}
    for y, g in stk.groupby(stk.start.dt.year):
        by_year[int(y)] = dict(n=len(g), nth1=int((g.nth == 1).sum()), nth2=int((g.nth >= 2).sum()), changed_method=int((g.kind == 'changed_method').sum()),
                              prepay_only=int((g.regime == 'prepay_only').sum()), med_tdays=float(g.n_tdays.median()),
                              match_min=g.match_min.value_counts(dropna=False).to_dict())
    post = [by_year[y]['n'] for y in (2022, 2023, 2024, 2025)]
    post_mean = float(np.mean(post))
    # 2021-10～2026-09 的年均（第 0 步定義：2021 年後年均＝2022～2025 四個完整年平均）
    gate = {y: dict(n=v['n'], ratio=round(v['n'] / post_mean, 3), pass50=bool(v['n'] >= 0.5 * post_mean)) for y, v in by_year.items() if 2005 <= y <= 2021}

    # ---- 與 disposal_5y 一致率（2021-09-24～2026-10-05，面板內）----
    B = pd.read_csv(X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
    B = B[B.code.isin(cols)]
    lo, hi = pd.Timestamp('2021-09-24'), pd.Timestamp('2026-10-05')
    A_ = stk[(stk.start >= lo) & (stk.start <= hi)]
    B_ = B[(B.start >= lo) & (B.start <= hi)]
    ka = set(zip(A_.code, A_.start)); kb = set(zip(B_.code, B_.start))
    kea = set(zip(A_.code, A_.start, A_.end)); keb = set(zip(B_.code, B_.start, B_.end))
    agree = dict(n_A=len(ka), n_B=len(kb), both=len(ka & kb), A_in_B=round(len(ka & kb) / max(len(ka), 1), 4), B_in_A=round(len(ka & kb) / max(len(kb), 1), 4),
                 jaccard=round(len(ka & kb) / max(len(ka | kb), 1), 4), exact_start_end_both=len(kea & keb),
                 only_A_sample=sorted([(c, str(d.date())) for c, d in ka - kb])[:15], only_B_sample=sorted([(c, str(d.date())) for c, d in kb - ka])[:15])
    # 公告日一致（同 code,start）
    mA = A_.set_index(['code', 'start'])['announce']; mB = B_.drop_duplicates(['code', 'start']).set_index(['code', 'start'])['date']
    jj = mA.index.intersection(mB.index)
    agree['announce_same_rate'] = round(float((mA.loc[jj].values == mB.loc[jj].values).mean()), 4) if len(jj) else None

    # ---- 與第 1 題名單一致率（若存在）----
    infra = {'exists': INFRA.exists()}
    if INFRA.exists():
        try:
            I = pd.read_csv(INFRA, dtype={'code': str})
            sc = [c for c in I.columns if 'start' in c.lower()][0]
            I[sc] = pd.to_datetime(I[sc])
            I = I[I.code.isin(cols)]
            for lab, a, b in [('2006-04_2021-08', '2006-04-01', '2021-08-31'), ('2005_2021-09', '2005-01-01', '2021-09-30')]:
                kA = set(zip(stk[(stk.start >= a) & (stk.start <= b)].code, stk[(stk.start >= a) & (stk.start <= b)].start))
                II = I[(I[sc] >= a) & (I[sc] <= b)]
                kI = set(zip(II.code, II[sc]))
                infra[lab] = dict(n_mine=len(kA), n_infra=len(kI), both=len(kA & kI), agree_vs_mine=round(len(kA & kI) / max(len(kA), 1), 4),
                                  agree_vs_infra=round(len(kA & kI) / max(len(kI), 1), 4), sha256=sha(INFRA))
        except Exception as e:
            infra['error'] = repr(e)

    # ---- 近似規則（逐字沿用 disposal_bear_longhistory）精確率／召回率，逐年 ----
    adj = pd.read_parquet(P / 'adj.parquet')
    twr = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index()
    tx = twr.reindex(adj.index).ffill()
    A6 = adj / adj.shift(6) - 1; m6 = tx / tx.shift(6) - 1
    att = (A6 >= 0.32) & (A6.sub(m6, axis=0) >= 0.20)
    trig = (att & att.shift(1, fill_value=False) & att.shift(2, fill_value=False)).to_numpy()
    T = len(adj.index); pc = adj.columns; prox = []
    for c in range(len(pc)):
        i = 0; col = trig[:, c]
        while i < T:
            if col[i]:
                s, e = i + 1, i + 10
                if e < T: prox.append((pc[c], adj.index[s], adj.index[e]))
                i = e + 1
            else:
                i += 1
    PX = pd.DataFrame(prox, columns=['code', 'start', 'end'])
    PX.to_csv(HERE / 'proxy_events.csv', index=False, date_format='%Y-%m-%d')
    def match(a, b):
        bs = {}
        for c, d in zip(b.code, b.start): bs.setdefault(c, []).append(d)
        return np.array([any(abs((d - x).days) <= 5 for x in bs.get(c, [])) for c, d in zip(a.code, a.start)])
    pr = {}
    for y in range(2005, 2027):
        a, b = pd.Timestamp(y, 1, 1), pd.Timestamp(y, 12, 31)
        p_ = PX[(PX.start >= a) & (PX.start <= b)]; t_ = stk[(stk.start >= a) & (stk.start <= b) & (stk.kind == 'regular')]
        if len(p_) == 0 or len(t_) == 0: continue
        pr[y] = dict(n_proxy=len(p_), n_true=len(t_), precision=round(float(match(p_, t_).mean()), 3), recall=round(float(match(t_, p_).mean()), 3))
    for lab, a, b in [('2006-04_2021-08', '2006-04-01', '2021-08-31'), ('2021-09_2026-09', '2021-09-24', '2026-09-30')]:
        p_ = PX[(PX.start >= a) & (PX.start <= b)]; t_ = stk[(stk.start >= a) & (stk.start <= b) & (stk.kind == 'regular')]
        pr[lab] = dict(n_proxy=len(p_), n_true=len(t_), precision=round(float(match(p_, t_).mean()), 3), recall=round(float(match(t_, p_).mean()), 3))

    freeze = dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        source_A=dict(repo='github.com/tsungmin2001/tw-disposition', path='raw/twse_YYYY.json', note='TWSE rwd announcement/punish 全年查詢封存；每檔 len(data)==total 已斷言',
                      file_sha256=files, rows_raw=n_raw, rows_dedup=len(df)),
        list_csv='disposal_hist_rejudge.csv', list_sha256=sha(HERE / 'disposal_hist_rejudge.csv'), proxy_csv_sha256=sha(HERE / 'proxy_events.csv'),
        n_in_panel=int(stk.shape[0]), by_year_in_panel=by_year, post2021_annual_mean_2022_2025=post_mean, gate_50pct=gate,
        agree_disposal_5y=agree, agree_infra_v3=infra, proxy_precision_recall=pr,
        note_regimes='2010-12 前處置措施只有「收足五成款券」（第一次）／「收足全部款券」（再次），沒有分盤撮合；2010-12 起第一次約每 5 分鐘、第二次約每 20 分鐘撮合。'
                     '「人工管制撮合」多為變更交易方法／全額交割股的處置。2011～2015 第一次處置 5 個營業日、第二次 10 個營業日；之後皆 10 個營業日；2026-08-10 起 5 個營業日。')
    (HERE / 'data_freeze.json').write_text(json.dumps(freeze, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(json.dumps({k: freeze[k] for k in ['n_in_panel', 'post2021_annual_mean_2022_2025', 'gate_50pct', 'agree_disposal_5y', 'agree_infra_v3']}, ensure_ascii=False, indent=0, default=str))
    print(json.dumps(pr, indent=0))
    print({y: (v['n'], v['nth1'], v['nth2'], v['changed_method'], v['med_tdays']) for y, v in by_year.items()})

if __name__ == '__main__':
    main()
