#!/usr/bin/env python3
"""data_exright_fix 共用：建官方日 K 面板（沿用 stlib.build_panel 格式，快取放本題 scratchpad/exr/）、
讀證交所除權除息計算結果表（TWT49U 公開鏡像：joeycchen72/taiwan-market-history data/twse-exright，o_0050_tilt_pit 已下載，只讀），
產生「修正後漲跌價差」的各版本。只讀共用資料，不改任何其他資料夾。一律 python3 -I -B。"""
import sys, gzip, io
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/exr')
EXR_DIR = AG / 'o_0050_tilt_pit' / 'raw' / 'jc_exright' / 'data' / 'twse-exright'
TOL = 0.006           # 參考價比對容忍（相對）：涵蓋跳動點四捨五入


def load_twt49u():
    fr = []
    for f in sorted(EXR_DIR.glob('*/*.csv.gz')):
        fr.append(pd.read_csv(f, dtype={'code': str}))
    d = pd.concat(fr, ignore_index=True)
    d['date'] = pd.to_datetime(d['date'])
    d = d[d['code'].str.fullmatch(r'[1-9]\d{3}')]
    d = d.drop_duplicates(['date', 'code'], keep='last')
    return d


def raw_panel():
    """回傳 dict（IDX, COLS, C, CHG, MK）＝ stlib.build_panel 產生的 npz（全期，未截到 END）。"""
    sys.path.insert(0, str(AG / 's1_shortterm_leaderboard'))
    import stlib
    stlib.S1 = TMP; stlib.cslib.TMP = TMP
    f = stlib.build_panel()
    z = np.load(f, allow_pickle=False)
    return {k: z[k] for k in ('IDX', 'COLS', 'C', 'CHG', 'MK')}, f


def classify(Z, ev):
    """逐格分類。回傳 DataFrame（每個「漲跌價差＝0 但收盤對前一成交收盤有變」的格子）＋ 除權息事件對照表。"""
    IDX = pd.DatetimeIndex(Z['IDX']); COLS = Z['COLS']; C = Z['C']; CHG = Z['CHG']; MK = Z['MK']
    T, N = C.shape
    has = np.isfinite(C) & (C > 0)
    CF = pd.DataFrame(np.where(has, C, np.nan)).ffill().to_numpy()
    prevC = np.vstack([np.full((1, N), np.nan), CF[:-1]])
    z = has & np.isfinite(prevC) & (np.nan_to_num(CHG) == 0) & (np.abs(C - prevC) > 1e-9)
    ti, ji = np.nonzero(z)
    cells = pd.DataFrame(dict(t=ti, j=ji, date=IDX[ti], code=COLS[ji], mk=np.where(MK[ti, ji] == 1, 'twse', 'tpex'),
                              close=C[ti, ji], prev=prevC[ti, ji]))
    ci = {c: j for j, c in enumerate(COLS)}; di = {d: i for i, d in enumerate(IDX)}
    e = ev[ev['code'].isin(ci) & ev['date'].isin(di)].copy()
    e['t'] = e['date'].map(di); e['j'] = e['code'].map(ci)
    t, j = e['t'].to_numpy(), e['j'].to_numpy()
    e['close'] = C[t, j]; e['chg'] = CHG[t, j]; e['prevC'] = prevC[t, j]; e['mk'] = MK[t, j]
    e['ref_official'] = e['close'] - e['chg']
    cells = cells.merge(e[['t', 'j', 'ref_price', 'prev_close', 'value', 'type']], on=['t', 'j'], how='left')
    cells['in_twt49u'] = cells['ref_price'].notna()
    cells['ret_close'] = cells['close'] / cells['prev'] - 1
    cells['ret_ref'] = cells['close'] / cells['ref_price'] - 1
    cells['prev_traded'] = has[np.maximum(ti - 1, 0), ji]
    cat = np.where(cells['in_twt49u'] & (np.abs(cells['close'] / cells['ref_price'] - 1) <= 1e-9), '真除權息_收盤剛好等於參考價（0 正確）',
          np.where(cells['in_twt49u'], '真除權息_漲跌價差漏填',
          np.where(~cells['prev_traded'], '資料錯_前一日無成交、漲跌價差沒算',
          np.where(cells['mk'] == 'tpex', '上櫃_研判除權息（無公開參考價來源）',
          np.where(np.abs(cells['ret_close']) > 0.105, '上市不在除權息表_漲跌超過 10%（減資等參考價重設）', '上市不在除權息表_其他')))))
    cells['cat'] = cat
    # 事件表：官方漲跌價差在除權息日是否反映參考價
    ok = e['close'].notna() & (e['close'] > 0)
    e = e[ok].copy()
    rel = lambda a, b: np.abs(a / b - 1)
    e['ecat'] = np.select(
        [e['chg'].isna(), (e['chg'] == 0) & (rel(e['close'], e['ref_price']) <= 1e-9), e['chg'] == 0,
         rel(e['ref_official'], e['ref_price']) <= TOL, rel(e['ref_official'], e['prevC']) <= 1e-6],
        ['漲跌價差缺值', '0（收盤＝參考價，正確）', '0（漏填）', '已反映參考價', '等於收盤−前收（漏扣股利）'], '其他')
    return cells, e


def corrected_chg(Z, ev, mode):
    """回傳（修正後的 CHG 複本, 改動格數）。z＝「漲跌價差＝0（或缺值記 0）且收盤對前一成交收盤有變」。
    mode='orig'：原樣（＝下界版「維持 0」）。
    mode='twt' ：只修除權息。上市在 TWT49U 的事件日、漲跌價差＝0 → CHG＝收盤 − TWT49U 參考價（含收盤沒變的事件日）。
    mode='main'：建議修法＝twt，再加上「前一日無成交、漲跌價差沒算」的上市格子 → CHG＝收盤 − 前一成交收盤。
                 超過漲跌幅 10.5% 的（停牌期間參考價被重設，如減資）、上櫃（不在股池）、其餘少數格子不動。
    mode='close'：收盤報酬版。所有 z 格子（上市＋上櫃）→ CHG＝收盤 − 前一成交收盤（除權息日當成股價下跌、不含股利）。"""
    IDX = pd.DatetimeIndex(Z['IDX']); COLS = Z['COLS']; C = Z['C']; CHG = Z['CHG'].copy(); MK = Z['MK']
    if mode == 'orig':
        return CHG, 0
    T, N = C.shape
    has = np.isfinite(C) & (C > 0)
    CF = pd.DataFrame(np.where(has, C, np.nan)).ffill().to_numpy()
    prevC = np.vstack([np.full((1, N), np.nan), CF[:-1]])
    zero = np.nan_to_num(CHG) == 0
    z = has & np.isfinite(prevC) & zero & (np.abs(C - prevC) > 1e-9)
    REFX = np.full((T, N), np.nan)
    ci = {c: j for j, c in enumerate(COLS)}; di = {d: i for i, d in enumerate(IDX)}
    e = ev[ev['code'].isin(ci) & ev['date'].isin(di)]
    REFX[e['date'].map(di).to_numpy(), e['code'].map(ci).to_numpy()] = e['ref_price'].to_numpy()
    inx = np.isfinite(REFX) & (MK == 1)
    prev_tr = np.vstack([np.zeros((1, N), bool), has[:-1]])
    n = 0
    if mode in ('twt', 'main'):
        m = has & inx & zero
        # TWT49U 的參考價是「除權息參考價」（含現金增資理論價）；收盤落在它 ±10.1% 以外，表示當天是現增除權、撮合基準是減除股利參考價，
        # 這時改用前一成交收盤（純現增時減除股利參考價＝前收；少數「權息＋現增」會少算股利，影響很小）。
        with np.errstate(invalid='ignore', divide='ignore'):
            rr = C / REFX
        oob = m & ~((rr >= 0.899) & (rr <= 1.101))
        REFX[oob] = prevC[oob]
        CHG[m] = C[m] - REFX[m]; n += int(m.sum())
        corrected_chg.n_oob = int(oob.sum())
        if mode == 'main':
            m3 = z & ~inx & ~prev_tr & (MK == 1) & (np.abs(C / prevC - 1) <= 0.105)
            CHG[m3] = C[m3] - prevC[m3]; n += int(m3.sum())
    elif mode == 'close':
        CHG[z] = C[z] - prevC[z]; n += int(z.sum())
    return CHG, n
