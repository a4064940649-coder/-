#!/usr/bin/env python3
"""營收可得時點分類器（只讀 raw/ 的時間戳檔）。每個 (訊號日 S, 代碼, 營收月) 回傳 AVAIL／UNDET／UNAVAIL／NONE 與證據。
版本（iterations.json）：
  V1 字面規則：AVAIL＝公告日 < D 或快照首見 ≤ S 13:30；UNDET＝公告日＝D 或首見在 S 收盤後、前一張在 S 之前；UNAVAIL＝公告日 > D 或首見 > S 24:00；
     D＝v2_core 期限日（D>S 日字面用期限日）；來源＝MOPS ann_date、Tasi、上游。
  V2 ＝V1＋新聞（嚴格有效位數數值閘門）。
  V3 ＝V2，但 UNAVAIL 只在「某張 ≥ S 13:30 的快照裡該檔確定不在」（T_prev ≥ S 13:30）或公告日 > S；首見 > S 24:00 但前一張 < S 13:30 → UNDET；首次加入檔案（無 T_prev）且首見 > S 13:30 → UNDET。
  V4 ＝V3，截止日一律用訊號日 S（D>S 日不用期限日）。← 預定凍結版
  V5 ＝V4，但新聞閘門放寬到 ±15%（敏感度，不凍結）。
  V6 ＝V4，但快照（Tasi／上游）的 UNAVAIL 改成「前一張快照的日期 > S（≥ S 之後的交易日）仍不在」：一致性閘門顯示 Tasi 的營收表比 MOPS 公告日晚 1 天
     （2026-09 營收 1,337／1,457 筆在公告日當晚 18:20 快照還不在、隔天才出現），所以「S 當晚快照不在」不能證明 S 13:30 前沒公告。← 凍結版
衝突（有來源 AVAIL、有來源 UNAVAIL）→ UNDET（標 CONFLICT）。第二截止 S 24:00 只作描述（second 欄）。"""
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw'
TZ = 'Asia/Taipei'

def _tp(s): return pd.to_datetime(s, utc=True, errors='coerce').dt.tz_convert(TZ).dt.tz_localize(None)

class Evidence:
    def __init__(self, news_tol='sig'):
        FS = pd.read_csv(RAW / 'ts_tasi_upstream_firstseen.csv.gz', dtype=str)
        FS['Tt'] = _tp(FS['T']); FS['Tp'] = _tp(FS['T_prev'])
        self.snap = {}
        for r in FS.itertuples():
            self.snap.setdefault((r.code, r.ym), []).append((r.source, r.Tt, r.Tp, r.reason))
        AN = pd.read_csv(RAW / 'ts_mops_ann_date.csv', dtype=str)
        self.ann = {(c, y): pd.Timestamp(a) for c, y, a in zip(AN.code, AN.ym, AN.ann_date)}
        NW = pd.read_csv(RAW / 'ts_news_mentions.csv.gz', dtype={'code': str, 'ym': str, 'art': str, 'src': str})
        NW['tsd'] = pd.to_datetime(NW.ts)
        NW['has_time'] = NW.has_time.astype(str).eq('True')
        tw = pd.read_csv('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv', dtype={'code': str})
        tw['ym'] = tw.month.str[:7]; tw = tw.drop_duplicates(['code', 'ym'])
        r5 = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/rev_5y.csv.gz', dtype={'code': str})
        TR = dict(zip(zip(tw.code, tw.ym), tw.revenue_k_twd)); TR5 = dict(zip(zip(r5.code, r5.ym), r5.revenue))
        def ok(code, ym, v):
            tv = [x for x in (TR.get((code, ym)), TR5.get((code, ym))) if x is not None and np.isfinite(x)]
            if not tv or not np.isfinite(v) or v <= 0: return False
            if news_tol == 'sig':
                s = str(int(round(v))); z = len(s) - len(s.rstrip('0')); tol = 0.5 * 10 ** z + 1.0
                return any(abs(v - t) <= tol for t in tv)
            return any(t > 0 and abs(v / t - 1) <= 0.15 for t in tv)
        NW['valid'] = [ok(c, y, v) for c, y, v in zip(NW.code, NW.ym, NW.rev_k)]
        self.news = {}
        for r in NW[NW.valid].itertuples():
            self.news.setdefault((r.code, r.ym), []).append((r.src, r.tsd, r.has_time))
        self.n_news_valid = int(NW.valid.sum()); self.n_news = int(len(NW))

    def classify(self, S, code, ym, version='V4', D=None, use_news=True):
        """S：訊號日（Timestamp，日）；D：期限日（V1～V2 字面規則用）。回傳 (state, detail dict)。"""
        cut_day = S if version in ('V4', 'V5', 'V6') else (D if D is not None else S)
        C1330 = cut_day + pd.Timedelta(hours=13, minutes=30)
        C24 = cut_day + pd.Timedelta(days=1)
        lit = version in ('V1', 'V2')
        st = []; det = {}
        a = self.ann.get((code, ym))
        if a is not None:
            s = 'AVAIL' if a < cut_day else ('UNDET' if a == cut_day else 'UNAVAIL')
            st.append(('mops', s)); det['ann_date'] = str(a.date())
        for src, Tt, Tp, rs in self.snap.get((code, ym), []):
            if pd.isna(Tt): continue
            if Tt <= C1330: s = 'AVAIL'
            elif lit:
                if Tt > C24: s = 'UNAVAIL'
                elif pd.notna(Tp) and Tp < cut_day: s = 'UNDET'
                elif pd.notna(Tp): s = 'UNDET'
                else: s = 'UNDET'
            elif version == 'V6':
                if pd.notna(Tp) and Tp.normalize() > cut_day: s = 'UNAVAIL'
                else: s = 'UNDET'
            else:
                if pd.notna(Tp) and Tp >= C1330: s = 'UNAVAIL'
                else: s = 'UNDET'
            # 上游 git 首次匯入（file_add）且首見遠晚於訊號日：沒有資訊
            if src == 'upstream' and pd.isna(Tp) and Tt > C24 + pd.Timedelta(days=3): s = 'NONE'
            if src == 'tasi' and pd.isna(Tp) and Tt > C24 + pd.Timedelta(days=3) and not lit: s = 'NONE'
            st.append((src, s)); det[src] = (str(Tt), str(Tp) if pd.notna(Tp) else '', rs)
        if use_news and version != 'V1':
            for src, ts, ht in self.news.get((code, ym), []):
                if ht:
                    s = 'AVAIL' if ts <= C1330 else ('UNDET' if ts < C24 else 'NONE')
                else:
                    dd = ts.normalize()
                    s = 'AVAIL' if dd < cut_day else ('UNDET' if dd == cut_day else 'NONE')
                st.append(('news_' + src, s))
                det.setdefault('news', []).append((src, str(ts), bool(ht)))
        ss = [s for _, s in st]
        if 'AVAIL' in ss and 'UNAVAIL' in ss: state = 'UNDET'; det['conflict'] = True
        elif 'AVAIL' in ss: state = 'AVAIL'
        elif 'UNAVAIL' in ss: state = 'UNAVAIL'
        elif 'UNDET' in ss: state = 'UNDET'
        else: state = 'NONE'
        # 第二截止（描述）：S 24:00 前是否已看得到
        second = None
        if state != 'AVAIL':
            seen24 = any(s == 'AVAIL' for _, s in st)
            for src, Tt, Tp, rs in self.snap.get((code, ym), []):
                if pd.notna(Tt) and Tt <= C24: seen24 = True
                if pd.notna(Tp) and Tp.normalize() > cut_day: second = 'ABSENT_ON_LATER_DAY'
            if a is not None and a <= cut_day: seen24 = True
            if a is not None and a > cut_day: second = 'ANN_AFTER_S'
            second = 'SEEN_BY_24' if seen24 else (second or 'UNKNOWN')
        det['sources'] = st
        return state, det, second
