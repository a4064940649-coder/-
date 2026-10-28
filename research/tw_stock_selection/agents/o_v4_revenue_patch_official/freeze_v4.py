#!/usr/bin/env python3
"""步驟 0（總監統一資料）：data_freeze_v4.json。在任何策略報酬之前執行。
固定：官方處置名單、P、X、終止上市名單、R_DEL、tw_revenue、rev_5y、disposal_5y、taiwanan 月營收、日線面板、0050，
以及本題的 R_FIX（長期年增率）、Y5_FIX（rev_5y 年增率，2021-03 起訊號用）、季營收代理、新聞抽取結果。
o4 其他三題以本檔的 v3_base 為底（V3 資料），R_FIX 為本題新增。"""
import sys, json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v4lib as L
K = L.K
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
base = L.check_v3_inputs()
P = K.P
panel = {f'panel/{n}.parquet': {'path': str(P / f'{n}.parquet'), 'sha256': sha(P / f'{n}.parquet')} for n in ('adj', 'close_raw', 'open', 'high', 'value', 'ret', 'flag', 'src')}
other = {'etf_0050': {'path': str(K.E50F), 'sha256': sha(K.E50F)},
         'stock_list': {'path': str(K.X.parent / 'stock_list.csv'), 'sha256': sha(K.X.parent / 'stock_list.csv')},
         'taiex_calendar': {'path': '/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', 'sha256': sha('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv')},
         'rev_hist': {'path': str(K.X / 'rev_hist.csv.gz'), 'sha256': sha(K.X / 'rev_hist.csv.gz')},
         'rev_announce_p_delay': {'path': str(L.AG / 'rev_announce' / 'p_delay.json'), 'sha256': sha(L.AG / 'rev_announce' / 'p_delay.json')},
         'rev_announce_firstseen': {'path': str(L.AG / 'rev_announce' / 'firstseen_2026-09.csv'), 'sha256': sha(L.AG / 'rev_announce' / 'firstseen_2026-09.csv')},
         'foreign_codes_src': {'path': str(L.AG / 'o_rev_survivorship' / 'data_freeze.json'), 'sha256': sha(L.AG / 'o_rev_survivorship' / 'data_freeze.json')}}
itd = json.loads((HERE / 'iterations_data.json').read_text(encoding='utf-8'))
for k in ('R_FIX', 'Y5_FIX'):
    assert sha(HERE / itd[k]['file']) == itd[k]['sha256']
mine = {n: {'file': n, 'sha256': sha(HERE / n)} for n in ('raw_news_mentions.parquet', 'raw_quarterly.parquet', 'yoy_qproxy.parquet', 'rfix_cells.csv',
                                                          'gap_pool_stockmonths.csv', 'gap_delisted.csv')}
fz = {'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
      'note': '在任何策略報酬之前凍結（本題第一次跑報酬在 PREREG 之後）。v3_base＝o4 各題共用的 V3 資料；R_FIX／Y5_FIX＝本題修補。',
      'v3_base': base, 'panel': panel, 'other': other,
      'R_FIX': itd['R_FIX'], 'Y5_FIX': itd['Y5_FIX'], 'merge_rule_selected': itd['selected'],
      'merge_rule_text': itd['rule'], 'this_topic_files': mine,
      'news_source': json.loads((HERE / 'raw_news_meta.json').read_text(encoding='utf-8')),
      'quarterly_source': json.loads((HERE / 'raw_quarterly_meta.json').read_text(encoding='utf-8')),
      'qproxy': json.loads((HERE / 'yoy_qproxy_meta.json').read_text(encoding='utf-8')),
      'not_reachable': ['web.archive.org（proxy 403；WebFetch 亦拒絕）', 'mops.twse.com.tw／mopsov.twse.com.tw（403）', 'goodinfo.tw、twse.com.tw（DNS 失敗）',
                        'gh api search（session 限 repo-scoped）→ 改用 GitHub MCP code search'],
      'disposal_rule_main': 'trade：公告日 ≤ 訊號日 d，且 start ≤ 成交日 e（d 的次一面板交易日）≤ end → 排除'}
(HERE / 'data_freeze_v4.json').write_text(json.dumps(fz, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps({k: fz[k] for k in ('timestamp', 'R_FIX', 'Y5_FIX', 'merge_rule_selected')}, ensure_ascii=False, indent=1))
