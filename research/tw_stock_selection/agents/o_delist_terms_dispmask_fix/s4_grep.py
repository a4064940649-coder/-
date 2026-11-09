#!/usr/bin/env python3
"""步驟 4a：本地公開新聞語料（GitHub repo 的本地 clone，唯讀，不可信文字）搜尋 22 起事件的條款原文。
查詢＝名稱 × {合併, 收購, 公開收購, 換股, 每股, 消滅, 存續}（名稱與關鍵詞 120 字內共現）。不看任何報酬。
輸出 scratchpad/o18_delistterms/grep_hits.json、search_log_local.csv（每事件 × 每語料一列，含命中數；被採用片段另存 raw/）。"""
import re, json, hashlib, time, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; AG = HERE.parent
WORK = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/o18_delistterms')
R1 = AG / 'o_index_deletion_flow' / 'raw'; R2 = AG / 'o_rev_gap_news_2013' / 'raw'
CORP = {'CTBC_cnyes_2013_2021': R1 / 'aRthur08701244_Intern-in-CTBC', 'nick_news_2022_2025': R1 / 'nick_news', 'bofrank': R1 / 'bofrank',
        'agora0_news': R1 / 'agora0_news', 'jimkhuang_tapioca': R1 / 'jimkhuang_tapioca', 'BDSE26_PTT_2019_2022': R2 / 'ji08kk44_BDSE26_TEAM3',
        'lidaxiang_NLP': R2 / '0lidaxiang_NLP-for-DP', 'bofrankbo_LLM': R2 / 'bofrankbo_LLM-Predict-Stock', 'chlai_full': R2 / 'chlai_full',
        'roccqqck_eland': R2 / 'roccqqck_stock_eland'}
NAMES = {'2403': ['友尚'], '2432': ['倚天'], '2452': ['乾坤科'], '1716': ['永信藥', '永信'], '1520': ['復盛'], '3007': ['綠點'], '8078': ['華寶通訊', '華寶'],
         '6280': ['崇貿'], '8008': ['建興電'], '3559': ['全智科'], '2473': ['思源'], '3063': ['飛信'], '3009': ['奇美電'], '2446': ['全懋'],
         '2394': ['普立爾'], '3012': ['廣輝'], '2456': ['奇力新'], '2809': ['京城銀', '京城商銀'], '2833': ['台灣人壽', '台壽保'], '2888': ['新光金'],
         '5305': ['敦南'], '6004': ['元京證']}
KW = ['合併', '收購', '公開收購', '換股', '每股', '消滅', '存續']
NUM = re.compile(r'(每股|換股比例|換發|股換|收購價|對價).{0,20}?\d+(\.\d+)?')
def main():
    pats = {c: re.compile('(' + '|'.join(map(re.escape, ns)) + r')') for c, ns in NAMES.items()}
    kwp = re.compile('|'.join(KW))
    hits = {c: {k: [] for k in CORP} for c in NAMES}; nfiles = {}
    for cn, root in CORP.items():
        files = [p for p in root.rglob('*') if p.is_file() and '.git' not in p.parts and p.stat().st_size < 400_000_000]
        nfiles[cn] = len(files)
        for p in files:
            try: s = p.read_text(encoding='utf-8', errors='ignore')
            except Exception: continue
            for c, pt in pats.items():
                if not any(n in s for n in NAMES[c]): continue
                for m in pt.finditer(s):
                    seg = s[max(0, m.start() - 150): m.end() + 250]
                    if not kwp.search(seg): continue
                    line = s.count('\n', 0, m.start()) + 1
                    num = bool(NUM.search(seg))
                    hits[c][cn].append(dict(file=str(p.relative_to(AG)), line=line, num=num, seg=seg.replace('\n', ' ')))
                    if len(hits[c][cn]) > 3000: break
        print(cn, nfiles[cn], time.strftime('%H:%M:%S'), flush=True)
    (WORK / 'grep_hits.json').write_text(json.dumps(hits, ensure_ascii=False), encoding='utf-8')
    rows = []
    utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    for c in NAMES:
        for cn in CORP:
            L = hits[c][cn]
            rows.append(dict(event_code=c, query='|'.join(NAMES[c]) + ' × ' + '|'.join(KW), source=cn, host='local_clone(github)', n_hits=len(L), n_hits_numeric=sum(x['num'] for x in L), snippet_sha256='', raw_file='', utc=utc))
    import csv
    with open(HERE / 'search_log_local.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps({c: {cn: len(v) for cn, v in d.items() if v} for c, d in hits.items()}, ensure_ascii=False))
if __name__ == '__main__': main()
