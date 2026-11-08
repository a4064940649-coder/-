#!/usr/bin/env python3
"""步驟二之一：在公開新聞鏡像（GitHub 開源 repo 的本地 clone，唯讀，視為不可信文字）中找每個下市事件的對價條款。
不看任何策略報酬。輸出 scratchpad/o17_delist/grep_hits.json（候選片段）；被採用為證據的原始檔另行複製到 raw/ 並記 sha256。
python3 -I -B s2_grep.py"""
import sys, re, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cz_lib as Z

ROOT = Z.AG / 'o_index_deletion_flow' / 'raw'
CORP = [ROOT / 'aRthur08701244_Intern-in-CTBC', ROOT / 'nick_news', ROOT / 'bofrank',
        Z.AG / 'o_rev_gap_news_2013' / 'raw']
K = r'(換股|換發|股份轉換|合併|收購|公開收購|每股|現金|對價|比例|基準日|終止上市|下市|新股上市|掛牌)'
EV = {
    '1520': ['復盛'], '1704': ['榮化'], '1716': ['永信'], '2311': ['日月光'], '2325': ['矽品'], '2394': ['普立爾'],
    '2403': ['友尚'], '2432': ['倚天'], '2446': ['全懋'], '2452': ['乾坤'], '2473': ['思源'], '2809': ['京城銀', '京城商銀'],
    '2823': ['中壽', '中國人壽'], '2833': ['台灣人壽', '台壽'], '2847': ['大眾銀'], '2888': ['新光金'], '3009': ['奇美電'],
    '3012': ['廣輝'], '3063': ['飛信'], '3474': ['華亞科'], '3697': ['晨星'], '6004': ['元京證'], '6286': ['立錡'],
    '1902': ['台紙'], '2448': ['晶電'], '2456': ['奇力新'], '3698': ['隆達'], '5305': ['敦南'],
}
NUM = re.compile(r'\d+(\.\d+)?\s*(元|股)')


def main():
    pats = {c: re.compile('(' + '|'.join(map(re.escape, ns)) + r').{0,60}' + K + r'.{0,80}') for c, ns in EV.items()}
    hits = {c: [] for c in EV}
    files = [p for d in CORP if d.exists() for p in d.rglob('*') if p.is_file() and '.git' not in p.parts and p.suffix in ('.csv', '.json', '.txt', '.md', '.html', '')]
    for p in files:
        try:
            s = p.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        for c, pt in pats.items():
            if not any(n in s for n in EV[c]): continue
            for m in pt.finditer(s):
                seg = s[max(0, m.start() - 120): m.end() + 120].replace('\n', ' ')
                score = len(NUM.findall(seg)) + 3 * bool(re.search(r'換股比例|換發|每股.{0,6}元|收購價|現金對價|1股|１股', seg))
                hits[c].append(dict(file=str(p.relative_to(Z.AG)), pos=m.start(), score=score, seg=seg))
    out = {}
    for c, L in hits.items():
        L.sort(key=lambda x: -x['score'])
        seen = set(); keep = []
        for x in L:
            k = x['seg'][100:220]
            if k in seen: continue
            seen.add(k); keep.append(x)
            if len(keep) >= 40: break
        out[c] = dict(n_raw=len(L), top=keep)
    (Z.WORK / 'grep_hits.json').write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding='utf-8')
    print({c: v['n_raw'] for c, v in out.items()})


if __name__ == '__main__':
    main()
