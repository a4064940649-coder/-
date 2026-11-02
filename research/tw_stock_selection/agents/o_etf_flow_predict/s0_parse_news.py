"""解析新聞封存中「臺灣高股息指數 成分股納入/刪除」段落（不可信文字，只取名稱與代碼，人工覆核）。"""
import sys, re, glob, json, csv
csv.field_size_limit(10**9)
ROOT='/home/user/-/research/tw_stock_selection/agents/o_index_deletion_flow/raw'
files=sorted(glob.glob(ROOT+'/aRthur08701244_Intern-in-CTBC/AI_Investment/OpenIEStanza_EventEmbedding/input/EB_EventExtraction2/*.csv'))+sorted(glob.glob(ROOT+'/nick_news/data/news/*.csv'))+sorted(glob.glob(ROOT+'/agora0_news/**/*.*',recursive=True))
rows=[]
for f in files:
    try: txt=open(f,encoding='utf-8',errors='ignore').read()
    except Exception: continue
    for m in re.finditer(r'(台灣|臺灣)高股息指數', txt):
        seg=txt[m.start():m.start()+700]
        # 截到下一個指數段落
        cut=re.search(r'[一二三四五六七八九十]、(?!.*高股息)', seg[10:])
        if cut: seg=seg[:10+cut.start()]
        add=re.search(r'納入\s*(\d+|[一二三四五六七八九十]+)\s*(支|檔)[：:]([^。；]*)', seg)
        dele=re.search(r'刪除\s*(\d+|[一二三四五六七八九十]+)\s*(支|檔)[：:]([^。；]*)', seg)
        if not (add or dele): continue
        # 日期：檔內最近一個 yyyymmdd_ 標記
        pre=txt[:m.start()]
        dm=re.findall(r'(20\d{6})_\d+', pre[-20000:])
        d=dm[-1] if dm else ''
        def codes(g): return re.findall(r'(\d{4})', g.group(3)) if g else []
        def names(g): return g.group(3) if g else ''
        rows.append({'file':f.split('/')[-1],'date':d,'add_codes':codes(add),'del_codes':codes(dele),'add_txt':names(add)[:200],'del_txt':names(dele)[:200]})
uniq={}
for r in rows:
    k=(tuple(r['add_codes']),tuple(r['del_codes']))
    if k not in uniq or (r['date'] and r['date']<uniq[k]['date']): uniq[k]=r
res=sorted(uniq.values(), key=lambda r:r['date'])
json.dump(res,open(sys.argv[1],'w'),ensure_ascii=False,indent=1)
for r in res: print(r['date'],r['file'],'ADD',r['add_codes'],'DEL',r['del_codes'])
