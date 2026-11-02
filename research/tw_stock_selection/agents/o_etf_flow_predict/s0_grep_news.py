"""從既有 GitHub 新聞封存（o_index_deletion_flow/raw，視為不可信）擷取臺灣高股息指數成分調整段落，供人工抄錄。只做文字擷取。"""
import sys, re, glob, csv, json
csv.field_size_limit(10**9)
ROOT='/home/user/-/research/tw_stock_selection/agents/o_index_deletion_flow/raw'
pats=[re.compile(r'高股息指數'), re.compile(r'0056')]
kw=re.compile(r'(納入|剔除|刪除|調入|調出|新增|移出|成分股)')
out=[]
files=glob.glob(ROOT+'/aRthur08701244_Intern-in-CTBC/AI_Investment/OpenIEStanza_EventEmbedding/input/EB_EventExtraction2/*.csv')+glob.glob(ROOT+'/nick_news/data/news/*.csv')+glob.glob(ROOT+'/agora0_news/**/*',recursive=True)
for f in files:
    try: txt=open(f,encoding='utf-8',errors='ignore').read()
    except Exception: continue
    for m in re.finditer(r'高股息指數|臺灣高股息|台灣高股息', txt):
        s=txt[max(0,m.start()-300):m.end()+600]
        if kw.search(s) and ('調整' in s or '納入' in s):
            out.append((f.split('/')[-1], s.replace('\n',' ')))
seen=set(); res=[]
for f,s in out:
    k=s[250:400]
    if k in seen: continue
    seen.add(k); res.append({'file':f,'snip':s})
json.dump(res,open(sys.argv[1],'w'),ensure_ascii=False,indent=0)
print(len(res))
