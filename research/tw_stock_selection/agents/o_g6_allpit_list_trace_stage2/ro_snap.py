# R0 唯讀快照：live_tracking/main_forward、o_forward_infra_live、o_g8p2a_protocol_addendum 逐檔 sha256＋mtime
import hashlib, json, os, sys, datetime
BASE='/home/user/-/research/tw_stock_selection/agents'
DIRS=['live_tracking/main_forward','o_forward_infra_live','o_g8p2a_protocol_addendum']
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{}}
for d in DIRS:
    for root,_,fs in os.walk(os.path.join(BASE,d)):
        for f in fs:
            p=os.path.join(root,f)
            try:
                h=hashlib.sha256(open(p,'rb').read()).hexdigest()
                out['files'][os.path.relpath(p,BASE)]={'sha256':h,'mtime':os.path.getmtime(p)}
            except Exception as e:
                out['files'][os.path.relpath(p,BASE)]={'err':str(e)}
out['n']=len(out['files'])
json.dump(out,open(sys.argv[1],'w'),ensure_ascii=False,indent=0)
print(out['n'],out['utc'])
