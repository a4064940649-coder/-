# -*- coding: utf-8 -*-
"""S3 獨立複製：掃官方日 K 每日檔的 name 欄，找出創新板股票（股名結尾是「-創」或「-KY創」等帶「創」字尾標記）。
用法：python3 -I -B rep_s3_names.py <out_json>
輸出：{代碼: {first, last, n_days, names:[...], last_name_any, last_date_any}}
只讀 git 物件，不讀研究員任何檔案。
"""
import sys, os, io, csv, subprocess, re, json, tempfile

REPO = "/home/user/-/research/tw_stock_selection/raw/jb7k6295r7-bot_tw-stock-data"
END = "2026-10-22"
PAT = re.compile(r"-(KY)?創$")


def main(out):
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1")
    ls = subprocess.run(["git", "-C", REPO, "ls-tree", "HEAD", "data/universe/daily/"],
                        capture_output=True, text=True, env=env, check=True).stdout
    files = []
    for line in ls.splitlines():
        meta, path = line.split("\t")
        m = re.search(r"(\d{4}-\d{2}-\d{2})\.csv$", path)
        if m and m.group(1) <= END:
            files.append((m.group(1), meta.split()[2]))
    files.sort()
    tf = tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(out))
    tf.write("\n".join(h for _, h in files) + "\n"); tf.close()
    p = subprocess.Popen(["git", "-C", REPO, "cat-file", "--batch"], stdin=open(tf.name, "rb"),
                         stdout=subprocess.PIPE, env=env)
    res = {}; loose = {}
    for d, h in files:
        hdr = p.stdout.readline().split()
        size = int(hdr[2])
        data = p.stdout.read(size).decode("utf-8")
        p.stdout.read(1)
        if "創" not in data:
            continue
        lines = data.split("\n")
        head = lines[0]
        sel = [x for x in lines[1:] if "創" in x]
        for r in csv.DictReader(io.StringIO(head + "\n" + "\n".join(sel))):
            sid = r["stock_id"]; nm = r["name"].strip()
            if len(sid) != 4 or not sid.isdigit() or sid[0] == "0" or sid[:2] == "91":
                continue
            if PAT.search(nm):
                e = res.setdefault(sid, dict(first=d, last=d, n_days=0, names=[], market=set()))
                e["last"] = d; e["n_days"] += 1; e["market"].add(r["market"])
                if nm not in e["names"]:
                    e["names"].append(nm)
            elif nm.endswith("創") and ("-" in nm):
                loose.setdefault(sid, set()).add(nm)          # 其他可疑寫法，只列出來人工看
    p.wait(); os.remove(tf.name)
    for e in res.values():
        e["market"] = sorted(e["market"])
    json.dump(dict(innov=res, other_suspect={k: sorted(v) for k, v in loose.items()}), open(out, "w"),
              ensure_ascii=False, indent=1)
    print("innovation-board codes:", len(res), "other suspect:", {k: sorted(v) for k, v in loose.items()})


if __name__ == "__main__":
    main(sys.argv[1])
