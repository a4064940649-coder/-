# -*- coding: utf-8 -*-
"""獨立複製（o_canslim_wide_exit）：從官方日 K 的 git 物件串流抽出 0050 的開高低收。
用法：python3 -I -B rw_e50.py <scratch_dir>   輸出 <scratch_dir>/e50_ohlc.csv
"""
import sys, os, subprocess, re

REPO = "/home/user/-/research/tw_stock_selection/raw/jb7k6295r7-bot_tw-stock-data"
END = "2026-10-22"


def main(out_dir):
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
    hl = os.path.join(out_dir, "_hashes_e50.txt")
    with open(hl, "w") as fh:
        fh.write("\n".join(h for _, h in files) + "\n")
    p = subprocess.Popen(["git", "-C", REPO, "cat-file", "--batch"], stdin=open(hl, "rb"),
                         stdout=subprocess.PIPE, env=env)
    out = ["date,open,high,low,close,volume,amount,change"]
    for d, h in files:
        hdr = p.stdout.readline().split()
        size = int(hdr[2])
        data = p.stdout.read(size).decode("utf-8")
        p.stdout.read(1)
        lines = data.split("\n")
        cols = lines[0].strip().split(",")
        ix = {c: k for k, c in enumerate(cols)}
        key = d + "_0050,"
        row = None
        for ln in lines[1:]:
            if ln.startswith(key):
                row = ln.strip().split(",")
                break
        if row is None:
            out.append(d + ",,,,,,,")
        else:
            out.append(",".join([d] + [row[ix[c]] for c in ("open", "high", "low", "close", "volume", "amount", "change")]))
    p.wait()
    with open(os.path.join(out_dir, "e50_ohlc.csv"), "w") as f:
        f.write("\n".join(out) + "\n")
    os.remove(hl)
    print("days", len(files))


if __name__ == "__main__":
    main(sys.argv[1])
