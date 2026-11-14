# -*- coding: utf-8 -*-
"""獨立複製：價格面板建置（從 git 物件串流讀官方日 K，不展開工作目錄）。
用法：python3 -I -B rep_panel.py <scratch_dir>
輸出：<scratch_dir>/panel.npz（暫存，結束時刪除）
"""
import sys, os, io, csv, subprocess, re
import numpy as np

REPO = "/home/user/-/research/tw_stock_selection/raw/jb7k6295r7-bot_tw-stock-data"
END = "2026-10-22"
MK = {"twse": 1, "tpex": 2}


def fnum(s):
    if s is None:
        return np.nan
    s = s.strip().replace(",", "")
    if s in ("", "--", "---", "-"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


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
    dates = [d for d, _ in files]
    T = len(dates)
    hl = os.path.join(out_dir, "_hashes.txt")
    with open(hl, "w") as fh:
        fh.write("\n".join(h for _, h in files) + "\n")
    p = subprocess.Popen(["git", "-C", REPO, "cat-file", "--batch"], stdin=open(hl, "rb"),
                         stdout=subprocess.PIPE, env=env)
    rows = []  # (t, code, mkt, o,h,l,c,vol,amt,chg, notrade)
    code_set = {}
    for t, (d, h) in enumerate(files):
        hdr = p.stdout.readline().split()
        size = int(hdr[2])
        data = p.stdout.read(size)
        p.stdout.read(1)
        rd = csv.DictReader(io.StringIO(data.decode("utf-8")))
        for r in rd:
            sid = r["stock_id"]
            if len(sid) != 4 or not sid.isdigit() or sid[0] == "0":
                continue
            mk = MK.get(r["market"])
            if mk is None:
                continue
            assert r["date"] == d
            if sid not in code_set:
                code_set[sid] = len(code_set)
            rows.append((t, sid, mk, fnum(r["open"]), fnum(r["high"]), fnum(r["low"]), fnum(r["close"]),
                         fnum(r["volume"]), fnum(r["amount"]), fnum(r["change"]),
                         1 if r["price_basis"].strip() else 0, r["limit"].strip()))
    p.wait()
    codes = sorted(code_set)
    ci = {c: i for i, c in enumerate(codes)}
    N = len(codes)
    def mat(fill=np.nan, dt=np.float64):
        return np.full((T, N), fill, dtype=dt)
    O, H, L, C, V, A, G = mat(), mat(), mat(), mat(), mat(), mat(), mat()
    M = mat(0, np.int8); P = mat(0, np.int8); NT = mat(0, np.int8); LF = mat(0, np.int8)
    lim = {"": 0, "up": 1, "down": 2, "flat": 3}
    for (t, sid, mk, o, h, l, c, v, a, g, nt, lf) in rows:
        i = ci[sid]
        O[t, i] = o; H[t, i] = h; L[t, i] = l; C[t, i] = c; V[t, i] = v; A[t, i] = a; G[t, i] = g
        M[t, i] = mk; P[t, i] = 1; NT[t, i] = nt; LF[t, i] = lim.get(lf, 9)
    np.savez_compressed(os.path.join(out_dir, "panel.npz"), dates=np.array(dates), codes=np.array(codes),
                        O=O, H=H, L=L, C=C, V=V, A=A, G=G, M=M, P=P, NT=NT, LF=LF)
    print("days", T, dates[0], dates[-1], "codes", N, "rows", len(rows))


if __name__ == "__main__":
    main(sys.argv[1])
