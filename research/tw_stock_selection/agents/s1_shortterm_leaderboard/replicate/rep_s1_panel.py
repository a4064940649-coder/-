# -*- coding: utf-8 -*-
"""S1 獨立複製：價格面板建置（自寫；從 git 物件串流讀官方日 K，逐檔直接寫入陣列，省記憶體）。
未讀、未 import 研究員的 stlib.py／run.py／cslib.py。
用法：python3 -I -B rep_s1_panel.py <scratch_dir>
輸出：<scratch_dir>/panel.npz（鍵名與 o_ext_canslim_xs/replicate/rep_panel.py 相同，可給 rep_lib.load_panel 讀）
     <scratch_dir>/names.json（代號→最後一次出現的名稱；判斷 KY 用）
"""
import sys, os, io, csv, subprocess, re, json, resource
import numpy as np

REPO = "/home/user/-/research/tw_stock_selection/raw/jb7k6295r7-bot_tw-stock-data"
END = "2026-10-22"
MK = {"twse": 1, "tpex": 2}
NMAX = 3200


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
    def mat(fill=np.nan, dt=np.float64):
        return np.full((T, NMAX), fill, dtype=dt)
    O, H, L, C, V, A, G = mat(), mat(), mat(), mat(), mat(), mat(), mat()
    M = mat(0, np.int8); P = mat(0, np.int8); NT = mat(0, np.int8); LF = mat(0, np.int8)
    lim = {"": 0, "up": 1, "down": 2, "flat": 3}
    ci = {}; names = {}; nrows = 0
    e50 = {}
    for t, (d, h) in enumerate(files):
        hdr = p.stdout.readline().split()
        size = int(hdr[2])
        data = p.stdout.read(size)
        p.stdout.read(1)
        rd = csv.DictReader(io.StringIO(data.decode("utf-8")))
        for r in rd:
            sid = r["stock_id"]
            if sid == "0050":
                e50[d] = (fnum(r["open"]), fnum(r["close"]))
            if len(sid) != 4 or not sid.isdigit() or sid[0] == "0" or sid[:2] == "91":
                continue
            mk = MK.get(r["market"])
            if mk is None:
                continue
            assert r["date"] == d
            i = ci.get(sid)
            if i is None:
                i = len(ci); ci[sid] = i
                assert i < NMAX
            names[sid] = r["name"]
            O[t, i] = fnum(r["open"]); H[t, i] = fnum(r["high"]); L[t, i] = fnum(r["low"]); C[t, i] = fnum(r["close"])
            V[t, i] = fnum(r["volume"]); A[t, i] = fnum(r["amount"]); G[t, i] = fnum(r["change"])
            M[t, i] = mk; P[t, i] = 1; NT[t, i] = 1 if r["price_basis"].strip() else 0
            LF[t, i] = lim.get(r["limit"].strip(), 9)
            nrows += 1
    p.wait()
    os.remove(hl)
    codes = sorted(ci)
    order = np.array([ci[c] for c in codes])
    np.savez_compressed(os.path.join(out_dir, "panel.npz"), dates=np.array(dates), codes=np.array(codes),
                        O=O[:, order], H=H[:, order], L=L[:, order], C=C[:, order], V=V[:, order], A=A[:, order],
                        G=G[:, order], M=M[:, order], P=P[:, order], NT=NT[:, order], LF=LF[:, order])
    json.dump(names, open(os.path.join(out_dir, "names.json"), "w"), ensure_ascii=False)
    json.dump(e50, open(os.path.join(out_dir, "e50_open_close.json"), "w"))
    print("days", T, dates[0], dates[-1], "codes", len(codes), "rows", nrows,
          "peakRSS_MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)


if __name__ == "__main__":
    main(sys.argv[1])
