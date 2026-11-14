#!/usr/bin/env python3
"""K 保管員：把 k_gen_plans.py 產在 scratchpad/k_stage3/ 的 V7X REAL 名單（plans_base.csv＝M1、plans_M2.csv＝M2）壓縮封存到 sealed/。
封存的是原 CSV 的**完整位元組**（欄 variant,j,seed,signal,exec,exec_idx,picks,inserted；沒有刪欄），gzip mtime=0（同內容必得同 sha256）。
封存前先核對：原 CSV 的 sha256 等於 integrate_o20/v7x_plans_check.json 記錄的當時 plans_base.csv／plans_M2.csv；封存後解壓回讀再核一次。
python3 -I -B k_seal_plans.py"""
import gzip, hashlib, io, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; AG = HERE.parent; SEAL = HERE / 'sealed'
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/k_stage3')
REF = json.loads((AG / 'integrate_o20/v7x_plans_check.json').read_text(encoding='utf-8'))['out']
out = {}
for dv, tag in (('base', 'M1'), ('M2', 'M2')):
    raw = (SCR / f'plans_{dv}.csv').read_bytes(); h = hashlib.sha256(raw).hexdigest()
    assert h == REF[dv]['sha256'], (dv, h, REF[dv]['sha256'])
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=9, mtime=0) as g: g.write(raw)
    f = SEAL / f'v7x_plans_{tag}.csv.gz'; f.write_bytes(buf.getvalue())
    back = gzip.decompress(f.read_bytes()); assert hashlib.sha256(back).hexdigest() == h and len(back) == len(raw)
    assert f.stat().st_size < 45 * 2**20
    out[tag] = dict(file=f'sealed/{f.name}', gz_bytes=f.stat().st_size, gz_sha256=hashlib.sha256(f.read_bytes()).hexdigest(), csv_bytes=len(raw), csv_sha256=h,
                    csv_sha256_eq_integrate_o20=True, integrate_o20_file=REF[dv]['file'], n_lines_incl_header=raw.count(b'\n'))
(HERE / 'k_seal_plans.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=1))
