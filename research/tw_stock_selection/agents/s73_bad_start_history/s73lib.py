#!/usr/bin/env python3
"""S73 共用：直接 import S65 的引擎副本（s65lib，不改），只把快取目錄改指到 scratchpad/s73/cache。一律 python3 -I -B。"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
sys.path.insert(0, str(AG / 's65_new_starter_entry'))
import s65lib as B                                          # noqa: E402
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s73')
CACHE = TMP / 'cache'
B.TMP = TMP; B.CACHE = CACHE
for _m, _a in ((B.s20lib, 'S20'), (B.s17lib, 'S17'), (B.s10lib, 'S10'), (B.s9lib, 'S9'), (B.s8lib, 'S8'), (B.s7lib, 'S7'), (B.s6lib, 'S6'), (B.s5lib, 'S5'),
               (B.s4lib, 'S4'), (B.s3lib, 'S3'), (B.s2lib, 'S2'), (B.stlib, 'S1'), (B.cslib, 'TMP')):
    setattr(_m, _a, CACHE)
NAVS = AG.parent.parent.parent / 'xq_scripts' / 'site' / 'navs_official.csv.gz'
CUR_S, CUR_B = -0.1327, -0.1285        # 網站模擬帳戶 2026-11-12～11-19：策略、0050（不在面板內，只當比較值）
