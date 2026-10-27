# -*- coding: utf-8 -*-
"""Best-effort recovery of truncated BIFF8 .xls (FSC case-98/99/102 are cut at 256KB).
Assumes the Workbook stream is stored in consecutive 512-byte sectors starting at the
first BOF record, and walks BIFF records (SST/CONTINUE/LABELSST/LABEL/NUMBER/RK/MULRK)
until bytes run out. Output: raw/fsc_cases/recovered_<name>.csv  (row, col, value)
Untrusted input: only parses bytes, never executes anything.
"""
import struct, sys, os, csv

def rk_val(rk):
    if rk & 2:
        v = rk >> 2
        if v & (1 << 29):
            v -= 1 << 30
        v = float(v)
    else:
        v = struct.unpack('<d', struct.pack('<Q', (rk & 0xFFFFFFFC) << 32))[0]
    if rk & 1:
        v /= 100.0
    return v


def read_unistr(buf, pos, nchars, conts):
    """read BIFF8 unicode string body (after cch); handles CONTINUE boundaries via conts list."""
    out = []
    flags = buf[pos]; pos += 1
    rich = flags & 0x08; ext = flags & 0x04; wide = flags & 0x01
    nrich = 0; next_ = 0
    if rich:
        nrich = struct.unpack_from('<H', buf, pos)[0]; pos += 2
    if ext:
        next_ = struct.unpack_from('<I', buf, pos)[0]; pos += 4
    got = 0
    while got < nchars:
        # boundary of current segment
        seg_end = next((c for c in conts if c > pos), len(buf))
        if wide:
            avail = (seg_end - pos) // 2
            take = min(avail, nchars - got)
            out.append(buf[pos:pos + 2 * take].decode('utf-16-le', errors='replace')); pos += 2 * take
        else:
            avail = seg_end - pos
            take = min(avail, nchars - got)
            out.append(buf[pos:pos + take].decode('latin-1')); pos += take
        got += take
        if got < nchars:
            if pos >= len(buf):
                break
            wide = buf[pos] & 1; pos += 1  # new option byte at CONTINUE
    pos += 4 * nrich + next_
    return ''.join(out), pos


def main(path, out):
    data = open(path, 'rb').read()
    # stream starts at first BOF (0x0809) aligned to sector boundary after header
    start = None
    for off in range(512, len(data), 512):
        if data[off:off + 2] == b'\x09\x08':
            start = off; break
    s = data[start:]
    pos = 0
    sst = []
    cells = []
    recs = []
    while pos + 4 <= len(s):
        rid, ln = struct.unpack_from('<HH', s, pos)
        body = s[pos + 4:pos + 4 + ln]
        if len(body) < ln:
            break
        recs.append((rid, body))
        pos += 4 + ln
    i = 0
    while i < len(recs):
        rid, body = recs[i]
        if rid == 0x00FC:  # SST with CONTINUEs
            buf = bytearray(body); conts = []
            j = i + 1
            while j < len(recs) and recs[j][0] == 0x003C:
                conts.append(len(buf)); buf += recs[j][1]; j += 1
            buf = bytes(buf)
            total, uniq = struct.unpack_from('<II', buf, 0)
            p = 8
            try:
                while len(sst) < uniq and p < len(buf):
                    # cch might sit right at a boundary
                    if p in conts:
                        pass
                    cch = struct.unpack_from('<H', buf, p)[0]; p += 2
                    sv, p = read_unistr(buf, p, cch, conts)
                    sst.append(sv)
            except Exception as e:
                print('sst stop', len(sst), e)
            i = j; continue
        if rid == 0x00FD and len(body) >= 10:
            r, c, xf, k = struct.unpack_from('<HHHI', body, 0)
            cells.append((r, c, sst[k] if k < len(sst) else ''))
        elif rid == 0x0203 and len(body) >= 14:
            r, c, xf = struct.unpack_from('<HHH', body, 0)
            cells.append((r, c, struct.unpack_from('<d', body, 6)[0]))
        elif rid == 0x027E and len(body) >= 10:
            r, c, xf, rk = struct.unpack_from('<HHHI', body, 0)
            cells.append((r, c, rk_val(rk)))
        elif rid == 0x00BD and len(body) >= 6:
            r, c0 = struct.unpack_from('<HH', body, 0)
            n = (len(body) - 6) // 6
            for k in range(n):
                xf, rk = struct.unpack_from('<HI', body, 4 + 6 * k)
                cells.append((r, c0 + k, rk_val(rk)))
        elif rid == 0x0204 and len(body) >= 8:  # LABEL (old)
            r, c, xf, cch = struct.unpack_from('<HHHH', body, 0)
            try:
                sv, _ = read_unistr(body, 8, cch, [])
            except Exception:
                sv = ''
            cells.append((r, c, sv))
        i += 1
    with open(out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['row', 'col', 'value'])
        for r, c, v in cells:
            w.writerow([r, c, v])
    print(path, 'bytes', len(data), 'records', len(recs), 'sst', len(sst), 'cells', len(cells),
          'maxrow', max((r for r, _, _ in cells), default=-1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
