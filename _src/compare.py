# -*- coding: utf-8 -*-
"""兩組圖逐字逐筆比對（完全不看 part_of）：只比較「同一筆在 A、B 兩圖的顏色」。

輸出每字：
  換色筆 n   —— 同一筆在兩邊主要顏色不同（語意改變）
  A/B 混色筆 —— 該筆在該圖中主色佔比 < 98%（視覺上有雜色／接縫）

用法：
  python3 _src/compare.py assets /tmp/auto_v6
  python3 _src/compare.py assets /tmp/auto_v6 湖 籃 慚
"""
import os
import sys

import build_parts4 as bp
import parts_auto as pa
from pngmap import lab_on_mask, stroke_parts

PURE = 0.98


def path_of(base, folder, name):
    return os.path.join(base, name) if base != "assets" else "assets/%s/%s" % (folder, name)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    A, B = sys.argv[1], sys.argv[2]
    only = set("".join(sys.argv[3:]))
    tot_switch = tot_mixA = tot_mixB = 0
    rows = []
    for rec in bp.DATA + bp.EXAMPLES:
        ch, parts = rec[0], rec[3]
        if only and ch not in only:
            continue
        folder = "chars" if rec in bp.DATA else "examples"
        name = "%05d.png" % ord(ch)
        pa_, pb_ = path_of(A, folder, name), path_of(B, folder, name)
        if not (os.path.exists(pa_) and os.path.exists(pb_)):
            continue
        data, mask, dist = pa.stroke_field(ch)
        a = stroke_parts(pa_, parts, mask, dist, ch)
        b = stroke_parts(pb_, parts, mask, dist, ch)
        names = [p[0] for p in parts]
        la, _ = lab_on_mask(pa_, parts, mask)
        lb, _ = lab_on_mask(pb_, parts, mask)
        gap = 100.0 * float((la[mask] != lb[mask]).mean())
        switch = [(i + 1, a[i][0], b[i][0]) for i in range(len(a))
                  if a[i][0] and b[i][0] and a[i][0] != b[i][0]]
        mixA = sum(1 for p, s in a if 0 < s < PURE)
        mixB = sum(1 for p, s in b if 0 < s < PURE)
        tot_switch += len(switch)
        tot_mixA += mixA
        tot_mixB += mixB
        if gap > 1.0:
            rows.append(ch)
            print("%-2s 像素差 %.1f%%（換色%d A混色%d B混色%d）" % (ch, gap, len(switch), mixA, mixB), flush=True)
        if switch or mixA or mixB:
            rows.append(ch)
            msg = "%-2s 換色%d A混色%d B混色%d" % (ch, len(switch), mixA, mixB)
            if switch:
                msg += "  " + "；".join("筆%02d %s→%s" % (i, names[x - 1], names[y - 1])
                                        for i, x, y in switch[:4])
            print(msg, flush=True)
    n = len([r for r in bp.DATA + bp.EXAMPLES if not only or r[0] in only])
    print("\n== 合計 ==  換色筆 %d  A(舊)混色筆 %d  B(新)混色筆 %d" % (tot_switch, tot_mixA, tot_mixB))
    print("有差異的字：%s" % (" ".join(rows) if rows else "—"))
    return 0


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.exit(main())
