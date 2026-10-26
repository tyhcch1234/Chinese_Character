# -*- coding: utf-8 -*-
"""逐筆診斷：對照「應該屬於哪個部件」（parts_auto 的 part_of）與「圖上實際顏色」。

用法：
  python3 _src/inspect_char.py 湖 assets/chars/28246.png
  python3 _src/inspect_char.py 湖 /tmp/auto_v2/28246.png
"""
import os
import sys

import numpy as np

import build_parts4 as bp
import parts_auto as pa
from audit import label_from_png


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    ch, path = sys.argv[1], sys.argv[2]
    rec = next(r for r in bp.DATA + bp.EXAMPLES if r[0] == ch)
    structure, parts = rec[1], rec[3]
    names = [p[0] for p in parts]
    roles = [p[1] for p in parts]

    data, mask, dist = pa.stroke_field(ch)
    part_of = pa.assign_parts(ch, structure, parts, data, mask, dist)
    lock = pa.identity_lock(ch)
    if lock and len(lock) == len(part_of):
        part_of = np.array([v if 1 <= v <= len(parts) else int(part_of[i])
                            for i, v in enumerate(lock)], np.int16)
    lab, alpha = label_from_png(path, parts)

    H, W = mask.shape
    s = min(bp.OUT / H, bp.OUT / W)
    w, h = max(1, int(W * s)), max(1, int(H * s))
    own = pa.stroke_own(ch) - 1
    box = np.zeros((bp.BOX, bp.BOX), np.uint8)
    box[(bp.BOX - h) // 2:(bp.BOX - h) // 2 + h,
        (bp.BOX - w) // 2:(bp.BOX - w) // 2 + w] = np.array(
        pa.Image.fromarray(np.clip(own, 0, 255).astype(np.uint8)).resize((w, h), pa.Image.NEAREST))
    o2 = np.array(pa.Image.fromarray(box).resize((lab.shape[1], lab.shape[0]), pa.Image.NEAREST))

    print("字 %s（%s）部件：%s" % (ch, structure,
                                 "、".join("%s[%s]" % (n, r) for n, r in zip(names, roles))))
    print("筆畫數 %d" % dist.shape[0])
    bad = 0
    for i in range(dist.shape[0]):
        sel = (o2 == i) & alpha
        tot = int(sel.sum())
        if tot < 20:
            continue
        vals, cnt = np.unique(lab[sel], return_counts=True)
        obs = {names[v - 1]: int(c) for v, c in zip(vals, cnt) if v > 0}
        want = names[part_of[i] - 1]
        ok = (len(obs) == 1 and list(obs)[0] == want and list(obs.values())[0] / tot > 0.95)
        if not ok:
            bad += 1
        dom = max(obs.items(), key=lambda kv: kv[1])[0] if obs else "-"
        share = max(obs.values()) / tot if obs else 0
        flag = "OK " if ok else "BAD"
        print("  %s 筆%02d  應為 %-4s  實見 %-4s (%.0f%%) %s" % (
            flag, i + 1, want, dom, 100 * share, "" if ok else obs))
    print("不合格筆畫 %d / %d" % (bad, dist.shape[0]))


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
