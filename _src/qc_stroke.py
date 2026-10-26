# -*- coding: utf-8 -*-
"""筆畫層純度：找「一筆被切成兩色」的殘留色塊，且該處沒有任何其他筆畫經過。

只看像素顏色 + 字型墨跡幾何，不讀 part_of。
判定：對第 i 筆的墨跡（分水嶺區域，腐蝕 2px 去掉邊緣），
      主色以外的像素若離「其他任何一筆」的中心線都 > 12px，
      就代表這裡沒有筆畫交叉，是錯色殘留（P1 違反）。
輸出：字、筆號、殘留面積、顏色、位置。

  python3 _src/qc_stroke.py             # 全部
  python3 _src/qc_stroke.py 氣聲國默烈   # 指定字
"""
import os
import sys

import numpy as np
from scipy import ndimage

import build_parts4 as bp
import parts_auto as pa
from pngmap import lab_on_mask

ST = ndimage.generate_binary_structure(2, 1)
NAME = {1: "red", 2: "blue", 3: "green", 4: "orange"}


def scan(ch, parts, path, gap=12.0, min_area=0.002):
    data, mask, dist = pa.stroke_field(ch)
    own = pa.stroke_own(ch)
    lab, _ = lab_on_mask(path, parts, mask)
    ink = mask.astype(bool)
    H, W = lab.shape
    ink_area = int(ink.sum())
    found = []
    for i in range(dist.shape[0]):
        reg = (own == i + 1) & ink
        if reg.sum() < 200:
            continue
        core = ndimage.binary_erosion(reg, ST, iterations=2) & (lab > 0)
        if core.sum() < 100:
            continue
        vals, cnt = np.unique(lab[core], return_counts=True)
        body = int(vals[np.argmax(cnt)])
        # 其他筆畫的最近距離
        others = np.delete(dist, i, axis=0).min(axis=0)
        stray = core & (lab != body) & (others > gap)
        if not stray.any():
            continue
        lb, nl = ndimage.label(stray, ST)
        for j in range(1, nl + 1):
            m = lb == j
            a = int(m.sum())
            if a < min_area * ink_area:
                continue
            ys, xs = np.nonzero(m)
            vals2, cnt2 = np.unique(lab[m], return_counts=True)
            found.append(dict(stroke=i + 1, body=body, wrong=int(vals2[np.argmax(cnt2)]),
                              area=a, x=float(xs.mean() / W), y=float(ys.mean() / H)))
    return found, ink_area


def main():
    only = set("".join(a for a in sys.argv[1:] if not a.startswith("-")))
    total = 0
    for rec in bp.DATA + bp.EXAMPLES:
        ch = rec[0]
        if only and ch not in only:
            continue
        folder = "chars" if rec in bp.DATA else "examples"
        path = "assets/%s/%05d.png" % (folder, ord(ch))
        if not os.path.exists(path):
            continue
        found, area = scan(ch, rec[3], path)
        if not found:
            continue
        total += 1
        txt = "  ".join("s%d %s>%s %dpx @(%.2f,%.2f)" % (
            f["stroke"], NAME[f["body"]], NAME[f["wrong"]], f["area"], f["x"], f["y"])
            for f in found)
        print("%s  %s" % (ch, txt))
    print("\n有殘留色的字：%d" % total)
    return 0


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.exit(main())
