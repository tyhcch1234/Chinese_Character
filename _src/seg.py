# -*- coding: utf-8 -*-
"""以字型墨跡做「測地分水嶺」，把墨跡切成逐筆區域（替代 hanzi 輪廓）。

為甚麼不用 hanzi 外形：它是另一套字體（「湖」的「古」橫起點 x≈109，源流明體約
x≈330），輪廓必然覆蓋鄰筆墨跡，任何以輪廓決定歸屬的方法都會切出楔形。

分水嶺的性質正好符合 P4「分界線只可落在空白走廊或筆畫交叉點」：
先在墨跡上求 distance transform（離筆畫邊界的深度），再以各筆中心線貼合後的
核心為種子泛濫——兩條筆畫相接處是窄頸（深度淺），泛濫會在那裡相遇，於窄頸切開，
而不是像 Voronoi 那樣用直線距離硬切。

用法：
  python3 _src/seg.py 湖      # 印出每筆像素數與最窄頸位置
"""
import os
import sys

import numpy as np
from scipy import ndimage

SEED_R = 3.0        # 種子半徑：中心線 3px 以內的墨跡像素
FALLBACK = 6.0


def ink_labels(mask, dist):
    """回傳 (labels, seeds)：labels 為 1..n 的逐筆區域，0 = 未分。"""
    from skimage.segmentation import watershed

    n = dist.shape[0]
    depth = ndimage.distance_transform_edt(mask)
    near = np.argmin(dist, axis=0)
    dmin = dist.min(axis=0)

    markers = np.zeros(mask.shape, np.int32)
    for i in range(n):
        sel = mask & (near == i) & (dmin <= SEED_R)
        if not sel.any():                       # 這筆太細 → 放寬
            sel = mask & (near == i) & (dmin <= FALLBACK)
        markers[sel] = i + 1

    # 有些筆完全被鄰筆的種子蓋過（墨跡相連得很緊），強制在它最近墨跡處放一個種子
    yy, xx = np.mgrid[-2:3, -2:3]
    disk = (yy ** 2 + xx ** 2) <= 4
    for i in range(n):
        if i + 1 in markers:
            continue
        dm = np.where(mask, dist[i], np.inf)
        if not np.isfinite(dm).any():
            continue
        y, x = np.unravel_index(np.argmin(dm), dm.shape)
        ys = slice(max(0, y - 2), min(mask.shape[0], y + 3))
        xs = slice(max(0, x - 2), min(mask.shape[1], x + 3))
        sub = markers[ys, xs]
        patch = disk[:sub.shape[0], :sub.shape[1]] & mask[ys, xs]
        sub[patch] = i + 1
        markers[ys, xs] = sub

    labels = watershed(-depth, markers, mask=mask, watershed_line=False)
    holes = mask & (labels == 0)
    if holes.any():                             # 沒被泛濫到的墨跡 → 最近中心線
        labels = np.where(holes, near.astype(np.int32) + 1, labels)
    return labels.astype(np.int16), markers


def main():
    import build_parts4 as bp
    import parts_auto as pa

    only = set("".join(sys.argv[1:]))
    for rec in bp.DATA + bp.EXAMPLES:
        ch = rec[0]
        if only and ch not in only:
            continue
        data, mask, dist = pa.stroke_field(ch)
        lab, seeds = ink_labels(mask, dist)
        counts = np.bincount(lab[mask], minlength=dist.shape[0] + 1)[1:]
        empty = [i + 1 for i, c in enumerate(counts) if c == 0]
        print("%-2s 筆數 %d  種子 %d  空筆 %s  最小筆 %d px"
              % (ch, dist.shape[0], len(np.unique(seeds)) - 1, empty or "—",
                 counts.min()), flush=True)


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
