# -*- coding: utf-8 -*-
"""以已驗收圖的「部件分區」為準，用字型墨跡重新上色（去接縫，不改語意）。

為甚麼不直接用管線重畫：逐筆外形配準（v9）實測更差——把每筆輪廓縮放到它自己的
墨跡面積之後，輪廓會互相覆蓋，混色筆由 61 升到 83。在拿到可靠的逐筆外形之前，
最安全而且真實可見的改善是：**沿用已驗收圖的部件分區（人類背書），只把上色方式
由「雙線性降取樣」改成「4× 超取樣 + 最近色降取樣」（P6）**，接縫因此消失。

用法：
  python3 _src/reskin.py /tmp/skin          # 輸出到 /tmp/skin
  python3 _src/reskin.py /tmp/skin 湖 籃     # 只做指定字
"""
import os
import sys

import numpy as np
from PIL import Image

import build_parts4 as bp
import parts_auto as pa
from pngmap import lab_on_mask

SS = 4          # 超取樣倍數


def reskin(path, parts, mask):
    H, W = mask.shape
    lab, _ = lab_on_mask(path, parts, mask)
    big = np.repeat(np.repeat(lab, SS, 0), SS, 1)
    bigmask = np.repeat(np.repeat(mask, SS, 0), SS, 1)

    rgba = np.zeros(big.shape + (4,), np.uint8)
    for k, p in enumerate(parts, start=1):
        rgba[(big == k)] = (*pa.COLORS[p[1]], 255)
    rgba[~bigmask] = (0, 0, 0, 0)

    s = min(bp.OUT / (H * SS), bp.OUT / (W * SS))
    w, h = max(1, int(W * SS * s)), max(1, int(H * SS * s))
    tile = np.array(Image.fromarray(rgba).resize((w, h), Image.NEAREST))
    canvas = np.zeros((bp.BOX, bp.BOX, 4), np.uint8)
    y0, x0 = (bp.BOX - h) // 2, (bp.BOX - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = tile
    return Image.fromarray(canvas)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    dst = sys.argv[1]
    only = set("".join(sys.argv[2:]))
    os.makedirs(dst, exist_ok=True)
    n = 0
    for rec in bp.DATA + bp.EXAMPLES:
        ch, parts = rec[0], rec[3]
        if only and ch not in only:
            continue
        folder = "chars" if rec in bp.DATA else "examples"
        src = "assets/%s/%05d.png" % (folder, ord(ch))
        if not os.path.exists(src):
            continue
        data, mask, dist = pa.stroke_field(ch)
        img = reskin(src, parts, mask)
        img.save(os.path.join(dst, "%05d.png" % ord(ch)), optimize=True)
        n += 1
        print("reskin", ch, flush=True)
    print("\n完成 %d 張 → %s" % (n, dst))
    return 0


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.exit(main())
