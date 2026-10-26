# -*- coding: utf-8 -*-
"""非循環的純度驗證：只看「筆畫幾何（字型）」與「PNG 像素顏色」，不看 part_of。

檢查三項（任何一項 > 0 即 fail）：
  1. 一筆多色：某筆墨跡中非主色佔比 ≥ 2% 的筆數
  2. 未著色：墨跡像素中顏色不屬於任何部件的像素數
  3. 顏色數：圖上出現的部件顏色數 ≠ 部件數

座標映射用 `pngmap`（render 的嚴格逆運算），不做整圖縮放。

用法：
  python3 _src/verify.py assets              # 檢查現有圖
  python3 _src/verify.py /tmp/auto_v4        # 檢查新輸出
  python3 _src/verify.py /tmp/auto_v4 湖 幫   # 指定字
"""
import os
import sys

import numpy as np
from scipy import ndimage

import build_parts4 as bp
import parts_auto as pa
from pngmap import lab_on_mask

MULTI_TOL = 0.02


_CUR = [None]


def measure(path, parts, mask, dist):
    lab, ink = lab_on_mask(path, parts, mask)
    # 只量筆畫內部：邊界 2px 的鋸齒／重取樣誤差不列入「未著色」
    ink = ndimage.binary_erosion(ink, iterations=2)
    own = pa.stroke_own(_CUR[0])
    owner0 = np.where(own > 0, own - 1, np.argmin(dist, axis=0))
    multi, unpainted = 0, 0
    for i in range(dist.shape[0]):
        sel = (owner0 == i) & ink
        tot = int(sel.sum())
        if tot < 20:
            continue
        painted = sel & (lab > 0)
        unpainted += tot - int(painted.sum())
        if painted.sum() < 20:
            continue
        vals, cnt = np.unique(lab[painted], return_counts=True)
        if cnt.max() / cnt.sum() < 1 - MULTI_TOL:
            multi += 1
    used = len(set(np.unique(lab).tolist()) - {0})
    return multi, unpainted, used


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    base = sys.argv[1]
    only = set("".join(sys.argv[2:]))
    fail, rows, total = 0, [], 0
    for rec in bp.DATA + bp.EXAMPLES:
        ch, structure, parts = rec[0], rec[1], rec[3]
        if only and ch not in only:
            continue
        folder = "chars" if rec in bp.DATA else "examples"
        name = "%05d.png" % ord(ch)
        path = os.path.join(base, name) if base != "assets" else "assets/%s/%s" % (folder, name)
        if not os.path.exists(path):
            continue
        total += 1
        data, mask, dist = pa.stroke_field(ch)
        _CUR[0] = ch
        multi, unpainted, used = measure(path, parts, mask, dist)
        k = len(parts)
        if multi or unpainted or used != k:
            fail += 1
            rows.append(ch)
            print("%-2s FAIL 多色筆%d 未著色%d 顏色數%d/%d" % (ch, multi, unpainted, used, k),
                  flush=True)
    print("\n%s：合格 %d / %d%s" % (base, total - fail, total,
                                  ("  不合格：" + " ".join(rows)) if rows else ""))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.exit(main())
