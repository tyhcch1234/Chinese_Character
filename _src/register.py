# -*- coding: utf-8 -*-
"""逐筆外形配準（帶遮擋懲罰）——P5 的落地。

上一版（v9）用「輪廓面積 = 該筆 Voronoi 面積」求縮放，等於把最近中心線的錯誤分區
再放大一次，混色筆由 61 升到 83。這裡改成**有目標函數的搜尋**：

    對每一筆，在 (scale, dx, dy) 上搜尋，使
      得分 = +1.0 × 輪廓蓋到的字型墨跡
             − 1.5 × 輪廓蓋到的空白
             − 2.0 × 輪廓蓋到的「別筆核心」
    「別筆核心」= 該像素的最近中心線是別筆，且距離 < CORE px。
    這樣輪廓會盡量貼住自己的墨跡，同時被罰「搶別人的筆」。

搜尋在 1/4 解析度上做（275×275 左右），每筆 5×5×5=125 個候選，全部 49 字約 1–2 分鐘。

用法：
  python3 _src/register.py 湖          # 印出一筆一筆的最終參數與得分
  python3 _src/register.py            # 全部字（只印改動幅度）
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

import build_parts4 as bp
import parts_auto as pa

DOWN = 4            # 搜尋解析度（1/DOWN）
CORE = 5.0          # 「別筆核心」半徑（全解析度 px）
S_GRID = (0.80, 0.90, 1.00, 1.10, 1.20)
D_GRID = (-8.0, -4.0, 0.0, 4.0, 8.0)


def _poly_arrays(data):
    out = []
    for d in data["strokes"]:
        polys = [np.array([(x, pa.BASELINE - y) for x, y in poly], float)
                 for poly in bp.svg_polys(d)]
        out.append([q for q in polys if len(q) >= 3])
    return out


def _centroid(polys):
    area = cx = cy = 0.0
    for q in polys:
        a = 0.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], -1))
                      - np.dot(q[:, 1], np.roll(q[:, 0], -1)))
        area += a
        cx += a * q[:, 0].mean()
        cy += a * q[:, 1].mean()
    if area <= 1e-6:
        return None
    return cx / area, cy / area, area


def solve(ch):
    """回傳 [(s, dx, dy, score)]，與 stroke 同序；無外形者為 None。"""
    data, mask, dist = pa.stroke_field(ch)
    H, W = mask.shape
    hl, wl = H // DOWN, W // DOWN

    # 全域仿射（與 stroke_field 完全相同，來自快取的 fit）
    z = np.load(os.path.join(pa.GEOM, "%05d.npz" % ord(ch)))
    s_best, tx_, ty_, ax_, cx_, cy_ = [float(v) for v in z["fit"]]

    def to_px(pt):
        return ((pt[0] - cx_) * s_best * ax_ + tx_, (pt[1] - cy_) * s_best + ty_)

    ink_l = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (wl, hl), Image.BILINEAR)) > 127
    core_l = np.array(Image.fromarray(
        (np.argmin(dist, axis=0) * 40).clip(0, 255).astype(np.uint8)).resize(
        (wl, hl), Image.NEAREST))
    near_index = np.argmin(dist, axis=0)
    core_mask = (dist.min(axis=0) < CORE)
    others = np.zeros((hl, wl), np.uint8)
    idx_l = np.array(Image.fromarray((near_index + 1).clip(0, 255).astype(np.uint8)).resize(
        (wl, hl), Image.NEAREST))
    cm_l = np.array(Image.fromarray((core_mask * 255).astype(np.uint8)).resize(
        (wl, hl), Image.NEAREST)) > 127

    polys = _poly_arrays(data)
    out = []
    for i, qs in enumerate(polys):
        c = _centroid(qs)
        if c is None:
            out.append(None)
            continue
        pcx, pcy, _ = c
        gx, gy = to_px((pcx, pcy))

        best = (-1e9, 1.0, 0.0, 0.0)
        for s in S_GRID:
            for dx in D_GRID:
                for dy in D_GRID:
                    lay = Image.new("L", (wl, hl), 0)
                    dr = ImageDraw.Draw(lay)
                    for q in qs:
                        dr.polygon([(((x - pcx) * s * s_best * ax_ + gx + dx) / DOWN,
                                     ((y - pcy) * s * s_best + gy + dy) / DOWN)
                                    for x, y in q], fill=255)
                    inside = np.array(lay) > 127
                    if not inside.any():
                        continue
                    hit = inside & ink_l
                    miss = inside & ~ink_l
                    steal = inside & cm_l & (idx_l != i + 1)
                    score = (hit.sum() - 1.5 * miss.sum() - 2.0 * steal.sum())
                    if score > best[0]:
                        best = (score, s, dx, dy)
        out.append((best[1], best[2], best[3], best[0]))
    return out


def main():
    only = "".join(sys.argv[1:])
    items = bp.DATA + bp.EXAMPLES
    for rec in items:
        ch = rec[0]
        if only and ch != only:
            continue
        sol = solve(ch)
        if not only:
            moved = sum(1 for s in sol if s and (abs(s[1]) > 1e-6 or abs(s[2]) > 1e-6
                                                 or abs(s[0] - 1) > 1e-6))
            print("%-2s 逐筆調整 %d/%d" % (ch, moved, len([s for s in sol if s])), flush=True)
        else:
            print("字 %s：%d 筆" % (ch, len(sol)))
            for i, s in enumerate(sol):
                print("  筆%02d %s" % (i + 1, "無外形" if s is None else
                                      "scale %.2f  dx %+5.1f  dy %+5.1f  score %.0f" % s))


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
