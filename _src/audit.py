# -*- coding: utf-8 -*-
"""筆畫級純度審核：不受鋸齒影響，直接量「一筆幾色 / 部件異色」。

用法：
  python3 _src/audit.py assets                        # 只審舊圖
  python3 _src/audit.py assets /tmp/auto_all          # 新舊對審
  python3 _src/audit.py assets /tmp/auto_all 思湖國    # 只審指定字
"""
import os
import sys

import numpy as np
from PIL import Image

import build_parts4 as bp
import parts_auto as pa

ORG = os.getcwd()
OUT, BOX = bp.OUT, bp.BOX


def label_from_png(path, parts):
    im = Image.open(path).convert("RGBA")
    rgba = np.array(im)
    rgb = rgba[:, :, :3].astype(int)
    alpha = rgba[:, :, 3] > 128
    lab = np.zeros(alpha.shape, np.uint8)
    for k, p in enumerate(parts, start=1):
        lab[(np.abs(rgb - np.array(pa.COLORS[p[1]], int)).sum(2) < 70) & alpha] = k
    return lab, alpha


def audit(ch, structure, parts, path):
    data, mask, dist = pa.stroke_field(ch)
    part_of = pa.assign_parts(ch, structure, parts, data, mask, dist)
    H, W = mask.shape
    s = min(OUT / H, OUT / W)
    w, h = max(1, int(W * s)), max(1, int(H * s))
    own, _, _ = pa.lane_owner(mask, dist, structure, len(parts))
    o = np.clip(own, 0, 255).astype(np.uint8)
    box = np.zeros((BOX, BOX), np.uint8)
    box[(BOX - h) // 2:(BOX - h) // 2 + h, (BOX - w) // 2:(BOX - w) // 2 + w] = np.array(
        Image.fromarray(o).resize((w, h), Image.NEAREST))

    lab, alpha = label_from_png(path, parts)
    o2 = np.array(Image.fromarray(box).resize((lab.shape[1], lab.shape[0]),
                                              Image.NEAREST))
    ink = alpha & (o2 >= 0)

    strokes_multi, worst = 0, 0.0
    n_stroke = int(dist.shape[0])
    for i in range(n_stroke):
        sel = ink & (o2 == i) & (lab > 0)
        tot = int(sel.sum())
        if tot < 20:
            continue
        vals, cnt = np.unique(lab[sel], return_counts=True)
        if len(vals) > 1:
            share = 1.0 - cnt.max() / cnt.sum()
            if share >= 0.05:
                strokes_multi += 1
                worst = max(worst, share)

    foreign, terr_tot = 0, 0
    for k in range(1, len(parts) + 1):
        strokes = [i for i in range(n_stroke) if part_of[i] == k]
        if not strokes:
            continue
        terr = np.isin(o2, strokes) & ink
        terr_tot += int(terr.sum())
        foreign += int((terr & (lab != k) & (lab > 0)).sum())

    unpainted = int((ink & (lab == 0)).sum())
    return dict(strokes_multi=strokes_multi, worst_share=worst,
                foreign=foreign, terr=terr_tot,
                foreign_pct=100.0 * foreign / max(1, terr_tot),
                unpainted=unpainted)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    sets = [sys.argv[1]] + ([sys.argv[2]] if len(sys.argv) > 2 and os.path.isdir(sys.argv[2]) else [])
    only = set("".join(sys.argv[2:])) if not sys.argv[2:3] or not os.path.isdir(sys.argv[2:3][0]) else set(
        "".join(sys.argv[3:]))
    items = bp.DATA + bp.EXAMPLES

    rows = []
    for rec in items:
        ch, structure, parts = rec[0], rec[1], rec[3]
        if only and ch not in only:
            continue
        folder = "chars" if rec in bp.DATA else "examples"
        rel = "assets/%s/%05d.png" % (folder, ord(ch))
        if not os.path.exists(rel):
            continue
        res = []
        for base in sets:
            path = rel if base == "assets" else os.path.join(base, "%05d.png" % ord(ch))
            res.append(audit(ch, structure, parts, path) if os.path.exists(path) else None)
        rows.append((ch, res))
        msg = "%-2s" % ch
        for base, r in zip(sets, res):
            if r is None:
                msg += "   %s: MISSING" % os.path.basename(base)
            else:
                msg += "   %s: 多色筆%d(最差%.0f%%) 異色%.2f%% 未著色%d" % (
                    os.path.basename(base), r["strokes_multi"], 100 * r["worst_share"],
                    r["foreign_pct"], r["unpainted"])
        print(msg, flush=True)

    print("\n== 總結 ==")
    for j, base in enumerate(sets):
        ok = [r for _, res in rows if res[j] and res[j]["strokes_multi"] == 0 and res[j]["foreign_pct"] == 0]
        multi = sum(r[j]["strokes_multi"] for _, r in rows if r[j])
        print("%-12s 合格 %d/%d  多色筆合計 %d  平均異色 %.2f%%" % (
            os.path.basename(base), len(ok), len(rows), multi,
            np.mean([r[j]["foreign_pct"] for _, r in rows if r[j]])))


if __name__ == "__main__":
    main()
