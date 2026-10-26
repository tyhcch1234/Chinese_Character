# -*- coding: utf-8 -*-
"""逐字修色：把已上線圖上「部件邊界畫錯」的像素改回正確部件色。

兩種規則，座標一律以墨跡外框歸一化（0–1，y 由上而下）：

  box(ch, frm, to, x0, x1, y0, y1)   矩形內由 frm 改為 to
  seed(ch, frm, to, x, y)            由該點所在的 frm 色連通塊整體改為 to

規則可重複執行（已是 to 色的像素不再匹配 frm），所以修完可以再跑一次確認 0 px。
每一條都對應 docs/method.md 記錄的一個具體錯切，不是泛用濾波。

  python3 _src/patch_colour.py --dry    # 只報告會改多少像素
  python3 _src/patch_colour.py          # 寫入 assets/ 並重建 index.html
"""
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

import build_parts4 as bp

COLORS = {"R": (192, 57, 43), "B": (43, 108, 176), "G": (47, 133, 90), "O": (183, 121, 31)}
ST = np.ones((3, 3), bool)

RULES = [
    # 國（結構示範）：囗 的上橫中段被「或」的矩形吃掉 → 整條上橫都是囗的紅。
    ("box", "國", "B", "R", (0.000, 1.000, 0.000, 0.085)),
    # 意（結構示範）：日 的右豎下半與底橫右端被判給「心」 → 回到「音」的藍。
    ("box", "意", "R", "B", (0.600, 0.850, 0.455, 0.665)),
    # 默：黑 的第三點被「犬」的矩形吃掉 → 回到黑的紅。
    ("box", "默", "B", "R", (0.240, 0.385, 0.720, 0.950)),
    # 默：第四點整點被判給「犬」（與犬的長撇不相連）→ 回到黑的紅。
    ("seed", "默", "B", "R", (0.450, 0.770)),
    ("seed", "默", "B", "R", (0.389, 0.855)),
    # 烈：刂 的豎鉤末端被「灬」的矩形吃掉 → 回到列的藍（灬第四點在 0.72 以下）。
    ("box", "烈", "R", "B", (0.780, 1.000, 0.600, 0.720)),
    # 喜：第二個口的底橫、右豎下段被「䒑一」的矩形吃掉 → 回到口的綠。
    ("box", "喜", "O", "G", (0.180, 0.860, 0.425, 0.525)),
    # 聲：耳 的頂橫左段與左豎上段被「殸」的矩形吃掉 → 回到耳的紅
    #     （x>0.12 以保留殸長撇壓在耳頂的筆尖）。
    ("box", "聲", "B", "R", (0.120, 0.400, 0.520, 0.580)),
    ("box", "聲", "B", "R", (0.220, 0.320, 0.560, 0.680)),
]


def folder_of(ch):
    return "chars" if any(r[0] == ch for r in bp.DATA) else "examples"


def apply_one(ch, dry):
    path = "assets/%s/%05d.png" % (folder_of(ch), ord(ch))
    if not os.path.exists(path):
        return 0, "missing"
    im = Image.open(path).convert("RGBA")
    a = np.array(im)
    alpha = a[:, :, 3] > 128
    ys, xs = np.nonzero(alpha)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    W, H = x1 - x0, y1 - y0
    total = 0
    for kind, _, frm, to, geo in [r for r in RULES if r[1] == ch]:
        rgb = a[:, :, :3].astype(int)
        m = (np.abs(rgb - np.array(COLORS[frm], int)).sum(2) < 10) & alpha
        if kind == "box":
            bx0, bx1, by0, by1 = geo
            box = np.zeros_like(m)
            box[int(y0 + by0 * H):int(y0 + by1 * H) + 1,
                int(x0 + bx0 * W):int(x0 + bx1 * W) + 1] = True
            sel = m & box
        else:
            sx, sy = int(x0 + geo[0] * W), int(y0 + geo[1] * H)
            lb, _ = ndimage.label(m, ST)
            k = lb[sy, sx]
            if k == 0:
                print("  ! %s seed (%.2f,%.2f) 不在 %s 色塊內" % (ch, geo[0], geo[1], frm))
                continue
            sel = lb == k
        n = int(sel.sum())
        a[sel, 0], a[sel, 1], a[sel, 2] = COLORS[to]
        total += n
        print("  %-2s %-4s %s>%s %5d px" % (ch, kind, frm, to, n))
    if total and not dry:
        Image.fromarray(a).save(path, optimize=True)
    return total, "ok"


def main():
    dry = "--dry" in sys.argv
    chars = sorted({r[1] for r in RULES})
    total = 0
    for ch in chars:
        n, st = apply_one(ch, dry)
        total += n
    print("\n合計 %d px%s" % (total, "（dry run，未寫入）" if dry else "，已寫入"))
    if not dry and total:
        os.system("%s _src/build_site.py" % sys.executable)
        print("index.html rebuilt")
    return 0


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.exit(main())
