# -*- coding: utf-8 -*-
"""逐字品質複檢：不看 part_of 當裁判，直接對「已發佈 PNG」量六件事。

  python3 _src/qc.py                # 量 40 字 + 9 張結構示範，印表
  python3 _src/qc.py --sheet DIR    # 另出 4 張放大對照圖（每張 10 字）
  python3 _src/qc.py 湖 籃           # 只量指定字

指標（全部以字型墨跡 + 像素顏色為準）：
  uncol   墨跡內未上色像素（黑洞）
  spill   墨跡外上色像素（位移／變粗）
  seam    opening(3×3) 損失像素（髮絲接縫）
  frag    面積 < 該色 1% 的孤島色塊數（楔形／碎片）
  mix     主要顏色佔有率 < 99.5% 的筆畫數（一筆多色）
  miss    宣告的部件顏色完全沒出現
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

import build_parts4 as bp
import parts_auto as pa
from pngmap import lab_on_mask

ST = ndimage.generate_binary_structure(2, 1)


def labels_on_font(ch, parts, path):
    data, mask, dist = pa.stroke_field(ch)
    lab, _ = lab_on_mask(path, parts, mask)
    return data, mask.astype(bool), dist, lab


def stroke_ink(dist, mask, i):
    """第 i 筆的墨跡（最近中心線）——只用作『這筆有幾種色』的量測區。"""
    return (np.argmin(dist, axis=0) == i) & mask


def measure(ch, parts, path):
    data, mask, dist, lab = labels_on_font(ch, parts, path)
    k = len(parts)
    ink_px = int(mask.sum())
    uncol = int((mask & (lab == 0)).sum())
    spill = int(((lab > 0) & ~mask).sum())

    seam = 0
    frag = []
    frag_px = 0
    for idx, p in enumerate(parts, start=1):
        m = lab == idx
        if not m.any():
            continue
        seam += int((m & ~ndimage.binary_opening(m, ST)).sum())
        nl_lab, nl = ndimage.label(m, ST)
        if nl > 1:
            areas = ndimage.sum(m, nl_lab, range(1, nl + 1))
            big = areas.max()
            for j, a in enumerate(areas, start=1):
                if a < 0.01 * big:
                    ys, xs = np.nonzero(nl_lab == j)
                    frag.append((p[0], int(a), float(xs.mean() / lab.shape[1]),
                                 float(ys.mean() / lab.shape[0])))
                    frag_px += int(a)

    # 一筆多色：用鎖定身分（若有）決定這筆屬於哪件；只看墨跡主色佔有率
    lock = pa.identity_lock(ch)
    mixed = 0
    wrong = []
    for i in range(dist.shape[0]):
        sel = stroke_ink(dist, mask, i) & (lab > 0)
        n = int(sel.sum())
        if n < 30:
            continue
        vals, cnt = np.unique(lab[sel], return_counts=True)
        j = int(np.argmax(cnt))
        share = float(cnt[j] / n)
        if share < 0.995:
            mixed += 1
            wrong.append((i + 1, int(vals[j]), round(share, 4)))
        if lock and i < len(lock) and vals[j] != lock[i] and cnt[j] / n > 0.5:
            wrong.append((i + 1, int(vals[j]), "lock=%d" % lock[i]))

    present = set(int(v) for v in np.unique(lab) if v)
    miss = [p[0] for idx, p in enumerate(parts, start=1) if idx not in present]
    return dict(char=ch, ink=ink_px, uncol=uncol, spill=spill, seam=seam,
                frag=len(frag), frag_px=frag_px, frag_at=frag[:4],
                strokes=dist.shape[0], mix=mixed, miss=miss, odd=wrong[:4])


def collect(only=None):
    rows = []
    for rec in bp.DATA + bp.EXAMPLES:
        ch, parts = rec[0], rec[3]
        if only and ch not in only:
            continue
        folder = "chars" if rec in bp.DATA else "examples"
        path = "assets/%s/%05d.png" % (folder, ord(ch))
        if not os.path.exists(path):
            continue
        r = measure(ch, parts, path)
        r["kind"] = folder
        rows.append(r)
    return rows


def sheet(rows, dst, per=10, cell=460, cols=2):
    os.makedirs(dst, exist_ok=True)
    font = ImageFont.load_default()
    for page in range(0, len(rows), per):
        chunk = rows[page:page + per]
        rowsn = (len(chunk) + cols - 1) // cols
        W, H = cols * cell, rowsn * (cell + 34)
        canvas = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(canvas)
        for i, r in enumerate(chunk):
            ch = r["char"]
            folder = "chars" if r["kind"] == "chars" else "examples"
            im = Image.open("assets/%s/%05d.png" % (folder, ord(ch))).convert("RGBA")
            ys, xs = np.nonzero(np.array(im)[:, :, 3] > 8)
            crop = im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
            s = (cell - 30) / max(crop.size)
            crop = crop.resize((max(1, int(crop.width * s)), max(1, int(crop.height * s))),
                               Image.LANCZOS)
            cx, cy = (i % cols) * cell, (i // cols) * (cell + 34)
            canvas.paste(crop, (cx + (cell - crop.width) // 2,
                                cy + (cell - crop.height) // 2), crop)
            flag = "!" if (r["uncol"] or r["frag"] or r["mix"] or r["miss"]) else ""
            d.text((cx + 8, cy + cell + 6),
                   "%s  uncol=%d frag=%d mix=%d %s" % (ch, r["uncol"], r["frag"], r["mix"], flag),
                   fill="black", font=font)
            d.rectangle([cx, cy, cx + cell - 1, cy + cell - 1], outline=(225, 225, 225))
        out = os.path.join(dst, "qc_%02d.png" % (page // per + 1))
        canvas.save(out)
        print("sheet", out, canvas.size, [r["char"] for r in chunk])


def main():
    args = [a for a in sys.argv[1:]]
    dst = None
    if "--sheet" in args:
        i = args.index("--sheet")
        dst = args[i + 1]
        del args[i:i + 2]
    only = set("".join(args))
    rows = collect(only)
    hdr = "%-3s %-8s %5s %5s %5s %4s %6s %4s %4s %s"
    print(hdr % ("字", "類", "uncol", "spill", "seam", "frag", "fragpx", "mix", "筆", "miss/odd"))
    for r in rows:
        print(hdr % (r["char"], "char" if r["kind"] == "chars" else "example",
                     r["uncol"], r["spill"], r["seam"], r["frag"], r["frag_px"],
                     r["mix"], r["strokes"], (r["miss"] or "") if not r["odd"] else r["odd"]))
    bad = [r for r in rows if r["uncol"] or r["frag"] or r["mix"] or r["miss"]]
    print("\n合計 %d 張，需複查 %d：%s" % (len(rows), len(bad), "".join(r["char"] for r in bad)))
    if dst:
        sheet(rows, dst)
    return 0


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.exit(main())
