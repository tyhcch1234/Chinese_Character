# -*- coding: utf-8 -*-
"""Decompose each character into colour-coded parts (blob aware) and render PNG parts."""
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from fontTools.ttLib import TTFont
from fontTools.pens.boundsPen import BoundsPen

FONT = "_font/GenRyuMin2TW-M.otf"
TT = TTFont(FONT)
GS = TT.getGlyphSet()
CMAP = TT.getBestCmap()
UPEM = TT["head"].unitsPerEm
RENDER = 1800      # supersampled raster size
OUT = 800          # final png size
BOX = 1000         # design box

COLORS = {"radical": (192, 57, 43), "part": (43, 108, 176),
          "part2": (47, 133, 90), "part3": (183, 121, 31)}

# name, structure, radical label, parts [(name, role, [x1,x2,y1,y2] fractions of ink box)]
DATA = [
    ("敏", "左右結構", "攴（攵）", [("每", "part", [0, .58, 0, 1]), ("攵", "radical", [.55, 1, 0, 1])]),
    ("終", "左右結構", "糸（糹）", [("糸", "radical", [0, .45, 0, 1]), ("冬", "part", [.42, 1, 0, 1])]),
    ("於", "左右結構", "方", [("方", "radical", [0, .44, 0, 1]), ("仒", "part", [.40, 1, 0, 1])]),
    ("操", "左右結構", "手（扌）", [("扌", "radical", [0, .30, 0, 1]), ("喿", "part", [.27, 1, 0, 1])]),
    ("場", "左右結構", "土", [("土", "radical", [0, .30, 0, 1]), ("昜", "part", [.26, 1, 0, 1])]),
    ("籃", "上下結構", "竹（⺮）", [("竹", "radical", [0, 1, 0, .22]), ("監", "part", [0, 1, .18, 1])]),
    ("球", "左右結構", "玉（王）", [("王", "radical", [0, .30, 0, 1]), ("求", "part", [.27, 1, 0, 1])]),
    ("架", "上下結構", "木", [("加", "part", [0, 1, 0, .47]), ("木", "radical", [0, 1, .43, 1])]),
    ("找", "左右結構", "手（扌）", [("扌", "radical", [0, .28, 0, 1]), ("戈", "part", [.24, 1, 0, 1])]),
    ("錶", "左右結構", "金", [("金", "radical", [0, .42, 0, 1]), ("表", "part", [.39, 1, 0, 1])]),
    ("喜", "上中下結構", "口", [("士", "part", [0, 1, 0, .30]), ("口", "part2", [0, 1, .28, .58]),
                                ("䒑一", "part3", [0, 1, .55, .72]), ("口", "radical", [0, 1, .70, 1])]),
    ("眉", "兩面包圍（左上）", "目", [("眉", "part", [0, 1, 0, 1]), ("目", "radical", [.33, 1, .27, 1])]),
    ("梢", "左右結構", "木", [("木", "radical", [0, .42, 0, 1]), ("肖", "part", [.38, 1, 0, 1])]),
    ("拿", "上下結構", "手", [("合", "part", [0, 1, 0, .54]), ("手", "radical", [0, 1, .50, 1])]),
    ("幫", "上下結構", "巾", [("封", "part", [0, 1, 0, .33]), ("帛（白＋巾）", "radical", [0, 1, .31, 1])]),
    ("助", "左右結構", "力", [("且", "part", [0, .52, 0, 1]), ("力", "radical", [.48, 1, 0, 1])]),
    ("帶", "上下結構", "巾", [("卅冖", "part", [0, 1, 0, .46]), ("巾", "radical", [0, 1, .44, 1])]),
    ("焦", "上下結構", "火（灬）", [("隹", "part", [0, 1, 0, .66]), ("灬", "radical", [0, 1, .62, 1])]),
    ("急", "上下結構", "心", [("⺈彐", "part", [0, 1, 0, .66]), ("心", "radical", [0, 1, .64, 1])]),
    ("慰", "上下結構", "心", [("尉", "part", [0, 1, 0, .76]), ("心", "radical", [0, 1, .72, 1])]),
    ("慚", "左右結構", "心（忄）", [("忄", "radical", [0, .16, 0, 1]), ("斬", "part", [.13, 1, 0, 1])]),
    ("愧", "左右結構", "心（忄）", [("忄", "radical", [0, .16, 0, 1]), ("鬼", "part", [.13, 1, 0, 1])]),
    ("面", "獨體字", "面", [("面", "radical", [0, 1, 0, 1])]),
    ("耳", "獨體字", "耳", [("耳", "radical", [0, 1, 0, 1])]),
    ("赤", "上下結構", "赤", [("土", "part", [0, 1, 0, .46]), ("灬", "radical", [0, 1, .42, 1])]),
    ("温", "左右結構", "水（氵）", [("氵", "radical", [0, .22, 0, 1]), ("昷", "part", [.18, 1, 0, 1])]),
    ("暖", "左右結構", "日", [("日", "radical", [0, .22, 0, 1]), ("爰", "part", [.18, 1, 0, 1])]),
    ("興", "上下結構", "臼", [("舁同", "radical", [0, 1, 0, .82]), ("八", "part", [0, 1, .80, 1])]),
    ("采", "上下結構", "爪（爫）", [("爫", "radical", [0, 1, 0, .33]), ("木", "part", [0, 1, .29, 1])]),
    ("烈", "上下結構", "火（灬）", [("列", "part", [0, 1, 0, .66]), ("灬", "radical", [0, 1, .62, 1])]),
    ("跌", "左右結構", "足", [("足", "radical", [0, .46, 0, 1]), ("失", "part", [.42, 1, 0, 1])]),
    ("倒", "左右結構", "人（亻）", [("亻", "radical", [0, .18, 0, 1]), ("到", "part", [.15, 1, 0, 1])]),
    ("默", "左右結構", "黑", [("黑", "radical", [0, .54, 0, 1]), ("犬", "part", [.50, 1, 0, 1])]),
    ("作", "左右結構", "人（亻）", [("亻", "radical", [0, .18, 0, 1]), ("乍", "part", [.15, 1, 0, 1])]),
    ("聲", "上下結構", "耳", [("殸", "part", [0, 1, 0, .56]), ("耳", "radical", [0, 1, .52, 1])]),
    ("貪", "上下結構", "貝", [("今", "part", [0, 1, 0, .36]), ("貝", "radical", [0, 1, .32, 1])]),
    ("玩", "左右結構", "玉（王）", [("王", "radical", [0, .30, 0, 1]), ("元", "part", [.27, 1, 0, 1])]),
    ("垂", "獨體字", "土", [("垂", "radical", [0, 1, 0, 1])]),
    ("喪", "獨體字", "口", [("喪", "radical", [0, 1, 0, 1])]),
    ("氣", "兩面包圍（右上）", "气", [("气", "radical", [0, 1, 0, 1]), ("米", "part", [.16, .88, .30, 1])]),
]


EXAMPLES = [
    ("明", "左右結構", "日", [("日", "radical", [0, .44, 0, 1]), ("月", "part", [.42, 1, 0, 1])]),
    ("湖", "左中右結構", "水（氵）", [("氵", "radical", [0, .24, 0, 1]), ("古", "part", [.22, .66, 0, 1]),
                                   ("月", "part2", [.64, 1, 0, 1])]),
    ("思", "上下結構", "心", [("田", "part", [0, 1, 0, .48]), ("心", "radical", [0, 1, .45, 1])]),
    ("意", "上中下結構", "心", [("音", "part", [0, 1, 0, .55]), ("心", "radical", [0, 1, .52, 1])]),
    ("品", "品字形結構", "口", [("口", "radical", [.20, .80, 0, .44]), ("口", "part", [0, .52, .42, 1]),
                                ("口", "part2", [.48, 1, .42, 1])]),
    ("國", "全包圍結構", "囗", [("囗", "radical", [0, 1, 0, 1]), ("或", "part", [.14, .88, .14, .88])]),
    ("閃", "三面包圍結構", "門", [("門", "radical", [0, 1, 0, 1]), ("人", "part", [.30, .78, .22, .92])]),
    ("病", "兩面包圍結構", "疒", [("疒", "radical", [0, 1, 0, 1]), ("丙", "part", [.36, .95, .32, .96])]),
    ("永", "獨體字", "水", [("永", "radical", [0, 1, 0, 1])]),
]

def ink_image(ch):
    n = RENDER
    fs = int(n * 1.35)
    font = ImageFont.truetype(FONT, fs)
    img = Image.new("L", (n * 2, n * 2), 0)
    d = ImageDraw.Draw(img)
    d.text((n // 2, n + n // 3), ch, font=font, fill=255, anchor="ls")
    bb = img.getbbox()
    assert bb, ch
    return img.crop(bb)

def colorize(ch, parts):
    img = ink_image(ch)
    arr = np.array(img)
    h, w = arr.shape
    mask = arr > 110
    lab, nlab = ndimage.label(mask, structure=np.ones((3, 3), bool))

    # region id per pixel (later parts override earlier ones)
    region = np.zeros((h, w), np.int16)
    for i, (_, _, (x1, x2, y1, y2)) in enumerate(parts, start=1):
        region[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)] = i

    assign = np.zeros((h, w), np.uint8)
    counts = ndimage.sum_labels(np.ones_like(arr), lab, index=np.arange(1, nlab + 1))
    for b in range(1, nlab + 1):
        ys, xs = np.where(lab == b)
        if len(ys) < 12:                       # dust / stray dots: follow pixel region
            assign[ys, xs] = region[ys, xs]
            continue
        vals = region[ys, xs]
        uniq, cnt = np.unique(vals, return_counts=True)
        order = np.argsort(-cnt)
        dominant = uniq[order[0]]
        minor = cnt[order[1]] / cnt.sum() if len(order) > 1 else 0.0
        if minor >= 0.25:
            assign[ys, xs] = vals             # spans a boundary -> split per pixel
        else:
            assign[ys, xs] = dominant

    rgba = np.zeros((h, w, 4), np.uint8)
    palette = [None] + [COLORS[p[1]] for p in parts]
    for i in range(1, len(parts) + 1):
        m = (assign == i) & mask
        rgba[m, 0], rgba[m, 1], rgba[m, 2] = palette[i]
        rgba[m, 3] = 255
    # soften edges from the anti-aliased source
    alpha = np.clip(arr.astype(np.float32) * 1.15, 0, 255)
    rgba[..., 3] = np.maximum(rgba[..., 3], (alpha * (rgba[..., 3] > 0)).astype(np.uint8))
    out = Image.fromarray(rgba, "RGBA")

    # fit into BOX
    s = min(860 / h, 900 / w)
    out = out.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    canvas = Image.new("RGBA", (BOX, BOX), (0, 0, 0, 0))
    canvas.paste(out, ((BOX - out.width) // 2, (BOX - out.height) // 2), out)
    return canvas.resize((OUT, OUT), Image.LANCZOS)

os.makedirs("assets/chars", exist_ok=True)
os.makedirs("assets/examples", exist_ok=True)

def build(items, folder, outfile):
    records = []
    for ch, struct, radical, parts in items:
        im = colorize(ch, parts)
        q = im.quantize(colors=32, method=Image.FASTOCTREE)
        q.save(f"{folder}/{ord(ch):05d}.png", optimize=True)
        records.append({"char": ch, "structure": struct, "radical": radical,
                        "parts": [{"name": p[0], "role": p[1]} for p in parts]})
        print("ok", ch)
    json.dump(records, open(outfile, "w"), ensure_ascii=False, indent=1)
    return records

records = build(DATA, "assets/chars", "chars.json")
examples = build(EXAMPLES, "assets/examples", "examples.json")
