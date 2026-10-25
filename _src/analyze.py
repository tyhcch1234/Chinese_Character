# -*- coding: utf-8 -*-
"""Compute per-character part boundaries (as fractions of the ink bounding box)
by snapping rough guesses to local minima of the ink profile."""
import json, sys
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
from fontTools.pens.boundsPen import BoundsPen

FONT = "_font/GenRyuMin2TW-M.otf"
TT = TTFont(FONT)
GS = TT.getGlyphSet()
CMAP = TT.getBestCmap()
UPEM = TT["head"].unitsPerEm
PILFONT = ImageFont.truetype(FONT, 400)
S = 400

def bounds(ch):
    gname = CMAP.get(ord(ch))
    if not gname:
        raise SystemExit("missing glyph: " + ch)
    bp = BoundsPen(GS)
    GS[gname].draw(bp)
    return bp.bounds  # xMin,yMin,xMax,yMax in font units

def ink(ch):
    img = Image.new("L", (S * 2, S * 2), 0)
    d = ImageDraw.Draw(img)
    d.text((S // 2, S // 2), ch, font=PILFONT, fill=255, anchor="ls")
    bbox = img.getbbox()
    if not bbox:
        raise SystemExit("empty render: " + ch)
    img = img.crop(bbox)
    return img

def profiles(img):
    w, h = img.size
    px = img.load()
    cols = [0] * w
    rows = [0] * h
    for y in range(h):
        for x in range(w):
            if px[x, y] > 40:
                cols[x] += 1
                rows[y] += 1
    mw = max(cols) or 1
    mh = max(rows) or 1
    return [c / mw for c in cols], [r / mh for r in rows]

def snap(prof, target, win=0.09):
    """Return the fraction (0..1) with least ink within target +/- win."""
    n = len(prof)
    lo = max(1, int((target - win) * n))
    hi = min(n - 2, int((target + win) * n))
    if hi <= lo:
        lo, hi = max(1, int(target * n) - 3), min(n - 2, int(target * n) + 3)
    best, bi = 1e9, int(target * n)
    for i in range(lo, hi + 1):
        v = prof[i]
        if v < best - 1e-9:
            best, bi = v, i
    return round(bi / (n - 1), 4), round(best, 4)

SPEC = [
    # (char, structure, axis, rough boundaries, part names/roles)
    ("敏", "左右", "x", [0.50], [("每", "part"), ("攵", "radical")]),
    ("終", "左右", "x", [0.46], [("糸", "radical"), ("冬", "part")]),
    ("於", "左右", "x", [0.44], [("方", "radical"), ("𠂉丶", "part")]),
    ("操", "左右", "x", [0.26], [("扌", "radical"), ("喿", "part")]),
    ("場", "左右", "x", [0.28], [("土", "radical"), ("昜", "part")]),
    ("籃", "上下", "y", [0.27], [("竹", "radical"), ("監", "part")]),
    ("球", "左右", "x", [0.30], [("王", "radical"), ("求", "part")]),
    ("架", "上下", "y", [0.46], [("加", "part"), ("木", "radical")]),
    ("找", "左右", "x", [0.26], [("扌", "radical"), ("戈", "part")]),
    ("錶", "左右", "x", [0.40], [("金", "radical"), ("表", "part")]),
    ("喜", "上中下", "y", [0.30, 0.52], [("士", "part"), ("口", "part2"), ("䒑一囗", "radical")]),
    ("眉", "半包圍", None, None, [("眉", "part"), ("目", "radical")]),
    ("梢", "左右", "x", [0.42], [("木", "radical"), ("肖", "part")]),
    ("拿", "上下", "y", [0.50], [("合", "part"), ("手", "radical")]),
    ("幫", "上下", "y", [0.42, 0.72], [("封", "part"), ("白", "part2"), ("巾", "radical")]),
    ("助", "左右", "x", [0.54], [("且", "part"), ("力", "radical")]),
    ("帶", "上中下", "y", [0.30, 0.44], [("卅", "part"), ("冖", "part2"), ("巾", "radical")]),
    ("焦", "上下", "y", [0.72], [("隹", "part"), ("灬", "radical")]),
    ("急", "上中下", "y", [0.22, 0.52], [("⺈", "part"), ("彐", "part2"), ("心", "radical")]),
    ("慰", "上下", "y", [0.72], [("尉", "part"), ("心", "radical")]),
    ("慚", "左右", "x", [0.18], [("忄", "radical"), ("斬", "part")]),
    ("愧", "左右", "x", [0.18], [("忄", "radical"), ("鬼", "part")]),
    ("面", "獨體", None, None, [("面", "radical")]),
    ("耳", "獨體", None, None, [("耳", "radical")]),
    ("赤", "上下", "y", [0.44], [("土", "part"), ("灬", "radical")]),
    ("温", "左右", "x", [0.20], [("氵", "radical"), ("昷", "part")]),
    ("暖", "左右", "x", [0.20], [("日", "radical"), ("爰", "part")]),
    ("興", "上下", "y", [0.42, 0.86], [("舁", "radical"), ("同", "part"), ("八", "part2")]),
    ("采", "上下", "y", [0.34], [("爫", "radical"), ("木", "part")]),
    ("烈", "上下", "y", [0.70], [("列", "part"), ("灬", "radical")]),
    ("跌", "左右", "x", [0.44], [("足", "radical"), ("失", "part")]),
    ("倒", "左右", "x", [0.18], [("亻", "radical"), ("到", "part")]),
    ("默", "左右", "x", [0.56], [("黑", "radical"), ("犬", "part")]),
    ("作", "左右", "x", [0.20], [("亻", "radical"), ("乍", "part")]),
    ("聲", "上下", "y", [0.62], [("殸", "part"), ("耳", "radical")]),
    ("貪", "上下", "y", [0.40], [("今", "part"), ("貝", "radical")]),
    ("玩", "左右", "x", [0.30], [("王", "radical"), ("元", "part")]),
    ("垂", "獨體", None, None, [("垂", "radical")]),
    ("喪", "獨體", None, None, [("喪", "radical")]),
    ("氣", "半包圍", None, None, [("气", "radical"), ("米", "part")]),
]

out = {}
for ch, struct, axis, bounds_rough, parts in SPEC:
    img = ink(ch)
    cols, rows = profiles(img)
    bb = bounds(ch)
    entry = {"char": ch, "structure": struct, "parts": []}
    if axis == "x":
        xs = [snap(cols, t)[0] for t in bounds_rough]
        resid = [snap(cols, t)[1] for t in bounds_rough]
        edges = [0.0] + xs + [1.0]
        for i, (name, role) in enumerate(parts):
            entry["parts"].append({"name": name, "role": role,
                                   "rect": [edges[i], edges[i + 1], 0.0, 1.0]})
        entry["_resid"] = resid
    elif axis == "y":
        ys = [snap(rows, t)[0] for t in bounds_rough]
        resid = [snap(rows, t)[1] for t in bounds_rough]
        edges = [0.0] + ys + [1.0]
        for i, (name, role) in enumerate(parts):
            entry["parts"].append({"name": name, "role": role,
                                   "rect": [0.0, 1.0, edges[i], edges[i + 1]]})
        entry["_resid"] = resid
    elif struct == "半包圍" and ch == "眉":
        # cap = everything; 目 = inner box
        x0, _ = snap(cols, 0.30, 0.10)
        x1, _ = snap(cols, 0.95, 0.06)
        y0, ry = snap(rows, 0.36, 0.10)
        y1, _ = snap(rows, 0.98, 0.06)
        entry["parts"] = [
            {"name": "眉", "role": "part", "rect": [0.0, 1.0, 0.0, 1.0]},
            {"name": "目", "role": "radical", "rect": [max(x0, 0.25), 1.0, y0, 1.0]},
        ]
        entry["_resid"] = [ry]
    elif struct == "半包圍" and ch == "氣":
        y0, ry = snap(rows, 0.30, 0.10)
        entry["parts"] = [
            {"name": "气", "role": "radical", "rect": [0.0, 1.0, 0.0, 1.0]},
            {"name": "米", "role": "part", "rect": [0.18, 0.88, y0, 0.97]},
        ]
        entry["_resid"] = [ry]
    else:
        entry["parts"] = [{"name": parts[0][0], "role": parts[0][1], "rect": [0, 1, 0, 1]}]
    entry["bbox"] = [round(v, 1) for v in bb]
    out[ch] = entry
    print(ch, struct, [p["rect"] for p in entry["parts"]], "resid", entry.get("_resid"))

json.dump(out, open("_font/parts_raw.json", "w"), ensure_ascii=False, indent=1)
print("saved")
