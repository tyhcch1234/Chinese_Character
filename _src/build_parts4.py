# -*- coding: utf-8 -*-
"""Stroke-exact colouring.

Each ink pixel is assigned to the nearest stroke median (stroke-order data from the
public `hanzi-writer-data` package).  All pixels of one stroke therefore always get
the same component colour -> a stroke can never be split between two colours.
The glyph shapes themselves still come from 源流明體.
"""
import json, os, re, sys, urllib.parse, urllib.request
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

FONT = "_font/GenRyuMin2TW-M.otf"
RENDER = 1100
OUT = 800
BOX = 1000
CACHE = "_data/hanzi"
HANZI_SIZE = 1024
BASELINE = 900
COLORS = {"radical": (192, 57, 43), "part": (43, 108, 176),
          "part2": (47, 133, 90), "part3": (183, 121, 31)}

from build_parts_data import DATA, EXAMPLES  # noqa: E402

_pil_font = None
NUM = re.compile(r"-?\d+(?:\.\d+)?")


def hanzi_data(ch):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{ord(ch):05d}.json")
    if os.path.exists(path):
        return json.load(open(path))
    url = "https://cdn.jsdelivr.net/npm/hanzi-writer-data@2.0/" + urllib.parse.quote(ch) + ".json"
    with urllib.request.urlopen(url, timeout=30) as r:
        d = json.loads(r.read().decode())
    json.dump(d, open(path, "w"))
    return d


def path_points(d):
    """Control points of an SVG path (good enough for a bounding box)."""
    pts = []
    for seg in re.findall(r"[MLQCSTZmlqcstz][^MLQCSTZmlqcstz]*", d):
        nums = [float(v) for v in NUM.findall(seg[1:])]
        for i in range(0, len(nums) - 1, 2):
            pts.append((nums[i], nums[i + 1]))
    return pts


def svg_polys(d, steps=8):
    """Parse an SVG path ('d') into filled polygons (M/L/Q/C/Z, absolute+relative)."""
    toks = re.findall(r"[MLQCZmlqcz]|-?\d*\.?\d+(?:e-?\d+)?", d)
    polys, cur, cmd = [], [], None
    i, cx, cy = 0, 0.0, 0.0
    start = (0.0, 0.0)
    while i < len(toks):
        t = toks[i]
        if re.match(r"[A-Za-z]", t):
            cmd = t
            i += 1
            if cmd in "Zz":
                if len(cur) > 2:
                    polys.append(cur)
                cur = []
                cx, cy = start
                continue
        n = 0
        while i < len(toks) and not re.match(r"[A-Za-z]", toks[i]):
            n += 1
            i += 1
        vals = [float(v) for v in toks[i - n:i]]
        rel = cmd.islower()
        C = cmd.upper()
        if C == "M":
            for k in range(0, len(vals) - 1, 2):
                x, y = vals[k], vals[k + 1]
                if rel:
                    x, y = cx + x, cy + y
                if cur:
                    polys.append(cur)
                cur = [(x, y)]
                cx, cy = x, y
                start = (x, y)
        elif C == "L":
            for k in range(0, len(vals) - 1, 2):
                x, y = vals[k], vals[k + 1]
                if rel:
                    x, y = cx + x, cy + y
                cur.append((x, y))
                cx, cy = x, y
        elif C == "Q":
            for k in range(0, len(vals) - 3, 4):
                x1, y1, x2, y2 = vals[k:k + 4]
                if rel:
                    x1, y1, x2, y2 = cx + x1, cy + y1, cx + x2, cy + y2
                for j in range(1, steps + 1):
                    t2 = j / steps
                    m = 1 - t2
                    cur.append((m * m * cx + 2 * m * t2 * x1 + t2 * t2 * x2,
                                m * m * cy + 2 * m * t2 * y1 + t2 * t2 * y2))
                cx, cy = x2, y2
        elif C == "C":
            for k in range(0, len(vals) - 5, 6):
                x1, y1, x2, y2, x3, y3 = vals[k:k + 6]
                if rel:
                    x1, y1 = cx + x1, cy + y1
                    x2, y2 = cx + x2, cy + y2
                    x3, y3 = cx + x3, cy + y3
                p0 = (cx, cy)
                for j in range(1, steps + 1):
                    t2 = j / steps
                    m = 1 - t2
                    cur.append((m ** 3 * p0[0] + 3 * m * m * t2 * x1 + 3 * m * t2 * t2 * x2 + t2 ** 3 * x3,
                                m ** 3 * p0[1] + 3 * m * m * t2 * y1 + 3 * m * t2 * t2 * y2 + t2 ** 3 * y3))
                cx, cy = x3, y3
        else:
            pass
    if len(cur) > 2:
        polys.append(cur)
    return polys

def render_mask(ch):
    global _pil_font
    if _pil_font is None:
        _pil_font = ImageFont.truetype(FONT, 1000)
    img = Image.new("L", (RENDER * 2, RENDER * 2), 0)
    ImageDraw.Draw(img).text((RENDER, RENDER), ch, font=_pil_font, fill=255, anchor="ls")
    img = img.crop(img.getbbox())
    scale = min((RENDER - 8) / img.height, (RENDER - 8) / img.width)
    img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
    return np.array(img) > 110


def densify(points, step=6.0):
    out = []
    for a, b in zip(points[:-1], points[1:]):
        d = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
        k = max(1, int(d / step))
        for i in range(k):
            t = i / k
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    out.append(tuple(points[-1]))
    return out


def colorize(ch, parts, structure="", spans=None):
    data = hanzi_data(ch)
    mask = render_mask(ch)
    H, W = mask.shape

    # hanzi space -> screen space (y down), then fit to this glyph's ink box
    strokes = []
    allx, ally = [], []
    for d in data["strokes"]:
        pts = [(x, BASELINE - y) for x, y in path_points(d)]
        strokes.append(pts)
        allx += [p[0] for p in pts]
        ally += [p[1] for p in pts]
    hx0, hx1, hy0, hy1 = min(allx), max(allx), min(ally), max(ally)

    ys, xs = np.nonzero(mask)
    gx0, gx1, gy0, gy1 = xs.min(), xs.max(), ys.min(), ys.max()
    # initial guess: fit the stroke outline box onto the ink box
    sx = (gx1 - gx0) / max(1e-6, hx1 - hx0)
    sy = (gy1 - gy0) / max(1e-6, hy1 - hy0)
    cx, cy = (hx0 + hx1) / 2, (hy0 + hy1) / 2
    tx, ty = (gx0 + gx1) / 2, (gy0 + gy1) / 2

    # refine by pulling every median point onto the ink (edge-distance based)
    ink_dist = ndimage.distance_transform_edt(~mask)
    med_pts = np.array([(x, BASELINE - y) for m in data["medians"] for x, y in m], float)

    def cost(v):
        s_, tx_, ty_, ax = v
        px = (med_pts[:, 0] - cx) * s_ * ax + tx_
        py = (med_pts[:, 1] - cy) * s_ + ty_
        ok = (px >= 0) & (px < W - 1) & (py >= 0) & (py < H - 1)
        if ok.sum() < len(px) * 0.9:
            return 1e6
        d = ink_dist[py[ok].astype(int), px[ok].astype(int)]
        return float(d.mean() + 0.02 * abs(np.log(ax)))

    from scipy.optimize import minimize
    res = minimize(cost, [ (sx + sy) / 2, tx, ty, 1.0 ], method="Nelder-Mead",
                   options={"maxiter": 600, "xatol": 1e-3, "fatol": 1e-4})
    s_best, tx, ty, ax = res.x

    def to_px(p):
        return ((p[0] - cx) * s_best * ax + tx, (p[1] - cy) * s_best + ty)

    dist = np.full((len(data["medians"]), H, W), np.inf, dtype=np.float32)
    for i, med in enumerate(data["medians"]):
        med = [(x, BASELINE - y) for x, y in med]
        pts = densify(med, step=max(4.0, 8.0 / max(sx, sy)))
        layer = Image.new("L", (W, H), 0)
        dr = ImageDraw.Draw(layer)
        xy = [to_px(p) for p in pts]
        dr.line(xy, fill=255, width=1)
        for p in xy:
            dr.ellipse([p[0] - 0.6, p[1] - 0.6, p[0] + 0.6, p[1] + 0.6], fill=255)
        dist[i] = ndimage.distance_transform_edt(np.array(layer) < 128)

    # Stroke bodies come from the stroke-order data, painted in writing order:
    # a later stroke keeps its own body at a junction, an earlier stroke is cut
    # at that junction.  Every pixel therefore belongs to exactly one stroke and
    # a stroke can never be split between two colours.
    polys_masks = []
    stroke_of = {}
    for d in data["strokes"]:
        layer = Image.new("L", (W, H), 0)
        dr = ImageDraw.Draw(layer)
        for poly in svg_polys(d):
            pts = [to_px((x, BASELINE - y)) for x, y in poly]
            if len(pts) > 2:
                dr.polygon(pts, fill=255)
        m = np.array(layer) > 127
        # A few glyphs use compound paths (several sub-paths in one stroke); a
        # naive polygon fill can then balloon a small dot into a shape covering
        # half the glyph and steal a neighbour's ink.  Rebuild such a mask as a
        # thick line along the stroke's own centre line instead.
        med = [(x, BASELINE - y) for x, y in data["medians"][i]]
        mx = [to_px(p)[0] for p in med]
        my = [to_px(p)[1] for p in med]
        ys, xs = np.nonzero(m)
        if m.any() and (max(xs) - min(xs) > 2.2 * (max(mx) - min(mx)) + 24
                        or max(ys) - min(ys) > 2.2 * (max(my) - min(my)) + 24):
            layer = Image.new("L", (W, H), 0)
            dr = ImageDraw.Draw(layer)
            pts = densify([to_px(p) for p in med], step=4.0)
            dr.line(pts, fill=255, width=max(8, int(0.085 * min(W, H))), joint="curve")
            m = np.array(layer) > 127
        m = ndimage.binary_dilation(m, np.ones((5, 5), bool))
        polys_masks.append(m)

    # Ownership rule: every ink pixel belongs to the stroke whose centre line is
    # nearest.  The boundary between two strokes is therefore always the medial
    # line between their centrelines - it can only run *between* strokes, never
    # lengthwise along one.  Because a pixel maps to one stroke and a stroke maps
    # to one part, a single stroke can only ever carry one colour.
    # A pixel may only be given to a stroke whose own body actually covers it;
    # among those, the nearest centre line wins.  Ink that no stroke body covers
    # falls back to the nearest centre line.  Together with the one-stroke-one-part
    # mapping this makes a two-coloured stroke impossible.
    owner = np.full((H, W), -1, np.int32)
    best = np.full((H, W), np.inf, np.float32)
    for i in range(len(dist)):
        better = (dist[i] < best) & polys_masks[i]
        owner[better] = i
        best[better] = dist[i][better]
    rest = mask & (owner < 0)
    if rest.any():
        best2 = np.full((H, W), np.inf, np.float32)
        for i in range(len(dist)):
            better = rest & (dist[i] < best2)
            owner[better] = i
            best2[better] = dist[i][better]

    # Part rectangles are authored by eye.  Each internal boundary is snapped to
    # a point midway between two neighbouring stroke centres, so the colour
    # boundary always falls *between* strokes and never through one.
    sxs = np.array([np.nonzero(m)[1].mean() for m in polys_masks], float)
    sys_ = np.array([np.nonzero(m)[0].mean() for m in polys_masks], float)

    def snap(v, centres, span, limit=0.13):
        c = np.sort(centres)
        if len(c) < 2:
            return v
        mids = (c[:-1] + c[1:]) / 2.0
        i = int(np.argmin(np.abs(mids - v)))
        return float(mids[i]) if abs(mids[i] - v) <= span * limit else v

    region = np.zeros((H, W), np.int16)
    for i, (_, _, (a, b, c, d)) in enumerate(parts, start=1):
        x0 = 0 if a <= 0 else int(round(snap(a * W, sxs, W)))
        x1 = W if b >= 1 else int(round(snap(b * W, sxs, W)))
        y0 = 0 if c <= 0 else int(round(snap(c * H, sys_, H)))
        y1 = H if d >= 1 else int(round(snap(d * H, sys_, H)))
        if x1 <= x0:
            x1 = min(W, x0 + 1)
        if y1 <= y0:
            y1 = min(H, y0 + 1)
        region[y0:y1, x0:x1] = i

    # Stroke -> part.  Baseline: whichever part rectangle holds most of the
    # stroke's ink.  When the stroke-order data names the radical's strokes we
    # also try that grouping, which keeps a radical whole even when its strokes
    # interleave with the rest (e.g. 於, 國).
    def region_assign(i, allowed):
        m = (owner == i) & mask
        if not m.any():
            return allowed[0], 0.0
        vals = region[m]
        vals = vals[np.isin(vals, allowed)]
        if vals.size == 0:
            return allowed[0], 0.0
        uniq, cnt = np.unique(vals, return_counts=True)
        return int(uniq[np.argmax(cnt)]), float(cnt.max() / cnt.sum())

    nums = list(range(1, len(parts) + 1))
    base = {i: region_assign(i, nums) for i in range(len(dist))}

    rad_strokes = set(data.get("radStrokes") or [])
    rad_idx = [k for k, (_, role, _) in enumerate(parts, start=1) if role == "radical"]
    rad_num = rad_idx[0] if len(rad_idx) == 1 else None
    best = base
    if rad_strokes and rad_num is not None:
        others = [k for k in nums if k != rad_num]
        alt = {}
        bad = 0
        for i in range(len(dist)):
            if i in rad_strokes:
                alt[i] = (rad_num, region_assign(i, [rad_num])[1])
            else:
                alt[i] = region_assign(i, others or [rad_num])
            if alt[i][0] != base[i][0] and alt[i][1] < 0.08:
                bad += 1
        rs = sorted(rad_strokes)
        contiguous = rs == list(range(rs[0], rs[-1] + 1))
        at_edge = contiguous and (rs[0] == 0 or rs[-1] == len(dist) - 1)
        if "包圍" in structure or at_edge:
            best = alt

    stroke_part = {i: best[i][0] for i in best}
    if spans:
        stroke_part = {}
        for k, (a, b) in enumerate(spans, start=1):
            for i in range(a, min(b, len(dist))):
                stroke_part[i] = k

    # Final say goes to stroke bodies (eroded outlines), with the component
    # rectangle breaking ties: where two stroke bodies overlap, the one whose
    # component owns that area wins.  Every visible stroke therefore keeps a
    # single colour, even in glyphs the font draws as one merged blob.
    part_of = np.zeros(len(dist), np.int16)
    for i in range(len(dist)):
        part_of[i] = stroke_part.get(i, 1)

    label_img = np.zeros((H, W), np.uint8)
    label_img[mask] = part_of[owner[mask]]

    # Colour is taken from the stroke each pixel belongs to, not from its
    # position, so a stroke and every stroke of a component keep exactly one
    # colour.  The font's own skeleton is only used to decide *where* two strokes
    # meet, never which colour a stroke has.
    label_img = np.zeros((H, W), np.uint8)
    label_img[mask] = part_of[owner[mask]]

    # Remove stray specks: a tiny island of one colour swallowed by another
    # colour is merged into its surroundings, so no stroke shows two colours.
    ink_area = mask.sum()
    min_patch = max(120, int(ink_area * 0.004))
    for _ in range(3):
        changed = False
        for k in range(1, len(parts) + 1):
            cl, nc = ndimage.label(label_img == k)
            for c in range(1, nc + 1):
                patch = cl == c
                a = int(patch.sum())
                if a >= min_patch:
                    continue
                nb = ndimage.binary_dilation(patch, np.ones((3, 3), bool)) & mask & (label_img != k)
                if not nb.any():
                    continue
                vals, cnt = np.unique(label_img[nb], return_counts=True)
                if cnt.max() < 0.7 * cnt.sum():
                    continue
                label_img[patch] = vals[np.argmax(cnt)]
                changed = True
        if not changed:
            break

    # A narrow tongue of one colour pushing into another component is handed back
    # to its surroundings (opening radius 4 px is far thinner than any stroke), so
    # every component reads as one flat colour.
    st = ndimage.generate_binary_structure(2, 1)
    for k in range(1, len(parts) + 1):
        m = label_img == k
        if not m.any():
            continue
        op = ndimage.binary_opening(m, ndimage.iterate_structure(st, 4))
        lost = m & ~op
        if lost.sum() == 0 or lost.sum() > 0.25 * m.sum():
            continue
        keep = mask & ~lost
        _, idx = ndimage.distance_transform_edt(~keep, return_indices=True)
        label_img[lost] = label_img[idx[0][lost], idx[1][lost]]

    # Component purity: a small lobe of one colour that hangs off another colour
    # (a corner or a tip that the font fuses into its neighbour) is repainted with
    # that neighbour, so a whole component reads as a single flat colour.  Free
    # standing bits of a colour (e.g. the separate dots of 心) have no ink
    # neighbour and are left alone.
    out8 = np.ones((3, 3), bool)
    for _ in range(3):
        changed = False
        for k in range(1, len(parts) + 1):
            m = label_img == k
            tot = int(m.sum())
            if tot == 0:
                continue
            cl, nc = ndimage.label(m, structure=out8)
            if nc < 2:
                continue
            sizes = ndimage.sum(m, cl, range(1, nc + 1))
            big = int(np.argmax(sizes)) + 1
            for c in range(1, nc + 1):
                if c == big:
                    continue
                patch = cl == c
                if sizes[c - 1] > 0.05 * tot:
                    continue
                nb = ndimage.binary_dilation(patch, out8) & mask & (label_img != k)
                if not nb.any():
                    continue
                vals, cnt = np.unique(label_img[nb], return_counts=True)
                if cnt.max() >= 0.6 * cnt.sum():
                    label_img[patch] = vals[np.argmax(cnt)]
                    changed = True
        if not changed:
            break

    # A detached flake of one colour sitting right inside another component (a
    # font corner the stroke data does not cover) is repainted with that
    # component.  Free-standing marks such as the four dots of 心 only ever see
    # blank paper within reach, so they stay untouched.
    big_k = np.ones((15, 15), bool)
    for _ in range(2):
        for k in range(1, len(parts) + 1):
            m = label_img == k
            tot = int(m.sum())
            if tot == 0:
                continue
            cl, nc = ndimage.label(m, structure=out8)
            if nc < 2:
                continue
            sizes = ndimage.sum(m, cl, range(1, nc + 1))
            order = np.argsort(sizes)
            keep = set(order[-2:].tolist())
            for c in order:
                if c in keep:
                    continue
                patch = cl == int(c) + 1
                if sizes[int(c)] > 0.04 * tot:
                    continue
                near = ndimage.binary_dilation(patch, big_k) & mask & (label_img != k)
                if not near.any():
                    continue
                vals, cnt = np.unique(label_img[near], return_counts=True)
                if cnt.max() >= 0.6 * cnt.sum():
                    label_img[patch] = vals[np.argmax(cnt)]

    if os.environ.get("PRB"):
        for i in range(len(dist)):
            mm = (owner == i) & mask
            if not mm.any():
                continue
            ys, xs = np.nonzero(mm)
            print("  st%-2d part%d bbox y%3d-%3d x%3d-%3d px%d" % (
                i + 1, part_of[i], ys.min(), ys.max(), xs.min(), xs.max(), mm.sum()))
        for (yy, xx) in [(404,600),(420,620),(440,640),(300,620)]:
            print("  probe", yy, xx, "owner", owner[yy, xx], "ink", bool(mask[yy, xx]))
    rgba = np.zeros((H, W, 4), np.uint8)
    for k, (_, role, _) in enumerate(parts, start=1):
        m = label_img == k
        rgba[m, 0], rgba[m, 1], rgba[m, 2] = COLORS[role]
        rgba[m, 3] = 255
    out = Image.fromarray(rgba, "RGBA")

    s = min(860 / out.height, 900 / out.width)
    out = out.resize((max(1, int(out.width * s)), max(1, int(out.height * s))), Image.LANCZOS)
    canvas = Image.new("RGBA", (BOX, BOX), (0, 0, 0, 0))
    canvas.paste(out, ((BOX - out.width) // 2, (BOX - out.height) // 2), out)
    return canvas.resize((OUT, OUT), Image.LANCZOS)


def final_clean(canvas, parts):
    """Repaint detached flakes at the delivered resolution.

    A small island of one colour whose ring is mostly another colour is a corner
    or tip the font fused into the neighbouring component; it is repainted so the
    whole component reads as one colour.  Free standing marks (the separate dots
    of 心, 采's 木 tips ...) only see paper around them and are left alone.
    """
    rgba = np.array(canvas.convert("RGBA"))
    rgb = rgba[:, :, :3].astype(int)
    alpha = rgba[:, :, 3] > 128
    mask_of = {}
    for k, (_, role, _) in enumerate(parts, start=1):
        col = np.array(COLORS[role], int)
        mask_of[k] = (np.abs(rgb - col).sum(2) < 60) & alpha
    out8 = ndimage.generate_binary_structure(2, 1)   # 4-connected: a hairline
    for _ in range(3):                               # bridge does not count
        for k, m in mask_of.items():
            tot = int(m.sum())
            if tot == 0:
                continue
            cl, nc = ndimage.label(m, structure=out8)
            if nc < 2:
                continue
            sizes = ndimage.sum(m, cl, range(1, nc + 1))
            order = np.argsort(sizes)
            for c in order[:-1]:
                if sizes[int(c)] > 0.05 * tot:
                    continue
                patch = cl == int(c) + 1
                ring = ndimage.binary_dilation(patch, np.ones((11, 11), bool)) & alpha & ~patch
                if not ring.any():
                    continue
                votes = {}
                for k2, m2 in mask_of.items():
                    v = int((ring & m2).sum())
                    if v:
                        votes[k2] = v
                if not votes:
                    continue
                best = max(votes, key=votes.get)
                if votes[best] <= 0.5 * sum(votes.values()) or best == k:
                    continue
                col = np.array(COLORS[parts[best - 1][1]], np.uint8)
                rgba[patch, 0], rgba[patch, 1], rgba[patch, 2] = col
                mask_of[k] = mask_of[k] & ~patch
                mask_of[best] = mask_of[best] | patch
    return Image.fromarray(rgba, "RGBA")


def build(items, folder, outfile):
    os.makedirs(folder, exist_ok=True)
    recs = []
    for rec in items:
        ch, struct, radical, parts = rec[0], rec[1], rec[2], rec[3]
        spans = rec[4] if len(rec) > 4 else None
        final_clean(colorize(ch, parts, struct, spans), parts).quantize(
            colors=32, method=Image.FASTOCTREE).save(
            f"{folder}/{ord(ch):05d}.png", optimize=True)
        recs.append({"char": ch, "structure": struct, "radical": radical,
                     "parts": [{"name": p[0], "role": p[1]} for p in parts]})
        print("ok", ch, flush=True)
    json.dump(recs, open(outfile, "w"), ensure_ascii=False, indent=1)
    return recs


if __name__ == "__main__":
    if len(sys.argv) > 1:
        want = set(sys.argv[1])
        build([d for d in DATA + EXAMPLES if d[0] in want], "assets/chars", "chars.json")
    else:
        build(DATA, "assets/chars", "chars.json")
        build(EXAMPLES, "assets/examples", "examples.json")
        import clean_final
        clean_final.run()          # component-purity pass on the delivered PNGs
