# -*- coding: utf-8 -*-
"""Component-purity pass over the delivered PNGs.

Two defects are removed, so that one component always reads as a single flat
colour and no colour runs along the body of another component's stroke:

1. seams  - a 1-2 px filament of one colour lying along the edge of another
            colour (an artefact of the resize / palette step).  Filaments do not
            survive a 3x3 opening, so the pixels it removes are handed to the
            nearest remaining ink.
2. flakes - a detached blob of one colour whose 11 px ring is mostly another
            colour: a corner or tip the font fused into the neighbouring
            component.  Repainted with that neighbour.

Free standing marks (the four dots of 心, the tips of 木) only see paper around
them and are never touched.
"""
import json

import numpy as np
from PIL import Image
from scipy import ndimage

COLORS = {"radical": (192, 57, 43), "part": (43, 108, 176),
          "part2": (47, 133, 90), "part3": (183, 121, 31)}
FOUR = ndimage.generate_binary_structure(2, 1)
EIGHT = np.ones((3, 3), bool)


def _masks(rgb, alpha, parts):
    out = {}
    for k, p in enumerate(parts, start=1):
        col = np.array(COLORS[p["role"]], int)
        out[k] = (np.abs(rgb - col).sum(2) < 60) & alpha
    return out


def _paint(rgba, mask, role):
    col = COLORS[role]
    rgba[mask, 0], rgba[mask, 1], rgba[mask, 2] = col


def clean(path, parts):
    im = Image.open(path).convert("RGBA")
    rgba = np.array(im)
    alpha = rgba[:, :, 3] > 128
    rgb = rgba[:, :, :3].astype(int)
    mask_of = _masks(rgb, alpha, parts)
    changed = 0

    for _ in range(3):
        # ---- 1. seams: hairlines of one colour inside another ----
        lab = np.zeros(alpha.shape, np.int16)
        for k, m in mask_of.items():
            lab[m] = k
        for k in list(mask_of):
            m = mask_of[k]
            if not m.any():
                continue
            # radius 2 (5x5 cross) is needed: a seam left by the resize is 2-4 px
            # wide, so a 3x3 opening still lets it through.  No real stroke is that
            # thin, so nothing legitimate is lost.
            op = ndimage.binary_opening(m, ndimage.iterate_structure(FOUR, 2))
            lost = m & ~op
            if lost.sum() == 0 or lost.sum() > 0.35 * m.sum():
                continue
            keep = alpha & ~lost
            if not keep.any():
                continue
            _, idx = ndimage.distance_transform_edt(~keep, return_indices=True)
            lab[lost] = lab[idx[0][lost], idx[1][lost]]
            changed += int(lost.sum())

        # ---- 2. flakes: detached islands wrapped in another colour ----
        for k in list(mask_of):
            m = lab == k
            mask_of[k] = m
        for k in list(mask_of):
            m = mask_of[k]
            tot = int(m.sum())
            if tot == 0:
                continue
            cl, nc = ndimage.label(m, structure=FOUR)
            if nc < 2:
                continue
            sizes = ndimage.sum(m, cl, range(1, nc + 1))
            for c in np.argsort(sizes)[:-1]:
                if sizes[int(c)] > 0.05 * tot:
                    continue
                patch = cl == int(c) + 1
                ring = ndimage.binary_dilation(patch, np.ones((11, 11), bool)) & alpha & ~patch
                votes = {}
                for k2 in mask_of:
                    v = int((ring & (lab == k2)).sum())
                    if v:
                        votes[k2] = v
                if not votes:
                    continue
                best = max(votes, key=votes.get)
                if best == k or votes[best] <= 0.5 * sum(votes.values()):
                    continue
                lab[patch] = best
                mask_of[k] = mask_of[k] & ~patch
                mask_of[best] = mask_of[best] | patch
                changed += int(patch.sum())

        rgba[:, :, :3] = 0
        for k, p in enumerate(parts, start=1):
            _paint(rgba, lab == k, p["role"])

    if changed:
        Image.fromarray(rgba, "RGBA").quantize(colors=32, method=Image.FASTOCTREE).save(
            path, optimize=True)
    return changed


# A handful of glyphs fuse one component's brush tip into its neighbour so tightly
# that no pixel-level test separates them.  For those, the structural split is
# enforced geometrically: 上下結構 uses a horizontal frontier, 左右/左中右 a
# vertical one.  (char, axis, position, part_index held above the frontier)
HARD = {
    "思": ("y", 0.56, 1),      # 田 above, 心 below
    "湖": ("x", 0.30, 1),      # 氵 left, 古+月 right
}


def _hard_clip(path, parts, rule):
    axis, cut, idx = rule
    im = Image.open(path).convert("RGBA")
    rgba = np.array(im)
    alpha = rgba[:, :, 3] > 128
    h, w = alpha.shape
    sel = np.zeros((h, w), bool)
    if axis == "y":
        sel[int(round(cut * h)):, :] = True      # everything BELOW the frontier
    else:
        sel[:, int(round(cut * w)):] = True
    keep = np.ones((h, w), bool) if False else None
    # One-way sweep: above the frontier only the upper component may appear.  The
    # lower component's ink never reaches up there, so nothing legitimate moves -
    # and the upper component's own tips that hang below the frontier are left
    # alone.
    upper = alpha & ~sel
    col_upper = COLORS[parts[idx - 1]["role"]]
    rgb = rgba[:, :, :3].astype(int)
    foreign = upper & (np.abs(rgb - np.array(col_upper, int)).sum(2) >= 60)
    rgba[foreign, 0], rgba[foreign, 1], rgba[foreign, 2] = col_upper
    Image.fromarray(rgba, "RGBA").quantize(colors=32, method=Image.FASTOCTREE).save(
        path, optimize=True)


def run(only=None):
    total = 0
    for folder, meta in (("chars", "chars.json"), ("examples", "examples.json")):
        for rec in json.load(open(meta)):
            if only and rec["char"] not in only:
                continue
            n = clean("assets/%s/%05d.png" % (folder, ord(rec["char"])), rec["parts"])
            if rec["char"] in HARD:
                _hard_clip("assets/%s/%05d.png" % (folder, ord(rec["char"])),
                           rec["parts"], HARD[rec["char"]])
                print("  %s hard-split applied" % rec["char"])
            if n:
                print("  %s repaired %d px" % (rec["char"], n))
            total += n
    print("total repaired px", total)


if __name__ == "__main__":
    import sys
    run(set(sys.argv[1]) if len(sys.argv) > 1 else None)
