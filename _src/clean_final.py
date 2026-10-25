# -*- coding: utf-8 -*-
"""Component-purity pass over the delivered PNGs.

A detached flake of one colour whose 11 px ring is mostly another colour is a
corner or tip the font fused into the neighbouring component; it is repainted so
the whole component reads as one colour.  Free standing marks (the separate dots
of 心, tips of 木 ...) only see paper around them and are left alone.
"""
import json

import numpy as np
from PIL import Image
from scipy import ndimage

COLORS = {"radical": (192, 57, 43), "part": (43, 108, 176),
          "part2": (47, 133, 90), "part3": (183, 121, 31)}


def clean(path, parts):
    im = Image.open(path).convert("RGBA")
    rgba = np.array(im)
    rgb = rgba[:, :, :3].astype(int)
    alpha = rgba[:, :, 3] > 128
    mask_of = {}
    for k, p in enumerate(parts, start=1):
        col = np.array(COLORS[p["role"]], int)
        mask_of[k] = (np.abs(rgb - col).sum(2) < 60) & alpha
    out8 = ndimage.generate_binary_structure(2, 1)
    changed = 0
    for _ in range(4):
        for k, m in list(mask_of.items()):
            tot = int(m.sum())
            if tot == 0:
                continue
            cl, nc = ndimage.label(m, structure=out8)
            if nc < 2:
                continue
            sizes = ndimage.sum(m, cl, range(1, nc + 1))
            for c in np.argsort(sizes)[:-1]:
                if sizes[int(c)] > 0.05 * tot:
                    continue
                patch = cl == int(c) + 1
                ring = ndimage.binary_dilation(patch, np.ones((11, 11), bool)) & alpha & ~patch
                votes = {}
                for k2, m2 in mask_of.items():
                    v = int((ring & m2).sum())
                    if v:
                        votes[k2] = v
                if not votes:
                    continue
                best = max(votes, key=votes.get)
                if best == k or votes[best] <= 0.5 * sum(votes.values()):
                    continue
                col = np.array(COLORS[parts[best - 1]["role"]], np.uint8)
                rgba[patch, 0], rgba[patch, 1], rgba[patch, 2] = col
                mask_of[k] = mask_of[k] & ~patch
                mask_of[best] = mask_of[best] | patch
                changed += int(patch.sum())
    if changed:
        Image.fromarray(rgba, "RGBA").quantize(colors=32, method=Image.FASTOCTREE).save(
            path, optimize=True)
    return changed


def run():
    total = 0
    for folder, meta in (("chars", "chars.json"), ("examples", "examples.json")):
        for rec in json.load(open(meta)):
            n = clean("assets/%s/%05d.png" % (folder, ord(rec["char"])), rec["parts"])
            if n:
                print("  %s repaired %d px" % (rec["char"], n))
            total += n
    print("total repaired px", total)


if __name__ == "__main__":
    run()
