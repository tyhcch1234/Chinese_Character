# -*- coding: utf-8 -*-
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

FONT = "_font/GenRyuMin2TW-M.otf"
RENDER = 1800

def ink_image(ch):
    n = RENDER
    fs = int(n * 1.35)
    font = ImageFont.truetype(FONT, fs)
    img = Image.new("L", (n * 2, n * 2), 0)
    d = ImageDraw.Draw(img)
    d.text((n // 2, n + n // 3), ch, font=font, fill=255, anchor="ls")
    return img.crop(img.getbbox())

for ch in sys.argv[1:]:
    arr = np.array(ink_image(ch))
    h, w = arr.shape
    mask = arr > 110
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    print(f"== {ch}  blobs={n}  size={w}x{h}")
    objs = ndimage.find_objects(lab)
    for i, sl in enumerate(objs, start=1):
        ys, xs = sl
        area = int((lab[sl] == i).sum())
        if area < 40:
            continue
        cy = (ys.start + ys.stop) / 2 / h
        cx = (xs.start + xs.stop) / 2 / w
        print(f"  #{i:<3} area={area:<7} x[{xs.start/w:.2f},{xs.stop/w:.2f}] "
              f"y[{ys.start/h:.2f},{ys.stop/h:.2f}] c=({cx:.2f},{cy:.2f})")
