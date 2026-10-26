# -*- coding: utf-8 -*-
"""把成品 PNG 的顏色映回字型墨跡座標（render() 的嚴格逆運算）。

render() 的做法是：lab(H×W) --縮到 (w,h)--> 貼進 BOX×BOX 畫布中央。
所以逆運算必須「先裁出貼圖區域，再縮回 (H,W)」，不可以直接把整張 1000×1000
縮到 (H,W)——那樣會把四周留白一起縮進去，令墨跡錯位約 10%。
"""
import numpy as np
from PIL import Image

import build_parts4 as bp
import parts_auto as pa


def tile_geometry(mask):
    H, W = mask.shape
    s = min(bp.OUT / H, bp.OUT / W)
    w, h = max(1, int(W * s)), max(1, int(H * s))
    y0 = (bp.BOX - h) // 2
    x0 = (bp.BOX - w) // 2
    return s, w, h, x0, y0


def lab_on_mask(path, parts, mask):
    """回傳 (H,W) 的部件索引圖（0 = 無色）與布林墨跡。"""
    rgba = np.array(Image.open(path).convert("RGBA"))
    rgb = rgba[:, :, :3].astype(int)
    alpha = rgba[:, :, 3] > 128
    lab = np.zeros(rgb.shape[:2], np.uint8)
    for k, p in enumerate(parts, start=1):
        lab[(np.abs(rgb - np.array(pa.COLORS[p[1]], int)).sum(2) < 70) & alpha] = k
    H, W = mask.shape
    # 通用做法：把 PNG 的實際墨跡外框（alpha bbox）對到字型墨跡外框。
    # 舊版圖是 800×800 且字只佔 83%，新版是 1000×1000 且字佔 77%，
    # 直接整圖縮放會錯位 10–17%（顏色比對會全部報錯）。
    ys, xs = np.nonzero(alpha)
    if len(xs):
        src = lab[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    else:
        src = lab
    back = np.array(Image.fromarray(src).resize((W, H), Image.NEAREST))
    return back, mask.astype(bool)


def stroke_parts(path, parts, mask, dist, ch=None):
    """每一筆在圖上的主要顏色 → (部件索引, 佔比)；0 表示無法判定。"""
    lab, ink = lab_on_mask(path, parts, mask)
    if ch is not None:
        own = pa.stroke_own(ch)
        owner0 = np.where(own > 0, own - 1, np.argmin(dist, axis=0))
    else:
        owner0 = np.argmin(dist, axis=0)
    out = []
    for i in range(dist.shape[0]):
        sel = (owner0 == i) & ink & (lab > 0)
        if not sel.any():
            out.append((0, 0.0))
            continue
        vals, cnt = np.unique(lab[sel], return_counts=True)
        j = int(np.argmax(cnt))
        out.append((int(vals[j]), float(cnt[j] / cnt.sum())))
    return out
