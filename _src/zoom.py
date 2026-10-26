# -*- coding: utf-8 -*-
"""把兩張圖的同一區域並排放大，用肉眼檢查接縫／切邊。

用法：
  python3 _src/zoom.py 湖 assets/chars/28246.png /tmp/auto_v9/28246.png
  python3 _src/zoom.py 湖 A.png B.png 0.10 0.30 0.55 0.65   # x0 y0 x1 y1（相對比例）
輸出：_preview/zoom_<字>.png（先按 alpha bbox 對齊，再裁同一相對區域）
"""
import os
import sys

import numpy as np
from PIL import Image

OUT = 900   # 對齊後的正方形邊長


def aligned(path):
    im = Image.open(path).convert("RGBA")
    a = np.array(im)[:, :, 3] > 128
    ys, xs = np.nonzero(a)
    return im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)).resize(
        (OUT, OUT), Image.NEAREST)


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    ch, pa_, pb_ = sys.argv[1], sys.argv[2], sys.argv[3]
    box = [float(v) for v in sys.argv[4:8]] or [0.10, 0.20, 0.55, 0.65]
    x0, y0, x1, y1 = (int(box[0] * OUT), int(box[1] * OUT),
                      int(box[2] * OUT), int(box[3] * OUT))
    a, b = aligned(pa_).crop((x0, y0, x1, y1)), aligned(pb_).crop((x0, y0, x1, y1))
    w, h = a.size
    out = Image.new("RGBA", (2 * w + 14, h), (255, 255, 255, 255))
    out.paste(a, (0, 0))
    out.paste(b, (w + 14, 0))
    os.makedirs("_preview", exist_ok=True)
    dst = "_preview/zoom_%s.png" % ch
    out.convert("RGB").save(dst)
    print("%s  A=%s  B=%s" % (dst, pa_, pb_))
    return 0


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.exit(main())
