# -*- coding: utf-8 -*-
"""回歸測試：以「現有已驗收的圖」為語意基準，逐筆比對新管線的部件歸屬。

為什麼要這支：`audit.py` 的純度指標是循環論證——它用同一套 `part_of` 去定義
每一筆的領地，所以「顏色乾淨」不代表「顏色對」。真正獨立、且有人類背書的基準
只有一個：已經上線、用家看過的 49 張圖。

做法：對每一筆，取舊圖上該筆墨跡的主要顏色 → 舊圖認為它屬哪個部件；
再問新管線 `part_of` 認為它屬哪個部件。不一致就是語意回歸。

用法：
  python3 _src/regress.py                      # 全部字
  python3 _src/regress.py 湖 幫 思              # 指定字
  python3 _src/regress.py --quiet              # 只印總結
"""
import os
import sys

import numpy as np
from PIL import Image

import build_parts4 as bp
import parts_auto as pa

TOL = 0.85          # 舊圖某色佔比低於此值 → 視為「舊圖本身模糊」，不當基準


def old_part_per_stroke(path, parts, mask, dist, ch=None):
    """舊圖逐筆主色 → (部件索引, 佔比)；0 表示無法判定。"""
    from pngmap import stroke_parts
    return stroke_parts(path, parts, mask, dist, ch)


def main():
    quiet = "--quiet" in sys.argv
    only = set("".join(a for a in sys.argv[1:] if not a.startswith("-")))
    items = bp.DATA + bp.EXAMPLES
    bad_chars, bad_strokes, checked = [], 0, 0
    for rec in items:
        ch, structure, parts = rec[0], rec[1], rec[3]
        if only and ch not in only:
            continue
        folder = "chars" if rec in bp.DATA else "examples"
        path = "assets/%s/%05d.png" % (folder, ord(ch))
        if not os.path.exists(path):
            continue
        data, mask, dist = pa.stroke_field(ch)
        part_of = pa.assign_parts(ch, structure, parts, data, mask, dist)
        old = old_part_per_stroke(path, parts, mask, dist, ch)
        names = [p[0] for p in parts]
        msgs = []
        for i, (op, share) in enumerate(old):
            checked += 1
            if op == 0 or share < TOL:
                continue
            if op != part_of[i]:
                bad_strokes += 1
                msgs.append("筆%02d 舊=%s 新=%s(%.0f%%)" % (i + 1, names[op - 1], names[part_of[i] - 1], 100 * share))
        if msgs:
            bad_chars.append(ch)
            if not quiet:
                print("%-2s 回歸 %d 筆：%s" % (ch, len(msgs), "；".join(msgs)), flush=True)
    print("\n== 語意回歸 ==  有差異的字 %d / %d，差異筆畫 %d（可判定 %d 筆）" % (
        len(bad_chars), len([r for r in items if not only or r[0] in only]), bad_strokes, checked))
    if bad_chars:
        print("字：%s" % " ".join(bad_chars))


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
