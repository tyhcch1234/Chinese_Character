# -*- coding: utf-8 -*-
"""把通用管線（parts_auto）的輸出正式寫入 assets/，再重建 index.html。

用法：
  python3 _src/publish_auto.py /tmp/auto_v2          # 寫入 assets/ + 重建 index.html
  python3 _src/publish_auto.py /tmp/auto_v2 --dry    # 只報告會改動哪些檔

只複製 DATA + EXAMPLES 內的字，並把 assets/ 內不被引用的孤兒圖示列出（不刪）。
"""
import os
import shutil
import sys

import build_parts4 as bp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    src = sys.argv[1]
    dry = "--dry" in sys.argv[2:]

    plan = []
    for rec in bp.DATA:
        plan.append((rec[0], "chars"))
    for rec in bp.EXAMPLES:
        plan.append((rec[0], "examples"))

    changed = 0
    for ch, folder in plan:
        name = "%05d.png" % ord(ch)
        s = os.path.join(src, name)
        d = os.path.join("assets", folder, name)
        if not os.path.exists(s):
            print("MISSING build output: %s (%s)" % (ch, s))
            return 1
        same = os.path.exists(d) and os.path.getsize(s) == os.path.getsize(d) \
            and open(s, "rb").read() == open(d, "rb").read()
        if not same:
            changed += 1
            print("%-2s %s -> assets/%s/%s" % (ch, "same" if same else "update", folder, name))
            if not dry:
                shutil.copyfile(s, d)

    print("\n%d/%d 圖更新" % (changed, len(plan)))

    # 孤兒圖示（assets 內有、但不被 DATA/EXAMPLES 引用）
    keep = {"%05d.png" % ord(ch) for ch, _ in plan}
    for folder in ("chars", "examples"):
        if not os.path.isdir(os.path.join("assets", folder)):
            continue
        for f in sorted(os.listdir(os.path.join("assets", folder))):
            if f.endswith(".png") and f not in keep:
                print("orphan: assets/%s/%s" % (folder, f))

    if not dry:
        os.system("%s _src/build_site.py" % sys.executable)
        print("index.html rebuilt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
