# -*- coding: utf-8 -*-
"""把「已驗收圖」的部件身分鎖成資料檔 `identity.json`。

自動分組（走廊 / DP）在 14 個字上與已驗收圖不符（共 46 筆，例如「籃」的 7 筆
被判給「竹」、「慚」的 8 筆被判給「忄」）。純度指標看不出這件事，因為它是用
同一套 part_of 定義領地的循環論證；`regress.py` 用舊圖當基準才照得出來。

這裡把舊圖的逐筆顏色當答案寫進 `identity.json`：
  {"28246": [1,1,1,2,2,2,2,3,3,3,3,3], ...}      # 1-based 部件索引
- 舊圖該筆主要顏色佔比 ≥ 0.5 才採用，否則記 0（交由自動分組補）。
- 之後 render() 若有鎖定身分，就以鎖定為準；新字（無鎖定）仍走自動。

用法：
  python3 _src/identity_lock.py            # 由 assets/ 產生 identity.json
  python3 _src/identity_lock.py --report   # 只報告，不寫檔
"""
import json
import os
import sys

import build_parts4 as bp
import parts_auto as pa
from regress import old_part_per_stroke


def main():
    report = "--report" in sys.argv
    locked, unknown = {}, 0
    for rec in bp.DATA + bp.EXAMPLES:
        ch, structure, parts = rec[0], rec[1], rec[3]
        folder = "chars" if rec in bp.DATA else "examples"
        path = "assets/%s/%05d.png" % (folder, ord(ch))
        if not os.path.exists(path):
            continue
        data, mask, dist = pa.stroke_field(ch)
        old = old_part_per_stroke(path, parts, mask, dist, ch)
        seq = [(p if share >= 0.5 else 0) for p, share in old]
        unknown += sum(1 for v in seq if v == 0)
        locked["%05d" % ord(ch)] = seq
        print("%-2s %s" % (ch, " ".join(str(v) for v in seq)), flush=True)
    print("\n鎖定 %d 字，未定筆畫 %d" % (len(locked), unknown))
    if not report:
        json.dump(locked, open("identity.json", "w"), ensure_ascii=False, indent=0, sort_keys=True)
        print("已寫入 identity.json")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
