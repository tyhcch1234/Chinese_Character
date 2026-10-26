# -*- coding: utf-8 -*-
"""修 identity.json：把舊圖殘留的「單筆錯色」按 hanzi 筆畫資料（P5）糾正。

舊圖雖然是人看過的基準，但它本身有少量殘留錯誤（例如「温」皿底一筆仍是氵的
紅色、「愧」忄的第二點是藍色）。直接鎖定會把這些錯誤一併繼承，而且新的「一筆
一色」會讓它們更明顯。這裡只做兩條有資料依據的修正：

  R1 `radStrokes` 全部屬於部首部件（P5：身分聽筆畫資料）。
  R2 標為部首、但與部首其他筆畫相距很遠、且更接近另一部件的筆，改歸該部件。
     （「幫」的白＋巾相鄰，不會被 R2 誤改。）

用法：
  python3 _src/lock_repair.py            # 修正並覆寫 identity.json
  python3 _src/lock_repair.py --report   # 只列出會改動的筆
"""
import json
import os
import sys

import numpy as np

import build_parts4 as bp
import parts_auto as pa


def main():
    report = "--report" in sys.argv
    lock = json.load(open("identity.json"))
    changes = 0
    for rec in bp.DATA + bp.EXAMPLES:
        ch, structure, parts = rec[0], rec[1], rec[3]
        key = "%05d" % ord(ch)
        if key not in lock:
            continue
        data, mask, dist = pa.stroke_field(ch)
        seq = lock[key]
        if len(seq) != len(dist):
            print("%s 筆畫數不符（%d vs %d），跳過" % (ch, len(seq), len(dist)))
            continue
        names = [p[0] for p in parts]
        rad = list(data.get("radStrokes") or [])
        k = len(parts)
        ri = next((i for i, p in enumerate(parts, start=1) if p[1] == "radical"), 1)
        cent = pa.stroke_centroids(dist, mask)
        H, W = mask.shape
        norm = np.stack([cent[:, 0] / W, cent[:, 1] / H], axis=1)

        def d(a, b):
            return float(np.hypot(*(norm[a] - norm[b])))

        msgs = []
        # R1
        if rad and k > 1:
            for i in rad:
                if seq[i] != ri:
                    msgs.append("R1 筆%02d %s→%s" % (i + 1, names[seq[i] - 1], names[ri - 1]))
                    seq[i] = ri
            # R2
            others = [i for i in range(len(seq)) if seq[i] == ri and i not in rad]
            for i in others:
                if not rad:
                    break
                d_rad = min(d(i, j) for j in rad)
                best, best_d = None, 9.9
                for j in range(len(seq)):
                    if seq[j] == ri:
                        continue
                    dd = d(i, j)
                    if dd < best_d:
                        best, best_d = seq[j], dd
                if best is not None and d_rad > 0.15 and best_d < 0.5 * d_rad:
                    msgs.append("R2 筆%02d %s→%s（距部首%.2f，距%s %.2f）" % (
                        i + 1, names[ri - 1], names[best - 1], d_rad, names[best - 1], best_d))
                    seq[i] = best
        if msgs:
            changes += len(msgs)
            print("%-2s %s" % (ch, "；".join(msgs)), flush=True)
        lock[key] = seq
    print("\n共 %d 處修正" % changes)
    if not report:
        json.dump(lock, open("identity.json", "w"), ensure_ascii=False, indent=0, sort_keys=True)
        print("已覆寫 identity.json")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
