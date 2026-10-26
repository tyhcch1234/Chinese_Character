# -*- coding: utf-8 -*-
"""通用部件分色：由結構類型自動推導分界，不再需要手畫矩形 / spans / HARD 補丁。

六條不變量（見 docs/method.md）：
  P1 顏色 = f(筆畫)
  P2 部件 = 筆畫的集合，且為分割
  P3 部件領地 = 其筆畫中心線的 Voronoi
  P4 分界線落在空白走廊（由投影剖面求得，並用來驗證分組）
  P5 幾何聽字型、身分聽筆畫資料
  P6 降取樣用最近色，不混色（因此不需要接縫／碎片補丁）

用法：
  python3 _src/parts_auto.py --check              # 檢查現有 assets/ 的圖
  python3 _src/parts_auto.py --build OUTDIR       # 用通用規則重建到 OUTDIR
  python3 _src/parts_auto.py --build OUTDIR 思 湖  # 只做指定字
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

import build_parts4 as bp

COLORS = bp.COLORS
BASELINE = bp.BASELINE
OUT = bp.OUT
BOX = bp.BOX


# --------------------------------------------------------------------------- #
# 幾何：把 hanzi-writer 的筆畫中心線貼到字型墨跡上（P5：字型決定幾何）
# --------------------------------------------------------------------------- #
GEOM = "_data/geom2"        # 含逐筆輪廓歸屬（own），不進 git


def stroke_field(ch):
    """回傳 (data, mask, dist)：mask 為字型墨跡，dist[i] 為到第 i 筆中心線的距離。"""
    cache = os.path.join(GEOM, "%05d.npz" % ord(ch))
    data = bp.hanzi_data(ch)
    if os.path.exists(cache):
        z = np.load(cache)
        return data, z["mask"], z["dist"].astype(np.float32)

    mask = bp.render_mask(ch)
    H, W = mask.shape

    strokes, allx, ally = [], [], []
    for d in data["strokes"]:
        pts = [(x, BASELINE - y) for x, y in bp.path_points(d)]
        strokes.append(pts)
        allx += [p[0] for p in pts]
        ally += [p[1] for p in pts]
    hx0, hx1, hy0, hy1 = min(allx), max(allx), min(ally), max(ally)

    ys, xs = np.nonzero(mask)
    gx0, gx1, gy0, gy1 = xs.min(), xs.max(), ys.min(), ys.max()
    sx = (gx1 - gx0) / max(1e-6, hx1 - hx0)
    sy = (gy1 - gy0) / max(1e-6, hy1 - hy0)
    cx, cy = (hx0 + hx1) / 2.0, (hy0 + hy1) / 2.0
    tx, ty = (gx0 + gx1) / 2.0, (gy0 + gy1) / 2.0

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
    res = minimize(cost, [(sx + sy) / 2, tx, ty, 1.0], method="Nelder-Mead",
                   options={"maxiter": 600, "xatol": 1e-3, "fatol": 1e-4})
    s_best, tx, ty, ax = res.x

    def to_px(p):
        return ((p[0] - cx) * s_best * ax + tx, (p[1] - cy) * s_best + ty)

    dist = np.full((len(data["medians"]), H, W), np.inf, dtype=np.float32)
    for i, med in enumerate(data["medians"]):
        med = [(x, BASELINE - y) for x, y in med]
        pts = bp.densify(med, step=max(4.0, 8.0 / max(sx, sy)))
        layer = Image.new("L", (W, H), 0)
        dr = ImageDraw.Draw(layer)
        xy = [to_px(p) for p in pts]
        dr.line(xy, fill=255, width=1)
        for p in xy:
            dr.ellipse([p[0] - 0.6, p[1] - 0.6, p[0] + 0.6, p[1] + 0.6], fill=255)
        dist[i] = ndimage.distance_transform_edt(np.array(layer) < 128)

    # 逐筆輪廓歸屬（P1 的真正依據）：把每一筆自己的外形填色，後畫的蓋先畫的。
    # 只用最近中心線會在兩筆相鄰處「切邊」——例如「湖」的提會被判給「古」，
    # 出現一條藍色薄片。用筆畫本身的外形就不會。
    own = Image.new("I", (W, H), 0)
    dr_own = ImageDraw.Draw(own)
    for i, d in enumerate(data["strokes"]):
        for poly in bp.svg_polys(d):
            if len(poly) >= 3:
                dr_own.polygon([to_px((x, BASELINE - y)) for x, y in poly], fill=i + 1)
    own = np.array(own, np.int16)
    if (own[mask] == 0).any():                     # 字型墨跡超出行廓 → 退回最近中心線
        o0 = (np.argmin(dist, axis=0) + 1).astype(np.int16)
        own = np.where((own == 0) & mask, o0, own)

    os.makedirs(GEOM, exist_ok=True)
    np.savez_compressed(cache, mask=mask, dist=dist.astype(np.float16), own=own)
    return data, mask, dist


def stroke_own(ch):
    """每一筆的墨跡（1-based；0 = 不屬任何筆）。

    輪廓歸屬之後再加一條爭議區規則：若某像素雖落在第 i 筆輪廓內，但它離另一筆
    中心線明顯更近（近 2px 以上），就判給那一筆。這是 P5 的落地——字型的筆形
    比 hanzi 輪廓粗，兩者重疊處要聽字型（例：湖的提右上角不該被判給古的橫）。
    """
    stroke_field(ch)
    z = np.load(os.path.join(GEOM, "%05d.npz" % ord(ch)))
    own = z["own"].astype(np.int16)
    dist = z["dist"].astype(np.float32)
    H, W = own.shape
    near = (np.argmin(dist, axis=0) + 1).astype(np.int16)
    idx = np.clip(own - 1, 0, dist.shape[0] - 1)
    d_own = dist[idx, np.arange(H)[:, None], np.arange(W)[None, :]]
    d_near = dist.min(axis=0)
    sel = (own > 0) & (own != near) & (d_near + 2.0 < d_own)
    own[sel] = near[sel]
    return own


# --------------------------------------------------------------------------- #
# P4：由投影剖面找空白走廊
# --------------------------------------------------------------------------- #
def lanes(mask, axis, lo=0.22, hi=0.78):
    """空白走廊位置（0-1），由墨跡投影剖面的低谷求得，由深到淺。"""
    prof = (mask.sum(0) if axis == "x" else mask.sum(1)).astype(float)
    n = len(prof)
    peak = prof.max() or 1.0
    out = []
    for i in range(2, n - 2):
        pos = i / n
        if not (lo <= pos <= hi):
            continue
        if prof[i] <= prof[i - 1] and prof[i] <= prof[i + 1]:
            out.append((prof[i] / peak, pos))
    out.sort()
    keep, min_gap = [], 0.08
    for depth, pos in out:
        if depth > 0.35:               # 不是真的走廊
            continue
        if all(abs(pos - k) > min_gap for k in keep):
            keep.append(pos)
        if len(keep) >= 2:
            break
    return keep


def cuts(mask, axis, need, lo=0.12, hi=0.88):
    """挑出 need 條分界線（0-1），目標是「切到的墨跡最少」，不是「最深低谷」。

    為甚麼不用 `lanes()`：低谷只看局部深度，遇到跨件的長橫（湖的「一」）就會
    在部件內部挑到假走廊（0.26/0.36 相鄰兩條），令「古」被並進「月」。
    這裡改成全局最小割：在 lo..hi 內選 need 條線，使
      1. 合計切到的墨跡像素最少；
      2. 每條帶闊度 ≥ 10% 字寬；
      3. 每條帶至少佔 5% 墨跡（不許空帶）。
    分界線落在長橫最窄處（P4 允許「空白走廊或筆畫交叉點」），而身分仍由
    筆畫整體決定，所以線切過一筆不會把它分成兩色。
    """
    import itertools
    prof = (mask.sum(0) if axis == "x" else mask.sum(1)).astype(float)
    n = len(prof)
    total = prof.sum() or 1.0
    if need <= 0:
        return []
    lo_i, hi_i = int(lo * n), int(hi * n)
    cand = [i for i in range(lo_i, hi_i + 1)
            if prof[i] <= prof[i - 1] and prof[i] <= prof[i + 1]]
    if len(cand) > 60:
        cand = sorted(sorted(cand, key=lambda i: prof[i])[:60])
    wmin = 0.10 * n
    inkmin = 0.05 * total
    best = None
    for combo in itertools.combinations(cand, need):
        if any(combo[i + 1] - combo[i] < wmin for i in range(len(combo) - 1)):
            continue
        edges = [0] + list(combo) + [n]
        if any(edges[i + 1] - edges[i] < wmin for i in range(len(edges) - 1)):
            continue
        if any(prof[edges[i]:edges[i + 1]].sum() < inkmin for i in range(len(edges) - 1)):
            continue
        cost = sum(prof[c] for c in combo)
        if best is None or cost < best[0]:
            best = (cost, combo)
    if best is None:
        return []
    return [c / n for c in best[1]]


def stroke_centroids(dist, mask, own=None):
    if own is None:
        own = np.argmin(dist, axis=0) + 1
    out = []
    for i in range(dist.shape[0]):
        m = (own == i + 1) & mask
        if not m.any():
            ys, xs = np.nonzero(mask)
            out.append((xs.mean(), ys.mean()))
            continue
        ys, xs = np.nonzero(m)
        out.append((xs.mean(), ys.mean()))
    return np.array(out, float)


# --------------------------------------------------------------------------- #
# P2：把筆畫切成連續區塊（一維 k-means，容許的最優分段）
# --------------------------------------------------------------------------- #
def split_1d(values, k):
    """把已排序的一維數值切成 k 段連續區間，使組內平方誤差最小。"""
    v = np.asarray(values, float)
    n = len(v)
    if k <= 1 or n <= 1:
        return [list(range(n))]
    c = np.concatenate([[0.0], np.cumsum(v)])
    c2 = np.concatenate([[0.0], np.cumsum(v * v)])

    def cost(a, b):                      # 段 [a, b)
        m = b - a
        s = c[b] - c[a]
        return (c2[b] - c2[a]) - s * s / m

    INF = float("inf")
    best = [[INF] * (n + 1) for _ in range(k + 1)]
    prev = [[-1] * (n + 1) for _ in range(k + 1)]
    best[0][0] = 0.0
    for g in range(1, k + 1):
        for b in range(1, n + 1):
            for a in range(g - 1, b):
                if best[g - 1][a] == INF:
                    continue
                val = best[g - 1][a] + cost(a, b)
                if val < best[g][b]:
                    best[g][b] = val
                    prev[g][b] = a
    groups, b = [], n
    for g in range(k, 0, -1):
        a = prev[g][b]
        groups.append(list(range(a, b)))
        b = a
    return groups[::-1]


def assign_parts(ch, structure, parts, data, mask, dist):
    """回傳長度 = 筆畫數 的部件索引陣列（1-based）。"""
    n = dist.shape[0]
    k = len(parts)
    try:
        own = stroke_own(ch)
    except Exception:
        own = None
    cent = stroke_centroids(dist, mask, own)
    part_of = np.ones(n, np.int16)

    def radical_index():
        for i, p in enumerate(parts, start=1):
            if p[1] == "radical":
                return i
        return 1

    if k == 1 or n == 1:
        return part_of

    rad = set(data.get("radStrokes") or [])
    ri = radical_index()

    if "包圍" in structure:
        for i in range(n):
            part_of[i] = ri if i in rad else (2 if ri == 1 else 1)
        return part_of

    if "品字" in structure:
        order = np.argsort(cent[:, 1])
        g = split_1d(cent[order, 1], 2)
        top = [int(order[i]) for i in g[0]]
        bottom = [int(order[i]) for i in g[1]]
        for i in top:
            part_of[i] = 1
        sub = np.argsort(cent[bottom, 0])
        g2 = split_1d(cent[bottom][sub, 0], 2)
        for gi, gg in enumerate(g2, start=2):
            for j in gg:
                part_of[int(np.array(bottom)[sub[j]])] = gi
        return part_of

    axis = 0 if ("左右" in structure or "左中右" in structure) else 1
    need = max(1, min(3, k - 1))
    ln = cuts(mask, "x" if axis == 0 else "y", need)
    span = mask.shape[1] if axis == 0 else mask.shape[0]

    def dp_fallback():
        part_of[:] = 1
        order = np.argsort(cent[:, axis])
        for gi, g in enumerate(split_1d(cent[order, axis], k), start=1):
            for j in g:
                part_of[int(order[j])] = gi

    if len(ln) == need:
        # 硬約束（P4）：每一筆按重心落在哪一條帶分組，不再讓 DP 自由分群。
        edges = [0.0] + list(ln) + [1.0]
        for i in range(n):
            part_of[i] = int(min(k, np.searchsorted(edges, cent[i, axis] / span, side="right")))
        if len(set(part_of.tolist())) < k:      # 有帶空了 → 退回 DP
            dp_fallback()
    else:
        dp_fallback()

    # 部首必須整組落在同一個部件（P5：身分聽筆畫資料）
    if rad and len(rad) < n:
        target = part_of[sorted(rad)[0]]
        if any(part_of[i] != target for i in rad):
            for i in range(n):
                part_of[i] = ri if i in rad else 0
            rest = [i for i in range(n) if part_of[i] == 0]
            lefts = [p for p in range(1, k + 1) if p != ri] or [ri]
            sub = np.argsort(cent[rest, axis])
            gs = split_1d(cent[rest][sub, axis], len(lefts))
            for gi, g in enumerate(gs):
                for j in g:
                    part_of[rest[int(sub[j])]] = lefts[min(gi, len(lefts) - 1)]
        elif target != ri:
            for a in range(1, k + 1):
                if part_of[sorted(rad)[0]] == a:
                    for i in range(n):
                        if part_of[i] == a:
                            part_of[i] = ri
                        elif part_of[i] == ri:
                            part_of[i] = a
                    break
    return part_of


# --------------------------------------------------------------------------- #
# 繪製：P3 領地 + P6 最近色降取樣
# --------------------------------------------------------------------------- #
def lane_owner(mask, dist, structure, k, lx=None, ly=None):
    """P3+P4：先以走廊把平面分區，再在「本區的筆畫」之間取最近的。

    跨走廊的筆畫不能搶另一區的墨跡，所以一筆的領地不會橫越走廊。
    lx / ly 可由呼叫者給定（例如由筆畫重心分群的中點），否則自行找低谷。
    """
    H, W = mask.shape
    nneed = max(1, min(3, k - 1))
    if lx is None:
        lx = lanes(mask, "x")[:nneed] if ("左右" in structure or "左中右" in structure
                                          or "品字" in structure) else []
    if ly is None:
        ly = lanes(mask, "y")[:nneed] if ("上下" in structure or "上中下" in structure
                                          or "品字" in structure) else []
    cent = stroke_centroids(dist, mask)

    def signature(x, y):
        return (tuple(int(x > c * W) for c in lx)
                + tuple(int(y > c * H) for c in ly))

    ys_g, xs_g = np.mgrid[0:H, 0:W]
    px_sig = {tuple(int(x > c * W) for c in lx) + tuple(int(y > c * H) for c in ly)
              for x, y in ((0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1))}
    stro_sig = [signature(cent[i, 0], cent[i, 1]) for i in range(dist.shape[0])]

    owner = np.full((H, W), -1, np.int16)
    for sig in px_sig:
        members = [i for i in range(dist.shape[0]) if stro_sig[i] == sig]
        if not members:
            members = list(range(dist.shape[0]))
        sel = np.ones((H, W), bool)
        for j, c in enumerate(lx):
            sel &= (xs_g > c * W) == bool(sig[j])
        for j, c in enumerate(ly):
            sel &= (ys_g > c * H) == bool(sig[len(lx) + j])
        d = dist[members]
        sub = np.argmin(d, axis=0)
        owner[sel] = np.array(members)[sub[sel]]
    if (owner < 0).any():
        glob = np.argmin(dist, axis=0)
        owner[owner < 0] = glob[owner < 0]
    return owner, lx, ly


LOCK_FILE = "identity.json"
_LOCK = None


def identity_lock(ch):
    """已驗收圖鎖定的逐筆身分（1-based）；無鎖定時回 None。"""
    global _LOCK
    if _LOCK is None:
        _LOCK = json.load(open(LOCK_FILE)) if os.path.exists(LOCK_FILE) else {}
    seq = _LOCK.get("%05d" % ord(ch))
    return list(seq) if seq else None


def render(ch, structure, parts, path=None):
    data, mask, dist = stroke_field(ch)
    part_of = assign_parts(ch, structure, parts, data, mask, dist)
    lock = identity_lock(ch)
    if lock and len(lock) == len(part_of):
        # 有鎖定就用鎖定：自動分組只是候選，已驗收圖才是語意基準。
        k = len(parts)
        part_of = np.array([v if 1 <= v <= k else int(part_of[i])
                            for i, v in enumerate(lock)], np.int16)
    cent = stroke_centroids(dist, mask)
    H, W = mask.shape

    # P4：k 個部件需要 k-1 條分界線；但分界線只用來決定「身分」，
    # 不用來決定顏色區域——顏色一律跟隨筆畫（P1/P3），所以一條線切過長橫
    # 也不會把該筆分成兩色。
    # 顏色分區：走廊帶 + 帶內最近筆畫（P3/P4）。走廊由筆畫重心分群的中點給出，
    # 比「切口墨跡最少」穩定——後者會挑到部件內部的假低谷（湖的 0.36）。
    need = max(1, min(3, len(parts) - 1))
    axis = 0 if ("左右" in structure or "左中右" in structure) else 1
    cent = stroke_centroids(dist, mask, None)
    order = np.argsort(cent[:, axis])
    groups = split_1d(cent[order, axis], len(parts))
    span = W if axis == 0 else H
    cuts_ = []
    for gi in range(len(groups) - 1):
        hi_ = cent[order[groups[gi]]][:, axis].max()
        lo_ = cent[order[groups[gi + 1]]][:, axis].min()
        cuts_.append(((hi_ + lo_) / 2.0) / span)
    owner, _, _ = lane_owner(mask, dist, structure, len(parts),
                             lx=cuts_ if axis == 0 else None,
                             ly=cuts_ if axis == 1 else None)

    lab = np.zeros((H, W), np.uint8)
    lab[mask] = part_of[owner[mask]]

    im = Image.fromarray(lab, "L")
    s = min(OUT / im.height, OUT / im.width)
    w, h = max(1, int(im.width * s)), max(1, int(im.height * s))
    small = np.array(im.resize((w, h), Image.NEAREST))      # 最近色，不混色

    y0 = (BOX - h) // 2
    x0 = (BOX - w) // 2
    canvas = np.zeros((BOX, BOX, 4), np.uint8)
    tile = canvas[y0:y0 + h, x0:x0 + w]
    for k, part in enumerate(parts, start=1):
        m = small == k
        tile[m, 0], tile[m, 1], tile[m, 2] = COLORS[part[1]]
        tile[m, 3] = 255
    canvas[y0:y0 + h, x0:x0 + w] = tile
    out = Image.fromarray(canvas, "RGBA")
    if path:
        out.save(path, optimize=True)
    return out, dict(part_of=part_of, mask=mask, dist=dist, lanes={
        "x": lanes(mask, "x"), "y": lanes(mask, "y")})


# --------------------------------------------------------------------------- #
# 驗證（任何一項 > 0 即 fail）
# --------------------------------------------------------------------------- #
def quick_metrics(png_path, parts):
    """不看筆畫，只看圖：接縫、孤島、覆蓋。"""
    im = Image.open(png_path).convert("RGBA")
    rgba = np.array(im)
    rgb = rgba[:, :, :3].astype(int)
    alpha = rgba[:, :, 3] > 128
    out = {}
    seam = 0
    islands = 0
    total = 0
    for p in parts:
        m = (np.abs(rgb - np.array(COLORS[p[1]], int)).sum(2) < 60) & alpha
        if not m.any():
            continue
        total += int(m.sum())
        op = ndimage.binary_opening(m, ndimage.generate_binary_structure(2, 1))
        seam += int((m & ~op).sum())
        islands += ndimage.label(m, structure=ndimage.generate_binary_structure(2, 1))[1]
    out["seam_px"] = seam
    out["islands"] = islands
    out["coverage"] = int(abs(total - int(alpha.sum())))
    return out


def main():
    args = [a for a in sys.argv[1:]]
    if not args:
        print(__doc__)
        return
    items = bp.DATA + bp.EXAMPLES
    if args[0] == "--check":
        rows = []
        for rec in items:
            folder = "chars" if rec in bp.DATA else "examples"
            path = "assets/%s/%05d.png" % (folder, ord(rec[0]))
            if not os.path.exists(path):
                continue
            m = quick_metrics(path, rec[3])
            rows.append((rec[0], m))
        bad = [r for r in rows if r[1]["seam_px"] or r[1]["coverage"]]
        print("checked %d, failing %d" % (len(rows), len(bad)))
        for ch, m in bad[:15]:
            print("  %s seam=%d islands=%d cov=%d" % (ch, m["seam_px"], m["islands"], m["coverage"]))
        return

    if args[0] == "--ab":
        outdir = args[1]
        os.makedirs(outdir, exist_ok=True)
        rows = []
        for rec in items:
            ch, structure, parts = rec[0], rec[1], rec[3]
            folder = "chars" if rec in bp.DATA else "examples"
            old_path = "assets/%s/%05d.png" % (folder, ord(ch))
            new_path = os.path.join(outdir, "%05d.png" % ord(ch))
            if not os.path.exists(old_path):
                continue
            if not os.path.exists(new_path):
                render(ch, structure, parts, new_path)
            mo = quick_metrics(old_path, parts)
            mn = quick_metrics(new_path, parts)
            rows.append((ch, mo, mn))
        worse = [r for r in rows if r[2]["seam_px"] > r[1]["seam_px"] + 40]
        better = [r for r in rows if r[2]["seam_px"] + 40 < r[1]["seam_px"]]
        print("chars %d | seam better %d worse %d" % (len(rows), len(better), len(worse)))
        for ch in ("思", "湖", "幫", "國", "病", "喜", "架", "籃"):
            r = next((x for x in rows if x[0] == ch), None)
            if r:
                print("  %s old seam %3d -> new %3d | islands %d -> %d" % (
                    ch, r[1]["seam_px"], r[2]["seam_px"],
                    r[1]["islands"], r[2]["islands"]))
        print("worse list:", " ".join(r[0] for r in worse) or "-")
        return

    if args[0] == "--build":
        outdir = args[1]
        only = set(args[2:]) if len(args) > 2 else None
        os.makedirs(outdir, exist_ok=True)
        for rec in items:
            ch, structure, radical, parts = rec[0], rec[1], rec[2], rec[3]
            if only and ch not in only:
                continue
            p = os.path.join(outdir, "%05d.png" % ord(ch))
            _, info = render(ch, structure, parts, p)
            m = quick_metrics(p, parts)
            print("ok %s seam=%d islands=%d cov=%d lanes x=%s y=%s" % (
                ch, m["seam_px"], m["islands"], m["coverage"],
                ["%.2f" % v for v in info["lanes"]["x"][:2]],
                ["%.2f" % v for v in info["lanes"]["y"][:2]]), flush=True)


if __name__ == "__main__":
    main()
