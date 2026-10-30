import numpy as np

ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], np.float64)


def similarity_to_arc(src):
    """제품 MobileFaceNetModel.cpp similarityToArc 를 줄 단위로 옮긴 것 — 5점 → arcface_dst 최소제곱 닮음변환(2×3)."""
    ms, md = src.mean(0), ARCFACE_DST.mean(0)
    x, d = src - ms, ARCFACE_DST - md
    den = (x ** 2).sum()
    a = (x[:, 0] * d[:, 0] + x[:, 1] * d[:, 1]).sum() / den
    b = (x[:, 0] * d[:, 1] - x[:, 1] * d[:, 0]).sum() / den
    return np.array([[a, -b, md[0] - (a * ms[0] - b * ms[1])], [b, a, md[1] - (b * ms[0] + a * ms[1])]], np.float64)

def decode_yolox(out, S, sc, W, H):
    out = out.reshape(-1, 85); dets = []; i = 0
    for s in (8, 16, 32):
        g = S // s
        for gy in range(g):
            for gx in range(g):
                a = out[i]; i += 1; c = int(np.argmax(a[5:])); sco = a[4] * a[5 + c]
                if sco < 0.2: continue
                cx, cy = (a[0] + gx) * s, (a[1] + gy) * s; bw, bh = np.exp(a[2]) * s, np.exp(a[3]) * s
                dets.append((float(sco), c, max(0, (cx - bw / 2) / sc), max(0, (cy - bh / 2) / sc), min(W, (cx + bw / 2) / sc), min(H, (cy + bh / 2) / sc)))
    dets.sort(key=lambda d: -d[0]); keep = []
    for d in dets:
        if all(k[1] != d[1] or iou(k, d) <= 0.45 for k in keep): keep.append(d)
    return keep


def iou(a, b):
    ix = max(0, min(a[4], b[4]) - max(a[2], b[2])); iy = max(0, min(a[5], b[5]) - max(a[3], b[3])); n = ix * iy
    u = (a[4] - a[2]) * (a[5] - a[3]) + (b[4] - b[2]) * (b[5] - b[3]) - n
    return n / u if u > 0 else 0


def match(ref, tst):
    used, m, ious, sd = set(), 0, [], []
    for r in ref:
        best, bj = 0, -1
        for j, t in enumerate(tst):
            if j in used or t[1] != r[1]: continue
            v = iou(r, t)
            if v > best: best, bj = v, j
        if bj >= 0 and best > 0.5: used.add(bj); m += 1; ious.append(best); sd.append(abs(r[0] - tst[bj][0]))
    return {"ref": len(ref), "dets": len(tst), "matched": m, "min_iou": round(float(min(ious)), 3) if ious else None, "max_score_diff": round(float(max(sd)), 3) if sd else None}


