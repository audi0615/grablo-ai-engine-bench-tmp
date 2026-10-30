import numpy as np

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


