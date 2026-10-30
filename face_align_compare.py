"""
얼굴 인식 정렬 5점 — (a) 검출(YuNet, 긴 변 480) 5점 그대로 대 (b) 제품의 확대 재측정(MobileFaceNetModel::alignFiveFromCrop:
검출 상자 크롭을 4.4배 검은 정사각 캔버스 가운데 → 640² YuNet → IoU ≥ 0.3 검출의 5점). 인식은 SFace(ORT) 같은 것으로.
  CCTV 를 흉내 낸다: LFW 사진을 1920×1080 회색 프레임 안에 얼굴 폭 F px 로 축소해 넣는다(F = 60 · 80 · 120 · 200).
  (c) 검출 5점 + 크롭 정렬(제품 (b) 와 같은 정렬 원천 — 크롭 밖은 검정)도 함께 잰다: 큰 얼굴에서 (b) 를 대신할 후보.
  두 방식은 **같은 검출 결과**를 쓰고 5점만 다르다. 검출 실패 쌍은 두 방식에서 함께 빠진다.
  지표: LFW 표준 6000쌍 10-fold 정확도, TAR@FAR 1e-3 · 1e-2, 비용(YuNet 호출 수).
usage: python face_align_compare.py <models_dir> <out_dir> [long_side=480]
"""
import json, os, sys, time
import numpy as np, cv2, onnxruntime as ort
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import roc_curve
from bench_common import similarity_to_arc

M, O = sys.argv[1:3]
L = int(sys.argv[3]) if len(sys.argv) > 3 else 480
os.makedirs(O, exist_ok=True)
ort.set_default_logger_severity(3)
so = ort.SessionOptions(); so.intra_op_num_threads = 4
yunet = ort.InferenceSession(os.path.join(M, "yunet_2026may.onnx"), so, providers=["CPUExecutionProvider"])
sface = ort.InferenceSession(os.path.join(M, "face_recognition_sface_2021dec.onnx"), so, providers=["CPUExecutionProvider"])
YN = [o.name for o in yunet.get_outputs()]
FW, FH, ZOOM, AGREE = 1920, 1080, 4.4, 0.3
SIZES = [60, 80, 100, 120, 160, 200]


def iou_xywh(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])); iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    u = a[2] * a[3] + b[2] * b[3] - ix * iy
    return ix * iy / u if u > 0 else 0


def detect(rgb, long_side):
    """YuNetFaceModel::detectLocked 와 같다(long_side 0 = 640² 늘리기). 반환 [(box xywh, score, kps 5×2)] 원본 좌표."""
    h, w = rgb.shape[:2]
    if long_side <= 0:
        inW = inH = padW = padH = 640; small = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
    else:
        s = min(1.0, long_side / max(w, h)); inW, inH = max(1, round(w * s)), max(1, round(h * s))
        small = cv2.resize(rgb, (inW, inH), interpolation=cv2.INTER_AREA) if s < 1 else rgb
        padW, padH = ((inW - 1) // 32 + 1) * 32, ((inH - 1) // 32 + 1) * 32
    blob = np.zeros((1, 3, padH, padW), np.float32); blob[0, :, :inH, :inW] = cv2.cvtColor(small, cv2.COLOR_RGB2BGR).transpose(2, 0, 1)
    out = dict(zip(YN, yunet.run(None, {"input": blob}))); sx, sy = w / inW, h / inH
    boxes, scores, kps = [], [], []
    for st in (8, 16, 32):
        cols, rows = padW // st, padH // st; n = cols * rows
        cls = out[f"cls_{st}"].reshape(-1)[:n]; obj = out[f"obj_{st}"].reshape(-1)[:n]
        bb = out[f"bbox_{st}"].reshape(-1, 4); kp = out[f"kps_{st}"].reshape(-1, 10)
        sc = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))
        for i in np.where(sc >= 0.3)[0]:
            c, r = i % cols, i // cols
            cx, cy = (c + bb[i, 0]) * st, (r + bb[i, 1]) * st; bw, bh = np.exp(bb[i, 2]) * st, np.exp(bb[i, 3]) * st
            boxes.append([cx - bw / 2, cy - bh / 2, bw, bh]); scores.append(float(sc[i]))
            kps.append([((c + kp[i, 2 * j]) * st, (r + kp[i, 2 * j + 1]) * st) for j in range(5)])
    ib = [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in boxes]; kept = []
    for k in sorted(range(len(scores)), key=lambda k: -scores[k]):
        if all(iou_xywh(ib[k], ib[j]) <= 0.3 for j in kept): kept.append(k)
    return [([boxes[k][0] * sx, boxes[k][1] * sy, boxes[k][2] * sx, boxes[k][3] * sy], scores[k],
             np.array([(p[0] * sx, p[1] * sy) for p in kps[k]], np.float64)) for k in kept]


def zoom_kps(crop):
    """MobileFaceNetModel::alignFiveFromCrop 의 5점(크롭 좌표)."""
    h, w = crop.shape[:2]; S = int(round(ZOOM * max(w, h)))
    if S < 8: return None
    x0, y0 = (S - w) // 2, (S - h) // 2
    canvas = np.zeros((S, S, 3), np.uint8); canvas[y0:y0 + h, x0:x0 + w] = crop
    best, bo = None, 0.0
    for box, sc, kp in detect(canvas, 0):
        ov = iou_xywh([x0, y0, w, h], box)
        if ov >= AGREE and ov > bo: best, bo = kp, ov
    return None if best is None else best - np.array([x0, y0], np.float64)


def embed(rgb, kps):
    a = cv2.warpAffine(rgb, similarity_to_arc(kps), (112, 112), flags=cv2.INTER_LINEAR, borderValue=0)
    v = sface.run(None, {"data": a.astype(np.float32).transpose(2, 0, 1)[None]})[0].reshape(-1)
    return v / (np.linalg.norm(v) + 1e-12)


fetch_lfw_pairs(subset="10_folds", resize=0.05, data_home=os.path.join(O, "sk"))
home = os.path.join(O, "sk", "lfw_home")
rows = [l.split() for l in open(os.path.join(home, "pairs.txt")).read().splitlines()[1:] if l.strip()]
path = lambda n, i: os.path.join(home, "lfw_funneled", n, f"{n}_{int(i):04d}.jpg")
pairs = [(path(r[0], r[1]), path(r[0], r[2]), 1) if len(r) == 3 else (path(r[0], r[1]), path(r[2], r[3]), 0) for r in rows]
y = np.array([p[2] for p in pairs]); imgs = sorted({p[k] for p in pairs for k in (0, 1)}); idx = {p: i for i, p in enumerate(imgs)}
A = np.array([idx[p[0]] for p in pairs]); B = np.array([idx[p[1]] for p in pairs]); folds = np.repeat(np.arange(10), 600)

# LFW 원본에서 얼굴 폭(축소 배율 기준)
base_w = np.zeros(len(imgs))
for i, p in enumerate(imgs):
    d = detect(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), 0)
    if d: base_w[i] = min(d, key=lambda q: np.hypot(q[0][0] + q[0][2] / 2 - 125, q[0][1] + q[0][3] / 2 - 125))[0][2]
print(json.dumps({"images": len(imgs), "face_found_in_original": int((base_w > 0).sum()), "long_side": L}), flush=True)


def metrics(E, ok):
    sims = np.where(ok, (E[A] * E[B]).sum(1), -1.0); accs = []
    for k in range(10):
        tr = (folds != k) & ok; te = (folds == k) & ok
        th = np.quantile(sims[tr], np.linspace(0, 1, 2001)); best = max(th, key=lambda h: ((sims[tr] >= h) == y[tr]).mean())
        accs.append(((sims[te] >= best) == y[te]).mean())
    fpr, tpr, _ = roc_curve(y[ok], sims[ok])
    return {"acc10": round(float(np.mean(accs)) * 100, 2), "tar@far1e-2": round(float(np.interp(1e-2, fpr, tpr)), 4),
            "tar@far1e-3": round(float(np.interp(1e-3, fpr, tpr)), 4)}


res = {"long_side": L, "sizes": {}}
for F in SIZES:
    Ea = np.zeros((len(imgs), 128), np.float32); Eb = np.zeros_like(Ea); Ec = np.zeros_like(Ea); ok_img = np.zeros(len(imgs), bool)
    t0 = time.time(); zoom_fail = 0; det_w = np.zeros(len(imgs))
    for i, p in enumerate(imgs):
        if base_w[i] <= 0: continue
        src = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB); k = F / base_w[i]
        sm = cv2.resize(src, (max(1, round(250 * k)), max(1, round(250 * k))), interpolation=cv2.INTER_AREA)
        frame = np.full((FH, FW, 3), 114, np.uint8); ox, oy = (FW - sm.shape[1]) // 2, (FH - sm.shape[0]) // 2
        frame[oy:oy + sm.shape[0], ox:ox + sm.shape[1]] = sm
        c = np.array([FW / 2, FH / 2])
        dets = detect(frame, L)
        if not dets: continue
        box, sc, kp = min(dets, key=lambda q: np.hypot(q[0][0] + q[0][2] / 2 - c[0], q[0][1] + q[0][3] / 2 - c[1]))
        x1, y1 = max(0, int(box[0])), max(0, int(box[1])); x2, y2 = min(FW, int(box[0] + box[2])), min(FH, int(box[1] + box[3]))
        if x2 - x1 < 2 or y2 - y1 < 2: continue
        crop = frame[y1:y2, x1:x2]
        zk = zoom_kps(crop)
        if zk is None: zoom_fail += 1; continue      # 제품은 정렬 실패면 그 얼굴을 인식하지 않는다 — 두 방식에서 함께 뺀다
        Ea[i] = embed(frame, kp)                      # (a) 검출 5점, 원본 프레임에서 정렬
        Eb[i] = embed(crop, zk)                       # (b) 확대 재측정 5점, 크롭에서 정렬(제품과 같다)
        Ec[i] = embed(crop, kp - np.array([x1, y1], np.float64))   # (c) 검출 5점, 크롭에서 정렬 — (b) 와 같은 정렬 원천
        ok_img[i] = True; det_w[i] = box[2] * min(1.0, L / max(FW, FH))
    ok = ok_img[A] & ok_img[B]
    r = {"pairs_used": int(ok.sum()), "zoom_refind_fail": zoom_fail, "a_detector_kps": metrics(Ea, ok), "b_zoom_refine": metrics(Eb, ok),
         "c_detector_kps_crop": metrics(Ec, ok),
         "a_vs_b_cos_mean": round(float((Ea[ok_img] * Eb[ok_img]).sum(1).mean()), 4),
         "c_vs_b_cos_mean": round(float((Ec[ok_img] * Eb[ok_img]).sum(1).mean()), 4),
         "c_vs_b_cos_p05": round(float(np.quantile((Ec[ok_img] * Eb[ok_img]).sum(1), 0.05)), 4),
         "det_face_px_in_input_median": round(float(np.median(det_w[ok_img])), 1), "sec": round(time.time() - t0)}
    res["sizes"][F] = r; print(json.dumps({"face_px": F, **r}), flush=True)
json.dump(res, open(os.path.join(O, "face_align_compare.json"), "w"), indent=1)
