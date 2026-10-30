"""
인텔 맥에서 얼굴 Core ML(YuNet · SFace) 대 ORT 1.17.1 — 정확도와 속도. 인텔 맥에는 ANE 가 없어 Core ML 은 CPU · GPU 로 돈다.
  YuNet : WIDER FACE val 앞 N장, 제품 절차(tools/coreml_convert/face_coreml_check.py 와 같은 디코드 · 공식 NMS) — ORT 대 Core ML
          최종 검출 1:1 대조 + 추론 시간(긴 변 320/480/640)
  SFace : LFW 표준 6000쌍 10-fold — YuNet(ORT) 5점 → arcface_dst 112 정렬, 엔진마다 정확도 · TAR@FAR, ORT 대비 코사인, 시간
  Core ML 계산 장치: CPU_ONLY · CPU_AND_GPU · ALL (실패하면 오류 문자열을 남긴다)
usage: python face_coreml_intel.py <models_dir> <wider_val_images_dir> <out_dir> [yunet_images=300]
"""
import glob, json, os, platform, sys, time
import numpy as np, cv2, onnxruntime as ort, coremltools as ct
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import roc_curve
from bench_common import similarity_to_arc

M, WIDER, O = sys.argv[1:4]
NY = int(sys.argv[4]) if len(sys.argv) > 4 else 300
os.makedirs(O, exist_ok=True)
ort.set_default_logger_severity(3)
so = ort.SessionOptions(); so.intra_op_num_threads = 2   # 제품 getOptimalOrtVisionThreads()
y_ort = ort.InferenceSession(os.path.join(M, "yunet_2026may.onnx"), so, providers=["CPUExecutionProvider"])
s_ort = ort.InferenceSession(os.path.join(M, "face_recognition_sface_2021dec.onnx"), so, providers=["CPUExecutionProvider"])
NAMES = [o.name for o in y_ort.get_outputs()]
UNITS = ["CPU_ONLY", "CPU_AND_GPU", "ALL"]
res = {"cpu": platform.processor(), "platform": platform.platform(), "coremltools": ct.__version__, "ort": ort.__version__}
try:
    import subprocess; res["cpu_brand"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
except Exception: pass
print(json.dumps(res), flush=True)


def timeit(fn, n=20):
    for _ in range(3): fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); t.append((time.perf_counter() - t0) * 1e3)
    t.sort(); return round(t[len(t) // 2], 2)


def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])); iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    u = a[2] * a[3] + b[2] * b[3] - ix * iy
    return ix * iy / u if u > 0 else 0


def prep(rgb, L):
    h, w = rgb.shape[:2]; s = min(1.0, L / max(w, h))
    inW, inH = max(1, round(w * s)), max(1, round(h * s))
    small = cv2.resize(rgb, (inW, inH), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    padW, padH = ((inW - 1) // 32 + 1) * 32, ((inH - 1) // 32 + 1) * 32
    blob = np.zeros((1, 3, padH, padW), np.float32); blob[0, :, :inH, :inW] = cv2.cvtColor(small, cv2.COLOR_RGB2BGR).transpose(2, 0, 1)
    return blob, w / inW, h / inH, padW, padH


def decode(out, sx, sy, padW, padH):
    boxes, scores, kps = [], [], []
    for st in (8, 16, 32):
        cols, rows = padW // st, padH // st
        cls = out[f"cls_{st}"].reshape(-1); obj = out[f"obj_{st}"].reshape(-1)
        bb = out[f"bbox_{st}"].reshape(-1, 4); kp = out[f"kps_{st}"].reshape(-1, 10)
        sc_all = np.sqrt(np.clip(cls[:cols * rows], 0, 1) * np.clip(obj[:cols * rows], 0, 1))
        for i in np.where(sc_all >= 0.3)[0]:
            c, r = i % cols, i // cols
            cx, cy = (c + bb[i, 0]) * st, (r + bb[i, 1]) * st; bw, bh = np.exp(bb[i, 2]) * st, np.exp(bb[i, 3]) * st
            boxes.append([cx - bw / 2, cy - bh / 2, bw, bh]); scores.append(float(sc_all[i]))
            kps.append([((c + kp[i, 2 * j]) * st * sx, (r + kp[i, 2 * j + 1]) * st * sy) for j in range(5)])
    ib = [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in boxes]; kept = []
    for k in sorted(range(len(scores)), key=lambda k: -scores[k]):
        if all(iou(ib[k], ib[j]) <= 0.3 for j in kept): kept.append(k)
    return [([boxes[k][0] * sx, boxes[k][1] * sy, boxes[k][2] * sx, boxes[k][3] * sy], scores[k], kps[k]) for k in kept]


def y_run(engine, blob):
    if engine == "ort": return dict(zip(NAMES, y_ort.run(None, {"input": blob})))
    o = engine.predict({"input": blob}); return {n: np.asarray(o[n], np.float32) for n in NAMES}


def detect(engine, rgb, L):
    blob, sx, sy, pw, ph = prep(rgb, L)
    return decode(y_run(engine, blob), sx, sy, pw, ph), blob


# ── YuNet ────────────────────────────────────────────────────────────────────
wider = sorted(glob.glob(os.path.join(WIDER, "**", "*.jpg"), recursive=True))[:NY]
print(json.dumps({"wider_images": len(wider)}), flush=True)
wimgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in wider]
y_cml = {}
for u in UNITS:
    try: y_cml[u] = ct.models.MLModel(os.path.join(M, "yunet_2026may.mlpackage"), compute_units=getattr(ct.ComputeUnit, u))
    except Exception as e: res[f"yunet_{u}_load_error"] = str(e)[:300]
for L in (320, 480, 640):
    ref = [detect("ort", im, L) for im in wimgs]
    blob0 = ref[0][1]
    row = {"long_side": L, "ort_faces": sum(len(r[0]) for r in ref), "ort_ms": timeit(lambda: y_ort.run(None, {"input": blob0}))}
    for u, m in y_cml.items():
        try:
            n_c = matched = 0; w_iou, w_sc, w_kp = 1.0, 0.0, 0.0
            for im, (a, _) in zip(wimgs, ref):
                b, _ = detect(m, im, L); n_c += len(b); used = set()
                for ab, asc, akp in a:
                    best, bj = 0, -1
                    for j, (bb, _, _) in enumerate(b):
                        if j not in used and iou(ab, bb) > best: best, bj = iou(ab, bb), j
                    if bj < 0 or best < 0.5: continue
                    used.add(bj); matched += 1; bb, bsc, bkp = b[bj]
                    w_iou = min(w_iou, best); w_sc = max(w_sc, abs(asc - bsc))
                    w_kp = max(w_kp, max(np.hypot(p[0] - q[0], p[1] - q[1]) for p, q in zip(akp, bkp)))
            row[u] = {"faces": n_c, "matched": matched, "worst_iou": round(w_iou, 4), "worst_score_diff": round(w_sc, 4),
                      "worst_kp_px": round(float(w_kp), 3), "ms": timeit(lambda: m.predict({"input": blob0}))}
        except Exception as e:
            row[u] = {"error": str(e)[:300]}
    res[f"yunet_{L}"] = row; print(json.dumps({"yunet": row}), flush=True)

# ── SFace (LFW 6000쌍) ────────────────────────────────────────────────────────
fetch_lfw_pairs(subset="10_folds", resize=0.05, data_home=os.path.join(O, "sk"))
home = os.path.join(O, "sk", "lfw_home")
rows = [l.split() for l in open(os.path.join(home, "pairs.txt")).read().splitlines()[1:] if l.strip()]
path = lambda n, i: os.path.join(home, "lfw_funneled", n, f"{n}_{int(i):04d}.jpg")
pairs = [(path(r[0], r[1]), path(r[0], r[2]), 1) if len(r) == 3 else (path(r[0], r[1]), path(r[2], r[3]), 0) for r in rows]
y = np.array([p[2] for p in pairs]); imgs = sorted({p[k] for p in pairs for k in (0, 1)}); idx = {p: i for i, p in enumerate(imgs)}
crops = np.zeros((len(imgs), 1, 3, 112, 112), np.float32); ok_img = np.zeros(len(imgs), bool)
for i, p in enumerate(imgs):
    rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    dets, _ = detect("ort", rgb, 640)
    if not dets: continue
    box, sc, kp = min(dets, key=lambda d: np.hypot(d[0][0] + d[0][2] / 2 - 125, d[0][1] + d[0][3] / 2 - 125))   # 가운데 얼굴(LFW 규약)
    crops[i] = cv2.warpAffine(rgb, similarity_to_arc(np.array(kp, np.float64)), (112, 112), flags=cv2.INTER_LINEAR).astype(np.float32).transpose(2, 0, 1)[None]
    ok_img[i] = True
A = np.array([idx[p[0]] for p in pairs]); B = np.array([idx[p[1]] for p in pairs]); ok = ok_img[A] & ok_img[B]
folds = np.repeat(np.arange(10), 600)
res["lfw"] = {"images": len(imgs), "aligned": int(ok_img.sum()), "pairs_used": int(ok.sum())}
print(json.dumps(res["lfw"]), flush=True)


def evaluate(embed):
    E = np.zeros((len(imgs), 128), np.float32)
    for i in np.where(ok_img)[0]:
        v = embed(crops[i]).reshape(-1); E[i] = v / (np.linalg.norm(v) + 1e-12)
    sims = np.where(ok, (E[A] * E[B]).sum(1), -1.0); accs = []
    for k in range(10):
        tr = (folds != k) & ok; te = (folds == k) & ok
        th = np.quantile(sims[tr], np.linspace(0, 1, 2001)); best = max(th, key=lambda h: ((sims[tr] >= h) == y[tr]).mean())
        accs.append(((sims[te] >= best) == y[te]).mean())
    fpr, tpr, _ = roc_curve(y[ok], sims[ok])
    return E, {"acc10": round(float(np.mean(accs)) * 100, 2), "tar@far1e-3": round(float(np.interp(1e-3, fpr, tpr)), 4)}


E_ort, r = evaluate(lambda x: s_ort.run(None, {"data": x})[0]); r["ms"] = timeit(lambda: s_ort.run(None, {"data": crops[0]}))
res["sface_ort"] = r; print(json.dumps({"sface": "ort", **r}), flush=True)
for u in UNITS:
    try:
        m = ct.models.MLModel(os.path.join(M, "face_recognition_sface_2021dec.mlpackage"), compute_units=getattr(ct.ComputeUnit, u))
        E, r = evaluate(lambda x: np.asarray(m.predict({"data": x})["fc1"], np.float32))
        cos = (E[ok_img] * E_ort[ok_img]).sum(1)
        r.update({"cos_min": round(float(cos.min()), 6), "cos_mean": round(float(cos.mean()), 6), "ms": timeit(lambda: m.predict({"data": crops[0]}))})
    except Exception as e:
        r = {"error": str(e)[:300]}
    res[f"sface_{u}"] = r; print(json.dumps({"sface": u, **r}), flush=True)
json.dump(res, open(os.path.join(O, "face_coreml_intel.json"), "w"), indent=1)
