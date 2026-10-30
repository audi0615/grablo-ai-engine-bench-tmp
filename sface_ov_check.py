"""
SFace — ORT 1.17.1 대 OpenVINO CPU(f32 · 기본 정밀도) : 정확도(LFW 표준 6000쌍 10-fold) · ORT 대비 코사인 · 속도. AMD · Intel 러너용.
  정렬은 제품과 같다(YuNet ORT 5점 → arcface_dst 112, RGB 0~255 NCHW). 스레드 2(제품 getOptimalOrtVisionThreads).
usage: python sface_ov_check.py <models_dir> <out_dir>
"""
import json, os, platform, sys, time
import numpy as np, cv2, onnxruntime as ort, openvino as ov
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import roc_curve
from bench_common import similarity_to_arc

M, O = sys.argv[1:3]
os.makedirs(O, exist_ok=True)
ort.set_default_logger_severity(3)
so = ort.SessionOptions(); so.intra_op_num_threads = 2
yunet = ort.InferenceSession(os.path.join(M, "yunet_2026may.onnx"), so, providers=["CPUExecutionProvider"])
SP = os.path.join(M, "face_recognition_sface_2021dec.onnx")
s_ort = ort.InferenceSession(SP, so, providers=["CPUExecutionProvider"])
core = ov.Core()
res = {"cpu": core.get_property("CPU", "FULL_DEVICE_NAME"), "platform": platform.platform(), "openvino": ov.__version__, "ort": ort.__version__}
print(json.dumps(res), flush=True)


def ov_compile(prec):
    cfg = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": 2}
    if prec: cfg["INFERENCE_PRECISION_HINT"] = prec
    t0 = time.perf_counter(); c = core.compile_model(core.read_model(SP), "CPU", cfg)
    return c, round((time.perf_counter() - t0) * 1e3)


def kps640(rgb):
    h, w = rgb.shape[:2]; s = min(1.0, 640 / max(w, h)); iw, ih = round(w * s), round(h * s)
    sm = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    pw, ph = ((iw - 1) // 32 + 1) * 32, ((ih - 1) // 32 + 1) * 32
    b = np.zeros((1, 3, ph, pw), np.float32); b[0, :, :ih, :iw] = cv2.cvtColor(sm, cv2.COLOR_RGB2BGR).transpose(2, 0, 1)
    out = dict(zip([o.name for o in yunet.get_outputs()], yunet.run(None, {"input": b}))); best = None
    for st in (8, 16, 32):
        cols, rows = pw // st, ph // st; n = cols * rows
        sc = np.sqrt(np.clip(out[f"cls_{st}"].reshape(-1)[:n], 0, 1) * np.clip(out[f"obj_{st}"].reshape(-1)[:n], 0, 1))
        kp = out[f"kps_{st}"].reshape(-1, 10); bb = out[f"bbox_{st}"].reshape(-1, 4)
        for i in np.where(sc >= 0.3)[0]:
            c, r = i % cols, i // cols; cx, cy = (c + bb[i, 0]) * st / s, (r + bb[i, 1]) * st / s
            d = np.hypot(cx - w / 2, cy - h / 2)
            if best is None or d < best[0]: best = (d, np.array([((c + kp[i, 2 * j]) * st / s, (r + kp[i, 2 * j + 1]) * st / s) for j in range(5)]))
    return None if best is None else best[1]


fetch_lfw_pairs(subset="10_folds", resize=0.05, data_home=os.path.join(O, "sk"))
home = os.path.join(O, "sk", "lfw_home")
rows = [l.split() for l in open(os.path.join(home, "pairs.txt")).read().splitlines()[1:] if l.strip()]
path = lambda n, i: os.path.join(home, "lfw_funneled", n, f"{n}_{int(i):04d}.jpg")
pairs = [(path(r[0], r[1]), path(r[0], r[2]), 1) if len(r) == 3 else (path(r[0], r[1]), path(r[2], r[3]), 0) for r in rows]
y = np.array([p[2] for p in pairs]); imgs = sorted({p[k] for p in pairs for k in (0, 1)}); idx = {p: i for i, p in enumerate(imgs)}
A = np.array([idx[p[0]] for p in pairs]); B = np.array([idx[p[1]] for p in pairs]); folds = np.repeat(np.arange(10), 600)
X = np.zeros((len(imgs), 1, 3, 112, 112), np.float32); ok_img = np.zeros(len(imgs), bool)
for i, p in enumerate(imgs):
    rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB); k = kps640(rgb)
    if k is None: continue
    X[i] = cv2.warpAffine(rgb, similarity_to_arc(k), (112, 112), flags=cv2.INTER_LINEAR).astype(np.float32).transpose(2, 0, 1)[None]; ok_img[i] = True
ok = ok_img[A] & ok_img[B]


def evaluate(run):
    E = np.zeros((len(imgs), 128), np.float32)
    for i in np.where(ok_img)[0]:
        v = run(X[i]).reshape(-1); E[i] = v / (np.linalg.norm(v) + 1e-12)
    sims = np.where(ok, (E[A] * E[B]).sum(1), -1.0); accs = []
    for k in range(10):
        tr = (folds != k) & ok; te = (folds == k) & ok
        th = np.quantile(sims[tr], np.linspace(0, 1, 2001)); best = max(th, key=lambda h: ((sims[tr] >= h) == y[tr]).mean())
        accs.append(((sims[te] >= best) == y[te]).mean())
    fpr, tpr, _ = roc_curve(y[ok], sims[ok])
    return E, {"acc10": round(float(np.mean(accs)) * 100, 2), "tar@far1e-3": round(float(np.interp(1e-3, fpr, tpr)), 4)}


def timeit(fn, n=200):
    for _ in range(10): fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); t.append((time.perf_counter() - t0) * 1e3)
    t.sort(); return round(t[n // 2], 2)


E0, r = evaluate(lambda x: s_ort.run(None, {"data": x})[0]); r["ms"] = timeit(lambda: s_ort.run(None, {"data": X[0]}))
res["ort"] = r; print(json.dumps({"ort": r}), flush=True)
for prec in ("f32", ""):
    c, cms = ov_compile(prec); req = c.create_infer_request()
    E, r = evaluate(lambda x: req.infer({0: x})[c.outputs[0]])
    cos = (E[ok_img] * E0[ok_img]).sum(1)
    r.update({"cos_min": round(float(cos.min()), 6), "compile_ms": cms, "ms": timeit(lambda: req.infer({0: X[0]}))})
    res[f"ov_cpu_{prec or 'default'}"] = r; print(json.dumps({f"ov_cpu_{prec or 'default'}": r}), flush=True)
json.dump(res, open(os.path.join(O, "sface_ov_check.json"), "w"), indent=1)
