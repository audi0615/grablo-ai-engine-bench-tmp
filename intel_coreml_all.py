"""
인텔 맥 — Core ML(ALL · CPU_AND_GPU) 대 ORT 1.17.1 CPU(제품 2스레드), 제품이 Core ML 로 돌리는 여섯 계열 전부(2026-11-03).
  YOLOX nano · tiny · s / LPR 검출 · OCR / ReID / SigLIP 이미지 / SFace / YuNet 480 고정판
  1) 한 장 지연 p50 · 정확도(검출은 디코드 뒤 1:1 대조, 임베딩은 코사인, OCR 은 글자 일치)
  2) 동시 처리량 — 프로세스 N개가 YOLOX tiny 를 SEC 초 동안 돌린 총 추론 수(카메라 N대 흉내)
usage: python intel_coreml_all.py <models_dir> <img_dir> <out_dir>
"""
import json, multiprocessing as mp, os, platform, subprocess, sys, time
import numpy as np, cv2, onnxruntime as ort, coremltools as ct
from bench_common import decode_yolox, match

M, I, O = sys.argv[1:4]
SEC = float(os.environ.get("SEC", "3"))
CSEC = float(os.environ.get("CSEC", "20"))
os.makedirs(O, exist_ok=True)
ort.set_default_logger_severity(3)
UNITS = ["ALL", "CPU_AND_GPU"]
RES = {"info": {}, "single": [], "concurrent": []}


def sh(cmd):
    try: return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e: return str(e)


def info():
    RES["info"] = {"platform": platform.platform(), "cpu_brand": sh("sysctl -n machdep.cpu.brand_string"),
                   "ncpu": os.cpu_count(), "mem": sh("sysctl -n hw.memsize"), "gpu": sh("system_profiler SPDisplaysDataType | grep -E 'Chipset|Metal|VRAM|Vendor'"),
                   "coremltools": ct.__version__, "ort": ort.__version__}
    print(json.dumps(RES["info"]), flush=True)


def emit(d):
    print(json.dumps(d), flush=True); RES["single"].append(d)


def timeit(fn):
    for _ in range(3): fn()
    lat = []; t0 = time.perf_counter()
    while time.perf_counter() - t0 < SEC or len(lat) < 10:
        t = time.perf_counter(); fn(); lat.append((time.perf_counter() - t) * 1e3)
    lat.sort(); return round(lat[len(lat) // 2], 2)


def ort_sess(path, th=2):
    so = ort.SessionOptions(); so.intra_op_num_threads = th
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def cml(path, cu):
    t0 = time.perf_counter(); m = ct.models.MLModel(path, compute_units=getattr(ct.ComputeUnit, cu))
    return m, round((time.perf_counter() - t0) * 1e3)


def cos(a, b):
    a = np.asarray(a, np.float32).reshape(-1); b = np.asarray(b, np.float32).reshape(-1)
    return round(float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)), 6)


def letterbox(bgr, S):
    h, w = bgr.shape[:2]; sc = min(S / w, S / h); nw, nh = round(w * sc), round(h * sc)
    pad = np.full((S, S, 3), 114, np.uint8); pad[:nh, :nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return pad.transpose(2, 0, 1)[None].astype(np.float32), sc


imgs = [cv2.imread(os.path.join(I, f)) for f in ("bus.jpg", "dog.jpg", "zidane.jpg")]


def guard(name, fn):
    try: fn()
    except Exception as e: emit({"model": name, "error": str(e)[:300]})


def run_single():
    # ── YOLOX ────────────────────────────────────────────────────────────────────
    def yolox(m, S):
        s = ort_sess(os.path.join(M, f"yolox_{m}.onnx"))
        xs = [letterbox(im, S) for im in imgs]
        ref = [decode_yolox(s.run(None, {"images": x})[0], S, sc, im.shape[1], im.shape[0]) for (x, sc), im in zip(xs, imgs)]
        emit({"model": f"yolox_{m}", "engine": "ort2t", "p50": timeit(lambda: s.run(None, {"images": xs[0][0]})), "dets": [len(r) for r in ref]})
        for cu in UNITS:
            c, load = cml(os.path.join(M, f"yolox_{m}.mlpackage"), cu)
            acc = [match(r, decode_yolox(c.predict({"images": x})["output"], S, sc, im.shape[1], im.shape[0])) for r, (x, sc), im in zip(ref, xs, imgs)]
            emit({"model": f"yolox_{m}", "engine": f"coreml_{cu}", "load_ms": load, "p50": timeit(lambda: c.predict({"images": xs[0][0]})), "acc": acc})


    for m, S in (("nano", 416), ("tiny", 416), ("s", 640)):
        guard(f"yolox_{m}", lambda: yolox(m, S))


    # ── 임베딩(ReID · SigLIP · SFace) ─────────────────────────────────────────────
    crops = [im[y:y + im.shape[0] // 2, x:x + im.shape[1] // 2] for im in imgs for y in (0, im.shape[0] // 2) for x in (0, im.shape[1] // 2)]
    rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]


    def embed(name, onnx, pkg, X, inname):
        s = ort_sess(os.path.join(M, onnx)); oname = None
        A = [s.run(None, {inname: X[i:i + 1]}) for i in range(len(X))]
        emit({"model": name, "engine": "ort2t", "p50": timeit(lambda: s.run(None, {inname: X[0:1]}))})
        for cu in UNITS:
            c, load = cml(os.path.join(M, pkg), cu)
            cin = c.get_spec().description.input[0].name
            outs = [c.predict({cin: X[i:i + 1]}) for i in range(len(X))]
            # ORT 출력 중 Core ML 출력과 모양이 같은 것끼리 코사인(이름이 달라도 된다)
            cs = []
            for a, o in zip(A, outs):
                best = None
                for v in o.values():
                    v = np.asarray(v)
                    for t in a:
                        if np.asarray(t).size == v.size:
                            cc = cos(t, v); best = cc if best is None else max(best, cc)
                cs.append(best)
            emit({"model": name, "engine": f"coreml_{cu}", "load_ms": load, "p50": timeit(lambda: c.predict({cin: X[0:1]})),
                  "cos_min": min(x for x in cs if x is not None) if any(x is not None for x in cs) else None, "n": len(X)})


    Xr = np.stack([((cv2.resize(c, (128, 256)) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1) for c in rgb]).astype(np.float32)
    guard("reid", lambda: embed("reid", "osnet_ain_x1_0_msmt17.onnx", "osnet_ain_x1_0_msmt17.mlpackage", Xr, "images"))
    Xs = np.stack([((cv2.resize(c, (256, 256)) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1) for c in rgb]).astype(np.float32)
    guard("siglip", lambda: embed("siglip", "siglip2_b32_256_vision.onnx", "siglip2_b32_256_vision.mlpackage", Xs, "pixel_values"))
    sf = ort_sess(os.path.join(M, "face_recognition_sface_2021dec.onnx"))
    Xf = np.stack([cv2.resize(c, (112, 112)).transpose(2, 0, 1) for c in crops]).astype(np.float32)   # SFace 는 BGR 0~255
    guard("sface", lambda: embed("sface", "face_recognition_sface_2021dec.onnx", "face_recognition_sface_2021dec.mlpackage", Xf, sf.get_inputs()[0].name))


    # ── YuNet 480 고정판(원시 출력 12개 코사인) ─────────────────────────────────────
    def yunet():
        s = ort_sess(os.path.join(M, "yunet_2026may.onnx")); inname = s.get_inputs()[0].name; names = [o.name for o in s.get_outputs()]
        X = [cv2.resize(im, (480, 480)).transpose(2, 0, 1)[None].astype(np.float32) for im in imgs]
        A = [dict(zip(names, s.run(None, {inname: x}))) for x in X]
        emit({"model": "yunet480", "engine": "ort2t", "p50": timeit(lambda: s.run(None, {inname: X[0]}))})
        for cu in UNITS:
            c, load = cml(os.path.join(M, "yunet_2026may_480.mlpackage"), cu)
            cin = c.get_spec().description.input[0].name
            B = [c.predict({cin: x}) for x in X]
            cs = [cos(a[k], b[k]) for a, b in zip(A, B) for k in names if k in b]
            emit({"model": "yunet480", "engine": f"coreml_{cu}", "load_ms": load, "p50": timeit(lambda: c.predict({cin: X[0]})),
                  "cos_min": min(cs) if cs else None, "outputs_compared": len(cs)})


    guard("yunet480", yunet)


    # ── LPR(검출 · OCR) ───────────────────────────────────────────────────────────
    def lpr():
        plate = cv2.cvtColor(cv2.imread(os.path.join(I, "zz_plate.png")), cv2.COLOR_BGR2RGB)
        h, w = plate.shape[:2]; sc = min(384 / w, 384 / h); nw, nh = round(w * sc), round(h * sc); dw, dh = (384 - nw) / 2, (384 - nh) / 2
        top, left = round(dh - .1), round(dw - .1)
        xd = (cv2.copyMakeBorder(cv2.resize(plate, (nw, nh)), top, round(dh + .1), left, round(dw + .1), cv2.BORDER_CONSTANT, value=(114, 114, 114)).transpose(2, 0, 1)[None] / 255.0).astype(np.float32)
        s = ort_sess(os.path.join(M, "lpr/detector.onnx"), 4); rd = s.run(None, {"images": xd})[0]; r0 = rd[np.argmax(rd[:, 6])]
        emit({"model": "lpr_detector", "engine": "ort4t", "p50": timeit(lambda: s.run(None, {"images": xd})), "top_box": [round(float(v), 1) for v in r0[1:5]]})
        bx = [int(round((v - (left if i % 2 == 0 else top)) / sc)) for i, v in enumerate(r0[1:5])]
        crop = cv2.resize(plate[bx[1]:bx[3], bx[0]:bx[2]], (128, 64))[None]
        so_ = ort_sess(os.path.join(M, "lpr/ocr.onnx"), 4); po = so_.run(["plate"], {"input": crop})[0]
        emit({"model": "lpr_ocr", "engine": "ort4t", "p50": timeit(lambda: so_.run(["plate"], {"input": crop}))})
        for cu in UNITS:
            c, load = cml(os.path.join(M, "lpr_cm/detector.mlpackage"), cu); o = c.predict({"images": xd})
            b = np.asarray(o["boxes"]).reshape(-1, 4); scr = np.asarray(o["scores"]).reshape(-1); top_b = b[int(np.argmax(scr))]
            emit({"model": "lpr_detector", "engine": f"coreml_{cu}", "load_ms": load, "p50": timeit(lambda: c.predict({"images": xd})),
                  "top_box": [round(float(v), 1) for v in top_b], "max_score": round(float(scr.max()), 3), "ort_score": round(float(r0[6]), 3)})
            c2, load2 = cml(os.path.join(M, "lpr_cm/ocr.mlpackage"), cu)
            cin = c2.get_spec().description.input[0]; ov = None
            for arr in (crop.astype(np.int32), crop.astype(np.float32), crop.astype(np.float16)):   # 변환판 입력 자료형에 맞는 것
                try: ov = c2.predict({cin.name: arr}); break
                except Exception: continue
            pv = np.asarray(ov.get("plate", list(ov.values())[0])).reshape(po.shape)
            emit({"model": "lpr_ocr", "engine": f"coreml_{cu}", "load_ms": load2, "p50": timeit(lambda: c2.predict({cin.name: arr})),
                  "same_text": bool((po.argmax(-1) == pv.argmax(-1)).all())})


    guard("lpr", lpr)



# ── 동시 처리량(YOLOX tiny, 프로세스 N개 — 카메라 N대 흉내) ─────────────────────────
def worker(kind, q, start):
    import onnxruntime as o2, coremltools as c2
    x, _ = letterbox(cv2.imread(os.path.join(I, "bus.jpg")), 416)
    if kind == "ort2t":
        so = o2.SessionOptions(); so.intra_op_num_threads = 2
        s = o2.InferenceSession(os.path.join(M, "yolox_tiny.onnx"), so, providers=["CPUExecutionProvider"]); run = lambda: s.run(None, {"images": x})
    else:
        m = c2.models.MLModel(os.path.join(M, "yolox_tiny.mlpackage"), compute_units=getattr(c2.ComputeUnit, kind.split("_", 1)[1])); run = lambda: m.predict({"images": x})
    for _ in range(3): run()
    start.wait(); n = 0; t0 = time.perf_counter()
    while time.perf_counter() - t0 < CSEC: run(); n += 1
    q.put(n)


if __name__ == "__main__":
    info()
    run_single()
    ctx = mp.get_context("spawn")
    for N in (4, 8, 12):
        for kind in ("ort2t", "coreml_ALL", "coreml_CPU_AND_GPU"):
            try:
                q = ctx.Queue(); start = ctx.Event(); ps = [ctx.Process(target=worker, args=(kind, q, start)) for _ in range(N)]
                for p in ps: p.start()
                time.sleep(8 + N); start.set()
                tot = [q.get(timeout=CSEC + 120) for _ in ps]
                for p in ps: p.join(30)
                d = {"concurrent": N, "engine": kind, "total_fps": round(sum(tot) / CSEC, 2), "min_per_proc_fps": round(min(tot) / CSEC, 2)}
            except Exception as e:
                d = {"concurrent": N, "engine": kind, "error": str(e)[:200]}
            print(json.dumps(d), flush=True); RES["concurrent"].append(d)
    json.dump(RES, open(os.path.join(O, "intel_coreml_all.json"), "w"), indent=1)
