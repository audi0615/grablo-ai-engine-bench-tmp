"""
ORT 1.17.1(제품 판) 대 OpenVINO(Intel 공식) — 같은 ONNX, 같은 입력. 정확도는 ORT fp32 출력과 대조.
usage: python bench.py <models_dir> <img_dir> <out_dir> [seconds]
"""
import json, os, sys, time, platform, glob
import numpy as np, cv2
import onnxruntime as ort
CORE = None
OV_PREC = os.environ.get("OV_PREC", "")          # "" = OpenVINO 기본 정밀도, "f32" = 고정
SKIP_ORT = os.environ.get("SKIP_ORT") == "1"      # 두 번째 실행(정밀도만 바꿈)에서는 ORT 시간을 다시 재지 않는다
OV_NAME = "openvino" + ("_" + OV_PREC if OV_PREC else "")

M, I, O = sys.argv[1:4]
SEC = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
os.makedirs(O, exist_ok=True)
ort.set_default_logger_severity(3)
def timeit(fn):
    for _ in range(3): fn()
    lat = []; c0 = time.process_time(); t0 = time.perf_counter()
    while time.perf_counter() - t0 < SEC or len(lat) < 10:
        t = time.perf_counter(); fn(); lat.append((time.perf_counter() - t) * 1e3)
    lat.sort()
    return {"p50": round(lat[len(lat) // 2], 2), "p90": round(lat[len(lat) * 9 // 10], 2), "cpu_ms": round((time.process_time() - c0) * 1e3 / len(lat), 2)}


def ort_sess(path, threads):
    so = ort.SessionOptions(); so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def ov_model(path, threads, shapes=None):
    m = CORE.read_model(path)
    if shapes: m.reshape(shapes)
    cfg = {"PERFORMANCE_HINT": "LATENCY"}
    if threads: cfg["INFERENCE_NUM_THREADS"] = threads
    if OV_PREC: cfg["INFERENCE_PRECISION_HINT"] = OV_PREC   # 공식 속성 — 기본은 하드웨어가 정한다(ARM 은 f16)
    return CORE.compile_model(m, "CPU", cfg)


# ── 입력 ─────────────────────────────────────────────────────────────────────
def letterbox_yolox(bgr, S):
    h, w = bgr.shape[:2]; sc = min(S / w, S / h); nw, nh = round(w * sc), round(h * sc)
    pad = np.full((S, S, 3), 114, np.uint8); pad[:nh, :nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return pad.transpose(2, 0, 1)[None].astype(np.float32), sc


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


def main():
    global CORE
    try:
        import openvino as ov
        CORE = ov.Core()
    except Exception as e:  # Windows ARM64 등 휠이 없는 곳
        CORE = None
        print(json.dumps({"note": f"openvino unavailable: {e}"}))

    INFO = {"platform": platform.platform(), "machine": platform.machine(), "cpu": platform.processor(), "cores": os.cpu_count(),
            "ort": ort.__version__, "openvino": (ov.__version__ if CORE else None)}
    if CORE: INFO["ov_cpu"] = CORE.get_property("CPU", "FULL_DEVICE_NAME")
    print(json.dumps({"info": INFO}), flush=True)


    imgs = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(I, "*"))) if cv2.imread(p) is not None]
    results = []
    def emit(d):
        results.append(d); print(json.dumps(d), flush=True)

    # ── YOLOX (ORT 기준 출력도 여기서 만든다 — C++ ncnn 대조용으로 저장) ───────────────────────
    person_crops = []
    for m, S in (("nano", 416), ("tiny", 416), ("s", 640)):
        path = os.path.join(M, f"yolox_{m}.onnx"); ref_s = ort_sess(path, 4)
        x, sc = letterbox_yolox(imgs[0], S); x.tofile(os.path.join(O, f"in{S}.f32"))
        ref = ref_s.run(None, {"images": x})[0]; ref.astype(np.float32).tofile(os.path.join(O, f"ref_yolox_{m}.f32"))
        json.dump([sc, imgs[0].shape[1], imgs[0].shape[0]], open(os.path.join(O, f"meta{S}.json"), "w"))
        ref_d = decode_yolox(ref, S, sc, imgs[0].shape[1], imgs[0].shape[0])
        if m == "s":   # 사람 크롭(ReID·SigLIP 입력) — 모든 이미지에서
            for im in imgs:
                xi, sci = letterbox_yolox(im, S); di = decode_yolox(ref_s.run(None, {"images": xi})[0], S, sci, im.shape[1], im.shape[0])
                person_crops += [im[int(d[3]):int(d[5]), int(d[2]):int(d[4])] for d in di if d[1] == 0 and d[0] > 0.5 and d[4] - d[2] > 20 and d[5] - d[3] > 40]
        for th in (2, 4):
            s = ort_sess(path, th); SKIP_ORT or emit({"engine": "ort", "model": f"yolox_{m}", "threads": th, **timeit(lambda: s.run(None, {"images": x}))})
            if CORE:
                c = ov_model(path, th); r = c.create_infer_request(); o = r.infer({0: x})[0]
                emit({"engine": OV_NAME, "model": f"yolox_{m}", "threads": th, **timeit(lambda: r.infer({0: x})), "acc": match(ref_d, decode_yolox(o, S, sc, imgs[0].shape[1], imgs[0].shape[0]))})


    def embed_bench(name, path, X, inname, out_index, threads):
        s = ort_sess(path, threads)
        A = np.concatenate([s.run(None, {inname: X[i:i + 1]})[out_index] for i in range(len(X))])
        SKIP_ORT or emit({"engine": "ort", "model": name, "threads": threads, **timeit(lambda: s.run(None, {inname: X[0:1]}))})
        if CORE:
            c = ov_model(path, threads, {inname: list(X[0:1].shape)}); r = c.create_infer_request()
            C = np.concatenate([r.infer({inname: X[i:i + 1]})[c.outputs[out_index]] for i in range(len(X))])
            cos = (A * C).sum(1) / np.linalg.norm(A, axis=1) / np.linalg.norm(C, axis=1)
            emit({"engine": OV_NAME, "model": name, "threads": threads, **timeit(lambda: r.infer({inname: X[0:1]})), "acc": {"n": len(X), "cos_min": round(float(cos.min()), 6)}})


    rgb_crops = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in person_crops[:24]]
    emit({"note": f"person crops: {len(rgb_crops)}"})
    Xr = np.stack([((cv2.resize(c, (128, 256)) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1) for c in rgb_crops]).astype(np.float32)
    embed_bench("reid_osnet", os.path.join(M, "osnet_ain_x1_0_msmt17.onnx"), Xr, "images", 0, 2)
    Xs = np.stack([((cv2.resize(c, (256, 256)) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1) for c in rgb_crops]).astype(np.float32)
    embed_bench("siglip_vision", os.path.join(M, "siglip2_b32_256_vision.onnx"), Xs, "pixel_values", 1, 2)

    # ── LPR (detector 전체 그래프 · OCR) ─────────────────────────────────────────────
    plate = cv2.cvtColor(cv2.imread(os.path.join(I, "zz_plate.png")), cv2.COLOR_BGR2RGB)
    h, w = plate.shape[:2]; sc = min(384 / w, 384 / h); nw, nh = round(w * sc), round(h * sc); dw, dh = (384 - nw) / 2, (384 - nh) / 2
    xd = (cv2.copyMakeBorder(cv2.resize(plate, (nw, nh)), round(dh - .1), round(dh + .1), round(dw - .1), round(dw + .1), cv2.BORDER_CONSTANT, value=(114, 114, 114)).transpose(2, 0, 1)[None] / 255.0).astype(np.float32)
    for th in (4,):
        s = ort_sess(os.path.join(M, "lpr/detector.onnx"), th); rd = s.run(None, {"images": xd})[0]
        SKIP_ORT or emit({"engine": "ort", "model": "lpr_detector", "threads": th, **timeit(lambda: s.run(None, {"images": xd}))})
        r0 = rd[np.argmax(rd[:, 6])]; bx = [int(round((v - (round(dw - .1) if i % 2 == 0 else round(dh - .1))) / sc)) for i, v in enumerate(r0[1:5])]
        crop = cv2.resize(plate[bx[1]:bx[3], bx[0]:bx[2]], (128, 64))[None]
        so_ = ort_sess(os.path.join(M, "lpr/ocr.onnx"), th); po = so_.run(["plate"], {"input": crop})[0]
        SKIP_ORT or emit({"engine": "ort", "model": "lpr_ocr", "threads": th, **timeit(lambda: so_.run(["plate"], {"input": crop}))})
        if CORE:
            c = ov_model(os.path.join(M, "lpr/detector.onnx"), th); r = c.create_infer_request(); od = r.infer({0: xd})[0]
            emit({"engine": OV_NAME, "model": "lpr_detector", "threads": th, **timeit(lambda: r.infer({0: xd})),
                  "acc": {"rows_ref": int(len(rd)), "rows": int(len(od)), "max_box_diff": round(float(np.abs(od[np.argmax(od[:, 6])][1:5] - r0[1:5]).max()), 3) if len(od) else None}})
            c2 = ov_model(os.path.join(M, "lpr/ocr.onnx"), th, {"input": [1, 64, 128, 3]}); r2 = c2.create_infer_request(); pv = r2.infer({"input": crop})[c2.output("plate")]
            emit({"engine": OV_NAME, "model": "lpr_ocr", "threads": th, **timeit(lambda: r2.infer({"input": crop})),
                  "acc": {"same_text": bool((po.argmax(-1) == pv.argmax(-1)).all())}})

    json.dump({"info": INFO, "results": results}, open(os.path.join(O, "py_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
