# 1대(1갈래) 대 4대(4갈래 동시) — 컨트롤러와 같은 세션 구성:
#   ort : 세션 하나를 N 스레드가 동시에 run (SharedOrtSession)
#   ov  : 갈래마다 compiled model + infer request 따로 (OpenVINOModel 인스턴스), LATENCY 힌트, inference_num_threads
# 입력은 모든 엔진 · 갈래 수에 같은 사진 한 장. 결과는 jsonl 한 줄씩.
# usage: python conc.py <models_dir> <image> <out.jsonl> <seconds> <rounds>
import json, os, sys, threading, time
import cv2, numpy as np, onnxruntime as ort, openvino as ov

M, IMG, OUT, SEC, ROUNDS = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5])
cores = os.cpu_count() or 1
lpr_threads = min(8 if cores > 8 else 4, cores)          # PlatformAIAccel.h getOptimalAIThreads
CASES = [  # family, file, threads, H, W, raw(0~255), ov f32
    ('object_detection', 'yolox_tiny.onnx', 2, 416, 416, True, False),
    ('lpr', 'lpr/detector.onnx', lpr_threads, 384, 384, False, False),
    ('person_reid', 'osnet_ain_x1_0_msmt17.onnx', 2, 256, 128, False, True),
    ('semantic_vision', 'siglip2_b32_256_vision.onnx', 2, 256, 256, False, True),
    ('face_embedding', 'face_recognition_sface_2021dec.onnx', 2, 112, 112, False, True),
]
img = cv2.imread(IMG)

def tensor(h, w, raw):
    x = cv2.resize(img, (w, h)).astype(np.float32)
    if not raw: x /= 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

def runners(eng, path, lanes, th, h, w, raw, f32):
    if eng == 'ort':
        so = ort.SessionOptions(); so.intra_op_num_threads = th
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        s = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
        i = s.get_inputs()[0]
        sh = i.shape
        if len(sh) == 4:
            h = sh[2] if isinstance(sh[2], int) and sh[2] > 0 else h
            w = sh[3] if isinstance(sh[3], int) and sh[3] > 0 else w
        out = []
        for _ in range(lanes):
            x = tensor(h, w, raw)
            out.append(lambda x=x: s.run(None, {i.name: x}))
        return out, h, w
    core = ov.Core()
    out = []
    for _ in range(lanes):
        m = core.read_model(path)
        ps = m.input().get_partial_shape()
        if ps.rank.is_static and len(ps) == 4:
            if ps[2].is_static: h = ps[2].get_length()
            if ps[3].is_static: w = ps[3].get_length()
        m.reshape([1, 3, h, w])
        cfg = {'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_NUM_THREADS': th}
        if f32: cfg['INFERENCE_PRECISION_HINT'] = 'f32'
        r = core.compile_model(m, 'CPU', cfg).create_infer_request()
        x = tensor(h, w, raw)
        out.append(lambda r=r, x=x: r.infer({0: x}))
    return out, h, w

def bench(fns):
    for f in fns:
        for _ in range(3): f()
    stop = False
    lat = [[] for _ in fns]
    def loop(k):
        while not stop:
            a = time.perf_counter(); fns[k](); lat[k].append((time.perf_counter() - a) * 1000)
    ts = [threading.Thread(target=loop, args=(k,)) for k in range(len(fns))]
    c0 = time.process_time(); t0 = time.perf_counter()
    for t in ts: t.start()
    time.sleep(SEC); stop = True
    for t in ts: t.join()
    wall = time.perf_counter() - t0; cpu = time.process_time() - c0
    a = sorted(x for l in lat for x in l)
    return dict(calls=len(a), p50=round(a[len(a) // 2], 2), p90=round(a[len(a) * 9 // 10], 2),
                per_s=round(len(a) / wall, 1), cpu_cores=round(cpu / wall, 2))

with open(OUT, 'w') as fo:
    for rnd in range(1, ROUNDS + 1):
        for fam, f, th, h, w, raw, f32 in CASES:
            p = os.path.join(M, f)
            if not os.path.exists(p): continue
            for lanes in (1, 4):
                for eng in ('ort', 'ov'):
                    try:
                        fns, H, W = runners(eng, p, lanes, th, h, w, raw, f32)
                        r = bench(fns)
                    except Exception as e:
                        r = dict(error=str(e)[:200]); H = W = 0
                    row = dict(round=rnd, family=fam, engine=eng, lanes=lanes, threads=th, H=H, W=W, **r)
                    print(json.dumps(row), flush=True); fo.write(json.dumps(row) + '\n')
