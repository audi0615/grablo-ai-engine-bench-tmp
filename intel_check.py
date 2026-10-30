"""
Intel 에서 ReID 가 ORT 1.17.1 로 비정상적으로 느린 원인 확인 + 인텔 맥 Core ML 실측.
  1) ORT 기본 대 ORT `session.set_denormal_as_zero=1`(공식 세션 설정 키) — 속도 · 코사인(기본 대비)
  2) (macOS) Core ML mlpackage — CPU_ONLY / ALL — 속도 · 코사인(ORT 기본 대비)
입력: 실제 사람 크롭이 아니라도 비교는 같은 입력끼리라 공정하다 — 여기서는 bus.jpg 를 잘라 쓴다.
usage: python intel_check.py <models_dir> <img> <out_dir>
"""
import json, os, platform, sys, time
import numpy as np, cv2
import onnxruntime as ort

M, IMG, O = sys.argv[1:4]
os.makedirs(O, exist_ok=True)
ort.set_default_logger_severity(3)


def timeit(fn, sec=3.0):
    for _ in range(3): fn()
    lat = []; t0 = time.perf_counter()
    while time.perf_counter() - t0 < sec or len(lat) < 10:
        t = time.perf_counter(); fn(); lat.append((time.perf_counter() - t) * 1e3)
    lat.sort(); return round(lat[len(lat) // 2], 2)


def cos(a, b):
    a = a.reshape(len(a), -1); b = b.reshape(len(b), -1)
    return float(((a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)).min())


im = cv2.cvtColor(cv2.imread(IMG), cv2.COLOR_BGR2RGB); h, w = im.shape[:2]
crops = [im[y:y + h // 2, x:x + w // 4] for y in (0, h // 2) for x in (0, w // 4, w // 2, 3 * w // 4)]
CASES = {
    "reid": ("osnet_ain_x1_0_msmt17", "images", 0, "features",
             np.stack([((cv2.resize(c, (128, 256)) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1) for c in crops]).astype(np.float32)),
    "siglip": ("siglip2_b32_256_vision", "pixel_values", 1, "pooler_output",
               np.stack([((cv2.resize(c, (256, 256)) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1) for c in crops]).astype(np.float32)),
}
res = {"cpu": platform.processor(), "machine": platform.machine(), "platform": platform.platform()}
try:
    import subprocess; res["cpu_brand"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() or open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0]
except Exception: pass
print(json.dumps(res), flush=True)

for key, (name, inp, oidx, cml_out, X) in CASES.items():
    def sess(daz):
        so = ort.SessionOptions(); so.intra_op_num_threads = 2
        if daz: so.add_session_config_entry("session.set_denormal_as_zero", "1")
        return ort.InferenceSession(os.path.join(M, name + ".onnx"), so, providers=["CPUExecutionProvider"])
    s0, s1 = sess(False), sess(True)
    ref = np.concatenate([s0.run(None, {inp: X[i:i + 1]})[oidx] for i in range(len(X))])
    daz = np.concatenate([s1.run(None, {inp: X[i:i + 1]})[oidx] for i in range(len(X))])
    r = {"model": key, "ort_ms": timeit(lambda: s0.run(None, {inp: X[0:1]})),
         "ort_daz_ms": timeit(lambda: s1.run(None, {inp: X[0:1]})), "ort_daz_cos": round(cos(ref, daz), 6)}
    pkg = os.path.join(M, name + ".mlpackage")
    if sys.platform == "darwin" and os.path.isdir(pkg):
        import coremltools as ct
        for cu in ("CPU_ONLY", "ALL"):
            try:
                m = ct.models.MLModel(pkg, compute_units=getattr(ct.ComputeUnit, cu))
                out = np.concatenate([m.predict({inp: X[i:i + 1]})[cml_out] for i in range(len(X))])
                r[f"coreml_{cu}_ms"] = timeit(lambda: m.predict({inp: X[0:1]}))
                r[f"coreml_{cu}_cos"] = round(cos(ref, out.astype(np.float32)), 6)
            except Exception as e:
                r[f"coreml_{cu}_error"] = str(e)[:200]
    print(json.dumps(r), flush=True)
    res[key] = r
json.dump(res, open(os.path.join(O, "intel_check.json"), "w"), indent=1)
