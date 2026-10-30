"""ReID(OSNet-AIN) 를 ORT 프로파일러로 돌려 연산자 종류별 시간 합 — Intel 만 느린 원인 찾기.
usage: python ort_profile.py <osnet.onnx> <out_dir>"""
import json, os, sys, glob, collections, platform
import numpy as np, onnxruntime as ort
P, O = sys.argv[1:3]; os.makedirs(O, exist_ok=True)
so = ort.SessionOptions(); so.intra_op_num_threads = 2; so.enable_profiling = True; so.profile_file_prefix = os.path.join(O, "prof")
s = ort.InferenceSession(P, so, providers=["CPUExecutionProvider"])
x = np.random.default_rng(0).standard_normal((1, 3, 256, 128)).astype(np.float32)
for _ in range(25): s.run(None, {"images": x})
f = s.end_profiling(); ev = json.load(open(f))
by = collections.Counter(); cnt = collections.Counter()
for e in ev:
    a = e.get("args", {})
    if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time"):
        by[a.get("op_name", "?")] += e["dur"]; cnt[a.get("op_name", "?")] += 1
tot = sum(by.values())
out = {"cpu": platform.processor(), "ort": ort.__version__, "total_ms_per_run": round(tot / 25 / 1000, 2),
       "ops": [{"op": k, "ms_per_run": round(v / 25 / 1000, 2), "pct": round(100 * v / tot, 1), "calls_per_run": cnt[k] // 25} for k, v in by.most_common(12)]}
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(O, "profile_summary.json"), "w"), indent=1)
