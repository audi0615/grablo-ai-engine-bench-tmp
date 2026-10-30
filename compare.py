# C++ 벤치(ncnn·ORT) 출력 대 ORT fp32 기준(bench.py 가 저장) — YOLOX 검출 일치
import json, sys, numpy as np
from bench_common import decode_yolox, match
out_dir, model, S, test = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
sc, W, H = json.load(open(f"{out_dir}/meta{S}.json"))
ref = decode_yolox(np.fromfile(f"{out_dir}/ref_yolox_{model}.f32", np.float32), S, sc, W, H)
tst = decode_yolox(np.fromfile(test, np.float32), S, sc, W, H)
print(json.dumps(match(ref, tst)))
