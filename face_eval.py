"""
얼굴 인식 모델 비교 — LFW 표준 검증 6000쌍(10-fold), 원본 해상도 + CCTV 처럼 떨어뜨린 해상도.
  current  : 제품 mobilefacenet(ncnn, data→fc1, RGB, 그래프 안에서 (x-127.5)/128)
  sface    : OpenCV Zoo SFace(Apache-2.0) — OpenCV 공식 API(alignCrop · feature)
  auraface : fal/AuraFace-v1 glintr100(Apache-2.0) — InsightFace ArcFace 규약((x-127.5)/127.5, RGB)
정렬: 제품과 같은 YuNet 5점 → arcface_dst 닮음변환 112×112. 5점은 **원본 해상도에서 한 번** 재고
  해상도 조건마다 같은 점을 쓴다 — 인식 모델만 비교되도록.
usage: python face_eval.py <models_dir> <out_dir>
"""
import json, os, sys, time
import numpy as np, cv2
import onnxruntime as ort
import ncnn
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import roc_auc_score, roc_curve

M, O = sys.argv[1:3]
os.makedirs(O, exist_ok=True)
SCALES = [1.0, 0.4, 0.25]          # LFW 얼굴 ≈ 100px → 약 40px · 25px

ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)


def norm_crop(bgr, kps):
    m, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), ARCFACE_DST, method=cv2.LMEDS)
    return cv2.warpAffine(bgr, m, (112, 112), borderValue=0)


def degrade(bgr, s):
    if s == 1.0: return bgr
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def main():
    d = fetch_lfw_pairs(subset="10_folds", color=True, resize=1.0, slice_=None, data_home=os.path.join(O, "sk"))
    P = d.pairs; y = d.target.astype(int)
    if P.max() <= 1.5: P = P * 255
    P = P.astype(np.uint8)                       # (6000, 2, 250, 250, 3) RGB
    print(json.dumps({"pairs": int(len(y)), "same": int(y.sum())}), flush=True)

    det = cv2.FaceDetectorYN.create(os.path.join(M, "yunet_2023mar.onnx"), "", (250, 250), 0.6, 0.3, 5000)
    rec_sf = cv2.FaceRecognizerSF.create(os.path.join(M, "sface.onnx"), "")
    net = ncnn.Net(); net.opt.num_threads = 4
    net.load_param(os.path.join(M, "mobilefacenet.param")); net.load_model(os.path.join(M, "mobilefacenet.bin"))
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    aura = ort.InferenceSession(os.path.join(M, "auraface.onnx"), so, providers=["CPUExecutionProvider"])
    aura_in = aura.get_inputs()[0].name

    def detect(bgr):
        _, f = det.detect(bgr)
        if f is None or len(f) == 0: return None
        c = np.array([125, 125])
        f = min(f, key=lambda r: np.linalg.norm(r[0:2] + r[2:4] / 2 - c))   # 가운데 얼굴(LFW 규약)
        return f

    def emb_current(bgr, f):
        rgb = cv2.cvtColor(norm_crop(bgr, f[4:14].reshape(5, 2)), cv2.COLOR_BGR2RGB)
        ex = net.create_extractor(); ex.input("data", ncnn.Mat.from_pixels(rgb, ncnn.Mat.PixelType.PIXEL_RGB, 112, 112))
        _, out = ex.extract("fc1"); return np.array(out).reshape(-1)

    def emb_sface(bgr, f):
        return rec_sf.feature(rec_sf.alignCrop(bgr, f)).reshape(-1)

    def emb_aura(bgr, f):
        rgb = cv2.cvtColor(norm_crop(bgr, f[4:14].reshape(5, 2)), cv2.COLOR_BGR2RGB).astype(np.float32)
        x = ((rgb - 127.5) / 127.5).transpose(2, 0, 1)[None]
        return aura.run(None, {aura_in: x})[0].reshape(-1)

    MODELS = {"current_mobilefacenet": emb_current, "sface": emb_sface, "auraface": emb_aura}

    # 5점은 원본에서 한 번
    faces = np.zeros((len(y), 2, 15), np.float32); miss = 0
    for i in range(len(y)):
        for k in range(2):
            f = detect(cv2.cvtColor(P[i, k], cv2.COLOR_RGB2BGR))
            if f is None: miss += 1
            faces[i, k] = f if f is not None else np.nan
    print(json.dumps({"yunet_miss": miss}), flush=True)
    ok = ~np.isnan(faces).any(axis=(1, 2))

    folds = np.repeat(np.arange(10), 600)   # sklearn 10_folds 는 pairs.txt 순서 그대로 — 600쌍씩 한 fold
    results = []
    for s in SCALES:
        for name, fn in MODELS.items():
            sims = np.full(len(y), -1.0); t = []
            for i in np.where(ok)[0]:
                e = []
                for k in range(2):
                    bgr = degrade(cv2.cvtColor(P[i, k], cv2.COLOR_RGB2BGR), s)
                    f = faces[i, k].copy()
                    t0 = time.perf_counter(); v = fn(bgr, f); t.append(time.perf_counter() - t0)
                    e.append(v / (np.linalg.norm(v) + 1e-12))
                sims[i] = float(e[0] @ e[1])
            # 표준 10-fold: 9 fold 에서 정확도 최대 임계값 → 남은 fold 정확도
            accs = []
            for k in range(10):
                tr = (folds != k) & ok; te = (folds == k) & ok
                th = np.quantile(sims[tr], np.linspace(0, 1, 2001)); best = max(th, key=lambda h: ((sims[tr] >= h) == y[tr]).mean())
                accs.append(((sims[te] >= best) == y[te]).mean())
            fpr, tpr, _ = roc_curve(y[ok], sims[ok])
            tar = lambda far: float(np.interp(far, fpr, tpr))
            r = {"scale": s, "model": name, "acc10": round(float(np.mean(accs)) * 100, 2), "acc_std": round(float(np.std(accs)) * 100, 2),
                 "auc": round(float(roc_auc_score(y[ok], sims[ok])), 4), "tar@far1e-2": round(tar(1e-2), 4), "tar@far1e-3": round(tar(1e-3), 4),
                 "ms_per_face": round(float(np.median(t)) * 1e3, 2), "pairs": int(ok.sum())}
            results.append(r); print(json.dumps(r), flush=True)
    json.dump(results, open(os.path.join(O, "face_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
