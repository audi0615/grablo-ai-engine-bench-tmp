#!/usr/bin/env bash
# 모델·이미지 받기 (공식 출처·CDN)
set -euo pipefail
mkdir -p models/lpr img
for m in nano tiny s; do curl -sSfL -o models/yolox_$m.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_$m.onnx; done
for f in yolox_nano_v2.param yolox_nano_v2.bin yolox_tiny_v2.param yolox_tiny_v2.bin yolox_s.param yolox_s.bin osnet_ain_x1_0_msmt17.onnx siglip2_b32_256_vision.onnx; do curl -sSfL -o models/$f https://downloads.grablo.co/assets/vision/$f; done
curl -sSfL -o lpr.tar https://downloads.grablo.co/assets/lpr/grablo-lpr-latin-v1.tar && tar -xf lpr.tar && cp grablo-lpr-latin-v1/*.onnx models/lpr/
curl -sSfL -o img/bus.jpg https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg
curl -sSfL -o img/dog.jpg https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/main/assets/dog.jpg
curl -sSfL -o img/zidane.jpg https://github.com/ultralytics/assets/releases/download/v0.0.0/zidane.jpg
curl -sSfL -o img/zz_plate.png https://raw.githubusercontent.com/ankandrew/fast-alpr/master/assets/test_image.png
ls -la models models/lpr img
