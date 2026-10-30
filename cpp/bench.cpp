// YOLOX 엔진 비교 — 제품과 같은 ncnn(20260113, YoloV5Focus 커스텀 레이어) 대 ORT 1.17.1 CPU. Linux·Windows 공용.
// usage: bench <ncnn|ort> <model_dir> <nano|tiny|s> <input.f32> <out.f32> <threads> <seconds>
#include <ncnn/net.h>
#include <ncnn/layer.h>
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
static double cpuMs() { FILETIME c, e, k, u; GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u);
    auto t = [](FILETIME f) { return (double)(((unsigned long long)f.dwHighDateTime << 32) | f.dwLowDateTime) / 1e4; }; return t(k) + t(u); }
#else
#include <sys/resource.h>
static double cpuMs() { rusage r; getrusage(RUSAGE_SELF, &r); return (r.ru_utime.tv_sec + r.ru_stime.tv_sec) * 1e3 + (r.ru_utime.tv_usec + r.ru_stime.tv_usec) / 1e3; }
#endif
using Clock = std::chrono::steady_clock;
static double msSince(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
class YoloV5Focus : public ncnn::Layer {   // 제품 YoloxObjectModel.cpp 와 같다
public:
    YoloV5Focus() { one_blob_only = true; }
    virtual int forward(const ncnn::Mat& b, ncnn::Mat& t, const ncnn::Option& opt) const {
        int w = b.w, h = b.h, ch = b.c, ow = w / 2, oh = h / 2, oc = ch * 4;
        t.create(ow, oh, oc, 4u, 1, opt.blob_allocator); if (t.empty()) return -100;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < oc; p++) { const float* ptr = b.channel(p % ch).row((p / ch) % 2) + ((p / ch) / 2); float* o = t.channel(p);
            for (int i = 0; i < oh; i++) { for (int j = 0; j < ow; j++) { *o++ = *ptr; ptr += 2; } ptr += w; } }
        return 0; }
};
DEFINE_LAYER_CREATOR(YoloV5Focus)
int main(int argc, char** argv) {
    std::string be = argv[1], dir = argv[2], model = argv[3]; int threads = atoi(argv[6]); double sec = atof(argv[7]);
    const int S = model == "s" ? 640 : 416;
    std::vector<float> in((size_t)3 * S * S); { std::ifstream f(argv[4], std::ios::binary); f.read((char*)in.data(), in.size() * 4); }
    std::function<void(std::vector<float>&)> run; ncnn::Net net; Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "b"); Ort::Session* sess = nullptr;
    if (be == "ncnn") {
        net.opt.num_threads = threads; net.opt.use_vulkan_compute = false;
#if defined(__aarch64__) || defined(_M_ARM64)
        net.opt.use_fp16_packed = net.opt.use_fp16_storage = true; net.opt.use_fp16_arithmetic = ncnn::cpu_support_arm_asimdhp();
#endif
        net.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
        std::string base = dir + "/" + (model == "s" ? std::string("yolox_s") : "yolox_" + model + "_v2");
        if (net.load_param((base + ".param").c_str()) || net.load_model((base + ".bin").c_str())) { printf("{\"error\":\"ncnn load\"}\n"); return 1; }
        run = [&](std::vector<float>& out) { ncnn::Mat m(S, S, 3); for (int c = 0; c < 3; c++) memcpy(m.channel(c), in.data() + (size_t)c * S * S, (size_t)S * S * 4);
            ncnn::Extractor ex = net.create_extractor(); ex.input("images", m); ncnn::Mat o; ex.extract("output", o);
            out.resize((size_t)o.w * o.h); for (int i = 0; i < o.h; i++) memcpy(out.data() + (size_t)i * o.w, o.row(i), o.w * 4); };
    } else {
        Ort::SessionOptions so; so.SetIntraOpNumThreads(threads); so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        std::string p = dir + "/yolox_" + model + ".onnx";
#ifdef _WIN32
        std::wstring wp(p.begin(), p.end()); sess = new Ort::Session(env, wp.c_str(), so);
#else
        sess = new Ort::Session(env, p.c_str(), so);
#endif
        run = [&](std::vector<float>& out) { auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault); int64_t sh[4] = { 1, 3, S, S };
            Ort::Value t = Ort::Value::CreateTensor<float>(mem, in.data(), in.size(), sh, 4); const char* i[] = { "images" }; const char* o[] = { "output" };
            auto r = sess->Run(Ort::RunOptions{ nullptr }, i, &t, 1, o, 1); const float* d = r[0].GetTensorData<float>(); out.assign(d, d + r[0].GetTensorTypeAndShapeInfo().GetElementCount()); };
    }
    std::vector<float> out; run(out); { std::ofstream f(argv[5], std::ios::binary); f.write((char*)out.data(), out.size() * 4); }
    for (int i = 0; i < 5; i++) run(out);
    std::vector<double> lat; double c0 = cpuMs(); auto ts = Clock::now();
    while (msSince(ts) < sec * 1000 || lat.size() < 10) { auto t = Clock::now(); run(out); lat.push_back(msSince(t)); }
    double cpu = (cpuMs() - c0) / lat.size(); std::sort(lat.begin(), lat.end());
    printf("{\"engine\":\"%s\",\"model\":\"yolox_%s\",\"threads\":%d,\"p50\":%.2f,\"p90\":%.2f,\"cpu_ms\":%.2f}\n", be.c_str(), model.c_str(), threads, lat[lat.size() / 2], lat[lat.size() * 9 / 10], cpu);
    delete sess; return 0;
}
