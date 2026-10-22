// SAM 3 자동 라벨: 사진 목록 → 프롬프트마다 상자·점수·마스크(288×288 RLE) → JSONL 한 줄에 사진 하나.
// 읽기·리사이즈는 CPU 스레드 여럿, 모델은 GPU 하나. 이미 쓴 사진은 건너뛴다(이어 하기).
//
// 사용: sam3_label --weights <sam3 폴더> --prompts prompts.json --list 목록.txt --out 결과.jsonl
//                  [--thr 0.3] [--threads 8] [--f32] [--no-masks] [--limit N]
// prompts.json: {"classes": [{"id": 0, "name": "fire", "prompts": ["fire", "flame"]}, ...]}
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <fstream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "prep.h"
#include "sam3.h"
#include "tok.h"

using namespace sam3;
using json = nlohmann::json;

namespace {
struct Args {
  std::string weights, prompts, list, out;
  float thr = 0.3f;
  int threads = 8, limit = -1;
  bool f32 = false, masks = true;
};
Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto v = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("값 없음: " + k);
      return argv[++i];
    };
    if (k == "--weights") a.weights = v();
    else if (k == "--prompts") a.prompts = v();
    else if (k == "--list") a.list = v();
    else if (k == "--out") a.out = v();
    else if (k == "--thr") a.thr = std::stof(v());
    else if (k == "--threads") a.threads = std::stoi(v());
    else if (k == "--limit") a.limit = std::stoi(v());
    else if (k == "--f32") a.f32 = true;
    else if (k == "--no-masks") a.masks = false;
    else throw std::runtime_error("모르는 인자: " + k);
  }
  if (a.weights.empty() || a.prompts.empty() || a.list.empty() || a.out.empty())
    throw std::runtime_error("--weights --prompts --list --out 이 필요");
  return a;
}

struct Slot {
  std::string path;
  int w = 0, h = 0;
  bool ok = false;
  float* px = nullptr;  // 고정 메모리 [3, 1008, 1008]
};

// 행 우선 이진 마스크 → RLE(0 부터 시작하는 길이 교대)
std::vector<int> rle(const std::vector<uint8_t>& m) {
  std::vector<int> c;
  uint8_t cur = 0;
  int run = 0;
  for (uint8_t v : m) {
    if (v != cur) {
      c.push_back(run);
      run = 0;
      cur = v;
    }
    ++run;
  }
  c.push_back(run);
  return c;
}

template <typename T>
int run(const Args& A) {
  // ---- 프롬프트 ----
  std::ifstream fp(A.prompts);
  auto pj = json::parse(fp);
  std::vector<std::string> ptxt;
  std::vector<int> pcls;
  for (auto& c : pj["classes"])
    for (auto& p : c["prompts"]) {
      ptxt.push_back(p.get<std::string>());
      pcls.push_back(c["id"].get<int>());
    }
  int P = (int)ptxt.size();
  ClipTok tok(A.weights + "/vocab.json", A.weights + "/merges.txt");
  std::vector<int> ids((size_t)P * TLEN), mask((size_t)P * TLEN);
  for (int b = 0; b < P; ++b) tok.encode(ptxt[b], TLEN, &ids[(size_t)b * TLEN], &mask[(size_t)b * TLEN]);

  // ---- 목록·이어 하기 ----
  std::vector<std::string> list;
  {
    std::ifstream fl(A.list);
    std::string l;
    while (std::getline(fl, l))
      if (!l.empty()) list.push_back(l);
  }
  std::set<std::string> done;
  {
    std::ifstream fo(A.out);
    std::string l;
    while (std::getline(fo, l)) {
      try {
        done.insert(json::parse(l)["img"].get<std::string>());
      } catch (...) {
      }
    }
  }
  std::vector<std::string> todo;
  for (auto& p : list)
    if (!done.count(p)) todo.push_back(p);
  if (A.limit >= 0 && (int)todo.size() > A.limit) todo.resize(A.limit);
  std::fprintf(stderr, "사진 %zu 장 중 이미 %zu 장, 이번 %zu 장, 프롬프트 %d 개\n", list.size(), done.size(),
               todo.size(), P);
  {
    json meta = {{"model", "facebook/sam3"},  {"precision", A.f32 ? "f32" : "f16"}, {"thr", A.thr},
                 {"mask_size", L0},           {"prompts", ptxt},                    {"prompt_class", pcls},
                 {"classes", pj["classes"]}, {"box", "xyxy 픽셀(원본)"},           {"mask", "288×288 행 우선 RLE"}};
    std::ofstream(A.out + ".meta.json") << meta.dump(2) << "\n";
  }
  if (todo.empty()) return 0;

  // ---- 모델 ----
  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  SafeTensors W(A.weights + "/model.safetensors");
  Model<T> m(W, P, s);
  m.encode_text(ids.data(), mask.data(), P);

  // ---- 읽기 스레드 → 큐 ----
  const int NS = std::max(4, A.threads * 2);
  std::vector<Slot> slots(NS);
  for (auto& sl : slots) CK(cudaMallocHost(&sl.px, (size_t)3 * IMG * IMG * 4));
  std::mutex mu;
  std::condition_variable cv_free, cv_ready;
  std::deque<int> free_q, ready_q;
  for (int i = 0; i < NS; ++i) free_q.push_back(i);
  std::atomic<size_t> next{0};
  std::atomic<int> readers_left{A.threads};
  std::vector<std::thread> readers;
  for (int t = 0; t < A.threads; ++t)
    readers.emplace_back([&]() {
      std::vector<uint8_t> rgb;
      for (;;) {
        size_t i = next++;
        if (i >= todo.size()) break;
        int si;
        {
          std::unique_lock<std::mutex> lk(mu);
          cv_free.wait(lk, [&] { return !free_q.empty(); });
          si = free_q.front();
          free_q.pop_front();
        }
        Slot& sl = slots[si];
        sl.path = todo[i];
        sl.ok = load_rgb(sl.path, rgb, sl.w, sl.h) && sl.w > 0 && sl.h > 0;
        if (sl.ok) prepare_sam3(rgb.data(), sl.w, sl.h, sl.px);
        {
          std::lock_guard<std::mutex> lk(mu);
          ready_q.push_back(si);
        }
        cv_ready.notify_one();
      }
      if (--readers_left == 0) cv_ready.notify_all();
    });

  // ---- GPU 반복 ----
  std::ofstream fo(A.out, std::ios::app);
  Buf<float> dpx, dmask;
  dpx.alloc((size_t)3 * IMG * IMG);
  dmask.alloc((size_t)NQ * L0 * L0);
  std::vector<float> lg((size_t)P * NQ), bx((size_t)P * NQ * 4), pr(P), mk;
  std::vector<uint8_t> bin((size_t)L0 * L0);
  auto t0 = std::chrono::steady_clock::now();
  size_t n = 0, ndet = 0;
  for (;;) {
    int si;
    {
      std::unique_lock<std::mutex> lk(mu);
      cv_ready.wait(lk, [&] { return !ready_q.empty() || readers_left == 0; });
      if (ready_q.empty()) break;
      si = ready_q.front();
      ready_q.pop_front();
    }
    Slot& sl = slots[si];
    json row = {{"img", sl.path}};
    if (!sl.ok) {
      row["err"] = "read";
    } else {
      CK(cudaMemcpyAsync(dpx.p, sl.px, dpx.n * 4, cudaMemcpyHostToDevice, s));
      m.encode_image(dpx.p);
      m.detect();
      CK(cudaMemcpyAsync(lg.data(), m.logits.p, lg.size() * 4, cudaMemcpyDeviceToHost, s));
      CK(cudaMemcpyAsync(bx.data(), m.boxes.p, bx.size() * 4, cudaMemcpyDeviceToHost, s));
      CK(cudaMemcpyAsync(pr.data(), m.presence.p, pr.size() * 4, cudaMemcpyDeviceToHost, s));
      CK(cudaStreamSynchronize(s));
      row["w"] = sl.w;
      row["h"] = sl.h;
      json det = json::array();
      auto sg = [](float x) { return 1.f / (1.f + std::exp(-x)); };
      for (int b = 0; b < P; ++b) {
        float ps = sg(pr[b]);
        std::vector<int> qs;
        std::vector<float> sc;
        for (int q = 0; q < NQ; ++q) {
          float v = sg(lg[(size_t)b * NQ + q]) * ps;
          if (v > A.thr) qs.push_back(q), sc.push_back(v);
        }
        if (qs.empty()) continue;
        if (A.masks) {
          m.masks_for(b, qs, dmask.p);
          mk.resize(qs.size() * (size_t)L0 * L0);
          CK(cudaMemcpy(mk.data(), dmask.p, mk.size() * 4, cudaMemcpyDeviceToHost));
        }
        for (size_t j = 0; j < qs.size(); ++j) {
          const float* r = &bx[((size_t)b * NQ + qs[j]) * 4];
          json d = {{"p", b},
                    {"c", pcls[b]},
                    {"s", std::round(sc[j] * 1e4f) / 1e4f},
                    {"b",
                     {std::round(r[0] * sl.w * 10) / 10, std::round(r[1] * sl.h * 10) / 10,
                      std::round(r[2] * sl.w * 10) / 10, std::round(r[3] * sl.h * 10) / 10}}};
          if (A.masks) {
            const float* ml = &mk[j * (size_t)L0 * L0];
            for (size_t u = 0; u < bin.size(); ++u) bin[u] = ml[u] > 0.f;
            d["m"] = rle(bin);
          }
          det.push_back(d);
          ++ndet;
        }
      }
      row["det"] = det;
    }
    fo << row.dump() << "\n";
    {
      std::lock_guard<std::mutex> lk(mu);
      free_q.push_back(si);
    }
    cv_free.notify_one();
    if (++n % 200 == 0) {
      fo.flush();
      double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::fprintf(stderr, "%zu / %zu 장, %.1f 장/초, 검출 %zu, 남은 시간 %.0f 분\n", n, todo.size(), n / sec, ndet,
                   (todo.size() - n) / (n / sec) / 60);
    }
  }
  for (auto& t : readers) t.join();
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::fprintf(stderr, "끝: %zu 장, %.1f 초, %.1f 장/초, 검출 %zu\n", n, sec, n / sec, ndet);
  for (auto& sl : slots) cudaFreeHost(sl.px);
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Args a = parse(argc, argv);
    return a.f32 ? run<float>(a) : run<f16>(a);
  } catch (std::exception& e) {
    std::fprintf(stderr, "오류: %s\n", e.what());
    return 1;
  }
}
