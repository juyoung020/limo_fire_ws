// 인식 벤치마크: 시뮬 정답(limo_fire_sim GtCamera 의 /gt/*)과 YOLO 결과를 장마다 짝지어 지표를 쌓고, 끝나면 표(마크다운)로 낸다.
//   장면 탐지: 상자·마스크 각각 P·R·F1(운용 문턱, IoU 0.5), mAP50·mAP50-95(문턱 없이 전부), 맞은 것의 평균 IoU — 클래스별
//   속도: 처리 장/초, 지연(카메라 영상 도착 → 탐지 도착, 벽시계) 평균·p50·p95, 처리한 장 비율
//   물체 지도: object_mapper 가 지도에 올린 물체 vs 월드 데이터 파일 정답 — P·R·F1, 위치 오차
// 짝짓기·AP 는 학습기와 같은 det_metrics.hpp. 마스크 정답이 부정확한 클래스(그림판인 불·연기)는 마스크 지표에서 뺀다.
// 순찰은 하지 않는다(로봇을 움직이는 쪽은 따로). 끝낼 때(Ctrl+C) 또는 ~/report 서비스로 표를 쓴다
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <yaml-cpp/yaml.h>

#include "limo_fire_perception/det_metrics.hpp"

namespace limo_fire_perception
{

namespace
{
using Clock = std::chrono::steady_clock;
using Dets = vision_msgs::msg::Detection2DArray;
using Image = sensor_msgs::msg::Image;

int64_t key(const builtin_interfaces::msg::Time & t) {return static_cast<int64_t>(t.sec) * 1000000000LL + t.nanosec;}

det_metrics::Box box_of(const vision_msgs::msg::BoundingBox2D & b)
{
  return {static_cast<float>(b.center.position.x - b.size_x / 2), static_cast<float>(b.center.position.y - b.size_y / 2),
    static_cast<float>(b.center.position.x + b.size_x / 2), static_cast<float>(b.center.position.y + b.size_y / 2)};
}

// 좌표처럼 음수도 그대로 쓰는 숫자
std::string num(double v, int prec = 1)
{
  std::ostringstream s;
  s.setf(std::ios::fixed);
  s.precision(prec);
  s << v;
  return s.str();
}

// 지표: 음수(정답 없음 등)는 "-"
std::string fmt(double v, int prec = 3)
{
  if (v < 0) {return "-";}
  std::ostringstream s;
  s.setf(std::ios::fixed);
  s.precision(prec);
  s << v;
  return s.str();
}
}  // namespace

class PerceptionBenchmarkNode : public rclcpp::Node
{
public:
  explicit PerceptionBenchmarkNode(const rclcpp::NodeOptions & options)
  : Node("perception_benchmark", options)
  {
    class_names_ = declare_parameter<std::vector<std::string>>("class_names");
    const int nc = static_cast<int>(class_names_.size());
    operating_conf_ = declare_parameter<double>("operating_conf", 0.35);
    world_file_ = declare_parameter<std::string>("world_file", "");
    report_dir_ = declare_parameter<std::string>("report_dir", "");
    map_match_radius_ = declare_parameter<double>("map_match_radius", 1.5);
    const auto mask_exclude = declare_parameter<std::vector<std::string>>("mask_exclude", {"fire", "smoke"});
    mask_classes_.assign(nc, true);
    for (const auto & n : mask_exclude) {
      const int c = class_index(n);
      if (c >= 0) {mask_classes_[c] = false;}
    }
    box_all_ = std::make_unique<det_metrics::ApEval>(nc);
    box_op_ = std::make_unique<det_metrics::ApEval>(nc);
    mask_all_ = std::make_unique<det_metrics::ApEval>(nc);
    mask_op_ = std::make_unique<det_metrics::ApEval>(nc);
    for (auto & e : size_op_) {e = std::make_unique<det_metrics::ApEval>(nc);}

    const auto sensor = rclcpp::SensorDataQoS();
    gt_det_sub_ = create_subscription<Dets>(declare_parameter<std::string>("gt_detections_topic", "/gt/detections"), 20,
        [this](Dets::ConstSharedPtr m) {std::lock_guard<std::mutex> l(mtx_); gt_det_[key(m->header.stamp)] = m; try_eval();});
    gt_mask_sub_ = create_subscription<Image>(declare_parameter<std::string>("gt_mask_topic", "/gt/mask"), 20,
        [this](Image::ConstSharedPtr m) {std::lock_guard<std::mutex> l(mtx_); gt_mask_[key(m->header.stamp)] = m; try_eval();});
    pred_mask_sub_ = create_subscription<Image>(declare_parameter<std::string>("mask_topic", "/yolo_detector/mask"), 20,
        [this](Image::ConstSharedPtr m) {std::lock_guard<std::mutex> l(mtx_); pred_mask_[key(m->header.stamp)] = m; try_eval();});
    pred_det_sub_ = create_subscription<Dets>(declare_parameter<std::string>("detections_topic", "/yolo_detector/detections"),
        20, [this](Dets::ConstSharedPtr m) {
          const auto now = Clock::now();
          std::lock_guard<std::mutex> l(mtx_);
          const int64_t k = key(m->header.stamp);
          pred_det_[k] = m;
          if (n_pred_++ == 0) {t_first_ = now;}
          t_last_ = now;
          const auto it = cam_arrival_.find(k);
          if (it != cam_arrival_.end()) {latency_ms_.push_back(std::chrono::duration<double, std::milli>(now - it->second).count());}
          try_eval();
        });
    cam_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      declare_parameter<std::string>("camera_info_topic", "/camera/color/camera_info"), sensor,
      [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr m) {
        std::lock_guard<std::mutex> l(mtx_);
        cam_arrival_[key(m->header.stamp)] = Clock::now();
        if (cam_arrival_.size() > 300) {cam_arrival_.erase(cam_arrival_.begin());}
      });
    rclcpp::QoS latched(1);
    latched.transient_local();
    marker_sub_ = create_subscription<visualization_msgs::msg::MarkerArray>(
      declare_parameter<std::string>("markers_topic", "/object_mapper/markers"), latched,
      [this](visualization_msgs::msg::MarkerArray::ConstSharedPtr m) {
        std::lock_guard<std::mutex> l(mtx_);
        for (const auto & mk : m->markers) {
          if (mk.type != visualization_msgs::msg::Marker::MESH_RESOURCE || class_index(mk.ns) < 0) {continue;}
          if (mk.action == visualization_msgs::msg::Marker::ADD) {
            icons_[{mk.ns, mk.id}] = {mk.pose.position.x, mk.pose.position.y};
          } else {
            icons_.erase({mk.ns, mk.id});
          }
        }
      });
    report_srv_ = create_service<std_srvs::srv::Trigger>("~/report",
        [this](std_srvs::srv::Trigger::Request::SharedPtr, std_srvs::srv::Trigger::Response::SharedPtr res) {
          res->message = write_report();
          res->success = true;
        });
    timer_ = create_wall_timer(std::chrono::seconds(10), [this]() {
          std::lock_guard<std::mutex> l(mtx_);
          RCLCPP_INFO(get_logger(), "평가한 장 %d (정답 장 %d), 탐지 %d 장", frames_, gt_frames_, n_pred_);
        });
    RCLCPP_INFO(get_logger(), "인식 벤치마크 시작: 운용 문턱 %.2f, 마스크 제외 %zu 클래스", operating_conf_,
      std::count(mask_classes_.begin(), mask_classes_.end(), false));
  }

  // 끌 때: 마지막 표 뒤로 새로 평가한 장이 있을 때만 쓴다 (~/report 로 쓴 직후 끄면 같은 표가 두 번 생기지 않게)
  ~PerceptionBenchmarkNode() override
  {
    if (frames_ == frames_at_report_) {return;}
    const std::string r = write_report();
    std::printf("%s\n", r.c_str());
  }

private:
  int class_index(const std::string & n) const
  {
    const auto it = std::find(class_names_.begin(), class_names_.end(), n);
    return it == class_names_.end() ? -1 : static_cast<int>(it - class_names_.begin());
  }

  // 정답(상자·번호 영상)과 탐지가 같은 시각에 다 모였으면 한 장을 평가한다. 탐지가 없는 장은 마스크가 안 온다
  void try_eval()
  {
    for (auto it = gt_det_.begin(); it != gt_det_.end(); ) {
      const int64_t k = it->first;
      const auto gm = gt_mask_.find(k);
      const auto pd = pred_det_.find(k);
      const bool newest_is_later = !pred_det_.empty() && pred_det_.rbegin()->first > k + 500000000LL;
      if (gm == gt_mask_.end() || pd == pred_det_.end()) {
        // 탐지기가 이 장을 건너뛰었으면(최신 장만 처리) 0.5 초 뒤 버린다
        if (newest_is_later) {gt_mask_.erase(k); ++gt_frames_; it = gt_det_.erase(it);} else {++it;}
        continue;
      }
      Image::ConstSharedPtr pm;
      if (!pd->second->detections.empty()) {
        const auto f = pred_mask_.find(k);
        if (f == pred_mask_.end()) {
          if (newest_is_later) {pm = nullptr;} else {++it; continue;}
        } else {
          pm = f->second;
        }
      }
      eval(*it->second, *gm->second, *pd->second, pm.get());
      ++gt_frames_;
      gt_mask_.erase(gm);
      pred_det_.erase(pd);
      pred_mask_.erase(k);
      it = gt_det_.erase(it);
    }
    // 정답이 안 오는 탐지·마스크는 오래되면 버린다
    auto drop_old = [](auto & m) {
        while (m.size() > 100) {m.erase(m.begin());}
      };
    drop_old(pred_det_);
    drop_old(pred_mask_);
    drop_old(gt_mask_);
  }

  void eval(const Dets & gt, const Image & gt_ids, const Dets & pred, const Image * pred_ids)
  {
    ++frames_;
    std::vector<int> gcls;
    for (const auto & g : gt.detections) {gcls.push_back(class_index(g.results.empty() ? "" : g.results[0].hypothesis.class_id));}
    struct P {det_metrics::Pred p; det_metrics::Box b; int idx;};
    std::vector<P> ps;
    for (size_t i = 0; i < pred.detections.size(); ++i) {
      const auto & d = pred.detections[i];
      if (d.results.empty()) {continue;}
      const int c = class_index(d.results[0].hypothesis.class_id);
      if (c < 0) {continue;}
      ps.push_back({{static_cast<float>(d.results[0].hypothesis.score), c}, box_of(d.bbox), static_cast<int>(i)});
    }

    // 마스크 교집합: 픽셀마다 (예측 번호, 정답 번호) 짝을 센다. 번호 = detections 순서 + 1 (0 = 배경)
    const size_t np = pred.detections.size() + 1, ng = gt.detections.size() + 1;
    std::vector<int> inter(np * ng, 0);
    const bool masks = pred_ids && pred_ids->width == gt_ids.width && pred_ids->height == gt_ids.height;
    if (masks) {
      const size_t n = static_cast<size_t>(gt_ids.width) * gt_ids.height;
      for (size_t i = 0; i < n; ++i) {
        const size_t p = pred_ids->data[i], g = gt_ids.data[i];
        if (p < np && g < ng) {inter[p * ng + g]++;}
      }
    }
    std::vector<int> parea(np, 0), garea(ng, 0);
    for (size_t p = 0; p < np; ++p) {
      for (size_t g = 0; g < ng; ++g) {parea[p] += inter[p * ng + g]; garea[g] += inter[p * ng + g];}
    }
    if (!masks) {  // 예측 마스크가 없으면 정답 넓이만 따로 센다
      std::fill(garea.begin(), garea.end(), 0);
      for (auto v : gt_ids.data) {if (v < ng) {garea[v]++;}}
    }

    auto feed = [&](det_metrics::ApEval & box_ev, det_metrics::ApEval & mask_ev, float min_conf) {
        std::vector<det_metrics::Pred> pp;
        std::vector<float> biou, miou;
        for (const auto & p : ps) {
          if (p.p.conf < min_conf) {continue;}
          pp.push_back(p.p);
          for (size_t g = 0; g < gt.detections.size(); ++g) {
            biou.push_back(det_metrics::box_iou(p.b, box_of(gt.detections[g].bbox)));
            const int in = inter[(p.idx + 1) * ng + g + 1];
            const int un = parea[p.idx + 1] + garea[g + 1] - in;
            miou.push_back(un > 0 ? static_cast<float>(in) / un : 0.f);
          }
        }
        box_ev.add(pp, gcls, biou);
        mask_ev.add(pp, gcls, miou);
      };
    feed(*box_all_, *mask_all_, 0.f);
    feed(*box_op_, *mask_op_, static_cast<float>(operating_conf_));

    // 크기별 재현율(COCO 처럼 상자 넓이 32² · 96² px 로 나눔): 그 크기 정답만 넣고 맞은 비율만 본다
    for (int b = 0; b < 3; ++b) {
      std::vector<int> gc;
      std::vector<size_t> gi;
      for (size_t g = 0; g < gt.detections.size(); ++g) {
        const double a = gt.detections[g].bbox.size_x * gt.detections[g].bbox.size_y;
        if ((a < 32 * 32 ? 0 : a < 96 * 96 ? 1 : 2) == b) {gc.push_back(gcls[g]); gi.push_back(g);}
      }
      std::vector<det_metrics::Pred> pp;
      std::vector<float> io;
      for (const auto & p : ps) {
        if (p.p.conf < operating_conf_) {continue;}
        pp.push_back(p.p);
        for (size_t g : gi) {io.push_back(det_metrics::box_iou(p.b, box_of(gt.detections[g].bbox)));}
      }
      size_op_[b]->add(pp, gc, io);
    }
  }

  // 지도 물체(map) ↔ 월드 정답(세계 좌표): 클래스 같고 map_match_radius 안에서 가까운 순 1:1
  std::string map_section()
  {
    if (world_file_.empty()) {return "";}
    YAML::Node w;
    try {
      w = YAML::LoadFile(world_file_);
    } catch (const std::exception & e) {
      return "\n물체 지도: 월드 데이터 파일을 못 읽음 (" + std::string(e.what()) + ")\n";
    }
    const double sx = w["robot_start"]["x"].as<double>(), sy = w["robot_start"]["y"].as<double>();
    const double yaw = w["robot_start"]["yaw"].as<double>();
    struct G {std::string name; int cls; double x, y;};
    std::vector<G> gts;
    for (const auto & o : w["objects"]) {
      gts.push_back({o["name"].as<std::string>(), o["cls"].as<int>(), o["center"][0].as<double>(), o["center"][1].as<double>()});
    }
    struct C {double d; size_t g; std::string ns; int id;};
    std::vector<C> cands;
    std::map<std::pair<std::string, int>, std::pair<double, double>> world_icons;
    for (const auto & [k, p] : icons_) {
      const double wx = sx + std::cos(yaw) * p.first - std::sin(yaw) * p.second;
      const double wy = sy + std::sin(yaw) * p.first + std::cos(yaw) * p.second;
      world_icons[k] = {wx, wy};
      for (size_t g = 0; g < gts.size(); ++g) {
        if (class_names_[gts[g].cls] != k.first) {continue;}
        const double d = std::hypot(gts[g].x - wx, gts[g].y - wy);
        if (d < map_match_radius_) {cands.push_back({d, g, k.first, k.second});}
      }
    }
    std::sort(cands.begin(), cands.end(), [](const C & a, const C & b) {return a.d < b.d;});
    std::vector<char> gused(gts.size(), 0);
    std::map<std::pair<std::string, int>, double> matched;
    for (const auto & c : cands) {
      if (gused[c.g] || matched.count({c.ns, c.id})) {continue;}
      gused[c.g] = 1;
      matched[{c.ns, c.id}] = c.d;
    }
    const int nc = static_cast<int>(class_names_.size());
    std::vector<int> ngt(nc, 0), nmap(nc, 0), ntp(nc, 0);
    std::vector<double> esum(nc, 0), emax(nc, 0);
    for (const auto & g : gts) {ngt[g.cls]++;}
    for (const auto & [k, p] : world_icons) {nmap[class_index(k.first)]++;}
    for (const auto & [k, d] : matched) {
      const int c = class_index(k.first);
      ntp[c]++;
      esum[c] += d;
      emax[c] = std::max(emax[c], d);
    }
    std::ostringstream s;
    s << "\n## 물체 지도 (object_mapper 가 지도에 올린 물체 vs 월드 정답, 같은 클래스 " << fmt(map_match_radius_, 1)
      << " m 안 1:1)\n\n| 클래스 | 정답 | 지도 | 맞음 | P | R | F1 | 위치 오차 평균 (m) | 최대 (m) |\n|---|---|---|---|---|---|---|---|---|\n";
    int tg = 0, tm = 0, tt = 0;
    double te = 0, tx = 0;
    for (int c = 0; c < nc; ++c) {
      const double p = nmap[c] ? static_cast<double>(ntp[c]) / nmap[c] : -1, r = ngt[c] ? static_cast<double>(ntp[c]) / ngt[c] : -1;
      const double f = p > 0 && r > 0 ? 2 * p * r / (p + r) : (ngt[c] || nmap[c] ? 0 : -1);
      s << "| " << class_names_[c] << " | " << ngt[c] << " | " << nmap[c] << " | " << ntp[c] << " | " << fmt(p, 2) << " | "
        << fmt(r, 2) << " | " << fmt(f, 2) << " | " << (ntp[c] ? fmt(esum[c] / ntp[c], 2) : "-") << " | "
        << (ntp[c] ? fmt(emax[c], 2) : "-") << " |\n";
      tg += ngt[c]; tm += nmap[c]; tt += ntp[c]; te += esum[c]; tx = std::max(tx, emax[c]);
    }
    const double P = tm ? static_cast<double>(tt) / tm : 0, R = tg ? static_cast<double>(tt) / tg : 0;
    s << "| **전체** | " << tg << " | " << tm << " | " << tt << " | " << fmt(P, 2) << " | " << fmt(R, 2) << " | "
      << fmt(P + R > 0 ? 2 * P * R / (P + R) : 0, 2) << " | " << (tt ? fmt(te / tt, 2) : "-") << " | " << fmt(tx, 2) << " |\n";
    std::vector<std::string> missed, wrong;
    for (size_t g = 0; g < gts.size(); ++g) {if (!gused[g]) {missed.push_back(gts[g].name);}}
    for (const auto & [k, p] : world_icons) {
      if (!matched.count(k)) {
        std::ostringstream o;
        o << k.first << " (" << num(p.first) << ", " << num(p.second) << ")";
        wrong.push_back(o.str());
      }
    }
    auto join = [](const std::vector<std::string> & v) {
        std::string r;
        for (const auto & x : v) {r += (r.empty() ? "" : ", ") + x;}
        return r.empty() ? std::string("없음") : r;
      };
    s << "\n- 놓친 것: " << join(missed) << "\n- 잘못·중복 올라간 것 (세계 좌표): " << join(wrong) << "\n";
    return s.str();
  }

  std::string write_report()
  {
    std::lock_guard<std::mutex> l(mtx_);
    const auto ba = box_all_->compute(), bo = box_op_->compute();
    const auto ma = mask_all_->compute(mask_classes_), mo = mask_op_->compute(mask_classes_);
    const std::time_t now = std::time(nullptr);
    char date[32];
    std::strftime(date, sizeof(date), "%Y-%m-%d %H:%M", std::localtime(&now));
    std::string world = world_file_.substr(world_file_.find_last_of('/') + 1);
    world = world.substr(0, world.find('.'));

    std::ostringstream s;
    s << "# 인식 벤치마크 — " << (world.empty() ? "?" : world) << " (" << date << ")\n\n"
      << "- 정답: Webots Recognition 분할(limo_fire_sim GtCamera), 가려진 부분 빼고 보이는 픽셀\n"
      << "- 평가한 장 " << frames_ << " / 정답 장 " << gt_frames_ << " (탐지기가 처리한 장만), 정답 등장 " << ba.ngt
      << " 번 (장마다 보인 물체, 30 픽셀 이상)\n"
      << "- mAP 는 탐지기가 낸 것 전부(문턱 0.01), P·R·F1·IoU 는 운용 문턱 conf ≥ " << fmt(operating_conf_, 2) << ", IoU ≥ 0.5\n"
      << "- 마스크 지표에서 뺀 클래스(정답이 그림판 사각형이라 부정확):";
    for (size_t c = 0; c < mask_classes_.size(); ++c) {if (!mask_classes_[c]) {s << " " << class_names_[c];}}
    s << "\n\n## 요약\n\n| | P | R | F1 | mAP50 | mAP50-95 | 평균 IoU |\n|---|---|---|---|---|---|---|\n"
      << "| 상자 | " << fmt(bo.precision()) << " | " << fmt(bo.recall()) << " | " << fmt(bo.f1()) << " | " << fmt(ba.map50) << " | "
      << fmt(ba.map) << " | " << fmt(bo.mean_iou) << " |\n"
      << "| 마스크 | " << fmt(mo.precision()) << " | " << fmt(mo.recall()) << " | " << fmt(mo.f1()) << " | " << fmt(ma.map50)
      << " | " << fmt(ma.map) << " | " << fmt(mo.mean_iou) << " |\n";

    s << "\n## 클래스별\n\n| 클래스 | 정답 | 상자 P | R | F1 | AP50 | AP50-95 | IoU | 마스크 AP50 | AP50-95 | IoU |\n"
      << "|---|---|---|---|---|---|---|---|---|---|---|\n";
    for (size_t c = 0; c < class_names_.size(); ++c) {
      const auto & o = bo.cls[c];
      const double p = o.npred ? static_cast<double>(o.tp) / o.npred : -1, r = o.ngt ? static_cast<double>(o.tp) / o.ngt : -1;
      const double f = p > 0 && r > 0 ? 2 * p * r / (p + r) : (o.ngt ? 0 : -1);
      const bool mk = mask_classes_[c];
      s << "| " << class_names_[c] << " | " << o.ngt << " | " << fmt(p) << " | " << fmt(r) << " | " << fmt(f) << " | "
        << fmt(ba.cls[c].ap50) << " | " << fmt(ba.cls[c].ap) << " | " << (o.tp ? fmt(o.mean_iou) : "-") << " | "
        << (mk ? fmt(ma.cls[c].ap50) : "제외") << " | " << (mk ? fmt(ma.cls[c].ap) : "제외") << " | "
        << (mk && mo.cls[c].tp ? fmt(mo.cls[c].mean_iou) : (mk ? "-" : "제외")) << " |\n";
    }

    s << "\n## 크기별 재현율 (상자, 운용 문턱) — 멀리서 작게 보인 것을 놓치는지\n\n| 클래스 | 작음 < 32² px | 중간 | 큼 > 96² px |\n|---|---|---|---|\n";
    std::array<det_metrics::ApEval::Result, 3> sz;
    for (int b = 0; b < 3; ++b) {sz[b] = size_op_[b]->compute();}
    auto rec = [](const det_metrics::ApEval::ClassStat & c) {
        return c.ngt ? fmt(static_cast<double>(c.tp) / c.ngt, 2) + " (" + std::to_string(c.ngt) + ")" : std::string("-");
      };
    for (size_t c = 0; c < class_names_.size(); ++c) {
      s << "| " << class_names_[c] << " | " << rec(sz[0].cls[c]) << " | " << rec(sz[1].cls[c]) << " | " << rec(sz[2].cls[c]) << " |\n";
    }
    s << "| **전체** | " << fmt(sz[0].recall(), 2) << " | " << fmt(sz[1].recall(), 2) << " | " << fmt(sz[2].recall(), 2) << " |\n"
      << "\n괄호 = 정답 등장 수\n";

    std::vector<double> lat = latency_ms_;
    std::sort(lat.begin(), lat.end());
    auto pct = [&](double q) {return lat.empty() ? -1.0 : lat[std::min(lat.size() - 1, static_cast<size_t>(q * lat.size()))];};
    double avg = 0;
    for (double v : lat) {avg += v;}
    avg = lat.empty() ? -1 : avg / lat.size();
    const double sec = std::chrono::duration<double>(t_last_ - t_first_).count();
    s << "\n## 속도 (벽시계)\n\n| 처리 장/초 | 지연 평균 (ms) | p50 | p95 | 처리한 장 / 정답 장 |\n|---|---|---|---|---|\n| "
      << fmt(sec > 0 ? (n_pred_ - 1) / sec : -1, 1) << " | " << fmt(avg, 1) << " | " << fmt(pct(0.5), 1) << " | "
      << fmt(pct(0.95), 1) << " | " << fmt(gt_frames_ ? static_cast<double>(frames_) / gt_frames_ : -1, 2) << " |\n"
      << "\n지연 = 카메라 영상(camera_info) 도착 → 탐지 결과 도착. 시뮬은 실시간보다 느려(약 0.7 배) 영상 간격이 실물보다 길다\n";

    s << map_section();

    const std::string text = s.str();
    frames_at_report_ = frames_;
    if (!report_dir_.empty()) {
      char stamp[32];
      std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M", std::localtime(&now));
      const std::string path = report_dir_ + "/" + stamp + "_" + (world.empty() ? "sim" : world) + ".md";
      std::ofstream(path) << text;
      RCLCPP_INFO(get_logger(), "벤치마크 표: %s", path.c_str());
    }
    return text;
  }

  std::vector<std::string> class_names_;
  std::vector<bool> mask_classes_;
  double operating_conf_, map_match_radius_;
  std::string world_file_, report_dir_;
  std::unique_ptr<det_metrics::ApEval> box_all_, box_op_, mask_all_, mask_op_;
  std::array<std::unique_ptr<det_metrics::ApEval>, 3> size_op_;  // 작음 · 중간 · 큼

  std::mutex mtx_;
  std::map<int64_t, Dets::ConstSharedPtr> gt_det_, pred_det_;
  std::map<int64_t, Image::ConstSharedPtr> gt_mask_, pred_mask_;
  std::map<int64_t, Clock::time_point> cam_arrival_;
  std::map<std::pair<std::string, int>, std::pair<double, double>> icons_;
  std::vector<double> latency_ms_;
  int frames_ = 0, gt_frames_ = 0, n_pred_ = 0, frames_at_report_ = 0;
  Clock::time_point t_first_, t_last_;

  rclcpp::Subscription<Dets>::SharedPtr gt_det_sub_, pred_det_sub_;
  rclcpp::Subscription<Image>::SharedPtr gt_mask_sub_, pred_mask_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_sub_;
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr marker_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr report_srv_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace limo_fire_perception

RCLCPP_COMPONENTS_REGISTER_NODE(limo_fire_perception::PerceptionBenchmarkNode)
