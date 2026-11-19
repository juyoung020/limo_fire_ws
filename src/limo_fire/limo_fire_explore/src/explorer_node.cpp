// 합친 탐사 실물 노드 /explore/explorer (limo_fire_explore/README "합친 탐사 — explorer"): 위층 판단 모델이 로봇 자기 지도에서 목표를 고르고, 아래층 규칙 추종기
// (로봇 지도 A* plan_ref + path_step_* 추종기 — RobotEnv::low_act 계층형 가지)가 몬다. 2026-11-16 아래층 강화학습을 그만둠 (archive/rl/README.md).
// /scan 한 장 = 시뮬 걸음 하나(0.1 s). 시뮬 low_explore_check --upper decider 와 같은 코드·같은 값 (판단 모델이 못 답할 때 대신 고르는 규칙은 --upper cand):
//   관측·지도·판 끝      RobotEnv 합친 탐사 모드 (sim 헤더 그대로 — env_ref.cpp step_env + low_prepare_env, combine_run 설정)
//   위층 결정·목표 넘기기  scripts/low_explore_ctl.h (combine_run 의 환경 하나 몫을 옮긴 것, 규칙 가지)
// 걸음 차례 (장 k): step(감각, 장 k) → 판 끝·닿음 → 위층 걸음 뒤 → 깊이 → [결정 차례면 위층 결정 → set_cmd] → set_slot · low_prepare → low_act → 명령
// 명령을 0 으로 만드는 멈춤 조건:
//   꺼짐(enable false · 처음) · 리모컨이 오토가 아님(/rc_state buttons[rc_button] ≠ rc_auto_value, 또는 rc_timeout 넘게 안 옴 — 꺼짐으로 바뀜, 다시 오토가 돼도
//   enable true 를 따로 불러야 감) · 입력 끊김(라이다·TF·오도메트리 input_timeout 넘게 — 0, 돌아오면 이어 감) · 닿음(시뮬 contact.h sense_step 정의 —
//   멈추고 꺼짐, contact_resume_s > 0 이면 그만큼 뒤 저절로 켬) · 끝(지도 닫힘 + 진척 없음 10 s · 진척 없음 · 시간 — 시뮬 판 끝 규칙).
// 위층 목표가 없는 걸음(판 시작·프론티어 없음)은 시뮬과 같게 규칙 추종기의 제자리 회전.
// dry_run:=true (기본): /cmd_vel 발행자를 아예 안 만들고 같은 명령을 cmd_debug 에만.
// 컴포넌트: app.launch.xml 이 인식(object_detector · object_map)과 한 프로세스 · 한 스레드 실행기에 올린다 (docs/runtime_io.md).
//   늦게 뜬 쪽도 받아야 하는 토픽(transient_local)만 프로세스 안 전달을 끔.
// 지시 "○○ 로 가": /explore/goal 글 하나 (빈 글 = 지움) — 별칭으로 클래스를 풀어(class_alias.h) 판단 모델에 힌트로, 진행은 /explore/goal_status.
// debug:=false 면 RViz 용 표시 토픽(candidates · candidates_text · decider_image · goal_marker)과 cmd_debug(dry run 이 아닐 때)를 안 만든다.
// 위층: 로봇 안 llama-server 판단 모델 (scripts/low_decider.h — 사건 때 비동기 요청, 기다리는 동안 지금 목표로 몰고, 답이 오면 바꿈). 후보는 scripts/cand_view.h,
//   답이 없거나 잇따라 틀리면 규칙(Ctl::cand_pick — 후보 0 번, 지금 목표 붙듦)이 대신. candidates 에 후보 그림 (결정마다).
//   fuel · slot0 · cand 기준선 위층은 시뮬 도구(low_explore_check · combine_run)에만 남음 (2026-11-18 노드에서 뺌).
//   지시(instruction)가 있으면 물체 지도의 물체 · 보이는 사람도 후보 — 물체로 가기(scripts/task_sup.h) · 사람 따라가기(scripts/obj_follow.h)는 노드가 목표를 쥠.
#include <chrono>
#include <deque>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <limo_fire_perception/shared_image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <png.h>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include <rviz_2d_overlay_msgs/msg/overlay_text.hpp>

#include "limo_fire_explore/class_alias.h"
#include "limo_fire_explore/decider_client.h"
#include "limo_fire_explore/rc_watch.h"
#include "limo_fire_explore/robot_env.h"
#include "low_explore_ctl.h"
#include "low_node_rec.h"
#include "obj_follow.h"
#include "task_sup.h"

namespace limo_fire_explore {

class ExplorerNode : public rclcpp::Node {
 public:
  explicit ExplorerNode(const rclcpp::NodeOptions& options) : Node("explorer", options) {
    env_yaml_ = declare_parameter<std::string>("env_yaml", LIMO_FIRE_EXPLORE_ENV_YAML);
    dry_run_ = declare_parameter<bool>("dry_run", true);
    enabled_ = declare_parameter<bool>("enabled", false);
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    tf_timeout_ = declare_parameter<double>("tf_timeout", 0.2);
    depth_timeout_ = declare_parameter<double>("depth_timeout", 0.5);
    input_timeout_ = declare_parameter<double>("input_timeout", 0.3);
    rc_timeout_ = declare_parameter<double>("rc_timeout", 0.5);
    rc_button_ = (int)declare_parameter<int>("rc_button", 1);
    rc_auto_ = (int)declare_parameter<int>("rc_auto_value", 2);
    require_rc_ = declare_parameter<bool>("require_rc", true);
    rc_resub_s_ = declare_parameter<double>("rc_resubscribe_s", 2.0);   // 발행자가 있는데 이만큼 못 받으면 /rc_state 구독을 새로 만듦 (rc_watch.h)
    contact_resume_s_ = declare_parameter<double>("contact_resume_s", 0.0);
    max_speed_ = declare_parameter<double>("max_speed", -1.0);
    record_ = declare_parameter<std::string>("record", "");
    debug_ = declare_parameter<bool>("debug", false);   // RViz 표시 토픽 (끄면 안 만듦 — 판단·명령은 같음)
    aliases_ = parse_aliases(declare_parameter<std::vector<std::string>>("aliases", std::vector<std::string>{}));
    const bool camera_layer = declare_parameter<bool>("camera_layer", true);
    const std::string scan_topic = declare_parameter<std::string>("scan_topic", "/scan");
    const std::string odom_topic = declare_parameter<std::string>("odom_topic", "/odom");
    const std::string imu_topic = declare_parameter<std::string>("imu_topic", "/imu");
    const std::string depth_topic = declare_parameter<std::string>("depth_topic", "/camera/depth/image_raw");
    const std::string info_topic = declare_parameter<std::string>("depth_info_topic", "/camera/depth/camera_info");
    const std::string cmd_topic = declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");
    const std::string rc_topic = declare_parameter<std::string>("rc_topic", "/rc_state");
    // 닿음 정의 (학습 run.json con_cfg 와 같게 — 시뮬 env_cfg.cpp SIM_CON_* 기본값 위에 덮음. 음수 = env.yaml·환경변수 그대로)
    const int con_mask = (int)declare_parameter<int>("con_mask", -1);
    const int con_stall = (int)declare_parameter<int>("con_stall", -1);
    const double con_acc = declare_parameter<double>("con_acc", -1.0);
    // 위층 판단 모델 (scripts/low_decider.h). 사건 때만 물음 (event: 목표에 다가감 · 실패 · 후보 크게 바뀜 · 최대 간격), 지도 사진 + 후보 글
    const std::string dec_url = declare_parameter<std::string>("decider_url", "http://127.0.0.1:8171");
    const double dec_timeout = declare_parameter<double>("decider_timeout", 8.0);
    lxd::Params dp;
    dp.px = (int)declare_parameter<int>("decider_px", dp.px);   // 기본은 scripts/low_decider.h Params (시뮬·라벨과 같은 값)
    dp.compact = declare_parameter<bool>("decider_compact", dp.compact);   // 짧은 꼴 (미세조정 v1 꼴 · 로봇 약 2 s)
    dp.fmt = (int)declare_parameter<int>("decider_fmt", dp.fmt);           // 요청 꼴 1 = state JSON 객체 (v1) · 2 = state 글 하나 (짧은 표 — 토큰 적고 앞부분 캐시)
    dp.stop_cand = declare_parameter<bool>("decider_stop_cand", dp.stop_cand);   // 지시가 있을 때 멈춤 후보 (고르면 멈춤)
    dp.input = dec::kBoth;
    dp.match_r = (float)declare_parameter<double>("decider_match_r", 1.5);
    dp.hint = declare_parameter<bool>("decider_hint", true);          // 글의 "Rules of thumb"
    dp.mem_image = declare_parameter<bool>("decider_mem_image", true);   // 지도 사진에 지나온 길 · 지난 목표
    dp.image = declare_parameter<std::string>("decider_image", "A") == "B" ? 1 : 0;
    dp.max_cands = (int)declare_parameter<int>("decider_max_cands", dp.max_cands);   // 요청에 넣는 후보 수 상한 (0 = 다)
    dp.log_k = (int)declare_parameter<int>("decider_log_k", dp.log_k);   // 결정 기록 최근 K 개 (글 + 지도 번호), 0 = 끔
    hc_ = sim::load_env_cfg(env_yaml_);
    if (!camera_layer) hc_.cfg.cam_on = 0;
    if (con_mask >= 0) hc_.cfg.con_mask = con_mask;
    if (con_stall >= 0) hc_.cfg.con_stall = con_stall;
    if (con_acc >= 0.0) hc_.cfg.con_acc = (float)con_acc;
    if (dry_run_ && (hc_.cfg.con_mask & sim::kConStall)) {   // dry run: 명령이 안 나가 로봇이 안 움직이므로 "명령은 가라는데 바퀴가 섬"(멈춤 신호)은 늘 참 — 빼고 셈
      hc_.cfg.con_mask &= ~sim::kConStall;
      RCLCPP_WARN(get_logger(), "dry run: 닿음 신호에서 멈춤(명령 대비 바퀴 정지)을 뺌 — 명령이 실제로 안 나가서");
    }
    env_ = std::make_unique<RobotEnv>(hc_);   // 아래층 + 합친 탐사 모드
    const sim::EnvCfg& c = env_->cfg();
    if (!(max_speed_ > 0.0)) max_speed_ = c.act_v_fwd;   // 기본 = 시뮬 한도 (자르지 않음)
    lp_ = lxc::Params::from_cfg(c);
    lp_.use_cand = true;    // 위층 후보 scripts/cand_view.h (판단 모델 후보 · 대신 고르는 규칙은 그 0 번) — low_explore_check --upper decider|cand
    lp_.use_fuel = false;
    // 걸음마다 관측에서 쓰는 것만 (RobotEnv::set_obs): 계층형 후보 8 칸은 노드가 안 씀 (low_explore_check --node-obs 가 같은 설정으로 비교).
    // 걸음 사이 프론티어 칸 판정 기억은 RobotEnv 기본 켬 (값 같음 — low_explore_check --fc-check 가 처음부터 다시 센 값과 견줌)
    env_->set_obs(false);
    ctl_ = std::make_unique<lxc::Ctl>(lp_);
    {
      dp.mode = 1;   // event (scripts/low_decider.h — periodic 은 시뮬 비교에서 덮음이 같고 요청만 많아 실물에서 안 씀)
      dp.hold_extra = (int)declare_parameter<int>("decider_hold_extra", dp.hold_extra);      // 목표가 죽었는데 답이 오는 중이면 (지연 어림 + 이것) 걸음까지 기다림 (−1 = 바로 규칙)
      dp.ask_lead_s = (float)declare_parameter<double>("decider_ask_lead_s", dp.ask_lead_s);  // event: 남은 시간 ≤ 답 지연 + 이것 (s) 이면 미리 물음 (−1 = 끔)
      dp.obj_max = (int)declare_parameter<int>("decider_obj_max", dp.obj_max);              // 물체 후보 칸 (따라가기 포함, 합쳐 decider_max_cands)
      dp.obj_always = declare_parameter<bool>("decider_obj_always", dp.obj_always);          // 지시 없어도 물체 후보 (끄면 지시 없는 탐사 요청이 예전과 바이트 같음)
      dp.vis_s = (float)declare_parameter<double>("decider_vis_s", dp.vis_s);               // 이 안에 본 물체 = 지금 보임
      dp.sem_max = (int)declare_parameter<int>("decider_sem_max", dp.sem_max);              // 뜻 물체 칸 (0 = 끔 — 요청이 예전과 바이트 같음)
      dp.sem_thr = (float)declare_parameter<double>("decider_sem_thr", dp.sem_thr);         // 뜻 물체 칸 문턱 (맞음 확률)
      dp.front_max_instr = (int)declare_parameter<int>("decider_front_max_instr", dp.front_max_instr);   // 지시 있을 때 프론티어 칸 상한 (0 = 예전)
      dp.obj_fill = declare_parameter<bool>("decider_obj_fill", dp.obj_fill);               // 이름 물체 칸을 클래스 안 맞는 가까운 물체로도 채움
      dp.sem_first = declare_parameter<bool>("decider_sem_first", dp.sem_first);            // 뜻 물체 칸을 이름 물체 칸 앞 글자에
      dp.timeout_steps = std::max(1, (int)std::lround(dec_timeout / dt()));
      dec_up_ = std::make_unique<lxd::Upper>(dp);
      dec_obj_always_ = dp.obj_always;
      {   // 사람 따라가기 (scripts/obj_follow.h — 판단 모델이 kind follow 후보를 고르면)
        ofl::Params fp;
        fp.dist = (float)declare_parameter<double>("follow_dist", fp.dist);
        fp.stop = (float)declare_parameter<double>("follow_stop", fp.stop);
        fp.hyst = (float)declare_parameter<double>("follow_hyst", fp.hyst);
        fp.replan_s = (float)declare_parameter<double>("follow_replan_s", fp.replan_s);
        fp.gate = (float)declare_parameter<double>("follow_gate", fp.gate);
        fp.gate_v = (float)declare_parameter<double>("follow_gate_v", fp.gate_v);
        fp.lost_s = (float)declare_parameter<double>("follow_lost_s", fp.lost_s);
        fp.search_s = (float)declare_parameter<double>("follow_search_s", fp.search_s);
        fp.search_w = (float)declare_parameter<double>("follow_search_w", fp.search_w);
        fp.v_max = (float)declare_parameter<double>("follow_max_speed", fp.v_max);
        fp.reask_m = (float)declare_parameter<double>("follow_reask_m", fp.reask_m);
        fp.clear = (float)declare_parameter<double>("follow_clear", fp.clear);
        follower_ = ofl::Follower(fp);
        follow_classes_ = declare_parameter<std::vector<std::string>>("follow_classes", std::vector<std::string>{"person"});
      }
      dec_vis_s_ = dp.vis_s;
      dec_cli_ = std::make_unique<DeciderClient>(dec_url, (int)std::lround(dec_timeout * 1000.0));
      RCLCPP_INFO(get_logger(), "위층 판단 모델 %s · 사건 때 물음 · 후보 최대 %d · 한도 %.1f s — 기다리는 동안 지금 목표로, 목표가 죽으면 답을 기다리고 답이 잇따라 틀리면 규칙이 대신",
                  dec_url.c_str(), dp.max_cands, dp.timeout_steps * dt());
    }
    RCLCPP_INFO(get_logger(),
                "아래층 규칙 추종기 (A* + path_step) · 위층 판단 모델 · 결정 %d 걸음 · 피하기 %.2f m %.0f s · 닿음 마스크 %d "
                "멈춤 %d 걸음 가속 %.1f m/s² · 최고 속도 %.2f m/s%s — %s",
                lp_.decide_steps,
                lp_.bl_r, lp_.bl_steps * dt(), c.con_mask, c.con_stall, c.con_acc, max_speed_, max_speed_ < c.act_v_fwd ? " (시뮬 한도보다 낮춤)" : "",
                dry_run_ ? "dry run: /cmd_vel 안 냄, /explore/cmd_debug 에만" : "실제 주행: /cmd_vel 냄");
    if (!record_.empty()) rec_ = std::make_unique<lowrec::Writer>(record_, hc_);

    tf_buf_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    // 스레드 없이 이 노드의 실행기에서 받음 (노드 = 한 스레드). TF 는 다른 프로세스에서 오고 /tf_static 은 transient_local 이라 프로세스 안 전달을 끔 (object_map 과 같음)
    rclcpp::SubscriptionOptions tf_opt = tf2_ros::detail::get_default_transform_listener_sub_options<>();
    rclcpp::SubscriptionOptions tf_static_opt = tf2_ros::detail::get_default_transform_listener_static_sub_options<>();
    tf_opt.use_intra_process_comm = rclcpp::IntraProcessSetting::Disable;
    tf_static_opt.use_intra_process_comm = rclcpp::IntraProcessSetting::Disable;
    tf_lis_ = std::make_shared<tf2_ros::TransformListener>(*tf_buf_, this, false, tf2_ros::DynamicListenerQoS(), tf2_ros::StaticListenerQoS(), tf_opt, tf_static_opt);
    const auto sensor = rclcpp::SensorDataQoS();
    sub_scan_ = create_subscription<sensor_msgs::msg::LaserScan>(scan_topic, sensor, [this](sensor_msgs::msg::LaserScan::ConstSharedPtr m) { on_scan(m); });
    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(odom_topic, 20, [this](nav_msgs::msg::Odometry::ConstSharedPtr m) {
      odom_ = m;
      odom_t_ = now();
    });
    sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(imu_topic, 50, [this](sensor_msgs::msg::Imu::ConstSharedPtr m) { imu_ = m; });
    // 깊이: 같은 프로세스의 물체 지도(object_map)와 구독 하나를 나눠 씀 — 역직렬화 한 번, 같은 포인터 (limo_fire_perception/shared_image.hpp, SensorDataQoS)
    sub_depth_ = limo_fire_perception::SharedImageSubscription::create(this, depth_topic, [this](const sensor_msgs::msg::Image::ConstSharedPtr & m) { depth_ = m; });
    sub_info_ = create_subscription<sensor_msgs::msg::CameraInfo>(info_topic, sensor, [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr m) { info_ = m; });
    rc_topic_ = rc_topic;
    subscribe_rc();
    // 늦게 뜬 RViz·BT 도 마지막 값을 받게 transient_local 인 것: 프로세스 안 전달은 volatile 만 되므로 이것들만 끔 (밖으로만 나가는 토픽)
    rclcpp::PublisherOptions latched;
    latched.use_intra_process_comm = rclcpp::IntraProcessSetting::Disable;
    const auto tl = rclcpp::QoS(1).transient_local();
    if (!dry_run_) pub_cmd_vel_ = create_publisher<geometry_msgs::msg::Twist>(cmd_topic, 10);   // dry run 이면 발행자를 아예 안 만듦
    if (dry_run_ || debug_) pub_cmd_ = create_publisher<geometry_msgs::msg::TwistStamped>("cmd_debug", 10);   // dry run 에서 명령을 보는 곳
    pub_status_ = create_publisher<std_msgs::msg::String>("status", 10);
    pub_state_ = create_publisher<std_msgs::msg::String>("state", tl, latched);
    pub_done_ = create_publisher<std_msgs::msg::Bool>("done", tl, latched);
    if (debug_) {
      pub_goal_ = create_publisher<visualization_msgs::msg::MarkerArray>("goal_marker", 10);
      pub_cand_ = create_publisher<visualization_msgs::msg::MarkerArray>("candidates", 10);
      pub_dec_img_ = create_publisher<sensor_msgs::msg::Image>("decider_image", tl, latched);   // 판단 모델에 마지막으로 보낸 지도 그림 (보기만)
      pub_cand_txt_ = create_publisher<rviz_2d_overlay_msgs::msg::OverlayText>("candidates_text", tl, latched);
      publish_dec_placeholder();   // 첫 요청 전에는 회색 판 (RViz 가 "No Image" 로 비지 않게)
    }
    // 지시 글 /explore/goal (판단 모델 글에 넣음) · 판단 모델이 물체 후보를 고르면 그 설 자리로 가기
    goto_resume_ = declare_parameter<bool>("goto_resume", false);   // 도착하면 멈춤 (true = 탐사 이어 감)
    {   // 지시 일 감독 (scripts/task_sup.h — 시뮬 low_explore_check 와 같은 코드): 설 자리로 가기 · 바라보기 · 못 가면 실패 · 지시 시간 한도
      tsk::Params tp;
      tp.arrive_m = (float)declare_parameter<double>("goto_arrive_m", tp.arrive_m);
      tp.arrive_far_m = (float)declare_parameter<double>("goto_arrive_far_m", tp.arrive_far_m);
      tp.face_tol = (float)declare_parameter<double>("goto_face_tol", tp.face_tol);
      tp.face_timeout_s = (float)declare_parameter<double>("goto_face_timeout_s", tp.face_timeout_s);
      tp.stall_s = (float)declare_parameter<double>("goto_stall_s", tp.stall_s);
      tp.stall_m = (float)declare_parameter<double>("goto_stall_m", tp.stall_m);
      tp.nopath_s = (float)declare_parameter<double>("goto_nopath_s", tp.nopath_s);
      tp.max_fail = (int)declare_parameter<int>("goto_max_fail", tp.max_fail);
      tp.fail_ttl_s = (float)declare_parameter<double>("goto_fail_ttl_s", tp.fail_ttl_s);
      tp.instr_limit_s = (float)declare_parameter<double>("instruction_limit_s", tp.instr_limit_s);
      goto_ = tsk::Goto(tp);
      fails_ = tsk::Fails(tp);
      instr_ = tsk::Instr(tp);
    }
    pub_goal_status_ = create_publisher<std_msgs::msg::String>("goal_status", tl, latched);
    obj_stale_s_ = declare_parameter<double>("decider_obj_stale_s", 600.0);   // 이보다 오래 안 본 물체는 물체 후보에서 뺌 (s)
    sub_goal_ = create_subscription<std_msgs::msg::String>("goal", 10, [this](std_msgs::msg::String::ConstSharedPtr m) { on_goal(m->data); });
    // 물체 지도 (limo_fire_perception object_map): 지시가 있을 때 판단 모델 글에 아는 물체를 넣음.
    // 임무 BT 도 transient_local 로 받아 발행이 transient_local 이므로 프로세스 안 전달을 끔 (작은 글, 10 Hz)
    rclcpp::SubscriptionOptions latched_sub;
    latched_sub.use_intra_process_comm = rclcpp::IntraProcessSetting::Disable;
    sub_objects_ = create_subscription<vision_msgs::msg::Detection3DArray>("/perception/objects", tl,
      [this](vision_msgs::msg::Detection3DArray::ConstSharedPtr m) { objects_ = m; }, latched_sub);
    // 뜻 물체 칸 (decider_sem_max > 0 일 때만): 지시 글을 인식 object_search 에 묻고 (/perception/object_query), 물체마다 맞음 확률을 받음
    // (/perception/object_query_result — score = 물체 사진들 중 가장 높은 sigmoid(scale·cos + bias), limo_fire_perception/sem_score.hpp). 끄면 토픽을 안 만듦
    sem_strip_ = declare_parameter<std::vector<std::string>>("decider_sem_strip", std::vector<std::string>{});   // 지시 → 명사구 (class_alias.h query_phrase)
    if (dec_up_->params().sem_max > 0) {
      pub_query_ = create_publisher<std_msgs::msg::String>("/perception/object_query", tl, latched);   // 늦게 뜬 object_search 도 지금 지시를 받게
      sub_query_ = create_subscription<vision_msgs::msg::Detection3DArray>("/perception/object_query_result", tl,
        [this](vision_msgs::msg::Detection3DArray::ConstSharedPtr m) {
          // 지금 지시를 보낸 뒤에 셈한 결과만 (옛 지시로 셈해 오던 결과 · 늦게 붙어 받은 옛 결과가 새 지시 점수로 붙지 않게 — 같은 시계, object_search 는 셈한 때를 찍음)
          if (instruction_.empty() || rclcpp::Time(m->header.stamp, get_clock()->get_clock_type()) < query_t_) return;
          sem_.clear();
          for (const auto& d : m->detections) if (!d.results.empty()) sem_[d.id] = (float)d.results[0].hypothesis.score;
          obj_dirty_ = true;
        }, latched_sub);
    }
    srv_enable_ = create_service<std_srvs::srv::SetBool>("enable", [this](const std::shared_ptr<std_srvs::srv::SetBool::Request> q,
                                                                             std::shared_ptr<std_srvs::srv::SetBool::Response> r) { on_enable(q->data, r.get()); });
    watchdog_ = create_wall_timer(std::chrono::milliseconds(50), [this] { check_inputs(); });
    tf_retry_ = create_wall_timer(std::chrono::milliseconds(5), [this] { try_scans(); });   // 장 시각 TF 를 기다리는 장 (막지 않고 5 ms 마다)
    if (enabled_ && require_rc_) {   // 켠 채로 띄워도 리모컨 오토를 한 번 본 뒤에만
      enabled_ = false;
      RCLCPP_WARN(get_logger(), "enabled:=true 지만 리모컨 오토를 아직 못 봄 — /explore/enable true 로 켤 것");
    }
    set_state(enabled_ ? "running" : "idle", enabled_ ? "켬" : "꺼짐 — /explore/enable true (RViz 탐사 패널) 로 켬");
    std_msgs::msg::Bool d;
    d.data = false;
    pub_done_->publish(d);
  }

 private:
  double dt() const { return (double)env_->cfg().substeps * env_->cfg().dt; }

  // ---- 상태 (RViz 탐사 패널: state running 이면 켜짐, done, status)
  void set_state(const std::string& st, const std::string& why) {
    if (st != state_ || why != why_) RCLCPP_INFO(get_logger(), "상태 %s — %s", st.c_str(), why.c_str());
    state_ = st;
    why_ = why;
    std_msgs::msg::String m;
    m.data = st;
    pub_state_->publish(m);
  }
  bool running() const { return enabled_ && !finished_; }

  // 지시 "○○ 로 가" (/explore/goal): 글에서 클래스를 풀어 판단 모델에 힌트로 주고(물체 후보 차례에서 앞으로), 지시 글은 판단 모델 글에 그대로.
  // 빈 글 = 지시 지움 → 보통 탐사. 지시가 바뀌면 하던 따라가기·물체로 가기는 끝내고 판단 모델이 새로 고름 (바로 물음 — Upper "new instruction")
  void on_goal(const std::string& text) {
    if (!text.empty()) {
      target_class_ = class_of(text, aliases_, object_classes());
      dec_up_->set_class_hint(target_class_);
    }
    instruction_ = text;
    sem_.clear();   // 옛 지시의 점수는 버림 (새 답이 오면 다시 채움 — 그 전까지 뜻 칸은 빔)
    query_t_ = now();
    if (pub_query_) { std_msgs::msg::String q; q.data = text.empty() ? text : query_phrase(text, sem_strip_); pub_query_->publish(q); }
    end_follow("");
    end_goto();
    fails_.clear();
    if (instruction_.empty()) {
      instr_.stop();
      target_class_.clear();
      dec_up_->set_objects({});
      say_goto("지시 없음 — 보통 탐사");
    } else {
      instr_.start(now().seconds());
      say_goto("지시: " + instruction_ + (target_class_.empty() ? std::string(" (클래스 못 풂 — 판단 모델이 글을 읽음)") : " → 클래스 " + target_class_) +
               " — 판단 모델이 후보(지도 물체 옆 · 프론티어) 중에서 고름");
    }
    dec_up_->set_instruction(instruction_);
    obj_dirty_ = true;   // 다음 걸음에 물체 후보를 새로 채움 (판단 모델이 "new instruction" 으로 바로 물을 때 같이 들어가게)
  }
  // 지시의 클래스를 지금 물체 지도로 다시 풂: 영문 클래스 이름 지시("find the bed")는 별칭 표에 없고 물체 지도 클래스로만 풀리는데, 지시 때 그 물체를
  // 아직 못 봤으면 힌트가 끝까지 비어 이름 물체 칸이 안 참 (시뮬 332 짝에서 "봤지만 후보에 안 듦" 의 까닭 하나). 물체 후보 재료를 줄 때마다 (low_explore_check 와 같은 차례)
  void refresh_class() {
    if (instruction_.empty()) return;
    const std::string c = class_of(instruction_, aliases_, object_classes());
    if (c == target_class_) return;
    target_class_ = c;
    dec_up_->set_class_hint(c);
    if (!c.empty()) say_goto("지시: " + instruction_ + " → 클래스 " + c + " (물체 지도에서 풂)");
  }
  // 지금 물체 지도에 있는 클래스 이름 (영문 클래스 이름으로 지시해도 풀리게)
  std::vector<std::string> object_classes() const {
    std::vector<std::string> v;
    if (objects_)
      for (const auto& d : objects_->detections)
        if (!d.results.empty()) v.push_back(d.results[0].hypothesis.class_id);
    return v;
  }

  void send_zero(const char* why) {   // 멈춤: 0 명령 (실제 주행이면 /cmd_vel 도)
    if (pub_cmd_) {
      geometry_msgs::msg::TwistStamped c;
      c.header.stamp = now();
      c.header.frame_id = base_frame_;
      pub_cmd_->publish(c);
    }
    if (pub_cmd_vel_) pub_cmd_vel_->publish(geometry_msgs::msg::Twist());
    (void)why;
  }
  void stop(const std::string& st, const std::string& why, bool disable) {
    if (disable) enabled_ = false;
    send_zero(why.c_str());
    set_state(st, why);
  }

  void on_enable(bool on, std_srvs::srv::SetBool::Response* r) {
    if (on) {
      if (require_rc_ && !rc_auto_now()) {
        r->success = false;
        r->message = "리모컨이 오토(SWB 위)가 아니거나 /rc_state 가 안 옴 — 켜지 않음 · " + rc_info();
        RCLCPP_WARN(get_logger(), "%s", r->message.c_str());
        return;
      }
      if (finished_) {   // 끝난 뒤 다시 켜면 새 판: 다음 장에서 지금 자리로 판 시작
        finished_ = false;
        started_ = false;
        std_msgs::msg::Bool d;
        d.data = false;
        pub_done_->publish(d);
      }
      enabled_ = true;
      contact_t_ = -1.0;
      ctl_->force_redo();   // 다시 켜면 바로 위층 결정 (지도·피하는 칸은 그대로)
      set_state("running", "켬");
      r->success = true;
      r->message = dry_run_ ? "켬 (dry run — /explore/cmd_debug 에만)" : "켬 — 주행";
    } else {
      stop("idle", "꺼짐 (/explore/enable false)", true);
      r->success = true;
      r->message = "멈춤";
    }
  }

  bool rc_auto_now() const {
    if (!rc_) return false;
    if (rc_watch_.age_s(std::chrono::steady_clock::now()) > rc_timeout_) return false;
    return (int)rc_->buttons.size() > rc_button_ && rc_->buttons[rc_button_] == rc_auto_;
  }
  // /rc_state 구독 (best effort — rc_watch.h). 다시 구독할 때도 이것
  void subscribe_rc() {
    sub_rc_ = create_subscription<sensor_msgs::msg::Joy>(rc_topic_, limo_fire_explore::rc_qos(), [this](sensor_msgs::msg::Joy::ConstSharedPtr m) { on_rc(m); });
    rc_watch_.subscribed(std::chrono::steady_clock::now());
  }
  // 리모컨 받기 상태 한 줄 (status · 켜기 거절 글)
  std::string rc_info() const {
    char b[160];
    const double age = rc_watch_.age_s(std::chrono::steady_clock::now());
    std::snprintf(b, sizeof b, "리모컨 %s (마지막 %s 전, 받음 %ld, 발행자 %zu, 다시 구독 %d, buttons[%d]=%d)", rc_auto_now() ? "오토" : "오토 아님",
                  age > 1e8 ? "없음" : (std::to_string((int)std::lround(age * 1000)) + " ms").c_str(), rc_watch_.count(), sub_rc_ ? sub_rc_->get_publisher_count() : (size_t)0,
                  rc_watch_.resubs(), rc_button_, rc_ && (int)rc_->buttons.size() > rc_button_ ? rc_->buttons[rc_button_] : -1);
    return b;
  }
  void on_rc(const sensor_msgs::msg::Joy::ConstSharedPtr& m) {
    rc_ = m;
    rc_watch_.got(std::chrono::steady_clock::now());
    rc_auto_ok_ = rc_auto_now();
    if (!rc_auto_ok_ && require_rc_ && enabled_) stop("paused", "리모컨이 오토가 아님 (SWB) — 다시 오토로 바꾸고 /explore/enable true", true);
  }

  // 20 Hz: 입력 끊김 · 리모컨 끊김 · 닿음 뒤 저절로 켜기
  void check_inputs() {
    const rclcpp::Time t = now();
    if (sub_rc_ && rc_watch_.want_resub(std::chrono::steady_clock::now(), sub_rc_->get_publisher_count(), rc_resub_s_)) {
      RCLCPP_WARN(get_logger(), "/rc_state 발행자는 있는데 %.1f s 넘게 못 받음 — 구독을 새로 만듦 (%s)", rc_resub_s_, rc_info().c_str());
      subscribe_rc();
    }
    if (require_rc_ && enabled_ && !rc_auto_now()) stop("paused", "/rc_state 가 안 옴(" + std::to_string((int)std::lround(rc_timeout_ * 1000)) + " ms 넘게) 또는 오토 아님 — /explore/enable true 로 다시", true);
    const bool stale = (last_scan_t_.nanoseconds() == 0 || (t - last_scan_t_).seconds() > input_timeout_) ||
                       (last_tf_t_.nanoseconds() == 0 || (t - last_tf_t_).seconds() > input_timeout_) ||
                       (odom_t_.nanoseconds() == 0 || (t - odom_t_).seconds() > input_timeout_);
    if (running() && stale) {
      if (!stale_) RCLCPP_WARN(get_logger(), "입력 끊김 (라이다·TF·오도메트리 %.2f s 넘게) — 명령 0", input_timeout_);
      stale_ = true;
      send_zero("입력 끊김");
    } else if (stale_ && !stale) {
      stale_ = false;
      RCLCPP_INFO(get_logger(), "입력 돌아옴");
    }
    if (!enabled_ && contact_t_ >= 0.0 && contact_resume_s_ > 0.0 && t.seconds() - contact_t_ >= contact_resume_s_ && !finished_ && (!require_rc_ || rc_auto_now())) {
      enabled_ = true;
      contact_t_ = -1.0;
      ctl_->force_redo();
      set_state("running", "닿음 뒤 저절로 켬 (contact_resume_s)");
    }
  }

  Proprio proprio() const {
    Proprio p;
    p.wheel_v = (float)odom_->twist.twist.linear.x;
    p.wheel_vy = (float)odom_->twist.twist.linear.y;
    p.wheel_w = (float)odom_->twist.twist.angular.z;
    p.gyro_z = (float)imu_->angular_velocity.z;
    p.imu[0] = (float)imu_->linear_acceleration.x;
    p.imu[1] = (float)imu_->linear_acceleration.y;
    p.imu[2] = (float)imu_->linear_acceleration.z;
    p.imu[3] = (float)imu_->angular_velocity.x;
    p.imu[4] = (float)imu_->angular_velocity.y;
    tf2::Quaternion q(imu_->orientation.x, imu_->orientation.y, imu_->orientation.z, imu_->orientation.w);
    double r, pi, yw;
    tf2::Matrix3x3(q).getRPY(r, pi, yw);
    p.imu[5] = (float)r;
    p.imu[6] = (float)pi;
    return p;
  }
  static double yaw_of(const geometry_msgs::msg::Quaternion& q) {
    return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
  }

  // 장 도착: 장 시각의 map → base_link 가 오면 처리 (try_scans). 노드는 한 스레드라 TF 를 막고 기다리지 않고, 장과 그때의 감각 값(오도메트리·IMU·
  // 깊이 — 옛 막는 기다림 동안 바뀌지 않던 것)을 줄에 두고 tf_timeout 안에 TF 가 오면 차례대로 처리, 넘으면 그 장을 건너뜀
  void on_scan(const sensor_msgs::msg::LaserScan::ConstSharedPtr& m) {
    last_scan_t_ = now();
    if (finished_) return;
    if (!odom_ || !imu_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "오도메트리·IMU 기다림");
      return;
    }
    if (scans_.empty()) head_t_ = std::chrono::steady_clock::now();
    scans_.push_back(Pend{m, odom_, imu_, depth_, info_});
    try_scans();
  }
  void try_scans() {
    while (!scans_.empty()) {
      Pend& e = scans_.front();
      std::string err;
      geometry_msgs::msg::TransformStamped T;
      bool ok = false;
      const tf2::TimePoint ts = tf2_ros::fromMsg(e.scan->header.stamp);   // 기다리지 않는 BufferCore 판 (기다림 판은 TF 스레드가 있어야 함)
      if (tf_buf_->canTransform(map_frame_, base_frame_, ts, &err)) {
        try {
          T = tf_buf_->lookupTransform(map_frame_, base_frame_, ts);
          ok = true;
        } catch (const std::exception& ex) {
          err = ex.what();
        }
      }
      if (!ok && std::chrono::duration<double>(std::chrono::steady_clock::now() - head_t_).count() < tf_timeout_) return;   // 아직 기다림
      const Pend p = std::move(e);
      scans_.pop_front();
      head_t_ = std::chrono::steady_clock::now();
      if (!ok) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "TF %s → %s (장 시각) 없음: %s — 이 장 건너뜀", map_frame_.c_str(), base_frame_.c_str(), err.c_str());
        continue;
      }
      // 장이 왔을 때의 감각 값으로 처리 (그 사이 온 값은 다음 장에)
      auto o = odom_; auto i = imu_; auto d = depth_; auto f = info_;
      odom_ = p.odom; imu_ = p.imu; depth_ = p.depth; info_ = p.info;
      on_scan_tf(p.scan, T);
      odom_ = o; imu_ = i; depth_ = d; info_ = f;
    }
  }
  // ---- 걸음 하나 (장 한 장, 0.1 s) — 위에서 아래로: 감각 → 판 끝 → 깊이 → 위층 결정 → 아래층 추종 → 명령·상태 발행
  using clk = std::chrono::steady_clock;
  static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

  void on_scan_tf(const sensor_msgs::msg::LaserScan::ConstSharedPtr& m, const geometry_msgs::msg::TransformStamped& T) {
    if (finished_) return;
    last_tf_t_ = now();
    // 꺼짐·멈춤 동안은 판을 세움: 판 시계(진척 없음·시간·닫힘)는 켠 동안만 감. 판은 처음 켤 때 시작
    if (!running()) return;
    const Scan s = to_scan(*m, T);
    const Proprio p = proprio();
    const auto t0 = clk::now();
    if (!started_) start_episode(s, p);
    else if (!sense(s, p)) return;   // 판이 끝남
    const double step_ms = ms_since(t0);
    read_depth();
    const double dec_ms = decide();
    const auto t2 = clk::now();
    // 규칙 추종기: 경로 계획 후보 (combine_run 의 act[0] — 다음 결정까지 그대로). 따라가기 · 물체로 가기는 목표 칸을 노드가 줌 — 행동 칸 0 (low_follow 의 그 칸)
    env_->set_slot(follower_.active() || goto_.active() ? 0 : ctl_->slot());
    env_->low_prepare();
    const double obs_ms = ms_since(t2);
    float v = 0.f, w = 0.f;
    if (running()) env_->low_act(&v, &w);
    else env_->hold();   // 멈춤: 명령 0
    publish_cmd(m->header.stamp, v, w);
    publish_status(step_ms, obs_ms, dec_ms, v, w);
  }

  Scan to_scan(const sensor_msgs::msg::LaserScan& m, const geometry_msgs::msg::TransformStamped& T) const {
    Scan s;
    s.ranges.assign(m.ranges.begin(), m.ranges.end());
    s.intensities.assign(m.intensities.begin(), m.intensities.end());
    s.angle_min = m.angle_min;
    s.angle_increment = m.angle_increment;
    s.time_increment = m.time_increment;
    s.x = (float)T.transform.translation.x;
    s.y = (float)T.transform.translation.y;
    s.th = (float)yaw_of(T.transform.rotation);
    s.w = (float)imu_->angular_velocity.z;
    return s;
  }

  // 판 시작 (시뮬 리셋 관측 — 장 없음): 관측·결정만
  void start_episode(const Scan& s, const Proprio& p) {
    env_->reset(s.x, s.y, s.th, p, -2.f * s.angle_min);
    const sim::MapDev& md = env_->map();
    world_.gw = md.gw; world_.gh = md.gh; world_.ox = md.ox; world_.oy = md.oy;
    ctl_->restart();   // 새 판: 위층 피하는 칸·목표도 처음부터
    if (dec_up_) dec_up_->restart();
    ctl_->set_geom(md.gw, env_->cfg().res);
    if (rec_) rec_->reset(s.x, s.y, s.th, p, -2.f * s.angle_min, RobotEnv::Window{md.ox, md.oy, md.gw, md.gh});
    started_ = true;
    RCLCPP_INFO(get_logger(), "시작 자세 (%.2f, %.2f, %.1f°) — 지도 창 %.1f m", s.x, s.y, s.th * 57.2958, md.gw * env_->cfg().res);
  }

  // 감각 한 걸음 (지도·관측) + 닿음 · 물체로 가기 · 지시 감독 · 판 끝. 판이 끝나면 false
  bool sense(const Scan& s, const Proprio& p) {
    if (rec_) rec_->step(p, s);
    env_->step(p, s);
    const sim::EnvState& e = env_->state();
    // 닿음 (시뮬 contact.h sense_step — e.contact = 닿음 신호 & con_mask): 멈추고 끔
    if (e.contact && running()) {
      contact_t_ = now().seconds();
      char b[200];
      std::snprintf(b, sizeof b, "닿음 (신호 %d: 라이다 1 · 깊이 2 · 멈춤 4 · 지도 8 · 사람 16 · 바퀴·자이로 32 · 가속 64, 가속 %.1f m/s²) — %s", e.s_kind, e.a_h,
                    contact_resume_s_ > 0.0 ? "멈춤, 잠시 뒤 저절로 켬" : "멈춤, /explore/enable true 로 다시");
      stop("paused", b, true);
    }
    // 물체로 가기 (task_sup.h): 닿으면 물체 쪽으로 돌아 선 뒤 끝 (goto_resume 이면 탐사 이어 감, 아니면 멈춤), 못 가면 그 물체를 잠시 빼고 다시 물음
    if (goto_.active() && running()) {
      const tsk::Ev ev = goto_.step(now().seconds(), e.x + e.ex, e.y + e.ey, e.th + e.eth, e.goal_cell >= 0 && e.p_n <= 0,
                                    tsk::path_end(e, world_.ox, world_.oy, env_->cfg().res, env_->cfg().goal_tol));
      if (ev == tsk::kArrived) finish_goto();
      else if (ev == tsk::kFailed) fail_goto();
    }
    if (instr_.active() && running()) {   // 지시 감독: 실패가 잇따르거나 시간이 넘으면 멈추고 알림
      if (const char* g = instr_.give_up(now().seconds())) {
        end_follow("");
        end_goto();
        instr_.stop();
        say_goto(std::string(std::strcmp(g, "could not reach") == 0 ? "못 감: " : "못 찾음: ") + instruction_ + " — " + g);
        stop("paused", std::string("지시를 끝내지 못함 (") + g + ")", true);
      }
    }
    // 판 끝 (시뮬 finish_step 의 합친 탐사 규칙 — combine_run.cpp:1270). 물체로 가는 중이면 끝내지 않음
    if (goto_.active() || follower_.active()) {
    } else if (ctl_->rough_done(e.step) && instr_.active()) {   // 지시가 남았는데 "대충 다 앎": 판을 끝내지 않고 물체 후보로 (판단 모델에 물음), 물체도 없으면 못 찾음
      if (!dec_up_ || !objects_ || map_objects().empty()) {
        instr_.stop();
        say_goto("못 찾음: " + instruction_ + " — 다 탐사했는데 지도에 맞는 물체가 없음");
        stop("paused", "지시를 끝내지 못함 (다 탐사함)", true);
      } else if (!dec_up_->waiting() && e.step % lp_.decide_steps == 0) dec_up_->ask("exploration is done but the instruction is not");
    } else if (ctl_->rough_done(e.step)) {   // 후보 모드 "대충 다 앎" 끝 (Ctl::rough_done — low_explore_check 끝 코드 11 과 같은 판정)
      finish("끝: 대충 다 앎 (후보 0 · 남은 덩어리는 대충 아는 곳 안 구멍뿐 10 s)");
      return false;
    }
    const int dn = (goto_.active() || follower_.active() || instr_.active()) ? 0 : env_->done();
    if (dn == mdp::cur::kDeclareOk || dn == mdp::cur::kNoProgress || dn == mdp::cur::kTruncated) {
      finish(dn == mdp::cur::kDeclareOk ? "끝: 지도 닫힘 + 진척 없음 10 s (시뮬 성공 판정)" : dn == mdp::cur::kNoProgress ? "끝: 지도 진척 없음 (시뮬 실패 판정)" : "끝: 시간 한도");
      return false;
    }
    ctl_->after_step(e, env_->low_rl(), env_->low_ep());   // 판단 모델도 피하는 칸 · 다시 고름은 Ctl 이 셈
    return true;
  }
  void finish(const std::string& why) {
    finished_ = true;
    stop("done", why, true);
    std_msgs::msg::Bool d;
    d.data = true;
    pub_done_->publish(d);
  }

  // 깊이 (센서 입력): 장 시각에 가까운 깊이 영상 → 관측 깊이 · 깊이 카메라 층 격자 (없거나 오래되면 0 칸 = 구멍)
  void read_depth() {
    std::vector<uint8_t> dep((size_t)mdp::obs::DEPTH_SIZE, 0), ldep((size_t)env_->layer_n(), 0);
    if (depth_ && info_ && (now() - rclcpp::Time(depth_->header.stamp)).seconds() < depth_timeout_ && depth_->encoding == "16UC1") {
      env_->depth_from_image(reinterpret_cast<const uint16_t*>(depth_->data.data()), (int)depth_->width, (int)depth_->height, (int)(depth_->step / 2),
                             (float)info_->k[0], (float)info_->k[4], (float)info_->k[2], (float)info_->k[5], dep.data());
      if (!ldep.empty())
        env_->layer_from_image(reinterpret_cast<const uint16_t*>(depth_->data.data()), (int)depth_->width, (int)depth_->height, (int)(depth_->step / 2),
                               (float)info_->k[0], (float)info_->k[4], (float)info_->k[2], (float)info_->k[5], ldep.data());
    } else {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "깊이 영상 없음·오래됨·16UC1 아님 — 깊이 관측 0 (구멍)");
    }
    env_->set_depth(dep.data());
    if (rec_) rec_->depth(dep.data());
    if (!ldep.empty()) {
      env_->set_layer_depth(ldep.data());
      if (rec_) rec_->layer(ldep.data(), ldep.size());
    }
  }

  // 위층: 판단 모델 (기다리지 않음 — 요청은 비차단 소켓으로 걸음마다 조금씩, 답은 다음 걸음들에서 꺼냄) · 물체로 가기 · 따라가기.
  // 걸린 ms (꺼져 있으면 −1)
  double decide() {
    if (!running()) {
      if (env_->low_cmd().follow != 0) env_->set_cmd(sim::LowCmd{-1, 0});
      return -1.0;
    }
    const auto t1 = clk::now();
    const lxc::View v{&env_->cfg(), &world_, env_->lmap(), env_->trail(), env_->cam_layer(), &env_->state(), env_->cand(), (uint32_t)env_->cand_cur(), env_->reach_keep()};
    sim::LowCmd cmd;
    int why = 0;
    if ((!instruction_.empty() || dec_obj_always_) && objects_ && (env_->state().step % lp_.decide_steps == 0 || obj_dirty_)) {
      obj_dirty_ = false;
      refresh_class();
      dec_up_->set_known_objects(known_objects());
      dec_up_->set_objects(map_objects());   // 지도 물체 모두 (설 자리는 Upper::object_spots 가 탐사 지도에서, 칸 고르기는 Upper::pick_objects)
    }
    const bool stepped_change = dec_up_->step(v, *ctl_, *dec_cli_, &cmd, &why);
    if (stepped_change) {
      env_->set_cmd(cmd);
      why_dec_ = why;
      publish_goal(cmd.target);
    }
    if (env_->state().step % lp_.decide_steps == 0) publish_object_cands();
    if (dec_up_->take_stop_pick()) {   // 판단 모델이 멈춤 후보를 고름 ("멈춰" · "거기 서") → 따라가기 · 물체로 가기 끝, 꺼짐 (다시 켜면 이어 감)
      end_follow("");
      end_goto();
      instr_.stop();
      say_goto("판단 모델이 멈춤을 고름 — 멈춤 (지시: " + instruction_ + ")");
      stop("paused", "판단 모델이 멈춤을 고름", true);
    }
    lxd::Params::Obj pick;
    const bool picked = dec_up_->take_object_pick(&pick);
    if (follower_.active() && stepped_change) end_follow("판단 모델이 따라가기 대신 탐사 자리를 고름");
    if (goto_.active() && stepped_change) { end_goto(); say_goto("판단 모델이 물체 대신 탐사 자리를 고름"); }
    if (picked && pick.follow) {   // 판단 모델이 따라가기 후보를 고름 → 그 사람을 따라감 (목표는 노드가 쥠 — 판단 모델은 답만 받고, 사람이 많이 옮기면 다시 물음)
      end_follow("");
      follower_.start(pick.oid, pick.x, pick.y, now().seconds());
      dec_up_->set_external(true);
      char b[160];
      std::snprintf(b, sizeof b, "판단 모델이 따라가기를 고름: %s (%.2f, %.2f) — %.1f m 뒤에서 최대 %.1f m/s", pick.oid.c_str(), pick.x, pick.y, follower_.params().dist,
                    follower_.params().v_max);
      say_goto(b);
    } else if (picked) end_follow("");
    if (follower_.active()) follow_step();
    if (picked && !pick.follow) start_goto(pick);
    if (goto_.active()) apply_goto_cmd();
    if (dec_cli_->n_png() != n_png_pub_ && !dec_cli_->last_png().empty()) {   // 새 요청을 보냈으면 그 지도 그림을 RViz 로
      n_png_pub_ = dec_cli_->n_png();
      publish_dec_png(dec_cli_->last_png());
    }
    const double dec_ms = ms_since(t1);
    if (env_->state().step % lp_.decide_steps == 0) publish_cands();   // candidates: 판단 모델이 마지막으로 만든 후보 목록 그대로 (따로 안 만듦)
    return dec_ms;
  }
  // 판단 모델(또는 지시 쪽 규칙)이 물체 후보를 고름 → 그 설 자리로, 닿으면 물체를 바라보고 멈춤
  void start_goto(const lxd::Params::Obj& pick) {
    float gx = 0.f, gy = 0.f;
    const auto* d = find_object(pick.oid);
    if (!d || std::hypot(d->bbox.center.position.x - pick.x, d->bbox.center.position.y - pick.y) > 1.0) {   // 요청 뒤 물체가 지도에서 빠지거나 옮김 → 옛 자리로 안 가고 다시 물음
      say_goto("고른 물체가 지도에서 빠지거나 옮김: " + pick.oid + " — 다시 물음");
      dec_up_->ask("chosen object is gone or moved");
    } else if (env_->cell_xy(pick.vp, &gx, &gy)) {
      goto_.start(pick.oid, pick.oid, pick.cls, gx, gy, true, pick.x, pick.y, now().seconds());
      dec_up_->set_external(true);
      char b[160];
      std::snprintf(b, sizeof b, "물체 후보를 고름: %s — 설 자리 (%.2f, %.2f) 로%s", pick.oid.c_str(), gx, gy, dec_up_->n_fb_obj() > fb_obj_seen_ ? " (판단 모델 대신 규칙)" : "");
      fb_obj_seen_ = dec_up_->n_fb_obj();
      say_goto(b);
    }
  }

  // 명령 발행: 녹화 · 멈춤이면 0 · 속도 한도 → cmd_debug (dry run·debug) · /cmd_vel (실제 주행이고 켜짐·입력 정상일 때만)
  void publish_cmd(const builtin_interfaces::msg::Time& stamp, float& v, float& w) {
    if (rec_) {   // 그때 아래층 목적지도 (다시 못 내는 판단 모델 위층의 목표 — node_stats)
      float gx = NAN, gy = NAN;
      const sim::LowCmd& lc = env_->low_cmd();
      if (lc.follow != 0) env_->cell_xy(lc.target, &gx, &gy);
      rec_->act(v, w, gx, gy);
    }
    if (!running() || stale_) { v = 0.f; w = 0.f; }
    if (std::fabs(v) > max_speed_) v = v > 0.f ? (float)max_speed_ : -(float)max_speed_;   // max_speed (기본 = 시뮬 한도라 안 자름)
    geometry_msgs::msg::Twist t;
    t.linear.x = v;
    t.angular.z = w;
    if (pub_cmd_) {
      geometry_msgs::msg::TwistStamped cmd;
      cmd.header.stamp = stamp;
      cmd.header.frame_id = base_frame_;
      cmd.twist = t;
      pub_cmd_->publish(cmd);
    }
    if (pub_cmd_vel_ && running() && !stale_) pub_cmd_vel_->publish(t);
  }

  // /explore/status 한 줄 (10 걸음 = 1 s 마다)
  void publish_status(double step_ms, double obs_ms, double dec_ms, float v, float w) {
    sum_step_ += step_ms;
    sum_obs_ += obs_ms;
    if (dec_ms >= 0.0) { sum_dec_ += dec_ms; ++n_dec_; max_dec_ = std::max(max_dec_, dec_ms); }
    ++n_;
    if (n_ % 10 != 0) return;
    std_msgs::msg::String st;
    char buf[512];
    const sim::LowCmd& lc = env_->low_cmd();
    std::snprintf(buf, sizeof(buf), "%s (%s) · 리모컨 %s · 걸음 %d · 목표 칸 %d%s · 명령 v %.2f ω %.2f · 결정 %ld 통과 %ld 막힘 %ld · 평균 ms: 걸음 %.2f 관측 %.2f 위층 %.1f (최대 %.1f)",
                  state_.c_str(), why_.c_str(), rc_auto_now() ? "오토" : "오토 아님", env_->steps(), lc.target, why_dec_ == 2 ? " (목표 없음)" : "", v, w,
                  ctl_->n_dec(), ctl_->n_reach(), ctl_->n_stall(), sum_step_ / n_, sum_obs_ / n_, n_dec_ ? sum_dec_ / n_dec_ : 0.0, max_dec_);
    st.data = buf;
    st.data += " · " + rc_info();
    if (dec_up_) {
      char b2[300];
      std::snprintf(b2, sizeof b2, " · 판단 모델: 보냄 %ld 답 %ld (마지막 %.0f ms) 시간 끝 %ld 실패 %ld%s%s 후보 안 맞음 %ld 규칙 대신 %ld%s", dec_up_->n_sent(), dec_up_->n_ans(), dec_up_->last_ms(),
                    dec_up_->n_timeout(), dec_up_->n_fail(), dec_up_->last_err().empty() ? "" : " — ", dec_up_->last_err().c_str(), dec_up_->n_nomatch(), dec_up_->n_fallback(),
                    dec_up_->waiting() ? " · 기다리는 중" : "");
      st.data += b2;
    }
    pub_status_->publish(st);
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 10000, "%s", st.data.c_str());
  }

  // candidates (map 틀, 결정마다): DELETEALL 뒤 후보마다 구(시점) · 글(판단 모델 글자 · 번호 · 넓이/시간) · 시점→덩어리 대표 칸 선 · 덩어리 프론티어 칸(작은 상자),
  // 색 = 보통 초록 · 지금 목표 파랑 · 실패 기억(이 번호 포기 또는 1 m 안 실패 자리) 주황 · 쉬는 덩어리 회색, 판단 모델 마지막 답 시간 글, 로봇 자취(최근 60 s, 오래될수록 옅음)
  void publish_cands() {
    if (!debug_) return;
    const cand::Out& o = ctl_->cands();
    const cand::Mem& mem = ctl_->cand_mem();
    visualization_msgs::msg::MarkerArray ma;
    const rclcpp::Time t = now();
    const auto base = [&](const char* ns, int id, int type) {
      visualization_msgs::msg::Marker mk;
      mk.header.frame_id = map_frame_;
      mk.header.stamp = t;
      mk.ns = ns;
      mk.id = id;
      mk.type = type;
      mk.action = visualization_msgs::msg::Marker::ADD;
      mk.pose.orientation.w = 1.0;
      return mk;
    };
    visualization_msgs::msg::Marker del;
    del.header.frame_id = map_frame_;
    del.action = visualization_msgs::msg::Marker::DELETEALL;
    ma.markers.push_back(del);
    const auto color = [](visualization_msgs::msg::Marker& mk, float r, float g, float b, float a) { mk.color.r = r; mk.color.g = g; mk.color.b = b; mk.color.a = a; };
    const auto pt = [&](int cell) {
      geometry_msgs::msg::Point p;
      float x = 0.f, y = 0.f;
      env_->cell_xy(cell, &x, &y);
      p.x = x; p.y = y; p.z = 0.05;
      return p;
    };
    for (size_t i = 0; i < o.c.size(); ++i) {
      const cand::Cand& c = o.c[i];
      float r = 0.1f, g = 0.8f, b = 0.2f;
      if (c.aborts > 0 || c.fail_near > 0) { r = 1.0f; g = 0.55f; b = 0.0f; }
      if (c.id == ctl_->cur_id() && ctl_->cmd().follow != 0) { r = 0.1f; g = 0.4f; b = 1.0f; }
      auto sp = base("cand", (int)i, visualization_msgs::msg::Marker::SPHERE);
      sp.pose.position = pt(c.vp);
      sp.scale.x = sp.scale.y = sp.scale.z = 0.22;
      color(sp, r, g, b, 0.9f);
      ma.markers.push_back(sp);
      const bool asked = dec_up_ && dec_up_->last_pick_id() == c.id;   // 판단 모델의 마지막 답이 고른 후보
      char buf[96];
      std::snprintf(buf, sizeof buf, "%c#%d:%.0fm2/%.0fs%s%s", 'A' + (int)i, c.id, c.gain_m2, c.t_s, c.aborts > 0 ? "!" : "", asked ? "<" : "");   // RViz 글은 빈칸이 넓어 빈칸 없이
      const double tsz = 0.30;   // 글 높이 (m) — 지도 위에서 읽히게
      const geometry_msgs::msg::Point at = pt(c.vp);
      auto bg = base("cand_text_bg", (int)i, visualization_msgs::msg::Marker::CUBE);   // 글 밑 어두운 판 (TEXT 는 배경이 없음)
      bg.pose.position = at;
      bg.pose.position.x += 0.0;
      bg.pose.position.y += 0.30;
      bg.pose.position.z = 0.30;
      bg.scale.x = 0.55 * tsz * (double)std::strlen(buf) + 0.1;
      bg.scale.y = 1.25 * tsz;
      bg.scale.z = 0.01;
      color(bg, 0.f, 0.f, 0.f, 0.65f);
      ma.markers.push_back(bg);
      auto tx = base("cand_text", (int)i, visualization_msgs::msg::Marker::TEXT_VIEW_FACING);
      tx.pose.position = at;
      tx.pose.position.y += 0.30;
      tx.pose.position.z = 0.35;
      tx.scale.z = tsz;
      tx.text = buf;
      color(tx, std::min(1.f, r + 0.4f), std::min(1.f, g + 0.3f), std::min(1.f, b + 0.4f), 1.0f);   // 어두운 판 위 밝은 색
      ma.markers.push_back(tx);
      auto ln = base("cand_link", (int)i, visualization_msgs::msg::Marker::LINE_LIST);
      ln.points.push_back(pt(c.vp));
      ln.points.push_back(pt(c.anc));
      ln.scale.x = 0.02;
      color(ln, r, g, b, 0.8f);
      ma.markers.push_back(ln);
      auto cl = base("cand_frontier", (int)i, visualization_msgs::msg::Marker::CUBE_LIST);
      if (i < o.cells.size())
        for (size_t q = 0; q < o.cells[i].size(); q += 2) cl.points.push_back(pt(o.cells[i][q]));   // 2 칸마다 (메시지 크기)
      cl.scale.x = cl.scale.y = 0.05;
      cl.scale.z = 0.02;
      color(cl, r, g, b, 0.7f);
      ma.markers.push_back(cl);
    }
    {
      auto dm = base("cand_dormant", 0, visualization_msgs::msg::Marker::SPHERE_LIST);
      for (int cell : o.dormant) dm.points.push_back(pt(cell));
      dm.scale.x = dm.scale.y = dm.scale.z = 0.12;
      color(dm, 0.5f, 0.5f, 0.5f, 0.8f);
      if (!dm.points.empty()) ma.markers.push_back(dm);
    }
    if (!mem.trail.empty()) {   // 자취: 오래될수록 옅게
      auto tr = base("trail", 0, visualization_msgs::msg::Marker::LINE_STRIP);
      tr.scale.x = 0.03;
      const int n = (int)mem.trail.size();
      for (int k = 0; k < n; ++k) {
        geometry_msgs::msg::Point p;
        p.x = mem.trail[k].x; p.y = mem.trail[k].y; p.z = 0.02;
        tr.points.push_back(p);
        std_msgs::msg::ColorRGBA cc;
        cc.r = 0.8f; cc.g = 0.2f; cc.b = 0.8f; cc.a = 0.15f + 0.85f * (float)(k + 1) / n;
        tr.colors.push_back(cc);
      }
      color(tr, 0.8f, 0.2f, 0.8f, 1.0f);
      ma.markers.push_back(tr);
    }
    pub_cand_->publish(ma);
    {   // 상태 글 (화면 고정 · 어두운 바탕 — RViz TextOverlay "Candidates text"): 후보 수 · 쉬는 덩어리 · 후보 시간 · 지금 목표 · 판단 모델 마지막 답 글자·지연
      rviz_2d_overlay_msgs::msg::OverlayText ot;
      ot.action = rviz_2d_overlay_msgs::msg::OverlayText::ADD;
      ot.width = 640; ot.height = 90;
      ot.horizontal_alignment = rviz_2d_overlay_msgs::msg::OverlayText::LEFT;
      ot.vertical_alignment = rviz_2d_overlay_msgs::msg::OverlayText::TOP;
      ot.horizontal_distance = 10; ot.vertical_distance = 10;
      ot.bg_color.r = ot.bg_color.g = ot.bg_color.b = 0.f; ot.bg_color.a = 0.75f;
      ot.fg_color.r = 1.f; ot.fg_color.g = 1.f; ot.fg_color.b = 0.6f; ot.fg_color.a = 1.f;
      ot.text_size = 13.f; ot.line_width = 2; ot.font = "DejaVu Sans Mono";
      int cur_k = -1;
      for (size_t i = 0; i < o.c.size(); ++i) if (o.c[i].id == ctl_->cur_id()) cur_k = (int)i;
      const double dt_s = (double)env_->cfg().substeps * env_->cfg().dt;
      char buf[400];
      int n = std::snprintf(buf, sizeof buf, "candidates %zu  dormant %d  build %.1f ms%s\ngoal: %s", o.c.size(), o.n_dormant, o.ms, o.rotate ? "  (rotate)" : "",
                            cur_k >= 0 ? (std::string(1, (char)('A' + cur_k)) + " #" + std::to_string(ctl_->cur_id())).c_str() : "none");
      if (dec_up_) {
        const int as = dec_up_->last_ans_step();
        n += std::snprintf(buf + n, sizeof buf - (size_t)n, "\ndecider: %s #%d  %.1f s ago  latency %.1f s  ans %ld/%ld%s%s", dec_up_->last_choice().empty() ? "-" : dec_up_->last_choice().c_str(),
                           dec_up_->last_pick_id(), as >= 0 ? (env_->state().step - as) * dt_s : -1.0, dec_up_->last_ms() / 1000.0, dec_up_->n_ans(), dec_up_->n_sent(),
                           dec_up_->last_err().empty() ? "" : "  err: ", dec_up_->last_err().c_str());
        if (!dec_up_->last_trigger().empty() && n < (int)sizeof buf)
          n += std::snprintf(buf + n, sizeof buf - (size_t)n, "\nasked because: %s%s", dec_up_->last_trigger().c_str(), dec_up_->waiting() ? " (waiting)" : "");
      }
      ot.text = buf;
      if (!instruction_.empty()) ot.text += "\ngoal: " + instruction_ + " — " + goal_status_;   // 지시와 마지막 진행 (/explore/goal_status)
      pub_cand_txt_->publish(ot);
    }
  }

  // 물체 지도의 마지막으로 본 뒤 초: 물체 지도(/perception/object_map)가 낸 때 값(covariance[35]) + 그 뒤 지난 시간 (/perception/objects 는 10 Hz 로 늘 다시 나오므로 덧셈은 0.1 s 안팎)
  double seen_s_of(const vision_msgs::msg::Detection3D& d) const {
    const double age = objects_ ? std::max(0.0, (now() - rclcpp::Time(objects_->header.stamp, get_clock()->get_clock_type())).seconds()) : 0.0;
    return d.results[0].pose.covariance[35] + (objects_ && objects_->header.stamp.sec > 0 ? age : 0.0);
  }
  // 판단 모델 물체 후보 재료: 지도 물체 모두 (obj_stale_s 넘게 안 본 것 빼고), class_hint 클래스면 match 1
  std::vector<lxd::Params::Obj> map_objects() const {
    std::vector<lxd::Params::Obj> ob;
    for (const auto& d : objects_->detections) {
      if (d.results.empty()) continue;
      lxd::Params::Obj o;
      o.oid = d.id;
      o.cls = d.results[0].hypothesis.class_id;
      o.x = (float)d.bbox.center.position.x;
      o.y = (float)d.bbox.center.position.y;
      o.match = !target_class_.empty() && o.cls == target_class_ ? 1.f : 0.f;
      if (const auto it = sem_.find(o.oid); it != sem_.end()) o.sem = it->second;
      o.seen_s = (float)seen_s_of(d);
      if (o.seen_s > obj_stale_s_) continue;
      if (fails_.blocked(o.oid, o.cls, o.x, o.y, now().seconds())) continue;   // 못 간 물체는 잠시 뺌 (task_sup.h Fails)
      ob.push_back(o);
      if (is_follow_class(o.cls) && o.seen_s <= dec_vis_s_) { o.follow = true; ob.push_back(o); }   // 지금 보이는 사람: 따라가기 후보도
    }
    return ob;
  }
  bool is_follow_class(const std::string& c) const { for (const auto& f : follow_classes_) if (f == c) return true; return false; }
  // 따라가기 입력: 물체 지도의 따라갈 클래스 항목 (사람은 object_map 이 최신 측정으로 옮김)
  std::vector<ofl::Person> persons() const {
    std::vector<ofl::Person> v;
    if (!objects_) return v;
    for (const auto& d : objects_->detections)
      if (!d.results.empty() && is_follow_class(d.results[0].hypothesis.class_id))
        v.push_back({d.id, (float)d.bbox.center.position.x, (float)d.bbox.center.position.y, (float)seen_s_of(d)});
    return v;
  }
  // 따라가기 끝 (why 가 있으면 알림)
  void end_follow(const std::string& why) {
    if (!follower_.active() && why.empty()) return;
    follower_.stop();
    if (dec_up_) dec_up_->set_external(false);
    if (!why.empty()) say_goto(why);
  }
  // 따라가기 한 걸음: 아래층 명령을 정하고 (다시 묻기 · 포기 처리)
  void follow_step() {
    const sim::EnvState& e = env_->state();
    const float res = env_->cfg().res;
    const auto spot = [&](float x, float y, float px, float py, float* gx, float* gy) {
      const ofl::Params& fp = follower_.params();
      return ofl::spot_on_map(env_->lmap(), env_->cfg().size, world_.gw, world_.gh, world_.ox, world_.oy, res, x, y, px, py, fp.stop, fp.clear, fp.search, gx, gy);
    };
    const auto cell = [&](float x, float y) {
      const int cx = (int)std::floor((x - world_.ox) / res), cy = (int)std::floor((y - world_.oy) / res);
      return cx < 0 || cy < 0 || cx >= world_.gw || cy >= world_.gh ? -1 : cy * world_.gw + cx;
    };
    const ofl::Out o = follower_.step(now().seconds(), e.x + e.ex, e.y + e.ey, persons(), spot);
    follow_out_ = o;
    if (o.mode == ofl::kGiveUp) {
      env_->set_cmd(sim::LowCmd{-1, 1});
      end_follow("사람을 놓침: " + follower_.oid() + " — " + std::to_string((int)(follower_.params().lost_s + follower_.params().search_s)) + " s 못 찾아 멈춤");
      stop("paused", "따라가던 사람을 놓침 (찾기 시간 끝)", true);
      publish_follow();
      return;
    }
    const sim::LowCmd c = ofl::cmd_of(o, follower_.params(), cell);
    const sim::LowCmd& cur = env_->low_cmd();
    if (c.target != cur.target || c.follow != cur.follow || c.face != cur.face || c.w_idle != cur.w_idle || c.vmax != cur.vmax) {
      env_->set_cmd(c);
      publish_goal(c.target);
    }
    if (o.reask) dec_up_->ask("followed person moved");
    if (env_->state().step % lp_.decide_steps == 0) publish_follow();
  }
  // RViz candidates (ns follow): 따라가는 사람 (주황 원기둥 + 글 — 모드 · 거리), 없으면 지움
  void publish_follow() {
    if (!debug_) return;
    visualization_msgs::msg::MarkerArray ma;
    for (int k = 0; k < 2; ++k) {
      visualization_msgs::msg::Marker m;
      m.header.frame_id = map_frame_;
      m.header.stamp = now();
      m.ns = "follow";
      m.id = k;
      if (!follower_.active()) { m.action = m.DELETE; ma.markers.push_back(m); continue; }
      m.pose.position.x = follow_out_.px;
      m.pose.position.y = follow_out_.py;
      m.pose.orientation.w = 1.0;
      static const char* mn[5] = {"끔", "따라감", "섬", "찾는 중", "놓침"};
      if (k == 0) { m.type = m.CYLINDER; m.pose.position.z = 0.6; m.scale.x = m.scale.y = 0.4; m.scale.z = 1.2; m.color.r = 1.0f; m.color.g = 0.5f; m.color.a = 0.5f; }
      else {
        m.type = m.TEXT_VIEW_FACING; m.pose.position.z = 1.5; m.scale.z = 0.25; m.color.r = m.color.g = m.color.b = m.color.a = 1.0f;
        char b[96];
        std::snprintf(b, sizeof b, "%s %s %.1f m", follower_.oid().c_str(), follow_out_.searching ? mn[3] : mn[std::clamp(follow_out_.mode, 0, 4)], follow_out_.d);
        m.text = b;
      }
      ma.markers.push_back(m);
    }
    pub_cand_->publish(ma);
  }
  // 판단 모델 글에 넣을 아는 물체: 가까운 10 개 "chair-3 chair 1.2 m front-left seen 4 s" (+ " visible" — 지금 보임) — lxd::Upper::known_lines (시뮬과 같은 함수)
  std::vector<std::string> known_objects() const {
    const sim::EnvState& s = env_->state();
    std::vector<lxd::Upper::Known> k;
    for (const auto& d : objects_->detections)
      if (!d.results.empty())
        k.push_back({d.id, d.results[0].hypothesis.class_id, d.bbox.center.position.x, d.bbox.center.position.y, seen_s_of(d)});
    return lxd::Upper::known_lines(k, s.x + s.ex, s.y + s.ey, s.th + s.eth, dec_vis_s_);
  }
  // 판단 모델 요청의 물체 후보 글자 (A … — 프론티어 후보보다 앞 글자) 를 candidates 에 (ns cand_obj): 흰 바탕 원 + 글자 + 물체 이름
  void publish_object_cands() {
    if (!debug_) return;
    visualization_msgs::msg::MarkerArray ma;
    visualization_msgs::msg::Marker del;
    del.header.frame_id = map_frame_;
    del.ns = "cand_obj";
    del.action = visualization_msgs::msg::Marker::DELETEALL;
    ma.markers.push_back(del);
    const auto& ob = dec_up_->last_objects();
    for (size_t i = 0; i < ob.size() && (!instruction_.empty() || dec_obj_always_); ++i) {
      float x = 0.f, y = 0.f;
      if (!env_->cell_xy(ob[i].vp, &x, &y)) continue;
      for (int k = 0; k < 2; ++k) {
        visualization_msgs::msg::Marker m;
        m.header.frame_id = map_frame_;
        m.header.stamp = now();
        m.ns = "cand_obj";
        m.id = (int)i * 2 + k;
        m.pose.position.x = x;
        m.pose.position.y = y;
        m.pose.orientation.w = 1.0;
        if (k == 0) { m.type = m.CYLINDER; m.pose.position.z = 0.05; m.scale.x = m.scale.y = 0.42; m.scale.z = 0.02; m.color.r = m.color.g = m.color.b = 1.0f; m.color.a = 0.95f; }
        else { m.type = m.TEXT_VIEW_FACING; m.pose.position.z = 0.4; m.scale.z = 0.3; m.color.r = 0.1f; m.color.g = 0.1f; m.color.b = 0.1f; m.color.a = 1.0f;
               m.text = std::string(1, (char)('A' + (int)i)) + " " + ob[i].oid; }
        ma.markers.push_back(m);
      }
    }
    pub_cand_->publish(ma);
  }
  // 지도 좌표 → 탐사 지도 칸 (창 밖이면 −1)
  int xy_cell(float x, float y) const {
    const float res = env_->cfg().res;
    const int cx = (int)std::floor((x - world_.ox) / res), cy = (int)std::floor((y - world_.oy) / res);
    if (cx < 0 || cy < 0 || cx >= world_.gw || cy >= world_.gh) return -1;
    return cy * world_.gw + cx;
  }
  const vision_msgs::msg::Detection3D* find_object(const std::string& oid) const {
    if (!objects_) return nullptr;
    for (const auto& d : objects_->detections) if (d.id == oid && !d.results.empty()) return &d;
    return nullptr;
  }
  // 물체로 가기 명령 (task_sup.h — 판단 모델이 고른 물체의 설 자리 · 닿으면 물체 쪽으로 돌기)
  void apply_goto_cmd() {
    const sim::LowCmd c = goto_.cmd([this](float x, float y) { return xy_cell(x, y); });
    const sim::LowCmd& cur = env_->low_cmd();
    if (c.target != cur.target || c.follow != cur.follow || c.face != cur.face) {
      env_->set_cmd(c);
      publish_goal(c.target);
    }
  }
  void end_goto() {
    if (!goto_.active()) return;
    goto_.cancel();
    env_->set_cmd(sim::LowCmd{-1, 0});
    if (dec_up_) dec_up_->set_external(false);
  }
  void finish_goto() {
    if (dec_up_) dec_up_->set_external(false);
    env_->set_cmd(sim::LowCmd{-1, 0});
    instr_.stop();
    say_goto("도착: " + goto_.label() + (goto_resume_ ? " — 탐사를 이어 감" : " — 멈춤"));
    if (!goto_resume_) stop("paused", "물체에 도착 (goto_resume false)", true);
  }
  // 못 감 (막힘 · 경로 없음 — 뒤로 물러나지 않음): 그 물체를 잠시 후보에서 빼고 판단 모델에 다시 물음
  void fail_goto() {
    const double t = now().seconds();
    if (!goto_.oid().empty()) fails_.add(goto_.oid(), goto_.cls(), goto_.fx(), goto_.fy(), t);
    instr_.failed();
    if (dec_up_) {
      dec_up_->set_external(false);
      dec_up_->ask("could not reach " + (goto_.oid().empty() ? std::string("the spot") : goto_.oid()));
    }
    env_->set_cmd(sim::LowCmd{-1, 1});   // 목표 없음 — 다음 답까지 제자리
    char b[200];
    std::snprintf(b, sizeof b, "못 감: %s (%s) — 그 물체를 %.0f s 빼고 다시 물음 (이 지시 실패 %d/%d)", goto_.label().c_str(), goto_.why(), goto_.params().fail_ttl_s,
                  instr_.fails(), goto_.params().max_fail);
    say_goto(b);
  }
  void say_goto(const std::string& s) {
    std_msgs::msg::String m;
    m.data = s;
    pub_goal_status_->publish(m);
    goal_status_ = s;
    RCLCPP_INFO(get_logger(), "%s", s.c_str());
  }
  void publish_goal(int cell) {
    if (!debug_) return;
    visualization_msgs::msg::MarkerArray ma;
    visualization_msgs::msg::Marker mk;
    mk.header.frame_id = map_frame_;
    mk.header.stamp = now();
    mk.ns = "goal";
    mk.id = 0;
    float x = 0.f, y = 0.f;
    if (cell >= 0 && env_->cell_xy(cell, &x, &y)) {
      mk.type = visualization_msgs::msg::Marker::CYLINDER;
      mk.action = visualization_msgs::msg::Marker::ADD;
      mk.pose.position.x = x;
      mk.pose.position.y = y;
      mk.pose.orientation.w = 1.0;
      mk.scale.x = mk.scale.y = 0.3;
      mk.scale.z = 0.05;
      mk.color.g = 1.0f;
      mk.color.a = 0.8f;
    } else {
      mk.action = visualization_msgs::msg::Marker::DELETE;
    }
    ma.markers.push_back(mk);
    pub_goal_->publish(ma);
  }

  std::string env_yaml_, map_frame_, base_frame_, record_, state_, why_;
  bool dry_run_ = true, enabled_ = false, started_ = false, finished_ = false, stale_ = false, require_rc_ = true, rc_auto_ok_ = false;
  double tf_timeout_ = 0.2, depth_timeout_ = 0.5, input_timeout_ = 0.3, rc_timeout_ = 0.5, contact_resume_s_ = 0.0, max_speed_ = -1.0, contact_t_ = -1.0;
  double sum_step_ = 0, sum_obs_ = 0, sum_dec_ = 0, max_dec_ = 0;
  long n_ = 0, n_dec_ = 0;
  int rc_button_ = 1, rc_auto_ = 2, why_dec_ = 0;
  sim::EnvHostCfg hc_;
  sim::World world_;
  std::unique_ptr<RobotEnv> env_;
  lxc::Params lp_;
  std::unique_ptr<lxc::Ctl> ctl_;
  std::unique_ptr<lxd::Upper> dec_up_;
  std::unique_ptr<DeciderClient> dec_cli_;
  std::unique_ptr<lowrec::Writer> rec_;
  std::shared_ptr<tf2_ros::Buffer> tf_buf_;
  std::shared_ptr<tf2_ros::TransformListener> tf_lis_;
  sensor_msgs::msg::Imu::ConstSharedPtr imu_;
  nav_msgs::msg::Odometry::ConstSharedPtr odom_;
  sensor_msgs::msg::Image::ConstSharedPtr depth_;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr info_;
  sensor_msgs::msg::Joy::ConstSharedPtr rc_;
  rclcpp::Time last_scan_t_{0, 0, RCL_ROS_TIME}, last_tf_t_{0, 0, RCL_ROS_TIME}, odom_t_{0, 0, RCL_ROS_TIME};
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_scan_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  std::shared_ptr<limo_fire_perception::SharedImageSubscription> sub_depth_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_info_;
  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr sub_rc_;
  limo_fire_explore::RcWatch rc_watch_;
  std::string rc_topic_;
  double rc_resub_s_ = 2.0;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_cmd_vel_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr pub_cmd_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_status_, pub_state_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_dec_img_;
  // 판단 모델에 보낸 PNG 를 rgb8 로 풀어 decider_image 로 (보기만 — 요청에는 손대지 않음)
  void publish_dec_png(const std::vector<uint8_t>& png) {
    if (!debug_) return;
    png_image pi{};
    pi.version = PNG_IMAGE_VERSION;
    if (!png_image_begin_read_from_memory(&pi, png.data(), png.size())) return;
    pi.format = PNG_FORMAT_RGB;
    sensor_msgs::msg::Image im;
    im.data.resize(PNG_IMAGE_SIZE(pi));
    if (!png_image_finish_read(&pi, nullptr, im.data.data(), 0, nullptr)) { png_image_free(&pi); return; }
    im.header.stamp = now();
    im.header.frame_id = "map";
    im.width = pi.width;
    im.height = pi.height;
    im.encoding = "rgb8";
    im.step = pi.width * 3;
    pub_dec_img_->publish(im);
  }
  void publish_dec_placeholder() {
    sensor_msgs::msg::Image im;
    im.header.stamp = now();
    im.header.frame_id = "map";
    im.width = im.height = 288;
    im.encoding = "rgb8";
    im.step = im.width * 3;
    im.data.assign((size_t)im.step * im.height, 96);
    pub_dec_img_->publish(im);
  }
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_goal_status_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_goal_;
  std::string target_class_, goal_status_;
  Aliases aliases_;
  bool debug_ = false;
  double obj_stale_s_ = 600.0;
  rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_objects_, sub_query_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_query_;
  std::map<std::string, float> sem_;   // 지금 지시에 대한 물체별 뜻 맞음 확률 (object_search 답)
  rclcpp::Time query_t_;                 // 지금 지시를 object_search 에 보낸 때 (on_goal 이 now() 로 — 지시가 빈 동안은 안 견줌)
  std::vector<std::string> sem_strip_;  // decider_sem_strip
  vision_msgs::msg::Detection3DArray::ConstSharedPtr objects_;
  bool obj_dirty_ = false;
  ofl::Follower follower_;
  ofl::Out follow_out_;
  std::vector<std::string> follow_classes_;
  bool dec_obj_always_ = false;   // decider_obj_always
  double dec_vis_s_ = 1.0;        // decider_vis_s
  bool goto_resume_ = false;
  tsk::Goto goto_;
  tsk::Fails fails_;
  tsk::Instr instr_;
  long fb_obj_seen_ = 0;
  std::string instruction_;
  long n_png_pub_ = 0;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_done_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_goal_, pub_cand_;
  rclcpp::Publisher<rviz_2d_overlay_msgs::msg::OverlayText>::SharedPtr pub_cand_txt_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr srv_enable_;
  rclcpp::TimerBase::SharedPtr watchdog_, tf_retry_;
  struct Pend {   // TF 를 기다리는 장과 그때의 감각 값
    sensor_msgs::msg::LaserScan::ConstSharedPtr scan;
    nav_msgs::msg::Odometry::ConstSharedPtr odom;
    sensor_msgs::msg::Imu::ConstSharedPtr imu;
    sensor_msgs::msg::Image::ConstSharedPtr depth;
    sensor_msgs::msg::CameraInfo::ConstSharedPtr info;
  };
  std::deque<Pend> scans_;
  std::chrono::steady_clock::time_point head_t_;
};

}  // namespace limo_fire_explore

RCLCPP_COMPONENTS_REGISTER_NODE(limo_fire_explore::ExplorerNode)
