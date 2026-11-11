// sensor_view — ESP32 시험 코드(firmware/sensor_check)가 USB 시리얼로 내는 줄을 화면에 그린다.
//   열화상 "F,ta,t1..t768" (0.1 °C, GY-MCU90640 32×24) → 색 지도 + 색 막대 + 가장 뜨거운 곳 + 마우스 자리 온도
//   레이더 "T,x1,y1,v1,x2,y2,v2,x3,y3,v3" (mm, cm/s, HLK-LD2450) → 시야 부채꼴 위 표적 점
//   상태 "S,..." → 아래 글줄
// 받은 줄 종류에 따라 화면이 바뀐다. 포트를 안 주면 /dev/ttyACM* 첫 번째, 끊기면 1 초마다 다시 연다.
// 키: m = 열화상 좌우 반전 · f = 색 범위 고정(15~40 °C)/자동 · q = 끝
#include <fcntl.h>
#include <glob.h>
#include <termios.h>
#include <unistd.h>

#include <QApplication>
#include <QImage>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QTimer>
#include <QWidget>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kCols = 32, kRows = 24;     // MLX90640
constexpr int kImgW = 640, kImgH = 480;   // 열화상 그림 (20 배)
constexpr int kBarW = 90;                 // 색 막대
constexpr int kInfoH = 110;               // 아래 글줄
constexpr double kRadarRangeMm = 6000.0;  // LD2450 최대 6 m, 좌우 ±60° (매뉴얼 2.1)
constexpr double kRadarFovDeg = 60.0;

// 시리얼 포트 (POSIX). 열기·읽기 실패는 상태 글로 알리고 다시 연다
class Serial {
 public:
  Serial(std::string port, int baud) : want_(std::move(port)), baud_(baud) {}
  ~Serial() { close_fd(); }

  // 새로 받은 완성된 줄들
  std::vector<std::string> poll(std::string& status) {
    std::vector<std::string> lines;
    if (fd_ < 0 && !open_port(status)) return lines;
    char buf[8192];
    for (;;) {
      const ssize_t n = ::read(fd_, buf, sizeof(buf));
      if (n > 0) {
        pending_.append(buf, static_cast<size_t>(n));
        continue;
      }
      if (n == 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) {   // 보드가 빠짐
        status = "연결 끊김, 다시 연결 중… (" + port_ + ")";
        close_fd();
      }
      break;
    }
    size_t start = 0, nl;
    while ((nl = pending_.find('\n', start)) != std::string::npos) {
      std::string l = pending_.substr(start, nl - start);
      if (!l.empty() && l.back() == '\r') l.pop_back();
      lines.push_back(std::move(l));
      start = nl + 1;
    }
    pending_.erase(0, start);
    if (pending_.size() > 65536) pending_.clear();   // 줄바꿈 없는 쓰레기
    return lines;
  }
  const std::string& port() const { return port_; }

 private:
  bool open_port(std::string& status) {
    const auto now = std::chrono::steady_clock::now();
    if (now - last_try_ < std::chrono::seconds(1)) return false;
    last_try_ = now;
    port_ = want_;
    if (port_.empty()) {
      glob_t g{};
      if (::glob("/dev/ttyACM*", 0, nullptr, &g) == 0 && g.gl_pathc > 0) port_ = g.gl_pathv[0];
      globfree(&g);
      if (port_.empty()) {
        status = "보드 없음 (/dev/ttyACM* 이 없음) — USB 를 꽂아 주세요";
        return false;
      }
    }
    fd_ = ::open(port_.c_str(), O_RDONLY | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) {
      status = port_ + " 못 엶: " + std::strerror(errno) + (errno == EACCES ? " (dialout 그룹?)" : "");
      return false;
    }
    termios t{};
    tcgetattr(fd_, &t);
    cfmakeraw(&t);
    const speed_t sp = baud_ == 115200 ? B115200 : baud_ == 460800 ? B460800 : B921600;
    cfsetispeed(&t, sp);
    cfsetospeed(&t, sp);
    tcsetattr(fd_, TCSANOW, &t);
    pending_.clear();
    status.clear();
    return true;
  }
  void close_fd() {
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
  }

  std::string want_, port_, pending_;
  int baud_, fd_ = -1;
  std::chrono::steady_clock::time_point last_try_{};
};

// 열화상에서 흔히 쓰는 "아이언" 색: 검정 → 남색 → 보라 → 빨강 → 주황 → 노랑 → 흰색
std::array<QRgb, 256> make_lut() {
  struct Stop { double x; int r, g, b; };
  const Stop s[] = {{0.0, 0, 0, 10},     {0.15, 30, 0, 100},  {0.35, 140, 0, 150}, {0.55, 230, 40, 40},
                    {0.75, 255, 140, 0}, {0.9, 255, 230, 50}, {1.0, 255, 255, 255}};
  std::array<QRgb, 256> lut{};
  for (int i = 0; i < 256; ++i) {
    const double x = i / 255.0;
    for (size_t k = 0; k + 1 < std::size(s); ++k) {
      if (x <= s[k + 1].x) {
        const double f = (x - s[k].x) / (s[k + 1].x - s[k].x);
        lut[i] = qRgb(int(s[k].r + f * (s[k + 1].r - s[k].r)), int(s[k].g + f * (s[k + 1].g - s[k].g)),
                      int(s[k].b + f * (s[k + 1].b - s[k].b)));
        break;
      }
    }
  }
  return lut;
}

bool parse_ints(const std::string& line, std::vector<int>& out) {
  out.clear();
  const char* p = line.c_str() + 2;   // "F," · "T," 뒤
  while (*p) {
    char* end;
    const long v = std::strtol(p, &end, 10);
    if (end == p) return false;
    out.push_back(static_cast<int>(v));
    p = (*end == ',') ? end + 1 : end;
    if (*end != ',' && *end) return false;
  }
  return true;
}

class View : public QWidget {
 public:
  View(std::string port, int baud) : serial_(std::move(port), baud), lut_(make_lut()) {
    setWindowTitle("sensor_view — 열화상 · 레이더");
    setFixedSize(kImgW + kBarW, kImgH + kInfoH);
    setMouseTracking(true);
    auto* t = new QTimer(this);
    connect(t, &QTimer::timeout, this, [this] { tick(); });
    t->start(30);
    fps_t0_ = std::chrono::steady_clock::now();
  }

 protected:
  void keyPressEvent(QKeyEvent* e) override {
    if (e->key() == Qt::Key_M) mirror_ = !mirror_;
    else if (e->key() == Qt::Key_F) fixed_ = !fixed_;
    else if (e->key() == Qt::Key_Q) close();
    update();
  }
  void mouseMoveEvent(QMouseEvent* e) override { mouse_ = e->pos(); }
  void leaveEvent(QEvent*) override { mouse_ = QPoint(-1, -1); }

  void paintEvent(QPaintEvent*) override {
    QPainter p(this);
    p.fillRect(rect(), QColor(17, 17, 17));
    p.setRenderHint(QPainter::Antialiasing);
    QFont f = p.font();
    f.setPixelSize(15);
    f.setBold(true);
    p.setFont(f);
    QString info;
    if (mode_ == Mode::Thermal) info = draw_thermal(p);
    else if (mode_ == Mode::Radar) info = draw_radar(p);
    else info = "아직 데이터 없음 — 열화상(F 줄)·레이더(T 줄) 시험 코드가 올라간 보드를 꽂아 주세요";
    const QString st = QString::fromStdString(status_.empty() ? serial_.port() : status_);
    p.setPen(QColor(220, 220, 220));
    p.drawText(QRect(8, kImgH + 6, width() - 16, kInfoH - 6), Qt::AlignLeft | Qt::TextWordWrap,
               info + QString("\n%1   %2 프레임/s").arg(st).arg(fps_, 0, 'f', 1));
  }

 private:
  enum class Mode { None, Thermal, Radar };

  void tick() {
    for (const auto& l : serial_.poll(status_)) {
      if (l.rfind("F,", 0) == 0 && parse_ints(l, ints_) && ints_.size() == 1 + kCols * kRows) {
        ta_ = ints_[0] / 10.0;
        for (int i = 0; i < kCols * kRows; ++i) temp_[i] = ints_[1 + i] / 10.0;
        mode_ = Mode::Thermal;
        ++frames_;
      } else if (l.rfind("T,", 0) == 0 && parse_ints(l, ints_) && ints_.size() == 9) {
        for (int k = 0; k < 9; ++k) radar_[k] = ints_[k];
        mode_ = Mode::Radar;
        ++frames_;
      } else if (l.rfind("S,", 0) == 0) {
        sline_ = QString::fromStdString(l.substr(2));
      }
    }
    const auto now = std::chrono::steady_clock::now();
    const double dt = std::chrono::duration<double>(now - fps_t0_).count();
    if (dt >= 1.0) {
      fps_ = (frames_ - fps_frames_) / dt;
      fps_frames_ = frames_;
      fps_t0_ = now;
    }
    update();
  }

  // 매뉴얼: 점 순서 (Col1,Row1)…(Col32,Row1),(Col1,Row2)… 이고 (Col1,Row1) 은 그림의 오른쪽 위 → 기본은 좌우를 뒤집어 그림
  double at(int r, int c) const { return temp_[r * kCols + (mirror_ ? kCols - 1 - c : c)]; }

  QString draw_thermal(QPainter& p) {
    // 그림용 2×2 이웃 평균: MLX90640 은 화소 절반씩(체스판) 번갈아 읽어 두 묶음 사이에 0.1~0.3 °C 차 → 바둑판 무늬가 사라진다.
    // 숫자(최고·최저·마우스)는 원래 값
    double sm[kRows][kCols], lo = 1e9, hi = -1e9, rlo = 1e9, rhi = -1e9;
    int hr = 0, hc = 0;
    for (int r = 0; r < kRows; ++r)
      for (int c = 0; c < kCols; ++c) {
        const int r1 = std::min(r + 1, kRows - 1), c1 = std::min(c + 1, kCols - 1);
        sm[r][c] = (at(r, c) + at(r, c1) + at(r1, c) + at(r1, c1)) / 4;
        lo = std::min(lo, sm[r][c]);
        hi = std::max(hi, sm[r][c]);
        rlo = std::min(rlo, at(r, c));
        if (at(r, c) > rhi) { rhi = at(r, c); hr = r; hc = c; }
      }
    if (fixed_) { lo = 15.0; hi = 40.0; }
    const double span = std::max(0.5, hi - lo);
    QImage small(kCols, kRows, QImage::Format_RGB32);
    for (int r = 0; r < kRows; ++r)
      for (int c = 0; c < kCols; ++c)
        small.setPixel(c, r, lut_[std::clamp(int(255 * (sm[r][c] - lo) / span), 0, 255)]);
    p.drawImage(QRect(0, 0, kImgW, kImgH), small.scaled(kImgW, kImgH, Qt::IgnoreAspectRatio, Qt::SmoothTransformation));

    // 가장 뜨거운 곳 (초록 십자 + 온도), 가운데 (흰 원)
    const double x = (hc + 0.5) * kImgW / kCols, y = (hr + 0.5) * kImgH / kRows;
    p.setPen(QPen(QColor(0, 255, 120), 3));
    p.drawLine(QPointF(x - 18, y), QPointF(x + 18, y));
    p.drawLine(QPointF(x, y - 18), QPointF(x, y + 18));
    const QString hot = QString("최고 %1°C").arg(rhi, 0, 'f', 1);
    const QRectF tr(std::min(x + 14, kImgW - 110.0), std::max(4.0, y - 34), 104, 26);
    p.fillRect(tr, Qt::black);
    p.drawText(tr, Qt::AlignCenter, hot);
    p.setPen(QPen(Qt::white, 2));
    p.drawEllipse(QPointF(kImgW / 2.0, kImgH / 2.0), 6, 6);

    // 색 막대
    const int top = 30, bot = kImgH - 30;
    for (int yy = top; yy < bot; ++yy) {
      p.setPen(QColor(lut_[std::clamp(255 * (bot - yy) / (bot - top), 0, 255)]));
      p.drawLine(kImgW + 15, yy, kImgW + 35, yy);
    }
    p.setPen(QColor(230, 230, 230));
    p.drawText(kImgW + 10, 20, "온도");
    for (int k = 0; k <= 4; ++k) {
      const double fr = k / 4.0;
      p.drawText(kImgW + 40, int(bot - fr * (bot - top)) + 5, QString("%1°").arg(lo + fr * span, 0, 'f', span < 8 ? 1 : 0));
    }

    QString here;
    if (mouse_.x() >= 0 && mouse_.x() < kImgW && mouse_.y() >= 0 && mouse_.y() < kImgH) {
      const int c = mouse_.x() * kCols / kImgW, r = mouse_.y() * kRows / kImgH;
      here = QString("   마우스 자리 %1°C").arg(at(r, c), 0, 'f', 1);
    }
    return QString("열화상  최저 %1°C  최고 %2°C  가운데 %3°C%4   센서 자체 %5°C   %6\n"
                   "색: 검정·보라 = 차가움 → 빨강·노랑 = 따뜻함 → 흰색 = 가장 뜨거움 │ 초록 십자 = 가장 뜨거운 곳 │ 흰 원 = 가운데 │ "
                   "m 좌우 반전(%7) · f 색 범위(%8) · q 끝")
        .arg(rlo, 0, 'f', 1).arg(rhi, 0, 'f', 1).arg(at(kRows / 2, kCols / 2), 0, 'f', 1).arg(here).arg(ta_, 0, 'f', 1)
        .arg(sline_).arg(mirror_ ? "켬" : "끔").arg(fixed_ ? "15~40°C 고정" : "자동");
  }

  QString draw_radar(QPainter& p) {
    const double ox = (kImgW + kBarW) / 2.0, oy = kImgH - 30, scale = (kImgH - 60) / kRadarRangeMm;
    auto px = [&](double xmm, double ymm) { return QPointF(ox + xmm * scale, oy - ymm * scale); };
    p.setPen(QColor(85, 85, 85));
    for (int r = 1000; r <= int(kRadarRangeMm); r += 1000) {
      const double rp = r * scale;
      p.drawArc(QRectF(ox - rp, oy - rp, 2 * rp, 2 * rp), int((90 - kRadarFovDeg) * 16), int(2 * kRadarFovDeg * 16));
      p.drawText(QPointF(ox + 4, oy - rp - 4), QString("%1 m").arg(r / 1000));
    }
    for (double a : {-kRadarFovDeg, -30.0, 0.0, 30.0, kRadarFovDeg}) {
      const double rad = a * M_PI / 180.0;
      p.drawLine(QPointF(ox, oy), px(kRadarRangeMm * std::sin(rad), kRadarRangeMm * std::cos(rad)));
    }
    p.setBrush(QColor(255, 204, 0));
    p.setPen(Qt::NoPen);
    p.drawEllipse(QPointF(ox, oy), 6, 6);
    const QColor colors[3] = {QColor(230, 25, 75), QColor(60, 180, 75), QColor(67, 99, 216)};
    QString lines;
    for (int k = 0; k < 3; ++k) {
      const int x = radar_[3 * k], y = radar_[3 * k + 1], v = radar_[3 * k + 2];
      if (x == 0 && y == 0) {
        lines += QString("[%1] 없음   ").arg(k + 1);
        continue;
      }
      p.setBrush(colors[k]);
      p.setPen(QPen(Qt::white, 1));
      p.drawEllipse(px(x, y), 10, 10);
      p.drawText(px(x, y) + QPointF(-4, -14), QString::number(k + 1));
      lines += QString("[%1] x %2 mm  y %3 mm  %4 m  %5°  %6 cm/s   ")
                   .arg(k + 1).arg(x).arg(y).arg(std::hypot(x, y) / 1000.0, 0, 'f', 2)
                   .arg(std::atan2(x, y) * 180.0 / M_PI, 0, 'f', 1).arg(v);
    }
    return "레이더  " + lines + "\n노란 점 = 레이더(앞면 = 금색 패치 쪽) │ 부채꼴 = 시야 ±60°, 6 m │ 점 = 표적(최대 3) │ q 끝";
  }

  Serial serial_;
  std::array<QRgb, 256> lut_;
  std::string status_;
  QString sline_;
  std::vector<int> ints_;
  std::array<double, kCols * kRows> temp_{};
  std::array<int, 9> radar_{};
  double ta_ = 0.0, fps_ = 0.0;
  long frames_ = 0, fps_frames_ = 0;
  std::chrono::steady_clock::time_point fps_t0_;
  Mode mode_ = Mode::None;
  bool mirror_ = true, fixed_ = false;
  QPoint mouse_{-1, -1};
};

}  // namespace

int main(int argc, char** argv) {
  QApplication app(argc, argv);
  std::string port;
  int baud = 921600;   // 열화상 시험 코드 USB 속도. USB 직결(CDC) 보드는 속도를 무시한다
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--baud" && i + 1 < argc) baud = std::atoi(argv[++i]);
    else if (a == "-h" || a == "--help") {
      std::printf("사용: sensor_view [포트(기본: /dev/ttyACM* 첫 번째)] [--baud 921600|460800|115200]\n");
      return 0;
    } else port = a;
  }
  View v(port, baud);
  v.show();
  return app.exec();
}
