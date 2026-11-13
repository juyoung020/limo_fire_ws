"""Build a clearly labeled overview from the Wego MPPI results and MuJoCo GUI replay."""

from pathlib import Path
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "mujoco-gui-demo.mp4"
OUTPUT = HERE / "wego-nav-mujoco-workflow-overview.mp4"
WIDTH, HEIGHT, FPS = 1440, 830, 30
FONT = Path(r"C:\Windows\Fonts\malgun.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\malgunbd.ttf")
BG = (12, 22, 38)
CYAN = (61, 210, 220)
WHITE = (244, 247, 250)
MUTED = (183, 197, 211)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT), size=size)


def card(path: Path, eyebrow: str, title: str, lines: list[str], footer: str) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((68, 64, WIDTH - 68, HEIGHT - 64), radius=30, fill=(20, 37, 59), outline=(45, 73, 101), width=2)
    draw.rounded_rectangle((106, 105, 119, 153), radius=6, fill=CYAN)
    draw.text((141, 104), eyebrow, font=font(27, True), fill=CYAN)
    draw.text((106, 198), title, font=font(49, True), fill=WHITE)
    y = 302
    for line in lines:
        draw.ellipse((112, y + 11, 126, y + 25), fill=CYAN)
        draw.text((151, y), line, font=font(29), fill=WHITE)
        y += 65
    draw.line((106, HEIGHT - 169, WIDTH - 106, HEIGHT - 169), fill=(54, 77, 101), width=2)
    draw.text((106, HEIGHT - 137), footer, font=font(21), fill=MUTED)
    image.save(path)


def make_overlay(path: Path) -> None:
    image = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 76), fill=(8, 18, 31, 222))
    draw.text((38, 19), "MUJOCO GUI 재생", font=font(27, True), fill=CYAN)
    draw.text((WIDTH - 555, 23), "탐지 연출 · 조준 · 발사 · 전도", font=font(21), fill=WHITE)
    draw.rectangle((0, HEIGHT - 73, WIDTH, HEIGHT), fill=(8, 18, 31, 224))
    draw.text((38, HEIGHT - 54), "고정 경로와 시뮬레이터 표적 cue를 재생합니다. 실제 카메라 탐지/Nav2 연결 장면은 아닙니다.", font=font(19), fill=WHITE)
    image.save(path)


def run(ffmpeg: str, args: list[str]) -> None:
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory(prefix="wego-mujoco-video-") as tmp:
        temp = Path(tmp)
        intro_png, outro_png, overlay_png = temp / "intro.png", temp / "outro.png", temp / "overlay.png"
        intro_mp4, scene_mp4, outro_mp4 = temp / "intro.mp4", temp / "scene.mp4", temp / "outro.mp4"
        card(intro_png, "통합 흐름 개요 · Nav2 결과 + MuJoCo 시연", "탐색·이동에서 표적 전도까지", [
            "Wego / Nav2 MPPI: 지정 목표 2곳 주행 성공 (각 16초)",
            "MuJoCo: 탐지 cue 이후 조준·발사, 충돌 및 전도 재생",
            "두 실행은 별도 환경의 검증 결과를 영상으로 엮은 것입니다",
        ], "팀의 탐색·주행 방향과 발사 시뮬레이션을 한 편에서 확인합니다")
        card(outro_png, "검증 범위", "결과와 다음 연결 지점", [
            "Wego 실제 실행: MPPI waypoint 2/2 성공 · frontier 후보 1건 취소",
            "MuJoCo 재생: 45° 이상 0.20초 유지 조건으로 전도 판정 성공",
            "아직 미통합: 카메라/YOLO 탐지 결과를 Nav2 목표와 발사 제어에 연결",
        ], "이 영상은 통합 동작 제안/결과 요약이며, 단일 end-to-end 실험 영상은 아닙니다")
        make_overlay(overlay_png)
        run(ffmpeg, ["-loop", "1", "-framerate", str(FPS), "-i", str(intro_png), "-t", "6", "-r", str(FPS), "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p", str(intro_mp4)])
        run(ffmpeg, ["-i", str(SOURCE), "-i", str(overlay_png), "-filter_complex", f"[0:v]fps={FPS},scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2[base];[base][1:v]overlay=0:0,format=yuv420p[v]", "-map", "[v]", "-an", "-t", "36", "-r", str(FPS), "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p", str(scene_mp4)])
        run(ffmpeg, ["-loop", "1", "-framerate", str(FPS), "-i", str(outro_png), "-t", "7", "-r", str(FPS), "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p", str(outro_mp4)])
        concat = temp / "concat.txt"
        concat.write_text("".join(f"file '{p.as_posix()}'\n" for p in (intro_mp4, scene_mp4, outro_mp4)), encoding="utf-8")
        run(ffmpeg, ["-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", "-movflags", "+faststart", str(OUTPUT)])
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
