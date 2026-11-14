import os
from PIL import Image, ImageDraw, ImageFont
import sim_gif as G
F = ImageFont.truetype(r"C:\Windows\Fonts\malgunbd.ttf", 22)
fr = G.scene(30.0, 30.0, 2, "v5: 레버(주황)가 핀을 밀어 장전 → 걸쇠(빨강)가 잡음 → 레버가 걸쇠를 눌러 발사",
             dist=0.24, look=(-0.06, 0.0, 0.05), step_fast=0.015, elev=0, az=270)
fr += G.scene(15.0, 15.0, 1, "v5 전체: 15° 발사", dist=0.48, look=(0.07, 0.0, 0.02), step_fast=0.03, elev=-6, az=90)
out = []
for im, du in fr:
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((600, 470, 950, 590), 10, fill=(20, 20, 24))
    for i, (t, c) in enumerate((("주황 = 레버(서보 축에 끼움)", (255, 150, 40)), ("빨강 = 걸쇠", (255, 90, 90)), ("노랑 줄 = 고무줄", (255, 230, 0)))):
        d.text((615, 480 + i * 34), t, font=F, fill=c)
    out.append((im.convert("P", palette=Image.ADAPTIVE, colors=128), max(du, 50)))
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "render", "v5_launch.gif")
out[0][0].save(p, save_all=True, append_images=[o[0] for o in out[1:]], duration=[o[1] for o in out], loop=0, optimize=True)
print(p, len(out), round(os.path.getsize(p) / 1e6, 1), "MB")
