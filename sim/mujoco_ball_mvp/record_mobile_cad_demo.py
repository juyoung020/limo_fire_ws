"""Patrol, truth/FOV target cue, LIMO aiming animation and elastic-launcher shot."""
import argparse
import csv
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import launch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--preview', action='store_true', help='write preview PNGs without encoding the MP4')
parser.add_argument('--output', type=Path, default=Path.home() / 'Videos' / 'MuJoCo' / 'mobile_demo',
                    help='directory for generated video, scene, events and traces')
parser.add_argument('--drag-coefficient', type=float, default=0.45,
                    help='constant unspun-sphere Cd used for the shot (0 disables drag)')
parser.add_argument('--air-density', type=float, default=1.2044, help='air density in kg/m^3')
cli = parser.parse_args()
OUT = cli.output.expanduser().resolve()
OUT.mkdir(parents=True, exist_ok=True)
if cli.drag_coefficient < 0 or not math.isfinite(cli.drag_coefficient):
    parser.error('--drag-coefficient must be finite and nonnegative')
if cli.air_density <= 0 or not math.isfinite(cli.air_density):
    parser.error('--air-density must be finite and positive')


def geom(parent, **attrs):
    attrs.setdefault('contype', '0')
    attrs.setdefault('conaffinity', '0')
    return ET.SubElement(parent, 'geom', {k: str(v) for k,v in attrs.items()})


xml = ET.parse(ROOT / 'scene_elastic_launcher.xml')
world = xml.getroot().find('worldbody')
limo_root = world.find("body[@name='limo_yaw_root']")
limo_root.set('mocap', 'true')
limo_root.attrib.pop('pos')
pitch_root = world.find(".//body[@name='launcher_pitch_root']")
pitch_root.set('quat', '1 0 0 0')
ball_body = world.find("body[@name='ball']")
ball_radius = float(ball_body.find("geom[@name='ball_geom']").get('size')) * 1000
target_body = world.find("body[@name='fire_target']")
target_x, target_y = map(float, target_body.get('pos').split()[:2])
for mesh in xml.getroot().findall('./asset/mesh'):
    mesh.set('file', str(ROOT / mesh.get('file')))
# Visual room around the actual LIMO, launcher and flame target.
for name,pos,size in [
    ('north','0 1.55 0.2','2.5 0.055 0.2'),
    ('west','-2.45 0 0.2','0.055 1.55 0.2'),
    ('east','2.45 0 0.2','0.055 1.55 0.2'),
    ('island','0.25 -1.20 0.12','0.48 0.20 0.12'),
    ('divider','0.1 1.1 0.18','0.38 0.07 0.18'),
]:
    geom(world, name=name, type='box', pos=pos, size=size, rgba='0.32 0.39 0.47 1')
xml.write(OUT / 'mobile_scene.xml', encoding='utf-8')

# Record real integration states by calling the existing detector/step loop.
shot_speed,shot_pitch=4.2,15.0
args = launch.arguments(['--headless', '--elastic-launcher', '--scene', str(OUT/'mobile_scene.xml'),
                         '--speed',str(shot_speed),'--pitch',str(shot_pitch),
                         '--drag-coefficient',str(cli.drag_coefficient),
                         '--air-density',str(cli.air_density),
                          '--csv', str(OUT/'shot.csv'), '--summary', str(OUT/'shot_summary.json')])
states, models = [], []
original_step = mujoco.mj_step


def capture(model, data):
    if not states:
        models.append(model)
        states.append((float(data.time),data.qpos.copy(),data.qvel.copy()))
    original_step(model,data)
    states.append((float(data.time),data.qpos.copy(),data.qvel.copy()))


mujoco.mj_step = capture
try:
    result = launch.run(args)
finally:
    mujoco.mj_step = original_step
if not result['toppled']:
    raise RuntimeError('Actual shot did not topple target')

model = models[0]
model.vis.global_.offwidth, model.vis.global_.offheight = 1280,720
model.vis.headlight.ambient[:] = (0.4,0.4,0.4)
model.vis.headlight.diffuse[:] = (0.7,0.7,0.7)
data = mujoco.MjData(model)
ball_adr = model.joint('ball_free').qposadr[0]
ball_vel_adr = model.joint('ball_free').dofadr[0]
tilt_adr = model.joint('target_hinge').qposadr[0]
state_times = np.array([s[0] for s in states])


def wrap(a):
    return (a+math.pi)%(2*math.pi)-math.pi


# Patrol waypoints; the target sensor uses a truth/FOV proxy for 0.25 seconds.
path = [(-0.95,-1.1),(-0.95,0),(0.0,0)]
x,y,yaw = -1.8,-1.1,0.0
index = 0
travel = 0.0
dt = 1/60
seen_for = 0.0
detected = None
stopped = None
shot_time = None
timeline = []
events = []
for k in range(2400):
    t = k*dt
    speed = 0.0
    if t >= 1.0 and stopped is None:
        wx,wy=path[index]
        dx,dy=wx-x,wy-y
        distance=math.hypot(dx,dy)
        desired=math.atan2(dy,dx)
        error=wrap(desired-yaw)
        yaw += max(-1.2*dt,min(1.2*dt,error))
        if abs(error)<0.08:
            speed=min(0.45,distance/dt)
            x += speed*dt*math.cos(yaw)
            y += speed*dt*math.sin(yaw)
            travel += speed*dt
        if distance<0.008:
            x,y=wx,wy
            if index<len(path)-1:
                index+=1
            else:
                stopped=t
                yaw=0
                events.append(dict(event='stopped',time_s=t))
    scan=math.radians(55)*math.sin(t*2.0)
    target_yaw=math.atan2(target_y-y,target_x-x)
    sensor_yaw = yaw+scan if detected is None else target_yaw
    target_range=math.hypot(target_x-x,target_y-y)
    visible=target_range<=2.3 and abs(wrap(target_yaw-sensor_yaw))<=math.radians(25)
    if detected is None:
        seen_for = seen_for+dt if visible else 0.0
        if seen_for>=0.25:
            detected=t
            events.append(dict(event='truth_fov_detected',time_s=t,range_m=target_range))
    pitch=0.0
    stage='SEARCH'
    if detected is not None:
        stage='DETECTED' if t-detected<0.8 else 'APPROACH'
    if stopped is not None:
        stage='AIM'
        aim_f=min(1.0,max(0.0,(t-stopped)/1.2))
        aim_f=aim_f*aim_f*(3-2*aim_f)
        yaw_start=math.radians(-25.0)
        yaw=yaw_start+(target_yaw-yaw_start)*aim_f
        pitch=math.radians(-5.0+20.0*aim_f)
        sensor_yaw=yaw
        if t-stopped>=2.5 and detected is not None:
            shot_time=t
            events.append(dict(event='fired',time_s=t,speed_m_s=shot_speed,pitch_deg=shot_pitch))
            timeline.append(dict(t=t,x=x,y=y,yaw=yaw,pan=sensor_yaw,pitch=pitch,
                                 travel=travel,stage='FIRE',speed=0,range_m=target_range))
            break
    timeline.append(dict(t=t,x=x,y=y,yaw=yaw,pan=sensor_yaw,pitch=pitch,
                         travel=travel,stage=stage,speed=speed,range_m=target_range))
if shot_time is None:
    raise RuntimeError('Patrol did not reach detected + stopped + aimed firing condition')
with (OUT/'mission.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(timeline[0]))
    writer.writeheader();writer.writerows(timeline)
events.append(dict(event='topple_confirmed',time_s=shot_time+result['toppled_at_s']))
(OUT/'events.json').write_text(json.dumps(dict(events=events,
    movement='prescribed kinematic waypoint patrol; no wheel dynamics or obstacle avoidance',
        detection='truth position + 2.3 m range + 50 degree FOV + 0.25 second hold; YOLO image inference is not connected',
    shot='elastic-launcher MuJoCo physics, documented mechanism animation, unchanged 45 degree / 0.20 second detector'),indent=2),encoding='utf-8')

camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
font=r'C:\Windows\Fonts\malgun.ttf'
large=ImageFont.truetype(font,30);normal=ImageFont.truetype(font,23);small=ImageFont.truetype(font,18)
labels={'SEARCH':'순찰하며 표적 탐색','DETECTED':'표적 탐지(모의 센서)','APPROACH':'발사 위치로 접근',
        'AIM':'차체 회전 · 발사기 조준','FIRE':'공 발사','FLIGHT':'공 비행 중',
        'CONTACT':'표적 명중','SUCCESS':'전도 성공'}


def decorate(renderer, mission_t, slow=False):
    pre=timeline[min(len(timeline)-1,max(0,int(mission_t/dt)))]
    x,y,yaw=pre['x'],pre['y'],pre['yaw']
    pitch=pre['pitch']
    phase=pre['stage']
    flight=max(0,mission_t-shot_time)
    si=max(0,min(len(states)-1,int(np.searchsorted(state_times,flight,side='right')-1)))
    stamp,qpos,qvel=states[si]
    data.time=stamp if mission_t>=shot_time else max(0.0,mission_t-stopped)
    data.qpos[:]=qpos;data.qvel[:]=qvel
    if mission_t>=shot_time:
        phase='FIRE'
        if flight>=0.035:phase='FLIGHT'
        if flight>=result['contact_time_s']:phase='CONTACT'
        if flight>=result['toppled_at_s']:phase='SUCCESS'
    args.yaw=math.degrees(yaw);args.pitch=math.degrees(pitch)
    yaw_quat,pitch_quat,_=launch.launcher_aim_quats(args)
    root_id=model.body('limo_yaw_root').id
    root_mocap=int(model.body_mocapid[root_id])
    data.mocap_pos[root_mocap]=(x,y,0.0)
    data.mocap_quat[root_mocap]=yaw_quat
    model.body_quat[model.body('launcher_pitch_root').id]=pitch_quat
    launch.animate_launcher(model,data,args)
    for name in ('launcher_lever_visual','launcher_sear_visual','launcher_plunger_visual'):
        mid=int(model.body_mocapid[model.body(name).id])
        data.mocap_pos[mid,:2] += (x,y)
    if mission_t<shot_time:
        data.qpos[ball_adr:ball_adr+3]=np.array([x,y,0.0])+launch.launcher_local_to_world(args,(0.042,0.0,0.071))
        data.qvel[ball_vel_adr:ball_vel_adr+6]=0.0
    mujoco.mj_forward(model,data)
    blend=max(0,min(1,(mission_t-stopped)/1.2))
    blend=blend*blend*(3-2*blend)
    camera.lookat[:]=(-.2+.9*blend,-.1+.1*blend,.14+.08*blend)
    camera.distance=5.5-2.7*blend
    camera.azimuth=115-25*blend
    camera.elevation=-43+15*blend
    ball_world=data.qpos[ball_adr:ball_adr+3].copy()
    if shot_time<=mission_t<shot_time+0.45:
        focus=(ball_world+np.array([1.22,0,.36]))/2
        flight_blend=min(1,(mission_t-shot_time)/0.12)
        camera.lookat[:]=focus
        camera.distance=3.4-0.9*flight_blend
        camera.azimuth=90
        camera.elevation=-24
    renderer.update_scene(data,camera=camera)
    if shot_time<=mission_t<shot_time+0.36:
        # Render-only trail follows recorded integration states; it never changes physics.
        trail_start=max(0,si-80)
        sample_indices=list(range(trail_start,si,8))
        points=[states[j][1][ball_adr:ball_adr+3].copy() for j in sample_indices]
        points.append(ball_world)
        for j in range(len(points)-1):
            geom_id=renderer.scene.geoms[renderer.scene.ngeom]
            alpha=.3+.7*(j+1)/max(1,len(points)-1)
            mujoco.mjv_initGeom(geom_id,mujoco.mjtGeom.mjGEOM_LINE,np.zeros(3),np.zeros(3),
                                np.eye(3).flatten(),np.array([1,.27,.035,alpha]))
            mujoco.mjv_connector(geom_id,mujoco.mjtGeom.mjGEOM_LINE,5,points[j],points[j+1])
            renderer.scene.ngeom+=1
    # FOV rays are visual markers only, not physical contacts.
    if mission_t<shot_time:
        for angle in (yaw-math.radians(25),yaw,yaw+math.radians(25)):
            g=renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_LINE,np.zeros(3),np.zeros(3),np.eye(3).flatten(),np.array([0.1,.85,.85,.55]))
            mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_LINE,2,np.array([x,y,.025]),
                                np.array([x+2.3*math.cos(angle),y+2.3*math.sin(angle),.025]))
            renderer.scene.ngeom+=1
    picture=Image.fromarray(renderer.render());draw=ImageDraw.Draw(picture,'RGBA')
    draw.rectangle((0,0,1280,106),fill=(12,21,32,242))
    draw.text((28,12),'LIMO 순찰 · 표적 탐지 · 본체 조준 · 발사',font=large,fill='white')
    draw.text((30,61),'실제 LIMO / 프린트 발사장치 / 불꽃 표적 모델 · 탐지는 truth/FOV 모의 센서',font=normal,fill=(185,211,230))
    draw.rounded_rectangle((990,22,1250,73),radius=12,fill=(30,120,91,240) if phase=='SUCCESS' else (37,76,115,240))
    draw.text((1007,32),'0.1× 느린 재생' if slow else labels[phase],font=normal,fill='white')
    draw.rectangle((0,600,1280,720),fill=(12,21,32,245))
    tilt=abs(math.degrees(float(data.qpos[tilt_adr])))
    shown_speed=pre['speed'] if mission_t<shot_time else float(np.linalg.norm(data.qvel[ball_vel_adr:ball_vel_adr+3]))
    draw.text((30,612),f"{labels[phase]}   |   시간 {mission_t:5.2f}s   |   속도 {shown_speed:.2f}m/s   |   표적 기울기 {tilt:.1f}°",font=normal,fill='white')
    text='실제 YOLO 추론은 미연결 · 사전 설정 경로 주행 및 모의 시야 탐지'
    if mission_t>=shot_time:text=f'발사 {shot_speed:.1f}m/s · Pitch {shot_pitch:.0f}° · Cd {cli.drag_coefficient:.2f}  |  궤적은 MuJoCo 계산'
    if phase=='SUCCESS':text='전도 판정: 45° 이상 기울기를 0.20초 연속 유지'
    draw.text((30,650),text,font=small,fill=(175,207,227))
    draw.text((30,683),'탁구공 40 mm / 2.7 g · 추정 설계 모델',font=small,fill=(155,172,191))
    return np.array(picture)


with mujoco.Renderer(model,height=720,width=1280) as renderer:
    preview_times=[1.5,detected+.3,stopped+.8,shot_time+.7,
                   shot_time+result['toppled_at_s']+0.4]
    for i,t in enumerate(preview_times):
        Image.fromarray(decorate(renderer,t)).save(OUT/f'preview_{i}.png')
    if not cli.preview:
        destination=OUT/'mobile_search_aim_fire.mp4'
        writer=imageio_ffmpeg.write_frames(str(destination),(1280,720),fps=30,codec='libx264',
                     quality=8,pix_fmt_out='yuv420p',output_params=['-movflags','+faststart'])
        writer.send(None);n=0
        try:
            for frame in range(math.ceil((shot_time+3)*30)):
                writer.send(decorate(renderer,frame/30));n+=1
            for frame in range(144):
                writer.send(decorate(renderer,shot_time+frame/300,slow=True));n+=1
            for frame in range(60):
                writer.send(decorate(renderer,shot_time+1.2));n+=1
        finally:
            writer.close()
        metadata=dict(video=str(destination),frames=n,duration_s=n/30,fps=30,
                      shot_time_s=shot_time,detected_at_s=detected,stopped_at_s=stopped,
                      toppled=result['toppled'],confirmed_at_s=shot_time+result['toppled_at_s'],
                      slow_motion_factor=0.1,
                      cad_source='provided launcher and flame-target STL files; AgileX LIMO visual meshes',
                      cad_scale='source launcher and robot meshes converted from millimetres to metres',ball_diameter_m=ball_radius*2/1000,
                      ball_mass_kg=0.0027,launcher_elevation_deg=shot_pitch,launcher_speed_m_s=shot_speed,
                      drag_coefficient=cli.drag_coefficient,air_density_kg_m3=cli.air_density,
                      drag_model='constant Cd quadratic drag, unspun sphere; sensitivity assumption, not calibrated',
                      video_ball_color='white MuJoCo sphere; orange trail is render-only',
                      trajectory='orange 3D line segments from recorded MuJoCo states; render-only',
                      attribution='LIMO source: AgileX Robotics limo_ros2; launcher and target from user-provided STL files',
                      moving_base_model='prescribed kinematic waypoint patrol',
                      detector='truth position + 2.3m range + 50 degree FOV; YOLO image inference not connected',
                      launcher_model='elastic launcher visual sequence; force/band dynamics not simulated')
        (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        print(json.dumps(metadata,indent=2))
print('Events:',json.dumps(events))
