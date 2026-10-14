"""Offline visual evidence from fitted SMPL and actual paired video/cloud."""
import json
from pathlib import Path
import cv2
import numpy as np


EDGES = [(0,1),(0,2),(0,3),(1,4),(2,5),(3,6),(4,7),(5,8),(6,9),(7,10),
         (8,11),(9,12),(9,13),(9,14),(12,15),(13,16),(14,17),(16,18),(17,19),
         (18,20),(19,21),(20,22),(21,23)]


def make_review(session, calibration, window_dir, max_frames=3):
    window_dir = Path(window_dir)
    rows = [json.loads(line) for line in (window_dir/'quality.jsonl').read_text(encoding='utf-8').splitlines()]
    if not rows:
        return None
    available = sorted({row['lidar_index'] for row in rows})
    selected = [available[i] for i in np.linspace(0,len(available)-1,min(max_frames,len(available))).astype(int)]
    output = window_dir/'review'
    output.mkdir(exist_ok=True)
    frames = []
    colors = [(20,160,255),(220,90,60),(80,220,80),(200,80,200)]
    for lidar_index in selected:
        athletes = [row for row in rows if row['lidar_index']==lidar_index]
        cameras = {}
        geometry = []
        for camera in ('cam1','cam2'):
            image = session.read_image(camera,athletes[0]['cameras'][camera]['record_index'])
            for k,row in enumerate(athletes):
                with np.load(row['fit_path']) as fit:
                    vertices = fit['vertices']; joints = fit['joints']; cloud = fit['source_cloud']
                uv, depth = calibration.project(cloud,camera)
                for point in uv[depth>0][::max(1,len(uv)//2000)]:
                    x,y = np.round(point).astype(int)
                    if 0<=x<image.shape[1] and 0<=y<image.shape[0]:
                        cv2.circle(image,(x,y),1,(0,255,255),-1)
                uv,depth = calibration.project(vertices[::6],camera)
                for point in uv[depth>0]:
                    x,y = np.round(point).astype(int)
                    if 0<=x<image.shape[1] and 0<=y<image.shape[0]:
                        cv2.circle(image,(x,y),1,colors[k%len(colors)],-1)
                juv,jdepth = calibration.project(joints,camera)
                for i,j in EDGES:
                    if min(jdepth[i],jdepth[j])>0:
                        cv2.line(image,tuple(np.round(juv[i]).astype(int)),tuple(np.round(juv[j]).astype(int)),
                                 colors[k%len(colors)],2,cv2.LINE_AA)
                if camera=='cam2':
                    geometry.append({'cloud':np.round(cloud,4).tolist(),'vertices':np.round(vertices[::3],4).tolist(),
                                     'joints':np.round(joints,4).tolist(),'accepted':row['accepted'],
                                     'mean_mm':round(row['qa']['final_p2surface_mean_m']*1000,1),
                                     'p90_mm':round(row['qa']['final_p2surface_p90_m']*1000,1),
                                     'reprojection_px':(round(row['qa']['reprojection_mean_px'],1)
                                                        if row['qa']['reprojection_mean_px'] is not None else None),
                                     'motion_id':row['motion_id']})
            cv2.putText(image,f'{camera} scan {lidar_index} | yellow: real LiDAR; orange/blue: SMPL',
                        (25,35),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
            name = f'{lidar_index:06d}_{camera}.jpg'
            cv2.imwrite(str(output/name),image)
            cameras[camera] = name
        frames.append({'lidar_index':lidar_index,'cameras':cameras,'athletes':geometry})
    projection_values = [r['qa']['reprojection_mean_px'] for r in rows if r['qa']['reprojection_mean_px'] is not None]
    summary = {'fitted':len(rows),'accepted':sum(r['accepted'] for r in rows),
               'mean_surface_mm':round(np.mean([r['qa']['final_p2surface_mean_m'] for r in rows])*1000,2),
               'mean_p90_surface_mm':round(np.mean([r['qa']['final_p2surface_p90_m'] for r in rows])*1000,2),
               'mean_reprojection_px':round(float(np.mean(projection_values)),2) if projection_values else None,
               'calibration_digest':calibration.data['snapshot_digest']}
    template = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>羽毛球 SMPL / LiDAR 配对检查</title>
<style>body{margin:24px;background:#121821;color:#e8edf3;font:15px system-ui}button,select{padding:8px;margin:5px;background:#263243;color:white;border:1px solid #62768c}#c{background:#0c1119;width:100%;height:580px}.layout{display:grid;grid-template-columns:1fr 1fr;gap:20px}img{width:100%}small{color:#afbdce}pre{white-space:pre-wrap}</style>
<h1>羽毛球 SMPL 与真实点云</h1><p>拖动旋转，滚轮缩放。黄色为真实 LiDAR 返回点；橙/蓝为拟合 SMPL。双相机叠加使用同一冻结 K/d/R/t。</p>
<select id="frame"></select><button id="cloud">点云 开/关</button><button id="mesh">SMPL 开/关</button>
<div class="layout"><div><canvas id="c"></canvas><pre id="metrics"></pre></div><div><img id="cam1"><img id="cam2"></div></div>
<small>标注属于视频/点云拟合伪标签。当前时钟是设备报告共同时间轴；曝光及逐点扫描运动补偿未验。背面及遮挡部位由视频先验约束。</small>
<script>const frames=__FRAMES__,summary=__SUMMARY__,edges=__EDGES__;
let yaw=.7,pitch=.15,scale=120,showCloud=true,showMesh=true,selected=0;
const canvas=document.getElementById('c'),ctx=canvas.getContext('2d');
const selector=document.getElementById('frame');frames.forEach((f,i)=>selector.add(new Option('LiDAR 扫描 '+f.lidar_index,i)));
function draw(){canvas.width=canvas.clientWidth*devicePixelRatio;canvas.height=canvas.clientHeight*devicePixelRatio;ctx.scale(devicePixelRatio,devicePixelRatio);ctx.clearRect(0,0,canvas.clientWidth,canvas.clientHeight);
const f=frames[selected],all=f.athletes.flatMap(a=>a.cloud),center=[0,1,2].map(k=>all.reduce((s,p)=>s+p[k],0)/all.length);
function project(p){const x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2];const u=x*Math.cos(yaw)-y*Math.sin(yaw),v=x*Math.sin(yaw)+y*Math.cos(yaw);return [canvas.clientWidth/2+u*scale,canvas.clientHeight/2-(z*Math.cos(pitch)-v*Math.sin(pitch))*scale];}
const colors=['#ffac32','#5096f8','#80dd80','#d889dd'];f.athletes.forEach((a,i)=>{if(showCloud){ctx.fillStyle='#ffff55';a.cloud.forEach(p=>{const q=project(p);ctx.fillRect(q[0],q[1],1.6,1.6)})}if(showMesh){ctx.fillStyle=colors[i%4];a.vertices.forEach(p=>{const q=project(p);ctx.fillRect(q[0],q[1],1,1)});ctx.strokeStyle=colors[i%4];ctx.lineWidth=2;edges.forEach(e=>{const p=project(a.joints[e[0]]),q=project(a.joints[e[1]]);ctx.beginPath();ctx.moveTo(...p);ctx.lineTo(...q);ctx.stroke()})}});
document.getElementById('cam1').src=f.cameras.cam1;document.getElementById('cam2').src=f.cameras.cam2;
document.getElementById('metrics').textContent=JSON.stringify({summary,frame:f.lidar_index,athletes:f.athletes.map(({cloud,vertices,joints,...a})=>a)},null,2);}
selector.onchange=()=>{selected=+selector.value;draw()};document.getElementById('cloud').onclick=()=>{showCloud=!showCloud;draw()};document.getElementById('mesh').onclick=()=>{showMesh=!showMesh;draw()};let drag=null;
canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};canvas.onpointerup=()=>drag=null;canvas.onpointermove=e=>{if(drag){yaw+=(e.clientX-drag[0])*.008;pitch+=(e.clientY-drag[1])*.008;drag=[e.clientX,e.clientY];draw()}};
canvas.onwheel=e=>{e.preventDefault();scale=Math.max(15,Math.min(600,scale*Math.exp(-e.deltaY*.001)));draw()};window.onresize=draw;draw();</script></html>'''
    html = template.replace('__FRAMES__',json.dumps(frames,ensure_ascii=False)).replace('__SUMMARY__',json.dumps(summary)).replace('__EDGES__',json.dumps(EDGES))
    (output/'index.html').write_text(html,encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return {'path':str(output/'index.html'),'summary':summary}
