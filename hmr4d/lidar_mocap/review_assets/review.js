/* Offline WebGL2 review. All geometry and photos come from recorded scans. */
'use strict';
const manifest=window.REVIEW_MANIFEST,$=id=>document.getElementById(id);
let clip=null,clipNumber=-1,index=0,playing=false,mediaTime=0,lastTick=0,loadGeneration=0;
let visibleMotion=null;
const embedded=new URLSearchParams(location.search).has('embedded');
if(embedded)document.body.classList.add('embedded');
const canvas=$('scene'),gl=canvas.getContext('webgl2',{antialias:true,preserveDrawingBuffer:true});
if(!gl){$('status').textContent='此浏览器未提供 WebGL2；请使用 Chrome 或 Edge。';throw new Error('WebGL2 unavailable');}
const decoder=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0)).buffer;
const floats=s=>new Float32Array(decoder(s));
const faces=new Uint16Array(decoder(manifest.faces));
const lineIndices=new Uint16Array(faces.length*2);
for(let i=0,j=0;i<faces.length;i+=3){const a=faces[i],b=faces[i+1],c=faces[i+2];lineIndices.set([a,b,b,c,c,a],j);j+=6;}
const vertex=`#version 300 es
layout(location=0) in vec3 aPosition;layout(location=1) in vec3 aNormal;
uniform mat4 uMatrix;uniform float uPointSize;out vec3 vNormal;
void main(){gl_Position=uMatrix*vec4(aPosition,1.0);gl_PointSize=uPointSize;vNormal=aNormal;}`;
const fragment=`#version 300 es
precision highp float;in vec3 vNormal;uniform vec4 uColor;uniform float uLit;uniform float uPoints;out vec4 outputColor;
void main(){if(uPoints>0.5&&distance(gl_PointCoord,vec2(.5))>.5)discard;
float shade=mix(1.0,.35+.65*max(0.0,dot(normalize(vNormal),normalize(vec3(-.4,-.8,1.0)))),uLit);
outputColor=vec4(uColor.rgb*shade,uColor.a);}`;
function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;}
const program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,vertex));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,fragment));gl.linkProgram(program);
if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(program));
gl.useProgram(program);const uniforms={};for(const k of ['uMatrix','uPointSize','uColor','uLit','uPoints'])uniforms[k]=gl.getUniformLocation(program,k);
const positionBuffer=gl.createBuffer(),normalBuffer=gl.createBuffer(),indexBuffer=gl.createBuffer();
let center=[0,0,0],distance=10,yaw=3.5,pitch=.26,defaultView=null;
const sub=(a,b)=>a.map((v,i)=>v-b[i]),dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const norm=a=>{const n=Math.hypot(...a)||1;return a.map(v=>v/n);};
function multiply(a,b){const m=new Float32Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)m[c*4+r]+=a[k*4+r]*b[c*4+k];return m;}
function matrix(){const eye=center.map((v,i)=>v+distance*[Math.cos(pitch)*Math.cos(yaw),Math.cos(pitch)*Math.sin(yaw),Math.sin(pitch)][i]);
 const f=norm(sub(center,eye)),s=norm(cross(f,[0,0,1])),u=cross(s,f);
 const view=new Float32Array([s[0],u[0],-f[0],0,s[1],u[1],-f[1],0,s[2],u[2],-f[2],0,-dot(s,eye),-dot(u,eye),dot(f,eye),1]);
 const n=.03,far=200,t=1/Math.tan(Math.PI/8),aspect=canvas.width/canvas.height;
 const projection=new Float32Array([t/aspect,0,0,0,0,t,0,0,0,0,(far+n)/(n-far),-1,0,0,2*far*n/(n-far),0]);return multiply(projection,view);}
function normals(positions){const n=new Float32Array(positions.length);
 for(let i=0;i<faces.length;i+=3){const a=faces[i]*3,b=faces[i+1]*3,c=faces[i+2]*3;
 const ab=[positions[b]-positions[a],positions[b+1]-positions[a+1],positions[b+2]-positions[a+2]],ac=[positions[c]-positions[a],positions[c+1]-positions[a+1],positions[c+2]-positions[a+2]],v=cross(ab,ac);
 for(const o of [a,b,c]){n[o]+=v[0];n[o+1]+=v[1];n[o+2]+=v[2];}}
 for(let i=0;i<n.length;i+=3){const len=Math.hypot(n[i],n[i+1],n[i+2])||1;n[i]/=len;n[i+1]/=len;n[i+2]/=len;}return n;}
function unpackFrame(frame){if(frame._decoded)return;frame._cloud=floats(frame.cloud);
 for(const a of frame.athletes){a._vertices=floats(a.vertices);a._joints=floats(a.joints);a._cloud=floats(a.cloud);a._normals=normals(a._vertices);}frame._decoded=true;}
function drawArray(points,color,mode,indices=null,normal=null,pointSize=2){
 gl.bindBuffer(gl.ARRAY_BUFFER,positionBuffer);gl.bufferData(gl.ARRAY_BUFFER,points,gl.STREAM_DRAW);gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);
 if(normal){gl.bindBuffer(gl.ARRAY_BUFFER,normalBuffer);gl.bufferData(gl.ARRAY_BUFFER,normal,gl.STREAM_DRAW);gl.enableVertexAttribArray(1);gl.vertexAttribPointer(1,3,gl.FLOAT,false,0,0);}else{gl.disableVertexAttribArray(1);gl.vertexAttrib3f(1,0,0,1);}
 gl.uniform4fv(uniforms.uColor,color);gl.uniform1f(uniforms.uLit,normal?1:0);gl.uniform1f(uniforms.uPoints,mode===gl.POINTS?1:0);gl.uniform1f(uniforms.uPointSize,pointSize*devicePixelRatio);
 if(indices){gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,indexBuffer);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,indices,gl.STREAM_DRAW);gl.drawElements(mode,indices.length,gl.UNSIGNED_SHORT,0);}else gl.drawArrays(mode,0,points.length/3);
}
const plane=manifest.reference_plane;
const floorLines=[];const po=plane.origin_viewer_m,pn=plane.normal_viewer;
for(let d=-10;d<=10;d++){for(const xy of [[po[0]+d,po[1]-10],[po[0]+d,po[1]+10],[po[0]-10,po[1]+d],[po[0]+10,po[1]+d]])floorLines.push(xy[0],xy[1],po[2]-((xy[0]-po[0])*pn[0]+(xy[1]-po[1])*pn[1])/pn[2]);}
function draw(){if(!clip)return;const frame=clip.frames[index];unpackFrame(frame);
 const athletes=frame.athletes.filter(a=>!visibleMotion||a.motion_id===visibleMotion);
 const w=Math.round(canvas.clientWidth*devicePixelRatio),h=Math.round(canvas.clientHeight*devicePixelRatio);if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
 gl.viewport(0,0,w,h);gl.clearColor(.035,.055,.085,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.uniformMatrix4fv(uniforms.uMatrix,false,matrix());
 drawArray(new Float32Array(floorLines),[.20,.27,.36,.55],gl.LINES);
 if($('mesh').checked){gl.enable(gl.CULL_FACE);gl.cullFace(gl.BACK);for(const a of athletes){drawArray(a._vertices,[...a.color.map(c=>c/255),Number($('opacity').value)],gl.TRIANGLES,faces,a._normals);}gl.disable(gl.CULL_FACE);}
 if($('wire').checked)for(const a of athletes)drawArray(a._vertices,[...a.color.map(c=>c/255),.6],gl.LINES,lineIndices);
 if($('background').checked)drawArray(frame._cloud,[.52,.60,.69,.65],gl.POINTS,null,null,1.4);
 if($('points').checked)for(const a of athletes)drawArray(a._cloud,[1,.94,.30,1],gl.POINTS,null,null,2.2);
 if($('skeleton')?.checked){gl.disable(gl.DEPTH_TEST);for(const a of athletes){
  const bones=new Float32Array(manifest.edges.flatMap(([x,y])=>[...a._joints.slice(x*3,x*3+3),...a._joints.slice(y*3,y*3+3)]));
  drawArray(bones,[...a.color.map(c=>c/255),1],gl.LINES);
  drawArray(a._joints.slice(0,3),[...a.color.map(c=>c/255),1],gl.POINTS,null,null,12);
  drawArray(a._joints.slice(0,3),[1,1,1,1],gl.POINTS,null,null,4);
 }gl.enable(gl.DEPTH_TEST);}
 window.REVIEW_STATE={clip:clip.id,frame:index,scan:frame.lidar_index,frameCount:clip.frames.length,duration:clip.duration_s,meshCount:frame.athletes.length,visibleMeshCount:athletes.length,visibleMotion,rejectedActors:frame.athletes.filter(a=>!a.accepted).map(a=>a.motion_id),vertexCount:frame.athletes.reduce((n,a)=>n+a._vertices.length/3,0),pointCount:frame.point_count,playing,mediaTime,webglError:gl.getError()};
 if(embedded)reportEmbedded();
}
function orient(){const min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];
 for(const f of clip.frames)for(const a of f.athletes){const p=floats(a.vertices);for(let i=0;i<p.length;i+=3)for(let k=0;k<3;k++){min[k]=Math.min(min[k],p[i+k]);max[k]=Math.max(max[k],p[i+k]);}}
 yaw=3.5;pitch=.26;
 if(!Number.isFinite(min[0])){center=po.slice();distance=18;}else{center=min.map((v,i)=>(v+max[i])/2);
  const e=[Math.cos(pitch)*Math.cos(yaw),Math.cos(pitch)*Math.sin(yaw),Math.sin(pitch)],s=norm(cross(e.map(v=>-v),[0,0,1])),u=cross(s,e.map(v=>-v));
  const tanY=Math.tan(Math.PI/8),tanX=tanY*canvas.clientWidth/canvas.clientHeight;distance=5;
  for(let mask=0;mask<8;mask++){const d=center.map((v,k)=>(mask&(1<<k)?max[k]:min[k])-v);distance=Math.max(distance,dot(e,d)+Math.max(Math.abs(dot(s,d))/tanX,Math.abs(dot(u,d))/tanY));}distance*=1.10;
 }
 defaultView={center:center.slice(),distance,yaw,pitch};
}
const fmt=(n,d=2)=>Number.isFinite(n)?n.toFixed(d):'未量化';
function showFrame(i){if(!clip)return;index=Math.max(0,Math.min(clip.frames.length-1,i));const f=clip.frames[index];$('timeline').value=index;
 if(f.cam1_image)$('cam1').src=f.cam1_image;if(f.cam2_image)$('cam2').src=f.cam2_image;
 $('clock').textContent=`${mediaTime.toFixed(2)} / ${clip.duration_s.toFixed(2)} 秒 · 帧 ${index+1}/${clip.frames.length}`;
 const rejected=f.athletes.filter(a=>!a.accepted).map(a=>a.motion_id.split(':').slice(-2).join(':'));
 $('status').textContent=`${manifest.clips[clipNumber].session_id} · 扫描 ${f.lidar_index} · ${f.point_count} 个原始返回点 · ${f.athletes.length} 份拟合${f.athletes.length?'':'（本帧无拟合，未补帧）'}${rejected.length?' · 拒收轨迹 '+rejected.join(', '):''}`;
 const html=f.athletes.map(a=>`<tr><td>${a.motion_id.split(':').slice(-2).join(':')}</td><td class="${a.accepted?'good':'bad'}">${a.accepted?'通过':'拒收'}</td><td>${fmt(a.qa.final_p2surface_mean_m*1000)} mm</td><td>${fmt(a.qa.final_p2surface_p90_m*1000)} mm</td><td>${fmt(a.qa.reprojection_mean_px)} px</td></tr>`).join('');
 $('metrics').innerHTML='<table><tr><th>窗口内轨迹</th><th>质量门禁</th><th>点到表面 mean</th><th>本帧点到表面 P90</th><th>Cam2 拟合残差</th></tr>'+html+'</table>';
 $('provenance').textContent=JSON.stringify({scan_mid_us:f.scan_mid_us,cameras:f.cameras,geometry_interpolated:false,athletes:f.athletes.map(a=>({motion_id:a.motion_id,accepted:a.accepted,rejection_reasons:a.rejection_reasons,qa:a.qa}))},null,2);draw();
}
function seek(i){if(!clip)return;playing=false;$('play').textContent='播放';mediaTime=clip.frames[Math.max(0,Math.min(i,clip.frames.length-1))].offset_s;showFrame(i);}
window.REVIEW_CLIP_READY=data=>{if(data.id!==manifest.clips[clipNumber].id)return;clip=data;index=0;mediaTime=0;playing=false;orient();$('timeline').max=clip.frames.length-1;$('play').textContent='播放';showFrame(0);};
function loadClip(i){playing=false;clipNumber=i;const generation=++loadGeneration;clip=null;window.REVIEW_STATE=null;gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);$('status').textContent='加载真实网格与连续帧…';
 [...$('clips').children].forEach((b,n)=>b.classList.toggle('active',n===i));const script=document.createElement('script');script.src=manifest.clips[i].id+'/data.js';
 script.onerror=()=>{if(generation===loadGeneration)$('status').textContent='片段数据未能加载，请保留完整输出文件夹。';};
 script.onload=()=>script.remove();document.body.appendChild(script);
}
// These hooks support reproducible browser validation and ordinary controls.
window.REVIEW_LOAD_CLIP=loadClip;window.REVIEW_SEEK=seek;window.REVIEW_PLAY=()=>{$('play').click();};
window.REVIEW_GET_VIEW=()=>({center:center.slice(),distance,yaw,pitch});
window.REVIEW_SET_VIEW=view=>{center=view.center.slice();distance=view.distance;yaw=view.yaw;pitch=view.pitch;draw();};
window.REVIEW_SET_MOTION=motion=>{visibleMotion=motion||null;draw();};
window.REVIEW_SET_CONTROL=(id,value)=>{if($(id)){if($(id).type==='checkbox')$(id).checked=value;else $(id).value=value;draw();}};
// The seven-source online adapter sends the same packed native-frame contract
// as offline data.js. Reuse full mesh normals/topology, point cloud, controls
// and camera math without creating a second 3D renderer.
window.REVIEW_STREAM_FRAME=frame=>{const changed=!clip||clip.id!==frame.session_id;
 clipNumber=0;clip={id:frame.session_id,frames:[frame],duration_s:frame.offset_s||0};index=0;mediaTime=frame.offset_s||0;
 if(changed)orient();showFrame(0);window.REVIEW_STATE.streaming=true;
 window.REVIEW_STATE.displayedBackgroundPoints=frame.displayed_background_points;};
function reportEmbedded(){if(!clip||!window.REVIEW_STATE)return;parent.postMessage({type:'review-state',state:window.REVIEW_STATE,view:window.REVIEW_GET_VIEW(),
 times:clip.frames.map(f=>({lidar_index:f.lidar_index,scan_mid_us:f.scan_mid_us,offset_s:f.offset_s})),
 images:['cam1','cam2'].map(id=>({src:$(id).getAttribute('src'),ready:$(id).complete&&$(id).naturalWidth>0}))},'*');}
if(embedded){$('cam1').onload=reportEmbedded;$('cam2').onload=reportEmbedded;
 window.addEventListener('message',event=>{if(event.source!==parent||event.data?.type!=='review-command')return;const {command,value}=event.data;
  if(command==='seek')seek(value);else if(command==='view')window.REVIEW_SET_VIEW(value);else if(command==='motion')window.REVIEW_SET_MOTION(value);else if(command==='control')window.REVIEW_SET_CONTROL(value.id,value.value);
 });}
function notifyView(){if(embedded)parent.postMessage({type:'review-view',view:window.REVIEW_GET_VIEW()},'*');}
$('play').onclick=()=>{if(!clip)return;playing=!playing;lastTick=performance.now();$('play').textContent=playing?'暂停':'播放';draw();};
$('timeline').oninput=()=>seek(Number($('timeline').value));$('prev').onclick=()=>clip&&seek(index-1);$('next').onclick=()=>clip&&seek(index+1);
for(const id of ['mesh','wire','points','background','opacity','skeleton'])if($(id))$(id).oninput=draw;
$('reset').onclick=()=>{if(defaultView){({center,distance,yaw,pitch}=defaultView);center=center.slice();draw();notifyView();}};
let drag=null;canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);};canvas.onpointerup=()=>drag=null;
canvas.onpointermove=e=>{if(drag){yaw-=(e.clientX-drag[0])*.008;pitch=Math.max(-.1,Math.min(1.45,pitch+(e.clientY-drag[1])*.007));drag=[e.clientX,e.clientY];draw();notifyView();}};
canvas.onwheel=e=>{e.preventDefault();distance=Math.max(2,Math.min(100,distance*Math.exp(e.deltaY*.001)));draw();notifyView();};window.onresize=draw;
document.onkeydown=e=>{if(embedded&&['Space','ArrowRight','ArrowLeft'].includes(e.code)){e.preventDefault();parent.postMessage({type:'review-transport',command:e.code},'*');return;}
 if(e.code==='Space'&&!['INPUT','SELECT'].includes(document.activeElement.tagName)){e.preventDefault();$('play').click();}else if(e.code==='ArrowRight'&&clip)seek(index+1);else if(e.code==='ArrowLeft'&&clip)seek(index-1);};
function animate(now){if(playing&&clip){mediaTime+=(now-lastTick)/1000*Number($('speed').value);if(mediaTime>=clip.duration_s){if($('loop').checked)mediaTime%=clip.duration_s;else{mediaTime=clip.duration_s;playing=false;$('play').textContent='播放';}}
 let i=0;while(i+1<clip.frames.length&&clip.frames[i+1].offset_s<=mediaTime)i++;if(i!==index)showFrame(i);else{$('clock').textContent=`${mediaTime.toFixed(2)} / ${clip.duration_s.toFixed(2)} 秒 · 帧 ${index+1}/${clip.frames.length}`;if(window.REVIEW_STATE){window.REVIEW_STATE.mediaTime=mediaTime;window.REVIEW_STATE.playing=playing;}}}lastTick=now;requestAnimationFrame(animate);}requestAnimationFrame(animate);
$('subtitle').textContent=`${manifest.clip_count} 个 ${manifest.clip_duration_s} 秒片段 · 双相机原始影像 + 6890 顶点 SMPL 表面 · 原生约 10 Hz，无插值`;
manifest.clips.forEach((c,i)=>{const button=document.createElement('button');button.innerHTML=`${String(i+1).padStart(2,'0')} · ${c.duration_s.toFixed(0)} 秒<small>${c.session_id}</small><small>${c.summary.scans} 扫描 · mean ${fmt(c.summary.mean_surface_mm)} mm</small>`;button.onclick=()=>loadClip(i);$('clips').appendChild(button);});
if(manifest.review_title)document.querySelector('h1').textContent=manifest.review_title;
if(manifest.review_description)$('subtitle').textContent=manifest.review_description;
if(manifest.review_notice){const notice=document.createElement('p');notice.textContent=manifest.review_notice.message+' ';const a=document.createElement('a');a.href=manifest.review_notice.link;a.textContent=manifest.review_notice.label;notice.append(a);document.querySelector('header').append(notice);}
manifest.clips.forEach((c,i)=>{if(c.label){const label=document.createElement('small');label.textContent=c.label;$('clips').children[i].appendChild(label);}});
const totals=manifest.clips.reduce((s,c)=>({frames:s.frames+c.summary.scans,fits:s.fits+c.summary.fitted_person_frames,ok:s.ok+c.summary.accepted}),{frames:0,fits:0,ok:0});
$('headline').innerHTML=`<div class="stat"><strong>${manifest.clip_count} × ${manifest.clip_duration_s}s</strong><small>连续片段 · ${totals.frames} 个实际扫描</small></div><div class="stat"><strong>${totals.ok} / ${totals.fits}</strong><small>预览中的通过／全部拟合</small></div><div class="stat"><strong>同一份世界网格</strong><small>两机分别使用冻结 K/d/R/t</small></div>`;
const description=document.createElement('p');description.textContent='Cam2 像素误差是参与拟合的去畸变关键点残差。Cam1 原图叠加与独立 RGB 关键帧检查需分开理解；检测器一致性不等于人体真值。';$('report').appendChild(description);
const quality=manifest.quality_report,cam1=manifest.cam1_check;
if(quality){const speed=quality.efficiency,passed=quality.quality.passed,stages=passed.surface_stages,stats=document.createElement('div');
 stats.innerHTML=`<p>正式批跑已完成 ${quality.completed_windows.length} 个窗口：${passed.person_scan_count} 个通过样本；统计水印 ${quality.progress_watermark.updated_utc}。</p>
 <table><tr><th>指标</th><th>实际结果</th><th>范围与含义</th></tr>
 <tr><td>端到端效率</td><td>${fmt(speed.completed_source_seconds/60)} 分钟视频 → ${fmt(speed.wall_seconds/60)} 分钟壁钟</td><td>${fmt(speed.passed_person_scans_per_wall_second)} 个通过人物／扫描每秒；含 teacher、拟合及 I/O</td></tr>
 <tr><td>可见点到全三角面</td><td>${fmt(stages.cloud_center_aligned_before_optimization.frame_mean_m.mean*1000)} → ${fmt(stages.final_smpl_all_triangles.frame_mean_m.mean*1000)} mm</td><td>中心平移后／优化后；平均每帧 P90 为 ${fmt(stages.final_smpl_all_triangles.frame_p90_m.mean*1000)} mm</td></tr>
 <tr><td>Cam2 关键点拟合残差</td><td>mean ${fmt(passed.primary_cam2_reprojection_px.mean)} px</td><td>去畸变像素；参与了优化</td></tr>
 ${cam1?.completed?`<tr><td>Cam1 RGB 检测器一致性</td><td>mean ${fmt(cam1.summary.pixel_error.mean)} / median ${fmt(cam1.summary.pixel_error.median)} / P90 ${fmt(cam1.summary.pixel_error.p90)} px</td><td>${cam1.summary.checked_frames} 个关键帧；关节覆盖 ${fmt(cam1.summary.coverage_of_candidate_17_joints*100,1)}%；未回馈拟合</td></tr>`:''}
 </table><p class="muted">两机焦距不同，不直接横比像素数。距离是可见表面拟合残差，不是 3D 关节误差或遮挡部位真值。</p>
 <p><a href="quality_efficiency.md">完整效率／质量报告</a> · <a href="cam1_check.md">Cam1 逐片段检查</a></p>`;$('report').appendChild(stats);
}
for(const [title,data] of [['效率与质量报告',manifest.quality_report],['Cam1 独立关键帧检查',manifest.cam1_check]]){const detail=document.createElement('details');const summary=document.createElement('summary');summary.textContent=title;detail.appendChild(summary);const pre=document.createElement('pre');pre.textContent=data?JSON.stringify(data,null,2):'尚未量化；查看报告文件或后续更新。';detail.appendChild(pre);$('report').appendChild(detail);}
if(!manifest.stream_mode)loadClip(Math.max(0,Math.min(manifest.clips.length-1,Number(new URLSearchParams(location.search).get('clip'))||0)));
