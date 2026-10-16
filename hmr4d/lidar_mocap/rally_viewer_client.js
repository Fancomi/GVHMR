/* Native source playback and exact scan transactions; the existing review.js
 * owns WebGL meshes, topology/normals, real clouds and orbit controls. */
'use strict';
const $=id=>document.getElementById(id);
const videos=[$('cam1'),$('cam2')],canvases=[$('overlay1'),$('overlay2')];
const colors=['#ffaa40','#62adff','#94e69d','#df9bff'];
let manifest,faces,sourceId,source,clip,loading=false,playing=false,lastSync=0;
let reviewTime=0,lastTick=0,lastFrameRequest=0,frameBusy=false,generation=0,lastData=null;
const paired=()=>$('mode').value==='paired';
function clock(value,cam,inverse=false){
 const a=source.cameras[cam].clock_anchors,k=inverse?1:0,o=1-k;let l=0,r=a.length-1;
 while(l+1<r){const m=(l+r)>>1;if(a[m][k]<=value)l=m;else r=m;}
 return a[l][o]+(value-a[l][k])*(a[r][o]-a[l][o])/(a[r][k]-a[l][k]);
}
function commonTime(camera=1){return clock(videos[camera-1].currentTime,'cam'+camera,true);}
function clearOverlay(){canvases.forEach(c=>c.getContext('2d').clearRect(0,0,c.width,c.height));}
function pause(){playing=false;videos.forEach(v=>v.pause());$('play').textContent='播放';}
function mediaReady(v){return v.readyState>=1?Promise.resolve():new Promise((resolve,reject)=>{
 v.addEventListener('loadedmetadata',resolve,{once:true});v.addEventListener('error',()=>reject(Error('视频载入失败')),{once:true});});}
async function moveVideo(v,t){
 await mediaReady(v);t=Math.max(0,Math.min(v.duration-.02,t));if(Math.abs(v.currentTime-t)<.000001)return;
 await new Promise((resolve,reject)=>{const done=()=>{clearTimeout(timer);resolve();};
 const timer=setTimeout(()=>reject(Error('原生帧解码超时')),15000);v.addEventListener('seeked',done,{once:true});v.currentTime=t;});
}
function drawProjection(data){
 clearOverlay();canvases.forEach((canvas,i)=>{const ctx=canvas.getContext('2d'),cam='cam'+(i+1);
  if($('projectCloud').checked){ctx.fillStyle='#77ceda';(data.background_projection?.[cam]||[]).forEach(p=>ctx.fillRect(p[0],p[1],1.5,1.5));}
  data.athletes.forEach((a,k)=>{const g=a.cameras[cam],color=colors[k%colors.length];
   ctx.fillStyle=color;
   if(g.mesh.length===6890){
    // Every projected triangle uses the same winding, so a single nonzero
    // path fills their union without front/back faces erasing each other.
    ctx.beginPath();for(let j=0;j<faces.length;j+=3){const p=g.mesh[faces[j]],q=g.mesh[faces[j+1]],r=g.mesh[faces[j+2]];
     if(!p||!q||!r)continue;const positive=(q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])>=0;
     ctx.moveTo(...p);ctx.lineTo(...(positive?q:r));ctx.lineTo(...(positive?r:q));ctx.closePath();}
    ctx.globalAlpha=.24;ctx.fill('nonzero');ctx.globalAlpha=1;
   }else g.mesh.forEach(p=>{if(p)ctx.fillRect(p[0]-1,p[1]-1,2,2);});
   ctx.fillStyle='#ffff55';g.cloud.forEach(p=>{if(p)ctx.fillRect(p[0]-1,p[1]-1,2,2);});
   ctx.strokeStyle=color;ctx.lineWidth=3;manifest.edges.forEach(([a,b])=>{if(g.joints[a]&&g.joints[b]){
    ctx.beginPath();ctx.moveTo(...g.joints[a]);ctx.lineTo(...g.joints[b]);ctx.stroke();}});
  });
 });
}
async function sceneReady(){for(let i=0;i<200;i++){
 if($('scene').contentWindow.REVIEW_STREAM_FRAME)return;
 await new Promise(resolve=>setTimeout(resolve,50));}throw Error('三维播放器未就绪');}
async function frameAt(t){
 if(frameBusy)return;frameBusy=true;const version=generation,sid=sourceId;
 try{
  const data=await(await fetch('/api/geometry?source='+sid+'&t='+t)).json();
  if(version!==generation)return;
  window.lastPairedFrame=data;lastData=data;
  if(!data.frame){clearOverlay();$('fitInfo').textContent='该扫描缺少双相机原生时间配对。';return;}
  await Promise.all(videos.map((v,i)=>moveVideo(v,data.native['cam'+(i+1)].pts)));
  if(version!==generation)return;
  await sceneReady();$('scene').contentWindow.REVIEW_STREAM_FRAME(data.frame);drawProjection(data);
  const fitted=data.athletes.map((a,i)=>'人体'+(i+1)+'：'+a.mean_mm.toFixed(1)+' mm，'+(a.accepted?'拟合 QA 通过':'已排除')).join('；');
  $('fitInfo').textContent='扫描 '+data.lidar_index+' · 原生帧 '+data.native.cam1.record_index+' / '+data.native.cam2.record_index+
   ' · 实测 '+data.frame.point_count.toLocaleString()+' 点（三维背景显示 '+data.frame.displayed_background_points.toLocaleString()+' 个实际返回点） · '+
   (fitted||'该帧 SMPL 待生成或位于拟合回合之外');
 }catch(e){$('error').textContent=e.message;}finally{frameBusy=false;}
}
async function seek(t){
 pause();generation++;clearOverlay();reviewTime=Math.max(0,Math.min(source.duration-.03,t));$('seek').value=reviewTime;
 if(paired()){
  while(frameBusy)await new Promise(resolve=>setTimeout(resolve,20));await frameAt(reviewTime);
 }else await Promise.all(videos.map((v,i)=>moveVideo(v,clock(reviewTime,'cam'+(i+1)))));
}
async function selectSource(sid,rallyIndex){
 loading=true;pause();generation++;sourceId=sid;source=manifest.sources[sid];$('source').value=sid;$('seek').max=source.duration;
 $('rally').replaceChildren(...source.rallies.map(c=>new Option('回合 '+c.index+' · '+(c.end-c.begin).toFixed(1)+'s',c.index)));
 videos.forEach((v,i)=>{v.src=source.cameras['cam'+(i+1)].url;v.load();});await Promise.all(videos.map(mediaReady));
 await selectRally(rallyIndex||source.rallies[0].index);loading=false;
}
async function selectRally(index){
 clip=source.rallies.find(c=>c.index==index);$('rally').value=index;await seek(Math.max(0,clip.begin));
 history.replaceState(null,'','?source='+sourceId+'&rally='+index);
}
async function play(){
 playing=true;lastTick=performance.now();$('play').textContent='暂停';
 if(paired()){videos.forEach(v=>v.pause());return;}
 clearOverlay();videos.forEach(v=>v.playbackRate=+$('speed').value);
 try{await Promise.all(videos.map(v=>v.play()));}catch(e){pause();$('error').textContent=e.message;}
}
async function pairedFrame(){pause();$('mode').value='paired';reviewTime=commonTime();await seek(reviewTime);}
function update(now){
 if(source&&!loading){
  if(playing&&paired())reviewTime+=Math.max(0,now-lastTick)/1000*+$('speed').value;
  const t=paired()?reviewTime:commonTime();$('seek').value=t;$('time').textContent=t.toFixed(2)+' / '+source.duration.toFixed(2)+' s';
  if(playing){
   if(paired()){if(now-lastFrameRequest>=100&&!frameBusy){lastFrameRequest=now;frameAt(t);}}
   else if(now-lastSync>200){const t2=commonTime(2),rate=+$('speed').value;
    if(Math.abs(t2-t)>.045)videos[1].currentTime=clock(t,'cam2');
    else videos[1].playbackRate=rate*Math.max(.98,Math.min(1.02,1+(t-t2)*.4));lastSync=now;}
   const end=$('loop').checked&&clip?Math.min(source.duration-.03,clip.end):source.duration-.03;
   if(t>=end){if($('loop').checked)seek(Math.max(0,clip.begin)).then(play);else pause();}
  }
 }
 lastTick=now;requestAnimationFrame(update);
}
async function status(){try{
 const s=await(await fetch('/api/status')).json();$('accepted').textContent=s.paired_accepted==null?'构建中':s.paired_accepted.toLocaleString();
 const labels={pending:'待处理',running:'正在处理',reuse_pending:'已有拟合，待复用核验',reused:'已复用',fitted:'拟合完成',failed:'失败',teacher_complete:'关键点完成',complete:'数据完成'};
 $('status').replaceChildren(...Object.entries(s.sources).map(([sid,a])=>{const tr=document.createElement('tr');
  const n=a.fitted??a.completed_fit_person_frames,ok=a.accepted??a.completed_fit_qa_pass;
  [sid,a.rallies,a.rally_unique_paired_scans,labels[a.state]||a.state,n??'—',ok??'—',n==null?'—':n-ok].forEach(value=>{const td=document.createElement('td');td.textContent=value;tr.append(td);});
  tr.onclick=()=>selectSource(sid);tr.style.cursor='pointer';tr.title=(a.failures||[]).join('\n');return tr;}
 ));
 }catch(e){$('error').textContent=e.message;}}
$('source').onchange=()=>selectSource($('source').value);$('rally').onchange=()=>selectRally(+$('rally').value);
$('play').onclick=()=>playing?pause():play();$('seek').onchange=()=>seek(+$('seek').value);
$('back').onclick=()=>seek(Math.max(0,(paired()?reviewTime:commonTime())-1));
$('forward').onclick=()=>seek(Math.min(source.duration-.03,(paired()?reviewTime:commonTime())+1));$('exact').onclick=pairedFrame;
$('previous').onclick=()=>selectRally(source.rallies[Math.max(0,source.rallies.indexOf(clip)-1)].index);
$('next').onclick=()=>selectRally(source.rallies[Math.min(source.rallies.length-1,source.rallies.indexOf(clip)+1)].index);
$('speed').onchange=()=>{videos.forEach(v=>v.playbackRate=+$('speed').value);};
$('mode').onchange=()=>seek(commonTime());$('projectCloud').onchange=()=>{if(lastData)drawProjection(lastData);};
window.viewer={selectSource,selectRally,seek,play,pause,pairedFrame,commonTime,get sourceId(){return sourceId;},get loading(){return loading;}};
(async()=>{try{
 manifest=await(await fetch('/api/manifest')).json();faces=new Uint16Array(Uint8Array.from(atob(manifest.faces),c=>c.charCodeAt(0)).buffer);
 $('rallyMinutes').textContent=manifest.rally_window_minutes.toFixed(2)+' 分钟';$('inputScans').textContent=manifest.unique_paired_input_scans.toLocaleString()+' 帧';
 $('source').replaceChildren(...Object.keys(manifest.sources).map(s=>new Option(s,s)));
 const q=new URLSearchParams(location.search);await selectSource(q.get('source')||Object.keys(manifest.sources)[0],+q.get('rally')||undefined);
 window.viewerReady=true;status();setInterval(status,10000);lastTick=performance.now();requestAnimationFrame(update);
 }catch(e){$('error').textContent=e.stack;}})();
