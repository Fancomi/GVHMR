/* Native source playback and exact scan transactions; the existing review.js
 * owns WebGL meshes, topology/normals, real clouds and orbit controls. */
'use strict';
const $=id=>document.getElementById(id);
const videos=[$('cam1'),$('cam2')],canvases=[$('overlay1'),$('overlay2')];
// Visible RGB is a frozen native-frame canvas. Hidden media elements can seek
// without exposing a new RGB frame beneath an older point/mesh projection.
const nativeFrames=videos.map((v,i)=>{const c=document.createElement('canvas');
 c.id='nativeFrame'+(i+1);c.width=1920;c.height=1080;c.style.visibility='hidden';v.after(c);return c;});
const scopeLabel=document.createElement('label');scopeLabel.innerHTML='<input type="checkbox" id="allPoints">全部源点';$('projectCloud').parentElement.after(scopeLabel);
function projectedRenderer(canvas){
 const gl=canvas.getContext('webgl2',{alpha:true,premultipliedAlpha:false,antialias:true,preserveDrawingBuffer:true});
 if(!gl)throw Error('投影叠加需要 WebGL2');
 function compile(type,text){const s=gl.createShader(type);gl.shaderSource(s,text);gl.compileShader(s);
  if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;}
 const program=gl.createProgram();
 gl.attachShader(program,compile(gl.VERTEX_SHADER,'#version 300 es\nin vec2 p;uniform float pointSize;void main(){gl_Position=vec4(p.x/960.0-1.0,1.0-p.y/540.0,0.0,1.0);gl_PointSize=pointSize;}'));
 gl.attachShader(program,compile(gl.FRAGMENT_SHADER,'#version 300 es\nprecision highp float;uniform vec4 color;out vec4 c;void main(){c=color;}'));
 gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));
 const position=gl.getAttribLocation(program,'p'),color=gl.getUniformLocation(program,'color'),size=gl.getUniformLocation(program,'pointSize');
 const points=gl.createBuffer(),indices=gl.createBuffer();
 function clear(){gl.viewport(0,0,canvas.width,canvas.height);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);}
 function draw(array,rgba,mode,pointSize=2,faceIndices=null){
  gl.useProgram(program);gl.disable(gl.BLEND);gl.disable(gl.DEPTH_TEST);gl.disable(gl.CULL_FACE);
  const flat=array instanceof Float32Array?array:new Float32Array(array.flat());
  gl.bindBuffer(gl.ARRAY_BUFFER,points);gl.bufferData(gl.ARRAY_BUFFER,flat,gl.STREAM_DRAW);
  gl.enableVertexAttribArray(position);gl.vertexAttribPointer(position,2,gl.FLOAT,false,0,0);
  gl.uniform4fv(color,rgba);gl.uniform1f(size,pointSize);
  if(faceIndices){gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,indices);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,faceIndices,gl.STREAM_DRAW);gl.drawElements(mode,faceIndices.length,gl.UNSIGNED_SHORT,0);}
  else gl.drawArrays(mode,0,flat.length/2);
 }
 function render(data,cam){clear();let fullMeshes=0;
  if($('projectCloud').checked)draw(data.background_projection?.[cam]||[],[.35,.79,.85,.65],gl.POINTS,1.6);
  data.athletes.forEach((a,k)=>{const g=a.cameras[cam],rgb=[[1,.62,.19],[.31,.64,1],[.5,.88,.53],[.85,.52,.89]][k%4];
   if(g._meshFlat&&g._meshFlat.length===13780){
    let usable=faces;if(g._meshValid.some(v=>!v)){const valid=[];for(let j=0;j<faces.length;j+=3)
     if(g._meshValid[faces[j]]&&g._meshValid[faces[j+1]]&&g._meshValid[faces[j+2]])valid.push(faces[j],faces[j+1],faces[j+2]);usable=new Uint16Array(valid);}
    // Constant alpha is written once per rasterized pixel, without additive
    // blending across overlapping front/back faces: an exact silhouette union.
    draw(g._meshFlat,[...rgb,.24],gl.TRIANGLES,1,usable);fullMeshes++;
   }else draw(g.mesh.filter(Boolean),[...rgb,1],gl.POINTS,2);
   draw(g.cloud.filter(Boolean),[1,1,.25,1],gl.POINTS,2.3);
   const bones=[];manifest.edges.forEach(([a,b])=>{if(g.joints[a]&&g.joints[b])bones.push(g.joints[a],g.joints[b]);});
   draw(bones,[...rgb,1],gl.LINES);
  });return {camera:cam,scan:data.lidar_index,nativeRecord:data.native[cam].record_index,fullMeshes,webglError:gl.getError()};
 }
 return {clear,render};
}
const projectionRenderers=canvases.map(projectedRenderer);
const colors=['#ffaa40','#62adff','#94e69d','#df9bff'];
let manifest,faces,sourceId,source,clip,loading=false,playing=false,lastSync=0;
let reviewTime=0,lastTick=0,lastFrameRequest=0,frameBusy=false,generation=0,lastData=null;
let resolveReady;const ready=new Promise(resolve=>{resolveReady=resolve;});
const readyFrames=new Map(),prefetchTasks=new Map();let nextPrefetchIndex=null;
const scope=()=>$('allPoints').checked?'all':'roi';
let commitTimes=[],playToken=0,bufferWaitStart=null,bufferWaits=[];
function clearReadyFrames(){for(const task of prefetchTasks.values())task.controller.abort();prefetchTasks.clear();nextPrefetchIndex=null;
 for(const b of readyFrames.values())b.decoded.forEach(x=>x.close());readyFrames.clear();}
function bufferState(){const frames=[...readyFrames.values()].sort((a,b)=>a.data.frame.offset_s-b.data.frame.offset_s);
 return {ready:frames.length,ahead_seconds:frames.length?Math.max(0,frames.at(-1).data.frame.offset_s-reviewTime):0,pending:prefetchTasks.size,scope:scope(),commits:commitTimes,
  starvation_gaps:commitTimes.slice(1).map((c,i)=>c.time-commitTimes[i].time).filter(g=>g>250),
  buffer_underruns:bufferWaits.length+(bufferWaitStart==null?0:1),buffer_wait_ms:bufferWaits.reduce((n,v)=>n+v,0)+(bufferWaitStart==null?0:performance.now()-bufferWaitStart)};}
async function decodeBundle(data){
 for(const a of data.athletes)for(const g of Object.values(a.cameras))if(g.mesh_xy_f32){
  g._meshFlat=new Float32Array(Uint8Array.from(atob(g.mesh_xy_f32),c=>c.charCodeAt(0)).buffer);
  g._meshValid=Uint8Array.from(atob(g.mesh_valid_u8),c=>c.charCodeAt(0));
  let rows=null;Object.defineProperty(g,'mesh',{get(){return rows||(rows=Array.from(g._meshValid,(v,i)=>v?[g._meshFlat[i*2],g._meshFlat[i*2+1]]:null));}});
 }
 const decoded=await Promise.all(['cam1','cam2'].map(async cam=>{const image=data.native_images[cam],native=data.native[cam];
  if(image.record_index!==native.record_index||Math.abs(image.media_pts_us/1e6-native.pts)>.000002)throw Error('原生图像与扫描配对不一致');
  return createImageBitmap(await(await fetch('data:image/jpeg;base64,'+image.jpeg_base64)).blob());}));
 await sceneReady();await new Promise(resolve=>requestAnimationFrame(resolve));
 $('scene').contentWindow.REVIEW_PREPARE_FRAME(data.frame);
 return {data,decoded};
}
function startPrefetch(index){
 const version=generation,sid=sourceId,pointScope=scope(),controller=new AbortController(),task={controller,remaining:Math.min(4,source.scan_times.length-index)};prefetchTasks.set(index,task);
 task.promise=(async()=>{try{
  const t=source.scan_times[index];
  const response=await fetch('/api/bundles?source='+sid+'&t='+t+'&count=4&scope='+pointScope,{signal:controller.signal});
  if(!response.ok)throw Error('帧包载入失败');const batch=await response.json();
  for(const data of batch.frames){if(version!==generation||controller.signal.aborted)return;if(!data.frame){task.remaining--;continue;}
   if(readyFrames.has(data.lidar_index)){task.remaining--;continue;}
   const bundle=await decodeBundle(data);
   if(version!==generation||controller.signal.aborted){bundle.decoded.forEach(b=>b.close());return;}
   readyFrames.set(data.lidar_index,bundle);
   task.remaining--;
   while(readyFrames.size>16){const key=Math.min(...readyFrames.keys());readyFrames.get(key).decoded.forEach(b=>b.close());readyFrames.delete(key);}
  }
 }catch(e){if(e.name!=='AbortError')$('error').textContent=e.message;}
 finally{if(prefetchTasks.get(index)===task)prefetchTasks.delete(index);}})();
 return task.promise;
}
function fillBuffer(){
 if(nextPrefetchIndex==null)return;
 const ahead=[...readyFrames.keys()].filter(i=>i>(lastData?.lidar_index??-1)).length;
 let reserved=ahead+[...prefetchTasks.values()].reduce((n,t)=>n+t.remaining,0);
 while(prefetchTasks.size<3&&reserved<=12&&nextPrefetchIndex<source.scan_times.length){
  const index=nextPrefetchIndex;nextPrefetchIndex+=4;reserved+=4;startPrefetch(index);
 }
}
async function prefetchFrames(){
 if(nextPrefetchIndex==null)nextPrefetchIndex=(lastData?.lidar_index??0)+1;
 fillBuffer();
 for(let i=0;i<500;i++){
  const ahead=[...readyFrames.keys()].filter(k=>k>(lastData?.lidar_index??-1)).length;
  if(ahead>=12||(!prefetchTasks.size&&nextPrefetchIndex>=source.scan_times.length))return;
  if(!prefetchTasks.size)fillBuffer();await new Promise(resolve=>setTimeout(resolve,20));
 }
 throw Error('播放缓存载入超时');
}
function commitBundle(bundle){
 const {data,decoded}=bundle;
 nativeFrames.forEach((c,i)=>{c.getContext('2d',{alpha:false}).drawImage(decoded[i],0,0,c.width,c.height);c.style.visibility='visible';});
 $('scene').contentWindow.REVIEW_STREAM_FRAME(data.frame);drawProjection(data);
 lastData=data;window.lastPairedFrame=data;
 window.committedPairedFrame={sourceId,scan:data.lidar_index,lidar_index:data.lidar_index,scan_mid_us:data.scan_mid_us,native:data.native,point_scope:data.frame.point_scope,
  images:Object.fromEntries(Object.entries(data.native_images).map(([cam,{jpeg_base64,...meta}])=>[cam,meta]))};
 if(playing){commitTimes.push({scan:data.lidar_index,time:performance.now()});if(commitTimes.length>150)commitTimes.shift();}
 const fitted=data.athletes.map((a,i)=>'人体'+(i+1)+'：'+a.mean_mm.toFixed(1)+' mm，'+(a.accepted?'拟合 QA 通过':'已排除')).join('；');
 $('fitInfo').textContent='扫描 '+data.lidar_index+' · 原生帧 '+data.native.cam1.record_index+' / '+data.native.cam2.record_index+
  ' · '+(data.frame.point_scope==='roi'?'正式 ROI ':'全部源点 ')+data.frame.point_count.toLocaleString()+' / 原始 '+data.frame.raw_point_count.toLocaleString()+' 点 · '+
  (fitted||'该帧 SMPL 待生成或位于拟合回合之外');
}
const paired=()=>$('mode').value==='paired';
function clock(value,cam,inverse=false){
 const a=source.cameras[cam].clock_anchors,k=inverse?1:0,o=1-k;let l=0,r=a.length-1;
 while(l+1<r){const m=(l+r)>>1;if(a[m][k]<=value)l=m;else r=m;}
 return a[l][o]+(value-a[l][k])*(a[r][o]-a[l][o])/(a[r][k]-a[l][k]);
}
function commonTime(camera=1){return paired()&&lastData?.native?clock(lastData.native['cam'+camera].pts,'cam'+camera,true):clock(videos[camera-1].currentTime,'cam'+camera,true);}
function clearOverlay(){projectionRenderers.forEach(r=>r.clear());}
function stopPlayback(){playToken++;playing=false;videos.forEach(v=>v.pause());$('play').textContent='播放';}
async function pause(){const desired=reviewTime;stopPlayback();
 while(frameBusy)await new Promise(resolve=>setTimeout(resolve,10));
 if(paired()&&lastData?.frame&&Math.abs(lastData.frame.offset_s-desired)>.15)await frameAt(desired);
 if(paired()&&lastData?.frame)reviewTime=lastData.frame.offset_s;
}
function mediaReady(v){return v.readyState>=1?Promise.resolve():new Promise((resolve,reject)=>{
 v.addEventListener('loadedmetadata',resolve,{once:true});v.addEventListener('error',()=>reject(Error('视频载入失败')),{once:true});});}
async function moveVideo(v,t){
 await mediaReady(v);t=Math.max(0,Math.min(v.duration-.02,t));if(Math.abs(v.currentTime-t)<.000001)return;
 await new Promise((resolve,reject)=>{const done=()=>{clearTimeout(timer);resolve();};
 const timer=setTimeout(()=>reject(Error('原生帧解码超时')),15000);v.addEventListener('seeked',done,{once:true});v.currentTime=t;});
}
function drawProjection(data){
 window.projectionStates=projectionRenderers.map((r,i)=>r.render(data,'cam'+(i+1)));
}
async function sceneReady(){for(let i=0;i<200;i++){
 if($('scene').contentWindow.REVIEW_STREAM_FRAME)return;
 await new Promise(resolve=>setTimeout(resolve,50));}throw Error('三维播放器未就绪');}
async function frameAt(t){
 if(frameBusy)return;frameBusy=true;const version=generation,sid=sourceId;
 try{
  const data=await(await fetch('/api/geometry?source='+sid+'&t='+t+'&scope='+scope())).json();
  if(version!==generation)return;
  if(!data.frame){clearOverlay();$('fitInfo').textContent='该扫描缺少双相机原生时间配对。';return;}
  const bundle=await decodeBundle(data);await new Promise(resolve=>requestAnimationFrame(resolve));
  if(version===generation)commitBundle(bundle);bundle.decoded.forEach(b=>b.close());
 }catch(e){$('error').textContent=e.message;}finally{frameBusy=false;}
}
async function seek(t){
 stopPlayback();generation++;clearReadyFrames();reviewTime=Math.max(0,Math.min(source.duration-.03,t));$('seek').value=reviewTime;
 if(paired()){
  while(frameBusy)await new Promise(resolve=>setTimeout(resolve,20));await frameAt(reviewTime);
 }else {await Promise.all(videos.map((v,i)=>moveVideo(v,clock(reviewTime,'cam'+(i+1)))));nativeFrames.forEach(c=>c.style.visibility='hidden');clearOverlay();}
}
async function selectSource(sid,rallyIndex){
 ['source','rally','previous','next','play','exact','mode'].forEach(id=>$(id).disabled=true);
 loading=true;stopPlayback();generation++;sourceId=sid;source=manifest.sources[sid];$('source').value=sid;$('seek').max=source.duration;
 $('rally').replaceChildren(...source.rallies.map(c=>new Option('回合 '+c.index+' · '+(c.end-c.begin).toFixed(1)+'s',c.index)));
 videos.forEach((v,i)=>{v.src=source.cameras['cam'+(i+1)].url;v.load();});await Promise.all(videos.map(mediaReady));
 await selectRally(rallyIndex||source.rallies[0].index);loading=false;
 ['source','rally','previous','next','play','exact','mode'].forEach(id=>$(id).disabled=false);
}
async function selectRally(index){
 clip=source.rallies.find(c=>c.index==index);$('rally').value=index;await seek(Math.max(0,clip.begin));
 history.replaceState(null,'','?source='+sourceId+'&rally='+index);
}
async function play(){
 if(paired()){videos.forEach(v=>v.pause());$('play').textContent='准备播放';
  const version=generation,token=++playToken;if(bufferState().ahead_seconds<.6)await prefetchFrames();if(version!==generation||token!==playToken)return;
  commitTimes=[];bufferWaitStart=null;bufferWaits=[];playing=true;lastTick=performance.now();$('play').textContent='暂停';return;}
 playing=true;lastTick=performance.now();$('play').textContent='暂停';
 nativeFrames.forEach(c=>c.style.visibility='hidden');clearOverlay();videos.forEach(v=>v.playbackRate=+$('speed').value);
 try{await Promise.all(videos.map(v=>v.play()));}catch(e){pause();$('error').textContent=e.message;}
}
async function pairedFrame(){const t=commonTime();stopPlayback();$('mode').value='paired';await seek(t);}
function update(now){
 if(source&&!loading){
  if(playing&&paired())reviewTime+=Math.max(0,now-lastTick)/1000*+$('speed').value;
  const t=paired()?reviewTime:commonTime();if(!paired())reviewTime=t;$('seek').value=t;$('time').textContent=t.toFixed(2)+' / '+source.duration.toFixed(2)+' s';
  if(playing){
   if(paired()){
    const frames=[...readyFrames.values()].sort((a,b)=>a.data.frame.offset_s-b.data.frame.offset_s);
    let candidate=null;for(const b of frames)if(b.data.frame.offset_s<=t)candidate=b;
    if(t-(lastData?.frame.offset_s??t)>.16&&(!candidate||candidate.data.lidar_index===lastData?.lidar_index)){
     if(bufferWaitStart==null)bufferWaitStart=now;
    }
    if(candidate&&candidate.data.lidar_index!==lastData?.lidar_index){commitBundle(candidate);lastFrameRequest=now;
     if(bufferWaitStart!=null){bufferWaits.push(now-bufferWaitStart);bufferWaitStart=null;}
     for(const [key,b] of readyFrames)if(key<candidate.data.lidar_index-1){b.decoded.forEach(x=>x.close());readyFrames.delete(key);}}
    fillBuffer();
   }
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
$('play').onclick=()=>playing||$('play').textContent==='准备播放'?pause():play();$('seek').onchange=()=>seek(+$('seek').value);
$('back').onclick=()=>seek(Math.max(0,(paired()?reviewTime:commonTime())-1));
$('forward').onclick=()=>seek(Math.min(source.duration-.03,(paired()?reviewTime:commonTime())+1));$('exact').onclick=pairedFrame;
$('previous').onclick=()=>selectRally(source.rallies[Math.max(0,source.rallies.indexOf(clip)-1)].index);
$('next').onclick=()=>selectRally(source.rallies[Math.min(source.rallies.length-1,source.rallies.indexOf(clip)+1)].index);
$('speed').onchange=()=>{videos.forEach(v=>v.playbackRate=+$('speed').value);};
$('mode').onchange=()=>seek(reviewTime);$('projectCloud').onchange=()=>{if(lastData)drawProjection(lastData);};
$('allPoints').onchange=()=>seek(commonTime());
window.viewer={ready,selectSource:async(...args)=>{await ready;return selectSource(...args);},selectRally,seek,play,pause,pairedFrame,commonTime,bufferState,get sourceId(){return sourceId;},get loading(){return loading;}};
(async()=>{try{
 manifest=await(await fetch('/api/manifest')).json();faces=new Uint16Array(Uint8Array.from(atob(manifest.faces),c=>c.charCodeAt(0)).buffer);
 $('rallyMinutes').textContent=manifest.rally_window_minutes.toFixed(2)+' 分钟';$('inputScans').textContent=manifest.unique_paired_input_scans.toLocaleString()+' 帧';
 $('source').replaceChildren(...Object.keys(manifest.sources).map(s=>new Option(s,s)));
 const q=new URLSearchParams(location.search);await selectSource(q.get('source')||Object.keys(manifest.sources)[0],+q.get('rally')||undefined);
 window.viewerReady=true;resolveReady();status();setInterval(status,10000);lastTick=performance.now();requestAnimationFrame(update);
 }catch(e){$('error').textContent=e.stack;}})();
