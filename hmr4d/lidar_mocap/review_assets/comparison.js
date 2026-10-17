/* Atomic six-view comparison of recorded native frames and stored full SMPL. */
'use strict';
const $=id=>document.getElementById(id),manifest=window.COMPARISON_MANIFEST;
const unpack=s=>new Float32Array(Uint8Array.from(atob(s),c=>c.charCodeAt(0)).buffer);
const faces=new Uint16Array(Uint8Array.from(atob(manifest.faces),c=>c.charCodeAt(0)).buffer);
const workers=[$('oldWorker'),$('newWorker')],workerReady=[false,false],requests=new Map();let requestId=0;
const projections=['old','new'].map(prefix=>[1,2].map(i=>createProjectedRenderer($(prefix+'Projection'+i),()=>({faces,edges:manifest.edges}),()=>$('projectCloud').checked,{personPoints:()=>$('points').checked,opacity:()=>.4})));
let clip=null,selected=0,index=0,playing=false,busy=false,generation=0,view=null,defaultView=null,lastTick=0;
let cached=new Map(),tasks=new Map(),commits=[],waitingMs=0;const readyLimit=6;
let viewPending=false,viewScheduled=false;
function queueViewRender(){viewPending=true;if(viewScheduled)return;viewScheduled=true;
 requestAnimationFrame(()=>{viewScheduled=false;if(busy)return;viewPending=false;clearCache();action(()=>seek(index));});}
function fail(error){$('error').textContent=error?.message||String(error);playing=false;$('play').textContent='播放';}
function action(fn){Promise.resolve().then(fn).catch(fail);}
window.addEventListener('message',event=>{
 const which=workers.findIndex(w=>w.contentWindow===event.source);if(which<0)return;
 if(event.data?.type==='comparison-ready')workerReady[which]=true;
 if(event.data?.type!=='comparison-rendered')return;
 const pending=requests.get(event.data.requestId);if(!pending){event.data.bitmap?.close();return;}
 requests.delete(event.data.requestId);clearTimeout(pending.timer);
 if(event.data.error)pending.reject(Error(event.data.error));else pending.resolve(event.data);
});
async function ready(){const start=performance.now();while(!workerReady.every(Boolean)){
 if(performance.now()-start>20000)throw Error('三维渲染器未就绪');
 workers.forEach(w=>w.contentWindow.postMessage({type:'comparison-ping'},'*'));await new Promise(r=>setTimeout(r,100));
}}
function workerRender(which,frame,currentView){return new Promise((resolve,reject)=>{
 const id=++requestId,timer=setTimeout(()=>{requests.delete(id);reject(Error('三维渲染超时'));},20000);
 requests.set(id,{resolve,reject,timer});workers[which].contentWindow.postMessage({type:'comparison-render',requestId:id,frame,view:currentView,
 controls:{background:true,points:$('points').checked,mesh:true,skeleton:true,opacity:.55}},'*');
});}
function physical(a){const id=Number(a.motion_id.split(':').at(-2));return Number(clip.aliases[id]??id);}
function shown(a){return ($('failed').checked||a.accepted)&&(!$('person').value||physical(a)===Number($('person').value));}
function showActors(data){const result={...data,athletes:data.athletes.filter(shown),frame:{...data.frame}};
 result.frame.athletes=data.frame.athletes.filter(shown);return result;}
function decodeGeometry(data){for(const a of data.athletes)for(const g of Object.values(a.cameras))if(!g._meshFlat){
 g._meshFlat=unpack(g.mesh_xy_f32);g._meshValid=Uint8Array.from(atob(g.mesh_valid_u8),c=>c.charCodeAt(0));
 }}
function dispose(bundle){bundle.rgb.forEach(b=>b.close());bundle.scenes.forEach(b=>b.bitmap.close());}
function clearCache(){generation++;for(const b of cached.values())dispose(b);cached.clear();tasks.clear();}
function fitView(){if(!clip)return;const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
 for(const f of clip.frames)for(const version of ['before','after'])for(const a of f[version].frame.athletes){
  if(!shown(a))continue;const points=unpack(a.vertices);
  for(let i=0;i<points.length;i+=3)for(let k=0;k<3;k++){lo[k]=Math.min(lo[k],points[i+k]);hi[k]=Math.max(hi[k],points[i+k]);}
 }
 if(!Number.isFinite(lo[0])){lo.splice(0,3,-5,-5,0);hi.splice(0,3,5,5,2);}
 const center=lo.map((v,k)=>(v+hi[k])/2),yaw=3.5,pitch=.35;
 const eye=[Math.cos(pitch)*Math.cos(yaw),Math.cos(pitch)*Math.sin(yaw),Math.sin(pitch)];
 const right=[Math.sin(yaw),-Math.cos(yaw),0],up=[-Math.sin(pitch)*Math.cos(yaw),-Math.sin(pitch)*Math.sin(yaw),Math.cos(pitch)];
 const rect=$('old3d').getBoundingClientRect(),ty=Math.tan(Math.PI/8)*.88,tx=ty*rect.width/rect.height;
 const dot=(a,b)=>a.reduce((n,v,i)=>n+v*b[i],0);let distance=5;
 for(let m=0;m<8;m++){const d=center.map((v,k)=>((m&(1<<k))?hi[k]:lo[k])-v);distance=Math.max(distance,dot(eye,d)+Math.max(Math.abs(dot(right,d))/tx,Math.abs(dot(up,d))/ty));}
 view={center,distance,yaw,pitch};defaultView=structuredClone(view);resizeWorkers();
}
function resizeWorkers(){const rect=$('old3d').getBoundingClientRect();for(const w of workers){w.style.width=Math.round(rect.width)+'px';w.style.height=Math.round(rect.height)+'px';}}
async function prepare(i){if(cached.has(i))return cached.get(i);if(tasks.has(i))return tasks.get(i);
 const token=generation,data=clip.frames[i],currentView=structuredClone(view);
 const task=(async()=>{
  await ready();const before=showActors(data.before),after=showActors(data.after);
  after.frame.cloud=before.frame.cloud;after.background_projection=before.background_projection;
  decodeGeometry(before);decodeGeometry(after);
  const rgb=await Promise.all(['cam1','cam2'].map(async cam=>{
   const im=data.native_images[cam],native=data.before.native[cam];
   if(im.record_index!==native.record_index||im.media_pts_us!==Math.round(native.pts*1e6))throw Error('原生图像时间戳不匹配');
   return createImageBitmap(await(await fetch('data:image/jpeg;base64,'+im.jpeg_base64)).blob());
  }));
  let scenes;try{scenes=await Promise.all([workerRender(0,before.frame,currentView),workerRender(1,after.frame,currentView)]);}
  catch(error){rgb.forEach(b=>b.close());throw error;}
  const bundle={data,before,after,rgb,scenes};
  if(token!==generation){dispose(bundle);return null;}cached.set(i,bundle);return bundle;
 })().finally(()=>{if(tasks.get(i)===task)tasks.delete(i);});tasks.set(i,task);return task;
}
function prune(){for(const [i,b] of cached)if(i<index-1||i>index+readyLimit){dispose(b);cached.delete(i);}}
function prefetch(){if(!clip||busy)return;for(let i=index+1;i<Math.min(clip.frames.length,index+4);i++)if(!cached.has(i)&&!tasks.has(i))prepare(i).catch(fail);}
function cameraZoom(){if(!clip)return;
 for(const [ci,cam] of ['cam1','cam2'].entries()){
  let transform='none';if($('zoom').checked){
   const points=[];for(const f of clip.frames)for(const v of ['before','after'])for(const a of f[v].athletes)
    if(shown(a))points.push(...a.cameras[cam].joints.filter(Boolean));
   if(points.length){const lo=[Math.min(...points.map(p=>p[0]))-80,Math.min(...points.map(p=>p[1]))-65];
    const hi=[Math.max(...points.map(p=>p[0]))+80,Math.max(...points.map(p=>p[1]))+65];
    const scale=Math.max(1,Math.min(1920/(hi[0]-lo[0]),1080/(hi[1]-lo[1])));
    const x=(960-(hi[0]+lo[0])/2*scale)/1920*100,y=(540-(hi[1]+lo[1])/2*scale)/1080*100;
    transform=`translate(${x}%,${y}%) scale(${scale})`;
   }
  }
  for(const prefix of ['old','new'])for(const type of ['Rgb','Projection'])$(prefix+type+(ci+1)).style.transform=transform;
 }
}
function commit(bundle,i){const rows=[bundle.before,bundle.after],projectionStates=[];
 rows.forEach((data,v)=>{const canvas=$(v?'new3d':'old3d'),bitmap=bundle.scenes[v].bitmap;
  if(canvas.width!==bitmap.width)canvas.width=bitmap.width;if(canvas.height!==bitmap.height)canvas.height=bitmap.height;
  canvas.getContext('2d',{alpha:false}).drawImage(bitmap,0,0);
  ['cam1','cam2'].forEach((cam,c)=>{const rgb=$((v?'new':'old')+'Rgb'+(c+1));rgb.getContext('2d',{alpha:false}).drawImage(bundle.rgb[c],0,0);
   projectionStates.push(projections[v][c].render(data,cam));});
 });
 index=i;$('seek').value=i;$('clock').textContent=`${bundle.data.offset_s.toFixed(2)} / 5.00 s · ${i+1}/50`;
 $('status').textContent=`${clip.source_id} · 扫描 ${bundle.data.scan} · ${playing?'播放中':'暂停'} · 两行共用同一原生 RGB`;
 const describe=(data,label)=>{const actors=data.frame.athletes.map(a=>{
  const d=window.REPAIR_DIAGNOSTICS?.[(data===bundle.before?'before':'after')+'|'+a.motion_id+'|'+bundle.data.scan];
  const leg=d&&$('person').value?` / 腿 ${Number.isFinite(d.cam1)?d.cam1.toFixed(1):'—'}|${Number.isFinite(d.cam2)?d.cam2.toFixed(1):'—'}px${d.inconsistent?'（观测冲突）':''}${d.corrected?' 已纠左右':''}`:(d?.inconsistent?'（腿观测冲突）':'');
  return `${a.motion_id.split(':').at(-2)}: ${(a.qa.final_p2surface_mean_m*1000).toFixed(1)}mm${leg}`;});
  const rejected=data===bundle.before?bundle.data.before.frame.athletes.filter(a=>!a.accepted).length:bundle.data.after.frame.athletes.filter(a=>!a.accepted).length;
  return `${label} ID / 点面残差 ${actors.join(' · ')||'无通过拟合'}${rejected?' · 拒收 '+rejected+' 份':''}`;};
 $('metrics').textContent=describe(bundle.before,'前')+'　｜　'+describe(bundle.after,'后');
 const state={clip:clip.id,index:i,scan:bundle.data.scan,native:bundle.data.before.native,
  native_images:Object.fromEntries(Object.entries(bundle.data.native_images).map(([k,v])=>[k,{record_index:v.record_index,media_pts_us:v.media_pts_us,raw_bgr_sha256:v.raw_bgr_sha256,jpeg_sha256:v.jpeg_sha256}])),
  scenes:bundle.scenes.map(s=>s.state),projections:projectionStates,before_ids:bundle.before.athletes.map(a=>a.motion_id),after_ids:bundle.after.athletes.map(a=>a.motion_id),
  source_reports:clip.source_reports,geometry_interpolated:false};
 commits.push({scan:state.scan,clip:clip.id,time:performance.now()});if(commits.length>1000)commits.shift();
 window.COMPARISON_STATE={...state,playing,commits:commits.slice(),waiting_ms:waitingMs};window.COMPARISON_LAST_BUNDLE=bundle;
 $('evidence').textContent=JSON.stringify({frame:state,quality:window.REPAIR_QUALITY?.cohorts,quality_scope:window.REPAIR_QUALITY?.scope,limits:manifest.limits},null,2);prune();prefetch();
}
async function seek(i,pause=true){if(!clip)return;if(pause){playing=false;$('play').textContent='播放';}const token=generation;
 busy=true;i=Math.max(0,Math.min(clip.frames.length-1,i));$('status').textContent='准备同帧六视图…';
 const start=performance.now();try{const bundle=await prepare(i);if(bundle&&token===generation){waitingMs+=performance.now()-start;commit(bundle,i);}}
 finally{if(token===generation){busy=false;if(viewPending)queueViewRender();else prefetch();}}}
function options(){if(!clip)return;clearCache();cameraZoom();action(()=>seek(index));}
window.COMPARISON_CLIP_READY=data=>{if(data.id!==manifest.clips[selected].id)return;clip=data;index=0;
 $('person').innerHTML='<option value="">全部</option>';
 const ids=new Set(data.frames.flatMap(f=>f.after.frame.athletes.map(physical)));
 [...ids].sort((a,b)=>a-b).forEach(id=>{const option=document.createElement('option');option.value=id;option.textContent='ID '+id;$('person').appendChild(option);});
 fitView();cameraZoom();action(()=>seek(0));};
function loadClip(i){playing=false;clearCache();clip=null;busy=false;selected=i;$('clips').value=String(i);$('play').textContent='播放';
 window.COMPARISON_STATE=null;window.COMPARISON_LAST_BUNDLE=null;
 if(manifest.clips[i].state!=='complete'){fail(Error('该片段正在重算；刷新页面可查看完成状态。'));return;}
 $('error').textContent='';$('status').textContent='加载 50 个真实扫描…';const script=document.createElement('script');script.src=manifest.clips[i].id+'/data.js';
 script.onload=()=>script.remove();script.onerror=()=>{script.remove();fail(Error('片段数据加载失败'));};document.body.appendChild(script);
}
function toggle(){if(!clip)return;playing=!playing;lastTick=performance.now();$('play').textContent=playing?'暂停':'播放';if(window.COMPARISON_STATE)window.COMPARISON_STATE.playing=playing;prefetch();}
async function tick(now){if(playing&&clip&&!busy){
 const interval=(index+1<clip.frames.length?(clip.frames[index+1].offset_s-clip.frames[index].offset_s)*1000:100)/Number($('speed').value);
 if(now-lastTick>=interval){
 let next=index+1;if(next===clip.frames.length){if($('loop').checked)next=0;else{toggle();requestAnimationFrame(tick);return;}}
 const buffered=cached.has(next);lastTick+=interval;
 action(async()=>{const started=performance.now();await seek(next,false);if(!buffered)lastTick+=performance.now()-started;});
 }}requestAnimationFrame(tick);}
manifest.clips.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.disabled=c.state!=='complete';o.textContent=`${i+1}. ${c.source_id} · 扫描 ${c.lidar_indices[0]}${c.state==='complete'?'':'（重算中）'}`;$('clips').appendChild(o);});
const complete=manifest.clips.filter(c=>c.state==='complete').length;$('progress').textContent=`${complete} / ${manifest.clips.length} 段 · 每段 5 秒 · 原生 10 Hz`;
$('clips').onchange=()=>loadClip(Number($('clips').value));$('play').onclick=toggle;$('prev').onclick=()=>action(()=>seek(index-1));$('next').onclick=()=>action(()=>seek(index+1));
$('seek').oninput=()=>action(()=>seek(Number($('seek').value)));$('failed').onchange=()=>{fitView();options();};$('projectCloud').onchange=options;$('zoom').onchange=cameraZoom;
$('reset').onclick=()=>{fitView();options();};
$('person').onchange=()=>{if($('person').value)$('zoom').checked=true;fitView();options();};
$('points').onchange=options;
for(const id of ['old3d','new3d']){const canvas=$(id);let drag=null;
 canvas.onpointerdown=e=>{if(!view)return;playing=false;$('play').textContent='播放';drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);};
 canvas.onpointermove=e=>{if(!drag)return;view.yaw-=(e.clientX-drag[0])*.008;view.pitch=Math.max(-.1,Math.min(1.45,view.pitch+(e.clientY-drag[1])*.007));drag=[e.clientX,e.clientY];queueViewRender();};
 canvas.onpointerup=()=>{drag=null;};canvas.onwheel=e=>{e.preventDefault();if(!view)return;view.distance=Math.max(2,Math.min(100,view.distance*Math.exp(e.deltaY*.001)));queueViewRender();};
}
document.onkeydown=e=>{if(['INPUT','SELECT'].includes(document.activeElement.tagName)||e.repeat)return;if(e.code==='Space'){e.preventDefault();toggle();}else if(e.code==='ArrowRight')action(()=>seek(index+1));else if(e.code==='ArrowLeft')action(()=>seek(index-1));};
window.onresize=()=>{if(clip){fitView();options();}};
window.comparison={load:loadClip,seek,play:()=>{if(!playing)toggle();},pause:()=>{if(playing)toggle();},state:()=>window.COMPARISON_STATE};
loadClip(Math.max(0,manifest.clips.findIndex(c=>c.state==='complete')));requestAnimationFrame(tick);
if(location.protocol!=='file:')setInterval(async()=>{
 if(manifest.clips.every(c=>c.state==='complete'))return;
 try{const latest=await(await fetch('manifest.json',{cache:'no-store'})).json();
  latest.clips.forEach((c,i)=>{if(c.state==='complete'&&manifest.clips[i].state!=='complete'){
   Object.assign(manifest.clips[i],c);const option=$('clips').options[i];option.disabled=false;option.textContent=option.textContent.replace('（重算中）','');
  }});$('progress').textContent=`${manifest.clips.filter(c=>c.state==='complete').length} / ${manifest.clips.length} 段 · 每段 5 秒 · 原生 10 Hz`;
 }catch(error){console.debug(error);}
},5000);
