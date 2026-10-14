'use strict';
const manifest=window.SYNC_MANIFEST,$=id=>document.getElementById(id),frames=[];
let ready=false,index=0,playing=false,mediaTime=0,lastTick=0,times=[],defaultView=null,lastView='',diagnostic=null;
const states=new Map();
const labels=['第一条 · 橙色','第二条 · 蓝色','第三条 · 绿色','第四条 · 紫色'];
const motions=Object.keys(manifest.clips[0].motion_colors);
motions.forEach((id,i)=>{const option=document.createElement('option');option.value=id;option.textContent=labels[i];$('athlete').appendChild(option);const label=document.createElement('span');const dot=document.createElement('i');dot.className='dot';dot.style.background=`rgb(${manifest.clips[0].motion_colors[id].join(',')})`;label.append(dot,labels[i]);$('legend').append(label);});
manifest.clips.forEach((clip,i)=>{const column=document.createElement('section');column.className='column';const title=document.createElement('h2');title.textContent=clip.label;const frame=document.createElement('iframe');frame.title=clip.label;frame.src=`viewer.html?embedded=1&clip=${i}`;column.append(title,frame);$('columns').append(column);frames.push(frame);});
function command(command,value){frames.forEach(f=>f.contentWindow.postMessage({type:'review-command',command,value},'*'));}
function state(){window.SYNC_STATE={ready,frame:index,scan:times[index]?.lidar_index,playing,mediaTime,duration:manifest.clip_duration_s,views:frames.map((_,i)=>states.get(i)?.state||null),images:frames.map((_,i)=>states.get(i)?.images||[])};}
function show(i){index=Math.max(0,Math.min(times.length-1,i));command('seek',index);$('timeline').value=index;$('clock').textContent=`${mediaTime.toFixed(2)} / 5.00 秒 · 扫描 ${times[index].lidar_index}`;$('status').textContent=`三栏扫描 ${times[index].lidar_index} · 同一物理时间 ${times[index].scan_mid_us} · 四人全部有实际拟合`;
 state();if(diagnostic)drawCharts();}
function seek(i){if(!ready)return;playing=false;$('play').textContent='播放';mediaTime=times[Math.max(0,Math.min(i,times.length-1))].offset_s;show(i);}
window.SYNC_SEEK=seek;window.SYNC_PLAY=()=>{if(!ready)return;playing=!playing;lastTick=performance.now();$('play').textContent=playing?'暂停':'播放';state();};
$('play').onclick=window.SYNC_PLAY;$('timeline').oninput=()=>seek(Number($('timeline').value));$('prev').onclick=()=>seek(index-1);$('next').onclick=()=>seek(index+1);
$('athlete').onchange=()=>{command('motion',$('athlete').value);if(diagnostic)drawCharts();};
for(const id of ['mesh','points','skeleton','wire'])$(id).onchange=()=>command('control',{id,value:$(id).checked});
$('reset').onclick=()=>{if(defaultView){lastView=JSON.stringify(defaultView);command('view',defaultView);}};
function animate(now){if(ready){
 if(playing){mediaTime+=(now-lastTick)/1000*Number($('speed').value);if(mediaTime>=5){if($('loop').checked)mediaTime%=5;else{mediaTime=5;playing=false;$('play').textContent='播放';}}let i=0;while(i+1<times.length&&times[i+1].offset_s<=mediaTime)i++;if(i!==index)show(i);else{$('clock').textContent=`${mediaTime.toFixed(2)} / 5.00 秒 · 扫描 ${times[index].lidar_index}`;state();}}}lastTick=now;requestAnimationFrame(animate);}requestAnimationFrame(animate);
window.addEventListener('message',event=>{const n=frames.findIndex(f=>f.contentWindow===event.source);if(n<0)return;
 if(event.data?.type==='review-transport'){if(event.data.command==='Space')window.SYNC_PLAY();else seek(index+(event.data.command==='ArrowRight'?1:-1));return;}
 if(event.data?.type==='review-view'){lastView=JSON.stringify(event.data.view);command('view',event.data.view);return;}
 if(event.data?.type!=='review-state')return;
 const payload=event.data;if(payload.state.clip!==manifest.clips[n].id)return;states.set(n,payload);
 if(!ready&&states.size===frames.length){times=states.get(0).times;defaultView=states.get(0).view;lastView=JSON.stringify(defaultView);ready=true;$('timeline').max=times.length-1;command('view',defaultView);show(0);}
 state();
});
const variantNames=['old','fixed_beta_benchmark','production'],curveColors=['#a6b5c9','#f5bf67','#90c8ff'];
function norm(v){return Math.hypot(...v);}
function residual(previous,current,key){const a=previous.pelvis_by_variant_m[key],b=current.pelvis_by_variant_m[key];return norm(b.map((x,i)=>x-a[i]-(current.proxy_m[i]-previous.proxy_m[i])))*100;}
function drawCharts(){if(!diagnostic)return;for(const [trackId,track] of Object.entries(diagnostic.tracks)){
 const canvas=document.getElementById('chart'+trackId),ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height,pad=35;
 const rows=track.timeline.filter(r=>r.clip_offset_s>=0&&r.clip_offset_s<5),points=variantNames.map(key=>rows.slice(1).map((r,i)=>({x:r.clip_offset_s,y:residual(rows[i],r,key)})));
 const ymax=Math.max(5,...points.flat().map(r=>r.y))*1.12;ctx.clearRect(0,0,w,h);ctx.fillStyle='#a6b5c9';ctx.font='12px system-ui';
 for(let i=0;i<=4;i++){const y=h-pad-i*(h-2*pad)/4;ctx.strokeStyle='#34455c';ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(w-pad,y);ctx.stroke();ctx.fillText((ymax*i/4).toFixed(0),5,y+4);}
 points.forEach((series,j)=>{ctx.strokeStyle=curveColors[j];ctx.lineWidth=j===2?2.5:1.5;ctx.beginPath();series.forEach((p,i)=>{const x=pad+p.x/5*(w-2*pad),y=h-pad-p.y/ymax*(h-2*pad);if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y);});ctx.stroke();});
 if(ready){ctx.strokeStyle='#ffffff';const x=pad+times[index].offset_s/5*(w-2*pad);ctx.beginPath();ctx.moveTo(x,pad);ctx.lineTo(x,h-pad);ctx.stroke();}
 ctx.fillText('0 秒',pad,h-8);ctx.fillText('5 秒',w-pad-25,h-8);ctx.fillText('cm',5,18);
 }}
diagnostic=window.SYNC_DIAGNOSIS||null;
if(diagnostic){
 const section=$('summary'),title=document.createElement('h2');title.textContent='整段四人的位移一致性';section.append(title);
 const table=document.createElement('table');table.innerHTML='<tr><th>轨迹</th><th>原拟合 P90</th><th>固定体型修正 P90</th><th>当前流水线 P90</th></tr>'+Object.entries(diagnostic.tracks).map(([id,t])=>`<tr><td>${labels[Number(id)-1]}</td>${variantNames.map(k=>`<td>${(t.variants[k].clip50.pelvis_minus_cloud_proxy_step_m.p90*100).toFixed(2)} cm</td>`).join('')}</tr>`).join('');section.append(table);
 const note=document.createElement('p');note.className='muted';note.textContent='指标是相邻实际扫描的“骨盆位移减点云中位位置位移”的长度。原拟合、固定体型实验、当前流水线使用相同帧范围，点云参照不等于骨盆真值；指标反映估计与可见表面运动的一致性。';section.append(note);
 const legend=document.createElement('p');legend.innerHTML=curveColors.map((c,i)=>`<span style="color:${c}">${['原拟合','固定体型修正','当前流水线'][i]}</span>`).join(' · ');$('charts').append(legend);
 Object.entries(diagnostic.tracks).forEach(([id,t])=>{const title=document.createElement('h2');title.textContent=labels[Number(id)-1]+' · 骨盆/点云位移差';const canvas=document.createElement('canvas');canvas.id='chart'+id;canvas.className='jumpchart';canvas.width=1400;canvas.height=220;
  canvas.onclick=e=>{if(ready){const x=(e.clientX-canvas.getBoundingClientRect().left)/canvas.clientWidth;const time=Math.max(0,Math.min(5,(x*1400-35)/(1400-70)*5));let i=0;while(i+1<times.length&&times[i+1].offset_s<time)i++;seek(i);}};$('charts').append(title,canvas);});drawCharts();
}
[['橙色 14–15',1,14],['橙色 37–38',1,37],['橙色 39–40',1,39],['蓝色 23–24–25',2,23]].forEach(([label,track,scan])=>{const button=document.createElement('button');button.textContent=label;button.onclick=()=>{if(!ready)return;$('athlete').value=motions[track-1];$('athlete').onchange();const i=times.findIndex(t=>t.lidar_index===scan);if(i>=0)seek(i);};$('events').append(button,' ');});
