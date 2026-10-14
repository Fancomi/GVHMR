/* Validate all portable clips in an isolated Chrome for Testing session. */
import {spawn} from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createServer} from 'node:net';

const args=process.argv.slice(2),option=(key,fallback)=>args.includes(key)?args[args.indexOf(key)+1]:fallback;
const review=path.resolve(option('--review-dir','outputs/badminton_corner_fix_v1/sequence_review'));
const helper=option('--browser-helper','D:/PROJECT/LIDAR/lidar_project/scripts/checks/lib/browser_session.mjs');
const {resolveTestBrowser,verifyBrowserIdentity}=await import(pathToFileURL(path.resolve(helper)).href);
const chrome=resolveTestBrowser({override:option('--browser','D:/PROJECT/LIDAR/lidar_project/build/tools/chrome-for-testing/chrome-win64/chrome.exe')});
const profile=path.join(review,'browser-profile'),reservation=createServer();
await new Promise(r=>reservation.listen(0,'127.0.0.1',r));const port=reservation.address().port;await new Promise(r=>reservation.close(r));
const child=spawn(chrome,['--headless=new','--no-sandbox','--no-first-run','--enable-automation','--enable-unsafe-swiftshader','--use-angle=swiftshader',
 '--window-size=1640,1180','--remote-debugging-port='+port,'--user-data-dir='+profile],{windowsHide:true,stdio:['ignore','ignore','pipe']});
child.stderr.pipe(fs.createWriteStream(path.join(review,'browser_stderr.log')));
const pause=ms=>new Promise(r=>setTimeout(r,ms));let browser,page;
let nextId=0;const pending=new Map(),exceptions=[];
function connect(url){return new Promise((done,reject)=>{const ws=new WebSocket(url);ws.addEventListener('open',()=>done(ws));ws.addEventListener('error',reject);});}
function attach(ws){ws.addEventListener('message',e=>{const m=JSON.parse(e.data);if(m.method==='Runtime.exceptionThrown')exceptions.push(m.params.exceptionDetails.text+': '+(m.params.exceptionDetails.exception?.description||''));
 if(pending.has(m.id)){const item=pending.get(m.id);clearTimeout(item.timer);pending.delete(m.id);m.error?item.reject(new Error(m.error.message)):item.done(m.result);}});}
function send(ws,method,params={}){return new Promise((done,reject)=>{const id=++nextId,timer=setTimeout(()=>{pending.delete(id);reject(new Error(method+' timed out'));},20000);pending.set(id,{done,reject,timer});ws.send(JSON.stringify({id,method,params}));});}
async function evaluate(expression){const result=await send(page,'Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(result.exceptionDetails)throw new Error(result.exceptionDetails.exception?.description||'Page evaluation failed');return result.result.value;}
async function ready(id,frame){for(let i=0;i<150;i++){await pause(100);const r=await evaluate(`(()=>{const s=window.REVIEW_STATE;return {state:s,images:[...document.images].map(i=>({src:i.getAttribute('src'),ready:i.complete&&i.naturalWidth>0})),buttons:document.querySelectorAll('#clips button').length}})()`);
 if(r.state?.clip===id&&(frame===undefined||r.state.frame===frame)&&r.images.every(i=>i.ready)&&r.images.every(i=>i.src.startsWith(id+'/')&&(frame===undefined||i.src.includes('/'+String(frame).padStart(3,'0')+'_'))))return r;
 if(exceptions.length)throw new Error(exceptions.join('\n'));}throw new Error('Clip did not render: '+id+' frame '+frame);}
try{
 let version;for(let i=0;i<80;i++){try{const r=await fetch(`http://127.0.0.1:${port}/json/version`);if(r.ok){version=await r.json();break;}}catch{}await pause(150);}
 if(!version)throw new Error('CFT startup failed');browser=await connect(version.webSocketDebuggerUrl);attach(browser);
 const identity=await send(browser,'Browser.getBrowserCommandLine'),processes=await send(browser,'SystemInfo.getProcessInfo');verifyBrowserIdentity({binary:chrome,pid:child.pid,profile,port},identity.arguments,processes.processInfo);
 const target=await send(browser,'Target.createTarget',{url:'about:blank'}),tabs=await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
 page=await connect(tabs.find(t=>t.id===target.targetId).webSocketDebuggerUrl);attach(page);await send(page,'Runtime.enable');await send(page,'Page.enable');
 await send(page,'Page.navigate',{url:pathToFileURL(path.join(review,'index.html')).href});await ready('clip01',0);
 const clips=await evaluate('window.REVIEW_MANIFEST.clips.map(c=>c.id)'),checked=[];
 for(let i=0;i<clips.length;i++){
  await evaluate(`window.REVIEW_LOAD_CLIP(${i})`);const first=await ready(clips[i],0),last=first.state.frameCount-1;
  const states=[first];for(const f of [Math.floor(last/2),last]){await evaluate(`window.REVIEW_SEEK(${f})`);states.push(await ready(clips[i],f));}
  if(states.some(s=>s.state.webglError!==0||s.state.duration!==5))throw new Error('WebGL or duration mismatch');
  if(states[0].state.scan===states[2].state.scan)throw new Error('Continuous frames did not advance');
  const pixels=await evaluate(`(()=>{const p=new Uint8Array(gl.drawingBufferWidth*gl.drawingBufferHeight*4);gl.readPixels(0,0,gl.drawingBufferWidth,gl.drawingBufferHeight,gl.RGBA,gl.UNSIGNED_BYTE,p);let colored=0;for(let j=0;j<p.length;j+=4)if(Math.max(p[j],p[j+1],p[j+2])-Math.min(p[j],p[j+1],p[j+2])>45&&Math.max(p[j],p[j+1],p[j+2])>70)colored++;return {colored,total:p.length/4};})()`);
  if(!pixels.colored)throw new Error('3D canvas has no colored body/point content');
  checked.push({id:clips[i],frame_count:first.state.frameCount,duration_s:first.state.duration,states:states.map(s=>s.state),images_loaded:true,canvas_pixels:pixels});
  if([0,4,9].includes(i)){await evaluate(`window.REVIEW_SEEK(${Math.floor(last/2)})`);await ready(clips[i],Math.floor(last/2));const layout=await send(page,'Page.getLayoutMetrics');const size=layout.cssContentSize;const shot=await send(page,'Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip:{x:0,y:0,width:size.width,height:size.height,scale:1}});fs.writeFileSync(path.join(review,clips[i]+'_screenshot.png'),Buffer.from(shot.data,'base64'));}
 }
 await evaluate('window.REVIEW_LOAD_CLIP(0)');await ready('clip01',0);await evaluate('window.REVIEW_PLAY()');await pause(1200);const animated=await evaluate('window.REVIEW_STATE');
 if(animated.frame<5||!animated.playing)throw new Error('Playback failed to advance on native timestamps');await evaluate('window.REVIEW_PLAY()');
 const result={passed:true,checked_clips:checked.length,clips:checked,playback: animated,exceptions};fs.writeFileSync(path.join(review,'browser_verification.json'),JSON.stringify(result,null,2));
 console.log(`Verified ${checked.length} clips: real first/middle/last photos, WebGL geometry, 5-second durations and advancing playback`);
 await send(browser,'Browser.close');browser.close();browser=null;page.close();page=null;
}finally{if(page)page.close();if(browser)browser.close();for(let i=0;i<30&&child.exitCode===null;i++)await pause(100);if(child.exitCode===null)child.kill();}
