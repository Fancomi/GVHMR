/* Validate portable production clips in an isolated, explicitly supplied browser. */
import {spawn,execFileSync} from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createServer} from 'node:net';
import {parseArgs} from 'node:util';

const args=process.argv.slice(2);
if(args.includes('--help')||args.includes('-h')){
 console.log(`Usage: node tools/check_badminton_reviews.mjs --browser <executable> [options]

Validate every production clip's first/middle/last photos, WebGL content,
manifest duration, and native-time playback using Node's built-in CDP client.
The Chrome for Testing executable is an external asset supplied by the caller.

Options:
  --browser <path>       Required browser executable; no implicit browser lookup
  --review-dir <path>    Review directory (default: outputs/badminton_corner_fix_v1_continuity/sequence_review)
  --viewer              Open historical viewer.html instead of index.html
  --cleanup-profile     Remove this run's isolated browser profile after exit
  --help, -h            Show help without launching a browser or writing files

Writes browser_verification.json (viewer_browser_verification.json with --viewer),
representative screenshots, and browser_stderr.log into the review directory.`);
 process.exit(0);
}
const {values}=parseArgs({args,options:{
 browser:{type:'string'},'review-dir':{type:'string',default:'outputs/badminton_corner_fix_v1_continuity/sequence_review'},
 viewer:{type:'boolean'},'cleanup-profile':{type:'boolean'},help:{type:'boolean',short:'h'},
}});
if(!values.browser)throw new Error('--browser <executable> is required; use --help for usage');
const realpath=p=>fs.realpathSync.native(p);
const samePath=(a,b)=>process.platform==='win32'?a.toLowerCase()===b.toLowerCase():a===b;
const chrome=realpath(path.resolve(values.browser));
if(!fs.statSync(chrome).isFile())throw new Error('Browser executable must be a file');
const review=realpath(path.resolve(values['review-dir']));
if(!fs.statSync(review).isDirectory())throw new Error('Review path must be a directory');
const entry=path.join(review,values.viewer?'viewer.html':'index.html');
if(!fs.statSync(entry).isFile())throw new Error('Review entry page must be a file');
const profile=fs.mkdtempSync(path.join(review,'browser-profile-'));
const reservation=createServer();
await new Promise((done,reject)=>{reservation.once('error',reject);reservation.listen(0,'127.0.0.1',done);});
const port=reservation.address().port;
await new Promise((done,reject)=>reservation.close(error=>error?reject(error):done()));
const child=spawn(chrome,['--headless=new','--no-sandbox','--no-first-run','--enable-automation','--enable-unsafe-swiftshader','--use-angle=swiftshader',
 '--window-size=1640,1180','--remote-debugging-port='+port,'--user-data-dir='+profile],{windowsHide:true,stdio:['ignore','ignore','pipe']});
let spawnError;
child.once('error',error=>{spawnError=error;});
child.stderr.pipe(fs.createWriteStream(path.join(review,'browser_stderr.log')));
const pause=ms=>new Promise(done=>setTimeout(done,ms));
const running=()=>!spawnError&&child.exitCode===null&&child.signalCode===null;
let browser,page,nextId=0;
const pending=new Map(),exceptions=[];

function commandLineOption(arguments_,name){
 const found=[];
 for(let i=1;i<arguments_.length;i++){
  if(arguments_[i]===name)found.push(arguments_[i+1]);
  else if(arguments_[i].startsWith(name+'='))found.push(arguments_[i].slice(name.length+1));
 }
 if(found.length!==1||typeof found[0]!=='string'||!found[0])throw new Error('Expected exactly one browser '+name+' argument');
 return found[0];
}
function verifyBrowserIdentity(arguments_,processInfo){
 if(!Array.isArray(arguments_)||!arguments_.length||typeof arguments_[0]!=='string'||!samePath(realpath(arguments_[0]),chrome)){
  throw new Error('CDP browser executable differs from the explicitly supplied executable');
 }
 const actualProfile=commandLineOption(arguments_,'--user-data-dir');
 if(actualProfile!==profile||!samePath(realpath(actualProfile),realpath(profile))){
  throw new Error('CDP browser user-data-dir differs from this run\'s isolated profile');
 }
 if(commandLineOption(arguments_,'--remote-debugging-port')!==String(port))throw new Error('CDP browser debug port differs from the reserved port');
 const browserProcesses=Array.isArray(processInfo)?processInfo.filter(item=>item.type==='browser'):[];
 if(browserProcesses.length!==1||Number(browserProcesses[0].id)!==child.pid||!running()){
  throw new Error('CDP browser process PID differs from the spawned browser child');
 }
 return {binary:chrome,pid:child.pid,profile,port};
}
function connect(url){return new Promise((done,reject)=>{const ws=new WebSocket(url);ws.addEventListener('open',()=>done(ws));ws.addEventListener('error',reject);});}
function attach(ws){ws.addEventListener('message',event=>{
 const message=JSON.parse(event.data);
 if(message.method==='Runtime.exceptionThrown')exceptions.push(message.params.exceptionDetails.text+': '+(message.params.exceptionDetails.exception?.description||''));
 if(pending.has(message.id)){
  const item=pending.get(message.id);clearTimeout(item.timer);pending.delete(message.id);
  message.error?item.reject(new Error(message.error.message)):item.done(message.result);
 }
});}
function send(ws,method,params={}){return new Promise((done,reject)=>{
 const id=++nextId,timer=setTimeout(()=>{pending.delete(id);reject(new Error(method+' timed out'));},20000);
 pending.set(id,{done,reject,timer});
 try{ws.send(JSON.stringify({id,method,params}));}catch(error){clearTimeout(timer);pending.delete(id);reject(error);}
});}
async function evaluate(expression){
 const result=await send(page,'Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
 if(result.exceptionDetails)throw new Error(result.exceptionDetails.exception?.description||'Page evaluation failed');
 return result.result.value;
}
async function manifestClips(){
 for(let i=0;i<150;i++){
  const clips=await evaluate('window.REVIEW_MANIFEST?.clips?.map(c=>({id:c.id,duration_s:c.duration_s}))');
  if(Array.isArray(clips)&&clips.length){
   const ids=new Set();
   for(const clip of clips){
    if(typeof clip.id!=='string'||!/^[-a-zA-Z0-9_]+$/.test(clip.id)||ids.has(clip.id))throw new Error('Review clip IDs must be unique, safe filenames');
    if(!Number.isFinite(clip.duration_s)||clip.duration_s<=0)throw new Error('Review clip has an invalid manifest duration_s: '+clip.id);
    ids.add(clip.id);
   }
   return clips;
  }
  if(exceptions.length)throw new Error(exceptions.join('\n'));
  await pause(100);
 }
 throw new Error('Review manifest did not load any clips');
}
async function ready(id,frame){
 for(let i=0;i<150;i++){
  await pause(100);
  const result=await evaluate(`(()=>{const s=window.REVIEW_STATE;return {state:s,images:[...document.images].map(i=>({src:i.getAttribute('src'),ready:i.complete&&i.naturalWidth>0})),buttons:document.querySelectorAll('#clips button').length}})()`);
  if(exceptions.length)throw new Error(exceptions.join('\n'));
  if(result.state?.clip===id&&(frame===undefined||result.state.frame===frame)&&result.images.length>=2&&
    result.images.every(image=>image.ready&&typeof image.src==='string'&&image.src.startsWith(id+'/')&&
      (frame===undefined||image.src.includes('/'+String(frame).padStart(3,'0')+'_'))))return result;
 }
 throw new Error('Clip did not render: '+id+' frame '+frame);
}
async function screenshot(id){
 const layout=await send(page,'Page.getLayoutMetrics'),size=layout.cssContentSize;
 const shot=await send(page,'Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip:{x:0,y:0,width:size.width,height:size.height,scale:1}});
 fs.writeFileSync(path.join(review,id+'_screenshot.png'),Buffer.from(shot.data,'base64'));
}

try{
 let version;
 for(let i=0;i<80;i++){
  if(spawnError)throw spawnError;
  if(!running())throw new Error('Browser exited before its CDP endpoint became available');
  try{const response=await fetch(`http://127.0.0.1:${port}/json/version`);if(response.ok){version=await response.json();break;}}catch{}
  await pause(150);
 }
 if(!version)throw new Error('Browser CDP startup failed');
 browser=await connect(version.webSocketDebuggerUrl);attach(browser);
 const identity=await send(browser,'Browser.getBrowserCommandLine'),processes=await send(browser,'SystemInfo.getProcessInfo');
 const verifiedIdentity=verifyBrowserIdentity(identity.arguments,processes.processInfo);
 const target=await send(browser,'Target.createTarget',{url:'about:blank'});
 const tabs=await (await fetch(`http://127.0.0.1:${port}/json/list`)).json(),tab=tabs.find(item=>item.id===target.targetId);
 if(!tab?.webSocketDebuggerUrl)throw new Error('Spawned browser did not expose the created review target');
 page=await connect(tab.webSocketDebuggerUrl);attach(page);
 await send(page,'Runtime.enable');await send(page,'Page.enable');
 await send(page,'Page.navigate',{url:pathToFileURL(entry).href});
 const clips=await manifestClips(),checked=[];
 for(let i=0;i<clips.length;i++){
  const clip=clips[i];
  await evaluate(`window.REVIEW_LOAD_CLIP(${i})`);
  const first=await ready(clip.id,0),last=first.state.frameCount-1;
  if(!Number.isInteger(last)||last<1)throw new Error('Continuous clip must contain at least two native frames: '+clip.id);
  const states=[first];
  for(const frame of [Math.floor(last/2),last]){
   await evaluate(`window.REVIEW_SEEK(${frame})`);states.push(await ready(clip.id,frame));
  }
  if(states.some(item=>item.state.webglError!==0||!Number.isFinite(item.state.duration)||Math.abs(item.state.duration-clip.duration_s)>1e-9)){
   throw new Error('WebGL or manifest duration mismatch: '+clip.id);
  }
  if(states.some(item=>item.buttons!==clips.length))throw new Error('Review clip navigation is incomplete');
  if(states[0].state.scan===states[2].state.scan)throw new Error('Continuous scans did not advance: '+clip.id);
  const pixels=await evaluate(`(()=>{const p=new Uint8Array(gl.drawingBufferWidth*gl.drawingBufferHeight*4);gl.readPixels(0,0,gl.drawingBufferWidth,gl.drawingBufferHeight,gl.RGBA,gl.UNSIGNED_BYTE,p);let colored=0;for(let j=0;j<p.length;j+=4)if(Math.max(p[j],p[j+1],p[j+2])-Math.min(p[j],p[j+1],p[j+2])>45&&Math.max(p[j],p[j+1],p[j+2])>70)colored++;return {colored,total:p.length/4,webglError:gl.getError()};})()`);
  if(!pixels.colored||pixels.webglError!==0)throw new Error('3D canvas has no valid colored body/point content: '+clip.id);
  checked.push({id:clip.id,frame_count:first.state.frameCount,duration_s:first.state.duration,expected_duration_s:clip.duration_s,states:states.map(item=>item.state),images_loaded:true,canvas_pixels:pixels});
  if([0,4,9].includes(i)||i===clips.length-1){
   await evaluate(`window.REVIEW_SEEK(${Math.floor(last/2)})`);await ready(clip.id,Math.floor(last/2));await screenshot(clip.id);
  }
 }
 await evaluate('window.REVIEW_LOAD_CLIP(0)');await ready(clips[0].id,0);await evaluate('window.REVIEW_PLAY()');
 let animated;
 const playbackStart=Date.now(),minimumTime=Math.min(1,clips[0].duration_s/2),timeout=Math.max(1500,Math.min(15000,clips[0].duration_s*1100));
 while(Date.now()-playbackStart<timeout){
  await pause(50);animated=await evaluate('window.REVIEW_STATE');
  if(exceptions.length)throw new Error(exceptions.join('\n'));
  if(animated?.playing&&animated.frame>0&&animated.mediaTime>=minimumTime)break;
 }
 if(!animated?.playing||animated.frame<=0||!Number.isFinite(animated.mediaTime)||animated.mediaTime<minimumTime){
  throw new Error('Playback failed to advance on native timestamps');
 }
 await evaluate('window.REVIEW_PLAY()');
 const result={passed:true,browser:verifiedIdentity,checked_clips:checked.length,clips:checked,playback:animated,exceptions};
 fs.writeFileSync(path.join(review,values.viewer?'viewer_browser_verification.json':'browser_verification.json'),JSON.stringify(result,null,2));
 console.log(`Verified ${checked.length} clips: real first/middle/last photos, WebGL geometry, manifest durations and advancing playback`);
 await send(browser,'Browser.close');browser.close();browser=null;page.close();page=null;
}finally{
 if(page)page.close();if(browser)browser.close();
 for(let i=0;i<30&&running();i++)await pause(100);
 if(running()){
  child.kill();
  for(let i=0;i<30&&running();i++)await pause(100);
 }
 if(values['cleanup-profile']&&fs.existsSync(profile)){
  if(running())throw new Error('Browser is still running; isolated profile was retained');
  const profileStat=fs.lstatSync(profile),resolved=realpath(profile);
  if(!profileStat.isDirectory()||profileStat.isSymbolicLink()||!samePath(path.dirname(resolved),review)||path.basename(resolved)!==path.basename(profile)){
   throw new Error('Browser cache path left the verified review directory or became a link');
  }
  const literal="'"+resolved.replaceAll("'","''")+"'";
  if(process.platform==='win32'){
   execFileSync('powershell',['-NoProfile','-Command',`$browserCache = Get-Item -LiteralPath ${literal} -ErrorAction Stop; if (($browserCache.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Browser cache is a link' }`],{windowsHide:true,stdio:'pipe'});
   // Chromium can release session-file handles after the main PID exits.
   await pause(1500);
   for(let attempt=0;;attempt++){
    try{execFileSync('powershell',['-NoProfile','-Command',`Remove-Item -LiteralPath ${literal} -Recurse -Force -ErrorAction Stop`],{windowsHide:true,stdio:'pipe'});break;}
    catch(error){if(attempt>=2)throw error;await pause(1500);}
   }
   const stillExists=execFileSync('powershell',['-NoProfile','-Command',`Test-Path -LiteralPath ${literal}`],{windowsHide:true,encoding:'utf8'}).trim();
   if(stillExists!=='False')throw new Error('Browser profile cleanup failed');
  }else{
   fs.rmSync(resolved,{recursive:true,force:true});
   if(fs.existsSync(resolved))throw new Error('Browser profile cleanup failed');
  }
 }
}
