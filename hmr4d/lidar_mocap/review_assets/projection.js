/* Shared full-SMPL/native-camera projection renderer. */
'use strict';
window.createProjectedRenderer=function(canvas,geometry,showCloud,options={}){
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
 function render(data,cam){clear();let fullMeshes=0;const {faces,edges}=geometry();
  if(showCloud())draw(data.background_projection?.[cam]||[],[.35,.79,.85,.65],gl.POINTS,1.6);
  data.athletes.forEach(a=>{const g=a.cameras[cam],rgb=a.color.map(c=>c/255);
   if(g._meshFlat&&g._meshFlat.length===13780){
    let usable=faces;if(g._meshValid.some(v=>!v)){const valid=[];for(let j=0;j<faces.length;j+=3)
     if(g._meshValid[faces[j]]&&g._meshValid[faces[j+1]]&&g._meshValid[faces[j+2]])valid.push(faces[j],faces[j+1],faces[j+2]);usable=new Uint16Array(valid);}
    // Constant alpha is written once per rasterized pixel, without additive
    // blending across overlapping front/back faces: an exact silhouette union.
    draw(g._meshFlat,[...rgb,options.opacity?.()??.24],gl.TRIANGLES,1,usable);fullMeshes++;
   }else draw(g.mesh.filter(Boolean),[...rgb,1],gl.POINTS,2);
   if(options.personPoints?.()??true)draw(g.cloud.filter(Boolean),[1,1,.25,1],gl.POINTS,2.3);
   const bones=[];edges.forEach(([a,b])=>{if(g.joints[a]&&g.joints[b])bones.push(g.joints[a],g.joints[b]);});
   draw(bones,[...rgb,1],gl.LINES);
  });return {camera:cam,scan:data.lidar_index,nativeRecord:data.native[cam].record_index,fullMeshes,webglError:gl.getError()};
 }
 return {clear,render};
}
;
