// Stage-1 cross-check: feed the service's player_0.json through label_mocap's
// OWN forwardSmpl (GL-frame LBS) + projection, and confirm the resulting 2D
// joints land in the image / bbox. This validates the CV->GL conversion in
// infer_core, using the exact downstream math rather than a re-derivation.
import { readFile } from 'node:fs/promises';
import { loadModelFromFiles } from '../../../label_mocap/smpl_core/smpl_model.js';
import { forwardSmpl } from '../../../label_mocap/smpl_core/lbs.js';
import { projectPoint } from '../../../label_mocap/label/src/scene/projection.js';

const jsonPath = process.argv[2];
if (!jsonPath) { console.error('usage: node xcheck_label_mocap.mjs <player_0.json>'); process.exit(1); }

const model = await loadModelFromFiles(
  new URL('../../../label_mocap/smpl_web_viewer/public/models/smpl_neutral.meta.json', import.meta.url),
  async (u) => new Uint8Array(await readFile(u)));

const doc = JSON.parse(await readFile(jsonPath, 'utf8'));
const img = doc.images[0];
const ann = doc.annotations[0];
const K = img.cam_K;
const W = img.width, H = img.height;

const frame = {
  root_pos: ann.root_pos,
  root_rota: ann.root_rota,
  body_pose: ann.body_pose,
  betas: ann.betas,
};
const out = forwardSmpl(model, frame);  // GL-frame joints (Y up, -Z fwd)

let inImg = 0, behind = 0;
const xs = [], ys = [];
for (let j = 0; j < 24; j++) {
  const p = [out.joints[j * 3], out.joints[j * 3 + 1], out.joints[j * 3 + 2]];
  if (p[2] >= 0) { behind++; continue; }
  const [u, v] = projectPoint(p, K);
  xs.push(u); ys.push(v);
  if (u >= 0 && u <= W && v >= 0 && v <= H) inImg++;
}
const bx = ann.bbox[0], by = ann.bbox[1], bw = ann.bbox[2], bh = ann.bbox[3];
let inBox = 0;
for (let i = 0; i < xs.length; i++) if (xs[i] >= bx && xs[i] <= bx + bw && ys[i] >= by && ys[i] <= by + bh) inBox++;

console.log(`image WxH = ${W}x${H}, cam_K fx=${K.fx.toFixed(1)} cx=${K.cx} cy=${K.cy}`);
console.log(`label_mocap LBS joints: x[${Math.min(...xs).toFixed(0)},${Math.max(...xs).toFixed(0)}] y[${Math.min(...ys).toFixed(0)},${Math.max(...ys).toFixed(0)}]`);
console.log(`behind camera: ${behind}/24, inside image: ${inImg}/${xs.length}, inside bbox: ${inBox}/${xs.length}`);
