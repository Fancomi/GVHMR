// Verify the root_pos fix: feed the service's root_pos/root_rota/body_pose/betas
// into label_mocap's own forwardSmpl, project the pelvis (joint 0), and compare
// to the service's keypoints[0]. Before the fix the pelvis was ~100-120px off
// vertically (transl vs posed-pelvis); after, it should be ~0.
import { readFile } from 'node:fs/promises';
import { loadModelFromFiles } from '../../../label_mocap/smpl_core/smpl_model.js';
import { forwardSmpl } from '../../../label_mocap/smpl_core/lbs.js';
import { projectPoint } from '../../../label_mocap/label/src/scene/projection.js';

const jsonPath = process.argv[2];
const model = await loadModelFromFiles(
  new URL('../../../label_mocap/smpl_web_viewer/public/models/smpl_neutral.meta.json', import.meta.url),
  async (u) => new Uint8Array(await readFile(u)));

const doc = JSON.parse(await readFile(jsonPath, 'utf8'));
const img = doc.images[0], ann = doc.annotations[0], K = img.cam_K;

const out = forwardSmpl(model, {
  root_pos: ann.root_pos, root_rota: ann.root_rota,
  body_pose: ann.body_pose, betas: ann.betas,
});

// Our pelvis (joint 0) projected with the service's K.
const p0 = [out.joints[0], out.joints[1], out.joints[2]];
const [u, v] = projectPoint(p0, K);
// Service's own keypoints[0] (pelvis).
const su = ann.keypoints[0], sv = ann.keypoints[1];

const dx = u - su, dy = v - sv;
console.log(`cam_K fx=${K.fx.toFixed(1)} cx=${K.cx} cy=${K.cy}`);
console.log(`OURS pelvis -> (${u.toFixed(1)}, ${v.toFixed(1)})`);
console.log(`CLOUD kp[0]  -> (${su.toFixed(1)}, ${sv.toFixed(1)})`);
console.log(`delta dx=${dx.toFixed(1)}  dy=${dy.toFixed(1)}  |d|=${Math.hypot(dx, dy).toFixed(1)} px`);
