// Canonical owned-browser wrapper for the Python fresh native playback verifier.
// Uses the calibration repository's existing Chrome for Testing lifecycle helper.
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { createServer } from 'node:net';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const tools = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const forwarded = [];
let python = process.env.GVHMR_PYTHON || resolve(tools, '../.venv/Scripts/python.exe');
let helper = process.env.LIDAR_BROWSER_SESSION_HELPER ||
  resolve(tools, '../../../LIDAR/lidar_project/scripts/checks/lib/browser_session.mjs');
for (let index = 0; index < args.length; index++) {
  if (['--python', '--browser-session-helper'].includes(args[index])) {
    const option = args[index];
    if (!args[index + 1] || args[index + 1].startsWith('--')) throw new Error(`${option} requires a path`);
    if (option === '--python') python = args[++index];
    else helper = args[++index];
  } else {
    if (args[index] === '--cdp-url') throw new Error('The wrapper allocates its own CDP endpoint');
    forwarded.push(args[index]);
  }
}
if (forwarded.includes('--help') || forwarded.includes('-h')) {
  const code = await new Promise((done, fail) => {
    const child = spawn(python, ['-B', resolve(tools, 'verify_badminton_fresh_playback.py'), '--help'],
      { stdio: 'inherit', shell: false, windowsHide: true });
    child.once('error', fail);
    child.once('exit', status => done(status ?? 1));
  });
  process.exit(code);
}
if (!existsSync(helper)) throw new Error('Provide the existing browser_session.mjs with --browser-session-helper');
const { startBrowser } = await import(pathToFileURL(resolve(helper)).href);
const port = await new Promise((done, fail) => {
  const server = createServer();
  server.once('error', fail);
  server.listen(0, '127.0.0.1', () => {
    const selected = server.address().port;
    server.close(error => error ? fail(error) : done(selected));
  });
});
let owned;
try {
  owned = await startBrowser({ port, args: ['--headless=new', '--autoplay-policy=no-user-gesture-required'] });
  const code = await new Promise((done, fail) => {
    const child = spawn(python, ['-B', resolve(tools, 'verify_badminton_fresh_playback.py'),
      ...forwarded, '--cdp-url', `http://127.0.0.1:${port}`], { stdio: 'inherit', shell: false, windowsHide: true });
    child.once('error', fail);
    child.once('exit', (status, signal) => signal ? fail(new Error(`Verifier stopped by ${signal}`)) : done(status ?? 1));
  });
  process.exitCode = code;
} finally {
  if (owned) await owned.close();
}
