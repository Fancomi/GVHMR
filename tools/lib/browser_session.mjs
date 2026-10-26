// Owned browser lifecycle; adapted from Fancomi lidar_project, commit 5f80bf3.
// Never close a browser without checking its binary, PID, profile and CDP port.
import { spawn } from 'node:child_process';
import { existsSync, statSync, readFileSync, writeFileSync, unlinkSync, mkdtempSync, mkdirSync, rmSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import { createServer } from 'node:net';

const pause = ms => new Promise(done => setTimeout(done, ms));
const alive = pid => { try { process.kill(pid, 0); return true; } catch (e) { return e.code !== 'ESRCH'; } };
const repoRoot = fileURLToPath(new URL('../../', import.meta.url));
const sessionRoot = join(repoRoot, 'outputs/browser_sessions');
const profileRoot = join(sessionRoot, 'profiles');
const statePath = join(sessionRoot, 'active.json');
const profilePrefix = 'gvhmr_review_profile_';

// Only the project-owned Chrome for Testing is discovered automatically.
// An explicit override names a separately installed test browser, never a fallback.
export function resolveTestBrowser({ override = process.env.GVHMR_CHROME || process.env.LIDAR_CHROME, root = repoRoot,
    platform = process.platform, arch = process.arch } = {}) {
  const relative = platform === 'win32' ? 'chrome-win64/chrome.exe'
    : platform === 'darwin' ? `chrome-mac-${arch === 'arm64' ? 'arm64' : 'x64'}/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing`
    : platform === 'linux' ? 'chrome-linux64/chrome' : null;
  const binary = override ? resolve(override) : relative && join(root, 'build/tools/chrome-for-testing', relative);
  if (!binary || !existsSync(binary) || !statSync(binary).isFile()) {
    throw new Error(`Independent Chrome for Testing not found: ${binary || platform}. Install it under build/tools/chrome-for-testing or set GVHMR_CHROME to a dedicated test binary; system Chrome/Edge fallback is disabled.`);
  }
  return binary;
}

export function recordBrowserEvent(path, state, event, detail = {}) {
  const previous = existsSync(path) ? JSON.parse(readFileSync(path, 'utf8')) : {};
  const time = new Date().toISOString();
  const record = { ...previous, ...state, event, time, ...detail,
    events: [...(previous.events || []), { event, time, ...detail }] };
  writeFileSync(path, JSON.stringify(record, null, 2) + '\n');
  console.log('[browser] ' + JSON.stringify({ ...record, evidence: path }));
}

export function verifyBrowserIdentity(state, commandLine, processes) {
  const sameBinary = typeof state.binary === 'string' && typeof commandLine[0] === 'string' &&
    (process.platform === 'win32' ? resolve(commandLine[0]).toLowerCase() === resolve(state.binary).toLowerCase()
      : resolve(commandLine[0]) === resolve(state.binary));
  if (!sameBinary || !commandLine.includes('--user-data-dir=' + state.profile) ||
      !commandLine.includes('--remote-debugging-port=' + state.port) ||
      !processes.some(p => p.type === 'browser' && p.id === state.pid)) {
    throw new Error('Refusing to close browser: registered binary/PID/profile/port do not match CDP identity');
  }
}

async function version(port) {
  try {
    const response = await fetch(`http://127.0.0.1:${port}/json/version`, { signal: AbortSignal.timeout(750) });
    if (!response.ok) return null;
    return await response.json();
  } catch { return null; }
}

async function command(url, method) {
  return new Promise((done, reject) => {
    const socket = new WebSocket(url);
    let settled = false, sent = false, result = {};
    const finish = (error, value, closeSocket = true) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      for (const [event, handler] of Object.entries(handlers)) socket.removeEventListener(event, handler);
      if (closeSocket) {
        try { socket.close(); } catch (closeError) { error ||= closeError; }
      }
      error ? reject(error) : done(value);
    };
    const handlers = {
      open() {
        try { socket.send(JSON.stringify({ id: 1, method })); sent = true; }
        catch (error) { finish(error); }
      },
      message(event) {
        try {
          const reply = JSON.parse(event.data);
          if (reply.id !== 1) return;
          if (reply.error) { finish(new Error(reply.error.message)); return; }
          // Chrome must finish its shutdown handshake. Closing this socket on
          // the reply can interrupt Browser.close while the browser still lives.
          if (method === 'Browser.close') result = reply.result || {};
          else finish(null, reply.result);
        } catch (error) { finish(error); }
      },
      error() { finish(new Error(`CDP ${method} failed`)); },
      close() {
        if (method === 'Browser.close' && sent) finish(null, result, false);
        else finish(new Error(`CDP ${method} disconnected before replying`), undefined, false);
      },
    };
    const timer = setTimeout(() => finish(new Error(`CDP ${method} timed out`)), 3000);
    for (const [event, handler] of Object.entries(handlers)) socket.addEventListener(event, handler);
  });
}

async function waitGone(state, child) {
  for (let n = 0; n < 100; ++n) {
    const exited = child ? child.exitCode !== null || child.signalCode !== null : !alive(state.pid);
    if (exited && !await version(state.port)) return;
    await pause(100);
  }
  throw new Error(`Owned browser PID ${state.pid} did not exit; refusing to launch another browser`);
}

async function closeOwned(state, child = null) {
  const info = await version(state.port);
  if (!info) {
    if (!alive(state.pid) || (child && (child.exitCode !== null || child.signalCode !== null))) return;
    // A freshly spawned child belongs to this exact ChildProcess handle.
    if (child) { child.kill(); await waitGone(state, child); return; }
    throw new Error('Registered browser is alive but CDP is unavailable; cannot verify its profile safely');
  }
  const url = info.webSocketDebuggerUrl;
  const args = await command(url, 'Browser.getBrowserCommandLine');
  const processes = await command(url, 'SystemInfo.getProcessInfo');
  verifyBrowserIdentity(state, args.arguments || [], processes.processInfo || []);
  try {
    await command(url, 'Browser.close');
    await waitGone(state, child);
  } catch (error) {
    // Only this fresh ChildProcess handle permits bounded termination. A
    // registry PID alone never authorizes killing a browser that did not exit.
    if (!child || child.exitCode !== null || child.signalCode !== null) throw error;
    if (state.evidence) recordBrowserEvent(state.evidence, state, 'close-fallback', { error: error.message });
    child.kill();
    await waitGone(state, child);
  }
}

function validState(state) {
  return state.version === 1 && Number.isInteger(state.pid) && state.pid > 0 &&
    Number.isInteger(state.port) && state.port > 0 && state.port < 65536 &&
    typeof state.profile === 'string' && dirname(resolve(state.profile)) === resolve(profileRoot) &&
    resolve(state.profile).startsWith(join(profileRoot, profilePrefix));
}

function removeLocalProfile(state) {
  // Only this repository's generated profile children may be removed.
  if (!validState(state) || dirname(resolve(state.profile)) !== resolve(profileRoot)) return;
  // Windows can release profile handles shortly after the browser process exits.
  rmSync(state.profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 });
}

export async function startBrowser({ chrome, port, args = [] }) {
  chrome = resolveTestBrowser({ override: chrome });
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Invalid CDP port');
  if (args.some(arg => /^--(user-data-dir|remote-debugging-port|remote-debugging-pipe)(=|$)/.test(arg))) {
    throw new Error('Browser identity flags are controlled by browser_session');
  }
  for (const registeredPath of [statePath]) {
    if (!existsSync(registeredPath)) continue;
    const state = JSON.parse(readFileSync(registeredPath, 'utf8'));
    if (!validState(state)) throw new Error('Invalid owned-browser state; refusing to close any browser');
    await closeOwned(state);
    if (state.evidence) recordBrowserEvent(state.evidence, state, 'closed-before-replacement');
    removeLocalProfile(state);
    unlinkSync(registeredPath);
  }
  // Refuse an unrelated listener before spawn; never attach by port alone.
  await new Promise((done, reject) => {
    const server = createServer();
    server.once('error', () => reject(new Error(`CDP port ${port} is already occupied`)));
    server.listen(port, '127.0.0.1', () => server.close(done));
  });
  mkdirSync(profileRoot, { recursive: true });
  mkdirSync(dirname(statePath), { recursive: true });
  const profile = mkdtempSync(join(profileRoot, profilePrefix));
  // This is the interactive debug browser requested by the user. Hiding its
  // window keeps document.visibilityState hidden and can suspend MSE startup.
  // Unattended checks opt into Chrome's --headless=new instead.
  const child = spawn(chrome, ['--remote-debugging-port=' + port, '--user-data-dir=' + profile, '--enable-automation', ...args], { stdio: 'ignore', detached: true, windowsHide: false });
  await new Promise((done, reject) => { child.once('spawn', done); child.once('error', reject); });
  const state = { version: 1, pid: child.pid, port, profile, binary: chrome };
  try { writeFileSync(statePath, JSON.stringify(state), { flag: 'wx' }); }
  catch (error) {
    child.kill();
    await waitGone(state, child);
    throw error;
  }
  try {
    const evidenceDir = join(sessionRoot, 'evidence');
    mkdirSync(evidenceDir, { recursive: true });
    state.evidence = join(evidenceDir, `${Date.now()}-${child.pid}.json`);
    writeFileSync(statePath, JSON.stringify(state));
    recordBrowserEvent(state.evidence, state, 'spawned');
    let info = null;
    for (let n = 0; n < 100; ++n) {
      if (child.exitCode !== null || child.signalCode !== null) throw new Error('Owned browser exited before CDP became ready');
      info = await version(port);
      if (info) break;
      await pause(100);
    }
    if (!info) throw new Error('Owned browser CDP did not become ready');
    const commandLine = await command(info.webSocketDebuggerUrl, 'Browser.getBrowserCommandLine');
    const processes = await command(info.webSocketDebuggerUrl, 'SystemInfo.getProcessInfo');
    verifyBrowserIdentity(state, commandLine.arguments || [], processes.processInfo || []);
    recordBrowserEvent(state.evidence, state, 'ready', { commandLine: commandLine.arguments, browser: info.Browser });
  } catch (error) {
    // Never send Browser.close to an endpoint whose ownership is in doubt.
    // The ChildProcess handle still identifies our own freshly spawned child.
    if (child.exitCode === null && child.signalCode === null) child.kill();
    for (let n = 0; n < 100 && child.exitCode === null && child.signalCode === null; ++n) await pause(100);
    if (child.exitCode !== null || child.signalCode !== null) unlinkSync(statePath);
    if (state.evidence) recordBrowserEvent(state.evidence, state, 'startup-failed', { error: error.message });
    throw error;
  }
  return {
    profile, pid: state.pid, binary: chrome, evidence: state.evidence,
    async close({ keepOpen = false, cleanup = true } = {}) {
      if (keepOpen) { recordBrowserEvent(state.evidence, state, 'kept-open'); child.unref(); return; }
      await closeOwned(state, child);
      recordBrowserEvent(state.evidence, state, 'closed');
      if (cleanup) {
        removeLocalProfile(state);
        if (existsSync(statePath) && JSON.parse(readFileSync(statePath, 'utf8')).pid === state.pid) unlinkSync(statePath);
      }
    },
  };
}
