import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';
import { runInNewContext } from 'node:vm';

import { resolveTestBrowser, verifyBrowserIdentity } from '../tools/lib/browser_session.mjs';

test('dedicated Chrome discovery uses only the declared install or explicit binary', () => {
  const root = mkdtempSync(join(tmpdir(), 'gvhmr-browser-test-'));
  try {
    const binary = join(root, 'build/tools/chrome-for-testing/chrome-win64/chrome.exe');
    assert.throws(() => resolveTestBrowser({ root, platform: 'win32', override: '' }), /Chrome for Testing not found/);
    mkdirSync(dirname(binary), { recursive: true });
    writeFileSync(binary, 'test binary path only; never executed');
    assert.equal(resolveTestBrowser({ root, platform: 'win32', override: '' }), binary);
    assert.equal(resolveTestBrowser({ platform: 'unsupported', override: binary }), binary);
    assert.throws(() => resolveTestBrowser({ root, platform: 'unsupported', override: '' }), /not found/);
    assert.throws(() => resolveTestBrowser({ root, override: join(root, 'missing.exe') }), /not found/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('closing requires the exact owned binary, profile, CDP port and browser PID', () => {
  const state = { binary: fileURLToPath(import.meta.url), profile: '/owned-profile', port: 54321, pid: 123 };
  const args = [state.binary, '--user-data-dir=' + state.profile, '--remote-debugging-port=' + state.port];
  const processes = [{ type: 'browser', id: state.pid }];
  verifyBrowserIdentity(state, args, processes);
  for (const field of ['binary', 'profile', 'port', 'pid']) {
    const changed = { ...state, [field]: typeof state[field] === 'number' ? state[field] + 1 : state[field] + '-other' };
    assert.throws(() => verifyBrowserIdentity(changed, args, processes), /Refusing to close browser/);
  }
  assert.throws(() => verifyBrowserIdentity({ ...state, binary: undefined }, args, processes), /Refusing to close/);
  assert.throws(() => verifyBrowserIdentity(state, args, [{ type: 'renderer', id: state.pid }]), /Refusing to close/);
});

test('CDP close waits for server shutdown and settles once on errors or timeout', async () => {
  const code = readFileSync(new URL('../tools/lib/browser_session.mjs', import.meta.url), 'utf8');
  const commandCode = code.slice(code.indexOf('async function command('), code.indexOf('\nasync function waitGone('));
  for (const scenario of ['server-close', 'protocol-error', 'timeout', 'regular-reply']) {
    let socket, expire, cleared = 0, settled = false;
    class MockSocket {
      constructor() { socket = this; this.listeners = new Map(); this.closes = 0; }
      addEventListener(event, callback) { this.listeners.set(event, callback); }
      removeEventListener(event, callback) { assert.equal(this.listeners.get(event), callback); this.listeners.delete(event); }
      send(message) { this.request = JSON.parse(message); }
      close() { this.closes++; this.dispatch('close'); }
      dispatch(event, data = {}) { this.listeners.get(event)?.(data); }
    }
    const command = runInNewContext(`(${commandCode})`, {
      WebSocket: MockSocket, setTimeout: callback => { expire = callback; return 1; },
      clearTimeout: () => { cleared++; },
    });
    const pending = command('ws://unused', scenario === 'regular-reply' ? 'SystemInfo.getProcessInfo' : 'Browser.close');
    pending.then(() => { settled = true; }, () => { settled = true; });
    socket.dispatch('open');
    assert.equal(socket.request.id, 1);
    if (scenario === 'server-close') {
      socket.dispatch('message', { data: JSON.stringify({ id: 1, result: {} }) });
      await Promise.resolve();
      assert.equal(settled, false, 'a reply must not complete Browser.close');
      assert.equal(socket.closes, 0, 'the client must not interrupt browser shutdown');
      socket.dispatch('close');
      await pending;
      assert.equal(socket.closes, 0);
    } else if (scenario === 'protocol-error') {
      socket.dispatch('message', { data: JSON.stringify({ id: 1, error: { message: 'closing denied' } }) });
      await assert.rejects(pending, /closing denied/);
      assert.equal(socket.closes, 1);
    } else if (scenario === 'timeout') {
      expire();
      await assert.rejects(pending, /timed out/);
      assert.equal(socket.closes, 1);
    } else {
      socket.dispatch('message', { data: JSON.stringify({ id: 1, result: { processInfo: [] } }) });
      const result = await pending;
      assert.equal(result.processInfo.length, 0);
      assert.equal(socket.closes, 1);
    }
    socket.dispatch('close');
    socket.dispatch('error');
    assert.equal(cleared, 1);
    assert.equal(socket.listeners.size, 0);
  }
});

test('failed shutdown can terminate only a fresh owned handle after identity verification', async () => {
  const code = readFileSync(new URL('../tools/lib/browser_session.mjs', import.meta.url), 'utf8');
  const closeCode = code.slice(code.indexOf('async function closeOwned('), code.indexOf('\nfunction validState('));
  for (const scenario of ['fresh-child', 'registry-only', 'wrong-identity']) {
    const state = { binary: fileURLToPath(import.meta.url), profile: '/owned-profile', port: 54321, pid: 123,
      evidence: '/unused-evidence' };
    let kills = 0, closeRequests = 0, exitChecks = 0;
    const events = [];
    const child = { exitCode: null, signalCode: null, kill() { kills++; this.exitCode = 1; } };
    const closeOwned = runInNewContext(`(${closeCode})`, {
      version: async () => ({ webSocketDebuggerUrl: 'ws://unused' }), verifyBrowserIdentity,
      command: async (_url, method) => {
        if (method === 'Browser.getBrowserCommandLine') return { arguments: [state.binary,
          '--user-data-dir=' + state.profile, '--remote-debugging-port=' + state.port] };
        if (method === 'SystemInfo.getProcessInfo') return { processInfo: [
          { type: 'browser', id: scenario === 'wrong-identity' ? state.pid + 1 : state.pid },
        ] };
        closeRequests++;
        throw new Error('shutdown timed out');
      },
      waitGone: async (_state, owned) => { assert.equal(owned, child); assert.equal(owned.exitCode, 1); exitChecks++; },
      recordBrowserEvent: (_path, _state, event) => events.push(event),
    });
    const pending = closeOwned(state, scenario === 'registry-only' ? null : child);
    if (scenario === 'fresh-child') {
      await pending;
      assert.deepEqual(events, ['close-fallback']);
      assert.equal(kills, 1);
      assert.equal(exitChecks, 1);
    } else {
      await assert.rejects(pending, scenario === 'registry-only' ? /shutdown timed out/ : /Refusing to close/);
      assert.equal(kills, 0);
      assert.equal(exitChecks, 0);
    }
    assert.equal(closeRequests, scenario === 'wrong-identity' ? 0 : 1);
  }
});

test('the real viewer initializer honors exact scan links inside and outside rally bounds', async () => {
  const code = readFileSync(new URL('../hmr4d/lidar_mocap/rally_viewer_client.js', import.meta.url), 'utf8');
  const initializer = code.slice(code.lastIndexOf('(async()=>{try{'));
  for (const [scan, expectedTime, expectedRally, expectedLoop] of [
    [3, 30, 1, false], [1, 10, 2, true], [99, null, 1, true], ['invalid', null, 1, true],
  ]) {
    const elements = new Map();
    const history = [];
    const source = { scan_times: [0, 10, 20, 30], rallies: [
      { index: 1, begin: 0, end: 5 }, { index: 2, begin: 10, end: 20 },
    ] };
    const scope = {
      URLSearchParams, Uint16Array, Uint8Array, atob, manifest: null, faces: null,
      source, sourceId: null, clip: null, lastTick: 0,
      window: {}, location: { search: '?source=actual&rally=1&scan=' + scan },
      fetch: async () => ({ json: async () => ({ faces: '', rally_window_minutes: 1,
        unique_paired_input_scans: 4, sources: { actual: source } }) }),
      $: id => { if (!elements.has(id)) elements.set(id, { checked: true, replaceChildren() {} }); return elements.get(id); },
      Option: function () {}, history: { replaceState: (...args) => history.push(args.at(-1)) },
      resolveReady() {}, status() {}, setInterval() {}, requestAnimationFrame() {}, update() {},
      performance: { now: () => 0 },
      selectSource: async (sid, rally) => { scope.sourceId = sid; scope.clip = source.rallies.find(r => r.index === rally); },
      selectRally: async index => { scope.clip = source.rallies.find(r => r.index === index); },
      seek: async time => { scope.sought = time; },
    };
    await runInNewContext(initializer, scope);
    assert.equal(scope.window.viewerReady, true, elements.get('error')?.textContent);
    assert.equal(scope.sought ?? null, expectedTime);
    assert.equal(scope.clip.index, expectedRally);
    assert.equal(elements.get('loop')?.checked ?? true, expectedLoop);
    if (expectedTime !== null) assert.equal(history.at(-1), `?source=actual&rally=${expectedRally}&scan=${scan}`);
  }
});
