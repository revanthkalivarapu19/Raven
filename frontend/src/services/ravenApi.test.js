import test from 'node:test';
import assert from 'node:assert/strict';
import { followUp, verify } from './ravenApi.js';

const validResult = {
  verdict: 'real', confidence: 85, status: 'Reviewed.', why: 'Supported.', evidence: [],
};

function setBackendMode(enabled) {
  globalThis.window = { __RAVEN_USE_REAL_BACKEND__: enabled };
}

function response({ ok = true, status = 200, body = validResult } = {}) {
  return { ok, status, json: async () => body };
}

test('real backend success returns the API response', async () => {
  setBackendMode(true);
  globalThis.fetch = async () => response();
  assert.deepEqual(await verify({ text: 'claim' }), validResult);
});

test('real backend non-2xx does not return a demo result', async () => {
  setBackendMode(true);
  globalThis.fetch = async () => response({ ok: false, status: 500 });
  await assert.rejects(verify({ text: 'Scientists developed a new AI model to detect cancer earlier.' }),
    /Verification service is unavailable/);
});

test('real backend network failure does not return a demo result', async () => {
  setBackendMode(true);
  globalThis.fetch = async () => { throw new TypeError('private network details'); };
  await assert.rejects(verify({ text: 'claim' }), /Verification service is unavailable/);
});

test('malformed real backend response does not return a demo result', async () => {
  setBackendMode(true);
  globalThis.fetch = async () => response({ body: { verdict: 'real' } });
  await assert.rejects(verify({ text: 'claim' }), /Verification service is unavailable/);
});

test('explicit demo mode retains static verification fallback', async () => {
  setBackendMode(false);
  globalThis.fetch = async () => { throw new Error('fetch must not be called in demo mode'); };
  const result = await verify({ text: 'Scientists developed a new AI model to detect cancer earlier.' });
  assert.equal(result.verdict, 'real');
  assert.equal(result.evidence[0].source, 'Reuters');
});

test('real backend follow-up failure does not return canned text', async () => {
  setBackendMode(true);
  globalThis.fetch = async () => response({ ok: false, status: 503 });
  await assert.rejects(followUp({ question: 'Which source is strongest?' }),
    /Verification service is unavailable/);
});
