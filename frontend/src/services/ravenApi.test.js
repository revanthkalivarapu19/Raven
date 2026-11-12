import test from 'node:test';
import assert from 'node:assert/strict';
import { followUp, recheckClaim, routeConversation, verify } from './ravenApi.js';
import { classifyInputIntent } from './followUpIntent.js';

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

test('routes explanation and source questions without treating them as claims', () => {
  assert.equal(classifyInputIntent('Why is this result uncertain?'), 'EXPLAIN_RESULT');
  assert.equal(classifyInputIntent('What evidence was considered?'), 'ASK_ABOUT_SOURCE');
  assert.equal(classifyInputIntent('Which source supports that?'), 'ASK_ABOUT_SOURCE');
});

test('routes source search and verification commands to recheck', () => {
  assert.equal(classifyInputIntent('Check other sources.'), 'MORE_EVIDENCE');
  assert.equal(classifyInputIntent('Verify the claim again.'), 'MORE_EVIDENCE');
  assert.equal(classifyInputIntent('Can you search more sources?'), 'MORE_EVIDENCE');
  assert.equal(classifyInputIntent('Check other sources and verify the claim again.'), 'MORE_EVIDENCE');
  assert.equal(classifyInputIntent('Find more evidence.'), 'MORE_EVIDENCE');
});

test('routes a new factual statement to a new claim', () => {
  assert.equal(classifyInputIntent('Smoking increases the risk of lung cancer.'), 'NEW_VERIFICATION');
  assert.equal(classifyInputIntent('Did the Federal Reserve raise rates in 2022?'), 'NEW_VERIFICATION');
});

test('routes general conversation without verification', () => {
  assert.equal(classifyInputIntent('Hello!'), 'GENERAL_QUERY');
  assert.equal(classifyInputIntent('How can you help me?'), 'GENERAL_QUERY');
});

test('backend conversation router receives each supported intent and current session context', async () => {
  setBackendMode(true);
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, body: JSON.parse(options.body) });
    const requestBody = calls.at(-1).body;
    return response({ body: requestBody.intent === 'NEW_VERIFICATION' || requestBody.intent === 'MORE_EVIDENCE'
      ? validResult
      : { intent: requestBody.intent, answer: 'Handled without verification.' } });
  };
  const intents = [
    'NEW_VERIFICATION', 'EXPLAIN_RESULT', 'MORE_EVIDENCE',
    'ASK_ABOUT_SOURCE', 'GENERAL_QUERY',
  ];
  for (const intent of intents) {
    await routeConversation({
      intent, message: 'turn text', sessionState: { current_claim: 'stored claim' },
      currentResult: validResult,
    });
  }
  assert.deepEqual(calls.map((call) => call.url), intents.map(() => '/api/conversation'));
  assert.deepEqual(calls.map((call) => call.body.intent), intents);
  assert.equal(calls[2].body.session_state.current_claim, 'stored claim');
  assert.equal(calls[3].body.current_result.verdict, 'real');
});

test('recheck sends the original claim text to the verification API', async () => {
  setBackendMode(true);
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, body: JSON.parse(options.body) });
    return response();
  };
  const originalClaim = 'Smoking increases the risk of lung cancer.';
  await recheckClaim({ text: originalClaim, image: null });
  assert.equal(calls[0].url, '/api/verify');
  assert.equal(calls[0].body.text, originalClaim);
  assert.notEqual(calls[0].body.text, 'Check other sources and verify the claim again.');
});
