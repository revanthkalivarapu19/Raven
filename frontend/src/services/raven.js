/**
 * RAVEN service layer — the boundary between the interface and verification.
 *
 * Everything the conversation asks for goes through `request()` below. No
 * component decides a verdict, builds evidence, or writes reasoning; they ask
 * for a turn and render what comes back. Swapping the mock for the real
 * backend should mean editing this file and nothing else.
 *
 * ── Replacing the mock ────────────────────────────────────────────────
 * `request()` is the seam. Its signature and its return value are the contract:
 * keep them and every component keeps working.
 *
 *   request({ kind: 'assessment' | 'explanation' | 'evidence' | 'answer',
 *             payload }) -> Promise<Turn>
 *
 * The real transport is not defined yet, so the shapes below are a contract
 * rather than invented endpoints. Nothing here assumes a method, path or
 * streaming model. See README.md for the full shape reference.
 */

import { SCENARIOS, STARTERS, detectScenario } from './mockData'

/**
 * The entire simulation surface, in one place. Delete this block, and the
 * mock branches it guards, when a real backend exists.
 */
export const MOCK = {
  enabled: true,
  /** Per interaction, so the interface is exercised against both a short wait
   *  and a long one. [min, max] in milliseconds. */
  latency: {
    assessment: [1900, 2600],
    explanation: [1100, 1600],
    evidence: [1300, 1900],
  },
  /** A message matching this makes the next turn fail, so the error state and
   *  the retry path can be reached on purpose. Documented in README.md. */
  failureTrigger: /\bsimulate (error|failure)\b/i,
}

/* -------------------------------------------------------------------------
   Errors
   ------------------------------------------------------------------------- */

/** The error shape the interface knows how to render. */
export class RavenError extends Error {
  constructor(userMessage, { code = 'unavailable', retryable = true } = {}) {
    super(code)
    this.name = 'RavenError'
    this.userMessage = userMessage
    this.code = code
    this.retryable = retryable
  }
}

const SOURCES_UNREACHABLE = () =>
  new RavenError(
    'I couldn’t reach the sources just now. That’s a problem on my side, not with your claim.',
    { code: 'sources-unreachable' },
  )

/* -------------------------------------------------------------------------
   Mock helpers
   ------------------------------------------------------------------------- */

function wait(range) {
  const [min, max] = range
  const ms = min + Math.random() * (max - min)
  return new Promise((resolve) => setTimeout(resolve, ms))
}

/** Resolves a scenario from a claim string, so a follow-up stays in the same
 *  context as the check it belongs to. Falls back to the general scenario. */
function scenarioFor(claim = '', verdict = null) {
  const q = claim.toLowerCase()

  if (/(gold|rbi|coin|lakh|whatsapp|prize)/.test(q)) return SCENARIOS.financial
  if (/(strike|bus|transport|bengaluru|bangalore)/.test(q)) return SCENARIOS.local
  if (/(who|mpox|emergency|outbreak|vaccine)/.test(q)) return SCENARIOS.health
  if (/(photo|picture|image|bridge)/.test(q) || verdict === 'Fake') return SCENARIOS.image
  return SCENARIOS.general
}

function turnFrom(scenario, kind) {
  if (kind === 'assessment') {
    return {
      kind,
      verdict: scenario.verdict,
      intro: scenario.intro,
      claim: scenario.claim,
      confidence: scenario.confidence,
      reasoning: scenario.reasoning,
      evidence: scenario.evidence,
      followUps: scenario.followUps,
    }
  }

  if (kind === 'explanation') {
    return {
      kind,
      verdict: scenario.verdict,
      intro: null,
      reasoning: scenario.why,
      followUps: scenario.followUps.filter((item) => item.action !== 'why'),
    }
  }

  // 'evidence' — the wider search: sources first, then what it did not find.
  return {
    kind,
    verdict: scenario.verdict,
    intro: scenario.more.intro,
    evidence: scenario.more.evidence,
    note: scenario.more.note,
    followUps: scenario.followUps.filter((item) => item.action !== 'evidence'),
  }
}

function article(kind = '') {
  return /^[aeiou]/i.test(kind) ? 'an' : 'a'
}

function readableDate(value) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return String(value)
  return new Intl.DateTimeFormat(undefined, {
    day: 'numeric',
    month: 'long',
    year: 'numeric',
  }).format(date)
}

/* -------------------------------------------------------------------------
   The seam
   ------------------------------------------------------------------------- */

/**
 * The one entry point the interface uses.
 *
 * @param {Object} request
 * @param {'assessment'|'explanation'|'evidence'|'answer'} request.kind
 * @param {Object} request.payload  Shape depends on `kind` — see README.md.
 * @returns {Promise<Object>} Turn — shape depends on `kind` — see README.md.
 * @throws {RavenError} when the sources cannot be reached.
 */
export async function request({ kind, payload = {} }) {
  if (!MOCK.enabled) {
    // Where the real call goes. The mock below is the only thing removed:
    //
    //   const response = await fetch(`${API}/…`, {
    //     method: 'POST',
    //     headers: { 'content-type': 'application/json' },
    //     body: JSON.stringify({ kind, ...payload }),
    //   })
    //   if (!response.ok) throw SOURCES_UNREACHABLE()
    //   return response.json()
    //
    throw SOURCES_UNREACHABLE()
  }

  switch (kind) {
    case 'assessment':
      return mockAssessment(payload)
    case 'explanation':
      return mockExplanation(payload)
    case 'evidence':
      return mockEvidence(payload)
    case 'answer':
      return mockAnswer(payload)
    default:
      throw new RavenError('That request could not be understood.', {
        code: 'bad-request',
        retryable: false,
      })
  }
}

/* -------------------------------------------------------------------------
   Mock responses
   ------------------------------------------------------------------------- */

/** payload: { text?: string, image?: { dataUrl, name, width, height } | null } */
async function mockAssessment({ text = '', image = null }) {
  await wait(MOCK.latency.assessment)

  if (MOCK.failureTrigger.test(text)) throw SOURCES_UNREACHABLE()

  return turnFrom(SCENARIOS[detectScenario(text, image)], 'assessment')
}

/** payload: { claim: string, verdict: 'Real'|'Fake'|'Unverified' } */
async function mockExplanation({ claim, verdict }) {
  await wait(MOCK.latency.explanation)
  return turnFrom(scenarioFor(claim, verdict), 'explanation')
}

/** payload: { claim: string } */
async function mockEvidence({ claim }) {
  await wait(MOCK.latency.evidence)
  return turnFrom(scenarioFor(claim), 'evidence')
}

/** payload: { source: EvidenceItem } */
async function mockAnswer({ source }) {
  await wait(MOCK.latency.explanation)

  const kind = source.kind?.toLowerCase() ?? 'source'

  return {
    kind: 'answer',
    intro: 'Here’s what this source is.',
    reasoning: [
      `“${source.title}” was published by ${source.publisher} on ${readableDate(source.published)}. It is ${article(kind)} ${kind}.`,
      'The passage shown in the answer is the part that speaks to the claim directly. The rest of the document may cover other things, and the link goes to the original so you can read all of it.',
      'What matters about a source is whether its wording is specific enough to check, and whether it was written by the body that would actually know — not how widely it has been shared.',
    ],
    followUps: [],
  }
}

/* -------------------------------------------------------------------------
   Opening suggestions
   ------------------------------------------------------------------------- */

/** Suggested openings for the empty state. */
export function getStarters() {
  return STARTERS.map(({ id, label, detail }) => ({ id, label, detail }))
}
