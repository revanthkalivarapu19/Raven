/**
 * User-facing language for verdicts, confidence and progress.
 *
 * Kept in one place so the interface never improvises wording — and so that
 * nothing from the internal architecture (intents, agents, pipelines) can
 * leak into what a person reads.
 */

export const VERDICTS = {
  Real: {
    key: 'real',
    label: 'Real',
    stance: 'The sources support this claim',
    meaning: 'Reliable sources state the same thing, and nothing credible contradicts it.',
  },
  Fake: {
    key: 'fake',
    label: 'Fake',
    stance: 'The sources contradict this claim',
    meaning: 'Reliable sources contradict the claim, or it traces back to something fabricated.',
  },
  Unverified: {
    key: 'unverified',
    label: 'Unverified',
    stance: 'The evidence doesn’t settle this',
    meaning: 'The evidence is thin, too recent, or genuinely divided.',
  },
}

export function verdictMeta(verdict) {
  return VERDICTS[verdict] ?? VERDICTS.Unverified
}

/**
 * Turns a 0–1 reading into a word and a sentence a person can act on.
 *
 * The sentence depends on the verdict: high confidence in a Fake verdict means
 * the sources agree that the claim is wrong, which is a different statement
 * from sources agreeing that a claim is true.
 */
export function confidenceReading(value, verdict = null) {
  if (value >= 0.85) {
    return {
      level: 'High',
      note:
        verdict === 'Fake'
          ? 'The sources are consistent and specific about why this is wrong.'
          : 'The sources agree, and nothing credible disagrees.',
    }
  }
  if (value >= 0.6) {
    return {
      level: 'Moderate',
      note: 'Most sources point the same way, but there are gaps.',
    }
  }
  if (value >= 0.35) {
    return { level: 'Low', note: 'The evidence is thin, or the sources disagree.' }
  }
  return { level: 'Very low', note: 'Almost nothing addresses this claim directly.' }
}

/**
 * What the reasoning under an answer is called, per kind of answer.
 *
 * Named after the question it answers — a reader should know what the section
 * is for before they start it, and never be told which internal step produced
 * it.
 */
export const EXPLANATION_LABELS = {
  assessment: 'Why I read it this way',
  explanation: 'The reasoning',
  evidence: 'What the further sources show',
  answer: 'About this source',
}

export function explanationLabel(kind) {
  return EXPLANATION_LABELS[kind] ?? EXPLANATION_LABELS.assessment
}

/** What a source says about the claim, in plain words. */
export const STANCES = {
  supports: 'Supports the claim',
  contradicts: 'Contradicts the claim',
  unclear: 'Doesn’t settle it',
}

export function stanceLabel(stance) {
  return STANCES[stance] ?? STANCES.unclear
}

/**
 * Progress lines shown while a turn is being produced. They describe what is
 * happening in ordinary language — no agent names, no pipeline stages.
 */
export const STAGES = {
  assessment: [
    'Looking for sources',
    'Reading what they say',
    'Comparing the accounts',
    'Weighing the evidence',
  ],
  explanation: ['Going back over the sources', 'Checking the reasoning holds'],
  evidence: ['Searching further', 'Reading the additional sources'],
  answer: ['Looking that source up'],
}

export function stagesFor(kind) {
  return STAGES[kind] ?? STAGES.assessment
}
