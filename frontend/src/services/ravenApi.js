/**
 * RAVEN API Service Layer
 * Connects the frontend interface to fact-checking services.
 * Includes frontend demo/mock fallback data for demonstration prototype.
 */

/* ==========================================================================
   FRONTEND DEMO / MOCK DATA (Isolate for clean removal once backend connects)
   ========================================================================== */

const DEMO_CLAIM_1_CANCER = {
  verdict: 'real',
  confidence: 78,
  status:
    'Multiple retrieved sources are broadly consistent with the central claim. The evidence agrees on the reported development and no major contradiction was identified.',
  why: 'Multiple retrieved sources are broadly consistent with the central claim. The evidence agrees on the reported development and no major contradiction was identified.',
  evidence: [
    {
      source: 'Reuters',
      stance: 'support',
      relevance: 92,
      trust: 92,
      excerpt:
        'Scientists develop a new AI model to detect cancer earlier, according to a published study.',
      note: 'Corroborates findings published in primary peer-reviewed clinical research.',
    },
    {
      source: 'BBC',
      stance: 'support',
      relevance: 89,
      trust: 89,
      excerpt:
        'Researchers report promising results from an AI model designed to identify cancer at an earlier stage.',
      note: 'Independent science reporting quoting lead institutional researchers.',
    },
    {
      source: 'The Guardian',
      stance: 'support',
      relevance: 76,
      trust: 76,
      excerpt:
        'Researchers report progress toward earlier cancer detection using artificial intelligence.',
      note: 'Provides comparative context on clinical deployment timelines.',
    },
  ],
};

const DEMO_CLAIM_2_COFFEE = {
  verdict: 'uncertain',
  confidence: 54,
  status:
    'Some evidence reports an association with cardiovascular outcomes, but the retrieved material does not establish a direct or universal prevention effect.',
  why: 'Some evidence reports an association with cardiovascular outcomes, but the retrieved material does not establish a direct or universal prevention effect.',
  evidence: [
    {
      source: 'WHO',
      stance: 'limiting',
      relevance: 81,
      trust: 81,
      excerpt:
        'The available evidence does not establish the broad prevention effect stated by the claim.',
      note: 'Global health guidance notes lack of randomized clinical prevention evidence.',
    },
    {
      source: 'Nature',
      stance: 'unclear',
      relevance: 74,
      trust: 74,
      excerpt:
        'Observational findings indicate an association, but do not by themselves establish causality.',
      note: 'Highlights confounding lifestyle factors present across long-term observational surveys.',
    },
    {
      source: 'JAMA',
      stance: 'limiting',
      relevance: 69,
      trust: 69,
      excerpt:
        'Further research is needed before a broad preventive effect can be established.',
      note: 'Clinical cardiology review emphasizing dose-response variability and population differences.',
    },
  ],
};

const DEMO_CLAIM_3_ALZHEIMER = {
  verdict: 'real',
  confidence: 82,
  status:
    'Clinical trial phase-3 results published in medical journals confirm therapeutic efficacy for the newly approved treatment.',
  why: 'Multiple peer-reviewed clinical trials corroborate measurable reduction in disease progression markers.',
  evidence: [
    {
      source: 'FDA Announcement',
      stance: 'support',
      relevance: 95,
      trust: 96,
      excerpt:
        'The FDA approved the novel therapy following demonstrated reduction of cognitive decline in Phase 3 trials.',
      note: 'Official regulatory approval documentation with clinical trial milestones.',
    },
    {
      source: 'The Lancet',
      stance: 'support',
      relevance: 88,
      trust: 92,
      excerpt:
        'Multi-center clinical trials demonstrate statistically significant reduction in biomarker progression over 18 months.',
      note: 'Peer-reviewed international clinical trial findings across diverse patient cohorts.',
    },
    {
      source: 'Associated Press',
      stance: 'support',
      relevance: 78,
      trust: 85,
      excerpt:
        'Neurology centers begin protocol adoption following new treatment regulatory milestone.',
      note: 'Journalistic summary of clinical rollout across regional medical networks.',
    },
  ],
};

const DEMO_CLAIM_4_ARTICLE = {
  verdict: 'uncertain',
  confidence: 61,
  status:
    'The article presents selective quotes that partially substantiate context but omit key qualifying caveats from the source report.',
  why: 'Retrieved source text contains material limitations that weaken the headline assertion.',
  evidence: [
    {
      source: 'Reuters Fact Check',
      stance: 'unclear',
      relevance: 88,
      trust: 90,
      excerpt:
        'The original briefing included qualifying conditions that were omitted in subsequent syndication.',
      note: 'Editorial comparison of primary press release vs viral headline claims.',
    },
    {
      source: 'FactCheck.org',
      stance: 'limiting',
      relevance: 82,
      trust: 88,
      excerpt:
        'Cross-referencing the underlying study reveals the findings were limited to preliminary laboratory models.',
      note: 'Methodological review identifying unverified generalizations in popular reporting.',
    },
    {
      source: 'AP News',
      stance: 'limiting',
      relevance: 75,
      trust: 82,
      excerpt:
        'Expert commentary cautions against broad generalizations before human trials conclude.',
      note: 'Independent specialist review corroborating initial research boundaries.',
    },
  ],
};

const DEMO_CLAIM_DEFAULT = {
  verdict: 'real',
  confidence: 78,
  status:
    'Multiple retrieved sources are broadly consistent with the central claim. The evidence agrees on the reported development and no major contradiction was identified.',
  why: 'Multiple retrieved sources are broadly consistent with the central claim. The evidence agrees on the reported development and no major contradiction was identified.',
  evidence: [
    {
      source: 'Reuters',
      stance: 'support',
      relevance: 92,
      trust: 92,
      excerpt:
        'Independent reporting corroborates the central facts in the submitted claim with primary public records.',
      note: 'Wire service coverage cross-referenced against official publications.',
    },
    {
      source: 'BBC',
      stance: 'support',
      relevance: 89,
      trust: 89,
      excerpt:
        'Institutional statements confirm the underlying events and timeline described in the claim.',
      note: 'Verified journalistic account with direct sourcing from responsible bodies.',
    },
    {
      source: 'The Guardian',
      stance: 'support',
      relevance: 76,
      trust: 76,
      excerpt:
        'Primary reports and corroborating documentation indicate broad factual alignment without major discrepancies.',
      note: 'Secondary editorial analysis corroborating core assertions.',
    },
  ],
};

let claimTurnIndex = 0;

export function resetDemoSession() {
  claimTurnIndex = 0;
}

/**
 * Match a claim string to realistic static demo data.
 * @param {string} text
 * @returns {Object} Static demo result
 */
function getStaticDemoResult(text = '') {
  const lower = (text || '').toLowerCase();

  if (lower.includes('cancer') || lower.includes('ai model') || lower.includes('earlier')) {
    return DEMO_CLAIM_1_CANCER;
  }
  if (lower.includes('coffee') || lower.includes('heart') || lower.includes('drink')) {
    return DEMO_CLAIM_2_COFFEE;
  }
  if (lower.includes('alzheimer')) {
    return DEMO_CLAIM_3_ALZHEIMER;
  }
  if (lower.includes('article') || lower.includes('support')) {
    return DEMO_CLAIM_4_ARTICLE;
  }

  // If user inputs generic text ("Hello", arbitrary claim, etc.), cycle through realistic claims
  const cycle = [DEMO_CLAIM_1_CANCER, DEMO_CLAIM_2_COFFEE, DEMO_CLAIM_3_ALZHEIMER];
  const result = cycle[claimTurnIndex % cycle.length] || DEMO_CLAIM_DEFAULT;
  claimTurnIndex++;
  return result;
}

/* ==========================================================================
   SERVICE API EXPORTS
   ========================================================================== */

/**
 * Verify a claim using text and/or an attached screenshot.
 * // TODO(backend): Connect real backend pipeline via POST /api/verify
 * 
 * @param {Object} params
 * @param {string} params.text - The claim text in English or Telugu
 * @param {Object|string|null} params.image - Image file or data URL
 * @returns {Promise<Object>} Verification result matching Result data shape
 */
export async function verify({ text = '', image: _image = null } = {}) {
  // Short realistic processing delay
  await new Promise((resolve) => setTimeout(resolve, 600));

  // 1. Attempt real backend if configured
  try {
    if (typeof window !== 'undefined' && window.__RAVEN_USE_REAL_BACKEND__) {
      const response = await fetch('/api/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, image: _image }),
      });
      if (response.ok) {
        const data = await response.json();
        if (data && data.verdict) {
          return data;
        }
      }
    }
  } catch {
    // Backend offline or unreachable: fall through to demo fallback
  }

  // 2. Frontend demo / mock fallback
  return getStaticDemoResult(text);
}

/**
 * Answer a follow-up question related to a previous result.
 * // TODO(backend): Connect real backend pipeline via POST /api/followup
 * 
 * @param {Object} params
 * @param {string} params.question - Follow-up user question
 * @param {Object} params.result - The parent verification result
 * @param {Array} params.turns - Conversation history turns
 * @returns {Promise<string>} Plain text answer under RAVEN label
 */
export async function followUp({ question = '', result: _result = null, turns: _turns = [] } = {}) {
  await new Promise((resolve) => setTimeout(resolve, 400));

  // 1. Attempt real backend if configured
  try {
    if (typeof window !== 'undefined' && window.__RAVEN_USE_REAL_BACKEND__) {
      const response = await fetch('/api/followup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, result: _result }),
      });
      if (response.ok) {
        const data = await response.json();
        if (data && typeof data.answer === 'string') {
          return data.answer;
        }
      }
    }
  } catch {
    // Backend offline or unreachable: fall through to demo fallback
  }

  // 2. Frontend demo contextual follow-up matching
  const lower = (question || '').toLowerCase();

  // Contextual comparison between claims
  if (lower.includes('compare') || lower.includes('first claim') || lower.includes('previous claim')) {
    return 'The first claim has stronger agreement across the retrieved sources, while the second uses broader causal wording and has more limiting evidence. This is why the confidence is lower for the second claim.';
  }

  // Follow-up on confidence percentage
  if (lower.includes('confidence') || lower.includes('why only') || lower.includes('78%') || lower.includes('54%')) {
    return 'The confidence score is derived from publisher credibility scores, direct quotation relevance, and consensus across independent reports. While the research is corroborated by major institutions, preliminary clinical stages and sample boundaries account for the remaining margin.';
  }

  // Follow-up on strongest source
  if (lower.includes('strongest') || lower.includes('which source') || lower.includes('best source') || lower.includes('most credible')) {
    return 'The highest-rated source in the retrieved record is Reuters (92%), followed closely by the BBC (89%), owing to direct citations of published institutional findings.';
  }

  // Follow-up on coffee / heart disease / uncertainty
  if (lower.includes('coffee') || lower.includes('heart') || lower.includes('uncertain')) {
    return 'The WHO and JAMA reviews emphasize that observational studies show associative correlation rather than clinical causation. Confounding lifestyle factors mean a universal prevention effect cannot be established with certainty.';
  }

  // General fallback follow-up
  return 'The retrieved evidence addresses the core assertions in this claim. Independent reporting and primary documentation corroborate the factual timeline without major contradictions identified.';
}
