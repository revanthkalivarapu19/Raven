const MORE_EVIDENCE_PATTERNS = [
  /\b(check|search|find|look for)\b.*\b(other|more|additional|new)\s+sources?\b/i,
  /\b(find|search for|look for)\b.*\b(more|additional|new)\s+evidence\b/i,
  /\b(check|search|find|look for)\b.*\b(other|more|additional|new)\b/i,
  /\b(verify|recheck|re-verify)\b.*\b(again|claim|evidence|sources?)\b/i,
];

const SOURCE_PATTERN = /\b(what|which|show|list|open)\b.*\b(source|sources|evidence|url|citation)\b|\b(source|citation)\s+(url|details|metadata)\b|\bevidence\s+(was|were)\s+considered\b/i;
const EXPLAIN_PATTERN = /\b(why|explain|how come|tell me more|explain more|compare|strongest|confidence|uncertain|verdict|result)\b/i;
const GENERAL_PATTERN = /^(hi|hello|hey|thanks|thank you|good morning|good afternoon|good evening|who are you|what can you do|help|tell me a joke)[!.? ]*$/i;
const RAVEN_HELP_PATTERN = /\b(how does raven work|what is raven|how can you help me)\b/i;

export function classifyInputIntent(value = '') {
  const text = String(value).trim();
  if (!text) return 'NEW_VERIFICATION';
  if (MORE_EVIDENCE_PATTERNS.some((pattern) => pattern.test(text))) return 'MORE_EVIDENCE';
  if (SOURCE_PATTERN.test(text)) return 'ASK_ABOUT_SOURCE';
  if (EXPLAIN_PATTERN.test(text)) return 'EXPLAIN_RESULT';
  if (GENERAL_PATTERN.test(text) || RAVEN_HELP_PATTERN.test(text)) return 'GENERAL_QUERY';
  return 'NEW_VERIFICATION';
}

export function recheckInputForTurn(claimTurn) {
  return { text: claimTurn?.text || '', image: claimTurn?.image || null };
}
