const RECHECK_PATTERNS = [
  /\b(check|search|find|look for)\b.*\b(other|more|additional|new)\s+sources?\b/i,
  /\b(check|verify|recheck|re-verify|search|find)\b.*\b(again|claim|evidence|sources?)\b/i,
  /\b(find|search for|look for)\b.*\b(more|additional|new)\s+evidence\b/i,
  /\b(check|search|find|look for)\b.*\b(other|more|additional|new)\b/i,
  /\b(verify|recheck|re-verify)\b/i,
];

export function classifyInputIntent(value = '') {
  const text = String(value).trim();
  if (!text) return 'NEW_CLAIM';

  if (RECHECK_PATTERNS.some((pattern) => pattern.test(text))) return 'RECHECK_CLAIM';
  if (/\b(what|which|show|list)\b.*\b(evidence|sources?)\b|\bevidence\s+(was|were)\s+considered\b|\b(which|what)\s+source\b/i.test(text)) {
    return 'SHOW_EVIDENCE';
  }
  if (/\b(why|explain|how come|tell me more|explain more|compare|first claim|previous claim|strongest|what does)\b|\b(uncertain|confidence|result)\b/i.test(text)) {
    return 'EXPLAIN_RESULT';
  }
  return 'NEW_CLAIM';
}

export function recheckInputForTurn(claimTurn) {
  return { text: claimTurn?.text || '', image: claimTurn?.image || null };
}
