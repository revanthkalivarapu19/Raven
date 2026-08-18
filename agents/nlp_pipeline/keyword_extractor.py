"""
Extracts key terms/phrases from an existing spaCy Doc.
"""

def extract_keywords(doc, max_keywords: int = 10) -> list[str]:
    """
    Return a list of deduplicated keyword/keyphrase strings,
    derived from noun chunks and proper nouns in the Doc.
    Limit the result to max_keywords items.
    Guard against doc being None (return empty list).
    """
    if doc is None:
        return []
        
    candidates = []
    chunk_tokens = set()
    
    # 1. Collect all doc.noun_chunks as candidate keyphrases
    for chunk in doc.noun_chunks:
        text = chunk.text.strip()
        # Avoid purely stopword or single char chunks
        if len(text) > 1 and not all(token.is_stop for token in chunk):
            candidates.append(text)
            # Record tokens covered by this noun chunk
            for token in chunk:
                chunk_tokens.add(token.i)
                
    # 2. Collect individual tokens (PROPN, NOUN) not covered by a noun chunk
    for token in doc:
        if token.i not in chunk_tokens:
            if token.pos_ in ("PROPN", "NOUN"):
                text = token.text.strip()
                if len(text) > 1 and not token.is_stop:
                    candidates.append(text)
                    
    # 3. Deduplicate (case-insensitive) while preserving first-seen order
    seen = set()
    deduped_keywords = []
    for candidate in candidates:
        lower_candidate = candidate.lower()
        if lower_candidate not in seen:
            seen.add(lower_candidate)
            deduped_keywords.append(candidate)
            
    # 4. Filter out chunks that are purely stopwords or single characters.
    # (Already handled during collection above)
            
    # 5. Return up to max_keywords items
    return deduped_keywords[:max_keywords]

if __name__ == "__main__":
    from agents.nlp_pipeline.loader import get_doc
    
    sample_text = "Prime Minister Modi announced a new scheme in Hyderabad on Monday, according to officials."
    doc = get_doc(sample_text)
    
    keywords = extract_keywords(doc)
    print("Keywords:")
    for kw in keywords:
        print(kw)
