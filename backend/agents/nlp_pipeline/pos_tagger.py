"""
Extracts part-of-speech tags from an existing spaCy Doc.
"""

def extract_pos_tags(doc) -> list[dict]:
    """
    Return a list of dicts, one per token:
        {"text": token.text, "pos": token.pos_, "tag": token.tag_}
    Guard against doc being None (return empty list).
    """
    if doc is None:
        return []
    return [{"text": token.text, "pos": token.pos_, "tag": token.tag_} for token in doc]

if __name__ == "__main__":
    from agents.nlp_pipeline.loader import get_doc
    
    sample_text = "The quick brown fox jumps over the lazy dog."
    doc = get_doc(sample_text)
    
    pos_tags = extract_pos_tags(doc)
    print("POS Tags:")
    for pt in pos_tags:
        print(pt)
