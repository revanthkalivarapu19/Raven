"""
Extracts dependency relations from an existing spaCy Doc.
"""

def extract_dependencies(doc) -> list[dict]:
    """
    Return a list of dicts, one per token:
        {
            "text": token.text,
            "dep": token.dep_,
            "head": token.head.text,
            "pos": token.pos_
        }
    Guard against doc being None (return empty list).
    """
    if doc is None:
        return []
    return [{
        "text": token.text,
        "dep": token.dep_,
        "head": token.head.text,
        "pos": token.pos_
    } for token in doc]

if __name__ == "__main__":
    from agents.nlp_pipeline.loader import get_doc
    
    sample_text = "The quick brown fox jumps over the lazy dog."
    doc = get_doc(sample_text)
    
    deps = extract_dependencies(doc)
    print("Dependencies:")
    for dep in deps:
        print(dep)
