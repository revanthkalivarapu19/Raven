"""
Extracts tokens and sentences from an existing spaCy Doc.
"""

def extract_tokens(doc) -> list[str]:
    """Return a list of token text strings from the Doc."""
    if doc is None:
        return []
    return [token.text for token in doc]

def extract_sentences(doc) -> list[str]:
    """Return a list of sentence strings from the Doc."""
    if doc is None:
        return []
    return [sent.text for sent in doc.sents]

if __name__ == "__main__":
    from agents.nlp_pipeline.loader import get_doc

    sample_text = "This is sentence one. And here is sentence two!"
    doc = get_doc(sample_text)
    
    tokens = extract_tokens(doc)
    sentences = extract_sentences(doc)
    
    print("Tokens:", tokens)
    print("Sentences:", sentences)
