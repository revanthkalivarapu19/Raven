"""
Extracts named entities from an existing spaCy Doc.
"""

def extract_entities(doc) -> list[dict]:
    """
    Return a list of dicts, one per named entity found:
        {"text": ent.text, "label": ent.label_}
    Guard against doc being None (return empty list).
    """
    if doc is None:
        return []
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

if __name__ == "__main__":
    from agents.nlp_pipeline.loader import get_doc
    
    sample_text = "Prime Minister Modi met officials at the United Nations in New York on Monday."
    doc = get_doc(sample_text)
    
    entities = extract_entities(doc)
    print("Entities:")
    for ent in entities:
        print(ent)
