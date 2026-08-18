"""
Orchestrates the full NLP pipeline stage.
"""
import logging
from agents.nlp_pipeline.loader import get_doc
from agents.nlp_pipeline.tokenizer import extract_tokens, extract_sentences
from agents.nlp_pipeline.pos_tagger import extract_pos_tags
from agents.nlp_pipeline.ner import extract_entities
from agents.nlp_pipeline.dependency_parser import extract_dependencies
from agents.nlp_pipeline.keyword_extractor import extract_keywords

logger = logging.getLogger(__name__)

def run_nlp_pipeline(processed_text: str) -> dict:
    """
    Run the full NLP pipeline on processed_text and return a
    structured dictionary:

       {
    "text": processed_text,
    "tokens": [...],
    "sentences": [...],
    "pos_tags": [...],
    "entities": [...],
    "dependencies": [...],
    "keywords": [...]
}
    Steps:
        1. Call loader.get_doc(processed_text) ONCE to get the Doc.
        2. If the Doc is None (empty/invalid input), raise a
           clear ValueError rather than silently returning
           empty results.
        3. Pass the SAME Doc object into each extractor function
           from tokenizer.py, pos_tagger.py, ner.py,
           dependency_parser.py, and keyword_extractor.py.
        4. Combine all results into the dictionary above and
           return it.
    """
    if processed_text is None:
        raise ValueError("Input text cannot be None.")
    
    # Check if empty or whitespace using strip()
    if not str(processed_text).strip():
        raise ValueError("Input text cannot be empty.")
        
    logger.info(f"Received text of length {len(processed_text)} for NLP processing.")
    
    # 1. Call loader.get_doc(processed_text) ONCE to get the Doc.
    doc = get_doc(processed_text)
    
    # 2. If the Doc is None (empty/invalid input), raise a clear ValueError
    if doc is None:
        raise ValueError("Failed to create a spaCy Doc from the input text.")
        
    logger.info("Created spaCy Doc successfully.")
    
    # 3. Pass the SAME Doc object into each extractor function
    tokens = extract_tokens(doc)
    logger.info("Extraction complete: tokens")
    
    sentences = extract_sentences(doc)
    logger.info("Extraction complete: sentences")
    
    pos_tags = extract_pos_tags(doc)
    logger.info("Extraction complete: pos_tags")
    
    entities = extract_entities(doc)
    logger.info("Extraction complete: entities")
    
    dependencies = extract_dependencies(doc)
    logger.info("Extraction complete: dependencies")
    
    keywords = extract_keywords(doc)
    logger.info("Extraction complete: keywords")
    
    logger.info("NLP pipeline finished successfully.")
    
    # 4. Combine all results into the dictionary and return it.
    return {
        "text": processed_text,
        "tokens": tokens,
        "sentences": sentences,
        "pos_tags": pos_tags,
        "entities": entities,
        "dependencies": dependencies,
        "keywords": keywords
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test 1: Normal English news-style sentence
    print("--- Test 1: Normal text ---")
    sample_text = "Prime Minister Modi announced a new scheme in Hyderabad on Monday, according to officials."
    result = run_nlp_pipeline(sample_text)
    print("\nResult dictionary:")
    print("text:", result["text"])
    print("tokens:", result["tokens"][:5], "... (truncated for brevity)")
    print("sentences:", result["sentences"])
    print("pos_tags:", result["pos_tags"][:3], "... (truncated)")
    print("entities:", result["entities"])
    print("dependencies:", result["dependencies"][:3], "... (truncated)")
    print("keywords:", result["keywords"])
    
    # Test 2: Empty string input
    print("\n--- Test 2: Empty string ---")
    try:
        run_nlp_pipeline("   ")
        print("FAIL: Expected ValueError for empty string")
    except ValueError as e:
        print(f"PASS: Caught ValueError - {e}")
        
    # Test 3: None input
    print("\n--- Test 3: None input ---")
    try:
        run_nlp_pipeline(None)
        print("FAIL: Expected ValueError for None")
    except ValueError as e:
        print(f"PASS: Caught ValueError - {e}")
