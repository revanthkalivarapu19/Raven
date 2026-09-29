"""
Loads the spaCy English model once, and provides a function to process text into a Doc object.
"""
import spacy

# Load the model ONCE at module level
nlp = spacy.load("en_core_web_sm")

def get_doc(text: str):
    """
    Run the loaded spaCy model on text and return the Doc object.
    Guard against None/empty text - return None in that case.
    """
    if not text:
        return None
    return nlp(text)

if __name__ == "__main__":
    sample_text = "This is a sample sentence to test the spaCy loader."
    doc = get_doc(sample_text)
    if doc:
        print(f"Text: {doc.text}")
        print(f"Length: {len(doc)} tokens")
    else:
        print("Doc is None")
