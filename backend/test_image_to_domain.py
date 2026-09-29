from agents.input_processing.ocr import extract_text_from_image
from agents.nlp_pipeline.nlp_pipeline import run_nlp_pipeline
from agents.claim_extraction.claim_extraction_agent import ClaimExtractionAgent
from agents.domain_detection.domain_agent import DomainDetectionAgent


def main():

    # ==========================================
    # 1. IMAGE INPUT
    # ==========================================

    image_path = "agents/input_processing/1.png"

    extracted_text = extract_text_from_image(image_path)

    print("\n========== OCR RESULT ==========")
    print("Extracted Text:", extracted_text)

    if not extracted_text.strip():
        print("No text found in image.")
        return

    # ==========================================
    # 2. NLP PIPELINE
    # ==========================================

    nlp_result = run_nlp_pipeline(extracted_text)

    print("\n========== NLP PIPELINE RESULT ==========")
    print("Text:", nlp_result["text"])
    print("Tokens:", nlp_result["tokens"])
    print("Sentences:", nlp_result["sentences"])
    print("POS Tags:", nlp_result["pos_tags"])
    print("Entities:", nlp_result["entities"])
    print("Dependencies:", nlp_result["dependencies"])
    print("Keywords:", nlp_result["keywords"])  

    # ==========================================
    # 3. CLAIM EXTRACTION
    # ==========================================

    claim_agent = ClaimExtractionAgent()

    extracted_claim = claim_agent.extract_claim(
        nlp_result["text"]
    )

    print("\n========== CLAIM EXTRACTION ==========")
    print("Extracted Claim:", extracted_claim)

    if not extracted_claim:
        print("No claim could be extracted.")
        return

    # ==========================================
    # 4. DOMAIN DETECTION
    # ==========================================

    domain_agent = DomainDetectionAgent()

    domain_result = domain_agent.detect_domain(
        extracted_claim
    )

    print("\n========== DOMAIN DETECTION ==========")
    print("Domain     :", domain_result["domain"])
    print("Confidence :", domain_result["confidence"])
    print("Reason     :", domain_result["reason"])


if __name__ == "__main__":
    main()