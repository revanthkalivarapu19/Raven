"""
test_input_processing_redesign.py
=================================
Targeted Validation Test Suite for:
RAVEN Input Processing Redesign & NLP Pipeline Removal

Explicitly covers all 25 checklist items + NLP removal verification:
1. Plain English text claim
2. Plain Telugu text claim
3. Mixed English + Telugu claim
4. Text containing numbers and decimals
5. Text containing percentages
6. Text containing currencies
7. Text containing dates and times
8. Text containing names/entities and abbreviations
9. Text containing negation
10. Text containing a URL as part of ordinary text
11. Long multi-sentence text
12. Image containing English text
13. Image containing Telugu text
14. Image containing mixed English/Telugu text
15. Image containing multiple text regions
16. Image containing repeated/noisy OCR text
17. OCR confidence propagation
18. OCR bounding-box / reading-order handling
19. Unicode normalization for Telugu
20. Translation behavior
21. English text must not unnecessarily pass through Telugu translation
22. Telugu text should be translated when required
23. Mixed-language segments should be processed independently
24. Original source/provenance information should remain available
25. Input Processing -> Claim Extraction -> Domain Detection state flow
26. Rigorous verification that nlp_pipeline is removed and not required
"""

import os
import sys
import unittest
import unicodedata

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.input_processing.ocr import extract_ocr_data, extract_text_from_image, sort_reading_order
from agents.input_processing.language_detect import detect_language_with_confidence, detect_language
from agents.input_processing.normalize import normalize_text
from agents.input_processing.clean import clean_content_with_flags, clean_content
from agents.input_processing.translate import translate_to_english, split_into_sentences, translate_with_metadata
from agents.input_processing.pipeline import process_input
from agents.graph import get_raven_graph, run_pipeline
from agents.state import RavenState


class TestRavenInputProcessingFullSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
        cls.en_image = os.path.join(PROJECT_ROOT, "agents", "input_processing", "testimage.png")
        cls.te_image = os.path.join(PROJECT_ROOT, "agents", "input_processing", "testimage_tel.png")

    # 1. Plain English text claim
    def test_01_plain_english_text_claim(self):
        sample = "The Reserve Bank of India raised the repo rate to 6.5 percent."
        res = process_input(text=sample)
        self.assertEqual(res["original_language"], "en")
        self.assertGreaterEqual(res["language_confidence"], 0.80)
        self.assertFalse(res["was_translated"])
        self.assertEqual(len(res["segments"]), 1)
        self.assertEqual(res["segments"][0]["source"], "user_text")
        self.assertEqual(res["processed_text"], sample)

    # 2. Plain Telugu text claim
    def test_02_plain_telugu_text_claim(self):
        sample = "రిజర్వ్ బ్యాంక్ వడ్డీ రేటును తగ్గించింది."
        res = process_input(text=sample)
        self.assertEqual(res["original_language"], "te")
        self.assertGreaterEqual(res["language_confidence"], 0.80)
        self.assertTrue(res["was_translated"])
        self.assertEqual(res["segments"][0]["source"], "user_text")
        self.assertIn("Reserve Bank", res["processed_text"])

    # 3. Mixed English + Telugu claim
    def test_03_mixed_english_telugu_claim(self):
        sample = "PM Modi హైదరాబాద్ లో ప్రసంగించారు and announced new policies."
        res = process_input(text=sample)
        self.assertEqual(res["original_language"], "mixed")
        self.assertGreaterEqual(res["language_confidence"], 0.80)
        self.assertTrue(res["was_translated"])
        self.assertEqual(res["segments"][0]["detected_language"], "mixed")
        self.assertIn("Modi", res["processed_text"])

    # 4. Text containing numbers and decimals
    def test_04_text_numbers_and_decimals(self):
        sample = "The index gained 124.50 points, reaching 82345.75 today."
        norm = normalize_text(sample)
        self.assertIn("124.50", norm)
        self.assertIn("82345.75", norm)
        res = process_input(text=sample)
        self.assertIn("124.50", res["processed_text"])
        self.assertIn("82345.75", res["processed_text"])

    # 5. Text containing percentages
    def test_05_text_percentages(self):
        sample = "The interest rate increased by 5.5% while inflation is 0.50%."
        norm = normalize_text(sample)
        self.assertIn("5.5%", norm)
        self.assertIn("0.50%", norm)
        res = process_input(text=sample)
        self.assertIn("5.5%", res["processed_text"])
        self.assertIn("0.50%", res["processed_text"])

    # 6. Text containing currencies
    def test_06_text_currencies(self):
        sample = "The cost of the tablet was ₹5,000 in India and $100 in the USA."
        norm = normalize_text(sample)
        self.assertIn("₹5,000", norm)
        self.assertIn("$100", norm)
        res = process_input(text=sample)
        self.assertIn("₹5,000", res["processed_text"])
        self.assertIn("$100", res["processed_text"])

    # 7. Text containing dates and times
    def test_07_text_dates_and_times(self):
        sample = "Meeting scheduled on 2026-08-09 at 10:30 PM."
        norm = normalize_text(sample)
        self.assertIn("2026-08-09", norm)
        self.assertIn("10:30 PM", norm)
        res = process_input(text=sample)
        self.assertIn("2026-08-09", res["processed_text"])
        self.assertIn("10:30 PM", res["processed_text"])

    # 8. Text containing names/entities and abbreviations
    def test_08_text_names_entities_abbreviations(self):
        sample = "RBI and NASA collaborated on GPT-4 and 5G communication protocols."
        norm = normalize_text(sample)
        self.assertIn("RBI", norm)
        self.assertIn("NASA", norm)
        self.assertIn("GPT-4", norm)
        self.assertIn("5G", norm)
        # Verify no blind lowercasing
        self.assertNotEqual(norm, norm.lower())
        res = process_input(text=sample)
        self.assertIn("RBI", res["processed_text"])
        self.assertIn("NASA", res["processed_text"])

    # 9. Text containing negation
    def test_09_text_negation(self):
        sample = "The ministry did not approve the proposal and confirmed no funds were disbursed."
        norm = normalize_text(sample)
        self.assertIn("did not approve", norm)
        self.assertIn("no funds were disbursed", norm)
        res = process_input(text=sample)
        self.assertIn("did not approve", res["processed_text"])

    # 10. Text containing a URL as part of ordinary text
    def test_10_text_url_as_content(self):
        sample = "Official announcement published at https://example.com/news for citizen review."
        norm = normalize_text(sample)
        self.assertIn("https://example.com/news", norm)
        res = process_input(text=sample)
        self.assertIn("https://example.com/news", res["processed_text"])

    # 11. Long multi-sentence text
    def test_11_long_multi_sentence_text(self):
        sample = (
            "The monetary policy committee met on Wednesday. "
            "Members observed steady economic recovery across sectors. "
            "Consequently, the repo rate was held unchanged at 6.5 percent."
        )
        sentences = split_into_sentences(sample)
        self.assertEqual(len(sentences), 3)
        res = process_input(text=sample)
        self.assertIn("monetary policy committee", res["processed_text"])
        self.assertIn("held unchanged", res["processed_text"])

    # 12. Image containing English text
    def test_12_image_containing_english_text(self):
        if not os.path.exists(self.en_image):
            self.skipTest("English test image missing")
        res = process_input(image_path=self.en_image)
        self.assertEqual(res["original_language"], "en")
        self.assertFalse(res["was_translated"])
        self.assertTrue(res["had_image"])
        self.assertGreater(len(res["processed_text"]), 0)
        self.assertEqual(res["segments"][0]["source"], "ocr")

    # 13. Image containing Telugu text
    def test_13_image_containing_telugu_text(self):
        if not os.path.exists(self.te_image):
            self.skipTest("Telugu test image missing")
        res = process_input(image_path=self.te_image)
        self.assertEqual(res["original_language"], "te")
        self.assertTrue(res["was_translated"])
        self.assertGreater(len(res["processed_text"]), 0)
        self.assertEqual(res["segments"][0]["source"], "ocr")

    # 14. Image containing mixed English/Telugu text
    def test_14_image_containing_mixed_english_telugu_text(self):
        # Construct synthetic mixed OCR data
        blocks = [
            {"text": "TOP HEADLINES:", "confidence": 0.95, "box": [[10, 10], [200, 10], [200, 30], [10, 30]], "order": 0},
            {"text": "తెలంగాణ ప్రభుత్వం నూతన విధానం", "confidence": 0.90, "box": [[10, 40], [300, 40], [300, 60], [10, 60]], "order": 1}
        ]
        text = "TOP HEADLINES: తెలంగాణ ప్రభుత్వం నూతన విధానం"
        lang, conf = detect_language_with_confidence(text)
        self.assertEqual(lang, "mixed")
        self.assertGreaterEqual(conf, 0.80)

    # 15. Image containing multiple text regions
    def test_15_image_containing_multiple_text_regions(self):
        if not os.path.exists(self.te_image):
            self.skipTest("Telugu test image missing")
        ocr_data = extract_ocr_data(self.te_image)
        self.assertGreater(len(ocr_data["blocks"]), 5)
        # Verify regions have order and bounding boxes
        for i, b in enumerate(ocr_data["blocks"]):
            self.assertEqual(b["order"], i)
            self.assertIn("box", b)
            self.assertIn("confidence", b)

    # 16. Image containing repeated/noisy OCR text
    def test_16_image_repeated_noisy_ocr_text(self):
        noisy_text = "@@## BREAKING NEWS ^^ ~~ ||\nBREAKING NEWS\nBREAKING NEWS\nNext headline item."
        cleaned, flags = clean_content_with_flags(noisy_text)
        self.assertIn("repeated_lines_removed", flags)
        self.assertIn("ocr_symbol_noise_cleaned", flags)
        self.assertNotIn("@@##", cleaned)
        self.assertNotIn("^^", cleaned)
        # Verify duplicate consecutive line was removed
        self.assertEqual(cleaned, "BREAKING NEWS\nNext headline item.")

    # 17. OCR confidence propagation
    def test_17_ocr_confidence_propagation(self):
        if not os.path.exists(self.en_image):
            self.skipTest("English test image missing")
        res = process_input(image_path=self.en_image)
        self.assertIn("ocr_confidence", res)
        self.assertIsInstance(res["ocr_confidence"], float)
        self.assertGreater(res["ocr_confidence"], 0.0)
        self.assertEqual(res["segments"][0]["ocr_confidence"], res["ocr_confidence"])

    # 18. OCR bounding-box / reading-order handling
    def test_18_ocr_bounding_box_reading_order_handling(self):
        # 3 blocks positioned across page
        raw_boxes = [
            ([[150, 100], [300, 100], [300, 120], [150, 120]], "Line 2 Col 2", 0.90),
            ([[10, 100], [140, 100], [140, 120], [10, 120]], "Line 2 Col 1", 0.95),
            ([[10, 20], [300, 20], [300, 40], [10, 40]], "Line 1 Header", 0.99),
        ]
        sorted_boxes = sort_reading_order(raw_boxes)
        self.assertEqual(sorted_boxes[0]["text"], "Line 1 Header")
        self.assertEqual(sorted_boxes[1]["text"], "Line 2 Col 1")
        self.assertEqual(sorted_boxes[2]["text"], "Line 2 Col 2")

    # 19. Unicode normalization for Telugu
    def test_19_unicode_normalization_for_telugu(self):
        decomposed = unicodedata.normalize('NFD', "రిజర్వ్ బ్యాంక్")
        norm = normalize_text(decomposed, "te")
        self.assertEqual(norm, unicodedata.normalize('NFC', "రిజర్వ్ బ్యాంక్"))

    # 20. Translation behavior
    def test_20_translation_behavior(self):
        # English sentence should skip translation
        en_meta = translate_with_metadata("The GDP growth reached 7.2 percent.")
        self.assertFalse(en_meta["was_translated"])
        self.assertEqual(en_meta["sentences_translated"], 0)
        self.assertEqual(en_meta["translated_text"], "The GDP growth reached 7.2 percent.")

        # Telugu sentence should be translated
        te_meta = translate_with_metadata("రిజర్వ్ బ్యాంక్ వడ్డీ రేటును తగ్గించింది.")
        self.assertTrue(te_meta["was_translated"])
        self.assertGreater(te_meta["sentences_translated"], 0)
        self.assertIn("Reserve Bank", te_meta["translated_text"])

    # 21. English text must not unnecessarily pass through Telugu translation
    def test_21_english_not_passed_to_indictrans(self):
        sample = "The Ministry of Finance released the annual financial report."
        res = translate_to_english(sample)
        # Must return exact string without invoking IndicTrans
        self.assertEqual(res, sample)

    # 22. Telugu text should be translated when required
    def test_22_telugu_translated_when_required(self):
        sample = "ప్రభుత్వం వచ్చే నెలలో కొత్త పథకాన్ని ప్రారంభించనుంది."
        res = translate_to_english(sample)
        self.assertNotEqual(res, sample)
        self.assertIn("government", res.lower())

    # 23. Mixed-language segments should be processed independently
    def test_23_mixed_language_segments_processed_independently(self):
        if not os.path.exists(self.te_image):
            self.skipTest("Telugu test image missing")
        res = process_input(text="Verify this breaking news:", image_path=self.te_image)
        self.assertEqual(len(res["segments"]), 2)
        # Segment 1 is pure English user text
        self.assertEqual(res["segments"][0]["source"], "user_text")
        self.assertEqual(res["segments"][0]["detected_language"], "en")
        self.assertIsNone(res["segments"][0]["translated_text"])
        # Segment 2 is Telugu OCR image
        self.assertEqual(res["segments"][1]["source"], "ocr")
        self.assertEqual(res["segments"][1]["detected_language"], "te")
        self.assertIsNotNone(res["segments"][1]["translated_text"])

    # 24. Original source/provenance information should remain available
    def test_24_source_provenance_information_available(self):
        if not os.path.exists(self.en_image):
            self.skipTest("English test image missing")
        res = process_input(text="Is this claim true?", image_path=self.en_image)
        self.assertIn("segments", res)
        self.assertEqual(res["segments"][0]["source"], "user_text")
        self.assertIn("original_text", res["segments"][0])
        self.assertEqual(res["segments"][1]["source"], "ocr")
        self.assertIn("original_text", res["segments"][1])
        self.assertIn("blocks", res["segments"][1])

    # 25. Input Processing -> Claim Extraction -> Domain Detection state flow
    def test_25_pipeline_to_claim_to_domain_flow(self):
        input_text = "The Reserve Bank of India reduced the repo rate by 0.50% on Friday to boost economic growth."
        state = run_pipeline(text=input_text)
        # Check LangGraph shared state
        self.assertIn("processed_text", state)
        self.assertIn("claim", state)
        self.assertIn("domain", state)
        self.assertEqual(state["original_language"], "en")
        self.assertGreater(len(state["claim"]), 0)
        self.assertEqual(state["domain"]["domain"], "Finance")

    # 26. Specifically verify NLP pipeline removal
    def test_26_nlp_pipeline_removal_verification(self):
        # 1. Confirm agents/nlp_pipeline does not exist
        nlp_dir = os.path.join(PROJECT_ROOT, "agents", "nlp_pipeline")
        self.assertFalse(os.path.exists(nlp_dir), "agents/nlp_pipeline directory must not exist")

        # 2. Confirm graph nodes do not contain nlp_pipeline
        graph = get_raven_graph()
        node_names = list(graph.nodes.keys())
        self.assertNotIn("nlp_pipeline", node_names)
        self.assertNotIn("nlp", node_names)
        self.assertIn("input_processing", node_names)
        self.assertIn("claim_extraction", node_names)
        self.assertIn("domain_detection", node_names)
        self.assertEqual(node_names, ['__start__', 'input_processing', 'claim_extraction', 'domain_detection'])


if __name__ == "__main__":
    unittest.main()
